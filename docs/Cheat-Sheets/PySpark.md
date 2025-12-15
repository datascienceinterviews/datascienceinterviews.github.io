
---
title: PySpark Cheat Sheet
description: A comprehensive reference guide for PySpark, covering setup, data loading, transformations, actions, SQL, MLlib, structured streaming, performance tuning, and more.
---

# PySpark Cheat Sheet

[TOC]

This cheat sheet provides an exhaustive overview of the PySpark API, covering essential concepts, code snippets, and best practices for efficient data processing and machine learning with Apache Spark. It aims to be a one-stop reference for common tasks.

## Getting Started

### PySpark Architecture

```
    ┌─────────────────────────────────────────────────┐
    │              Driver Program                      │
    │  ┌──────────────────────────────────────────┐   │
    │  │        SparkContext/SparkSession         │   │
    │  └──────────────┬───────────────────────────┘   │
    └─────────────────┼───────────────────────────────┘
                      │
           ┌──────────┴──────────┐
           │   Cluster Manager   │
           │  (YARN/Mesos/K8s)   │
           └──────────┬──────────┘
                      │
        ┌─────────────┼─────────────┐
        ↓             ↓             ↓
    ┌─────────┐  ┌─────────┐  ┌─────────┐
    │ Worker  │  │ Worker  │  │ Worker  │
    │  Node   │  │  Node   │  │  Node   │
    │┌───────┐│  │┌───────┐│  │┌───────┐│
    ││Executor││ ││Executor││ ││Executor││
    ││ Cache  ││  ││ Cache  ││  ││ Cache  ││
    ││ Tasks  ││  ││ Tasks  ││  ││ Tasks  ││
    │└───────┘│  │└───────┘│  │└───────┘│
    └─────────┘  └─────────┘  └─────────┘
```

### Installation

```bash
# Basic installation
pip install pyspark

# With specific version
pip install pyspark==3.5.0

# With additional dependencies
pip install pyspark[sql,ml,streaming]
```

Using virtual environment:

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate     # On Windows

# Install PySpark
pip install pyspark
```

### SparkSession (Recommended for Spark 2.0+)

```python
from pyspark.sql import SparkSession

# Basic session
spark = SparkSession.builder \
    .appName("MyPySparkApp") \
    .master("local[*]") \
    .getOrCreate()

# With configurations
spark = SparkSession.builder \
    .appName("MyPySparkApp") \
    .master("local[*]") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .enableHiveSupport() \
    .getOrCreate()

# Access SparkContext from SparkSession
sc = spark.sparkContext

# Get Spark version
print(spark.version)
```

### SparkContext (Legacy API)

```python
from pyspark import SparkContext, SparkConf

# Configuration
conf = SparkConf() \
    .setAppName("MyPySparkApp") \
    .setMaster("local[*]") \
    .set("spark.executor.memory", "4g")

# Create SparkContext
sc = SparkContext(conf=conf)

# Create SparkSession from existing context
from pyspark.sql import SparkSession
spark = SparkSession(sparkContext=sc)
```

### Stopping Spark

```python
# Stop SparkSession (also stops SparkContext)
spark.stop()

# Stop SparkContext only
sc.stop()
```

## Data Loading

### Data Loading Flow

```
    ┌──────────────┐
    │ Data Sources │
    └──────┬───────┘
           │
    ┌──────┴──────────────────────────────────┐
    │                                          │
    ↓                ↓            ↓            ↓
┌────────┐      ┌──────┐    ┌────────┐   ┌──────────┐
│  Text  │      │ CSV  │    │ JSON   │   │  JDBC    │
│ Files  │      │Files │    │ Files  │   │Database  │
└───┬────┘      └───┬──┘    └───┬────┘   └────┬─────┘
    │               │           │             │
    └───────────────┴───────────┴─────────────┘
                    │
                    ↓
           ┌────────────────┐
           │  Spark Reader  │
           └────────┬───────┘
                    │
                    ↓
           ┌────────────────┐
           │   DataFrame    │
           └────────────────┘
```

### Loading from Text Files

```python
# Using SparkContext (RDD)
lines = sc.textFile("path/to/textfile.txt")
lines = sc.textFile("hdfs://path/to/textfile.txt")  # HDFS
lines = sc.textFile("s3a://bucket/path/file.txt")   # S3

# Multiple files
lines = sc.textFile("path/to/directory/*.txt")

# Using SparkSession (DataFrame)
df = spark.read.text("path/to/textfile.txt")

# Read multiple text files
df = spark.read.text(["file1.txt", "file2.txt"])
```

### Loading from CSV Files

```python
# Basic CSV read
df = spark.read.csv("path/to/file.csv", header=True, inferSchema=True)

# With detailed options
df = spark.read.csv(
    "path/to/file.csv",
    header=True,
    inferSchema=True,
    sep=",",
    quote='"',
    escape="\\",
    nullValue="NULL",
    dateFormat="yyyy-MM-dd",
    timestampFormat="yyyy-MM-dd HH:mm:ss"
)

# Alternative syntax
df = spark.read \
    .format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("mode", "DROPMALFORMED") \
    .load("path/to/file.csv")

# With explicit schema
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

schema = StructType([
    StructField("name", StringType(), True),
    StructField("age", IntegerType(), True),
    StructField("city", StringType(), True)
])

df = spark.read.csv("path/to/file.csv", schema=schema, header=True)

# Read multiple CSV files
df = spark.read.csv("path/to/directory/*.csv", header=True, inferSchema=True)
```

### Loading from JSON Files

```python
# Basic JSON read
df = spark.read.json("path/to/file.json")

# With options
df = spark.read \
    .format("json") \
    .option("multiLine", "true") \
    .option("mode", "PERMISSIVE") \
    .load("path/to/file.json")

# Read JSON lines (one JSON object per line)
df = spark.read.json("path/to/jsonlines.jsonl")

# With schema
df = spark.read.json("path/to/file.json", schema=schema)

# Read multiple JSON files
df = spark.read.json("path/to/directory/*.json")
```

### Loading from Parquet Files

```python
# Basic Parquet read (preserves schema automatically)
df = spark.read.parquet("path/to/file.parquet")

# Read multiple Parquet files
df = spark.read.parquet("path/to/directory/*.parquet")

# Read partitioned data
df = spark.read.parquet("path/to/partitioned_data")

# With merge schema (for schema evolution)
df = spark.read \
    .option("mergeSchema", "true") \
    .parquet("path/to/file.parquet")
```

### Loading from ORC Files

```python
# Basic ORC read
df = spark.read.orc("path/to/file.orc")

# With options
df = spark.read \
    .format("orc") \
    .option("mergeSchema", "true") \
    .load("path/to/file.orc")

# Read multiple ORC files
df = spark.read.orc("path/to/directory/*.orc")
```

### Loading from Avro Files

```python
# Basic Avro read
df = spark.read.format("avro").load("path/to/file.avro")

# Read multiple Avro files
df = spark.read.format("avro").load("path/to/directory/*.avro")
```

### Loading from JDBC Database

```python
# Basic JDBC read
df = spark.read.format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5432/mydatabase") \
    .option("dbtable", "mytable") \
    .option("user", "myuser") \
    .option("password", "mypassword") \
    .option("driver", "org.postgresql.Driver") \
    .load()

# With query
df = spark.read.format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5432/mydatabase") \
    .option("query", "SELECT * FROM mytable WHERE age > 25") \
    .option("user", "myuser") \
    .option("password", "mypassword") \
    .load()

# With partitioning for parallel reads
df = spark.read.format("jdbc") \
    .option("url", "jdbc:mysql://localhost:3306/mydb") \
    .option("dbtable", "mytable") \
    .option("user", "myuser") \
    .option("password", "mypassword") \
    .option("partitionColumn", "id") \
    .option("lowerBound", "1") \
    .option("upperBound", "100000") \
    .option("numPartitions", "10") \
    .load()

# Common JDBC URLs
# PostgreSQL: jdbc:postgresql://host:5432/database
# MySQL: jdbc:mysql://host:3306/database
# SQL Server: jdbc:sqlserver://host:1433;databaseName=database
# Oracle: jdbc:oracle:thin:@host:1521:database
```

### Loading from Delta Lake

```python
from delta.tables import DeltaTable

# Read Delta table
df = spark.read.format("delta").load("path/to/delta_table")

# Alternative: Use DeltaTable
deltaTable = DeltaTable.forPath(spark, "path/to/delta_table")
df = deltaTable.toDF()

# Read specific version (time travel)
df = spark.read.format("delta") \
    .option("versionAsOf", 0) \
    .load("path/to/delta_table")

# Read at specific timestamp
df = spark.read.format("delta") \
    .option("timestampAsOf", "2023-01-01 00:00:00") \
    .load("path/to/delta_table")
```

### Loading from Other Sources

```python
# Hive table
df = spark.table("database.tablename")
df = spark.sql("SELECT * FROM database.tablename")

# From Pandas DataFrame
import pandas as pd
pandas_df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
spark_df = spark.createDataFrame(pandas_df)

# From Python collections
data = [("Alice", 25), ("Bob", 30)]
df = spark.createDataFrame(data, ["name", "age"])

# From RDD
rdd = sc.parallelize([("Alice", 25), ("Bob", 30)])
df = spark.createDataFrame(rdd, ["name", "age"])
```

## DataFrames

### DataFrame Lifecycle

```
    ┌──────────────┐
    │ Data Source  │
    └──────┬───────┘
           │
           ↓
    ┌──────────────┐
    │   Load Data  │
    └──────┬───────┘
           │
           ↓
    ┌──────────────┐
    │ Transform    │────┐
    └──────┬───────┘    │
           │            │
           ↓            │
    ┌──────────────┐    │
    │   Action     │    │
    └──────┬───────┘    │
           │            │
           ↓ (Trigger)  │
    ┌──────────────┐    │
    │   Execute    │    │
    │  Lazy Eval   │    │
    └──────┬───────┘    │
           │            │
           ↓            │
    ┌──────────────┐    │
    │   Result     │    │
    └──────────────┘    │
                        │
           More Transforms? ──┘
```

### Creating DataFrames

From tuple list:

```python
# With column names
data = [("Alice", 30), ("Bob", 25), ("Charlie", 35)]
df = spark.createDataFrame(data, ["Name", "Age"])

# Inferred schema
data = [("Alice", 30, "Engineer"), ("Bob", 25, "Doctor")]
df = spark.createDataFrame(data)
df.show()
```

From RDD:

```python
# Basic RDD to DataFrame
data = [("Alice", 30), ("Bob", 25)]
rdd = spark.sparkContext.parallelize(data)
df = spark.createDataFrame(rdd, schema=["Name", "Age"])

# With explicit schema
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

schema = StructType([
    StructField("Name", StringType(), nullable=False),
    StructField("Age", IntegerType(), nullable=True)
])
df = spark.createDataFrame(rdd, schema=schema)
```

From list of dictionaries:

```python
data = [
    {"Name": "Alice", "Age": 30, "City": "NYC"},
    {"Name": "Bob", "Age": 25, "City": "LA"}
]
df = spark.createDataFrame(data)
```

From Pandas DataFrame:

```python
import pandas as pd

# Convert Pandas to Spark DataFrame
pandas_df = pd.DataFrame({
    "Name": ["Alice", "Bob"],
    "Age": [30, 25]
})
spark_df = spark.createDataFrame(pandas_df)

# Convert Spark to Pandas DataFrame
pandas_df = spark_df.toPandas()
```

From Row objects:

```python
from pyspark.sql import Row

# Create Row objects
data = [
    Row(Name="Alice", Age=30),
    Row(Name="Bob", Age=25)
]
df = spark.createDataFrame(data)
```

Empty DataFrame with schema:

```python
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

schema = StructType([
    StructField("Name", StringType(), True),
    StructField("Age", IntegerType(), True)
])
df = spark.createDataFrame([], schema)

### DataFrame Operations

Basic inspection:

```python
# Display rows
df.show()                    # Show first 20 rows
df.show(5)                   # Show first 5 rows
df.show(truncate=False)      # Show without truncating
df.show(vertical=True)       # Vertical format

# Schema and structure
df.printSchema()             # Print schema tree
df.schema                    # Get schema object
df.columns                   # Get column names list
df.dtypes                    # Get column types [(name, type)]

# Statistics
df.count()                   # Count rows
df.describe().show()         # Summary statistics
df.summary("count", "mean", "stddev", "min", "max").show()

# Sample data
df.head(5)                   # Return first 5 rows as list
df.first()                   # Return first row
df.take(5)                   # Return first 5 rows as list
df.tail(5)                   # Return last 5 rows (use carefully)
```

Column operations:

```python
from pyspark.sql.functions import col, lit

# Select columns
df.select("name", "age").show()
df.select(df.name, df.age).show()
df.select(col("name"), col("age")).show()

# Add new column
df.withColumn("age_plus_10", col("age") + 10)
df.withColumn("full_name", col("first_name") + " " + col("last_name"))
df.withColumn("status", lit("active"))  # Add constant value

# Rename columns
df.withColumnRenamed("old_name", "new_name")
df.toDF("new_name1", "new_name2")  # Rename all columns

# Drop columns
df.drop("column1")
df.drop("column1", "column2")

# Cast column type
df.withColumn("age", col("age").cast("string"))
df.withColumn("price", col("price").cast("double"))
```

Filtering and conditions:

```python
from pyspark.sql.functions import col

# Filter rows
df.filter(df["age"] > 25)
df.filter(col("age") > 25)
df.where(col("age") > 25)

# Multiple conditions
df.filter((col("age") > 25) & (col("city") == "NYC"))
df.filter((col("age") < 20) | (col("age") > 60))
df.filter(~(col("status") == "inactive"))

# String operations
df.filter(col("name").startswith("A"))
df.filter(col("name").endswith("son"))
df.filter(col("name").contains("li"))

# Null checks
df.filter(col("age").isNull())
df.filter(col("age").isNotNull())

# In operator
df.filter(col("city").isin(["NYC", "LA", "SF"]))
```

Sorting and ordering:

```python
from pyspark.sql.functions import col, desc, asc

# Sort ascending
df.orderBy("age")
df.sort("age")
df.orderBy(col("age").asc())

# Sort descending
df.orderBy(col("age").desc())
df.sort(col("age").desc())

# Multiple columns
df.orderBy(["age", "name"])
df.orderBy(col("age").desc(), col("name").asc())
```

Aggregations:

```python
from pyspark.sql.functions import count, sum, avg, min, max, countDistinct

# Group by
df.groupBy("department").count()
df.groupBy("department").agg(avg("salary"), max("age"))

# Multiple aggregations
df.groupBy("department").agg(
    count("*").alias("total_employees"),
    avg("salary").alias("avg_salary"),
    max("salary").alias("max_salary"),
    min("salary").alias("min_salary")
)

# Without groupBy (global aggregation)
df.agg(avg("salary"), max("age"))
df.select(avg("salary"), count("*"))

# Distinct count
df.agg(countDistinct("department"))
```

Joins:

```
    Join Types Flow:
    
    ┌──────────┐        ┌──────────┐
    │   df1    │        │   df2    │
    └────┬─────┘        └─────┬────┘
         │                    │
         └──────────┬─────────┘
                    │
           ┌────────┴────────┐
           │  Join Operation │
           └────────┬────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
        ↓           ↓           ↓
    ┌───────┐  ┌────────┐  ┌──────┐
    │ inner │  │  left  │  │right │
    └───────┘  │ outer  │  │outer │
               └────────┘  └──────┘
                    │
                    ↓
              ┌──────────┐
              │   full   │
              │  outer   │
              └──────────┘
```

```python
# Inner join (default)
df1.join(df2, df1.id == df2.id)
df1.join(df2, "id")  # If column name is same

# Left outer join
df1.join(df2, df1.id == df2.id, "left")
df1.join(df2, df1.id == df2.id, "left_outer")

# Right outer join
df1.join(df2, df1.id == df2.id, "right")
df1.join(df2, df1.id == df2.id, "right_outer")

# Full outer join
df1.join(df2, df1.id == df2.id, "full")
df1.join(df2, df1.id == df2.id, "full_outer")

# Left semi join (like SQL IN)
df1.join(df2, df1.id == df2.id, "left_semi")

# Left anti join (like SQL NOT IN)
df1.join(df2, df1.id == df2.id, "left_anti")

# Cross join (Cartesian product)
df1.crossJoin(df2)

# Multiple join conditions
df1.join(df2, (df1.id == df2.id) & (df1.date == df2.date))
```

Set operations:

```python
# Union (combines all rows, includes duplicates)
df1.union(df2)

# Union by name (matches by column name)
df1.unionByName(df2)

# Intersection (rows in both)
df1.intersect(df2)

# Subtract (rows in df1 but not in df2)
df1.subtract(df2)

# Distinct (remove duplicates)
df.distinct()
df.dropDuplicates()
df.dropDuplicates(["name", "age"])  # Based on specific columns
```

Advanced operations:

```python
from pyspark.sql import StorageLevel

# Sampling
df.sample(withReplacement=False, fraction=0.5, seed=42)
df.sample(0.3)  # 30% sample

# Random split
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Limit
df.limit(10)

# Pivoting
df.groupBy("date").pivot("category").sum("amount")
df.groupBy("date").pivot("category", ["A", "B"]).sum("amount")

# Rollup (hierarchical aggregation)
df.rollup("country", "city").agg(sum("sales"))

# Cube (multi-dimensional aggregation)
df.cube("year", "quarter").agg(sum("revenue"))

# Caching
df.cache()                              # Cache in memory
df.persist(StorageLevel.MEMORY_AND_DISK)  # Custom storage
df.unpersist()                          # Remove from cache
df.persist().count()                    # Cache and trigger computation
```

Handling nulls:

```python
# Drop rows with nulls
df.na.drop()                      # Drop if any null
df.na.drop(how="all")             # Drop if all nulls
df.na.drop(subset=["age", "name"]) # Drop if null in specific columns

# Fill nulls
df.na.fill(0)                     # Fill all nulls with 0
df.na.fill({"age": 0, "name": "Unknown"})  # Fill by column
df.fillna(0)                      # Alternative syntax

# Replace values
df.na.replace(["old1", "old2"], ["new1", "new2"])
df.replace(to_replace={"old": "new"}, subset=["column"])
```

### Built-in Functions

```python
from pyspark.sql.functions import (
    col, lit, when, coalesce, concat, concat_ws,
    lower, upper, trim, length, substring, regexp_replace,
    split, explode, array_contains,
    year, month, dayofmonth, current_date, current_timestamp,
    datediff, date_add, date_sub, to_date, to_timestamp,
    round, ceil, floor, abs, sqrt, pow,
    md5, sha1, sha2, base64, unbase64
)

# String functions
df.withColumn("lower_name", lower(col("name")))
df.withColumn("upper_name", upper(col("name")))
df.withColumn("trimmed", trim(col("name")))
df.withColumn("name_length", length(col("name")))
df.withColumn("first_3", substring(col("name"), 1, 3))
df.withColumn("cleaned", regexp_replace(col("text"), "[^a-zA-Z]", ""))
df.withColumn("full_name", concat(col("first"), lit(" "), col("last")))
df.withColumn("full_name", concat_ws(" ", col("first"), col("last")))

# Array functions
df.withColumn("words", split(col("sentence"), " "))
df.withColumn("word", explode(col("words")))
df.withColumn("has_item", array_contains(col("items"), "target"))

# Date functions
df.withColumn("year", year(col("date")))
df.withColumn("month", month(col("date")))
df.withColumn("day", dayofmonth(col("date")))
df.withColumn("today", current_date())
df.withColumn("now", current_timestamp())
df.withColumn("days_diff", datediff(col("end_date"), col("start_date")))
df.withColumn("next_week", date_add(col("date"), 7))
df.withColumn("last_week", date_sub(col("date"), 7))

# Math functions
df.withColumn("rounded", round(col("value"), 2))
df.withColumn("ceiling", ceil(col("value")))
df.withColumn("floor", floor(col("value")))
df.withColumn("absolute", abs(col("value")))
df.withColumn("square_root", sqrt(col("value")))

# Conditional operations
df.withColumn("category",
    when(col("age") < 18, "minor")
    .when(col("age") < 65, "adult")
    .otherwise("senior")
)

# Null handling
df.withColumn("filled", coalesce(col("col1"), col("col2"), lit("default")))
```

### User-Defined Functions (UDFs)

```
    UDF Processing Flow:
    
    ┌───────────────┐
    │ Python UDF    │
    │  (Slow)       │
    └───────┬───────┘
            │
            ↓ Serialization
    ┌───────────────┐
    │  JVM Executor │
    └───────┬───────┘
            │
            ↓ Row by Row
    ┌───────────────┐
    │  Result       │
    └───────────────┘
    
    Pandas UDF (Faster):
    
    ┌───────────────┐
    │ Pandas UDF    │
    │ (Vectorized)  │
    └───────┬───────┘
            │
            ↓ Apache Arrow
    ┌───────────────┐
    │  JVM Executor │
    └───────┬───────┘
            │
            ↓ Batch Processing
    ┌───────────────┐
    │  Result       │
    └───────────────┘
```

Standard Python UDF:

```python
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, IntegerType, DoubleType

# Simple UDF
def to_upper(s):
    if s is not None:
        return s.upper()
    return None

to_upper_udf = udf(to_upper, StringType())
df = df.withColumn("upper_name", to_upper_udf(col("name")))

# UDF with decorator
@udf(returnType=IntegerType())
def calculate_age(birth_year):
    from datetime import datetime
    current_year = datetime.now().year
    return current_year - birth_year

df = df.withColumn("age", calculate_age(col("birth_year")))

# UDF with multiple arguments
@udf(returnType=DoubleType())
def calculate_discount(price, discount_pct):
    if price and discount_pct:
        return price * (1 - discount_pct / 100)
    return price

df = df.withColumn("final_price", calculate_discount(col("price"), col("discount")))
```

Pandas UDF (Vectorized, faster):

```python
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType, DoubleType
import pandas as pd

# Series to Series
@pandas_udf(StringType())
def to_upper_pandas(series: pd.Series) -> pd.Series:
    return series.str.upper()

df = df.withColumn("upper_name", to_upper_pandas(col("name")))

# Multiple columns
@pandas_udf(DoubleType())
def calculate_discount(price: pd.Series, discount: pd.Series) -> pd.Series:
    return price * (1 - discount / 100)

df = df.withColumn("final_price", calculate_discount(col("price"), col("discount")))

# Iterator of Series (for large data)
from typing import Iterator

@pandas_udf(DoubleType())
def complex_calculation(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
    for batch in iterator:
        # Process batch
        yield batch * 2

df = df.withColumn("doubled", complex_calculation(col("value")))
```

Best practices for UDFs:

```python
# Register UDF for SQL use
spark.udf.register("to_upper_sql", to_upper, StringType())
df.createOrReplaceTempView("my_table")
result = spark.sql("SELECT to_upper_sql(name) FROM my_table")

# Use built-in functions when possible (much faster)
# Instead of UDF for upper:
df.withColumn("upper_name", upper(col("name")))  # Preferred

# Cache if using UDF multiple times
df_cached = df.cache()
df_cached = df_cached.withColumn("col1", my_udf(col("x")))
df_cached = df_cached.withColumn("col2", my_udf(col("y")))
```

### GroupBy Operations

```python
from pyspark.sql.functions import (
    count, sum, avg, max, min, mean, stddev,
    collect_list, collect_set, countDistinct
)

# Single column grouping
df.groupBy("department").count()
df.groupBy("department").sum("salary")
df.groupBy("department").avg("salary")
df.groupBy("department").max("salary")
df.groupBy("department").min("salary")

# Multiple column grouping
df.groupBy("department", "city").count()

# Multiple aggregations
df.groupBy("department").agg(
    count("*").alias("total_employees"),
    avg("salary").alias("avg_salary"),
    max("salary").alias("max_salary"),
    min("salary").alias("min_salary"),
    sum("bonus").alias("total_bonus"),
    countDistinct("employee_id").alias("unique_employees")
)

# Collect values into list/set
df.groupBy("department").agg(
    collect_list("employee_name").alias("employees"),
    collect_set("skill").alias("unique_skills")
)

# Group and filter
df.groupBy("department") \
    .agg(avg("salary").alias("avg_salary")) \
    .filter(col("avg_salary") > 50000)
```

### Window Functions

```
    Window Function Partitioning:
    
    ┌────────────────────────────────────┐
    │       Full DataFrame               │
    └────────────┬───────────────────────┘
                 │
         Partition By
                 │
    ┌────────────┼────────────┐
    │            │            │
    ↓            ↓            ↓
┌─────────┐  ┌─────────┐  ┌─────────┐
│ Dept A  │  │ Dept B  │  │ Dept C  │
│ Window  │  │ Window  │  │ Window  │
└────┬────┘  └────┬────┘  └────┬────┘
     │            │            │
  Order By     Order By     Order By
     │            │            │
     ↓            ↓            ↓
  Ranking     Ranking     Ranking
```

```python
from pyspark.sql import Window
from pyspark.sql.functions import (
    row_number, rank, dense_rank, percent_rank,
    lag, lead, first, last,
    sum, avg, max, min, count
)

# Define window specifications
# Partition by department, order by salary descending
windowSpec = Window.partitionBy("department").orderBy(col("salary").desc())

# Ranking functions
df.withColumn("row_num", row_number().over(windowSpec))
df.withColumn("rank", rank().over(windowSpec))
df.withColumn("dense_rank", dense_rank().over(windowSpec))
df.withColumn("percent_rank", percent_rank().over(windowSpec))

# Analytic functions
df.withColumn("prev_salary", lag("salary", 1).over(windowSpec))
df.withColumn("next_salary", lead("salary", 1).over(windowSpec))
df.withColumn("first_salary", first("salary").over(windowSpec))
df.withColumn("last_salary", last("salary").over(windowSpec))

# Aggregate window functions
df.withColumn("dept_total_salary", sum("salary").over(windowSpec))
df.withColumn("dept_avg_salary", avg("salary").over(windowSpec))
df.withColumn("dept_max_salary", max("salary").over(windowSpec))

# Running totals
runningTotalWindow = Window \
    .partitionBy("department") \
    .orderBy("date") \
    .rowsBetween(Window.unboundedPreceding, Window.currentRow)

df.withColumn("running_total", sum("amount").over(runningTotalWindow))

# Moving average (last 3 rows)
movingAvgWindow = Window \
    .partitionBy("department") \
    .orderBy("date") \
    .rowsBetween(-2, 0)  # 2 rows before and current row

df.withColumn("moving_avg", avg("value").over(movingAvgWindow))

# Range-based window (value-based, not row-based)
rangeWindow = Window \
    .partitionBy("department") \
    .orderBy("salary") \
    .rangeBetween(-1000, 1000)  # Within 1000 of current salary

df.withColumn("peers_count", count("*").over(rangeWindow))

# No partitioning (entire dataset)
globalWindow = Window.orderBy("date")
df.withColumn("global_rank", rank().over(globalWindow))

# Top N per group using window functions
windowSpec = Window.partitionBy("department").orderBy(col("salary").desc())
top_earners = df.withColumn("rank", rank().over(windowSpec)) \
    .filter(col("rank") <= 3)
```

### SQL Queries

Register DataFrame as a temporary view:

```python
df.createOrReplaceTempView("my_table")
```

Run SQL queries:

```python
result_df = spark.sql("SELECT Name, Age FROM my_table WHERE Age > 25")
result_df.show()
```

## RDDs (Resilient Distributed Datasets)

### RDD vs DataFrame

```
    ┌─────────────────────────────────────────┐
    │                RDD                      │
    │  • Low-level API                        │
    │  • No schema                            │
    │  • Type-safe (in Scala)                 │
    │  • No optimization                      │
    │  • Manual optimization needed           │
    └─────────────────────────────────────────┘
                    VS
    ┌─────────────────────────────────────────┐
    │             DataFrame                   │
    │  • High-level API                       │
    │  • Schema-based                         │
    │  • Catalyst optimizer                   │
    │  • Tungsten execution                   │
    │  • Better performance                   │
    └─────────────────────────────────────────┘
```

### Creating RDDs

```python
# From Python collection
data = [1, 2, 3, 4, 5]
rdd = sc.parallelize(data)

# With specific number of partitions
rdd = sc.parallelize(data, numSlices=4)

# From text file
rdd = sc.textFile("path/to/file.txt")
rdd = sc.textFile("path/to/directory/*.txt")

# From whole text files (returns filename, content pairs)
rdd = sc.wholeTextFiles("path/to/directory")

# From sequence file
rdd = sc.sequenceFile("path/to/file")

# Empty RDD
empty_rdd = sc.emptyRDD()

# From range
rdd = sc.parallelize(range(1, 100))
```

### RDD Transformations (Lazy)

```
    Transformation Pipeline:
    
    ┌──────────┐
    │   RDD    │
    └────┬─────┘
         │
         ↓ map()
    ┌────────────┐
    │ Transformed│
    └────┬───────┘
         │
         ↓ filter()
    ┌────────────┐
    │  Filtered  │
    └────┬───────┘
         │
         ↓ reduce() (Action)
    ┌────────────┐
    │  Execute   │
    │  Pipeline  │
    └────────────┘
```

Basic transformations:

```python
# Map: Apply function to each element
rdd.map(lambda x: x * 2)
rdd.map(lambda x: (x, 1))  # Create pairs

# FlatMap: Map and flatten results
rdd.flatMap(lambda x: x.split())
rdd.flatMap(lambda x: range(x))

# Filter: Keep elements matching condition
rdd.filter(lambda x: x > 2)
rdd.filter(lambda x: x % 2 == 0)

# MapPartitions: Apply function to each partition
def process_partition(iterator):
    yield sum(iterator)
rdd.mapPartitions(process_partition)

# MapPartitionsWithIndex: Include partition index
def process_with_index(index, iterator):
    yield (index, sum(iterator))
rdd.mapPartitionsWithIndex(process_with_index)

# Distinct: Remove duplicates
rdd.distinct()

# Sample: Random sample
rdd.sample(withReplacement=False, fraction=0.5, seed=42)

# Union: Combine RDDs
rdd1.union(rdd2)

# Intersection: Common elements
rdd1.intersection(rdd2)

# Subtract: Elements in rdd1 but not rdd2
rdd1.subtract(rdd2)

# Cartesian: Cartesian product
rdd1.cartesian(rdd2)
```

Sorting and partitioning:

```python
# Sort by value
rdd.sortBy(lambda x: x, ascending=True)

# Repartition: Change number of partitions (shuffle)
rdd.repartition(10)

# Coalesce: Reduce partitions (no shuffle)
rdd.coalesce(1)

# Partition by custom function
rdd.partitionBy(numPartitions=10, partitionFunc=lambda x: x % 10)

# Zip: Combine element-wise
rdd1.zip(rdd2)  # Same partitioning required

# ZipWithIndex: Add indices
rdd.zipWithIndex()

# ZipWithUniqueId: Add unique IDs
rdd.zipWithUniqueId()
```

### RDD Actions (Trigger Computation)

```python
# Collect: Return all elements (use carefully!)
rdd.collect()

# Count: Number of elements
rdd.count()

# First: First element
rdd.first()

# Take: First N elements
rdd.take(5)

# Top: Top N elements
rdd.top(5)

# TakeOrdered: N smallest elements
rdd.takeOrdered(5)
rdd.takeOrdered(5, key=lambda x: -x)  # Largest

# TakeSample: Random sample
rdd.takeSample(withReplacement=False, num=5, seed=42)

# Reduce: Aggregate elements
rdd.reduce(lambda x, y: x + y)
rdd.reduce(lambda x, y: x if x > y else y)  # Max

# Fold: Like reduce with initial value
rdd.fold(0, lambda x, y: x + y)

# Aggregate: Different types for accumulator and elements
rdd.aggregate(
    (0, 0),  # Initial value
    lambda acc, value: (acc[0] + value, acc[1] + 1),  # Seq op
    lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1])  # Comb op
)

# Foreach: Apply function to each element (no return)
rdd.foreach(lambda x: print(x))

# ForeachPartition: Apply function to each partition
def print_partition(iterator):
    for item in iterator:
        print(item)
rdd.foreachPartition(print_partition)

# CountByValue: Count occurrences of each value
rdd.countByValue()

# SaveAsTextFile: Save to files
rdd.saveAsTextFile("path/to/output")

# SaveAsSequenceFile: Save as Hadoop SequenceFile
rdd.saveAsSequenceFile("path/to/output")
```

### Pair RDD Operations

```python
# Create pair RDD
words = sc.parallelize(["hello", "world", "hello"])
pairs = words.map(lambda x: (x, 1))

# ReduceByKey: Combine values for each key
word_counts = pairs.reduceByKey(lambda x, y: x + y)

# GroupByKey: Group all values for each key (avoid if possible)
pairs.groupByKey().mapValues(list)

# AggregateByKey: Aggregate with initial value
pairs.aggregateByKey(
    0,  # Initial value
    lambda acc, value: acc + value,  # Seq func
    lambda acc1, acc2: acc1 + acc2   # Comb func
)

# FoldByKey: Fold with initial value
pairs.foldByKey(0, lambda x, y: x + y)

# CombineByKey: Most general aggregation
def create_combiner(x):
    return (x, 1)

def merge_value(acc, x):
    return (acc[0] + x, acc[1] + 1)

def merge_combiners(acc1, acc2):
    return (acc1[0] + acc2[0], acc1[1] + acc2[1])

pairs.combineByKey(create_combiner, merge_value, merge_combiners)

# MapValues: Transform only values
pairs.mapValues(lambda x: x * 2)

# FlatMapValues: FlatMap only values
pairs.flatMapValues(lambda x: range(x))

# Keys: Get all keys
pairs.keys()

# Values: Get all values
pairs.values()

# SortByKey: Sort by key
pairs.sortByKey(ascending=True)

# Join: Inner join
rdd1 = sc.parallelize([("a", 1), ("b", 2)])
rdd2 = sc.parallelize([("a", 3), ("c", 4)])
rdd1.join(rdd2)  # Result: [("a", (1, 3))]

# LeftOuterJoin
rdd1.leftOuterJoin(rdd2)  # Result: [("a", (1, Some(3))), ("b", (2, None))]

# RightOuterJoin
rdd1.rightOuterJoin(rdd2)  # Result: [("a", (Some(1), 3)), ("c", (None, 4))]

# FullOuterJoin
rdd1.fullOuterJoin(rdd2)

# Cogroup: Group from multiple RDDs
rdd1.cogroup(rdd2)

# Pair RDD actions
pairs.countByKey()        # Count values per key
pairs.collectAsMap()      # Collect as dictionary
pairs.lookup("hello")     # Get all values for key
```

### RDD Persistence

```python
from pyspark import StorageLevel

# Cache in memory
rdd.cache()  # Same as persist(StorageLevel.MEMORY_ONLY)

# Persist with storage level
rdd.persist(StorageLevel.MEMORY_AND_DISK)
rdd.persist(StorageLevel.MEMORY_ONLY)
rdd.persist(StorageLevel.DISK_ONLY)
rdd.persist(StorageLevel.MEMORY_AND_DISK_SER)  # Serialized

# Check if persisted
rdd.is_cached

# Unpersist
rdd.unpersist()

# Get number of partitions
rdd.getNumPartitions()

# Checkpoint (for lineage truncation)
sc.setCheckpointDir("path/to/checkpoint")
rdd.checkpoint()
```

## Writing Data

### Write Modes

```
    Write Mode Decision Tree:
    
    ┌──────────────────┐
    │  Target Exists?  │
    └────────┬─────────┘
             │
        ┌────┴────┐
        ↓         ↓
      Yes        No
        │         │
        ↓         ↓
    ┌───────┐  ┌────────┐
    │ Mode? │  │ Create │
    └───┬───┘  └────────┘
        │
    ┌───┴────────────────┐
    │                    │
    ↓                    ↓
┌──────────┐      ┌────────────┐
│overwrite │      │   append   │
└──────────┘      └────────────┘
    │                    │
    ↓                    ↓
┌──────────┐      ┌────────────┐
│ Replace  │      │  Add New   │
│   All    │      │   Rows     │
└──────────┘      └────────────┘
    
    ↓ error         ↓ ignore
┌──────────┐   ┌────────────┐
│  Throw   │   │  Skip if   │
│  Error   │   │  Exists    │
└──────────┘   └────────────┘
```

### Writing DataFrames

CSV files:

```python
# Basic write
df.write.csv("path/to/output", header=True, mode="overwrite")

# With options
df.write \
    .format("csv") \
    .option("header", "true") \
    .option("sep", ",") \
    .option("quote", '"') \
    .option("escape", "\\") \
    .option("compression", "gzip") \
    .option("dateFormat", "yyyy-MM-dd") \
    .mode("overwrite") \
    .save("path/to/output")

# Single file output
df.coalesce(1).write.csv("path/to/output", header=True)
```

JSON files:

```python
# Basic write
df.write.json("path/to/output", mode="overwrite")

# With options
df.write \
    .format("json") \
    .option("compression", "gzip") \
    .option("dateFormat", "yyyy-MM-dd") \
    .mode("overwrite") \
    .save("path/to/output")
```

Parquet files (recommended for Spark):

```python
# Basic write
df.write.parquet("path/to/output", mode="overwrite")

# With compression
df.write \
    .format("parquet") \
    .option("compression", "snappy")  # gzip, snappy, lzo, none
    .mode("overwrite") \
    .save("path/to/output")

# Partitioned write
df.write \
    .partitionBy("year", "month") \
    .parquet("path/to/output", mode="overwrite")

# With bucketing
df.write \
    .bucketBy(10, "user_id") \
    .sortBy("timestamp") \
    .saveAsTable("user_events")
```

ORC files:

```python
# Basic write
df.write.orc("path/to/output", mode="overwrite")

# With compression
df.write \
    .format("orc") \
    .option("compression", "snappy") \
    .mode("overwrite") \
    .save("path/to/output")
```

Avro files:

```python
# Basic write
df.write.format("avro").save("path/to/output", mode="overwrite")

# With compression
df.write \
    .format("avro") \
    .option("compression", "snappy") \
    .save("path/to/output", mode="overwrite")
```

JDBC database:

```python
# Basic write
df.write.format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5432/mydatabase") \
    .option("dbtable", "mytable") \
    .option("user", "myuser") \
    .option("password", "mypassword") \
    .option("driver", "org.postgresql.Driver") \
    .mode("overwrite") \
    .save()

# With batching for better performance
df.write.format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5432/mydatabase") \
    .option("dbtable", "mytable") \
    .option("user", "myuser") \
    .option("password", "mypassword") \
    .option("batchsize", "10000") \
    .option("isolationLevel", "NONE") \
    .mode("append") \
    .save()

# Truncate and insert
df.write.format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5432/mydatabase") \
    .option("dbtable", "mytable") \
    .option("user", "myuser") \
    .option("password", "mypassword") \
    .option("truncate", "true") \
    .mode("overwrite") \
    .save()
```

Delta Lake:

```python
# Basic write
df.write.format("delta").mode("overwrite").save("path/to/delta_table")

# Append mode
df.write.format("delta").mode("append").save("path/to/delta_table")

# With partitioning
df.write \
    .format("delta") \
    .partitionBy("date") \
    .mode("overwrite") \
    .save("path/to/delta_table")

# Write as managed table
df.write.format("delta").saveAsTable("my_delta_table")

# Merge (upsert) operation
from delta.tables import DeltaTable

deltaTable = DeltaTable.forPath(spark, "path/to/delta_table")
deltaTable.alias("target").merge(
    df.alias("source"),
    "target.id = source.id"
).whenMatchedUpdate(set={
    "value": "source.value",
    "updated_at": "source.updated_at"
}).whenNotMatchedInsert(values={
    "id": "source.id",
    "value": "source.value",
    "updated_at": "source.updated_at"
}).execute()
```

Hive tables:

```python
# Save as managed table
df.write.saveAsTable("database.tablename", mode="overwrite")

# Save as external table
df.write \
    .option("path", "hdfs://path/to/data") \
    .saveAsTable("database.tablename", mode="overwrite")

# Partitioned table
df.write \
    .partitionBy("year", "month") \
    .saveAsTable("database.tablename", mode="overwrite")

# Insert into existing table
df.write.insertInto("database.tablename", overwrite=True)
```

### Write Modes

```python
# Overwrite: Replace existing data
df.write.mode("overwrite").parquet("path/to/output")

# Append: Add to existing data
df.write.mode("append").parquet("path/to/output")

# Error (default): Throw error if exists
df.write.mode("error").parquet("path/to/output")
df.write.mode("errorifexists").parquet("path/to/output")

# Ignore: Skip if exists
df.write.mode("ignore").parquet("path/to/output")
```

### Writing RDDs

```python
# Save as text file
rdd.saveAsTextFile("path/to/output")

# With compression
rdd.saveAsTextFile("path/to/output", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

# Save as pickle file
rdd.saveAsPickleFile("path/to/output")

# Save as sequence file (Hadoop format)
rdd.saveAsSequenceFile("path/to/output")

# Save as Hadoop file with custom format
rdd.saveAsHadoopFile(
    "path/to/output",
    "org.apache.hadoop.mapred.TextOutputFormat"
)

# Convert to DataFrame and write
rdd.toDF(["column1", "column2"]).write.parquet("path/to/output")
```

### Partitioning Strategies

```python
# Partition by column values
df.write \
    .partitionBy("year", "month", "day") \
    .parquet("path/to/output")

# Control number of output files
df.repartition(10).write.parquet("path/to/output")

# Single output file (use carefully!)
df.coalesce(1).write.parquet("path/to/output")

# Partition and control file size
df.repartition("year", "month") \
    .write \
    .partitionBy("year", "month") \
    .parquet("path/to/output")
```

## Spark SQL

### Creating Tables

From DataFrame:

```python
df.write.saveAsTable("my_table")
```

Using SQL:

```python
spark.sql("CREATE TABLE my_table (name STRING, age INT) USING parquet")
```

### Inserting Data

From DataFrame:

```python
df.write.insertInto("my_table")
```

Using SQL:

```python
spark.sql("INSERT INTO my_table VALUES ('Alice', 30)")
```

### Selecting Data

```python
spark.sql("SELECT * FROM my_table").show()
```

### Filtering Data

```python
spark.sql("SELECT * FROM my_table WHERE age > 25").show()
```

### Aggregating Data

```python
spark.sql("SELECT name, AVG(age) FROM my_table GROUP BY name").show()
```

### Joining Tables

```python
spark.sql("SELECT * FROM table1 JOIN table2 ON table1.key = table2.key").show()
```

### Window Functions in SQL

```python
spark.sql("""
SELECT
    name,
    age,
    RANK() OVER (ORDER BY age DESC) as age_rank
FROM
    my_table
""").show()
```

## Spark MLlib

### ML Pipeline Flow

```
    ┌───────────────┐
    │   Raw Data    │
    └───────┬───────┘
            │
            ↓
    ┌───────────────┐
    │ StringIndexer │  (Categorical → Numeric)
    └───────┬───────┘
            │
            ↓
    ┌───────────────┐
    │VectorAssembler│  (Combine Features)
    └───────┬───────┘
            │
            ↓
    ┌───────────────┐
    │ StandardScaler│  (Normalize)
    └───────┬───────┘
            │
            ↓
    ┌───────────────┐
    │  Train/Test   │
    │     Split     │
    └───────┬───────┘
            │
        ┌───┴───┐
        ↓       ↓
    ┌──────┐ ┌──────┐
    │Train │ │ Test │
    └──┬───┘ └───┬──┘
       │         │
       ↓         │
    ┌──────┐    │
    │Model │    │
    └──┬───┘    │
       │        │
       └────────┴──→ Predictions
```

### Data Preparation

Feature transformers:

```python
from pyspark.ml.feature import (
    StringIndexer, VectorAssembler, StandardScaler,
    MinMaxScaler, Normalizer, OneHotEncoder,
    Tokenizer, StopWordsRemover, CountVectorizer,
    IDF, PCA, Imputer
)

# StringIndexer: Convert categories to indices
indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
indexed_df = indexer.fit(df).transform(df)

# Handle unseen labels
indexer = StringIndexer(
    inputCol="category",
    outputCol="categoryIndex",
    handleInvalid="keep"  # "skip" or "error"
)

# OneHotEncoder: Convert indices to binary vectors
encoder = OneHotEncoder(inputCols=["categoryIndex"], outputCols=["categoryVec"])
encoded_df = encoder.fit(indexed_df).transform(indexed_df)

# VectorAssembler: Combine features into vector
assembler = VectorAssembler(
    inputCols=["feature1", "feature2", "feature3"],
    outputCol="features",
    handleInvalid="skip"  # "keep" or "error"
)
output_df = assembler.transform(df)

# Imputer: Fill missing values
imputer = Imputer(
    inputCols=["age", "income"],
    outputCols=["age_imputed", "income_imputed"],
    strategy="mean"  # "median" or "mode"
)
imputer_model = imputer.fit(df)
imputed_df = imputer_model.transform(df)

# StandardScaler: Standardize features (mean=0, std=1)
scaler = StandardScaler(
    inputCol="features",
    outputCol="scaledFeatures",
    withStd=True,
    withMean=True
)
scaler_model = scaler.fit(df)
scaled_df = scaler_model.transform(df)

# MinMaxScaler: Scale to range [0, 1]
min_max_scaler = MinMaxScaler(
    inputCol="features",
    outputCol="scaledFeatures",
    min=0.0,
    max=1.0
)
min_max_model = min_max_scaler.fit(df)
scaled_df = min_max_model.transform(df)

# Normalizer: Normalize to unit norm
normalizer = Normalizer(
    inputCol="features",
    outputCol="normFeatures",
    p=2.0  # L2 norm (p=1.0 for L1)
)
normalized_df = normalizer.transform(df)

# PCA: Dimensionality reduction
pca = PCA(
    k=3,  # Number of principal components
    inputCol="features",
    outputCol="pcaFeatures"
)
pca_model = pca.fit(df)
pca_df = pca_model.transform(df)
```

Text processing:

```python
from pyspark.ml.feature import (
    Tokenizer, RegexTokenizer, StopWordsRemover,
    CountVectorizer, HashingTF, IDF, Word2Vec, NGram
)

# Tokenizer: Split text into words
tokenizer = Tokenizer(inputCol="text", outputCol="words")
tokenized_df = tokenizer.transform(df)

# RegexTokenizer: Tokenize with regex
regex_tokenizer = RegexTokenizer(
    inputCol="text",
    outputCol="words",
    pattern="\\W"  # Split on non-word characters
)

# StopWordsRemover: Remove common words
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
removed_df = remover.transform(tokenized_df)

# CountVectorizer: Bag of words
cv = CountVectorizer(
    inputCol="filtered",
    outputCol="rawFeatures",
    vocabSize=10000,
    minDF=2.0  # Minimum document frequency
)
cv_model = cv.fit(removed_df)
cv_df = cv_model.transform(removed_df)

# TF-IDF: Term frequency - inverse document frequency
# First HashingTF
from pyspark.ml.feature import HashingTF
hashing_tf = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
tf_df = hashing_tf.transform(removed_df)

# Then IDF
idf = IDF(inputCol="rawFeatures", outputCol="features")
idf_model = idf.fit(tf_df)
tfidf_df = idf_model.transform(tf_df)

# Word2Vec: Word embeddings
word2vec = Word2Vec(
    vectorSize=100,
    minCount=5,
    inputCol="filtered",
    outputCol="features"
)
w2v_model = word2vec.fit(removed_df)
w2v_df = w2v_model.transform(removed_df)

# NGram: Generate n-grams
ngram = NGram(n=2, inputCol="words", outputCol="ngrams")
ngram_df = ngram.transform(tokenized_df)
```

### Feature Extraction

*   `Tokenizer`: Splits strings into words.
*   `StopWordsRemover`: Removes stop words.
*   `CountVectorizer`: Converts text documents to vectors of term counts.
*   `IDF`: Computes Inverse Document Frequency.
*   `Word2Vec`: Learns vector representations of words.
*   `NGram`: Generates n-grams from input sequences.

### Feature Scaling

*   `StandardScaler`: Standardizes features by removing the mean and scaling to unit variance.
*   `MinMaxScaler`: Transforms features by scaling each feature to a given range.
*   `MaxAbsScaler`: Scales each feature to the [-1, 1] range by dividing through the largest maximum absolute value in each feature.
*   `Normalizer`: Normalizes each sample to unit norm.

### Feature Selection

*   `VectorSlicer`: Creates a new feature vector by selecting a subset of features from an existing vector.
*   `RFormula`: Implements the R formula string syntax for selecting features.
*   `PCA`: Reduces the dimensionality of feature vectors using Principal Component Analysis.

### Classification

```
    Classification Model Selection:
    
    ┌──────────────────┐
    │  Problem Type?   │
    └────────┬─────────┘
             │
    ┌────────┴────────┐
    │                 │
    ↓                 ↓
┌─────────┐      ┌─────────┐
│ Binary  │      │  Multi  │
│  Class  │      │  Class  │
└────┬────┘      └────┬────┘
     │                │
     ↓                ↓
┌──────────────────────────┐
│ Linear Separable?        │
└────────┬─────────────────┘
         │
    ┌────┴─────┐
    ↓          ↓
  Yes         No
    │          │
    ↓          ↓
┌────────┐  ┌──────────┐
│Logistic│  │Random    │
│Regress │  │Forest/   │
│        │  │GBT/SVM   │
└────────┘  └──────────┘
```

Logistic Regression:

```python
from pyspark.ml.classification import LogisticRegression

# Binary classification
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=0.01,        # Regularization parameter
    elasticNetParam=0.5,  # 0=L2, 1=L1, 0.5=mix
    threshold=0.5         # Classification threshold
)
model = lr.fit(training_data)
predictions = model.transform(test_data)

# Multi-class classification
lr_multi = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    family="multinomial"
)

# Access model parameters
print(f"Coefficients: {model.coefficients}")
print(f"Intercept: {model.intercept}")

# Summary
training_summary = model.summary
print(f"Accuracy: {training_summary.accuracy}")
print(f"Area under ROC: {training_summary.areaUnderROC}")
```

Decision Tree:

```python
from pyspark.ml.classification import DecisionTreeClassifier

dt = DecisionTreeClassifier(
    featuresCol="features",
    labelCol="label",
    maxDepth=5,           # Maximum depth
    maxBins=32,           # Number of bins for continuous features
    impurity="gini",      # "gini" or "entropy"
    minInstancesPerNode=1 # Minimum instances per node
)
model = dt.fit(training_data)
predictions = model.transform(test_data)

# Feature importance
print(f"Feature importances: {model.featureImportances}")

# Tree structure
print(model.toDebugString)
```

Random Forest:

```python
from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=100,         # Number of trees
    maxDepth=5,
    maxBins=32,
    subsamplingRate=1.0,  # Fraction of training data
    featureSubsetStrategy="auto",  # "all", "sqrt", "log2", "onethird"
    seed=42
)
model = rf.fit(training_data)
predictions = model.transform(test_data)

# Feature importance
print(f"Feature importances: {model.featureImportances}")

# Access individual trees
print(f"Number of trees: {model.numTrees}")
print(f"Total nodes: {model.totalNumNodes}")
```

Gradient-Boosted Trees (GBT):

```python
from pyspark.ml.classification import GBTClassifier

gbt = GBTClassifier(
    featuresCol="features",
    labelCol="label",
    maxIter=20,           # Number of boosting iterations
    maxDepth=5,
    stepSize=0.1,         # Learning rate
    subsamplingRate=1.0,
    featureSubsetStrategy="all"
)
model = gbt.fit(training_data)
predictions = model.transform(test_data)

# Feature importance
print(f"Feature importances: {model.featureImportances}")
```

Naive Bayes:

```python
from pyspark.ml.classification import NaiveBayes

nb = NaiveBayes(
    featuresCol="features",
    labelCol="label",
    smoothing=1.0,        # Laplace smoothing
    modelType="multinomial"  # "multinomial", "bernoulli", "gaussian"
)
model = nb.fit(training_data)
predictions = model.transform(test_data)
```

Linear Support Vector Machine:

```python
from pyspark.ml.classification import LinearSVC

svm = LinearSVC(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=0.01,
    threshold=0.0
)
model = svm.fit(training_data)
predictions = model.transform(test_data)
```

Multilayer Perceptron:

```python
from pyspark.ml.classification import MultilayerPerceptronClassifier

# Define network architecture
# [input_size, hidden_layer1, hidden_layer2, ..., output_size]
layers = [10, 20, 15, 3]  # 10 features, 2 hidden layers, 3 classes

mlp = MultilayerPerceptronClassifier(
    layers=layers,
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    blockSize=128,        # Block size for stacking input data
    seed=42
)
model = mlp.fit(training_data)
predictions = model.transform(test_data)
```

### Regression

Linear Regression:

```python
from pyspark.ml.regression import LinearRegression

lr = LinearRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=0.01,        # L2 regularization
    elasticNetParam=0.0,  # 0=L2 (Ridge), 1=L1 (Lasso)
    standardization=True,
    fitIntercept=True
)
model = lr.fit(training_data)
predictions = model.transform(test_data)

# Model statistics
print(f"Coefficients: {model.coefficients}")
print(f"Intercept: {model.intercept}")
print(f"RMSE: {model.summary.rootMeanSquaredError}")
print(f"R2: {model.summary.r2}")
print(f"Mean Absolute Error: {model.summary.meanAbsoluteError}")
```

Decision Tree Regression:

```python
from pyspark.ml.regression import DecisionTreeRegressor

dt = DecisionTreeRegressor(
    featuresCol="features",
    labelCol="label",
    maxDepth=5,
    maxBins=32,
    minInstancesPerNode=1,
    varianceCol="variance"  # Output variance column
)
model = dt.fit(training_data)
predictions = model.transform(test_data)

# Feature importance
print(f"Feature importances: {model.featureImportances}")
```

Random Forest Regression:

```python
from pyspark.ml.regression import RandomForestRegressor

rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="label",
    numTrees=100,
    maxDepth=5,
    maxBins=32,
    subsamplingRate=1.0,
    featureSubsetStrategy="auto",
    seed=42
)
model = rf.fit(training_data)
predictions = model.transform(test_data)

# Feature importance
print(f"Feature importances: {model.featureImportances}")
```

Gradient-Boosted Trees Regression:

```python
from pyspark.ml.regression import GBTRegressor

gbt = GBTRegressor(
    featuresCol="features",
    labelCol="label",
    maxIter=20,
    maxDepth=5,
    stepSize=0.1,
    lossType="squared",   # "squared", "absolute"
    subsamplingRate=1.0
)
model = gbt.fit(training_data)
predictions = model.transform(test_data)

# Feature importance
print(f"Feature importances: {model.featureImportances}")
```

Generalized Linear Regression:

```python
from pyspark.ml.regression import GeneralizedLinearRegression

glr = GeneralizedLinearRegression(
    featuresCol="features",
    labelCol="label",
    family="gaussian",     # "gaussian", "binomial", "poisson", "gamma"
    link="identity",       # Link function
    maxIter=100,
    regParam=0.01
)
model = glr.fit(training_data)
predictions = model.transform(test_data)
```

Isotonic Regression:

```python
from pyspark.ml.regression import IsotonicRegression

iso = IsotonicRegression(
    featuresCol="features",
    labelCol="label",
    isotonic=True  # True for ascending, False for descending
)
model = iso.fit(training_data)
predictions = model.transform(test_data)
```

### Clustering

K-Means:

```python
from pyspark.ml.clustering import KMeans

kmeans = KMeans(
    featuresCol="features",
    predictionCol="cluster",
    k=3,                  # Number of clusters
    maxIter=20,
    seed=42,
    initMode="k-means||"  # "k-means||" or "random"
)
model = kmeans.fit(data)
predictions = model.transform(data)

# Cluster centers
centers = model.clusterCenters()
print(f"Cluster Centers: {centers}")

# Within set sum of squared errors
wssse = model.summary.trainingCost
print(f"WSSSE: {wssse}")

# Cluster sizes
cluster_sizes = predictions.groupBy("cluster").count().show()
```

Bisecting K-Means:

```python
from pyspark.ml.clustering import BisectingKMeans

bkm = BisectingKMeans(
    featuresCol="features",
    k=3,
    maxIter=20,
    minDivisibleClusterSize=1.0
)
model = bkm.fit(data)
predictions = model.transform(data)
```

Gaussian Mixture Model:

```python
from pyspark.ml.clustering import GaussianMixture

gmm = GaussianMixture(
    featuresCol="features",
    k=3,                  # Number of mixture components
    maxIter=100,
    seed=42,
    probabilityCol="probability",
    predictionCol="cluster"
)
model = gmm.fit(data)
predictions = model.transform(data)

# Mixture weights
print(f"Weights: {model.weights}")

# Gaussian parameters
print(f"Gaussians: {model.gaussiansDF.show()}")
```

Latent Dirichlet Allocation (LDA):

```python
from pyspark.ml.clustering import LDA

lda = LDA(
    featuresCol="features",
    k=10,                 # Number of topics
    maxIter=100,
    optimizer="online",   # "online" or "em"
    learningOffset=1024.0,
    learningDecay=0.51
)
model = lda.fit(data)

# Topics
topics = model.describeTopics(maxTermsPerTopic=10)
topics.show(truncate=False)

# Transform documents
transformed = model.transform(data)
```

### Recommendation

```
    Collaborative Filtering Flow:
    
    ┌────────────────────┐
    │ User-Item Matrix   │
    │  (Sparse Ratings)  │
    └──────────┬─────────┘
               │
               ↓
    ┌──────────────────────┐
    │  Matrix Factorization│
    │        (ALS)          │
    └──────┬────────┬───────┘
           │        │
      ┌────┘        └────┐
      ↓                  ↓
┌──────────┐      ┌──────────┐
│   User   │      │   Item   │
│ Factors  │      │ Factors  │
└─────┬────┘      └────┬─────┘
      │                │
      └────────┬───────┘
               │
               ↓ (Dot Product)
    ┌──────────────────────┐
    │  Predicted Ratings   │
    └──────────────────────┘
```

Alternating Least Squares (ALS):

```python
from pyspark.ml.recommendation import ALS

als = ALS(
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    maxIter=10,
    regParam=0.01,        # Regularization parameter
    rank=10,              # Number of latent factors
    coldStartStrategy="drop",  # "drop" or "nan"
    nonnegative=False,    # Force non-negative factors
    implicitPrefs=False,  # True for implicit feedback
    alpha=1.0             # Confidence parameter (implicit)
)
model = als.fit(training_data)
predictions = model.transform(test_data)

# Get user factors
user_factors = model.userFactors
user_factors.show()

# Get item factors
item_factors = model.itemFactors
item_factors.show()

# Recommend top N items for all users
user_recs = model.recommendForAllUsers(10)
user_recs.show(truncate=False)

# Recommend top N users for all items
item_recs = model.recommendForAllItems(10)

# Recommend for specific users
specific_users = training_data.select("userId").distinct().limit(3)
user_subset_recs = model.recommendForUserSubset(specific_users, 10)

# Handle cold start
predictions_with_nan = model.setColdStartStrategy("nan").transform(test_data)
```

### Evaluation

Binary Classification:

```python
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# ROC-AUC
evaluator_roc = BinaryClassificationEvaluator(
    rawPredictionCol="rawPrediction",
    labelCol="label",
    metricName="areaUnderROC"
)
auc = evaluator_roc.evaluate(predictions)
print(f"Area Under ROC: {auc}")

# PR-AUC
evaluator_pr = BinaryClassificationEvaluator(
    rawPredictionCol="rawPrediction",
    labelCol="label",
    metricName="areaUnderPR"
)
pr_auc = evaluator_pr.evaluate(predictions)
print(f"Area Under PR: {pr_auc}")
```

Multiclass Classification:

```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Accuracy
accuracy_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)
accuracy = accuracy_eval.evaluate(predictions)
print(f"Accuracy: {accuracy}")

# F1 Score
f1_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)
f1 = f1_eval.evaluate(predictions)
print(f"F1 Score: {f1}")

# Weighted Precision
precision_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedPrecision"
)
precision = precision_eval.evaluate(predictions)

# Weighted Recall
recall_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedRecall"
)
recall = recall_eval.evaluate(predictions)

# Confusion Matrix
from pyspark.mllib.evaluation import MulticlassMetrics
metrics = MulticlassMetrics(predictions.select("prediction", "label").rdd)
confusion_matrix = metrics.confusionMatrix()
print(f"Confusion Matrix:\n{confusion_matrix}")
```

Regression:

```python
from pyspark.ml.evaluation import RegressionEvaluator

# RMSE
rmse_eval = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="rmse"
)
rmse = rmse_eval.evaluate(predictions)
print(f"RMSE: {rmse}")

# MSE
mse_eval = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="mse"
)
mse = mse_eval.evaluate(predictions)

# MAE
mae_eval = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="mae"
)
mae = mae_eval.evaluate(predictions)

# R2
r2_eval = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="r2"
)
r2 = r2_eval.evaluate(predictions)
print(f"R2: {r2}")
```

Clustering:

```python
from pyspark.ml.evaluation import ClusteringEvaluator

# Silhouette Score
evaluator = ClusteringEvaluator(
    featuresCol="features",
    predictionCol="prediction",
    metricName="silhouette",
    distanceMeasure="squaredEuclidean"  # or "cosine"
)
silhouette = evaluator.evaluate(predictions)
print(f"Silhouette Score: {silhouette}")
```

Ranking (Recommendation):

```python
from pyspark.ml.evaluation import RankingEvaluator

# Precision at K
precision_eval = RankingEvaluator(
    predictionCol="recommendations",
    labelCol="items",
    metricName="precisionAtK",
    k=10
)
precision_at_k = precision_eval.evaluate(predictions)

# Mean Average Precision
map_eval = RankingEvaluator(
    predictionCol="recommendations",
    labelCol="items",
    metricName="meanAveragePrecision"
)
map_score = map_eval.evaluate(predictions)

# NDCG
ndcg_eval = RankingEvaluator(
    predictionCol="recommendations",
    labelCol="items",
    metricName="ndcgAtK",
    k=10
)
ndcg = ndcg_eval.evaluate(predictions)
```

### Cross-Validation and Hyperparameter Tuning

```
    Cross-Validation Flow:
    
    ┌──────────────┐
    │ Training Data│
    └──────┬───────┘
           │
           ↓ Split into K folds
    ┌──────────────────┐
    │ Fold1│Fold2│...  │
    └──────┬───────────┘
           │
           ↓ For each param combo
    ┌──────────────────┐
    │ Train on K-1     │
    │ Validate on 1    │
    └──────┬───────────┘
           │
           ↓ Repeat K times
    ┌──────────────────┐
    │ Average metrics  │
    └──────┬───────────┘
           │
           ↓
    ┌──────────────────┐
    │ Best Parameters  │
    └──────────────────┘
```

K-Fold Cross-Validation:

```python
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Create estimator
lr = LogisticRegression(featuresCol="features", labelCol="label")

# Build parameter grid
paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.1, 0.01, 0.001]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .addGrid(lr.maxIter, [50, 100]) \
    .build()

# Create evaluator
evaluator = BinaryClassificationEvaluator(
    rawPredictionCol="rawPrediction",
    labelCol="label",
    metricName="areaUnderROC"
)

# Cross validator
crossval = CrossValidator(
    estimator=lr,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=5,           # Number of folds
    seed=42,
    parallelism=2         # Number of parallel fits
)

# Fit and get best model
cvModel = crossval.fit(training_data)
best_model = cvModel.bestModel

# Make predictions
predictions = cvModel.transform(test_data)

# Get best parameters
print(f"Best Params: {best_model.extractParamMap()}")

# Get average metrics per parameter combination
avg_metrics = cvModel.avgMetrics
print(f"Average Metrics: {avg_metrics}")
```

Train-Validation Split:

```python
from pyspark.ml.tuning import TrainValidationSplit

# Faster than CrossValidator (single split)
tvs = TrainValidationSplit(
    estimator=lr,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    trainRatio=0.8,       # 80% for training, 20% for validation
    seed=42,
    parallelism=2
)

tvsModel = tvs.fit(training_data)
best_model = tvsModel.bestModel
predictions = tvsModel.transform(test_data)
```

### ML Pipelines

```
    Pipeline Architecture:
    
    ┌──────────────┐
    │   Raw Data   │
    └──────┬───────┘
           │
           ↓ Stage 1
    ┌──────────────┐
    │  Transformer │
    └──────┬───────┘
           │
           ↓ Stage 2
    ┌──────────────┐
    │  Transformer │
    └──────┬───────┘
           │
           ↓ Stage 3 (Final)
    ┌──────────────┐
    │  Estimator   │
    │  (Model)     │
    └──────┬───────┘
           │
           ↓
    ┌──────────────┐
    │ Predictions  │
    └──────────────┘
```

Basic Pipeline:

```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression

# Stage 1: Index categorical column
indexer = StringIndexer(
    inputCol="category",
    outputCol="categoryIndex",
    handleInvalid="keep"
)

# Stage 2: Assemble features
assembler = VectorAssembler(
    inputCols=["categoryIndex", "feature1", "feature2"],
    outputCol="features",
    handleInvalid="skip"
)

# Stage 3: Scale features
scaler = StandardScaler(
    inputCol="features",
    outputCol="scaledFeatures",
    withStd=True,
    withMean=True
)

# Stage 4: Train model
lr = LogisticRegression(
    featuresCol="scaledFeatures",
    labelCol="label",
    maxIter=100
)

# Create pipeline
pipeline = Pipeline(stages=[indexer, assembler, scaler, lr])

# Fit pipeline
model = pipeline.fit(training_data)

# Transform test data
predictions = model.transform(test_data)

# Access stages
stages = model.stages
lr_model = stages[-1]  # Get the trained LogisticRegression model
```

Pipeline with Cross-Validation:

```python
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Create pipeline
pipeline = Pipeline(stages=[indexer, assembler, scaler, lr])

# Parameter grid for the entire pipeline
paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .addGrid(scaler.withMean, [True, False]) \
    .build()

# Cross-validation
crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=3
)

# Fit and get best pipeline
cvModel = crossval.fit(training_data)
predictions = cvModel.transform(test_data)
```

### Model Persistence

```python
from pyspark.ml import PipelineModel

# Save model
model.save("path/to/model")

# Load model
loaded_model = PipelineModel.load("path/to/model")

# Use loaded model
predictions = loaded_model.transform(test_data)

# Save specific estimator
lr_model.save("path/to/lr_model")

# Load specific estimator
from pyspark.ml.classification import LogisticRegressionModel
loaded_lr = LogisticRegressionModel.load("path/to/lr_model")

# Save to HDFS
model.save("hdfs://namenode:8020/models/my_model")

# Save to S3
model.save("s3a://bucket-name/models/my_model")
```

## Structured Streaming

### Streaming Architecture

```
    ┌────────────────┐
    │ Data Source    │
    │ (Kafka/Socket) │
    └────────┬───────┘
             │ Continuous
             ↓ Stream
    ┌────────────────┐
    │ Spark Streaming│
    │   DataFrame    │
    └────────┬───────┘
             │
             ↓ Transform
    ┌────────────────┐
    │ Aggregations   │
    │   Windows      │
    └────────┬───────┘
             │
             ↓ Output
    ┌────────────────┐
    │  Sink          │
    │ (Files/DB)     │
    └────────────────┘
             ↓
    ┌────────────────┐
    │  Checkpoint    │
    │  (Recovery)    │
    └────────────────┘
```

### Reading Streaming Data

From socket:

```python
# Read from TCP socket
lines = spark.readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 9999) \
    .load()
```

From files:

```python
# CSV files
df = spark.readStream \
    .format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .schema(predefined_schema) \
    .load("path/to/directory")

# JSON files
df = spark.readStream \
    .format("json") \
    .option("maxFilesPerTrigger", 1) \
    .load("path/to/directory")

# Parquet files
df = spark.readStream \
    .format("parquet") \
    .schema(predefined_schema) \
    .load("path/to/directory")

# Text files
df = spark.readStream \
    .format("text") \
    .load("path/to/directory")
```

From Kafka:

```python
# Read from Kafka topic
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "topic1") \
    .option("startingOffsets", "earliest") \
    .load()

# Multiple topics
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "topic1,topic2,topic3") \
    .load()

# Topic pattern
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribePattern", "topic.*") \
    .load()

# Parse Kafka message
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StructField, StringType

schema = StructType([
    StructField("name", StringType()),
    StructField("age", StringType())
])

df = df.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")
```

### Stream Processing

Basic transformations:

```python
from pyspark.sql.functions import explode, split, col, length

# Split and explode
words = df.select(explode(split(col("value"), " ")).alias("word"))

# Filter
filtered = df.filter(col("age") > 25)

# Select
selected = df.select("name", "age")

# With new columns
enhanced = df.withColumn("word_length", length(col("word")))
```

Aggregations:

```python
from pyspark.sql.functions import count, avg, sum, max, min

# Group by and count
wordCounts = words.groupBy("word").count()

# Multiple aggregations
stats = df.groupBy("category").agg(
    count("*").alias("count"),
    avg("value").alias("avg_value"),
    max("value").alias("max_value")
)
```

Window operations:

```python
from pyspark.sql.functions import window, col

# Tumbling window (non-overlapping)
windowedCounts = df \
    .withWatermark("timestamp", "10 minutes") \
    .groupBy(
        window(col("timestamp"), "10 minutes"),
        col("category")
    ) \
    .count()

# Sliding window (overlapping)
slidingCounts = df \
    .withWatermark("timestamp", "10 minutes") \
    .groupBy(
        window(col("timestamp"), "10 minutes", "5 minutes"),  # window, slide
        col("category")
    ) \
    .count()

# Session window (event-based gaps)
from pyspark.sql.functions import session_window
sessionCounts = df \
    .withWatermark("timestamp", "10 minutes") \
    .groupBy(
        session_window(col("timestamp"), "5 minutes"),  # gap duration
        col("user_id")
    ) \
    .count()
```

Watermarking (handling late data):

```python
from pyspark.sql.functions import window, col

# Define watermark (data older than 10 minutes will be dropped)
df_with_watermark = df \
    .withWatermark("timestamp", "10 minutes") \
    .groupBy(
        window(col("timestamp"), "5 minutes"),
        col("device_id")
    ) \
    .count()
```

### Writing Streaming Data

```
    Output Modes:
    
    ┌──────────────────┐
    │  Query Type?     │
    └────────┬─────────┘
             │
    ┌────────┴────────┐
             │
    ┌────────┴─────────────────┐
    │                          │
    ↓ (With Aggregation)       ↓ (No Aggregation)
┌────────┐                 ┌────────┐
│complete│                 │ append │
│  OR    │                 └────────┘
│ update │
└────────┘
```

Console output:

```python
# Display results in console
query = wordCounts.writeStream \
    .outputMode("complete") \
    .format("console") \
    .option("truncate", "false") \
    .option("numRows", 20) \
    .start()

query.awaitTermination()
```

File output:

```python
# Parquet
query = df.writeStream \
    .outputMode("append") \
    .format("parquet") \
    .option("path", "path/to/output") \
    .option("checkpointLocation", "path/to/checkpoint") \
    .start()

# CSV
query = df.writeStream \
    .outputMode("append") \
    .format("csv") \
    .option("path", "path/to/output") \
    .option("checkpointLocation", "path/to/checkpoint") \
    .option("header", "true") \
    .start()

# Partitioned output
query = df.writeStream \
    .outputMode("append") \
    .format("parquet") \
    .partitionBy("year", "month") \
    .option("path", "path/to/output") \
    .option("checkpointLocation", "path/to/checkpoint") \
    .start()
```

Kafka output:

```python
# Write to Kafka
from pyspark.sql.functions import to_json, struct

kafka_df = df.select(
    col("key"),
    to_json(struct("*")).alias("value")
)

query = kafka_df.writeStream \
    .outputMode("append") \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "output_topic") \
    .option("checkpointLocation", "path/to/checkpoint") \
    .start()
```

Memory output (for debugging):

```python
# Store in memory table
query = wordCounts.writeStream \
    .outputMode("complete") \
    .format("memory") \
    .queryName("word_counts") \
    .start()

# Query the memory table
spark.sql("SELECT * FROM word_counts").show()
```

ForeachBatch (custom logic):

```python
def process_batch(batch_df, batch_id):
    # Custom processing for each micro-batch
    batch_df.write \
        .format("jdbc") \
        .option("url", "jdbc:postgresql://localhost:5432/db") \
        .option("dbtable", "my_table") \
        .mode("append") \
        .save()
    print(f"Processed batch {batch_id}")

query = df.writeStream \
    .outputMode("update") \
    .foreachBatch(process_batch) \
    .start()
```

Foreach (row-by-row):

```python
from pyspark.sql.streaming import ForeachWriter

class MyForeachWriter(ForeachWriter):
    def open(self, partition_id, epoch_id):
        # Open connection
        return True
    
    def process(self, row):
        # Process each row
        print(row)
    
    def close(self, error):
        # Close connection
        pass

query = df.writeStream \
    .foreach(MyForeachWriter()) \
    .start()
```

### Output Modes

```python
# Append: Only new rows (no aggregation or with watermark)
.outputMode("append")

# Complete: All rows every time (aggregations only)
.outputMode("complete")

# Update: Only updated rows (aggregations only)
.outputMode("update")
```

### Query Management

```python
# Start query
query = df.writeStream \
    .outputMode("complete") \
    .format("console") \
    .start()

# Wait for termination
query.awaitTermination()

# Wait with timeout
query.awaitTermination(timeout=60)  # 60 seconds

# Stop query
query.stop()

# Check status
print(query.status)
print(query.lastProgress)
print(query.recentProgress)

# Get active streams
active_streams = spark.streams.active
for stream in active_streams:
    print(stream.id)
    print(stream.name)

# Stop all streams
for stream in spark.streams.active:
    stream.stop()

# Query with trigger
query = df.writeStream \
    .outputMode("complete") \
    .format("console") \
    .trigger(processingTime="10 seconds") \
    .start()

# Available triggers:
# .trigger(processingTime="10 seconds")  # Micro-batch every 10 sec
# .trigger(once=True)                    # Process once and stop
# .trigger(continuous="1 second")         # Continuous processing
```

## Performance Tuning

### Performance Optimization Flow

```
    ┌──────────────────┐
    │  Identify        │
    │  Bottleneck      │
    └────────┬─────────┘
             │
    ┌────────┴────────┐
    │                 │
    ↓                 ↓
┌─────────┐     ┌──────────┐
│ Spark   │     │ Monitor  │
│  UI     │     │ Metrics  │
└────┬────┘     └─────┬────┘
     │                │
     └────────┬───────┘
              │
    ┌─────────┴──────────┐
    │                    │
    ↓                    ↓
┌─────────┐        ┌──────────┐
│ Memory  │        │ Shuffle  │
│ Issue   │        │ Issue    │
└────┬────┘        └─────┬────┘
     │                   │
     ↓                   ↓
┌─────────┐        ┌──────────┐
│ Cache   │        │Partition │
│Optimize │        │Optimize  │
└─────────┘        └──────────┘
     │                   │
     └───────┬───────────┘
             │
             ↓
    ┌────────────────┐
    │  Re-measure    │
    │  Performance   │
    └────────────────┘
```

### Data Partitioning

```python
# Check current partitions
print(f"Number of partitions: {df.rdd.getNumPartitions()}")

# Repartition (increases or decreases, causes shuffle)
df_repartitioned = df.repartition(100)
df_repartitioned = df.repartition("user_id")  # By column

# Coalesce (only decreases, no shuffle)
df_coalesced = df.coalesce(10)

# Optimal partition size: 128MB - 1GB per partition
# Rule of thumb: 2-3x number of cores

# Partition by column (for joins and aggregations)
df_partitioned = df.repartition("date", "category")

# Check partition distribution
df.groupBy(spark_partition_id()).count().show()

# Custom partitioner for RDDs
from pyspark import HashPartitioner
rdd = rdd.partitionBy(100, lambda key: hash(key) % 100)
```

### Caching and Persistence

```python
from pyspark import StorageLevel

# Cache in memory (default: MEMORY_AND_DISK)
df_cached = df.cache()

# Persist with storage level
df.persist(StorageLevel.MEMORY_ONLY)        # Memory only
df.persist(StorageLevel.MEMORY_AND_DISK)    # Memory, overflow to disk
df.persist(StorageLevel.DISK_ONLY)          # Disk only
df.persist(StorageLevel.MEMORY_AND_DISK_SER) # Serialized in memory
df.persist(StorageLevel.OFF_HEAP)           # Off-heap memory

# Unpersist
df.unpersist()

# Check if cached
print(df.is_cached)

# Best practices:
# 1. Cache DataFrames used multiple times
# 2. Cache after expensive operations (joins, aggregations)
# 3. Unpersist when no longer needed
# 4. Cache before count() to trigger computation

# Example
df_filtered = df.filter(col("age") > 25).cache()
df_filtered.count()  # Trigger caching
result1 = df_filtered.groupBy("city").count()
result2 = df_filtered.groupBy("country").avg("salary")
```

### Broadcast Variables

```python
# Broadcast small datasets (< 2GB) to all executors
small_data = spark.sparkContext.broadcast(small_lookup_dict)

# Use in RDD operations
rdd.map(lambda x: small_data.value.get(x))

# Use in DataFrame operations
from pyspark.sql.functions import udf, broadcast

# Broadcast join (for small tables)
result = large_df.join(broadcast(small_df), "key")

# Manual broadcast
broadcast_var = spark.sparkContext.broadcast(small_df.collect())

# Unpersist broadcast variable
broadcast_var.unpersist()

# Best practices:
# - Broadcast tables < 10MB automatically
# - Manually broadcast for tables up to 2GB
# - Use for dimension tables in star schema joins
```

### Accumulators

```python
# Create accumulator
counter = spark.sparkContext.accumulator(0)
sum_accumulator = spark.sparkContext.accumulator(0.0)

# Use in operations
def process_row(row):
    counter.add(1)
    return row

rdd.foreach(process_row)
print(f"Processed {counter.value} rows")

# Custom accumulator
from pyspark.accumulators import AccumulatorParam

class SetAccumulatorParam(AccumulatorParam):
    def zero(self, value):
        return set()
    
    def addInPlace(self, val1, val2):
        val1.update(val2)
        return val1

set_accumulator = spark.sparkContext.accumulator(set(), SetAccumulatorParam())

# Note: Only use accumulators in actions, not transformations
```

### Memory Management

```python
# Check memory usage
spark.sparkContext._conf.get("spark.executor.memory")
spark.sparkContext._conf.get("spark.driver.memory")

# Memory tuning configurations:
spark = SparkSession.builder \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.memory.storageFraction", "0.5") \
    .config("spark.executor.memoryOverhead", "1g") \
    .getOrCreate()

# Memory regions:
# - Execution: 60% of spark.memory.fraction (shuffles, joins, sorts)
# - Storage: 40% of spark.memory.fraction (cache, broadcasts)
# - User: 1 - spark.memory.fraction (user objects)

# Best practices:
# 1. executor.memory = 4-8GB optimal
# 2. Too large executors cause GC issues
# 3. Monitor via Spark UI > Executors tab
```

### Shuffle Optimization

```python
# Control shuffle partitions
spark.conf.set("spark.sql.shuffle.partitions", "200")

# Default: 200 (often too many for small data)
# Rule: Set to ~2x number of cores for small data
# For large data: Aim for 128MB-1GB per partition

# Reduce shuffles:
# 1. Use broadcast joins for small tables
result = large_df.join(broadcast(small_df), "key")

# 2. Pre-partition before multiple operations
df_partitioned = df.repartition("key").cache()
result1 = df_partitioned.groupBy("key").sum()
result2 = df_partitioned.groupBy("key").avg()

# 3. Avoid wide transformations when possible
# Wide: groupBy, join, distinct, repartition
# Narrow: filter, select, map

# 4. Coalesce writer partitions
df.coalesce(1).write.parquet("output")

# Adaptive Query Execution (Spark 3.0+)
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
```

### Serialization

```python
# Use Kryo serialization (faster than Java serialization)
spark = SparkSession.builder \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.kryoserializer.buffer.max", "512m") \
    .getOrCreate()

# Register custom classes
from pyspark import SparkConf

conf = SparkConf()
conf.set("spark.kryo.registrationRequired", "false")
conf.registerKryoClasses([MyClass1, MyClass2])
```

### Data Skew Handling

```python
# Detect skew
df.groupBy("key").count().orderBy(col("count").desc()).show()

# Solution 1: Salting (add random prefix)
from pyspark.sql.functions import rand, floor

df_salted = df.withColumn("salt", (rand() * 10).cast("int"))
df_salted = df_salted.withColumn("salted_key", 
    concat(col("key"), lit("_"), col("salt")))

# Join with salted keys
result = df1_salted.join(df2_salted, "salted_key")

# Solution 2: Broadcast the skewed side
result = df1.join(broadcast(df2), "key")

# Solution 3: Adaptive Query Execution
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
```

### Query Optimization

```python
# Use DataFrames over RDDs (Catalyst optimizer)
# Good: df.filter(col("age") > 25)
# Avoid: rdd.filter(lambda x: x[0] > 25)

# Filter early
df_filtered = df.filter(col("age") > 25).select("name", "city")  # Good
# Avoid: df.select("name", "city").filter(col("age") > 25)

# Select only needed columns
df.select("col1", "col2")  # Good
# Avoid: df.select("*")

# Use built-in functions over UDFs
df.withColumn("upper", upper(col("name")))  # Good
# Avoid: df.withColumn("upper", my_udf(col("name")))

# Predicate pushdown
df = spark.read.parquet("data") \
    .filter(col("year") == 2023)  # Pushed to storage

# Column pruning
df = spark.read.parquet("data") \
    .select("name", "age")  # Only read needed columns

# Explain query plan
df.explain()
df.explain(mode="extended")  # More details
df.explain(mode="cost")      # With statistics
```

### Monitoring and Debugging

```python
# Spark UI: http://localhost:4040

# Show execution plan
df.explain()

# Show physical plan
df.explain(mode="formatted")

# Enable event logs
spark = SparkSession.builder \
    .config("spark.eventLog.enabled", "true") \
    .config("spark.eventLog.dir", "file:///tmp/spark-events") \
    .getOrCreate()

# Log level
spark.sparkContext.setLogLevel("WARN")  # ERROR, WARN, INFO, DEBUG

# Checkpoint for long lineages
spark.sparkContext.setCheckpointDir("path/to/checkpoint")
df.checkpoint()  # Truncate lineage
```

## Common Issues and Debugging

### Issue Resolution Flow

```
    ┌─────────────┐
    │   Error?    │
    └──────┬──────┘
           │
    ┌──────┴─────────────────┐
    │                        │
    ↓                        ↓
┌─────────┐            ┌──────────┐
│  OOM    │            │ Slow     │
│ Error   │            │ Perform  │
└────┬────┘            └────┬─────┘
     │                      │
     ↓                      ↓
┌─────────┐            ┌──────────┐
│Increase │            │Check     │
│ Memory  │            │Spark UI  │
└────┬────┘            └────┬─────┘
     │                      │
     ↓ Still Fails?         ↓
┌─────────┐            ┌──────────┐
│Optimize │            │Optimize  │
│ Code    │            │Query     │
└─────────┘            └──────────┘
```

### Out of Memory Errors

Executor OOM:

```python
# Symptoms:
# - java.lang.OutOfMemoryError: Java heap space
# - Container killed by YARN

# Solutions:

# 1. Increase executor memory
spark = SparkSession.builder \
    .config("spark.executor.memory", "8g") \
    .config("spark.executor.memoryOverhead", "2g") \
    .getOrCreate()

# 2. Increase partitions (reduce data per partition)
spark.conf.set("spark.sql.shuffle.partitions", "400")
df = df.repartition(400)

# 3. Avoid collect() on large datasets
# Bad: data = df.collect()
# Good: df.write.parquet("output")

# 4. Use persist() strategically
df.cache()  # Only if reused multiple times

# 5. Process data in batches
for batch in df.randomSplit([0.1] * 10):
    batch.write.mode("append").parquet("output")
```

Driver OOM:

```python
# Symptoms:
# - OutOfMemoryError in driver
# - collect() or take(large_n) fails

# Solutions:

# 1. Increase driver memory
spark = SparkSession.builder \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# 2. Avoid collecting large data to driver
# Bad: results = df.collect()
# Good: df.write.parquet("output")

# 3. Use limit() before collect()
sample = df.limit(1000).collect()

# 4. Don't broadcast large variables
# Bad: bc = spark.sparkContext.broadcast(large_data)  # > 2GB
# Good: Use join instead

# 5. Check for data leaks in UDFs
# Bad: global_list = []  # Grows indefinitely
```

### Slow Performance

Identify bottleneck:

```python
# 1. Check Spark UI (http://localhost:4040)
# - Jobs tab: See failed/slow jobs
# - Stages tab: Identify slow stages
# - Storage tab: Check cache usage
# - Executors tab: Check resource usage
# - SQL tab: View query plans

# 2. Enable query execution time logging
spark.conf.set("spark.sql.debug.maxToStringFields", "100")

# 3. Explain query plan
df.explain()
df.explain(mode="extended")

# 4. Check for skew
df.groupBy("key").count().orderBy(col("count").desc()).show()

# 5. Monitor GC time
# Check Executor tab in Spark UI for GC time
# If GC time > 10% execution time, tune memory
```

Common performance fixes:

```python
# 1. Use DataFrame API over RDD
# Bad: rdd.map().filter().collect()
# Good: df.select().filter().collect()

# 2. Filter early
# Good: df.filter(col("year") == 2023).select("name")
# Bad: df.select("name").filter(col("year") == 2023)

# 3. Avoid UDFs when possible
# Bad: df.withColumn("upper", my_udf(col("name")))
# Good: df.withColumn("upper", upper(col("name")))

# 4. Use broadcast joins
# Good: large_df.join(broadcast(small_df), "key")

# 5. Adjust shuffle partitions
spark.conf.set("spark.sql.shuffle.partitions", "200")

# 6. Enable adaptive query execution
spark.conf.set("spark.sql.adaptive.enabled", "true")
```

### Serialization Errors

```python
# Symptoms:
# - Task not serializable
# - NotSerializableException

# Solutions:

# 1. Don't reference class members in RDD/DataFrame operations
# Bad:
class MyClass:
    def __init__(self):
        self.data = some_data
    
    def process(self, rdd):
        return rdd.map(lambda x: x + self.data)  # Serializes entire class

# Good:
class MyClass:
    def __init__(self):
        self.data = some_data
    
    def process(self, rdd):
        local_data = self.data  # Extract to local variable
        return rdd.map(lambda x: x + local_data)

# 2. Use broadcast for non-serializable objects
non_serializable_obj = MyComplexObject()
bc_obj = spark.sparkContext.broadcast(non_serializable_obj)
rdd.map(lambda x: bc_obj.value.process(x))

# 3. Make class serializable
import pickle

class MyClass:
    def __reduce__(self):
        return (self.__class__, (self.data,))
```

### Data Skew

```python
# Symptoms:
# - Few tasks take much longer than others
# - Some executors idle while others work
# - Uneven data distribution

# Detect skew:
df.groupBy("key").count() \
    .orderBy(col("count").desc()) \
    .show(20)

# Solution 1: Salting
from pyspark.sql.functions import rand, floor, concat, lit

# Add random salt to skewed key
df_salted = df.withColumn("salt", (rand() * 10).cast("int")) \
    .withColumn("salted_key", concat(col("key"), lit("_"), col("salt")))

# Solution 2: Separate hot keys
hot_keys = ["key1", "key2", "key3"]
df_hot = df.filter(col("key").isin(hot_keys))
df_normal = df.filter(~col("key").isin(hot_keys))

# Process separately
result_hot = df_hot.join(broadcast(small_df), "key")
result_normal = df_normal.join(small_df, "key")
result = result_hot.union(result_normal)

# Solution 3: Adaptive skew join (Spark 3.0+)
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionFactor", "5")
```

### Connection Issues

```python
# JDBC connection issues:
# 1. Check driver is available
# 2. Increase timeout
df = spark.read.format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5432/db") \
    .option("connectTimeout", "60") \
    .option("socketTimeout", "60") \
    .load()

# 3. Use connection pooling
df.write.format("jdbc") \
    .option("numPartitions", "10") \
    .option("batchsize", "10000") \
    .save()

# Kafka connection issues:
# 1. Check broker reachability
# 2. Verify topic exists
# 3. Check security settings
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("failOnDataLoss", "false") \
    .load()
```

### Common Error Messages

```python
# "py4j.protocol.Py4JJavaError"
# - Java exception in Spark
# - Read the caused by section for root cause

# "Analysis exception"
# - Column not found
# - Check column names and case sensitivity

# "Stage cancelled"
# - Job failed or manually stopped
# - Check previous errors

# "Executor lost"
# - Executor crashed (OOM, network issue)
# - Check executor logs

# "No space left on device"
# - Disk full on executor/driver
# - Increase disk space or clean temp files

# "Failed to bind to port"
# - Port already in use
# - Change port or stop conflicting process
```

## Spark Configuration

### Configuration Hierarchy

```
    ┌─────────────────────┐
    │  spark-defaults.conf│  (Lowest priority)
    └──────────┬──────────┘
               │
               ↓ Overridden by
    ┌─────────────────────┐
    │ SparkConf in code   │
    └──────────┬──────────┘
               │
               ↓ Overridden by
    ┌─────────────────────┐
    │ spark-submit flags  │
    └──────────┬──────────┘
               │
               ↓ Overridden by
    ┌─────────────────────┐
    │ Runtime config      │  (Highest priority)
    └─────────────────────┘
```

### Essential Configurations

Application settings:

```python
from pyspark import SparkConf
from pyspark.sql import SparkSession

# Using SparkConf
conf = SparkConf() \
    .setAppName("MyApp") \
    .setMaster("local[*]")  # local[*], yarn, mesos, k8s

spark = SparkSession.builder.config(conf=conf).getOrCreate()

# Using builder pattern
spark = SparkSession.builder \
    .appName("MyApp") \
    .master("local[*]") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

# Runtime configuration
spark.conf.set("spark.sql.shuffle.partitions", "200")
print(spark.conf.get("spark.sql.shuffle.partitions"))
```

Memory configurations:

```python
spark = SparkSession.builder \
    .config("spark.driver.memory", "4g") \
    .config("spark.driver.maxResultSize", "2g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.executor.memoryOverhead", "2g") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.memory.storageFraction", "0.5") \
    .getOrCreate()

# Memory breakdown:
# Total Executor Memory = executor.memory + memoryOverhead
# Spark Memory = executor.memory × memory.fraction
# Storage Memory = Spark Memory × storageFraction
# Execution Memory = Spark Memory × (1 - storageFraction)
```

Executor configurations:

```python
spark = SparkSession.builder \
    .config("spark.executor.cores", "4") \
    .config("spark.executor.instances", "10") \
    .config("spark.dynamicAllocation.enabled", "true") \
    .config("spark.dynamicAllocation.minExecutors", "2") \
    .config("spark.dynamicAllocation.maxExecutors", "50") \
    .config("spark.dynamicAllocation.initialExecutors", "10") \
    .getOrCreate()

# Optimal executor sizing:
# - 4-5 cores per executor
# - 4-8 GB memory per executor
# - Leave 1 core and 1GB for OS on each node
```

Shuffle and parallelism:

```python
spark = SparkSession.builder \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.default.parallelism", "100") \
    .config("spark.sql.files.maxPartitionBytes", "134217728")  # 128MB \
    .config("spark.sql.files.openCostInBytes", "4194304")  # 4MB \
    .getOrCreate()

# Tuning guidelines:
# shuffle.partitions: 2-3x number of cores (small data)
#                     aim for 128MB-1GB per partition (large data)
# default.parallelism: 2-3x number of cores
```

Serialization:

```python
spark = SparkSession.builder \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.kryo.registrationRequired", "false") \
    .config("spark.kryoserializer.buffer.max", "512m") \
    .getOrCreate()
```

Adaptive Query Execution (AQE):

```python
spark = SparkSession.builder \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.minPartitionNum", "1") \
    .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "134217728") \
    .config("spark.sql.adaptive.skewJoin.enabled", "true") \
    .config("spark.sql.adaptive.skewJoin.skewedPartitionFactor", "5") \
    .config("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "256MB") \
    .getOrCreate()
```

Broadcast settings:

```python
spark = SparkSession.builder \
    .config("spark.sql.autoBroadcastJoinThreshold", "10485760")  # 10MB \
    .config("spark.sql.broadcastTimeout", "300") \
    .getOrCreate()

# Set to -1 to disable auto broadcast
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
```

Storage and checkpointing:

```python
spark = SparkSession.builder \
    .config("spark.local.dir", "/tmp/spark") \
    .config("spark.eventLog.enabled", "true") \
    .config("spark.eventLog.dir", "file:///tmp/spark-events") \
    .config("spark.history.fs.logDirectory", "file:///tmp/spark-events") \
    .getOrCreate()

# Set checkpoint directory
spark.sparkContext.setCheckpointDir("hdfs://path/to/checkpoint")
```

Network and timeout:

```python
spark = SparkSession.builder \
    .config("spark.network.timeout", "800s") \
    .config("spark.executor.heartbeatInterval", "60s") \
    .config("spark.rpc.askTimeout", "600s") \
    .config("spark.sql.broadcastTimeout", "600") \
    .getOrCreate()
```

Compression:

```python
spark = SparkSession.builder \
    .config("spark.io.compression.codec", "snappy")  # snappy, lz4, gzip \
    .config("spark.rdd.compress", "true") \
    .config("spark.sql.inMemoryColumnarStorage.compressed", "true") \
    .config("spark.sql.parquet.compression.codec", "snappy") \
    .getOrCreate()
```

Dynamic allocation:

```python
spark = SparkSession.builder \
    .config("spark.dynamicAllocation.enabled", "true") \
    .config("spark.dynamicAllocation.minExecutors", "1") \
    .config("spark.dynamicAllocation.maxExecutors", "100") \
    .config("spark.dynamicAllocation.initialExecutors", "10") \
    .config("spark.dynamicAllocation.executorIdleTimeout", "60s") \
    .config("spark.dynamicAllocation.schedulerBacklogTimeout", "1s") \
    .getOrCreate()
```

### Common Configuration Patterns

Local development:

```python
spark = SparkSession.builder \
    .appName("LocalDev") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "10") \
    .getOrCreate()
```

Production (YARN):

```python
spark = SparkSession.builder \
    .appName("ProductionApp") \
    .master("yarn") \
    .config("spark.submit.deployMode", "cluster") \
    .config("spark.executor.memory", "8g") \
    .config("spark.executor.cores", "4") \
    .config("spark.executor.instances", "20") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "400") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()
```

Streaming application:

```python
spark = SparkSession.builder \
    .appName("StreamingApp") \
    .config("spark.streaming.backpressure.enabled", "true") \
    .config("spark.streaming.kafka.maxRatePerPartition", "1000") \
    .config("spark.sql.streaming.checkpointLocation", "/path/to/checkpoint") \
    .getOrCreate()
```

### View and Update Configuration

```python
# Get all configurations
all_conf = spark.sparkContext.getConf().getAll()
for key, value in all_conf:
    print(f"{key}: {value}")

# Get specific configuration
value = spark.conf.get("spark.sql.shuffle.partitions")

# Set configuration at runtime (only for SQL configurations)
spark.conf.set("spark.sql.shuffle.partitions", "100")

# Check if configuration is modifiable
spark.conf.isModifiable("spark.sql.shuffle.partitions")  # True
spark.conf.isModifiable("spark.executor.memory")  # False (static)
```

## Tips and Best Practices

### Development Best Practices

```
    Development Workflow:
    
    ┌──────────────┐
    │ Local Dev    │
    │(Small Sample)│
    └──────┬───────┘
           │
           ↓ Test
    ┌──────────────┐
    │  Unit Tests  │
    └──────┬───────┘
           │
           ↓ Deploy
    ┌──────────────┐
    │ Staging      │
    │(Medium Data) │
    └──────┬───────┘
           │
           ↓ Validate
    ┌──────────────┐
    │ Production   │
    │ (Full Data)  │
    └──────────────┘
```

Code organization:

```python
# Use virtual environments
python -m venv venv
source venv/bin/activate
pip install pyspark

# Project structure
my_pyspark_project/
├── src/
│   ├── __init__.py
│   ├── transformations.py
│   ├── utils.py
│   └── config.py
├── tests/
│   ├── test_transformations.py
│   └── test_utils.py
├── configs/
│   ├── dev.conf
│   └── prod.conf
├── requirements.txt
└── main.py

# Use meaningful names
# Good
user_purchases_df = spark.read.parquet("user_purchases")
active_users = user_purchases_df.filter(col("status") == "active")

# Bad
df1 = spark.read.parquet("data")
df2 = df1.filter(col("x") == "y")
```

Testing:

```python
# Unit tests with pytest
import pytest
from pyspark.sql import SparkSession
from src.transformations import filter_active_users

@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder \
        .master("local[2]") \
        .appName("test") \
        .getOrCreate()

def test_filter_active_users(spark):
    # Create test data
    data = [
        ("user1", "active"),
        ("user2", "inactive"),
        ("user3", "active")
    ]
    df = spark.createDataFrame(data, ["user_id", "status"])
    
    # Apply transformation
    result = filter_active_users(df)
    
    # Assert
    assert result.count() == 2
    assert result.filter(col("status") == "inactive").count() == 0
```

### Performance Best Practices

DO's:

```python
# ✓ Use DataFrame API over RDD
df.filter(col("age") > 25)  # Good

# ✓ Filter early and select only needed columns
df.filter(col("year") == 2023).select("name", "age")

# ✓ Use built-in functions
df.withColumn("upper", upper(col("name")))

# ✓ Broadcast small tables
large_df.join(broadcast(small_df), "key")

# ✓ Cache when reusing DataFrames
df_cached = df.filter(col("age") > 25).cache()
df_cached.count()  # Materialize
result1 = df_cached.groupBy("city").count()
result2 = df_cached.groupBy("country").avg("salary")

# ✓ Use vectorized Pandas UDFs
@pandas_udf(StringType())
def to_upper(s: pd.Series) -> pd.Series:
    return s.str.upper()

# ✓ Partition appropriately
df.repartition(100, "user_id")

# ✓ Enable Adaptive Query Execution
spark.conf.set("spark.sql.adaptive.enabled", "true")

# ✓ Use Parquet format
df.write.parquet("output")

# ✓ Specify schema when reading
df = spark.read.schema(schema).parquet("input")
```

DON'Ts:

```python
# ✗ Avoid collect() on large datasets
# data = df.collect()  # Bad

# ✗ Avoid Python UDFs when built-in exists
# df.withColumn("upper", my_udf(col("name")))  # Bad

# ✗ Avoid groupByKey (use reduceByKey)
# rdd.groupByKey()  # Bad

# ✗ Don't select all columns when not needed
# df.select("*").filter(...)  # Bad

# ✗ Avoid unnecessary shuffles
# df.repartition(10).repartition(20)  # Bad

# ✗ Don't cache everything
# df1.cache()  # Only if reused multiple times
# df2.cache()
# df3.cache()

# ✗ Avoid wide transformations when possible
# Multiple joins/groupBy in sequence without optimization
```

### Data Quality Practices

```python
# Validate schema
expected_schema = StructType([
    StructField("id", IntegerType(), False),
    StructField("name", StringType(), False)
])

if df.schema != expected_schema:
    raise ValueError("Schema mismatch")

# Check for nulls
null_counts = df.select([
    count(when(col(c).isNull(), c)).alias(c) 
    for c in df.columns
])
null_counts.show()

# Data validation
from pyspark.sql.functions import col, when, count

# Check data ranges
df.select(
    min("age"),
    max("age"),
    avg("age")
).show()

# Identify duplicates
duplicate_count = df.groupBy("id").count() \
    .filter(col("count") > 1) \
    .count()

# Add data quality flags
df_with_quality = df.withColumn(
    "quality_flag",
    when(col("age").isNull(), "missing_age")
    .when(col("age") < 0, "invalid_age")
    .when(col("age") > 150, "outlier_age")
    .otherwise("valid")
)
```

### Production Checklist

```python
# 1. Enable event logging
spark = SparkSession.builder \
    .config("spark.eventLog.enabled", "true") \
    .config("spark.eventLog.dir", "hdfs://path/to/logs") \
    .getOrCreate()

# 2. Set appropriate log level
spark.sparkContext.setLogLevel("WARN")

# 3. Use checkpointing for streaming
query = df.writeStream \
    .option("checkpointLocation", "hdfs://path/to/checkpoint") \
    .start()

# 4. Handle failures gracefully
from pyspark.sql.utils import AnalysisException

try:
    df = spark.read.parquet("input")
except AnalysisException as e:
    logger.error(f"Failed to read data: {e}")
    raise

# 5. Monitor query execution
df.explain()
df.explain(mode="cost")

# 6. Use appropriate file formats
# - Parquet: Best for analytics (columnar)
# - ORC: Good for Hive integration
# - Avro: Good for row-based operations
# - Delta: Best for ACID operations

# 7. Partition large datasets
df.write \
    .partitionBy("year", "month", "day") \
    .parquet("output")

# 8. Set reasonable timeouts
spark.conf.set("spark.network.timeout", "600s")
spark.conf.set("spark.sql.broadcastTimeout", "600")

# 9. Clean up resources
df.unpersist()
spark.catalog.clearCache()
spark.stop()
```

### Security Best Practices

```python
# 1. Don't hardcode credentials
# Bad
df = spark.read.jdbc(url, table, properties={"user": "admin", "password": "pass123"})

# Good - use environment variables
import os
properties = {
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}
df = spark.read.jdbc(url, table, properties=properties)

# 2. Enable encryption
spark = SparkSession.builder \
    .config("spark.authenticate", "true") \
    .config("spark.network.crypto.enabled", "true") \
    .config("spark.io.encryption.enabled", "true") \
    .getOrCreate()

# 3. Use secure connections
df = spark.read.format("jdbc") \
    .option("url", "jdbc:postgresql://host:5432/db?ssl=true") \
    .load()

# 4. Audit data access
logger.info(f"User {username} accessed table {table_name}")
```

### Key Recommendations

1. **Always prefer DataFrames over RDDs** for automatic optimization
2. **Filter early, select only needed columns** to reduce data movement
3. **Use built-in functions** instead of UDFs when possible
4. **Broadcast small tables** (<2GB) for efficient joins
5. **Cache strategically** only when DataFrames are reused multiple times
6. **Monitor Spark UI** to identify and fix bottlenecks
7. **Use Parquet** as default storage format for better performance
8. **Enable Adaptive Query Execution** for automatic optimization
9. **Partition wisely** based on query patterns (typically by date)
10. **Test with sample data** before running on full datasets
