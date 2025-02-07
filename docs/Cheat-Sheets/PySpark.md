
---
title: PySpark Cheat Sheet
description: A comprehensive reference guide for PySpark, covering setup, data loading, transformations, actions, SQL, MLlib, structured streaming, performance tuning, and more.
---

# PySpark Cheat Sheet

[TOC]

This cheat sheet provides an exhaustive overview of the PySpark API, covering essential concepts, code snippets, and best practices for efficient data processing and machine learning with Apache Spark. It aims to be a one-stop reference for common tasks.

## Getting Started

### Installation

```bash
pip install pyspark
```

Consider using a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate  # On Windows
```

### SparkSession

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("MyPySparkApp") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

# To use an existing SparkContext:
# spark = SparkSession(sparkContext=sc)
```

### SparkContext

```python
from pyspark import SparkContext, SparkConf

conf = SparkConf().setAppName("MyPySparkApp").setMaster("local[*]")
sc = SparkContext(conf=conf)

# To use SparkSession:
# from pyspark.sql import SparkSession
# spark = SparkSession(sparkContext=sc)
```

### Stopping SparkSession/SparkContext

```python
spark.stop()  # For SparkSession
sc.stop()     # For SparkContext
```

## Data Loading

### Loading from Text Files

```python
# SparkContext
lines = sc.textFile("path/to/my/textfile.txt")

# SparkSession
df = spark.read.text("path/to/my/textfile.txt")
```

### Loading from CSV Files

```python
# SparkSession
df = spark.read.csv("path/to/my/csvfile.csv", header=True, inferSchema=True)```

### Loading from JSON Files

```python
# SparkSession
df = spark.read.json("path/to/my/jsonfile.json")
```

### Loading from Parquet Files

```python
# SparkSession
df = spark.read.parquet("path/to/my/parquetfile.parquet")
```

### Loading from ORC Files

```python
# SparkSession
df = spark.read.orc("path/to/my/orcfile.orc")
```

### Loading from Avro Files

```python
# SparkSession
df = spark.read.format("avro").load("path/to/my/avrofile.avro")
```

### Loading from JDBC

```python
# SparkSession
df = spark.read.format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5432/mydatabase") \
    .option("dbtable", "mytable") \
    .option("user", "myuser") \
    .option("password", "mypassword") \
    .load()
```

### Loading from Delta Lake

```python
from delta.tables import DeltaTable

deltaTable = DeltaTable.forPath(spark, "path/to/my/delta_table")
df = deltaTable.toDF()
```

## DataFrames

### Creating DataFrames

From RDD:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Example").getOrCreate()
data = [("Alice", 30), ("Bob", 25)]
rdd = spark.sparkContext.parallelize(data)
df = spark.createDataFrame(rdd, schema=["Name", "Age"])
```

From List of Dictionaries:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Example").getOrCreate()
data = [{"Name": "Alice", "Age": 30}, {"Name": "Bob", "Age": 25}]
df = spark.createDataFrame(data)
```

### DataFrame Operations

*   `df.show()`: Displays the DataFrame.
*   `df.printSchema()`: Prints the schema of the DataFrame.
*   `df.columns`: Returns a list of column names.
*   `df.count()`: Returns the number of rows.
*   `df.describe()`: Computes summary statistics.
*   `df.summary("count", "mean", "stddev", "min", "max", "25%", "50%", "75%")`: Computes descriptive statistics.
*   `df.select("column1", "column2")`: Selects specific columns.
*   `df.withColumn("new_column", df["column1"] + df["column2"])`: Adds a new column.
*   `df.withColumnRenamed("old_name", "new_name")`: Renames a column.
*   `df.drop("column1")`: Drops a column.
*   `df.filter(df["age"] > 25)`: Filters rows based on a condition.
*   `df.where(df["age"] > 25)`: Another way to filter rows.
*   `df.groupBy("column1").count()`: Groups data and counts occurrences.
*   `df.orderBy("column1", ascending=False)`: Orders data by a column.
*   `df.sort("column1", ascending=False)`: Another way to order data.
*   `df.limit(10)`: Limits the number of rows.
*   `df.distinct()`: Removes duplicate rows.
*   `df.union(other_df)`: Unions two DataFrames (requires same schema).
*   `df.unionByName(other_df)`: Unions two DataFrames by column name.
*   `df.intersect(other_df)`: Returns the intersection of two DataFrames.
*   `df.subtract(other_df)`: Returns the rows in `df` but not in `other_df`.
*   `df.join(other_df, df["key"] == other_df["key"], how="inner")`: Joins two DataFrames.
*   `df.crossJoin(other_df)`: Performs a Cartesian product join.
*   `df.agg({"age": "avg"})`: Performs aggregation functions (avg, min, max, sum, etc.).
*   `df.rollup("column1", "column2")`: Creates rollup aggregates.
*   `df.cube("column1", "column2")`: Creates cube aggregates.
*   `df.pivot("column1", values=["value1", "value2"])`: Pivots a DataFrame.
*   `df.sample(withReplacement=False, fraction=0.5, seed=None)`: Samples a fraction of rows.
*   `df.randomSplit([0.8, 0.2], seed=None)`: Splits the DataFrame into multiple DataFrames randomly.
*   `df.cache()`: Caches the DataFrame in memory.
*   `df.persist(StorageLevel.MEMORY_AND_DISK)`: Persists the DataFrame with a specific storage level.
*   `df.unpersist()`: Removes the DataFrame from the cache.

### Applying Python Functions (UDFs)

```python
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

def to_upper(s):
    return s.upper()

to_upper_udf = udf(to_upper, StringType())

df = df.withColumn("upper_name", to_upper_udf(df["Name"]))
```

### Applying Pandas UDFs (Vectorized UDFs)

```python
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType
import pandas as pd

@pandas_udf(StringType())
def to_upper_pandas(series: pd.Series) -> pd.Series:
    return series.str.upper()

df = df.withColumn("upper_name", to_upper_pandas(df["Name"]))
```

### GroupBy Operations

```python
from pyspark.sql.functions import avg, max, min, sum, count

# Group by a single column
grouped_df = df.groupBy("Department")
grouped_df.count().show()

# Group by multiple columns
grouped_df = df.groupBy("Department", "City")
grouped_df.agg(avg("Salary"), sum("Bonus")).show()

# Applying aggregation functions
from pyspark.sql.functions import col
df.groupBy("Department") \
  .agg(avg(col("Salary")).alias("Average Salary"),
       sum(col("Bonus")).alias("Total Bonus")) \
  .show()

# Window functions
from pyspark.sql import Window
from pyspark.sql.functions import rank, dense_rank, row_number

windowSpec  = Window.partitionBy("Department").orderBy(col("Salary").desc())
df.withColumn("rank",rank().over(windowSpec)).show()
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

### Creating RDDs

From a List:

```python
data = [1, 2, 3, 4, 5]
rdd = sc.parallelize(data)
```

From a Text File:

```python
rdd = sc.textFile("path/to/my/textfile.txt")
```

### RDD Transformations

*   `rdd.map(lambda x: x * 2)`: Applies a function to each element.
*   `rdd.filter(lambda x: x > 2)`: Filters elements based on a condition.
*   `rdd.flatMap(lambda x: x.split())`: Flattens and maps elements.
*   `rdd.distinct()`: Removes duplicate elements.
*   `rdd.sample(withReplacement=False, fraction=0.5)`: Samples elements.
*   `rdd.union(other_rdd)`: Unions two RDDs.
*   `rdd.intersection(other_rdd)`: Intersects two RDDs.
*   `rdd.subtract(other_rdd)`: Subtracts one RDD from another.
*   `rdd.cartesian(other_rdd)`: Computes the Cartesian product of two RDDs.
*   `rdd.sortBy(lambda x: x, ascending=False)`: Sorts elements.
*   `rdd.repartition(numPartitions=4)`: Changes the number of partitions.
*   `rdd.coalesce(numPartitions=1)`: Decreases the number of partitions.
*   `rdd.pipe(command)`: Pipes each element to a shell command.
*   `rdd.zip(other_rdd)`: Zips two RDDs together.
*   `rdd.zipWithIndex()`: Zips the RDD with its element indices.

### RDD Actions

*   `rdd.collect()`: Returns all elements as a list.
*   `rdd.count()`: Returns the number of elements.
*   `rdd.first()`: Returns the first element.
*   `rdd.take(3)`: Returns the first N elements.
*   `rdd.top(3)`: Returns the top N elements.
*   `rdd.reduce(lambda x, y: x + y)`: Reduces elements using a function.
*   `rdd.fold(zeroValue, op)`: Folds elements using a function and a zero value.
*   `rdd.aggregate(zeroValue, seqOp, combOp)`: Aggregates elements using sequence and combination functions.
*   `rdd.foreach(lambda x: print(x))`: Applies a function to each element.
*   `rdd.saveAsTextFile("path/to/output")`: Saves the RDD as a text file.
*   `rdd.saveAsPickleFile("path/to/output")`: Saves the RDD as a serialized Python object file.
*   `rdd.countByKey()`: Returns the count of each key (for pair RDDs).
*   `rdd.collectAsMap()`: Returns the elements as a dictionary (for pair RDDs).
*   `rdd.lookup(key)`: Returns the values for a given key (for pair RDDs).

### Pair RDDs

*   `rdd.map(lambda x: (x, 1))`: Creates a pair RDD.
*   `rdd.reduceByKey(lambda x, y: x + y)`: Reduces values for each key.
*   `rdd.groupByKey()`: Groups values for each key.
*   `rdd.aggregateByKey(zeroValue, seqFunc, combFunc)`: Aggregates values for each key.
*   `rdd.foldByKey(zeroValue, func)`: Folds values for each key.
*   `rdd.combineByKey(createCombiner, mergeValue, mergeCombiners)`: Generic combine function for each key.
*   `rdd.sortByKey()`: Sorts by key.
*   `rdd.join(other_rdd)`: Joins two pair RDDs.
*   `rdd.leftOuterJoin(other_rdd)`: Performs a left outer join.
*   `rdd.rightOuterJoin(other_rdd)`: Performs a right outer join.
*   `rdd.fullOuterJoin(other_rdd)`: Performs a full outer join.
*   `rdd.cogroup(other_rdd)`: Groups values for each key in multiple RDDs.

## Writing Data

### Writing DataFrames

To CSV:

```python
df.write.csv("path/to/output/csv", header=True, mode="overwrite")
```

To JSON:

```python
df.write.json("path/to/output/json", mode="overwrite")
```

To Parquet:

```python
df.write.parquet("path/to/output/parquet", mode="overwrite")
```

To ORC:

```python
df.write.orc("path/to/output/orc", mode="overwrite")
```

To Avro:

```python
df.write.format("avro").save("path/to/output/avro", mode="overwrite")
```

To JDBC:

```python
df.write.format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5432/mydatabase") \
    .option("dbtable", "mytable") \
    .option("user", "myuser") \
    .option("password", "mypassword") \
    .mode("overwrite") \
    .save()
```

To Delta Lake:

```python
df.write.format("delta").mode("overwrite").save("path/to/delta_table")
```

### Writing RDDs

```python
rdd.saveAsTextFile("path/to/output")
rdd.saveAsPickleFile("path/to/output")
rdd.saveAsSequenceFile("path/to/output") # For Hadoop SequenceFile format
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

### Data Preparation

```python
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler

# StringIndexer
indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
indexed_df = indexer.fit(df).transform(df)

# VectorAssembler
assembler = VectorAssembler(inputCols=["feature1", "feature2"], outputCol="features")
output_df = assembler.transform(indexed_df)

# StandardScaler
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                        withStd=True, withMean=True)
scalerModel = scaler.fit(output_df)
scaled_df = scalerModel.transform(output_df)
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

Logistic Regression:

```python
from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(featuresCol="features", labelCol="label")
model = lr.fit(training_data)
predictions = model.transform(test_data)
```

Decision Tree:

```python
from pyspark.ml.classification import DecisionTreeClassifier

dt = DecisionTreeClassifier(featuresCol="features", labelCol="label")
model = dt.fit(training_data)
predictions = model.transform(test_data)
```

Random Forest:

```python
from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(featuresCol="features", labelCol="label")
model = rf.fit(training_data)
predictions = model.transform(test_data)
```

Gradient-Boosted Trees (GBT):

```python
from pyspark.ml.classification import GBTClassifier

gbt = GBTClassifier(featuresCol="features", labelCol="label")
model = gbt.fit(training_data)
predictions = model.transform(test_data)
```

Multilayer Perceptron Classifier (MLPC):

```python
from pyspark.ml.classification import MultilayerPerceptronClassifier

layers = [4, 5, 4, 3] # Input size, hidden layers, output size
mlp = MultilayerPerceptronClassifier(layers=layers, featuresCol='features', labelCol='label')
model = mlp.fit(training_data)
predictions = model.transform(test_data)
```

### Regression

Linear Regression:

```python
from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol="features", labelCol="label")
model = lr.fit(training_data)
predictions = model.transform(test_data)
```

Decision Tree Regression:

```python
from pyspark.ml.regression import DecisionTreeRegressor

dt = DecisionTreeRegressor(featuresCol="features", labelCol="label")
model = dt.fit(training_data)
predictions = model.transform(test_data)
```

Random Forest Regression:

```python
from pyspark.ml.regression import RandomForestRegressor

rf = RandomForestRegressor(featuresCol="features", labelCol="label")
model = rf.fit(training_data)
predictions = model.transform(test_data)
```

Gradient-Boosted Trees (GBT) Regression:

```python
from pyspark.ml.regression import GBTRegressor

gbt = GBTRegressor(featuresCol="features", labelCol="label")
model = gbt.fit(training_data)
predictions = model.transform(test_data)
```

### Clustering

K-Means:

```python
from pyspark.ml.clustering import KMeans

kmeans = KMeans(k=3, featuresCol="features")
model = kmeans.fit(data)
predictions = model.transform(data)
```

Gaussian Mixture Model (GMM):

```python
from pyspark.ml.clustering import GaussianMixture

gmm = GaussianMixture().setK(2).setSeed(538009335)
model = gmm.fit(data)
predictions = model.transform(data)
```

### Recommendation

Alternating Least Squares (ALS):

```python
from pyspark.ml.recommendation import ALS

als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating")
model = als.fit(training_data)
predictions = model.transform(test_data)
```

### Evaluation

Classification Metrics:

```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %s" % (accuracy))
```

Regression Metrics:

```python
from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("RMSE = %s" % (rmse))
```

Clustering Metrics:

```python
from pyspark.ml.evaluation import ClusteringEvaluator

evaluator = ClusteringEvaluator(featuresCol="features")
silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))
```

### Cross-Validation

```python
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .addGrid(lr.fitIntercept, [False, True]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

crossval = CrossValidator(estimator=lr,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)

cvModel = crossval.fit(training_data)
predictions = cvModel.transform(test_data)
```

### Pipelines

```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression

# Define stages
indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
assembler = VectorAssembler(inputCols=["feature1", "feature2"], outputCol="features")
lr = LogisticRegression(featuresCol="features", labelCol="label")

# Create pipeline
pipeline = Pipeline(stages=[indexer, assembler, lr])

# Fit the pipeline
model = pipeline.fit(training_data)

# Transform the data
predictions = model.transform(test_data)
```

### Model Persistence

```python
model.save("path/to/my/model")
loaded_model = PipelineModel.load("path/to/my/model")
```

## Structured Streaming

### Reading Data

```python
df = spark.readStream.format("socket") \
    .option("host", "localhost") \
    .option("port", 9999) \
    .load()
```

### Processing Data

```python
from pyspark.sql.functions import explode, split

words = df.select(explode(split(df.value, " ")).alias("word"))
wordCounts = words.groupBy("word").count()
```

### Writing Data

```python
query = wordCounts.writeStream \
    .outputMode("complete") \
    .format("console") \
    .start()

query.awaitTermination()
```

### Available Output Modes

*   `"append"`: Only new rows are written to the sink.
*   `"complete"`: All rows are written to the sink every time there are updates.
*   `"update"`: Only updated rows are written to the sink.

### Available Sinks

*   `"console"`: Prints to the console.
*   `"memory"`: Stores the output in memory.
*   `"parquet"`, `"csv"`, `"json"`, `"jdbc"`: Writes to files or databases.

## Performance Tuning

### Data Partitioning

*   Use `repartition()` or `coalesce()` to control the number of partitions.
*   Partition data based on frequently used keys.

### Caching

*   Use `cache()` or `persist()` to store intermediate results in memory or on disk.

### Broadcast Variables

*   Use `sc.broadcast()` to broadcast small datasets to all executors.

### Accumulators

*   Use `sc.accumulator()` to create global counters.

### Memory Management

*   Tune `spark.executor.memory` and `spark.driver.memory` to allocate sufficient memory.
*   Avoid creating large objects in the driver.

### Shuffle Optimization

*   Tune `spark.sql.shuffle.partitions` to control the number of shuffle partitions.
*   Use `mapPartitions` to perform operations on each partition.

### Data Serialization

*   Use Kryo serialization for better performance.

### Garbage Collection

*   Tune garbage collection settings to reduce GC overhead.

## Common Issues and Debugging

*   **Out of Memory Errors:** Increase executor memory or reduce the amount of data being processed.
*   **Slow Performance:** Analyze the Spark UI to identify bottlenecks.
*   **Serialization Errors:** Ensure that all objects being serialized are serializable.
*   **Data Skew:** Partition data to distribute it evenly across executors.
*   **Driver OOM:** Increase driver memory or reduce the amount of data being collected to the driver.

## Spark Configuration

### SparkConf Options

*   `spark.app.name`: Application name.
*   `spark.master`: Spark master URL.
*   `spark.executor.memory`: Memory per executor.
*   `spark.driver.memory`: Memory for the driver process.
*   `spark.executor.cores`: Number of cores per executor.
*   `spark.default.parallelism`: Default number of partitions.
*   `spark.sql.shuffle.partitions`: Number of partitions to use when shuffling data for joins or aggregations.
*   `spark.serializer`: Serializer class name (e.g., `org.apache.spark.serializer.KryoSerializer`).
*   `spark.driver.maxResultSize`: Maximum size of the result that the driver can collect.
*   `spark.kryoserializer.buffer.max`: Maximum buffer size for Kryo serialization.
*   `spark.sql.adaptive.enabled`: Enables adaptive query execution.
*   `spark.sql.adaptive.coalescePartitions.enabled`: Enables adaptive partition coalescing.

## Tips and Best Practices

*   Use virtual environments to isolate project dependencies.
*   Use meaningful names for variables and functions.
*   Follow the DRY (Don't Repeat Yourself) principle.
*   Write unit tests to ensure code quality.
*   Use a consistent coding style.
*   Document your code.
*   Use a version control system (e.g., Git).
*   Use appropriate data types for your data.
*   Optimize your Spark configuration for your workload.
*   Use caching to improve performance.
*   Use partitioning to distribute data evenly.
*   Avoid shuffling data unnecessarily.
*   Use broadcast variables for small datasets.
*   Use accumulators for global counters.
*   Use the Spark UI to monitor your application.
*   Use a logging framework to log events and errors.
*   Use a security framework to protect your data.
*   Use a resource manager (e.g., YARN, Mesos, Kubernetes) to manage your cluster.
*   Use a deployment tool to deploy your application to production.
*   Monitor your application for performance issues.
*   Use a CDN (Content Delivery Network) for static files.
*   Optimize database queries.
*   Use asynchronous tasks for long-running operations.
*   Implement proper logging and error handling.
*   Regularly update PySpark and its dependencies.
*   Use a security scanner to identify potential vulnerabilities.
*   Follow security best practices.
*   Use a reverse proxy like Nginx or Apache in front of your Spark application.
*   Use a load balancer for high availability.
*   Automate deployments using tools like Fabric or Ansible.
*   Use a monitoring tool like Prometheus or Grafana.
*   Implement health checks for your application.
*   Use a CDN for static assets.
*   Cache frequently accessed data.
*   Use a database connection pool.
*   Optimize your database queries.
*   Use a task queue for long-running tasks.
*   Use a background worker for asynchronous tasks.
*   Use a message queue for inter-process communication.
*   Use a service discovery tool for microservices.
*   Use a containerization tool like Docker.
*   Use an orchestration tool like Kubernetes.
*   Use Delta Lake for reliable data lakes.
*   Use Apache Arrow for faster data transfer between Python and Spark.
*   Use vectorized UDFs for better performance.
*   Use adaptive query execution (AQE) to optimize queries at runtime.
*   Use cost-based optimization (CBO) to choose the best query plan.
