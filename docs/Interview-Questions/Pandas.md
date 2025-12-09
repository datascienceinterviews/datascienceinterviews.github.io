
---
title: Pandas Interview Questions
description: 100+ Pandas interview questions for cracking Data Science and Python Developer interviews
---

# Pandas Interview Questions

<!-- [TOC] -->

This document provides a curated list of Pandas interview questions commonly asked in technical interviews for Data Science, Data Analysis, Machine Learning, and Python Developer roles. It covers fundamental concepts to advanced data manipulation techniques, including rigorous "brutally difficult" questions for senior roles.

This is updated frequently but right now this is the most exhaustive list of type of questions being asked.

---

| Sno | Question Title | Practice Links | Companies Asking | Difficulty | Topics |
|-----|----------------|----------------|------------------|------------|--------|
| 1 | What is Pandas and why is it used? | [Pandas Docs](https://pandas.pydata.org/docs/getting_started/overview.html) | Google, Amazon, Meta, Netflix | Easy | Basics, Introduction |
| 2 | Difference between Series and DataFrame | [GeeksforGeeks](https://www.geeksforgeeks.org/python-pandas-dataframe/) | Google, Amazon, Meta, Microsoft | Easy | Data Structures |
| 3 | How to create a DataFrame from dictionary? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) | Amazon, Google, Flipkart | Easy | DataFrame Creation |
| 4 | Difference between loc and iloc | [Stack Overflow](https://stackoverflow.com/questions/28757389/pandas-loc-vs-iloc-vs-at-vs-iat) | Google, Amazon, Meta, Apple, Netflix | Easy | Indexing, Selection |
| 5 | How to read CSV, Excel, JSON files? | [Pandas Docs](https://pandas.pydata.org/docs/user_guide/io.html) | Most Tech Companies | Easy | Data I/O |
| 6 | How to handle missing values (NaN)? | [Real Python](https://realpython.com/python-pandas-missing-data/) | Google, Amazon, Meta, Netflix, Apple | Medium | Missing Data, fillna, dropna |
| 7 | Difference between dropna() and fillna() | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html) | Amazon, Google, Microsoft | Easy | Missing Data |
| 8 | Explain GroupBy in Pandas | [Real Python](https://realpython.com/pandas-groupby/) | Google, Amazon, Meta, Netflix, Apple | Medium | GroupBy, Aggregation |
| 9 | How to merge two DataFrames? | [Pandas Docs](https://pandas.pydata.org/docs/user_guide/merging.html) | Google, Amazon, Meta, Microsoft | Medium | Merging, Joining |
| 10 | Difference between merge(), join(), concat() | [Stack Overflow](https://stackoverflow.com/questions/40468069/pandas-merge-join-and-concat-difference) | Google, Amazon, Meta | Medium | Merging, Joining, Concatenation |
| 11 | How to apply a function to DataFrame? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html) | Google, Amazon, Meta, Netflix | Medium | apply, applymap, map |
| 12 | Difference between apply(), map(), applymap() | [GeeksforGeeks](https://www.geeksforgeeks.org/difference-between-map-applymap-and-apply-methods-in-pandas/) | Google, Amazon, Microsoft | Medium | Data Transformation |
| 13 | How to rename columns in DataFrame? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rename.html) | Most Tech Companies | Easy | Column Operations |
| 14 | How to sort DataFrame by column values? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html) | Most Tech Companies | Easy | Sorting |
| 15 | How to filter rows based on conditions? | [Pandas Docs](https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html) | Google, Amazon, Meta, Netflix | Easy | Filtering, Boolean Indexing |
| 16 | How to remove duplicate rows? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html) | Amazon, Google, Microsoft | Easy | Data Cleaning, Deduplication |
| 17 | How to change data types of columns? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.astype.html) | Most Tech Companies | Easy | Data Types |
| 18 | What is the difference between copy() and view? | [Stack Overflow](https://stackoverflow.com/questions/27673231/what-does-the-copy-method-do-in-pandas) | Google, Amazon, Meta | Medium | Memory Management |
| 19 | Explain pivot tables in Pandas | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.pivot_table.html) | Amazon, Google, Microsoft, Netflix | Medium | Pivot Tables, Reshaping |
| 20 | Difference between pivot() and pivot_table() | [Stack Overflow](https://stackoverflow.com/questions/30960338/pandas-pivot-vs-pivot-table-what-is-the-difference) | Google, Amazon, Meta | Medium | Reshaping |
| 21 | How to handle datetime data in Pandas? | [Pandas Docs](https://pandas.pydata.org/docs/user_guide/timeseries.html) | Google, Amazon, Netflix, Meta | Medium | DateTime, Time Series |
| 22 | How to create a date range? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.date_range.html) | Amazon, Netflix, Google | Easy | DateTime |
| 23 | What is MultiIndex (Hierarchical Indexing)? | [Pandas Docs](https://pandas.pydata.org/docs/user_guide/advanced.html) | Google, Amazon, Meta | Hard | MultiIndex, Hierarchical Data |
| 24 | How to reset and set index? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.reset_index.html) | Most Tech Companies | Easy | Indexing |
| 25 | How to perform rolling window calculations? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html) | Google, Amazon, Netflix, Apple | Medium | Rolling Windows, Time Series |
| 26 | How to calculate moving averages? | [GeeksforGeeks](https://www.geeksforgeeks.org/how-to-calculate-moving-average-in-a-pandas-dataframe/) | Google, Amazon, Netflix, Apple | Medium | Rolling Windows, Finance |
| 27 | How to perform resampling on time series? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html) | Google, Amazon, Netflix | Medium | Resampling, Time Series |
| 28 | Difference between transform() and apply() | [Stack Overflow](https://stackoverflow.com/questions/27517425/apply-vs-transform-on-a-group-object) | Google, Amazon, Meta | Hard | GroupBy, Data Transformation |
| 29 | How to create bins with cut() and qcut()? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.cut.html) | Google, Amazon, Meta | Medium | Discretization, Binning |
| 30 | How to handle categorical data? | [Pandas Docs](https://pandas.pydata.org/docs/user_guide/categorical.html) | Google, Amazon, Meta, Netflix | Medium | Categorical Data, Memory |
| 31 | How to one-hot encode categorical data? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html) | Google, Amazon, Meta, Microsoft | Easy | Feature Engineering, ML |
| 32 | How to read data from SQL database? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.read_sql.html) | Amazon, Google, Microsoft | Medium | Database I/O |
| 33 | How to export DataFrame to various formats? | [Pandas Docs](https://pandas.pydata.org/docs/user_guide/io.html) | Most Tech Companies | Easy | Data Export |
| 34 | How to handle large datasets efficiently? | [Towards Data Science](https://towardsdatascience.com/why-and-how-to-use-pandas-with-large-data-9594c2b3f7f0) | Google, Amazon, Netflix, Meta | Hard | Performance, Memory Optimization |
| 35 | What is Categorical dtype and when to use it? | [Pandas Docs](https://pandas.pydata.org/docs/user_guide/categorical.html) | Google, Amazon, Meta | Medium | Data Types, Memory Optimization |
| 36 | How to optimize memory usage in Pandas? | [Medium](https://medium.com/bigdatarepublic/advanced-pandas-optimize-speed-and-memory-a654b53be4c9) | Google, Amazon, Netflix | Hard | Memory Optimization |
| 37 | Difference between inplace=True and returning copy | [Stack Overflow](https://stackoverflow.com/questions/43893457/pandas-inplace-true-vs-creating-a-new-variable) | Most Tech Companies | Easy | DataFrame Modification |
| 38 | How to use query() method for filtering? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html) | Google, Amazon, Meta | Easy | Filtering, Query |
| 39 | How to work with string data (str accessor)? | [Pandas Docs](https://pandas.pydata.org/docs/user_guide/text.html) | Google, Amazon, Meta, Netflix | Medium | String Operations |
| 40 | How to use str accessor methods? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.Series.str.html) | Amazon, Google, Microsoft | Medium | String Operations |
| 41 | How to split and expand string columns? | [GeeksforGeeks](https://www.geeksforgeeks.org/python-pandas-split-a-list-values-into-multiple-columns/) | Amazon, Google, Meta | Medium | String Operations, Data Cleaning |
| 42 | How to use melt() for unpivoting data? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.melt.html) | Google, Amazon, Meta | Medium | Reshaping, Unpivoting |
| 43 | How to use stack() and unstack()? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.stack.html) | Google, Amazon, Meta | Medium | Reshaping, MultiIndex |
| 44 | How to cross-tabulate with crosstab()? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.crosstab.html) | Google, Amazon, Meta | Medium | Cross Tabulation, Analysis |
| 45 | How to calculate correlations? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html) | Google, Amazon, Meta, Netflix | Easy | Statistical Analysis |
| 46 | How to calculate descriptive statistics? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html) | Most Tech Companies | Easy | Statistical Analysis |
| 47 | How to use agg() for multiple aggregations? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.agg.html) | Google, Amazon, Meta, Netflix | Medium | Aggregation |
| 48 | How to use named aggregations? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.agg.html) | Google, Amazon, Meta | Medium | GroupBy, Named Aggregation |
| 49 | How to handle timezone-aware datetime? | [Pandas Docs](https://pandas.pydata.org/docs/user_guide/timeseries.html#time-zone-handling) | Google, Amazon, Netflix | Medium | DateTime, Timezones |
| 50 | How to interpolate missing values? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html) | Google, Amazon, Netflix | Medium | Missing Data, Interpolation |
| 51 | How to forward fill and backward fill? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ffill.html) | Amazon, Netflix, Google | Easy | Missing Data, Time Series |
| 52 | How to use where() and mask() methods? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.where.html) | Google, Amazon, Meta | Medium | Conditional Operations |
| 53 | How to clip values in DataFrame? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.clip.html) | Amazon, Google, Meta | Easy | Data Transformation |
| 54 | How to rank values in Pandas? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rank.html) | Google, Amazon, Meta, Netflix | Easy | Ranking |
| 55 | How to calculate percentage change? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pct_change.html) | Google, Amazon, Netflix, Apple | Easy | Time Series, Finance |
| 56 | How to shift and lag data? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shift.html) | Google, Amazon, Netflix | Easy | Time Series, Lag Features |
| 57 | How to calculate cumulative statistics? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.cumsum.html) | Google, Amazon, Meta, Netflix | Easy | Cumulative Operations |
| 58 | How to use explode() for list columns? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.explode.html) | Google, Amazon, Meta | Medium | List Operations, Data Preprocessing |
| 59 | How to sample data from DataFrame? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html) | Google, Amazon, Meta, Netflix | Easy | Sampling |
| 60 | How to detect and handle outliers? | [Towards Data Science](https://towardsdatascience.com/detecting-and-treating-outliers-in-python-1c66d55f0b8d) | Google, Amazon, Meta, Netflix | Medium | Outlier Detection, Data Cleaning |
| 61 | How to normalize/standardize data? | [GeeksforGeeks](https://www.geeksforgeeks.org/python-pandas-dataframe-normalization/) | Google, Amazon, Meta, Microsoft | Medium | Feature Engineering, ML |
| 62 | How to use eval() for efficient operations? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.eval.html) | Google, Amazon, Meta | Hard | Performance Optimization |
| 63 | How to perform element-wise operations? | [Pandas Docs](https://pandas.pydata.org/docs/getting_started/intro_tutorials/05_add_columns.html) | Most Tech Companies | Easy | Arithmetic Operations |
| 64 | Why vectorized operations are faster than loops? | [Real Python](https://realpython.com/numpy-scipy-pandas-differences/) | Google, Amazon, Meta | Medium | Performance, Vectorization |
| 65 | How to profile Pandas code performance? | [Pandas Docs](https://pandas.pydata.org/docs/user_guide/enhancingperf.html) | Google, Amazon, Netflix | Hard | Performance Profiling |
| 66 | How to use pipe() for method chaining? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pipe.html) | Google, Amazon, Meta | Medium | Method Chaining |
| 67 | How to handle SettingWithCopyWarning? | [Pandas Docs](https://pandas.pydata.org/docs/user_guide/indexing.html#returning-a-view-versus-a-copy) | Google, Amazon, Meta, Microsoft | Medium | Common Errors, Debugging |
| 68 | How to compare two DataFrames? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.compare.html) | Amazon, Google, Microsoft | Medium | Data Comparison, Validation |
| 69 | How to combine DataFrames with different schemas? | [Stack Overflow](https://stackoverflow.com/questions/23668427/pandas-three-way-joining-multiple-dataframes-on-columns) | Google, Amazon, Meta | Medium | Merging, Schema Alignment |
| 70 | How to create conditional columns? | [GeeksforGeeks](https://www.geeksforgeeks.org/adding-new-column-to-existing-dataframe-in-pandas/) | Most Tech Companies | Easy | Data Transformation |
| 71 | How to use np.where() with Pandas? | [Real Python](https://realpython.com/numpy-array-programming/) | Google, Amazon, Meta, Netflix | Easy | Conditional Operations |
| 72 | How to use np.select() for multiple conditions? | [Stack Overflow](https://stackoverflow.com/questions/19913659/pandas-conditional-creation-of-a-series-dataframe-column) | Google, Amazon, Meta | Medium | Conditional Operations |
| 73 | How to count value frequencies? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html) | Most Tech Companies | Easy | Data Exploration |
| 74 | How to find unique values and nunique()? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.nunique.html) | Most Tech Companies | Easy | Data Exploration |
| 75 | How to check for null values? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isnull.html) | Most Tech Companies | Easy | Missing Data |
| 76 | How to use any() and all() methods? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.any.html) | Google, Amazon, Meta | Easy | Boolean Operations |
| 77 | How to select specific columns? | [Pandas Docs](https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html) | Most Tech Companies | Easy | Column Selection |
| 78 | How to drop columns or rows? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html) | Most Tech Companies | Easy | Data Cleaning |
| 79 | How to use assign() for creating new columns? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.assign.html) | Google, Amazon, Meta | Easy | Column Creation |
| 80 | How to use idxmax() and idxmin()? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.idxmax.html) | Google, Amazon, Meta, Netflix | Easy | Indexing |
| 81 | Why is iterating over rows slow? | [Stack Overflow](https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas) | Google, Amazon, Meta | Medium | Performance |
| 82 | How to use iterrows() and itertuples()? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iterrows.html) | Amazon, Google, Microsoft | Easy | Iteration |
| 83 | How to vectorize custom functions? | [Real Python](https://realpython.com/numpy-scipy-pandas-intro/) | Google, Amazon, Meta | Hard | Performance Optimization |
| 84 | How to use Pandas with NumPy? | [Pandas Docs](https://pandas.pydata.org/docs/user_guide/basics.html) | Google, Amazon, Meta, Netflix | Easy | NumPy Integration |
| 85 | How to flatten hierarchical index? | [Stack Overflow](https://stackoverflow.com/questions/14507794/how-to-flatten-multiindex-columns) | Google, Amazon, Meta | Medium | MultiIndex |
| 86 | How to group by multiple columns? | [Pandas Docs](https://pandas.pydata.org/docs/user_guide/groupby.html) | Most Tech Companies | Easy | GroupBy |
| 87 | How to filter groups after GroupBy? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.filter.html) | Google, Amazon, Meta | Medium | GroupBy, Filtering |
| 88 | How to get first/last n rows per group? | [Stack Overflow](https://stackoverflow.com/questions/20067636/pandas-dataframe-get-first-row-of-each-group) | Google, Amazon, Meta, Netflix | Medium | GroupBy |
| 89 | How to handle JSON with nested structures? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.json_normalize.html) | Amazon, Google, Meta | Medium | JSON Processing |
| 90 | How to read/write Parquet files? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html) | Google, Amazon, Netflix, Meta | Easy | File I/O, Big Data |
| 91 | Difference between Parquet, CSV, and Feather | [Towards Data Science](https://towardsdatascience.com/the-best-format-to-save-pandas-data-414dca023e0d) | Google, Amazon, Netflix | Medium | File Formats, Performance |
| 92 | How to use chunksize for large files? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html) | Google, Amazon, Netflix, Meta | Medium | Large Data Processing |
| 93 | How to use nsmallest() and nlargest()? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.nlargest.html) | Google, Amazon, Meta | Easy | Selection |
| 94 | How to calculate weighted average? | [Stack Overflow](https://stackoverflow.com/questions/10951341/pandas-dataframe-aggregate-function-using-multiple-columns) | Google, Amazon, Netflix, Apple | Medium | Aggregation, Finance |
| 95 | How to perform window functions like SQL? | [Pandas Docs](https://pandas.pydata.org/docs/user_guide/window.html) | Google, Amazon, Meta, Netflix | Medium | Window Functions |
| 96 | How to join on nearest key (asof join)? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.merge_asof.html) | Google, Amazon, Netflix, Apple | Hard | Joining, Time Series |
| 97 | How to use combine_first() for data merging? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.combine_first.html) | Amazon, Google, Microsoft | Medium | Merging |
| 98 | How to create period indices? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.period_range.html) | Google, Amazon, Netflix | Medium | Time Series |
| 99 | How to use Timedelta for time differences? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.html) | Google, Amazon, Netflix | Easy | DateTime |
| 100 | How to set display options globally? | [Pandas Docs](https://pandas.pydata.org/docs/user_guide/options.html) | Most Tech Companies | Easy | Display Options |
| 101 | What is method chaining and when to use it? | [Tom Augspurger Blog](https://tomaugspurger.github.io/method-chaining.html) | Google, Amazon, Meta | Medium | Method Chaining, Clean Code |
| 102 | How to calculate month-over-month change? | [StrataScratch](https://www.stratascratch.com/) | Google, Amazon, Meta, Netflix | Medium | Time Series, Analytics |
| 103 | How to find customers with highest orders? | [DataLemur](https://datalemur.com/) | Amazon, Google, Meta, Netflix | Medium | GroupBy, Aggregation |
| 104 | **[HARD]** How to calculate retention metrics efficiently? | [StrataScratch](https://www.stratascratch.com/) | Meta, Netflix, Amazon, Google | Hard | Cohort Analysis, Time Series |
| 105 | **[HARD]** How to implement A/B test analysis? | [Towards Data Science](https://towardsdatascience.com/) | Meta, Google, Netflix, Amazon | Hard | Statistical Analysis, Testing |
| 106 | **[HARD]** How to optimize memory with `category` types? | [Pandas Docs](https://pandas.pydata.org/docs/user_guide/categorical.html) | Google, Amazon, Netflix | Hard | Memory Optimization |
| 107 | **[HARD]** How to implement cohort analysis? | [Towards Data Science](https://towardsdatascience.com/) | Meta, Netflix, Amazon, Google | Hard | Cohort Analysis |
| 108 | **[HARD]** How to calculate funnel drop-off rates? | [StrataScratch](https://www.stratascratch.com/) | Meta, Google, Amazon, Netflix | Hard | Funnel Analysis, Analytics |
| 109 | **[HARD]** How to implement custom testing using `assert_frame_equal`? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.testing.assert_frame_equal.html) | Google, Amazon, Microsoft | Hard | Testing, Quality |
| 110 | **[HARD]** How to handle sparse data structures? | [Pandas Docs](https://pandas.pydata.org/docs/user_guide/sparse.html) | Google, Amazon, Netflix | Hard | Sparse Data, Memory |
| 111 | **[HARD]** How to use Numba/JIT with Pandas? | [Pandas Docs](https://pandas.pydata.org/docs/user_guide/enhancingperf.html) | Google, Amazon, Hedge Funds | Hard | Performance |
| 112 | **[HARD]** How to implement custom accessors? | [Pandas Docs](https://pandas.pydata.org/docs/development/extending.html) | Google, Amazon, Meta | Hard | Extending Pandas |
| 113 | **[HARD]** How to use Swifter for parallel processing? | [Swifter Docs](https://github.com/jmcarpenter2/swifter) | Google, Amazon, Uber | Hard | Parallelism |
| 114 | **[HARD]** Explain Pandas Block Manager structure | [Pandas Wiki](https://github.com/pandas-dev/pandas/wiki) | Google, Amazon, Meta | Hard | Internals |
| 115 | **[HARD]** How Copy-on-Write (CoW) works in Pandas 2.0+? | [Pandas Docs](https://pandas.pydata.org/docs/user_guide/copy_on_write.html) | Google, Meta, Microsoft | Hard | Internals, Performance |
| 116 | **[HARD]** How to use PyArrow backend for performance? | [Pandas Docs](https://pandas.pydata.org/docs/user_guide/pyarrow.html) | Google, Amazon, Databricks | Hard | Performance, Arrow |
| 117 | **[HARD]** How to implement custom index types? | [Pandas Docs](https://pandas.pydata.org/docs/development/extending.html) | Google, Amazon | Hard | Extending Pandas |
| 118 | **[HARD]** How to optimize MultiIndex slicing performance? | [Pandas Docs](https://pandas.pydata.org/docs/user_guide/advanced.html) | Google, Amazon, Hedge Funds | Hard | Optimization |
| 119 | **[HARD]** `groupby().transform()` internal mechanics vs `apply()` | [Pandas Docs](https://pandas.pydata.org/docs/user_guide/groupby.html) | Google, Amazon, Meta | Hard | Deep Dive |
| 120 | **[HARD]** How to implement rolling window with `raw=True`? | [Pandas Docs](https://pandas.pydata.org/docs/reference/api/pandas.core.window.rolling.Rolling.apply.html) | Google, Amazon, Hedge Funds | Hard | Optimization |
| 121 | **[HARD]** How to extend Pandas with custom plotting backends? | [Pandas Docs](https://pandas.pydata.org/docs/development/extending.html) | Google, Amazon | Hard | Extending Pandas |
| 122 | **[HARD]** How to handle time series offset aliases? | [Pandas Docs](https://pandas.pydata.org/docs/user_guide/timeseries.html) | Google, Amazon, Hedge Funds | Hard | Time Series |
| 123 | **[HARD]** How to use Dask DataFrames for out-of-core computing? | [Dask Docs](https://docs.dask.org/en/stable/dataframe.html) | Google, Amazon, Netflix | Hard | Big Data |
| 124 | **[HARD]** How to optimize chained assignment performance? | [Pandas Docs](https://pandas.pydata.org/docs/user_guide/indexing.html) | Google, Amazon, Meta | Hard | Optimization |
| 125 | **[HARD]** Nullable integers/floats implementation? | [Pandas Docs](https://pandas.pydata.org/docs/user_guide/integer_na.html) | Google, Amazon, Microsoft | Hard | Internals |
| 126 | **[HARD]** How to use Cython with Pandas? | [Pandas Docs](https://pandas.pydata.org/docs/user_guide/enhancingperf.html) | Google, Amazon, HFT Firms | Hard | Performance |
| 127 | **[HARD]** Comparison of Parquet vs Feather vs ORC? | [Apache Arrow](https://arrow.apache.org/) | Google, Amazon, Netflix | Hard | Systems |

---

## Code Examples

### 1. Memory Optimization
```python
import pandas as pd
import numpy as np

# Typical large dataframe creation
df = pd.DataFrame({
    'category': np.random.choice(['A', 'B', 'C'], size=1000000),
    'value': np.random.randn(1000000)
})

# Memory usage before optimization
print(df.memory_usage(deep=True).sum() / 1024**2, "MB")

# Optimize by converting object to category
df['category'] = df['category'].astype('category')

# Memory usage after optimization
print(df.memory_usage(deep=True).sum() / 1024**2, "MB")
```

### 2. Method Chaining for Clean Code
```python
# Instead of multiple intermediate variables
df = (
    pd.read_csv('data.csv')
    .query('status == "active"')
    .assign(
        year=lambda x: pd.to_datetime(x['date']).dt.year,
        total_cost=lambda x: x['price'] * x['quantity']
    )
    .groupby(['year', 'region'])
    .agg(total_revenue=('total_cost', 'sum'))
    .reset_index()
    .sort_values('total_revenue', ascending=False)
)
```

### 3. Parallel Processing with Swifter
```python
import pandas as pd
import swifter

df = pd.DataFrame({'text': ['some text'] * 100000})

def heavy_processing(text):
    # Simulate heavy work
    return text.upper()[::-1]

# Automatic parallelization
df['processed'] = df['text'].swifter.apply(heavy_processing)
```

---

## Questions asked in Google interview
- How would you optimize a Pandas operation running slowly on large dataset?
- Explain the difference between merge() and join()
- Write code to calculate rolling averages with different window sizes
- How would you handle a DataFrame with 100 million rows?
- Explain memory optimization techniques
- Write code to perform complex GroupBy with multiple aggregations
- Explain the internal data structure of DataFrame
- How would you implement feature engineering pipelines?
- Write code to calculate year-over-year growth
- Explain vectorized operations and their importance
- How to handle SettingWithCopyWarning?
- Write code to perform window functions similar to SQL

## Questions asked in Amazon interview
- Write code to merge multiple DataFrames with different schemas
- How would you calculate year-over-year growth?
- Explain how to handle time series data with irregular intervals
- Write code to identify and remove duplicate records
- How would you implement a moving average crossover strategy?
- Explain the difference between transform() and apply()
- Write code to pivot data for sales analysis
- How would you handle categorical variables with high cardinality?
- Explain how to optimize for memory efficiency
- Write code to perform cohort analysis

## Questions asked in Meta interview
- Write code to analyze user engagement data
- How would you calculate conversion funnels?
- Explain how to handle large-scale data processing
- Write code to resample time series data
- How would you implement A/B testing analysis?
- Explain method chaining and its benefits
- Write code to calculate retention metrics
- How would you handle hierarchical data structures?
- Explain vectorization benefits over loops
- Write code to analyze network data

## Questions asked in Microsoft interview
- Explain the SettingWithCopyWarning and how to avoid it
- Write code to perform window functions similar to SQL
- How would you handle timezone conversions?
- Explain the difference between views and copies
- Write code to implement custom aggregation functions
- How would you optimize Pandas for production?
- Explain multi-level indexing use cases
- Write code to compare two DataFrames
- How would you handle missing data in time series?
- Explain eval() and query() methods

## Questions asked in Netflix interview
- Write code to analyze viewing patterns and user behavior
- How would you calculate streaming quality metrics?
- Explain how to handle messy data from multiple sources
- Write code to implement collaborative filtering preprocessing
- How would you analyze content performance across regions?
- Explain time series decomposition
- Write code to calculate customer lifetime value
- How would you handle data for recommendation systems?
- Explain rolling window calculations for real-time analytics
- Write code to analyze A/B test results

## Questions asked in Apple interview
- Write code to perform data validation on imported data
- How would you implement data quality checks?
- Explain how to handle multi-format data imports
- Write code to analyze product performance metrics
- How would you implement data anonymization?
- Explain best practices for production Pandas code
- Write code to create automated data reports
- How would you handle data versioning?
- Explain memory management for large DataFrames
- Write code to implement time-based partitioning

## Questions asked in Flipkart interview
- Write code to analyze e-commerce transaction data
- How would you calculate GMV metrics?
- Explain handling high-cardinality categorical data
- Write code to analyze customer purchase patterns
- How would you implement product recommendation preprocessing?
- Explain data aggregation for dashboard analytics

## Questions asked in LinkedIn interview
- Write code to analyze professional network connections
- How would you calculate engagement metrics for posts?
- Explain how to handle user activity data
- Write code to implement skill-based matching
- How would you analyze job posting performance?
- Explain data preprocessing for NLP tasks

---

## Additional Resources

- [Official Pandas Documentation](https://pandas.pydata.org/docs/)
- [Minimally Sufficient Pandas](https://medium.com/dunder-data/minimally-sufficient-pandas-a8e67f2a2428)
- [Modern Pandas](https://tomaugspurger.github.io/modern-1-intro.html)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
- [Effective Pandas (Book)](https://store.metasnake.com/effective-pandas-book)
