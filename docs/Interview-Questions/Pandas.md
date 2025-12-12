
---
title: Pandas Interview Questions
description: 100+ Pandas interview questions for cracking Data Science and Python Developer interviews
---

# Pandas Interview Questions

<!-- [TOC] -->

This document provides a curated list of Pandas interview questions commonly asked in technical interviews for Data Science, Data Analysis, Machine Learning, and Python Developer roles. It covers fundamental concepts to advanced data manipulation techniques, including rigorous "brutally difficult" questions for senior roles.

This is updated frequently but right now this is the most exhaustive list of type of questions being asked.

---

## Premium Interview Questions

### Explain loc vs iloc - Key Difference - Google, Amazon, Meta Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Indexing`, `Selection`, `Core` | **Asked by:** Google, Amazon, Meta, Apple, Netflix

??? success "View Answer"

    **loc vs iloc:**
    
    | Method | Type | Example |
    |--------|------|---------|
    | `loc` | Label-based | `df.loc['row_label', 'col_name']` |
    | `iloc` | Integer position | `df.iloc[0, 1]` |
    
    **Examples:**
    
    ```python
    import pandas as pd
    
    df = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'city': ['NY', 'LA', 'SF']
    }, index=['a', 'b', 'c'])
    
    # loc - uses labels
    df.loc['a', 'name']           # 'Alice'
    df.loc['a':'b', 'name':'age'] # Rows a-b, columns name-age
    df.loc[df['age'] > 25]        # Boolean filtering
    
    # iloc - uses integer positions
    df.iloc[0, 0]                 # 'Alice' (first row, first col)
    df.iloc[0:2, 0:2]             # First 2 rows, first 2 cols
    df.iloc[[0, 2], [0, 1]]       # Specific rows and cols
    ```
    
    **Key Difference:**
    
    - `loc` includes end of slice: `df.loc['a':'c']` includes 'c'
    - `iloc` excludes end: `df.iloc[0:2]` excludes index 2
    
    **Common Mistake:**
    
    ```python
    # WRONG: mixing loc with integers on non-integer index
    df.loc[0]  # KeyError if index is ['a', 'b', 'c']
    
    # CORRECT
    df.iloc[0]  # Always works with position
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** DataFrame navigation fundamentals.
        
        **Strong answer signals:**
        
        - Knows loc is inclusive, iloc exclusive
        - Can use boolean indexing with loc
        - Avoids common mistakes
        - Mentions at/iat for scalar access

---

### How Do You Handle Missing Values in Pandas? - Google, Amazon, Netflix Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Missing Data`, `Data Cleaning`, `fillna` | **Asked by:** Google, Amazon, Meta, Netflix, Apple

??? success "View Answer"

    **Detecting Missing Values:**
    
    ```python
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame({
        'A': [1, np.nan, 3, np.nan],
        'B': ['x', None, 'y', 'z']
    })
    
    # Detection
    df.isnull()              # Boolean mask
    df.isnull().sum()        # Count per column
    df.isnull().sum().sum()  # Total count
    df.isna().any()          # Any missing per column
    ```
    
    **Handling Strategies:**
    
    ```python
    # 1. Drop rows with any missing
    df.dropna()
    
    # 2. Drop rows where specific columns are missing
    df.dropna(subset=['A'])
    
    # 3. Drop only if all values missing
    df.dropna(how='all')
    
    # 4. Fill with constant
    df.fillna(0)
    df.fillna({'A': 0, 'B': 'unknown'})
    
    # 5. Fill with statistics
    df['A'].fillna(df['A'].mean())
    df['A'].fillna(df['A'].median())
    df['A'].fillna(df['A'].mode()[0])
    
    # 6. Forward/backward fill (time series)
    df.ffill()  # Forward fill
    df.bfill()  # Backward fill
    
    # 7. Interpolation
    df['A'].interpolate(method='linear')
    ```
    
    **Best Practices:**
    
    | Scenario | Strategy |
    |----------|----------|
    | Random missing | Mean/median imputation |
    | Time series | Forward fill or interpolate |
    | Categorical | Mode or 'Unknown' category |
    | Many missing | Consider dropping column |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Data cleaning expertise.
        
        **Strong answer signals:**
        
        - Chooses strategy based on data type
        - Knows dropna vs fillna trade-offs
        - Uses appropriate interpolation for time series
        - Considers impact on analysis

---

### Explain GroupBy in Pandas - Split-Apply-Combine - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `GroupBy`, `Aggregation`, `Split-Apply-Combine` | **Asked by:** Google, Amazon, Meta, Netflix, Apple

??? success "View Answer"

    **GroupBy Concept:**
    
    1. **Split**: Divide data into groups
    2. **Apply**: Apply function to each group
    3. **Combine**: Combine results
    
    ```python
    import pandas as pd
    
    df = pd.DataFrame({
        'category': ['A', 'A', 'B', 'B', 'A'],
        'product': ['x', 'y', 'x', 'y', 'x'],
        'sales': [100, 150, 200, 250, 120],
        'quantity': [10, 15, 20, 25, 12]
    })
    
    # Basic groupby
    df.groupby('category')['sales'].sum()
    
    # Multiple columns
    df.groupby(['category', 'product'])['sales'].mean()
    
    # Multiple aggregations
    df.groupby('category').agg({
        'sales': ['sum', 'mean', 'count'],
        'quantity': 'sum'
    })
    
    # Named aggregations (cleaner output)
    df.groupby('category').agg(
        total_sales=('sales', 'sum'),
        avg_sales=('sales', 'mean'),
        order_count=('sales', 'count')
    )
    
    # Custom functions
    df.groupby('category')['sales'].apply(lambda x: x.max() - x.min())
    ```
    
    **Transform vs Apply:**
    
    ```python
    # Transform: returns same shape (broadcast back)
    df['sales_normalized'] = df.groupby('category')['sales'].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    
    # Apply: returns any shape
    df.groupby('category').apply(lambda g: g.nlargest(2, 'sales'))
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Data aggregation skills.
        
        **Strong answer signals:**
        
        - Explains split-apply-combine
        - Uses named aggregations
        - Knows transform vs apply
        - Can handle multi-level groupby

---

### Difference Between merge(), join(), and concat() - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Merging`, `Joining`, `Concatenation` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Comparison:**
    
    | Method | Use Case | Key Difference |
    |--------|----------|---------------|
    | `merge()` | SQL-like joins on columns | Column-based |
    | `join()` | Join on index | Index-based |
    | `concat()` | Stack DataFrames | No key matching |
    
    **Examples:**
    
    ```python
    import pandas as pd
    
    df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'val1': [1, 2, 3]})
    df2 = pd.DataFrame({'key': ['A', 'B', 'D'], 'val2': [4, 5, 6]})
    
    # MERGE - column-based joining
    pd.merge(df1, df2, on='key', how='inner')  # Only A, B
    pd.merge(df1, df2, on='key', how='left')   # Keep all from df1
    pd.merge(df1, df2, on='key', how='outer')  # Keep all
    pd.merge(df1, df2, left_on='key', right_on='key')  # Different column names
    
    # JOIN - index-based (set index first)
    df1.set_index('key').join(df2.set_index('key'))
    
    # CONCAT - stacking
    pd.concat([df1, df2])              # Vertical (default axis=0)
    pd.concat([df1, df2], axis=1)      # Horizontal
    pd.concat([df1, df2], ignore_index=True)  # Reset index
    ```
    
    **Join Types:**
    
    ```
    Inner: Only matching keys
    Left:  All from left + matching from right
    Right: All from right + matching from left
    Outer: All from both (union)
    ```
    
    **Indicator for debugging:**
    
    ```python
    result = pd.merge(df1, df2, on='key', how='outer', indicator=True)
    # _merge column shows: 'left_only', 'right_only', 'both'
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Data integration skills.
        
        **Strong answer signals:**
        
        - Knows when to use each method
        - Can explain join types
        - Uses indicator for debugging
        - Handles different column names

---

### How to Apply Functions to DataFrames? apply(), map(), applymap() - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Data Transformation`, `Functions` | **Asked by:** Google, Amazon, Microsoft

??? success "View Answer"

    **Method Comparison:**
    
    | Method | Scope | Use Case |
    |--------|-------|----------|
    | `apply()` | Row/Column | Complex transformations |
    | `map()` | Series only | Element-wise mapping |
    | `applymap()` | Element-wise | Simple element ops (deprecated, use `map`) |
    
    **Examples:**
    
    ```python
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': ['x', 'y', 'z']
    })
    
    # apply() - works on columns or rows
    df[['A', 'B']].apply(np.sum)           # Sum each column
    df[['A', 'B']].apply(np.sum, axis=1)   # Sum each row
    df[['A', 'B']].apply(lambda x: x.max() - x.min())  # Custom function
    
    # map() - Series only, element-wise
    df['C'].map({'x': 'X', 'y': 'Y', 'z': 'Z'})  # Dictionary mapping
    df['A'].map(lambda x: x ** 2)                 # Lambda function
    
    # For DataFrames, use apply with axis
    df[['A', 'B']].apply(lambda x: x ** 2)  # Each column
    
    # In Pandas 2.1+, use map() instead of applymap()
    df[['A', 'B']].map(lambda x: x * 2)
    ```
    
    **Performance Tip:**
    
    ```python
    # SLOW - apply with lambda
    df['A'].apply(lambda x: x ** 2)
    
    # FAST - vectorized operation
    df['A'] ** 2
    
    # Use apply only when vectorization isn't possible
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Data transformation proficiency.
        
        **Strong answer signals:**
        
        - Prefers vectorized operations
        - Knows applymap deprecated in 2.1+
        - Uses map for Series, apply for DataFrame
        - Understands axis parameter

---

### How to Optimize Memory Usage in Pandas? - Google, Amazon, Netflix Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Memory Optimization`, `Performance` | **Asked by:** Google, Amazon, Netflix

??? success "View Answer"

    **Check Memory Usage:**
    
    ```python
    import pandas as pd
    
    df = pd.read_csv('large_file.csv')
    
    # Memory per column
    df.memory_usage(deep=True)
    
    # Total memory in MB
    df.memory_usage(deep=True).sum() / 1024**2
    ```
    
    **Optimization Techniques:**
    
    ```python
    # 1. Use appropriate dtypes on read
    df = pd.read_csv('file.csv', dtype={
        'id': 'int32',           # Instead of int64
        'value': 'float32',      # Instead of float64
        'category': 'category'   # Instead of object
    })
    
    # 2. Convert existing columns
    df['category'] = df['category'].astype('category')
    
    # 3. Downcast numeric types
    df['int_col'] = pd.to_numeric(df['int_col'], downcast='integer')
    df['float_col'] = pd.to_numeric(df['float_col'], downcast='float')
    
    # 4. Use sparse dtypes for mostly-empty columns
    sparse_col = pd.arrays.SparseArray([0, 0, 0, 1, 0, 0, 0, 0])
    ```
    
    **Memory Reduction Example:**
    
    ```python
    def reduce_memory(df):
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type == 'object':
                if df[col].nunique() / len(df) < 0.5:  # Low cardinality
                    df[col] = df[col].astype('category')
            
            elif col_type == 'int64':
                df[col] = pd.to_numeric(df[col], downcast='integer')
            
            elif col_type == 'float64':
                df[col] = pd.to_numeric(df[col], downcast='float')
        
        return df
    ```
    
    **Savings:**
    
    | From | To | Reduction |
    |------|-----|-----------|
    | int64 | int32 | 50% |
    | float64 | float32 | 50% |
    | object (strings) | category | 90%+ |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Production-ready skills.
        
        **Strong answer signals:**
        
        - Knows category dtype for strings
        - Uses downcast for numerics
        - Checks memory before/after
        - Considers trade-offs (precision)

---

### How to Handle Large Datasets That Don't Fit in Memory? - Google, Amazon, Netflix Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Large Data`, `Chunking`, `Performance` | **Asked by:** Google, Amazon, Netflix, Meta

??? success "View Answer"

    **Chunked Processing:**
    
    ```python
    import pandas as pd
    
    # Process file in chunks
    chunk_size = 100_000
    chunks = []
    
    for chunk in pd.read_csv('huge_file.csv', chunksize=chunk_size):
        # Process each chunk
        processed = chunk.groupby('category')['value'].sum()
        chunks.append(processed)
    
    # Combine results
    result = pd.concat(chunks).groupby(level=0).sum()
    ```
    
    **Use Efficient File Formats:**
    
    ```python
    # Parquet - columnar, compressed
    df.to_parquet('data.parquet', compression='snappy')
    df = pd.read_parquet('data.parquet', columns=['col1', 'col2'])  # Read subset
    
    # Feather - fast read/write
    df.to_feather('data.feather')
    df = pd.read_feather('data.feather')
    ```
    
    **Dask for Out-of-Core Computing:**
    
    ```python
    import dask.dataframe as dd
    
    # Lazy loading - doesn't load into memory
    ddf = dd.read_csv('huge_file.csv')
    
    # Same Pandas API
    result = ddf.groupby('category')['value'].sum().compute()
    ```
    
    **PyArrow Backend (Pandas 2.0+):**
    
    ```python
    # Use PyArrow for better memory efficiency
    df = pd.read_csv('file.csv', dtype_backend='pyarrow')
    
    # Or convert existing
    df = df.convert_dtypes(dtype_backend='pyarrow')
    ```
    
    **Comparison:**
    
    | Approach | Use Case |
    |----------|----------|
    | Chunking | Simple aggregations |
    | Parquet | Columnar queries |
    | Dask | Complex operations |
    | Polars | Speed-critical |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Big data handling.
        
        **Strong answer signals:**
        
        - Knows chunking for simple cases
        - Uses Parquet for columnar access
        - Mentions Dask/Polars for scale
        - Understands memory vs I/O trade-offs

---

### Explain Pivot Tables in Pandas - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Pivot Tables`, `Reshaping` | **Asked by:** Amazon, Google, Microsoft, Netflix

??? success "View Answer"

    **pivot_table() - Flexible Reshaping:**
    
    ```python
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame({
        'date': ['2023-01', '2023-01', '2023-02', '2023-02'],
        'product': ['A', 'B', 'A', 'B'],
        'region': ['East', 'East', 'West', 'West'],
        'sales': [100, 150, 200, 250],
        'quantity': [10, 15, 20, 25]
    })
    
    # Basic pivot table
    pt = pd.pivot_table(
        df,
        values='sales',
        index='date',
        columns='product',
        aggfunc='sum'
    )
    
    # Multiple aggregations
    pt = pd.pivot_table(
        df,
        values=['sales', 'quantity'],
        index='date',
        columns='product',
        aggfunc={'sales': 'sum', 'quantity': 'mean'}
    )
    
    # With totals
    pt = pd.pivot_table(
        df,
        values='sales',
        index='date',
        columns='product',
        aggfunc='sum',
        margins=True,
        margins_name='Total'
    )
    ```
    
    **pivot() vs pivot_table():**
    
    | | pivot() | pivot_table() |
    |-|---------|---------------|
    | Duplicates | Error | Handles with aggfunc |
    | Aggregation | No | Yes |
    | Fill value | No | Yes |
    
    ```python
    # pivot() - simple reshape (no duplicates allowed)
    df.pivot(index='date', columns='product', values='sales')
    
    # pivot_table() - handles duplicates
    df.pivot_table(index='date', columns='product', values='sales', aggfunc='mean')
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Data reshaping skills.
        
        **Strong answer signals:**
        
        - Uses pivot_table for aggregation
        - Knows margins for totals
        - Understands when to use each
        - Can reverse with melt()

---

### How to Work with DateTime Data in Pandas? - Google, Amazon, Netflix Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `DateTime`, `Time Series` | **Asked by:** Google, Amazon, Netflix, Meta

??? success "View Answer"

    **Creating DateTime:**
    
    ```python
    import pandas as pd
    
    # Parse strings to datetime
    df['date'] = pd.to_datetime(df['date_string'])
    df['date'] = pd.to_datetime(df['date_string'], format='%Y-%m-%d')
    
    # Create date range
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    dates = pd.date_range(start='2023-01-01', periods=12, freq='ME')
    ```
    
    **DateTime Accessors (.dt):**
    
    ```python
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract components
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek  # 0=Monday
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = df['date'].dt.dayofweek >= 5
    
    # Formatting
    df['month_name'] = df['date'].dt.month_name()
    df['formatted'] = df['date'].dt.strftime('%Y-%m')
    ```
    
    **Time Zone Handling:**
    
    ```python
    # Localize (no timezone -> timezone)
    df['date'] = df['date'].dt.tz_localize('UTC')
    
    # Convert between timezones
    df['date_est'] = df['date'].dt.tz_convert('US/Eastern')
    ```
    
    **Resampling:**
    
    ```python
    # Resample time series
    df.set_index('date').resample('M')['sales'].sum()  # Monthly
    df.set_index('date').resample('Q')['sales'].mean()  # Quarterly
    df.set_index('date').resample('W')['sales'].agg(['sum', 'mean'])
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Time series manipulation.
        
        **Strong answer signals:**
        
        - Uses .dt accessor for components
        - Handles timezones correctly
        - Knows resampling frequencies
        - Can calculate time differences

---

### What is SettingWithCopyWarning and How to Avoid It? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Common Errors`, `Best Practices` | **Asked by:** Google, Amazon, Meta, Microsoft

??? success "View Answer"

    **The Problem:**
    
    ```python
    import pandas as pd
    
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    
    # This triggers warning
    filtered = df[df['A'] > 1]
    filtered['B'] = 100  # Warning: setting on copy
    ```
    
    **Why It Happens:**
    
    Pandas can't always tell if you're working with a view or a copy.
    Modifying a view might not update the original (or might unexpectedly).
    
    **Solutions:**
    
    ```python
    # Solution 1: Explicit copy
    filtered = df[df['A'] > 1].copy()
    filtered['B'] = 100  # Safe
    
    # Solution 2: Use .loc for assignment
    df.loc[df['A'] > 1, 'B'] = 100  # Modifies original
    
    # Solution 3: Chain in one line
    df = df[df['A'] > 1].assign(B=100)
    
    # Pandas 2.0+ with Copy-on-Write
    pd.options.mode.copy_on_write = True
    filtered = df[df['A'] > 1]
    filtered['B'] = 100  # Creates copy automatically
    ```
    
    **Best Practices:**
    
    | Want to... | Use |
    |------------|-----|
    | Modify original | `df.loc[condition, 'col'] = value` |
    | Create new DataFrame | `df[condition].copy()` |
    | Chain operations | `.assign()` method |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of views vs copies.
        
        **Strong answer signals:**
        
        - Explains view vs copy concept
        - Uses .loc for in-place modification
        - Uses .copy() when needed
        - Knows Copy-on-Write in Pandas 2.0

---

### How to Use Rolling Windows for Time Series? - Google, Amazon, Netflix Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Rolling Windows`, `Time Series`, `Finance` | **Asked by:** Google, Amazon, Netflix, Apple

??? success "View Answer"

    **Rolling Window Calculations:**
    
    ```python
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=100),
        'price': np.random.randn(100).cumsum() + 100
    })
    df.set_index('date', inplace=True)
    
    # Moving averages
    df['MA_7'] = df['price'].rolling(window=7).mean()
    df['MA_30'] = df['price'].rolling(window=30).mean()
    
    # Other statistics
    df['rolling_std'] = df['price'].rolling(7).std()
    df['rolling_max'] = df['price'].rolling(7).max()
    df['rolling_sum'] = df['price'].rolling(7).sum()
    
    # Minimum periods (handle NaN at start)
    df['MA_7_min3'] = df['price'].rolling(window=7, min_periods=3).mean()
    
    # Centered window
    df['MA_centered'] = df['price'].rolling(window=7, center=True).mean()
    ```
    
    **Custom Rolling Functions:**
    
    ```python
    # Custom function with apply
    df['rolling_range'] = df['price'].rolling(7).apply(
        lambda x: x.max() - x.min()
    )
    
    # Faster with raw=True (NumPy array)
    df['rolling_custom'] = df['price'].rolling(7).apply(
        lambda x: np.percentile(x, 75), raw=True
    )
    ```
    
    **Exponential Weighted Average:**
    
    ```python
    # EMA - more weight to recent values
    df['EMA_7'] = df['price'].ewm(span=7).mean()
    df['EMA_decay'] = df['price'].ewm(alpha=0.1).mean()
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Time series analysis skills.
        
        **Strong answer signals:**
        
        - Uses min_periods for edge cases
        - Knows EWM for exponential weighting
        - Uses raw=True for performance
        - Can implement trading signals

---

### How to Efficiently Use Query and Eval? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Filtering`, `Query`, `Performance` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **query() - String-Based Filtering:**
    
    ```python
    import pandas as pd
    
    df = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'department': ['Sales', 'IT', 'Sales'],
        'salary': [50000, 60000, 55000]
    })
    
    # Standard filtering (verbose)
    df[(df['age'] > 25) & (df['department'] == 'Sales')]
    
    # query() - cleaner syntax
    df.query('age > 25 and department == "Sales"')
    
    # Using variables with @
    min_age = 25
    dept = 'Sales'
    df.query('age > @min_age and department == @dept')
    
    # Column names with spaces
    df.query('`Column Name` > 10')
    ```
    
    **eval() - Efficient Expression Evaluation:**
    
    ```python
    # Create new column without intermediate copies
    df.eval('bonus = salary * 0.1')
    
    # Multiple expressions
    df.eval('''
        bonus = salary * 0.1
        total_comp = salary + bonus
        age_group = age // 10 * 10
    ''', inplace=True)
    
    # Conditional expressions
    df.eval('is_senior = age >= 30')
    ```
    
    **Performance:**
    
    ```python
    # eval uses numexpr for large DataFrames
    # Faster for: large datasets, complex expressions
    # Similar for: small datasets, simple operations
    
    # Check if numexpr is available
    import pandas as pd
    print(pd.get_option('compute.use_numexpr'))
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Clean code and performance.
        
        **Strong answer signals:**
        
        - Uses query for readable filters
        - Uses @ for variable interpolation
        - Knows eval for complex expressions
        - Understands when it's faster

---

### How to Work with String Data in Pandas? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `String Operations`, `Text Processing` | **Asked by:** Google, Amazon, Meta, Netflix

??? success "View Answer"

    **String Accessor (.str):**
    
    ```python
    import pandas as pd
    
    df = pd.DataFrame({
        'name': ['John Smith', 'Jane Doe', 'Bob Johnson'],
        'email': ['john@example.com', 'jane@test.com', 'bob@sample.org']
    })
    
    # Case transformations
    df['name'].str.lower()
    df['name'].str.upper()
    df['name'].str.title()
    
    # String matching
    df['name'].str.contains('John')
    df['name'].str.startswith('J')
    df['email'].str.endswith('.com')
    
    # Extraction
    df['first_name'] = df['name'].str.split(' ').str[0]
    df['domain'] = df['email'].str.extract(r'@(\w+\.\w+)')
    
    # Replacement
    df['name'].str.replace('John', 'Jonathan')
    df['email'].str.replace(r'@\w+', '@company', regex=True)
    
    # Length and padding
    df['name'].str.len()
    df['name'].str.pad(width=20, side='right', fillchar='.')
    ```
    
    **Split and Expand:**
    
    ```python
    # Split into multiple columns
    df[['first', 'last']] = df['name'].str.split(' ', expand=True)
    
    # Split into list (no expand)
    df['name_parts'] = df['name'].str.split(' ')
    ```
    
    **Regular Expressions:**
    
    ```python
    # Extract groups
    df['phone'] = pd.Series(['123-456-7890', '987-654-3210'])
    df[['area', 'exchange', 'number']] = df['phone'].str.extract(
        r'(\d{3})-(\d{3})-(\d{4})'
    )
    
    # Find all matches
    df['numbers'] = df['phone'].str.findall(r'\d+')
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Text data handling.
        
        **Strong answer signals:**
        
        - Uses .str accessor consistently
        - Knows expand parameter for split
        - Can write regex patterns
        - Handles edge cases (NaN, empty)

---

### Difference Between transform() and apply() in GroupBy - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `GroupBy`, `Data Transformation` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Key Difference:**
    
    | Method | Output Shape | Broadcast |
    |--------|--------------|-----------|
    | `transform()` | Same as input | Yes |
    | `apply()` | Any shape | No |
    
    **transform() - Broadcasts Back:**
    
    ```python
    import pandas as pd
    
    df = pd.DataFrame({
        'group': ['A', 'A', 'B', 'B'],
        'value': [10, 20, 30, 40]
    })
    
    # Normalize within groups
    df['normalized'] = df.groupby('group')['value'].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    
    # Add group statistics as new column
    df['group_mean'] = df.groupby('group')['value'].transform('mean')
    df['group_sum'] = df.groupby('group')['value'].transform('sum')
    df['pct_of_group'] = df['value'] / df.groupby('group')['value'].transform('sum')
    
    # Result: same number of rows as original
    ```
    
    **apply() - Flexible Output:**
    
    ```python
    # Return aggregated result
    df.groupby('group')['value'].apply(lambda x: x.sum())
    
    # Return different shape per group
    df.groupby('group').apply(lambda g: g.nlargest(1, 'value'))
    
    # Return multiple values per group
    df.groupby('group')['value'].apply(lambda x: pd.Series({
        'mean': x.mean(),
        'range': x.max() - x.min()
    }))
    ```
    
    **When to Use Each:**
    
    | Use Case | Method |
    |----------|--------|
    | Add group stat as column | `transform()` |
    | Normalize within groups | `transform()` |
    | Filter/rank within groups | `transform()` |
    | Custom aggregation | `apply()` |
    | Return subset of rows | `apply()` |

    !!! tip "Interviewer's Insight"
        **What they're testing:** GroupBy internals.
        
        **Strong answer signals:**
        
        - Explains broadcast behavior
        - Uses transform for normalization
        - Knows performance difference
        - Can implement ranking within groups

---

### How to Create Bins and Categories with cut() and qcut()? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Discretization`, `Binning` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **cut() - Fixed-Width Bins:**
    
    ```python
    import pandas as pd
    import numpy as np
    
    ages = pd.Series([5, 17, 25, 35, 45, 55, 65, 75, 85])
    
    # Explicit bin edges
    bins = [0, 18, 35, 50, 65, 100]
    labels = ['Child', 'Young Adult', 'Adult', 'Middle Aged', 'Senior']
    
    pd.cut(ages, bins=bins, labels=labels)
    
    # Equal-width bins
    pd.cut(ages, bins=5)  # 5 equal-width bins
    
    # Include lowest value
    pd.cut(ages, bins=bins, labels=labels, include_lowest=True)
    
    # Return bin boundaries
    pd.cut(ages, bins=5, retbins=True)
    ```
    
    **qcut() - Quantile-Based Bins:**
    
    ```python
    # Equal-sized bins (same number of items)
    values = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
    
    pd.qcut(values, q=4)  # Quartiles
    pd.qcut(values, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    pd.qcut(values, q=[0, 0.25, 0.5, 0.75, 1])  # Custom quantiles
    
    # Handle duplicates
    pd.qcut(values, q=4, duplicates='drop')
    ```
    
    **Comparison:**
    
    | Method | Bin Size | Use Case |
    |--------|----------|----------|
    | `cut()` | Fixed width | Age groups, price ranges |
    | `qcut()` | Equal count | Percentiles, balanced groups |
    
    **Example:**
    
    ```python
    # cut: bins by value range
    pd.cut([1, 5, 10, 50, 100], bins=3)
    # (0.9, 34], (0.9, 34], (0.9, 34], (34, 67], (67, 100]
    
    # qcut: bins by count
    pd.qcut([1, 5, 10, 50, 100], q=3)
    # (0.99, 7.5], (0.99, 7.5], (7.5, 30], (30, 100], (30, 100]
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Data discretization.
        
        **Strong answer signals:**
        
        - Knows when to use each method
        - Uses meaningful labels
        - Handles edge cases (duplicates)
        - Explains equal-width vs equal-count

---

### How to Handle Categorical Data Efficiently? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Categorical Data`, `Memory` | **Asked by:** Google, Amazon, Meta, Netflix

??? success "View Answer"

    **Creating Categorical:**
    
    ```python
    import pandas as pd
    
    # Convert to categorical
    df = pd.DataFrame({'status': ['active', 'inactive', 'active', 'pending'] * 10000})
    
    # Memory before
    print(df.memory_usage(deep=True).sum())  # ~360KB
    
    # Convert
    df['status'] = df['status'].astype('category')
    
    # Memory after
    print(df.memory_usage(deep=True).sum())  # ~40KB (90% reduction!)
    ```
    
    **Ordered Categories:**
    
    ```python
    from pandas.api.types import CategoricalDtype
    
    # Define ordered category
    size_type = CategoricalDtype(
        categories=['small', 'medium', 'large', 'xlarge'],
        ordered=True
    )
    
    df['size'] = df['size'].astype(size_type)
    
    # Now comparisons work
    df[df['size'] > 'medium']  # Returns 'large' and 'xlarge'
    df['size'].min(), df['size'].max()
    ```
    
    **get_dummies() for One-Hot Encoding:**
    
    ```python
    # One-hot encoding
    df = pd.DataFrame({'color': ['red', 'blue', 'green', 'red']})
    
    pd.get_dummies(df, columns=['color'])
    # Result: color_blue, color_green, color_red columns
    
    # Drop first to avoid multicollinearity
    pd.get_dummies(df, columns=['color'], drop_first=True)
    
    # Prefix
    pd.get_dummies(df, columns=['color'], prefix='c')
    ```
    
    **Category Operations:**
    
    ```python
    # Add/remove categories
    df['status'].cat.add_categories(['new_status'])
    df['status'].cat.remove_unused_categories()
    
    # Rename categories
    df['status'].cat.rename_categories({'active': 'ACTIVE'})
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Efficient data handling.
        
        **Strong answer signals:**
        
        - Uses category for memory savings
        - Knows ordered categories for comparisons
        - Uses get_dummies for ML
        - Understands when to use category type

---

### How to Use MultiIndex (Hierarchical Indexing)? - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `MultiIndex`, `Hierarchical Data` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Creating MultiIndex:**
    
    ```python
    import pandas as pd
    
    # From tuples
    index = pd.MultiIndex.from_tuples([
        ('A', 2021), ('A', 2022), ('B', 2021), ('B', 2022)
    ], names=['category', 'year'])
    
    df = pd.DataFrame({'sales': [100, 150, 200, 250]}, index=index)
    
    # From product (all combinations)
    index = pd.MultiIndex.from_product(
        [['A', 'B'], [2021, 2022, 2023]],
        names=['category', 'year']
    )
    
    # From GroupBy
    df = pd.DataFrame({
        'category': ['A', 'A', 'B', 'B'],
        'year': [2021, 2022, 2021, 2022],
        'sales': [100, 150, 200, 250]
    })
    df_multi = df.set_index(['category', 'year'])
    ```
    
    **Selecting with MultiIndex:**
    
    ```python
    # Select outer level
    df.loc['A']  # All years for category A
    
    # Select specific combination
    df.loc[('A', 2021)]
    
    # Cross-section
    df.xs(2021, level='year')  # All categories for 2021
    
    # Slice
    df.loc['A':'B']  # Range of outer index
    ```
    
    **Flatten MultiIndex:**
    
    ```python
    # Reset to regular columns
    df.reset_index()
    
    # Flatten column MultiIndex
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    ```
    
    **Stack/Unstack:**
    
    ```python
    # Wide to long (stack)
    df.stack()
    
    # Long to wide (unstack)
    df.unstack()
    df.unstack(level='year')
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Complex data structures.
        
        **Strong answer signals:**
        
        - Creates MultiIndex efficiently
        - Uses xs for cross-sections
        - Knows stack/unstack
        - Can flatten when needed

---

### How to Profile and Optimize Pandas Performance? - Google, Amazon, Netflix Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Performance`, `Profiling`, `Optimization` | **Asked by:** Google, Amazon, Netflix

??? success "View Answer"

    **Profiling Tools:**
    
    ```python
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame(np.random.randn(100000, 10))
    
    # Time operations
    %timeit df.sum()
    
    # Line-by-line profiling
    %lprun -f function_name function_name(df)
    
    # Memory profiling
    %memit df.copy()
    ```
    
    **Common Optimizations:**
    
    ```python
    # 1. Avoid loops - use vectorization
    # SLOW
    for i in range(len(df)):
        df.loc[i, 'new_col'] = df.loc[i, 'A'] * 2
    
    # FAST
    df['new_col'] = df['A'] * 2
    
    # 2. Use NumPy for complex operations
    # SLOW
    df['result'] = df.apply(lambda row: complex_function(row), axis=1)
    
    # FAST
    df['result'] = np.where(df['A'] > 0, df['A'] * 2, df['A'] * 3)
    
    # 3. Use categorical for strings
    df['category'] = df['category'].astype('category')
    
    # 4. Read only needed columns
    df = pd.read_csv('file.csv', usecols=['col1', 'col2'])
    
    # 5. Use query() for complex filters
    df.query('A > 0 and B < 100')  # Faster than boolean indexing
    ```
    
    **Advanced Techniques:**
    
    ```python
    # Numba JIT compilation
    from numba import jit
    
    @jit(nopython=True)
    def fast_calculation(arr):
        result = np.empty(len(arr))
        for i in range(len(arr)):
            result[i] = arr[i] ** 2 + arr[i] * 2
        return result
    
    df['result'] = fast_calculation(df['A'].values)
    
    # Swifter for automatic parallelization
    import swifter
    df['result'] = df['A'].swifter.apply(complex_function)
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Production optimization skills.
        
        **Strong answer signals:**
        
        - Profiles before optimizing
        - Prefers vectorization over loops
        - Uses NumPy for speed
        - Knows Numba/Swifter for edge cases

---

### How to Use melt() for Unpivoting Data? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Reshaping`, `Unpivoting` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **melt() - Wide to Long Format:**
    
    ```python
    import pandas as pd
    
    # Wide format
    df_wide = pd.DataFrame({
        'name': ['Alice', 'Bob'],
        '2021_sales': [100, 200],
        '2022_sales': [150, 250],
        '2023_sales': [180, 300]
    })
    
    # Convert to long format
    df_long = pd.melt(
        df_wide,
        id_vars=['name'],
        value_vars=['2021_sales', '2022_sales', '2023_sales'],
        var_name='year',
        value_name='sales'
    )
    
    # Result:
    #    name        year  sales
    # 0  Alice  2021_sales    100
    # 1    Bob  2021_sales    200
    # ...
    ```
    
    **Clean up melted data:**
    
    ```python
    df_long['year'] = df_long['year'].str.replace('_sales', '').astype(int)
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Data reshaping skills.
        
        **Strong answer signals:**
        
        - Knows melt reverses pivot
        - Uses id_vars for fixed columns
        - Cleans variable names after melting

---

### How to Use stack() and unstack()? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Reshaping`, `MultiIndex` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **stack() - Pivot columns to rows:**
    
    ```python
    import pandas as pd
    
    df = pd.DataFrame({
        'A': [1, 2],
        'B': [3, 4]
    }, index=['row1', 'row2'])
    
    # Stack columns into rows
    stacked = df.stack()
    # Result: MultiIndex Series
    # row1  A    1
    #       B    3
    # row2  A    2
    #       B    4
    ```
    
    **unstack() - Pivot rows to columns:**
    
    ```python
    # Reverse operation
    unstacked = stacked.unstack()
    
    # Unstack specific level
    df_multi = df.set_index([['cat1', 'cat1', 'cat2', 'cat2'], [1, 2, 1, 2]])
    df_multi.unstack(level=0)  # Unstack first level
    df_multi.unstack(level=1)  # Unstack second level
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** MultiIndex manipulation.
        
        **Strong answer signals:**
        
        - Understands stack/unstack relationship
        - Can specify which level to unstack
        - Uses for reshaping complex data

---

### How to Cross-Tabulate with crosstab()? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Cross Tabulation`, `Analysis` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **crosstab() - Frequency Tables:**
    
    ```python
    import pandas as pd
    
    df = pd.DataFrame({
        'gender': ['M', 'F', 'M', 'F', 'M', 'F'],
        'department': ['Sales', 'Sales', 'IT', 'IT', 'Sales', 'HR'],
        'salary': [50000, 55000, 60000, 65000, 52000, 48000]
    })
    
    # Simple frequency table
    pd.crosstab(df['gender'], df['department'])
    
    # With margins (totals)
    pd.crosstab(df['gender'], df['department'], margins=True)
    
    # With aggregation
    pd.crosstab(
        df['gender'], 
        df['department'], 
        values=df['salary'],
        aggfunc='mean'
    )
    
    # Normalize to percentages
    pd.crosstab(df['gender'], df['department'], normalize='all')  # All cells
    pd.crosstab(df['gender'], df['department'], normalize='index')  # By row
    pd.crosstab(df['gender'], df['department'], normalize='columns')  # By column
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Contingency table creation.
        
        **Strong answer signals:**
        
        - Uses normalize for percentages
        - Knows margins for totals
        - Can aggregate with values/aggfunc

---

### How to Use explode() for List Columns? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `List Operations`, `Data Preprocessing` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **explode() - Unnest Lists:**
    
    ```python
    import pandas as pd
    
    df = pd.DataFrame({
        'name': ['Alice', 'Bob'],
        'skills': [['Python', 'SQL', 'ML'], ['Java', 'Scala']]
    })
    
    # Explode list column
    df_exploded = df.explode('skills')
    
    # Result:
    #    name  skills
    # 0  Alice  Python
    # 0  Alice     SQL
    # 0  Alice      ML
    # 1    Bob    Java
    # 1    Bob   Scala
    
    # Reset index after explode
    df_exploded = df.explode('skills').reset_index(drop=True)
    
    # Explode multiple columns (same length)
    df = pd.DataFrame({
        'id': [1, 2],
        'values': [[1, 2], [3, 4]],
        'labels': [['a', 'b'], ['c', 'd']]
    })
    df.explode(['values', 'labels'])
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Nested data handling.
        
        **Strong answer signals:**
        
        - Resets index after explode
        - Handles multiple list columns
        - Knows reverse: groupby + list aggregation

---

### How to Handle JSON with Nested Structures? - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `JSON Processing`, `Nested Data` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **json_normalize() - Flatten Nested JSON:**
    
    ```python
    import pandas as pd
    from pandas import json_normalize
    
    data = [
        {
            'id': 1,
            'name': 'Alice',
            'address': {
                'city': 'NYC',
                'zip': '10001'
            },
            'orders': [{'id': 101, 'amount': 50}, {'id': 102, 'amount': 75}]
        }
    ]
    
    # Flatten top level
    df = json_normalize(data)
    
    # Flatten with nested paths
    df = json_normalize(
        data,
        record_path='orders',  # Explode this array
        meta=['id', 'name', ['address', 'city']],  # Include these fields
        meta_prefix='user_'
    )
    
    # Read JSON file
    df = pd.read_json('data.json')
    df = pd.read_json('data.json', orient='records')
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Real-world data handling.
        
        **Strong answer signals:**
        
        - Uses json_normalize for nested
        - Knows record_path and meta
        - Handles different JSON orientations

---

### How to Use Method Chaining for Clean Code? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Method Chaining`, `Clean Code` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Method Chaining Pattern:**
    
    ```python
    import pandas as pd
    
    # Instead of multiple intermediate variables
    df = (
        pd.read_csv('data.csv')
        .query('status == "active"')
        .assign(
            year=lambda x: pd.to_datetime(x['date']).dt.year,
            total=lambda x: x['price'] * x['quantity']
        )
        .groupby(['year', 'region'])
        .agg(revenue=('total', 'sum'))
        .reset_index()
        .sort_values('revenue', ascending=False)
    )
    ```
    
    **pipe() for Custom Functions:**
    
    ```python
    def add_features(df):
        return df.assign(
            log_value=np.log1p(df['value']),
            is_high=df['value'] > df['value'].median()
        )
    
    def filter_outliers(df, column, n_std=3):
        mean, std = df[column].mean(), df[column].std()
        return df[(df[column] - mean).abs() <= n_std * std]
    
    result = (
        df
        .pipe(add_features)
        .pipe(filter_outliers, 'value', n_std=2)
    )
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Clean, maintainable code.
        
        **Strong answer signals:**
        
        - Uses parentheses for multi-line chains
        - Uses pipe() for custom functions
        - Avoids intermediate variables

---

### How to Read/Write Parquet Files? - Google, Amazon, Netflix Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `File I/O`, `Big Data`, `Parquet` | **Asked by:** Google, Amazon, Netflix, Meta

??? success "View Answer"

    **Parquet - Columnar Format:**
    
    ```python
    import pandas as pd
    
    # Write to Parquet
    df.to_parquet('data.parquet')
    df.to_parquet('data.parquet', compression='snappy')  # Default
    df.to_parquet('data.parquet', compression='gzip')    # Smaller
    
    # Read Parquet
    df = pd.read_parquet('data.parquet')
    
    # Read only specific columns (fast!)
    df = pd.read_parquet('data.parquet', columns=['col1', 'col2'])
    
    # With filters (predicate pushdown)
    df = pd.read_parquet(
        'data.parquet',
        filters=[('year', '==', 2023)]
    )
    ```
    
    **Parquet vs CSV:**
    
    | Feature | Parquet | CSV |
    |---------|---------|-----|
    | Size | ~10x smaller | Larger |
    | Read speed | Faster | Slower |
    | Column selection | Fast | Must read all |
    | Data types | Preserved | Lost |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Efficient data storage.
        
        **Strong answer signals:**
        
        - Uses Parquet for large datasets
        - Reads only needed columns
        - Knows compression options

---

### How to Use assign() for Creating New Columns? - Google, Amazon Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Column Creation`, `Method Chaining` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **assign() - Create Columns in Chain:**
    
    ```python
    import pandas as pd
    
    df = pd.DataFrame({
        'price': [100, 200, 300],
        'quantity': [2, 3, 1]
    })
    
    # Create multiple columns
    df = df.assign(
        total=lambda x: x['price'] * x['quantity'],
        discounted=lambda x: x['total'] * 0.9,
        tax=lambda x: x['total'] * 0.1
    )
    
    # Reference previous assignments
    df = df.assign(
        subtotal=lambda x: x['price'] * x['quantity'],
        tax=lambda x: x['subtotal'] * 0.1,
        total=lambda x: x['subtotal'] + x['tax']
    )
    ```
    
    **assign() vs direct assignment:**
    
    ```python
    # Direct assignment (modifies in place)
    df['new_col'] = df['price'] * 2
    
    # assign() (returns new DataFrame, original unchanged)
    df_new = df.assign(new_col=df['price'] * 2)
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Functional programming style.
        
        **Strong answer signals:**
        
        - Uses lambda for dependent columns
        - Prefers assign for method chaining
        - Knows it returns new DataFrame

---

### How to Calculate Percentage Change and Cumulative Stats? - Google, Amazon, Netflix Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Time Series`, `Finance` | **Asked by:** Google, Amazon, Netflix, Apple

??? success "View Answer"

    **Percentage Change:**
    
    ```python
    import pandas as pd
    
    df = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=5),
        'price': [100, 105, 102, 110, 108]
    })
    
    # Daily percentage change
    df['pct_change'] = df['price'].pct_change()
    
    # Percentage change over N periods
    df['pct_change_3'] = df['price'].pct_change(periods=3)
    
    # Fill first NaN
    df['pct_change'] = df['price'].pct_change().fillna(0)
    ```
    
    **Cumulative Statistics:**
    
    ```python
    # Cumulative sum
    df['cumsum'] = df['price'].cumsum()
    
    # Cumulative product (for returns)
    df['cumulative_return'] = (1 + df['pct_change']).cumprod() - 1
    
    # Cumulative max/min
    df['cummax'] = df['price'].cummax()
    df['cummin'] = df['price'].cummin()
    
    # Drawdown
    df['drawdown'] = df['price'] / df['price'].cummax() - 1
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Financial calculations.
        
        **Strong answer signals:**
        
        - Uses pct_change for returns
        - Knows cumprod for cumulative returns
        - Can calculate drawdowns

---

### How to Shift and Lag Data? - Google, Amazon, Netflix Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Time Series`, `Lag Features` | **Asked by:** Google, Amazon, Netflix

??? success "View Answer"

    **shift() - Create Lag/Lead Features:**
    
    ```python
    import pandas as pd
    
    df = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=5),
        'value': [10, 20, 30, 40, 50]
    })
    
    # Previous value (lag)
    df['prev_value'] = df['value'].shift(1)
    df['prev_2'] = df['value'].shift(2)
    
    # Next value (lead)
    df['next_value'] = df['value'].shift(-1)
    
    # Calculate difference from previous
    df['diff'] = df['value'] - df['value'].shift(1)
    # Same as: df['value'].diff()
    
    # Shift with fill
    df['prev_filled'] = df['value'].shift(1, fill_value=0)
    ```
    
    **Use Cases:**
    
    | Operation | Formula |
    |-----------|---------|
    | Lag-1 | `df['col'].shift(1)` |
    | Difference | `df['col'].diff()` |
    | % Change | `df['col'].pct_change()` |
    | Rolling difference | `df['col'] - df['col'].shift(n)` |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Feature engineering for time series.
        
        **Strong answer signals:**
        
        - Creates lag features for ML
        - Knows shift vs diff vs pct_change
        - Handles NaN from shifting

---

### How to Sample Data from DataFrame? - Google, Amazon Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Sampling`, `Data Exploration` | **Asked by:** Google, Amazon, Meta, Netflix

??? success "View Answer"

    **sample() - Random Sampling:**
    
    ```python
    import pandas as pd
    
    df = pd.DataFrame({'A': range(1000)})
    
    # Sample n rows
    sample = df.sample(n=100)
    
    # Sample fraction
    sample = df.sample(frac=0.1)  # 10%
    
    # With replacement (bootstrap)
    bootstrap = df.sample(n=1000, replace=True)
    
    # Reproducible sampling
    sample = df.sample(n=100, random_state=42)
    
    # Weighted sampling
    df['weight'] = [0.1] * 500 + [0.9] * 500
    sample = df.sample(n=100, weights='weight')
    ```
    
    **Stratified Sampling:**
    
    ```python
    # Sample within groups
    df.groupby('category').sample(n=10)
    df.groupby('category').sample(frac=0.1)
    
    # Stratified with sklearn
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, stratify=df['category'], test_size=0.2)
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Data sampling techniques.
        
        **Strong answer signals:**
        
        - Uses random_state for reproducibility
        - Knows weighted sampling
        - Uses stratified for imbalanced data

---

### How to Detect and Handle Outliers? - Google, Amazon, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Outlier Detection`, `Data Cleaning` | **Asked by:** Google, Amazon, Meta, Netflix

??? success "View Answer"

    **Detection Methods:**
    
    ```python
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame({'value': [1, 2, 3, 100, 4, 5, -50, 6]})
    
    # Z-score method
    z_scores = (df['value'] - df['value'].mean()) / df['value'].std()
    outliers = df[abs(z_scores) > 3]
    
    # IQR method
    Q1 = df['value'].quantile(0.25)
    Q3 = df['value'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df['value'] < lower) | (df['value'] > upper)]
    
    # Percentile method
    outliers = df[(df['value'] < df['value'].quantile(0.01)) | 
                  (df['value'] > df['value'].quantile(0.99))]
    ```
    
    **Handling Outliers:**
    
    ```python
    # Remove
    df_clean = df[(df['value'] >= lower) & (df['value'] <= upper)]
    
    # Cap (winsorize)
    df['value_capped'] = df['value'].clip(lower=lower, upper=upper)
    
    # Replace with NaN
    df.loc[abs(z_scores) > 3, 'value'] = np.nan
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Data quality handling.
        
        **Strong answer signals:**
        
        - Knows IQR and Z-score methods
        - Chooses method based on distribution
        - Considers domain knowledge

---

### How to Normalize and Standardize Data? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Feature Engineering`, `ML Preprocessing` | **Asked by:** Google, Amazon, Meta, Microsoft

??? success "View Answer"

    **Normalization (Min-Max Scaling):**
    
    ```python
    import pandas as pd
    
    df = pd.DataFrame({'value': [10, 20, 30, 40, 50]})
    
    # Scale to [0, 1]
    df['normalized'] = (df['value'] - df['value'].min()) / \
                       (df['value'].max() - df['value'].min())
    
    # Using sklearn
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df['normalized'] = scaler.fit_transform(df[['value']])
    ```
    
    **Standardization (Z-score):**
    
    ```python
    # Scale to mean=0, std=1
    df['standardized'] = (df['value'] - df['value'].mean()) / df['value'].std()
    
    # Using sklearn
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df['standardized'] = scaler.fit_transform(df[['value']])
    ```
    
    **When to Use:**
    
    | Method | Use When |
    |--------|----------|
    | Min-Max | Bounded range needed, no outliers |
    | Z-score | Normal distribution, has outliers |
    | Robust | Many outliers (uses median/IQR) |

    !!! tip "Interviewer's Insight"
        **What they're testing:** ML preprocessing.
        
        **Strong answer signals:**
        
        - Knows difference between methods
        - Uses sklearn for production
        - Considers outliers in choice

---

### How to Use where() and mask() Methods? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Conditional Operations`, `Data Transformation` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **where() - Keep values where True:**
    
    ```python
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
    
    # Keep values >= 3, replace others with NaN
    df['A'].where(df['A'] >= 3)
    # Result: [NaN, NaN, 3, 4, 5]
    
    # Replace with specific value
    df['A'].where(df['A'] >= 3, other=0)
    # Result: [0, 0, 3, 4, 5]
    ```
    
    **mask() - Replace values where True:**
    
    ```python
    # Opposite of where
    # Replace values >= 3 with NaN
    df['A'].mask(df['A'] >= 3)
    # Result: [1, 2, NaN, NaN, NaN]
    
    # Replace with specific value
    df['A'].mask(df['A'] >= 3, other=999)
    # Result: [1, 2, 999, 999, 999]
    ```
    
    **vs np.where:**
    
    ```python
    # np.where for if-else
    df['result'] = np.where(df['A'] >= 3, 'high', 'low')
    
    # Multiple conditions: np.select
    conditions = [df['A'] < 2, df['A'] < 4]
    choices = ['low', 'medium']
    df['category'] = np.select(conditions, choices, default='high')
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Conditional data manipulation.
        
        **Strong answer signals:**
        
        - Knows where keeps, mask replaces
        - Uses np.where for if-else
        - Uses np.select for multiple conditions

---

### How to Rank Values in Pandas? - Google, Amazon Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Ranking`, `Analysis` | **Asked by:** Google, Amazon, Meta, Netflix

??? success "View Answer"

    **rank() Method:**
    
    ```python
    import pandas as pd
    
    df = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie', 'David'],
        'score': [85, 90, 85, 95]
    })
    
    # Default: average rank for ties
    df['rank'] = df['score'].rank()
    # [1.5, 3, 1.5, 4] - tied scores get average
    
    # Rank methods for ties
    df['rank_min'] = df['score'].rank(method='min')    # [1, 3, 1, 4]
    df['rank_max'] = df['score'].rank(method='max')    # [2, 3, 2, 4]
    df['rank_first'] = df['score'].rank(method='first') # [1, 3, 2, 4]
    df['rank_dense'] = df['score'].rank(method='dense') # [1, 2, 1, 3]
    
    # Descending rank
    df['rank_desc'] = df['score'].rank(ascending=False)
    
    # Rank within groups
    df['rank_in_group'] = df.groupby('category')['score'].rank()
    
    # Percentile rank
    df['percentile'] = df['score'].rank(pct=True)
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Data ranking skills.
        
        **Strong answer signals:**
        
        - Knows different tie-breaking methods
        - Uses dense for no gaps
        - Can rank within groups

---

### How to Find First/Last N Rows Per Group? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `GroupBy`, `Selection` | **Asked by:** Google, Amazon, Meta, Netflix

??? success "View Answer"

    **nlargest/nsmallest per Group:**
    
    ```python
    import pandas as pd
    
    df = pd.DataFrame({
        'category': ['A', 'A', 'A', 'B', 'B', 'B'],
        'value': [10, 30, 20, 40, 60, 50]
    })
    
    # Top 2 per group using nlargest
    df.groupby('category').apply(lambda x: x.nlargest(2, 'value'))
    
    # Top N using head after sort
    df.sort_values('value', ascending=False).groupby('category').head(2)
    
    # First/last row per group
    df.groupby('category').first()
    df.groupby('category').last()
    df.groupby('category').nth(0)  # First row
    df.groupby('category').nth(-1)  # Last row
    ```
    
    **Using rank:**
    
    ```python
    # More efficient for large data
    df['rank'] = df.groupby('category')['value'].rank(method='first', ascending=False)
    top_2 = df[df['rank'] <= 2]
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Efficient group operations.
        
        **Strong answer signals:**
        
        - Uses nlargest/nsmallest for simplicity
        - Uses rank for large data
        - Knows first(), last(), nth()

---

### How to Use nsmallest() and nlargest()? - Google, Amazon Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Selection`, `Performance` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **nlargest/nsmallest - Efficient Selection:**
    
    ```python
    import pandas as pd
    
    df = pd.DataFrame({
        'product': ['A', 'B', 'C', 'D', 'E'],
        'sales': [100, 500, 200, 800, 300],
        'profit': [10, 50, 30, 70, 20]
    })
    
    # Top 3 by sales
    df.nlargest(3, 'sales')
    
    # Bottom 3 by profit
    df.nsmallest(3, 'profit')
    
    # By multiple columns (tiebreaker)
    df.nlargest(3, ['sales', 'profit'])
    
    # Keep='first' (default), 'last', 'all'
    df.nlargest(3, 'sales', keep='all')  # Include all ties
    ```
    
    **Performance:**
    
    ```python
    # nlargest is O(n log k) - faster than full sort
    df.nlargest(10, 'value')  # Faster
    
    # Full sort is O(n log n)
    df.sort_values('value', ascending=False).head(10)  # Slower
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Efficient data selection.
        
        **Strong answer signals:**
        
        - Knows nlargest is faster than sort+head
        - Uses keep='all' for ties
        - Applies to Series and DataFrame

---

### How to Calculate Weighted Average? - Google, Amazon, Netflix Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Aggregation`, `Finance` | **Asked by:** Google, Amazon, Netflix, Apple

??? success "View Answer"

    **Weighted Average:**
    
    ```python
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame({
        'product': ['A', 'B', 'C'],
        'price': [10, 20, 30],
        'quantity': [100, 50, 25]
    })
    
    # Weighted average price
    weighted_avg = np.average(df['price'], weights=df['quantity'])
    # or
    weighted_avg = (df['price'] * df['quantity']).sum() / df['quantity'].sum()
    
    # Weighted average per group
    def weighted_avg_func(group, value_col, weight_col):
        return np.average(group[value_col], weights=group[weight_col])
    
    df.groupby('category').apply(
        weighted_avg_func, 'price', 'quantity'
    )
    ```
    
    **Named Aggregation with Weighted Average:**
    
    ```python
    def weighted_mean(df, value_col, weight_col):
        return (df[value_col] * df[weight_col]).sum() / df[weight_col].sum()
    
    result = df.groupby('category').apply(
        lambda x: pd.Series({
            'weighted_price': weighted_mean(x, 'price', 'quantity'),
            'total_quantity': x['quantity'].sum()
        })
    )
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Custom aggregations.
        
        **Strong answer signals:**
        
        - Uses np.average with weights
        - Can apply to groups
        - Handles edge cases (zero weights)

---

### How to Perform Window Functions Like SQL? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Window Functions`, `Analytics` | **Asked by:** Google, Amazon, Meta, Netflix

??? success "View Answer"

    **SQL-like Window Functions:**
    
    ```python
    import pandas as pd
    
    df = pd.DataFrame({
        'category': ['A', 'A', 'A', 'B', 'B', 'B'],
        'date': pd.date_range('2023-01-01', periods=6),
        'value': [10, 20, 30, 40, 50, 60]
    })
    
    # ROW_NUMBER
    df['row_num'] = df.groupby('category').cumcount() + 1
    
    # RANK
    df['rank'] = df.groupby('category')['value'].rank(method='min')
    
    # DENSE_RANK
    df['dense_rank'] = df.groupby('category')['value'].rank(method='dense')
    
    # LEAD / LAG
    df['prev_value'] = df.groupby('category')['value'].shift(1)
    df['next_value'] = df.groupby('category')['value'].shift(-1)
    
    # Running SUM / AVG
    df['running_sum'] = df.groupby('category')['value'].cumsum()
    df['running_avg'] = df.groupby('category')['value'].expanding().mean().values
    
    # Percent of total
    df['pct_of_cat'] = df['value'] / df.groupby('category')['value'].transform('sum')
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Analytics skills.
        
        **Strong answer signals:**
        
        - Maps SQL functions to Pandas
        - Uses transform for same-shape output
        - Combines groupby with cumulative ops

---

### How to Compare Two DataFrames? - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Data Comparison`, `Validation` | **Asked by:** Amazon, Google, Microsoft

??? success "View Answer"

    **compare() Method (Pandas 1.1+):**
    
    ```python
    import pandas as pd
    
    df1 = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
    df2 = pd.DataFrame({'A': [1, 2, 4], 'B': ['a', 'x', 'c']})
    
    # Show differences
    df1.compare(df2)
    # Shows 'self' and 'other' for differences
    
    # Keep all rows
    df1.compare(df2, keep_equal=True)
    
    # Keep all columns
    df1.compare(df2, keep_shape=True)
    ```
    
    **Testing Equality:**
    
    ```python
    # Check if equal
    df1.equals(df2)  # Returns True/False
    
    # Element-wise comparison
    df1 == df2  # Boolean DataFrame
    (df1 == df2).all().all()  # All equal?
    
    # For testing
    from pandas.testing import assert_frame_equal
    assert_frame_equal(df1, df2, check_dtype=False)
    ```
    
    **Find Differences:**
    
    ```python
    # Rows in df1 not in df2
    df1[~df1.isin(df2).all(axis=1)]
    
    # Using merge
    merged = df1.merge(df2, indicator=True, how='outer')
    only_in_df1 = merged[merged['_merge'] == 'left_only']
    only_in_df2 = merged[merged['_merge'] == 'right_only']
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Data validation skills.
        
        **Strong answer signals:**
        
        - Uses compare() for visual diff
        - Uses assert_frame_equal for tests
        - Can find specific differences

---

### How to Handle SettingWithCopyWarning Properly? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Best Practices`, `Common Errors` | **Asked by:** Google, Amazon, Meta, Microsoft

??? success "View Answer"

    **Copy-on-Write in Pandas 2.0+:**
    
    ```python
    import pandas as pd
    
    # Enable CoW globally
    pd.options.mode.copy_on_write = True
    
    # Now this is safe
    df2 = df[df['A'] > 0]
    df2['B'] = 100  # Creates copy automatically
    
    # Original df unchanged
    ```
    
    **Best Practices Summary:**
    
    ```python
    # 1. Modifying original - use .loc
    df.loc[df['A'] > 0, 'B'] = 100
    
    # 2. Creating new DataFrame - use .copy()
    df_new = df[df['A'] > 0].copy()
    df_new['B'] = 100
    
    # 3. Method chaining - use .assign()
    df_new = df.query('A > 0').assign(B=100)
    
    # 4. Never chain indexing
    # BAD
    df[df['A'] > 0]['B'] = 100
    # GOOD
    df.loc[df['A'] > 0, 'B'] = 100
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Production-safe code.
        
        **Strong answer signals:**
        
        - Knows CoW in Pandas 2.0+
        - Uses .loc consistently
        - Avoids chained indexing

---

### How to Use Sparse Data Structures? - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Sparse Data`, `Memory Optimization` | **Asked by:** Google, Amazon, Netflix

??? success "View Answer"

    **Sparse Arrays - For Mostly-Null Data:**
    
    ```python
    import pandas as pd
    import numpy as np
    
    # Create sparse array
    arr = pd.arrays.SparseArray([0, 0, 1, 0, 0, 0, 2, 0, 0, 0])
    
    # Memory savings
    print(arr.memory_usage())  # Much smaller than dense
    
    # Sparse Series
    s = pd.Series(pd.arrays.SparseArray([0] * 1000000 + [1]))
    print(s.memory_usage())  # ~16 bytes vs ~8MB
    
    # Convert existing
    df_sparse = df.astype(pd.SparseDtype('float', fill_value=0))
    ```
    
    **Use Cases:**
    
    | Scenario | Memory Savings |
    |----------|---------------|
    | 90% zeros | ~10x |
    | 99% zeros | ~100x |
    | One-hot encoded | Massive |
    
    **Operations:**
    
    ```python
    # Most operations work normally
    s_sparse.sum()
    s_sparse.mean()
    
    # Convert back to dense
    s_dense = s_sparse.sparse.to_dense()
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Advanced memory optimization.
        
        **Strong answer signals:**
        
        - Uses for mostly-zero data
        - Knows fill_value parameter
        - Understands trade-offs

---

### How to Implement Custom Aggregation Functions? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Aggregation`, `Custom Functions` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Custom Aggregation:**
    
    ```python
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame({
        'category': ['A', 'A', 'B', 'B'],
        'value': [10, 20, 30, 40]
    })
    
    # Lambda function
    df.groupby('category')['value'].agg(lambda x: x.max() - x.min())
    
    # Named function
    def range_func(x):
        return x.max() - x.min()
    
    df.groupby('category')['value'].agg(range_func)
    
    # Multiple custom aggregations
    df.groupby('category')['value'].agg([
        ('range', lambda x: x.max() - x.min()),
        ('cv', lambda x: x.std() / x.mean()),  # Coefficient of variation
        ('iqr', lambda x: x.quantile(0.75) - x.quantile(0.25))
    ])
    
    # Named aggregation with custom
    df.groupby('category').agg(
        value_range=('value', lambda x: x.max() - x.min()),
        value_cv=('value', lambda x: x.std() / x.mean())
    )
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Advanced aggregation skills.
        
        **Strong answer signals:**
        
        - Uses named aggregations
        - Defines reusable functions
        - Combines built-in and custom

---

### How to Perform Asof Joins (Nearest Key Join)? - Google, Amazon, Netflix Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Joining`, `Time Series` | **Asked by:** Google, Amazon, Netflix, Apple

??? success "View Answer"

    **merge_asof() - Join on Nearest Key:**
    
    ```python
    import pandas as pd
    
    # Trade data
    trades = pd.DataFrame({
        'time': pd.to_datetime(['2023-01-01 10:00:01', '2023-01-01 10:00:03', 
                                '2023-01-01 10:00:05']),
        'ticker': ['AAPL', 'AAPL', 'AAPL'],
        'quantity': [100, 200, 150]
    })
    
    # Quote data
    quotes = pd.DataFrame({
        'time': pd.to_datetime(['2023-01-01 10:00:00', '2023-01-01 10:00:02', 
                                '2023-01-01 10:00:04']),
        'ticker': ['AAPL', 'AAPL', 'AAPL'],
        'bid': [149.0, 149.5, 150.0]
    })
    
    # Join trade with most recent quote
    result = pd.merge_asof(
        trades.sort_values('time'),
        quotes.sort_values('time'),
        on='time',
        by='ticker',
        direction='backward'  # Use most recent quote
    )
    ```
    
    **Direction Options:**
    
    | Direction | Meaning |
    |-----------|---------|
    | backward | Previous/equal key |
    | forward | Next/equal key |
    | nearest | Closest key |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Time series joining.
        
        **Strong answer signals:**
        
        - Sorts data before asof join
        - Uses by= for grouping
        - Knows direction parameter

---

### How to Calculate Cohort Metrics? - Meta, Netflix, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Cohort Analysis`, `Time Series` | **Asked by:** Meta, Netflix, Amazon, Google

??? success "View Answer"

    **Cohort Retention Analysis:**
    
    ```python
    import pandas as pd
    
    df = pd.DataFrame({
        'user_id': [1, 1, 1, 2, 2, 3],
        'order_date': pd.to_datetime([
            '2023-01-05', '2023-02-10', '2023-03-15',
            '2023-01-20', '2023-03-25', '2023-02-01'
        ])
    })
    
    # Get first purchase date (cohort)
    df['cohort'] = df.groupby('user_id')['order_date'].transform('min')
    df['cohort_month'] = df['cohort'].dt.to_period('M')
    df['order_month'] = df['order_date'].dt.to_period('M')
    
    # Calculate months since cohort
    df['months_since_cohort'] = (
        df['order_month'].astype(int) - df['cohort_month'].astype(int)
    )
    
    # Create cohort pivot
    cohort_data = df.groupby(['cohort_month', 'months_since_cohort']).agg(
        users=('user_id', 'nunique')
    ).reset_index()
    
    cohort_pivot = cohort_data.pivot_table(
        index='cohort_month',
        columns='months_since_cohort',
        values='users'
    )
    
    # Calculate retention rates
    cohort_sizes = cohort_pivot[0]
    retention = cohort_pivot.divide(cohort_sizes, axis=0)
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Business analytics skills.
        
        **Strong answer signals:**
        
        - Calculates cohort from first action
        - Uses period for month grouping
        - Creates retention matrix

---

### How to Implement A/B Test Analysis in Pandas? - Meta, Google, Netflix Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `A/B Testing`, `Statistical Analysis` | **Asked by:** Meta, Google, Netflix, Amazon

??? success "View Answer"

    **A/B Test Analysis:**
    
    ```python
    import pandas as pd
    import numpy as np
    from scipy import stats
    
    df = pd.DataFrame({
        'user_id': range(10000),
        'variant': np.random.choice(['control', 'treatment'], 10000),
        'converted': np.random.binomial(1, [0.10] * 5000 + [0.12] * 5000)
    })
    
    # Summary statistics per variant
    summary = df.groupby('variant').agg(
        users=('user_id', 'count'),
        conversions=('converted', 'sum'),
        conversion_rate=('converted', 'mean')
    )
    
    # Statistical test
    control = df[df['variant'] == 'control']['converted']
    treatment = df[df['variant'] == 'treatment']['converted']
    
    # Chi-squared test for proportions
    contingency = pd.crosstab(df['variant'], df['converted'])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    
    # Or use proportion z-test
    from statsmodels.stats.proportion import proportions_ztest
    count = [treatment.sum(), control.sum()]
    nobs = [len(treatment), len(control)]
    z_stat, p_value = proportions_ztest(count, nobs)
    
    # Confidence interval for lift
    lift = (treatment.mean() - control.mean()) / control.mean()
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Statistical analysis skills.
        
        **Strong answer signals:**
        
        - Calculates conversion rates
        - Uses appropriate statistical test
        - Interprets p-value correctly

---

### How to Use Copy-on-Write (CoW) in Pandas 2.0+? - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Internals`, `Performance` | **Asked by:** Google, Meta, Microsoft

??? success "View Answer"

    **Copy-on-Write Explained:**
    
    ```python
    import pandas as pd
    
    # Enable globally
    pd.options.mode.copy_on_write = True
    
    # Or use context manager
    with pd.option_context('mode.copy_on_write', True):
        df2 = df[['A', 'B']]
        df2['A'] = 100  # Creates copy only when modified
    ```
    
    **Benefits:**
    
    | Aspect | Without CoW | With CoW |
    |--------|-------------|----------|
    | Memory | Copies on slice | Shares until modified |
    | Safety | Ambiguous | Always safe |
    | Speed | Unnecessary copies | Lazy copies |
    
    **Behavior:**
    
    ```python
    pd.options.mode.copy_on_write = True
    
    # Views share data until modification
    df = pd.DataFrame({'A': [1, 2, 3]})
    df2 = df[['A']]  # No copy yet
    
    df2['A'] = 100  # Copy created here
    print(df)  # Original unchanged!
    
    # No more SettingWithCopyWarning
    df[df['A'] > 1]['A'] = 99  # Safe, no effect on df
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Modern Pandas knowledge.
        
        **Strong answer signals:**
        
        - Knows CoW mechanism
        - Understands lazy copying
        - Knows it's default in Pandas 3.0

---

### How to Use PyArrow Backend for Better Performance? - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Performance`, `Arrow` | **Asked by:** Google, Amazon, Databricks

??? success "View Answer"

    **PyArrow Backend (Pandas 2.0+):**
    
    ```python
    import pandas as pd
    
    # Read with PyArrow backend
    df = pd.read_csv('data.csv', dtype_backend='pyarrow')
    df = pd.read_parquet('data.parquet', dtype_backend='pyarrow')
    
    # Convert existing DataFrame
    df_arrow = df.convert_dtypes(dtype_backend='pyarrow')
    
    # Check types
    df_arrow.dtypes
    # int64[pyarrow], string[pyarrow], etc.
    ```
    
    **Benefits:**
    
    | Feature | NumPy Backend | PyArrow Backend |
    |---------|---------------|-----------------|
    | String memory | High | Low |
    | Null handling | float64 trick | Native |
    | Interop | Limited | Arrow ecosystem |
    
    **String Performance:**
    
    ```python
    # PyArrow strings are much more efficient
    df['text_col']  # With PyArrow: less memory, faster ops
    
    # Native null support
    df_arrow['int_col']  # Can have Int64 with nulls
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Cutting-edge Pandas.
        
        **Strong answer signals:**
        
        - Knows Arrow for string efficiency
        - Uses for interop with Spark/Arrow
        - Understands when to use

---

## Quick Reference: 127+ Interview Questions

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

??? success "View Code Example"


    **Difficulty:** 🟢 Easy | **Tags:** `Code Example` | **Asked by:** Code Pattern
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

??? success "View Code Example"


    **Difficulty:** 🟢 Easy | **Tags:** `Code Example` | **Asked by:** Code Pattern
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

??? success "View Code Example"


    **Difficulty:** 🟢 Easy | **Tags:** `Code Example` | **Asked by:** Code Pattern
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

### Explain MultiIndex in Pandas - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `MultiIndex`, `Hierarchical Data`, `Advanced Indexing` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **MultiIndex** creates hierarchical row/column labels, enabling multi-dimensional data in 2D DataFrames. Essential for time series, grouped data, and pivot operations.

    **Complete Examples:**

    ```python
    import pandas as pd
    import numpy as np
    
    # 1. Creating MultiIndex
    # From tuples
    index = pd.MultiIndex.from_tuples([
        ('USA', 'NY'), ('USA', 'CA'), ('UK', 'London'), ('UK', 'Manchester')
    ], names=['Country', 'City'])
    
    df = pd.DataFrame({
        'Population': [8_000_000, 4_000_000, 9_000_000, 500_000],
        'GDP': [1500, 3000, 500, 80]
    }, index=index)
    
    print(df)
    
    # 2. From product
    idx = pd.MultiIndex.from_product(
        [['2023', '2024'], ['Q1', 'Q2', 'Q3', 'Q4']],
        names=['Year', 'Quarter']
    )
    
    sales_df = pd.DataFrame({
        'Revenue': np.random.randint(100, 500, 8),
        'Profit': np.random.randint(10, 100, 8)
    }, index=idx)
    
    # 3. Indexing with MultiIndex
    # Access by level
    print(df.loc['USA'])  # All USA cities
    print(df.loc[('USA', 'NY')])  # Specific city
    
    # Cross-section (xs)
    print(df.xs('USA', level='Country'))
    print(df.xs('NY', level='City'))
    
    # 4. Slicing MultiIndex
    # Slice first level
    print(df.loc['USA':'UK'])
    
    # Slice second level
    print(df.loc[('USA', 'CA'):('UK', 'London')])
    
    # 5. Stack and Unstack
    # Create sample data
    data = pd.DataFrame({
        'Year': [2023, 2023, 2024, 2024],
        'Quarter': ['Q1', 'Q2', 'Q1', 'Q2'],
        'Revenue': [100, 150, 120, 180]
    })
    
    # Set multiindex
    data_idx = data.set_index(['Year', 'Quarter'])
    
    # Unstack: move index level to columns
    unstacked = data_idx.unstack()
    print(unstacked)
    
    # Stack: move column level to index
    stacked = unstacked.stack()
    print(stacked)
    
    # 6. GroupBy with MultiIndex
    df_sales = pd.DataFrame({
        'Region': ['North', 'North', 'South', 'South'] * 2,
        'Product': ['A', 'B', 'A', 'B'] * 2,
        'Quarter': ['Q1', 'Q1', 'Q1', 'Q1', 'Q2', 'Q2', 'Q2', 'Q2'],
        'Revenue': [100, 150, 120, 180, 110, 160, 130, 190]
    })
    
    # Create MultiIndex from groupby
    grouped = df_sales.groupby(['Region', 'Product']).sum()
    print(grouped)
    
    # 7. Sorting MultiIndex
    # Sort by all levels
    df_sorted = df.sort_index()
    
    # Sort by specific level
    df_sorted = df.sort_index(level='City')
    
    # Sort descending
    df_sorted = df.sort_index(ascending=False)
    
    # 8. Swapping Levels
    # Swap index levels
    df_swapped = df.swaplevel('Country', 'City')
    print(df_swapped)
    
    # 9. Resetting Index
    # Remove MultiIndex
    df_reset = df.reset_index()
    print(df_reset)
    
    # Keep one level
    df_reset_partial = df.reset_index(level='City')
    print(df_reset_partial)
    
    # 10. Advanced Operations
    # Sum across levels
    print(df.sum(level='Country'))
    
    # Mean across levels
    print(df.mean(level='City'))
    
    # Apply function per level
    result = df.groupby(level='Country').agg({
        'Population': 'sum',
        'GDP': 'mean'
    })
    print(result)
    ```

    **Creating MultiIndex:**

    | Method | Use Case | Example |
    |--------|----------|---------|
    | **from_tuples** | Explicit labels | `pd.MultiIndex.from_tuples([('A', 1)])` |
    | **from_product** | Cartesian product | `pd.MultiIndex.from_product([['A', 'B'], [1, 2]])` |
    | **from_arrays** | Separate arrays | `pd.MultiIndex.from_arrays([['A', 'B'], [1, 2]])` |
    | **set_index** | From columns | `df.set_index(['col1', 'col2'])` |

    **Common Operations:**

    | Operation | Description | Example |
    |-----------|-------------|---------|
    | **loc** | Access by labels | `df.loc[('USA', 'NY')]` |
    | **xs** | Cross-section | `df.xs('USA', level=0)` |
    | **unstack** | Level to columns | `df.unstack()` |
    | **stack** | Columns to level | `df.stack()` |
    | **swaplevel** | Swap index levels | `df.swaplevel(0, 1)` |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Advanced indexing and hierarchical data structures.
        
        **Strong answer signals:**
        
        - Knows from_tuples, from_product, set_index methods
        - Can access specific levels with loc and xs()
        - Understands stack/unstack transformations
        - Mentions GroupBy creates MultiIndex naturally
        - Uses sort_index(level=...) for sorting
        - Applies to time series and pivot tables

---

### Explain pd.merge() Parameters - Amazon, Microsoft Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Merge`, `Join`, `Data Combination` | **Asked by:** Amazon, Microsoft, Google, Meta

??? success "View Answer"

    **pd.merge()** combines DataFrames using SQL-style joins. Parameters: **how** (join type), **on** (keys), **left_on/right_on**, **suffixes**, **indicator**.

    **Complete Examples:**

    ```python
    import pandas as pd
    
    # Sample data
    customers = pd.DataFrame({
        'customer_id': [1, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Charlie', 'David'],
        'city': ['NYC', 'LA', 'Chicago', 'Boston']
    })
    
    orders = pd.DataFrame({
        'order_id': [101, 102, 103, 104, 105],
        'customer_id': [1, 2, 2, 5, 1],
        'amount': [100, 200, 150, 300, 50]
    })
    
    # 1. Inner Join (default)
    inner = pd.merge(customers, orders, on='customer_id', how='inner')
    print("Inner join:")
    print(inner)
    # Only matching records (customer_id 1, 2)
    
    # 2. Left Join
    left = pd.merge(customers, orders, on='customer_id', how='left')
    print("\nLeft join:")
    print(left)
    # All customers, NaN for unmatched orders
    
    # 3. Right Join
    right = pd.merge(customers, orders, on='customer_id', how='right')
    print("\nRight join:")
    print(right)
    # All orders, NaN for unmatched customers
    
    # 4. Outer Join
    outer = pd.merge(customers, orders, on='customer_id', how='outer')
    print("\nOuter join:")
    print(outer)
    # All records, NaN where no match
    
    # 5. Different Column Names
    products = pd.DataFrame({
        'product_id': [1, 2, 3],
        'name': ['Widget', 'Gadget', 'Tool']
    })
    
    order_items = pd.DataFrame({
        'order_id': [101, 102],
        'prod_id': [1, 2],
        'quantity': [5, 3]
    })
    
    merged = pd.merge(
        products, order_items,
        left_on='product_id',
        right_on='prod_id',
        how='inner'
    )
    print("\nDifferent column names:")
    print(merged)
    
    # 6. Multiple Keys
    df1 = pd.DataFrame({
        'year': [2023, 2023, 2024],
        'quarter': ['Q1', 'Q2', 'Q1'],
        'revenue': [100, 150, 120]
    })
    
    df2 = pd.DataFrame({
        'year': [2023, 2023, 2024],
        'quarter': ['Q1', 'Q2', 'Q2'],
        'profit': [20, 30, 25]
    })
    
    multi_key = pd.merge(df1, df2, on=['year', 'quarter'], how='outer')
    print("\nMultiple keys:")
    print(multi_key)
    
    # 7. Suffixes for Duplicate Columns
    left_df = pd.DataFrame({
        'key': ['A', 'B'],
        'value': [1, 2],
        'extra': [10, 20]
    })
    
    right_df = pd.DataFrame({
        'key': ['A', 'C'],
        'value': [3, 4],
        'extra': [30, 40]
    })
    
    with_suffixes = pd.merge(
        left_df, right_df,
        on='key',
        how='outer',
        suffixes=('_left', '_right')
    )
    print("\nWith suffixes:")
    print(with_suffixes)
    
    # 8. Indicator Column
    with_indicator = pd.merge(
        customers, orders,
        on='customer_id',
        how='outer',
        indicator=True
    )
    print("\nWith indicator:")
    print(with_indicator)
    # Shows source of each row
    
    # Custom indicator name
    custom_indicator = pd.merge(
        customers, orders,
        on='customer_id',
        how='outer',
        indicator='source'
    )
    
    # 9. Index-based Merge
    left_idx = pd.DataFrame({'A': [1, 2]}, index=['a', 'b'])
    right_idx = pd.DataFrame({'B': [3, 4]}, index=['a', 'c'])
    
    idx_merge = pd.merge(
        left_idx, right_idx,
        left_index=True,
        right_index=True,
        how='outer'
    )
    print("\nIndex merge:")
    print(idx_merge)
    
    # 10. Validate Merge
    # Ensure one-to-one merge
    try:
        pd.merge(
            customers, orders,
            on='customer_id',
            validate='one_to_one'  # Will fail (one customer, many orders)
        )
    except Exception as e:
        print(f"Validation error: {e}")
    
    # Valid one-to-many
    valid = pd.merge(
        customers, orders,
        on='customer_id',
        validate='one_to_many'  # OK
    )
    ```

    **Join Types:**

    | how | SQL Equivalent | Description | Use Case |
    |-----|---------------|-------------|----------|
    | **inner** | INNER JOIN | Only matches | Common records |
    | **left** | LEFT JOIN | All left + matches | Keep all left |
    | **right** | RIGHT JOIN | All right + matches | Keep all right |
    | **outer** | FULL OUTER JOIN | All records | Keep everything |
    | **cross** | CROSS JOIN | Cartesian product | All combinations |

    **Key Parameters:**

    | Parameter | Description | Example |
    |-----------|-------------|---------|
    | **on** | Common column(s) | `on='customer_id'` |
    | **left_on/right_on** | Different names | `left_on='id', right_on='cust_id'` |
    | **how** | Join type | `how='left'` |
    | **suffixes** | Duplicate columns | `suffixes=('_x', '_y')` |
    | **indicator** | Show merge source | `indicator=True` |
    | **validate** | Check cardinality | `validate='one_to_one'` |

    !!! tip "Interviewer's Insight"
        **What they're testing:** SQL join knowledge and data combination strategies.
        
        **Strong answer signals:**
        
        - Knows all join types: inner, left, right, outer
        - Uses left_on/right_on for different column names
        - Applies suffixes to avoid duplicate columns
        - Mentions indicator=True for tracking merge source
        - Knows validate parameter for cardinality checks
        - Can merge on index with left_index/right_index

---

### Explain apply() vs map() vs applymap() - Google, Netflix Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `apply`, `map`, `Performance` | **Asked by:** Google, Netflix, Meta, Amazon

??? success "View Answer"

    **map()** for Series element-wise. **apply()** for Series/DataFrame row/column-wise. **applymap()** (deprecated) for DataFrame element-wise. Use vectorized operations when possible.

    **Complete Examples:**

    ```python
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': [10, 20, 30, 40],
        'C': ['a', 'b', 'c', 'd']
    })
    
    # 1. Series.map() - Element-wise
    # Map with dict
    df['C_mapped'] = df['C'].map({'a': 'apple', 'b': 'banana', 'c': 'cherry'})
    print(df)
    
    # Map with function
    df['A_squared'] = df['A'].map(lambda x: x ** 2)
    
    # 2. Series.apply() - Element-wise (like map)
    df['A_cubed'] = df['A'].apply(lambda x: x ** 3)
    
    # More complex function
    def categorize(value):
        if value < 2:
            return 'low'
        elif value < 4:
            return 'medium'
        else:
            return 'high'
    
    df['category'] = df['A'].apply(categorize)
    
    # 3. DataFrame.apply() - Row/Column-wise
    # Apply to columns (axis=0, default)
    col_sums = df[['A', 'B']].apply(sum)
    print("Column sums:", col_sums)
    
    # Apply to rows (axis=1)
    row_sums = df[['A', 'B']].apply(sum, axis=1)
    print("Row sums:", row_sums)
    
    # Return Series from apply
    def compute_stats(row):
        return pd.Series({
            'sum': row['A'] + row['B'],
            'product': row['A'] * row['B']
        })
    
    stats = df.apply(compute_stats, axis=1)
    print("\nStats:")
    print(stats)
    
    # 4. DataFrame.applymap() - Element-wise (deprecated)
    # Use df.map() in pandas 2.1+
    numeric_df = df[['A', 'B']]
    
    # Old way (deprecated):
    # result = numeric_df.applymap(lambda x: x * 2)
    
    # New way:
    result = numeric_df.map(lambda x: x * 2)
    print("\nElement-wise operation:")
    print(result)
    
    # 5. Performance Comparison
    import time
    
    df_large = pd.DataFrame({
        'A': np.random.randint(0, 100, 100000),
        'B': np.random.randint(0, 100, 100000)
    })
    
    # Vectorized (fastest)
    start = time.time()
    result_vec = df_large['A'] * 2 + df_large['B']
    vec_time = time.time() - start
    print(f"Vectorized: {vec_time:.4f}s")
    
    # apply (slower)
    start = time.time()
    result_apply = df_large.apply(lambda row: row['A'] * 2 + row['B'], axis=1)
    apply_time = time.time() - start
    print(f"apply(): {apply_time:.4f}s ({apply_time/vec_time:.0f}x slower)")
    
    # 6. When to Use Each
    # map: Series, simple transformations, dict mapping
    status_map = {1: 'active', 2: 'inactive', 3: 'pending'}
    df['status'] = df['A'].map(status_map)
    
    # apply: Series, more complex functions, multiple outputs
    def complex_transform(x):
        return x ** 2 if x % 2 == 0 else x ** 3
    
    df['complex'] = df['A'].apply(complex_transform)
    
    # DataFrame.apply: row/column aggregations, multi-column operations
    def calculate_score(row):
        base = row['A'] * row['B']
        bonus = 10 if row['A'] > 2 else 0
        return base + bonus
    
    df['score'] = df.apply(calculate_score, axis=1)
    
    # 7. Vectorized Alternative (Best Performance)
    # Instead of apply for simple operations:
    # Slow:
    # df['sum'] = df.apply(lambda row: row['A'] + row['B'], axis=1)
    
    # Fast:
    df['sum'] = df['A'] + df['B']
    
    # Instead of apply with conditions:
    # Slow:
    # df['flag'] = df['A'].apply(lambda x: 'high' if x > 2 else 'low')
    
    # Fast:
    df['flag'] = np.where(df['A'] > 2, 'high', 'low')
    
    # 8. Result Type Parameter
    df_nums = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    
    # Return DataFrame
    result_df = df_nums.apply(lambda x: x * 2, result_type='expand')
    print(result_df)
    
    # Return list (broadcast)
    result_broadcast = df_nums.apply(lambda x: [1, 2], result_type='broadcast')
    print(result_broadcast)
    ```

    **Method Comparison:**

    | Method | Scope | Axis | Use Case | Speed |
    |--------|-------|------|----------|-------|
    | **Series.map()** | Element | N/A | Dict mapping, simple transform | Medium |
    | **Series.apply()** | Element | N/A | Complex functions | Medium |
    | **DataFrame.apply()** | Row/Column | 0/1 | Multi-column, aggregations | Slow |
    | **DataFrame.map()** | Element | N/A | Element-wise transform | Medium |
    | **Vectorized** | All | N/A | Math operations | **Fast** |

    **Performance Tips:**

    | Slow | Fast | Speedup |
    |------|------|---------|
    | `df.apply(lambda r: r['A']+r['B'], axis=1)` | `df['A'] + df['B']` | 100x+ |
    | `df['A'].apply(lambda x: x*2)` | `df['A'] * 2` | 50x+ |
    | `df['A'].apply(lambda x: 'Y' if x>5 else 'N')` | `np.where(df['A']>5, 'Y', 'N')` | 20x+ |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Performance optimization and function application methods.
        
        **Strong answer signals:**
        
        - Distinguishes map() for Series vs apply() for both
        - Knows applymap() is deprecated (use df.map())
        - Understands axis=0 (columns) vs axis=1 (rows)
        - Prefers vectorized operations (100x+ faster)
        - Uses np.where() instead of apply for conditionals
        - Only uses apply when vectorization impossible

---

### Explain pd.cut() vs pd.qcut() - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Binning`, `Discretization`, `Data Preparation` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **pd.cut()** bins data into equal-width intervals. **pd.qcut()** bins into equal-frequency quantiles. Use cut for fixed ranges, qcut for balanced distribution.

    **Complete Examples:**

    ```python
    import pandas as pd
    import numpy as np
    
    # Sample data
    ages = np.array([18, 22, 25, 27, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80])
    scores = np.array([45, 67, 89, 72, 81, 55, 93, 60, 77, 85, 92, 68, 74, 82, 90])
    
    # 1. pd.cut() - Equal-width Bins
    # Create 3 bins of equal width
    age_bins = pd.cut(ages, bins=3)
    print("cut() with 3 bins:")
    print(age_bins)
    print(age_bins.value_counts())
    
    # 2. pd.cut() with Custom Labels
    age_categories = pd.cut(
        ages,
        bins=[0, 30, 50, 100],
        labels=['Young', 'Middle-aged', 'Senior']
    )
    print("\nCustom labels:")
    print(age_categories)
    
    # 3. pd.cut() with right=False
    # Default: intervals are right-closed (a, b]
    # With right=False: intervals are left-closed [a, b)
    bins_left = pd.cut(ages, bins=3, right=False)
    print("\nLeft-closed intervals:")
    print(bins_left)
    
    # 4. pd.qcut() - Equal-frequency Quantiles
    # Create 4 quantiles (quartiles)
    score_quartiles = pd.qcut(scores, q=4)
    print("\nqcut() quartiles:")
    print(score_quartiles)
    print(score_quartiles.value_counts())
    
    # 5. pd.qcut() with Custom Labels
    score_categories = pd.qcut(
        scores,
        q=4,
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    print("\nQuantile labels:")
    print(score_categories)
    
    # 6. Specific Quantiles
    # Deciles (10 groups)
    deciles = pd.qcut(scores, q=10)
    print("\nDeciles:")
    print(deciles.value_counts())
    
    # Custom quantiles
    custom_quantiles = pd.qcut(
        scores,
        q=[0, 0.25, 0.5, 0.75, 1.0],
        labels=['Q1', 'Q2', 'Q3', 'Q4']
    )
    
    # 7. Include Lowest
    # Include lowest value in first bin
    with_lowest = pd.cut(
        ages,
        bins=[18, 30, 50, 80],
        include_lowest=True
    )
    print("\nInclude lowest:")
    print(with_lowest)
    
    # 8. Duplicate Edges Handling
    # When data has duplicates, qcut may fail
    data_with_dups = np.array([1, 1, 1, 1, 2, 3, 4, 5])
    
    # This would fail:
    # pd.qcut(data_with_dups, q=4)
    
    # Use duplicates='drop'
    safe_qcut = pd.qcut(data_with_dups, q=4, duplicates='drop')
    print("\nWith duplicates:")
    print(safe_qcut)
    
    # 9. Return Bins
    # Get bin edges
    values, bin_edges = pd.cut(ages, bins=3, retbins=True)
    print("\nBin edges:")
    print(bin_edges)
    
    # 10. Practical Example: Customer Segmentation
    df = pd.DataFrame({
        'customer_id': range(1, 16),
        'age': ages,
        'purchase_amount': scores * 10,
        'frequency': np.random.randint(1, 20, 15)
    })
    
    # Age groups (equal width)
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 30, 50, 100],
        labels=['Young', 'Middle', 'Senior']
    )
    
    # Spending tiers (equal frequency)
    df['spending_tier'] = pd.qcut(
        df['purchase_amount'],
        q=3,
        labels=['Low', 'Medium', 'High']
    )
    
    # Frequency quintiles
    df['freq_quintile'] = pd.qcut(
        df['frequency'],
        q=5,
        labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
        duplicates='drop'
    )
    
    print("\nCustomer Segmentation:")
    print(df)
    
    # Analysis by segment
    print("\nAverage purchase by age group:")
    print(df.groupby('age_group')['purchase_amount'].mean())
    
    print("\nCustomers per spending tier:")
    print(df['spending_tier'].value_counts())
    ```

    **cut() vs qcut():**

    | Aspect | pd.cut() | pd.qcut() |
    |--------|----------|-----------|
    | **Method** | Equal-width intervals | Equal-frequency quantiles |
    | **Bins** | Same size ranges | Same number of items |
    | **Use Case** | Fixed ranges (e.g., age groups) | Balanced distribution |
    | **Example** | 0-20, 20-40, 40-60 | Each bin has ~same count |
    | **Distribution** | Uneven counts | Even counts |

    **Key Parameters:**

    | Parameter | cut() | qcut() | Description |
    |-----------|-------|--------|-------------|
    | **bins/q** | ✅ | ✅ | Number of bins/quantiles |
    | **labels** | ✅ | ✅ | Custom category names |
    | **right** | ✅ | ✅ | Right-closed intervals (default True) |
    | **include_lowest** | ✅ | ✅ | Include smallest value |
    | **duplicates** | ❌ | ✅ | Handle duplicate edges |
    | **retbins** | ✅ | ✅ | Return bin edges |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Data discretization and binning strategies.
        
        **Strong answer signals:**
        
        - Knows cut() creates equal-width intervals
        - Knows qcut() creates equal-frequency quantiles
        - Uses cut for fixed ranges (age groups, scores)
        - Uses qcut for balanced distributions (percentiles)
        - Applies custom labels parameter
        - Handles duplicates with duplicates='drop' in qcut

---

### Explain Categorical Data Type - Meta, Netflix Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Categorical`, `Memory Optimization`, `Performance` | **Asked by:** Meta, Netflix, Google, Amazon

??? success "View Answer"

    **Categorical** dtype stores repeated string values as integers with mapping, reducing memory usage by 90%+ for low-cardinality columns. Essential for large datasets.

    **Complete Examples:**

    ```python
    import pandas as pd
    import numpy as np
    
    # 1. Creating Categorical
    # From list
    colors = pd.Categorical(['red', 'blue', 'red', 'green', 'blue', 'red'])
    print("Categorical:", colors)
    print("Categories:", colors.categories)
    print("Codes:", colors.codes)
    
    # 2. DataFrame Column
    df = pd.DataFrame({
        'product': ['A', 'B', 'A', 'C', 'B', 'A'] * 10000,
        'size': ['S', 'M', 'L', 'M', 'S', 'L'] * 10000
    })
    
    # Memory before
    memory_before = df.memory_usage(deep=True).sum() / 1024**2
    print(f"\nMemory before: {memory_before:.2f} MB")
    
    # Convert to categorical
    df['product'] = df['product'].astype('category')
    df['size'] = df['size'].astype('category')
    
    # Memory after
    memory_after = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory after: {memory_after:.2f} MB")
    print(f"Reduction: {(1 - memory_after/memory_before)*100:.1f}%")
    
    # 3. Ordered Categories
    sizes = pd.Categorical(
        ['S', 'M', 'L', 'M', 'S', 'XL'],
        categories=['S', 'M', 'L', 'XL'],
        ordered=True
    )
    
    print("\nOrdered categorical:")
    print(sizes)
    print("S < M:", sizes[0] < sizes[1])
    
    # 4. Adding/Removing Categories
    cat = pd.Categorical(['a', 'b', 'c'])
    
    # Add category
    cat = cat.add_categories(['d'])
    print("\nAdded category:", cat.categories)
    
    # Remove category
    cat = cat.remove_categories(['d'])
    print("Removed category:", cat.categories)
    
    # Set categories
    cat = cat.set_categories(['a', 'b', 'c', 'd', 'e'])
    print("Set categories:", cat.categories)
    
    # 5. Reordering Categories
    df_survey = pd.DataFrame({
        'satisfaction': ['Good', 'Bad', 'Excellent', 'Good', 'Bad', 'Fair']
    })
    
    df_survey['satisfaction'] = pd.Categorical(
        df_survey['satisfaction'],
        categories=['Bad', 'Fair', 'Good', 'Excellent'],
        ordered=True
    )
    
    # Now can sort meaningfully
    df_sorted = df_survey.sort_values('satisfaction')
    print("\nSorted by category order:")
    print(df_sorted)
    
    # 6. Groupby with Categories
    df_sales = pd.DataFrame({
        'region': pd.Categorical(['North', 'South', 'North', 'East']),
        'sales': [100, 200, 150, 180]
    })
    
    # Add unused category
    df_sales['region'] = df_sales['region'].cat.add_categories(['West'])
    
    # Groupby includes empty categories
    grouped = df_sales.groupby('region', observed=False).sum()
    print("\nGroupby with empty category:")
    print(grouped)
    
    # Exclude empty categories
    grouped_observed = df_sales.groupby('region', observed=True).sum()
    print("\nGroupby observed only:")
    print(grouped_observed)
    
    # 7. Renaming Categories
    df_status = pd.DataFrame({
        'status': pd.Categorical(['A', 'B', 'A', 'C'])
    })
    
    # Rename
    df_status['status'] = df_status['status'].cat.rename_categories({
        'A': 'Active',
        'B': 'Blocked',
        'C': 'Closed'
    })
    print("\nRenamed categories:")
    print(df_status)
    
    # 8. Performance Comparison
    # String operations
    df_str = pd.DataFrame({
        'category': ['A', 'B', 'C'] * 100000
    })
    
    df_cat = df_str.copy()
    df_cat['category'] = df_cat['category'].astype('category')
    
    # Comparison speed
    import time
    
    start = time.time()
    _ = df_str['category'] == 'A'
    str_time = time.time() - start
    
    start = time.time()
    _ = df_cat['category'] == 'A'
    cat_time = time.time() - start
    
    print(f"\nString comparison: {str_time:.4f}s")
    print(f"Categorical comparison: {cat_time:.4f}s")
    print(f"Speedup: {str_time/cat_time:.1f}x")
    
    # 9. When to Use Categorical
    # Good: low cardinality (few unique values)
    df_good = pd.DataFrame({
        'country': ['USA', 'UK', 'Canada'] * 10000  # 3 unique
    })
    df_good['country'] = df_good['country'].astype('category')
    
    # Bad: high cardinality (many unique values)
    df_bad = pd.DataFrame({
        'user_id': [f'user_{i}' for i in range(10000)]  # All unique
    })
    # Don't convert to categorical - no benefit!
    
    # 10. Practical Example: Survey Data
    df_survey = pd.DataFrame({
        'age_group': pd.Categorical(
            ['18-25', '26-35', '18-25', '36-45', '26-35'],
            categories=['18-25', '26-35', '36-45', '46-55', '55+'],
            ordered=True
        ),
        'satisfaction': pd.Categorical(
            ['Good', 'Bad', 'Excellent', 'Good', 'Fair'],
            categories=['Bad', 'Fair', 'Good', 'Excellent'],
            ordered=True
        ),
        'product': pd.Categorical(['A', 'B', 'A', 'C', 'B'])
    })
    
    # Efficient aggregation
    result = df_survey.groupby(['age_group', 'product'], observed=False).size()
    print("\nSurvey analysis:")
    print(result)
    ```

    **Benefits:**

    | Benefit | Description | Example Savings |
    |---------|-------------|-----------------|
    | **Memory** | Stores as int codes | 90%+ reduction |
    | **Speed** | Faster comparisons | 2-10x faster |
    | **Ordering** | Meaningful sorts | Survey ratings |
    | **Groupby** | Include empty categories | Complete reports |

    **When to Use:**

    | Use Categorical | Don't Use Categorical |
    |-----------------|----------------------|
    | Low cardinality (<50% unique) | High cardinality (90%+ unique) |
    | Repeated values | Mostly unique values |
    | Surveys, ratings | User IDs, timestamps |
    | Country, department | Email addresses |
    | Fixed set of values | Free-form text |

    **Key Operations:**

    | Operation | Method | Example |
    |-----------|--------|---------|
    | **Create** | `astype('category')` | `df['col'].astype('category')` |
    | **Add category** | `.cat.add_categories()` | `.cat.add_categories(['new'])` |
    | **Remove** | `.cat.remove_categories()` | `.cat.remove_categories(['old'])` |
    | **Rename** | `.cat.rename_categories()` | `.cat.rename_categories({'A': 'Active'})` |
    | **Reorder** | `.cat.reorder_categories()` | `.cat.reorder_categories(['S', 'M', 'L'])` |
    | **Set ordered** | `.cat.as_ordered()` | `.cat.as_ordered()` |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Memory optimization and categorical data handling.
        
        **Strong answer signals:**
        
        - Knows categorical stores strings as integer codes with mapping
        - Mentions 90%+ memory reduction for low-cardinality columns
        - Uses ordered=True for meaningful sorting (e.g., ratings)
        - Applies when <50% unique values with repetition
        - Avoids for high cardinality data (unique IDs)
        - Uses .cat accessor for category management
        - Sets observed=False to include empty categories in groupby

---

### Explain Window Functions (Rolling, Expanding, EWMA) - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Window Functions`, `Time Series`, `Rolling Statistics` | **Asked by:** Google, Amazon, Meta, Netflix

??? success "View Answer"

    **Window functions** compute statistics over sliding, expanding, or exponentially weighted windows. Essential for time series analysis, moving averages, and trend detection.

    **Complete Examples:**

    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Sample time series data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'sales': np.random.randint(80, 120, 100) + np.sin(np.arange(100) * 0.1) * 20,
        'temperature': np.random.normal(70, 10, 100)
    })
    df.set_index('date', inplace=True)
    
    # 1. Rolling Window (Moving Average)
    # 7-day moving average
    df['sales_ma7'] = df['sales'].rolling(window=7).mean()
    
    # 30-day moving average
    df['sales_ma30'] = df['sales'].rolling(window=30).mean()
    
    print("Rolling means:")
    print(df[['sales', 'sales_ma7', 'sales_ma30']].head(35))
    
    # 2. Rolling with min_periods
    # Calculate even with fewer than 7 values
    df['sales_ma7_min'] = df['sales'].rolling(window=7, min_periods=1).mean()
    
    # 3. Rolling Aggregations
    df['sales_roll_std'] = df['sales'].rolling(window=7).std()
    df['sales_roll_min'] = df['sales'].rolling(window=7).min()
    df['sales_roll_max'] = df['sales'].rolling(window=7).max()
    df['sales_roll_sum'] = df['sales'].rolling(window=7).sum()
    
    # Multiple aggregations
    rolling_stats = df['sales'].rolling(window=7).agg(['mean', 'std', 'min', 'max'])
    print("\nRolling statistics:")
    print(rolling_stats.head(10))
    
    # 4. Expanding Window
    # Cumulative average (all data up to current point)
    df['sales_cumavg'] = df['sales'].expanding().mean()
    
    # Cumulative sum
    df['sales_cumsum'] = df['sales'].expanding().sum()
    
    # Cumulative max
    df['sales_cummax'] = df['sales'].expanding().max()
    
    print("\nExpanding statistics:")
    print(df[['sales', 'sales_cumavg', 'sales_cumsum', 'sales_cummax']].head(10))
    
    # 5. Exponentially Weighted Moving Average (EWMA)
    # More weight to recent values
    df['sales_ewma'] = df['sales'].ewm(span=7).mean()
    
    # Different smoothing parameters
    df['sales_ewm_fast'] = df['sales'].ewm(span=3).mean()  # Fast response
    df['sales_ewm_slow'] = df['sales'].ewm(span=14).mean()  # Slow response
    
    # Alpha parameter (0-1)
    df['sales_ewm_alpha'] = df['sales'].ewm(alpha=0.3).mean()
    
    # 6. Center Parameter
    # Center the window (for smoothing, not forecasting)
    df['sales_centered'] = df['sales'].rolling(window=7, center=True).mean()
    
    print("\nCentered vs non-centered:")
    print(df[['sales', 'sales_ma7', 'sales_centered']].iloc[3:8])
    
    # 7. Rolling with Custom Function
    def range_ratio(x):
        """Range divided by mean."""
        return (x.max() - x.min()) / x.mean() if x.mean() != 0 else 0
    
    df['sales_volatility'] = df['sales'].rolling(window=7).apply(range_ratio)
    
    # 8. Time-based Window
    # 7-day window (calendar days, not data points)
    df['sales_7d'] = df['sales'].rolling('7D').mean()
    
    # 14-day window
    df['sales_14d'] = df['sales'].rolling('14D').mean()
    
    # 9. Shift and Diff
    # Previous value
    df['sales_prev'] = df['sales'].shift(1)
    
    # Change from previous day
    df['sales_change'] = df['sales'].diff()
    
    # Percentage change
    df['sales_pct_change'] = df['sales'].pct_change()
    
    # Change from 7 days ago
    df['sales_change_7d'] = df['sales'].diff(7)
    
    print("\nChanges:")
    print(df[['sales', 'sales_prev', 'sales_change', 'sales_pct_change']].head(10))
    
    # 10. Practical Example: Anomaly Detection
    # Bollinger Bands
    window = 20
    df['bb_mid'] = df['sales'].rolling(window=window).mean()
    df['bb_std'] = df['sales'].rolling(window=window).std()
    df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * 2)
    
    # Detect anomalies
    df['anomaly'] = ((df['sales'] > df['bb_upper']) | 
                     (df['sales'] < df['bb_lower']))
    
    print("\nAnomalies detected:")
    print(df[df['anomaly']][['sales', 'bb_upper', 'bb_lower']])
    
    # 11. Multi-column Rolling
    # Rolling correlation
    df['corr_sales_temp'] = df['sales'].rolling(window=30).corr(df['temperature'])
    
    # 12. GroupBy + Rolling
    # Add categories
    df['category'] = np.random.choice(['A', 'B'], len(df))
    
    # Rolling average per category
    df['sales_ma_by_cat'] = df.groupby('category')['sales'].rolling(window=7).mean().reset_index(0, drop=True)
    ```

    **Window Types:**

    | Type | Method | Use Case | Example |
    |------|--------|----------|---------|
    | **Rolling** | `.rolling(window)` | Moving average | 7-day MA |
    | **Expanding** | `.expanding()` | Cumulative stats | Running total |
    | **EWMA** | `.ewm(span/alpha)` | Weighted average | Trend smoothing |

    **Common Rolling Operations:**

    | Operation | Method | Description |
    |-----------|--------|-------------|
    | **Mean** | `.rolling(n).mean()` | Moving average |
    | **Sum** | `.rolling(n).sum()` | Moving total |
    | **Std** | `.rolling(n).std()` | Moving volatility |
    | **Min/Max** | `.rolling(n).min/max()` | Moving range |
    | **Corr** | `.rolling(n).corr()` | Moving correlation |
    | **Apply** | `.rolling(n).apply(func)` | Custom function |

    **Key Parameters:**

    | Parameter | Description | Example |
    |-----------|-------------|---------|
    | **window** | Window size | `window=7` |
    | **min_periods** | Min values needed | `min_periods=1` |
    | **center** | Center the window | `center=True` |
    | **span** | EWMA span | `span=7` |
    | **alpha** | EWMA smoothing | `alpha=0.3` |

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "rolling(): moving average over fixed window"
        - "expanding(): cumulative statistics"
        - "ewm(): exponentially weighted (recent values weighted more)"
        - "min_periods: handle NaN at start"
        - "center=True: for smoothing (not forecasting)"
        - "Time-based: `rolling('7D')` for calendar days"
        - "Shift: `df['prev'] = df['col'].shift(1)`"
        - "Diff: `df['change'] = df['col'].diff()`"
        - "Use for: moving averages, anomaly detection, trend analysis"

---

### Explain DataFrame Indexing (.loc vs .iloc vs .at vs .iat) - Microsoft, Amazon Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Indexing`, `DataFrame Access`, `Performance` | **Asked by:** Microsoft, Amazon, Google, Meta

??? success "View Answer"

    **.loc** accesses by label, **.iloc** by integer position, **.at/.iat** for single values (faster). Use loc for label-based, iloc for position-based indexing.

    **Complete Examples:**

    ```python
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': ['a', 'b', 'c', 'd', 'e']
    }, index=['row1', 'row2', 'row3', 'row4', 'row5'])
    
    # 1. .loc - Label-based
    # Single value
    print(df.loc['row1', 'A'])  # 1
    
    # Row
    print(df.loc['row2'])  # Series
    
    # Multiple rows
    print(df.loc[['row1', 'row3']])
    
    # Slice (inclusive)
    print(df.loc['row1':'row3'])  # Includes row3!
    
    # Boolean indexing
    print(df.loc[df['A'] > 2])
    
    # Multiple columns
    print(df.loc['row1', ['A', 'B']])
    
    # 2. .iloc - Position-based
    # Single value
    print(df.iloc[0, 0])  # 1
    
    # Row by position
    print(df.iloc[1])  # Second row
    
    # Multiple rows
    print(df.iloc[[0, 2, 4]])
    
    # Slice (exclusive end)
    print(df.iloc[0:3])  # Rows 0, 1, 2 (not 3)
    
    # Negative indexing
    print(df.iloc[-1])  # Last row
    
    # 2D slicing
    print(df.iloc[1:3, 0:2])  # Rows 1-2, columns 0-1
    
    # 3. .at - Fast single value (label)
    value = df.at['row1', 'A']
    print(value)  # 1
    
    # Set value
    df.at['row1', 'A'] = 100
    
    # 4. .iat - Fast single value (position)
    value = df.iat[0, 0]
    print(value)  # 100
    
    # Set value
    df.iat[0, 0] = 1
    
    # 5. Performance Comparison
    import time
    
    large_df = pd.DataFrame(np.random.rand(10000, 100))
    
    # .loc timing
    start = time.time()
    for _ in range(1000):
        _ = large_df.loc[500, 50]
    loc_time = time.time() - start
    
    # .at timing (faster)
    start = time.time()
    for _ in range(1000):
        _ = large_df.at[500, 50]
    at_time = time.time() - start
    
    print(f".loc: {loc_time:.4f}s")
    print(f".at: {at_time:.4f}s ({loc_time/at_time:.1f}x slower)")
    
    # 6. Setting Values
    # Set single value
    df.loc['row1', 'A'] = 999
    
    # Set multiple values
    df.loc['row1', ['A', 'B']] = [1, 2]
    
    # Set column
    df.loc[:, 'A'] = 0
    
    # Set with condition
    df.loc[df['A'] > 2, 'B'] = 100
    
    # 7. Adding Rows/Columns
    # Add row
    df.loc['row6'] = [6, 60, 'f']
    
    # Add column
    df.loc[:, 'D'] = [1, 2, 3, 4, 5, 6]
    
    # 8. Common Pitfalls
    # Chain indexing (bad!)
    # df[df['A'] > 2]['B'] = 999  # SettingWithCopyWarning
    
    # Correct way
    df.loc[df['A'] > 2, 'B'] = 999
    
    # 9. MultiIndex
    df_multi = pd.DataFrame({
        'value': [1, 2, 3, 4]
    }, index=pd.MultiIndex.from_tuples([
        ('A', 1), ('A', 2), ('B', 1), ('B', 2)
    ]))
    
    # Access with tuple
    print(df_multi.loc[('A', 1)])
    
    # Slice first level
    print(df_multi.loc['A'])
    
    # 10. Boolean + Label
    # Combine conditions
    result = df.loc[(df['A'] > 1) & (df['B'] < 50), ['A', 'C']]
    print(result)
    ```

    **Comparison:**

    | Method | Indexing Type | Speed | Use Case |
    |--------|---------------|-------|----------|
    | **.loc** | Label-based | Medium | General access by label |
    | **.iloc** | Position-based | Medium | Access by position |
    | **.at** | Label-based | **Fast** | Single value by label |
    | **.iat** | Position-based | **Fast** | Single value by position |

    **Key Differences:**

    | Aspect | .loc | .iloc |
    |--------|------|-------|
    | **Input** | Labels | Integers |
    | **Slice** | Inclusive end | Exclusive end |
    | **Example** | `df.loc['row1':'row3']` | `df.iloc[0:3]` |
    | **Boolean** | ✅ Yes | ❌ No |

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "loc: label-based, iloc: position-based"
        - "loc slicing: inclusive end"
        - "iloc slicing: exclusive end (like Python)"
        - "at/iat: faster for single values"
        - "Use loc with boolean indexing"
        - "Avoid chained indexing: `df[col][row]`"
        - "Correct: `df.loc[row, col]`"
        - "at/iat: 10-100x faster for scalar access"

    !!! tip "Interviewer's Insight"
        **What they're testing:** DataFrame indexing fundamentals and performance.
        
        **Strong answer signals:**
        
        - Knows loc uses labels, iloc uses integer positions
        - Mentions at/iat are 10-100x faster for single values
        - Understands loc slicing is inclusive, iloc is exclusive
        - Avoids chained indexing to prevent SettingWithCopyWarning
        - Uses loc for boolean indexing and filtering

---

### Explain pd.concat() vs pd.merge() - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `concat`, `merge`, `Data Combination` | **Asked by:** Amazon, Google, Meta, Microsoft

??? success "View Answer"

    **pd.concat()** stacks DataFrames vertically/horizontally. **pd.merge()** combines based on keys (SQL joins). Use concat for stacking, merge for joining on columns.

    **Complete Examples:**

    ```python
    import pandas as pd
    
    # Sample data
    df1 = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    
    df2 = pd.DataFrame({
        'A': [7, 8, 9],
        'B': [10, 11, 12]
    })
    
    df3 = pd.DataFrame({
        'C': [13, 14, 15],
        'D': [16, 17, 18]
    })
    
    # 1. concat() - Vertical (default axis=0)
    vertical = pd.concat([df1, df2])
    print("Vertical concat:")
    print(vertical)
    # Stacks rows, keeps all columns
    
    # Reset index
    vertical_reset = pd.concat([df1, df2], ignore_index=True)
    print("\nWith reset index:")
    print(vertical_reset)
    
    # 2. concat() - Horizontal (axis=1)
    horizontal = pd.concat([df1, df3], axis=1)
    print("\nHorizontal concat:")
    print(horizontal)
    # Side-by-side, matches by index
    
    # 3. concat() with Keys
    keyed = pd.concat([df1, df2], keys=['first', 'second'])
    print("\nWith keys:")
    print(keyed)
    # Creates MultiIndex
    
    # 4. concat() with Missing Columns
    df4 = pd.DataFrame({
        'A': [1, 2],
        'C': [3, 4]  # Different column
    })
    
    mixed = pd.concat([df1, df4])
    print("\nMissing columns (filled with NaN):")
    print(mixed)
    
    # Inner join (only common columns)
    inner = pd.concat([df1, df4], join='inner')
    print("\nInner join:")
    print(inner)
    
    # 5. merge() - Inner Join
    customers = pd.DataFrame({
        'customer_id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie']
    })
    
    orders = pd.DataFrame({
        'order_id': [101, 102, 103],
        'customer_id': [1, 2, 4],
        'amount': [100, 200, 300]
    })
    
    inner_merge = pd.merge(customers, orders, on='customer_id')
    print("\nInner merge:")
    print(inner_merge)
    # Only matching customer_ids (1, 2)
    
    # 6. merge() - Outer Join
    outer_merge = pd.merge(customers, orders, on='customer_id', how='outer')
    print("\nOuter merge:")
    print(outer_merge)
    # All records, NaN for missing
    
    # 7. When to Use Each
    # concat: Same structure, stack them
    jan = pd.DataFrame({'sales': [100, 200]})
    feb = pd.DataFrame({'sales': [150, 250]})
    mar = pd.DataFrame({'sales': [120, 220]})
    
    quarterly = pd.concat([jan, feb, mar], ignore_index=True)
    print("\nQuarterly sales (concat):")
    print(quarterly)
    
    # merge: Different tables, join on key
    products = pd.DataFrame({
        'product_id': [1, 2],
        'name': ['Widget', 'Gadget']
    })
    
    sales = pd.DataFrame({
        'product_id': [1, 2, 1],
        'quantity': [5, 3, 2]
    })
    
    sales_detail = pd.merge(sales, products, on='product_id')
    print("\nSales detail (merge):")
    print(sales_detail)
    
    # 8. concat Multiple DataFrames
    dfs = [df1, df2, df1, df2]
    result = pd.concat(dfs, ignore_index=True)
    print("\nMultiple concat:")
    print(result)
    
    # 9. Performance: concat vs append
    # append (deprecated) - slow in loop
    # result = pd.DataFrame()
    # for df in dfs:
    #     result = result.append(df)  # Slow!
    
    # concat - fast
    result = pd.concat(dfs)  # Much faster
    
    # 10. Practical Example
    # Monthly data files
    data_2023_q1 = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=90, freq='D'),
        'revenue': range(90)
    })
    
    data_2023_q2 = pd.DataFrame({
        'date': pd.date_range('2023-04-01', periods=91, freq='D'),
        'revenue': range(100, 191)
    })
    
    # Combine time series
    full_year = pd.concat([data_2023_q1, data_2023_q2], ignore_index=True)
    
    # Add metadata
    metadata = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=181, freq='D'),
        'holiday': [False] * 181
    })
    
    # Merge with metadata
    complete = pd.merge(full_year, metadata, on='date', how='left')
    print("\nComplete data:")
    print(complete.head())
    ```

    **concat() vs merge():**

    | Aspect | concat() | merge() |
    |--------|----------|---------|
    | **Purpose** | Stack DataFrames | Join on keys |
    | **Direction** | Vertical/Horizontal | Based on keys |
    | **Use Case** | Append rows, add columns | SQL-style joins |
    | **Key Required** | No | Yes |
    | **Example** | Monthly data → yearly | Customers + Orders |

    **concat() Parameters:**

    | Parameter | Description | Example |
    |-----------|-------------|---------|
    | **axis** | 0=vertical, 1=horizontal | `axis=0` |
    | **ignore_index** | Reset index | `ignore_index=True` |
    | **keys** | Create MultiIndex | `keys=['a', 'b']` |
    | **join** | 'outer' (default) or 'inner' | `join='inner'` |

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "concat: stack DataFrames (same structure)"
        - "merge: join on keys (different tables)"
        - "concat axis=0: vertical, axis=1: horizontal"
        - "concat ignore_index: reset index"
        - "merge for SQL-style joins"
        - "concat join='inner': only common columns"
        - "Use concat for: time series, appending rows"
        - "Use merge for: combining related tables"
        - "concat list of DFs: faster than repeated append"

---

### Explain GroupBy Aggregation Functions - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `GroupBy`, `Aggregation`, `Statistics` | **Asked by:** Google, Meta, Netflix, Amazon

??? success "View Answer"

    **GroupBy** splits data into groups, applies functions, combines results. Supports multiple aggregations, custom functions, and named aggregations with **agg()**.

    **Complete Examples:**

    ```python
    import pandas as pd
    import numpy as np
    
    # Sample sales data
    df = pd.DataFrame({
        'region': ['North', 'North', 'South', 'South', 'East', 'East'] * 5,
        'product': ['A', 'B', 'A', 'B', 'A', 'B'] * 5,
        'sales': np.random.randint(100, 500, 30),
        'quantity': np.random.randint(1, 20, 30),
        'cost': np.random.randint(50, 200, 30)
    })
    
    # 1. Basic GroupBy
    # Single aggregation
    region_sales = df.groupby('region')['sales'].sum()
    print("Sales by region:")
    print(region_sales)
    
    # Multiple columns
    by_region_product = df.groupby(['region', 'product'])['sales'].sum()
    print("\nSales by region and product:")
    print(by_region_product)
    
    # 2. Multiple Aggregations
    # Same function on all columns
    summary = df.groupby('region').sum()
    print("\nSum by region:")
    print(summary)
    
    # Multiple functions
    stats = df.groupby('region')['sales'].agg(['sum', 'mean', 'std', 'count'])
    print("\nMultiple stats:")
    print(stats)
    
    # 3. Different Functions per Column
    aggregations = df.groupby('region').agg({
        'sales': 'sum',
        'quantity': 'mean',
        'cost': ['min', 'max']
    })
    print("\nDifferent functions per column:")
    print(aggregations)
    
    # 4. Named Aggregations (Clean Column Names)
    result = df.groupby('region').agg(
        total_sales=('sales', 'sum'),
        avg_quantity=('quantity', 'mean'),
        max_cost=('cost', 'max'),
        sales_count=('sales', 'count')
    )
    print("\nNamed aggregations:")
    print(result)
    
    # 5. Custom Aggregation Functions
    def range_func(x):
        return x.max() - x.min()
    
    def coefficient_of_variation(x):
        return x.std() / x.mean() if x.mean() != 0 else 0
    
    custom = df.groupby('region')['sales'].agg([
        ('total', 'sum'),
        ('average', 'mean'),
        ('range', range_func),
        ('cv', coefficient_of_variation)
    ])
    print("\nCustom functions:")
    print(custom)
    
    # 6. Lambda Functions
    result = df.groupby('region').agg({
        'sales': [
            ('total', 'sum'),
            ('weighted_avg', lambda x: (x * df.loc[x.index, 'quantity']).sum() / df.loc[x.index, 'quantity'].sum())
        ]
    })
    print("\nLambda aggregation:")
    print(result)
    
    # 7. Transform (Keep Original Shape)
    # Add group mean to each row
    df['region_avg_sales'] = df.groupby('region')['sales'].transform('mean')
    
    # Normalize within group
    df['sales_normalized'] = df.groupby('region')['sales'].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    
    print("\nWith transform:")
    print(df[['region', 'sales', 'region_avg_sales', 'sales_normalized']].head())
    
    # 8. Filter Groups
    # Keep only groups with more than 5 records
    filtered = df.groupby('region').filter(lambda x: len(x) > 5)
    print("\nFiltered groups:")
    print(filtered.groupby('region').size())
    
    # Keep groups with high average sales
    high_sales = df.groupby('region').filter(lambda x: x['sales'].mean() > 250)
    
    # 9. Apply (Most Flexible)
    def top_products(group):
        """Get top 2 products by sales."""
        return group.nlargest(2, 'sales')
    
    top_per_region = df.groupby('region').apply(top_products)
    print("\nTop products per region:")
    print(top_per_region)
    
    # 10. GroupBy with MultiIndex
    multi_group = df.groupby(['region', 'product']).agg({
        'sales': ['sum', 'mean'],
        'quantity': 'sum'
    })
    print("\nMultiIndex groupby:")
    print(multi_group)
    
    # Flatten MultiIndex columns
    multi_group.columns = ['_'.join(col).strip() for col in multi_group.columns.values]
    print("\nFlattened columns:")
    print(multi_group)
    
    # 11. Cumulative Operations
    # Cumulative sum within group
    df['cumulative_sales'] = df.groupby('region')['sales'].cumsum()
    
    # Ranking within group
    df['sales_rank'] = df.groupby('region')['sales'].rank(ascending=False)
    
    print("\nCumulative operations:")
    print(df[['region', 'sales', 'cumulative_sales', 'sales_rank']].head(10))
    
    # 12. Practical Example: Sales Report
    report = df.groupby('region').agg(
        total_revenue=('sales', 'sum'),
        avg_sale=('sales', 'mean'),
        total_units=('quantity', 'sum'),
        num_transactions=('sales', 'count'),
        max_sale=('sales', 'max'),
        min_sale=('sales', 'min')
    )
    
    # Add calculated columns
    report['revenue_per_unit'] = report['total_revenue'] / report['total_units']
    report['avg_units_per_transaction'] = report['total_units'] / report['num_transactions']
    
    print("\nSales report:")
    print(report)
    ```

    **Common Aggregation Functions:**

    | Function | Description | Example |
    |----------|-------------|---------|
    | **sum** | Total | `groupby().sum()` |
    | **mean** | Average | `groupby().mean()` |
    | **count** | Count non-null | `groupby().count()` |
    | **size** | Count all (including null) | `groupby().size()` |
    | **std** | Standard deviation | `groupby().std()` |
    | **min/max** | Min/Max | `groupby().max()` |
    | **first/last** | First/Last value | `groupby().first()` |
    | **nunique** | Unique count | `groupby().nunique()` |

    **GroupBy Methods:**

    | Method | Purpose | Returns |
    |--------|---------|---------|
    | **agg()** | Apply aggregations | Aggregated data |
    | **transform()** | Keep original shape | Same shape as input |
    | **filter()** | Filter groups | Subset of data |
    | **apply()** | Custom function | Custom result |

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "groupby(): split-apply-combine pattern"
        - "agg(): apply multiple functions"
        - "Named agg: `total_sales=('sales', 'sum')`"
        - "transform(): keep original shape"
        - "filter(): keep/remove entire groups"
        - "apply(): most flexible, returns custom structure"
        - "Multiple grouping: `groupby(['col1', 'col2'])`"
        - "Custom functions: pass callable to agg()"
        - "Flatten MultiIndex: `'_'.join(col)`"

---

### Explain Pandas Query Method - Amazon, Netflix Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Query`, `Filtering`, `Performance` | **Asked by:** Amazon, Netflix, Google, Meta

??? success "View Answer"

    **query()** filters DataFrames using string expressions (SQL-like). Cleaner syntax than boolean indexing, supports variables with @, and can be faster for large DataFrames.

    **Complete Examples:**

    ```python
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25, 30, 35, 40, 28],
        'salary': [50000, 60000, 70000, 80000, 55000],
        'department': ['Sales', 'IT', 'Sales', 'IT', 'HR'],
        'experience': [2, 5, 8, 12, 3]
    })
    
    # 1. Basic Query
    # Simple condition
    result = df.query('age > 30')
    print("Age > 30:")
    print(result)
    
    # Multiple conditions (AND)
    result = df.query('age > 25 and salary > 55000')
    print("\nAge > 25 AND salary > 55000:")
    print(result)
    
    # OR condition
    result = df.query('department == "IT" or salary > 70000')
    print("\nIT OR high salary:")
    print(result)
    
    # 2. Comparison with Boolean Indexing
    # Boolean indexing (verbose)
    result_bool = df[(df['age'] > 25) & (df['salary'] > 55000)]
    
    # Query (cleaner)
    result_query = df.query('age > 25 and salary > 55000')
    
    # Both produce same result
    print("\nResults equal:", result_bool.equals(result_query))
    
    # 3. Using Variables with @
    min_age = 30
    min_salary = 60000
    
    result = df.query('age > @min_age and salary > @min_salary')
    print("\nWith variables:")
    print(result)
    
    # List membership
    departments = ['IT', 'HR']
    result = df.query('department in @departments')
    print("\nDepartment in list:")
    print(result)
    
    # 4. String Operations
    # String contains
    result = df.query('name.str.contains("a", case=False)', engine='python')
    print("\nName contains 'a':")
    print(result)
    
    # String startswith
    result = df.query('name.str.startswith("D")', engine='python')
    print("\nName starts with 'D':")
    print(result)
    
    # 5. Between
    # Age between 25 and 35
    result = df.query('25 <= age <= 35')
    print("\nAge between 25 and 35:")
    print(result)
    
    # 6. Not Equal
    result = df.query('department != "Sales"')
    print("\nNot Sales:")
    print(result)
    
    # 7. Complex Expressions
    # Calculate in query
    result = df.query('salary / experience > 10000')
    print("\nSalary per year experience > 10k:")
    print(result)
    
    # 8. Index Queries
    df_indexed = df.set_index('name')
    
    # Query index
    result = df_indexed.query('index == "Alice"')
    print("\nQuery index:")
    print(result)
    
    # 9. Performance Comparison
    large_df = pd.DataFrame({
        'A': np.random.randint(0, 100, 100000),
        'B': np.random.randint(0, 100, 100000),
        'C': np.random.randint(0, 100, 100000)
    })
    
    import time
    
    # Boolean indexing
    start = time.time()
    _ = large_df[(large_df['A'] > 50) & (large_df['B'] < 30)]
    bool_time = time.time() - start
    
    # Query
    start = time.time()
    _ = large_df.query('A > 50 and B < 30')
    query_time = time.time() - start
    
    print(f"\nBoolean: {bool_time:.4f}s")
    print(f"Query: {query_time:.4f}s")
    
    # 10. Inplace Modification
    df_copy = df.copy()
    
    # Filter and reassign
    df_filtered = df_copy.query('age > 30')
    
    # Can't use inplace=True with query (returns filtered view)
    
    # 11. Backticks for Column Names with Spaces
    df_spaces = pd.DataFrame({
        'First Name': ['Alice', 'Bob'],
        'Last Name': ['Smith', 'Jones'],
        'Age': [25, 30]
    })
    
    result = df_spaces.query('`First Name` == "Alice"')
    print("\nWith spaces in column names:")
    print(result)
    
    # 12. Practical Example: Data Analysis
    sales_df = pd.DataFrame({
        'product': ['A', 'B', 'C', 'A', 'B', 'C'] * 10,
        'region': ['North', 'South'] * 30,
        'sales': np.random.randint(100, 1000, 60),
        'units': np.random.randint(1, 50, 60),
        'quarter': ['Q1', 'Q2', 'Q3', 'Q4'] * 15
    })
    
    # Complex business query
    target_sales = 500
    target_region = 'North'
    
    high_performers = sales_df.query(
        'sales > @target_sales and region == @target_region and units > 20'
    )
    
    print("\nHigh performers:")
    print(high_performers)
    
    # Chained queries
    result = (sales_df
              .query('region == "North"')
              .query('sales > 500')
              .query('quarter in ["Q1", "Q2"]'))
    
    print("\nChained queries:")
    print(result)
    ```

    **query() vs Boolean Indexing:**

    | Aspect | query() | Boolean Indexing |
    |--------|---------|------------------|
    | **Syntax** | `df.query('age > 30')` | `df[df['age'] > 30]` |
    | **Readability** | Cleaner, SQL-like | More verbose |
    | **Performance** | Faster (large data) | Slower |
    | **Variables** | Use `@` | Direct |
    | **Complex** | More readable | Harder to read |

    **Query Operators:**

    | Operator | Description | Example |
    |----------|-------------|---------|
    | **==, !=** | Equality | `'age == 30'` |
    | **>, <, >=, <=** | Comparison | `'salary > 50000'` |
    | **and, or, not** | Logical | `'age > 30 and salary < 60000'` |
    | **in** | Membership | `'dept in @departments'` |
    | **@** | Variable reference | `'age > @min_age'` |

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "query(): SQL-like string expressions"
        - "Cleaner than boolean indexing"
        - "Use @ for variables: `query('age > @min_age')`"
        - "and/or instead of &/|"
        - "Faster for large DataFrames (numexpr)"
        - "Backticks for spaces: `` `column name` ``"
        - "Can query index: `query('index == \"value\"')`"
        - "engine='python' for string methods"
        - "More readable for complex conditions"

---

### Explain Pandas Memory Usage Optimization - Netflix, Google Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Memory`, `Performance`, `Optimization` | **Asked by:** Netflix, Google, Meta, Amazon

??? success "View Answer"

    **Memory optimization** reduces DataFrame memory footprint through: downcasting numeric types, categorical dtype for strings, sparse arrays, chunking, and efficient file formats (parquet).

    **Complete Examples:**

    ```python
    import pandas as pd
    import numpy as np
    
    # 1. Check Memory Usage
    df = pd.DataFrame({
        'int_col': np.random.randint(0, 100, 1000000),
        'float_col': np.random.rand(1000000),
        'str_col': np.random.choice(['A', 'B', 'C'], 1000000),
        'date_col': pd.date_range('2020-01-01', periods=1000000, freq='min')
    })
    
    # Memory before
    memory_before = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory before: {memory_before:.2f} MB")
    print("\nMemory by column:")
    print(df.memory_usage(deep=True))
    
    # 2. Downcast Numeric Types
    # int64 → int8/int16/int32
    df['int_col'] = pd.to_numeric(df['int_col'], downcast='integer')
    
    # float64 → float32
    df['float_col'] = pd.to_numeric(df['float_col'], downcast='float')
    
    memory_after_numeric = df.memory_usage(deep=True).sum() / 1024**2
    print(f"\nAfter numeric downcast: {memory_after_numeric:.2f} MB")
    print(f"Saved: {memory_before - memory_after_numeric:.2f} MB")
    
    # 3. Categorical for Low Cardinality Strings
    df['str_col'] = df['str_col'].astype('category')
    
    memory_after_cat = df.memory_usage(deep=True).sum() / 1024**2
    print(f"After categorical: {memory_after_cat:.2f} MB")
    print(f"Total saved: {(1 - memory_after_cat/memory_before)*100:.1f}%")
    
    # 4. Automatic Optimization Function
    def optimize_dataframe(df):
        """Automatically optimize DataFrame dtypes."""
        optimized = df.copy()
        
        for col in optimized.columns:
            col_type = optimized[col].dtype
            
            # Numeric columns
            if col_type != 'object' and col_type.name != 'category':
                c_min = optimized[col].min()
                c_max = optimized[col].max()
                
                # Integer
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        optimized[col] = optimized[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        optimized[col] = optimized[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        optimized[col] = optimized[col].astype(np.int32)
                
                # Float
                else:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        optimized[col] = optimized[col].astype(np.float32)
            
            # String to categorical
            elif col_type == 'object':
                num_unique = optimized[col].nunique()
                num_total = len(optimized[col])
                if num_unique / num_total < 0.5:  # Less than 50% unique
                    optimized[col] = optimized[col].astype('category')
        
        return optimized
    
    # Test optimization
    df_test = pd.DataFrame({
        'id': range(100000),
        'category': np.random.choice(['A', 'B', 'C', 'D'], 100000),
        'value': np.random.rand(100000) * 100
    })
    
    print(f"\nBefore optimization: {df_test.memory_usage(deep=True).sum()/1024**2:.2f} MB")
    df_optimized = optimize_dataframe(df_test)
    print(f"After optimization: {df_optimized.memory_usage(deep=True).sum()/1024**2:.2f} MB")
    
    # 5. Sparse Arrays
    # For data with many zeros/NaN
    df_sparse = pd.DataFrame({
        'A': pd.arrays.SparseArray([0, 0, 1, 0, 0, 0, 2, 0, 0, 0] * 100000)
    })
    
    # Regular array
    df_dense = pd.DataFrame({
        'A': [0, 0, 1, 0, 0, 0, 2, 0, 0, 0] * 100000
    })
    
    print(f"\nDense: {df_dense.memory_usage(deep=True).sum()/1024**2:.2f} MB")
    print(f"Sparse: {df_sparse.memory_usage(deep=True).sum()/1024**2:.2f} MB")
    
    # 6. Chunking for Large Files
    chunk_size = 10000
    
    # Process in chunks
    def process_large_file(filename, chunksize=10000):
        """Process large CSV in chunks."""
        results = []
        for chunk in pd.read_csv(filename, chunksize=chunksize):
            # Process chunk
            processed = chunk[chunk['value'] > 50]
            results.append(processed)
        return pd.concat(results, ignore_index=True)
    
    # 7. Efficient File Formats
    # Save test data
    df_test.to_csv('test.csv', index=False)
    df_test.to_parquet('test.parquet')
    
    import os
    csv_size = os.path.getsize('test.csv') / 1024**2
    parquet_size = os.path.getsize('test.parquet') / 1024**2
    
    print(f"\nCSV size: {csv_size:.2f} MB")
    print(f"Parquet size: {parquet_size:.2f} MB")
    print(f"Parquet is {csv_size/parquet_size:.1f}x smaller")
    
    # 8. Column Selection
    # Read only needed columns
    # df_subset = pd.read_csv('large_file.csv', usecols=['col1', 'col2'])
    
    # 9. Date Parsing
    # Don't parse dates if not needed
    # df = pd.read_csv('file.csv', parse_dates=False)
    
    # 10. Practical Example: Large Dataset
    print("\n=== Optimization Summary ===")
    
    # Original
    df_original = pd.DataFrame({
        'user_id': range(1000000),
        'country': np.random.choice(['USA', 'UK', 'Canada', 'Germany'], 1000000),
        'age': np.random.randint(18, 80, 1000000),
        'score': np.random.rand(1000000) * 100,
        'status': np.random.choice(['active', 'inactive'], 1000000)
    })
    
    mem_original = df_original.memory_usage(deep=True).sum() / 1024**2
    print(f"Original: {mem_original:.2f} MB")
    
    # Optimized
    df_opt = df_original.copy()
    df_opt['user_id'] = pd.to_numeric(df_opt['user_id'], downcast='integer')
    df_opt['age'] = pd.to_numeric(df_opt['age'], downcast='integer')
    df_opt['score'] = pd.to_numeric(df_opt['score'], downcast='float')
    df_opt['country'] = df_opt['country'].astype('category')
    df_opt['status'] = df_opt['status'].astype('category')
    
    mem_opt = df_opt.memory_usage(deep=True).sum() / 1024**2
    print(f"Optimized: {mem_opt:.2f} MB")
    print(f"Reduction: {(1 - mem_opt/mem_original)*100:.1f}%")
    
    # Cleanup
    os.remove('test.csv')
    os.remove('test.parquet')
    ```

    **Optimization Techniques:**

    | Technique | Memory Savings | Use Case |
    |-----------|----------------|----------|
    | **Downcast int64→int8** | 87.5% | Small integers |
    | **Downcast float64→float32** | 50% | Precision not critical |
    | **String→Categorical** | 90%+ | Low cardinality |
    | **Sparse arrays** | 90%+ | Many zeros/NaN |
    | **Parquet format** | 50-90% | File storage |
    | **Chunking** | No limit | Process huge files |

    **Data Type Sizes:**

    | Type | Size | Range |
    |------|------|-------|
    | **int8** | 1 byte | -128 to 127 |
    | **int16** | 2 bytes | -32,768 to 32,767 |
    | **int32** | 4 bytes | -2B to 2B |
    | **int64** | 8 bytes | -9E18 to 9E18 |
    | **float32** | 4 bytes | ~7 decimals |
    | **float64** | 8 bytes | ~15 decimals |

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "memory_usage(deep=True): check actual usage"
        - "Downcast: int64→int8/16/32, float64→float32"
        - "Categorical: 90%+ savings for low-cardinality strings"
        - "Sparse arrays: for data with many zeros"
        - "Chunking: process files larger than RAM"
        - "Parquet: smaller, faster than CSV"
        - "read_csv usecols: load only needed columns"
        - "Automatic optimization: check ranges, convert types"
        - "Rule: <50% unique → categorical"

---

### Explain Pandas Eval Method - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `eval`, `Performance`, `Expression Evaluation` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **eval()** evaluates string expressions efficiently using numexpr backend. Much faster than standard operations for large DataFrames, especially with complex arithmetic.

    **Complete Examples:**

    ```python
    import pandas as pd
    import numpy as np
    import time
    
    # Create large DataFrame
    n = 1000000
    df = pd.DataFrame({
        'A': np.random.rand(n),
        'B': np.random.rand(n),
        'C': np.random.rand(n),
        'D': np.random.rand(n)
    })
    
    # 1. Basic eval()
    # Standard operation
    result_standard = df['A'] + df['B'] * df['C'] - df['D']
    
    # eval operation
    result_eval = df.eval('A + B * C - D')
    
    # Results are equal
    print("Results equal:", np.allclose(result_standard, result_eval))
    
    # 2. Performance Comparison
    # Standard
    start = time.time()
    _ = df['A'] + df['B'] * df['C'] - df['D']
    standard_time = time.time() - start
    
    # eval
    start = time.time()
    _ = df.eval('A + B * C - D')
    eval_time = time.time() - start
    
    print(f"\nStandard: {standard_time:.4f}s")
    print(f"eval: {eval_time:.4f}s")
    print(f"Speedup: {standard_time/eval_time:.1f}x")
    
    # 3. Column Assignment
    # Create new column
    df.eval('E = A + B', inplace=True)
    print("\nNew column E:")
    print(df[['A', 'B', 'E']].head())
    
    # Multiple assignments
    df.eval('''
        F = A * 2
        G = B + C
    ''', inplace=True)
    
    # 4. Using Variables with @
    multiplier = 2.5
    threshold = 0.5
    
    result = df.eval('A * @multiplier > @threshold')
    print("\nWith variables:")
    print(result.head())
    
    # 5. Comparison Operations
    # Boolean mask
    mask = df.eval('A > 0.5 and B < 0.3')
    filtered = df[mask]
    print("\nFiltered rows:", len(filtered))
    
    # Multiple conditions
    complex_mask = df.eval('(A > 0.5 and B < 0.3) or (C > 0.8 and D < 0.2)')
    
    # 6. Local vs Global Variables
    local_var = 10
    
    # Access local variable
    df.eval('H = A * @local_var', inplace=True)
    
    # 7. DataFrame.eval() vs pd.eval()
    # DataFrame.eval
    df_result = df.eval('A + B')
    
    # pd.eval with multiple DataFrames
    df2 = pd.DataFrame({
        'X': np.random.rand(n),
        'Y': np.random.rand(n)
    })
    
    # Doesn't work directly with multiple DataFrames in df.eval
    # Use standard operations or merge first
    
    # 8. Assignment Operators
    # Add to existing column
    df.eval('A = A + 1', inplace=True)
    
    # Conditional assignment (use where instead)
    df['I'] = df.eval('A if A > 0.5 else B')
    
    # 9. Complex Expressions
    # Multi-line expressions
    df.eval('''
        temp1 = A + B
        temp2 = C * D
        result = temp1 / temp2
    ''', inplace=True)
    
    # 10. When eval() is Faster
    # Good: Complex arithmetic
    start = time.time()
    _ = df.eval('A + B * C / D - (A * B) + (C / D)')
    eval_fast = time.time() - start
    
    start = time.time()
    _ = df['A'] + df['B'] * df['C'] / df['D'] - (df['A'] * df['B']) + (df['C'] / df['D'])
    standard_slow = time.time() - start
    
    print(f"\nComplex arithmetic:")
    print(f"eval: {eval_fast:.4f}s")
    print(f"Standard: {standard_slow:.4f}s")
    print(f"Speedup: {standard_slow/eval_fast:.1f}x")
    
    # Bad: Simple operations, small data
    small_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    
    # Not faster for small data
    _ = small_df.eval('A + B')
    _ = small_df['A'] + small_df['B']  # Similar speed
    
    # 11. Supported Operations
    print("\n=== Supported Operations ===")
    df_ops = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    
    # Arithmetic
    print("Arithmetic:", df_ops.eval('A + B * 2').tolist())
    
    # Comparison
    print("Comparison:", df_ops.eval('A > 1').tolist())
    
    # Logical
    print("Logical:", df_ops.eval('A > 1 and B < 6').tolist())
    
    # 12. Limitations
    # Can't use:
    # - String methods: df.eval('A.str.upper()')  # Error
    # - Complex functions: df.eval('A.apply(func)')  # Error
    # - Method chaining
    
    # Use standard operations instead for these
    
    # 13. Practical Example
    # Financial calculations
    portfolio = pd.DataFrame({
        'shares': np.random.randint(10, 1000, 10000),
        'price': np.random.rand(10000) * 100,
        'cost_basis': np.random.rand(10000) * 80,
        'dividend': np.random.rand(10000) * 5
    })
    
    # Calculate portfolio metrics
    portfolio.eval('''
        market_value = shares * price
        cost_value = shares * cost_basis
        unrealized_gain = market_value - cost_value
        gain_pct = (unrealized_gain / cost_value) * 100
        annual_dividend = shares * dividend
        yield_pct = (annual_dividend / market_value) * 100
    ''', inplace=True)
    
    print("\nPortfolio metrics:")
    print(portfolio[['market_value', 'unrealized_gain', 'gain_pct', 'yield_pct']].head())
    ```

    **eval() vs Standard:**

    | Aspect | eval() | Standard |
    |--------|--------|----------|
    | **Syntax** | String expression | Python operations |
    | **Speed** | Faster (large data) | Slower |
    | **Memory** | More efficient | More allocations |
    | **Readability** | SQL-like, cleaner | More verbose |
    | **Best for** | Complex arithmetic | Simple operations |

    **When to Use eval():**

    | Use eval() | Don't Use eval() |
    |------------|------------------|
    | Large DataFrames (>100k rows) | Small DataFrames |
    | Complex arithmetic | Simple operations |
    | Multiple operations | Single operation |
    | Chained calculations | String methods |
    | Memory constrained | Custom functions |

    **Supported Operations:**

    | Operation | Example |
    |-----------|---------|
    | **Arithmetic** | `+`, `-`, `*`, `/`, `**`, `%` |
    | **Comparison** | `>`, `<`, `>=`, `<=`, `==`, `!=` |
    | **Logical** | `and`, `or`, `not`, `&`, `|`, `~` |
    | **Variables** | `@variable_name` |
    | **Assignment** | `new_col = expression` |

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "eval(): string expressions using numexpr"
        - "Faster for large DataFrames (100k+ rows)"
        - "Best for complex arithmetic: 2-4x speedup"
        - "More memory efficient"
        - "Use @ for variables: `eval('A * @multiplier')`"
        - "inplace=True: modify DataFrame"
        - "Limitations: no string methods, custom functions"
        - "Not faster for small data or simple ops"
        - "Cleaner syntax for complex calculations"

---

### Explain Pandas Time Series Resampling - Meta, Netflix Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Time Series`, `Resampling`, `Aggregation` | **Asked by:** Meta, Netflix, Google, Amazon

??? success "View Answer"

    **Resample()** converts time series to different frequencies (upsampling/downsampling). Essential for time series aggregation, business reporting, and data alignment.

    **Complete Examples:**

    ```python
    import pandas as pd
    import numpy as np
    
    # Create time series data
    dates = pd.date_range('2024-01-01', periods=365, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'sales': np.random.randint(100, 500, 365) + np.sin(np.arange(365) * 0.1) * 50,
        'customers': np.random.randint(10, 100, 365),
        'temperature': 70 + np.sin(np.arange(365) * 0.017) * 20 + np.random.randn(365) * 5
    }).set_index('date')
    
    # 1. Downsampling (Higher → Lower Frequency)
    # Daily → Weekly
    weekly = df.resample('W').sum()
    print("Weekly totals:")
    print(weekly.head())
    
    # Daily → Monthly
    monthly = df.resample('M').sum()
    print("\nMonthly totals:")
    print(monthly.head())
    
    # Daily → Quarterly
    quarterly = df.resample('Q').sum()
    print("\nQuarterly totals:")
    print(quarterly)
    
    # 2. Different Aggregations
    # Multiple aggregations
    monthly_stats = df.resample('M').agg({
        'sales': ['sum', 'mean', 'max'],
        'customers': 'sum',
        'temperature': 'mean'
    })
    print("\nMonthly statistics:")
    print(monthly_stats)
    
    # 3. Upsampling (Lower → Higher Frequency)
    # Create sparse data
    weekly_data = pd.DataFrame({
        'value': [100, 200, 150, 180]
    }, index=pd.date_range('2024-01-01', periods=4, freq='W'))
    
    # Fill forward
    daily_ffill = weekly_data.resample('D').ffill()
    print("\nUpsampled (forward fill):")
    print(daily_ffill.head(10))
    
    # Interpolate
    daily_interp = weekly_data.resample('D').interpolate()
    print("\nUpsampled (interpolate):")
    print(daily_interp.head(10))
    
    # 4. Common Frequency Codes
    freq_examples = {
        'D': 'Daily',
        'W': 'Weekly',
        'M': 'Month end',
        'MS': 'Month start',
        'Q': 'Quarter end',
        'QS': 'Quarter start',
        'Y': 'Year end',
        'H': 'Hourly',
        'T' or 'min': 'Minute',
        'S': 'Second'
    }
    
    # 5. Business Day Resampling
    # Business days only (Mon-Fri)
    business_weekly = df.resample('W-MON').sum()  # Week ending Monday
    print("\nWeekly (business days):")
    print(business_weekly.head())
    
    # Business month
    business_monthly = df.resample('BM').sum()  # Business month end
    
    # 6. Offset Resampling
    # 2-week periods
    biweekly = df.resample('2W').sum()
    print("\nBi-weekly totals:")
    print(biweekly.head())
    
    # 10-day periods
    ten_day = df.resample('10D').sum()
    
    # 7. Rolling with Resample
    # Monthly average with rolling
    monthly_rolling = df.resample('M').mean().rolling(window=3).mean()
    print("\n3-month moving average:")
    print(monthly_rolling.head(6))
    
    # 8. Groupby with Resample
    # Add category
    df['region'] = np.random.choice(['North', 'South'], len(df))
    
    # Resample per group
    regional_monthly = df.groupby('region').resample('M').sum()
    print("\nRegional monthly sales:")
    print(regional_monthly.head())
    
    # 9. Custom Aggregation
    def custom_agg(x):
        """Custom aggregation function."""
        return pd.Series({
            'total': x.sum(),
            'average': x.mean(),
            'peak': x.max(),
            'days': len(x)
        })
    
    custom_monthly = df['sales'].resample('M').apply(custom_agg)
    print("\nCustom aggregation:")
    print(custom_monthly.head())
    
    # 10. Label and Closed Parameters
    # Default: label='right', closed='right'
    # Period labeled by right edge, closed on right
    
    # Label by left edge
    left_label = df.resample('W', label='left').sum()
    
    # Closed on left
    left_closed = df.resample('W', closed='left').sum()
    
    # 11. Origin and Offset
    # Start resampling from specific point
    custom_origin = df.resample('10D', origin='2024-01-05').sum()
    
    # 12. Practical Example: Business Reporting
    sales_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='H'),
        'revenue': np.random.randint(100, 1000, 1000),
        'orders': np.random.randint(1, 20, 1000)
    }).set_index('timestamp')
    
    # Daily summary
    daily_report = sales_data.resample('D').agg({
        'revenue': ['sum', 'mean', 'max'],
        'orders': ['sum', 'count']
    })
    
    # Add calculated metrics
    daily_report['avg_order_value'] = (
        daily_report[('revenue', 'sum')] / daily_report[('orders', 'sum')]
    )
    
    print("\nDaily business report:")
    print(daily_report.head())
    
    # Weekly summary with week-over-week growth
    weekly_report = sales_data.resample('W').sum()
    weekly_report['revenue_growth'] = weekly_report['revenue'].pct_change() * 100
    weekly_report['orders_growth'] = weekly_report['orders'].pct_change() * 100
    
    print("\nWeekly growth report:")
    print(weekly_report.head())
    
    # 13. Combining with Transform
    # Add group statistics back to original data
    df['monthly_avg'] = df.groupby(df.index.to_period('M'))['sales'].transform('mean')
    
    print("\nWith monthly average:")
    print(df[['sales', 'monthly_avg']].head(35))
    ```

    **Frequency Codes:**

    | Code | Meaning | Example |
    |------|---------|---------|
    | **D** | Day | Daily data |
    | **W** | Week | Weekly summary |
    | **M** | Month end | Monthly totals |
    | **MS** | Month start | Start of month |
    | **Q** | Quarter end | Quarterly reports |
    | **Y** | Year end | Annual data |
    | **H** | Hour | Hourly aggregation |
    | **T/min** | Minute | Minute-level data |
    | **S** | Second | Second-level data |
    | **B** | Business day | Weekdays only |

    **Resampling Methods:**

    | Method | Description | Use Case |
    |--------|-------------|----------|
    | **sum()** | Total | Revenue, quantities |
    | **mean()** | Average | Prices, temperatures |
    | **first()** | First value | Opening price |
    | **last()** | Last value | Closing price |
    | **max()/min()** | Peak/Low | Daily highs/lows |
    | **count()** | Count | Transactions |
    | **ffill()** | Forward fill | Upsampling |
    | **interpolate()** | Interpolation | Smooth upsampling |

    **Upsampling vs Downsampling:**

    | Type | Direction | Aggregation | Example |
    |------|-----------|-------------|---------|
    | **Downsampling** | High → Low freq | Required | Daily → Monthly |
    | **Upsampling** | Low → High freq | Fill/interpolate | Monthly → Daily |

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "resample(): convert time series frequency"
        - "Downsampling: higher → lower (daily → monthly)"
        - "Upsampling: lower → higher (monthly → daily)"
        - "Frequency codes: D, W, M, Q, Y, H"
        - "Aggregation: sum, mean, max for downsampling"
        - "Fill methods: ffill, bfill, interpolate for upsampling"
        - "label: 'left' or 'right' edge labeling"
        - "Business days: 'B', 'BM' for business calendars"
        - "Combine with groupby for category-wise resampling"

---

### Explain Pandas Pivot Tables - Microsoft, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Pivot Tables`, `Aggregation`, `Reshaping` | **Asked by:** Microsoft, Amazon, Google, Meta

??? success "View Answer"

    **pivot_table()** summarizes data with row/column groupings and aggregations (like Excel). **pivot()** reshapes without aggregation. Use pivot_table for reports, pivot for simple reshaping.

    **Complete Examples:**

    ```python
    import pandas as pd
    import numpy as np
    
    # Sample sales data
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100, freq='D').repeat(3),
        'region': np.tile(['North', 'South', 'East'], 100),
        'product': np.tile(['A', 'B', 'C'], 100),
        'sales': np.random.randint(100, 1000, 300),
        'quantity': np.random.randint(1, 50, 300),
        'profit': np.random.randint(10, 200, 300)
    })
    
    # 1. Basic pivot_table
    # Sales by region and product
    pivot = pd.pivot_table(
        df,
        values='sales',
        index='region',
        columns='product',
        aggfunc='sum'
    )
    print("Sales by region and product:")
    print(pivot)
    
    # 2. Multiple Aggregations
    # Multiple functions
    multi_agg = pd.pivot_table(
        df,
        values='sales',
        index='region',
        columns='product',
        aggfunc=['sum', 'mean', 'count']
    )
    print("\nMultiple aggregations:")
    print(multi_agg)
    
    # 3. Multiple Values
    # Different columns
    multi_values = pd.pivot_table(
        df,
        values=['sales', 'profit'],
        index='region',
        columns='product',
        aggfunc='sum'
    )
    print("\nMultiple values:")
    print(multi_values)
    
    # 4. Margins (Totals)
    # Add row and column totals
    with_totals = pd.pivot_table(
        df,
        values='sales',
        index='region',
        columns='product',
        aggfunc='sum',
        margins=True,
        margins_name='Total'
    )
    print("\nWith totals:")
    print(with_totals)
    
    # 5. Fill Missing Values
    # Replace NaN
    filled = pd.pivot_table(
        df,
        values='sales',
        index='region',
        columns='product',
        aggfunc='sum',
        fill_value=0
    )
    print("\nFilled with zeros:")
    print(filled)
    
    # 6. Multiple Index Levels
    # Hierarchical index
    df['month'] = df['date'].dt.to_period('M')
    
    multi_index = pd.pivot_table(
        df,
        values='sales',
        index=['month', 'region'],
        columns='product',
        aggfunc='sum'
    )
    print("\nMulti-index:")
    print(multi_index.head(10))
    
    # 7. Custom Aggregation
    def range_func(x):
        return x.max() - x.min()
    
    custom_agg = pd.pivot_table(
        df,
        values='sales',
        index='region',
        columns='product',
        aggfunc=range_func
    )
    print("\nCustom aggregation (range):")
    print(custom_agg)
    
    # 8. pivot() vs pivot_table()
    # Simple data (unique index/column combinations)
    simple_df = pd.DataFrame({
        'row': ['A', 'A', 'B', 'B'],
        'col': ['X', 'Y', 'X', 'Y'],
        'value': [1, 2, 3, 4]
    })
    
    # pivot (no aggregation, must be unique)
    pivoted = simple_df.pivot(index='row', columns='col', values='value')
    print("\npivot() result:")
    print(pivoted)
    
    # pivot_table (with aggregation, handles duplicates)
    dup_df = pd.DataFrame({
        'row': ['A', 'A', 'A', 'B'],
        'col': ['X', 'X', 'Y', 'X'],
        'value': [1, 2, 3, 4]
    })
    
    # pivot would fail (duplicates)
    # pivoted = dup_df.pivot(index='row', columns='col', values='value')  # Error!
    
    # pivot_table handles it
    pivoted_agg = dup_df.pivot_table(
        index='row',
        columns='col',
        values='value',
        aggfunc='sum'
    )
    print("\npivot_table() with duplicates:")
    print(pivoted_agg)
    
    # 9. Observed Parameter
    # Add categorical column
    df['category'] = pd.Categorical(
        np.random.choice(['Cat1', 'Cat2', 'Cat3'], len(df)),
        categories=['Cat1', 'Cat2', 'Cat3', 'Cat4']  # Cat4 unused
    )
    
    # Include unused categories
    with_unused = pd.pivot_table(
        df,
        values='sales',
        index='category',
        aggfunc='sum',
        observed=False
    )
    print("\nWith unused categories:")
    print(with_unused)
    
    # Exclude unused
    observed_only = pd.pivot_table(
        df,
        values='sales',
        index='category',
        aggfunc='sum',
        observed=True
    )
    print("\nObserved only:")
    print(observed_only)
    
    # 10. Practical Example: Business Report
    sales_data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=365, freq='D'),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 365),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Food'], 365),
        'revenue': np.random.randint(1000, 10000, 365),
        'units_sold': np.random.randint(10, 200, 365)
    })
    
    # Add month and quarter
    sales_data['month'] = sales_data['date'].dt.to_period('M')
    sales_data['quarter'] = sales_data['date'].dt.to_period('Q')
    
    # Quarterly report
    quarterly_report = pd.pivot_table(
        sales_data,
        values=['revenue', 'units_sold'],
        index='quarter',
        columns='region',
        aggfunc={
            'revenue': 'sum',
            'units_sold': 'sum'
        },
        margins=True,
        fill_value=0
    )
    
    print("\nQuarterly business report:")
    print(quarterly_report)
    
    # Monthly category performance
    category_report = pd.pivot_table(
        sales_data,
        values='revenue',
        index='month',
        columns='product_category',
        aggfunc=['sum', 'mean', 'count'],
        margins=True
    )
    
    print("\nMonthly category report:")
    print(category_report.head(6))
    ```

    **pivot() vs pivot_table():**

    | Aspect | pivot() | pivot_table() |
    |--------|---------|---------------|
    | **Aggregation** | No | Yes |
    | **Duplicates** | Must be unique | Handles duplicates |
    | **Use Case** | Simple reshape | Summarization |
    | **Example** | Wide to long | Excel pivot table |
    | **Parameters** | index, columns, values | + aggfunc, margins |

    **Common Parameters:**

    | Parameter | Description | Example |
    |-----------|-------------|---------|
    | **values** | Column(s) to aggregate | `values='sales'` |
    | **index** | Row grouping | `index='region'` |
    | **columns** | Column grouping | `columns='product'` |
    | **aggfunc** | Aggregation function | `aggfunc='sum'` |
    | **fill_value** | Replace NaN | `fill_value=0` |
    | **margins** | Add totals | `margins=True` |

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "pivot_table(): Excel-like summarization"
        - "pivot(): reshape without aggregation"
        - "index: row grouping, columns: column grouping"
        - "aggfunc: 'sum', 'mean', 'count', or custom"
        - "margins=True: add row/column totals"
        - "fill_value=0: replace NaN"
        - "pivot_table handles duplicates, pivot doesn't"
        - "Multiple values: list in values parameter"
        - "Use for: reports, summaries, cross-tabulation"

---

### Explain pd.melt() - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `melt`, `Reshaping`, `Wide to Long` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **melt()** transforms wide format to long format (unpivots). Opposite of pivot. Essential for tidy data, plotting, and analysis requiring long format.

    **Complete Examples:**

    ```python
    import pandas as pd
    
    # Wide format data
    df_wide = pd.DataFrame({
        'student': ['Alice', 'Bob', 'Charlie'],
        'math': [85, 92, 78],
        'science': [88, 95, 82],
        'english': [90, 87, 85]
    })
    
    print("Wide format:")
    print(df_wide)
    
    # 1. Basic melt
    df_long = pd.melt(df_wide, id_vars=['student'])
    print("\nLong format (basic):")
    print(df_long)
    
    # 2. Custom Column Names
    df_long = pd.melt(
        df_wide,
        id_vars=['student'],
        var_name='subject',
        value_name='score'
    )
    print("\nWith custom names:")
    print(df_long)
    
    # 3. Select Specific Columns to Melt
    df_long = pd.melt(
        df_wide,
        id_vars=['student'],
        value_vars=['math', 'science'],  # Only these columns
        var_name='subject',
        value_name='score'
    )
    print("\nSelected columns:")
    print(df_long)
    
    # 4. Multiple ID Variables
    df_multi = pd.DataFrame({
        'student': ['Alice', 'Bob', 'Charlie'],
        'class': ['A', 'A', 'B'],
        'math': [85, 92, 78],
        'science': [88, 95, 82]
    })
    
    df_long = pd.melt(
        df_multi,
        id_vars=['student', 'class'],
        var_name='subject',
        value_name='score'
    )
    print("\nMultiple ID vars:")
    print(df_long)
    
    # 5. melt() + pivot() (Round Trip)
    # Wide → Long
    df_long = pd.melt(df_wide, id_vars=['student'])
    
    # Long → Wide
    df_back = df_long.pivot(
        index='student',
        columns='variable',
        values='value'
    ).reset_index()
    
    print("\nRound trip (original):")
    print(df_wide)
    print("\nRound trip (restored):")
    print(df_back)
    
    # 6. Real Example: Time Series
    sales_wide = pd.DataFrame({
        'product': ['Widget', 'Gadget', 'Tool'],
        '2023-Q1': [100, 150, 80],
        '2023-Q2': [120, 160, 90],
        '2023-Q3': [110, 170, 85],
        '2023-Q4': [130, 180, 95]
    })
    
    sales_long = pd.melt(
        sales_wide,
        id_vars=['product'],
        var_name='quarter',
        value_name='sales'
    )
    
    print("\nTime series (long format):")
    print(sales_long)
    
    # Convert quarter to datetime
    sales_long['quarter'] = pd.PeriodIndex(sales_long['quarter'], freq='Q').to_timestamp()
    
    # 7. Handling Missing Values
    df_missing = pd.DataFrame({
        'id': [1, 2, 3],
        'A': [10, None, 30],
        'B': [40, 50, None]
    })
    
    # Default: keeps NaN
    df_long_nan = pd.melt(df_missing, id_vars=['id'])
    print("\nWith NaN:")
    print(df_long_nan)
    
    # Drop NaN
    df_long_clean = pd.melt(df_missing, id_vars=['id']).dropna()
    print("\nNaN dropped:")
    print(df_long_clean)
    
    # 8. Practical Example: Survey Data
    survey = pd.DataFrame({
        'respondent_id': [1, 2, 3],
        'age': [25, 30, 35],
        'q1_satisfaction': [4, 5, 3],
        'q2_likelihood': [5, 4, 4],
        'q3_quality': [4, 5, 5]
    })
    
    # Melt only question columns
    survey_long = pd.melt(
        survey,
        id_vars=['respondent_id', 'age'],
        value_vars=['q1_satisfaction', 'q2_likelihood', 'q3_quality'],
        var_name='question',
        value_name='response'
    )
    
    # Clean question names
    survey_long['question'] = survey_long['question'].str.replace('q\\d+_', '', regex=True)
    
    print("\nSurvey data (long):")
    print(survey_long)
    
    # 9. Performance with Large Data
    # melt is efficient for large datasets
    import numpy as np
    
    large_df = pd.DataFrame({
        'id': range(10000),
        **{f'col_{i}': np.random.rand(10000) for i in range(50)}
    })
    
    import time
    start = time.time()
    large_melted = pd.melt(large_df, id_vars=['id'])
    melt_time = time.time() - start
    
    print(f"\nMelted 10k rows x 50 cols in {melt_time:.4f}s")
    print(f"Result shape: {large_melted.shape}")
    
    # 10. Use Cases Comparison
    print("\n=== When to Use ===")
    print("Wide format:")
    print("- Human-readable reports")
    print("- Pivot tables")
    print("- Matrix operations")
    
    print("\nLong format (melt):")
    print("- Analysis in Pandas/SQL")
    print("- Plotting with seaborn/plotly")
    print("- GroupBy operations")
    print("- Statistical modeling")
    ```

    **Wide vs Long Format:**

    | Format | Structure | Use Case | Example |
    |--------|-----------|----------|---------|
    | **Wide** | Many columns | Reports, humans | Math: 85, Science: 88 |
    | **Long** | Few columns, many rows | Analysis, computers | Subject: Math, Score: 85 |

    **melt() Parameters:**

    | Parameter | Description | Example |
    |-----------|-------------|---------|
    | **id_vars** | Columns to keep | `id_vars=['student']` |
    | **value_vars** | Columns to melt | `value_vars=['math', 'science']` |
    | **var_name** | Name for variable column | `var_name='subject'` |
    | **value_name** | Name for value column | `value_name='score'` |

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "melt(): wide → long format (unpivot)"
        - "id_vars: columns to keep fixed"
        - "value_vars: columns to melt (optional, default all)"
        - "var_name: name for melted column names"
        - "value_name: name for values"
        - "Opposite of pivot()/pivot_table()"
        - "Use for: plotting, analysis, tidy data"
        - "Long format better for: GroupBy, seaborn, SQL"
        - "Round trip: melt() + pivot() = original"

---

### Explain pd.crosstab() - Amazon, Microsoft Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `crosstab`, `Frequency Tables`, `Statistics` | **Asked by:** Amazon, Microsoft, Google, Meta

??? success "View Answer"

    **crosstab()** computes cross-tabulation of two or more factors. Shows frequency distributions. Similar to pivot_table but specialized for counts and proportions.

    **Complete Examples:**

    ```python
    import pandas as pd
    import numpy as np
    
    # Sample data
    df = pd.DataFrame({
        'gender': np.random.choice(['M', 'F'], 200),
        'education': np.random.choice(['HS', 'Bachelor', 'Master', 'PhD'], 200),
        'employed': np.random.choice(['Yes', 'No'], 200),
        'age_group': np.random.choice(['18-25', '26-35', '36-50', '50+'], 200)
    })
    
    # 1. Basic Crosstab (Counts)
    ct = pd.crosstab(df['gender'], df['education'])
    print("Basic crosstab:")
    print(ct)
    
    # 2. With Margins (Totals)
    ct_margins = pd.crosstab(
        df['gender'],
        df['education'],
        margins=True,
        margins_name='Total'
    )
    print("\nWith totals:")
    print(ct_margins)
    
    # 3. Normalize (Percentages)
    # Normalize all (percentages of total)
    ct_norm_all = pd.crosstab(
        df['gender'],
        df['education'],
        normalize='all'
    )
    print("\nNormalized (all):")
    print(ct_norm_all)
    
    # Normalize by rows
    ct_norm_index = pd.crosstab(
        df['gender'],
        df['education'],
        normalize='index'
    )
    print("\nNormalized (by row):")
    print(ct_norm_index)
    
    # Normalize by columns
    ct_norm_columns = pd.crosstab(
        df['gender'],
        df['education'],
        normalize='columns'
    )
    print("\nNormalized (by column):")
    print(ct_norm_columns)
    
    # 4. Multiple Row/Column Variables
    ct_multi = pd.crosstab(
        [df['gender'], df['age_group']],
        [df['education'], df['employed']]
    )
    print("\nMulti-level crosstab:")
    print(ct_multi)
    
    # 5. With Values (Like pivot_table)
    df['income'] = np.random.randint(30000, 120000, len(df))
    
    ct_values = pd.crosstab(
        df['gender'],
        df['education'],
        values=df['income'],
        aggfunc='mean'
    )
    print("\nWith values (mean income):")
    print(ct_values)
    
    # 6. Multiple Aggregations
    ct_multi_agg = pd.crosstab(
        df['gender'],
        df['education'],
        values=df['income'],
        aggfunc=['mean', 'median', 'count']
    )
    print("\nMultiple aggregations:")
    print(ct_multi_agg)
    
    # 7. Dropna Parameter
    df_with_nan = df.copy()
    df_with_nan.loc[::10, 'education'] = None
    
    # Include NaN
    ct_with_nan = pd.crosstab(
        df_with_nan['gender'],
        df_with_nan['education'],
        dropna=False
    )
    print("\nWith NaN:")
    print(ct_with_nan)
    
    # Exclude NaN
    ct_no_nan = pd.crosstab(
        df_with_nan['gender'],
        df_with_nan['education'],
        dropna=True
    )
    print("\nExcluding NaN:")
    print(ct_no_nan)
    
    # 8. Practical Example: A/B Test Analysis
    ab_test = pd.DataFrame({
        'variant': np.random.choice(['A', 'B'], 1000),
        'converted': np.random.choice(['Yes', 'No'], 1000, p=[0.15, 0.85]),
        'device': np.random.choice(['Mobile', 'Desktop'], 1000)
    })
    
    # Conversion rates
    conversion_ct = pd.crosstab(
        ab_test['variant'],
        ab_test['converted'],
        normalize='index'
    )
    print("\nA/B test conversion rates:")
    print(conversion_ct)
    
    # By device
    device_ct = pd.crosstab(
        [ab_test['variant'], ab_test['device']],
        ab_test['converted'],
        normalize='index',
        margins=True
    )
    print("\nBy device:")
    print(device_ct)
    
    # 9. Chi-Square Test
    # Statistical independence test
    from scipy.stats import chi2_contingency
    
    ct_test = pd.crosstab(df['gender'], df['employed'])
    chi2, p_value, dof, expected = chi2_contingency(ct_test)
    
    print(f"\nChi-square test:")
    print(f"Chi2: {chi2:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Independent: {p_value > 0.05}")
    
    # 10. Comparison with pivot_table
    # crosstab
    ct = pd.crosstab(df['gender'], df['education'])
    
    # Equivalent pivot_table
    pt = df.pivot_table(
        index='gender',
        columns='education',
        aggfunc='size',
        fill_value=0
    )
    
    print("\ncrosstab == pivot_table:", ct.equals(pt))
    ```

    **crosstab() vs pivot_table():**

    | Aspect | crosstab() | pivot_table() |
    |--------|------------|---------------|
    | **Purpose** | Frequency counts | General aggregation |
    | **Default aggfunc** | Count | Mean |
    | **Normalize** | ✅ Built-in | ❌ Manual |
    | **Input** | Arrays or Series | DataFrame columns |
    | **Use Case** | Categorical analysis | Numeric summarization |

    **normalize Parameter:**

    | Value | Meaning | Use Case |
    |-------|---------|----------|
    | **'all'** | % of total | Overall distribution |
    | **'index'** | % of row total | Row-wise comparison |
    | **'columns'** | % of column total | Column-wise comparison |

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "crosstab(): frequency table of categorical variables"
        - "Default: counts, can aggregate with values/aggfunc"
        - "normalize: 'all', 'index', 'columns' for percentages"
        - "margins=True: add row/column totals"
        - "Useful for: categorical analysis, A/B testing, chi-square"
        - "vs pivot_table: specialized for counts"
        - "Multiple variables: pass lists to index/columns"
        - "Chi-square test: test independence of variables"

---

### Explain pd.get_dummies() for One-Hot Encoding - Google, Netflix Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `One-Hot Encoding`, `Feature Engineering`, `ML Preprocessing` | **Asked by:** Google, Netflix, Meta, Amazon

??? success "View Answer"

    **get_dummies()** converts categorical variables to binary indicator columns (one-hot encoding). Essential for machine learning with categorical features.

    **Complete Examples:**

    ```python
    import pandas as pd
    import numpy as np
    
    # Sample data
    df = pd.DataFrame({
        'color': ['red', 'blue', 'green', 'red', 'blue'],
        'size': ['S', 'M', 'L', 'M', 'S'],
        'price': [10, 20, 15, 12, 18]
    })
    
    print("Original data:")
    print(df)
    
    # 1. Basic One-Hot Encoding
    dummies = pd.get_dummies(df['color'])
    print("\nOne-hot encoded:")
    print(dummies)
    
    # 2. Add Prefix
    dummies = pd.get_dummies(df['color'], prefix='color')
    print("\nWith prefix:")
    print(dummies)
    
    # 3. Encode Entire DataFrame
    df_encoded = pd.get_dummies(df, columns=['color', 'size'])
    print("\nFull DataFrame encoded:")
    print(df_encoded)
    
    # 4. Drop First (Avoid Multicollinearity)
    # Drop one category to avoid dummy variable trap
    dummies_drop = pd.get_dummies(
        df,
        columns=['color', 'size'],
        drop_first=True
    )
    print("\nDrop first category:")
    print(dummies_drop)
    
    # 5. Custom Prefix for Multiple Columns
    dummies_custom = pd.get_dummies(
        df,
        columns=['color', 'size'],
        prefix=['col', 'sz']
    )
    print("\nCustom prefixes:")
    print(dummies_custom)
    
    # 6. Handle Unknown Categories
    # Training data
    train = pd.DataFrame({
        'category': ['A', 'B', 'C', 'A', 'B']
    })
    
    train_encoded = pd.get_dummies(train, prefix='cat')
    print("\nTraining encoded:")
    print(train_encoded)
    
    # Test data with new category 'D'
    test = pd.DataFrame({
        'category': ['A', 'B', 'D']
    })
    
    test_encoded = pd.get_dummies(test, prefix='cat')
    print("\nTest encoded (has 'D'):")
    print(test_encoded)
    
    # Align columns
    test_aligned = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)
    print("\nTest aligned with training:")
    print(test_aligned)
    
    # 7. Dummy NA
    # Include NaN as category
    df_na = pd.DataFrame({
        'color': ['red', 'blue', None, 'red', 'blue']
    })
    
    dummies_na = pd.get_dummies(df_na, dummy_na=True)
    print("\nWith NaN as category:")
    print(dummies_na)
    
    # 8. Sparse Matrix (Memory Efficient)
    # For high cardinality
    df_large = pd.DataFrame({
        'category': np.random.choice(list('ABCDEFGHIJ'), 10000)
    })
    
    # Dense (default)
    dense = pd.get_dummies(df_large)
    print(f"\nDense memory: {dense.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Sparse
    sparse = pd.get_dummies(df_large, sparse=True)
    print(f"Sparse memory: {sparse.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # 9. Practical Example: ML Preprocessing
    # Customer data
    customers = pd.DataFrame({
        'customer_id': [1, 2, 3, 4, 5],
        'country': ['USA', 'UK', 'Canada', 'USA', 'UK'],
        'membership': ['Gold', 'Silver', 'Bronze', 'Gold', 'Silver'],
        'age': [25, 35, 45, 30, 40],
        'purchases': [10, 5, 8, 12, 6]
    })
    
    # Encode categorical, keep numeric
    customers_encoded = pd.get_dummies(
        customers,
        columns=['country', 'membership'],
        drop_first=True,  # Avoid multicollinearity
        prefix=['country', 'tier']
    )
    
    print("\nML-ready data:")
    print(customers_encoded)
    
    # 10. Comparison with Categorical
    # get_dummies: expands to multiple columns
    encoded = pd.get_dummies(df['color'])
    print("\nget_dummies (wide):")
    print(encoded)
    
    # Categorical: single column with integer codes
    categorical = df['color'].astype('category')
    print("\nCategorical (compact):")
    print(categorical)
    print("Codes:", categorical.cat.codes.tolist())
    ```

    **get_dummies() Parameters:**

    | Parameter | Description | Example |
    |-----------|-------------|---------|
    | **data** | DataFrame or Series | `pd.get_dummies(df)` |
    | **columns** | Columns to encode | `columns=['color']` |
    | **prefix** | Column name prefix | `prefix='cat'` |
    | **drop_first** | Avoid multicollinearity | `drop_first=True` |
    | **dummy_na** | Include NaN as category | `dummy_na=True` |
    | **sparse** | Use sparse matrix | `sparse=True` |

    **When to drop_first:**

    | Use Case | drop_first | Reason |
    |----------|------------|--------|
    | **Linear models** | ✅ True | Avoid multicollinearity |
    | **Tree models** | ❌ False | Can handle all dummies |
    | **Neural networks** | ❌ False | Usually fine |
    | **Interpretation** | ❌ False | Easier to understand |

    **get_dummies() vs Categorical:**

    | Aspect | get_dummies() | Categorical |
    |--------|---------------|-------------|
    | **Format** | Multiple columns | Single column |
    | **Storage** | More space | Less space |
    | **ML** | Ready to use | Need encoding |
    | **Interpretability** | Clear | Needs mapping |

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "get_dummies(): one-hot encoding (categorical → binary)"
        - "Creates column per category with 0/1"
        - "prefix: add meaningful column names"
        - "drop_first=True: avoid dummy variable trap (linear models)"
        - "dummy_na=True: treat NaN as separate category"
        - "sparse=True: save memory for high cardinality"
        - "Align train/test: use reindex() for new categories"
        - "vs LabelEncoder: ordinal encoding (single column)"
        - "Use for: ML models that need numeric input"

---

### Explain SettingWithCopyWarning - Meta, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Warning`, `DataFrame Copies`, `Best Practices` | **Asked by:** Meta, Google, Amazon, Netflix

??? success "View Answer"

    **SettingWithCopyWarning** occurs when modifying a DataFrame slice that might be a view or copy. Use **.loc** for assignment and **.copy()** when needed to avoid ambiguity.

    **Complete Examples:**

    ```python
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': ['a', 'b', 'c', 'd', 'e']
    })
    
    # 1. The Problem: Chained Indexing
    # BAD: This triggers warning
    # df[df['A'] > 2]['B'] = 999  # SettingWithCopyWarning!
    
    # GOOD: Use .loc
    df.loc[df['A'] > 2, 'B'] = 999
    print("Correct assignment:")
    print(df)
    
    # Reset for more examples
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': ['a', 'b', 'c', 'd', 'e']
    })
    
    # 2. View vs Copy
    # Sometimes slice is a view, sometimes a copy
    # Pandas can't always tell which
    
    # This might be a view
    subset1 = df[df['A'] > 2]
    
    # This might be a copy
    subset2 = df[['A', 'B']]
    
    # Modifying triggers warning (is it view or copy?)
    # subset1['B'] = 999  # Warning!
    
    # 3. Explicit Copy
    # Make explicit copy to avoid warning
    subset_copy = df[df['A'] > 2].copy()
    subset_copy['B'] = 999  # No warning
    print("\nExplicit copy:")
    print(subset_copy)
    print("\nOriginal unchanged:")
    print(df)
    
    # 4. Using .loc (Recommended)
    # Always use .loc for assignment
    df_correct = df.copy()
    
    # Single condition
    df_correct.loc[df_correct['A'] > 2, 'B'] = 999
    
    # Multiple conditions
    df_correct.loc[(df_correct['A'] > 2) & (df_correct['B'] < 50), 'C'] = 'updated'
    
    print("\nUsing .loc:")
    print(df_correct)
    
    # 5. Function Modifications
    # BAD: Modifying without copy
    def process_bad(data):
        # If data is a slice, this warns
        data['new_col'] = data['A'] * 2
        return data
    
    # GOOD: Explicit copy
    def process_good(data):
        data = data.copy()
        data['new_col'] = data['A'] * 2
        return data
    
    subset = df[df['A'] > 2]
    # result_bad = process_bad(subset)  # Warning
    result_good = process_good(subset)  # No warning
    print("\nProcessed with copy:")
    print(result_good)
    
    # 6. When to Copy
    print("\n=== When to Use .copy() ===")
    
    # Slice you'll modify
    working_df = df[df['A'] > 2].copy()
    working_df['B'] = 100  # Safe
    
    # Function that modifies
    def transform(data):
        data = data.copy()  # Safe
        data['transformed'] = data['A'] ** 2
        return data
    
    # 7. Detecting Views vs Copies
    # Check if shares memory
    df_original = pd.DataFrame({'A': [1, 2, 3]})
    
    # View (shares memory)
    view = df_original['A']
    print(f"\nView shares memory: {np.shares_memory(df_original['A'].values, view.values)}")
    
    # Copy (doesn't share)
    copy = df_original['A'].copy()
    print(f"Copy shares memory: {np.shares_memory(df_original['A'].values, copy.values)}")
    
    # 8. Common Patterns
    print("\n=== Best Practices ===")
    
    # Pattern 1: Filtering then modifying
    # BAD:
    # filtered = df[df['A'] > 2]
    # filtered['B'] = 100  # Warning
    
    # GOOD:
    filtered = df[df['A'] > 2].copy()
    filtered['B'] = 100  # No warning
    
    # Pattern 2: Direct modification
    # BEST (if you want to modify original):
    df.loc[df['A'] > 2, 'B'] = 100
    
    # Pattern 3: Function returning modified data
    def clean_data(df):
        """Always copy at start of function."""
        df = df.copy()
        df['cleaned'] = df['A'] * 2
        return df
    
    # 9. Disabling Warning (Not Recommended)
    # Only for debugging
    # pd.options.mode.chained_assignment = None  # Disable
    # pd.options.mode.chained_assignment = 'warn'  # Default
    # pd.options.mode.chained_assignment = 'raise'  # Raise error
    
    # 10. Real-World Example
    sales_df = pd.DataFrame({
        'product': ['A', 'B', 'C', 'D', 'E'],
        'sales': [100, 200, 150, 300, 250],
        'region': ['North', 'South', 'North', 'South', 'North']
    })
    
    # Task: Add 10% bonus to North region
    
    # BAD:
    # north = sales_df[sales_df['region'] == 'North']
    # north['sales'] = north['sales'] * 1.1  # Warning!
    
    # GOOD Option 1: Use .loc on original
    sales_df.loc[sales_df['region'] == 'North', 'sales'] *= 1.1
    print("\nSales with bonus:")
    print(sales_df)
    
    # GOOD Option 2: Explicit copy if you need subset
    north_copy = sales_df[sales_df['region'] == 'North'].copy()
    north_copy['sales'] *= 1.1
    print("\nNorth region (copy):")
    print(north_copy)
    ```

    **Common Causes:**

    | Pattern | Problem | Solution |
    |---------|---------|----------|
    | `df[cond][col] = val` | Chained indexing | `df.loc[cond, col] = val` |
    | `subset['col'] = val` | Unclear if view/copy | `subset = df[...].copy()` |
    | `func(df[...])` | Function modifies slice | Copy in function |

    **Best Practices:**

    | Scenario | Recommended Approach |
    |----------|---------------------|
    | **Modify original** | `df.loc[condition, column] = value` |
    | **Modify subset** | `subset = df[...].copy()` |
    | **Function** | Copy at start: `df = df.copy()` |
    | **Read-only** | No copy needed |

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "SettingWithCopyWarning: ambiguous view vs copy"
        - "Chained indexing bad: `df[cond][col] = val`"
        - "Use .loc: `df.loc[cond, col] = val`"
        - "Explicit copy when needed: `.copy()`"
        - "View: shares memory, copy: independent"
        - "Always copy at function start if modifying"
        - "Warning helps avoid silent bugs"
        - "Never disable warning without understanding"
        - "Rule: if modifying slice, use .loc or .copy()"

---

### Explain Pandas String Methods (.str accessor) - Amazon, Microsoft Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `String Methods`, `Text Processing`, `Data Cleaning` | **Asked by:** Amazon, Microsoft, Google, Meta

??? success "View Answer"

    **.str accessor** provides vectorized string operations on Series. Essential for text cleaning, pattern matching, and feature extraction. Much faster than Python loops.

    **Complete Examples:**

    ```python
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame({
        'name': ['  John Doe  ', 'jane smith', 'BOB WILSON', 'Alice Johnson'],
        'email': ['john@example.com', 'jane@test.com', 'bob@EXAMPLE.COM', 'alice@test.com'],
        'phone': ['123-456-7890', '(555) 123-4567', '555.123.4567', '1234567890'],
        'address': ['123 Main St, NYC, NY', '456 Oak Ave, LA, CA', '789 Elm Rd, Austin, TX', '321 Pine Dr, Miami, FL']
    })
    
    # 1. Basic String Operations
    # Lower/upper case
    df['name_lower'] = df['name'].str.lower()
    df['name_upper'] = df['name'].str.upper()
    
    # Title case
    df['name_title'] = df['name'].str.title()
    
    # Strip whitespace
    df['name_clean'] = df['name'].str.strip()
    
    print("Case conversions:")
    print(df[['name', 'name_lower', 'name_title', 'name_clean']].head())
    
    # 2. Contains / Startswith / Endswith
    # Check if contains pattern
    df['has_john'] = df['name'].str.lower().str.contains('john')
    
    # Starts with
    df['starts_with_j'] = df['name'].str.lower().str.startswith('j')
    
    # Ends with
    df['gmail'] = df['email'].str.endswith('@example.com')
    
    print("\nPattern matching:")
    print(df[['name', 'has_john', 'starts_with_j']])
    
    # 3. Replace
    # Simple replace
    df['email_masked'] = df['email'].str.replace('@', '[at]')
    
    # Regex replace
    df['phone_clean'] = df['phone'].str.replace(r'[\(\)\-\.\s]', '', regex=True)
    
    print("\nReplace:")
    print(df[['email', 'email_masked', 'phone', 'phone_clean']])
    
    # 4. Extract with Regex
    # Extract domain from email
    df['domain'] = df['email'].str.extract(r'@(\w+\.\w+)')
    
    # Extract area code
    df['area_code'] = df['phone_clean'].str.extract(r'^(\d{3})')
    
    # Multiple groups
    df[['first_name', 'last_name']] = df['name_clean'].str.extract(r'(\w+)\s+(\w+)')
    
    print("\nExtract:")
    print(df[['email', 'domain', 'first_name', 'last_name']])
    
    # 5. Split
    # Split on space
    name_parts = df['name_clean'].str.split()
    print("\nSplit (list):")
    print(name_parts)
    
    # Expand to columns
    df[['first', 'last']] = df['name_clean'].str.split(n=1, expand=True)
    
    # Split address
    df[['street', 'city', 'state']] = df['address'].str.split(',', expand=True)
    
    print("\nSplit (expanded):")
    print(df[['first', 'last', 'city']])
    
    # 6. Slice
    # First 3 characters
    df['name_abbr'] = df['name_clean'].str[:3]
    
    # Last 4 of phone
    df['phone_last4'] = df['phone_clean'].str[-4:]
    
    print("\nSlice:")
    print(df[['name_clean', 'name_abbr', 'phone_last4']])
    
    # 7. Length and Find
    # String length
    df['name_length'] = df['name_clean'].str.len()
    
    # Find position
    df['at_position'] = df['email'].str.find('@')
    
    print("\nLength and find:")
    print(df[['name_clean', 'name_length', 'email', 'at_position']])
    
    # 8. Count Occurrences
    # Count character
    df['num_spaces'] = df['address'].str.count(' ')
    
    # Count pattern
    df['num_digits'] = df['phone'].str.count(r'\d')
    
    print("\nCount:")
    print(df[['address', 'num_spaces']])
    
    # 9. Padding
    # Pad with zeros
    order_ids = pd.Series([1, 42, 123])
    padded = order_ids.astype(str).str.zfill(5)
    print("\nZero-padded:")
    print(padded)
    
    # Pad left/right
    padded_left = order_ids.astype(str).str.pad(width=5, side='left', fillchar='0')
    padded_right = order_ids.astype(str).str.pad(width=5, side='right', fillchar='*')
    
    # 10. Practical Example: Data Cleaning
    messy_data = pd.DataFrame({
        'customer_name': ['  john DOE  ', 'Jane_Smith', 'bob-wilson'],
        'email': ['JOHN@EXAMPLE.COM', 'jane.smith@test', 'bob@example.COM'],
        'phone': ['(555)123-4567', '555 123 4567', '5551234567'],
        'zip_code': ['12345', '678', '90210']
    })
    
    # Clean everything
    cleaned = messy_data.copy()
    
    # Names: strip, title case, replace special chars
    cleaned['customer_name'] = (cleaned['customer_name']
                                .str.strip()
                                .str.replace(r'[_-]', ' ', regex=True)
                                .str.title())
    
    # Email: lowercase, validate format
    cleaned['email'] = cleaned['email'].str.lower()
    cleaned['email_valid'] = cleaned['email'].str.match(r'^[\w\.-]+@[\w\.-]+\.\w+$')
    
    # Phone: extract digits only, format
    phone_digits = cleaned['phone'].str.replace(r'\D', '', regex=True)
    cleaned['phone_formatted'] = phone_digits.str.replace(
        r'^(\d{3})(\d{3})(\d{4})$',
        r'(\1) \2-\3',
        regex=True
    )
    
    # Zip: pad with zeros
    cleaned['zip_code'] = cleaned['zip_code'].str.zfill(5)
    
    print("\nCleaned data:")
    print(cleaned)
    
    # 11. Performance Comparison
    import time
    
    large_series = pd.Series(['test_string'] * 100000)
    
    # Vectorized (fast)
    start = time.time()
    _ = large_series.str.upper()
    vectorized_time = time.time() - start
    
    # Loop (slow)
    start = time.time()
    _ = [s.upper() for s in large_series]
    loop_time = time.time() - start
    
    print(f"\nVectorized: {vectorized_time:.4f}s")
    print(f"Loop: {loop_time:.4f}s ({loop_time/vectorized_time:.0f}x slower)")
    ```

    **Common String Methods:**

    | Method | Description | Example |
    |--------|-------------|---------|
    | **lower/upper/title** | Case conversion | `.str.lower()` |
    | **strip/lstrip/rstrip** | Remove whitespace | `.str.strip()` |
    | **contains** | Check if contains | `.str.contains('word')` |
    | **replace** | Replace substring | `.str.replace('old', 'new')` |
    | **extract** | Regex extract | `.str.extract(r'(\d+)')` |
    | **split** | Split string | `.str.split(',')` |
    | **[:n]** | Slice | `.str[:5]` |
    | **len** | Length | `.str.len()` |
    | **find** | Find position | `.str.find('@')` |
    | **count** | Count occurrences | `.str.count('a')` |

    **Regex Operations:**

    | Method | Purpose | Example |
    |--------|---------|---------|
    | **contains** | Match pattern | `.str.contains(r'\d{3}')` |
    | **extract** | Extract groups | `.str.extract(r'(\w+)@(\w+)')` |
    | **replace** | Replace pattern | `.str.replace(r'\s+', ' ')` |
    | **match** | Full match | `.str.match(r'^\d{5}$')` |
    | **findall** | Find all matches | `.str.findall(r'\d+')` |

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - ".str accessor: vectorized string operations"
        - "Much faster than Python loops (10-100x)"
        - "Common: lower(), strip(), replace(), contains()"
        - "extract(): use regex with capturing groups"
        - "split(expand=True): split into columns"
        - "replace(regex=True): regex patterns"
        - "Use for: data cleaning, feature extraction"
        - "Returns Series of same length (NaN for missing)"
        - "Performance: always prefer .str over loops"

---

### Explain pd.explode() - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `explode`, `List Columns`, `Data Transformation` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **explode()** transforms list-like column into separate rows. Each list element becomes a new row with duplicated other column values. Essential for nested data structures.

    **Complete Examples:**

    ```python
    import pandas as pd
    import numpy as np
    
    # 1. Basic explode
    df = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie'],
        'hobbies': [['reading', 'swimming'], ['gaming'], ['cooking', 'painting', 'music']]
    })
    
    print("Original (lists in column):")
    print(df)
    
    exploded = df.explode('hobbies')
    print("\nExploded:")
    print(exploded)
    
    # 2. Reset Index
    exploded_reset = df.explode('hobbies').reset_index(drop=True)
    print("\nWith reset index:")
    print(exploded_reset)
    
    # 3. Multiple Columns with Lists
    df_multi = pd.DataFrame({
        'id': [1, 2],
        'tags': [['a', 'b'], ['c', 'd', 'e']],
        'scores': [[10, 20], [30, 40, 50]]
    })
    
    # Explode both columns (must be same length per row)
    exploded_multi = df_multi.explode(['tags', 'scores'])
    print("\nExplode multiple columns:")
    print(exploded_multi)
    
    # 4. With Missing Values
    df_missing = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie'],
        'items': [['apple', 'banana'], None, []]
    })
    
    exploded_na = df_missing.explode('items')
    print("\nWith missing/empty:")
    print(exploded_na)
    
    # 5. Practical Example: User Tags
    users = pd.DataFrame({
        'user_id': [101, 102, 103],
        'username': ['alice', 'bob', 'charlie'],
        'interests': [
            ['python', 'data science', 'ML'],
            ['javascript', 'web dev'],
            ['design', 'UX', 'CSS', 'HTML']
        ]
    })
    
    # Explode to analyze tag frequency
    user_interests = users.explode('interests')
    
    tag_counts = user_interests['interests'].value_counts()
    print("\nTag frequency:")
    print(tag_counts)
    
    # 6. E-commerce Orders
    orders = pd.DataFrame({
        'order_id': [1, 2, 3],
        'customer': ['Alice', 'Bob', 'Charlie'],
        'products': [
            ['Laptop', 'Mouse'],
            ['Keyboard'],
            ['Monitor', 'Cable', 'Stand']
        ],
        'quantities': [
            [1, 2],
            [1],
            [1, 3, 1]
        ],
        'prices': [
            [1000, 25],
            [75],
            [300, 10, 50]
        ]
    })
    
    # Explode order items
    order_items = orders.explode(['products', 'quantities', 'prices'])
    order_items = order_items.reset_index(drop=True)
    
    # Calculate line totals
    order_items['line_total'] = (
        order_items['quantities'].astype(int) * 
        order_items['prices'].astype(int)
    )
    
    print("\nOrder items:")
    print(order_items)
    
    # Total per order
    order_totals = order_items.groupby('order_id')['line_total'].sum()
    print("\nOrder totals:")
    print(order_totals)
    
    # 7. JSON/Nested Data
    # Simulating API response
    api_data = pd.DataFrame({
        'post_id': [1, 2],
        'author': ['Alice', 'Bob'],
        'comments': [
            [
                {'user': 'Bob', 'text': 'Great!'},
                {'user': 'Charlie', 'text': 'Awesome!'}
            ],
            [
                {'user': 'Alice', 'text': 'Thanks!'}
            ]
        ]
    })
    
    # Explode comments
    comments_df = api_data.explode('comments').reset_index(drop=True)
    
    # Extract comment fields
    comments_df['commenter'] = comments_df['comments'].apply(lambda x: x['user'] if x else None)
    comments_df['comment_text'] = comments_df['comments'].apply(lambda x: x['text'] if x else None)
    
    comments_df = comments_df.drop('comments', axis=1)
    
    print("\nExtracted comments:")
    print(comments_df)
    
    # 8. Time Series Events
    events = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=3, freq='D'),
        'events': [
            ['login', 'purchase', 'logout'],
            ['login'],
            ['login', 'view', 'view', 'logout']
        ]
    })
    
    event_log = events.explode('events').reset_index(drop=True)
    
    # Event frequency by day
    event_freq = event_log.groupby(['date', 'events']).size().unstack(fill_value=0)
    print("\nEvent frequency:")
    print(event_freq)
    
    # 9. Performance Note
    # explode is efficient, but watch for very large lists
    import time
    
    # Create test data
    large_df = pd.DataFrame({
        'id': range(1000),
        'values': [[1, 2, 3, 4, 5] for _ in range(1000)]
    })
    
    start = time.time()
    _ = large_df.explode('values')
    explode_time = time.time() - start
    
    print(f"\nExploded 1000 rows × 5 values in {explode_time:.4f}s")
    
    # 10. Reverse: groupby + list
    # Explode → Group back to lists
    exploded_data = pd.DataFrame({
        'category': ['A', 'A', 'B', 'B', 'B'],
        'value': [1, 2, 3, 4, 5]
    })
    
    grouped = exploded_data.groupby('category')['value'].apply(list).reset_index()
    print("\nGrouped back to lists:")
    print(grouped)
    ```

    **explode() Characteristics:**

    | Aspect | Description |
    |--------|-------------|
    | **Input** | Column with lists/arrays/sets |
    | **Output** | One row per list element |
    | **Index** | Duplicates original index |
    | **Other cols** | Repeated for each element |
    | **Empty list** | Creates one row with NaN |
    | **None/NaN** | Creates one row with NaN |

    **Common Patterns:**

    | Pattern | Use Case |
    |---------|----------|
    | **User tags** | Analyze interests, categories |
    | **Order items** | E-commerce line items |
    | **JSON arrays** | Flatten nested API data |
    | **Multi-value fields** | Convert to normalized form |

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "explode(): list column → separate rows"
        - "Each list element becomes new row"
        - "Other columns duplicated"
        - "Index preserved (duplicated)"
        - "reset_index(drop=True) for clean index"
        - "Can explode multiple columns (same lengths)"
        - "Empty list → NaN row"
        - "Reverse: groupby().apply(list)"
        - "Use for: nested data, JSON arrays, multi-value fields"

---

### Explain Pandas Pipe Method - Meta, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Pipe`, `Method Chaining`, `Clean Code` | **Asked by:** Meta, Google, Netflix

??? success "View Answer"

    **pipe()** enables method chaining with custom functions. Pass DataFrame through pipeline of functions for clean, readable data transformations.

    **Complete Examples:**

    ```python
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie', 'David'],
        'age': [25, 30, 35, 40],
        'salary': [50000, 60000, 70000, 80000],
        'department': ['Sales', 'IT', 'Sales', 'IT']
    })
    
    # 1. Without pipe (nested functions - hard to read)
    def add_bonus(df, rate=0.1):
        df = df.copy()
        df['bonus'] = df['salary'] * rate
        return df
    
    def calculate_total(df):
        df = df.copy()
        df['total'] = df['salary'] + df['bonus']
        return df
    
    # Nested (ugly)
    result = calculate_total(add_bonus(df, rate=0.15))
    
    # 2. With pipe (clean, readable)
    result = (df
              .pipe(add_bonus, rate=0.15)
              .pipe(calculate_total))
    
    print("With pipe:")
    print(result)
    
    # 3. Multiple Transformations
    def filter_by_age(df, min_age):
        return df[df['age'] >= min_age]
    
    def add_age_group(df):
        df = df.copy()
        df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 100], 
                                  labels=['Young', 'Middle', 'Senior'])
        return df
    
    def calculate_metrics(df):
        df = df.copy()
        df['salary_per_age'] = df['salary'] / df['age']
        return df
    
    # Clean pipeline
    result = (df
              .pipe(filter_by_age, min_age=28)
              .pipe(add_age_group)
              .pipe(calculate_metrics))
    
    print("\nPipeline result:")
    print(result)
    
    # 4. Mixing Built-in and Custom Functions
    result = (df
              .pipe(lambda d: d[d['age'] > 25])  # Lambda
              .assign(bonus=lambda d: d['salary'] * 0.1)  # Built-in
              .pipe(lambda d: d.sort_values('bonus', ascending=False))
              .reset_index(drop=True))
    
    print("\nMixed pipeline:")
    print(result)
    
    # 5. Pipe with Arguments and Kwargs
    def apply_discount(df, column, discount_rate, condition_col, condition_val):
        df = df.copy()
        mask = df[condition_col] == condition_val
        df.loc[mask, column] = df.loc[mask, column] * (1 - discount_rate)
        return df
    
    result = (df
              .pipe(apply_discount, 
                    column='salary',
                    discount_rate=0.05,
                    condition_col='department',
                    condition_val='Sales'))
    
    print("\nWith arguments:")
    print(result)
    
    # 6. Error Handling in Pipeline
    def validate_data(df):
        """Validate DataFrame."""
        assert not df['salary'].isna().any(), "Salary has NaN"
        assert (df['age'] > 0).all(), "Invalid age"
        return df
    
    def clean_data(df):
        """Clean DataFrame."""
        df = df.copy()
        df = df.dropna()
        df = df[df['age'] > 0]
        return df
    
    # Safe pipeline
    result = (df
              .pipe(clean_data)
              .pipe(validate_data)
              .pipe(add_bonus, rate=0.1))
    
    # 7. Practical Example: Data Processing Pipeline
    sales_data = pd.DataFrame({
        'order_id': range(1, 101),
        'customer_id': np.random.randint(1, 21, 100),
        'product': np.random.choice(['A', 'B', 'C'], 100),
        'quantity': np.random.randint(1, 10, 100),
        'price': np.random.randint(10, 100, 100),
        'date': pd.date_range('2024-01-01', periods=100, freq='D')
    })
    
    def add_revenue(df):
        df = df.copy()
        df['revenue'] = df['quantity'] * df['price']
        return df
    
    def add_date_features(df):
        df = df.copy()
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.day_name()
        return df
    
    def filter_high_value(df, min_revenue=500):
        return df[df['revenue'] >= min_revenue]
    
    def add_customer_segment(df):
        df = df.copy()
        customer_totals = df.groupby('customer_id')['revenue'].sum()
        segments = pd.qcut(customer_totals, q=3, labels=['Low', 'Medium', 'High'])
        df['segment'] = df['customer_id'].map(segments)
        return df
    
    # Complete pipeline
    processed = (sales_data
                 .pipe(add_revenue)
                 .pipe(add_date_features)
                 .pipe(filter_high_value, min_revenue=300)
                 .pipe(add_customer_segment)
                 .pipe(lambda d: d.sort_values(['segment', 'revenue'], 
                                               ascending=[True, False])))
    
    print("\nProcessed sales data:")
    print(processed.head(10))
    
    # 8. Reusable Pipelines
    def sales_pipeline(df, min_revenue=500):
        """Reusable sales processing pipeline."""
        return (df
                .pipe(add_revenue)
                .pipe(add_date_features)
                .pipe(filter_high_value, min_revenue=min_revenue)
                .pipe(add_customer_segment))
    
    # Apply to different datasets
    result1 = sales_pipeline(sales_data, min_revenue=300)
    result2 = sales_pipeline(sales_data, min_revenue=600)
    
    print(f"\nWith min_revenue=300: {len(result1)} rows")
    print(f"With min_revenue=600: {len(result2)} rows")
    
    # 9. pipe() vs Method Chaining
    # Method chaining (only built-in methods)
    result_chain = (df
                    .query('age > 25')
                    .assign(bonus=lambda d: d['salary'] * 0.1)
                    .sort_values('bonus'))
    
    # pipe (custom functions)
    result_pipe = (df
                   .pipe(lambda d: d[d['age'] > 25])
                   .pipe(add_bonus, rate=0.1)
                   .pipe(lambda d: d.sort_values('bonus')))
    
    print("\nBoth approaches work:")
    print(f"Chain: {len(result_chain)} rows")
    print(f"Pipe: {len(result_pipe)} rows")
    
    # 10. Performance Note
    # pipe() has minimal overhead
    import time
    
    large_df = pd.DataFrame({
        'A': range(100000),
        'B': range(100000)
    })
    
    # With pipe
    start = time.time()
    _ = (large_df
         .pipe(lambda d: d[d['A'] > 50000])
         .pipe(lambda d: d.assign(C=d['A'] + d['B'])))
    pipe_time = time.time() - start
    
    # Without pipe
    start = time.time()
    temp = large_df[large_df['A'] > 50000]
    _ = temp.assign(C=temp['A'] + temp['B'])
    direct_time = time.time() - start
    
    print(f"\nPipe: {pipe_time:.4f}s")
    print(f"Direct: {direct_time:.4f}s")
    print("(Virtually same performance)")
    ```

    **pipe() Benefits:**

    | Benefit | Description |
    |---------|-------------|
    | **Readability** | Linear, top-to-bottom flow |
    | **Reusability** | Functions can be reused |
    | **Testability** | Each function testable |
    | **Maintainability** | Easy to add/remove steps |
    | **Debugging** | Comment out steps easily |

    **pipe() vs Method Chaining:**

    | Aspect | pipe() | Method Chaining |
    |--------|--------|-----------------|
    | **Functions** | Custom functions | Built-in methods |
    | **Flexibility** | Very flexible | Limited to methods |
    | **Arguments** | Pass parameters | Use kwargs |
    | **Use Case** | Complex pipelines | Simple transformations |

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "pipe(): pass DataFrame through functions"
        - "Enables method chaining with custom functions"
        - "Clean, readable data pipelines"
        - "Pass arguments: `.pipe(func, arg1, arg2)`"
        - "Better than nested functions: `f(g(h(df)))`"
        - "Combine with built-in methods: assign, query"
        - "Each function: take df, return df"
        - "Benefits: readability, reusability, testability"
        - "Minimal performance overhead"

---

### Explain pd.json_normalize() - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `JSON`, `Nested Data`, `Flattening` | **Asked by:** Amazon, Google, Meta, Netflix

??? success "View Answer"

    **json_normalize()** flattens nested JSON/dict structures into DataFrames. Essential for API responses, nested documents, and semi-structured data.

    **Complete Examples:**

    ```python
    import pandas as pd
    import json
    
    # 1. Simple Nested JSON
    data = [
        {'name': 'Alice', 'age': 25, 'address': {'city': 'NYC', 'zip': '10001'}},
        {'name': 'Bob', 'age': 30, 'address': {'city': 'LA', 'zip': '90001'}}
    ]
    
    df = pd.json_normalize(data)
    print("Flattened JSON:")
    print(df)
    
    # 2. Deep Nesting
    data = [
        {
            'id': 1,
            'user': {
                'name': 'Alice',
                'contact': {
                    'email': 'alice@example.com',
                    'phone': '123-456-7890'
                }
            }
        },
        {
            'id': 2,
            'user': {
                'name': 'Bob',
                'contact': {
                    'email': 'bob@example.com',
                    'phone': '098-765-4321'
                }
            }
        }
    ]
    
    df = pd.json_normalize(data)
    print("\nDeep nesting:")
    print(df)
    
    # 3. Custom Separator
    df = pd.json_normalize(data, sep='_')
    print("\nCustom separator:")
    print(df)
    
    # 4. Arrays in JSON
    data = [
        {'name': 'Alice', 'scores': [90, 85, 88]},
        {'name': 'Bob', 'scores': [78, 82, 80]}
    ]
    
    # Note: Arrays stay as lists
    df = pd.json_normalize(data)
    print("\nWith arrays:")
    print(df)
    
    # Explode to separate rows
    df_exploded = df.explode('scores').reset_index(drop=True)
    print("\nExploded scores:")
    print(df_exploded)
    
    # 5. Nested Arrays of Objects
    data = [
        {
            'order_id': 1,
            'customer': 'Alice',
            'items': [
                {'product': 'Laptop', 'price': 1000},
                {'product': 'Mouse', 'price': 25}
            ]
        },
        {
            'order_id': 2,
            'customer': 'Bob',
            'items': [
                {'product': 'Keyboard', 'price': 75}
            ]
        }
    ]
    
    df = pd.json_normalize(
        data,
        record_path='items',
        meta=['order_id', 'customer']
    )
    print("\nNested arrays of objects:")
    print(df)
    
    # 6. Multiple Levels of Nesting
    data = [
        {
            'id': 1,
            'name': 'Alice',
            'orders': [
                {
                    'order_id': 101,
                    'items': [
                        {'product': 'A', 'qty': 2},
                        {'product': 'B', 'qty': 1}
                    ]
                }
            ]
        }
    ]
    
    # Flatten orders
    df_orders = pd.json_normalize(
        data,
        record_path='orders',
        meta=['id', 'name']
    )
    print("\nFlattened orders:")
    print(df_orders)
    
    # Further flatten items
    # Need to work with the items column
    items_data = []
    for _, row in df_orders.iterrows():
        for item in row['items']:
            items_data.append({
                'id': row['id'],
                'name': row['name'],
                'order_id': row['order_id'],
                'product': item['product'],
                'qty': item['qty']
            })
    
    df_items = pd.DataFrame(items_data)
    print("\nFully flattened:")
    print(df_items)
    
    # 7. Real Example: GitHub API Response
    github_response = [
        {
            'id': 1,
            'name': 'repo1',
            'owner': {
                'login': 'alice',
                'id': 123,
                'url': 'https://github.com/alice'
            },
            'stats': {
                'stars': 100,
                'forks': 20,
                'watchers': 50
            }
        },
        {
            'id': 2,
            'name': 'repo2',
            'owner': {
                'login': 'bob',
                'id': 456,
                'url': 'https://github.com/bob'
            },
            'stats': {
                'stars': 500,
                'forks': 100,
                'watchers': 250
            }
        }
    ]
    
    df = pd.json_normalize(github_response, sep='_')
    print("\nGitHub repos:")
    print(df)
    
    # 8. Handling Missing Keys
    data = [
        {'name': 'Alice', 'age': 25, 'city': 'NYC'},
        {'name': 'Bob', 'age': 30},  # Missing 'city'
        {'name': 'Charlie', 'city': 'LA'}  # Missing 'age'
    ]
    
    df = pd.json_normalize(data)
    print("\nMissing keys (NaN):")
    print(df)
    
    # 9. Real Example: Twitter/X API
    tweets = [
        {
            'id': 1,
            'text': 'Hello World',
            'user': {'name': 'Alice', 'followers': 1000},
            'entities': {
                'hashtags': ['hello', 'world'],
                'mentions': ['@bob']
            },
            'metrics': {'likes': 50, 'retweets': 10}
        },
        {
            'id': 2,
            'text': 'Python is great',
            'user': {'name': 'Bob', 'followers': 5000},
            'entities': {
                'hashtags': ['python'],
                'mentions': []
            },
            'metrics': {'likes': 200, 'retweets': 50}
        }
    ]
    
    df = pd.json_normalize(tweets, sep='_')
    print("\nTweets data:")
    print(df[['id', 'text', 'user_name', 'user_followers', 'metrics_likes']])
    
    # 10. Performance with Large JSON
    # json_normalize is efficient
    import time
    
    large_data = [
        {
            'id': i,
            'data': {
                'value': i * 2,
                'meta': {
                    'timestamp': '2024-01-01',
                    'source': 'api'
                }
            }
        }
        for i in range(10000)
    ]
    
    start = time.time()
    df = pd.json_normalize(large_data, sep='_')
    normalize_time = time.time() - start
    
    print(f"\nNormalized 10k records in {normalize_time:.4f}s")
    print(f"Result shape: {df.shape}")
    ```

    **json_normalize() Parameters:**

    | Parameter | Description | Example |
    |-----------|-------------|---------|
    | **data** | JSON data (list of dicts) | `[{'a': 1}]` |
    | **record_path** | Path to nested records | `record_path='items'` |
    | **meta** | Columns to include from parent | `meta=['id', 'name']` |
    | **sep** | Separator for flattened keys | `sep='_'` (default '.') |
    | **max_level** | Max nesting depth | `max_level=1` |

    **Common Patterns:**

    | Pattern | Solution |
    |---------|----------|
    | **Simple nesting** | `json_normalize(data)` |
    | **Nested arrays** | `json_normalize(data, record_path='items', meta=['id'])` |
    | **Deep nesting** | Flatten in multiple steps |
    | **Arrays of values** | Use with `.explode()` |

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "json_normalize(): flatten nested JSON/dicts"
        - "Converts nested structures to flat DataFrame"
        - "record_path: path to nested array"
        - "meta: parent columns to include"
        - "sep: separator for flattened names (default '.')"
        - "Use with explode() for nested arrays"
        - "Essential for: API responses, NoSQL data"
        - "Handles missing keys (creates NaN)"
        - "Use for: Twitter API, GitHub API, MongoDB docs"

---

### Explain pd.to_datetime() Options - Microsoft, Amazon Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `DateTime`, `Parsing`, `Time Series` | **Asked by:** Microsoft, Amazon, Google, Meta

??? success "View Answer"

    **to_datetime()** parses strings/numbers to datetime objects. Supports multiple formats, error handling, timezone conversions, and performance optimizations with format parameter.

    **Complete Examples:**

    ```python
    import pandas as pd
    import numpy as np
    
    # 1. Basic Parsing
    # Automatic format detection
    dates = pd.Series(['2024-01-15', '2024-02-20', '2024-03-25'])
    parsed = pd.to_datetime(dates)
    print("Basic parsing:")
    print(parsed)
    print(f"Type: {parsed.dtype}")
    
    # 2. Different Formats
    # Common formats
    formats = {
        'ISO': ['2024-01-15', '2024-02-20'],
        'US': ['01/15/2024', '02/20/2024'],
        'EU': ['15/01/2024', '20/02/2024'],
        'Text': ['Jan 15, 2024', 'Feb 20, 2024']
    }
    
    for name, dates in formats.items():
        parsed = pd.to_datetime(dates)
        print(f"\n{name} format:")
        print(parsed)
    
    # 3. Specify Format (Much Faster!)
    dates = pd.Series(['20240115', '20240220'] * 10000)
    
    import time
    
    # Without format (slow, tries multiple formats)
    start = time.time()
    _ = pd.to_datetime(dates)
    auto_time = time.time() - start
    
    # With format (fast, direct parsing)
    start = time.time()
    _ = pd.to_datetime(dates, format='%Y%m%d')
    format_time = time.time() - start
    
    print(f"\nAuto detection: {auto_time:.4f}s")
    print(f"With format: {format_time:.4f}s ({auto_time/format_time:.0f}x faster!)")
    
    # 4. Common Format Codes
    print("\n=== Format Codes ===")
    examples = {
        '%Y-%m-%d': '2024-01-15',
        '%d/%m/%Y': '15/01/2024',
        '%m/%d/%Y': '01/15/2024',
        '%Y%m%d': '20240115',
        '%d-%b-%Y': '15-Jan-2024',
        '%Y-%m-%d %H:%M:%S': '2024-01-15 14:30:00',
        '%d/%m/%Y %I:%M %p': '15/01/2024 02:30 PM'
    }
    
    for fmt, example in examples.items():
        parsed = pd.to_datetime(example, format=fmt)
        print(f"{fmt:25} → {example:25} → {parsed}")
    
    # 5. Error Handling
    mixed_dates = pd.Series(['2024-01-15', '2024-02-20', 'not a date', '2024-03-25'])
    
    # errors='raise' (default) - raises exception
    # try: pd.to_datetime(mixed_dates); except: pass
    
    # errors='ignore' - returns original if can't parse
    ignore_result = pd.to_datetime(mixed_dates, errors='ignore')
    print("\nerrors='ignore':")
    print(ignore_result)
    print(f"Types: {[type(x).__name__ for x in ignore_result]}")
    
    # errors='coerce' - NaT for invalid
    coerce_result = pd.to_datetime(mixed_dates, errors='coerce')
    print("\nerrors='coerce':")
    print(coerce_result)
    
    # 6. From Components
    df = pd.DataFrame({
        'year': [2024, 2024, 2024],
        'month': [1, 2, 3],
        'day': [15, 20, 25]
    })
    
    # Combine columns
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    print("\nFrom components:")
    print(df)
    
    # 7. Unix Timestamps
    timestamps = pd.Series([1704067200, 1706745600, 1709337600])
    
    # Convert (unit='s' for seconds)
    dates = pd.to_datetime(timestamps, unit='s')
    print("\nFrom Unix timestamp:")
    print(dates)
    
    # Milliseconds
    timestamps_ms = timestamps * 1000
    dates_ms = pd.to_datetime(timestamps_ms, unit='ms')
    
    # 8. Timezone Handling
    # Parse with timezone
    dates = pd.Series(['2024-01-15 10:00:00', '2024-02-20 15:30:00'])
    
    # UTC
    utc = pd.to_datetime(dates, utc=True)
    print("\nUTC:")
    print(utc)
    
    # Convert to timezone
    eastern = utc.dt.tz_convert('America/New_York')
    print("\nEastern Time:")
    print(eastern)
    
    # Localize (add timezone to naive)
    naive = pd.to_datetime(dates)
    localized = naive.dt.tz_localize('America/Los_Angeles')
    print("\nLocalized to Pacific:")
    print(localized)
    
    # 9. Day First / Year First
    ambiguous_dates = pd.Series(['01-02-2024', '03-04-2024'])
    
    # Default: month first (US)
    us_format = pd.to_datetime(ambiguous_dates)
    print("\nMonth first (US):")
    print(us_format)
    
    # Day first (EU)
    eu_format = pd.to_datetime(ambiguous_dates, dayfirst=True)
    print("\nDay first (EU):")
    print(eu_format)
    
    # 10. Practical Example: Log File Parsing
    log_data = pd.DataFrame({
        'timestamp': [
            '2024-01-15 10:30:45.123',
            '2024-01-15 10:31:12.456',
            '2024-01-15 10:32:03.789'
        ],
        'event': ['login', 'click', 'logout']
    })
    
    # Parse with milliseconds
    log_data['timestamp'] = pd.to_datetime(
        log_data['timestamp'],
        format='%Y-%m-%d %H:%M:%S.%f'
    )
    
    # Extract time components
    log_data['hour'] = log_data['timestamp'].dt.hour
    log_data['minute'] = log_data['timestamp'].dt.minute
    log_data['second'] = log_data['timestamp'].dt.second
    
    # Calculate duration
    log_data['duration'] = log_data['timestamp'].diff()
    
    print("\nLog data:")
    print(log_data)
    ```

    **Common Format Codes:**

    | Code | Meaning | Example |
    |------|---------|---------|
    | **%Y** | 4-digit year | 2024 |
    | **%y** | 2-digit year | 24 |
    | **%m** | Month (01-12) | 01 |
    | **%d** | Day (01-31) | 15 |
    | **%H** | Hour 24h (00-23) | 14 |
    | **%I** | Hour 12h (01-12) | 02 |
    | **%M** | Minute (00-59) | 30 |
    | **%S** | Second (00-59) | 45 |
    | **%f** | Microsecond | 123456 |
    | **%p** | AM/PM | PM |
    | **%b** | Month abbr | Jan |
    | **%B** | Month full | January |

    **Key Parameters:**

    | Parameter | Description | Example |
    |-----------|-------------|---------|
    | **format** | Explicit format | `format='%Y-%m-%d'` |
    | **errors** | Error handling | `errors='coerce'` |
    | **dayfirst** | Day before month | `dayfirst=True` |
    | **yearfirst** | Year first | `yearfirst=True` |
    | **utc** | Parse as UTC | `utc=True` |
    | **unit** | Timestamp unit | `unit='s'` |

    **Error Handling:**

    | Option | Behavior | Use Case |
    |--------|----------|----------|
    | **raise** | Raise exception | Ensure all valid |
    | **ignore** | Return original | Keep as string |
    | **coerce** | Return NaT | Fill invalid with NaT |

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "to_datetime(): parse strings/numbers to datetime"
        - "format parameter: specify format for speed (10-100x faster)"
        - "Format codes: %Y, %m, %d, %H, %M, %S"
        - "errors='coerce': NaT for invalid dates"
        - "dayfirst=True: European format (day/month/year)"
        - "unit='s': parse Unix timestamps"
        - "utc=True: parse as UTC"
        - "From dict: pass DataFrame with year/month/day columns"
        - "Always specify format for large datasets"

---

### Explain pd.read_csv() Optimization - Google, Netflix Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `File I/O`, `Performance`, `Large Files` | **Asked by:** Google, Netflix, Amazon, Meta

??? success "View Answer"

    **read_csv()** optimization reduces memory and time. Key techniques: dtype specification, usecols, nrows, chunksize, efficient parsers, and using alternative formats.

    **Complete Examples:**

    ```python
    import pandas as pd
    import numpy as np
    import time
    
    # Create sample CSV
    sample_df = pd.DataFrame({
        'id': range(100000),
        'category': np.random.choice(['A', 'B', 'C'], 100000),
        'value': np.random.rand(100000),
        'text': ['some text'] * 100000,
        'date': pd.date_range('2020-01-01', periods=100000, freq='h')
    })
    sample_df.to_csv('sample.csv', index=False)
    
    # 1. Basic (Slow)
    start = time.time()
    df = pd.read_csv('sample.csv')
    basic_time = time.time() - start
    basic_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    print(f"Basic: {basic_time:.4f}s, {basic_memory:.2f} MB")
    print(df.dtypes)
    
    # 2. Specify dtypes (Faster, Less Memory)
    start = time.time()
    df_typed = pd.read_csv(
        'sample.csv',
        dtype={
            'id': 'int32',  # vs int64
            'category': 'category',  # vs object
            'value': 'float32'  # vs float64
        },
        parse_dates=['date']
    )
    typed_time = time.time() - start
    typed_memory = df_typed.memory_usage(deep=True).sum() / 1024**2
    
    print(f"\nWith dtypes: {typed_time:.4f}s, {typed_memory:.2f} MB")
    print(f"Memory saved: {(1 - typed_memory/basic_memory)*100:.1f}%")
    print(df_typed.dtypes)
    
    # 3. Load Only Needed Columns
    start = time.time()
    df_cols = pd.read_csv(
        'sample.csv',
        usecols=['id', 'category', 'value']
    )
    cols_time = time.time() - start
    cols_memory = df_cols.memory_usage(deep=True).sum() / 1024**2
    
    print(f"\nSelected columns: {cols_time:.4f}s, {cols_memory:.2f} MB")
    
    # 4. Load First N Rows (Testing)
    df_sample = pd.read_csv('sample.csv', nrows=1000)
    print(f"\nFirst 1000 rows: {df_sample.shape}")
    
    # 5. Chunking (For Very Large Files)
    def process_chunks(filename, chunksize=10000):
        """Process file in chunks."""
        total = 0
        for chunk in pd.read_csv(filename, chunksize=chunksize):
            # Process each chunk
            total += chunk['value'].sum()
        return total
    
    start = time.time()
    result = process_chunks('sample.csv', chunksize=10000)
    chunk_time = time.time() - start
    print(f"\nChunked processing: {chunk_time:.4f}s, result: {result:.2f}")
    
    # 6. C Engine vs Python Engine
    # C engine (default, faster)
    start = time.time()
    _ = pd.read_csv('sample.csv', engine='c')
    c_time = time.time() - start
    
    # Python engine (more features, slower)
    start = time.time()
    _ = pd.read_csv('sample.csv', engine='python')
    python_time = time.time() - start
    
    print(f"\nC engine: {c_time:.4f}s")
    print(f"Python engine: {python_time:.4f}s")
    
    # 7. Low Memory Mode
    # Good for files with inconsistent types
    df_low_mem = pd.read_csv('sample.csv', low_memory=True)
    
    # 8. Date Parsing Optimization
    # Bad: parse after loading
    start = time.time()
    df1 = pd.read_csv('sample.csv')
    df1['date'] = pd.to_datetime(df1['date'])
    parse_after_time = time.time() - start
    
    # Good: parse during loading
    start = time.time()
    df2 = pd.read_csv('sample.csv', parse_dates=['date'])
    parse_during_time = time.time() - start
    
    print(f"\nParse after: {parse_after_time:.4f}s")
    print(f"Parse during: {parse_during_time:.4f}s")
    
    # 9. Compression
    # Save compressed
    sample_df.to_csv('sample.csv.gz', index=False, compression='gzip')
    
    # Read compressed (automatic detection)
    start = time.time()
    df_compressed = pd.read_csv('sample.csv.gz')
    compressed_time = time.time() - start
    
    import os
    csv_size = os.path.getsize('sample.csv') / 1024**2
    gz_size = os.path.getsize('sample.csv.gz') / 1024**2
    
    print(f"\nCSV size: {csv_size:.2f} MB")
    print(f"Gzip size: {gz_size:.2f} MB ({(1-gz_size/csv_size)*100:.1f}% smaller)")
    print(f"Load time: {compressed_time:.4f}s")
    
    # 10. Alternative: Parquet (Much Better!)
    # Save as parquet
    sample_df.to_parquet('sample.parquet', index=False)
    
    # Read parquet
    start = time.time()
    df_parquet = pd.read_parquet('sample.parquet')
    parquet_time = time.time() - start
    parquet_size = os.path.getsize('sample.parquet') / 1024**2
    
    print(f"\nParquet size: {parquet_size:.2f} MB")
    print(f"Parquet load: {parquet_time:.4f}s")
    print(f"Parquet vs CSV: {basic_time/parquet_time:.1f}x faster!")
    
    # 11. Skip Rows
    # Skip first N rows
    df_skip = pd.read_csv('sample.csv', skiprows=1000)
    
    # Skip by condition (Python engine)
    df_skip_func = pd.read_csv(
        'sample.csv',
        skiprows=lambda x: x % 2 == 0,  # Every other row
        engine='python'
    )
    
    # 12. Practical Recommendations
    print("\n=== Best Practices ===")
    print("1. Specify dtypes (especially category for low-cardinality)")
    print("2. Use usecols to load only needed columns")
    print("3. Use chunksize for files larger than RAM")
    print("4. parse_dates during loading, not after")
    print("5. Consider parquet for repeated reads (10x faster)")
    print("6. Use compression for storage (gzip, snappy)")
    print("7. Test with nrows first to design pipeline")
    print("8. Use low_memory=False to avoid dtype warnings")
    
    # Cleanup
    import os
    for file in ['sample.csv', 'sample.csv.gz', 'sample.parquet']:
        if os.path.exists(file):
            os.remove(file)
    ```

    **Optimization Techniques:**

    | Technique | Speed Improvement | Memory Savings | Use Case |
    |-----------|-------------------|----------------|----------|
    | **dtype** | 10-20% | 50-90% | Specify types |
    | **usecols** | 30-50% | 50-80% | Select columns |
    | **chunksize** | N/A | 90%+ | Larger than RAM |
    | **parse_dates** | 20-30% | - | Date columns |
    | **engine='c'** | 2-5x | - | Default, fastest |
    | **parquet** | 5-10x | 50-80% | Best format |

    **Key Parameters:**

    | Parameter | Purpose | Example |
    |-----------|---------|---------|
    | **dtype** | Specify types | `dtype={'col': 'int32'}` |
    | **usecols** | Select columns | `usecols=['a', 'b']` |
    | **nrows** | Limit rows | `nrows=1000` |
    | **chunksize** | Read in chunks | `chunksize=10000` |
    | **parse_dates** | Parse dates | `parse_dates=['date']` |
    | **low_memory** | Avoid warnings | `low_memory=False` |
    | **engine** | Parser | `engine='c'` (fast) |

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "Specify dtypes: 50-90% memory savings"
        - "usecols: load only needed columns"
        - "chunksize: process files larger than RAM"
        - "parse_dates: faster than parsing after load"
        - "category dtype: 90%+ savings for low cardinality"
        - "int32 vs int64: 50% memory savings"
        - "float32 vs float64: 50% memory savings"
        - "Parquet: 5-10x faster than CSV"
        - "nrows: test with sample first"
        - "engine='c': default, fastest"

---

### Explain Pandas Style for Formatting - Meta, Amazon Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Styling`, `Formatting`, `Visualization` | **Asked by:** Meta, Amazon, Microsoft

??? success "View Answer"

    **style** API formats DataFrame output with colors, highlights, and conditional formatting. Creates publication-ready tables, heatmaps, and visual reports.

    **Complete Examples:**

    ```python
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame({
        'Product': ['A', 'B', 'C', 'D', 'E'],
        'Sales': [10000, 15000, 8000, 12000, 18000],
        'Profit': [2000, 3500, 1200, 2800, 4200],
        'Growth': [0.15, -0.05, 0.22, 0.08, 0.18],
        'Target': [12000, 14000, 10000, 13000, 16000]
    })
    
    # 1. Basic Number Formatting
    styled = df.style.format({
        'Sales': '${:,.0f}',
        'Profit': '${:,.0f}',
        'Growth': '{:.1%}'
    })
    print("Formatted (display in notebook):")
    # styled  # Display in Jupyter
    
    # 2. Highlight Maximum
    styled = df.style.highlight_max(
        subset=['Sales', 'Profit'],
        color='lightgreen'
    )
    
    # 3. Highlight Minimum
    styled = df.style.highlight_min(
        subset=['Sales', 'Profit'],
        color='lightcoral'
    )
    
    # 4. Background Gradient
    styled = df.style.background_gradient(
        subset=['Sales', 'Profit'],
        cmap='YlOrRd'
    )
    
    # 5. Color Negative Values
    def color_negative_red(val):
        color = 'red' if val < 0 else 'black'
        return f'color: {color}'
    
    styled = df.style.applymap(
        color_negative_red,
        subset=['Growth']
    )
    
    # 6. Conditional Formatting (Row-wise)
    def highlight_missed_target(row):
        """Highlight rows that missed target."""
        if row['Sales'] < row['Target']:
            return ['background-color: #ffcccb'] * len(row)
        else:
            return [''] * len(row)
    
    styled = df.style.apply(highlight_missed_target, axis=1)
    
    # 7. Bar Charts in Cells
    styled = df.style.bar(
        subset=['Sales'],
        color='lightblue'
    )
    
    # 8. Combining Multiple Styles
    styled = (df.style
              .format({
                  'Sales': '${:,.0f}',
                  'Profit': '${:,.0f}',
                  'Growth': '{:.1%}',
                  'Target': '${:,.0f}'
              })
              .background_gradient(subset=['Sales'], cmap='Blues')
              .bar(subset=['Profit'], color='lightgreen')
              .highlight_max(subset=['Growth'], color='yellow'))
    
    # 9. Custom Function (Element-wise)
    def color_profit(val):
        """Color profit levels."""
        if val > 3000:
            return 'background-color: #90EE90'
        elif val > 2000:
            return 'background-color: #FFFFCC'
        else:
            return 'background-color: #FFB6C1'
    
    styled = df.style.applymap(color_profit, subset=['Profit'])
    
    # 10. Practical Example: Sales Dashboard
    dashboard_df = pd.DataFrame({
        'Region': ['North', 'South', 'East', 'West'],
        'Q1_Sales': [120000, 95000, 105000, 110000],
        'Q2_Sales': [135000, 88000, 115000, 125000],
        'YoY_Growth': [0.12, -0.08, 0.15, 0.10],
        'Target_Met': ['Yes', 'No', 'Yes', 'Yes']
    })
    
    def highlight_yes_green(val):
        if val == 'Yes':
            return 'background-color: lightgreen'
        elif val == 'No':
            return 'background-color: lightcoral'
        return ''
    
    styled = (dashboard_df.style
              .format({
                  'Q1_Sales': '${:,.0f}',
                  'Q2_Sales': '${:,.0f}',
                  'YoY_Growth': '{:.1%}'
              })
              .background_gradient(subset=['Q1_Sales', 'Q2_Sales'], cmap='YlGn')
              .applymap(highlight_yes_green, subset=['Target_Met'])
              .set_caption('Quarterly Sales Dashboard'))
    
    # 11. Export to HTML/Excel
    # To HTML
    html = styled.to_html()
    # with open('report.html', 'w') as f:
    #     f.write(html)
    
    # To Excel (with xlsxwriter)
    # styled.to_excel('report.xlsx', engine='xlsxwriter')
    
    # 12. Hiding Index/Columns
    styled = (df.style
              .hide_index()
              .format({'Sales': '${:,.0f}'}))
    
    # 13. Setting Properties
    styled = (df.style
              .set_properties(**{
                  'background-color': 'lightblue',
                  'color': 'black',
                  'border-color': 'white'
              }, subset=['Product']))
    
    # 14. Tooltips (Hover Text)
    # Note: More advanced, requires custom CSS
    
    # 15. Practical Report
    financial_df = pd.DataFrame({
        'Metric': ['Revenue', 'COGS', 'Gross Profit', 'Operating Exp', 'Net Income'],
        '2023': [1000000, 600000, 400000, 250000, 150000],
        '2024': [1200000, 680000, 520000, 280000, 240000],
        'Change': [0.20, 0.13, 0.30, 0.12, 0.60]
    })
    
    def color_change(val):
        if val > 0.15:
            return 'color: green; font-weight: bold'
        elif val < 0:
            return 'color: red; font-weight: bold'
        return ''
    
    styled = (financial_df.style
              .format({
                  '2023': '${:,.0f}',
                  '2024': '${:,.0f}',
                  'Change': '{:+.1%}'
              })
              .applymap(color_change, subset=['Change'])
              .set_caption('Financial Report 2023-2024')
              .set_table_styles([
                  {'selector': 'caption',
                   'props': [('font-size', '16px'), ('font-weight', 'bold')]}
              ]))
    
    print("\nStyle API commonly used in Jupyter notebooks for visual reports")
    ```

    **Common Style Methods:**

    | Method | Purpose | Example |
    |--------|---------|---------|
    | **format** | Number formatting | `.format({'col': '{:.2f}'})` |
    | **highlight_max** | Highlight maximum | `.highlight_max(color='green')` |
    | **highlight_min** | Highlight minimum | `.highlight_min(color='red')` |
    | **background_gradient** | Color scale | `.background_gradient(cmap='Blues')` |
    | **bar** | In-cell bar chart | `.bar(color='blue')` |
    | **applymap** | Element-wise function | `.applymap(func, subset=['col'])` |
    | **apply** | Row/column function | `.apply(func, axis=1)` |

    **Format Strings:**

    | Format | Output | Use Case |
    |--------|--------|----------|
    | **`{:.2f}`** | 12.35 | 2 decimals |
    | **`{:,.0f}`** | 1,234 | Thousands separator |
    | **`${:,.2f}`** | $1,234.56 | Currency |
    | **`{:.1%}`** | 12.3% | Percentage |
    | **`{:+.1f}`** | +12.3 | Show sign |

    **Color Maps:**

    | cmap | Description |
    |------|-------------|
    | **'YlOrRd'** | Yellow-Orange-Red (heat) |
    | **'Blues'** | Light to dark blue |
    | **'Greens'** | Light to dark green |
    | **'RdYlGn'** | Red-Yellow-Green (diverging) |
    | **'viridis'** | Perceptually uniform |

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "style: format DataFrame for display"
        - "format(): number formatting (${:,.0f}, {:.1%})"
        - "highlight_max/min(): highlight extremes"
        - "background_gradient(): color scale"
        - "bar(): in-cell bar charts"
        - "applymap(): element-wise styling"
        - "apply(): row/column-wise styling"
        - "Method chaining for multiple styles"
        - "Export: .to_html(), .to_excel()"
        - "Use in: Jupyter reports, dashboards"

---

### Explain Pandas Merge Indicators - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Merge`, `Join`, `Data Quality` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Merge indicator** tracks source of each row (left_only, right_only, both). Essential for data validation, join analysis, and identifying mismatches.

    **Complete Examples:**

    ```python
    import pandas as pd
    
    # Sample data
    customers = pd.DataFrame({
        'customer_id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'city': ['NYC', 'LA', 'Chicago', 'Boston', 'Seattle']
    })
    
    orders = pd.DataFrame({
        'order_id': [101, 102, 103, 104, 105, 106],
        'customer_id': [1, 2, 2, 6, 7, 1],
        'amount': [100, 200, 150, 300, 250, 175]
    })
    
    # 1. Basic Merge with Indicator
    merged = pd.merge(
        customers,
        orders,
        on='customer_id',
        how='outer',
        indicator=True
    )
    
    print("Merge with indicator:")
    print(merged)
    print("\nIndicator values:")
    print(merged['_merge'].value_counts())
    
    # 2. Custom Indicator Name
    merged = pd.merge(
        customers,
        orders,
        on='customer_id',
        how='outer',
        indicator='source'
    )
    
    print("\nCustom indicator name:")
    print(merged[['customer_id', 'name', 'order_id', 'source']])
    
    # 3. Analyze Merge Results
    # Customers without orders
    no_orders = merged[merged['source'] == 'left_only']
    print("\nCustomers without orders:")
    print(no_orders[['customer_id', 'name']])
    
    # Orders without customer match
    orphan_orders = merged[merged['source'] == 'right_only']
    print("\nOrphan orders (no matching customer):")
    print(orphan_orders[['order_id', 'customer_id', 'amount']])
    
    # Successfully matched
    matched = merged[merged['source'] == 'both']
    print("\nMatched records:")
    print(matched[['customer_id', 'name', 'order_id', 'amount']])
    
    # 4. Data Quality Check
    def analyze_merge(left, right, on, how='outer'):
        """Analyze merge quality."""
        merged = pd.merge(left, right, on=on, how=how, indicator=True)
        
        stats = merged['_merge'].value_counts()
        total = len(merged)
        
        print("=== Merge Analysis ===")
        print(f"Total rows: {total}")
        print(f"Left only: {stats.get('left_only', 0)} ({stats.get('left_only', 0)/total*100:.1f}%)")
        print(f"Right only: {stats.get('right_only', 0)} ({stats.get('right_only', 0)/total*100:.1f}%)")
        print(f"Both: {stats.get('both', 0)} ({stats.get('both', 0)/total*100:.1f}%)")
        
        return merged
    
    result = analyze_merge(customers, orders, on='customer_id')
    
    # 5. Validation After Merge
    merged = pd.merge(customers, orders, on='customer_id', how='outer', indicator=True)
    
    # Assert no orphan orders
    orphans = merged[merged['_merge'] == 'right_only']
    if len(orphans) > 0:
        print(f"\n⚠️  WARNING: {len(orphans)} orders without customer!")
        print(orphans[['order_id', 'customer_id']])
    
    # Assert all customers have orders
    no_orders = merged[merged['_merge'] == 'left_only']
    if len(no_orders) > 0:
        print(f"\nINFO: {len(no_orders)} customers without orders")
    
    # 6. Practical Example: Product Catalog Sync
    catalog = pd.DataFrame({
        'sku': ['A001', 'A002', 'A003', 'A004'],
        'product': ['Widget', 'Gadget', 'Tool', 'Device'],
        'price': [10, 20, 15, 25]
    })
    
    inventory = pd.DataFrame({
        'sku': ['A001', 'A002', 'A005', 'A006'],
        'warehouse': ['NYC', 'LA', 'NYC', 'LA'],
        'quantity': [100, 50, 75, 30]
    })
    
    sync = pd.merge(catalog, inventory, on='sku', how='outer', indicator='status')
    
    print("\n=== Catalog Sync Report ===")
    print(f"Products in catalog only: {(sync['status']=='left_only').sum()}")
    print(f"Products in inventory only: {(sync['status']=='right_only').sum()}")
    print(f"Products in both: {(sync['status']=='both').sum()}")
    
    # Products to add to inventory
    to_add = sync[sync['status'] == 'left_only']
    print("\nAdd to inventory:")
    print(to_add[['sku', 'product']])
    
    # Products to remove from inventory
    to_remove = sync[sync['status'] == 'right_only']
    print("\nRemove from inventory:")
    print(to_remove[['sku', 'warehouse']])
    
    # 7. Multiple Key Merge
    sales = pd.DataFrame({
        'region': ['North', 'South', 'North', 'East'],
        'product': ['A', 'B', 'A', 'C'],
        'sales': [100, 200, 150, 180]
    })
    
    targets = pd.DataFrame({
        'region': ['North', 'South', 'West'],
        'product': ['A', 'B', 'A'],
        'target': [120, 180, 100]
    })
    
    comparison = pd.merge(
        sales,
        targets,
        on=['region', 'product'],
        how='outer',
        indicator=True
    )
    
    print("\nSales vs Targets:")
    print(comparison)
    
    # Missing targets
    no_target = comparison[comparison['_merge'] == 'left_only']
    print("\nSales without targets:")
    print(no_target[['region', 'product', 'sales']])
    
    # 8. Time Series Alignment
    actual = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=5, freq='D'),
        'actual': [100, 110, 105, 115, 120]
    })
    
    forecast = pd.DataFrame({
        'date': pd.date_range('2024-01-02', periods=5, freq='D'),
        'forecast': [108, 106, 116, 122, 125]
    })
    
    comparison = pd.merge(
        actual,
        forecast,
        on='date',
        how='outer',
        indicator=True
    ).sort_values('date')
    
    print("\nActual vs Forecast:")
    print(comparison)
    
    # Calculate accuracy only where both exist
    both_exist = comparison[comparison['_merge'] == 'both']
    both_exist['error'] = abs(both_exist['actual'] - both_exist['forecast'])
    print("\nForecast accuracy:")
    print(both_exist[['date', 'actual', 'forecast', 'error']])
    ```

    **Indicator Values:**

    | Value | Meaning | Description |
    |-------|---------|-------------|
    | **'left_only'** | In left only | No match in right |
    | **'right_only'** | In right only | No match in left |
    | **'both'** | In both | Successfully matched |

    **Common Use Cases:**

    | Use Case | Check For |
    |----------|-----------|
    | **Data quality** | Orphan records |
    | **ETL validation** | Missing matches |
    | **Catalog sync** | Products to add/remove |
    | **Customer orders** | Customers without orders |
    | **Join analysis** | Match rate |

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "indicator=True: add _merge column"
        - "Values: 'left_only', 'right_only', 'both'"
        - "Use for: data quality checks, join validation"
        - "Filter by indicator: find unmatched records"
        - "Custom name: `indicator='source'`"
        - "Essential for: ETL pipelines, data validation"
        - "Check match rate: `value_counts()` on _merge"
        - "Find orphans: `merged[merged['_merge']=='right_only']`"
        - "Always use with how='outer' for full picture"

---

### Explain DataFrame.where() and mask() - Meta, Microsoft Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Conditional Operations`, `Filtering`, `Data Replacement` | **Asked by:** Meta, Microsoft, Google

??? success "View Answer"

    **where()** and **mask()** replace values based on conditions. where() keeps True values, mask() replaces True values. Alternative to boolean indexing for conditional replacement.

    **Complete Examples:**

    ```python
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': [100, 200, 300, 400, 500]
    })
    
    # 1. where() - Keep values where condition is True
    # Replace False values with NaN
    result = df.where(df > 20)
    print("where (keep > 20):")
    print(result)
    
    # 2. where() with replacement value
    result = df.where(df > 20, 0)
    print("\nwhere with 0:")
    print(result)
    
    # 3. mask() - Replace values where condition is True
    # Opposite of where()
    result = df.mask(df > 20)
    print("\nmask (replace > 20 with NaN):")
    print(result)
    
    # 4. mask() with replacement value
    result = df.mask(df > 20, 0)
    print("\nmask with 0:")
    print(result)
    
    # 5. Practical Example: Outlier Capping
    data = pd.DataFrame({
        'values': [1, 5, 10, 100, 15, 200, 8, 300]
    })
    
    # Cap at 50
    capped = data['values'].where(data['values'] <= 50, 50)
    print("\nCapped at 50:")
    print(capped)
    
    # Or using mask
    capped2 = data['values'].mask(data['values'] > 50, 50)
    print("\nSame with mask:")
    print(capped2)
    
    # 6. Element-wise Operations
    df = pd.DataFrame({
        'temperature': [-5, 0, 10, 25, 35, 42],
        'humidity': [20, 40, 60, 80, 90, 95]
    })
    
    # Replace extreme temperatures
    df['temp_adjusted'] = df['temperature'].where(
        (df['temperature'] >= 0) & (df['temperature'] <= 40)
    )
    print("\nTemperature filtering:")
    print(df)
    
    # 7. Using Functions
    # where/mask can take callable
    result = df['temperature'].where(lambda x: x > 0)
    print("\nUsing lambda:")
    print(result)
    
    # 8. Multiple Conditions
    sales = pd.DataFrame({
        'product': ['A', 'B', 'C', 'D', 'E'],
        'quantity': [10, 50, 100, 200, 5],
        'price': [10, 20, 15, 25, 30]
    })
    
    # Flag high-value sales
    sales['category'] = 'Normal'
    sales['category'] = sales['category'].where(
        sales['quantity'] * sales['price'] < 1000,
        'High Value'
    )
    print("\nSales categories:")
    print(sales)
    
    # 9. vs Boolean Indexing
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50]
    })
    
    # Boolean indexing (modifies subset)
    df_bool = df.copy()
    df_bool.loc[df_bool['A'] > 3, 'B'] = 0
    print("\nBoolean indexing:")
    print(df_bool)
    
    # where (returns new, preserves shape)
    df_where = df.copy()
    df_where['B'] = df_where['B'].where(df_where['A'] <= 3, 0)
    print("\nWith where:")
    print(df_where)
    
    # 10. Practical: Data Cleaning
    data = pd.DataFrame({
        'age': [-5, 25, 150, 30, 200, 45],
        'salary': [50000, -10000, 80000, 90000, 1000000, 75000]
    })
    
    # Valid ranges
    data['age_clean'] = data['age'].where(
        (data['age'] >= 0) & (data['age'] <= 120)
    )
    data['salary_clean'] = data['salary'].where(
        (data['salary'] >= 0) & (data['salary'] <= 500000)
    )
    print("\nData cleaning:")
    print(data)
    
    # 11. Winsorization
    # Cap extreme values
    data = pd.Series([1, 2, 3, 4, 5, 100, 200, 300])
    
    lower = data.quantile(0.05)
    upper = data.quantile(0.95)
    
    winsorized = (data
                  .where(data >= lower, lower)
                  .where(data <= upper, upper))
    print("\nWinsorized:")
    print(winsorized)
    
    # 12. Time Series Example
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    ts = pd.DataFrame({
        'date': dates,
        'value': [10, 50, 100, 200, 150, 180, 90, 70, 60, 55]
    })
    
    # Flag anomalies
    mean = ts['value'].mean()
    std = ts['value'].std()
    
    ts['anomaly'] = 'Normal'
    ts['anomaly'] = ts['anomaly'].where(
        abs(ts['value'] - mean) <= 2 * std,
        'Anomaly'
    )
    print("\nAnomaly detection:")
    print(ts)
    
    # 13. Performance Note
    # where/mask are efficient (vectorized)
    large_df = pd.DataFrame(np.random.randn(100000, 3), columns=['A', 'B', 'C'])
    
    import time
    
    # where()
    start = time.time()
    _ = large_df.where(large_df > 0, 0)
    where_time = time.time() - start
    
    # Boolean indexing (slower for full replacement)
    start = time.time()
    temp = large_df.copy()
    temp[temp < 0] = 0
    bool_time = time.time() - start
    
    print(f"\nwhere(): {where_time:.4f}s")
    print(f"Boolean: {bool_time:.4f}s")
    ```

    **where() vs mask():**

    | Method | Behavior | Equivalent |
    |--------|----------|------------|
    | **where(cond)** | Keep True, replace False | `df[cond]` |
    | **mask(cond)** | Replace True, keep False | `df[~cond]` |

    **Use Cases:**

    | Use Case | Method | Example |
    |----------|--------|---------|
    | **Outlier capping** | where | `s.where(s <= 100, 100)` |
    | **Filter & replace** | where | `df.where(df > 0, 0)` |
    | **Anomaly removal** | mask | `df.mask(is_anomaly)` |
    | **Winsorization** | where (twice) | Cap both tails |

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "where(): keep True, replace False with other"
        - "mask(): replace True with other, keep False"
        - "Default other: NaN"
        - "where(cond, value): replace False with value"
        - "Preserves DataFrame shape (vs boolean indexing)"
        - "Vectorized: efficient on large data"
        - "Use for: outlier capping, data cleaning, winsorization"
        - "Can take callable: `where(lambda x: x > 0)`"
        - "mask() = where(~cond)"

---

### Explain nlargest() and nsmallest() - Amazon, Netflix Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Selection`, `Sorting`, `Top-N` | **Asked by:** Amazon, Netflix, Google, Meta

??? success "View Answer"

    **nlargest()** and **nsmallest()** select top/bottom N rows by column(s). Faster than sorting for small N. Useful for top products, customers, values.

    **Complete Examples:**

    ```python
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame({
        'product': ['A', 'B', 'C', 'D', 'E', 'F'],
        'sales': [1000, 5000, 3000, 8000, 2000, 6000],
        'profit': [200, 1000, 600, 1500, 400, 1200],
        'region': ['North', 'South', 'North', 'East', 'West', 'South']
    })
    
    # 1. Top 3 by Sales
    top_sales = df.nlargest(3, 'sales')
    print("Top 3 by sales:")
    print(top_sales)
    
    # 2. Bottom 3 by Sales
    bottom_sales = df.nsmallest(3, 'sales')
    print("\nBottom 3 by sales:")
    print(bottom_sales)
    
    # 3. Multiple Columns (Tiebreaker)
    # First by sales, then by profit
    top_multi = df.nlargest(3, ['sales', 'profit'])
    print("\nTop 3 by sales, then profit:")
    print(top_multi)
    
    # 4. vs sort_values + head
    import time
    
    # Large dataset
    large_df = pd.DataFrame({
        'value': np.random.randn(1000000)
    })
    
    # nlargest (fast)
    start = time.time()
    _ = large_df.nlargest(10, 'value')
    nlargest_time = time.time() - start
    
    # sort + head (slow)
    start = time.time()
    _ = large_df.sort_values('value', ascending=False).head(10)
    sort_time = time.time() - start
    
    print(f"\nnlargest: {nlargest_time:.4f}s")
    print(f"sort + head: {sort_time:.4f}s ({sort_time/nlargest_time:.1f}x slower)")
    
    # 5. Series nlargest/nsmallest
    sales_series = pd.Series([100, 500, 300, 800, 200], 
                             index=['A', 'B', 'C', 'D', 'E'])
    
    print("\nTop 3 products:")
    print(sales_series.nlargest(3))
    
    print("\nBottom 2 products:")
    print(sales_series.nsmallest(2))
    
    # 6. Practical: Top Customers
    customers = pd.DataFrame({
        'customer_id': range(1, 101),
        'total_spent': np.random.randint(100, 10000, 100),
        'orders': np.random.randint(1, 50, 100),
        'last_order_days': np.random.randint(0, 365, 100)
    })
    
    # Top 10 spenders
    top_customers = customers.nlargest(10, 'total_spent')
    print("\nTop 10 customers by spending:")
    print(top_customers[['customer_id', 'total_spent', 'orders']])
    
    # Most frequent buyers
    frequent_buyers = customers.nlargest(10, 'orders')
    print("\nTop 10 frequent buyers:")
    print(frequent_buyers[['customer_id', 'orders', 'total_spent']])
    
    # 7. GroupBy + nlargest
    # Top 2 products per region
    top_per_region = df.groupby('region').apply(
        lambda x: x.nlargest(2, 'sales')
    ).reset_index(drop=True)
    print("\nTop 2 products per region:")
    print(top_per_region)
    
    # 8. Practical: Stock Analysis
    stocks = pd.DataFrame({
        'ticker': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META'],
        'price': [175, 140, 380, 145, 250, 350],
        'volume': [50000000, 25000000, 30000000, 45000000, 80000000, 20000000],
        'change_pct': [2.5, -1.2, 1.8, 0.5, 5.2, -2.1]
    })
    
    # Top gainers
    print("\nTop 3 gainers:")
    print(stocks.nlargest(3, 'change_pct')[['ticker', 'change_pct']])
    
    # Top losers
    print("\nTop 3 losers:")
    print(stocks.nsmallest(3, 'change_pct')[['ticker', 'change_pct']])
    
    # Most traded
    print("\nMost traded:")
    print(stocks.nlargest(3, 'volume')[['ticker', 'volume']])
    
    # 9. Keep Parameter
    # Handle duplicates
    df_dup = pd.DataFrame({
        'name': ['A', 'B', 'C', 'D', 'E'],
        'score': [10, 20, 20, 15, 10]
    })
    
    print("\nOriginal with duplicates:")
    print(df_dup)
    
    # keep='first' (default)
    print("\nTop 2 (keep='first'):")
    print(df_dup.nlargest(2, 'score', keep='first'))
    
    # keep='last'
    print("\nTop 2 (keep='last'):")
    print(df_dup.nlargest(2, 'score', keep='last'))
    
    # keep='all' (may return more than n)
    print("\nTop 2 (keep='all'):")
    print(df_dup.nlargest(2, 'score', keep='all'))
    
    # 10. Real-World: Sales Leaderboard
    sales_data = pd.DataFrame({
        'salesperson': [f'Person_{i}' for i in range(1, 51)],
        'q1_sales': np.random.randint(10000, 50000, 50),
        'q2_sales': np.random.randint(10000, 50000, 50),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 50)
    })
    
    sales_data['total_sales'] = sales_data['q1_sales'] + sales_data['q2_sales']
    
    # Overall top 10
    leaderboard = sales_data.nlargest(10, 'total_sales')
    print("\nTop 10 salespeople:")
    print(leaderboard[['salesperson', 'total_sales', 'region']])
    
    # Top 3 per region
    regional_leaders = sales_data.groupby('region').apply(
        lambda x: x.nlargest(3, 'total_sales')
    ).reset_index(drop=True)
    print("\nTop 3 per region:")
    print(regional_leaders[['region', 'salesperson', 'total_sales']])
    ```

    **Performance Comparison:**

    | Dataset Size | Top N | nlargest() | sort_values() + head() |
    |--------------|-------|------------|------------------------|
    | 10K rows | 10 | 0.001s | 0.005s (5x slower) |
    | 100K rows | 10 | 0.005s | 0.050s (10x slower) |
    | 1M rows | 10 | 0.05s | 0.5s (10x slower) |
    | 1M rows | 100 | 0.08s | 0.5s (6x slower) |

    **When to Use:**

    | Scenario | Best Method |
    |----------|-------------|
    | **Top/bottom N (N << size)** | nlargest/nsmallest |
    | **Need all sorted** | sort_values |
    | **Multiple columns** | Both work, nlargest faster |
    | **Very small N (<100)** | nlargest (much faster) |

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "nlargest(n, 'col'): top N rows by column"
        - "nsmallest(n, 'col'): bottom N rows"
        - "Faster than sort_values + head for small N"
        - "Multiple columns: `nlargest(n, ['col1', 'col2'])`"
        - "keep parameter: 'first', 'last', 'all' for ties"
        - "Works on Series and DataFrame"
        - "Use for: top products, customers, leaderboards"
        - "O(n log k) vs O(n log n) for sorting"
        - "GroupBy + nlargest: top N per group"

---

### Explain Pandas rank() Method - Microsoft, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Ranking`, `Statistics`, `Order` | **Asked by:** Microsoft, Google, Amazon

??? success "View Answer"

    **rank()** assigns ranks to values in Series/DataFrame. Supports multiple methods for handling ties (average, min, max, first, dense). Essential for leaderboards, percentiles, and relative comparisons.

    **Complete Examples:**

    ```python
    import pandas as pd
    import numpy as np
    
    # Sample data with ties
    s = pd.Series([100, 200, 200, 300, 400, 400, 400])
    
    # 1. Default Ranking (average for ties)
    print("Original values:")
    print(s.values)
    print("\nDefault rank (average):")
    print(s.rank())
    
    # 2. Different Ranking Methods
    methods = ['average', 'min', 'max', 'first', 'dense']
    
    df = pd.DataFrame({'value': s})
    for method in methods:
        df[f'rank_{method}'] = s.rank(method=method)
    
    print("\nAll ranking methods:")
    print(df)
    
    # 3. Ascending vs Descending
    scores = pd.Series([85, 92, 78, 92, 95])
    
    print("\nScores:")
    print(scores.values)
    
    # ascending=True (default): 1 = smallest
    print("\nRank ascending (1 = lowest):")
    print(scores.rank())
    
    # ascending=False: 1 = largest
    print("\nRank descending (1 = highest):")
    print(scores.rank(ascending=False))
    
    # 4. Percentile Ranks
    # pct=True: ranks from 0 to 1
    print("\nPercentile ranks:")
    print(scores.rank(pct=True))
    
    # 5. Practical: Student Grades
    students = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'math': [85, 92, 78, 92, 95],
        'english': [88, 85, 90, 82, 95]
    })
    
    # Rank by math (1 = best)
    students['math_rank'] = students['math'].rank(ascending=False, method='min')
    students['english_rank'] = students['english'].rank(ascending=False, method='min')
    
    # Overall rank (sum of ranks, lower is better)
    students['overall_rank'] = (students['math_rank'] + students['english_rank']).rank()
    
    print("\nStudent rankings:")
    print(students)
    
    # 6. GroupBy Ranking
    sales = pd.DataFrame({
        'region': ['North', 'North', 'South', 'South', 'East', 'East'],
        'salesperson': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
        'sales': [1000, 1500, 800, 1200, 900, 1100]
    })
    
    # Rank within each region
    sales['rank_in_region'] = sales.groupby('region')['sales'].rank(ascending=False)
    
    print("\nRanking within regions:")
    print(sales)
    
    # 7. Handling NaN
    s_nan = pd.Series([100, np.nan, 200, 300, np.nan])
    
    # na_option='keep' (default): NaN stays NaN
    print("\nWith NaN (keep):")
    print(s_nan.rank())
    
    # na_option='top': NaN gets smallest rank
    print("\nWith NaN (top):")
    print(s_nan.rank(na_option='top'))
    
    # na_option='bottom': NaN gets largest rank
    print("\nWith NaN (bottom):")
    print(s_nan.rank(na_option='bottom'))
    
    # 8. Practical: Stock Performance
    stocks = pd.DataFrame({
        'ticker': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'],
        'return_1m': [5.2, -1.5, 3.8, 2.1, 8.5],
        'return_6m': [12.5, 8.2, 15.1, 10.3, 25.2],
        'volatility': [0.25, 0.30, 0.22, 0.28, 0.45]
    })
    
    # Rank returns (higher is better)
    stocks['return_1m_rank'] = stocks['return_1m'].rank(ascending=False)
    stocks['return_6m_rank'] = stocks['return_6m'].rank(ascending=False)
    
    # Rank volatility (lower is better)
    stocks['vol_rank'] = stocks['volatility'].rank(ascending=True)
    
    # Combined score (lower is better)
    stocks['combined_rank'] = (
        stocks['return_1m_rank'] + 
        stocks['return_6m_rank'] + 
        stocks['vol_rank']
    ).rank()
    
    print("\nStock rankings:")
    print(stocks)
    
    # 9. Quantile-based Ranking
    values = pd.Series(np.random.randn(100))
    
    # Percentile ranks
    values_df = pd.DataFrame({
        'value': values,
        'percentile': values.rank(pct=True),
        'quartile': pd.qcut(values.rank(pct=True), q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    })
    
    print("\nQuantile-based ranking:")
    print(values_df.head(10))
    
    # 10. Leaderboard Example
    leaderboard = pd.DataFrame({
        'player': [f'Player_{i}' for i in range(1, 11)],
        'score': [1000, 1500, 1200, 1500, 1800, 900, 1500, 1200, 1000, 2000]
    })
    
    # Dense ranking (no gaps)
    leaderboard['rank_dense'] = leaderboard['score'].rank(method='dense', ascending=False)
    
    # Min ranking (standard)
    leaderboard['rank_standard'] = leaderboard['score'].rank(method='min', ascending=False)
    
    leaderboard = leaderboard.sort_values('score', ascending=False)
    
    print("\nLeaderboard:")
    print(leaderboard)
    
    # 11. Performance Comparison
    large_series = pd.Series(np.random.randn(1000000))
    
    import time
    
    start = time.time()
    _ = large_series.rank()
    rank_time = time.time() - start
    
    start = time.time()
    _ = large_series.sort_values().reset_index(drop=True)
    sort_time = time.time() - start
    
    print(f"\nrank(): {rank_time:.4f}s")
    print(f"sort: {sort_time:.4f}s")
    ```

    **Ranking Methods:**

    | Method | Description | Example [100, 200, 200, 300] |
    |--------|-------------|------------------------------|
    | **average** | Average of ranks | [1.0, 2.5, 2.5, 4.0] |
    | **min** | Minimum rank | [1, 2, 2, 4] |
    | **max** | Maximum rank | [1, 3, 3, 4] |
    | **first** | Order they appear | [1, 2, 3, 4] |
    | **dense** | No gaps | [1, 2, 2, 3] |

    **Key Parameters:**

    | Parameter | Options | Use Case |
    |-----------|---------|----------|
    | **method** | average, min, max, first, dense | Tie handling |
    | **ascending** | True, False | Rank direction |
    | **pct** | True, False | Percentile ranks (0-1) |
    | **na_option** | keep, top, bottom | NaN handling |

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "rank(): assign ranks to values"
        - "ascending=False: 1 = highest (for scores)"
        - "method='min': standard ranking (gaps for ties)"
        - "method='dense': no gaps (1, 2, 2, 3)"
        - "method='first': order of appearance"
        - "pct=True: percentile ranks (0.0 to 1.0)"
        - "Use with groupby(): rank within groups"
        - "na_option: handle NaN ('keep', 'top', 'bottom')"
        - "Use for: leaderboards, percentiles, relative comparisons"

---

### Explain pd.cut() vs pd.qcut() - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Binning`, `Discretization`, `Feature Engineering` | **Asked by:** Amazon, Google, Meta, Netflix

??? success "View Answer"

    **cut()** bins data into equal-width intervals. **qcut()** bins into equal-sized quantiles. cut() for fixed ranges, qcut() for balanced distribution.

    **Complete Examples:**

    ```python
    import pandas as pd
    import numpy as np
    
    ages = pd.Series([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80])
    
    # 1. pd.cut() - Equal-Width Bins
    # Bins have same width, different counts
    age_bins = pd.cut(ages, bins=4)
    print("cut() - Equal width:")
    print(age_bins)
    print("\nValue counts:")
    print(age_bins.value_counts().sort_index())
    
    # 2. pd.qcut() - Equal-Frequency Bins
    # Bins have different widths, same counts
    age_quantiles = pd.qcut(ages, q=4)
    print("\nqcut() - Equal frequency:")
    print(age_quantiles)
    print("\nValue counts:")
    print(age_quantiles.value_counts().sort_index())
    
    # 3. Custom Bin Edges (cut)
    custom_bins = pd.cut(ages, bins=[0, 18, 35, 60, 100], 
                         labels=['Child', 'Young Adult', 'Adult', 'Senior'])
    print("\nCustom bins:")
    print(pd.DataFrame({'age': ages, 'category': custom_bins}))
    
    # 4. Percentile-based (qcut)
    # Quartiles
    quartiles = pd.qcut(ages, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    print("\nQuartiles:")
    print(pd.DataFrame({'age': ages, 'quartile': quartiles}))
    
    # 5. Practical: Income Brackets
    incomes = pd.Series(np.random.randint(20000, 200000, 100))
    
    # Equal-width bins
    income_cut = pd.cut(incomes, bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    print("\nIncome bins (equal width):")
    print(income_cut.value_counts())
    
    # Equal-frequency bins (balanced)
    income_qcut = pd.qcut(incomes, q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    print("\nIncome bins (equal frequency):")
    print(income_qcut.value_counts())
    
    # 6. Include/Exclude Boundaries
    scores = pd.Series([0, 25, 50, 75, 100])
    
    # Right edge included (default)
    bins_right = pd.cut(scores, bins=[0, 50, 100], right=True)
    print("\nRight edge included:")
    print(bins_right)
    
    # Left edge included
    bins_left = pd.cut(scores, bins=[0, 50, 100], right=False)
    print("\nLeft edge included:")
    print(bins_left)
    
    # 7. Retbins - Get Bin Edges
    data = pd.Series(np.random.randn(100))
    
    # Get bins used
    binned, bin_edges = pd.cut(data, bins=5, retbins=True)
    print("\nBin edges:")
    print(bin_edges)
    
    # 8. Practical: Credit Score Categories
    credit_scores = pd.Series(np.random.randint(300, 850, 1000))
    
    # Standard credit score ranges
    credit_bins = pd.cut(
        credit_scores,
        bins=[300, 580, 670, 740, 800, 850],
        labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
    )
    
    print("\nCredit score distribution:")
    print(credit_bins.value_counts().sort_index())
    
    # 9. Handling Duplicates in qcut
    # qcut can fail with many duplicates
    data_dup = pd.Series([1, 1, 1, 2, 2, 3, 4, 5, 5, 5])
    
    try:
        pd.qcut(data_dup, q=4)
    except ValueError as e:
        print(f"\nqcut error: {e}")
    
    # Solution: duplicates='drop'
    binned_dup = pd.qcut(data_dup, q=4, duplicates='drop')
    print("\nWith duplicates='drop':")
    print(binned_dup.value_counts())
    
    # 10. Machine Learning Feature Engineering
    df = pd.DataFrame({
        'customer_id': range(1, 101),
        'age': np.random.randint(18, 80, 100),
        'income': np.random.randint(20000, 150000, 100),
        'spend': np.random.randint(100, 5000, 100)
    })
    
    # Create features
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 100], labels=['Young', 'Middle', 'Senior'])
    df['income_quartile'] = pd.qcut(df['income'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    df['spend_level'] = pd.qcut(df['spend'], q=3, labels=['Low', 'Medium', 'High'])
    
    print("\nFeature engineering:")
    print(df.head(10))
    
    # Distribution
    print("\nFeature distributions:")
    print(df[['age_group', 'income_quartile', 'spend_level']].apply(lambda x: x.value_counts()))
    ```

    **cut() vs qcut():**

    | Aspect | cut() | qcut() |
    |--------|-------|--------|
    | **Bins** | Equal width | Equal frequency |
    | **Use Case** | Fixed ranges | Balanced groups |
    | **Distribution** | Uneven | Even |
    | **Example** | Age groups | Quartiles |
    | **Custom Edges** | Yes | No (only q) |

    **Common Patterns:**

    | Task | Solution |
    |------|----------|
    | **Age groups** | `cut(ages, [0, 18, 65, 100])` |
    | **Quartiles** | `qcut(data, q=4)` |
    | **Credit scores** | `cut(scores, [300, 580, 670, 740, 800, 850])` |
    | **Income percentiles** | `qcut(income, q=10)` (deciles) |
    | **Equal groups for ML** | `qcut(feature, q=5)` |

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "cut(): equal-width bins, fixed ranges"
        - "qcut(): equal-frequency bins, quantiles"
        - "cut() bins can have different counts"
        - "qcut() bins have same counts (balanced)"
        - "cut() for: age groups, custom ranges"
        - "qcut() for: quartiles, percentiles, balanced features"
        - "labels: custom category names"
        - "retbins=True: return bin edges"
        - "qcut with duplicates: use duplicates='drop'"
        - "Use for: feature engineering, binning, discretization"

---

### Explain Pandas cumsum(), cumprod(), cummax(), cummin() - Meta, Microsoft Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Cumulative Operations`, `Time Series`, `Running Totals` | **Asked by:** Meta, Microsoft, Amazon, Google

??? success "View Answer"

    **Cumulative functions** compute running totals, products, max, min. Essential for time series analysis, running totals, cumulative statistics, and financial calculations.

    **Complete Examples:**

    ```python
    import pandas as pd
    import numpy as np
    
    data = pd.Series([1, 2, 3, 4, 5])
    
    # 1. cumsum() - Cumulative Sum
    print("Original:")
    print(data.values)
    print("\ncumsum():")
    print(data.cumsum().values)
    # [1, 3, 6, 10, 15]
    
    # 2. cumprod() - Cumulative Product
    print("\ncumprod():")
    print(data.cumprod().values)
    # [1, 2, 6, 24, 120]
    
    # 3. cummax() - Cumulative Maximum
    data_random = pd.Series([3, 1, 4, 1, 5, 2, 6])
    print("\nOriginal:")
    print(data_random.values)
    print("\ncummax():")
    print(data_random.cummax().values)
    # [3, 3, 4, 4, 5, 5, 6]
    
    # 4. cummin() - Cumulative Minimum
    print("\ncummin():")
    print(data_random.cummin().values)
    # [3, 1, 1, 1, 1, 1, 1]
    
    # 5. Practical: Running Total Sales
    sales = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=7, freq='D'),
        'daily_sales': [100, 150, 120, 180, 200, 160, 190]
    })
    
    sales['cumulative_sales'] = sales['daily_sales'].cumsum()
    
    print("\nRunning sales total:")
    print(sales)
    
    # 6. Financial: Cumulative Returns
    returns = pd.Series([0.05, -0.02, 0.03, 0.01, -0.01, 0.04])
    
    # Cumulative product for compound returns
    cumulative_return = (1 + returns).cumprod() - 1
    
    print("\nDaily returns:")
    print(returns.values)
    print("\nCumulative returns:")
    print(cumulative_return.values)
    
    # 7. Peak Detection (cummax)
    stock_prices = pd.Series([100, 105, 103, 108, 110, 107, 115, 112])
    
    df = pd.DataFrame({
        'price': stock_prices,
        'peak': stock_prices.cummax()
    })
    df['drawdown'] = (df['price'] - df['peak']) / df['peak']
    
    print("\nDrawdown analysis:")
    print(df)
    
    # 8. GroupBy Cumulative
    orders = pd.DataFrame({
        'customer': ['A', 'A', 'B', 'A', 'B', 'B'],
        'amount': [100, 200, 150, 300, 100, 200]
    })
    
    orders['cumulative_per_customer'] = orders.groupby('customer')['amount'].cumsum()
    
    print("\nCumulative per customer:")
    print(orders)
    
    # 9. Time Series: Running Metrics
    ts = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10, freq='D'),
        'value': np.random.randint(10, 50, 10)
    })
    
    ts['running_total'] = ts['value'].cumsum()
    ts['running_avg'] = ts['value'].expanding().mean()
    ts['running_max'] = ts['value'].cummax()
    ts['running_min'] = ts['value'].cummin()
    
    print("\nTime series running metrics:")
    print(ts)
    
    # 10. Practical: Inventory Tracking
    inventory = pd.DataFrame({
        'transaction': ['Sale', 'Purchase', 'Sale', 'Sale', 'Purchase', 'Sale'],
        'quantity': [-10, 50, -15, -20, 40, -12]
    })
    
    inventory['stock_level'] = 100 + inventory['quantity'].cumsum()
    
    print("\nInventory levels:")
    print(inventory)
    
    # 11. Year-to-Date Calculations
    monthly = pd.DataFrame({
        'month': pd.date_range('2024-01-01', periods=12, freq='MS'),
        'revenue': np.random.randint(50000, 100000, 12)
    })
    
    monthly['ytd_revenue'] = monthly['revenue'].cumsum()
    
    print("\nYear-to-date revenue:")
    print(monthly)
    
    # 12. Performance Comparison
    large_series = pd.Series(np.random.randn(1000000))
    
    import time
    
    # cumsum (fast, O(n))
    start = time.time()
    _ = large_series.cumsum()
    cumsum_time = time.time() - start
    
    # Manual cumsum (slow)
    start = time.time()
    result = []
    total = 0
    for val in large_series:
        total += val
        result.append(total)
    manual_time = time.time() - start
    
    print(f"\ncumsum(): {cumsum_time:.4f}s")
    print(f"Manual: {manual_time:.4f}s ({manual_time/cumsum_time:.1f}x slower)")
    ```

    **Cumulative Functions:**

    | Function | Description | Example [1,2,3,4] |
    |----------|-------------|-------------------|
    | **cumsum()** | Running sum | [1, 3, 6, 10] |
    | **cumprod()** | Running product | [1, 2, 6, 24] |
    | **cummax()** | Running maximum | [1, 2, 3, 4] |
    | **cummin()** | Running minimum | [1, 1, 1, 1] |

    **Common Use Cases:**

    | Use Case | Function | Example |
    |----------|----------|---------|
    | **Running totals** | cumsum() | Sales YTD |
    | **Compound returns** | cumprod() | Investment growth |
    | **Peak tracking** | cummax() | All-time high |
    | **Drawdown** | price - cummax() | Portfolio decline |
    | **Inventory** | cumsum() | Stock levels |
    | **Streak tracking** | Custom with cumsum | Winning streaks |

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "cumsum(): running sum/total"
        - "cumprod(): running product (compound growth)"
        - "cummax(): running maximum (peak)"
        - "cummin(): running minimum (trough)"
        - "Use with groupby(): cumulative per group"
        - "Financial: cumprod() for compound returns"
        - "Drawdown: price - cummax()"
        - "O(n) complexity: efficient"
        - "Common in: time series, finance, inventory"

---

### Explain pd.concat() axis Parameter - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Concatenation`, `Joining`, `Data Combination` | **Asked by:** Google, Amazon, Meta, Microsoft

??? success "View Answer"

    **concat() axis parameter** controls concatenation direction. **axis=0** (default) stacks vertically (rows). **axis=1** stacks horizontally (columns). Essential for combining datasets.

    **Complete Examples:**

    ```python
    import pandas as pd
    import numpy as np
    
    df1 = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    
    df2 = pd.DataFrame({
        'A': [7, 8, 9],
        'B': [10, 11, 12]
    })
    
    df3 = pd.DataFrame({
        'C': [13, 14, 15],
        'D': [16, 17, 18]
    })
    
    # 1. axis=0 (Vertical Stacking - Rows)
    result_v = pd.concat([df1, df2], axis=0)
    print("Vertical concatenation (axis=0):")
    print(result_v)
    
    # Reset index
    result_v_reset = pd.concat([df1, df2], axis=0, ignore_index=True)
    print("\nWith ignore_index=True:")
    print(result_v_reset)
    
    # 2. axis=1 (Horizontal Stacking - Columns)
    result_h = pd.concat([df1, df3], axis=1)
    print("\nHorizontal concatenation (axis=1):")
    print(result_h)
    
    # 3. Mismatched Indexes
    df_a = pd.DataFrame({'X': [1, 2, 3]}, index=['a', 'b', 'c'])
    df_b = pd.DataFrame({'Y': [4, 5, 6]}, index=['b', 'c', 'd'])
    
    # axis=1: outer join (default)
    result = pd.concat([df_a, df_b], axis=1)
    print("\nMismatched indexes (axis=1):")
    print(result)
    
    # axis=1 with inner join
    result_inner = pd.concat([df_a, df_b], axis=1, join='inner')
    print("\nWith join='inner':")
    print(result_inner)
    
    # 4. Multiple DataFrames
    df_list = [
        pd.DataFrame({'value': [1, 2]}),
        pd.DataFrame({'value': [3, 4]}),
        pd.DataFrame({'value': [5, 6]})
    ]
    
    # Vertical
    combined_v = pd.concat(df_list, axis=0, ignore_index=True)
    print("\nMultiple DataFrames (vertical):")
    print(combined_v)
    
    # 5. Practical: Combining Regional Data
    north = pd.DataFrame({
        'product': ['A', 'B', 'C'],
        'sales': [100, 200, 150]
    })
    
    south = pd.DataFrame({
        'product': ['A', 'B', 'C'],
        'sales': [120, 180, 160]
    })
    
    # Vertical (add rows)
    all_regions_v = pd.concat([north, south], axis=0, keys=['North', 'South'])
    print("\nRegional data (vertical):")
    print(all_regions_v)
    
    # Horizontal (add columns)
    all_regions_h = pd.concat(
        [north.set_index('product')['sales'], 
         south.set_index('product')['sales']], 
        axis=1, 
        keys=['North', 'South']
    )
    print("\nRegional data (horizontal):")
    print(all_regions_h)
    
    # 6. Keys Parameter
    q1 = pd.DataFrame({'revenue': [100, 200]})
    q2 = pd.DataFrame({'revenue': [150, 250]})
    
    quarterly = pd.concat([q1, q2], axis=0, keys=['Q1', 'Q2'])
    print("\nWith keys (MultiIndex):")
    print(quarterly)
    
    # 7. Names Parameter
    quarterly_named = pd.concat(
        [q1, q2], 
        axis=0, 
        keys=['Q1', 'Q2'],
        names=['Quarter', 'ID']
    )
    print("\nWith names:")
    print(quarterly_named)
    
    # 8. Practical: Time Series Concatenation
    jan = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=5, freq='D'),
        'value': [10, 12, 11, 13, 14]
    })
    
    feb = pd.DataFrame({
        'date': pd.date_range('2024-02-01', periods=5, freq='D'),
        'value': [15, 16, 14, 17, 18]
    })
    
    full_data = pd.concat([jan, feb], axis=0, ignore_index=True)
    print("\nTime series concatenation:")
    print(full_data)
    
    # 9. Verify No Data Loss
    def safe_concat(dfs, axis):
        """Concat with verification."""
        result = pd.concat(dfs, axis=axis)
        
        if axis == 0:
            expected_rows = sum(len(df) for df in dfs)
            assert len(result) == expected_rows, "Row count mismatch!"
        else:
            expected_cols = sum(len(df.columns) for df in dfs)
            assert len(result.columns) == expected_cols, "Column count mismatch!"
        
        return result
    
    result = safe_concat([df1, df2], axis=0)
    print(f"\nSafe concat: {result.shape}")
    
    # 10. Performance Comparison
    dfs = [pd.DataFrame({'A': range(1000)}) for _ in range(100)]
    
    import time
    
    # concat (fast)
    start = time.time()
    _ = pd.concat(dfs, axis=0)
    concat_time = time.time() - start
    
    # append (deprecated, slow)
    # result = dfs[0]
    # for df in dfs[1:]:
    #     result = result.append(df)
    
    print(f"\nconcat time: {concat_time:.4f}s")
    print("(append is deprecated, use concat)")
    ```

    **axis Parameter:**

    | axis | Direction | Stacking | Use Case |
    |------|-----------|----------|----------|
    | **0** | Vertical | Add rows | Combine datasets |
    | **1** | Horizontal | Add columns | Add features |

    **Common Parameters:**

    | Parameter | Purpose | Example |
    |-----------|---------|---------|
    | **axis** | Direction (0=rows, 1=cols) | `axis=0` |
    | **ignore_index** | Reset index | `ignore_index=True` |
    | **keys** | Add MultiIndex | `keys=['A', 'B']` |
    | **join** | 'outer' or 'inner' | `join='inner'` |
    | **names** | Name MultiIndex levels | `names=['level1']` |

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "axis=0: vertical stacking (add rows)"
        - "axis=1: horizontal stacking (add columns)"
        - "ignore_index=True: reset index"
        - "join='outer': keep all (default)"
        - "join='inner': keep common only"
        - "keys: add MultiIndex for source tracking"
        - "Use for: combining datasets, regional data"
        - "axis=0 + ignore_index: clean row numbers"
        - "axis=1: must have compatible indexes"
        - "Prefer concat over append (deprecated)"

---

### Explain Pandas Sparse Data Structures - Netflix, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Memory Optimization`, `Sparse Arrays`, `Performance` | **Asked by:** Netflix, Google, Amazon

??? success "View Answer"

    **Sparse arrays** store only non-zero/non-default values, saving memory for mostly-empty data. Essential for one-hot encoding, matrices with many zeros, and large categorical features.

    **Complete Examples:**

    ```python
    import pandas as pd
    import numpy as np
    
    # 1. Create Sparse Array
    # Most values are 0 (or another fill_value)
    dense = pd.Series([0, 0, 1, 0, 0, 2, 0, 0, 0, 3])
    
    # Convert to sparse
    sparse = dense.astype(pd.SparseDtype("int64", fill_value=0))
    
    print("Dense memory:")
    print(f"{dense.memory_usage(deep=True):,} bytes")
    print("\nSparse memory:")
    print(f"{sparse.memory_usage(deep=True):,} bytes")
    print(f"\nMemory savings: {(1 - sparse.memory_usage(deep=True)/dense.memory_usage(deep=True))*100:.1f}%")
    
    # 2. Sparse DataFrame
    # One-hot encoded data (mostly zeros)
    df = pd.DataFrame({
        'A': pd.Categorical(['cat', 'dog', 'cat', 'bird'] * 1000)
    })
    
    # Dense one-hot encoding
    dense_dummies = pd.get_dummies(df['A'])
    
    # Sparse one-hot encoding
    sparse_dummies = pd.get_dummies(df['A'], sparse=True)
    
    print("\nOne-hot encoding:")
    print(f"Dense: {dense_dummies.memory_usage(deep=True).sum():,} bytes")
    print(f"Sparse: {sparse_dummies.memory_usage(deep=True).sum():,} bytes")
    print(f"Savings: {(1 - sparse_dummies.memory_usage(deep=True).sum()/dense_dummies.memory_usage(deep=True).sum())*100:.1f}%")
    
    # 3. SparseArray Properties
    sparse_arr = pd.arrays.SparseArray([0, 0, 1, 2, 0, 0, 0, 3, 0])
    
    print("\nSparse array:")
    print(f"Values: {sparse_arr}")
    print(f"Density: {sparse_arr.density:.2%}")
    print(f"Fill value: {sparse_arr.fill_value}")
    print(f"Non-zero count: {sparse_arr.npoints}")
    
    # 4. Operations on Sparse Data
    s1 = pd.Series([0, 0, 1, 0, 2]).astype(pd.SparseDtype("int64", 0))
    s2 = pd.Series([0, 1, 0, 0, 3]).astype(pd.SparseDtype("int64", 0))
    
    # Arithmetic (stays sparse)
    result = s1 + s2
    print("\nArithmetic on sparse:")
    print(result)
    print(f"Result is sparse: {isinstance(result.dtype, pd.SparseDtype)}")
    
    # 5. Practical: User-Item Matrix
    # Typical recommendation system matrix (very sparse)
    users = 10000
    items = 1000
    
    # Simulate ratings (only 1% have ratings)
    n_ratings = int(users * items * 0.01)
    user_ids = np.random.randint(0, users, n_ratings)
    item_ids = np.random.randint(0, items, n_ratings)
    ratings = np.random.randint(1, 6, n_ratings)
    
    # Dense matrix (BAD for sparse data)
    # Would need: users * items * 8 bytes = 80 MB
    
    # Sparse representation (GOOD)
    sparse_df = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'rating': ratings
    })
    
    print(f"\nSparse ratings:")
    print(f"Memory: {sparse_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"vs full matrix: ~{users * items * 8 / 1024**2:.2f} MB")
    print(f"Savings: ~{(1 - sparse_df.memory_usage(deep=True).sum()/(users*items*8))*100:.1f}%")
    
    # 6. Convert Between Dense and Sparse
    dense = pd.Series([0, 1, 0, 2, 0])
    
    # To sparse
    sparse = dense.astype(pd.SparseDtype("int64", fill_value=0))
    print("\nConverted to sparse:")
    print(sparse)
    
    # Back to dense
    dense_again = sparse.astype("int64")
    print("\nBack to dense:")
    print(dense_again)
    
    # 7. Sparse with Different Fill Values
    # If most values are -1
    data = pd.Series([-1, -1, 5, -1, -1, 10, -1, -1])
    sparse_neg = data.astype(pd.SparseDtype("int64", fill_value=-1))
    
    print("\nSparse with fill_value=-1:")
    print(sparse_neg)
    print(f"Density: {sparse_neg.density:.2%}")
    
    # 8. Practical: Feature Engineering
    # High-cardinality categorical → sparse one-hot
    df = pd.DataFrame({
        'user_id': range(1000),
        'country': np.random.choice(['US', 'UK', 'CA', 'AU', 'DE'] * 200, 1000)
    })
    
    # Sparse one-hot encoding
    country_sparse = pd.get_dummies(df['country'], prefix='country', sparse=True)
    
    print("\nSparse feature engineering:")
    print(f"Shape: {country_sparse.shape}")
    print(f"Memory: {country_sparse.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    # 9. When to Use Sparse
    def analyze_sparsity(series):
        """Check if sparse makes sense."""
        zero_count = (series == 0).sum()
        total = len(series)
        sparsity = zero_count / total
        
        print(f"Sparsity: {sparsity:.1%}")
        if sparsity > 0.9:
            print("✓ Good candidate for sparse")
        elif sparsity > 0.5:
            print("⚠ Maybe use sparse")
        else:
            print("✗ Not sparse enough")
        
        return sparsity
    
    data1 = pd.Series(np.random.randint(0, 2, 1000))  # 50% zeros
    data2 = pd.Series(np.random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 1000))  # 90% zeros
    
    print("\nData 1:")
    analyze_sparsity(data1)
    
    print("\nData 2:")
    analyze_sparsity(data2)
    
    # 10. Limitations
    print("\n=== Sparse Limitations ===")
    print("- Arithmetic can densify results")
    print("- Some operations require conversion to dense")
    print("- Not all pandas functions support sparse")
    print("- Best for: ≥90% sparsity")
    print("- Use cases: one-hot encoding, user-item matrices, NLP features")
    ```

    **When to Use Sparse:**

    | Sparsity | Recommendation | Example |
    |----------|----------------|---------|
    | **>95%** | Definitely use sparse | User-item matrix |
    | **90-95%** | Likely beneficial | One-hot encoding |
    | **50-90%** | Maybe | Depends on operations |
    | **<50%** | Don't use sparse | Regular data |

    **Memory Comparison:**

    | Data | Dense | Sparse | Savings |
    |------|-------|--------|---------|
    | **99% zeros** | 1 GB | ~10 MB | ~99% |
    | **95% zeros** | 1 GB | ~50 MB | ~95% |
    | **90% zeros** | 1 GB | ~100 MB | ~90% |

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "Sparse: store only non-zero/non-default values"
        - "Memory savings: proportional to sparsity"
        - "pd.get_dummies(sparse=True): sparse one-hot"
        - "SparseDtype(dtype, fill_value): specify type"
        - "Use when: >90% same value (often zeros)"
        - "density: fraction of non-sparse values"
        - "npoints: count of non-sparse values"
        - "Use cases: one-hot encoding, user-item matrix, NLP"
        - "Limitation: some ops require densification"
        - "Trade-off: memory vs computation speed"

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
