
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
