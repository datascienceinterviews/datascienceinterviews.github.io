
---
title: NumPy Interview Questions
description: 100+ NumPy interview questions for cracking Data Science, Machine Learning, and Quant Developer interviews
---

# NumPy Interview Questions

<!-- [TOC] -->

This document provides a curated list of NumPy interview questions commonly asked in technical interviews for Data Science, Quantitative Analyst, Machine Learning Engineer, and High-Performance Computing roles. It covers everything from array manipulation to advanced linear algebra and memory management.

This is updated frequently but right now this is the most exhaustive list of type of questions being asked.

---

## Premium Interview Questions

### What is NumPy and Why is it Faster Than Python Lists? - Google, Amazon Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Basics`, `Performance` | **Asked by:** Google, Amazon, Meta, Netflix

??? success "View Answer"

    **NumPy Overview:**
    
    NumPy is Python's fundamental package for scientific computing, built on C for speed.
    
    **Why Faster Than Lists:**
    
    | Feature | Python List | NumPy Array |
    |---------|-------------|-------------|
    | Storage | Scattered pointers | Contiguous memory |
    | Type | Mixed types | Homogeneous |
    | Operations | Python loops | Vectorized C |
    | Memory | ~8x more | Compact |
    
    ```python
    import numpy as np
    import time
    
    # Python list
    py_list = list(range(1000000))
    start = time.time()
    result = [x * 2 for x in py_list]
    print(f"List: {time.time() - start:.4f}s")
    
    # NumPy array
    np_arr = np.arange(1000000)
    start = time.time()
    result = np_arr * 2
    print(f"NumPy: {time.time() - start:.4f}s")
    # NumPy is ~100x faster!
    ```
    
    **Key Benefits:**
    
    - Vectorized operations (no loops)
    - Broadcasting for shape compatibility
    - BLAS/LAPACK for linear algebra
    - Memory-efficient storage

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of performance fundamentals.
        
        **Strong answer signals:**
        
        - Mentions contiguous memory
        - Explains vectorization
        - Knows SIMD/BLAS acceleration

---

### Explain Broadcasting in NumPy - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Broadcasting`, `Vectorization` | **Asked by:** Google, Amazon, Meta, Apple

??? success "View Answer"

    **Broadcasting Rules:**
    
    1. Arrays are compared element-wise from trailing dimensions
    2. Dimensions are compatible if equal or one is 1
    3. Missing dimensions are treated as 1
    
    ```python
    import numpy as np
    
    # Example 1: Scalar broadcast
    a = np.array([1, 2, 3])
    b = 10
    print(a + b)  # [11, 12, 13]
    
    # Example 2: Row + Column broadcast
    row = np.array([[1, 2, 3]])          # Shape (1, 3)
    col = np.array([[10], [20], [30]])   # Shape (3, 1)
    print(row + col)
    # [[11, 12, 13],
    #  [21, 22, 23],
    #  [31, 32, 33]]
    
    # Example 3: Distance matrix
    A = np.array([[1, 2], [3, 4], [5, 6]])  # (3, 2)
    B = np.array([[0, 0], [1, 1]])           # (2, 2)
    
    # Expand dims for broadcasting
    diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]  # (3, 2, 2)
    distances = np.sqrt(np.sum(diff**2, axis=-1))     # (3, 2)
    ```
    
    **Common Error:**
    
    ```python
    a = np.array([1, 2, 3])      # Shape (3,)
    b = np.array([1, 2, 3, 4])   # Shape (4,)
    a + b  # ValueError: shapes not compatible
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Core NumPy understanding.
        
        **Strong answer signals:**
        
        - States broadcasting rules clearly
        - Uses np.newaxis for dimension expansion
        - Avoids unnecessary loops

---

### Difference Between flatten() and ravel() - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Array Manipulation`, `Memory` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Key Difference:**
    
    | Method | Returns | Memory |
    |--------|---------|--------|
    | `flatten()` | Always copy | Safe to modify |
    | `ravel()` | View if possible | May share memory |
    
    ```python
    import numpy as np
    
    a = np.array([[1, 2], [3, 4]])
    
    # flatten() - always returns copy
    flat = a.flatten()
    flat[0] = 99
    print(a)  # [[1, 2], [3, 4]] - unchanged
    
    # ravel() - returns view when possible
    raveled = a.ravel()
    raveled[0] = 99
    print(a)  # [[99, 2], [3, 4]] - changed!
    
    # ravel() returns copy if needed
    b = np.array([[1, 2], [3, 4]], order='F')  # Fortran order
    raveled = b.ravel()  # Copy needed for C-contiguous
    ```
    
    **When to Use:**
    
    - `flatten()`: Safe, need independent copy
    - `ravel()`: Performance-critical, won't modify
    - `reshape(-1)`: Similar to ravel, explicit

    !!! tip "Interviewer's Insight"
        **What they're testing:** Memory management awareness.
        
        **Strong answer signals:**
        
        - Knows view vs copy difference
        - Considers memory layout (C vs F)
        - Uses ravel for read-only operations

---

### How to Perform Matrix Multiplication in NumPy? - Google, Amazon Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Linear Algebra` | **Asked by:** Most Tech Companies

??? success "View Answer"

    **Matrix Multiplication Methods:**
    
    ```python
    import numpy as np
    
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    
    # Method 1: @ operator (Python 3.5+)
    C = A @ B
    
    # Method 2: np.matmul()
    C = np.matmul(A, B)
    
    # Method 3: np.dot() - works for 1D and 2D
    C = np.dot(A, B)
    
    # All give same result:
    # [[19, 22],
    #  [43, 50]]
    ```
    
    **Difference Between Methods:**
    
    | Operation | @/matmul | np.dot |
    |-----------|----------|--------|
    | Matrix-matrix | Same | Same |
    | Batched (3D+) | Batch multiply | Different behavior |
    | Vector-vector | Inner product | Same |
    
    ```python
    # Batched matrix multiplication
    batch_A = np.random.rand(10, 3, 4)  # 10 matrices of 3x4
    batch_B = np.random.rand(10, 4, 5)  # 10 matrices of 4x5
    
    result = batch_A @ batch_B  # Shape: (10, 3, 5)
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Linear algebra basics.
        
        **Strong answer signals:**
        
        - Uses @ operator for clarity
        - Knows batch multiplication
        - Understands shape requirements

---

### What is np.where() and How to Use It? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Conditional Logic`, `Vectorization` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **np.where() - Vectorized If-Else:**
    
    ```python
    import numpy as np
    
    arr = np.array([1, 2, 3, 4, 5])
    
    # Basic: np.where(condition, if_true, if_false)
    result = np.where(arr > 3, arr * 10, arr)
    # [1, 2, 3, 40, 50]
    
    # Return indices where condition is True
    indices = np.where(arr > 3)
    # (array([3, 4]),)
    
    # Multiple conditions
    result = np.where(arr < 2, 'small',
                      np.where(arr < 4, 'medium', 'large'))
    # ['small', 'medium', 'medium', 'large', 'large']
    
    # Better for multiple conditions: np.select
    conditions = [arr < 2, arr < 4, arr >= 4]
    choices = ['small', 'medium', 'large']
    result = np.select(conditions, choices)
    ```
    
    **Use Cases:**
    
    - Replace values conditionally
    - Find indices of elements
    - Vectorized if-else operations

    !!! tip "Interviewer's Insight"
        **What they're testing:** Vectorized conditional logic.
        
        **Strong answer signals:**
        
        - Uses np.where instead of loops
        - Knows np.select for multiple conditions
        - Returns indices with single argument

---

### How to Handle Memory Layout (C vs Fortran Order)? - Google, HFT Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Internals`, `Performance` | **Asked by:** HFT Firms, Google, Amazon

??? success "View Answer"

    **Memory Order:**
    
    | Order | Row Major (C) | Column Major (F) |
    |-------|---------------|------------------|
    | Access | Row by row | Column by column |
    | Default | NumPy, C | MATLAB, Fortran |
    | Contiguous | Rows in memory | Columns in memory |
    
    ```python
    import numpy as np
    
    a = np.array([[1, 2, 3], [4, 5, 6]])
    
    # Check order
    print(a.flags['C_CONTIGUOUS'])  # True (default)
    print(a.flags['F_CONTIGUOUS'])  # False
    
    # Create Fortran order
    b = np.array([[1, 2, 3], [4, 5, 6]], order='F')
    print(b.flags['F_CONTIGUOUS'])  # True
    
    # Performance implication
    # Row access faster in C order
    for row in a:
        process(row)  # Fast
    
    # Column access faster in F order
    for col in b.T:
        process(col)  # Fast in F order
    ```
    
    **When It Matters:**
    
    - BLAS/LAPACK calls expect specific order
    - Cache efficiency for iteration
    - Interfacing with C/Fortran code

    !!! tip "Interviewer's Insight"
        **What they're testing:** Low-level optimization knowledge.
        
        **Strong answer signals:**
        
        - Knows cache locality implications
        - Checks contiguity for performance
        - Uses appropriate order for access pattern

---

### How to Use np.einsum() for Einstein Summation? - Google, DeepMind Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Advanced Linear Algebra` | **Asked by:** Google, DeepMind, OpenAI

??? success "View Answer"

    **einsum - Einstein Summation:**
    
    ```python
    import numpy as np
    
    A = np.random.rand(3, 4)
    B = np.random.rand(4, 5)
    
    # Matrix multiplication: C_ij = sum_k A_ik * B_kj
    C = np.einsum('ik,kj->ij', A, B)
    
    # Transpose
    A_T = np.einsum('ij->ji', A)
    
    # Trace (diagonal sum)
    trace = np.einsum('ii->', np.eye(3))  # 3.0
    
    # Outer product
    outer = np.einsum('i,j->ij', np.arange(3), np.arange(4))
    
    # Batch matrix multiply
    batch_A = np.random.rand(10, 3, 4)
    batch_B = np.random.rand(10, 4, 5)
    result = np.einsum('bij,bjk->bik', batch_A, batch_B)
    
    # Attention mechanism (simplified)
    Q = np.random.rand(8, 64)  # queries
    K = np.random.rand(8, 64)  # keys
    attention = np.einsum('qd,kd->qk', Q, K)  # (8, 8)
    ```
    
    **Common Patterns:**
    
    | Pattern | Operation |
    |---------|-----------|
    | `ij->ji` | Transpose |
    | `ij,jk->ik` | Matrix multiply |
    | `ii->` | Trace |
    | `i,j->ij` | Outer product |
    | `bij,bjk->bik` | Batch matmul |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Advanced tensor operations.
        
        **Strong answer signals:**
        
        - Writes einsum notation correctly
        - Uses for attention/transformers
        - Knows when einsum is more readable

---

### How to Generate Random Numbers in NumPy? - Google, Amazon Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Random Sampling`, `Reproducibility` | **Asked by:** Most Tech Companies

??? success "View Answer"

    **Modern Random API (NumPy 1.17+):**
    
    ```python
    import numpy as np
    
    # Create a generator
    rng = np.random.default_rng(seed=42)
    
    # Uniform [0, 1)
    rng.random(size=(3, 4))
    
    # Integers
    rng.integers(low=0, high=10, size=5)
    
    # Normal distribution
    rng.normal(loc=0, scale=1, size=100)
    
    # Choice/sampling
    rng.choice(['a', 'b', 'c'], size=10, replace=True)
    
    # Shuffle
    arr = np.arange(10)
    rng.shuffle(arr)
    ```
    
    **Legacy API (still common):**
    
    ```python
    np.random.seed(42)              # Global seed
    np.random.rand(3, 4)            # Uniform [0, 1)
    np.random.randn(3, 4)           # Standard normal
    np.random.randint(0, 10, 5)     # Integers
    np.random.choice([1, 2, 3], 5)  # Random choice
    ```
    
    **Generator vs RandomState:**
    
    | Feature | Generator (new) | RandomState (old) |
    |---------|-----------------|-------------------|
    | Thread-safe | Yes | No |
    | Algorithm | PCG64 | Mersenne Twister |
    | Reproducible | Per-generator | Global state |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Reproducibility awareness.
        
        **Strong answer signals:**
        
        - Uses default_rng() for new code
        - Sets seed for reproducibility
        - Knows thread-safety differences

---

### How to Copy vs View Arrays? - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Memory Management` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **View vs Copy:**
    
    ```python
    import numpy as np
    
    a = np.array([1, 2, 3, 4, 5])
    
    # VIEW - shares memory
    view = a[1:4]      # Slicing creates view
    view[0] = 99
    print(a)           # [1, 99, 3, 4, 5] - modified!
    
    # COPY - independent memory
    copy = a[1:4].copy()
    copy[0] = 100
    print(a)           # [1, 99, 3, 4, 5] - unchanged
    
    # Check if view or copy
    print(view.base is a)  # True (view)
    print(copy.base)       # None (copy)
    ```
    
    **Operations That Create Views:**
    
    | Operation | View | Copy |
    |-----------|------|------|
    | Slicing | ✅ | |
    | reshape() | ✅ (if possible) | |
    | transpose() | ✅ | |
    | Boolean indexing | | ✅ |
    | Fancy indexing | | ✅ |
    | flatten() | | ✅ |
    
    **Force Copy:**
    
    ```python
    # Explicit copy
    b = a.copy()
    b = np.array(a, copy=True)
    
    # Ensure contiguous copy
    b = np.ascontiguousarray(a)
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Memory safety awareness.
        
        **Strong answer signals:**
        
        - Knows which ops create views
        - Uses .base to check sharing
        - Uses .copy() for safety

---

### How to Solve Linear Equations with NumPy? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Linear Algebra` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Solving Ax = b:**
    
    ```python
    import numpy as np
    
    # System: 2x + y = 5, x - y = 1
    A = np.array([[2, 1], [1, -1]])
    b = np.array([5, 1])
    
    # Method 1: np.linalg.solve (recommended)
    x = np.linalg.solve(A, b)
    print(x)  # [2., 1.]
    
    # Method 2: Inverse (less stable, slower)
    x = np.linalg.inv(A) @ b
    
    # Verify solution
    print(np.allclose(A @ x, b))  # True
    ```
    
    **Least Squares (overdetermined):**
    
    ```python
    # More equations than unknowns
    A = np.array([[1, 1], [1, 2], [1, 3]])
    b = np.array([1, 2, 2.5])
    
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    ```
    
    **Eigendecomposition:**
    
    ```python
    A = np.array([[4, 2], [1, 3]])
    
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Verify: A @ v = λ * v
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        print(np.allclose(A @ v, eigenvalues[i] * v))  # True
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Numerical linear algebra.
        
        **Strong answer signals:**
        
        - Uses solve() not inv()
        - Knows lstsq for overdetermined
        - Verifies solutions with allclose

---

### How to Compute Statistics with NumPy? - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Statistics` | **Asked by:** Most Tech Companies

??? success "View Answer"

    **Basic Statistics:**
    
    ```python
    import numpy as np
    
    data = np.array([2, 4, 6, 8, 10, 2, 4])
    
    # Central tendency
    np.mean(data)        # 5.14
    np.median(data)      # 4.0
    
    # Spread
    np.std(data)         # 2.67
    np.var(data)         # 7.12
    
    # Range
    np.min(data), np.max(data)  # 2, 10
    np.ptp(data)                # 8 (peak-to-peak)
    
    # Percentiles
    np.percentile(data, [25, 50, 75])  # [2.5, 4.0, 7.0]
    np.quantile(data, [0.25, 0.5, 0.75])
    ```
    
    **With NaN handling:**
    
    ```python
    data_nan = np.array([1, 2, np.nan, 4])
    
    np.mean(data_nan)     # nan
    np.nanmean(data_nan)  # 2.33 (ignores NaN)
    np.nanstd(data_nan)   # 1.25
    np.nanmax(data_nan)   # 4.0
    ```
    
    **Along Axis:**
    
    ```python
    matrix = np.array([[1, 2], [3, 4], [5, 6]])
    
    np.mean(matrix, axis=0)  # [3., 4.] - per column
    np.mean(matrix, axis=1)  # [1.5, 3.5, 5.5] - per row
    np.mean(matrix)          # 3.5 - overall
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Data analysis basics.
        
        **Strong answer signals:**
        
        - Uses nan-safe functions
        - Understands axis parameter
        - Knows percentile vs quantile

---

### How to Use Stride Tricks for Efficient Sliding Windows? - HFT, Google Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Internals`, `Performance` | **Asked by:** HFT Firms, Google, Amazon

??? success "View Answer"

    **Sliding Window Without Copying:**
    
    ```python
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view
    
    # New API (NumPy 1.20+)
    arr = np.arange(10)
    windows = sliding_window_view(arr, window_shape=3)
    # [[0, 1, 2],
    #  [1, 2, 3],
    #  [2, 3, 4], ...]
    
    # Rolling mean
    rolling_mean = windows.mean(axis=1)
    ```
    
    **Manual Stride Tricks:**
    
    ```python
    from numpy.lib.stride_tricks import as_strided
    
    def sliding_window_manual(arr, window_size):
        shape = (len(arr) - window_size + 1, window_size)
        strides = (arr.strides[0], arr.strides[0])
        return as_strided(arr, shape=shape, strides=strides)
    
    # 2D sliding windows (for images)
    def sliding_window_2d(arr, window_shape):
        h, w = arr.shape
        wh, ww = window_shape
        shape = (h - wh + 1, w - ww + 1, wh, ww)
        strides = arr.strides + arr.strides
        return as_strided(arr, shape=shape, strides=strides)
    ```
    
    **Caution:**
    
    - as_strided can create invalid memory access
    - Result is read-only view
    - Use sliding_window_view when possible

    !!! tip "Interviewer's Insight"
        **What they're testing:** Advanced memory optimization.
        
        **Strong answer signals:**
        
        - Uses sliding_window_view for new code
        - Understands stride meaning
        - Knows safety concerns

---

### How to Use np.vectorize()? When Should You Avoid It? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Vectorization`, `Performance` | **Asked by:** Google, Amazon

??? success "View Answer"

    **np.vectorize - Convenience, Not Performance:**
    
    ```python
    import numpy as np
    
    # Custom function
    def my_func(x):
        if x < 0:
            return 0
        elif x < 10:
            return x
        else:
            return 10
    
    # Vectorize it
    vectorized_func = np.vectorize(my_func)
    
    arr = np.array([-5, 3, 15, 7])
    result = vectorized_func(arr)  # [0, 3, 10, 7]
    ```
    
    **Important: vectorize is NOT faster than loops!**
    
    ```python
    # This is just as slow as a loop
    np.vectorize(lambda x: x**2)(arr)
    
    # This is actually fast (true vectorization)
    arr ** 2
    ```
    
    **When to Use:**
    
    | Use Case | Better Alternative |
    |----------|-------------------|
    | Complex branching | np.where, np.select |
    | Simple math | Native array ops |
    | External library | Consider Numba |
    
    **True Vectorization:**
    
    ```python
    # Instead of vectorize
    result = np.where(arr < 0, 0, np.where(arr < 10, arr, 10))
    result = np.clip(arr, 0, 10)  # Even simpler!
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Performance awareness.
        
        **Strong answer signals:**
        
        - Knows vectorize is for convenience only
        - Uses np.where/select for branching
        - Prefers native operations

---

### How to Concatenate and Stack Arrays? - Google, Amazon Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Array Manipulation` | **Asked by:** Google, Amazon, Microsoft

??? success "View Answer"

    **Concatenation:**
    
    ```python
    import numpy as np
    
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    
    # 1D concatenation
    np.concatenate([a, b])  # [1, 2, 3, 4, 5, 6]
    
    # 2D concatenation
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    
    np.concatenate([A, B], axis=0)  # Stack vertically (4, 2)
    np.concatenate([A, B], axis=1)  # Stack horizontally (2, 4)
    ```
    
    **Stacking:**
    
    ```python
    # vstack - vertical (axis=0)
    np.vstack([a, b])      # [[1,2,3], [4,5,6]]
    
    # hstack - horizontal (axis=1)
    np.hstack([A, B])      # [[1,2,5,6], [3,4,7,8]]
    
    # dstack - depth (axis=2)
    np.dstack([A, B])      # Shape: (2, 2, 2)
    
    # stack - new axis
    np.stack([a, b], axis=0)  # Shape: (2, 3)
    np.stack([a, b], axis=1)  # Shape: (3, 2)
    ```
    
    **Difference:**
    
    | Function | Behavior |
    |----------|----------|
    | concatenate | Join along existing axis |
    | stack | Create new axis |
    | vstack/hstack | Convenience for specific axis |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Array manipulation fluency.
        
        **Strong answer signals:**
        
        - Knows concatenate vs stack
        - Uses appropriate function for task
        - Understands axis parameter

---

### How to Use Boolean and Fancy Indexing? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Indexing`, `Selection` | **Asked by:** Most Tech Companies

??? success "View Answer"

    **Boolean Indexing:**
    
    ```python
    import numpy as np
    
    arr = np.array([1, 2, 3, 4, 5, 6])
    
    # Boolean mask
    mask = arr > 3
    print(mask)         # [False, False, False, True, True, True]
    print(arr[mask])    # [4, 5, 6]
    
    # Direct condition
    arr[arr > 3]        # [4, 5, 6]
    arr[arr % 2 == 0]   # [2, 4, 6]
    
    # Combined conditions
    arr[(arr > 2) & (arr < 5)]  # [3, 4]
    arr[(arr < 2) | (arr > 4)]  # [1, 5, 6]
    
    # Assignment
    arr[arr > 3] = 0    # [1, 2, 3, 0, 0, 0]
    ```
    
    **Fancy Indexing (Integer Arrays):**
    
    ```python
    arr = np.array([10, 20, 30, 40, 50])
    
    # Select specific indices
    indices = np.array([0, 2, 4])
    arr[indices]  # [10, 30, 50]
    
    # 2D fancy indexing
    matrix = np.arange(12).reshape(3, 4)
    rows = np.array([0, 2])
    cols = np.array([1, 3])
    matrix[rows, cols]  # [1, 11] - elements (0,1) and (2,3)
    ```
    
    **Important:**
    
    - Boolean indexing returns **copy**
    - Fancy indexing returns **copy**
    - Slicing returns **view**

    !!! tip "Interviewer's Insight"
        **What they're testing:** Flexible data selection.
        
        **Strong answer signals:**
        
        - Uses & | for combining conditions
        - Knows it returns copies
        - Uses for efficient filtering

---

### How to Perform SVD (Singular Value Decomposition)? - Google, Amazon, Netflix Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Linear Algebra`, `Dimensionality Reduction` | **Asked by:** Google, Amazon, Netflix

??? success "View Answer"

    **SVD in NumPy:**
    
    ```python
    import numpy as np
    
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    # Full SVD
    U, s, Vt = np.linalg.svd(A)
    
    # U: (m, m) orthogonal matrix
    # s: (min(m,n),) singular values (diagonal)
    # Vt: (n, n) orthogonal matrix
    
    # Reconstruct A
    S = np.zeros_like(A, dtype=float)
    np.fill_diagonal(S, s)
    A_reconstructed = U @ S @ Vt
    
    print(np.allclose(A, A_reconstructed))  # True
    ```
    
    **Truncated SVD (Low-Rank Approximation):**
    
    ```python
    # Keep top k components
    k = 2
    U_k = U[:, :k]
    s_k = s[:k]
    Vt_k = Vt[:k, :]
    
    A_approx = U_k @ np.diag(s_k) @ Vt_k
    
    # Compression ratio
    original = A.shape[0] * A.shape[1]
    compressed = k * (A.shape[0] + A.shape[1] + 1)
    ```
    
    **Applications:**
    
    - Image compression
    - Recommendation systems
    - Noise reduction
    - Latent semantic analysis

    !!! tip "Interviewer's Insight"
        **What they're testing:** Advanced linear algebra applications.
        
        **Strong answer signals:**
        
        - Explains U, s, Vt meaning
        - Knows truncated SVD use case
        - Mentions applications

---

### How to Use np.argsort() and np.argmax()? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Sorting`, `Indexing` | **Asked by:** Most Tech Companies

??? success "View Answer"

    **argsort - Indices That Would Sort:**
    
    ```python
    import numpy as np
    
    arr = np.array([30, 10, 50, 20, 40])
    
    # Get sorted indices
    sorted_idx = np.argsort(arr)
    print(sorted_idx)  # [1, 3, 0, 4, 2]
    
    # Use to sort
    arr[sorted_idx]    # [10, 20, 30, 40, 50]
    
    # Descending order
    arr[np.argsort(arr)[::-1]]  # [50, 40, 30, 20, 10]
    
    # Top k elements
    k = 3
    top_k_idx = np.argsort(arr)[-k:][::-1]
    arr[top_k_idx]  # [50, 40, 30]
    ```
    
    **argmax/argmin:**
    
    ```python
    arr = np.array([30, 10, 50, 20, 40])
    
    np.argmax(arr)  # 2 (index of 50)
    np.argmin(arr)  # 1 (index of 10)
    
    # 2D array
    matrix = np.array([[1, 5, 3], [8, 2, 7]])
    np.argmax(matrix)          # 3 (flat index of 8)
    np.argmax(matrix, axis=0)  # [1, 0, 1] (per column)
    np.argmax(matrix, axis=1)  # [1, 0] (per row)
    
    # Convert flat index to 2D
    flat_idx = np.argmax(matrix)
    np.unravel_index(flat_idx, matrix.shape)  # (1, 0)
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Index-based operations.
        
        **Strong answer signals:**
        
        - Uses argsort for ranking
        - Gets top-k efficiently
        - Knows unravel_index for nd

---

### How to Use Memory Mapping for Large Files? - Google, Amazon, Netflix Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Big Data`, `I/O` | **Asked by:** Google, Amazon, Netflix

??? success "View Answer"

    **np.memmap - Memory-Mapped Files:**
    
    ```python
    import numpy as np
    
    # Create memory-mapped file
    shape = (10000, 10000)
    dtype = np.float32
    
    # Write mode
    mmap_write = np.memmap('large_data.dat', dtype=dtype, 
                           mode='w+', shape=shape)
    mmap_write[:] = np.random.rand(*shape)
    del mmap_write  # Flush to disk
    
    # Read mode
    mmap_read = np.memmap('large_data.dat', dtype=dtype, 
                          mode='r', shape=shape)
    
    # Access without loading entire file
    subset = mmap_read[1000:2000, 1000:2000]
    mean_value = mmap_read.mean()  # Streams through file
    ```
    
    **Benefits:**
    
    | Feature | Regular Array | Memory Map |
    |---------|---------------|------------|
    | RAM usage | Full size | Only accessed pages |
    | Startup | Load all | Instant |
    | Sharing | Copy per process | Shared |
    
    **Use Cases:**
    
    - Files larger than RAM
    - Multi-process data sharing
    - Random access to large datasets

    !!! tip "Interviewer's Insight"
        **What they're testing:** Large data handling.
        
        **Strong answer signals:**
        
        - Knows when to use memmap
        - Understands virtual memory
        - Uses for out-of-core computing

---

### How to Perform FFT (Fast Fourier Transform)? - Google, Amazon, HFT Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Signal Processing` | **Asked by:** Google, Amazon, HFT Firms

??? success "View Answer"

    **FFT in NumPy:**
    
    ```python
    import numpy as np
    
    # Time domain signal
    t = np.linspace(0, 1, 1000)
    freq1, freq2 = 5, 50  # Hz
    signal = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)
    
    # FFT
    fft_result = np.fft.fft(signal)
    
    # Frequency axis
    n = len(signal)
    freq = np.fft.fftfreq(n, d=t[1]-t[0])
    
    # Power spectrum
    power = np.abs(fft_result) ** 2
    
    # Only positive frequencies
    positive_freq = freq[:n//2]
    positive_power = power[:n//2]
    ```
    
    **2D FFT (Images):**
    
    ```python
    image = np.random.rand(256, 256)
    
    fft_2d = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft_2d)  # Center low frequencies
    
    # Inverse FFT
    reconstructed = np.fft.ifft2(fft_2d)
    ```
    
    **rfft for Real Data:**
    
    ```python
    # More efficient for real input
    rfft_result = np.fft.rfft(signal)  # Only positive frequencies
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Signal processing knowledge.
        
        **Strong answer signals:**
        
        - Uses rfft for real data
        - Knows fftfreq for frequency axis
        - Can interpret power spectrum

---

### How to Use Structured Arrays? - Google, HFT Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Advanced Data Types` | **Asked by:** Google, Amazon, HFT Firms

??? success "View Answer"

    **Structured Arrays - Mixed Types:**
    
    ```python
    import numpy as np
    
    # Define dtype
    dt = np.dtype([
        ('name', 'U10'),        # Unicode string, 10 chars
        ('age', 'i4'),          # 32-bit integer
        ('weight', 'f8'),       # 64-bit float
        ('active', 'bool')
    ])
    
    # Create array
    data = np.array([
        ('Alice', 25, 65.5, True),
        ('Bob', 30, 80.0, False),
        ('Charlie', 35, 75.2, True)
    ], dtype=dt)
    
    # Access by field name
    print(data['name'])    # ['Alice', 'Bob', 'Charlie']
    print(data['age'])     # [25, 30, 35]
    
    # Filter
    active = data[data['active']]
    
    # Modify
    data['age'] += 1
    ```
    
    **Record Arrays (easier access):**
    
    ```python
    rec = data.view(np.recarray)
    print(rec.name)  # Attribute access
    print(rec.age)
    ```
    
    **Use Cases:**
    
    - Tabular data without Pandas overhead
    - Memory-mapped complex structures
    - Binary file formats

    !!! tip "Interviewer's Insight"
        **What they're testing:** Advanced NumPy knowledge.
        
        **Strong answer signals:**
        
        - Knows when to use vs Pandas
        - Defines custom dtypes
        - Uses for binary I/O

---

### How to Compute Norms and Distances? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Linear Algebra`, `Metrics` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Vector Norms:**
    
    ```python
    import numpy as np
    
    v = np.array([3, 4])
    
    # L2 norm (Euclidean)
    np.linalg.norm(v)        # 5.0
    np.linalg.norm(v, ord=2)  # Same
    
    # L1 norm (Manhattan)
    np.linalg.norm(v, ord=1)  # 7.0
    
    # L∞ norm (Max)
    np.linalg.norm(v, ord=np.inf)  # 4.0
    ```
    
    **Pairwise Distances:**
    
    ```python
    A = np.array([[0, 0], [1, 1], [2, 2]])
    B = np.array([[0, 1], [1, 0]])
    
    # Broadcasting approach
    diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=-1))
    
    # Using scipy (faster for large arrays)
    from scipy.spatial.distance import cdist
    distances = cdist(A, B, metric='euclidean')
    ```
    
    **Matrix Norms:**
    
    ```python
    M = np.array([[1, 2], [3, 4]])
    
    np.linalg.norm(M, 'fro')  # Frobenius norm
    np.linalg.norm(M, 2)      # Spectral norm (largest singular value)
    np.linalg.norm(M, 1)      # Max column sum
    np.linalg.norm(M, np.inf) # Max row sum
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Distance calculations.
        
        **Strong answer signals:**
        
        - Uses broadcasting for pairwise
        - Knows different norm types
        - Uses scipy.cdist for scale

---

### How to Reshape Arrays Efficiently? - Google, Amazon Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Array Manipulation` | **Asked by:** Most Tech Companies

??? success "View Answer"

    **reshape() Basics:**
    
    ```python
    import numpy as np
    
    arr = np.arange(12)
    
    # Reshape to 2D
    arr.reshape(3, 4)   # 3 rows, 4 cols
    arr.reshape(4, -1)  # 4 rows, auto-calculate cols
    arr.reshape(-1, 6)  # auto rows, 6 cols
    
    # Reshape vs resize
    reshaped = arr.reshape(3, 4)  # Returns view if possible
    arr.resize(3, 4)              # Modifies in place
    
    # newaxis for dimension expansion
    arr = np.array([1, 2, 3])
    arr[:, np.newaxis]  # Column vector (3, 1)
    arr[np.newaxis, :]  # Row vector (1, 3)
    ```
    
    **Common Patterns:**
    
    ```python
    # Flatten to 1D
    arr.reshape(-1)
    arr.ravel()
    
    # Add batch dimension
    arr[np.newaxis, ...]  # (1, ...)
    
    # Channel-first to channel-last
    img = np.random.rand(3, 224, 224)  # C, H, W
    img.transpose(1, 2, 0)              # H, W, C
    ```

    !!! tip "Interviewer's Insight"
        - Uses -1 for auto-dimension
        - Knows reshape returns view when possible
        - Uses newaxis for broadcasting

---

### How to Use np.partition() for Partial Sorting? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Sorting`, `Algorithms` | **Asked by:** Google, Amazon

??? success "View Answer"

    **partition - Faster Than Full Sort:**
    
    ```python
    import numpy as np
    
    arr = np.array([30, 10, 50, 20, 40])
    
    # Partition: elements < kth are left, > kth are right
    k = 2
    np.partition(arr, k)
    # [10, 20, 30, 50, 40] - 30 is in correct position
    
    # Get k smallest elements (unordered)
    k_smallest = np.partition(arr, k)[:k]  # [10, 20]
    
    # Get k largest
    k_largest = np.partition(arr, -k)[-k:]  # [40, 50]
    
    # argpartition - get indices
    idx = np.argpartition(arr, k)
    arr[idx[:k]]  # k smallest values
    ```
    
    **Complexity:**
    
    | Operation | Time |
    |-----------|------|
    | sort | O(n log n) |
    | partition | O(n) |
    
    **Use Case:**
    
    - Top-k elements without full sort
    - Median finding
    - Quick selection

    !!! tip "Interviewer's Insight"
        - Uses partition for efficiency
        - Knows O(n) vs O(n log n)
        - Uses for top-k problems

---

### How to Use np.searchsorted() for Binary Search? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Algorithms`, `Searching` | **Asked by:** Google, Amazon, HFT Firms

??? success "View Answer"

    **searchsorted - Binary Search:**
    
    ```python
    import numpy as np
    
    # Sorted array
    arr = np.array([1, 3, 5, 7, 9])
    
    # Find insertion point
    np.searchsorted(arr, 4)   # 2 (insert before 5)
    np.searchsorted(arr, 5)   # 2 (insert before existing 5)
    
    # side='right' - insert after equal values
    np.searchsorted(arr, 5, side='right')  # 3
    
    # Multiple search values
    np.searchsorted(arr, [2, 6, 8])  # [1, 3, 4]
    
    # Binning data
    bins = np.array([0, 10, 20, 30, 100])
    values = np.array([5, 15, 25, 50])
    bin_indices = np.searchsorted(bins, values) - 1
    ```
    
    **Applications:**
    
    - Histogram binning
    - Finding nearest neighbor
    - Merging sorted arrays

    !!! tip "Interviewer's Insight"
        - Uses for O(log n) lookup
        - Knows side parameter
        - Uses for binning operations

---

### How to Use np.clip() for Bounding Values? - Google, Amazon Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Array Manipulation` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **clip - Limit Value Range:**
    
    ```python
    import numpy as np
    
    arr = np.array([-5, 0, 5, 10, 15])
    
    # Clip to range [0, 10]
    np.clip(arr, 0, 10)  # [0, 0, 5, 10, 10]
    
    # Only lower bound
    np.clip(arr, 0, None)  # [0, 0, 5, 10, 15]
    
    # Only upper bound
    np.clip(arr, None, 10)  # [-5, 0, 5, 10, 10]
    
    # In-place clipping
    np.clip(arr, 0, 10, out=arr)
    ```
    
    **ML Use Cases:**
    
    ```python
    # Gradient clipping
    gradients = np.clip(gradients, -1.0, 1.0)
    
    # Probability bounds
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    
    # Image pixel normalization
    pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    ```

    !!! tip "Interviewer's Insight"
        - Uses for gradient clipping
        - Uses for outlier handling
        - Knows in-place option

---

### How to Use np.roll() for Circular Shifting? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Array Manipulation` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **roll - Circular Shift:**
    
    ```python
    import numpy as np
    
    arr = np.array([1, 2, 3, 4, 5])
    
    # Shift right by 2
    np.roll(arr, 2)   # [4, 5, 1, 2, 3]
    
    # Shift left by 2
    np.roll(arr, -2)  # [3, 4, 5, 1, 2]
    
    # 2D roll
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    np.roll(matrix, 1, axis=0)  # Shift rows down
    np.roll(matrix, 1, axis=1)  # Shift columns right
    ```
    
    **Applications:**
    
    ```python
    # Lag features for time series
    data = np.array([10, 20, 30, 40, 50])
    lag_1 = np.roll(data, 1)
    lag_1[0] = np.nan  # Handle boundary
    
    # Circular convolution
    # Ring buffer operations
    ```

    !!! tip "Interviewer's Insight"
        - Uses for time series lag
        - Knows axis parameter for 2D
        - Handles boundary conditions

---

### How to Use np.meshgrid()? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Geometry`, `Plotting` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **meshgrid - Coordinate Grids:**
    
    ```python
    import numpy as np
    
    x = np.array([1, 2, 3])
    y = np.array([10, 20])
    
    X, Y = np.meshgrid(x, y)
    # X = [[1, 2, 3],
    #      [1, 2, 3]]
    # Y = [[10, 10, 10],
    #      [20, 20, 20]]
    
    # Evaluate function at all points
    Z = X ** 2 + Y ** 2
    
    # Create coordinate pairs
    coords = np.stack([X.ravel(), Y.ravel()], axis=1)
    # [[1, 10], [2, 10], [3, 10], [1, 20], [2, 20], [3, 20]]
    ```
    
    **Use Cases:**
    
    ```python
    # Image pixel coordinates
    height, width = 100, 200
    y, x = np.meshgrid(range(height), range(width), indexing='ij')
    
    # Distance from center
    cx, cy = width // 2, height // 2
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    ```

    !!! tip "Interviewer's Insight"
        - Uses for grid evaluations
        - Knows indexing='ij' for matrix indexing
        - Uses for coordinate generation

---

### How to Perform Cumulative Operations? - Google, Amazon Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Statistics` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Cumulative Functions:**
    
    ```python
    import numpy as np
    
    arr = np.array([1, 2, 3, 4, 5])
    
    # Cumulative sum
    np.cumsum(arr)     # [1, 3, 6, 10, 15]
    
    # Cumulative product
    np.cumprod(arr)    # [1, 2, 6, 24, 120]
    
    # Cumulative min/max
    np.minimum.accumulate(arr)  # Running minimum
    np.maximum.accumulate(arr)  # Running maximum
    
    # 2D cumulative
    matrix = np.array([[1, 2], [3, 4]])
    np.cumsum(matrix, axis=0)  # Along rows
    np.cumsum(matrix, axis=1)  # Along columns
    ```
    
    **Financial Applications:**
    
    ```python
    # Cumulative returns
    returns = np.array([0.01, -0.02, 0.03, 0.01])
    cumulative = np.cumprod(1 + returns) - 1
    
    # Drawdown calculation
    prices = np.array([100, 110, 105, 115, 108])
    running_max = np.maximum.accumulate(prices)
    drawdown = (prices - running_max) / running_max
    ```

    !!! tip "Interviewer's Insight"
        - Uses for running calculations
        - Knows ufunc.accumulate pattern
        - Applies to financial metrics

---

### How to Handle NaN Values Efficiently? - Google, Amazon, Netflix Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Missing Data` | **Asked by:** Google, Amazon, Netflix

??? success "View Answer"

    **NaN-Safe Functions:**
    
    ```python
    import numpy as np
    
    arr = np.array([1, 2, np.nan, 4, np.nan, 6])
    
    # Detection
    np.isnan(arr)      # [False, False, True, False, True, False]
    np.isnan(arr).sum()  # 2 NaN values
    
    # Ignore NaN in calculations
    np.nanmean(arr)    # 3.25
    np.nanstd(arr)     # 1.92
    np.nanmax(arr)     # 6.0
    np.nansum(arr)     # 13.0
    
    # Replace NaN
    np.nan_to_num(arr, nan=0)              # [1, 2, 0, 4, 0, 6]
    np.where(np.isnan(arr), 0, arr)        # Same
    arr[np.isnan(arr)] = np.nanmean(arr)   # Fill with mean
    ```
    
    **Working with Infinity:**
    
    ```python
    arr = np.array([1, np.inf, -np.inf, np.nan])
    
    np.isinf(arr)      # [False, True, True, False]
    np.isfinite(arr)   # [True, False, False, False]
    
    np.nan_to_num(arr, nan=0, posinf=999, neginf=-999)
    ```

    !!! tip "Interviewer's Insight"
        - Uses nan-prefixed functions
        - Knows difference from regular functions
        - Handles inf separately

---

### How to Perform Set Operations? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Set Operations` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **NumPy Set Operations:**
    
    ```python
    import numpy as np
    
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([3, 4, 5, 6, 7])
    
    # Intersection
    np.intersect1d(a, b)  # [3, 4, 5]
    
    # Union
    np.union1d(a, b)      # [1, 2, 3, 4, 5, 6, 7]
    
    # Difference (in a but not b)
    np.setdiff1d(a, b)    # [1, 2]
    
    # Symmetric difference
    np.setxor1d(a, b)     # [1, 2, 6, 7]
    
    # Membership test
    np.isin(a, [2, 4])    # [False, True, False, True, False]
    ```
    
    **Unique Values:**
    
    ```python
    arr = np.array([1, 2, 2, 3, 3, 3])
    
    np.unique(arr)                    # [1, 2, 3]
    values, counts = np.unique(arr, return_counts=True)
    values, indices = np.unique(arr, return_index=True)
    values, inverse = np.unique(arr, return_inverse=True)
    ```

    !!! tip "Interviewer's Insight"
        - Uses for array comparisons
        - Knows unique with return options
        - Uses isin for filtering

---

### How to Use np.apply_along_axis()? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Iteration` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **apply_along_axis - Apply Function to Slices:**
    
    ```python
    import numpy as np
    
    def custom_normalize(x):
        return (x - x.min()) / (x.max() - x.min())
    
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    
    # Apply along columns (axis=0)
    np.apply_along_axis(custom_normalize, 0, matrix)
    
    # Apply along rows (axis=1)
    np.apply_along_axis(custom_normalize, 1, matrix)
    ```
    
    **When to Use:**
    
    ```python
    # Custom function not available as ufunc
    def top_3_mean(x):
        return np.mean(np.sort(x)[-3:])
    
    data = np.random.rand(100, 10)
    result = np.apply_along_axis(top_3_mean, 1, data)
    ```
    
    **Performance Note:**
    
    ```python
    # Prefer vectorized operations when possible
    # SLOW
    np.apply_along_axis(np.sum, 1, matrix)
    
    # FAST
    np.sum(matrix, axis=1)
    ```

    !!! tip "Interviewer's Insight"
        - Uses for custom functions only
        - Prefers built-in axis parameter
        - Knows performance implications

---

### How to Work with Complex Numbers? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Data Types` | **Asked by:** Google, Amazon, HFT Firms

??? success "View Answer"

    **Complex Number Operations:**
    
    ```python
    import numpy as np
    
    # Create complex array
    z = np.array([1+2j, 3+4j, 5+6j])
    
    # Real and imaginary parts
    z.real  # [1., 3., 5.]
    z.imag  # [2., 4., 6.]
    
    # Magnitude and phase
    np.abs(z)     # [2.24, 5., 7.81]
    np.angle(z)   # Phase in radians
    
    # Conjugate
    np.conj(z)    # [1-2j, 3-4j, 5-6j]
    
    # Create from magnitude and phase
    magnitude = np.array([1, 2, 3])
    phase = np.array([0, np.pi/4, np.pi/2])
    z = magnitude * np.exp(1j * phase)
    ```
    
    **FFT Applications:**
    
    ```python
    signal = np.random.rand(100)
    fft = np.fft.fft(signal)
    
    # Power spectrum
    power = np.abs(fft) ** 2
    
    # Phase spectrum
    phase = np.angle(fft)
    ```

    !!! tip "Interviewer's Insight"
        - Accesses .real and .imag attributes
        - Uses abs for magnitude
        - Applies to signal processing

---

### How to Use np.polynomial for Curve Fitting? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Curve Fitting` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Polynomial Fitting:**
    
    ```python
    import numpy as np
    
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([1, 3, 7, 13, 21])  # y ≈ x² + x + 1
    
    # Fit polynomial of degree 2
    coeffs = np.polyfit(x, y, deg=2)  # [1., 1., 1.]
    
    # Evaluate polynomial
    x_new = np.linspace(0, 4, 100)
    y_new = np.polyval(coeffs, x_new)
    
    # Modern API (NumPy 1.4+)
    from numpy.polynomial import polynomial as P
    
    coeffs = P.polyfit(x, y, deg=2)
    y_new = P.polyval(x_new, coeffs)
    ```
    
    **Polynomial Objects:**
    
    ```python
    # Create polynomial: 2x² + 3x + 1
    p = np.poly1d([2, 3, 1])
    
    p(2)       # Evaluate at x=2: 15
    p.roots    # Find roots
    p.deriv()  # Derivative: 4x + 3
    p.integ()  # Integral
    ```

    !!! tip "Interviewer's Insight"
        - Uses polyfit for fitting
        - Knows modern polynomial API
        - Can find roots and derivatives

---

### How to Save and Load NumPy Arrays? - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `File I/O` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Save/Load Methods:**
    
    ```python
    import numpy as np
    
    arr = np.random.rand(100, 100)
    
    # Single array (.npy)
    np.save('array.npy', arr)
    loaded = np.load('array.npy')
    
    # Multiple arrays (.npz)
    np.savez('arrays.npz', x=arr, y=arr*2)
    data = np.load('arrays.npz')
    print(data['x'], data['y'])
    
    # Compressed (.npz)
    np.savez_compressed('arrays_compressed.npz', arr=arr)
    
    # Text files
    np.savetxt('array.txt', arr, delimiter=',')
    loaded = np.loadtxt('array.txt', delimiter=',')
    ```
    
    **Format Comparison:**
    
    | Format | Speed | Size | Human Readable |
    |--------|-------|------|----------------|
    | .npy | Fast | Small | No |
    | .npz | Fast | Small | No |
    | .txt | Slow | Large | Yes |
    | .csv | Slow | Large | Yes |

    !!! tip "Interviewer's Insight"
        - Uses .npy for single arrays
        - Uses .npz for multiple
        - Knows compression options

---

### How to Use np.allclose() for Float Comparisons? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Logic`, `Testing` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Float Comparison:**
    
    ```python
    import numpy as np
    
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.00001, 2.00001, 3.00001])
    
    # Direct comparison often fails
    a == b  # [False, False, False]
    
    # Use allclose
    np.allclose(a, b)  # True (within tolerance)
    np.allclose(a, b, rtol=1e-7)  # False (stricter)
    
    # Element-wise close
    np.isclose(a, b)  # [True, True, True]
    
    # Parameters
    # rtol: relative tolerance
    # atol: absolute tolerance
    # |a - b| <= atol + rtol * |b|
    ```
    
    **Testing Applications:**
    
    ```python
    # Unit test assertions
    def test_inverse():
        A = np.random.rand(3, 3)
        A_inv = np.linalg.inv(A)
        assert np.allclose(A @ A_inv, np.eye(3))
    
    # Numerical algorithm validation
    result_v1 = algorithm_v1(data)
    result_v2 = algorithm_v2(data)
    assert np.allclose(result_v1, result_v2)
    ```

    !!! tip "Interviewer's Insight"
        - Uses for float comparisons
        - Knows tolerance parameters
        - Uses in testing

---

### How to Use np.select() for Multiple Conditions? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Advanced Logic` | **Asked by:** Google, Amazon

??? success "View Answer"

    **np.select - Multi-Condition Selection:**
    
    ```python
    import numpy as np
    
    arr = np.array([1, 5, 15, 25, 35])
    
    # Multiple conditions
    conditions = [
        arr < 10,
        arr < 20,
        arr < 30
    ]
    choices = ['small', 'medium', 'large']
    
    result = np.select(conditions, choices, default='xlarge')
    # ['small', 'small', 'medium', 'large', 'xlarge']
    ```
    
    **Numeric Operations:**
    
    ```python
    x = np.array([-2, -1, 0, 1, 2])
    
    conditions = [x < 0, x == 0, x > 0]
    choices = [x ** 2, 0, x ** 3]  # Different formula per condition
    
    result = np.select(conditions, choices)
    # [4, 1, 0, 1, 8]
    ```
    
    **vs np.where Nesting:**
    
    ```python
    # Cleaner than nested where
    # Instead of:
    np.where(arr < 10, 'small',
             np.where(arr < 20, 'medium',
                      np.where(arr < 30, 'large', 'xlarge')))
    
    # Use np.select
    ```

    !!! tip "Interviewer's Insight"
        - Uses for cleaner multi-condition
        - Applies different operations per condition
        - Prefers over nested where

---

### How to Create Custom ufuncs? - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Extending NumPy` | **Asked by:** Google, Amazon, Research

??? success "View Answer"

    **Custom Universal Functions:**
    
    ```python
    import numpy as np
    
    # Using np.frompyfunc (slow but flexible)
    def custom_op(x, y):
        if x > y:
            return x - y
        return x + y
    
    custom_ufunc = np.frompyfunc(custom_op, 2, 1)
    result = custom_ufunc(np.array([1, 5, 3]), np.array([2, 3, 4]))
    
    # Using np.vectorize with signature
    @np.vectorize
    def custom_func(x, y):
        return x ** 2 + y ** 2
    ```
    
    **For Performance: Use Numba:**
    
    ```python
    from numba import vectorize
    
    @vectorize
    def fast_custom(x, y):
        return x ** 2 + y ** 2
    
    result = fast_custom(a, b)  # Truly fast
    ```
    
    **Create from Existing ufunc:**
    
    ```python
    # Reduce operations
    np.add.reduce([1, 2, 3, 4])     # Sum: 10
    np.multiply.reduce([1, 2, 3, 4]) # Product: 24
    
    # Outer product
    np.add.outer([1, 2], [10, 20, 30])
    ```

    !!! tip "Interviewer's Insight"
        - Uses Numba for real performance
        - Knows frompyfunc limitations
        - Uses ufunc methods (reduce, outer)

---

### How to Use np.diff() for Discrete Differences? - Google, Amazon, HFT Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Time Series` | **Asked by:** Google, Amazon, HFT Firms

??? success "View Answer"

    **Discrete Difference:**
    
    ```python
    import numpy as np
    
    arr = np.array([1, 3, 6, 10, 15])
    
    # First difference
    np.diff(arr)     # [2, 3, 4, 5]
    
    # Second difference
    np.diff(arr, n=2)  # [1, 1, 1]
    
    # Along axis
    matrix = np.array([[1, 2, 4], [3, 5, 8]])
    np.diff(matrix, axis=0)  # Difference between rows
    np.diff(matrix, axis=1)  # Difference between columns
    ```
    
    **Financial Applications:**
    
    ```python
    prices = np.array([100, 102, 101, 105, 103])
    
    # Price changes
    changes = np.diff(prices)  # [2, -1, 4, -2]
    
    # Returns
    returns = np.diff(prices) / prices[:-1]
    
    # Log returns
    log_returns = np.diff(np.log(prices))
    ```

    !!! tip "Interviewer's Insight"
        - Uses for time series analysis
        - Knows n parameter for higher order
        - Applies to financial calculations

---

### How to Use np.convolve()? - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Signal Processing` | **Asked by:** Google, Amazon, CV Companies

??? success "View Answer"

    **Convolution:**
    
    ```python
    import numpy as np
    
    signal = np.array([1, 2, 3, 4, 5])
    kernel = np.array([1, 0, -1])
    
    # Full convolution
    np.convolve(signal, kernel, mode='full')    # Length: n + m - 1
    
    # Same size as input
    np.convolve(signal, kernel, mode='same')    # Length: max(n, m)
    
    # Only complete overlap
    np.convolve(signal, kernel, mode='valid')   # Length: max(n,m) - min(n,m) + 1
    ```
    
    **Applications:**
    
    ```python
    # Moving average
    window = np.ones(5) / 5
    smoothed = np.convolve(signal, window, mode='valid')
    
    # Edge detection (discrete derivative)
    edge_kernel = np.array([1, -1])
    edges = np.convolve(signal, edge_kernel, mode='same')
    ```
    
    **2D Convolution (scipy):**
    
    ```python
    from scipy.signal import convolve2d
    
    image = np.random.rand(100, 100)
    kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])  # Sobel
    edges = convolve2d(image, kernel, mode='same')
    ```

    !!! tip "Interviewer's Insight"
        - Knows mode parameter effects
        - Uses for signal smoothing
        - Uses scipy for 2D images

---

### How to Use np.histogram()? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Statistics` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Histogram Calculation:**
    
    ```python
    import numpy as np
    
    data = np.random.randn(1000)
    
    # Basic histogram
    counts, bin_edges = np.histogram(data, bins=10)
    
    # Custom bins
    counts, bin_edges = np.histogram(data, bins=[-3, -1, 0, 1, 3])
    
    # With density (normalized)
    density, bin_edges = np.histogram(data, bins=50, density=True)
    
    # Bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ```
    
    **2D Histogram:**
    
    ```python
    x = np.random.randn(1000)
    y = np.random.randn(1000)
    
    hist2d, xedges, yedges = np.histogram2d(x, y, bins=20)
    
    # N-dimensional
    data = np.random.randn(1000, 3)
    histnd, edges = np.histogramdd(data, bins=10)
    ```

    !!! tip "Interviewer's Insight"
        - Uses for distribution analysis
        - Knows density parameter
        - Uses histogram2d for correlations

---

### How to Use np.digitize() for Binning? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Discretization` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **digitize - Assign to Bins:**
    
    ```python
    import numpy as np
    
    data = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    bins = np.array([1, 2, 3, 4])
    
    # Get bin indices
    indices = np.digitize(data, bins)
    # [0, 1, 2, 3, 4]
    # 0.5 < 1, so bin 0
    # 1 <= 1.5 < 2, so bin 1
    
    # Right=True: bins are closed on right
    np.digitize(data, bins, right=True)
    ```
    
    **Practical Example:**
    
    ```python
    ages = np.array([5, 15, 25, 35, 45, 55, 65])
    age_bins = [0, 18, 30, 50, 100]
    labels = ['child', 'young', 'adult', 'senior']
    
    bin_idx = np.digitize(ages, age_bins) - 1  # 0-indexed
    categories = np.array(labels)[bin_idx]
    ```

    !!! tip "Interviewer's Insight"
        - Uses for categorical encoding
        - Knows right parameter
        - Uses with labels for categorization

---

### How to use np.einsum for complex operations? - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Advanced`, `Linear Algebra` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **einsum = Einstein summation notation for tensor operations**
    
    ```python
    import numpy as np
    
    # Matrix multiplication
    C = np.einsum('ij,jk->ik', A, B)
    
    # Batch matrix multiplication
    C = np.einsum('bij,bjk->bik', batch_A, batch_B)
    
    # Trace
    trace = np.einsum('ii->', A)
    
    # Outer product
    outer = np.einsum('i,j->ij', a, b)
    
    # Element-wise multiply and sum
    dot = np.einsum('i,i->', a, b)
    ```
    
    **Benefit:** Single function for many tensor operations.

    !!! tip "Interviewer's Insight"
        Uses einsum for complex tensor operations efficiently.

---

### How to use np.lib.stride_tricks? - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Advanced`, `Memory` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **Create views with custom strides (zero-copy)**
    
    ```python
    from numpy.lib.stride_tricks import sliding_window_view
    
    # Sliding windows (efficient)
    arr = np.array([1, 2, 3, 4, 5, 6])
    windows = sliding_window_view(arr, window_shape=3)
    # array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
    
    # Use for rolling calculations, convolutions
    ```
    
    **Caution:** Views share memory; modifications affect original.

    !!! tip "Interviewer's Insight"
        Uses sliding_window_view for memory-efficient operations.

---

### How to use np.frompyfunc for custom ufuncs? - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Advanced` | **Asked by:** Google, Amazon

??? success "View Answer"

    ```python
    import numpy as np
    
    # Create custom ufunc from Python function
    def custom_func(x, y):
        return x ** 2 + y
    
    ufunc = np.frompyfunc(custom_func, nin=2, nout=1)
    result = ufunc(arr1, arr2)  # Broadcasts automatically
    
    # For better performance, use np.vectorize with signature
    vfunc = np.vectorize(custom_func, otypes=[float])
    ```
    
    **Note:** Still slower than true ufuncs (no C-level optimization).

    !!! tip "Interviewer's Insight"
        Knows when to use numba.vectorize for real performance.

---

### How to use np.select for multiple conditions? - Most Tech Companies Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Conditional` | **Asked by:** Most Tech Companies

??? success "View Answer"

    ```python
    import numpy as np
    
    grades = np.array([92, 78, 65, 45, 88])
    
    conditions = [
        grades >= 90,
        grades >= 80,
        grades >= 70,
        grades >= 60
    ]
    choices = ['A', 'B', 'C', 'D']
    
    result = np.select(conditions, choices, default='F')
    # ['A', 'C', 'F', 'F', 'B']
    ```
    
    More readable than nested np.where.

    !!! tip "Interviewer's Insight"
        Uses np.select for cleaner multi-condition logic.

---

### How to use np.piecewise? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Functions` | **Asked by:** Google, Amazon

??? success "View Answer"

    ```python
    import numpy as np
    
    x = np.linspace(-2, 2, 100)
    
    # Define piecewise function
    y = np.piecewise(x,
        [x < 0, x >= 0],
        [lambda x: x**2, lambda x: x + 1]
    )
    # x<0: x^2, x>=0: x+1
    ```
    
    Useful for implementing mathematical functions with different formulas.

    !!! tip "Interviewer's Insight"
        Uses for piecewise mathematical functions.

---

### How to optimize memory with dtype selection? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Memory`, `Performance` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    ```python
    import numpy as np
    
    # Memory comparison
    arr_float64 = np.zeros(1_000_000, dtype=np.float64)  # 8 MB
    arr_float32 = np.zeros(1_000_000, dtype=np.float32)  # 4 MB
    arr_float16 = np.zeros(1_000_000, dtype=np.float16)  # 2 MB
    
    # Integer optimization
    small_ints = np.array([1, 2, 3], dtype=np.int8)  # 1 byte each
    
    # Downcast carefully
    arr = arr.astype(np.float32, copy=False)  # In-place when possible
    ```

    !!! tip "Interviewer's Insight"
        Chooses smallest dtype that maintains precision.

---

### How to implement rolling/sliding window operations? - Google, Amazon, Quantitative Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Windowing`, `Time Series`, `Performance` | **Asked by:** Google, Amazon, Citadel, Two Sigma

??? success "View Answer"

    **Using stride tricks (fastest):**

    ```python
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view
    
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    # Modern NumPy (1.20+)
    windows = sliding_window_view(data, window_shape=3)
    # [[1, 2, 3],
    #  [2, 3, 4],
    #  [3, 4, 5], ...]]
    
    rolling_mean = windows.mean(axis=1)
    rolling_max = windows.max(axis=1)
    
    # Old method with as_strided
    from numpy.lib.stride_tricks import as_strided
    
    def rolling_window(a, window):
        shape = (a.size - window + 1, window)
        strides = (a.strides[0], a.strides[0])
        return as_strided(a, shape=shape, strides=strides)
    ```

    **2D (image) sliding windows:**

    ```python
    # For convolution/pooling operations
    image = np.random.rand(10, 10)
    patches = sliding_window_view(image, window_shape=(3, 3))
    # Shape: (8, 8, 3, 3) - each position has 3x3 patch
    ```

    !!! tip "Interviewer's Insight"
        Uses stride tricks for O(1) memory. Knows sliding_window_view (NumPy 1.20+). Critical for financial/signal processing roles.

---

### Explain np.argpartition() vs np.argsort() - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Algorithms`, `Sorting`, `Performance` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Key Difference:**

    | Method | Time | Use Case |
    |--------|------|----------|
    | `argsort()` | O(n log n) | Need fully sorted |
    | `argpartition()` | O(n) | Need top/bottom k |

    ```python
    arr = np.array([7, 2, 9, 1, 5, 8])
    
    # Full sort - O(n log n)
    sorted_idx = np.argsort(arr)
    # [3, 1, 4, 0, 5, 2] - fully sorted indices
    
    # Partition - O(n) - much faster!
    k = 3  # Find 3 smallest
    partitioned_idx = np.argpartition(arr, k)
    # [3, 1, 4, 0, 5, 2] or similar
    # Guarantee: first k are smallest (not sorted among themselves)
    
    top_k_idx = partitioned_idx[:k]
    top_k_values = arr[top_k_idx]
    
    # For sorted top k:
    top_k_sorted = top_k_idx[np.argsort(arr[top_k_idx])]
    ```

    **Real interview question (Google):**

    ```python
    # Find top 10 values in 1 million array
    data = np.random.rand(1_000_000)
    
    # Slow: O(n log n)
    top_10 = np.sort(data)[-10:]
    
    # Fast: O(n)
    indices = np.argpartition(data, -10)[-10:]
    top_10 = data[indices]
    ```

    !!! tip "Interviewer's Insight"
        Knows when full sort is overkill. Uses argpartition for top-k problems. Mentions quickselect algorithm underneath.

---

### How to use np.searchsorted() for binary search? - Google, Amazon, HFT Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Binary Search`, `Algorithms` | **Asked by:** Google, Amazon, Jane Street, Citadel

??? success "View Answer"

    ```python
    import numpy as np
    
    sorted_arr = np.array([1, 3, 5, 7, 9, 11])
    
    # Find insertion index
    idx = np.searchsorted(sorted_arr, 6)  # Returns 3
    # Means: insert 6 at index 3 to keep sorted
    
    # Find multiple values
    values = np.array([2, 6, 10])
    indices = np.searchsorted(sorted_arr, values)
    # [1, 3, 4]
    
    # Side='right' for rightmost insertion point
    np.searchsorted(sorted_arr, 5, side='right')  # 3 (after 5)
    np.searchsorted(sorted_arr, 5, side='left')   # 2 (before 5)
    
    # Check if values exist
    idx = np.searchsorted(sorted_arr, 5)
    exists = (idx < len(sorted_arr)) and (sorted_arr[idx] == 5)
    ```

    **Real application - Time series:**

    ```python
    # Find events within time window
    timestamps = np.array([10, 20, 30, 40, 50, 60])  # Sorted
    
    start_idx = np.searchsorted(timestamps, 25, side='left')
    end_idx = np.searchsorted(timestamps, 55, side='right')
    
    events_in_window = timestamps[start_idx:end_idx]
    # [30, 40, 50]
    ```

    !!! tip "Interviewer's Insight"
        Uses for efficient lookups in sorted data. Knows O(log n) complexity. Mentions side parameter for duplicates.

---

### How to calculate pairwise distances efficiently? - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Broadcasting`, `Linear Algebra`, `ML` | **Asked by:** Google, Meta, Netflix

??? success "View Answer"

    **Problem:** Calculate distances between all pairs of points.

    **Approach 1: Broadcasting (memory intensive)**

    ```python
    X = np.random.rand(100, 3)  # 100 points in 3D
    
    # Expand dims for broadcasting
    X_expanded = X[:, np.newaxis, :]  # (100, 1, 3)
    diff = X_expanded - X[np.newaxis, :, :]  # (100, 100, 3)
    distances = np.sqrt(np.sum(diff**2, axis=-1))  # (100, 100)
    ```

    **Approach 2: Using formula (memory efficient)**

    ```python
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    X_sq = np.sum(X**2, axis=1)  # (100,)
    X_sq_col = X_sq[:, np.newaxis]  # (100, 1)
    X_sq_row = X_sq[np.newaxis, :]  # (1, 100)
    
    dot_product = X @ X.T  # (100, 100)
    distances_sq = X_sq_col + X_sq_row - 2 * dot_product
    distances = np.sqrt(np.clip(distances_sq, 0, None))  # Clip for numerical stability
    ```

    **Approach 3: scipy (production)**

    ```python
    from scipy.spatial.distance import cdist
    distances = cdist(X, X, metric='euclidean')
    ```

    | Method | Memory | Speed | Use Case |
    |--------|--------|-------|----------|
    | Broadcasting | O(n²×d) | Medium | Small n |
    | Formula | O(n²) | Fast | Large n |
    | scipy | O(n²) | Fastest | Production |

    !!! tip "Interviewer's Insight"
        Knows multiple approaches. Mentions memory vs speed tradeoff. Uses BLAS-optimized matrix multiplication.

---

### What is the difference between np.stack(), np.concatenate(), vstack(), hstack()? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Array Manipulation`, `Stacking` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Quick comparison:**

    | Function | Axis | New Dim? | Arrays Must Have Same Shape? |
    |----------|------|----------|-------------------------------|
    | `stack()` | Creates new | Yes | Yes |
    | `concatenate()` | Specified | No | All except concat axis |
    | `vstack()` | 0 (rows) | No | Columns match |
    | `hstack()` | 1 (cols) | No | Rows match |

    ```python
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    
    # stack() - creates NEW dimension
    np.stack([a, b], axis=0)  # [[1,2,3], [4,5,6]], shape (2,3)
    np.stack([a, b], axis=1)  # [[1,4], [2,5], [3,6]], shape (3,2)
    
    # concatenate() - extends existing dimension
    np.concatenate([a, b])  # [1,2,3,4,5,6], shape (6,)
    
    # 2D example
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    
    # vstack - vertically (more rows)
    np.vstack([A, B])
    # [[1, 2],
    #  [3, 4],
    #  [5, 6],
    #  [7, 8]]
    
    # hstack - horizontally (more columns)
    np.hstack([A, B])
    # [[1, 2, 5, 6],
    #  [3, 4, 7, 8]]
    
    # dstack - depth (3rd dimension)
    np.dstack([A, B])  # shape (2, 2, 2)
    ```

    !!! tip "Interviewer's Insight"
        Knows stack creates new axis. Uses vstack/hstack for readability. Mentions column_stack for 1D arrays.

---

### How to use np.meshgrid() and where is it useful? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Grid Generation`, `Plotting`, `Scientific Computing` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Purpose:** Create coordinate matrices from coordinate vectors.

    ```python
    import numpy as np
    
    x = np.array([1, 2, 3])
    y = np.array([4, 5])
    
    # Create 2D grid
    X, Y = np.meshgrid(x, y)
    
    # X = [[1, 2, 3],    Y = [[4, 4, 4],
    #      [1, 2, 3]]         [5, 5, 5]]
    
    # Every (X[i,j], Y[i,j]) is a coordinate
    ```

    **Real applications:**

    ```python
    # 1. Function evaluation on grid
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate function z = f(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))  # 2D sinc function
    
    # 2. Distance from origin
    distances = np.sqrt(X**2 + Y**2)
    
    # 3. Machine learning feature grid
    x1_range = np.linspace(0, 10, 100)
    x2_range = np.linspace(0, 10, 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    
    # Predict on grid for decision boundary
    grid_points = np.c_[X1.ravel(), X2.ravel()]
    predictions = model.predict(grid_points)
    predictions = predictions.reshape(X1.shape)
    ```

    !!! tip "Interviewer's Insight"
        Uses for plotting, contour plots, decision boundaries. Knows indexing='xy' vs 'ij' parameter.

---

### Explain np.clip() and use cases - Google, Amazon Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Array Manipulation`, `Data Cleaning` | **Asked by:** Google, Amazon, Meta, Netflix

??? success "View Answer"

    **Purpose:** Limit values to a range [min, max].

    ```python
    arr = np.array([1, 5, 10, 15, 20])
    
    # Clip to [5, 15]
    clipped = np.clip(arr, 5, 15)
    # [5, 5, 10, 15, 15]
    
    # One-sided clipping
    np.clip(arr, 10, None)  # [10, 10, 10, 15, 20] - only minimum
    np.clip(arr, None, 10)  # [1, 5, 10, 10, 10] - only maximum
    ```

    **Real applications:**

    ```python
    # 1. Image processing - pixel values in [0, 255]
    image = np.random.randn(100, 100) * 100 + 128
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    # 2. Gradient clipping (deep learning)
    gradients = model.compute_gradients()
    gradients = np.clip(gradients, -1.0, 1.0)
    
    # 3. Outlier handling
    data = np.random.randn(1000)
    # Clip to 3 standard deviations
    mean, std = data.mean(), data.std()
    data_clipped = np.clip(data, mean - 3*std, mean + 3*std)
    
    # 4. Winsorization (statistics)
    lower = np.percentile(data, 5)
    upper = np.percentile(data, 95)
    data_winsorized = np.clip(data, lower, upper)
    ```

    !!! tip "Interviewer's Insight"
        Uses for outlier handling, numerical stability. Knows winsorization vs trimming difference.

---

### How to use np.percentile() vs np.quantile()? - Google, Amazon, Netflix Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Statistics`, `Percentiles` | **Asked by:** Google, Amazon, Netflix

??? success "View Answer"

    **Difference:** Same function, different input scale.

    | Function | Input Range | Example |
    |----------|-------------|---------|
    | `percentile()` | 0-100 | 25, 50, 75 |
    | `quantile()` | 0-1 | 0.25, 0.5, 0.75 |

    ```python
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    # Percentile (0-100)
    np.percentile(data, 25)  # 3.25 (25th percentile)
    np.percentile(data, [25, 50, 75])  # [3.25, 5.5, 7.75]
    
    # Quantile (0-1)
    np.quantile(data, 0.25)  # 3.25 (same)
    np.quantile(data, [0.25, 0.5, 0.75])  # [3.25, 5.5, 7.75]
    
    # Interpolation methods
    np.percentile(data, 25, method='linear')     # Default
    np.percentile(data, 25, method='lower')      # Use lower value
    np.percentile(data, 25, method='higher')     # Use higher value
    np.percentile(data, 25, method='midpoint')   # Average of bounds
    np.percentile(data, 25, method='nearest')    # Closest value
    ```

    **Common use cases:**

    ```python
    # IQR (Interquartile Range)
    Q1, Q3 = np.percentile(data, [25, 75])
    IQR = Q3 - Q1
    
    # Outlier detection
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (data < lower_bound) | (data > upper_bound)
    ```

    !!! tip "Interviewer's Insight"
        Knows interpolation methods. Uses for outlier detection, box plots, data profiling.

---

### What is np.einsum() and when to use it? - Google, DeepMind, OpenAI Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Einstein Summation`, `Linear Algebra`, `Deep Learning` | **Asked by:** Google, DeepMind, OpenAI, Meta

??? success "View Answer"

    **Einstein summation notation** - concise tensor operations.

    **Basic examples:**

    ```python
    A = np.random.rand(3, 4)
    B = np.random.rand(4, 5)
    
    # Matrix multiplication: C[i,j] = Σ_k A[i,k] * B[k,j]
    C = np.einsum('ik,kj->ij', A, B)
    # Same as: A @ B
    
    # Transpose
    At = np.einsum('ij->ji', A)
    
    # Trace (diagonal sum)
    M = np.random.rand(5, 5)
    trace = np.einsum('ii->', M)
    # Same as: np.trace(M)
    
    # Element-wise product (Hadamard)
    result = np.einsum('ij,ij->ij', A, A)
    # Same as: A * A
    
    # Sum along axis
    row_sums = np.einsum('ij->i', A)  # Sum columns
    col_sums = np.einsum('ij->j', A)  # Sum rows
    ```

    **Advanced patterns:**

    ```python
    # Batch matrix multiply
    batch_A = np.random.rand(10, 3, 4)
    batch_B = np.random.rand(10, 4, 5)
    result = np.einsum('bij,bjk->bik', batch_A, batch_B)
    
    # Attention mechanism (simplified)
    Q = np.random.rand(8, 64)  # Queries
    K = np.random.rand(8, 64)  # Keys
    V = np.random.rand(8, 64)  # Values
    
    scores = np.einsum('qd,kd->qk', Q, K)  # Attention scores
    # After softmax:
    output = np.einsum('qk,kd->qd', scores, V)
    
    # Outer product
    a = np.array([1, 2, 3])
    b = np.array([4, 5])
    outer = np.einsum('i,j->ij', a, b)
    ```

    **When to use:**

    - Complex tensor operations
    - Batch operations
    - Transformers/attention
    - Custom contractions

    !!! tip "Interviewer's Insight"
        Reads einsum notation fluently. Knows when it's clearer than matmul. Mentions optimize='optimal' parameter for speed.

---

### How to use np.unique() effectively? - Google, Amazon Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Array Operations`, `Data Analysis` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    ```python
    arr = np.array([3, 1, 2, 3, 1, 2, 2, 4])
    
    # Just unique values (sorted)
    np.unique(arr)  # [1, 2, 3, 4]
    
    # With counts
    values, counts = np.unique(arr, return_counts=True)
    # values: [1, 2, 3, 4]
    # counts: [2, 3, 2, 1]
    
    # With original indices
    values, indices = np.unique(arr, return_inverse=True)
    # indices: [2, 0, 1, 2, 0, 1, 1, 3]
    # Reconstruction: values[indices] == arr
    
    # With first occurrence index
    values, first_idx = np.unique(arr, return_index=True)
    # first_idx: [1, 2, 0, 7] - where each value first appears
    
    # All at once
    values, indices, inverse, counts = np.unique(
        arr, return_index=True, return_inverse=True, return_counts=True
    )
    ```

    **Common patterns:**

    ```python
    # Most frequent element
    values, counts = np.unique(arr, return_counts=True)
    most_frequent = values[np.argmax(counts)]
    
    # Value counts as dict
    dict(zip(*np.unique(arr, return_counts=True)))
    # {1: 2, 2: 3, 3: 2, 4: 1}
    
    # Unique rows in 2D
    matrix = np.array([[1, 2], [3, 4], [1, 2]])
    unique_rows = np.unique(matrix, axis=0)
    # [[1, 2],
    #  [3, 4]]
    ```

    !!! tip "Interviewer's Insight"
        Uses return_counts for frequency analysis. Knows axis parameter for unique rows. Mentions sorting behavior.

---

### How to efficiently check array equality with tolerance? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Comparison`, `Numerical Precision` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Problem:** Floating point comparisons need tolerance.

    ```python
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0000001, 2.0000001, 3.0000001])
    
    # Exact comparison (usually fails for floats)
    np.array_equal(a, b)  # False
    
    # With tolerance - element-wise
    np.isclose(a, b)  # [True, True, True]
    np.isclose(a, b, rtol=1e-5, atol=1e-8)
    # rtol: relative tolerance
    # atol: absolute tolerance
    # Formula: |a - b| <= (atol + rtol * |b|)
    
    # All elements close
    np.allclose(a, b)  # True
    np.allclose(a, b, rtol=1e-5, atol=1e-8)
    
    # Manual implementation
    np.all(np.abs(a - b) < 1e-6)
    ```

    **Common pitfalls:**

    ```python
    # NaN handling
    a = np.array([1.0, np.nan, 3.0])
    b = np.array([1.0, np.nan, 3.0])
    
    np.allclose(a, b)  # False! NaN != NaN
    np.allclose(a, b, equal_nan=True)  # True
    
    # Infinity
    a = np.array([1.0, np.inf])
    b = np.array([1.0, np.inf])
    np.allclose(a, b)  # True (inf == inf)
    ```

    !!! tip "Interviewer's Insight"
        Uses allclose for arrays, isclose for element-wise. Knows equal_nan parameter. Explains rtol vs atol.

---

### How to use np.apply_along_axis() vs vectorization? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Iteration`, `Performance` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **apply_along_axis:** Apply function to 1D slices.

    ```python
    matrix = np.random.rand(5, 3)
    
    # Custom function along axis
    def normalize(x):
        return (x - x.mean()) / x.std()
    
    # Apply to each row
    normalized = np.apply_along_axis(normalize, 1, matrix)
    
    # Apply to each column
    normalized = np.apply_along_axis(normalize, 0, matrix)
    ```

    **Vectorization is faster:**

    ```python
    # Slow - apply_along_axis uses loop
    result = np.apply_along_axis(np.sum, 1, matrix)
    
    # Fast - vectorized
    result = matrix.sum(axis=1)
    
    # Slow
    result = np.apply_along_axis(lambda x: x.max() - x.min(), 1, matrix)
    
    # Fast
    result = matrix.max(axis=1) - matrix.min(axis=1)
    ```

    **When to use apply_along_axis:**

    - No vectorized alternative exists
    - Complex custom logic per slice
    - Readability over performance

    ```python
    # Example: Can't easily vectorize
    def custom_func(row):
        if row[0] > 0:
            return row.sum()
        else:
            return row.prod()
    
    result = np.apply_along_axis(custom_func, 1, matrix)
    ```

    !!! tip "Interviewer's Insight"
        Prefers vectorization when possible. Knows apply_along_axis is syntactic sugar over loops. Mentions numba/cython for complex cases.

---

### What are structured arrays and when to use them? - Google, Amazon, HFT Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Data Types`, `Structured Data` | **Asked by:** Google, Amazon, Jane Street, Two Sigma

??? success "View Answer"

    **Structured arrays:** Multiple fields per element (like struct in C).

    ```python
    # Define dtype with named fields
    dt = np.dtype([('name', 'U10'), ('age', 'i4'), ('salary', 'f8')])
    
    # Create structured array
    employees = np.array([
        ('Alice', 30, 50000.0),
        ('Bob', 25, 45000.0),
        ('Charlie', 35, 60000.0)
    ], dtype=dt)
    
    # Access by field name
    employees['name']    # ['Alice', 'Bob', 'Charlie']
    employees['age']     # [30, 25, 35]
    employees['salary']  # [50000., 45000., 60000.]
    
    # Index like normal array
    employees[0]  # ('Alice', 30, 50000.)
    employees[0]['name']  # 'Alice'
    
    # Sorting by field
    sorted_by_age = np.sort(employees, order='age')
    
    # Multi-field sort
    sorted_multi = np.sort(employees, order=['age', 'salary'])
    ```

    **Use cases:**

    ```python
    # 1. CSV-like data
    data = np.genfromtxt('data.csv', delimiter=',', 
                         dtype=[('date', 'U10'), ('price', 'f8')])
    
    # 2. Time series with metadata
    dt = np.dtype([('timestamp', 'datetime64[s]'), 
                   ('value', 'f8'), 
                   ('quality', 'i1')])
    
    # 3. Financial tick data
    dt = np.dtype([('symbol', 'U10'), ('price', 'f8'), 
                   ('volume', 'i8'), ('timestamp', 'i8')])
    ```

    **Advantages vs alternatives:**

    | | Structured Array | DataFrame | Dict of Arrays |
    |---|------------------|-----------|----------------|
    | Speed | Fast | Slower | Slow |
    | Memory | Compact | More | Most |
    | Features | Basic | Rich | Flexible |

    !!! tip "Interviewer's Insight"
        Uses for high-performance structured data. Knows it's faster than Pandas for simple operations. Mentions record arrays (rec.array).

---

### How to use np.histogram() for binning data? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Statistics`, `Data Analysis` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    ```python
    data = np.random.randn(1000)
    
    # Basic histogram
    counts, bin_edges = np.histogram(data, bins=10)
    # counts: frequency in each bin
    # bin_edges: boundaries (length = bins + 1)
    
    # Custom bin edges
    counts, bins = np.histogram(data, bins=[-3, -2, -1, 0, 1, 2, 3])
    
    # Equal-width bins in range
    counts, bins = np.histogram(data, bins=20, range=(-3, 3))
    
    # Density (probability)
    density, bins = np.histogram(data, bins=10, density=True)
    # density * bin_width = probability
    ```

    **2D histogram:**

    ```python
    x = np.random.randn(1000)
    y = np.random.randn(1000)
    
    H, xedges, yedges = np.histogram2d(x, y, bins=20)
    # H: 2D counts array
    ```

    **Use cases:**

    ```python
    # Discretize continuous data
    ages = np.array([25, 30, 35, 40, 45, 50, 55])
    bins = [0, 30, 40, 50, 100]
    bin_indices = np.digitize(ages, bins)
    # [1, 1, 2, 2, 3, 3, 3]
    
    # Equal-frequency binning (quantile-based)
    quantiles = np.percentile(data, [0, 25, 50, 75, 100])
    counts, _ = np.histogram(data, bins=quantiles)
    ```

    !!! tip "Interviewer's Insight"
        Knows histogram vs histogram2d. Uses digitize for binning. Mentions density normalization.

---

### Explain np.convolve() vs scipy.signal.convolve() - Google, Amazon, CV Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Signal Processing`, `Convolution` | **Asked by:** Google, Amazon, Meta, CV companies

??? success "View Answer"

    **1D Convolution:**

    ```python
    signal = np.array([1, 2, 3, 4, 5])
    kernel = np.array([0.5, 1, 0.5])
    
    # NumPy convolution
    result = np.convolve(signal, kernel, mode='same')
    
    # Modes:
    # 'full': full convolution (length = m + n - 1)
    # 'same': same length as signal
    # 'valid': only where fully overlapping
    ```

    **Mode comparison:**

    ```python
    signal = [1, 2, 3, 4, 5]
    kernel = [a, b, c]
    
    # 'full' (length 7):
    # [1*c, 1*b+2*c, 1*a+2*b+3*c, ...]
    
    # 'same' (length 5):
    # centered version of full
    
    # 'valid' (length 3):
    # [1*a+2*b+3*c, 2*a+3*b+4*c, 3*a+4*b+5*c]
    ```

    **Applications:**

    ```python
    # Moving average
    window = np.ones(5) / 5
    smoothed = np.convolve(signal, window, mode='same')
    
    # Edge detection (diff filter)
    kernel = np.array([1, 0, -1])
    edges = np.convolve(signal, kernel, mode='same')
    ```

    **NumPy vs scipy:**

    | Feature | NumPy | scipy |
    |---------|-------|-------|
    | Dimensions | 1D only | 1D, 2D, nD |
    | Methods | Direct | FFT, auto, etc |
    | Speed | Good | Faster (FFT) |

    !!! tip "Interviewer's Insight"
        Knows convolve vs correlate difference. Uses scipy for 2D/image processing. Mentions FFT-based convolution for large kernels.

---

### How to use np.pad() for array padding? - Google, CV Companies Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Array Manipulation`, `Image Processing` | **Asked by:** Google, Amazon, OpenCV roles

??? success "View Answer"

    ```python
    arr = np.array([1, 2, 3, 4, 5])
    
    # Constant padding
    np.pad(arr, pad_width=2, mode='constant', constant_values=0)
    # [0, 0, 1, 2, 3, 4, 5, 0, 0]
    
    # Edge mode (repeat edge values)
    np.pad(arr, 2, mode='edge')
    # [1, 1, 1, 2, 3, 4, 5, 5, 5]
    
    # Reflect mode (mirror)
    np.pad(arr, 2, mode='reflect')
    # [3, 2, 1, 2, 3, 4, 5, 4, 3]
    
    # Wrap mode (circular)
    np.pad(arr, 2, mode='wrap')
    # [4, 5, 1, 2, 3, 4, 5, 1, 2]
    ```

    **2D padding (images):**

    ```python
    image = np.random.rand(28, 28)
    
    # Pad all sides equally
    padded = np.pad(image, 2, mode='constant')  # (32, 32)
    
    # Different padding per dimension
    padded = np.pad(image, ((2, 2), (3, 3)), mode='constant')
    # Top/bottom: 2, Left/right: 3
    
    # Asymmetric padding
    padded = np.pad(image, ((1, 2), (3, 4)), mode='edge')
    # Top: 1, Bottom: 2, Left: 3, Right: 4
    ```

    **Custom padding function:**

    ```python
    def custom_pad(vector, pad_width, iaxis, kwargs):
        """Custom padding logic"""
        vector[:pad_width[0]] = -1  # Left pad
        vector[-pad_width[1]:] = -1  # Right pad
    
    padded = np.pad(arr, 2, custom_pad)
    ```

    !!! tip "Interviewer's Insight"
        Knows modes for different use cases. Uses for convolution "same" output. Mentions symmetric vs reflect difference.

---

### How to efficiently find indices of multiple values? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Indexing`, `Search` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Problem:** Find indices where array contains specific values.

    **Method 1: np.where()**

    ```python
    arr = np.array([1, 2, 3, 2, 4, 2, 5])
    
    # Single value
    indices = np.where(arr == 2)[0]  # [1, 3, 5]
    
    # Multiple values using isin
    values_to_find = [2, 4]
    mask = np.isin(arr, values_to_find)
    indices = np.where(mask)[0]  # [1, 3, 4, 5]
    ```

    **Method 2: nonzero()**

    ```python
    indices = np.nonzero(arr == 2)[0]  # Same as where
    
    # Multi-dimensional
    matrix = np.array([[1, 2], [3, 2], [2, 5]])
    rows, cols = np.where(matrix == 2)
    # rows: [0, 1, 2], cols: [1, 1, 0]
    ```

    **Method 3: argwhere() for coordinates**

    ```python
    coords = np.argwhere(matrix == 2)
    # [[0, 1],
    #  [1, 1],
    #  [2, 0]]
    ```

    **Performance comparison:**

    ```python
    # Fast - vectorized
    mask = np.isin(large_array, values)
    indices = np.where(mask)[0]
    
    # Slow - loop
    indices = [i for i, x in enumerate(large_array) if x in values]
    ```

    !!! tip "Interviewer's Insight"
        Uses isin for multiple values. Knows where returns tuple. Prefers argwhere for coordinate pairs.

---

### How to use np.gradient() for numerical differentiation? - Google, Amazon, Scientific Computing Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Calculus`, `Numerical Methods` | **Asked by:** Google, Amazon, Quantitative firms

??? success "View Answer"

    ```python
    # 1D gradient
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x)
    
    # Numerical derivative
    dy_dx = np.gradient(y, x)
    # Approximates cos(x)
    
    # Central difference formula
    # dy/dx ≈ (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])
    
    # With uniform spacing
    y = np.array([1, 4, 9, 16, 25])  # x^2
    gradient = np.gradient(y)  # [3, 4, 5, 6, 7] ≈ 2x
    ```

    **2D gradient (images):**

    ```python
    image = np.random.rand(100, 100)
    
    # Gradients in both directions
    grad_y, grad_x = np.gradient(image)
    
    # Gradient magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Gradient direction
    direction = np.arctan2(grad_y, grad_x)
    ```

    **Higher-order derivatives:**

    ```python
    y = np.array([1, 4, 9, 16, 25])  # x^2
    
    # First derivative
    dy = np.gradient(y)  # ≈ 2x
    
    # Second derivative
    d2y = np.gradient(dy)  # ≈ 2 (constant)
    ```

    !!! tip "Interviewer's Insight"
        Uses for numerical differentiation. Knows it's second-order accurate. Mentions edge handling with forward/backward differences.

---

### Explain memory views with np.asarray() vs np.array() - Google, HFT Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Memory Management`, `Views` | **Asked by:** Google, Amazon, Jane Street, Citadel

??? success "View Answer"

    **Key difference:**

    | Function | Creates Copy? | When |
    |----------|---------------|------|
    | `array()` | Always | Input is list or needs conversion |
    | `asarray()` | Only if needed | Returns view when possible |

    ```python
    # From list - both create copy
    lst = [1, 2, 3]
    a1 = np.array(lst)
    a2 = np.asarray(lst)
    
    # From array - different behavior
    arr = np.array([1, 2, 3])
    
    a1 = np.array(arr)     # Creates copy
    a2 = np.asarray(arr)   # Returns view (same object!)
    
    print(a1 is arr)  # False
    print(a2 is arr)  # True
    
    # Modify a2 affects arr
    a2[0] = 999
    print(arr[0])  # 999!
    ```

    **Memory efficiency:**

    ```python
    large_arr = np.random.rand(10**7)
    
    # Creates copy - doubles memory!
    copy1 = np.array(large_arr)
    
    # No copy - same memory
    view1 = np.asarray(large_arr)
    
    # Force copy
    copy2 = np.array(large_arr, copy=True)  # NumPy 2.0+
    ```

    **Type conversion:**

    ```python
    arr = np.array([1, 2, 3], dtype=np.int32)
    
    # asarray with dtype change creates copy
    arr_f64 = np.asarray(arr, dtype=np.float64)  # Copy
    print(arr_f64 is arr)  # False
    
    # Same dtype - no copy
    arr_same = np.asarray(arr, dtype=np.int32)  # View
    print(arr_same is arr)  # True
    ```

    !!! tip "Interviewer's Insight"
        Uses asarray in functions accepting array-like. Knows it prevents unnecessary copies. Mentions asanyarray for subclasses.

---

### How to use np.savez() for multiple arrays efficiently? - Google, Amazon Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `File I/O`, `Serialization` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    ```python
    # Save multiple arrays
    X_train = np.random.rand(1000, 10)
    y_train = np.random.randint(0, 2, 1000)
    X_test = np.random.rand(200, 10)
    y_test = np.random.randint(0, 2, 200)
    
    # Uncompressed .npz
    np.savez('data.npz', 
             X_train=X_train, y_train=y_train,
             X_test=X_test, y_test=y_test)
    
    # Compressed .npz (slower save, smaller file)
    np.savez_compressed('data_compressed.npz',
                        X_train=X_train, y_train=y_train)
    
    # Load
    data = np.load('data.npz')
    X_train = data['X_train']
    y_train = data['y_train']
    
    # Or unpack
    with np.load('data.npz') as data:
        X_train = data['X_train']
        y_train = data['y_train']
    
    # List keys
    print(data.files)  # ['X_train', 'y_train', 'X_test', 'y_test']
    ```

    **Single array (.npy):**

    ```python
    # Save
    np.save('array.npy', X_train)
    
    # Load
    X_train = np.load('array.npy')
    ```

    **Comparison:**

    | Format | Arrays | Compression | Speed | Use Case |
    |--------|--------|-------------|-------|----------|
    | .npy | 1 | No | Fastest | Single array |
    | .npz | Multiple | Optional | Fast | Dataset |
    | pickle | Any Python | Optional | Slow | Complex objects |

    !!! tip "Interviewer's Insight"
        Uses savez for datasets. Knows compression tradeoff. Mentions allow_pickle=False for security.

---

### How to use np.fromfunction() for array initialization? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Array Creation`, `Functional` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Create arrays using function of indices:**

    ```python
    # Simple example
    def f(i, j):
        return i + j
    
    arr = np.fromfunction(f, (3, 4), dtype=int)
    # [[0, 1, 2, 3],
    #  [1, 2, 3, 4],
    #  [2, 3, 4, 5]]
    
    # Multiplication table
    def mult_table(i, j):
        return (i + 1) * (j + 1)
    
    table = np.fromfunction(mult_table, (10, 10), dtype=int)
    
    # Distance from center
    def distance_from_center(i, j, center_i=50, center_j=50):
        return np.sqrt((i - center_i)**2 + (j - center_j)**2)
    
    image = np.fromfunction(distance_from_center, (100, 100))
    ```

    **vs other methods:**

    ```python
    # Using meshgrid (equivalent but explicit)
    i, j = np.mgrid[0:3, 0:4]
    arr = i + j
    
    # Using indices
    i, j = np.indices((3, 4))
    arr = i + j
    ```

    !!! tip "Interviewer's Insight"
        Uses for pattern generation. Knows indices are passed as separate arguments. Mentions lambda for simple cases.

---

### How to handle large arrays that don't fit in memory? - Google, Amazon, Netflix Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Memory Management`, `Big Data`, `Performance` | **Asked by:** Google, Amazon, Netflix, Data Engineering roles

??? success "View Answer"

    **Strategy 1: Memory mapping**

    ```python
    # Create memory-mapped array (on disk)
    large_array = np.memmap('large_file.dat', 
                            dtype='float32',
                            mode='w+',
                            shape=(100000000, 100))
    
    # Works like normal array but uses disk
    large_array[0:1000] = np.random.rand(1000, 100)
    
    # Read-only mode for existing file
    readonly_array = np.memmap('large_file.dat',
                               dtype='float32',
                               mode='r',
                               shape=(100000000, 100))
    ```

    **Strategy 2: Chunked processing**

    ```python
    def process_in_chunks(filename, chunk_size=10000):
        data = np.load(filename, mmap_mode='r')  # Memory-mapped
        n_samples = len(data)
        
        results = []
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            chunk = data[start:end]
            
            # Process chunk
            result = chunk.mean(axis=1)
            results.append(result)
        
        return np.concatenate(results)
    ```

    **Strategy 3: Generators**

    ```python
    def data_generator(filename, batch_size=32):
        """Yield batches without loading all data"""
        data = np.load(filename, mmap_mode='r')
        n_samples = len(data)
        
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            yield data[start:end]
    
    # Use in training loop
    for batch in data_generator('huge_dataset.npy'):
        model.train(batch)
    ```

    **Strategy 4: Sparse arrays**

    ```python
    from scipy.sparse import csr_matrix
    
    # For mostly-zero arrays
    dense = np.zeros((10000, 10000))
    dense[::100, ::100] = 1  # Only 10000 non-zeros
    
    # Much less memory
    sparse = csr_matrix(dense)
    ```

    !!! tip "Interviewer's Insight"
        Uses memmap for out-of-core processing. Knows chunking strategies. Mentions Dask for parallel chunked operations.

---

## Quick Reference: 100+ NumPy Interview Questions

| Sno | Question Title | Practice Links | Companies Asking | Difficulty | Topics |
|-----|----------------|----------------|------------------|------------|--------|
| 1 | What is NumPy and why is it faster than lists? | [NumPy Docs](https://numpy.org/doc/stable/user/whatisnumpy.html) | Google, Amazon, Meta, Netflix | Easy | Basics, Performance |
| 2 | Difference between list vs NumPy array | [GeeksforGeeks](https://www.geeksforgeeks.org/difference-between-list-and-numpy-array/) | Google, Amazon, Microsoft | Easy | Data Structures |
| 3 | How to create specific arrays (zeros, ones, eye)? | [NumPy Docs](https://numpy.org/doc/stable/reference/routines.array-creation.html) | Most Tech Companies | Easy | Array Creation |
| 4 | What is broadcasting in NumPy? | [NumPy Docs](https://numpy.org/doc/stable/user/basics.broadcasting.html) | Google, Amazon, Meta, Apple | Medium | Broadcasting, Vectorization |
| 5 | How to handle shapes and reshaping? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html) | Most Tech Companies | Easy | Array Manipulation |
| 6 | What are ufuncs (universal functions)? | [NumPy Docs](https://numpy.org/doc/stable/reference/ufuncs.html) | Google, Amazon, OpenAI | Medium | ufuncs, Vectorization |
| 7 | How to check memory usage of an array? | [Stack Overflow](https://stackoverflow.com/questions/11285613/selecting-multiple-columns-in-a-pandas-dataframe) | Google, Amazon, Netflix | Easy | Memory, Performance |
| 8 | Difference between flatten() and ravel() | [Stack Overflow](https://stackoverflow.com/questions/28930465/what-is-the-difference-between-flatten-and-ravel-functions-in-numpy) | Google, Amazon, Meta | Medium | Array Manipulation |
| 9 | How to perform matrix multiplication? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html) | Most Tech Companies | Easy | Linear Algebra |
| 10 | What is dot product vs cross product? | [NumPy Docs](https://numpy.org/doc/stable/reference/routines.linalg.html) | Google, Amazon, Meta | Medium | Linear Algebra |
| 11 | How to stack arrays (vstack, hstack)? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.vstack.html) | Google, Amazon, Microsoft | Easy | Array Manipulation |
| 12 | What is broadcasting error? | [Stack Overflow](https://stackoverflow.com/questions/29954263/what-does-the-valueerror-operands-could-not-be-broadcast-together-mean) | Google, Amazon, Meta | Easy | Debugging |
| 13 | How to generate random numbers? | [NumPy Docs](https://numpy.org/doc/stable/reference/random/index.html) | Most Tech Companies | Easy | Random Sampling |
| 14 | Difference between rand(), randn(), randint() | [GeeksforGeeks](https://www.geeksforgeeks.org/numpy-random-rand-vs-numpy-random-random/) | Google, Amazon, Meta | Easy | Random Sampling |
| 15 | How to set random seed? | [NumPy Docs](https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html) | Google, Amazon, Netflix | Easy | Reproducibility |
| 16 | How to find unique values and counts? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.unique.html) | Google, Amazon, Meta | Easy | Array Operations |
| 17 | How to calculate mean, median, std? | [NumPy Docs](https://numpy.org/doc/stable/reference/routines.statistics.html) | Most Tech Companies | Easy | Statistics |
| 18 | How to perform element-wise comparison? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.greater.html) | Google, Amazon, Meta | Easy | Boolean Operations |
| 19 | How to filter array with boolean indexing? | [NumPy Docs](https://numpy.org/doc/stable/user/basics.indexing.html) | Most Tech Companies | Easy | Indexing |
| 20 | How to use where() for conditional selection? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.where.html) | Google, Amazon, Meta | Medium | Conditional Logic |
| 21 | How to sort arrays? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.sort.html) | Most Tech Companies | Easy | Sorting |
| 22 | Difference between sort() methods (quicksort etc)? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.sort.html) | Google, Amazon, HFT Firms | Medium | Algorithms |
| 23 | How to get indices of sorted elements (argsort)? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.argsort.html) | Google, Amazon, Meta | Medium | Sorting, Indexing |
| 24 | How to find min/max values and their indices? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html) | Most Tech Companies | Easy | Statistics |
| 25 | How to calculate percentiles and quantiles? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.percentile.html) | Google, Amazon, Netflix, Apple | Medium | Statistics |
| 26 | How to save and load arrays (.npy, .npz)? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.save.html) | Google, Amazon, Meta | Easy | File I/O |
| 27 | How to read text/CSV with NumPy? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html) | Google, Amazon, Microsoft | Medium | File I/O |
| 28 | What is the difference between copy and view? | [NumPy Docs](https://numpy.org/doc/stable/user/basics.copies.html) | Google, Amazon, Meta | Hard | Memory Management |
| 29 | How to transpose a matrix? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html) | Most Tech Companies | Easy | Linear Algebra |
| 30 | How to compute inverse of a matrix? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html) | Google, Amazon, Meta | Medium | Linear Algebra |
| 31 | How to solve linear equations? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html) | Google, Amazon, Meta | Medium | Linear Algebra |
| 32 | How to calculate eigenvalues and eigenvectors? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html) | Google, Amazon, HFT Firms | Hard | Linear Algebra |
| 33 | How to compute determinant? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.linalg.det.html) | Google, Amazon, Meta | Easy | Linear Algebra |
| 34 | How to perform singular value decomposition (SVD)? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html) | Google, Amazon, Netflix | Hard | Linear Algebra |
| 35 | How to calculate inner and outer products? | [NumPy Docs](https://numpy.org/doc/stable/reference/routines.linalg.html) | Google, Amazon, Meta | Medium | Linear Algebra |
| 36 | How to use nan-safe functions (nanmean, etc)? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.nanmean.html) | Google, Amazon, Netflix | Medium | Missing Data |
| 37 | How to replace values meeting a condition? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.place.html) | Google, Amazon, Meta | Easy | Array Manipulation |
| 38 | How to pad an array? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.pad.html) | Google, Amazon, CV Companies | Medium | Image Processing |
| 39 | How to repeat elements or arrays? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.repeat.html) | Google, Amazon, Meta | Easy | Array Manipulation |
| 40 | How to split arrays? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.split.html) | Google, Amazon, Meta | Easy | Array Manipulation |
| 41 | How to use meshgrid? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html) | Google, Amazon, Meta | Medium | Plotting, Geometry |
| 42 | How to perform cumulative sum/product? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html) | Google, Amazon, Meta | Easy | Statistics |
| 43 | How to use diff() for discrete difference? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.diff.html) | Google, Amazon, HFT Firms | Medium | Time Series |
| 44 | How to compute histogram? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.histogram.html) | Google, Amazon, Meta | Medium | Statistics |
| 45 | How to digitize/bin data? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html) | Google, Amazon, Meta | Medium | Statistics |
| 46 | How to set print options? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.set_printoptions.html) | Google, Amazon | Easy | Display |
| 47 | How to use apply_along_axis? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.apply_along_axis.html) | Google, Amazon, Meta | Medium | Iteration |
| 48 | How to handle complex numbers? | [NumPy Docs](https://numpy.org/doc/stable/user/basics.types.html) | Google, Amazon, HFT Firms | Medium | Data Types |
| 49 | How to change data type (astype)? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html) | Most Tech Companies | Easy | Data Types |
| 50 | What are structured arrays? | [NumPy Docs](https://numpy.org/doc/stable/user/basics.rec.html) | Google, Amazon, HFT Firms | Hard | Advanced Data Types |
| 51 | What is None vs np.nan? | [Stack Overflow](https://stackoverflow.com/questions/17506163/how-to-check-if-float-nan-is-nan) | Google, Amazon, Microsoft | Easy | Basics |
| 52 | How to check if array is empty? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.size.html) | Google, Amazon | Easy | Basics |
| 53 | How to use expand_dims() and squeeze()? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html) | Google, Amazon, Meta, CV Companies | Medium | Shape Manipulation |
| 54 | How to use vectorization for performance? | [Real Python](https://realpython.com/numpy-array-programming/) | Google, Amazon, Meta | Medium | Performance |
| 55 | How to optimize memory with strides? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.as_strided.html) | Google, Amazon, HFT Firms | Hard | Internals |
| 56 | How to use matrix power? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_power.html) | Google, Amazon | Easy | Linear Algebra |
| 57 | How to compute trace? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.trace.html) | Google, Amazon | Easy | Linear Algebra |
| 58 | How to compute norm of vector/matrix? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html) | Google, Amazon, Meta | Medium | Linear Algebra |
| 59 | How to solve least squares problem? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html) | Google, Amazon, Meta | Medium | Optimization |
| 60 | How to use clip() to limit values? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.clip.html) | Google, Amazon, Meta | Easy | Array Manipulation |
| 61 | How to use roll() to shift elements? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.roll.html) | Google, Amazon, Meta | Medium | Array Manipulation |
| 62 | How to use tile() to construct array? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.tile.html) | Google, Amazon | Medium | Array Manipulation |
| 63 | How to use logical operations (and, or, xor)? | [NumPy Docs](https://numpy.org/doc/stable/reference/routines.logic.html) | Google, Amazon, Meta | Easy | Logic |
| 64 | How to use isclose() for float comparison? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.isclose.html) | Google, Amazon, Meta | Medium | Logic, Precision |
| 65 | How to use allclose() for array comparison? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html) | Google, Amazon, Meta | Medium | Logic, Testin |
| 66 | How to perform set operations (union, intersect)? | [NumPy Docs](https://numpy.org/doc/stable/reference/routines.set.html) | Google, Amazon, Meta | Medium | Set Operations |
| 67 | How to use indices() to return grid indices? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.indices.html) | Google, Amazon | Hard | Advanced Indexing |
| 68 | How to use unravel_index()? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.unravel_index.html) | Google, Amazon | Medium | Shape Manipulation |
| 69 | How to use ravel_multi_index()? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.ravel_multi_index.html) | Google, Amazon | Hard | Shape Manipulation |
| 70 | How to use diagonal() to extract diagonals? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.diagonal.html) | Google, Amazon | Easy | Linear Algebra |
| 71 | How to create mask arrays? | [NumPy Docs](https://numpy.org/doc/stable/reference/maskedarray.html) | Google, Amazon, Meta | Medium | Masked Arrays |
| 72 | How to use polyfit() and polyval()? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html) | Google, Amazon, Meta | Medium | Curve Fitting |
| 73 | How to perform convolution? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.convolve.html) | Google, Amazon, CV Companies | Hard | Signal Processing |
| 74 | How to use correlate()? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.correlate.html) | Google, Amazon | Hard | Signal Processing |
| 75 | How to use fft() for Fourier Transform? | [NumPy Docs](https://numpy.org/doc/stable/reference/routines.fft.html) | Google, Amazon, HFT Firms | Hard | Signal Processing |
| 76 | How to use piecewise() functions? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.piecewise.html) | Google, Amazon | Medium | Advanced Logic |
| 77 | How to use select() for multiple conditions? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.select.html) | Google, Amazon | Medium | Advanced Logic |
| 78 | How to use einsum() for Einstein summation? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) | Google, Amazon, Meta, Research | Hard | Advanced Linear Algebra |
| 79 | How to use tensordot()? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.tensordot.html) | Google, Amazon, Research | Hard | Deep Learning |
| 80 | How to use kronecker product (kron)? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.kron.html) | Google, Amazon | Medium | Linear Algebra |
| 81 | How to use gradient() to compute gradient? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.gradient.html) | Google, Amazon, Meta | Medium | Calculus |
| 82 | How to use trapz() for integration? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.trapz.html) | Google, Amazon | Medium | Calculus |
| 83 | How to use interp() for linear interpolation? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.interp.html) | Google, Amazon | Medium | Math |
| 84 | How to Use broadcasting with newaxis? | [NumPy Docs](https://numpy.org/doc/stable/reference/constants.html#numpy.newaxis) | Google, Amazon, Meta | Medium | Broadcasting |
| 85 | How to use array_split()? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.array_split.html) | Google, Amazon | Easy | Array Manipulation |
| 86 | How to use column_stack() and row_stack()? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.column_stack.html) | Google, Amazon | Easy | Array Manipulation |
| 87 | How to use dstack() (depth stacking)? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.dstack.html) | Google, Amazon, CV Companies | Medium | Array Manipulation |
| 88 | How to use vsplit() and hsplit()? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.vsplit.html) | Google, Amazon | Easy | Array Manipulation |
| 89 | How to use rollaxis() vs moveaxis()? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.moveaxis.html) | Google, Amazon, Research | Medium | Shape Manipulation |
| 90 | How to use swapaxes()? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.swapaxes.html) | Google, Amazon | Easy | Shape Manipulation |
| 91 | How to use fromiter() to create array? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.fromiter.html) | Google, Amazon | Medium | Array Creation |
| 92 | How to use frombuffer()? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.frombuffer.html) | Google, Amazon, HFT Firms | Hard | Internals, I/O |
| 93 | How to use partition() and argpartition()? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.partition.html) | Google, Amazon | Medium | Sorting |
| 94 | How to use searchsorted() for binary search? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html) | Google, Amazon, HFT Firms | Medium | Algorithms |
| 95 | How to use extract() based on condition? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.extract.html) | Google, Amazon | Medium | Filtering |
| 96 | How to use count_nonzero()? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.count_nonzero.html) | Google, Amazon | Easy | Basics |
| 97 | How to use copysign()? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.copysign.html) | Google, Amazon | Medium | Math |
| 98 | How to use fmax() and fmin()? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.fmax.html) | Google, Amazon | Medium | Math |
| 99 | How to use nan_to_num()? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html) | Google, Amazon, Netflix | Medium | Data Cleaning |
| 100| How to use correlate() vs convolve()? | [NumPy Docs](https://numpy.org/doc/stable/reference/routines.statistics.html) | Google, Amazon | Hard | Signal Processing |
| 101 | **[HARD]** How to implement custom ufuncs? | [NumPy Docs](https://numpy.org/doc/stable/user/c-info.ufunc-tutorial.html) | Google, Amazon, Research | Hard | Extending NumPy |
| 102 | **[HARD]** Explain C vs Fortran memory layout (contiguous)? | [NumPy Docs](https://numpy.org/doc/stable/user/basics.copies.html) | HFT Firms, Google, Amazon | Hard | Internals, Performance |
| 103 | **[HARD]** How to use `as_strided` for sliding windows? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.as_strided.html) | HFT Firms, Google, Amazon | Hard | Internals |
| 104 | **[HARD]** How to map large files with `memmap`? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html) | Google, Amazon, Netflix | Hard | Big Data, I/O |
| 105 | **[HARD]** Explain `einsum` index notation differences? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) | Google, DeepMind, OpenAI | Hard | Advanced Math |
| 106 | **[HARD]** How to efficiently broadcast without allocation? | [NumPy Docs](https://numpy.org/doc/stable/user/basics.broadcasting.html) | Google, Amazon | Hard | Performance |
| 107 | **[HARD]** How to link with optimized BLAS/LAPACK? | [NumPy Docs](https://numpy.org/doc/stable/user/building.html) | Google, Amazon, Research | Hard | Performance, Build |
| 108 | **[HARD]** How to use Structured Arrays for mixed data? | [NumPy Docs](https://numpy.org/doc/stable/user/basics.rec.html) | Google, Amazon, HFT Firms | Hard | Advanced Data Types |
| 109 | **[HARD]** How to vectorizing non-trivial objects properly? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html) | Google, Amazon | Hard | Performance |
| 110 | **[HARD]** How to manage floating point precision issues? | [NumPy Docs](https://numpy.org/doc/stable/user/basics.types.html) | HFT Firms, Research | Hard | Numerics |
| 111 | **[HARD]** How to implement cache blocking for operations? | [Intel Guides](https://software.intel.com/content/www/us/en/develop/articles/optimize-memory-access-with-intel-distribution-for-python-scipy.html) | HFT Firms, HPC | Hard | CPU Arch, Performance |
| 112 | **[HARD]** How to use Numba `@jit` with structured arrays? | [Numba Docs](https://numba.pydata.org/) | Google, HFT Firms | Hard | Optimization |
| 113 | **[HARD]** Explain the difference between `Generator` vs `RandomState`? | [NumPy Docs](https://numpy.org/doc/stable/reference/random/index.html) | Google, Amazon, Research | Hard | Randomness |
| 114 | **[HARD]** How to implement thread-safe random number generation? | [NumPy Docs](https://numpy.org/doc/stable/reference/random/multithreading.html) | Google, Amazon, HFT Firms | Hard | Parallelism |
| 115 | **[HARD]** How to use `np.frompyfunc` vs `np.vectorize`? | [Stack Overflow](https://stackoverflow.com/questions/35215161/most-efficient-way-to-map-function-over-numpy-array) | Google, Amazon | Hard | Performance |
| 116 | **[HARD]** How to debug stride-related issues? | [NumPy Docs](https://numpy.org/doc/stable/reference/arrays.ndarray.html) | HFT Firms, Google | Hard | Debugging |
| 117 | **[HARD]** How to optimize reduction operations (`keepdims`)? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.sum.html) | Google, Amazon | Hard | Optimization |
| 118 | **[HARD]** How to interface NumPy with C/C++ pointers? | [NumPy Docs](https://numpy.org/doc/stable/user/c-info.html) | HFT Firms, Google, Amazon | Hard | Interop |
| 119 | **[HARD]** How to use bitwise operations on packed arrays? | [NumPy Docs](https://numpy.org/doc/stable/reference/generated/numpy.packbits.html) | Google, Amazon | Hard | Optimization |
| 120 | **[HARD]** How to implement boolean masking without copies? | [NumPy Docs](https://numpy.org/doc/stable/user/basics.indexing.html) | Google, Amazon | Hard | Memory |

---

## Code Examples

### 1. Advanced Broadcasting

??? success "View Code Example"


    **Difficulty:** 🟢 Easy | **Tags:** `Code Example` | **Asked by:** Code Pattern
    ```python
    import numpy as np

    # Calculating distance matrix between two sets of points
    # A: (3, 2), B: (4, 2)
    A = np.array([[1,1], [2,2], [3,3]])
    B = np.array([[4,4], [5,5], [6,6], [7,7]])

    # Shape manipulation for broadcasting
    # shape (3,1,2) - shape (1,4,2) -> shape (3,4,2)
    diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]

    # Summing squares along last axis: shape (3,4)
    dists = np.sum(diff**2, axis=-1)
    print(dists)
    ```

### 2. Efficient Sliding Window (Stride Tricks)

??? success "View Code Example"


    **Difficulty:** 🟢 Easy | **Tags:** `Code Example` | **Asked by:** Code Pattern
    ```python
    import numpy as np
    from numpy.lib.stride_tricks import as_strided

    def sliding_window(arr, window_size):
        """
        Efficiently create sliding windows without copying data.
        """
        stride = arr.strides[0]
        shape = (len(arr) - window_size + 1, window_size)
        strides = (stride, stride)
        return as_strided(arr, shape=shape, strides=strides)

    arr = np.arange(10)
    print(sliding_window(arr, 3))
    ```

### 3. Einstein Summation

??? success "View Code Example"


    **Difficulty:** 🟢 Easy | **Tags:** `Code Example` | **Asked by:** Code Pattern
    ```python
    import numpy as np

    A = np.random.rand(2, 3)
    B = np.random.rand(3, 4)
    C = np.random.rand(2, 4)

    # Matrix multiplication: A @ B
    res_matmul = np.einsum('ik,kj->ij', A, B)

    # Dot product of rows in A and C
    res_dot = np.einsum('ij,ij->i', A, C)

    print("Matmul shape:", res_matmul.shape)
    print("Row dot shape:", res_dot.shape)
    ```

---

## Questions asked in Google interview
- How would you implement convolution from scratch using stride tricks?
- Explain the memory layout of C vs Fortran arrays in NumPy
- Write code to efficiently calculate pairwise distances
- How to handle numerical stability in large matrix operations?
- Explain broadcasting rules with examples
- How would you optimize a slow loop over arrays?
- Write code to perform image padding manually
- How to implement moving average without loops?
- Explain the usage of einsum vs dot product
- How to handle large datasets that don't fit in RAM?

## Questions asked in Amazon interview
- Write code to implement sparse matrix multiplication
- How would you generate non-uniform random numbers?
- Explain vectorized boolean operations
- Write code to filter values without creating a copy
- How to optimize array concatenation in a loop?
- Explain eigen decomposition implementation details
- Write code to solve system of linear equations
- How to handle missing values in numeric arrays?
- Explain performance difference between float32 vs float64
- Write code to normalize a matrix row-wise

## Questions asked in Meta interview
- How would you implement efficient array sorting?
- Explain structured arrays and their use cases
- Write code to compute histograms on multidimensional data
- How to implement custom reduction functions?
- Explain caching effects on array operations
- Write code to rotate an image represented as an array
- How to handle overflow in integer arrays?
- Explain how ufuncs work internally
- Write code to efficiently slicing multi-dimensional arrays
- How to implement vectorized string operations?

## Questions asked in Microsoft interview
- Explain the role of BLAS/LAPACK in NumPy
- Write code to compute the inverse of a matrix
- How to create a view of an array with different data type?
- Explain memory mapping for large files
- Write code to perform fast fourier transform
- How to implement a custom random number generator?
- Explain broadcasting errors and how to fix them
- Write code to compute cross-correlation
- How to optimize dot product for sparse vectors?
- Explain how to use `np.where` for complex conditions

## Questions asked in HFT Firms (e.g., Jane Street, Citadel)
- How to optimize stride usage for cache locality?
- Write code to implement order management system logic with arrays
- Explain floating point precision pitfalls in financial calc
- How to minimize memory allocations in critical paths?
- Write code to implement rolling window statistics efficiently
- how to use Numba to accelerate NumPy logic?
- Explain SIMD instructions usage in NumPy
- Write code to process tick data efficiently
- How to handle NaN propagation in accumulation?
- Explain the difference between `np.random.rand` and `np.random.Generator`

---

## Additional Resources

- [Official NumPy Documentation](https://numpy.org/doc/stable/)
- [From Python to NumPy](https://www.labri.fr/perso/nrougier/from-python-to-numpy/)
- [100 NumPy Exercises](https://github.com/rougier/numpy-100)
- [Scipy Lecture Notes](http://scipy-lectures.org/)
- [NumPy Visualization](https://github.com/rougier/numpy-tutorial)
