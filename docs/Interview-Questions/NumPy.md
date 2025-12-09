
---
title: NumPy Interview Questions
description: 100+ NumPy interview questions for cracking Data Science, Machine Learning, and Quant Developer interviews
---

# NumPy Interview Questions

<!-- [TOC] -->

This document provides a curated list of NumPy interview questions commonly asked in technical interviews for Data Science, Quantitative Analyst, Machine Learning Engineer, and High-Performance Computing roles. It covers everything from array manipulation to advanced linear algebra and memory management.

This is updated frequently but right now this is the most exhaustive list of type of questions being asked.

---

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
