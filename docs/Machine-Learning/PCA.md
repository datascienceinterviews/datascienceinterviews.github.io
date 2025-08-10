---
title: PCA (Principal Component Analysis)
description: Comprehensive guide to Principal Component Analysis with mathematical intuition, implementations, and interview questions.
comments: true
---

# 🎯 Principal Component Analysis (PCA)

PCA is a fundamental dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space while preserving maximum variance, making it invaluable for data visualization, noise reduction, and feature extraction.

**Resources:** [Scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) | [Elements of Statistical Learning - Chapter 14](https://web.stanford.edu/~hastie/ElemStatLearn/) | [Pattern Recognition and Machine Learning - Chapter 12](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)

##  Summary

Principal Component Analysis (PCA) is an unsupervised linear dimensionality reduction technique that identifies the principal components (directions of maximum variance) in high-dimensional data. It projects the original data onto a lower-dimensional subspace defined by these components, effectively reducing the number of features while retaining as much information as possible.

**Key characteristics:**
- **Dimensionality reduction**: Reduces the number of features while preserving information
- **Variance maximization**: Finds directions that capture maximum variance in data
- **Linear transformation**: Uses linear combinations of original features
- **Orthogonal components**: Principal components are orthogonal to each other
- **Data compression**: Enables efficient storage and transmission of data
- **Noise reduction**: Can filter out noise by discarding low-variance components

**Applications:**
- Data visualization (reducing to 2D/3D for plotting)
- Image compression and processing
- Feature extraction for machine learning
- Exploratory data analysis
- Noise reduction and signal processing
- Face recognition systems
- Stock market analysis
- Gene expression analysis

**Types:**
- **Standard PCA**: Linear dimensionality reduction using covariance matrix
- **Kernel PCA**: Non-linear extension using kernel methods
- **Sparse PCA**: Incorporates sparsity constraints on components
- **Incremental PCA**: For large datasets that don't fit in memory

## >� Intuition

### How PCA Works

Imagine you have a dataset of house prices with features like size, number of rooms, age, etc. Some features might be highly correlated (e.g., size and number of rooms). PCA finds new "directions" (principal components) that best capture the variation in your data. The first principal component captures the most variation, the second captures the next most variation (orthogonal to the first), and so on.

Think of it like finding the best angle to photograph a 3D object on a 2D photo - you want the angle that preserves the most information about the object's shape.

### Mathematical Foundation

#### 1. Covariance Matrix

For a dataset $X \in \mathbb{R}^{n \times d}$ (n samples, d features), first center the data:
$$\bar{X} = X - \mathbf{1}\mu^T$$

where $\mu = \frac{1}{n}\sum_{i=1}^{n} X_i$ is the mean vector.

The covariance matrix is:
$$C = \frac{1}{n-1}\bar{X}^T\bar{X}$$

#### 2. Eigenvalue Decomposition

PCA finds the eigenvalues and eigenvectors of the covariance matrix:
$$C\mathbf{v} = \lambda\mathbf{v}$$

Where:
- $\mathbf{v}$ are the eigenvectors (principal components)
- $\lambda$ are the eigenvalues (explained variance)

#### 3. Principal Components

The eigenvectors $\mathbf{v}_1, \mathbf{v}_2, ..., \mathbf{v}_d$ ordered by decreasing eigenvalues $\lambda_1 \geq \lambda_2 \geq ... \geq \lambda_d$ are the principal components.

#### 4. Dimensionality Reduction

To reduce to $k$ dimensions, select the first $k$ eigenvectors:
$$W = [\mathbf{v}_1, \mathbf{v}_2, ..., \mathbf{v}_k] \in \mathbb{R}^{d \times k}$$

Transform the data:
$$Z = \bar{X}W \in \mathbb{R}^{n \times k}$$

#### 5. Reconstruction

The original data can be approximated as:
$$\hat{X} = ZW^T + \mathbf{1}\mu^T$$

#### 6. Explained Variance Ratio

The proportion of variance explained by the first $k$ components:
$$\text{Explained Variance Ratio} = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{d} \lambda_i}$$

## =" Implementation using Libraries

### Scikit-learn Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_blobs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns

# Load and prepare data
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names

# Standardize the features (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Get explained variance ratio
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("Explained Variance by Component:")
for i, var in enumerate(explained_variance):
    print(f"PC{i+1}: {var:.3f} ({var*100:.1f}%)")

print(f"\nCumulative Explained Variance:")
for i, cum_var in enumerate(cumulative_variance):
    print(f"First {i+1} components: {cum_var:.3f} ({cum_var*100:.1f}%)")

# Visualize explained variance
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.bar(range(1, len(explained_variance) + 1), explained_variance)
plt.title('Explained Variance by Principal Component')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.legend()
plt.tight_layout()
plt.show()

# 2D visualization using first 2 components
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
colors = ['red', 'green', 'blue']
for i, color in enumerate(colors):
    plt.scatter(X_pca_2d[y == i, 0], X_pca_2d[y == i, 1], 
                c=color, label=iris.target_names[i], alpha=0.6)
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2f} variance)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2f} variance)')
plt.title('PCA: Iris Dataset in 2D')
plt.legend()
plt.grid(True)
plt.show()

# Component interpretation
components_df = pd.DataFrame(
    pca_2d.components_.T,
    columns=['PC1', 'PC2'],
    index=feature_names
)
print("\nPrincipal Component Loadings:")
print(components_df)

# Biplot (features and data points)
def biplot(X_pca, components, feature_names, y):
    plt.figure(figsize=(12, 8))
    
    # Plot data points
    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], 
                   c=color, label=iris.target_names[i], alpha=0.6)
    
    # Plot feature vectors
    for i, feature in enumerate(feature_names):
        plt.arrow(0, 0, components[i, 0]*3, components[i, 1]*3,
                 head_width=0.1, head_length=0.1, fc='black', ec='black')
        plt.text(components[i, 0]*3.2, components[i, 1]*3.2, feature,
                fontsize=12, ha='center', va='center')
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA Biplot - Iris Dataset')
    plt.legend()
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.show()

biplot(X_pca_2d, pca_2d.components_, feature_names, y)
```

### Dimensionality Reduction for Classification

```python
# Compare classification performance with and without PCA
def compare_with_without_pca(X, y, n_components=2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Without PCA
    rf_original = RandomForestClassifier(random_state=42)
    rf_original.fit(X_train_scaled, y_train)
    y_pred_original = rf_original.predict(X_test_scaled)
    accuracy_original = accuracy_score(y_test, y_pred_original)
    
    # With PCA
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    rf_pca = RandomForestClassifier(random_state=42)
    rf_pca.fit(X_train_pca, y_train)
    y_pred_pca = rf_pca.predict(X_test_pca)
    accuracy_pca = accuracy_score(y_test, y_pred_pca)
    
    print(f"Results Comparison:")
    print(f"Original features ({X.shape[1]}): {accuracy_original:.3f}")
    print(f"PCA features ({n_components}): {accuracy_pca:.3f}")
    print(f"Variance explained by PCA: {pca.explained_variance_ratio_.sum():.3f}")
    print(f"Dimensionality reduction: {X.shape[1]} -> {n_components} " +
          f"({(1 - n_components/X.shape[1])*100:.1f}% reduction)")

compare_with_without_pca(X_scaled, y, n_components=2)
```

## � From Scratch Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

class PCAFromScratch:
    def __init__(self, n_components=None):
        """
        Principal Component Analysis implementation from scratch
        
        Parameters:
        n_components: Number of components to keep (if None, keep all)
        """
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.singular_values_ = None
        
    def fit(self, X):
        """
        Fit PCA on the training data
        
        Parameters:
        X: Training data of shape (n_samples, n_features)
        """
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Compute covariance matrix
        n_samples = X.shape[0]
        covariance_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Sort by decreasing eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store results
        if self.n_components is None:
            self.n_components = len(eigenvalues)
        
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = (
            self.explained_variance_ / np.sum(eigenvalues)
        )
        
        # For compatibility with sklearn
        self.singular_values_ = np.sqrt(self.explained_variance_ * (n_samples - 1))
        
        return self
    
    def transform(self, X):
        """
        Transform the data to the principal component space
        
        Parameters:
        X: Data to transform of shape (n_samples, n_features)
        
        Returns:
        X_transformed: Transformed data of shape (n_samples, n_components)
        """
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)
    
    def fit_transform(self, X):
        """
        Fit PCA and transform the data
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_transformed):
        """
        Transform the data back to original space (reconstruction)
        
        Parameters:
        X_transformed: Data in PC space of shape (n_samples, n_components)
        
        Returns:
        X_reconstructed: Reconstructed data in original space
        """
        return np.dot(X_transformed, self.components_) + self.mean_
    
    def get_covariance(self):
        """
        Get the covariance matrix of the data in PC space
        """
        return np.dot(self.components_ * self.explained_variance_,
                     self.components_.T)

# Demonstration with synthetic data
np.random.seed(42)

# Create correlated 2D data
mean = [0, 0]
cov = [[3, 2.5], [2.5, 3]]
X_synthetic = np.random.multivariate_normal(mean, cov, 300)

# Apply custom PCA
pca_custom = PCAFromScratch(n_components=2)
X_pca_custom = pca_custom.fit_transform(X_synthetic)

# Compare with sklearn
from sklearn.decomposition import PCA
pca_sklearn = PCA(n_components=2)
X_pca_sklearn = pca_sklearn.fit_transform(X_synthetic)

print("Custom PCA Results:")
print("Explained variance ratio:", pca_custom.explained_variance_ratio_)
print("Components shape:", pca_custom.components_.shape)

print("\nSklearn PCA Results:")
print("Explained variance ratio:", pca_sklearn.explained_variance_ratio_)
print("Components shape:", pca_sklearn.components_.shape)

print("\nDifference in results (should be close to zero):")
print("Explained variance ratio diff:", 
      np.abs(pca_custom.explained_variance_ratio_ - 
             pca_sklearn.explained_variance_ratio_).max())

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original data
axes[0].scatter(X_synthetic[:, 0], X_synthetic[:, 1], alpha=0.7)
axes[0].set_title('Original Data')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].grid(True)

# PCA transformed data
axes[1].scatter(X_pca_custom[:, 0], X_pca_custom[:, 1], alpha=0.7)
axes[1].set_title('PCA Transformed Data')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].grid(True)

# Original data with principal components
axes[2].scatter(X_synthetic[:, 0], X_synthetic[:, 1], alpha=0.7)
mean_point = pca_custom.mean_

# Plot principal component directions
for i in range(2):
    direction = pca_custom.components_[i] * 3 * np.sqrt(pca_custom.explained_variance_[i])
    axes[2].arrow(mean_point[0], mean_point[1], 
                  direction[0], direction[1],
                  head_width=0.2, head_length=0.3, 
                  fc=f'C{i+1}', ec=f'C{i+1}', linewidth=2,
                  label=f'PC{i+1}')

axes[2].set_title('Original Data with Principal Components')
axes[2].set_xlabel('Feature 1')
axes[2].set_ylabel('Feature 2')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()
```

### Advanced Features Implementation

```python
def reconstruction_error_analysis(X, max_components=None):
    """
    Analyze reconstruction error vs number of components
    """
    if max_components is None:
        max_components = min(X.shape) - 1
    
    errors = []
    components_range = range(1, max_components + 1)
    
    for n_comp in components_range:
        pca = PCAFromScratch(n_components=n_comp)
        X_transformed = pca.fit_transform(X)
        X_reconstructed = pca.inverse_transform(X_transformed)
        
        # Calculate reconstruction error (mean squared error)
        error = np.mean((X - X_reconstructed) ** 2)
        errors.append(error)
    
    return components_range, errors

# Example with iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
X_iris = iris.data

# Standardize
X_iris_scaled = (X_iris - np.mean(X_iris, axis=0)) / np.std(X_iris, axis=0)

# Analyze reconstruction error
components, errors = reconstruction_error_analysis(X_iris_scaled, max_components=4)

plt.figure(figsize=(10, 6))
plt.plot(components, errors, 'bo-', linewidth=2, markersize=8)
plt.title('Reconstruction Error vs Number of Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Mean Squared Reconstruction Error')
plt.grid(True)
plt.xticks(components)
plt.show()

print("Reconstruction Errors:")
for comp, error in zip(components, errors):
    print(f"{comp} components: {error:.6f}")
```

## � Assumptions and Limitations

### Key Assumptions

1. **Linear relationships**: PCA assumes linear relationships between variables
2. **Variance equals importance**: Higher variance directions are assumed to be more important
3. **Orthogonal components**: Principal components are orthogonal (perpendicular)
4. **Gaussian distribution**: Works best with normally distributed data
5. **Standardization**: Features should be on similar scales (usually requires standardization)

### Limitations

1. **Linear transformation only**: Cannot capture non-linear relationships
   - **Solution**: Use Kernel PCA or other non-linear techniques

2. **Interpretability loss**: Principal components are linear combinations of original features
   - **Solution**: Use factor analysis or sparse PCA for more interpretable components

3. **Sensitive to scaling**: Features with larger scales dominate the principal components
   - **Solution**: Always standardize features before applying PCA

4. **Information loss**: Dimensionality reduction inherently loses some information
   - **Assessment**: Monitor explained variance ratio and reconstruction error

5. **Outlier sensitivity**: Outliers can significantly affect principal components
   - **Solution**: Use robust PCA variants or outlier detection/removal

6. **No guarantee of class separation**: PCA maximizes variance, not class separability
   - **Alternative**: Use Linear Discriminant Analysis (LDA) for classification tasks

### Comparison with Other Techniques

| Method | Linear | Supervised | Interpretable | Non-linear |
|--------|--------|------------|---------------|------------|
| PCA |  |  | Partial |  |
| LDA |  |  | Partial |  |
| t-SNE |  |  |  |  |
| UMAP |  |  |  |  |
| Factor Analysis |  |  |  |  |
| ICA |  |  |  |  |

**When to avoid PCA:**
- When original features have clear business meaning that must be preserved
- With categorical or ordinal data without proper encoding
- When non-linear relationships are important
- With very sparse data (consider specialized sparse PCA)
- When you need exactly interpretable features for regulatory compliance

## ❓ Interview Questions

??? question "What is the mathematical intuition behind PCA and how does it work?"

    **Answer:** PCA finds the directions (principal components) in the data that capture the maximum variance. Mathematically, it performs eigenvalue decomposition on the covariance matrix:
    
    1. **Center the data**: Subtract the mean from each feature
    2. **Compute covariance matrix**: C = (X^T * X) / (n-1)
    3. **Find eigenvalues and eigenvectors**: C*v = �*v
    4. **Sort by eigenvalues**: Largest eigenvalues correspond to directions with most variance
    5. **Project data**: Transform original data onto selected eigenvectors
    
    The key insight is that eigenvectors of the covariance matrix are orthogonal directions of maximum variance, and eigenvalues represent the amount of variance explained by each direction.

??? question "Why do we need to standardize features before applying PCA?"

    **Answer:** Features must be standardized because PCA is sensitive to the scale of variables:
    
    - **Scale dominance**: Features with larger scales (e.g., income in dollars vs age in years) will dominate the principal components
    - **Variance bias**: PCA maximizes variance, so large-scale features appear to have more "importance"
    - **Covariance matrix distortion**: The covariance matrix will be dominated by high-variance features
    
    **Example**: Without standardization, if you have height (cm, ~170) and weight (kg, ~70), height will dominate simply due to larger numerical values, not because it's more important.
    
    **Solution**: Use z-score standardization: (x - �) / � for each feature.

??? question "How do you choose the optimal number of principal components?"

    **Answer:** Several methods exist for selecting the number of components:
    
    1. **Explained Variance Threshold**: Keep components explaining 80-95% of variance
    2. **Elbow Method**: Plot explained variance vs components, look for "elbow" point
    3. **Kaiser Rule**: Keep components with eigenvalues > 1 (for standardized data)
    4. **Scree Plot**: Visual inspection of eigenvalue decay
    5. **Cross-validation**: Use downstream task performance to select optimal number
    6. **Business requirements**: Based on computational constraints or interpretability needs
    
    **Code example**:
    ```python
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumsum_var >= 0.95) + 1  # 95% variance
    ```

??? question "What's the difference between PCA and Linear Discriminant Analysis (LDA)?"

    **Answer:** Key differences:
    
    | Aspect | PCA | LDA |
    |--------|-----|-----|
    | **Type** | Unsupervised | Supervised |
    | **Objective** | Maximize variance | Maximize class separation |
    | **Input** | Features only | Features + labels |
    | **Components** | Up to min(n_features, n_samples) | Up to (n_classes - 1) |
    | **Use case** | Dimensionality reduction | Classification preprocessing |
    
    **When to use each**:
    - **PCA**: Data exploration, compression, noise reduction, visualization
    - **LDA**: Classification tasks, when you want to maximize class separability
    
    **Example**: For 3-class iris dataset, LDA can find at most 2 components, while PCA can find up to 4.

??? question "How do you interpret the principal components and their loadings?"

    **Answer:** Principal components and loadings provide insights into data structure:
    
    **Loadings (Component coefficients)**:
    - Show contribution of each original feature to each PC
    - Values range typically from -1 to 1
    - Large absolute values indicate strong influence
    
    **Interpretation steps**:
    1. **Examine loading values**: Which features contribute most to each PC?
    2. **Look for patterns**: Do related features load together?
    3. **Name components**: Based on dominant features (e.g., "size factor", "ratio factor")
    
    **Example interpretation**:
    ```
    PC1 loadings: [0.8 height, 0.7 weight, 0.1 age] � "Physical size factor"
    PC2 loadings: [0.2 height, -0.1 weight, 0.9 age] � "Age factor"
    ```

??? question "What are the limitations of PCA and when should you not use it?"

    **Answer:** Major limitations and alternatives:
    
    **Limitations**:
    1. **Linear only**: Cannot capture non-linear relationships � Use Kernel PCA, t-SNE
    2. **Variance ` Importance**: High variance doesn't always mean importance � Use domain knowledge
    3. **Loss of interpretability**: PCs are combinations of original features � Use Sparse PCA, Factor Analysis
    4. **Outlier sensitive**: Outliers can skew components � Use Robust PCA
    5. **No class consideration**: Doesn't consider target variable � Use LDA for classification
    
    **When NOT to use PCA**:
    - Categorical data without proper encoding
    - When original features must be preserved (regulatory requirements)
    - Very sparse data (many zeros)
    - Non-linear relationships are crucial
    - Small datasets (overfitting risk)

??? question "How do you handle missing values when applying PCA?"

    **Answer:** Several strategies for missing data in PCA:
    
    **1. Complete Case Analysis**:
    ```python
    # Remove rows with any missing values
    X_complete = X.dropna()
    ```
    
    **2. Imputation before PCA**:
    ```python
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    ```
    
    **3. Iterative Imputation**:
    ```python
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    imputer = IterativeImputer()
    X_imputed = imputer.fit_transform(X)
    ```
    
    **4. PCA with missing values** (specialized methods):
    - Use algorithms like NIPALS (Nonlinear Iterative Partial Least Squares)
    - Probabilistic PCA that handles missing values directly
    
    **Best practice**: Analyze missing data patterns first, then choose appropriate strategy based on data characteristics.

??? question "Explain the relationship between PCA and Singular Value Decomposition (SVD)."

    **Answer:** PCA and SVD are mathematically related:
    
    **SVD decomposition** of centered data matrix X:
    ```
    X = U * � * V^T
    ```
    Where:
    - U: Left singular vectors
    - �: Singular values (diagonal matrix)
    - V: Right singular vectors
    
    **Connection to PCA**:
    - **Principal components** = columns of V
    - **Explained variance** = (singular values)� / (n-1)
    - **Transformed data** = U * �
    
    **Advantages of SVD approach**:
    1. More numerically stable
    2. Computationally efficient for tall matrices
    3. Doesn't require computing covariance matrix explicitly
    4. Better for sparse data
    
    **Implementation**:
    ```python
    # Using SVD for PCA
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    components = Vt  # Principal components
    explained_variance = (s ** 2) / (n - 1)
    ```

??? question "How do you evaluate the quality of PCA results?"

    **Answer:** Multiple metrics assess PCA quality:
    
    **1. Explained Variance Ratio**:
    ```python
    total_variance_explained = sum(pca.explained_variance_ratio_)
    print(f"Total variance explained: {total_variance_explained:.3f}")
    ```
    
    **2. Reconstruction Error**:
    ```python
    X_reconstructed = pca.inverse_transform(X_pca)
    mse = np.mean((X_original - X_reconstructed) ** 2)
    ```
    
    **3. Silhouette Score** (if labels available):
    ```python
    from sklearn.metrics import silhouette_score
    score = silhouette_score(X_pca, labels)
    ```
    
    **4. Downstream Task Performance**:
    - Compare classifier accuracy before/after PCA
    - Monitor if important patterns are preserved
    
    **5. Visual Assessment**:
    - Scree plots for eigenvalue decay
    - Biplots for feature relationships
    - 2D/3D scatter plots for cluster visualization
    
    **Quality indicators**:
    -  First few PCs explain >80% variance
    -  Smooth eigenvalue decay (no sudden drops)
    -  Components are interpretable
    -  Downstream performance maintained

## >� Examples

### Real-world Example: Image Compression with PCA

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA

# Load face images dataset
faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X_faces = faces.data
y_faces = faces.target

print(f"Dataset shape: {X_faces.shape}")
print(f"Original image dimensions: {int(np.sqrt(X_faces.shape[1]))}x{int(np.sqrt(X_faces.shape[1]))}")

# Apply PCA with different numbers of components
n_components_list = [10, 50, 100, 200, 400]
fig, axes = plt.subplots(2, len(n_components_list) + 1, figsize=(18, 6))

# Original image
sample_idx = 0
original_image = X_faces[sample_idx].reshape(64, 64)
axes[0, 0].imshow(original_image, cmap='gray')
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

# Show compression ratios
compression_ratios = []
reconstruction_errors = []

for i, n_comp in enumerate(n_components_list):
    # Apply PCA
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_faces)
    X_reconstructed = pca.inverse_transform(X_pca)
    
    # Calculate compression ratio and error
    original_size = X_faces.shape[1]  # 4096 pixels
    compressed_size = n_comp + original_size * n_comp  # components + loadings
    compression_ratio = original_size / compressed_size
    
    mse = np.mean((X_faces - X_reconstructed) ** 2)
    
    compression_ratios.append(compression_ratio)
    reconstruction_errors.append(mse)
    
    # Display reconstructed image
    reconstructed_image = X_reconstructed[sample_idx].reshape(64, 64)
    axes[0, i + 1].imshow(reconstructed_image, cmap='gray')
    axes[0, i + 1].set_title(f'{n_comp} PCs\n({pca.explained_variance_ratio_.sum():.2f} var)')
    axes[0, i + 1].axis('off')

# Plot metrics
axes[1, 0].remove()  # Remove empty subplot
axes[1, 1].bar(range(len(n_components_list)), compression_ratios)
axes[1, 1].set_title('Compression Ratio')
axes[1, 1].set_xlabel('Number of Components')
axes[1, 1].set_xticks(range(len(n_components_list)))
axes[1, 1].set_xticklabels(n_components_list)

axes[1, 2].plot(n_components_list, reconstruction_errors, 'ro-')
axes[1, 2].set_title('Reconstruction Error')
axes[1, 2].set_xlabel('Number of Components')
axes[1, 2].set_ylabel('MSE')

# Explained variance
axes[1, 3].remove()
axes[1, 4].remove()
ax_combined = plt.subplot(2, len(n_components_list) + 1, (2*len(n_components_list) + 4, 2*len(n_components_list) + 6))

# Plot cumulative explained variance
pca_full = PCA()
pca_full.fit(X_faces)
cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)

ax_combined.plot(range(1, min(201, len(cumsum_var) + 1)), 
                cumsum_var[:200], 'b-', linewidth=2)
ax_combined.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
ax_combined.set_title('Cumulative Explained Variance')
ax_combined.set_xlabel('Number of Components')
ax_combined.set_ylabel('Cumulative Variance')
ax_combined.legend()
ax_combined.grid(True)

plt.tight_layout()
plt.show()

# Print summary statistics
print("\nImage Compression Results:")
for n_comp, ratio, error in zip(n_components_list, compression_ratios, reconstruction_errors):
    print(f"{n_comp:3d} components: {ratio:4.1f}x compression, MSE: {error:.6f}")
```

### Market Analysis Example

```python
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Download stock data for tech companies
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
end_date = datetime.now()
start_date = end_date - timedelta(days=365*2)  # 2 years of data

# Download stock returns
stock_data = {}
for ticker in tickers:
    try:
        stock = yf.download(ticker, start=start_date, end=end_date)
        stock_data[ticker] = stock['Adj Close'].pct_change().dropna()
    except:
        print(f"Could not download data for {ticker}")

# Create returns dataframe
returns_df = pd.DataFrame(stock_data)
returns_df = returns_df.dropna()

print(f"Stock returns data shape: {returns_df.shape}")
print("\nBasic statistics:")
print(returns_df.describe())

# Apply PCA to stock returns
from sklearn.preprocessing import StandardScaler

# Standardize returns
scaler = StandardScaler()
returns_scaled = scaler.fit_transform(returns_df)

# Fit PCA
pca_stocks = PCA()
returns_pca = pca_stocks.fit_transform(returns_scaled)

# Analyze results
explained_var = pca_stocks.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

print(f"\nPCA Results for Stock Returns:")
print("Explained Variance by Component:")
for i, var in enumerate(explained_var[:5]):
    print(f"PC{i+1}: {var:.3f} ({var*100:.1f}%)")

print(f"\nFirst 3 components explain {cumulative_var[2]:.3f} ({cumulative_var[2]*100:.1f}%) of variance")

# Component interpretation
components_df = pd.DataFrame(
    pca_stocks.components_[:3].T,  # First 3 components
    columns=['PC1 (Market)', 'PC2 (Tech vs Value)', 'PC3 (Volatility)'],
    index=returns_df.columns
)

print("\nComponent Loadings (Stock Exposure to Factors):")
print(components_df.round(3))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Explained variance
axes[0, 0].bar(range(1, len(explained_var) + 1), explained_var)
axes[0, 0].set_title('Explained Variance by Component')
axes[0, 0].set_xlabel('Principal Component')
axes[0, 0].set_ylabel('Explained Variance Ratio')

# Cumulative variance
axes[0, 1].plot(range(1, len(cumulative_var) + 1), cumulative_var, 'bo-')
axes[0, 1].axhline(y=0.95, color='r', linestyle='--', label='95%')
axes[0, 1].set_title('Cumulative Explained Variance')
axes[0, 1].set_xlabel('Number of Components')
axes[0, 1].legend()

# Component loadings heatmap
import seaborn as sns
sns.heatmap(components_df.T, annot=True, cmap='coolwarm', center=0,
            ax=axes[1, 0], cbar_kws={'label': 'Loading'})
axes[1, 0].set_title('Component Loadings Heatmap')

# Factor scores over time
factor_scores = pd.DataFrame(returns_pca[:, :3], 
                           index=returns_df.index,
                           columns=['Market Factor', 'Style Factor', 'Volatility Factor'])

axes[1, 1].plot(factor_scores.index, factor_scores['Market Factor'], 
                label='Market Factor', alpha=0.7)
axes[1, 1].plot(factor_scores.index, factor_scores['Style Factor'], 
                label='Style Factor', alpha=0.7)
axes[1, 1].set_title('Principal Component Scores Over Time')
axes[1, 1].set_ylabel('Factor Score')
axes[1, 1].legend()
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Portfolio analysis using PCA
print("\nPortfolio Risk Analysis using PCA:")

# Risk contribution by component
portfolio_weights = np.ones(len(tickers)) / len(tickers)  # Equal weights
portfolio_return_std = np.dot(portfolio_weights, returns_df.std())

print(f"Portfolio return volatility: {portfolio_return_std:.4f}")

# Factor exposures
factor_exposures = np.dot(portfolio_weights, components_df.values)
print("Portfolio factor exposures:")
for i, exposure in enumerate(factor_exposures):
    print(f"  PC{i+1}: {exposure:.3f}")
```

## 📚 References

- **Books:**
  - [The Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/) by Hastie, Tibshirani, and Friedman - Chapter 14
  - [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) by Christopher Bishop - Chapter 12
  - [Introduction to Statistical Learning](https://www.statlearning.com/) by James, Witten, Hastie, and Tibshirani - Chapter 10

- **Documentation:**
  - [Scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
  - [Scikit-learn Decomposition Guide](https://scikit-learn.org/stable/modules/decomposition.html)
  - [NumPy Linear Algebra](https://numpy.org/doc/stable/reference/routines.linalg.html)

- **Tutorials:**
  - [PCA Explained Visually](https://setosa.io/ev/principal-component-analysis/)
  - [A Tutorial on Principal Component Analysis](https://arxiv.org/abs/1404.1100)
  - [PCA with Python Tutorial](https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60)

- **Research Papers:**
  - Pearson, K. (1901). On Lines and Planes of Closest Fit to Systems of Points in Space
  - Hotelling, H. (1933). Analysis of a complex of statistical variables into principal components
  - Jolliffe, I.T. (2002). Principal Component Analysis, Second Edition

- **Advanced Topics:**
  - [Kernel PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html)
  - [Sparse PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html)
  - [Incremental PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html)

- **Online Courses:**
  - [Machine Learning Course - Stanford CS229](http://cs229.stanford.edu/)
  - [Dimensionality Reduction - Coursera](https://www.coursera.org/learn/machine-learning)
  - [Advanced Machine Learning - edX](https://www.edx.org/course/machine-learning)