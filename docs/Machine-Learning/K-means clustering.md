---
title: K-means Clustering
description: Comprehensive guide to K-means Clustering with mathematical intuition, implementations, and interview questions.
comments: true
---

# 📘 K-means Clustering

K-means is a popular unsupervised learning algorithm that partitions data into k clusters by grouping similar data points together and identifying underlying patterns in the data.

**Resources:** [Scikit-learn K-means](https://scikit-learn.org/stable/modules/clustering.html#k-means) | [Pattern Recognition and Machine Learning - Chapter 9](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf)

## ✍️ Summary

K-means clustering is an unsupervised machine learning algorithm that aims to partition n observations into k clusters where each observation belongs to the cluster with the nearest centroid (cluster center). It's one of the simplest and most popular clustering algorithms.

**Key characteristics:**
- **Unsupervised**: No labeled data required
- **Centroid-based**: Uses cluster centers to define clusters
- **Hard clustering**: Each point belongs to exactly one cluster
- **Iterative**: Uses an expectation-maximization approach
- **Distance-based**: Uses Euclidean distance (typically)

**Applications:**
- Customer segmentation in marketing
- Image segmentation and compression  
- Market research and analysis
- Data preprocessing for other ML algorithms
- Gene sequencing analysis
- Recommendation systems

**When to use K-means:**
- When you know the approximate number of clusters
- When clusters are roughly spherical and similar sized
- When you need interpretable results
- When computational efficiency is important

## 🧠 Intuition

### How K-means Works

K-means follows a simple iterative process:

1. **Initialize** k cluster centroids randomly
2. **Assign** each data point to the nearest centroid
3. **Update** centroids by calculating the mean of assigned points
4. **Repeat** steps 2-3 until centroids stop moving significantly

### Mathematical Foundation

#### 1. Objective Function

K-means minimizes the Within-Cluster Sum of Squares (WCSS):

$$J = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2$$

Where:
- $J$ is the objective function to minimize
- $k$ is the number of clusters
- $C_i$ is the set of points in cluster $i$
- $\mu_i$ is the centroid of cluster $i$
- $||x - \mu_i||^2$ is the squared Euclidean distance

#### 2. Centroid Update Rule

The centroid of cluster $i$ is updated as:

$$\mu_i = \frac{1}{|C_i|} \sum_{x \in C_i} x$$

Where $|C_i|$ is the number of points in cluster $i$.

#### 3. Distance Calculation

Euclidean distance between point $x$ and centroid $\mu_i$:

$$d(x, \mu_i) = \sqrt{\sum_{j=1}^{d} (x_j - \mu_{i,j})^2}$$

Where $d$ is the number of dimensions.

#### 4. Convergence Criteria

The algorithm stops when:
- Centroids don't move significantly: $||\mu_i^{(t+1)} - \mu_i^{(t)}|| < \epsilon$
- Maximum number of iterations reached
- No points change cluster assignments

### Algorithm Complexity

- **Time Complexity**: $O(n \cdot k \cdot d \cdot t)$
  - $n$: number of data points
  - $k$: number of clusters  
  - $d$: number of dimensions
  - $t$: number of iterations

- **Space Complexity**: $O(n \cdot d + k \cdot d)$

## 🔢 Implementation using Libraries

### Using Scikit-learn

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA

# Generate sample data
np.random.seed(42)
X, y_true = make_blobs(
    n_samples=300, 
    centers=4, 
    n_features=2, 
    cluster_std=0.8,
    random_state=42
)

# Basic K-means clustering
kmeans = KMeans(
    n_clusters=4,
    init='k-means++',      # Smart initialization
    n_init=10,             # Number of initializations
    max_iter=300,          # Maximum iterations
    tol=1e-4,              # Convergence tolerance
    random_state=42
)

# Fit the model
kmeans.fit(X)

# Get predictions and centroids
y_pred = kmeans.labels_
centroids = kmeans.cluster_centers_
inertia = kmeans.inertia_  # WCSS value

print(f"Inertia (WCSS): {inertia:.2f}")
print(f"Silhouette Score: {silhouette_score(X, y_pred):.3f}")

# Visualize results
plt.figure(figsize=(15, 5))

# Original data with true labels
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.7)
plt.title('True Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# K-means results
plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
plt.scatter(centroids[:, 0], centroids[:, 1], 
           marker='x', s=200, linewidths=3, color='red')
plt.title('K-means Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Comparison
plt.subplot(1, 3, 3)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.5, label='True')
plt.scatter(centroids[:, 0], centroids[:, 1], 
           marker='x', s=200, linewidths=3, color='red', label='Centroids')
plt.title('Comparison')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.tight_layout()
plt.show()

# Calculate accuracy (for labeled data comparison)
accuracy = adjusted_rand_score(y_true, y_pred)
print(f"Adjusted Rand Index: {accuracy:.3f}")
```

### Determining Optimal Number of Clusters

#### Elbow Method

```python
def plot_elbow_method(X, max_k=10):
    """Plot elbow method to find optimal k"""
    wcss = []
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, wcss, 'bo-')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.grid(True, alpha=0.3)
    
    # Find elbow point using second derivative
    second_derivative = np.diff(wcss, 2)
    elbow_point = np.argmax(second_derivative) + 2
    plt.axvline(x=elbow_point, color='red', linestyle='--', 
                label=f'Elbow at k={elbow_point}')
    plt.legend()
    plt.show()
    
    return wcss

wcss_values = plot_elbow_method(X, max_k=10)
```

#### Silhouette Method

```python
def plot_silhouette_method(X, max_k=10):
    """Plot silhouette scores for different k values"""
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_scores, 'go-')
    plt.title('Silhouette Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.grid(True, alpha=0.3)
    
    # Find optimal k
    optimal_k = k_range[np.argmax(silhouette_scores)]
    plt.axvline(x=optimal_k, color='red', linestyle='--', 
                label=f'Optimal k={optimal_k}')
    plt.legend()
    plt.show()
    
    return silhouette_scores

silhouette_values = plot_silhouette_method(X, max_k=10)
```

### Real-world Example: Iris Dataset

```python
# Load Iris dataset
iris = load_iris()
X_iris = iris.data
y_true_iris = iris.target

# Standardize features
scaler = StandardScaler()
X_iris_scaled = scaler.fit_transform(X_iris)

# Apply K-means
kmeans_iris = KMeans(n_clusters=3, random_state=42, n_init=10)
y_pred_iris = kmeans_iris.fit_predict(X_iris_scaled)

# Evaluate
silhouette_iris = silhouette_score(X_iris_scaled, y_pred_iris)
ari_iris = adjusted_rand_score(y_true_iris, y_pred_iris)

print(f"Iris Dataset Results:")
print(f"Silhouette Score: {silhouette_iris:.3f}")
print(f"Adjusted Rand Index: {ari_iris:.3f}")

# Visualize using PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_iris_scaled)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true_iris, cmap='viridis', alpha=0.7)
plt.title('True Species (PCA)')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')

plt.subplot(1, 3, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred_iris, cmap='viridis', alpha=0.7)
plt.title('K-means Clusters (PCA)')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')

# Feature importance
feature_names = iris.feature_names
centroids_original = scaler.inverse_transform(kmeans_iris.cluster_centers_)

plt.subplot(1, 3, 3)
for i, centroid in enumerate(centroids_original):
    plt.plot(feature_names, centroid, 'o-', label=f'Cluster {i}')
plt.title('Cluster Centroids (Original Features)')
plt.xlabel('Features')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Print cluster characteristics
print("\nCluster Centroids (Original Scale):")
for i, centroid in enumerate(centroids_original):
    print(f"Cluster {i}:")
    for feature, value in zip(feature_names, centroid):
        print(f"  {feature}: {value:.2f}")
    print()
```

### Mini-batch K-means for Large Datasets

```python
from sklearn.cluster import MiniBatchKMeans
import time

# Generate larger dataset
X_large, _ = make_blobs(n_samples=10000, centers=5, n_features=10, random_state=42)

# Compare standard K-means vs Mini-batch K-means
print("Comparing K-means vs Mini-batch K-means:")

# Standard K-means
start_time = time.time()
kmeans_standard = KMeans(n_clusters=5, random_state=42, n_init=10)
labels_standard = kmeans_standard.fit_predict(X_large)
time_standard = time.time() - start_time

# Mini-batch K-means
start_time = time.time()
kmeans_mini = MiniBatchKMeans(n_clusters=5, batch_size=100, random_state=42)
labels_mini = kmeans_mini.fit_predict(X_large)
time_mini = time.time() - start_time

print(f"Standard K-means - Time: {time_standard:.3f}s, Inertia: {kmeans_standard.inertia_:.2f}")
print(f"Mini-batch K-means - Time: {time_mini:.3f}s, Inertia: {kmeans_mini.inertia_:.2f}")
print(f"Speedup: {time_standard/time_mini:.1f}x")

# Compare clustering quality
ari_comparison = adjusted_rand_score(labels_standard, labels_mini)
print(f"Agreement between methods (ARI): {ari_comparison:.3f}")
```

## ⚙️ From Scratch Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

class KMeansFromScratch:
    """K-means clustering implementation from scratch"""
    
    def __init__(self, k=3, max_iters=100, tol=1e-4, init='k-means++', random_state=None):
        """
        Initialize K-means clusterer
        
        Parameters:
        -----------
        k : int, default=3
            Number of clusters
        max_iters : int, default=100
            Maximum number of iterations
        tol : float, default=1e-4
            Tolerance for convergence
        init : str, default='k-means++'
            Initialization method ('random' or 'k-means++')
        random_state : int, default=None
            Random seed for reproducibility
        """
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.init = init
        self.random_state = random_state
        
        # Initialize attributes
        self.centroids = None
        self.labels = None
        self.inertia_ = None
        self.n_iter_ = 0
        
    def _initialize_centroids(self, X):
        """Initialize centroids using specified method"""
        if self.random_state:
            np.random.seed(self.random_state)
            
        n_samples, n_features = X.shape
        
        if self.init == 'random':
            # Random initialization
            min_vals = np.min(X, axis=0)
            max_vals = np.max(X, axis=0)
            centroids = np.random.uniform(min_vals, max_vals, (self.k, n_features))
            
        elif self.init == 'k-means++':
            # K-means++ initialization for better initial centroids
            centroids = np.zeros((self.k, n_features))
            
            # Choose first centroid randomly
            centroids[0] = X[np.random.randint(n_samples)]
            
            for i in range(1, self.k):
                # Calculate distances from each point to nearest centroid
                distances = np.array([min([np.linalg.norm(x - c)**2 for c in centroids[:i]]) 
                                    for x in X])
                
                # Choose next centroid with probability proportional to squared distance
                probabilities = distances / distances.sum()
                cumulative_probabilities = probabilities.cumsum()
                r = np.random.rand()
                
                for j, p in enumerate(cumulative_probabilities):
                    if r < p:
                        centroids[i] = X[j]
                        break
        else:
            raise ValueError("init must be 'random' or 'k-means++'")
            
        return centroids
    
    def _assign_clusters(self, X, centroids):
        """Assign each point to the nearest centroid"""
        # Calculate distances from each point to each centroid
        distances = cdist(X, centroids, metric='euclidean')
        
        # Assign each point to the nearest centroid
        labels = np.argmin(distances, axis=1)
        
        return labels
    
    def _update_centroids(self, X, labels):
        """Update centroids based on current cluster assignments"""
        centroids = np.zeros((self.k, X.shape[1]))
        
        for i in range(self.k):
            # Find points belonging to cluster i
            cluster_points = X[labels == i]
            
            if len(cluster_points) > 0:
                # Update centroid as mean of cluster points
                centroids[i] = np.mean(cluster_points, axis=0)
            else:
                # Keep old centroid if no points assigned to cluster
                centroids[i] = self.centroids[i]
                
        return centroids
    
    def _calculate_inertia(self, X, labels, centroids):
        """Calculate within-cluster sum of squares (inertia)"""
        inertia = 0
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centroids[i])**2)
        return inertia
    
    def fit(self, X):
        """Fit K-means clustering to data"""
        # Initialize centroids
        self.centroids = self._initialize_centroids(X)
        
        # Store convergence history
        self.centroid_history = [self.centroids.copy()]
        self.inertia_history = []
        
        for iteration in range(self.max_iters):
            # Assign points to clusters
            labels = self._assign_clusters(X, self.centroids)
            
            # Update centroids
            new_centroids = self._update_centroids(X, labels)
            
            # Calculate inertia
            inertia = self._calculate_inertia(X, labels, new_centroids)
            self.inertia_history.append(inertia)
            
            # Check for convergence
            centroid_shift = np.linalg.norm(new_centroids - self.centroids)
            
            if centroid_shift < self.tol:
                self.n_iter_ = iteration + 1
                break
                
            # Update centroids
            self.centroids = new_centroids
            self.centroid_history.append(self.centroids.copy())
            
        # Final assignments
        self.labels = self._assign_clusters(X, self.centroids)
        self.inertia_ = self._calculate_inertia(X, self.labels, self.centroids)
        
        return self
    
    def predict(self, X):
        """Predict cluster labels for new data"""
        if self.centroids is None:
            raise ValueError("Model must be fitted before predicting")
            
        return self._assign_clusters(X, self.centroids)
    
    def fit_predict(self, X):
        """Fit model and return cluster labels"""
        self.fit(X)
        return self.labels
    
    def plot_convergence(self):
        """Plot convergence of inertia over iterations"""
        if not hasattr(self, 'inertia_history'):
            raise ValueError("Model must be fitted first")
            
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.inertia_history) + 1), self.inertia_history, 'b-o')
        plt.title('K-means Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Inertia (WCSS)')
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def plot_clusters(self, X):
        """Visualize clusters (works for 2D data)"""
        if X.shape[1] != 2:
            raise ValueError("Visualization only works for 2D data")
            
        if self.labels is None:
            raise ValueError("Model must be fitted first")
            
        plt.figure(figsize=(10, 8))
        
        # Plot points colored by cluster
        colors = plt.cm.viridis(np.linspace(0, 1, self.k))
        for i in range(self.k):
            cluster_points = X[self.labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       c=[colors[i]], alpha=0.7, label=f'Cluster {i}')
        
        # Plot centroids
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], 
                   c='red', marker='x', s=200, linewidths=3, label='Centroids')
        
        plt.title(f'K-means Clustering (k={self.k})')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# Example usage of custom implementation
print("Testing custom K-means implementation:")

# Generate sample data
np.random.seed(42)
X_test, y_true_test = make_blobs(n_samples=300, centers=3, n_features=2, 
                                cluster_std=1.0, random_state=42)

# Fit custom K-means
kmeans_custom = KMeansFromScratch(k=3, max_iters=100, init='k-means++', random_state=42)
labels_custom = kmeans_custom.fit_predict(X_test)

print(f"Custom K-means converged in {kmeans_custom.n_iter_} iterations")
print(f"Final inertia: {kmeans_custom.inertia_:.2f}")

# Compare with sklearn
from sklearn.cluster import KMeans
kmeans_sklearn = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_sklearn = kmeans_sklearn.fit_predict(X_test)

print(f"Sklearn K-means inertia: {kmeans_sklearn.inertia_:.2f}")
print(f"Agreement between implementations: {adjusted_rand_score(labels_custom, labels_sklearn):.3f}")

# Visualize results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_true_test, cmap='viridis', alpha=0.7)
plt.title('True Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(1, 3, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=labels_custom, cmap='viridis', alpha=0.7)
plt.scatter(kmeans_custom.centroids[:, 0], kmeans_custom.centroids[:, 1], 
           marker='x', s=200, linewidths=3, color='red')
plt.title('Custom K-means')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(1, 3, 3)
plt.scatter(X_test[:, 0], X_test[:, 1], c=labels_sklearn, cmap='viridis', alpha=0.7)
plt.scatter(kmeans_sklearn.cluster_centers_[:, 0], kmeans_sklearn.cluster_centers_[:, 1], 
           marker='x', s=200, linewidths=3, color='red')
plt.title('Sklearn K-means')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

# Plot convergence
kmeans_custom.plot_convergence()
```

## ⚠️ Assumptions and Limitations

### Assumptions

1. **Spherical Clusters**: K-means assumes clusters are roughly spherical and have similar sizes
2. **Similar Variance**: Clusters should have similar variance in all directions
3. **Isotropic Clusters**: Equal variance in all dimensions
4. **Fixed Number of Clusters**: You must specify k in advance
5. **Euclidean Distance**: Uses Euclidean distance metric (sensitive to scale)

### Limitations

1. **Sensitive to Initialization**: Can converge to local optima
2. **Requires Preprocessing**: Sensitive to feature scaling and outliers
3. **Difficulty with Non-spherical Clusters**: Performs poorly on elongated or irregular shapes
4. **Fixed k**: Need to know or estimate the number of clusters
5. **Sensitive to Outliers**: Outliers can significantly affect centroids
6. **Equal Cluster Size Assumption**: Tends to create clusters of similar sizes

### Comparison with Other Clustering Algorithms

| Algorithm | Advantages | Disadvantages | Best Use Cases |
|-----------|------------|---------------|----------------|
| **K-means** | Fast, simple, works well with spherical clusters | Requires k, sensitive to initialization, assumes spherical clusters | Customer segmentation, image compression |
| **Hierarchical** | No need to specify k, creates dendrogram | Slow O(n³), sensitive to noise | Small datasets, understanding cluster hierarchy |
| **DBSCAN** | Finds arbitrary shaped clusters, robust to outliers | Sensitive to parameters, struggles with varying densities | Anomaly detection, irregular shaped clusters |
| **Gaussian Mixture** | Soft clustering, handles elliptical clusters | More complex, requires knowing k | When cluster overlap is expected |

### When NOT to Use K-means

- **Non-spherical clusters**: Use DBSCAN or spectral clustering
- **Varying cluster densities**: Use DBSCAN
- **Unknown number of clusters**: Use hierarchical clustering or DBSCAN
- **Categorical data**: Use K-modes or mixed-type clustering
- **High-dimensional data**: Consider dimensionality reduction first

## 💡 Interview Questions

??? question "**1. Explain the K-means algorithm step by step.**"
    
    **Answer:**
    
    K-means follows these steps:
    
    1. **Initialization**: Choose k cluster centers (centroids) randomly or using k-means++
    2. **Assignment**: Assign each data point to the nearest centroid based on Euclidean distance
    3. **Update**: Recalculate centroids as the mean of all points assigned to each cluster
    4. **Convergence Check**: Repeat steps 2-3 until centroids stop moving significantly or max iterations reached
    
    **Mathematical formulation:**
    - Objective: Minimize $J = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2$
    - Centroid update: $\mu_i = \frac{1}{|C_i|} \sum_{x \in C_i} x$
    
    **Time complexity**: O(n·k·d·t) where n=samples, k=clusters, d=dimensions, t=iterations

??? question "**2. What are the main assumptions and limitations of K-means?**"
    
    **Answer:**
    
    **Assumptions:**
    - Clusters are spherical and have similar sizes
    - Features have similar variances (isotropic)
    - Number of clusters k is known
    - Data is continuous and suitable for Euclidean distance
    
    **Limitations:**
    - Sensitive to initialization (can converge to local optima)
    - Requires specifying k in advance
    - Assumes spherical clusters of similar size
    - Sensitive to outliers and feature scaling
    - Poor performance on non-convex clusters
    - Hard clustering (each point belongs to exactly one cluster)

??? question "**3. How do you determine the optimal number of clusters (k)?**"
    
    **Answer:**
    
    **Methods to determine optimal k:**
    
    1. **Elbow Method**: Plot WCSS vs k, look for the "elbow" point
    2. **Silhouette Method**: Choose k with highest average silhouette score
    3. **Gap Statistic**: Compare within-cluster dispersion with expected dispersion
    4. **Information Criteria**: Use AIC/BIC for model selection
    5. **Domain Knowledge**: Use business/domain expertise
    
    **Elbow Method Example:**
    ```python
    wcss = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k)
        wcss.append(kmeans.fit(X).inertia_)
    # Plot and look for elbow
    ```

??? question "**4. What is the difference between K-means and K-means++?**"
    
    **Answer:**
    
    **K-means++ is an initialization method, not a different algorithm:**
    
    **Standard K-means initialization:**
    - Randomly selects k points as initial centroids
    - Can lead to poor convergence and local optima
    - Results may vary significantly between runs
    
    **K-means++ initialization:**
    - First centroid chosen randomly
    - Subsequent centroids chosen with probability proportional to squared distance from nearest existing centroid
    - Provides better initial centroids, leading to better final clustering
    - More consistent results across multiple runs
    - Typically converges faster with better final objective value

??? question "**5. How does K-means handle outliers and what can you do about it?**"
    
    **Answer:**
    
    **How K-means handles outliers:**
    - Outliers significantly affect centroid positions since centroids are calculated as means
    - Can cause centroids to shift away from main cluster mass
    - May create clusters around outliers
    - Reduces overall clustering quality
    
    **Solutions:**
    
    1. **Preprocessing:**
       - Remove outliers using IQR, Z-score, or isolation forest
       - Use robust scaling instead of standard scaling
    
    2. **Alternative algorithms:**
       - Use K-medoids (uses medians instead of means)
       - Use DBSCAN (treats outliers as noise)
    
    3. **Outlier-aware variants:**
       - Trimmed K-means (removes certain percentage of farthest points)
       - Robust K-means with M-estimators

??? question "**6. Compare K-means with Hierarchical clustering.**"
    
    **Answer:**
    
    | Aspect | K-means | Hierarchical |
    |--------|---------|--------------|
    | **k specification** | Must specify k | No need to specify k |
    | **Time complexity** | O(nkdt) | O(n³) for agglomerative |
    | **Shape assumption** | Spherical clusters | Any shape |
    | **Scalability** | Good for large datasets | Poor for large datasets |
    | **Deterministic** | No (depends on initialization) | Yes |
    | **Output** | Flat partitioning | Dendrogram hierarchy |
    | **Interpretability** | Cluster centers | Hierarchy of merges |
    | **Memory usage** | Low | High O(n²) |

??? question "**7. What is the objective function of K-means and how is it optimized?**"
    
    **Answer:**
    
    **Objective Function (Within-Cluster Sum of Squares):**
    $$J = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2$$
    
    **Optimization:**
    - K-means uses Lloyd's algorithm (Expectation-Maximization)
    - **E-step**: Assign points to nearest centroids (minimize J w.r.t. cluster assignments)
    - **M-step**: Update centroids as cluster means (minimize J w.r.t. centroids)
    
    **Key properties:**
    - Guaranteed to converge (objective function decreases monotonically)
    - May converge to local minimum
    - Convergence to global optimum not guaranteed
    - Each step reduces or maintains the objective value

??? question "**8. How do you evaluate the quality of K-means clustering?**"
    
    **Answer:**
    
    **Internal Metrics (no ground truth needed):**
    
    1. **Silhouette Score**: Measures how similar points are to their own cluster vs other clusters
       - Range: [-1, 1], higher is better
    
    2. **Within-Cluster Sum of Squares (WCSS)**: Lower is better
    
    3. **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster variance
    
    **External Metrics (with ground truth):**
    
    1. **Adjusted Rand Index (ARI)**: Measures similarity to true clustering
    2. **Normalized Mutual Information (NMI)**: Information-theoretic measure
    3. **Fowlkes-Mallows Index**: Geometric mean of precision and recall
    
    **Example:**
    ```python
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    silhouette = silhouette_score(X, labels)
    ch_score = calinski_harabasz_score(X, labels)
    ```

??? question "**9. What is Mini-batch K-means and when would you use it?**"
    
    **Answer:**
    
    **Mini-batch K-means:**
    - Variant that uses small random batches of data for updates
    - Updates centroids using only a subset of data points in each iteration
    - Significantly faster than standard K-means
    - Slightly lower quality but much more scalable
    
    **When to use:**
    - Large datasets where standard K-means is too slow
    - Online/streaming data scenarios
    - When approximate results are acceptable
    - Limited computational resources
    
    **Trade-offs:**
    - **Pros**: Much faster, memory efficient, good for large datasets
    - **Cons**: Slightly less accurate, may need more iterations for convergence
    
    **Example:**
    ```python
    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=100)
    ```

??? question "**10. How does feature scaling affect K-means clustering?**"
    
    **Answer:**
    
    **Impact of feature scaling:**
    - K-means uses Euclidean distance, which is sensitive to feature scales
    - Features with larger scales dominate the distance calculation
    - Can lead to poor clustering where high-scale features determine clusters
    
    **Example:**
    ```python
    # Age: 20-80, Income: 20000-100000
    # Income will dominate distance calculation
    ```
    
    **Solutions:**
    
    1. **StandardScaler**: Mean=0, Std=1
       ```python
       from sklearn.preprocessing import StandardScaler
       scaler = StandardScaler()
       X_scaled = scaler.fit_transform(X)
       ```
    
    2. **MinMaxScaler**: Scale to [0,1]
       ```python
       from sklearn.preprocessing import MinMaxScaler
       scaler = MinMaxScaler()
       X_scaled = scaler.fit_transform(X)
       ```
    
    3. **RobustScaler**: Uses median and IQR (robust to outliers)
    
    **Best practice:** Always scale features before applying K-means

## 🧠 Examples

### Example 1: Customer Segmentation

```python
# Generate customer data
np.random.seed(42)
n_customers = 1000

# Features: Age, Income, Spending Score
ages = np.random.normal(40, 12, n_customers)
incomes = np.random.normal(60000, 20000, n_customers)
spending_scores = np.random.normal(50, 25, n_customers)

# Create DataFrame
customer_data = pd.DataFrame({
    'Age': ages,
    'Annual_Income': incomes,
    'Spending_Score': spending_scores
})

# Add some correlation (higher income -> higher spending for some customers)
mask = np.random.choice(n_customers, size=int(0.6 * n_customers), replace=False)
customer_data.loc[mask, 'Spending_Score'] += (customer_data.loc[mask, 'Annual_Income'] - 60000) / 1000

print("Customer Data Summary:")
print(customer_data.describe())

# Prepare data for clustering
X_customers = customer_data.values
scaler = StandardScaler()
X_customers_scaled = scaler.fit_transform(X_customers)

# Determine optimal number of clusters
def analyze_optimal_k(X, max_k=10):
    """Analyze optimal k using multiple methods"""
    wcss = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Elbow method
    ax1.plot(k_range, wcss, 'bo-')
    ax1.set_title('Elbow Method')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('WCSS')
    ax1.grid(True, alpha=0.3)
    
    # Silhouette method
    ax2.plot(k_range, silhouette_scores, 'ro-')
    ax2.set_title('Silhouette Analysis')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.grid(True, alpha=0.3)
    
    # Find optimal k
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
    ax2.axvline(x=optimal_k_silhouette, color='red', linestyle='--', 
                label=f'Optimal k={optimal_k_silhouette}')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    return optimal_k_silhouette, silhouette_scores

optimal_k, _ = analyze_optimal_k(X_customers_scaled, max_k=8)
print(f"Optimal number of clusters: {optimal_k}")

# Apply K-means with optimal k
kmeans_customers = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
customer_segments = kmeans_customers.fit_predict(X_customers_scaled)

# Add cluster labels to original data
customer_data['Segment'] = customer_segments

# Analyze segments
print(f"\nCustomer Segmentation Results (k={optimal_k}):")
print(f"Silhouette Score: {silhouette_score(X_customers_scaled, customer_segments):.3f}")

segment_analysis = customer_data.groupby('Segment').agg({
    'Age': ['mean', 'std'],
    'Annual_Income': ['mean', 'std'],
    'Spending_Score': ['mean', 'std'],
    'Segment': 'count'
}).round(2)

segment_analysis.columns = ['Age_Mean', 'Age_Std', 'Income_Mean', 'Income_Std', 
                           'Spending_Mean', 'Spending_Std', 'Count']
print("\nSegment Characteristics:")
print(segment_analysis)

# Visualize segments
fig = plt.figure(figsize=(18, 12))

# 2D scatter plots
feature_pairs = [
    ('Age', 'Annual_Income'),
    ('Age', 'Spending_Score'),
    ('Annual_Income', 'Spending_Score')
]

for i, (feat1, feat2) in enumerate(feature_pairs):
    ax = fig.add_subplot(2, 3, i+1)
    
    for segment in range(optimal_k):
        segment_data = customer_data[customer_data['Segment'] == segment]
        ax.scatter(segment_data[feat1], segment_data[feat2], 
                  alpha=0.6, label=f'Segment {segment}')
    
    ax.set_xlabel(feat1)
    ax.set_ylabel(feat2)
    ax.set_title(f'{feat1} vs {feat2}')
    ax.legend()
    ax.grid(True, alpha=0.3)

# 3D scatter plot
ax = fig.add_subplot(2, 3, 4, projection='3d')
colors = plt.cm.viridis(np.linspace(0, 1, optimal_k))

for segment in range(optimal_k):
    segment_data = customer_data[customer_data['Segment'] == segment]
    ax.scatter(segment_data['Age'], segment_data['Annual_Income'], 
              segment_data['Spending_Score'], c=[colors[segment]], 
              alpha=0.6, label=f'Segment {segment}')

ax.set_xlabel('Age')
ax.set_ylabel('Annual Income')
ax.set_zlabel('Spending Score')
ax.set_title('3D Customer Segments')
ax.legend()

# Segment size distribution
ax = fig.add_subplot(2, 3, 5)
segment_counts = customer_data['Segment'].value_counts().sort_index()
ax.bar(range(optimal_k), segment_counts.values, color=colors[:optimal_k])
ax.set_xlabel('Segment')
ax.set_ylabel('Number of Customers')
ax.set_title('Segment Size Distribution')
ax.set_xticks(range(optimal_k))

# Radar chart for segment characteristics
ax = fig.add_subplot(2, 3, 6, projection='polar')

# Normalize features for radar chart
centroids_original = scaler.inverse_transform(kmeans_customers.cluster_centers_)
features = ['Age', 'Annual_Income', 'Spending_Score']

# Normalize each feature to 0-1 scale for visualization
normalized_centroids = np.zeros_like(centroids_original)
for i, feature in enumerate(features):
    min_val = customer_data[feature].min()
    max_val = customer_data[feature].max()
    normalized_centroids[:, i] = (centroids_original[:, i] - min_val) / (max_val - min_val)

angles = np.linspace(0, 2*np.pi, len(features), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

for segment in range(optimal_k):
    values = normalized_centroids[segment].tolist()
    values += values[:1]  # Complete the circle
    ax.plot(angles, values, 'o-', linewidth=2, label=f'Segment {segment}')
    ax.fill(angles, values, alpha=0.25)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(features)
ax.set_title('Segment Characteristics (Normalized)')
ax.legend()

plt.tight_layout()
plt.show()

# Business insights
print("\nBusiness Insights:")
for segment in range(optimal_k):
    segment_data = segment_analysis.loc[segment]
    print(f"\nSegment {segment} ({segment_data['Count']} customers):")
    print(f"  Average Age: {segment_data['Age_Mean']:.1f} years")
    print(f"  Average Income: ${segment_data['Income_Mean']:,.0f}")
    print(f"  Average Spending Score: {segment_data['Spending_Mean']:.1f}")
    
    # Generate insights based on characteristics
    if segment_data['Age_Mean'] < 35 and segment_data['Spending_Mean'] > 60:
        print(f"  → Young high spenders - target for premium products")
    elif segment_data['Income_Mean'] > 70000 and segment_data['Spending_Mean'] < 40:
        print(f"  → High income, low spending - potential for marketing campaigns")
    elif segment_data['Age_Mean'] > 50 and segment_data['Spending_Mean'] > 60:
        print(f"  → Mature high spenders - focus on quality and service")
    else:
        print(f"  → Standard customers - balanced approach")
```

### Example 2: Image Color Quantization

```python
from sklearn.datasets import load_sample_image
import matplotlib.image as mpimg

def quantize_image_colors(image_path=None, n_colors=8):
    """Reduce number of colors in an image using K-means"""
    
    # Load image (use sample image if path not provided)
    if image_path is None:
        # Use sklearn sample image
        china = load_sample_image("china.jpg")
        image = china / 255.0  # Normalize to [0, 1]
    else:
        image = mpimg.imread(image_path)
        if image.max() > 1:
            image = image / 255.0
    
    print(f"Original image shape: {image.shape}")
    
    # Reshape image to be a list of pixels
    original_shape = image.shape
    image_2d = image.reshape(-1, 3)  # Flatten to (n_pixels, 3)
    
    print(f"Number of pixels: {image_2d.shape[0]:,}")
    print(f"Original colors: {len(np.unique(image_2d, axis=0)):,} unique colors")
    
    # Apply K-means clustering
    print(f"Reducing to {n_colors} colors using K-means...")
    
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    labels = kmeans.fit_predict(image_2d)
    
    # Replace each pixel with its cluster center
    quantized_colors = kmeans.cluster_centers_[labels]
    quantized_image = quantized_colors.reshape(original_shape)
    
    # Calculate compression ratio
    original_unique_colors = len(np.unique(image_2d, axis=0))
    compression_ratio = original_unique_colors / n_colors
    
    print(f"Compression ratio: {compression_ratio:.1f}x")
    print(f"Final colors: {n_colors}")
    
    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title(f'Original Image\n({original_unique_colors:,} colors)')
    axes[0, 0].axis('off')
    
    # Quantized image
    axes[0, 1].imshow(quantized_image)
    axes[0, 1].set_title(f'Quantized Image\n({n_colors} colors)')
    axes[0, 1].axis('off')
    
    # Difference
    difference = np.abs(image - quantized_image)
    axes[0, 2].imshow(difference)
    axes[0, 2].set_title('Absolute Difference')
    axes[0, 2].axis('off')
    
    # Color palette
    palette = kmeans.cluster_centers_.reshape(1, n_colors, 3)
    axes[1, 0].imshow(palette)
    axes[1, 0].set_title('Color Palette')
    axes[1, 0].axis('off')
    
    # Color distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    colors_rgb = kmeans.cluster_centers_
    
    axes[1, 1].bar(range(n_colors), counts, color=colors_rgb, alpha=0.8)
    axes[1, 1].set_title('Color Frequency')
    axes[1, 1].set_xlabel('Color Index')
    axes[1, 1].set_ylabel('Pixel Count')
    
    # MSE plot for different number of colors
    color_range = range(2, 17)
    mse_values = []
    
    for n_c in color_range:
        temp_kmeans = KMeans(n_clusters=n_c, random_state=42, n_init=5)
        temp_labels = temp_kmeans.fit_predict(image_2d)
        temp_quantized = temp_kmeans.cluster_centers_[temp_labels]
        mse = np.mean((image_2d - temp_quantized) ** 2)
        mse_values.append(mse)
    
    axes[1, 2].plot(color_range, mse_values, 'bo-')
    axes[1, 2].axvline(x=n_colors, color='red', linestyle='--', 
                       label=f'Selected k={n_colors}')
    axes[1, 2].set_title('MSE vs Number of Colors')
    axes[1, 2].set_xlabel('Number of Colors')
    axes[1, 2].set_ylabel('Mean Squared Error')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return quantized_image, kmeans

# Apply color quantization
quantized_img, color_kmeans = quantize_image_colors(n_colors=16)

print("\nColor palette (RGB values):")
for i, color in enumerate(color_kmeans.cluster_centers_):
    print(f"Color {i}: RGB({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})")
```

## 📚 References

### Academic Papers
- Lloyd, S. (1982). "Least squares quantization in PCM". *IEEE Transactions on Information Theory*
- Arthur, D. & Vassilvitskii, S. (2007). "K-means++: The advantages of careful seeding". *SODA '07*
- MacQueen, J. (1967). "Some methods for classification and analysis of multivariate observations"

### Books
- **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning*. Chapter 14.3
- **Bishop, C.** (2006). *Pattern Recognition and Machine Learning*. Chapter 9
- **Murphy, K.** (2012). *Machine Learning: A Probabilistic Perspective*. Chapter 25

### Online Resources
- [Scikit-learn K-means Documentation](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [K-means Clustering: Algorithm, Applications, Evaluation Methods](https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a)
- [An Introduction to Statistical Learning with R](https://www.statlearning.com/) - Chapter 10
- [Stanford CS229 Machine Learning Course Notes](http://cs229.stanford.edu/notes/cs229-notes7a.pdf)

### Tutorials and Guides
- [K-means Clustering in Python: A Practical Guide](https://realpython.com/k-means-clustering-python/)
- [Clustering Algorithms Comparison](https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html)
- [How to Determine the Optimal Number of Clusters](https://blog.cambridgespark.com/how-to-determine-the-optimal-number-of-clusters-for-k-means-clustering-14f27070048f)

### Interactive Visualizations
- [K-means Interactive Demo](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/)
- [Clustering Visualization Tool](https://www.cs.cmu.edu/~mmv/clustering.html)
