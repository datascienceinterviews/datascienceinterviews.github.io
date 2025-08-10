---
title: k-Nearest Neighbors (kNN)
description: Comprehensive guide to k-Nearest Neighbors algorithm with mathematical intuition, implementations, and interview questions.
comments: true
---

# 📘 k-Nearest Neighbors (kNN)

k-Nearest Neighbors (kNN) is a simple, versatile, non-parametric algorithm used for both classification and regression tasks that makes predictions based on the majority class or average value of its k closest training examples.

**Resources:** [Scikit-learn kNN](https://scikit-learn.org/stable/modules/neighbors.html) | [Elements of Statistical Learning - Chapter 13](https://web.stanford.edu/~hastie/ElemStatLearn/)

## ✍️ Summary

k-Nearest Neighbors (kNN) is an instance-based, lazy learning algorithm that delays all computation until prediction time. The core idea is simple: similar instances tend to have similar outputs. For classification, kNN predicts a class by finding the most common class among the k-closest neighbors. For regression, it predicts a value by averaging the values of its k-nearest neighbors.

**Key characteristics:**
- **Non-parametric**: Makes no assumptions about data distribution
- **Lazy learner**: No explicit training phase
- **Instance-based**: Stores all training examples for prediction
- **Intuitive**: Easy to understand and implement
- **Versatile**: Works for both classification and regression

**Applications:**
- Recommendation systems
- Credit scoring
- Medical diagnosis
- Anomaly detection
- Image classification
- Pattern recognition
- Gene expression analysis

## 🧠 Intuition

### How kNN Works

1. **Store**: Remember all training examples
2. **Distance**: Calculate distances between new example and all stored examples
3. **Neighbors**: Find k nearest neighbors based on distance
4. **Decision**: Make prediction based on neighbors (majority vote or average)

### Mathematical Foundation

#### 1. Distance Metrics

**Euclidean Distance (most common):**
$$d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

**Manhattan Distance:**
$$d(x, y) = \sum_{i=1}^{n} |x_i - y_i|$$

**Minkowski Distance (generalization):**
$$d(x, y) = \left(\sum_{i=1}^{n} |x_i - y_i|^p\right)^{1/p}$$
- p = 1: Manhattan distance
- p = 2: Euclidean distance

**Hamming Distance (for categorical features):**
$$d(x, y) = \sum_{i=1}^{n} \mathbb{1}(x_i \neq y_i)$$

#### 2. Decision Rules

**For Classification:**
$$\hat{y} = \text{mode}(y_i), \text{ where } i \in \text{top-}k \text{ nearest neighbors}$$

**For Regression:**
$$\hat{y} = \frac{1}{k} \sum_{i=1}^{k} y_i, \text{ where } i \in \text{top-}k \text{ nearest neighbors}$$

**Weighted kNN:**
$$\hat{y} = \frac{\sum_{i=1}^{k} w_i y_i}{\sum_{i=1}^{k} w_i}, \text{ where } w_i = \frac{1}{d(x, x_i)^2}$$

### Algorithm Steps

1. **Choose k**: Determine appropriate number of neighbors
2. **Calculate distances**: Measure distance between query point and all training samples
3. **Find neighbors**: Identify k closest points
4. **Make prediction**: Classify by majority vote or predict by average

## 🔢 Implementation using Libraries

### Using Scikit-learn

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_boston, make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# ---- Classification Example ----
# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features (important for distance-based algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train classifier
knn_clf = KNeighborsClassifier(
    n_neighbors=5,            # number of neighbors
    weights='uniform',        # or 'distance' for weighted voting
    algorithm='auto',         # or 'ball_tree', 'kd_tree', 'brute'
    leaf_size=30,             # affects speed of tree algorithms
    p=2                       # power parameter for Minkowski distance
)

knn_clf.fit(X_train_scaled, y_train)

# Make predictions
y_pred = knn_clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Visualize decision boundaries (for 2 features)
def plot_decision_boundary(X, y, model, scaler, feature_indices=[0, 1]):
    h = 0.02  # step size in the mesh
    
    # Select two features
    X_selected = X[:, feature_indices]
    X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    X_train_sel_scaled = scaler.fit_transform(X_train_sel)
    X_test_sel_scaled = scaler.transform(X_test_sel)
    
    # Train model on selected features
    model.fit(X_train_sel_scaled, y_train_sel)
    
    # Create a mesh grid
    x_min, x_max = X_selected[:, 0].min() - 1, X_selected[:, 0].max() + 1
    y_min, y_max = X_selected[:, 1].min() - 1, X_selected[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Scale mesh grid
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_points_scaled = scaler.transform(mesh_points)
    
    # Predict
    Z = model.predict(mesh_points_scaled)
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    
    # Plot training points
    scatter = plt.scatter(X_selected[:, 0], X_selected[:, 1], c=y, 
                 edgecolors='k', cmap=plt.cm.Paired)
    
    plt.xlabel(f'Feature {feature_indices[0]} ({iris.feature_names[feature_indices[0]]})')
    plt.ylabel(f'Feature {feature_indices[1]} ({iris.feature_names[feature_indices[1]]})')
    plt.title(f'Decision Boundary with k={model.n_neighbors}')
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.show()

# Plot decision boundary
plot_decision_boundary(X, y, KNeighborsClassifier(n_neighbors=5), 
                      StandardScaler(), [0, 1])

# Finding optimal k value
k_values = list(range(1, 31))
train_accuracies = []
test_accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    
    # Training accuracy
    y_train_pred = knn.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_accuracies.append(train_acc)
    
    # Testing accuracy
    y_test_pred = knn.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_accuracies.append(test_acc)

# Plot accuracy vs k
plt.figure(figsize=(12, 6))
plt.plot(k_values, train_accuracies, label='Training Accuracy', marker='o')
plt.plot(k_values, test_accuracies, label='Testing Accuracy', marker='s')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs k for kNN Classifier')
plt.legend()
plt.grid(True)
plt.show()

print(f"Best k: {k_values[np.argmax(test_accuracies)]}")

# ---- Regression Example ----

# Create synthetic regression data
np.random.seed(42)
X_reg = np.random.rand(100, 1) * 10
y_reg = 2 * X_reg.squeeze() + 3 + np.random.randn(100) * 2

# Split regression data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Create and train regression model
knn_reg = KNeighborsRegressor(n_neighbors=3)
knn_reg.fit(X_train_reg, y_train_reg)

# Make predictions
y_pred_reg = knn_reg.predict(X_test_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse:.3f}")

# Plot regression results
plt.figure(figsize=(10, 6))
plt.scatter(X_reg, y_reg, c='b', label='Data')
plt.scatter(X_test_reg, y_test_reg, c='g', marker='s', label='Test Data')
plt.scatter(X_test_reg, y_pred_reg, c='r', marker='^', label='Predictions')

# Plot predictions for a fine-grained range
X_range = np.linspace(0, 10, 1000).reshape(-1, 1)
y_range_pred = knn_reg.predict(X_range)
plt.plot(X_range, y_range_pred, c='orange', label='kNN Predictions')

plt.xlabel('X')
plt.ylabel('y')
plt.title('kNN Regression (k=3)')
plt.legend()
plt.grid(True)
plt.show()
```

## ⚙️ From Scratch Implementation

```python
import numpy as np
from collections import Counter
from scipy.spatial.distance import cdist

class KNNFromScratch:
    def __init__(self, k=5, distance_metric='euclidean', weights='uniform', algorithm_type='classification'):
        """
        k-Nearest Neighbors algorithm implementation from scratch
        
        Parameters:
        k (int): Number of neighbors to use
        distance_metric (str): 'euclidean', 'manhattan', or 'minkowski'
        weights (str): 'uniform' or 'distance'
        algorithm_type (str): 'classification' or 'regression'
        """
        self.k = k
        self.distance_metric = distance_metric
        self.weights = weights
        self.algorithm_type = algorithm_type
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """Store training data (lazy learning)"""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
    
    def _calculate_distances(self, X):
        """Calculate distances between test points and all training points"""
        return cdist(X, self.X_train, metric=self.distance_metric)
    
    def _get_neighbors(self, distances):
        """Get indices of k-nearest neighbors"""
        return np.argsort(distances, axis=1)[:, :self.k]
    
    def _get_weights(self, distances, neighbor_indices):
        """Get weights for neighbors"""
        if self.weights == 'uniform':
            # All neighbors have equal weight
            return np.ones((distances.shape[0], self.k))
        elif self.weights == 'distance':
            # Weights are inverse of distances
            neighbor_distances = np.take_along_axis(
                distances, neighbor_indices, axis=1)
            
            # Avoid division by zero
            neighbor_distances = np.maximum(neighbor_distances, 1e-10)
            
            # Weights = 1/distance
            weights = 1.0 / neighbor_distances
            
            # Normalize weights
            row_sums = weights.sum(axis=1, keepdims=True)
            return weights / row_sums
    
    def predict(self, X):
        """Make predictions for test data"""
        X = np.array(X)
        
        # Calculate distances
        distances = self._calculate_distances(X)
        
        # Get indices of k-nearest neighbors
        neighbor_indices = self._get_neighbors(distances)
        
        # Get weights
        weights = self._get_weights(distances, neighbor_indices)
        
        # Get labels of neighbors
        neighbor_labels = self.y_train[neighbor_indices]
        
        if self.algorithm_type == 'classification':
            return self._predict_classification(neighbor_labels, weights)
        else:  # regression
            return self._predict_regression(neighbor_labels, weights)
    
    def _predict_classification(self, neighbor_labels, weights):
        """Predict class labels using weighted voting"""
        predictions = []
        
        for i in range(neighbor_labels.shape[0]):
            if self.weights == 'uniform':
                # Majority vote
                most_common = Counter(neighbor_labels[i]).most_common(1)
                predictions.append(most_common[0][0])
            else:  # 'distance'
                # Weighted vote
                class_weights = {}
                for j in range(self.k):
                    label = neighbor_labels[i, j]
                    weight = weights[i, j]
                    class_weights[label] = class_weights.get(label, 0) + weight
                
                # Get class with highest weight
                predictions.append(max(class_weights, key=class_weights.get))
        
        return np.array(predictions)
    
    def _predict_regression(self, neighbor_labels, weights):
        """Predict values using weighted average"""
        # Weighted average: sum(weights * values) / sum(weights)
        return np.sum(neighbor_labels * weights, axis=1)
    
    def score(self, X, y):
        """Calculate accuracy (classification) or R² (regression)"""
        y_pred = self.predict(X)
        
        if self.algorithm_type == 'classification':
            # Classification accuracy
            return np.mean(y_pred == y)
        else:  # regression
            # R² score
            y_mean = np.mean(y)
            ss_total = np.sum((y - y_mean) ** 2)
            ss_residual = np.sum((y - y_pred) ** 2)
            return 1 - (ss_residual / ss_total)

# Example usage
if __name__ == "__main__":
    # Generate sample classification data
    np.random.seed(42)
    X = np.random.rand(100, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    
    # Split data
    indices = np.random.permutation(len(X))
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[indices[:train_size]], X[indices[train_size:]]
    y_train, y_test = y[indices[:train_size]], y[indices[train_size:]]
    
    # Train custom kNN classifier
    knn = KNNFromScratch(k=3, algorithm_type='classification')
    knn.fit(X_train, y_train)
    
    # Evaluate
    y_pred = knn.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Classification Accuracy: {accuracy:.3f}")
    
    # Generate sample regression data
    X_reg = np.random.rand(100, 1) * 10
    y_reg = 2 * X_reg.squeeze() + 3 + np.random.randn(100)
    
    # Split regression data
    indices_reg = np.random.permutation(len(X_reg))
    X_train_reg = X_reg[indices_reg[:train_size]]
    X_test_reg = X_reg[indices_reg[train_size:]]
    y_train_reg = y_reg[indices_reg[:train_size]]
    y_test_reg = y_reg[indices_reg[train_size:]]
    
    # Train custom kNN regressor
    knn_reg = KNNFromScratch(k=3, algorithm_type='regression', weights='distance')
    knn_reg.fit(X_train_reg, y_train_reg)
    
    # Evaluate
    y_pred_reg = knn_reg.predict(X_test_reg)
    mse = np.mean((y_pred_reg - y_test_reg) ** 2)
    rmse = np.sqrt(mse)
    r2 = knn_reg.score(X_test_reg, y_test_reg)
    print(f"Regression RMSE: {rmse:.3f}")
    print(f"Regression R²: {r2:.3f}")
```

## ⚠️ Assumptions and Limitations

### Assumptions

1. **Similarity-proximity assumption**: Similar instances are located close to each other in feature space
2. **Equal feature importance**: All features contribute equally to distance calculations
3. **Locally constant function**: Target function is assumed to be locally constant

### Limitations

1. **Curse of dimensionality**: Performance degrades in high-dimensional spaces
2. **Computational cost**: Calculating distances to all training samples is expensive
3. **Memory intensive**: Stores all training data
4. **Sensitive to irrelevant features**: All features contribute to distance
5. **Sensitive to scale**: Features with larger scales dominate distance calculations
6. **Imbalanced data**: Majority class dominates in classification
7. **Parameter sensitivity**: Results highly dependent on choice of k

### Comparison with Other Models

| Algorithm | Advantages vs kNN | Disadvantages vs kNN |
|-----------|-------------------|----------------------|
| **Decision Trees** | Handles irrelevant features, fast prediction | Less accurate for complex boundaries |
| **SVM** | Works well in high dimensions, handles non-linear patterns | Complex tuning, black box model |
| **Naive Bayes** | Very fast, works well with high dimensions | Assumes feature independence |
| **Linear Regression** | Simple, interpretable | Only captures linear relationships |
| **Neural Networks** | Captures complex patterns, automatic feature learning | Needs more data, complex tuning |

## 💡 Interview Questions

??? question "1. What's the difference between a lazy and eager learning algorithm, and where does kNN fit?"

    **Answer:**
    
    **Lazy learning algorithms** delay all computation until prediction time:
    - No explicit training phase
    - Store all training examples in memory
    - Computation happens at prediction time
    - Examples: k-Nearest Neighbors, Case-Based Reasoning
    
    **Eager learning algorithms** create a model during training:
    - Generalize from training data during training phase
    - Discard training data after model is built
    - Fast predictions using the pre-built model
    - Examples: Decision Trees, Neural Networks, SVM
    
    **kNN is a lazy learner because:**
    - It doesn't build a model during training
    - It simply stores all training examples
    - Computations (distance calculations, neighbor finding) happen during prediction
    - Each prediction requires scanning the entire training set
    
    This gives kNN certain characteristics:
    - Slow predictions (especially with large training sets)
    - Fast training (just stores data)
    - Adapts naturally to new training data
    - No information loss from generalization

??? question "2. How do you choose the optimal value of k in kNN?"

    **Answer:**
    
    **Methods for choosing k:**
    
    **1. Cross-validation:**
    - Most common approach
    - Split data into training and validation sets
    - Train models with different k values
    - Choose k with best validation performance
    - Example: k-fold cross-validation
    
    **2. Square root heuristic:**
    - Rule of thumb: k ≈ √n, where n is training set size
    - Quick starting point for experimentation
    
    **3. Elbow method:**
    - Plot error rate against different k values
    - Look for "elbow" where error rate stabilizes
    
    **Considerations when choosing k:**
    
    - **Small k values:**
      - More flexible decision boundaries
      - Can lead to overfitting
      - More sensitive to noise
    
    - **Large k values:**
      - Smoother decision boundaries
      - Can lead to underfitting
      - Computationally more expensive
    
    - **Odd vs. even:**
      - For binary classification, use odd k to avoid ties
    
    - **Domain knowledge:**
      - Consider problem characteristics
      - Some domains benefit from specific k ranges
    
    **Implementation example:**
    ```python
    from sklearn.model_selection import cross_val_score
    
    # Find optimal k
    k_range = range(1, 31)
    k_scores = []
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
        k_scores.append(scores.mean())
    
    best_k = k_range[np.argmax(k_scores)]
    ```

??? question "3. Why is feature scaling important for kNN, and how would you implement it?"

    **Answer:**
    
    **Importance of feature scaling:**
    
    - **Distance domination:** Features with larger scales will dominate the distance calculation
    - **Equal contribution:** Scaling ensures all features contribute equally
    - **Improved accuracy:** Properly scaled features generally lead to better performance
    
    **Example:**
    Consider two features: age (0-100) and income (0-1,000,000)
    - Without scaling, income differences will completely overwhelm age differences
    - After scaling, both contribute proportionally to their importance
    
    **Common scaling methods:**
    
    **1. Min-Max Scaling (Normalization):**
    $$X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}$$
    - Scales features to range [0,1]
    - Good when distribution is not Gaussian
    
    **2. Z-score Standardization:**
    $$X_{scaled} = \frac{X - \mu}{\sigma}$$
    - Transforms to mean=0, std=1
    - Good for normally distributed features
    
    **3. Robust Scaling:**
    $$X_{scaled} = \frac{X - median}{IQR}$$
    - Uses median and interquartile range
    - Robust to outliers
    
    **Implementation:**
    ```python
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    
    # Z-score standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Min-Max scaling
    minmax = MinMaxScaler()
    X_scaled = minmax.fit_transform(X)
    
    # Robust scaling
    robust = RobustScaler()
    X_scaled = robust.fit_transform(X)
    ```
    
    **Important considerations:**
    - Always fit scaler on training data only
    - Apply same transformation to test data
    - Different scaling methods may be appropriate for different features
    - Categorical features may need special handling (e.g., one-hot encoding)

??? question "4. How can kNN handle categorical features?"

    **Answer:**
    
    **Approaches for handling categorical features in kNN:**
    
    **1. One-Hot Encoding:**
    - Convert categorical variables to binary columns
    - Each category becomes its own binary feature
    - Increases dimensionality
    - Example: Color (Red, Blue, Green) → [1,0,0], [0,1,0], [0,0,1]
    
    ```python
    from sklearn.preprocessing import OneHotEncoder
    
    encoder = OneHotEncoder(sparse=False)
    X_cat_encoded = encoder.fit_transform(X_categorical)
    ```
    
    **2. Distance metrics for mixed data:**
    - Use specialized metrics that handle mixed data types
    - Gower distance: combines different metrics for different types
      - Euclidean for numeric features
      - Hamming for categorical features
    
    ```python
    def gower_distance(x, y, categorical_features):
        numeric_dist = np.sqrt(np.sum(((x[~categorical_features] - 
                                       y[~categorical_features]) ** 2)))
        categ_dist = np.sum(x[categorical_features] != y[categorical_features])
        return (numeric_dist + categ_dist) / len(x)
    ```
    
    **3. Ordinal Encoding:**
    - Assign integers to categories
    - Only appropriate when categories have natural ordering
    - Example: Size (Small, Medium, Large) → 1, 2, 3
    
    ```python
    from sklearn.preprocessing import OrdinalEncoder
    
    encoder = OrdinalEncoder()
    X_cat_encoded = encoder.fit_transform(X_categorical)
    ```
    
    **4. Feature hashing:**
    - Hash categorical values to fixed-length vectors
    - Useful for high-cardinality features
    
    **5. Custom distance functions:**
    - Implement specialized distance measures
    - Can apply different distance metrics to different features
    
    **Best practices:**
    - Consider feature relevance when choosing encoding
    - For small categorical sets, one-hot encoding often works best
    - For high cardinality, consider embeddings or hashing
    - Test different approaches with cross-validation

??? question "5. What happens when kNN is applied to high-dimensional data, and how can you address these issues?"

    **Answer:**
    
    **The Curse of Dimensionality in kNN:**
    
    - **Distance concentration:** As dimensions increase, distances between points become more similar
    - **Sparsity:** Points become farther apart, requiring more data to maintain density
    - **Computational cost:** Calculating distances in high dimensions is expensive
    - **Irrelevant features:** More dimensions increase the chance of including irrelevant features
    
    In high dimensions:
    - All points tend to be equidistant from each other
    - Concept of "nearest neighbor" becomes less meaningful
    - Distance calculations become computationally expensive
    - Model accuracy degrades significantly
    
    **Solutions to address high-dimensionality:**
    
    **1. Dimensionality reduction:**
    ```python
    from sklearn.decomposition import PCA
    
    # Reduce to 10 dimensions
    pca = PCA(n_components=10)
    X_reduced = pca.fit_transform(X)
    ```
    
    **2. Feature selection:**
    ```python
    from sklearn.feature_selection import SelectKBest, f_classif
    
    # Select top 10 features
    selector = SelectKBest(f_classif, k=10)
    X_selected = selector.fit_transform(X, y)
    ```
    
    **3. Use approximate nearest neighbors:**
    ```python
    from sklearn.neighbors import NearestNeighbors
    
    # Use ball tree algorithm
    nn = NearestNeighbors(algorithm='ball_tree')
    ```
    
    **4. Locality Sensitive Hashing (LSH):**
    - Maps similar items to same buckets with high probability
    - Allows approximate nearest neighbor search
    
    **5. Feature weighting:**
    - Assign different weights to features based on importance
    - Can use feature importance from other models
    
    **6. Use distance metrics suited for high dimensions:**
    - Cosine similarity for text data
    - Manhattan distance often works better than Euclidean in high dimensions
    
    **7. Increase training data:**
    - More data helps combat sparsity issues

??? question "6. How does weighted kNN differ from standard kNN, and when would you use it?"

    **Answer:**
    
    **Standard kNN vs. Weighted kNN:**
    
    **Standard kNN:**
    - All k neighbors contribute equally to prediction
    - Classification: simple majority vote
    - Regression: simple average of neighbor values
    
    **Weighted kNN:**
    - Neighbors contribute based on their distance
    - Closer neighbors have greater influence
    - Weight typically inversely proportional to distance
    
    **Mathematical formulation:**
    
    For standard kNN:
    $$\hat{y} = \frac{1}{k} \sum_{i=1}^{k} y_i$$
    
    For weighted kNN:
    $$\hat{y} = \frac{\sum_{i=1}^{k} w_i y_i}{\sum_{i=1}^{k} w_i}$$
    
    Where common weight functions include:
    - Inverse distance: $w_i = \frac{1}{d(x, x_i)}$
    - Inverse squared distance: $w_i = \frac{1}{d(x, x_i)^2}$
    - Exponential: $w_i = e^{-d(x, x_i)}$
    
    **When to use weighted kNN:**
    
    - **Uneven distribution:** When training data is unevenly distributed
    - **Varying importance:** When certain neighbors should have more influence
    - **Noisy data:** Reduces impact of outliers
    - **Class imbalance:** Helps with imbalanced classification problems
    - **Boundary regions:** Improves accuracy near decision boundaries
    
    **Implementation in scikit-learn:**
    ```python
    # Standard kNN
    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    
    # Weighted kNN
    knn_weighted = KNeighborsClassifier(n_neighbors=5, weights='distance')
    ```
    
    **Custom weight functions:**
    ```python
    def custom_weight(distances):
        return np.exp(-distances)
    
    knn_custom = KNeighborsClassifier(
        n_neighbors=5, 
        weights=custom_weight
    )
    ```

??? question "7. Compare and contrast kNN with other classification algorithms like Decision Trees and SVM."

    **Answer:**
    
    | Aspect | kNN | Decision Trees | SVM |
    |--------|-----|----------------|-----|
    | **Learning Type** | Lazy (instance-based) | Eager | Eager |
    | **Training Speed** | Very fast (just stores data) | Moderate | Slow (especially with non-linear kernels) |
    | **Prediction Speed** | Slow (distance calculation) | Fast | Fast (except with many support vectors) |
    | **Memory Requirements** | High (stores all training data) | Low | Moderate (stores support vectors) |
    | **Interpretability** | Moderate | High (can visualize tree) | Low (especially with kernels) |
    | **Handling Non-linearity** | Naturally handles non-linear boundaries | Steps (hierarchical) | Kernels transform to linearly separable space |
    | **Feature Scaling** | Critical | Not needed | Important |
    | **Outlier Sensitivity** | High | Low | Low (with proper C value) |
    | **Missing Value Handling** | Poor | Good | Poor |
    | **High Dimensions** | Poor (curse of dimensionality) | Good (feature selection) | Good (regularization) |
    | **Imbalanced Data** | Poor | Moderate | Good (with class weights) |
    | **Categorical Features** | Needs encoding | Handles naturally | Needs encoding |
    | **Overfitting Risk** | High with small k | High with deep trees | Low with proper regularization |
    | **Hyperparameter Tuning** | Simple (mainly k) | Moderate | Complex |
    
    **Key Contrasts:**
    
    **kNN vs. Decision Trees:**
    - kNN: Instance-based, creates complex non-linear boundaries
    - Trees: Rule-based, creates axis-parallel decision boundaries
    - kNN relies on distance; Trees use feature thresholds
    - Trees automatically handle feature importance; kNN treats all equally
    
    **kNN vs. SVM:**
    - kNN: Local patterns based on neighbors
    - SVM: Global patterns based on support vectors
    - kNN: Simple but computationally expensive at prediction time
    - SVM: Complex optimization but efficient predictions
    
    **When to choose each:**
    
    **Choose kNN when:**
    - Small to medium dataset
    - Low dimensionality
    - Complex non-linear decision boundaries
    - Quick implementation needed
    - Prediction speed not critical
    
    **Choose Decision Trees when:**
    - Feature importance needed
    - Interpretability is critical
    - Mixed feature types
    - Fast predictions required
    
    **Choose SVM when:**
    - High-dimensional data
    - Complex boundaries
    - Memory efficiency needed
    - Strong theoretical guarantees required

??? question "8. Explain how to implement kNN for large datasets where the data doesn't fit in memory."

    **Answer:**
    
    **Strategies for large-scale kNN:**
    
    **1. Data sampling techniques:**
    - Use a representative subset of training data
    - Condensed nearest neighbors: only keep points that affect decision boundaries
    - Edited nearest neighbors: remove noisy samples
    
    ```python
    from imblearn.under_sampling import CondensedNearestNeighbour
    
    cnn = CondensedNearestNeighbour(n_neighbors=5)
    X_reduced, y_reduced = cnn.fit_resample(X, y)
    ```
    
    **2. Approximate nearest neighbor methods:**
    - Locality-Sensitive Hashing (LSH)
    - Random projection trees
    - Product quantization
    
    ```python
    # Using Annoy library for approximate nearest neighbors
    from annoy import AnnoyIndex
    
    # Build index
    f = X.shape[1]  # dimensionality
    t = AnnoyIndex(f, 'euclidean')
    for i, x in enumerate(X):
        t.add_item(i, x)
    t.build(10)  # 10 trees
    
    # Query
    indices = t.get_nns_by_vector(query_vector, 5)  # find 5 nearest neighbors
    ```
    
    **3. KD-Trees and Ball Trees:**
    - Space-partitioning data structures
    - Log(n) query time for low dimensions
    - Still struggle in high dimensions
    
    ```python
    from sklearn.neighbors import KNeighborsClassifier
    
    knn = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree', 
                               leaf_size=30, n_jobs=-1)
    ```
    
    **4. Distributed computing:**
    - Parallelize distance computations
    - Map-reduce framework for kNN
    
    **5. GPU acceleration:**
    - Leverage GPU for parallel distance calculations
    - Libraries like FAISS from Facebook
    
    ```python
    # Using FAISS for GPU-accelerated nearest neighbor search
    import faiss
    
    # Convert to float32
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    # Build index
    index = faiss.IndexFlatL2(X_train.shape[1])
    index.add(X_train)
    
    # Search
    k = 5
    distances, indices = index.search(X_test, k)
    ```
    
    **6. Incremental learning:**
    - Process data in batches
    - Update model with new chunks of data
    
    **7. Database-backed implementations:**
    - Store data in database
    - Use database's indexing capabilities
    - SQL or NoSQL solutions
    
    **8. Feature reduction:**
    - Apply dimensionality reduction first
    - PCA, t-SNE, or UMAP to reduce dimensions
    
    **Best practices:**
    - Consider problem requirements (accuracy vs. speed)
    - Benchmark different approaches
    - Combine multiple strategies
    - Use specialized libraries for large-scale kNN

## 🧠 Examples

### Example 1: Handwritten Digit Recognition

```python
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load digit dataset
digits = load_digits()
X, y = digits.data, digits.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train kNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)

# Evaluate
accuracy = knn.score(X_test, y_test)
print(f"Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10),
            yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Visualize some predictions
fig, axes = plt.subplots(4, 5, figsize=(12, 8))
for i, ax in enumerate(axes.flat):
    if i < len(X_test):
        ax.imshow(X_test[i].reshape(8, 8), cmap='binary')
        pred = knn.predict(X_test[i].reshape(1, -1))[0]
        true = y_test[i]
        color = 'green' if pred == true else 'red'
        ax.set_title(f'Pred: {pred}, True: {true}', color=color)
        ax.axis('off')
plt.tight_layout()
plt.show()

# Test with different k values
k_values = list(range(1, 21))
train_scores = []
test_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_scores.append(knn.score(X_train, y_train))
    test_scores.append(knn.score(X_test, y_test))

plt.figure(figsize=(10, 6))
plt.plot(k_values, train_scores, 'o-', label='Training Accuracy')
plt.plot(k_values, test_scores, 's-', label='Testing Accuracy')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.title('kNN Accuracy vs k for Digit Recognition')
plt.legend()
plt.grid(True)
plt.show()
```

### Example 2: Anomaly Detection

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Generate sample data with outliers
X_inliers, _ = make_blobs(n_samples=300, centers=1, cluster_std=2.0, random_state=42)
X_outliers = np.random.uniform(low=-15, high=15, size=(15, 2))
X = np.vstack([X_inliers, X_outliers])

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit nearest neighbors model
k = 5
nn = NearestNeighbors(n_neighbors=k)
nn.fit(X_scaled)

# Calculate distance to k-th nearest neighbor
distances, indices = nn.kneighbors(X_scaled)
k_distance = distances[:, k-1]

# Set threshold for anomaly detection
threshold = np.percentile(k_distance, 95)  # 95th percentile
anomalies = k_distance > threshold

# Plot results
plt.figure(figsize=(12, 8))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c='blue', label='Normal points')
plt.scatter(X_scaled[anomalies, 0], X_scaled[anomalies, 1], 
            c='red', s=100, marker='x', label='Detected Anomalies')

# Plot true outliers
true_outliers = np.arange(len(X_inliers), len(X))
plt.scatter(X_scaled[true_outliers, 0], X_scaled[true_outliers, 1], 
            edgecolors='green', s=140, facecolors='none', 
            linewidth=2, marker='o', label='True Outliers')

plt.title('kNN-based Anomaly Detection')
plt.legend()
plt.grid(True)
plt.show()

# Evaluation
true_anomalies = np.zeros(len(X), dtype=bool)
true_anomalies[true_outliers] = True

# True Positives, False Positives, etc.
TP = np.sum(np.logical_and(anomalies, true_anomalies))
FP = np.sum(np.logical_and(anomalies, ~true_anomalies))
TN = np.sum(np.logical_and(~anomalies, ~true_anomalies))
FN = np.sum(np.logical_and(~anomalies, true_anomalies))

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")
```

## 📚 References

- **Books:**
  - [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) by Christopher Bishop
  - [The Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/) by Hastie, Tibshirani, & Friedman
  - [Machine Learning: A Probabilistic Perspective](https://www.cs.ubc.ca/~murphyk/MLbook/) by Kevin Murphy

- **Documentation:**
  - [Scikit-learn Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html)
  - [SciPy Spatial Distance Functions](https://docs.scipy.org/doc/scipy/reference/spatial.distance.html)

- **Tutorials:**
  - [KNN Algorithm - How KNN Algorithm Works](https://www.javatpoint.com/k-nearest-neighbor-algorithm-for-machine-learning)
  - [Complete Guide to kNN in Python](https://machinelearningmastery.com/k-nearest-neighbors-for-machine-learning/)
  - [KNN from Scratch in Python](https://www.analyticsvidhya.com/blog/2018/03/introduction-k-neighbours-algorithm-clustering/)

- **Research Papers:**
  - Fix, E., & Hodges, J. L. (1951). Discriminatory analysis, nonparametric discrimination: Consistency properties.
  - Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. IEEE Transactions on Information Theory, 13(1), 21-27.
  - Weinberger, K. Q., & Saul, L. K. (2009). Distance metric learning for large margin nearest neighbor classification. Journal of Machine Learning Research, 10, 207-244.

- **Online Courses:**
  - [Machine Learning by Andrew Ng (Coursera)](https://www.coursera.org/learn/machine-learning)
  - [KNN Algorithm - StatQuest with Josh Starmer](https://www.youtube.com/watch?v=HVXime0nQeI)
