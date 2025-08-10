---
title: Support Vector Machines (SVM)
description: Comprehensive guide to Support Vector Machines with mathematical intuition, kernel tricks, implementations, and interview questions.
comments: true
---

# ⚔️ Support Vector Machines (SVM)

Support Vector Machines are powerful supervised learning algorithms that find the optimal decision boundary by maximizing the margin between classes, capable of handling both linear and non-linear classification and regression problems through kernel methods.

**Resources:** [Scikit-learn SVM](https://scikit-learn.org/stable/modules/svm.html) | [Support Vector Networks Paper](https://link.springer.com/article/10.1007/BF00994018) | [Elements of Statistical Learning - Chapter 12](https://web.stanford.edu/~hastie/ElemStatLearn/)

##  Summary

Support Vector Machine (SVM) is a discriminative classifier that finds the optimal hyperplane to separate different classes by maximizing the margin (distance) between the closest points of each class. The algorithm focuses on the most informative data points (support vectors) rather than using all training data, making it efficient and robust.

**Key characteristics:**
- **Maximum margin classifier**: Finds the hyperplane with largest margin
- **Support vector focus**: Only depends on support vectors, not all training data
- **Kernel trick**: Can handle non-linear decision boundaries using kernel functions
- **Regularization**: Built-in regularization through the C parameter
- **Versatile**: Works for classification, regression, and outlier detection
- **Memory efficient**: Stores only support vectors, not entire dataset

**Applications:**
- Text classification and sentiment analysis
- Image classification and computer vision
- Bioinformatics and gene classification  
- Handwriting recognition
- Face detection and recognition
- Document classification
- Spam email filtering
- Medical diagnosis
- Financial market analysis

**Types:**
- **Linear SVM**: For linearly separable data
- **Soft Margin SVM**: Handles non-separable data with slack variables
- **Kernel SVM**: Non-linear classification using kernel methods
- **SVR (Support Vector Regression)**: For regression tasks
- **One-Class SVM**: For anomaly detection and novelty detection

## >� Intuition

### How SVM Works

Imagine you're trying to separate two groups of people in a room. Instead of just drawing any line between them, SVM finds the "widest corridor" that separates the groups. The people standing closest to this corridor (support vectors) determine where the boundary should be. Everyone else could leave the room, and the boundary would stay the same.

For non-linearly separable data, SVM uses the "kernel trick" - it projects the data into a higher-dimensional space where a linear separator can be found, then maps the decision boundary back to the original space.

### Mathematical Foundation

#### 1. Linear SVM - Hard Margin

For a binary classification problem with training data $\{(x_i, y_i)\}_{i=1}^n$ where $y_i \in \{-1, +1\}$:

**Decision boundary**: $w^T x + b = 0$

**Classification rule**: $f(x) = \text{sign}(w^T x + b)$

**Margin**: The distance from the hyperplane to the nearest data point is $\frac{1}{||w||}$

**Optimization problem** (Hard Margin):
$$\min_{w,b} \frac{1}{2}||w||^2$$

**Subject to**: $y_i(w^T x_i + b) \geq 1, \quad \forall i = 1,...,n$

#### 2. Soft Margin SVM

For non-separable data, introduce slack variables $\xi_i \geq 0$:

**Optimization problem** (Soft Margin):
$$\min_{w,b,\xi} \frac{1}{2}||w||^2 + C\sum_{i=1}^n \xi_i$$

**Subject to**: 
- $y_i(w^T x_i + b) \geq 1 - \xi_i, \quad \forall i$
- $\xi_i \geq 0, \quad \forall i$

Where $C$ is the regularization parameter controlling the trade-off between margin maximization and training error minimization.

#### 3. Dual Formulation (Lagrangian)

The primal problem is converted to dual form using Lagrange multipliers $\alpha_i$:

**Dual optimization problem**:
$$\max_{\alpha} \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j x_i^T x_j$$

**Subject to**:
- $0 \leq \alpha_i \leq C, \quad \forall i$
- $\sum_{i=1}^n \alpha_i y_i = 0$

**Decision function**:
$$f(x) = \text{sign}\left(\sum_{i=1}^n \alpha_i y_i x_i^T x + b\right)$$

#### 4. Kernel Trick

Replace the dot product $x_i^T x_j$ with a kernel function $K(x_i, x_j)$:

**Decision function with kernels**:
$$f(x) = \text{sign}\left(\sum_{i=1}^n \alpha_i y_i K(x_i, x) + b\right)$$

**Common kernel functions**:

**Linear**: $K(x, z) = x^T z$

**Polynomial**: $K(x, z) = (x^T z + c)^d$

**RBF (Radial Basis Function)**: $K(x, z) = \exp\left(-\gamma ||x - z||^2\right)$

**Sigmoid**: $K(x, z) = \tanh(\gamma x^T z + c)$

#### 5. Support Vector Regression (SVR)

For regression, use $\varepsilon$-insensitive loss:

**Optimization problem**:
$$\min_{w,b,\xi,\xi^*} \frac{1}{2}||w||^2 + C\sum_{i=1}^n (\xi_i + \xi_i^*)$$

**Subject to**:
- $y_i - w^T x_i - b \leq \varepsilon + \xi_i$
- $w^T x_i + b - y_i \leq \varepsilon + \xi_i^*$
- $\xi_i, \xi_i^* \geq 0$

## =" Implementation using Libraries

### Scikit-learn Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
from sklearn.datasets import make_classification, make_regression

# Classification Example with Iris Dataset
iris = datasets.load_iris()
X_iris = iris.data[:, :2]  # Use only first 2 features for visualization
y_iris = iris.target

# Binary classification (setosa vs non-setosa)
y_binary = (y_iris != 0).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_iris, y_binary, test_size=0.3, random_state=42, stratify=y_binary
)

# Standardize features (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("SVM Classification Example:")
print(f"Training data shape: {X_train_scaled.shape}")
print(f"Test data shape: {X_test_scaled.shape}")

# Train different SVM models
svm_models = {
    'Linear SVM': SVC(kernel='linear', C=1.0, random_state=42),
    'RBF SVM': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
    'Polynomial SVM': SVC(kernel='poly', degree=3, C=1.0, random_state=42),
    'Sigmoid SVM': SVC(kernel='sigmoid', C=1.0, random_state=42)
}

results = {}
for name, model in svm_models.items():
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'n_support': len(model.support_),
        'support_vectors': model.support_vectors_
    }
    
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Number of support vectors: {len(model.support_)}")

# Detailed analysis of best model (RBF SVM)
best_model = results['RBF SVM']['model']
y_pred_best = best_model.predict(X_test_scaled)

print(f"\nDetailed Results for RBF SVM:")
print(f"Classification Report:")
print(classification_report(y_test, y_pred_best))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - RBF SVM')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
```

### Decision Boundary Visualization

```python
def plot_svm_decision_boundary(X, y, model, title, scaler=None):
    """Plot SVM decision boundary with support vectors"""
    if scaler:
        X = scaler.transform(X)
    
    plt.figure(figsize=(10, 8))
    
    # Create mesh for decision boundary
    h = 0.01  # Step size in mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    # Predict on mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    plt.contour(xx, yy, Z, colors='black', linewidths=0.5)
    
    # Plot data points
    colors = ['red', 'blue']
    for i, color in enumerate(colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, 
                   marker='o', label=f'Class {i}', alpha=0.7)
    
    # Highlight support vectors
    if hasattr(model, 'support_vectors_'):
        plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                   s=100, facecolors='none', edgecolors='black', linewidth=2,
                   label='Support Vectors')
    
    plt.xlabel('Feature 1 (standardized)')
    plt.ylabel('Feature 2 (standardized)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Plot decision boundaries for different kernels
for name, result in results.items():
    plot_svm_decision_boundary(X_train, y_train, result['model'], 
                              f'{name} - Decision Boundary', scaler)
```

### Hyperparameter Tuning

```python
# Comprehensive hyperparameter tuning for RBF SVM
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf']
}

# Grid search with cross-validation
grid_search = GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print(f"\nHyperparameter Tuning Results:")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.3f}")

# Evaluate best model
best_svm = grid_search.best_estimator_
y_pred_tuned = best_svm.predict(X_test_scaled)
tuned_accuracy = accuracy_score(y_test, y_pred_tuned)

print(f"Test accuracy with best parameters: {tuned_accuracy:.3f}")
print(f"Number of support vectors: {len(best_svm.support_)}")

# Visualize hyperparameter effects
results_df = pd.DataFrame(grid_search.cv_results_)

# Heatmap of C vs gamma performance
pivot_table = results_df.pivot_table(
    values='mean_test_score',
    index='param_gamma', 
    columns='param_C'
)

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='viridis')
plt.title('SVM Performance: C vs Gamma')
plt.xlabel('C (Regularization)')
plt.ylabel('Gamma (Kernel coefficient)')
plt.show()
```

### Support Vector Regression (SVR)

```python
# Generate regression dataset
X_reg, y_reg = make_regression(
    n_samples=500, n_features=1, noise=0.1, 
    random_state=42
)

# Split data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

# Standardize
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

# Train SVR models
svr_models = {
    'Linear SVR': SVR(kernel='linear', C=100, epsilon=0.1),
    'RBF SVR': SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1),
    'Polynomial SVR': SVR(kernel='poly', degree=3, C=100, epsilon=0.1)
}

print(f"\nSupport Vector Regression Results:")

plt.figure(figsize=(15, 5))

for i, (name, model) in enumerate(svr_models.items()):
    # Train model
    model.fit(X_train_reg_scaled, y_train_reg)
    
    # Predictions
    y_pred_reg = model.predict(X_test_reg_scaled)
    
    # Evaluate
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_reg, y_pred_reg)
    
    print(f"{name}: RMSE = {rmse:.3f}, R� = {r2:.3f}, Support Vectors = {len(model.support_)}")
    
    # Plot results
    plt.subplot(1, 3, i+1)
    
    # Sort data for smooth line plotting
    X_plot = X_test_reg_scaled
    sort_idx = np.argsort(X_plot[:, 0])
    
    plt.scatter(X_test_reg_scaled, y_test_reg, alpha=0.6, label='Actual', color='blue')
    plt.scatter(X_test_reg_scaled, y_pred_reg, alpha=0.6, label='Predicted', color='red')
    plt.plot(X_plot[sort_idx], y_pred_reg[sort_idx], color='red', linewidth=2)
    
    # Highlight support vectors
    if len(model.support_) > 0:
        support_X = X_train_reg_scaled[model.support_]
        support_y = y_train_reg[model.support_]
        plt.scatter(support_X, support_y, s=100, facecolors='none', 
                   edgecolors='black', linewidth=2, label='Support Vectors')
    
    plt.xlabel('Feature (standardized)')
    plt.ylabel('Target')
    plt.title(f'{name}\nR� = {r2:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Multi-class Classification

```python
# Multi-class classification with full iris dataset
X_multi = iris.data
y_multi = iris.target

X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi, y_multi, test_size=0.3, random_state=42, stratify=y_multi
)

# Standardize
scaler_multi = StandardScaler()
X_train_multi_scaled = scaler_multi.fit_transform(X_train_multi)
X_test_multi_scaled = scaler_multi.transform(X_test_multi)

# Train multi-class SVM
svm_multi = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
svm_multi.fit(X_train_multi_scaled, y_train_multi)

# Predictions
y_pred_multi = svm_multi.predict(X_test_multi_scaled)

# Evaluate
accuracy_multi = accuracy_score(y_test_multi, y_pred_multi)

print(f"\nMulti-class Classification Results:")
print(f"Accuracy: {accuracy_multi:.3f}")
print(f"Total support vectors: {len(svm_multi.support_)}")
print(f"Support vectors per class: {svm_multi.n_support_}")

print(f"\nClassification Report:")
print(classification_report(y_test_multi, y_pred_multi, 
                          target_names=iris.target_names))

# Confusion matrix
plt.figure(figsize=(8, 6))
cm_multi = confusion_matrix(y_test_multi, y_pred_multi)
sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title('Multi-class SVM - Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
```

## � From Scratch Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

class SVMFromScratch:
    def __init__(self, C=1.0, kernel='linear', gamma='scale', degree=3, max_iter=1000):
        """
        Support Vector Machine implementation from scratch
        
        Parameters:
        C: Regularization parameter
        kernel: Kernel function ('linear', 'rbf', 'poly')
        gamma: Kernel coefficient for RBF and polynomial kernels
        degree: Degree for polynomial kernel
        max_iter: Maximum number of iterations
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.max_iter = max_iter
        
        # Model parameters
        self.alpha = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.support_vector_alpha = None
        self.b = None
        self.X_train = None
        self.y_train = None
        
    def _kernel_function(self, x1, x2):
        """Compute kernel function between two vectors"""
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        
        elif self.kernel == 'rbf':
            if self.gamma == 'scale':
                gamma = 1.0 / (x1.shape[0] * np.var(x1))
            else:
                gamma = self.gamma
            return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)
        
        elif self.kernel == 'poly':
            if self.gamma == 'scale':
                gamma = 1.0 / (x1.shape[0] * np.var(x1))
            else:
                gamma = self.gamma
            return (gamma * np.dot(x1, x2) + 1) ** self.degree
        
        else:
            raise ValueError("Unsupported kernel type")
    
    def _compute_kernel_matrix(self, X1, X2):
        """Compute kernel matrix between two sets of points"""
        n1, n2 = X1.shape[0], X2.shape[0]
        kernel_matrix = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                kernel_matrix[i, j] = self._kernel_function(X1[i], X2[j])
        
        return kernel_matrix
    
    def fit(self, X, y):
        """
        Train SVM using SMO (Sequential Minimal Optimization) algorithm
        Simplified implementation
        """
        n_samples, n_features = X.shape
        self.X_train = X
        self.y_train = y
        
        # Convert labels to -1 and 1
        y = np.where(y <= 0, -1, 1)
        
        # Initialize alpha
        self.alpha = np.zeros(n_samples)
        self.b = 0
        
        # Compute kernel matrix
        K = self._compute_kernel_matrix(X, X)
        
        # SMO algorithm (simplified)
        for iteration in range(self.max_iter):
            alpha_prev = np.copy(self.alpha)
            
            for j in range(n_samples):
                # Calculate prediction for point j
                prediction_j = np.sum(self.alpha * y * K[:, j]) + self.b
                
                # Calculate error
                E_j = prediction_j - y[j]
                
                # Check KKT conditions
                if (y[j] * E_j < -1e-3 and self.alpha[j] < self.C) or \
                   (y[j] * E_j > 1e-3 and self.alpha[j] > 0):
                    
                    # Select second alpha randomly
                    i = j
                    while i == j:
                        i = np.random.randint(0, n_samples)
                    
                    # Calculate prediction and error for point i
                    prediction_i = np.sum(self.alpha * y * K[:, i]) + self.b
                    E_i = prediction_i - y[i]
                    
                    # Save old alphas
                    alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]
                    
                    # Compute bounds
                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])
                    
                    if L == H:
                        continue
                    
                    # Compute eta
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    
                    # Update alpha_j
                    self.alpha[j] = self.alpha[j] - (y[j] * (E_i - E_j)) / eta
                    
                    # Clip alpha_j
                    self.alpha[j] = max(L, min(H, self.alpha[j]))
                    
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    # Update alpha_i
                    self.alpha[i] = self.alpha[i] + y[i] * y[j] * (alpha_j_old - self.alpha[j])
                    
                    # Update bias
                    b1 = self.b - E_i - y[i] * (self.alpha[i] - alpha_i_old) * K[i, i] - \
                         y[j] * (self.alpha[j] - alpha_j_old) * K[i, j]
                    
                    b2 = self.b - E_j - y[i] * (self.alpha[i] - alpha_i_old) * K[i, j] - \
                         y[j] * (self.alpha[j] - alpha_j_old) * K[j, j]
                    
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
            
            # Check convergence
            if np.allclose(self.alpha, alpha_prev, atol=1e-5):
                break
        
        # Identify support vectors
        support_vector_indices = np.where(self.alpha > 1e-5)[0]
        self.support_vectors = X[support_vector_indices]
        self.support_vector_labels = y[support_vector_indices]
        self.support_vector_alpha = self.alpha[support_vector_indices]
        
        print(f"Training completed in {iteration + 1} iterations")
        print(f"Number of support vectors: {len(support_vector_indices)}")
        
    def predict(self, X):
        """Make predictions on new data"""
        if self.support_vectors is None:
            raise ValueError("Model has not been trained yet")
        
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        
        for i in range(n_samples):
            prediction = 0
            for j in range(len(self.support_vectors)):
                prediction += (self.support_vector_alpha[j] * 
                             self.support_vector_labels[j] * 
                             self._kernel_function(X[i], self.support_vectors[j]))
            predictions[i] = prediction + self.b
        
        return np.sign(predictions).astype(int)
    
    def decision_function(self, X):
        """Return decision function values"""
        if self.support_vectors is None:
            raise ValueError("Model has not been trained yet")
        
        n_samples = X.shape[0]
        decisions = np.zeros(n_samples)
        
        for i in range(n_samples):
            decision = 0
            for j in range(len(self.support_vectors)):
                decision += (self.support_vector_alpha[j] * 
                           self.support_vector_labels[j] * 
                           self._kernel_function(X[i], self.support_vectors[j]))
            decisions[i] = decision + self.b
        
        return decisions

# Demonstration with synthetic dataset
np.random.seed(42)

# Generate synthetic dataset
X_demo, y_demo = make_classification(
    n_samples=200, n_features=2, n_redundant=0, n_informative=2,
    random_state=42, n_clusters_per_class=1
)

# Convert to binary classification
y_demo = np.where(y_demo == 0, -1, 1)

# Split data
X_train_demo, X_test_demo, y_train_demo, y_test_demo = train_test_split(
    X_demo, y_demo, test_size=0.3, random_state=42
)

# Standardize
scaler_demo = StandardScaler()
X_train_demo_scaled = scaler_demo.fit_transform(X_train_demo)
X_test_demo_scaled = scaler_demo.transform(X_test_demo)

# Train custom SVM
print("Training Custom SVM:")
svm_custom = SVMFromScratch(C=1.0, kernel='rbf', gamma=1.0)
svm_custom.fit(X_train_demo_scaled, y_train_demo)

# Predictions
y_pred_custom = svm_custom.predict(X_test_demo_scaled)

# Compare with sklearn
from sklearn.svm import SVC
svm_sklearn = SVC(kernel='rbf', C=1.0, gamma=1.0)
svm_sklearn.fit(X_train_demo_scaled, y_train_demo)
y_pred_sklearn = svm_sklearn.predict(X_test_demo_scaled)

# Evaluate
accuracy_custom = np.mean(y_pred_custom == y_test_demo)
accuracy_sklearn = np.mean(y_pred_sklearn == y_test_demo)

print(f"\nComparison Results:")
print(f"Custom SVM accuracy: {accuracy_custom:.3f}")
print(f"Sklearn SVM accuracy: {accuracy_sklearn:.3f}")
print(f"Difference: {abs(accuracy_custom - accuracy_sklearn):.3f}")

# Visualize results
def plot_svm_comparison(X, y, svm_custom, svm_sklearn, title_custom, title_sklearn):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Create mesh
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Plot custom SVM
    Z_custom = svm_custom.decision_function(mesh_points)
    Z_custom = Z_custom.reshape(xx.shape)
    
    ax1.contourf(xx, yy, Z_custom, levels=50, alpha=0.3, cmap=plt.cm.RdYlBu)
    ax1.contour(xx, yy, Z_custom, levels=[0], colors='black', linewidths=2)
    
    # Plot data points
    colors = ['red', 'blue']
    for i, color in enumerate([-1, 1]):
        idx = np.where(y == color)[0]
        ax1.scatter(X[idx, 0], X[idx, 1], c=colors[i], marker='o', 
                   label=f'Class {color}', alpha=0.7)
    
    # Highlight support vectors
    if svm_custom.support_vectors is not None:
        ax1.scatter(svm_custom.support_vectors[:, 0], svm_custom.support_vectors[:, 1],
                   s=100, facecolors='none', edgecolors='black', linewidth=2,
                   label='Support Vectors')
    
    ax1.set_title(title_custom)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot sklearn SVM
    Z_sklearn = svm_sklearn.decision_function(mesh_points)
    Z_sklearn = Z_sklearn.reshape(xx.shape)
    
    ax2.contourf(xx, yy, Z_sklearn, levels=50, alpha=0.3, cmap=plt.cm.RdYlBu)
    ax2.contour(xx, yy, Z_sklearn, levels=[0], colors='black', linewidths=2)
    
    for i, color in enumerate([-1, 1]):
        idx = np.where(y == color)[0]
        ax2.scatter(X[idx, 0], X[idx, 1], c=colors[i], marker='o', 
                   label=f'Class {color}', alpha=0.7)
    
    # Highlight support vectors
    ax2.scatter(svm_sklearn.support_vectors_[:, 0], svm_sklearn.support_vectors_[:, 1],
               s=100, facecolors='none', edgecolors='black', linewidth=2,
               label='Support Vectors')
    
    ax2.set_title(title_sklearn)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

plot_svm_comparison(X_train_demo_scaled, y_train_demo, svm_custom, svm_sklearn,
                   f'Custom SVM (Acc: {accuracy_custom:.3f})',
                   f'Sklearn SVM (Acc: {accuracy_sklearn:.3f})')
```

## � Assumptions and Limitations

### Key Assumptions

1. **Margin maximization is optimal**: Assumes that maximizing margin leads to better generalization
2. **Support vector sufficiency**: Only support vectors matter for the decision boundary
3. **Kernel validity**: Chosen kernel should satisfy Mercer's conditions
4. **Feature scaling**: SVM is sensitive to feature scales (assumes standardized features)
5. **Data quality**: Assumes training data is representative of test distribution

### Limitations

1. **Computational complexity**: O(n�) training complexity for SMO algorithm
   - **Impact**: Slow on large datasets (>10,000 samples)
   - **Solution**: Use approximate methods, sub-sampling, or linear SVM

2. **Memory requirements**: Stores support vectors and kernel matrix
   - **Impact**: Memory issues with large datasets or complex kernels
   - **Solution**: Use linear kernels, feature selection, or incremental learning

3. **No probabilistic output**: Standard SVM provides only class predictions
   - **Solution**: Use Platt scaling or cross-validation for probability estimates

4. **Sensitive to feature scaling**: Different scales can dominate the kernel
   - **Solution**: Always standardize features before training

5. **Hyperparameter sensitivity**: Performance heavily depends on C and kernel parameters
   - **Solution**: Use cross-validation for hyperparameter tuning

6. **Limited interpretability**: Kernel SVMs create complex decision boundaries
   - **Alternative**: Use linear SVM or other interpretable models when needed

### Comparison with Other Algorithms

| Algorithm | Training Speed | Prediction Speed | Memory Usage | Interpretability | Non-linear Capability |
|-----------|----------------|------------------|---------------|------------------|----------------------|
| SVM | Slow (O(n�)) | Fast | High | Low (kernel) | High |
| Logistic Regression | Fast | Very Fast | Low | High | Low |
| Random Forest | Medium | Medium | Medium | Medium | High |
| Neural Networks | Slow | Fast | High | Very Low | Very High |
| k-NN | Very Fast | Slow | Medium | High | High |
| Naive Bayes | Very Fast | Very Fast | Low | High | Low |

**When to use SVM:**
-  High-dimensional data
-  Clear margin of separation exists  
-  More features than samples
-  Non-linear relationships (with kernels)
-  Robust to outliers needed

**When to avoid SVM:**
- L Very large datasets (>100k samples)
- L Noisy data with overlapping classes
- L Need probability estimates
- L Real-time prediction requirements
- L Interpretability is crucial

## ❓ Interview Questions

??? question "Explain the mathematical intuition behind SVM and the concept of margin maximization."

    **Answer:** SVM finds the hyperplane that separates classes with maximum margin:
    
    **Mathematical foundation**:
    1. **Decision boundary**: $w^T x + b = 0$
    2. **Margin**: Distance from hyperplane to nearest points = $\frac{1}{||w||}$
    3. **Optimization**: Maximize margin = Minimize $\frac{1}{2}||w||^2$
    4. **Constraints**: Ensure correct classification: $y_i(w^T x_i + b) \geq 1$
    
    **Intuition**: 
    - Larger margins � better generalization (statistical learning theory)
    - Only support vectors (points on margin) determine decision boundary
    - All other points could be removed without changing the model
    
    **Why maximize margin?**
    - Provides robustness against small perturbations
    - Reduces VC dimension � better generalization bounds
    - Unique solution (convex optimization problem)

??? question "What is the kernel trick and how does it enable SVM to handle non-linear data?"

    **Answer:** The kernel trick allows SVM to handle non-linear data without explicitly computing high-dimensional transformations:
    
    **The trick**:
    1. **Replace dot products** in dual formulation with kernel function: $x_i^T x_j � K(x_i, x_j)$
    2. **Implicit mapping**: $K(x_i, x_j) = �(x_i)^T �(x_j)$ where � maps to higher dimension
    3. **No explicit computation** of �(x) needed
    
    **Popular kernels**:
    ```python
    # Linear: K(x,z) = x^T z
    # Polynomial: K(x,z) = (x^T z + c)^d
    # RBF: K(x,z) = exp(-�||x-z||�)
    # Sigmoid: K(x,z) = tanh(�x^T z + c)
    ```
    
    **Example**: RBF kernel maps data to infinite-dimensional space, allowing separation of any finite dataset
    
    **Advantages**:
    - Computational efficiency (no explicit mapping)
    - Handles complex non-linear relationships
    - Mathematical elegance through Mercer's theorem
    
    **Limitations**: 
    - Kernel choice is crucial
    - Interpretability decreases
    - Hyperparameter tuning becomes more complex

??? question "How do you choose appropriate hyperparameters (C, gamma, kernel) for SVM?"

    **Answer:** Systematic approach to SVM hyperparameter tuning:
    
    **Key hyperparameters**:
    
    **1. Regularization parameter C**:
    - **Small C**: Soft margin, more misclassifications allowed, prevents overfitting
    - **Large C**: Hard margin, fewer misclassifications, risk of overfitting
    - **Typical range**: [0.1, 1, 10, 100, 1000]
    
    **2. Kernel parameter gamma (for RBF/poly)**:
    - **Small gamma**: Far-reaching influence, smoother boundaries
    - **Large gamma**: Close influence, complex boundaries, overfitting risk
    - **Typical values**: ['scale', 'auto', 0.001, 0.01, 0.1, 1]
    
    **3. Kernel selection**:
    ```python
    # Linear: Good for high-dimensional, linearly separable data
    # RBF: Default choice, good for most non-linear problems
    # Polynomial: Specific polynomial relationships
    # Sigmoid: Neural network-like behavior
    ```
    
    **Tuning strategy**:
    ```python
    # Grid search with cross-validation
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'poly', 'linear']
    }
    GridSearchCV(SVC(), param_grid, cv=5)
    ```
    
    **Best practices**:
    - Start with default parameters
    - Use cross-validation for unbiased estimates
    - Consider computational constraints
    - Validate on separate test set

??? question "What's the difference between hard margin and soft margin SVM?"

    **Answer:** Key differences in handling non-separable data:
    
    **Hard Margin SVM**:
    - **Assumption**: Data is linearly separable
    - **Constraint**: All points correctly classified: $y_i(w^T x_i + b) \geq 1$
    - **Objective**: $\min \frac{1}{2}||w||^2$
    - **Problem**: No solution exists if data isn't separable
    - **Use case**: Clean, separable data
    
    **Soft Margin SVM**:
    - **Assumption**: Data may have noise/overlap
    - **Slack variables**: $�_i e 0$ allow constraint violations
    - **Modified constraints**: $y_i(w^T x_i + b) \geq 1 - �_i$
    - **Objective**: $\min \frac{1}{2}||w||^2 + C\sum �_i$
    - **Trade-off**: Margin maximization vs. training error
    
    **C parameter controls**:
    - $C � $: Approaches hard margin (no violations)
    - $C � 0$: Allows many violations (maximum margin)
    
    **Practical impact**:
    ```python
    # Hard margin equivalent
    SVC(C=1e6)  # Very large C
    
    # Soft margin
    SVC(C=1.0)   # Balanced trade-off
    ```
    
    **When to use**:
    - **Hard margin**: Perfect data, small datasets
    - **Soft margin**: Real-world data (recommended)

??? question "How does SVM handle multi-class classification?"

    **Answer:** SVM is inherently binary, but extends to multi-class using two main strategies:
    
    **1. One-vs-Rest (OvR)**:
    - Train K binary classifiers (K = number of classes)
    - Each classifier: "Class i vs All other classes"
    - Prediction: Class with highest decision function score
    - **Pros**: Simple, efficient
    - **Cons**: Imbalanced datasets per classifier
    
    ```python
    # Automatic in sklearn
    SVC()  # Uses OvR by default
    
    # Explicit
    from sklearn.multiclass import OneVsRestClassifier
    OneVsRestClassifier(SVC())
    ```
    
    **2. One-vs-One (OvO)**:
    - Train K(K-1)/2 binary classifiers
    - Each classifier: "Class i vs Class j"
    - Prediction: Majority voting among all classifiers
    - **Pros**: Balanced datasets, often more accurate
    - **Cons**: More classifiers to train
    
    ```python
    # In sklearn
    SVC(decision_function_shape='ovo')
    
    # Explicit
    from sklearn.multiclass import OneVsOneClassifier
    OneVsOneClassifier(SVC())
    ```
    
    **Comparison**:
    | Aspect | OvR | OvO |
    |--------|-----|-----|
    | **Classifiers** | K | K(K-1)/2 |
    | **Training time** | Faster | Slower |
    | **Prediction time** | Faster | Slower |
    | **Accuracy** | Good | Often better |
    | **Memory** | Less | More |
    
    **Decision function**:
    - OvR: Use raw scores from each classifier
    - OvO: Aggregate pairwise comparisons

??? question "What are the advantages and disadvantages of different SVM kernels?"

    **Answer:** Comprehensive comparison of SVM kernels:
    
    **Linear Kernel**: $K(x,z) = x^T z$
    
    **Advantages**:
    -  Fast training and prediction
    -  Interpretable (weights have meaning)
    -  Good for high-dimensional data
    -  Less prone to overfitting
    -  No hyperparameters to tune
    
    **Disadvantages**:
    - L Only linear decision boundaries
    - L Poor for complex non-linear relationships
    
    **Use when**: Text classification, high-dimensional data, linear relationships
    
    **RBF (Gaussian) Kernel**: $K(x,z) = \exp(-\gamma||x-z||^2)$
    
    **Advantages**:
    -  Handles non-linear relationships
    -  Universal approximator
    -  Works well as default choice
    -  Smooth decision boundaries
    
    **Disadvantages**:
    - L Requires hyperparameter tuning (�)
    - L Can overfit with large �
    - L Less interpretable
    - L Slower than linear
    
    **Use when**: Non-linear data, default choice for most problems
    
    **Polynomial Kernel**: $K(x,z) = (x^T z + c)^d$
    
    **Advantages**:
    -  Good for specific polynomial relationships
    -  Interpretable degree parameter
    -  Can capture interactions
    
    **Disadvantages**:
    - L Computationally expensive for high degrees
    - L Numerical instability
    - L Less general than RBF
    - L Multiple hyperparameters
    
    **Use when**: Known polynomial relationships in data
    
    **Sigmoid Kernel**: $K(x,z) = \tanh(\gamma x^T z + c)$
    
    **Advantages**:
    -  Neural network-like behavior
    -  S-shaped decision boundaries
    
    **Disadvantages**:
    - L Not positive semi-definite (violates Mercer's condition)
    - L Can be unstable
    - L Often outperformed by RBF
    - L Limited practical use
    
    **Selection guidelines**:
    1. Start with RBF (default choice)
    2. Try linear if high-dimensional
    3. Use polynomial for specific domain knowledge
    4. Avoid sigmoid unless specific need

??? question "How do you handle imbalanced datasets with SVM?"

    **Answer:** Several strategies for handling class imbalance in SVM:
    
    **1. Class weight balancing**:
    ```python
    # Automatic balancing
    SVC(class_weight='balanced')
    
    # Manual weights
    SVC(class_weight={0: 1, 1: 10})  # 10x weight for minority class
    
    # Effect: Increases penalty for misclassifying minority class
    ```
    
    **2. Resampling techniques**:
    ```python
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    
    # Oversample minority class
    smote = SMOTE()
    X_balanced, y_balanced = smote.fit_resample(X, y)
    
    # Undersample majority class  
    undersampler = RandomUnderSampler()
    X_balanced, y_balanced = undersampler.fit_resample(X, y)
    ```
    
    **3. Threshold adjustment**:
    ```python
    # Use decision function for custom thresholds
    scores = svm.decision_function(X_test)
    # Instead of scores > 0, use scores > custom_threshold
    predictions = (scores > optimal_threshold).astype(int)
    ```
    
    **4. Cost-sensitive learning**:
    - Modify C parameter per class
    - Different misclassification costs
    ```python
    # Higher C for minority class
    SVC(C=100, class_weight={0: 1, 1: 5})
    ```
    
    **5. Evaluation metrics**:
    ```python
    # Don't use accuracy for imbalanced data
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import roc_auc_score
    
    # Use precision, recall, F1-score, AUC
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred)
    auc = roc_auc_score(y_true, decision_scores)
    ```
    
    **Best practices**:
    - Combine multiple techniques
    - Use stratified cross-validation
    - Focus on minority class performance
    - Consider ensemble methods as alternative

??? question "Explain the computational complexity of SVM training and prediction."

    **Answer:** Detailed complexity analysis:
    
    **Training Complexity**:
    
    **SMO Algorithm** (most common):
    - **Time**: O(n�) to O(n�) depending on data
    - **Average case**: O(n���) for most datasets
    - **Worst case**: O(n�) for very difficult datasets
    - **Space**: O(n�) for kernel matrix storage
    
    **Factors affecting training time**:
    ```python
    # Dataset size (most important)
    n_samples = 1000    # Fast
    n_samples = 100000  # Very slow
    
    # Kernel complexity
    kernel='linear'     # Fastest
    kernel='rbf'        # Medium  
    kernel='poly'       # Slower
    
    # Hyperparameters
    C=0.1              # Faster (more violations allowed)
    C=1000             # Slower (strict constraints)
    ```
    
    **Prediction Complexity**:
    - **Time**: O(n_support_vectors � n_features)
    - **Typical**: Much faster than training
    - **Linear kernel**: O(n_features) - very fast
    - **Non-linear**: O(n_sv � n_features) - depends on support vectors
    
    **Memory Requirements**:
    ```python
    # Kernel matrix: n � n � 8 bytes (for RBF/poly)
    memory_gb = (n_samples ** 2 * 8) / (1024**3)
    
    # For 10,000 samples: ~0.75 GB
    # For 100,000 samples: ~75 GB (impractical)
    ```
    
    **Scalability solutions**:
    1. **Linear SVM**: Use for n > 10,000
    2. **Sampling**: Train on subset of data
    3. **Online SVM**: Incremental learning algorithms
    4. **Approximate methods**: Nystr�m approximation
    5. **Alternative algorithms**: Random Forest, XGBoost for large data
    
    **Practical guidelines**:
    - n < 1,000: Any kernel works
    - 1,000 < n < 10,000: RBF with tuning
    - n > 10,000: Consider linear SVM or alternatives
    - n > 100,000: Use other algorithms

??? question "How do you interpret and visualize SVM results?"

    **Answer:** Multiple approaches for SVM interpretation:
    
    **1. Decision boundaries (2D visualization)**:
    ```python
    def plot_svm_boundary(X, y, model):
        # Create mesh
        h = 0.01
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Predict on mesh
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot boundary and margins
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
        
        # Highlight support vectors
        plt.scatter(model.support_vectors_[:, 0], 
                   model.support_vectors_[:, 1],
                   s=100, facecolors='none', edgecolors='black')
    ```
    
    **2. Support vector analysis**:
    ```python
    print(f"Number of support vectors: {len(model.support_)}")
    print(f"Support vector ratio: {len(model.support_)/len(X_train):.2%}")
    print(f"Support vectors per class: {model.n_support_}")
    
    # High ratio might indicate:
    # - Complex decision boundary
    # - Noisy data  
    # - Need for different kernel/parameters
    ```
    
    **3. Feature importance (linear kernel only)**:
    ```python
    if model.kernel == 'linear':
        # Coefficients indicate feature importance
        feature_importance = abs(model.coef_[0])
        
        plt.barh(feature_names, feature_importance)
        plt.title('Linear SVM Feature Importance')
    ```
    
    **4. Decision function analysis**:
    ```python
    # Distance from hyperplane
    decision_scores = model.decision_function(X_test)
    
    # Confidence interpretation
    # |score| > 1: High confidence
    # |score| < 1: Low confidence (near boundary)
    
    plt.hist(decision_scores, bins=30)
    plt.axvline(x=0, color='red', linestyle='--', label='Decision boundary')
    plt.axvline(x=1, color='orange', linestyle='--', label='Margin')
    plt.axvline(x=-1, color='orange', linestyle='--')
    ```
    
    **5. Hyperparameter sensitivity analysis**:
    ```python
    # Plot performance vs hyperparameters
    C_values = [0.1, 1, 10, 100]
    scores = []
    
    for C in C_values:
        model = SVC(C=C, kernel='rbf')
        score = cross_val_score(model, X, y, cv=5).mean()
        scores.append(score)
    
    plt.plot(C_values, scores)
    plt.xlabel('C (log scale)')
    plt.xscale('log')
    plt.ylabel('Cross-validation accuracy')
    ```
    
    **6. Error analysis**:
    ```python
    # Analyze misclassified points
    y_pred = model.predict(X_test)
    misclassified = X_test[y_test != y_pred]
    
    # Are they near the decision boundary?
    decision_scores_errors = model.decision_function(misclassified)
    print(f"Average distance from boundary: {np.mean(abs(decision_scores_errors))}")
    ```

## >� Examples

### Real-world Example: Text Classification

```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Load text dataset (subset of 20 newsgroups)
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
newsgroups = fetch_20newsgroups(
    subset='all',
    categories=categories, 
    shuffle=True,
    random_state=42,
    remove=('headers', 'footers', 'quotes')
)

print("Text Classification with SVM")
print(f"Dataset shape: {len(newsgroups.data)} documents")
print(f"Categories: {newsgroups.target_names}")
print(f"Class distribution:")
for i, name in enumerate(newsgroups.target_names):
    count = sum(newsgroups.target == i)
    print(f"  {name}: {count} documents")

# Split data
X_text, X_test_text, y_text, y_test_text = train_test_split(
    newsgroups.data, newsgroups.target, 
    test_size=0.2, random_state=42, stratify=newsgroups.target
)

# Create pipeline with TF-IDF and SVM
text_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=10000,      # Limit vocabulary
        min_df=2,                # Ignore rare words  
        max_df=0.95,             # Ignore too common words
        stop_words='english',     # Remove stop words
        ngram_range=(1, 2)       # Use unigrams and bigrams
    )),
    ('svm', SVC(
        kernel='linear',         # Linear works well for text
        C=1.0,
        random_state=42
    ))
])

# Train model
print("\nTraining SVM text classifier...")
text_pipeline.fit(X_text, y_text)

# Predictions
y_pred_text = text_pipeline.predict(X_test_text)

# Evaluate
accuracy_text = np.mean(y_pred_text == y_test_text)
print(f"Test accuracy: {accuracy_text:.3f}")

# Detailed classification report
print(f"\nClassification Report:")
print(classification_report(y_test_text, y_pred_text, 
                          target_names=newsgroups.target_names))

# Confusion matrix
plt.figure(figsize=(8, 6))
cm_text = confusion_matrix(y_test_text, y_pred_text)
sns.heatmap(cm_text, annot=True, fmt='d', cmap='Blues',
            xticklabels=newsgroups.target_names,
            yticklabels=newsgroups.target_names)
plt.title('Text Classification - Confusion Matrix')
plt.ylabel('True Category')
plt.xlabel('Predicted Category')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Feature importance analysis (most important words)
feature_names = text_pipeline.named_steps['tfidf'].get_feature_names_out()
svm_model = text_pipeline.named_steps['svm']

# For each class, show most important features
n_features = 10
for i, category in enumerate(newsgroups.target_names):
    if hasattr(svm_model, 'coef_'):
        # Get coefficients for this class (one-vs-rest)
        if len(svm_model.classes_) == 2:
            coef = svm_model.coef_[0] if i == 1 else -svm_model.coef_[0]
        else:
            coef = svm_model.coef_[i]
        
        # Get top features
        top_positive_indices = coef.argsort()[-n_features:][::-1]
        top_negative_indices = coef.argsort()[:n_features]
        
        print(f"\nMost important features for '{category}':")
        print("Positive indicators:")
        for idx in top_positive_indices:
            print(f"  {feature_names[idx]}: {coef[idx]:.3f}")
        
        print("Negative indicators:")
        for idx in top_negative_indices:
            print(f"  {feature_names[idx]}: {coef[idx]:.3f}")

# Cross-validation performance
cv_scores = cross_val_score(text_pipeline, X_text, y_text, cv=5)
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Example predictions with confidence
sample_texts = [
    "I believe in God and Jesus Christ",
    "The graphics card is not working properly",
    "This medical treatment showed promising results",
    "There is no scientific evidence for the existence of God"
]

print(f"\nExample Predictions:")
for text in sample_texts:
    prediction = text_pipeline.predict([text])[0]
    decision_score = text_pipeline.decision_function([text])
    predicted_category = newsgroups.target_names[prediction]
    
    print(f"\nText: '{text[:50]}...'")
    print(f"Predicted: {predicted_category}")
    print(f"Decision scores: {decision_score[0]}")
```

### Image Classification Example

```python
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load face recognition dataset
faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X_faces = faces.data
y_faces = faces.target

print("Face Recognition with SVM")
print(f"Dataset shape: {X_faces.shape}")
print(f"Number of people: {len(np.unique(y_faces))}")
print(f"Image dimensions: 64x64 pixels")

# Visualize some sample faces
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_faces[i].reshape(64, 64), cmap='gray')
    ax.set_title(f'Person {y_faces[i]}')
    ax.axis('off')
plt.suptitle('Sample Face Images')
plt.tight_layout()
plt.show()

# Split data
X_train_faces, X_test_faces, y_train_faces, y_test_faces = train_test_split(
    X_faces, y_faces, test_size=0.25, random_state=42, stratify=y_faces
)

# Apply PCA for dimensionality reduction (faces are high-dimensional)
n_components = 150  # Reduce from 4096 to 150 dimensions
pca_faces = PCA(n_components=n_components, whiten=True, random_state=42)
X_train_pca = pca_faces.fit_transform(X_train_faces)
X_test_pca = pca_faces.transform(X_test_faces)

print(f"\nDimensionality reduction:")
print(f"Original dimensions: {X_train_faces.shape[1]}")
print(f"Reduced dimensions: {X_train_pca.shape[1]}")
print(f"Variance explained: {pca_faces.explained_variance_ratio_.sum():.3f}")

# Train SVM classifier
svm_faces = SVC(kernel='rbf', C=1000, gamma=0.005, random_state=42)
svm_faces.fit(X_train_pca, y_train_faces)

# Predictions
y_pred_faces = svm_faces.predict(X_test_pca)

# Evaluate
accuracy_faces = accuracy_score(y_test_faces, y_pred_faces)
print(f"\nFace Recognition Results:")
print(f"Accuracy: {accuracy_faces:.3f}")
print(f"Number of support vectors: {len(svm_faces.support_)}")
print(f"Support vector ratio: {len(svm_faces.support_)/len(X_train_pca):.2%}")

# Visualize some predictions
fig, axes = plt.subplots(3, 6, figsize=(15, 9))
for i, ax in enumerate(axes.flat):
    if i < len(X_test_faces):
        # Show original image
        ax.imshow(X_test_faces[i].reshape(64, 64), cmap='gray')
        
        # Get prediction and confidence
        true_label = y_test_faces[i] 
        pred_label = y_pred_faces[i]
        decision_score = svm_faces.decision_function([X_test_pca[i]])
        confidence = np.max(decision_score)
        
        # Color border based on correctness
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f'True: {true_label}, Pred: {pred_label}\nConf: {confidence:.2f}', 
                    color=color, fontsize=8)
        ax.axis('off')
        
        # Add border
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(3)
    else:
        ax.axis('off')

plt.suptitle('Face Recognition Predictions (Green=Correct, Red=Incorrect)')
plt.tight_layout()
plt.show()

# Analyze errors
incorrect_indices = np.where(y_test_faces != y_pred_faces)[0]
print(f"\nError Analysis:")
print(f"Total errors: {len(incorrect_indices)}")

if len(incorrect_indices) > 0:
    # Show decision scores for incorrect predictions
    incorrect_scores = svm_faces.decision_function(X_test_pca[incorrect_indices])
    avg_incorrect_confidence = np.mean(np.max(incorrect_scores, axis=1))
    
    correct_indices = np.where(y_test_faces == y_pred_faces)[0]
    correct_scores = svm_faces.decision_function(X_test_pca[correct_indices])
    avg_correct_confidence = np.mean(np.max(correct_scores, axis=1))
    
    print(f"Average confidence for correct predictions: {avg_correct_confidence:.3f}")
    print(f"Average confidence for incorrect predictions: {avg_incorrect_confidence:.3f}")
    
    # Plot confidence distribution
    plt.figure(figsize=(10, 6))
    plt.hist(np.max(correct_scores, axis=1), bins=20, alpha=0.7, 
             label='Correct predictions', color='green')
    plt.hist(np.max(incorrect_scores, axis=1), bins=20, alpha=0.7, 
             label='Incorrect predictions', color='red')
    plt.xlabel('Maximum Decision Score (Confidence)')
    plt.ylabel('Frequency')
    plt.title('Confidence Distribution: Correct vs Incorrect Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Hyperparameter sensitivity analysis
C_values = [1, 10, 100, 1000]
gamma_values = [0.001, 0.005, 0.01, 0.05]

results_grid = np.zeros((len(C_values), len(gamma_values)))

print(f"\nHyperparameter sensitivity analysis:")
for i, C in enumerate(C_values):
    for j, gamma in enumerate(gamma_values):
        svm_temp = SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)
        svm_temp.fit(X_train_pca, y_train_faces)
        score = svm_temp.score(X_test_pca, y_test_faces)
        results_grid[i, j] = score
        print(f"C={C:4}, gamma={gamma:.3f}: {score:.3f}")

# Visualize hyperparameter effects
plt.figure(figsize=(8, 6))
sns.heatmap(results_grid, 
           xticklabels=[f'{g:.3f}' for g in gamma_values],
           yticklabels=C_values,
           annot=True, fmt='.3f', cmap='viridis')
plt.title('Face Recognition Accuracy: C vs Gamma')
plt.xlabel('Gamma')
plt.ylabel('C')
plt.show()
```

### Regression Example with Support Vector Regression (SVR)

```python
from sklearn.svm import SVR
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Generate synthetic regression dataset with noise
np.random.seed(42)
X_reg, y_reg = make_regression(
    n_samples=300, 
    n_features=1, 
    noise=15,
    random_state=42
)

# Add some outliers
outlier_indices = np.random.choice(len(X_reg), size=20, replace=False)
y_reg[outlier_indices] += np.random.normal(0, 50, size=20)

# Sort for plotting
sort_indices = np.argsort(X_reg[:, 0])
X_reg_sorted = X_reg[sort_indices]
y_reg_sorted = y_reg[sort_indices]

print("Support Vector Regression Example")
print(f"Dataset shape: {X_reg.shape}")
print(f"Target range: [{y_reg.min():.1f}, {y_reg.max():.1f}]")

# Split data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

# Standardize features
scaler_svr = StandardScaler()
X_train_reg_scaled = scaler_svr.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_svr.transform(X_test_reg)

# Train different SVR models
svr_models = {
    'Linear SVR': SVR(kernel='linear', C=100, epsilon=0.1),
    'RBF SVR': SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1),
    'Polynomial SVR': SVR(kernel='poly', degree=3, C=100, epsilon=0.1)
}

plt.figure(figsize=(15, 10))

for i, (name, model) in enumerate(svr_models.items()):
    # Train model
    model.fit(X_train_reg_scaled, y_train_reg)
    
    # Predictions
    y_pred_train = model.predict(X_train_reg_scaled)
    y_pred_test = model.predict(X_test_reg_scaled)
    
    # Evaluate
    train_r2 = r2_score(y_train_reg, y_pred_train)
    test_r2 = r2_score(y_test_reg, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_test))
    test_mae = mean_absolute_error(y_test_reg, y_pred_test)
    
    print(f"\n{name} Results:")
    print(f"Train R�: {train_r2:.3f}")
    print(f"Test R�: {test_r2:.3f}")
    print(f"Test RMSE: {test_rmse:.3f}")
    print(f"Test MAE: {test_mae:.3f}")
    print(f"Support vectors: {len(model.support_)} ({len(model.support_)/len(X_train_reg_scaled)*100:.1f}%)")
    
    # Plot results
    plt.subplot(2, 3, i+1)
    
    # Create smooth line for predictions
    X_plot = scaler_svr.transform(X_reg_sorted.reshape(-1, 1))
    y_plot = model.predict(X_plot)
    
    # Plot data points
    plt.scatter(X_train_reg_scaled, y_train_reg, alpha=0.6, color='blue', 
                label='Training data', s=30)
    plt.scatter(X_test_reg_scaled, y_test_reg, alpha=0.6, color='red', 
                label='Test data', s=30)
    
    # Plot prediction line
    plt.plot(X_plot, y_plot, color='green', linewidth=2, label='SVR prediction')
    
    # Highlight support vectors
    if len(model.support_) > 0:
        support_X = X_train_reg_scaled[model.support_]
        support_y = y_train_reg[model.support_]
        plt.scatter(support_X, support_y, s=100, facecolors='none', 
                   edgecolors='black', linewidth=2, label='Support vectors')
    
    plt.xlabel('Feature (standardized)')
    plt.ylabel('Target')
    plt.title(f'{name}\nR� = {test_r2:.3f}, RMSE = {test_rmse:.1f}')
    plt.legend()
    plt.grid(True, alpha=0.3)

# Plot residuals analysis
plt.subplot(2, 3, 4)
best_model = svr_models['RBF SVR']  # Use RBF as best model
y_pred_best = best_model.predict(X_test_reg_scaled)
residuals = y_test_reg - y_pred_best

plt.scatter(y_pred_best, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot (RBF SVR)')
plt.grid(True, alpha=0.3)

# Plot actual vs predicted
plt.subplot(2, 3, 5)
plt.scatter(y_test_reg, y_pred_best, alpha=0.6)
plt.plot([y_test_reg.min(), y_test_reg.max()], 
         [y_test_reg.min(), y_test_reg.max()], 'r--', linewidth=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted (RBF SVR)')
plt.grid(True, alpha=0.3)

# Plot epsilon-tube visualization
plt.subplot(2, 3, 6)
epsilon = best_model.epsilon

# Sort data for smooth plotting
sort_idx = np.argsort(X_test_reg_scaled[:, 0])
X_sorted = X_test_reg_scaled[sort_idx]
y_pred_sorted = y_pred_best[sort_idx]

plt.scatter(X_test_reg_scaled, y_test_reg, alpha=0.6, color='blue', 
           label='Test data')
plt.plot(X_sorted, y_pred_sorted, color='green', linewidth=2, 
         label='SVR prediction')
plt.fill_between(X_sorted[:, 0], y_pred_sorted - epsilon, y_pred_sorted + epsilon,
                alpha=0.3, color='yellow', label=f'�-tube (�={epsilon})')

plt.xlabel('Feature (standardized)')
plt.ylabel('Target')
plt.title('SVR with �-insensitive Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Hyperparameter tuning for SVR
print(f"\nSVR Hyperparameter Analysis:")

# Test different epsilon values
epsilon_values = [0.01, 0.1, 0.5, 1.0, 2.0]
C_values = [1, 10, 100, 1000]

best_score = -np.inf
best_params = {}

for epsilon in epsilon_values:
    for C in C_values:
        svr_temp = SVR(kernel='rbf', C=C, epsilon=epsilon, gamma='scale')
        svr_temp.fit(X_train_reg_scaled, y_train_reg)
        score = svr_temp.score(X_test_reg_scaled, y_test_reg)
        
        if score > best_score:
            best_score = score
            best_params = {'C': C, 'epsilon': epsilon}
        
        print(f"C={C:4}, �={epsilon:4.2f}: R� = {score:.3f}, "
              f"Support vectors: {len(svr_temp.support_):3d}")

print(f"\nBest parameters: {best_params}")
print(f"Best R� score: {best_score:.3f}")
```

## 📚 References

- **Original Papers:**
  - [Support-Vector Networks](https://link.springer.com/article/10.1007/BF00994018) by Cortes & Vapnik (1995)
  - [The Nature of Statistical Learning Theory](https://www.springer.com/gp/book/9780387987804) by Vladimir Vapnik (1995)
  - [SMO Algorithm](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/smo-book.pdf) by John Platt (1998)

- **Books:**
  - [The Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/) by Hastie, Tibshirani, and Friedman - Chapter 12
  - [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) by Christopher Bishop - Chapter 7
  - [Learning with Kernels](https://mitpress.mit.edu/books/learning-kernels) by Sch�lkopf and Smola

- **Documentation:**
  - [Scikit-learn SVM Guide](https://scikit-learn.org/stable/modules/svm.html)
  - [Scikit-learn SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
  - [Scikit-learn SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)

- **Tutorials and Guides:**
  - [SVM Tutorial - Andrew Ng](http://cs229.stanford.edu/notes/cs229-notes3.pdf)
  - [Understanding SVM](https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47)
  - [Kernel Methods Tutorial](http://people.cs.uchicago.edu/~niyogi/papersps/NiyogiKernelMethods.pdf)

- **Advanced Topics:**
  - [One-Class SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html) for anomaly detection
  - [Nu-SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html) alternative parameterization
  - [Linear SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) for large datasets

- **Research Papers:**
  - Sch�lkopf, B., & Smola, A. J. (2002). Learning with kernels: Support vector machines
  - Chang, C. C., & Lin, C. J. (2011). LIBSVM: A library for support vector machines
  - Fan, R. E., Chang, K. W., Hsieh, C. J., Wang, X. R., & Lin, C. J. (2008). LIBLINEAR: A library for large linear classification

- **Online Courses:**
  - [Machine Learning Course - Stanford CS229](http://cs229.stanford.edu/)
  - [SVM in Machine Learning - Coursera](https://www.coursera.org/learn/machine-learning)
  - [Statistical Learning - edX](https://www.edx.org/course/statistical-learning)

- **Implementations:**
  - [scikit-learn](https://scikit-learn.org/stable/) (Python)
  - [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) (C++, multiple language bindings)
  - [e1071](https://cran.r-project.org/web/packages/e1071/index.html) (R package)