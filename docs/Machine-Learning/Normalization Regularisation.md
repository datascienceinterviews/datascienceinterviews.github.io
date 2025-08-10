---
title: Normalization and Regularisation
description: Comprehensive guide to feature normalization and model regularization techniques with mathematical foundations, implementations, and interview questions.
comments: true
---

# =Ø Normalization and Regularisation

Normalization and regularisation are fundamental techniques in machine learning: normalization ensures features are on similar scales for optimal algorithm performance, while regularisation prevents overfitting by constraining model complexity.

**Resources:** [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html) | [Regularization in Deep Learning](https://www.deeplearningbook.org/contents/regularization.html) | [Elements of Statistical Learning - Chapter 3](https://web.stanford.edu/~hastie/ElemStatLearn/)

##  Summary

**Normalization** (Feature Scaling) transforms features to similar scales, ensuring no single feature dominates due to its scale. **Regularisation** adds penalty terms to the loss function to prevent overfitting by constraining model complexity.

### Normalization
Feature scaling is crucial when features have different units, ranges, or variances. Without normalization, algorithms like gradient descent, SVM, and k-NN can be severely affected by scale differences.

**Common normalization techniques:**
- **Standardization (Z-score)**: Mean = 0, Standard deviation = 1
- **Min-Max scaling**: Scale to [0,1] range
- **Robust scaling**: Uses median and IQR, resistant to outliers
- **Unit vector scaling**: Scale to unit norm
- **Quantile transformation**: Map to uniform or normal distribution

### Regularisation
Regularisation prevents overfitting by adding penalty terms that discourage complex models, leading to better generalization on unseen data.

**Common regularisation techniques:**
- **L1 Regularization (Lasso)**: Promotes sparsity, feature selection
- **L2 Regularization (Ridge)**: Shrinks coefficients, handles multicollinearity
- **Elastic Net**: Combines L1 and L2 penalties
- **Dropout**: Randomly deactivates neurons (neural networks)
- **Early stopping**: Stop training before overfitting occurs

**Applications:**
- Feature preprocessing for all ML algorithms
- Linear models (Ridge, Lasso, Elastic Net)
- Neural networks (dropout, batch normalization)
- Tree-based models (pruning)
- Computer vision and NLP pipelines

## >à Intuition

### Normalization Intuition

Imagine you're comparing houses using price (in hundreds of thousands) and square footage (in thousands). Without normalization, price variations (20-800) might overshadow square footage variations (1-5), causing algorithms to ignore the latter feature entirely.

**Example**: In k-NN, Euclidean distance between houses:
- Without normalization: Distance dominated by price differences
- With normalization: Both features contribute meaningfully to distance

### Regularisation Intuition

Think of regularisation like speed limits on roads. Without limits (regularisation), drivers (models) might go too fast (overfit) and crash. Regularisation enforces "speed limits" on model complexity, ensuring safer (more generalizable) performance.

**Analogy**: 
- **No regularisation**: Memorizing exam answers ’ fails on new questions
- **With regularisation**: Understanding concepts ’ succeeds on new questions

### Mathematical Foundation

#### 1. Normalization Techniques

**Standardization (Z-score normalization)**:
$$z = \frac{x - \mu}{\sigma}$$

Where $\mu$ is mean and $\sigma$ is standard deviation.

**Min-Max scaling**:
$$x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

**Robust scaling**:
$$x_{robust} = \frac{x - \text{median}(x)}{\text{IQR}(x)}$$

Where IQR is the interquartile range.

**Unit vector scaling**:
$$x_{unit} = \frac{x}{||x||_2}$$

#### 2. Regularisation Mathematics

**L1 Regularization (Lasso)**:
$$\text{Loss} = \text{MSE} + \lambda \sum_{i=1}^{n} |w_i|$$

**L2 Regularization (Ridge)**:
$$\text{Loss} = \text{MSE} + \lambda \sum_{i=1}^{n} w_i^2$$

**Elastic Net**:
$$\text{Loss} = \text{MSE} + \lambda_1 \sum_{i=1}^{n} |w_i| + \lambda_2 \sum_{i=1}^{n} w_i^2$$

Where $\lambda$ controls regularisation strength.

#### 3. Effect on Gradients

**L1 gradient** (creates sparsity):
$$\frac{\partial}{\partial w_i} \lambda |w_i| = \lambda \cdot \text{sign}(w_i)$$

**L2 gradient** (shrinks coefficients):
$$\frac{\partial}{\partial w_i} \lambda w_i^2 = 2\lambda w_i$$

## =" Implementation using Libraries

### Normalization with Scikit-learn

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, 
    Normalizer, QuantileTransformer, PowerTransformer
)
from sklearn.datasets import make_classification, load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Generate sample dataset with different scales
np.random.seed(42)
n_samples = 1000

# Create features with vastly different scales
data = {
    'income': np.random.normal(50000, 15000, n_samples),      # Mean ~50k
    'age': np.random.normal(35, 10, n_samples),               # Mean ~35
    'debt_ratio': np.random.uniform(0, 1, n_samples),         # Range [0,1]
    'credit_score': np.random.normal(700, 100, n_samples),    # Mean ~700
    'num_accounts': np.random.poisson(5, n_samples)           # Count data
}

df = pd.DataFrame(data)

print("Feature Scaling Demonstration")
print("Original data statistics:")
print(df.describe())
print(f"\nFeature ranges:")
for col in df.columns:
    print(f"{col:15}: [{df[col].min():.2f}, {df[col].max():.2f}]")

# Visualize original distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

original_data = df.values

scalers = {
    'Original': None,
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler(),
    'Normalizer': Normalizer(),
    'QuantileTransformer': QuantileTransformer(output_distribution='normal')
}

scaled_data = {}

for i, (name, scaler) in enumerate(scalers.items()):
    if scaler is None:
        data_transformed = original_data
        title_stats = "Original Data"
    else:
        data_transformed = scaler.fit_transform(original_data)
        title_stats = f"Mean: {data_transformed.mean():.2f}, Std: {data_transformed.std():.2f}"
    
    scaled_data[name] = data_transformed
    
    # Plot first feature (income) for each transformation
    axes[i].hist(data_transformed[:, 0], bins=30, alpha=0.7, 
                density=True, edgecolor='black')
    axes[i].set_title(f'{name}\n{title_stats}')
    axes[i].set_xlabel('Income (transformed)')
    axes[i].set_ylabel('Density')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Compare performance impact on different algorithms
X = df.values
y = (df['income'] > df['income'].median()).astype(int)  # Binary target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Test different scalers with k-NN (scale-sensitive algorithm)
print(f"\nImpact of normalization on k-NN classifier:")

results = {}
for name, scaler in scalers.items():
    if scaler is None:
        X_train_scaled = X_train
        X_test_scaled = X_test
    else:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
    # Train k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    results[name] = accuracy
    print(f"{name:20}: {accuracy:.3f}")

# Visualize results
plt.figure(figsize=(10, 6))
methods = list(results.keys())
accuracies = list(results.values())

bars = plt.bar(methods, accuracies, color='skyblue', edgecolor='navy', alpha=0.7)
plt.title('k-NN Performance with Different Scaling Methods')
plt.ylabel('Accuracy')
plt.xlabel('Scaling Method')
plt.xticks(rotation=45)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
             f'{acc:.3f}', ha='center', va='bottom')

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Advanced Normalization Techniques

```python
# Outlier-robust scaling comparison
np.random.seed(42)

# Create data with outliers
normal_data = np.random.normal(0, 1, 1000)
outliers = np.random.normal(10, 1, 50)  # Extreme outliers
data_with_outliers = np.concatenate([normal_data, outliers])

# Compare different scalers on data with outliers
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Original data
axes[0,0].hist(data_with_outliers, bins=50, alpha=0.7, edgecolor='black')
axes[0,0].set_title('Original Data (with outliers)')
axes[0,0].set_ylabel('Frequency')

# StandardScaler (sensitive to outliers)
standard_scaler = StandardScaler()
data_standard = standard_scaler.fit_transform(data_with_outliers.reshape(-1, 1)).flatten()
axes[0,1].hist(data_standard, bins=50, alpha=0.7, edgecolor='black')
axes[0,1].set_title(f'StandardScaler\nMean: {data_standard.mean():.2f}, Std: {data_standard.std():.2f}')

# RobustScaler (resistant to outliers)
robust_scaler = RobustScaler()
data_robust = robust_scaler.fit_transform(data_with_outliers.reshape(-1, 1)).flatten()
axes[1,0].hist(data_robust, bins=50, alpha=0.7, edgecolor='black')
axes[1,0].set_title(f'RobustScaler\nMedian: {np.median(data_robust):.2f}, IQR: {np.percentile(data_robust, 75) - np.percentile(data_robust, 25):.2f}')
axes[1,0].set_ylabel('Frequency')
axes[1,0].set_xlabel('Scaled Values')

# QuantileTransformer (maps to uniform distribution)
quantile_transformer = QuantileTransformer(output_distribution='uniform')
data_quantile = quantile_transformer.fit_transform(data_with_outliers.reshape(-1, 1)).flatten()
axes[1,1].hist(data_quantile, bins=50, alpha=0.7, edgecolor='black')
axes[1,1].set_title(f'QuantileTransformer (Uniform)\nRange: [{data_quantile.min():.2f}, {data_quantile.max():.2f}]')
axes[1,1].set_xlabel('Scaled Values')

plt.tight_layout()
plt.show()

# Demonstrate PowerTransformer for non-normal data
from scipy.stats import skew

# Create skewed data
skewed_data = np.random.exponential(2, 1000)
print(f"Original skewness: {skew(skewed_data):.3f}")

# Apply different transformations
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original
axes[0].hist(skewed_data, bins=30, alpha=0.7, edgecolor='black', density=True)
axes[0].set_title(f'Original Data\nSkewness: {skew(skewed_data):.3f}')
axes[0].set_ylabel('Density')

# Box-Cox transformation
power_transformer_box = PowerTransformer(method='box-cox')
data_box_cox = power_transformer_box.fit_transform(skewed_data.reshape(-1, 1)).flatten()
axes[1].hist(data_box_cox, bins=30, alpha=0.7, edgecolor='black', density=True)
axes[1].set_title(f'Box-Cox Transform\nSkewness: {skew(data_box_cox):.3f}')

# Yeo-Johnson transformation (can handle negative values)
power_transformer_yj = PowerTransformer(method='yeo-johnson')
data_yj = power_transformer_yj.fit_transform(skewed_data.reshape(-1, 1)).flatten()
axes[2].hist(data_yj, bins=30, alpha=0.7, edgecolor='black', density=True)
axes[2].set_title(f'Yeo-Johnson Transform\nSkewness: {skew(data_yj):.3f}')

for ax in axes:
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Values')

plt.tight_layout()
plt.show()
```

### Regularisation Implementation

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import validation_curve
import warnings
warnings.filterwarnings('ignore')

# Generate regression dataset with potential for overfitting
np.random.seed(42)
n_samples = 100
n_features = 50

X_reg = np.random.randn(n_samples, n_features)
# Create target with only few features actually relevant
true_coef = np.zeros(n_features)
true_coef[:5] = [2, -3, 1, 4, -2]  # Only first 5 features matter
y_reg = X_reg @ true_coef + 0.1 * np.random.randn(n_samples)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

print("Regularization Comparison")
print(f"Dataset: {n_samples} samples, {n_features} features")
print(f"True non-zero coefficients: {np.sum(true_coef != 0)}")

# Compare different regularization techniques
models = {
    'Linear Regression': LinearRegression(),
    'Ridge (L2)': Ridge(alpha=1.0),
    'Lasso (L1)': Lasso(alpha=0.1),
    'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5)
}

results = {}
coefficients = {}

for name, model in models.items():
    # Train model
    model.fit(X_train_reg, y_train_reg)
    
    # Predictions
    y_pred_train = model.predict(X_train_reg)
    y_pred_test = model.predict(X_test_reg)
    
    # Evaluate
    train_mse = mean_squared_error(y_train_reg, y_pred_train)
    test_mse = mean_squared_error(y_test_reg, y_pred_test)
    train_r2 = r2_score(y_train_reg, y_pred_train)
    test_r2 = r2_score(y_test_reg, y_pred_test)
    
    # Store results
    results[name] = {
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'overfitting': train_r2 - test_r2  # Measure of overfitting
    }
    
    # Store coefficients
    if hasattr(model, 'coef_'):
        coefficients[name] = model.coef_
        non_zero = np.sum(np.abs(model.coef_) > 1e-5)
        results[name]['non_zero_coef'] = non_zero
    
    print(f"\n{name}:")
    print(f"  Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}")
    print(f"  Train MSE: {train_mse:.3f}, Test MSE: {test_mse:.3f}")
    print(f"  Overfitting gap: {train_r2 - test_r2:.3f}")
    if hasattr(model, 'coef_'):
        print(f"  Non-zero coefficients: {non_zero}/{n_features}")

# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Performance comparison
model_names = list(results.keys())
train_r2s = [results[name]['train_r2'] for name in model_names]
test_r2s = [results[name]['test_r2'] for name in model_names]

x = np.arange(len(model_names))
width = 0.35

axes[0,0].bar(x - width/2, train_r2s, width, label='Train R²', alpha=0.7)
axes[0,0].bar(x + width/2, test_r2s, width, label='Test R²', alpha=0.7)
axes[0,0].set_xlabel('Model')
axes[0,0].set_ylabel('R² Score')
axes[0,0].set_title('Train vs Test Performance')
axes[0,0].set_xticks(x)
axes[0,0].set_xticklabels(model_names, rotation=45)
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Overfitting comparison
overfitting_gaps = [results[name]['overfitting'] for name in model_names]
axes[0,1].bar(model_names, overfitting_gaps, alpha=0.7)
axes[0,1].set_ylabel('Overfitting Gap (Train R² - Test R²)')
axes[0,1].set_title('Overfitting Comparison')
axes[0,1].tick_params(axis='x', rotation=45)
axes[0,1].grid(True, alpha=0.3)

# Coefficient plots
# True coefficients vs estimated
axes[1,0].plot(true_coef, 'ko-', label='True coefficients', linewidth=2, markersize=6)
for name in ['Ridge (L2)', 'Lasso (L1)', 'Elastic Net']:
    if name in coefficients:
        axes[1,0].plot(coefficients[name], 'o-', label=name, alpha=0.7)
axes[1,0].set_xlabel('Feature Index')
axes[1,0].set_ylabel('Coefficient Value')
axes[1,0].set_title('Coefficient Comparison')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Sparsity comparison (number of non-zero coefficients)
sparsity_names = [name for name in model_names if name != 'Linear Regression']
non_zero_coefs = [results[name]['non_zero_coef'] for name in sparsity_names]
axes[1,1].bar(sparsity_names, non_zero_coefs, alpha=0.7)
axes[1,1].axhline(y=5, color='red', linestyle='--', label='True non-zero features')
axes[1,1].set_ylabel('Number of Non-zero Coefficients')
axes[1,1].set_title('Feature Selection (Sparsity)')
axes[1,1].tick_params(axis='x', rotation=45)
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Regularisation Path Analysis

```python
# Analyze how regularization strength affects coefficients
alphas = np.logspace(-4, 2, 50)

# Ridge regression path
ridge_coefs = []
ridge_scores = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_reg, y_train_reg)
    ridge_coefs.append(ridge.coef_)
    score = ridge.score(X_test_reg, y_test_reg)
    ridge_scores.append(score)

ridge_coefs = np.array(ridge_coefs)

# Lasso regression path
lasso_coefs = []
lasso_scores = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=2000)
    lasso.fit(X_train_reg, y_train_reg)
    lasso_coefs.append(lasso.coef_)
    score = lasso.score(X_test_reg, y_test_reg)
    lasso_scores.append(score)

lasso_coefs = np.array(lasso_coefs)

# Plot regularization paths
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Ridge coefficients path
for i in range(min(10, n_features)):  # Plot first 10 features
    axes[0,0].plot(alphas, ridge_coefs[:, i], label=f'Feature {i}' if i < 5 else "")
axes[0,0].set_xscale('log')
axes[0,0].set_xlabel('Regularization Strength (±)')
axes[0,0].set_ylabel('Coefficient Value')
axes[0,0].set_title('Ridge Regression Path')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Lasso coefficients path
for i in range(min(10, n_features)):  # Plot first 10 features
    axes[0,1].plot(alphas, lasso_coefs[:, i], label=f'Feature {i}' if i < 5 else "")
axes[0,1].set_xscale('log')
axes[0,1].set_xlabel('Regularization Strength (±)')
axes[0,1].set_ylabel('Coefficient Value')
axes[0,1].set_title('Lasso Regression Path')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Performance vs regularization strength
axes[1,0].plot(alphas, ridge_scores, 'b-', label='Ridge', linewidth=2)
axes[1,0].plot(alphas, lasso_scores, 'r-', label='Lasso', linewidth=2)
axes[1,0].set_xscale('log')
axes[1,0].set_xlabel('Regularization Strength (±)')
axes[1,0].set_ylabel('R² Score')
axes[1,0].set_title('Performance vs Regularization')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Number of non-zero coefficients (sparsity)
ridge_sparsity = [np.sum(np.abs(coef) > 1e-5) for coef in ridge_coefs]
lasso_sparsity = [np.sum(np.abs(coef) > 1e-5) for coef in lasso_coefs]

axes[1,1].plot(alphas, ridge_sparsity, 'b-', label='Ridge', linewidth=2)
axes[1,1].plot(alphas, lasso_sparsity, 'r-', label='Lasso', linewidth=2)
axes[1,1].axhline(y=5, color='green', linestyle='--', label='True non-zero features')
axes[1,1].set_xscale('log')
axes[1,1].set_xlabel('Regularization Strength (±)')
axes[1,1].set_ylabel('Number of Non-zero Coefficients')
axes[1,1].set_title('Sparsity vs Regularization')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nKey Insights:")
print(f"- Ridge: Shrinks coefficients but rarely makes them exactly zero")
print(f"- Lasso: Creates sparse solutions by setting coefficients to exactly zero")
print(f"- Optimal ± for Ridge: {alphas[np.argmax(ridge_scores)]:.4f}")
print(f"- Optimal ± for Lasso: {alphas[np.argmax(lasso_scores)]:.4f}")
```

## ™ From Scratch Implementation

### Custom Scalers Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

class StandardScalerFromScratch:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.fitted = False
    
    def fit(self, X):
        """Compute mean and standard deviation for later scaling"""
        X = np.array(X)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0, ddof=1)  # Sample standard deviation
        
        # Handle zero variance features
        self.scale_[self.scale_ == 0] = 1.0
        
        self.fitted = True
        return self
    
    def transform(self, X):
        """Scale features using computed statistics"""
        if not self.fitted:
            raise ValueError("Scaler has not been fitted yet.")
        
        X = np.array(X)
        return (X - self.mean_) / self.scale_
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_scaled):
        """Convert scaled features back to original scale"""
        if not self.fitted:
            raise ValueError("Scaler has not been fitted yet.")
        
        X_scaled = np.array(X_scaled)
        return X_scaled * self.scale_ + self.mean_

class MinMaxScalerFromScratch:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.scale_ = None
        self.data_min_ = None
        self.data_max_ = None
        self.fitted = False
    
    def fit(self, X):
        """Compute min and max for later scaling"""
        X = np.array(X)
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        
        # Compute scaling parameters
        data_range = self.data_max_ - self.data_min_
        data_range[data_range == 0] = 1.0  # Handle constant features
        
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / data_range
        self.min_ = self.feature_range[0] - self.data_min_ * self.scale_
        
        self.fitted = True
        return self
    
    def transform(self, X):
        """Scale features to specified range"""
        if not self.fitted:
            raise ValueError("Scaler has not been fitted yet.")
        
        X = np.array(X)
        return X * self.scale_ + self.min_
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_scaled):
        """Convert scaled features back to original scale"""
        if not self.fitted:
            raise ValueError("Scaler has not been fitted yet.")
        
        X_scaled = np.array(X_scaled)
        return (X_scaled - self.min_) / self.scale_

class RobustScalerFromScratch:
    def __init__(self):
        self.center_ = None
        self.scale_ = None
        self.fitted = False
    
    def fit(self, X):
        """Compute median and IQR for later scaling"""
        X = np.array(X)
        self.center_ = np.median(X, axis=0)
        
        # Compute IQR (75th percentile - 25th percentile)
        q75 = np.percentile(X, 75, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        self.scale_ = q75 - q25
        
        # Handle zero IQR
        self.scale_[self.scale_ == 0] = 1.0
        
        self.fitted = True
        return self
    
    def transform(self, X):
        """Scale features using median and IQR"""
        if not self.fitted:
            raise ValueError("Scaler has not been fitted yet.")
        
        X = np.array(X)
        return (X - self.center_) / self.scale_
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        return self.fit(X).transform(X)

# Demonstration with synthetic data
np.random.seed(42)

# Create data with outliers
normal_data = np.random.normal(10, 2, (100, 3))
outliers = np.array([[50, 5, 15], [60, 8, 20], [-20, 1, 5]])  # Add outliers
data = np.vstack([normal_data, outliers])

print("Custom Scalers Demonstration")
print(f"Original data shape: {data.shape}")
print(f"Original data statistics:")
print(f"Mean: {np.mean(data, axis=0)}")
print(f"Std: {np.std(data, axis=0)}")
print(f"Min: {np.min(data, axis=0)}")
print(f"Max: {np.max(data, axis=0)}")

# Test custom scalers
scalers_custom = {
    'Custom StandardScaler': StandardScalerFromScratch(),
    'Custom MinMaxScaler': MinMaxScalerFromScratch(),
    'Custom RobustScaler': RobustScalerFromScratch()
}

# Compare with sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

scalers_sklearn = {
    'Sklearn StandardScaler': StandardScaler(),
    'Sklearn MinMaxScaler': MinMaxScaler(),
    'Sklearn RobustScaler': RobustScaler()
}

# Test and compare
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i, (name, scaler) in enumerate(scalers_custom.items()):
    # Apply custom scaler
    data_scaled_custom = scaler.fit_transform(data)
    
    # Apply sklearn scaler for comparison
    sklearn_name = name.replace('Custom', 'Sklearn')
    sklearn_scaler = scalers_sklearn[sklearn_name]
    data_scaled_sklearn = sklearn_scaler.fit_transform(data)
    
    # Plot comparison for first feature
    axes[0, i].hist(data_scaled_custom[:, 0], bins=20, alpha=0.7, 
                    label='Custom', color='blue', density=True)
    axes[0, i].hist(data_scaled_sklearn[:, 0], bins=20, alpha=0.7, 
                    label='Sklearn', color='red', density=True)
    axes[0, i].set_title(f'{name}\n(Feature 1)')
    axes[0, i].legend()
    axes[0, i].grid(True, alpha=0.3)
    
    # Check numerical differences
    max_diff = np.max(np.abs(data_scaled_custom - data_scaled_sklearn))
    axes[1, i].scatter(data_scaled_custom.flatten(), 
                      data_scaled_sklearn.flatten(), alpha=0.6)
    axes[1, i].plot([data_scaled_custom.min(), data_scaled_custom.max()],
                    [data_scaled_custom.min(), data_scaled_custom.max()], 'r--')
    axes[1, i].set_xlabel('Custom Scaler')
    axes[1, i].set_ylabel('Sklearn Scaler')
    axes[1, i].set_title(f'Comparison\nMax diff: {max_diff:.2e}')
    axes[1, i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Test inverse transform
print(f"\nInverse Transform Test:")
scaler_test = StandardScalerFromScratch()
data_scaled = scaler_test.fit_transform(data)
data_reconstructed = scaler_test.inverse_transform(data_scaled)

reconstruction_error = np.max(np.abs(data - data_reconstructed))
print(f"Max reconstruction error: {reconstruction_error:.2e}")
print(" Inverse transform working correctly" if reconstruction_error < 1e-10 else " Inverse transform failed")
```

### Custom Regularised Regression

```python
class RidgeRegressionFromScratch:
    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        """Fit Ridge regression using closed-form solution"""
        X = np.array(X)
        y = np.array(y)
        
        if self.fit_intercept:
            # Add intercept term
            X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        else:
            X_with_intercept = X
        
        n_features = X_with_intercept.shape[1]
        
        # Ridge regression closed-form solution: (X'X + ±I)^(-1)X'y
        # Don't regularize the intercept term
        I = np.eye(n_features)
        if self.fit_intercept:
            I[0, 0] = 0  # Don't regularize intercept
        
        XTX_plus_alphaI = X_with_intercept.T @ X_with_intercept + self.alpha * I
        XTy = X_with_intercept.T @ y
        
        # Solve the system
        params = np.linalg.solve(XTX_plus_alphaI, XTy)
        
        if self.fit_intercept:
            self.intercept_ = params[0]
            self.coef_ = params[1:]
        else:
            self.intercept_ = 0
            self.coef_ = params
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        X = np.array(X)
        return X @ self.coef_ + self.intercept_
    
    def score(self, X, y):
        """Calculate R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

class LassoRegressionFromScratch:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None
    
    def _soft_threshold(self, x, alpha):
        """Soft thresholding operator for L1 regularization"""
        return np.sign(x) * np.maximum(np.abs(x) - alpha, 0)
    
    def fit(self, X, y):
        """Fit Lasso regression using coordinate descent"""
        X = np.array(X)
        y = np.array(y)
        
        # Center the data
        X_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)
        X_centered = X - X_mean
        y_centered = y - y_mean
        
        n_samples, n_features = X_centered.shape
        
        # Initialize coefficients
        self.coef_ = np.zeros(n_features)
        
        # Precompute X'X diagonal for coordinate descent
        XTX_diag = np.sum(X_centered ** 2, axis=0)
        
        # Coordinate descent
        for iteration in range(self.max_iter):
            coef_old = self.coef_.copy()
            
            for j in range(n_features):
                if XTX_diag[j] == 0:
                    continue
                
                # Compute residual without j-th feature
                residual = y_centered - X_centered @ self.coef_ + self.coef_[j] * X_centered[:, j]
                
                # Update coefficient using soft thresholding
                rho = X_centered[:, j] @ residual
                self.coef_[j] = self._soft_threshold(rho, self.alpha * n_samples) / XTX_diag[j]
            
            # Check convergence
            if np.max(np.abs(self.coef_ - coef_old)) < self.tol:
                break
        
        # Calculate intercept
        self.intercept_ = y_mean - X_mean @ self.coef_
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        X = np.array(X)
        return X @ self.coef_ + self.intercept_
    
    def score(self, X, y):
        """Calculate R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

# Test custom regularized regression
np.random.seed(42)

# Generate test data
n_samples, n_features = 50, 20
X_test = np.random.randn(n_samples, n_features)
true_coef = np.random.randn(n_features)
true_coef[10:] = 0  # Make last 10 coefficients zero
y_test = X_test @ true_coef + 0.1 * np.random.randn(n_samples)

# Split data
X_train_test, X_val_test, y_train_test, y_val_test = train_test_split(
    X_test, y_test, test_size=0.3, random_state=42
)

print("Custom Regularized Regression Test")

# Test Ridge regression
ridge_custom = RidgeRegressionFromScratch(alpha=1.0)
ridge_custom.fit(X_train_test, y_train_test)

ridge_sklearn = Ridge(alpha=1.0)
ridge_sklearn.fit(X_train_test, y_train_test)

print(f"\nRidge Regression Comparison:")
print(f"Custom Ridge R²: {ridge_custom.score(X_val_test, y_val_test):.4f}")
print(f"Sklearn Ridge R²: {ridge_sklearn.score(X_val_test, y_val_test):.4f}")

coef_diff_ridge = np.max(np.abs(ridge_custom.coef_ - ridge_sklearn.coef_))
print(f"Max coefficient difference: {coef_diff_ridge:.2e}")

# Test Lasso regression
lasso_custom = LassoRegressionFromScratch(alpha=0.1, max_iter=2000)
lasso_custom.fit(X_train_test, y_train_test)

lasso_sklearn = Lasso(alpha=0.1, max_iter=2000)
lasso_sklearn.fit(X_train_test, y_train_test)

print(f"\nLasso Regression Comparison:")
print(f"Custom Lasso R²: {lasso_custom.score(X_val_test, y_val_test):.4f}")
print(f"Sklearn Lasso R²: {lasso_sklearn.score(X_val_test, y_val_test):.4f}")

# Compare sparsity
custom_nonzero = np.sum(np.abs(lasso_custom.coef_) > 1e-5)
sklearn_nonzero = np.sum(np.abs(lasso_sklearn.coef_) > 1e-5)
print(f"Custom Lasso non-zero coefficients: {custom_nonzero}")
print(f"Sklearn Lasso non-zero coefficients: {sklearn_nonzero}")

# Visualize coefficient comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Ridge coefficients
axes[0].scatter(ridge_custom.coef_, ridge_sklearn.coef_, alpha=0.7)
axes[0].plot([ridge_custom.coef_.min(), ridge_custom.coef_.max()],
             [ridge_custom.coef_.min(), ridge_custom.coef_.max()], 'r--')
axes[0].set_xlabel('Custom Ridge Coefficients')
axes[0].set_ylabel('Sklearn Ridge Coefficients')
axes[0].set_title('Ridge Regression Coefficients Comparison')
axes[0].grid(True, alpha=0.3)

# Lasso coefficients
axes[1].scatter(lasso_custom.coef_, lasso_sklearn.coef_, alpha=0.7)
axes[1].plot([lasso_custom.coef_.min(), lasso_custom.coef_.max()],
             [lasso_custom.coef_.min(), lasso_custom.coef_.max()], 'r--')
axes[1].set_xlabel('Custom Lasso Coefficients')
axes[1].set_ylabel('Sklearn Lasso Coefficients')
axes[1].set_title('Lasso Regression Coefficients Comparison')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

##   Assumptions and Limitations

### Normalization Assumptions and Limitations

#### Key Assumptions

1. **Feature independence**: Different scaling methods assume features are independent
2. **Distribution stability**: Scaling parameters computed on training data apply to test data
3. **Outlier handling**: StandardScaler assumes roughly normal distribution
4. **Missing values**: Most scalers require handling missing values beforehand

#### Limitations

1. **StandardScaler limitations**:
   - **Sensitive to outliers**: Outliers heavily influence mean and standard deviation
   - **Assumes normal distribution**: Works best with normally distributed data
   - **Solution**: Use RobustScaler for data with outliers

2. **MinMaxScaler limitations**:
   - **Very sensitive to outliers**: Single outlier can compress all other values
   - **Fixed range assumption**: Assumes test data falls within training range
   - **Solution**: Use robust scaling or outlier detection

3. **RobustScaler limitations**:
   - **Less efficient**: May not utilize full feature range
   - **Assumes symmetric distribution around median**
   - **Assessment**: Check if IQR-based scaling is appropriate

4. **Data leakage risk**: Fitting scaler on entire dataset before train/test split
   - **Critical error**: Using test data to compute scaling parameters
   - **Solution**: Always fit scaler only on training data

### Regularisation Assumptions and Limitations

#### Key Assumptions

1. **Smooth coefficient penalty**: Assumes large coefficients are undesirable
2. **Feature relevance**: L1 assumes many features are irrelevant (sparsity assumption)
3. **Linear relationship**: Regularization assumes linear model structure
4. **Homoscedastic errors**: Assumes constant error variance

#### Limitations

1. **L1 (Lasso) limitations**:
   - **Arbitrary feature selection**: With correlated features, randomly picks one
   - **Bias introduction**: Can be overly aggressive in shrinking coefficients
   - **Solution**: Use Elastic Net to combine L1 and L2

2. **L2 (Ridge) limitations**:
   - **No feature selection**: Shrinks but doesn't eliminate features
   - **Multicollinearity**: Distributes weight among correlated features
   - **Alternative**: Use Lasso for feature selection

3. **Hyperparameter sensitivity**: Performance heavily depends on regularization strength
   - **Challenge**: Requires careful tuning using cross-validation
   - **Solution**: Use automated hyperparameter optimization

4. **Computational complexity**: Some regularization methods scale poorly
   - **Impact**: Lasso coordinate descent can be slow on very high-dimensional data
   - **Solution**: Use specialized libraries or approximate methods

### Comparison of Techniques

| Aspect | StandardScaler | MinMaxScaler | RobustScaler | L1 (Lasso) | L2 (Ridge) |
|--------|----------------|--------------|--------------|------------|------------|
| **Outlier Sensitivity** | High | Very High | Low | Medium | Medium |
| **Preserves Distribution** | Yes | No | Partially | N/A | N/A |
| **Computational Cost** | Low | Low | Medium | High | Low |
| **Feature Selection** | N/A | N/A | N/A | Yes | No |
| **Interpretability** | N/A | N/A | N/A | High | Medium |

**When to use each technique:**

**Normalization:**
- **StandardScaler**: Normal distributions, no outliers, most ML algorithms
- **MinMaxScaler**: Bounded features needed, neural networks
- **RobustScaler**: Data with outliers, non-normal distributions
- **QuantileTransformer**: Heavy outliers, need uniform distribution

**Regularisation:**
- **Ridge**: Multicollinearity, want to keep all features
- **Lasso**: Feature selection needed, sparse solution desired
- **Elastic Net**: Correlated features, balanced selection and shrinkage

## =¡ Interview Questions

??? question "Why is feature normalization important in machine learning, and when might you skip it?"

    **Answer:** Feature normalization is crucial for algorithms sensitive to feature scales:
    
    **Why normalization matters**:
    1. **Scale sensitivity**: Algorithms like SVM, k-NN, neural networks use distance metrics
    2. **Convergence speed**: Gradient descent converges faster with normalized features
    3. **Numerical stability**: Prevents overflow/underflow in computations
    4. **Fair feature contribution**: Ensures all features contribute meaningfully
    
    **Example impact**:
    ```python
    # Without normalization
    features = [[50000, 25], [60000, 30]]  # [income, age]
    # Distance dominated by income differences
    
    # With normalization  
    features_norm = [[0.1, 0.2], [0.6, 0.8]]
    # Both features contribute to distance
    ```
    
    **When to skip normalization**:
    - **Tree-based models**: Decision trees, Random Forest, XGBoost (scale-invariant)
    - **Naive Bayes**: Works with original feature distributions
    - **Linear regression** with interpretability needs: Keep original coefficient meanings
    - **Count data**: When raw counts are meaningful (e.g., word frequencies)
    
    **Algorithm sensitivity**:
    - **Requires normalization**: SVM, k-NN, neural networks, PCA, clustering
    - **Doesn't require**: Tree-based models, Naive Bayes

??? question "Compare StandardScaler, MinMaxScaler, and RobustScaler. When would you use each?"

    **Answer:** Each scaler handles different data characteristics:
    
    **StandardScaler (Z-score normalization)**:
    - **Formula**: `(x - mean) / std`
    - **Result**: Mean = 0, Std = 1
    - **Best for**: Normally distributed data without outliers
    - **Use cases**: Most ML algorithms, when data follows Gaussian distribution
    
    **MinMaxScaler**:
    - **Formula**: `(x - min) / (max - min)`
    - **Result**: Range [0, 1] or custom range
    - **Best for**: Bounded output needed, neural networks
    - **Use cases**: Image processing, when you need specific value ranges
    
    **RobustScaler**:
    - **Formula**: `(x - median) / IQR`
    - **Result**: Median = 0, IQR-based scale
    - **Best for**: Data with outliers, non-normal distributions
    - **Use cases**: Financial data, medical data with extreme values
    
    **Comparison with outliers**:
    ```python
    data = [1, 2, 3, 4, 5, 100]  # Last value is outlier
    
    # StandardScaler: All values affected by outlier
    # MinMaxScaler: Most values compressed near 0
    # RobustScaler: Outlier has minimal impact on scaling
    ```
    
    **Decision framework**:
    1. **Check for outliers** ’ If many, use RobustScaler
    2. **Check distribution** ’ If normal, use StandardScaler  
    3. **Check requirements** ’ If bounded output needed, use MinMaxScaler
    4. **Algorithm requirements** ’ Neural networks often prefer MinMax

??? question "Explain the difference between L1 and L2 regularization. When would you use each?"

    **Answer:** L1 and L2 regularization differ in penalty function and effects:
    
    **L1 Regularization (Lasso)**:
    - **Penalty**: $\lambda \sum |w_i|$ (sum of absolute values)
    - **Effect**: Creates sparse solutions (sets coefficients to exactly zero)
    - **Gradient**: Constant magnitude, doesn't shrink with coefficient size
    - **Feature selection**: Automatically selects relevant features
    
    **L2 Regularization (Ridge)**:
    - **Penalty**: $\lambda \sum w_i^2$ (sum of squared values)
    - **Effect**: Shrinks coefficients but rarely zeros them
    - **Gradient**: Proportional to coefficient size
    - **Multicollinearity**: Handles correlated features by distributing weights
    
    **Mathematical intuition**:
    ```
    L1 gradient: /w (»|w|) = »·sign(w)    # Constant push toward zero
    L2 gradient: /w (»w²) = 2»w           # Proportional shrinkage
    ```
    
    **Visual difference**:
    - **L1 constraint region**: Diamond shape ’ creates sparsity at corners
    - **L2 constraint region**: Circle shape ’ shrinks uniformly
    
    **When to use L1**:
    -  Feature selection needed
    -  Interpretable sparse models
    -  High-dimensional data with irrelevant features
    -  Storage/computation constraints
    
    **When to use L2**:
    -  All features potentially relevant
    -  Multicollinearity present
    -  Stability over sparsity
    -  Better numerical properties
    
    **Elastic Net** combines both: $\alpha \rho ||w||_1 + \alpha(1-\rho)||w||_2^2$

??? question "How do you determine the optimal regularization strength (lambda/alpha)?"

    **Answer:** Several approaches for finding optimal regularization strength:
    
    **1. Cross-Validation (Most common)**:
    ```python
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import Ridge
    
    alphas = [0.1, 1.0, 10.0, 100.0]
    best_alpha = None
    best_score = -np.inf
    
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        scores = cross_val_score(model, X, y, cv=5)
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_alpha = alpha
    ```
    
    **2. Grid Search with Cross-Validation**:
    ```python
    from sklearn.model_selection import GridSearchCV
    
    param_grid = {'alpha': np.logspace(-4, 4, 20)}
    grid_search = GridSearchCV(Ridge(), param_grid, cv=5)
    grid_search.fit(X, y)
    optimal_alpha = grid_search.best_params_['alpha']
    ```
    
    **3. Regularization Path Analysis**:
    - Plot performance vs. regularization strength
    - Look for "elbow" in validation curve
    - Balance bias-variance tradeoff
    
    **4. Information Criteria (AIC/BIC)**:
    ```python
    # For model selection without separate validation set
    # AIC = 2k - 2ln(L)  where k=parameters, L=likelihood
    ```
    
    **5. Early Stopping**:
    - Monitor validation loss during training
    - Stop when validation loss stops improving
    - Implicit regularization through training time
    
    **Search strategies**:
    - **Coarse to fine**: Start with wide range, then narrow down
    - **Logarithmic spacing**: Use `np.logspace` for wide range exploration
    - **Nested CV**: Use inner CV for hyperparameter selection, outer CV for evaluation
    
    **Practical tips**:
    - Start with wide range: [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    - Use stratified CV for classification
    - Consider computational budget vs. accuracy needs
    - Validate final choice on completely separate test set

??? question "What is the bias-variance tradeoff in the context of regularization?"

    **Answer:** Regularization directly addresses the bias-variance tradeoff:
    
    **Bias-Variance Decomposition**:
    $$E[(y - \hat{f}(x))^2] = \text{Bias}^2 + \text{Variance} + \text{Noise}$$
    
    **Without Regularization**:
    - **Low bias**: Model can fit training data well
    - **High variance**: Model overfits, predictions vary greatly with training data
    - **Risk**: Poor generalization to new data
    
    **With Regularization**:
    - **Higher bias**: Model constrained, can't fit training data perfectly
    - **Lower variance**: More stable predictions across different training sets
    - **Goal**: Minimize total error = Bias² + Variance + Noise
    
    **Regularization effects**:
    ```python
    # No regularization (» = 0): High variance, low bias
    # Strong regularization (» >> 1): Low variance, high bias  
    # Optimal »: Minimizes bias² + variance
    ```
    
    **Visual intuition**:
    - **Underfit** (too much regularization): High bias, predictions too simple
    - **Overfit** (too little regularization): High variance, predictions too complex
    - **Just right**: Balanced complexity, good generalization
    
    **Practical example**:
    ```python
    # Polynomial regression with different regularization
    » = 0:    Perfect training fit, poor test performance (overfit)
    » = 0.1:  Good training fit, good test performance (balanced)  
    » = 100:  Poor training fit, poor test performance (underfit)
    ```
    
    **How to detect**:
    - **High variance**: Large gap between training and validation performance
    - **High bias**: Both training and validation performance are poor
    - **Optimal point**: Minimal validation error
    
    **Regularization strength effects**:
    - **Increasing »**: Reduces variance, increases bias
    - **Decreasing »**: Reduces bias, increases variance
    - **Sweet spot**: Cross-validation finds optimal balance

??? question "How does regularization help with multicollinearity, and what's the difference between Ridge and Lasso in handling it?"

    **Answer:** Regularization addresses multicollinearity differently depending on the type:
    
    **Multicollinearity problem**:
    - **Issue**: When features are highly correlated, ordinary least squares becomes unstable
    - **Effect**: Small changes in data cause large changes in coefficients
    - **Math**: $(X^T X)$ becomes nearly singular, leading to unstable $(X^T X)^{-1}$
    
    **Ridge Regression approach**:
    - **Solution**: Adds $\lambda I$ to $(X^T X)$, making it invertible
    - **Formula**: $(X^T X + \lambda I)^{-1} X^T y$
    - **Effect**: Distributes coefficients among correlated features
    - **Example**: If features A and B are identical, Ridge gives both coefficient = 0.5
    
    **Lasso Regression approach**:
    - **Solution**: L1 penalty forces sparsity
    - **Effect**: Arbitrarily picks one feature from correlated group
    - **Example**: If features A and B are identical, Lasso gives one coefficient = 1, other = 0
    - **Limitation**: Selection among correlated features is somewhat random
    
    **Practical example**:
    ```python
    # Highly correlated features: house size and number of rooms
    X = [[2000, 4], [2500, 5], [3000, 6]]  # [sqft, rooms]
    
    # Ridge: Both features get partial coefficients
    # Ridge coefficients: [0.7, 0.6] (both contribute)
    
    # Lasso: One feature dominates  
    # Lasso coefficients: [1.2, 0.0] (only sqft matters)
    ```
    
    **Elastic Net solution**:
    - **Combines both**: $\alpha \rho ||w||_1 + \alpha(1-\rho)||w||_2^2$
    - **Advantage**: Groups correlated features together (like Ridge) but maintains sparsity (like Lasso)
    - **Best of both**: Handles multicollinearity while doing feature selection
    
    **Comparison summary**:
    | Aspect | Ridge | Lasso | Elastic Net |
    |--------|-------|-------|-------------|
    | **Multicollinearity** | Distributes weights | Random selection | Groups + selects |
    | **Stability** | High | Can be unstable | High |
    | **Feature selection** | No | Yes | Yes |
    | **Interpretability** | Medium | High | High |
    
    **When to use each**:
    - **Ridge**: When you believe all features are relevant
    - **Lasso**: When you need automatic feature selection
    - **Elastic Net**: When you have groups of correlated features

??? question "Explain data leakage in the context of feature scaling and how to prevent it."

    **Answer:** Data leakage in feature scaling occurs when test set information influences the scaling parameters:
    
    **What is scaling data leakage?**
    - **Problem**: Using entire dataset (including test set) to compute scaling parameters
    - **Effect**: Model has indirect access to test set information
    - **Result**: Overly optimistic performance estimates
    
    **Common mistakes**:
    ```python
    # WRONG: Scaling before train/test split
    X_scaled = StandardScaler().fit_transform(X)  # Uses ALL data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
    
    # WRONG: Fitting scaler on combined data
    scaler = StandardScaler().fit(np.vstack([X_train, X_test]))
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    ```
    
    **Correct approach**:
    ```python
    # CORRECT: Split first, then scale
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit only on training
    X_test_scaled = scaler.transform(X_test)        # Transform using train params
    ```
    
    **Why this matters**:
    - **Statistical contamination**: Test set statistics influence training
    - **Optimistic bias**: Model appears better than it actually is
    - **Production problems**: Real-world performance differs from validation
    
    **Cross-validation considerations**:
    ```python
    # CORRECT: Scaling inside CV loop
    from sklearn.pipeline import Pipeline
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression())
    ])
    
    # Scaler fitted separately for each CV fold
    scores = cross_val_score(pipeline, X, y, cv=5)
    ```
    
    **Time series special case**:
    - **Problem**: Future information leaking to past predictions
    - **Solution**: Use forward-chaining validation, fit scaler only on past data
    
    **Impact magnitude**:
    - Usually small but can be significant with small datasets
    - More problematic with MinMaxScaler (uses min/max)
    - Less problematic with RobustScaler (uses median/IQR)
    
    **Detection methods**:
    - Compare performance with/without proper scaling separation
    - Check if test performance seems unrealistically high
    - Validate scaling parameters make sense for training data only

??? question "How do you handle categorical features when applying normalization?"

    **Answer:** Categorical features require special handling as traditional scaling methods don't apply:
    
    **Why traditional scaling fails**:
    - **No inherent order**: Categories like [Red, Blue, Green] have no meaningful distance
    - **Arbitrary encoding**: Label encoding creates fake ordinal relationships
    - **Scale meaningless**: Normalizing [1, 2, 3] for categories is nonsensical
    
    **Proper approaches**:
    
    **1. One-Hot Encoding** (most common):
    ```python
    from sklearn.preprocessing import OneHotEncoder
    
    # Original: ['Red', 'Blue', 'Red', 'Green']  
    # One-hot: [[1,0,0], [0,1,0], [1,0,0], [0,0,1]]
    
    ohe = OneHotEncoder(drop='first', sparse=False)  # Avoid multicollinearity
    X_categorical_encoded = ohe.fit_transform(X_categorical)
    
    # Then apply scaling to numerical features only
    ```
    
    **2. Target Encoding**:
    ```python
    # Replace category with mean target value for that category
    category_means = df.groupby('category')['target'].mean()
    df['category_encoded'] = df['category'].map(category_means)
    
    # Can then apply scaling to encoded values
    ```
    
    **3. Mixed data pipeline**:
    ```python
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    
    # Specify which columns are categorical vs numerical
    numerical_features = ['age', 'income', 'credit_score']
    categorical_features = ['city', 'job_type', 'education']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ]
    )
    
    X_processed = preprocessor.fit_transform(X)
    ```
    
    **4. Ordinal encoding** (only for ordinal categories):
    ```python
    # Only for inherently ordered categories
    education_mapping = {
        'High School': 1,
        'Bachelor': 2, 
        'Master': 3,
        'PhD': 4
    }
    
    df['education_ordinal'] = df['education'].map(education_mapping)
    # Can then apply scaling
    ```
    
    **Best practices**:
    - **Separate preprocessing**: Handle categorical and numerical features separately
    - **Pipeline usage**: Use sklearn pipelines to prevent data leakage
    - **High cardinality**: Consider target encoding or embedding for many categories
    - **Rare categories**: Group infrequent categories into "Other" before encoding
    - **Validation**: Ensure consistent categories between train/test sets
    
    **Common pitfalls**:
    - Scaling label-encoded categorical features
    - Forgetting to handle new categories in test set
    - Creating too many dummy variables (curse of dimensionality)
    - Data leakage in target encoding without proper CV

??? question "What are some advanced regularization techniques beyond L1/L2?"

    **Answer:** Several advanced regularization techniques beyond basic L1/L2:
    
    **1. Elastic Net**:
    - **Formula**: $\alpha \rho ||w||_1 + \alpha(1-\rho)||w||_2^2$
    - **Advantage**: Combines L1 sparsity with L2 stability
    - **Use case**: Correlated features where you want grouping + selection
    
    **2. Group Lasso**:
    - **Concept**: Regularizes groups of features together
    - **Formula**: $\lambda \sum_{g} ||w_g||_2$ where $g$ represents feature groups
    - **Effect**: Either selects entire group or zeros out entire group
    - **Use case**: Gene expression, image pixels, polynomial features
    
    **3. Fused Lasso (Total Variation)**:
    - **Formula**: $\lambda_1 ||w||_1 + \lambda_2 \sum_{i} |w_i - w_{i+1}|$
    - **Effect**: Promotes sparsity + smooth coefficient transitions
    - **Use case**: Time series, spatial data, signal processing
    
    **4. Nuclear Norm (Matrix Regularization)**:
    - **Formula**: $\lambda ||W||_*$ (sum of singular values)
    - **Effect**: Promotes low-rank solutions
    - **Use case**: Matrix completion, collaborative filtering
    
    **5. Dropout (Neural Networks)**:
    ```python
    # Randomly set neurons to zero during training
    def dropout(x, rate=0.5, training=True):
        if training:
            mask = np.random.binomial(1, 1-rate, x.shape)
            return x * mask / (1-rate)
        return x
    ```
    
    **6. Batch Normalization**:
    - **Concept**: Normalize layer inputs during training
    - **Effect**: Stabilizes training, acts as regularization
    - **Formula**: $\frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} * \gamma + \beta$
    
    **7. Data Augmentation**:
    ```python
    # Create additional training examples through transformations
    # Images: rotation, scaling, flipping
    # Text: synonym replacement, back-translation
    # Time series: jittering, warping
    ```
    
    **8. Early Stopping**:
    - **Method**: Stop training when validation loss stops improving
    - **Effect**: Prevents overfitting through limited training time
    - **Implementation**: Monitor validation loss, stop after patience epochs
    
    **9. Weight Decay**:
    - **Concept**: Gradually reduce all weights during training
    - **Formula**: $w_{t+1} = (1-\lambda)w_t - \alpha \nabla L$
    - **Effect**: Similar to L2 but applied during optimization
    
    **10. Spectral Normalization**:
    - **Method**: Constrain spectral norm of weight matrices
    - **Effect**: Stabilizes GAN training, improves generalization
    - **Use case**: Generative models, discriminator regularization
    
    **Advanced combinations**:
    ```python
    # Multi-task learning with shared regularization
    Loss = £ TaskLoss_i + »||W_shared||‚² + »‚||W_specific||
    
    # Adaptive regularization (learning »)
    » = »€ * exp(-decay * epoch)
    ```
    
    **Selection criteria**:
    - **Data structure**: Spatial/temporal data ’ Fused Lasso
    - **High dimensions**: Group Lasso, Nuclear norm
    - **Neural networks**: Dropout, Batch norm, Weight decay
    - **Interpretability needs**: L1, Group Lasso
    - **Stability needs**: L2, Elastic Net

## >à Examples

### Real-world Example: Customer Churn Prediction

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Generate realistic customer churn dataset
np.random.seed(42)
n_customers = 5000

# Create realistic customer features with different scales
data = {
    'customer_id': range(1, n_customers + 1),
    'age': np.random.normal(40, 15, n_customers).clip(18, 80),
    'monthly_charges': np.random.normal(70, 25, n_customers).clip(20, 200),
    'total_charges': np.random.normal(2500, 1500, n_customers).clip(100, 8000),
    'contract_length': np.random.choice([1, 12, 24], n_customers, p=[0.3, 0.4, 0.3]),
    'num_services': np.random.poisson(3, n_customers).clip(1, 8),
    'support_calls': np.random.poisson(2, n_customers),
    'payment_method': np.random.choice(['Credit Card', 'Bank Transfer', 'Cash', 'Check'], 
                                      n_customers, p=[0.4, 0.3, 0.2, 0.1]),
    'internet_type': np.random.choice(['DSL', 'Fiber', 'Cable', 'None'], 
                                     n_customers, p=[0.3, 0.3, 0.3, 0.1]),
    'senior_citizen': np.random.choice([0, 1], n_customers, p=[0.85, 0.15])
}

# Create target variable (churn) with realistic relationships
churn_prob = (
    0.1 +  # Base churn rate
    0.1 * (data['monthly_charges'] > 100) +  # High charges increase churn
    0.15 * (data['contract_length'] == 1) +  # Month-to-month increases churn
    0.1 * (data['support_calls'] > 3) +      # Many support calls indicate issues
    0.05 * data['senior_citizen'] +          # Seniors slightly more likely to churn
    -0.08 * (data['contract_length'] == 24)  # Long contracts reduce churn
).clip(0, 1)

data['churn'] = np.random.binomial(1, churn_prob, n_customers)

# Create DataFrame
df_churn = pd.DataFrame(data)

print("Customer Churn Prediction - Normalization & Regularization Demo")
print(f"Dataset shape: {df_churn.shape}")
print(f"Churn rate: {df_churn['churn'].mean():.1%}")
print("\nDataset overview:")
print(df_churn.describe())

# Analyze feature distributions and scales
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
numerical_features = ['age', 'monthly_charges', 'total_charges', 'contract_length', 'num_services', 'support_calls']

for i, feature in enumerate(numerical_features):
    row, col = i // 3, i % 3
    
    # Plot distribution
    axes[row, col].hist(df_churn[feature], bins=30, alpha=0.7, edgecolor='black')
    axes[row, col].set_title(f'{feature}\nRange: [{df_churn[feature].min():.0f}, {df_churn[feature].max():.0f}]')
    axes[row, col].set_ylabel('Frequency')
    axes[row, col].grid(True, alpha=0.3)

plt.suptitle('Feature Distributions (Before Normalization)', fontsize=14)
plt.tight_layout()
plt.show()

# Prepare features for modeling
X = df_churn.drop(['customer_id', 'churn'], axis=1)
y = df_churn['churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define preprocessing for different column types
numerical_features = ['age', 'monthly_charges', 'total_charges', 'contract_length', 'num_services', 'support_calls']
categorical_features = ['payment_method', 'internet_type', 'senior_citizen']

print(f"\nFeature preprocessing:")
print(f"Numerical features: {numerical_features}")
print(f"Categorical features: {categorical_features}")

# Test different scaling approaches
scalers_test = {
    'No Scaling': None,
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler()
}

preprocessing_results = {}

for scaler_name, scaler in scalers_test.items():
    print(f"\nTesting {scaler_name}:")
    
    # Create preprocessing pipeline
    if scaler is None:
        # No scaling - just handle categorical variables
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(drop='first', sparse=False), categorical_features)
            ],
            remainder='passthrough'  # Keep numerical features as-is
        )
    else:
        # Apply scaling to numerical features
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', scaler, numerical_features),
                ('cat', OneHotEncoder(drop='first', sparse=False), categorical_features)
            ]
        )
    
    # Create full pipeline with logistic regression
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    # Evaluate using cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc')
    
    # Fit and predict for detailed metrics
    pipeline.fit(X_train, y_train)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_proba)
    
    preprocessing_results[scaler_name] = {
        'cv_auc_mean': cv_scores.mean(),
        'cv_auc_std': cv_scores.std(),
        'test_auc': test_auc
    }
    
    print(f"  CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
    print(f"  Test AUC: {test_auc:.3f}")

# Compare preprocessing approaches
preprocessing_df = pd.DataFrame(preprocessing_results).T

plt.figure(figsize=(12, 6))

# CV performance comparison
plt.subplot(1, 2, 1)
methods = preprocessing_df.index
cv_means = preprocessing_df['cv_auc_mean']
cv_stds = preprocessing_df['cv_auc_std']

bars = plt.bar(methods, cv_means, yerr=cv_stds, capsize=5, alpha=0.7)
plt.title('Cross-Validation Performance by Scaling Method')
plt.ylabel('AUC Score')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Add value labels
for bar, mean in zip(bars, cv_means):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{mean:.3f}', ha='center', va='bottom')

# Test performance comparison
plt.subplot(1, 2, 2)
test_aucs = preprocessing_df['test_auc']
bars = plt.bar(methods, test_aucs, alpha=0.7, color='orange')
plt.title('Test Set Performance by Scaling Method')
plt.ylabel('AUC Score')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Add value labels
for bar, auc in zip(bars, test_aucs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{auc:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Now test different regularization techniques
print(f"\n" + "="*60)
print("REGULARIZATION COMPARISON")
print("="*60)

# Use best scaling method
best_scaler = StandardScaler()  # Typically works well

preprocessor_final = ColumnTransformer(
    transformers=[
        ('num', best_scaler, numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse=False), categorical_features)
    ]
)

# Apply preprocessing
X_train_processed = preprocessor_final.fit_transform(X_train)
X_test_processed = preprocessor_final.transform(X_test)

print(f"Processed feature shape: {X_train_processed.shape}")

# Test different regularization techniques
regularization_models = {
    'Logistic Regression (No Reg)': LogisticRegression(penalty=None, max_iter=1000, random_state=42),
    'Ridge (L2)': LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42),
    'Lasso (L1)': LogisticRegression(penalty='l1', solver='liblinear', C=1.0, max_iter=1000, random_state=42),
    'ElasticNet': LogisticRegression(penalty='elasticnet', l1_ratio=0.5, solver='saga', C=1.0, max_iter=1000, random_state=42)
}

regularization_results = {}

for model_name, model in regularization_models.items():
    print(f"\nTesting {model_name}:")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_processed, y_train, cv=5, scoring='roc_auc')
    
    # Fit model
    model.fit(X_train_processed, y_train)
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Count non-zero coefficients (sparsity)
    if hasattr(model, 'coef_'):
        non_zero_coefs = np.sum(np.abs(model.coef_) > 1e-5)
        total_coefs = model.coef_.shape[1]
        sparsity = 1 - (non_zero_coefs / total_coefs)
    else:
        non_zero_coefs = "N/A"
        sparsity = "N/A"
    
    regularization_results[model_name] = {
        'cv_auc_mean': cv_scores.mean(),
        'cv_auc_std': cv_scores.std(),
        'test_auc': test_auc,
        'non_zero_coefs': non_zero_coefs,
        'sparsity': sparsity,
        'model': model
    }
    
    print(f"  CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
    print(f"  Test AUC: {test_auc:.3f}")
    print(f"  Non-zero coefficients: {non_zero_coefs}")
    if sparsity != "N/A":
        print(f"  Sparsity: {sparsity:.1%}")

# Hyperparameter tuning for best model
print(f"\n" + "="*40)
print("HYPERPARAMETER TUNING")
print("="*40)

# Tune regularization strength for Ridge
param_grid = {'C': np.logspace(-3, 2, 20)}

grid_search = GridSearchCV(
    LogisticRegression(penalty='l2', max_iter=1000, random_state=42),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

grid_search.fit(X_train_processed, y_train)

print(f"Best regularization strength (C): {grid_search.best_params_['C']:.4f}")
print(f"Best CV AUC: {grid_search.best_score_:.3f}")

# Final model evaluation
best_model = grid_search.best_estimator_
y_pred_proba_final = best_model.predict_proba(X_test_processed)[:, 1]
y_pred_final = best_model.predict(X_test_processed)

final_auc = roc_auc_score(y_test, y_pred_proba_final)
print(f"Final test AUC: {final_auc:.3f}")

# ROC curve comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
# Plot ROC curves for different regularization methods
for model_name, results in regularization_results.items():
    model = results['model']
    y_proba = model.predict_proba(X_test_processed)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Regularization Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# Feature importance (coefficients) for Ridge regression
plt.subplot(1, 2, 2)
ridge_model = regularization_results['Ridge (L2)']['model']

# Get feature names after preprocessing
feature_names = (numerical_features + 
                list(preprocessor_final.named_transformers_['cat']
                    .get_feature_names_out(categorical_features)))

coefficients = ridge_model.coef_[0]
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'coefficient': coefficients,
    'abs_coefficient': np.abs(coefficients)
}).sort_values('abs_coefficient', ascending=False)

# Plot top 10 most important features
top_features = feature_importance.head(10)
bars = plt.barh(range(len(top_features)), top_features['coefficient'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Coefficient Value')
plt.title('Top 10 Feature Importance (Ridge)')
plt.grid(True, alpha=0.3)

# Color bars by sign
for i, (bar, coef) in enumerate(zip(bars, top_features['coefficient'])):
    bar.set_color('red' if coef < 0 else 'blue')

plt.tight_layout()
plt.show()

print(f"\nTop 10 Most Important Features:")
for i, (_, row) in enumerate(top_features.iterrows()):
    direction = "increases" if row['coefficient'] > 0 else "decreases"
    print(f"{i+1:2d}. {row['feature']:25} ’ {direction} churn risk (coef: {row['coefficient']:+.3f})")
```

### Financial Risk Assessment Example

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import seaborn as sns

# Generate realistic financial dataset with outliers
np.random.seed(42)
n_loans = 2000

# Create features with different scales and outlier patterns
data = {
    'loan_amount': np.random.lognormal(10, 1, n_loans),  # Log-normal (right-skewed)
    'annual_income': np.random.lognormal(10.5, 0.8, n_loans),  # Income distribution
    'credit_score': np.random.beta(2, 1, n_loans) * 550 + 300,  # Credit scores 300-850
    'debt_to_income': np.random.exponential(0.3, n_loans),  # Debt ratios
    'employment_years': np.random.gamma(2, 2, n_loans),  # Employment history
    'num_credit_lines': np.random.poisson(8, n_loans),  # Count of credit lines
    'loan_to_value': np.random.uniform(0.5, 0.95, n_loans),  # LTV ratio
    'market_volatility': np.random.normal(0.15, 0.05, n_loans).clip(0.05, 0.4)  # Market conditions
}

# Add some extreme outliers (data entry errors, unusual cases)
outlier_indices = np.random.choice(n_loans, size=50, replace=False)
data['annual_income'][outlier_indices[:25]] *= 10  # Very high income outliers
data['debt_to_income'][outlier_indices[25:]] *= 5  # Very high debt outliers

# Create target: default risk score (0-1, higher = more risky)
risk_score = (
    0.1 * (data['debt_to_income'] / np.mean(data['debt_to_income'])) +
    0.2 * (1 - (data['credit_score'] - 300) / 550) +
    0.15 * (data['loan_to_value']) +
    0.1 * (data['market_volatility'] / 0.4) +
    -0.05 * np.log(data['annual_income'] / np.mean(data['annual_income'])) +
    0.1 * np.random.normal(0, 1, n_loans)  # Random noise
).clip(0, 1)

data['risk_score'] = risk_score

# Create DataFrame
df_risk = pd.DataFrame(data)

print("Financial Risk Assessment - Robust Scaling & Regularization")
print(f"Dataset shape: {df_risk.shape}")
print("\nDataset summary with outliers:")
print(df_risk.describe())

# Identify outliers using IQR method
def identify_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (df[column] < lower_bound) | (df[column] > upper_bound)

# Check for outliers in key features
outlier_analysis = {}
for col in ['loan_amount', 'annual_income', 'debt_to_income']:
    outliers = identify_outliers(df_risk, col)
    outlier_analysis[col] = {
        'count': outliers.sum(),
        'percentage': (outliers.sum() / len(df_risk)) * 100
    }
    print(f"\n{col}: {outliers.sum()} outliers ({(outliers.sum()/len(df_risk)*100):.1f}%)")

# Visualize distributions and outliers
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
features = list(df_risk.columns[:-1])  # Exclude target

for i, feature in enumerate(features):
    row, col = i // 3, i % 3
    
    # Histogram
    axes[row, col].hist(df_risk[feature], bins=50, alpha=0.7, edgecolor='black')
    axes[row, col].set_title(f'{feature}')
    axes[row, col].set_ylabel('Frequency')
    
    # Mark outliers if applicable
    if feature in outlier_analysis:
        outliers = identify_outliers(df_risk, feature)
        if outliers.sum() > 0:
            outlier_values = df_risk.loc[outliers, feature]
            axes[row, col].axvline(df_risk[feature].quantile(0.25) - 1.5*(df_risk[feature].quantile(0.75)-df_risk[feature].quantile(0.25)), 
                                  color='red', linestyle='--', alpha=0.7, label='Outlier bounds')
            axes[row, col].axvline(df_risk[feature].quantile(0.75) + 1.5*(df_risk[feature].quantile(0.75)-df_risk[feature].quantile(0.25)), 
                                  color='red', linestyle='--', alpha=0.7)
            axes[row, col].legend()
    
    axes[row, col].grid(True, alpha=0.3)

plt.suptitle('Feature Distributions with Outliers Highlighted', fontsize=14)
plt.tight_layout()
plt.show()

# Prepare data
X = df_risk.drop('risk_score', axis=1)
y = df_risk['risk_score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Compare different scaling approaches on data with outliers
scalers_robust = {
    'No Scaling': None,
    'StandardScaler': StandardScaler(),
    'RobustScaler': RobustScaler(),
    'QuantileTransformer': QuantileTransformer(output_distribution='normal')
}

scaling_results = {}

print(f"\n" + "="*60)
print("SCALING COMPARISON ON DATA WITH OUTLIERS")
print("="*60)

for scaler_name, scaler in scalers_robust.items():
    print(f"\nTesting {scaler_name}:")
    
    if scaler is None:
        X_train_scaled = X_train
        X_test_scaled = X_test
    else:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
    # Train simple linear regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_score = lr.score(X_train_scaled, y_train)
    test_score = lr.score(X_test_scaled, y_test)
    y_pred = lr.predict(X_test_scaled)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    scaling_results[scaler_name] = {
        'train_r2': train_score,
        'test_r2': test_score,
        'rmse': rmse,
        'mae': mae
    }
    
    print(f"  Train R²: {train_score:.3f}")
    print(f"  Test R²: {test_score:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAE: {mae:.3f}")

# Visualize scaling effects on first few features
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
sample_features = ['loan_amount', 'annual_income', 'debt_to_income']

for i, scaler_name in enumerate(['StandardScaler', 'RobustScaler', 'QuantileTransformer']):
    if i >= 3:
        break
        
    scaler = scalers_robust[scaler_name]
    X_scaled_sample = scaler.fit_transform(X_train[sample_features])
    
    row, col = i // 2, i % 2
    
    # Plot first feature
    axes[row, col].hist(X_scaled_sample[:, 0], bins=30, alpha=0.7, edgecolor='black')
    axes[row, col].set_title(f'{scaler_name}\nTransformed: {sample_features[0]}')
    axes[row, col].set_ylabel('Frequency')
    axes[row, col].grid(True, alpha=0.3)
    
    # Add statistics
    mean_val = np.mean(X_scaled_sample[:, 0])
    std_val = np.std(X_scaled_sample[:, 0])
    axes[row, col].text(0.02, 0.95, f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}', 
                       transform=axes[row, col].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Performance comparison
axes[1, 1].bar(scaling_results.keys(), [v['test_r2'] for v in scaling_results.values()], alpha=0.7)
axes[1, 1].set_title('Test R² by Scaling Method')
axes[1, 1].set_ylabel('R² Score')
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Now test regularization with best scaler (RobustScaler typically best for outliers)
print(f"\n" + "="*60)
print("REGULARIZATION WITH ROBUST SCALING")
print("="*60)

robust_scaler = RobustScaler()
X_train_robust = robust_scaler.fit_transform(X_train)
X_test_robust = robust_scaler.transform(X_test)

# Test different regularization strengths
alphas = np.logspace(-4, 2, 20)

# Test Ridge, Lasso, and ElasticNet
regularization_models = {
    'Ridge': Ridge(),
    'Lasso': Lasso(max_iter=2000),
    'ElasticNet': ElasticNet(max_iter=2000, l1_ratio=0.5)
}

regularization_paths = {}

for model_name, base_model in regularization_models.items():
    print(f"\nAnalyzing {model_name} regularization path:")
    
    train_scores = []
    test_scores = []
    coefficients = []
    sparsity_levels = []
    
    for alpha in alphas:
        # Set regularization strength
        if hasattr(base_model, 'alpha'):
            model = base_model.__class__(alpha=alpha, max_iter=2000)
            if model_name == 'ElasticNet':
                model = base_model.__class__(alpha=alpha, l1_ratio=0.5, max_iter=2000)
        
        # Fit model
        model.fit(X_train_robust, y_train)
        
        # Evaluate
        train_score = model.score(X_train_robust, y_train)
        test_score = model.score(X_test_robust, y_test)
        
        train_scores.append(train_score)
        test_scores.append(test_score)
        coefficients.append(model.coef_.copy())
        
        # Calculate sparsity (proportion of near-zero coefficients)
        sparsity = np.sum(np.abs(model.coef_) < 1e-5) / len(model.coef_)
        sparsity_levels.append(sparsity)
    
    regularization_paths[model_name] = {
        'train_scores': train_scores,
        'test_scores': test_scores,
        'coefficients': np.array(coefficients),
        'sparsity': sparsity_levels
    }
    
    # Find best alpha
    best_idx = np.argmax(test_scores)
    best_alpha = alphas[best_idx]
    best_test_score = test_scores[best_idx]
    
    print(f"  Best alpha: {best_alpha:.4f}")
    print(f"  Best test R²: {best_test_score:.3f}")
    print(f"  Sparsity at best alpha: {sparsity_levels[best_idx]:.1%}")

# Visualize regularization paths
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for i, (model_name, results) in enumerate(regularization_paths.items()):
    # Performance vs regularization strength
    axes[0, i].plot(alphas, results['train_scores'], 'b-', label='Train', linewidth=2)
    axes[0, i].plot(alphas, results['test_scores'], 'r-', label='Test', linewidth=2)
    axes[0, i].set_xscale('log')
    axes[0, i].set_xlabel('Regularization Strength (±)')
    axes[0, i].set_ylabel('R² Score')
    axes[0, i].set_title(f'{model_name}: Performance vs Regularization')
    axes[0, i].legend()
    axes[0, i].grid(True, alpha=0.3)
    
    # Mark best alpha
    best_idx = np.argmax(results['test_scores'])
    axes[0, i].axvline(alphas[best_idx], color='green', linestyle='--', alpha=0.7, 
                      label=f'Best ±={alphas[best_idx]:.4f}')
    
    # Coefficient paths (show first 5 features)
    for j in range(min(5, results['coefficients'].shape[1])):
        axes[1, i].plot(alphas, results['coefficients'][:, j], 
                       label=f'Feature {j+1}' if i == 0 else "")
    
    axes[1, i].set_xscale('log')
    axes[1, i].set_xlabel('Regularization Strength (±)')
    axes[1, i].set_ylabel('Coefficient Value')
    axes[1, i].set_title(f'{model_name}: Coefficient Paths')
    if i == 0:
        axes[1, i].legend()
    axes[1, i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Final model selection and feature importance
print(f"\n" + "="*40)
print("FINAL MODEL ANALYSIS")
print("="*40)

# Select best Ridge model (usually most stable)
best_alpha_ridge = alphas[np.argmax(regularization_paths['Ridge']['test_scores'])]
final_model = Ridge(alpha=best_alpha_ridge)
final_model.fit(X_train_robust, y_train)

# Final evaluation
y_pred_final = final_model.predict(X_test_robust)
final_r2 = r2_score(y_test, y_pred_final)
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_final))
final_mae = mean_absolute_error(y_test, y_pred_final)

print(f"Final Ridge Model (± = {best_alpha_ridge:.4f}):")
print(f"  Test R²: {final_r2:.3f}")
print(f"  RMSE: {final_rmse:.3f}")
print(f"  MAE: {final_mae:.3f}")

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': final_model.coef_,
    'abs_coefficient': np.abs(final_model.coef_)
}).sort_values('abs_coefficient', ascending=False)

print(f"\nFeature Importance Ranking:")
for i, (_, row) in enumerate(feature_importance.iterrows()):
    direction = "increases" if row['coefficient'] > 0 else "decreases"
    print(f"{i+1:2d}. {row['feature']:20} ’ {direction} risk (coef: {row['coefficient']:+.4f})")

# Final visualization
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_final, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Risk Score')
plt.ylabel('Predicted Risk Score')
plt.title(f'Final Model Performance\nR² = {final_r2:.3f}')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
bars = plt.barh(range(len(feature_importance)), feature_importance['coefficient'])
plt.yticks(range(len(feature_importance)), feature_importance['feature'])
plt.xlabel('Coefficient Value')
plt.title('Feature Importance (Ridge Coefficients)')
plt.grid(True, alpha=0.3)

# Color bars by sign
for bar, coef in zip(bars, feature_importance['coefficient']):
    bar.set_color('red' if coef < 0 else 'blue')

plt.tight_layout()
plt.show()
```

## =Ú References

- **Books:**
  - [The Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/) by Hastie, Tibshirani, and Friedman - Chapters 3, 18
  - [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) by Christopher Bishop - Chapter 1, 5
  - [Deep Learning](https://www.deeplearningbook.org/) by Goodfellow, Bengio, and Courville - Chapter 7 (Regularization)

- **Documentation:**
  - [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
  - [Scikit-learn Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
  - [Scikit-learn Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)

- **Research Papers:**
  - [Regularization and variable selection via the elastic net](https://web.stanford.edu/~hastie/Papers/B67.2%20(2005)%20301-320%20Zou%20&%20Hastie.pdf) by Zou & Hastie (2005)
  - [Regression Shrinkage and Selection via the Lasso](https://statweb.stanford.edu/~tibs/lasso/lasso.pdf) by Tibshirani (1996)
  - [Ridge Regression: Biased Estimation for Nonorthogonal Problems](https://www.math.arizona.edu/~hzhang/math574m/Read/RidgeRegressionHoerlKennard1970.pdf) by Hoerl & Kennard (1970)

- **Tutorials and Guides:**
  - [Feature Scaling Techniques](https://towardsdatascience.com/all-about-feature-scaling-bcc0ad75cb35)
  - [Regularization in Machine Learning](https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a)
  - [Understanding the Bias-Variance Tradeoff](http://scott.fortmann-roe.com/docs/BiasVariance.html)

- **Advanced Topics:**
  - [Group Lasso](https://tibshirani.su.domains/ftp/group.pdf) by Yuan & Lin (2006)
  - [The Fused Lasso](https://web.stanford.edu/~hastie/Papers/FusedLasso.pdf) by Tibshirani et al. (2005)
  - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html) by Srivastava et al. (2014)

- **Online Courses:**
  - [Machine Learning Course - Stanford CS229](http://cs229.stanford.edu/)
  - [Statistical Learning - Stanford Online](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)
  - [Regularization - Coursera Machine Learning](https://www.coursera.org/learn/machine-learning)

- **Software and Tools:**
  - [scikit-learn](https://scikit-learn.org/stable/) (Python)
  - [glmnet](https://cran.r-project.org/web/packages/glmnet/index.html) (R package)
  - [TensorFlow/Keras](https://www.tensorflow.org/) (Deep learning regularization)
  - [PyTorch](https://pytorch.org/) (Deep learning regularization)