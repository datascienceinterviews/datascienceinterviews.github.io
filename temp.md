---
title: Logistic Regression
description: Comprehensive guide to Logistic Regression with mathematical intuition, implementations, and interview questions.
comments: true
---

# =Ø Logistic Regression

Logistic Regression is a statistical method used for binary and multiclass classification problems that models the probability of class membership using the logistic function.

**Resources:** [Scikit-learn Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) | [Stanford CS229 Notes](http://cs229.stanford.edu/notes2020spring/cs229-notes1.pdf)

##  Summary

Logistic Regression is a linear classifier that uses the logistic function (sigmoid) to map any real-valued number into a value between 0 and 1, making it suitable for probability estimation and classification tasks.

**Key characteristics:**
- **Probabilistic**: Outputs probabilities rather than direct classifications
- **Linear decision boundary**: Creates linear decision boundaries in feature space
- **No distributional assumptions**: Unlike linear regression, doesn't assume normal distribution of errors
- **Robust to outliers**: Less sensitive to outliers compared to linear regression
- **Interpretable**: Coefficients have direct interpretation as log-odds ratios

**Applications:**
- Medical diagnosis (disease/no disease)
- Marketing (click/no click, buy/don't buy)
- Finance (default/no default)
- Email classification (spam/ham)
- Customer churn prediction
- A/B test analysis

**Types:**
- **Binary Logistic Regression**: Two classes (0 or 1)
- **Multinomial Logistic Regression**: Multiple classes (>2)
- **Ordinal Logistic Regression**: Ordered categories

## >à Intuition

### How Logistic Regression Works

While linear regression predicts continuous values, logistic regression predicts the probability that an instance belongs to a particular category. It uses the logistic (sigmoid) function to constrain outputs between 0 and 1.

### Mathematical Foundation

#### 1. The Logistic Function (Sigmoid)

The sigmoid function maps any real number to a value between 0 and 1:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Where $z = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_px_p$

#### 2. Odds and Log-Odds

**Odds** represent the ratio of probability of success to probability of failure:
$$\text{Odds} = \frac{p}{1-p}$$

**Log-odds (logit)** is the natural logarithm of odds:
$$\text{logit}(p) = \ln\left(\frac{p}{1-p}\right) = z$$

#### 3. The Logistic Regression Model

For binary classification:
$$P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1x_1 + ... + \beta_px_p)}}$$

**Key insight**: Linear combination of features determines the log-odds, while the sigmoid function converts it to probability.

#### 4. Maximum Likelihood Estimation

Logistic regression uses maximum likelihood estimation (MLE) to find optimal parameters. The likelihood function for $n$ observations is:

$$L(\beta) = \prod_{i=1}^{n} P(y_i|x_i)^{y_i} \cdot (1-P(y_i|x_i))^{1-y_i}$$

**Log-likelihood** (easier to optimize):
$$\ell(\beta) = \sum_{i=1}^{n} [y_i \log(P(y_i|x_i)) + (1-y_i) \log(1-P(y_i|x_i))]$$

#### 5. Cost Function

The cost function (negative log-likelihood) for logistic regression is:
$$J(\beta) = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(h_\beta(x_i)) + (1-y_i) \log(1-h_\beta(x_i))]$$

Where $h_\beta(x_i) = \sigma(\beta^T x_i)$ is the hypothesis function.

#### 6. Gradient Descent

The gradient of the cost function with respect to parameters:
$$\frac{\partial J(\beta)}{\partial \beta_j} = \frac{1}{n} \sum_{i=1}^{n} (h_\beta(x_i) - y_i) x_{ij}$$

**Update rule**:
$$\beta_j := \beta_j - \alpha \frac{\partial J(\beta)}{\partial \beta_j}$$

### Algorithm Steps

1. **Initialize parameters** $\beta$ randomly or to zero
2. **Forward propagation**: Calculate predictions using sigmoid function
3. **Calculate cost** using log-likelihood
4. **Backward propagation**: Calculate gradients
5. **Update parameters** using gradient descent
6. **Repeat** until convergence

## =" Implementation using Libraries

### Using Scikit-learn

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, 
                           confusion_matrix, roc_curve, auc, 
                           precision_recall_curve)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Binary Classification Example
print("=" * 50)
print("BINARY LOGISTIC REGRESSION")
print("=" * 50)

# Generate sample data
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (important for logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train model
log_reg = LogisticRegression(
    random_state=42,
    max_iter=1000,
    solver='lbfgs'  # Good for small datasets
)

log_reg.fit(X_train_scaled, y_train)

# Make predictions
y_pred = log_reg.predict(X_test_scaled)
y_pred_proba = log_reg.predict_proba(X_test_scaled)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
print(f"\nCoefficients: {log_reg.coef_[0]}")
print(f"Intercept: {log_reg.intercept_[0]:.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
roc_auc = auc(fpr, tpr)

plt.subplot(1, 3, 2)
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

# Decision boundary visualization
plt.subplot(1, 3, 3)
h = 0.02
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = log_reg.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], 
           c=y_train, cmap='RdYlBu', edgecolors='black')
plt.title('Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

# Real-world example: Breast Cancer Dataset
print("\n" + "=" * 50)
print("REAL-WORLD EXAMPLE: BREAST CANCER CLASSIFICATION")
print("=" * 50)

# Load dataset
cancer = load_breast_cancer()
X_cancer = cancer.data
y_cancer = cancer.target

# Use subset of features for interpretability
feature_names = cancer.feature_names[:10]  # First 10 features
X_cancer_subset = X_cancer[:, :10]

print(f"Dataset shape: {X_cancer_subset.shape}")
print(f"Features: {list(feature_names)}")
print(f"Classes: {cancer.target_names}")

# Split and scale
X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer = train_test_split(
    X_cancer_subset, y_cancer, test_size=0.2, random_state=42, stratify=y_cancer
)

scaler_cancer = StandardScaler()
X_train_cancer_scaled = scaler_cancer.fit_transform(X_train_cancer)
X_test_cancer_scaled = scaler_cancer.transform(X_test_cancer)

# Train model
cancer_model = LogisticRegression(random_state=42, max_iter=1000)
cancer_model.fit(X_train_cancer_scaled, y_train_cancer)

# Predictions
y_pred_cancer = cancer_model.predict(X_test_cancer_scaled)
y_pred_proba_cancer = cancer_model.predict_proba(X_test_cancer_scaled)

print(f"Cancer Classification Accuracy: {accuracy_score(y_test_cancer, y_pred_cancer):.3f}")

# Feature importance (coefficients)
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': cancer_model.coef_[0],
    'Abs_Coefficient': np.abs(cancer_model.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print("\nFeature Importance (by coefficient magnitude):")
print(feature_importance)

# Multiclass Classification Example
print("\n" + "=" * 50)
print("MULTICLASS LOGISTIC REGRESSION")
print("=" * 50)

from sklearn.datasets import make_classification

# Generate multiclass data
X_multi, y_multi = make_classification(
    n_samples=1000,
    n_features=4,
    n_informative=3,
    n_redundant=1,
    n_classes=3,
    n_clusters_per_class=1,
    random_state=42
)

X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42, stratify=y_multi
)

scaler_multi = StandardScaler()
X_train_multi_scaled = scaler_multi.fit_transform(X_train_multi)
X_test_multi_scaled = scaler_multi.transform(X_test_multi)

# Train multiclass model
multi_model = LogisticRegression(
    multi_class='ovr',  # One-vs-Rest
    random_state=42,
    max_iter=1000
)

multi_model.fit(X_train_multi_scaled, y_train_multi)

# Evaluate
y_pred_multi = multi_model.predict(X_test_multi_scaled)
print(f"Multiclass Accuracy: {accuracy_score(y_test_multi, y_pred_multi):.3f}")
print("\nMulticlass Classification Report:")
print(classification_report(y_test_multi, y_pred_multi))

# Cross-validation
cv_scores = cross_val_score(multi_model, X_train_multi_scaled, y_train_multi, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'penalty': ['l1', 'l2'],        # Regularization type
    'solver': ['liblinear', 'saga'] # Solvers that support both L1 and L2
}

# Grid search
grid_search = GridSearchCV(
    LogisticRegression(random_state=42, max_iter=1000),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Use best model
best_model = grid_search.best_estimator_
best_pred = best_model.predict(X_test_scaled)
print("Best model test accuracy:", accuracy_score(y_test, best_pred))
```

##  From Scratch Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LogisticRegressionFromScratch:
    """Logistic Regression implementation from scratch"""
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, fit_intercept=True, verbose=False):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        
    def _add_intercept(self, X):
        """Add bias column to the feature matrix"""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def _sigmoid(self, z):
        """Sigmoid activation function"""
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _cost_function(self, h, y):
        """Calculate the logistic regression cost function"""
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-15
        h = np.clip(h, epsilon, 1 - epsilon)
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        """Train the logistic regression model"""
        # Add intercept term if needed
        if self.fit_intercept:
            X = self._add_intercept(X)
        
        # Initialize weights
        self.weights = np.zeros(X.shape[1])
        
        # Store cost history
        self.cost_history = []
        
        # Gradient descent
        for i in range(self.max_iterations):
            # Forward propagation
            z = np.dot(X, self.weights)
            h = self._sigmoid(z)
            
            # Calculate cost
            cost = self._cost_function(h, y)
            self.cost_history.append(cost)
            
            # Calculate gradient
            gradient = np.dot(X.T, (h - y)) / y.size
            
            # Update weights
            self.weights -= self.learning_rate * gradient
            
            # Print progress
            if self.verbose and i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost:.6f}")
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if self.fit_intercept:
            X = self._add_intercept(X)
        
        probabilities = self._sigmoid(np.dot(X, self.weights))
        return np.vstack([1 - probabilities, probabilities]).T
    
    def predict(self, X):
        """Make predictions"""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
    
    def score(self, X, y):
        """Calculate accuracy"""
        return (self.predict(X) == y).mean()

# Example usage of from-scratch implementation
print("=" * 60)
print("FROM SCRATCH IMPLEMENTATION")
print("=" * 60)

# Generate sample data
np.random.seed(42)
X_scratch, y_scratch = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

# Split and scale data
X_train_scratch, X_test_scratch, y_train_scratch, y_test_scratch = train_test_split(
    X_scratch, y_scratch, test_size=0.2, random_state=42
)

scaler_scratch = StandardScaler()
X_train_scratch_scaled = scaler_scratch.fit_transform(X_train_scratch)
X_test_scratch_scaled = scaler_scratch.transform(X_test_scratch)

# Train custom logistic regression
custom_lr = LogisticRegressionFromScratch(
    learning_rate=0.01,
    max_iterations=1000,
    verbose=True
)

custom_lr.fit(X_train_scratch_scaled, y_train_scratch)

# Make predictions
y_pred_scratch = custom_lr.predict(X_test_scratch_scaled)
y_pred_proba_scratch = custom_lr.predict_proba(X_test_scratch_scaled)

custom_accuracy = custom_lr.score(X_test_scratch_scaled, y_test_scratch)
print(f"\nCustom Logistic Regression Accuracy: {custom_accuracy:.3f}")
print(f"Final weights: {custom_lr.weights}")

# Compare with sklearn
sklearn_lr = LogisticRegression(random_state=42, max_iter=1000)
sklearn_lr.fit(X_train_scratch_scaled, y_train_scratch)
sklearn_pred = sklearn_lr.predict(X_test_scratch_scaled)
sklearn_accuracy = accuracy_score(y_test_scratch, sklearn_pred)

print(f"Scikit-learn Accuracy: {sklearn_accuracy:.3f}")
print(f"Accuracy difference: {abs(custom_accuracy - sklearn_accuracy):.4f}")

# Plot cost function
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(custom_lr.cost_history)
plt.title('Cost Function During Training')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.grid(True)

# Visualize decision boundary
plt.subplot(1, 2, 2)
h = 0.02
x_min, x_max = X_train_scratch_scaled[:, 0].min() - 1, X_train_scratch_scaled[:, 0].max() + 1
y_min, y_max = X_train_scratch_scaled[:, 1].min() - 1, X_train_scratch_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = custom_lr.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
plt.scatter(X_train_scratch_scaled[:, 0], X_train_scratch_scaled[:, 1], 
           c=y_train_scratch, cmap='RdYlBu', edgecolors='black')
plt.title('Custom Implementation Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

# Advanced from-scratch implementation with regularization
class RegularizedLogisticRegression:
    """Logistic Regression with L1 and L2 regularization"""
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, 
                 regularization=None, lambda_reg=0.01, fit_intercept=True):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.regularization = regularization  # 'l1', 'l2', or None
        self.lambda_reg = lambda_reg
        self.fit_intercept = fit_intercept
        
    def _add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _cost_function(self, h, y):
        epsilon = 1e-15
        h = np.clip(h, epsilon, 1 - epsilon)
        cost = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
        
        # Add regularization term
        if self.regularization == 'l1':
            # Don't regularize intercept term
            reg_term = self.lambda_reg * np.sum(np.abs(self.weights[1:]))
        elif self.regularization == 'l2':
            reg_term = self.lambda_reg * np.sum(self.weights[1:] ** 2)
        else:
            reg_term = 0
            
        return cost + reg_term
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self._add_intercept(X)
        
        self.weights = np.zeros(X.shape[1])
        self.cost_history = []
        
        for i in range(self.max_iterations):
            # Forward propagation
            z = np.dot(X, self.weights)
            h = self._sigmoid(z)
            
            # Calculate cost
            cost = self._cost_function(h, y)
            self.cost_history.append(cost)
            
            # Calculate gradient
            gradient = np.dot(X.T, (h - y)) / y.size
            
            # Add regularization to gradient
            if self.regularization == 'l1':
                gradient[1:] += self.lambda_reg * np.sign(self.weights[1:])
            elif self.regularization == 'l2':
                gradient[1:] += 2 * self.lambda_reg * self.weights[1:]
            
            # Update weights
            self.weights -= self.learning_rate * gradient
    
    def predict_proba(self, X):
        if self.fit_intercept:
            X = self._add_intercept(X)
        probabilities = self._sigmoid(np.dot(X, self.weights))
        return np.vstack([1 - probabilities, probabilities]).T
    
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# Test regularized implementation
print("\n" + "=" * 60)
print("REGULARIZED IMPLEMENTATION")
print("=" * 60)

# Test L1 regularization
l1_model = RegularizedLogisticRegression(
    learning_rate=0.01,
    max_iterations=1000,
    regularization='l1',
    lambda_reg=0.01
)

l1_model.fit(X_train_scratch_scaled, y_train_scratch)
l1_accuracy = l1_model.score(X_test_scratch_scaled, y_test_scratch)

# Test L2 regularization
l2_model = RegularizedLogisticRegression(
    learning_rate=0.01,
    max_iterations=1000,
    regularization='l2',
    lambda_reg=0.01
)

l2_model.fit(X_train_scratch_scaled, y_train_scratch)
l2_accuracy = l2_model.score(X_test_scratch_scaled, y_test_scratch)

print(f"L1 Regularized Accuracy: {l1_accuracy:.3f}")
print(f"L2 Regularized Accuracy: {l2_accuracy:.3f}")
print(f"No Regularization Accuracy: {custom_accuracy:.3f}")

print(f"\nL1 weights: {l1_model.weights}")
print(f"L2 weights: {l2_model.weights}")
print(f"No reg weights: {custom_lr.weights}")
```

##   Assumptions and Limitations

### Assumptions

1. **Linear relationship between logit and features**: The log-odds should be a linear combination of features
2. **Independence of observations**: Each observation should be independent
3. **No multicollinearity**: Features should not be highly correlated
4. **Large sample size**: Generally needs larger sample sizes than linear regression
5. **Binary or ordinal outcome**: Dependent variable should be categorical

### Limitations

1. **Linear decision boundary**:
   - Can only create linear decision boundaries
   - **Solution**: Feature engineering, polynomial features, or non-linear algorithms

2. **Sensitive to outliers**:
   - Extreme values can influence the model significantly
   - **Solution**: Robust scaling, outlier detection and removal

3. **Assumes no missing values**:
   - Cannot handle missing data directly
   - **Solution**: Imputation or algorithms that handle missing values

4. **Requires feature scaling**:
   - Features on different scales can bias the model
   - **Solution**: Standardization or normalization

5. **Perfect separation problems**:
   - When classes are perfectly separable, coefficients can become infinite
   - **Solution**: Regularization (L1/L2)

### Comparison with Other Algorithms

| Algorithm | Interpretability | Speed | Non-linear | Probability Output | Overfitting Risk |
|-----------|-----------------|-------|------------|-------------------|------------------|
| Logistic Regression | PPPPP | PPPPP | L |  | PP |
| Decision Trees | PPPP | PPPP |  |  | PPPP |
| Random Forest | PP | PPP |  |  | PP |
| SVM | PP | PP |  | L | PPP |
| Neural Networks | P | PP |  |  | PPPPP |

**When to use Logistic Regression:**
-  When you need interpretable results
-  For baseline models
-  When you have linear relationships
-  When you need probability estimates
-  With limited training data

**When to avoid:**
- L When relationships are highly non-linear
- L When you have very high-dimensional data
- L When interpretability is not important and accuracy is paramount

## =¡ Interview Questions

??? question "1. Explain the difference between Linear Regression and Logistic Regression."

    **Key Differences:**

    | Aspect | Linear Regression | Logistic Regression |
    |--------|------------------|-------------------|
    | **Purpose** | Predicts continuous values | Predicts probabilities/classes |
    | **Output range** | (-, +) | [0, 1] |
    | **Function** | Linear: y = ²X + µ | Logistic: p = 1/(1 + e^(-²X)) |
    | **Error distribution** | Normal | Binomial |
    | **Cost function** | Mean Squared Error | Log-likelihood |
    | **Parameters estimation** | Least squares | Maximum likelihood |
    | **Decision boundary** | Not applicable | Linear |

    **Mathematical relationship:**
    ```
    Linear Regression: y = ² + ²x + ²x + ... + ²x + µ
    
    Logistic Regression: log(p/(1-p)) = ² + ²x + ²x + ... + ²x
    ```

    **When to use each:**
    - **Linear Regression**: Predicting house prices, temperatures, stock prices
    - **Logistic Regression**: Email spam detection, medical diagnosis, customer churn

??? question "2. What is the sigmoid function and why is it used in logistic regression?"

    **Sigmoid Function:**
    $$\sigma(z) = \frac{1}{1 + e^{-z}}$$

    **Properties:**
    
    1. **Range [0,1]**: Perfect for probability estimation
    2. **S-shaped curve**: Smooth transition between 0 and 1  
    3. **Differentiable**: Enables gradient descent optimization
    4. **Asymptotic**: Approaches 0 and 1 but never reaches them

    **Why sigmoid is used:**

    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    z = np.linspace(-10, 10, 100)
    y = sigmoid(z)

    plt.figure(figsize=(8, 5))
    plt.plot(z, y, 'b-', linewidth=2)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Decision boundary')
    plt.xlabel('z (linear combination)')
    plt.ylabel('Ã(z) (probability)')
    plt.title('Sigmoid Function')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    ```

    **Mathematical advantages:**
    - Maps any real number to (0,1)
    - Derivative: Ã'(z) = Ã(z)(1 - Ã(z))
    - Smooth gradient for optimization
    - Interpretable as probability

??? question "3. How do you interpret the coefficients in logistic regression?"

    **Coefficient Interpretation:**

    **Raw Coefficients (²):**
    - Represent change in **log-odds** per unit change in feature
    - If ² = 0.5, then one unit increase in x increases log-odds by 0.5

    **Odds Ratios (e^²):**
    - More interpretable than raw coefficients
    - If OR = e^² = 2, the odds double with one unit increase in feature

    **Example interpretation:**
    ```python
    # Example: Email spam classification
    # Features: [word_count, has_links, sender_reputation]
    # Coefficients: [0.1, 1.2, -0.8]
    
    coefficients = [0.1, 1.2, -0.8]
    odds_ratios = np.exp(coefficients)
    
    interpretations = [
        f"word_count: ²={coefficients[0]}, OR={odds_ratios[0]:.2f}",
        f"has_links: ²={coefficients[1]}, OR={odds_ratios[1]:.2f}", 
        f"sender_reputation: ²={coefficients[2]}, OR={odds_ratios[2]:.2f}"
    ]
    
    for interp in interpretations:
        print(interp)
    ```

    **Interpretation:**
    - **word_count (²=0.1)**: Each additional word increases spam odds by 10%
    - **has_links (²=1.2)**: Having links increases spam odds by 232%  
    - **sender_reputation (²=-0.8)**: Better reputation decreases spam odds by 55%

    **Key points:**
    - Positive ²: Increases probability of positive class
    - Negative ²: Decreases probability of positive class
    - Magnitude indicates strength of effect
    - Sign indicates direction of effect

??? question "4. What is the difference between odds and probability?"

    **Definitions:**

    **Probability (p):** 
    - Range: [0, 1]
    - P(event occurs) = number of favorable outcomes / total outcomes

    **Odds:**
    - Range: [0, ]
    - Odds = P(event occurs) / P(event doesn't occur) = p / (1-p)

    **Mathematical relationship:**
    ```python
    def prob_to_odds(p):
        return p / (1 - p)
    
    def odds_to_prob(odds):
        return odds / (1 + odds)
    
    # Examples
    probabilities = [0.1, 0.25, 0.5, 0.75, 0.9]
    
    print("Probability  Odds conversion:")
    for p in probabilities:
        odds = prob_to_odds(p)
        print(f"P = {p:.2f}  Odds = {odds:.2f}")
    
    # Output:
    # P = 0.10  Odds = 0.11  (1:9 against)
    # P = 0.25  Odds = 0.33  (1:3 against) 
    # P = 0.50  Odds = 1.00  (1:1 even)
    # P = 0.75  Odds = 3.00  (3:1 for)
    # P = 0.90  Odds = 9.00  (9:1 for)
    ```

    **Log-odds (logit):**
    - Range: (-, +)  
    - logit(p) = log(p/(1-p)) = log(odds)
    - This is what logistic regression actually models

    **Why this matters:**
    - Logistic regression predicts log-odds (linear combination)
    - Sigmoid converts log-odds back to probability
    - Coefficients represent changes in log-odds, not probability

??? question "5. How does Maximum Likelihood Estimation work in logistic regression?"

    **Maximum Likelihood Estimation (MLE):**

    **Concept:** Find parameters that make the observed data most likely.

    **Likelihood function:**
    $$L(\beta) = \prod_{i=1}^{n} P(y_i|x_i)^{y_i} \cdot (1-P(y_i|x_i))^{1-y_i}$$

    **Log-likelihood (easier to optimize):**
    $$\ell(\beta) = \sum_{i=1}^{n} [y_i \log(P(y_i|x_i)) + (1-y_i) \log(1-P(y_i|x_i))]$$

    **Step-by-step process:**

    ```python
    def log_likelihood(y_true, y_pred):
        # Avoid log(0) by clipping predictions
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        ll = np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return ll
    
    # Example with simple data
    y_true = np.array([0, 0, 1, 1])
    
    # Poor predictions
    y_pred_bad = np.array([0.9, 0.8, 0.2, 0.1])
    ll_bad = log_likelihood(y_true, y_pred_bad)
    
    # Good predictions  
    y_pred_good = np.array([0.1, 0.2, 0.8, 0.9])
    ll_good = log_likelihood(y_true, y_pred_good)
    
    print(f"Bad predictions log-likelihood: {ll_bad:.3f}")
    print(f"Good predictions log-likelihood: {ll_good:.3f}")
    print(f"Good predictions have higher likelihood!")
    ```

    **Why MLE over least squares:**
    - Least squares assumes normal distribution of errors
    - MLE is appropriate for binary outcomes
    - Provides asymptotic properties (consistency, efficiency)
    - Naturally handles the [0,1] constraint of probabilities

    **Optimization:**
    - No closed-form solution (unlike linear regression)
    - Uses iterative methods: Newton-Raphson, gradient descent
    - Requires numerical optimization algorithms

??? question "6. What is regularization in logistic regression and why is it needed?"

    **Regularization:** Technique to prevent overfitting by adding penalty term to cost function.

    **Why regularization is needed:**

    1. **Perfect separation**: When classes are linearly separable, coefficients  
    2. **Overfitting**: High-dimensional data with few samples
    3. **Multicollinearity**: Correlated features cause unstable estimates
    4. **Numerical stability**: Prevents extreme coefficient values

    **Types of regularization:**

    **L1 Regularization (Lasso):**
    $$J(\beta) = -\ell(\beta) + \lambda \sum_{j=1}^{p} |\beta_j|$$

    ```python
    # L1 regularization promotes sparsity
    from sklearn.linear_model import LogisticRegression

    # Strong L1 regularization
    l1_model = LogisticRegression(penalty='l1', C=0.1, solver='liblinear')
    l1_model.fit(X_train, y_train)
    
    print("L1 Coefficients:", l1_model.coef_[0])
    print("Number of zero coefficients:", np.sum(l1_model.coef_[0] == 0))
    ```

    **L2 Regularization (Ridge):**
    $$J(\beta) = -\ell(\beta) + \lambda \sum_{j=1}^{p} \beta_j^2$$

    ```python
    # L2 regularization shrinks coefficients  
    l2_model = LogisticRegression(penalty='l2', C=0.1)
    l2_model.fit(X_train, y_train)
    
    print("L2 Coefficients:", l2_model.coef_[0])
    print("Coefficient magnitudes:", np.abs(l2_model.coef_[0]))
    ```

    **Elastic Net (L1 + L2):**
    $$J(\beta) = -\ell(\beta) + \lambda_1 \sum_{j=1}^{p} |\beta_j| + \lambda_2 \sum_{j=1}^{p} \beta_j^2$$

    **Key differences:**
    
    | Regularization | Effect | Use Case | Parameter in sklearn |
    |---------------|--------|----------|-------------------|
    | **L1** | Feature selection, sparse | High-dim data, feature selection | penalty='l1' |
    | **L2** | Shrinks coefficients | Multicollinearity, general | penalty='l2' |
    | **Elastic Net** | Combines both | Best of both worlds | penalty='elasticnet' |

    **C parameter:** Inverse of regularization strength
    - Large C = Less regularization (more complex model)
    - Small C = More regularization (simpler model)

??? question "7. How do you handle multiclass classification with logistic regression?"

    **Strategies for multiclass classification:**

    **1. One-vs-Rest (OvR):**
    - Train K binary classifiers (K = number of classes)
    - Each classifier: "Class i vs all other classes"  
    - Prediction: Class with highest probability

    ```python
    # One-vs-Rest implementation
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_iris

    iris = load_iris()
    X, y = iris.data, iris.target

    # OvR is default for multiclass
    ovr_model = LogisticRegression(multi_class='ovr')
    ovr_model.fit(X, y)
    
    print("OvR model shape:", ovr_model.coef_.shape)  # (3, 4) - 3 classes, 4 features
    print("Classes:", iris.target_names)
    ```

    **2. One-vs-One (OvO):**
    - Train K(K-1)/2 binary classifiers
    - Each classifier: "Class i vs Class j"
    - Prediction: Majority voting

    **3. Multinomial Logistic Regression:**
    - Single model that directly handles multiple classes
    - Uses softmax function instead of sigmoid
    - More efficient than OvR/OvO

    ```python
    # Multinomial approach
    multinomial_model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    multinomial_model.fit(X, y)
    
    # Softmax probabilities
    probabilities = multinomial_model.predict_proba(X[:5])
    print("Softmax probabilities:")
    print(probabilities)
    print("Row sums (should be 1):", probabilities.sum(axis=1))
    ```

    **Softmax function (for multinomial):**
    $$P(y_i = k) = \frac{e^{z_{ik}}}{\sum_{j=1}^{K} e^{z_{ij}}}$$

    **Comparison:**
    
    | Method | # Models | Training Time | Prediction Speed | Memory |
    |--------|----------|---------------|------------------|--------|
    | **OvR** | K | Fast | Fast | Low |
    | **OvO** | K(K-1)/2 | Slow | Medium | High |
    | **Multinomial** | 1 | Medium | Very Fast | Very Low |

    **When to use each:**
    - **OvR**: Default choice, works well in practice
    - **OvO**: When individual binary problems are easier
    - **Multinomial**: When classes are mutually exclusive and exhaustive

??? question "8. How do you evaluate a logistic regression model?"

    **Evaluation metrics for logistic regression:**

    **1. Classification Accuracy:**
    ```python
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_true, y_pred)
    ```

    **2. Confusion Matrix:**
    ```python
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    ```

    **3. Precision, Recall, F1-Score:**
    ```python
    from sklearn.metrics import classification_report, precision_recall_fscore_support
    
    print(classification_report(y_true, y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)
    ```

    **4. ROC Curve and AUC:**
    ```python
    from sklearn.metrics import roc_curve, auc, roc_auc_score
    
    # For binary classification
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Direct calculation
    auc_score = roc_auc_score(y_true, y_pred_proba[:, 1])
    ```

    **5. Precision-Recall Curve:**
    ```python
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba[:, 1])
    avg_precision = average_precision_score(y_true, y_pred_proba[:, 1])
    ```

    **6. Log-Loss:**
    ```python
    from sklearn.metrics import log_loss
    
    # Measures quality of probability predictions
    logloss = log_loss(y_true, y_pred_proba[:, 1])
    ```

    **7. Cross-Validation:**
    ```python
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    ```

    **When to use each metric:**
    
    | Metric | Use Case | Important When |
    |--------|----------|---------------|
    | **Accuracy** | Balanced datasets | Equal misclassification costs |
    | **Precision** | False positives costly | Spam detection, medical screening |
    | **Recall** | False negatives costly | Disease diagnosis, fraud detection |
    | **F1-Score** | Imbalanced data | Balance precision and recall |
    | **AUC-ROC** | Ranking quality | Overall discriminative ability |
    | **PR-AUC** | Imbalanced data | Focus on positive class |
    | **Log-Loss** | Probability quality | Calibrated probabilities needed |

??? question "9. What are the assumptions of logistic regression and how do you check them?"

    **Assumptions of Logistic Regression:**

    **1. Independence of observations:**
    - Each observation should be independent
    
    **Check:** 
    - Review data collection process
    - Look for time series or clustered data
    - Use Durbin-Watson test for time series

    ```python
    from statsmodels.stats.diagnostic import acorr_ljungbox
    
    # Check for autocorrelation in residuals
    residuals = y_true - y_pred_proba[:, 1]
    ljung_box = acorr_ljungbox(residuals, lags=10)
    print("Ljung-Box test p-values:", ljung_box['lb_pvalue'])
    ```

    **2. Linear relationship between logit and features:**
    - Log-odds should be linear combination of features
    
    **Check:** Box-Tidwell test, visual inspection
    ```python
    # Visual check: logit vs continuous features
    def logit(p):
        return np.log(p / (1 - p))
    
    # Group data by feature quantiles and calculate logit
    for feature in continuous_features:
        plt.figure(figsize=(8, 5))
        
        # Create quantile groups
        quantiles = pd.qcut(X[feature], q=10, duplicates='drop')
        grouped_mean = y.groupby(quantiles).mean()
        
        # Calculate logit (avoid 0 and 1)
        grouped_mean = np.clip(grouped_mean, 0.01, 0.99)
        logit_values = logit(grouped_mean)
        
        plt.scatter(grouped_mean.index, logit_values)
        plt.xlabel(f'{feature} (quantiles)')
        plt.ylabel('Logit')
        plt.title(f'Linearity Check: {feature}')
        plt.show()
    ```

    **3. No multicollinearity:**
    - Features should not be highly correlated
    
    **Check:** VIF (Variance Inflation Factor)
    ```python
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                       for i in range(X.shape[1])]
    
    print("VIF values:")
    print(vif_data.sort_values('VIF', ascending=False))
    print("\nRule of thumb: VIF > 10 indicates multicollinearity")
    ```

    **4. Large sample size:**
    - Need adequate samples per parameter
    
    **Rule of thumb:** At least 10-20 events per predictor variable

    ```python
    # Check sample size adequacy
    n_samples, n_features = X.shape
    n_events = np.sum(y == 1)  # For binary classification
    
    ratio = n_events / n_features
    print(f"Events per predictor: {ratio:.1f}")
    print("Adequate if > 10-20")
    ```

    **5. No influential outliers:**
    - Extreme values shouldn't dominate the model
    
    **Check:** Cook's distance, standardized residuals
    ```python
    from scipy import stats
    
    # Calculate standardized residuals
    y_pred_prob = model.predict_proba(X)[:, 1]
    residuals = y - y_pred_prob
    std_residuals = residuals / np.sqrt(y_pred_prob * (1 - y_pred_prob))
    
    # Identify outliers
    outlier_threshold = 2.5
    outliers = np.abs(std_residuals) > outlier_threshold
    
    print(f"Number of potential outliers: {np.sum(outliers)}")
    print(f"Percentage of outliers: {np.mean(outliers) * 100:.1f}%")
    ```

    **What to do if assumptions are violated:**
    
    | Assumption Violated | Solutions |
    |-------------------|-----------|
    | **Independence** | Use mixed-effects models, cluster-robust errors |
    | **Linearity** | Add polynomial terms, splines, or transform variables |
    | **Multicollinearity** | Remove correlated features, PCA, regularization |
    | **Sample size** | Collect more data, use regularization, simpler model |
    | **Outliers** | Remove outliers, use robust methods, transform data |

??? question "10. How does logistic regression handle imbalanced datasets?"

    **Challenges with imbalanced data:**
    - Model biased toward majority class
    - High accuracy but poor minority class recall
    - Misleading performance metrics

    **Solutions:**

    **1. Class weighting:**
    ```python
    # Automatically balance class weights
    balanced_model = LogisticRegression(class_weight='balanced')
    balanced_model.fit(X_train, y_train)
    
    # Manual class weights
    manual_weights = {0: 1, 1: 10}  # Give 10x weight to minority class
    weighted_model = LogisticRegression(class_weight=manual_weights)
    ```

    **2. Resampling techniques:**
    ```python
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek
    
    # Oversampling minority class
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
    
    # Undersampling majority class
    undersampler = RandomUnderSampler(random_state=42)
    X_under, y_under = undersampler.fit_resample(X_train, y_train)
    
    # Combined approach
    combined = SMOTETomek(random_state=42)
    X_combined, y_combined = combined.fit_resample(X_train, y_train)
    ```

    **3. Threshold tuning:**
    ```python
    from sklearn.metrics import precision_recall_curve
    
    # Find optimal threshold
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    # Maximize F1-score
    f1_scores = 2 * precision * recall / (precision + recall)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Use optimal threshold for predictions
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    ```

    **4. Ensemble methods:**
    ```python
    from sklearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier
    
    # Balanced bagging
    balanced_bagging = BalancedBaggingClassifier(
        base_estimator=LogisticRegression(),
        random_state=42
    )
    
    # Balanced random forest
    balanced_rf = BalancedRandomForestClassifier(random_state=42)
    ```

    **5. Evaluation metrics for imbalanced data:**
    ```python
    from sklearn.metrics import (classification_report, confusion_matrix, 
                                roc_auc_score, average_precision_score)
    
    # Focus on minority class performance
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # ROC-AUC (less affected by imbalance)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    # Precision-Recall AUC (better for imbalanced data)
    auc_pr = average_precision_score(y_test, y_pred_proba)
    
    print(f"ROC-AUC: {auc_roc:.3f}")
    print(f"PR-AUC: {auc_pr:.3f}")
    ```

    **Comparison of approaches:**
    
    | Method | Pros | Cons | When to Use |
    |--------|------|------|-------------|
    | **Class Weighting** | Simple, fast | May overfit minority | Small to moderate imbalance |
    | **SMOTE** | Creates synthetic samples | Potential overfitting | Moderate imbalance |
    | **Undersampling** | Fast, simple | Loss of information | Large datasets |
    | **Threshold Tuning** | No data modification | Requires validation set | Any imbalance level |
    | **Ensemble** | Often best performance | More complex | Severe imbalance |

    **Best practices:**
    - Use stratified cross-validation
    - Focus on precision, recall, F1-score, not just accuracy
    - Consider business cost of false positives vs false negatives
    - Use PR-AUC over ROC-AUC for severe imbalance
    - Combine multiple approaches (e.g., SMOTE + class weighting)

## >à Examples

### Example 1: Customer Churn Prediction

```python
# Customer churn prediction using logistic regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc

# Simulate customer churn data
np.random.seed(42)
n_customers = 2000

# Generate synthetic customer data
data = {
    'age': np.random.normal(40, 12, n_customers).astype(int),
    'tenure_months': np.random.exponential(24, n_customers).astype(int),
    'monthly_charges': np.random.normal(65, 20, n_customers),
    'total_charges': np.random.normal(1500, 800, n_customers),
    'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                     n_customers, p=[0.5, 0.3, 0.2]),
    'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
                                      n_customers, p=[0.4, 0.2, 0.2, 0.2]),
    'customer_service_calls': np.random.poisson(2, n_customers),
    'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], 
                                        n_customers, p=[0.4, 0.4, 0.2])
}

# Create churn based on logical rules (with noise)
churn_probability = (
    (data['contract_type'] == 'Month-to-month') * 0.3 +
    (data['customer_service_calls'] > 3) * 0.2 +
    (data['monthly_charges'] > 80) * 0.15 +
    (data['tenure_months'] < 12) * 0.25 +
    np.random.normal(0, 0.1, n_customers)  # Add noise
)

churn = (churn_probability > 0.5).astype(int)

# Create DataFrame
df_churn = pd.DataFrame(data)
df_churn['churn'] = churn

print("Customer Churn Dataset:")
print(df_churn.head())
print(f"\nChurn rate: {df_churn['churn'].mean():.2%}")
print(f"Dataset shape: {df_churn.shape}")

# Exploratory Data Analysis
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Numerical features
numerical_features = ['age', 'tenure_months', 'monthly_charges', 'total_charges']
for i, feature in enumerate(numerical_features):
    ax = axes[i//3, i%3]
    df_churn.boxplot(column=feature, by='churn', ax=ax)
    ax.set_title(f'{feature} by Churn Status')
    ax.set_xlabel('Churn (0=No, 1=Yes)')

# Categorical features
df_churn['contract_type'].value_counts().plot(kind='bar', ax=axes[1, 2])
axes[1, 2].set_title('Contract Type Distribution')
axes[1, 2].set_xlabel('Contract Type')

# Customer service calls vs churn
churn_by_calls = df_churn.groupby('customer_service_calls')['churn'].mean()
axes[1, 1].bar(churn_by_calls.index, churn_by_calls.values)
axes[1, 1].set_title('Churn Rate by Customer Service Calls')
axes[1, 1].set_xlabel('Number of Calls')
axes[1, 1].set_ylabel('Churn Rate')

plt.tight_layout()
plt.show()

# Data preprocessing
# Encode categorical variables
le_contract = LabelEncoder()
le_payment = LabelEncoder()
le_internet = LabelEncoder()

df_processed = df_churn.copy()
df_processed['contract_type_encoded'] = le_contract.fit_transform(df_churn['contract_type'])
df_processed['payment_method_encoded'] = le_payment.fit_transform(df_churn['payment_method'])
df_processed['internet_service_encoded'] = le_internet.fit_transform(df_churn['internet_service'])

# Select features for modeling
feature_columns = ['age', 'tenure_months', 'monthly_charges', 'total_charges',
                  'contract_type_encoded', 'payment_method_encoded', 
                  'customer_service_calls', 'internet_service_encoded']

X = df_processed[feature_columns]
y = df_processed['churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
churn_model = LogisticRegression(
    random_state=42,
    class_weight='balanced',  # Handle class imbalance
    max_iter=1000
)

churn_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = churn_model.predict(X_test_scaled)
y_pred_proba = churn_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate model
print("\n" + "="*50)
print("CHURN PREDICTION RESULTS")
print("="*50)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=['No Churn', 'Churn']))

# Feature importance analysis
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': churn_model.coef_[0],
    'Abs_Coefficient': np.abs(churn_model.coef_[0]),
    'Odds_Ratio': np.exp(churn_model.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Interpret results
print("\nBusiness Insights:")
for _, row in feature_importance.head(3).iterrows():
    feature = row['Feature']
    coef = row['Coefficient']
    odds_ratio = row['Odds_Ratio']
    
    if coef > 0:
        impact = "increases"
    else:
        impact = "decreases"
    
    print(f"- {feature}: {impact} churn odds by {abs(odds_ratio-1)*100:.1f}%")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Customer Churn - ROC Curve')
plt.legend(loc="lower right")

# Feature importance visualization
plt.subplot(1, 2, 2)
top_features = feature_importance.head(6)
colors = ['red' if x < 0 else 'green' for x in top_features['Coefficient']]
plt.barh(range(len(top_features)), top_features['Coefficient'], color=colors)
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Coefficient Value')
plt.title('Feature Importance (Logistic Regression Coefficients)')
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.show()

# Customer segments analysis
print("\n" + "="*50)
print("CUSTOMER SEGMENTATION ANALYSIS")
print("="*50)

# Predict churn for different customer segments
segments = {
    'New Month-to-month': {
        'age': 35, 'tenure_months': 3, 'monthly_charges': 70, 
        'total_charges': 210, 'contract_type_encoded': le_contract.transform(['Month-to-month'])[0],
        'payment_method_encoded': le_payment.transform(['Electronic check'])[0],
        'customer_service_calls': 5, 'internet_service_encoded': le_internet.transform(['Fiber optic'])[0]
    },
    'Loyal Two-year': {
        'age': 45, 'tenure_months': 36, 'monthly_charges': 60,
        'total_charges': 2160, 'contract_type_encoded': le_contract.transform(['Two year'])[0],
        'payment_method_encoded': le_payment.transform(['Bank transfer'])[0],
        'customer_service_calls': 1, 'internet_service_encoded': le_internet.transform(['DSL'])[0]
    }
}

for segment_name, segment_data in segments.items():
    segment_features = np.array([[segment_data[col] for col in feature_columns]])
    segment_scaled = scaler.transform(segment_features)
    churn_prob = churn_model.predict_proba(segment_scaled)[0, 1]
    
    print(f"{segment_name} customer:")
    print(f"  Churn probability: {churn_prob:.1%}")
    print(f"  Risk level: {'High' if churn_prob > 0.7 else 'Medium' if churn_prob > 0.3 else 'Low'}")
    print()
```

### Example 2: Medical Diagnosis Classification

```python
# Medical diagnosis using logistic regression
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np

print("="*60)
print("MEDICAL DIAGNOSIS: BREAST CANCER CLASSIFICATION")
print("="*60)

# Load breast cancer dataset
cancer = load_breast_cancer()
X_medical = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y_medical = cancer.target

print(f"Dataset Information:")
print(f"- Samples: {len(X_medical)}")
print(f"- Features: {len(cancer.feature_names)}")
print(f"- Classes: {cancer.target_names}")
print(f"- Class distribution: {np.bincount(y_medical)}")

# Focus on most clinically relevant features
clinical_features = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean compactness', 'mean concavity', 'mean concave points',
    'worst radius', 'worst perimeter', 'worst area'
]

X_clinical = X_medical[clinical_features]

# Split data
X_train_med, X_test_med, y_train_med, y_test_med = train_test_split(
    X_clinical, y_medical, test_size=0.2, random_state=42, stratify=y_medical
)

# Scale features
scaler_med = StandardScaler()
X_train_med_scaled = scaler_med.fit_transform(X_train_med)
X_test_med_scaled = scaler_med.transform(X_test_med)

# Train medical diagnosis model
medical_model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    C=1.0  # No strong regularization for medical application
)

medical_model.fit(X_train_med_scaled, y_train_med)

# Predictions
y_pred_med = medical_model.predict(X_test_med_scaled)
y_pred_proba_med = medical_model.predict_proba(X_test_med_scaled)

print(f"\nDiagnostic Model Performance:")
print(f"Accuracy: {accuracy_score(y_test_med, y_pred_med):.3f}")

# Medical-specific metrics
from sklearn.metrics import confusion_matrix, precision_score, recall_score

cm_med = confusion_matrix(y_test_med, y_pred_med)
tn, fp, fn, tp = cm_med.ravel()

# Medical terminology
sensitivity = recall_score(y_test_med, y_pred_med)  # True Positive Rate
specificity = tn / (tn + fp)  # True Negative Rate
ppv = precision_score(y_test_med, y_pred_med)  # Positive Predictive Value
npv = tn / (tn + fn)  # Negative Predictive Value

print(f"\nMedical Performance Metrics:")
print(f"Sensitivity (Recall): {sensitivity:.3f}")
print(f"Specificity: {specificity:.3f}")
print(f"Positive Predictive Value: {ppv:.3f}")
print(f"Negative Predictive Value: {npv:.3f}")

print(f"\nConfusion Matrix:")
print(f"                 Predicted")
print(f"               Benign  Malignant")
print(f"Actual Benign    {tn:2d}      {fp:2d}")
print(f"    Malignant    {fn:2d}      {tp:2d}")

# Clinical interpretation of features
feature_clinical = pd.DataFrame({
    'Clinical_Feature': clinical_features,
    'Coefficient': medical_model.coef_[0],
    'Odds_Ratio': np.exp(medical_model.coef_[0]),
    'Clinical_Impact': ['Increases malignancy risk' if c > 0 else 'Decreases malignancy risk' 
                       for c in medical_model.coef_[0]]
}).sort_values('Coefficient', key=abs, ascending=False)

print(f"\nClinical Feature Analysis:")
print(feature_clinical)

# Risk stratification
risk_thresholds = [0.3, 0.7]
risk_levels = []

for prob in y_pred_proba_med[:, 1]:
    if prob < risk_thresholds[0]:
        risk_levels.append('Low Risk')
    elif prob < risk_thresholds[1]:
        risk_levels.append('Moderate Risk')
    else:
        risk_levels.append('High Risk')

risk_df = pd.DataFrame({
    'Patient_ID': range(len(y_test_med)),
    'Actual': ['Malignant' if y == 1 else 'Benign' for y in y_test_med],
    'Predicted_Probability': y_pred_proba_med[:, 1],
    'Risk_Level': risk_levels
})

print(f"\nRisk Stratification Summary:")
print(risk_df['Risk_Level'].value_counts())

# Show some example cases
print(f"\nExample High-Risk Cases:")
high_risk_cases = risk_df[risk_df['Risk_Level'] == 'High Risk'].head(3)
for _, case in high_risk_cases.iterrows():
    print(f"Patient {case['Patient_ID']}: {case['Predicted_Probability']:.1%} malignancy risk "
          f"(Actual: {case['Actual']})")
```

## =Ú References

1. **Books:**
   - [The Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/) - Hastie, Tibshirani, Friedman
   - [An Introduction to Statistical Learning](https://www.statlearning.com/) - James, Witten, Hastie, Tibshirani
   - [Applied Logistic Regression](https://www.wiley.com/en-us/Applied+Logistic+Regression%2C+3rd+Edition-p-9780470582473) - Hosmer, Lemeshow, Sturdivant

2. **Academic Papers:**
   - [Maximum Likelihood Estimation](https://projecteuclid.org/euclid.aos/1176342752) - Original MLE theory
   - [Regularization Paths for Generalized Linear Models](https://web.stanford.edu/~hastie/Papers/glmnet.pdf) - Friedman et al.

3. **Online Resources:**
   - [Scikit-learn Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
   - [Stanford CS229 - Machine Learning](http://cs229.stanford.edu/notes2020spring/cs229-notes1.pdf)
   - [MIT 6.034 - Logistic Regression](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/)

4. **Interactive Tools:**
   - [Logistic Regression Visualization](https://playground.tensorflow.org/)
   - [Seeing Theory - Regression](https://seeing-theory.brown.edu/regression-analysis/)

5. **Video Lectures:**
   - [Andrew Ng - Machine Learning Course](https://www.coursera.org/learn/machine-learning)
   - [StatQuest - Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8)
   - [3Blue1Brown - Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

6. **Documentation:**
   - [Statsmodels - Logistic Regression](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.html)
   - [TensorFlow - Classification](https://www.tensorflow.org/tutorials/keras/classification)