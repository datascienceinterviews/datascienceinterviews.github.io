---
title: Random Forest
description: Comprehensive guide to Random Forest ensemble method with mathematical intuition, implementations, and interview questions.
comments: true
---

# =ÿ Random Forest

Random Forest is a powerful ensemble machine learning algorithm that builds multiple decision trees and combines their predictions to create a more robust and accurate model, reducing overfitting while maintaining interpretability.

**Resources:** [Scikit-learn Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) | [Random Forests Paper - Leo Breiman](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf) | [Elements of Statistical Learning - Chapter 15](https://web.stanford.edu/~hastie/ElemStatLearn/)

##  Summary

Random Forest is an ensemble learning method that combines multiple decision trees using two key techniques: **bagging** (bootstrap aggregating) and **random feature selection**. Each tree in the forest is trained on a bootstrap sample of the data and considers only a random subset of features at each split, reducing correlation between trees and improving generalization.

**Key characteristics:**
- **Ensemble method**: Combines multiple decision trees for better performance
- **Bootstrap sampling**: Each tree trained on different subset of data
- **Random feature selection**: Each split considers random subset of features
- **Reduces overfitting**: Averaging multiple trees reduces variance
- **Handles missing values**: Can work with datasets containing missing values
- **Feature importance**: Provides built-in feature importance metrics

**Applications:**
- Classification tasks (fraud detection, medical diagnosis)
- Regression problems (house price prediction, stock returns)
- Feature selection and ranking
- Outlier detection
- Missing value imputation
- Bioinformatics and genomics
- Natural language processing
- Computer vision

**Types:**
- **Random Forest Classifier**: For classification tasks
- **Random Forest Regressor**: For regression tasks
- **Extremely Randomized Trees (Extra Trees)**: Uses random thresholds for splits
- **Isolation Forest**: Specialized variant for anomaly detection

## >‡ Intuition

### How Random Forest Works

Imagine you're making an important decision and want multiple expert opinions. Instead of asking one expert (single decision tree), you ask 100 experts (trees), where each expert:
1. Has seen different training examples (bootstrap sampling)
2. Considers different aspects of the problem (random features)
3. Makes their own prediction

The final decision is made by majority vote (classification) or averaging (regression). This "wisdom of crowds" approach often performs better than any individual expert.

### Mathematical Foundation

#### 1. Bootstrap Aggregating (Bagging)

For a training dataset $D = \{(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)\}$:

1. **Bootstrap sampling**: Create $B$ bootstrap samples $D_1, D_2, ..., D_B$
   - Each $D_b$ contains $n$ samples drawn with replacement from $D$
   - Typically ~63.2% unique samples per bootstrap

2. **Train individual trees**: For each $D_b$, train tree $T_b$

3. **Aggregate predictions**:
   - **Classification**: $\hat{y} = \text{mode}(T_1(x), T_2(x), ..., T_B(x))$
   - **Regression**: $\hat{y} = \frac{1}{B}\sum_{b=1}^{B} T_b(x)$

#### 2. Random Feature Selection

At each node split, instead of considering all $p$ features, randomly select $m$ features where:
- **Classification**: $m = \sqrt{p}$ (typical default)
- **Regression**: $m = \frac{p}{3}$ (typical default)
- **Custom**: Can be tuned as hyperparameter

#### 3. Out-of-Bag (OOB) Error

For each sample not in bootstrap sample $D_b$ (out-of-bag samples):
- Use trees trained on $D_b$ to predict
- OOB error provides unbiased estimate of generalization error

$$\text{OOB Error} = \frac{1}{n}\sum_{i=1}^{n} L(y_i, \hat{y}_{i}^{OOB})$$

where $\hat{y}_{i}^{OOB}$ is the prediction for $x_i$ using only trees where $x_i$ was out-of-bag.

#### 4. Feature Importance

**Gini Importance** (Mean Decrease Impurity):
$$\text{Importance}(f) = \frac{1}{B}\sum_{b=1}^{B} \sum_{t \in T_b} p(t) \cdot \Delta I(t)$$

where $p(t)$ is the proportion of samples reaching node $t$, and $\Delta I(t)$ is the impurity decrease at node $t$ when splitting on feature $f$.

**Permutation Importance** (Mean Decrease Accuracy):
1. Calculate OOB error for original data
2. Randomly permute values of feature $f$ in OOB samples  
3. Calculate new OOB error with permuted feature
4. Importance = increase in OOB error

#### 5. Variance Reduction

For $B$ independent trees with variance $\sigma^2$, the ensemble variance is:
$$\text{Var}(\text{ensemble}) = \frac{\sigma^2}{B}$$

However, trees are correlated with correlation $\rho$:
$$\text{Var}(\text{ensemble}) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$

Random feature selection reduces $\rho$, improving variance reduction.

## =" Implementation using Libraries

### Scikit-learn Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_boston, make_classification
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Classification Example with Iris Dataset
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
feature_names = iris.feature_names

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_iris, y_iris, test_size=0.3, random_state=42, stratify=y_iris)

# Create and train Random Forest classifier
rf_classifier = RandomForestClassifier(
    n_estimators=100,          # Number of trees
    max_depth=None,            # No limit on tree depth
    min_samples_split=2,       # Min samples to split internal node
    min_samples_leaf=1,        # Min samples at leaf node
    max_features='sqrt',       # Features to consider at each split
    bootstrap=True,            # Use bootstrap sampling
    oob_score=True,           # Calculate out-of-bag score
    random_state=42
)

rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)
y_prob = rf_classifier.predict_proba(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
oob_score = rf_classifier.oob_score_

print(f"Random Forest Classification Results:")
print(f"Accuracy: {accuracy:.3f}")
print(f"Out-of-bag Score: {oob_score:.3f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_classifier.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nFeature Importance:")
print(feature_importance)

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
plt.title('Random Forest Feature Importance')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, 
            yticklabels=iris.target_names)
plt.title('Confusion Matrix - Random Forest')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
```

### Regression Example

```python
# Generate synthetic regression data
X_reg, y_reg = make_classification(
    n_samples=1000, n_features=20, n_informative=15, 
    n_redundant=5, noise=0.1, random_state=42
)

# Convert to regression problem
y_reg = y_reg.astype(float) + np.random.normal(0, 0.1, len(y_reg))

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42)

# Random Forest Regressor
rf_regressor = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    oob_score=True,
    random_state=42
)

rf_regressor.fit(X_train_reg, y_train_reg)

# Predictions
y_pred_reg = rf_regressor.predict(X_test_reg)

# Evaluate
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_reg, y_pred_reg)
oob_score_reg = rf_regressor.oob_score_

print(f"\nRandom Forest Regression Results:")
print(f"RMSE: {rmse:.3f}")
print(f"R≤ Score: {r2:.3f}")
print(f"Out-of-bag Score: {oob_score_reg:.3f}")

# Plot predictions vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.6)
plt.plot([y_test_reg.min(), y_test_reg.max()], 
         [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Random Forest Regression: Actual vs Predicted (R≤ = {r2:.3f})')
plt.grid(True)
plt.show()
```

### Hyperparameter Tuning

```python
# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Grid search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.3f}")

# Best model performance
best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test)
best_accuracy = accuracy_score(y_test, y_pred_best)

print(f"Test accuracy with best parameters: {best_accuracy:.3f}")
```

### Learning Curves Analysis

```python
from sklearn.model_selection import learning_curve

def plot_learning_curves(estimator, X, y, cv=5):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy', random_state=42)
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='blue')
    
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                     alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy Score')
    plt.title('Random Forest Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot learning curves for default Random Forest
plot_learning_curves(RandomForestClassifier(random_state=42), X_iris, y_iris)
```

## ô From Scratch Implementation

```python
import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats
import matplotlib.pyplot as plt

class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Feature index for split
        self.threshold = threshold  # Threshold value for split
        self.left = left           # Left child node
        self.right = right         # Right child node
        self.value = value         # Prediction value (for leaf nodes)

class DecisionTreeFromScratch:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                 max_features=None, task='classification'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.task = task
        self.tree = None
        
    def fit(self, X, y):
        self.n_features = X.shape[1]
        if self.max_features is None:
            self.max_features = self.n_features
        elif isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
                self.max_features = int(np.sqrt(self.n_features))
            elif self.max_features == 'log2':
                self.max_features = int(np.log2(self.n_features))
        
        self.tree = self._build_tree(X, y, depth=0)
        
    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           len(np.unique(y)) == 1:
            return DecisionTreeNode(value=self._leaf_value(y))
        
        # Random feature selection
        feature_indices = np.random.choice(n_features, self.max_features, replace=False)
        
        # Find best split
        best_feature, best_threshold = self._best_split(X, y, feature_indices)
        
        if best_feature is None:
            return DecisionTreeNode(value=self._leaf_value(y))
        
        # Create child nodes
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices
        
        # Check minimum samples per leaf
        if np.sum(left_indices) < self.min_samples_leaf or \
           np.sum(right_indices) < self.min_samples_leaf:
            return DecisionTreeNode(value=self._leaf_value(y))
        
        left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return DecisionTreeNode(best_feature, best_threshold, left_child, right_child)
    
    def _best_split(self, X, y, feature_indices):
        best_gini = float('inf')
        best_feature, best_threshold = None, None
        
        for feature_idx in feature_indices:
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                left_indices = X[:, feature_idx] <= threshold
                right_indices = ~left_indices
                
                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue
                
                # Calculate weighted impurity
                n_left, n_right = len(y[left_indices]), len(y[right_indices])
                n_total = n_left + n_right
                
                if self.task == 'classification':
                    gini_left = self._gini_impurity(y[left_indices])
                    gini_right = self._gini_impurity(y[right_indices])
                    weighted_gini = (n_left/n_total) * gini_left + (n_right/n_total) * gini_right
                else:  # regression
                    mse_left = self._mse(y[left_indices])
                    mse_right = self._mse(y[right_indices])
                    weighted_gini = (n_left/n_total) * mse_left + (n_right/n_total) * mse_right
                
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _gini_impurity(self, y):
        if len(y) == 0:
            return 0
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)
    
    def _mse(self, y):
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)
    
    def _leaf_value(self, y):
        if self.task == 'classification':
            return stats.mode(y)[0][0]
        else:
            return np.mean(y)
    
    def predict(self, X):
        return np.array([self._predict_sample(sample, self.tree) for sample in X])
    
    def _predict_sample(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

class RandomForestFromScratch:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', bootstrap=True,
                 task='classification', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.task = task
        self.random_state = random_state
        self.trees = []
        self.feature_importances_ = None
        
    def fit(self, X, y):
        if self.random_state:
            np.random.seed(self.random_state)
        
        self.n_samples, self.n_features = X.shape
        self.trees = []
        
        # Feature importance tracking
        feature_importance_sum = np.zeros(self.n_features)
        
        for i in range(self.n_estimators):
            # Bootstrap sampling
            if self.bootstrap:
                indices = np.random.choice(self.n_samples, self.n_samples, replace=True)
                X_bootstrap = X[indices]
                y_bootstrap = y[indices]
            else:
                X_bootstrap, y_bootstrap = X, y
            
            # Train tree
            tree = DecisionTreeFromScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                task=self.task
            )
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
            
        # Calculate feature importances (simplified version)
        self._calculate_feature_importance(X, y)
        
    def _calculate_feature_importance(self, X, y):
        """Simplified feature importance using permutation importance"""
        self.feature_importances_ = np.zeros(self.n_features)
        baseline_score = self._score(X, y)
        
        for feature_idx in range(self.n_features):
            # Permute feature
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, feature_idx])
            
            # Calculate score with permuted feature
            permuted_score = self._score(X_permuted, y)
            
            # Feature importance = decrease in performance
            self.feature_importances_[feature_idx] = baseline_score - permuted_score
        
        # Normalize
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= np.sum(self.feature_importances_)
    
    def _score(self, X, y):
        predictions = self.predict(X)
        if self.task == 'classification':
            return np.mean(predictions == y)
        else:
            return -np.mean((predictions - y) ** 2)  # Negative MSE
    
    def predict(self, X):
        # Collect predictions from all trees
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        
        if self.task == 'classification':
            # Majority vote
            return np.array([stats.mode(tree_predictions[:, i])[0][0] 
                           for i in range(X.shape[0])])
        else:
            # Average for regression
            return np.mean(tree_predictions, axis=0)
    
    def predict_proba(self, X):
        """For classification tasks, return class probabilities"""
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")
        
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        n_samples = X.shape[0]
        n_classes = len(np.unique(tree_predictions))
        
        probabilities = np.zeros((n_samples, n_classes))
        
        for i in range(n_samples):
            class_counts = Counter(tree_predictions[:, i])
            for class_val, count in class_counts.items():
                probabilities[i, int(class_val)] = count / self.n_estimators
        
        return probabilities

# Demonstration
np.random.seed(42)

# Create synthetic dataset
X_demo, y_demo = make_classification(
    n_samples=500, n_features=10, n_informative=8, 
    n_redundant=2, n_clusters_per_class=1, random_state=42
)

# Split data
X_train_demo, X_test_demo, y_train_demo, y_test_demo = train_test_split(
    X_demo, y_demo, test_size=0.3, random_state=42, stratify=y_demo
)

# Train custom Random Forest
rf_custom = RandomForestFromScratch(
    n_estimators=50,
    max_depth=10,
    max_features='sqrt',
    task='classification',
    random_state=42
)

rf_custom.fit(X_train_demo, y_train_demo)

# Predictions
y_pred_custom = rf_custom.predict(X_test_demo)
y_proba_custom = rf_custom.predict_proba(X_test_demo)

# Compare with sklearn
rf_sklearn = RandomForestClassifier(
    n_estimators=50, max_depth=10, max_features='sqrt', random_state=42
)
rf_sklearn.fit(X_train_demo, y_train_demo)
y_pred_sklearn = rf_sklearn.predict(X_test_demo)

# Evaluate
accuracy_custom = np.mean(y_pred_custom == y_test_demo)
accuracy_sklearn = np.mean(y_pred_sklearn == y_test_demo)

print(f"From Scratch Random Forest Results:")
print(f"Custom RF Accuracy: {accuracy_custom:.3f}")
print(f"Sklearn RF Accuracy: {accuracy_sklearn:.3f}")
print(f"Difference: {abs(accuracy_custom - accuracy_sklearn):.3f}")

# Feature importance comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(range(len(rf_custom.feature_importances_)), rf_custom.feature_importances_)
plt.title('Custom Random Forest - Feature Importance')
plt.xlabel('Feature Index')
plt.ylabel('Importance')

plt.subplot(1, 2, 2)
plt.bar(range(len(rf_sklearn.feature_importances_)), rf_sklearn.feature_importances_)
plt.title('Sklearn Random Forest - Feature Importance')
plt.xlabel('Feature Index')
plt.ylabel('Importance')

plt.tight_layout()
plt.show()
```

## † Assumptions and Limitations

### Key Assumptions

1. **Independence of errors**: Assumes that prediction errors are independent
2. **Representative training data**: Training data should represent the population
3. **Stable relationships**: Feature-target relationships remain consistent over time
4. **No data leakage**: Future information is not used to predict past events
5. **Appropriate tree depth**: Trees should be complex enough to capture patterns but not overfit

### Limitations

1. **Can overfit with very deep trees**
   - **Solution**: Limit max_depth, increase min_samples_split/leaf

2. **Biased toward features with more levels**
   - **Solution**: Use balanced datasets, consider feature scaling

3. **Difficulty with linear relationships**
   - **Alternative**: Consider linear models or feature engineering

4. **Less interpretable than single decision tree**
   - **Solution**: Use feature importance plots, partial dependence plots

5. **Memory intensive for large forests**
   - **Solution**: Use smaller n_estimators, consider online learning alternatives

6. **Slower prediction than single tree**
   - **Assessment**: Trade-off between accuracy and prediction speed

### Comparison with Other Algorithms

| Algorithm | Interpretability | Overfitting Risk | Training Time | Prediction Speed | Performance |
|-----------|------------------|------------------|---------------|------------------|-------------|
| Decision Tree | High | High | Fast | Very Fast | Moderate |
| Random Forest | Medium | Low | Medium | Medium | High |
| Gradient Boosting | Low | Medium | Slow | Fast | Very High |
| SVM | Low | Medium | Slow | Fast | High |
| Logistic Regression | High | Low | Fast | Very Fast | Moderate |
| Neural Networks | Very Low | High | Very Slow | Medium | Very High |

**When to use Random Forest:**
-  Mixed data types (numerical and categorical)
-  Non-linear relationships
-  Need feature importance insights
-  Robust performance without much tuning
-  Medium-sized datasets

**When to avoid Random Forest:**
- L Very large datasets (consider XGBoost, LightGBM)
- L Real-time prediction requirements
- L Linear relationships dominate
- L High interpretability requirements
- L Memory constraints

## =° Interview Questions

??? question "How does Random Forest reduce overfitting compared to a single decision tree?"

    **Answer:** Random Forest reduces overfitting through several mechanisms:
    
    1. **Bootstrap Aggregating (Bagging)**: Each tree sees different training samples, reducing variance
    2. **Random Feature Selection**: Each split considers random features, decorrelating trees
    3. **Ensemble Averaging**: Averaging multiple models reduces overall variance
    4. **Bias-Variance Tradeoff**: Trades slight bias increase for large variance reduction
    
    **Mathematical intuition**: If individual trees have variance √≤, ensemble variance is √≤/B for independent trees, or ¡√≤ + (1-¡)√≤/B for correlated trees. Random features reduce correlation ¡.
    
    **Practical impact**: Single tree might achieve 85% accuracy with high variance, while Random Forest with 100 trees achieves 92% accuracy with much lower variance.

??? question "What is the difference between Random Forest and Gradient Boosting?"

    **Answer:** Key differences:
    
    | Aspect | Random Forest | Gradient Boosting |
    |--------|---------------|-------------------|
    | **Training** | Parallel (independent trees) | Sequential (corrects previous errors) |
    | **Tree depth** | Usually deep trees | Usually shallow trees |
    | **Overfitting** | Resistant | Prone to overfitting |
    | **Speed** | Faster training/prediction | Slower |
    | **Hyperparameters** | Fewer to tune | More complex tuning |
    | **Performance** | Good out-of-box | Often higher with tuning |
    
    **Use cases**:
    - **Random Forest**: When you want robust performance without much tuning
    - **Gradient Boosting**: When you can invest time in hyperparameter tuning for maximum performance

??? question "How do you handle categorical features in Random Forest?"

    **Answer:** Several approaches for categorical features:
    
    **1. Label Encoding** (simple but problematic):
    ```python
    # Creates artificial ordering
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    X_encoded = le.fit_transform(categorical_column)
    ```
    
    **2. One-Hot Encoding** (recommended):
    ```python
    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder(drop='first')  # Avoid multicollinearity
    X_encoded = ohe.fit_transform(categorical_column)
    ```
    
    **3. Target Encoding** (for high cardinality):
    ```python
    # Replace category with mean target value
    mean_values = df.groupby('category')['target'].mean()
    df['category_encoded'] = df['category'].map(mean_values)
    ```
    
    **4. Native handling** (some implementations):
    - R's randomForest package handles categorical features natively
    - Use Gini impurity for categorical splits
    
    **Best practice**: Use one-hot encoding for low cardinality (<10 categories), target encoding for high cardinality.

??? question "Explain the concept of Out-of-Bag (OOB) error and its advantages."

    **Answer:** Out-of-Bag error provides unbiased performance estimation without separate validation set:
    
    **Process**:
    1. Bootstrap sampling leaves ~37% of data "out-of-bag" for each tree
    2. For each sample, use trees that didn't see it during training
    3. Make prediction using only those trees
    4. Calculate error across all OOB predictions
    
    **Advantages**:
    - **No data splitting needed**: Use full dataset for training
    - **Unbiased estimate**: Similar to k-fold cross-validation
    - **Computational efficiency**: No separate validation runs
    - **Early stopping**: Monitor OOB error during training
    
    **Code example**:
    ```python
    rf = RandomForestClassifier(oob_score=True)
    rf.fit(X, y)
    print(f"OOB Score: {rf.oob_score_}")  # Unbiased performance estimate
    ```
    
    **Limitation**: OOB error might be pessimistic for small datasets.

??? question "How do you interpret feature importance in Random Forest and what are its limitations?"

    **Answer:** Random Forest provides two types of feature importance:
    
    **1. Gini Importance (Mean Decrease Impurity)**:
    - Measures average impurity decrease when splitting on each feature
    - **Fast to compute** but can be **biased toward high-cardinality features**
    
    **2. Permutation Importance (Mean Decrease Accuracy)**:
    - Measures accuracy drop when feature values are randomly permuted
    - **More reliable** but **computationally expensive**
    
    **Interpretation example**:
    ```python
    # Gini importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Permutation importance
    from sklearn.inspection import permutation_importance
    perm_importance = permutation_importance(rf, X, y, n_repeats=10)
    ```
    
    **Limitations**:
    - **Biased toward numerical and high-cardinality categorical features**
    - **Doesn't capture feature interactions**
    - **Can be misleading with correlated features**
    - **Different from causal importance**
    
    **Best practices**: Use both types, validate with domain knowledge, consider SHAP values for individual predictions.

??? question "How do hyperparameters affect Random Forest performance and how do you tune them?"

    **Answer:** Key hyperparameters and their effects:
    
    **Tree-level parameters**:
    - **n_estimators**: More trees í better performance, diminishing returns after ~100-500
    - **max_depth**: Controls overfitting vs underfitting
    - **min_samples_split/leaf**: Higher values prevent overfitting
    - **max_features**: 'sqrt' (classification), 'log2', or fraction of features
    
    **Bootstrap parameters**:
    - **bootstrap**: True for bagging, False for pasting
    - **oob_score**: Enable for validation without holdout set
    
    **Tuning strategies**:
    
    **1. Grid Search** (systematic but slow):
    ```python
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(rf, param_grid, cv=5)
    ```
    
    **2. Random Search** (more efficient):
    ```python
    from sklearn.model_selection import RandomizedSearchCV
    param_dist = {
        'n_estimators': [50, 100, 200, 500],
        'max_depth': [3, 5, 10, 20, None]
    }
    random_search = RandomizedSearchCV(rf, param_dist, n_iter=20, cv=5)
    ```
    
    **3. Progressive tuning**: Start with n_estimators, then tree parameters, then bootstrap parameters.

??? question "What is the computational complexity of Random Forest training and prediction?"

    **Answer:** Computational complexity analysis:
    
    **Training complexity**:
    - **Single tree**: O(n ◊ m ◊ log n) where n=samples, m=features
    - **Random Forest**: O(B ◊ n ◊ m ◊ log n) where B=number of trees
    - **With random features**: O(B ◊ n ◊ m ◊ log n) for classification
    - **Parallelizable**: Trees can be trained independently
    
    **Prediction complexity**:
    - **Single tree**: O(log n) for balanced tree
    - **Random Forest**: O(B ◊ log n)
    - **Space complexity**: O(B ◊ tree_size)
    
    **Memory usage**:
    ```python
    # Approximate memory for Random Forest
    memory_mb = (n_estimators * max_depth * n_features * 8) / (1024**2)
    ```
    
    **Optimization strategies**:
    - **Parallel training**: Use n_jobs=-1
    - **Feature subsampling**: Reduces computation per split
    - **Early stopping**: Monitor OOB score
    - **Tree pruning**: Remove unnecessary nodes post-training
    
    **Scalability considerations**:
    - Linear scaling with number of trees (parallelizable)
    - Memory can be limiting factor for very large forests
    - Consider ensemble methods like ExtraTrees for speed

??? question "How does Random Forest handle missing values and what are the best practices?"

    **Answer:** Random Forest cannot directly handle missing values, requiring preprocessing:
    
    **Preprocessing strategies**:
    
    **1. Simple imputation**:
    ```python
    from sklearn.impute import SimpleImputer
    # Mean/median for numerical, mode for categorical
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    ```
    
    **2. Advanced imputation**:
    ```python
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    # Uses other features to predict missing values
    imputer = IterativeImputer(random_state=42)
    X_imputed = imputer.fit_transform(X)
    ```
    
    **3. Missing indicator**:
    ```python
    from sklearn.impute import MissingIndicator
    # Add binary features indicating missingness
    indicator = MissingIndicator()
    missing_mask = indicator.fit_transform(X)
    X_augmented = np.hstack([X_imputed, missing_mask])
    ```
    
    **4. Multiple imputation**:
    - Create multiple imputed datasets
    - Train separate models
    - Combine predictions
    
    **Native approaches** (limited implementations):
    - **Surrogate splits**: Use alternative features when primary feature is missing
    - **Missing as category**: Treat missing as separate category for categorical features
    
    **Best practices**:
    - Analyze missingness patterns (MCAR, MAR, MNAR)
    - Consider domain knowledge for imputation strategy
    - Validate imputation quality
    - Report missing value handling in model documentation

??? question "Compare Random Forest with other ensemble methods like AdaBoost and XGBoost."

    **Answer:** Comprehensive comparison of ensemble methods:
    
    | Aspect | Random Forest | AdaBoost | XGBoost |
    |--------|---------------|----------|---------|
    | **Algorithm type** | Bagging | Boosting | Gradient Boosting |
    | **Tree training** | Parallel | Sequential | Sequential |
    | **Error focus** | Reduces variance | Reduces bias | Reduces both |
    | **Overfitting** | Resistant | Can overfit | Regularized |
    | **Speed** | Fast | Medium | Medium-Fast |
    | **Hyperparameters** | Few, robust | Few | Many, needs tuning |
    | **Performance** | Good baseline | Good for weak learners | Often best |
    | **Interpretability** | Medium | Low | Low |
    
    **When to use each**:
    
    **Random Forest**:
    -  Quick baseline model
    -  Mixed data types
    -  Interpretability needed
    -  Robust performance without tuning
    
    **AdaBoost**:
    -  Weak learners available
    -  Binary classification
    -  Less prone to outliers than other boosting
    - L Sensitive to noise and outliers
    
    **XGBoost/LightGBM**:
    -  Maximum predictive performance
    -  Kaggle competitions
    -  Large datasets
    -  Advanced regularization needed
    - L Requires careful hyperparameter tuning
    
    **Performance hierarchy** (generally): XGBoost > Random Forest > AdaBoost > Single Tree
    
    **Complexity hierarchy**: XGBoost > AdaBoost > Random Forest > Single Tree

??? question "How do you handle class imbalance in Random Forest?"

    **Answer:** Several strategies for handling imbalanced datasets:
    
    **1. Built-in class balancing**:
    ```python
    rf = RandomForestClassifier(
        class_weight='balanced',  # Automatically adjusts weights
        random_state=42
    )
    # Or custom weights
    rf = RandomForestClassifier(
        class_weight={0: 1, 1: 10},  # More weight to minority class
        random_state=42
    )
    ```
    
    **2. Sampling strategies**:
    ```python
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    
    # Oversampling minority class
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Undersampling majority class
    undersampler = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = undersampler.fit_resample(X, y)
    ```
    
    **3. Stratified sampling**:
    ```python
    # Ensure each bootstrap sample maintains class proportions
    rf = RandomForestClassifier(
        bootstrap=True,
        # Manual stratified sampling in custom implementation
        random_state=42
    )
    ```
    
    **4. Threshold tuning**:
    ```python
    # Adjust decision threshold based on precision-recall tradeoff
    from sklearn.metrics import precision_recall_curve
    
    y_proba = rf.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    
    # Find optimal threshold
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    ```
    
    **5. Evaluation metrics**:
    - Use precision, recall, F1-score instead of accuracy
    - ROC-AUC and Precision-Recall curves
    - Stratified cross-validation
    
    **Best approach**: Combine multiple strategies and validate with appropriate metrics.

## >‡ Examples

### Real-world Example: Credit Risk Assessment

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Generate synthetic credit risk dataset
np.random.seed(42)
n_samples = 10000

# Create base features
X_base, y = make_classification(
    n_samples=n_samples, 
    n_features=15, 
    n_informative=12,
    n_redundant=3,
    n_clusters_per_class=1,
    weights=[0.85, 0.15],  # Imbalanced dataset (15% default rate)
    flip_y=0.02,  # Small amount of label noise
    random_state=42
)

# Create meaningful feature names
feature_names = [
    'income', 'debt_to_income', 'credit_score', 'employment_years',
    'loan_amount', 'loan_to_value', 'payment_history', 'age',
    'education_level', 'num_accounts', 'total_credit_limit',
    'credit_utilization', 'num_inquiries', 'delinquencies', 'bankruptcies'
]

# Create DataFrame
df_credit = pd.DataFrame(X_base, columns=feature_names)
df_credit['default'] = y

# Add some realistic transformations
df_credit['income'] = np.exp(df_credit['income']) * 1000  # Log-normal income
df_credit['credit_score'] = (df_credit['credit_score'] * 100 + 700).clip(300, 850)
df_credit['age'] = (df_credit['age'] * 15 + 35).clip(18, 80)

print("Credit Risk Dataset Overview:")
print(f"Dataset shape: {df_credit.shape}")
print(f"Default rate: {df_credit['default'].mean():.3f}")
print(f"\nFeature statistics:")
print(df_credit.describe())

# Prepare features and target
X = df_credit[feature_names].values
y = df_credit['default'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Train default rate: {y_train.mean():.3f}")
print(f"Test default rate: {y_test.mean():.3f}")

# Train Random Forest with class balancing
rf_credit = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    class_weight='balanced',  # Handle class imbalance
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

rf_credit.fit(X_train, y_train)

# Predictions
y_pred = rf_credit.predict(X_test)
y_pred_proba = rf_credit.predict_proba(X_test)[:, 1]

print(f"\nModel Performance:")
print(f"Out-of-bag Score: {rf_credit.oob_score_:.3f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_credit.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Feature importance
sns.barplot(data=feature_importance.head(10), y='feature', x='importance', ax=axes[0, 0])
axes[0, 0].set_title('Top 10 Feature Importances')
axes[0, 0].set_xlabel('Importance Score')

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
axes[0, 1].set_title('Confusion Matrix')
axes[0, 1].set_ylabel('True Label')
axes[0, 1].set_xlabel('Predicted Label')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[1, 0].set_xlim([0.0, 1.0])
axes[1, 0].set_ylim([0.0, 1.05])
axes[1, 0].set_xlabel('False Positive Rate')
axes[1, 0].set_ylabel('True Positive Rate')
axes[1, 0].set_title('ROC Curve')
axes[1, 0].legend(loc="lower right")

# Prediction distribution
axes[1, 1].hist(y_pred_proba[y_test == 0], bins=30, alpha=0.7, label='No Default', density=True)
axes[1, 1].hist(y_pred_proba[y_test == 1], bins=30, alpha=0.7, label='Default', density=True)
axes[1, 1].set_xlabel('Predicted Probability of Default')
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_title('Prediction Distribution by True Class')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# Business impact analysis
def calculate_profit(y_true, y_pred_proba, threshold=0.5, 
                    profit_approved_good=200, loss_approved_bad=-1000,
                    profit_rejected_good=0, profit_rejected_bad=0):
    """Calculate business profit based on loan decisions"""
    
    decisions = (y_pred_proba >= threshold).astype(int)  # 1: Approve, 0: Reject
    
    # Calculate profit for each scenario
    approved_good = ((decisions == 1) & (y_true == 0)).sum() * profit_approved_good
    approved_bad = ((decisions == 1) & (y_true == 1)).sum() * loss_approved_bad
    rejected_good = ((decisions == 0) & (y_true == 0)).sum() * profit_rejected_good
    rejected_bad = ((decisions == 0) & (y_true == 1)).sum() * profit_rejected_bad
    
    total_profit = approved_good + approved_bad + rejected_good + rejected_bad
    
    return {
        'total_profit': total_profit,
        'approved_good': approved_good,
        'approved_bad': approved_bad,
        'rejected_good': rejected_good,
        'rejected_bad': rejected_bad,
        'approval_rate': decisions.mean(),
        'threshold': threshold
    }

# Analyze profit across different thresholds
thresholds = np.linspace(0.1, 0.9, 17)
profits = []

for threshold in thresholds:
    profit_info = calculate_profit(y_test, y_pred_proba, threshold)
    profits.append(profit_info)

profits_df = pd.DataFrame(profits)

# Find optimal threshold
optimal_idx = profits_df['total_profit'].idxmax()
optimal_threshold = profits_df.loc[optimal_idx, 'threshold']
optimal_profit = profits_df.loc[optimal_idx, 'total_profit']

print(f"\nBusiness Impact Analysis:")
print(f"Optimal threshold: {optimal_threshold:.2f}")
print(f"Maximum profit: ${optimal_profit:,.0f}")
print(f"Approval rate at optimal threshold: {profits_df.loc[optimal_idx, 'approval_rate']:.1%}")

# Plot profit vs threshold
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(profits_df['threshold'], profits_df['total_profit'], 'bo-', linewidth=2)
plt.axvline(optimal_threshold, color='red', linestyle='--', 
            label=f'Optimal: {optimal_threshold:.2f}')
plt.xlabel('Decision Threshold')
plt.ylabel('Total Profit ($)')
plt.title('Profit vs Decision Threshold')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(profits_df['threshold'], profits_df['approval_rate'], 'go-', linewidth=2)
plt.axvline(optimal_threshold, color='red', linestyle='--', 
            label=f'Optimal: {optimal_threshold:.2f}')
plt.xlabel('Decision Threshold')
plt.ylabel('Loan Approval Rate')
plt.title('Approval Rate vs Decision Threshold')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

### House Price Prediction Example

```python
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import plot_partial_dependence

# Generate synthetic house price dataset
np.random.seed(42)
n_samples = 5000

X_house, y_house = make_regression(
    n_samples=n_samples,
    n_features=12,
    n_informative=10,
    noise=0.1,
    random_state=42
)

# Create meaningful feature names
house_features = [
    'square_feet', 'bedrooms', 'bathrooms', 'age_years',
    'lot_size', 'garage_spaces', 'school_rating', 'crime_rate',
    'distance_downtown', 'property_tax', 'neighborhood_income', 'walk_score'
]

# Transform features to realistic scales
X_house[:, 0] = (X_house[:, 0] * 800 + 2000).clip(800, 5000)  # Square feet
X_house[:, 1] = (X_house[:, 1] * 2 + 3).clip(1, 6).round()  # Bedrooms
X_house[:, 2] = (X_house[:, 2] * 1.5 + 2).clip(1, 4).round()  # Bathrooms
X_house[:, 3] = (X_house[:, 3] * 20 + 25).clip(0, 100)  # Age
X_house[:, 4] = (X_house[:, 4] * 0.3 + 0.25).clip(0.1, 2.0)  # Lot size (acres)

# Transform target to realistic house prices ($)
y_house = (y_house * 100000 + 400000).clip(150000, 1500000)

# Create DataFrame
df_houses = pd.DataFrame(X_house, columns=house_features)
df_houses['price'] = y_house

print("House Price Dataset Overview:")
print(f"Dataset shape: {df_houses.shape}")
print(f"Price statistics:")
print(df_houses['price'].describe())

# Split data
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_house, y_house, test_size=0.2, random_state=42
)

# Train Random Forest Regressor
rf_houses = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

rf_houses.fit(X_train_h, y_train_h)

# Predictions
y_pred_h = rf_houses.predict(X_test_h)

# Evaluate
mse = mean_squared_error(y_test_h, y_pred_h)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_h, y_pred_h)
r2 = r2_score(y_test_h, y_pred_h)

print(f"\nHouse Price Prediction Results:")
print(f"R≤ Score: {r2:.3f}")
print(f"RMSE: ${rmse:,.0f}")
print(f"MAE: ${mae:,.0f}")
print(f"MAPE: {np.mean(np.abs((y_test_h - y_pred_h) / y_test_h)) * 100:.1f}%")
print(f"Out-of-bag Score: {rf_houses.oob_score_:.3f}")

# Feature importance for house prices
house_importance = pd.DataFrame({
    'feature': house_features,
    'importance': rf_houses.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nFeature Importance for House Prices:")
print(house_importance)

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Feature importance
sns.barplot(data=house_importance, y='feature', x='importance', ax=axes[0, 0])
axes[0, 0].set_title('Feature Importance - House Prices')
axes[0, 0].set_xlabel('Importance Score')

# Actual vs Predicted
axes[0, 1].scatter(y_test_h, y_pred_h, alpha=0.6)
axes[0, 1].plot([y_test_h.min(), y_test_h.max()], 
                [y_test_h.min(), y_test_h.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('Actual Price ($)')
axes[0, 1].set_ylabel('Predicted Price ($)')
axes[0, 1].set_title(f'Actual vs Predicted (R≤ = {r2:.3f})')

# Residuals
residuals = y_test_h - y_pred_h
axes[1, 0].scatter(y_pred_h, residuals, alpha=0.6)
axes[1, 0].axhline(y=0, color='r', linestyle='--')
axes[1, 0].set_xlabel('Predicted Price ($)')
axes[1, 0].set_ylabel('Residuals ($)')
axes[1, 0].set_title('Residual Plot')

# Residual distribution
axes[1, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('Residuals ($)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Residual Distribution')
axes[1, 1].axvline(x=0, color='red', linestyle='--')

plt.tight_layout()
plt.show()

# Partial dependence plots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

top_features = house_importance.head(6)['feature'].tolist()
feature_indices = [house_features.index(f) for f in top_features]

for i, (feature_idx, feature_name) in enumerate(zip(feature_indices, top_features)):
    plot_partial_dependence(
        rf_houses, X_train_h, [feature_idx], 
        ax=axes[i], feature_names=[feature_name]
    )
    axes[i].set_title(f'Partial Dependence: {feature_name}')

plt.tight_layout()
plt.show()

# Model interpretation: Feature interactions
print(f"\nModel Interpretation:")
print(f"The model shows that {house_importance.iloc[0]['feature']} is the most important factor,")
print(f"explaining {house_importance.iloc[0]['importance']:.1%} of the house price variation.")

# Prediction confidence intervals (approximate)
def prediction_intervals(rf, X, confidence=0.95):
    """Calculate prediction intervals using tree-level predictions"""
    # Get predictions from all trees
    tree_predictions = np.array([tree.predict(X) for tree in rf.estimators_])
    
    # Calculate percentiles
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(tree_predictions, lower_percentile, axis=0)
    upper_bound = np.percentile(tree_predictions, upper_percentile, axis=0)
    
    return lower_bound, upper_bound

# Calculate 95% prediction intervals for test set
lower_bounds, upper_bounds = prediction_intervals(rf_houses, X_test_h)

# Coverage analysis
coverage = ((y_test_h >= lower_bounds) & (y_test_h <= upper_bounds)).mean()
avg_interval_width = np.mean(upper_bounds - lower_bounds)

print(f"\nPrediction Intervals (95% confidence):")
print(f"Coverage: {coverage:.1%}")
print(f"Average interval width: ${avg_interval_width:,.0f}")

# Show some examples
examples = pd.DataFrame({
    'Actual': y_test_h[:10],
    'Predicted': y_pred_h[:10],
    'Lower_95%': lower_bounds[:10],
    'Upper_95%': upper_bounds[:10]
})
examples['Within_Interval'] = (
    (examples['Actual'] >= examples['Lower_95%']) & 
    (examples['Actual'] <= examples['Upper_95%'])
)

print(f"\nSample Predictions with Intervals:")
print(examples.round(0))
```

## =⁄ References

- **Original Paper:**
  - [Random Forests](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf) by Leo Breiman (2001)
  - [Bagging Predictors](https://www.stat.berkeley.edu/~breiman/bagging.pdf) by Leo Breiman (1996)

- **Books:**
  - [The Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/) by Hastie, Tibshirani, and Friedman - Chapter 15
  - [Introduction to Statistical Learning](https://www.statlearning.com/) by James, Witten, Hastie, and Tibshirani - Chapter 8
  - [Hands-On Machine Learning](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) by AurÈlien GÈron - Chapter 7

- **Documentation:**
  - [Scikit-learn Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)
  - [Scikit-learn RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
  - [Scikit-learn RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

- **Tutorials and Guides:**
  - [Random Forest Algorithm Guide](https://towardsdatascience.com/understanding-random-forest-58381e0602d2)
  - [Random Forest Feature Importance](https://explained.ai/rf-importance/)
  - [Hyperparameter Tuning for Random Forest](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74)

- **Advanced Topics:**
  - [Extremely Randomized Trees](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)
  - [Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
  - [Random Forest Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection-using-selectfrommodel)

- **Research Papers:**
  - Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely randomized trees
  - Strobl, C., Boulesteix, A. L., Kneib, T., Augustin, T., & Zeileis, A. (2008). Conditional variable importance for random forests
  - Louppe, G. (2014). Understanding Random Forests: From Theory to Practice

- **Online Courses:**
  - [Machine Learning Course - Stanford CS229](http://cs229.stanford.edu/)
  - [Random Forest in Machine Learning - Coursera](https://www.coursera.org/learn/machine-learning)
  - [Applied Machine Learning - edX](https://www.edx.org/course/machine-learning)

- **Implementations:**
  - [scikit-learn](https://scikit-learn.org/stable/)
  - [R randomForest package](https://cran.r-project.org/web/packages/randomForest/index.html)
  - [XGBoost](https://xgboost.readthedocs.io/) (gradient boosting alternative)