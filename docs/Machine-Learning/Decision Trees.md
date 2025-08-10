---
title: Decision Trees
description: Comprehensive guide to Decision Trees with mathematical intuition, implementations, and interview questions.
comments: true
---

# 📘 Decision Trees

Decision Trees are versatile, interpretable machine learning algorithms that make predictions by learning simple decision rules inferred from data features, creating a tree-like model of decisions.

**Resources:** [Scikit-learn Decision Trees](https://scikit-learn.org/stable/modules/tree.html) | [Elements of Statistical Learning - Chapter 9](https://web.stanford.edu/~hastie/ElemStatLearn/)

## ✍️ Summary

Decision Trees are supervised learning algorithms that can be used for both classification and regression tasks. They work by recursively splitting the data into subsets based on feature values that best separate the target classes or minimize prediction error.

**Key characteristics:**
- **Interpretability**: Easy to understand and visualize decision paths
- **Non-parametric**: No assumptions about data distribution
- **Feature selection**: Automatically identifies important features
- **Handles mixed data**: Works with both numerical and categorical features
- **Non-linear relationships**: Can capture complex patterns

**Applications:**
- Medical diagnosis systems
- Credit approval decisions  
- Customer segmentation
- Feature selection
- Rule extraction
- Fraud detection

**Types:**
- **Classification Trees**: Predict discrete class labels
- **Regression Trees**: Predict continuous values

## 🧠 Intuition

### How Decision Trees Work

A Decision Tree learns by asking a series of binary questions about the features. Each internal node represents a test on a feature, each branch represents an outcome of the test, and each leaf node represents a class prediction or numerical value.

### Mathematical Foundation

#### 1. Splitting Criteria

**For Classification (Gini Impurity):**
$$\text{Gini}(S) = 1 - \sum_{i=1}^{c} p_i^2$$

Where:
- $S$ is the set of examples
- $c$ is the number of classes
- $p_i$ is the proportion of examples belonging to class $i$

**For Classification (Entropy):**
$$\text{Entropy}(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)$$

**For Regression (Mean Squared Error):**
$$\text{MSE}(S) = \frac{1}{|S|} \sum_{i=1}^{|S|} (y_i - \bar{y})^2$$

Where $\bar{y}$ is the mean of target values in set $S$.

#### 2. Information Gain

The algorithm selects the feature that maximizes information gain:

$$\text{InfoGain}(S, A) = \text{Impurity}(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \text{Impurity}(S_v)$$

Where:
- $A$ is the attribute (feature)
- $S_v$ is the subset of $S$ where attribute $A$ has value $v$

#### 3. Stopping Criteria

The tree stops growing when:
- All examples have the same class (pure node)
- No more features to split on
- Maximum depth reached
- Minimum samples per leaf reached
- Information gain below threshold

### Algorithm Steps

1. **Start with root node** containing all training data
2. **For each node**:
   - Calculate impurity measure
   - Find best feature and threshold to split
   - Create child nodes
3. **Recursively repeat** for child nodes
4. **Stop** when stopping criteria met
5. **Assign prediction** to leaf nodes

## 🔢 Implementation using Libraries

### Using Scikit-learn

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_iris, load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import export_text, plot_tree
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import seaborn as sns

# Classification Example
# Generate sample data
X, y = make_classification(
    n_samples=1000,
    n_features=4,
    n_informative=3,
    n_redundant=1,
    n_clusters_per_class=1,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train classifier
clf = DecisionTreeClassifier(
    criterion='gini',           # or 'entropy'
    max_depth=5,               # prevent overfitting
    min_samples_split=20,      # minimum samples to split
    min_samples_leaf=10,       # minimum samples in leaf
    random_state=42
)

clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize tree structure
plt.figure(figsize=(20, 10))
plot_tree(clf, 
          feature_names=[f'Feature_{i}' for i in range(X.shape[1])],
          class_names=['Class_0', 'Class_1'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Decision Tree Structure")
plt.show()

# Feature importance
feature_importance = clf.feature_importances_
features = [f'Feature_{i}' for i in range(X.shape[1])]

plt.figure(figsize=(10, 6))
plt.bar(features, feature_importance)
plt.title('Feature Importance')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.show()

print("\nFeature Importances:")
for feature, importance in zip(features, feature_importance):
    print(f"{feature}: {importance:.3f}")

# Real-world example with Iris dataset
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris, y_iris, test_size=0.3, random_state=42
)

# Train classifier
iris_clf = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,
    random_state=42
)
iris_clf.fit(X_train_iris, y_train_iris)

# Predictions
y_pred_iris = iris_clf.predict(X_test_iris)
iris_accuracy = accuracy_score(y_test_iris, y_pred_iris)
print(f"\nIris Dataset Accuracy: {iris_accuracy:.3f}")

# Print decision tree rules
tree_rules = export_text(iris_clf, 
                        feature_names=iris.feature_names)
print("\nDecision Tree Rules:")
print(tree_rules)

# Regression Example
from sklearn.datasets import make_regression

# Generate regression data
X_reg, y_reg = make_regression(
    n_samples=500,
    n_features=1,
    noise=10,
    random_state=42
)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Train regression tree
reg_tree = DecisionTreeRegressor(
    max_depth=3,
    min_samples_split=20,
    random_state=42
)
reg_tree.fit(X_train_reg, y_train_reg)

# Predictions
y_pred_reg = reg_tree.predict(X_test_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
print(f"\nRegression MSE: {mse:.3f}")

# Visualize regression tree predictions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_test_reg, y_test_reg, alpha=0.6, label='Actual')
plt.scatter(X_test_reg, y_pred_reg, alpha=0.6, label='Predicted')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Decision Tree Regression')
plt.legend()

# Show step-wise predictions
X_plot = np.linspace(X_reg.min(), X_reg.max(), 300).reshape(-1, 1)
y_plot = reg_tree.predict(X_plot)

plt.subplot(1, 2, 2)
plt.scatter(X_train_reg, y_train_reg, alpha=0.6, label='Training Data')
plt.plot(X_plot, y_plot, color='red', linewidth=2, label='Tree Prediction')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Decision Tree Regression Function')
plt.legend()
plt.tight_layout()
plt.show()
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}

# Grid search
grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Use best model
best_clf = grid_search.best_estimator_
best_pred = best_clf.predict(X_test)
print("Best model accuracy:", accuracy_score(y_test, best_pred))
```

## ⚙️ From Scratch Implementation

```python
import numpy as np
from collections import Counter

class Node:
    """Represents a node in the decision tree"""
    
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Feature index to split on
        self.threshold = threshold  # Threshold value for splitting
        self.left = left           # Left child node
        self.right = right         # Right child node
        self.value = value         # Value if leaf node (class or regression value)

class DecisionTreeFromScratch:
    """Decision Tree implementation from scratch"""
    
    def __init__(self, max_depth=3, min_samples_split=2, criterion='gini', tree_type='classification'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.tree_type = tree_type
        self.root = None
        
    def fit(self, X, y):
        """Train the decision tree"""
        self.n_features = X.shape[1]
        self.root = self._grow_tree(X, y, depth=0)
        
    def predict(self, X):
        """Make predictions on test data"""
        return np.array([self._predict_single(sample, self.root) for sample in X])
    
    def _grow_tree(self, X, y, depth):
        """Recursively grow the tree"""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_classes == 1):
            leaf_value = self._most_common_class(y) if self.tree_type == 'classification' else np.mean(y)
            return Node(value=leaf_value)
        
        # Find best split
        best_feature, best_threshold = self._best_split(X, y)
        
        # Create child splits
        left_indices, right_indices = self._split_data(X[:, best_feature], best_threshold)
        left_child = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        
        return Node(feature=best_feature, threshold=best_threshold, 
                   left=left_child, right=right_child)
    
    def _best_split(self, X, y):
        """Find the best feature and threshold to split on"""
        best_gain = -1
        best_feature, best_threshold = None, None
        
        for feature_idx in range(self.n_features):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            for threshold in thresholds:
                gain = self._information_gain(y, feature_values, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
                    
        return best_feature, best_threshold
    
    def _information_gain(self, y, feature_values, threshold):
        """Calculate information gain for a split"""
        # Parent impurity
        parent_impurity = self._calculate_impurity(y)
        
        # Create splits
        left_indices, right_indices = self._split_data(feature_values, threshold)
        
        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0
        
        # Calculate weighted impurity of children
        n = len(y)
        n_left, n_right = len(left_indices), len(right_indices)
        impurity_left = self._calculate_impurity(y[left_indices])
        impurity_right = self._calculate_impurity(y[right_indices])
        
        child_impurity = (n_left / n) * impurity_left + (n_right / n) * impurity_right
        
        # Information gain
        return parent_impurity - child_impurity
    
    def _calculate_impurity(self, y):
        """Calculate impurity based on criterion"""
        if self.tree_type == 'classification':
            if self.criterion == 'gini':
                return self._gini_impurity(y)
            elif self.criterion == 'entropy':
                return self._entropy(y)
        else:  # regression
            return self._mse(y)
    
    def _gini_impurity(self, y):
        """Calculate Gini impurity"""
        if len(y) == 0:
            return 0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)
    
    def _entropy(self, y):
        """Calculate entropy"""
        if len(y) == 0:
            return 0
            
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-8))  # Add small value to avoid log(0)
    
    def _mse(self, y):
        """Calculate Mean Squared Error"""
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)
    
    def _split_data(self, feature_values, threshold):
        """Split data based on threshold"""
        left_indices = np.where(feature_values <= threshold)[0]
        right_indices = np.where(feature_values > threshold)[0]
        return left_indices, right_indices
    
    def _most_common_class(self, y):
        """Return the most common class"""
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def _predict_single(self, sample, node):
        """Predict a single sample"""
        if node.value is not None:  # Leaf node
            return node.value
        
        if sample[node.feature] <= node.threshold:
            return self._predict_single(sample, node.left)
        else:
            return self._predict_single(sample, node.right)
    
    def print_tree(self, node=None, depth=0, side='root'):
        """Print tree structure"""
        if node is None:
            node = self.root
            
        if node.value is not None:
            print(f"{'  ' * depth}{side}: Predict {node.value}")
        else:
            print(f"{'  ' * depth}{side}: Feature_{node.feature} <= {node.threshold:.2f}")
            self.print_tree(node.left, depth + 1, 'left')
            self.print_tree(node.right, depth + 1, 'right')

# Example usage of from-scratch implementation
print("=" * 50)
print("FROM SCRATCH IMPLEMENTATION")
print("=" * 50)

# Generate sample data
np.random.seed(42)
X_sample = np.random.randn(200, 4)
y_sample = (X_sample[:, 0] + X_sample[:, 1] > 0).astype(int)

# Split data
X_train_scratch, X_test_scratch, y_train_scratch, y_test_scratch = train_test_split(
    X_sample, y_sample, test_size=0.2, random_state=42
)

# Train custom decision tree
custom_tree = DecisionTreeFromScratch(
    max_depth=3,
    min_samples_split=10,
    criterion='gini',
    tree_type='classification'
)

custom_tree.fit(X_train_scratch, y_train_scratch)

# Make predictions
y_pred_scratch = custom_tree.predict(X_test_scratch)
custom_accuracy = np.mean(y_pred_scratch == y_test_scratch)

print(f"Custom Decision Tree Accuracy: {custom_accuracy:.3f}")
print("\nTree Structure:")
custom_tree.print_tree()

# Compare with sklearn
sklearn_tree = DecisionTreeClassifier(max_depth=3, min_samples_split=10, random_state=42)
sklearn_tree.fit(X_train_scratch, y_train_scratch)
sklearn_pred = sklearn_tree.predict(X_test_scratch)
sklearn_accuracy = accuracy_score(y_test_scratch, sklearn_pred)

print(f"Scikit-learn Decision Tree Accuracy: {sklearn_accuracy:.3f}")
print(f"Difference: {abs(custom_accuracy - sklearn_accuracy):.3f}")
```

## ⚠️ Assumptions and Limitations

### Assumptions
1. **Feature relevance**: Assumes that features contain information relevant to the target
2. **Finite feature space**: Works with discrete or discretized continuous features
3. **Independent samples**: Training samples should be independent
4. **Consistent labeling**: No contradictory examples (same features, different labels)

### Limitations

1. **Overfitting**: 
   - Prone to creating overly complex trees that memorize training data
   - **Solution**: Pruning, max_depth, min_samples constraints

2. **Instability**: 
   - Small changes in data can result in very different trees
   - **Solution**: Ensemble methods (Random Forest, Gradient Boosting)

3. **Bias towards features with many levels**: 
   - Favor features with more possible split points
   - **Solution**: Use conditional inference trees or random feature selection

4. **Difficulty with linear relationships**: 
   - Inefficient at modeling linear relationships
   - **Solution**: Combine with linear models or use ensemble methods

5. **Imbalanced data issues**: 
   - May be biased towards majority class
   - **Solution**: Class weighting, resampling techniques

### Comparison with Other Algorithms

| Algorithm | Interpretability | Overfitting Risk | Performance | Training Speed |
|-----------|-----------------|------------------|-------------|----------------|
| Decision Trees | Very High | High | Medium | Fast |
| Random Forest | Medium | Low | High | Medium |
| SVM | Low | Medium | High | Slow |
| Logistic Regression | High | Low | Medium | Fast |
| Neural Networks | Very Low | High | Very High | Very Slow |

**When to use Decision Trees:**
- ✅ When interpretability is crucial
- ✅ Mixed data types (numerical + categorical)
- ✅ Feature selection is needed
- ✅ Non-linear relationships exist
- ✅ Quick prototyping needed

**When to avoid:**
- ❌ When accuracy is paramount (use ensembles instead)
- ❌ With very noisy data
- ❌ When dataset is very small
- ❌ Linear relationships dominate

## 💡 Interview Questions

??? question "1. What is the difference between Gini impurity and Entropy? When would you use each?"

    **Gini Impurity:**
    
    - Formula: $\text{Gini} = 1 - \sum_{i=1}^{c} p_i^2$
    - Range: [0, 0.5] for binary classification
    - Computationally faster (no logarithms)
    - Tends to isolate most frequent class
    - Default in scikit-learn

    **Entropy:**
    
    - Formula: $\text{Entropy} = -\sum_{i=1}^{c} p_i \log_2(p_i)$
    - Range: [0, 1] for binary classification  
    - More computationally expensive
    - Tends to create more balanced splits
    - Better theoretical foundation in information theory

    **When to use:**
    
    - **Gini**: When computational speed is important, when you want to isolate the most frequent class
    - **Entropy**: When you need more balanced trees, when theoretical interpretability matters

    Both typically produce similar trees in practice.

??? question "2. How do you prevent overfitting in Decision Trees?"

    **Pre-pruning (Early Stopping):**
    
    1. **max_depth**: Limit tree depth
    2. **min_samples_split**: Minimum samples required to split a node
    3. **min_samples_leaf**: Minimum samples required in a leaf node
    4. **max_features**: Limit features considered for splitting
    5. **min_impurity_decrease**: Minimum impurity decrease required for split

    **Post-pruning:**
    
    1. **Cost Complexity Pruning**: Remove branches that don't improve performance
    2. **Reduced Error Pruning**: Use validation set to prune nodes
    3. **Rule Post-pruning**: Convert tree to rules, then prune rules

    **Other techniques:**
    
    - **Cross-validation**: Use CV to select hyperparameters
    - **Ensemble methods**: Random Forest, Gradient Boosting
    - **Feature selection**: Remove irrelevant features
    - **Data augmentation**: Increase training data size

    **Example code:**
    ```python
    # Pre-pruning
    tree = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt'
    )

    # Post-pruning (cost complexity)
    path = tree.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas
    ```

??? question "3. Explain the algorithm for building a Decision Tree step by step."

    **Decision Tree Construction Algorithm:**

    1. **Initialize**: Start with root node containing all training samples

    2. **For each node**:
       - Check stopping criteria:
         - All samples have same class (pure node)
         - Maximum depth reached
         - Minimum samples threshold reached
         - No information gain possible
       
    3. **Find best split**:
       - For each feature:
         - For each possible threshold:
           - Calculate information gain
       - Select feature and threshold with highest gain

    4. **Split data**: Create left and right child nodes based on best split

    5. **Recursive expansion**: Repeat process for each child node

    6. **Assign predictions**: For leaf nodes, assign most common class (classification) or mean value (regression)

    **Pseudocode:**
    ```python
    def build_tree(data, labels, depth):
        if stopping_criteria_met:
            return create_leaf_node(labels)
        
        best_feature, best_threshold = find_best_split(data, labels)
        
        left_data, left_labels = split_left(data, labels, best_feature, best_threshold)
        right_data, right_labels = split_right(data, labels, best_feature, best_threshold)
        
        left_child = build_tree(left_data, left_labels, depth+1)
        right_child = build_tree(right_data, right_labels, depth+1)
        
        return create_internal_node(best_feature, best_threshold, left_child, right_child)
    ```

??? question "4. How do Decision Trees handle categorical features?"

    **Methods for handling categorical features:**

    **1. Binary encoding for each category:**
    ```python
    # For feature "Color" with values [Red, Blue, Green]
    # Create binary splits: "Is Red?", "Is Blue?", "Is Green?"
    ```

    **2. Subset-based splits:**
    - Consider all possible subsets of categories
    - Computationally expensive: 2^(k-1) - 1 possible splits for k categories
    - Used in algorithms like C4.5

    **3. Ordinal encoding:**
    - Assign numerical values to categories
    - Only appropriate if natural ordering exists
    - Example: [Low, Medium, High] → [1, 2, 3]

    **4. One-hot encoding (preprocessing):**
    - Convert each category to binary feature
    - Most common approach in scikit-learn

    **Implementation considerations:**
    - **Scikit-learn**: Requires preprocessing (one-hot encoding)
    - **XGBoost**: Native support for categorical features
    - **LightGBM**: Native categorical support

    **Example:**
    ```python
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer

    # Preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(), categorical_features),
        ('num', 'passthrough', numerical_features)
    ])

    X_processed = preprocessor.fit_transform(X)
    tree.fit(X_processed, y)
    ```

??? question "5. What are the advantages and disadvantages of Decision Trees compared to other algorithms?"

    **Advantages:**
    
    1. **High Interpretability**: Easy to understand and explain decisions
    2. **No assumptions**: No statistical assumptions about data distribution
    3. **Handles mixed data**: Both numerical and categorical features
    4. **Automatic feature selection**: Identifies important features
    5. **Non-linear relationships**: Captures complex patterns
    6. **Fast prediction**: O(log n) prediction time
    7. **Robust to outliers**: Splits are based on order, not actual values
    8. **No preprocessing**: No need for feature scaling or normalization

    **Disadvantages:**
    
    1. **Overfitting**: Tends to create overly complex models
    2. **Instability**: Small data changes cause different trees
    3. **Bias**: Favors features with more levels
    4. **Poor extrapolation**: Cannot predict beyond training data range
    5. **Limited expressiveness**: Axis-parallel splits only
    6. **Imbalanced data**: Biased towards majority class
    7. **Greedy algorithm**: May not find globally optimal tree

    **Comparison table:**
    
    | Aspect | Decision Trees | Random Forest | SVM | Logistic Regression |
    |--------|---------------|---------------|-----|-------------------|
    | Interpretability | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐ |
    | Accuracy | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
    | Speed | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
    | Overfitting Risk | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

??? question "6. How do you handle missing values in Decision Trees?"

    **Approaches for missing values:**

    **1. Preprocessing approaches:**
    ```python
    # Remove rows with missing values
    X_clean = X.dropna()

    # Imputation
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')  # or 'mean', 'most_frequent'
    X_imputed = imputer.fit_transform(X)
    ```

    **2. Algorithm-level handling:**
    - **Surrogate splits**: Use correlated features when primary feature is missing
    - **Probabilistic splits**: Send sample down both branches with appropriate probabilities
    - **Missing as category**: Treat missing as separate category

    **3. Advanced techniques:**
    ```python
    # XGBoost handles missing values natively
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.fit(X_with_missing, y)  # No preprocessing needed

    # Custom handling in decision tree
    class MissingValueTree:
        def split_with_missing(self, X, feature, threshold):
            # Handle missing values by going to majority direction
            mask = ~np.isnan(X[:, feature])
            left_indices = np.where((X[:, feature] <= threshold) & mask)[0]
            right_indices = np.where((X[:, feature] > threshold) & mask)[0]
            missing_indices = np.where(~mask)[0]
            
            # Assign missing to majority branch
            if len(left_indices) > len(right_indices):
                left_indices = np.concatenate([left_indices, missing_indices])
            else:
                right_indices = np.concatenate([right_indices, missing_indices])
                
            return left_indices, right_indices
    ```

    **Best practices:**
    - Understand the mechanism of missingness
    - Consider domain knowledge
    - Evaluate impact of different strategies
    - Monitor performance with validation data

??? question "7. Explain information gain and how it's calculated."

    **Information Gain** measures the reduction in impurity achieved by splitting on a particular feature.

    **Formula:**
    $$\text{Information Gain}(S, A) = \text{Impurity}(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \text{Impurity}(S_v)$$

    Where:
    - $S$ = current set of examples
    - $A$ = attribute/feature to split on
    - $S_v$ = subset where attribute $A$ has value $v$

    **Step-by-step calculation:**

    1. **Calculate parent impurity**:
       ```python
       def gini_impurity(y):
           _, counts = np.unique(y, return_counts=True)
           probabilities = counts / len(y)
           return 1 - np.sum(probabilities ** 2)
       
       parent_gini = gini_impurity(y_parent)
       ```

    2. **For each possible split**:
       ```python
       def information_gain(y_parent, y_left, y_right):
           n = len(y_parent)
           n_left, n_right = len(y_left), len(y_right)
           
           parent_impurity = gini_impurity(y_parent)
           
           weighted_child_impurity = (
               (n_left / n) * gini_impurity(y_left) + 
               (n_right / n) * gini_impurity(y_right)
           )
           
           return parent_impurity - weighted_child_impurity
       ```

    **Example:**
    ```
    Dataset: [Yes, Yes, No, Yes, No, No, Yes, No]
    Parent Gini = 1 - (4/8)² - (4/8)² = 0.5

    Split on Feature X <= 0.5:
    Left:  [Yes, Yes, Yes, Yes] → Gini = 0
    Right: [No, No, No, No]     → Gini = 0

    Information Gain = 0.5 - (4/8 * 0 + 4/8 * 0) = 0.5
    ```

    This split perfectly separates classes, achieving maximum information gain.

??? question "8. What is the difference between pre-pruning and post-pruning?"

    **Pre-pruning (Early Stopping):**
    - Stops tree growth **during** construction
    - Prevents overfitting by limiting growth
    - More efficient (less computation)
    - Risk of under-fitting

    **Common pre-pruning parameters:**
    ```python
    DecisionTreeClassifier(
        max_depth=5,                    # Maximum tree depth
        min_samples_split=20,           # Min samples to split node
        min_samples_leaf=10,            # Min samples in leaf
        max_features='sqrt',            # Features to consider
        min_impurity_decrease=0.01      # Min impurity decrease
    )
    ```

    **Post-pruning:**
    - Builds full tree first, then removes branches
    - More thorough exploration of tree space
    - Better performance but more computationally expensive
    - Lower risk of under-fitting

    **Post-pruning techniques:**

    1. **Cost Complexity Pruning:**
    ```python
    # Find optimal alpha using cross-validation
    path = tree.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas[:-1]  # Exclude max alpha

    trees = []
    for alpha in ccp_alphas:
        clf = DecisionTreeClassifier(ccp_alpha=alpha)
        clf.fit(X_train, y_train)
        trees.append(clf)

    # Select best alpha using validation
    scores = [accuracy_score(y_val, tree.predict(X_val)) for tree in trees]
    best_alpha = ccp_alphas[np.argmax(scores)]
    ```

    2. **Reduced Error Pruning:**
    - Use validation set to evaluate node removal
    - Remove nodes that improve validation performance

    **Comparison:**
    
    | Aspect | Pre-pruning | Post-pruning |
    |--------|-------------|--------------|
    | Computation | Faster | Slower |
    | Memory | Less | More |
    | Risk | Under-fitting | Over-fitting |
    | Performance | Good | Better |
    | Implementation | Simpler | Complex |

    **Recommendation:** Start with pre-pruning for quick results, use post-pruning for optimal performance.

??? question "9. How do you evaluate the performance of a Decision Tree?"

    **Classification Metrics:**

    1. **Accuracy**: Overall correctness
       ```python
       from sklearn.metrics import accuracy_score
       accuracy = accuracy_score(y_true, y_pred)
       ```

    2. **Precision, Recall, F1-score**: Class-specific performance
       ```python
       from sklearn.metrics import classification_report, precision_recall_fscore_support
       print(classification_report(y_true, y_pred))
       ```

    3. **Confusion Matrix**: Detailed error analysis
       ```python
       from sklearn.metrics import confusion_matrix
       import seaborn as sns
       cm = confusion_matrix(y_true, y_pred)
       sns.heatmap(cm, annot=True, fmt='d')
       ```

    4. **ROC Curve and AUC**: Threshold-independent evaluation
       ```python
       from sklearn.metrics import roc_curve, auc
       fpr, tpr, _ = roc_curve(y_true, y_proba)
       auc_score = auc(fpr, tpr)
       ```

    **Regression Metrics:**

    1. **Mean Squared Error (MSE)**:
       ```python
       from sklearn.metrics import mean_squared_error
       mse = mean_squared_error(y_true, y_pred)
       ```

    2. **R² Score**: Explained variance
       ```python
       from sklearn.metrics import r2_score
       r2 = r2_score(y_true, y_pred)
       ```

    **Cross-validation:**
    ```python
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    # Stratified K-fold for classification
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(tree, X, y, cv=skf, scoring='accuracy')
    print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

    # Multiple metrics
    from sklearn.model_selection import cross_validate
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    scores = cross_validate(tree, X, y, cv=skf, scoring=scoring)
    ```

    **Validation Curves:**
    ```python
    from sklearn.model_selection import validation_curve

    # Evaluate effect of max_depth
    param_range = range(1, 11)
    train_scores, val_scores = validation_curve(
        DecisionTreeClassifier(random_state=42),
        X, y, param_name='max_depth', param_range=param_range,
        cv=5, scoring='accuracy'
    )

    plt.plot(param_range, train_scores.mean(axis=1), label='Training')
    plt.plot(param_range, val_scores.mean(axis=1), label='Validation')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.legend()
    ```

    **Business metrics:**
    - Consider domain-specific metrics
    - Cost of false positives vs false negatives
    - Interpretability requirements
    - Prediction speed requirements

??? question "10. When would you choose Decision Trees over other algorithms, and when would you avoid them?"

    **Choose Decision Trees when:**

    1. **Interpretability is crucial**:
       - Medical diagnosis
       - Legal decisions  
       - Regulatory compliance
       - Business rule extraction

    2. **Mixed data types**:
       - Combination of numerical and categorical features
       - No need for extensive preprocessing

    3. **Feature selection needed**:
       - High-dimensional data
       - Unknown feature importance
       - Automatic relevance detection

    4. **Non-linear relationships**:
       - Complex interaction patterns
       - Threshold-based decisions
       - Rule-based logic

    5. **Quick prototyping**:
       - Fast training and prediction
       - Baseline model development
       - Proof of concept

    6. **Robust to outliers**:
       - Noisy data with extreme values
       - No assumptions about distribution

    **Avoid Decision Trees when:**

    1. **Maximum accuracy required**:
       - Use ensemble methods (Random Forest, XGBoost)
       - Deep learning for complex patterns
       - SVMs for high-dimensional data

    2. **Linear relationships dominate**:
       - Use linear/logistic regression
       - More efficient and interpretable for linear patterns

    3. **Very small datasets**:
       - High variance with limited data
       - Risk of overfitting
       - Simple models preferred

    4. **Stable predictions needed**:
       - High variance to data changes
       - Use ensemble methods for stability

    5. **Extrapolation required**:
       - Cannot predict outside training range
       - Use regression models for extrapolation

    **Decision Matrix:**
    ```
    Data Size:      Small → Avoid, Large → Consider
    Interpretability: High need → Choose, Low need → Consider alternatives  
    Accuracy req:   High → Ensemble, Medium → Consider
    Data type:      Mixed → Choose, Numerical only → Consider alternatives
    Stability:      Required → Avoid, Not critical → Consider
    Linear patterns: Dominant → Avoid, Minimal → Choose
    ```

    **Best practice:** Start with Decision Trees for understanding, then move to ensemble methods for performance.

## 🧠 Examples

### Example 1: Credit Approval System

```python
# Simulate credit approval data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# Generate synthetic credit data
np.random.seed(42)
n_samples = 1000

data = {
    'income': np.random.normal(50000, 20000, n_samples),
    'credit_score': np.random.normal(650, 100, n_samples),
    'age': np.random.normal(35, 10, n_samples),
    'debt_ratio': np.random.uniform(0, 0.8, n_samples),
    'employment_years': np.random.exponential(5, n_samples)
}

# Create approval logic (simplified)
approval = []
for i in range(n_samples):
    score = 0
    if data['income'][i] > 40000: score += 2
    if data['credit_score'][i] > 600: score += 3
    if data['age'][i] > 25: score += 1
    if data['debt_ratio'][i] < 0.4: score += 2
    if data['employment_years'][i] > 2: score += 1
    
    # Add some noise
    score += np.random.normal(0, 1)
    approval.append(1 if score > 5 else 0)

# Create DataFrame
df = pd.DataFrame(data)
df['approved'] = approval

print("Credit Approval Dataset:")
print(df.head())
print(f"\nApproval rate: {df['approved'].mean():.2%}")

# Train Decision Tree
X = df.drop('approved', axis=1)
y = df['approved']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create interpretable model
credit_tree = DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=50,
    min_samples_leaf=20,
    random_state=42
)

credit_tree.fit(X_train, y_train)

# Evaluate
y_pred = credit_tree.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Rejected', 'Approved']))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': credit_tree.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Extract decision rules
print("\nDecision Rules:")
tree_rules = export_text(credit_tree, feature_names=list(X.columns))
print(tree_rules)

# Visualize tree
plt.figure(figsize=(15, 10))
plot_tree(credit_tree, 
          feature_names=X.columns,
          class_names=['Rejected', 'Approved'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Credit Approval Decision Tree")
plt.show()

# Test specific cases
test_cases = pd.DataFrame({
    'income': [30000, 60000, 45000],
    'credit_score': [550, 750, 650],
    'age': [22, 40, 30],
    'debt_ratio': [0.6, 0.2, 0.3],
    'employment_years': [1, 8, 3]
})

predictions = credit_tree.predict(test_cases)
probabilities = credit_tree.predict_proba(test_cases)

print("\nTest Cases:")
for i, (idx, row) in enumerate(test_cases.iterrows()):
    result = "Approved" if predictions[i] == 1 else "Rejected"
    confidence = probabilities[i].max()
    print(f"Case {i+1}: {result} (Confidence: {confidence:.2%})")
    print(f"  Income: ${row['income']:,.0f}, Credit Score: {row['credit_score']:.0f}")
```

### Example 2: Medical Diagnosis

```python
# Medical diagnosis example using Decision Tree
from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load breast cancer dataset
cancer = load_breast_cancer()
X_cancer = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y_cancer = cancer.target

print("Medical Diagnosis Dataset:")
print(f"Features: {len(cancer.feature_names)}")
print(f"Samples: {len(X_cancer)}")
print(f"Classes: {cancer.target_names}")

# Focus on most interpretable features
important_features = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry'
]

X_medical = X_cancer[important_features]

# Train interpretable model
X_train_med, X_test_med, y_train_med, y_test_med = train_test_split(
    X_medical, y_cancer, test_size=0.2, random_state=42, stratify=y_cancer
)

medical_tree = DecisionTreeClassifier(
    max_depth=3,  # Keep simple for medical interpretability
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)

medical_tree.fit(X_train_med, y_train_med)

# Evaluate
y_pred_med = medical_tree.predict(X_test_med)
print(f"\nDiagnostic Accuracy: {accuracy_score(y_test_med, y_pred_med):.3f}")

# Medical decision rules
print("\nMedical Decision Rules:")
med_rules = export_text(medical_tree, 
                       feature_names=important_features)
print(med_rules)

# Feature importance for medical interpretation
med_importance = pd.DataFrame({
    'Medical Feature': important_features,
    'Clinical Importance': medical_tree.feature_importances_
}).sort_values('Clinical Importance', ascending=False)

print("\nClinical Feature Importance:")
for _, row in med_importance.iterrows():
    print(f"{row['Medical Feature']}: {row['Clinical Importance']:.3f}")

# Confusion matrix for medical evaluation
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test_med, y_pred_med)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', 
            xticklabels=['Malignant', 'Benign'],
            yticklabels=['Malignant', 'Benign'])
plt.title('Medical Diagnosis Confusion Matrix')
plt.ylabel('Actual Diagnosis')
plt.xlabel('Predicted Diagnosis')
plt.show()

# Calculate medical metrics
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)  # True Positive Rate
specificity = tn / (tn + fp)  # True Negative Rate
ppv = tp / (tp + fp)         # Positive Predictive Value
npv = tn / (tn + fn)         # Negative Predictive Value

print(f"\nMedical Performance Metrics:")
print(f"Sensitivity (Recall): {sensitivity:.3f}")
print(f"Specificity: {specificity:.3f}")
print(f"Positive Predictive Value: {ppv:.3f}")
print(f"Negative Predictive Value: {npv:.3f}")
```

## 📚 References

1. **Books:**
   - [The Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/) - Hastie, Tibshirani, Friedman
   - [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/) - Christopher Bishop
   - [Hands-On Machine Learning](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) - Aurélien Géron

2. **Academic Papers:**
   - [Induction of Decision Trees](https://link.springer.com/article/10.1007/BF00116251) - J.R. Quinlan (1986)
   - [C4.5: Programs for Machine Learning](https://www.amazon.com/C4-5-Programs-Machine-Learning-Kaufmann/dp/1558602380) - J.R. Quinlan (1993)

3. **Online Resources:**
   - [Scikit-learn Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
   - [CS229 Stanford - Decision Trees](http://cs229.stanford.edu/notes2020spring/cs229-notes-dt.pdf)
   - [Towards Data Science - Decision Trees Explained](https://towardsdatascience.com/decision-trees-in-machine-learning-641b9c4e8dcb)

4. **Interactive Tools:**
   - [Decision Tree Visualizer](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/)
   - [Teachable Machine](https://teachablemachine.withgoogle.com/)

5. **Video Lectures:**
   - [MIT 6.034 Artificial Intelligence - Decision Trees](https://www.youtube.com/watch?v=OBWL4oLT7Uc)
   - [StatQuest - Decision Trees](https://www.youtube.com/watch?v=7VeUMuQv1j0)
