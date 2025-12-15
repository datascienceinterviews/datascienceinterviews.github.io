
---
title: Scikit-learn Cheat Sheet
description: A comprehensive reference guide for Scikit-learn, covering data preprocessing, model selection, training, evaluation, and more.
---

# Scikit-learn Cheat Sheet

[TOC]

This cheat sheet provides an exhaustive overview of the Scikit-learn (sklearn) machine learning library, covering essential concepts, code examples, and best practices for efficient model building, training, evaluation, and deployment. It aims to be a one-stop reference for common tasks.

## Quick Reference

```
┌────────────────────────────────────────────────────────────┐
│             COMMON TASKS QUICK REFERENCE                   │
└────────────────────────────────────────────────────────────┘

Task                              Module / Function
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA PREPARATION
Load dataset                      datasets.load_iris()
Split data                        train_test_split()
Scale features                    StandardScaler()
Handle missing values             SimpleImputer()
Encode categories                 OneHotEncoder()

MODEL TRAINING
Linear classification             LogisticRegression()
Non-linear classification         RandomForestClassifier()
Regression                        LinearRegression()
Clustering                        KMeans()
Dimensionality reduction          PCA()

MODEL EVALUATION
Classification metrics            accuracy_score(), f1_score()
Regression metrics                r2_score(), mean_squared_error()
Cross-validation                  cross_val_score()
Confusion matrix                  confusion_matrix()

HYPERPARAMETER TUNING
Grid search                       GridSearchCV()
Random search                     RandomizedSearchCV()

PIPELINES & WORKFLOWS
Create pipeline                   Pipeline()
Ensemble methods                  VotingClassifier()

MODEL PERSISTENCE
Save model                        joblib.dump()
Load model                        joblib.load()
```

## Machine Learning Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    ML WORKFLOW WITH SKLEARN                     │
└─────────────────────────────────────────────────────────────────┘

    ┌──────────────────┐
    │   Load Data      │
    │ (datasets.load)  │
    └────────┬─────────┘
             ↓
    ┌──────────────────┐
    │ Explore & Clean  │
    │  (Imputation)    │
    └────────┬─────────┘
             ↓
    ┌──────────────────┐
    │ Preprocess Data  │
    │  (Scale/Encode)  │
    └────────┬─────────┘
             ↓
    ┌──────────────────┐
    │  Split Dataset   │
    │ (train_test_split)│
    └────────┬─────────┘
             ↓
    ┌──────────────────┐
    │  Select Model    │
    │ (Choose Algo)    │
    └────────┬─────────┘
             ↓
    ┌──────────────────┐
    │  Train Model     │
    │   (model.fit)    │
    └────────┬─────────┘
             ↓
    ┌──────────────────┐
    │  Evaluate Model  │
    │   (metrics)      │
    └────────┬─────────┘
             │
             ├────→ Poor Performance?
             │           ↓
             │      ┌──────────────────┐
             │      │ Tune Hyperparams │
             │      │  (GridSearchCV)  │
             │      └────────┬─────────┘
             │               │
             │               └──────────→ [Back to Train]
             ↓
    ┌──────────────────┐
    │   Deploy Model   │
    │  (joblib.dump)   │
    └──────────────────┘
```

## Getting Started

### Installation

```bash
pip install scikit-learn
```

### Importing Scikit-learn

```python
import sklearn
from sklearn import datasets  # For built-in datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
```

## Data Preprocessing

### Loading Data

#### Built-in Datasets

```python
from sklearn import datasets

# Classification dataset: Iris (150 samples, 4 features, 3 classes)
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Regression dataset: California Housing (20,640 samples, 8 features)
california_housing = datasets.fetch_california_housing()
X = california_housing.data
y = california_housing.target

# Image dataset: Handwritten digits (1,797 samples, 64 features, 10 classes)
digits = datasets.load_digits()
X = digits.data
y = digits.target
```

#### From Pandas DataFrame

```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("your_data.csv")
X = df.drop("target_column", axis=1)
y = df["target_column"]
```

### Splitting Data

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 80% training, 20% testing
```

### Feature Scaling

```
┌────────────────────────────────────────────────────────────┐
│              FEATURE SCALING METHODS                       │
└────────────────────────────────────────────────────────────┘

StandardScaler          MinMaxScaler         RobustScaler
(Mean=0, Std=1)        (Range: 0-1)         (Uses Median/IQR)
      │                     │                      │
      ↓                     ↓                      ↓
   z = (x-μ)/σ         x' = (x-min)/(max-min)  (x-median)/IQR
      │                     │                      │
      ↓                     ↓                      ↓
Best for: Normal      Best for: Bounded     Best for: Data
distributions         ranges needed         with outliers

Normalizer (L2): Scales each sample to unit norm
      x' = x / ||x||₂
Best for: When direction matters more than magnitude
```

#### Standardization

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

#### Min-Max Scaling

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

#### Robust Scaling

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

#### Normalization

```python
from sklearn.preprocessing import Normalizer

normalizer = Normalizer()
X_train_normalized = normalizer.fit_transform(X_train)
X_test_normalized = normalizer.transform(X_test)
```

### Handling Missing Values

#### Imputation (SimpleImputer)

```python
from sklearn.impute import SimpleImputer
import numpy as np

imputer = SimpleImputer(strategy="mean")  # Replace missing values with the mean
# Other strategies: "median", "most_frequent", "constant"
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
```

#### Imputation (KNNImputer)

```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
```

#### Dropping Missing Values

```python
import pandas as pd
# Assuming X_train and X_test are pandas DataFrames
X_train_dropped = X_train.dropna()
X_test_dropped = X_test.dropna()
```

### Encoding Categorical Features

```
┌────────────────────────────────────────────────────────────┐
│              ENCODING STRATEGIES                           │
└────────────────────────────────────────────────────────────┘

Original: ['cat', 'dog', 'bird', 'cat']

OneHotEncoder           OrdinalEncoder       LabelEncoder
      │                       │                    │
      ↓                       ↓                    ↓
┌───────────┐           ┌─────────┐           ┌─────┐
│ 1 0 0     │           │ 0       │           │ 0   │
│ 0 1 0     │           │ 1       │           │ 1   │
│ 0 0 1     │           │ 2       │           │ 2   │
│ 1 0 0     │           │ 0       │           │ 0   │
└───────────┘           └─────────┘           └─────┘
 cat dog bird           Ordered                Target
                        relationship           only

Use Case:              Use Case:             Use Case:
- No ordinal          - Ordered              - Target variable
  relationship          categories             encoding
- Creates sparse      - Tree-based           - Simple integer
  features              models                 mapping
```

#### One-Hot Encoding

```python
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Assuming X_train and X_test are pandas DataFrames
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # sparse=False for older versions
X_train_encoded = encoder.fit_transform(X_train[['categorical_feature']])
X_test_encoded = encoder.transform(X_test[['categorical_feature']])

# Or, using pandas:
X_train_encoded = pd.get_dummies(X_train, columns=['categorical_feature'])
X_test_encoded = pd.get_dummies(X_test, columns=['categorical_feature'])
```

#### Ordinal Encoding

```python
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder()
X_train_encoded = encoder.fit_transform(X_train[['ordinal_feature']])
X_test_encoded = encoder.transform(X_test[['ordinal_feature']])
```

#### Label Encoding (for target variable)

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)
```

### Feature Engineering

#### Polynomial Features

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
```

#### Custom Transformers

```python
from sklearn.preprocessing import FunctionTransformer
import numpy as np

def log_transform(x):
    return np.log1p(x)

log_transformer = FunctionTransformer(log_transform)
X_train_log = log_transformer.transform(X_train)
X_test_log = log_transformer.transform(X_test)
```

### Feature Selection

#### VarianceThreshold

```python
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.1)  # Remove features with variance below 0.1
X_train_selected = selector.fit_transform(X_train)
X_test_selected = selector.transform(X_test)
```

#### SelectKBest

```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k=5)  # Select top 5 features
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
```

#### SelectFromModel

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

estimator = LogisticRegression(penalty="l1", solver='liblinear')
selector = SelectFromModel(estimator)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
```

#### RFE (Recursive Feature Elimination)

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

estimator = LogisticRegression()
selector = RFE(estimator, n_features_to_select=5)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
```

## Model Selection and Training

```
┌────────────────────────────────────────────────────────────┐
│              MODEL SELECTION GUIDE                         │
└────────────────────────────────────────────────────────────┘

                    Problem Type?
                         │
         ┌───────────────┼───────────────┐
         ↓                               ↓
   Classification                   Regression
         │                               │
         │                               │
    Data Size?                      Data Size?
         │                               │
    ┌────┴────┐                     ┌────┴────┐
    ↓         ↓                     ↓         ↓
  Small     Large                 Small     Large
    │         │                     │         │
    │         │                     │         │
Linear?   Linear?                Linear?   Linear?
    │         │                     │         │
┌───┴───┐ ┌──┴───┐             ┌───┴───┐ ┌──┴───┐
↓       ↓ ↓      ↓             ↓       ↓ ↓      ↓
Yes    No  Yes   No            Yes    No  Yes   No
│       │  │     │              │       │  │     │
↓       ↓  ↓     ↓              ↓       ↓  ↓     ↓

CLASSIFICATION MODELS          REGRESSION MODELS
━━━━━━━━━━━━━━━━━━━━━          ━━━━━━━━━━━━━━━━━━
Small + Linear:                Small + Linear:
- Logistic Regression          - Linear Regression
- Linear SVM                   - Ridge/Lasso
                               - SVR (linear)
Small + Non-linear:
- SVM (RBF kernel)             Small + Non-linear:
- Decision Tree                - SVR (RBF kernel)
- Random Forest                - Decision Tree
                               - Random Forest
Large + Linear:
- SGD Classifier               Large + Linear:
- Logistic Regression          - SGD Regressor
                               - Linear Regression
Large + Non-linear:
- Random Forest                Large + Non-linear:
- Gradient Boosting            - Random Forest
- Neural Networks              - Gradient Boosting
                               - Neural Networks

Special Cases:
- Multi-class: OneVsRest, OneVsOne
- Imbalanced: Use class_weight='balanced'
- High dimensions: L1/L2 regularization
- Interpretability needed: Linear models, Decision Trees
```

### Linear Regression

```python
from sklearn.linear_model import LinearRegression

# Basic linear regression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Access coefficients and intercept
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# Make predictions
new_data = [[1.5, 2.0]]
prediction = model.predict(new_data)
```

### Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

# Binary classification
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Get probability estimates
y_pred_proba = model.predict_proba(X_test)  # Returns probabilities for each class
print(f"Probability estimates: {y_pred_proba[:5]}")

# Multi-class classification (One-vs-Rest by default)
model_multiclass = LogisticRegression(multi_class='ovr', max_iter=200, random_state=42)
model_multiclass.fit(X_train, y_train)

# Regularization: Control overfitting with C parameter (smaller C = stronger regularization)
model_regularized = LogisticRegression(C=0.1, penalty='l2', random_state=42, max_iter=200)
model_regularized.fit(X_train, y_train)
```

### Support Vector Machines (SVM)

```python
from sklearn.svm import SVC, SVR

# For classification
model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)

# For regression
model = SVR(kernel='linear', C=1.0)
model.fit(X_train, y_train)
```

### Decision Trees

```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# For classification
model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)

# For regression
model = DecisionTreeRegressor(max_depth=5)
model.fit(X_train, y_train)
```

### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# For classification
model = RandomForestClassifier(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)

# For regression
model = RandomForestRegressor(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)
```

### Gradient Boosting

```python
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# For classification
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)

# For regression
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)
```

### K-Nearest Neighbors (KNN)

```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# For classification
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# For regression
model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train, y_train)
```

### Naive Bayes

```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)
```

### Clustering (K-Means)

```python
from sklearn.cluster import KMeans

model = KMeans(n_clusters=3, random_state=42, n_init = 'auto') # Added n_init
model.fit(X_train)
labels = model.predict(X_test)
```

### Principal Component Analysis (PCA)

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
```

### Model Persistence

```python
import joblib

# Save the model
joblib.dump(model, 'my_model.pkl')

# Load the model
loaded_model = joblib.load('my_model.pkl')
```

## Model Evaluation

### Regression Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

### Classification Metrics

```
┌────────────────────────────────────────────────────────────┐
│              CONFUSION MATRIX & METRICS                    │
└────────────────────────────────────────────────────────────┘

                    Predicted
                 Positive  Negative
              ┌──────────┬──────────┐
Actual   Pos  │    TP    │    FN    │
              ├──────────┼──────────┤
         Neg  │    FP    │    TN    │
              └──────────┴──────────┘

Metrics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Accuracy    = (TP + TN) / (TP + TN + FP + FN)
            → Overall correctness

Precision   = TP / (TP + FP)
            → Of predicted positives, how many are correct?
            → Important when FP is costly

Recall      = TP / (TP + FN)
            → Of actual positives, how many did we catch?
            → Important when FN is costly

F1 Score    = 2 × (Precision × Recall) / (Precision + Recall)
            → Harmonic mean of Precision and Recall
            → Balanced metric for imbalanced data

Example Use Cases:
- Spam Detection: High Precision (avoid blocking real emails)
- Disease Screening: High Recall (catch all potential cases)
- Balanced: F1 Score
```

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')  # 'macro', 'micro', 'weighted' for multi-class
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Detailed report
report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(report)

# For multi-class classification
# Use average='weighted' to account for class imbalance
precision_multi = precision_score(y_test, y_pred, average='weighted')
```

### ROC Curve and AUC

```python
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Train model and get probability predictions
model = LogisticRegression(random_state=42, max_iter=200)
model.fit(X_train, y_train)

# For binary classification - get probabilities for positive class
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Model (AUC = {auc:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.show()

# For multi-class classification
from sklearn.metrics import roc_auc_score
y_pred_proba_multi = model.predict_proba(X_test)
auc_multi = roc_auc_score(y_test, y_pred_proba_multi, multi_class='ovr')
print(f"Multi-class AUC: {auc_multi:.3f}")
```

### Cross-Validation

```
┌────────────────────────────────────────────────────────────┐
│           K-FOLD CROSS-VALIDATION (K=5)                    │
└────────────────────────────────────────────────────────────┘

Full Dataset: [████████████████████████████████████████]

Fold 1:  [TEST][TRAIN][TRAIN][TRAIN][TRAIN]  →  Score₁
Fold 2:  [TRAIN][TEST][TRAIN][TRAIN][TRAIN]  →  Score₂
Fold 3:  [TRAIN][TRAIN][TEST][TRAIN][TRAIN]  →  Score₃
Fold 4:  [TRAIN][TRAIN][TRAIN][TEST][TRAIN]  →  Score₄
Fold 5:  [TRAIN][TRAIN][TRAIN][TRAIN][TEST]  →  Score₅
                        ↓
        Final Score = Mean(Score₁, Score₂, ..., Score₅)

Benefits:
- Every sample used for both training and testing
- More reliable performance estimate
- Reduces variance in model evaluation
```

```python
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

# K-Fold Cross-Validation
cv_scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation

# Stratified K-Fold (for classification - preserves class distribution)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv)

print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean():.2f}")
print(f"Std deviation: {cv_scores.std():.2f}")
```

### Learning Curves

```
┌────────────────────────────────────────────────────────────┐
│         BIAS-VARIANCE TRADEOFF & LEARNING CURVES           │
└────────────────────────────────────────────────────────────┘

UNDERFITTING (High Bias)          GOOD FIT           OVERFITTING (High Variance)
━━━━━━━━━━━━━━━━━━━━━━━━━━        ━━━━━━━━━━        ━━━━━━━━━━━━━━━━━━━━━━━━━━
Score                             Score                Score
  │                                 │                    │
1.0│                                1.0│                  1.0│     ┌─Train
  │  ┌────────── Train               │  ┌─Train             │    ╱
  │  │                                │ ╱                    │   ╱
0.5│  │                              0.5│                   0.5│  │
  │  └────────── Valid               │ └─Valid               │  └────Valid
  │                                   │                       │
0.0└────────────────                0.0└──────────          0.0└──────────
    Training Size                       Training Size           Training Size

Symptoms:                         Symptoms:                Symptoms:
- Low train score                 - High train score       - Very high train score
- Low valid score                 - High valid score       - Much lower valid score
- Similar scores                  - Similar scores         - Large gap
- Model too simple                - Right complexity       - Model too complex

Solutions:                        Keep it!                 Solutions:
- More features                                           - More training data
- More complex model                                      - Reduce features
- Reduce regularization                                   - Increase regularization
                                                         - Simpler model
                                                         - Early stopping
```

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

# Generate learning curves
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy', n_jobs=-1
)

# Calculate mean and standard deviation
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training score', marker='o')
plt.plot(train_sizes, test_mean, label='Cross-validation score', marker='s')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15)
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.legend(loc='best')
plt.title('Learning Curve')
plt.grid(True)
plt.show()

# Diagnose overfitting/underfitting
gap = train_mean[-1] - test_mean[-1]
if gap > 0.1:
    print("⚠ Model may be overfitting (large gap between train and test)")
elif test_mean[-1] < 0.6:
    print("⚠ Model may be underfitting (low scores on both sets)")
else:
    print("✓ Model appears to be fitting well")
```

### Validation Curves

```python
from sklearn.model_selection import validation_curve
import numpy as np

param_range = np.logspace(-6, -1, 5)
train_scores, test_scores = validation_curve(
    model, X, y, param_name="gamma", param_range=param_range,
    cv=5, scoring="accuracy")

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean, label='Training score')
plt.plot(param_range, test_mean, label='Cross-validation score')
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.1)
plt.xscale('log')
plt.xlabel('Parameter Value')
plt.ylabel('Score')
plt.legend()
plt.title('Validation Curve')
plt.show()
```

## Hyperparameter Tuning

```
┌────────────────────────────────────────────────────────────┐
│         HYPERPARAMETER TUNING STRATEGIES                   │
└────────────────────────────────────────────────────────────┘

GridSearchCV (Exhaustive Search)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Parameter Grid:
  C: [0.1, 1, 10]
  kernel: ['linear', 'rbf']
  gamma: [0.1, 1]
       ↓
Tests ALL combinations: 3 × 2 × 2 = 12 models
       │
       ├─→ Model(C=0.1, kernel='linear', gamma=0.1)
       ├─→ Model(C=0.1, kernel='linear', gamma=1)
       ├─→ Model(C=0.1, kernel='rbf', gamma=0.1)
       └─→ ... (9 more)
       ↓
   Cross-Validate each
       ↓
   Select Best Params

Best for: Small parameter space


RandomizedSearchCV (Random Sampling)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Parameter Distributions:
  C: uniform(0.1, 10)
  kernel: ['linear', 'rbf']
  gamma: log-uniform(0.001, 1)
       ↓
Sample N random combinations (e.g., n_iter=20)
       │
       ├─→ Model(C=3.2, kernel='rbf', gamma=0.034)
       ├─→ Model(C=7.1, kernel='linear', gamma=0.421)
       └─→ ... (18 more)
       ↓
   Cross-Validate each
       ↓
   Select Best Params

Best for: Large parameter space, continuous parameters


Comparison:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GridSearchCV         RandomizedSearchCV
- Exhaustive         - Sampling-based
- Guaranteed best    - May miss optimal
- Slow for large     - Faster
  parameter space
- Good for discrete  - Good for continuous
  parameters           parameters
```

### GridSearchCV

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Simple grid search
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': [0.1, 1, 'scale', 'auto']
}

grid_search = GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1  # Use all CPU cores
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.2f}")
best_model = grid_search.best_estimator_

# Evaluate on test set
test_score = best_model.score(X_test, y_test)
print(f"Test set score: {test_score:.2f}")

# Grid search with Pipeline (RECOMMENDED)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(random_state=42))
])

# Use double underscore to access pipeline step parameters
param_grid_pipeline = {
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['linear', 'rbf'],
    'classifier__gamma': ['scale', 'auto']
}

grid_search_pipeline = GridSearchCV(
    pipeline,
    param_grid_pipeline,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search_pipeline.fit(X_train, y_train)

print(f"\nPipeline best parameters: {grid_search_pipeline.best_params_}")
print(f"Pipeline best score: {grid_search_pipeline.best_score_:.2f}")
```

### RandomizedSearchCV

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(10, 200),
    'max_depth': [3, 5, 10, None],
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 11),
    'bootstrap': [True, False]
}

random_search = RandomizedSearchCV(RandomForestClassifier(), param_distributions=param_dist,
                                   n_iter=20, cv=5, scoring='accuracy', random_state=42, verbose=2)
random_search.fit(X_train, y_train)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best cross-validation score: {random_search.best_score_:.2f}")
best_model = random_search.best_estimator_
```

## Pipelines

```
┌────────────────────────────────────────────────────────────┐
│                    PIPELINE FLOW                           │
└────────────────────────────────────────────────────────────┘

Raw Data (X_train)
       │
       ↓
┌──────────────────┐
│ StandardScaler   │  Step 1: Transform
│  (fit_transform) │
└────────┬─────────┘
         ↓
  Scaled Data
         │
         ↓
┌──────────────────┐
│  Feature Select  │  Step 2: Transform
│  (fit_transform) │
└────────┬─────────┘
         ↓
  Selected Features
         │
         ↓
┌──────────────────┐
│   Classifier     │  Step 3: Fit
│     (fit)        │
└────────┬─────────┘
         ↓
   Trained Model

Benefits:
- Prevents data leakage (fit only on training data)
- Simplifies workflow
- Easy hyperparameter tuning with GridSearchCV
- Ensures consistent preprocessing
```

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC

# Create pipeline with multiple steps
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_classif, k=10)),
    ('classifier', SVC(kernel='rbf'))
])

# Fit and predict (all steps executed automatically)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Access individual steps
scaler = pipeline.named_steps['scaler']
classifier = pipeline.named_steps['classifier']
```

## Ensemble Methods

```
┌────────────────────────────────────────────────────────────┐
│              ENSEMBLE LEARNING STRATEGIES                  │
└────────────────────────────────────────────────────────────┘

BAGGING (Bootstrap Aggregating)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Training Data → [Random Samples] → Parallel Training
                      │
        ┌─────────────┼─────────────┐
        ↓             ↓             ↓
    Model 1       Model 2       Model 3
        │             │             │
        └─────────────┼─────────────┘
                      ↓
              Vote / Average
                      ↓
              Final Prediction

Examples: Random Forest, Bagging Classifier


BOOSTING (Sequential Learning)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Training Data → Model 1 → Reweight → Model 2 → Reweight → Model 3
                  │          ↓          │          ↓          │
                  └──────→ Focus on ←──┘          │          │
                           Errors                 │          │
                                                  └────→ Weighted
                                                         Combination
Examples: AdaBoost, Gradient Boosting


STACKING (Meta-Learning)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Training Data
       │
       ├───→ Model 1 (SVM)     → Prediction 1
       ├───→ Model 2 (Tree)    → Prediction 2    } Level 0
       └───→ Model 3 (KNN)     → Prediction 3
                  │
                  ↓
           [Predictions as Features]
                  │
                  ↓
          Meta-Model (LogReg)                    } Level 1
                  │
                  ↓
          Final Prediction


VOTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hard Voting: Majority class wins
Soft Voting: Average predicted probabilities
```

### Bagging

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

base_estimator = DecisionTreeClassifier(max_depth=5)
bagging = BaggingClassifier(base_estimator=base_estimator, n_estimators=10, random_state=42)
bagging.fit(X_train, y_train)
```

### Boosting (AdaBoost)

```python
from sklearn.ensemble import AdaBoostClassifier

adaboost = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42)
adaboost.fit(X_train, y_train)
```

### Stacking

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

estimators = [
    ('svm', SVC(kernel='linear', C=1.0)),
    ('dt', DecisionTreeClassifier(max_depth=5))
]
final_estimator = LogisticRegression()

stacking = StackingClassifier(estimators=estimators, final_estimator=final_estimator)
stacking.fit(X_train, y_train)
```

### Voting Classifier

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

estimator1 = LogisticRegression(solver='liblinear')
estimator2 = SVC(kernel='linear', C=1.0, probability=True) # probability=True for soft voting

voting = VotingClassifier(estimators=[('lr', estimator1), ('svc', estimator2)], voting='soft') # 'hard' for majority voting
voting.fit(X_train, y_train)
```

## Dimensionality Reduction

```
┌────────────────────────────────────────────────────────────┐
│        DIMENSIONALITY REDUCTION TECHNIQUES                 │
└────────────────────────────────────────────────────────────┘

                High-Dimensional Data
                         │
                         ↓
        ┌────────────────┼────────────────┐
        │                                 │
        ↓                                 ↓
    LINEAR                          NON-LINEAR
        │                                 │
  ┌─────┴─────┐                    ┌─────┴─────┐
  ↓           ↓                    ↓           ↓
 PCA         LDA                 t-SNE        NMF
  │           │                    │           │
  │           │                    │           │

PCA (Principal Component Analysis)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Unsupervised
- Finds directions of maximum variance
- Orthogonal components
- Preserves global structure
Use: General dimensionality reduction, visualization

LDA (Linear Discriminant Analysis)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Supervised (requires labels)
- Maximizes class separability
- Linear transformation
Use: Classification preprocessing, feature extraction

t-SNE (t-Distributed Stochastic Neighbor Embedding)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Non-linear
- Preserves local structure
- Computationally expensive
- Stochastic (different runs → different results)
Use: Visualization (2D/3D), cluster analysis

NMF (Non-negative Matrix Factorization)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Non-negative data only
- Parts-based representation
- Interpretable components
Use: Topic modeling, image analysis, recommender systems

Comparison:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Technique  Supervised  Linear  Preserves      Speed
PCA        No          Yes     Global         Fast
LDA        Yes         Yes     Separability   Fast
t-SNE      No          No      Local          Slow
NMF        No          No      Parts          Medium
```

### PCA

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Fit PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Explained variance
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.2%}")

# Determine optimal number of components
pca_full = PCA()
pca_full.fit(X_train)
cumsum = np.cumsum(pca_full.explained_variance_ratio_)

# Plot explained variance
plt.figure(figsize=(10, 6))
plt.plot(cumsum, marker='o')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Analysis')
plt.legend()
plt.grid(True)
plt.show()

# Choose components to retain 95% variance
n_components_95 = np.argmax(cumsum >= 0.95) + 1
print(f"Components needed for 95% variance: {n_components_95}")
```

### Linear Discriminant Analysis (LDA)

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train, y_train)  # Supervised, needs y_train
X_test_lda = lda.transform(X_test)
```

### t-distributed Stochastic Neighbor Embedding (t-SNE)

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
X_train_tsne = tsne.fit_transform(X_train)  # Usually only fit_transform
```

### Non-negative Matrix Factorization (NMF)

```python
from sklearn.decomposition import NMF

nmf = NMF(n_components=2, random_state=42)
X_train_nmf = nmf.fit_transform(X_train)
X_test_nmf = nmf.transform(X_test)
```

## Model Inspection

### Feature Importances

```python
# For tree-based models (RandomForest, GradientBoosting)
importances = model.feature_importances_
print(importances)

# For linear models (LogisticRegression, LinearRegression)
coefficients = model.coef_
print(coefficients)
```

### Partial Dependence Plots

```python
from sklearn.inspection import plot_partial_dependence

plot_partial_dependence(model, X_train, features=[0, 1])  # Plot for features 0 and 1
```

### Permutation Importance

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
print(result.importances_mean)
```

## Calibration

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=5) # 'sigmoid' is another method
calibrated_model.fit(X_train, y_train)
```

## Dummy Estimators

```python
from sklearn.dummy import DummyClassifier, DummyRegressor

# For classification
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)

# For regression
dummy_reg = DummyRegressor(strategy="mean")
dummy_reg.fit(X_train, y_train)
```

## Multi-label Classification

```python
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

multilabel_model = MultiOutputClassifier(RandomForestClassifier())
multilabel_model.fit(X_train, y_train) # y_train is a 2D array of shape (n_samples, n_labels)
```

## Multi-class and Multi-label Classification

```python
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

ovr_model = OneVsRestClassifier(SVC(kernel='linear'))
ovr_model.fit(X_train, y_train)
```

## Outlier Detection

```python
from sklearn.ensemble import IsolationForest

outlier_detector = IsolationForest(random_state=42)
outlier_detector.fit(X_train)
outliers = outlier_detector.predict(X_test) # 1 for inliers, -1 for outliers
```

## Semi-Supervised Learning

```python
from sklearn.semi_supervised import LabelPropagation

label_prop_model = LabelPropagation()
label_prop_model.fit(X_train, y_train) # y_train can contain -1 for unlabeled samples
```

## Common Use Cases

### Handling Imbalanced Data

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Method 1: Use class_weight parameter
model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=200)
model.fit(X_train, y_train)

# Method 2: Compute custom class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
model = LogisticRegression(class_weight=class_weight_dict, random_state=42, max_iter=200)
model.fit(X_train, y_train)

# Method 3: Resampling (requires imbalanced-learn library)
# from imblearn.over_sampling import SMOTE
# smote = SMOTE(random_state=42)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

### Feature Importance Analysis

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Get feature importances
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Print feature ranking
print("Feature ranking:")
for i, idx in enumerate(indices):
    print(f"{i+1}. Feature {idx}: {importances[idx]:.4f}")

# Visualize feature importances
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices])
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.show()
```

### Time Series Split for Sequential Data

```python
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
import numpy as np

# Create time series cross-validator
tscv = TimeSeriesSplit(n_splits=5)

# Perform cross-validation
scores = []
for train_idx, test_idx in tscv.split(X):
    X_train_ts, X_test_ts = X[train_idx], X[test_idx]
    y_train_ts, y_test_ts = y[train_idx], y[test_idx]

    model = LinearRegression()
    model.fit(X_train_ts, y_train_ts)
    score = model.score(X_test_ts, y_test_ts)
    scores.append(score)

print(f"Time series CV scores: {scores}")
print(f"Mean score: {np.mean(scores):.3f}")
```

### Custom Transformer for Pipeline

```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

# Create custom transformer
class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features=None):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if self.features is None:
            return np.log1p(X_copy)
        else:
            X_copy[:, self.features] = np.log1p(X_copy[:, self.features])
            return X_copy

# Use in pipeline
pipeline = Pipeline([
    ('log_transform', LogTransformer(features=[0, 1])),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42, max_iter=200))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

### Model Comparison

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import numpy as np

# Define models to compare
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=200),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42)
}

# Compare models using cross-validation
results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    results[name] = {
        'mean': scores.mean(),
        'std': scores.std(),
        'scores': scores
    }
    print(f"{name}: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Select best model
best_model_name = max(results, key=lambda x: results[x]['mean'])
print(f"\nBest model: {best_model_name}")
```

### Saving Multiple Models

```python
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Train and save multiple models
models = {
    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f'{name}_model.pkl')
    print(f"Saved {name} model")

# Load and use models
loaded_rf = joblib.load('random_forest_model.pkl')
loaded_gb = joblib.load('gradient_boosting_model.pkl')

# Ensemble predictions
rf_pred = loaded_rf.predict_proba(X_test)
gb_pred = loaded_gb.predict_proba(X_test)
ensemble_pred = (rf_pred + gb_pred) / 2
final_pred = ensemble_pred.argmax(axis=1)
```

## Complete End-to-End Example

```python
# Complete ML pipeline: Classification on Iris dataset
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# ═══════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════
iris = datasets.load_iris()
X, y = iris.data, iris.target
print(f"Dataset shape: {X.shape}")
print(f"Classes: {np.unique(y)}")

# ═══════════════════════════════════════════════════════════
# 2. SPLIT DATA
# ═══════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# ═══════════════════════════════════════════════════════════
# 3. CREATE PIPELINE
# ═══════════════════════════════════════════════════════════
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(random_state=42))
])

# ═══════════════════════════════════════════════════════════
# 4. HYPERPARAMETER TUNING
# ═══════════════════════════════════════════════════════════
param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['linear', 'rbf'],
    'classifier__gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.3f}")

# ═══════════════════════════════════════════════════════════
# 5. EVALUATE ON TEST SET
# ═══════════════════════════════════════════════════════════
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print(f"\nTest set accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ═══════════════════════════════════════════════════════════
# 6. CROSS-VALIDATION ON FULL DATASET
# ═══════════════════════════════════════════════════════════
cv_scores = cross_val_score(best_model, X, y, cv=5)
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# ═══════════════════════════════════════════════════════════
# 7. SAVE MODEL
# ═══════════════════════════════════════════════════════════
joblib.dump(best_model, 'iris_classifier.pkl')
print("\nModel saved as 'iris_classifier.pkl'")

# ═══════════════════════════════════════════════════════════
# 8. LOAD AND USE MODEL
# ═══════════════════════════════════════════════════════════
loaded_model = joblib.load('iris_classifier.pkl')
new_sample = [[5.1, 3.5, 1.4, 0.2]]  # Example: likely setosa
prediction = loaded_model.predict(new_sample)
print(f"\nPrediction for {new_sample}: {iris.target_names[prediction[0]]}")
```

## Tips and Best Practices

```
┌────────────────────────────────────────────────────────────┐
│                  BEST PRACTICES CHECKLIST                  │
└────────────────────────────────────────────────────────────┘

DATA PREPARATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
☑ Always split data BEFORE preprocessing
☑ Use stratify=y for imbalanced classification
☑ Handle missing values appropriately
☑ Scale features (especially for SVM, KNN, Linear models)
☑ Encode categorical variables correctly
☑ Check for data leakage

MODEL TRAINING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
☑ Start with simple baseline (DummyClassifier/Regressor)
☑ Use pipelines to prevent data leakage
☑ Apply cross-validation for reliable estimates
☑ Tune hyperparameters systematically
☑ Use appropriate random_state for reproducibility
☑ Consider class imbalance (class_weight='balanced')

MODEL EVALUATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
☑ Choose metrics appropriate for the task
☑ Never evaluate on training data
☑ Use confusion matrix for classification
☑ Check learning curves for overfitting/underfitting
☑ Compare with baseline model
☑ Consider computational cost vs. performance

COMMON PITFALLS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✗ Fitting scaler on entire dataset → Use pipeline
✗ Not using cross-validation → Unreliable estimates
✗ Ignoring class imbalance → Biased model
✗ Over-tuning on test set → Use validation set
✗ Not setting random_state → Non-reproducible results
✗ Forgetting to scale for distance-based models
```

### Key Guidelines

*   **Data Preprocessing:** Always preprocess your data (scaling, encoding, handling missing values) before training a model.
*   **Cross-Validation:** Use cross-validation to get a reliable estimate of your model's performance.
*   **Hyperparameter Tuning:** Use `GridSearchCV` or `RandomizedSearchCV` to find the best hyperparameters for your model.
*   **Pipelines:** Use pipelines to streamline your workflow and prevent data leakage.
*   **Model Persistence:** Save your trained models using `joblib` or `pickle`.
*   **Feature Importance:** Use feature importance techniques to understand which features are most important for your model.
*   **Regularization:** Use regularization techniques (L1, L2) to prevent overfitting.
*   **Ensemble Methods:** Combine multiple models to improve performance.
*   **Choose the Right Model:** Select a model that is appropriate for your data and task (see Model Selection Guide).
*   **Evaluate Your Model:** Use appropriate evaluation metrics for your task.
*   **Understand Your Data:** Spend time exploring and understanding your data before building a model.
*   **Start Simple:** Begin with a simple model and gradually increase complexity.
*   **Iterate:** Machine learning is an iterative process. Experiment with different models, features, and hyperparameters.
*   **Document Your Work:** Keep track of your experiments and results.
*   **Use Version Control:** Use Git to track changes to your code.
*   **Use Virtual Environments:** Isolate project dependencies.
*   **Read the Documentation:** The Scikit-learn documentation is excellent.