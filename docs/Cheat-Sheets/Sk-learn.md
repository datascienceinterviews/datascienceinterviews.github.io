
---
title: Scikit-learn Cheat Sheet
description: A comprehensive reference guide for Scikit-learn, covering data preprocessing, model selection, training, evaluation, and more.
---

# Scikit-learn Cheat Sheet

[TOC]

This cheat sheet provides an exhaustive overview of the Scikit-learn (sklearn) machine learning library, covering essential concepts, code snippets, and best practices for efficient model building, training, evaluation, and deployment. It aims to be a one-stop reference for common tasks.

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

iris = datasets.load_iris()
X = iris.data
y = iris.target

boston = datasets.load_boston() # Now deprecated, use fetch_california_housing
california_housing = datasets.fetch_california_housing()
X = california_housing.data
y = california_housing.target

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
X_test_imputed = imputer.transform(X_test)```

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

### Linear Regression

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='liblinear') # Add solver for smaller datasets
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
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

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
```

### ROC Curve and AUC

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# For binary classification
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

# K-Fold Cross-Validation
cv_scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation

# Stratified K-Fold (for classification)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv)

print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean():.2f}")
```

### Learning Curves

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10))

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, label='Training score')
plt.plot(train_sizes, test_mean, label='Cross-validation score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.legend()
plt.title('Learning Curve')
plt.show()
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

### GridSearchCV

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': [0.1, 1, 'scale', 'auto']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.2f}")
best_model = grid_search.best_estimator_
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

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

## Ensemble Methods

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

### PCA

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
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

## Tips and Best Practices

*   **Data Preprocessing:** Always preprocess your data (scaling, encoding, handling missing values) before training a model.
*   **Cross-Validation:** Use cross-validation to get a reliable estimate of your model's performance.
*   **Hyperparameter Tuning:** Use `GridSearchCV` or `RandomizedSearchCV` to find the best hyperparameters for your model.
*   **Pipelines:** Use pipelines to streamline your workflow and prevent data leakage.
*   **Model Persistence:** Save your trained models using `joblib` or `pickle`.
*   **Feature Importance:** Use feature importance techniques to understand which features are most important for your model.
*   **Regularization:** Use regularization techniques (L1, L2, Dropout) to prevent overfitting.
*   **Ensemble Methods:** Combine multiple models to improve performance.
*   **Choose the Right Model:** Select a model that is appropriate for your data and task.
*   **Evaluate Your Model:** Use appropriate evaluation metrics for your task.
*   **Understand Your Data:** Spend time exploring and understanding your data before building a model.
*   **Start Simple:** Begin with a simple model and gradually increase complexity.
*   **Iterate:** Machine learning is an iterative process. Experiment with different models, features, and hyperparameters.
*   **Document Your Work:** Keep track of your experiments and results.
*   **Use Version Control:** Use Git to track changes to your code.
*   **Use Virtual Environments:** Isolate project dependencies.
*   **Read the Documentation:** The Scikit-learn documentation is excellent.