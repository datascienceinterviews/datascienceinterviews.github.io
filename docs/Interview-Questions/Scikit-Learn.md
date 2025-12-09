
---
title: Scikit-Learn Interview Questions
description: 100+ Scikit-Learn interview questions for cracking Machine Learning, Data Science, and ML Engineer interviews
---

# Scikit-Learn Interview Questions

<!-- [TOC] -->

This document provides a curated list of Scikit-Learn interview questions commonly asked in technical interviews for Machine Learning Engineer, Data Scientist, and AI/ML roles. It covers fundamental concepts to advanced machine learning techniques, model evaluation, and production deployment.

This is updated frequently but right now this is the most exhaustive list of type of questions being asked.

---

| Sno | Question Title | Practice Links | Companies Asking | Difficulty | Topics |
|-----|----------------|----------------|------------------|------------|--------|
| 1 | What is Scikit-Learn and why is it popular? | [Scikit-Learn Docs](https://scikit-learn.org/stable/getting_started.html) | Google, Amazon, Meta, Netflix | Easy | Basics, Introduction |
| 2 | Explain the Scikit-Learn API design (fit, transform, predict) | [Scikit-Learn Docs](https://scikit-learn.org/stable/developers/develop.html) | Google, Amazon, Meta, Microsoft | Easy | API Design, Estimators |
| 3 | What are estimators, transformers, and predictors? | [Scikit-Learn Docs](https://scikit-learn.org/stable/developers/develop.html#estimators) | Google, Amazon, Meta | Easy | Core Concepts |
| 4 | How to split data into train and test sets? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) | Most Tech Companies | Easy | Data Splitting, train_test_split |
| 5 | What is cross-validation and why is it important? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/cross_validation.html) | Google, Amazon, Meta, Netflix, Apple | Medium | Cross-Validation, Model Evaluation |
| 6 | Difference between KFold, StratifiedKFold, GroupKFold | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators) | Google, Amazon, Meta | Medium | Cross-Validation Strategies |
| 7 | How to implement GridSearchCV for hyperparameter tuning? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) | Google, Amazon, Meta, Netflix | Medium | Hyperparameter Tuning |
| 8 | Difference between GridSearchCV and RandomizedSearchCV | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/grid_search.html) | Google, Amazon, Meta | Medium | Hyperparameter Tuning |
| 9 | What is a Pipeline and why should we use it? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/compose.html#pipeline) | Google, Amazon, Meta, Netflix, Apple | Medium | Pipeline, Preprocessing |
| 10 | How to create a custom transformer? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/preprocessing.html#custom-transformers) | Google, Amazon, Meta, Microsoft | Medium | Custom Transformers |
| 11 | Explain StandardScaler vs MinMaxScaler vs RobustScaler | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling) | Google, Amazon, Meta, Netflix | Easy | Feature Scaling |
| 12 | What is feature scaling and when is it necessary? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/preprocessing.html) | Most Tech Companies | Easy | Feature Scaling |
| 13 | How to handle missing values in Scikit-Learn? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/impute.html) | Google, Amazon, Meta, Netflix | Medium | Missing Data, Imputation |
| 14 | Difference between SimpleImputer and IterativeImputer | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/impute.html#iterative-imputer) | Google, Amazon, Meta | Medium | Imputation Strategies |
| 15 | How to encode categorical variables? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features) | Most Tech Companies | Easy | Encoding, Categorical Data |
| 16 | Difference between LabelEncoder and OneHotEncoder | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features) | Google, Amazon, Meta, Netflix | Easy | Categorical Encoding |
| 17 | What is OrdinalEncoder and when to use it? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html) | Google, Amazon, Meta | Easy | Ordinal Encoding |
| 18 | How to implement feature selection? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/feature_selection.html) | Google, Amazon, Meta, Netflix | Medium | Feature Selection |
| 19 | Explain SelectKBest and mutual_info_classif | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html) | Google, Amazon, Meta | Medium | Feature Selection |
| 20 | What is Recursive Feature Elimination (RFE)? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html) | Google, Amazon, Meta, Microsoft | Medium | Feature Selection, RFE |
| 21 | How to implement Linear Regression? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares) | Most Tech Companies | Easy | Linear Regression |
| 22 | What is Ridge Regression and when to use it? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression) | Google, Amazon, Meta, Netflix | Medium | Regularization, Ridge |
| 23 | What is Lasso Regression and when to use it? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/linear_model.html#lasso) | Google, Amazon, Meta, Netflix | Medium | Regularization, Lasso |
| 24 | Difference between Ridge (L2) and Lasso (L1) | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/linear_model.html) | Google, Amazon, Meta, Netflix, Apple | Medium | Regularization |
| 25 | What is ElasticNet regression? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/linear_model.html#elastic-net) | Google, Amazon, Meta | Medium | ElasticNet, Regularization |
| 26 | How to implement Logistic Regression? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) | Most Tech Companies | Easy | Logistic Regression, Classification |
| 27 | Explain the solver options in Logistic Regression | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) | Google, Amazon, Meta | Medium | Optimization Solvers |
| 28 | How to implement Decision Trees? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/tree.html) | Most Tech Companies | Easy | Decision Trees |
| 29 | What are the hyperparameters for Decision Trees? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) | Google, Amazon, Meta, Netflix | Medium | Hyperparameters, Trees |
| 30 | How to implement Random Forest? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/ensemble.html#random-forests) | Most Tech Companies | Medium | Random Forest, Ensemble |
| 31 | Difference between bagging and boosting | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/ensemble.html) | Google, Amazon, Meta, Netflix, Apple | Medium | Ensemble Methods |
| 32 | How to implement Gradient Boosting? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting) | Google, Amazon, Meta, Netflix | Medium | Gradient Boosting |
| 33 | Difference between GradientBoosting and HistGradientBoosting | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting) | Google, Amazon, Meta | Medium | Gradient Boosting Variants |
| 34 | How to implement Support Vector Machines (SVM)? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/svm.html) | Google, Amazon, Meta, Microsoft | Medium | SVM, Classification |
| 35 | Explain different kernel functions in SVM | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/svm.html#kernel-functions) | Google, Amazon, Meta | Medium | SVM Kernels |
| 36 | How to implement K-Nearest Neighbors (KNN)? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/neighbors.html) | Most Tech Companies | Easy | KNN, Classification |
| 37 | What is the curse of dimensionality? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/neighbors.html#curse-of-dimensionality) | Google, Amazon, Meta, Netflix | Medium | Dimensionality, KNN |
| 38 | How to implement Naive Bayes classifiers? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/naive_bayes.html) | Most Tech Companies | Easy | Naive Bayes |
| 39 | Difference between GaussianNB, MultinomialNB, BernoulliNB | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/naive_bayes.html) | Google, Amazon, Meta | Medium | Naive Bayes Variants |
| 40 | How to implement K-Means clustering? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/clustering.html#k-means) | Most Tech Companies | Easy | K-Means, Clustering |
| 41 | How to determine optimal number of clusters? | [Scikit-Learn Docs](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html) | Google, Amazon, Meta, Netflix | Medium | Elbow Method, Silhouette |
| 42 | What is DBSCAN and when to use it? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/clustering.html#dbscan) | Google, Amazon, Meta | Medium | DBSCAN, Clustering |
| 43 | Difference between K-Means and DBSCAN | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/clustering.html) | Google, Amazon, Meta, Netflix | Medium | Clustering Comparison |
| 44 | How to implement Hierarchical Clustering? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering) | Google, Amazon, Meta | Medium | Hierarchical Clustering |
| 45 | How to implement PCA (Principal Component Analysis)? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/decomposition.html#pca) | Google, Amazon, Meta, Netflix, Apple | Medium | PCA, Dimensionality Reduction |
| 46 | How to choose number of components in PCA? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/decomposition.html#pca) | Google, Amazon, Meta, Netflix | Medium | PCA, Variance Explained |
| 47 | What is t-SNE and when to use it? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/manifold.html#t-sne) | Google, Amazon, Meta, Netflix | Medium | t-SNE, Visualization |
| 48 | Difference between PCA and t-SNE | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/manifold.html) | Google, Amazon, Meta | Medium | Dimensionality Reduction |
| 49 | What is accuracy and when is it misleading? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score) | Most Tech Companies | Easy | Metrics, Accuracy |
| 50 | Explain precision, recall, and F1-score | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics) | Google, Amazon, Meta, Netflix, Apple | Medium | Classification Metrics |
| 51 | What is the ROC curve and AUC? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics) | Google, Amazon, Meta, Netflix, Apple | Medium | ROC, AUC |
| 52 | When to use precision vs recall? | [Scikit-Learn Docs](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html) | Google, Amazon, Meta, Netflix | Medium | Metrics Tradeoff |
| 53 | What is the confusion matrix? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix) | Most Tech Companies | Easy | Confusion Matrix |
| 54 | What is mean squared error (MSE) and RMSE? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-error) | Most Tech Companies | Easy | Regression Metrics |
| 55 | What is R² score (coefficient of determination)? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score) | Most Tech Companies | Easy | Regression Metrics |
| 56 | How to handle imbalanced datasets? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html) | Google, Amazon, Meta, Netflix, Apple | Medium | Imbalanced Data, class_weight |
| 57 | What is SMOTE and how does it work? | [Imbalanced-Learn](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html) | Google, Amazon, Meta | Medium | Oversampling, SMOTE |
| 58 | How to implement ColumnTransformer? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/compose.html#columntransformer-for-heterogeneous-data) | Google, Amazon, Meta, Netflix | Medium | Column Transformers |
| 59 | What is FeatureUnion and when to use it? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/compose.html#featureunion-composite-feature-spaces) | Google, Amazon, Meta | Medium | Feature Engineering |
| 60 | How to implement polynomial features? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/preprocessing.html#generating-polynomial-features) | Google, Amazon, Meta | Easy | Polynomial Features |
| 61 | What is learning curve and how to interpret it? | [Scikit-Learn Docs](https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html) | Google, Amazon, Meta, Netflix | Medium | Learning Curves, Diagnostics |
| 62 | What is validation curve? | [Scikit-Learn Docs](https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html) | Google, Amazon, Meta | Medium | Validation Curves |
| 63 | How to save and load models with joblib? | [Scikit-Learn Docs](https://scikit-learn.org/stable/model_persistence.html) | Most Tech Companies | Easy | Model Persistence |
| 64 | What is calibration and why is it important? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/calibration.html) | Google, Amazon, Meta, Netflix | Medium | Probability Calibration |
| 65 | How to use CalibratedClassifierCV? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html) | Google, Amazon, Meta | Medium | Calibration |
| 66 | What is VotingClassifier? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier) | Google, Amazon, Meta | Medium | Ensemble, Voting |
| 67 | What is StackingClassifier? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/ensemble.html#stacked-generalization) | Google, Amazon, Meta, Netflix | Hard | Ensemble, Stacking |
| 68 | How to implement AdaBoost? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/ensemble.html#adaboost) | Google, Amazon, Meta | Medium | AdaBoost, Ensemble |
| 69 | What is BaggingClassifier? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/ensemble.html#bagging-meta-estimator) | Google, Amazon, Meta | Medium | Bagging, Ensemble |
| 70 | How to extract feature importances? | [Scikit-Learn Docs](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html) | Google, Amazon, Meta, Netflix | Medium | Feature Importance |
| 71 | What is permutation importance? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/permutation_importance.html) | Google, Amazon, Meta, Netflix | Medium | Permutation Importance |
| 72 | How to implement multi-class classification? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/multiclass.html) | Most Tech Companies | Medium | Multi-class Classification |
| 73 | What is One-vs-Rest (OvR) strategy? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/multiclass.html#one-vs-the-rest) | Google, Amazon, Meta | Medium | Multiclass Strategies |
| 74 | What is One-vs-One (OvO) strategy? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/multiclass.html#one-vs-one) | Google, Amazon, Meta | Medium | Multiclass Strategies |
| 75 | How to implement multi-label classification? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/multiclass.html#multilabel-classification) | Google, Amazon, Meta, Netflix | Hard | Multi-label Classification |
| 76 | What is MultiOutputClassifier? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) | Google, Amazon, Meta | Medium | Multi-output |
| 77 | How to implement Gaussian Mixture Models (GMM)? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/mixture.html) | Google, Amazon, Meta | Medium | GMM, Clustering |
| 78 | What is Isolation Forest? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/outlier_detection.html#isolation-forest) | Google, Amazon, Meta, Netflix | Medium | Anomaly Detection |
| 79 | How to implement One-Class SVM for anomaly detection? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/outlier_detection.html#one-class-svm) | Google, Amazon, Meta | Medium | Anomaly Detection |
| 80 | What is Local Outlier Factor (LOF)? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/outlier_detection.html#local-outlier-factor) | Google, Amazon, Meta | Medium | Anomaly Detection |
| 81 | How to implement text classification with TF-IDF? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) | Google, Amazon, Meta, Netflix | Medium | Text Classification, TF-IDF |
| 82 | What is CountVectorizer vs TfidfVectorizer? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) | Google, Amazon, Meta, Netflix | Easy | Text Vectorization |
| 83 | How to use HashingVectorizer for large datasets? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/feature_extraction.html#vectorizing-a-large-text-corpus-with-the-hashing-trick) | Google, Amazon, Meta | Hard | Large-scale Text |
| 84 | What is SGDClassifier and when to use it? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/sgd.html) | Google, Amazon, Meta, Netflix | Medium | Online Learning, SGD |
| 85 | How to implement partial_fit for online learning? | [Scikit-Learn Docs](https://scikit-learn.org/stable/computing/scaling_strategies.html#incremental-learning) | Google, Amazon, Meta, Netflix | Hard | Online Learning |
| 86 | What is MLPClassifier for neural networks? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/neural_networks_supervised.html) | Google, Amazon, Meta | Medium | Neural Networks |
| 87 | How to set random_state for reproducibility? | [Scikit-Learn Docs](https://scikit-learn.org/stable/common_pitfalls.html#randomness) | Most Tech Companies | Easy | Reproducibility |
| 88 | What is make_pipeline vs Pipeline? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html) | Google, Amazon, Meta | Easy | Pipeline |
| 89 | How to get prediction probabilities? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.predict_proba) | Most Tech Companies | Easy | Probabilities |
| 90 | What is decision_function vs predict_proba? | [Scikit-Learn Docs](https://scikit-learn.org/stable/glossary.html#term-decision_function) | Google, Amazon, Meta | Medium | Prediction Methods |
| 91 | **[HARD]** How to implement custom scoring functions for GridSearchCV? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/model_evaluation.html#defining-your-scoring-strategy-from-metric-functions) | Google, Amazon, Meta, Netflix | Hard | Custom Metrics |
| 92 | **[HARD]** How to implement time series cross-validation (TimeSeriesSplit)? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) | Google, Amazon, Netflix, Apple | Hard | Time Series CV |
| 93 | **[HARD]** How to implement nested cross-validation? | [Scikit-Learn Docs](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html) | Google, Amazon, Meta | Hard | Nested CV, Model Selection |
| 94 | **[HARD]** How to optimize memory with sparse matrices? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/feature_extraction.html) | Google, Amazon, Meta, Netflix | Hard | Sparse Matrices, Memory |
| 95 | **[HARD]** How to implement custom transformers with TransformerMixin? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/preprocessing.html#custom-transformers) | Google, Amazon, Meta, Microsoft | Hard | Custom Transformers |
| 96 | **[HARD]** How to implement custom estimators with BaseEstimator? | [Scikit-Learn Docs](https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator) | Google, Amazon, Meta | Hard | Custom Estimators |
| 97 | **[HARD]** How to optimize hyperparameters with Bayesian optimization? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingGridSearchCV.html) | Google, Amazon, Meta, Netflix | Hard | Hyperparameter Optimization |
| 98 | **[HARD]** How to implement stratified sampling for imbalanced regression? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/cross_validation.html) | Google, Amazon, Meta | Hard | Stratified Sampling |
| 99 | **[HARD]** How to implement target encoding without data leakage? | [Category Encoders](https://contrib.scikit-learn.org/category_encoders/) | Google, Amazon, Meta, Netflix | Hard | Target Encoding, Leakage |
| 100| **[HARD]** How to implement cross-validation with grouped data? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html) | Google, Amazon, Meta, Netflix | Hard | GroupKFold, Data Leakage |
| 101 | **[HARD]** How to implement feature selection with embedded methods? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection-using-selectfrommodel) | Google, Amazon, Meta | Hard | Feature Selection |
| 102 | **[HARD]** How to handle high-cardinality categorical features? | [Stack Overflow](https://stackoverflow.com/questions/62006247/how-to-handle-high-cardinality-categorical-features) | Google, Amazon, Meta, Netflix | Hard | High Cardinality |
| 103 | **[HARD]** How to implement model interpretability with SHAP values? | [SHAP Docs](https://shap.readthedocs.io/) | Google, Amazon, Meta, Netflix, Apple | Hard | Model Interpretability, SHAP |
| 104 | **[HARD]** How to implement multivariate time series forecasting? | [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/multioutput.html) | Google, Amazon, Netflix | Hard | Time Series, Multi-output |
| 105 | **[HARD]** How to handle concept drift in production models? | [Towards Data Science](https://towardsdatascience.com/) | Google, Amazon, Meta, Netflix | Hard | Concept Drift, MLOps |
| 106 | **[HARD]** How to implement model monitoring for production? | [MLflow Docs](https://mlflow.org/) | Google, Amazon, Meta, Netflix, Apple | Hard | Model Monitoring, MLOps |
| 107 | **[HARD]** How to optimize inference latency for real-time predictions? | [Scikit-Learn Docs](https://scikit-learn.org/stable/computing/computational_performance.html) | Google, Amazon, Meta, Netflix | Hard | Latency, Performance |
| 108 | **[HARD]** How to implement A/B testing for model comparison? | [Towards Data Science](https://towardsdatascience.com/) | Google, Amazon, Meta, Netflix | Hard | A/B Testing, Experimentation |
| 109 | **[HARD]** How to handle data leakage in feature engineering? | [Kaggle](https://www.kaggle.com/learn/feature-engineering) | Google, Amazon, Meta, Netflix, Apple | Hard | Data Leakage, Feature Engineering |
| 110 | **[HARD]** How to implement model versioning and tracking? | [MLflow Docs](https://mlflow.org/) | Google, Amazon, Meta, Netflix | Hard | Model Versioning, MLOps |

---

## Code Examples

### 1. Building a Custom Transformer
```python
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
        
    def fit(self, X, y=None):
        self.Q1 = X.quantile(0.25)
        self.Q3 = X.quantile(0.75)
        self.IQR = self.Q3 - self.Q1
        return self
    
    def transform(self, X):
        return X[~((X < (self.Q1 - self.factor * self.IQR)) | 
                   (X > (self.Q3 + self.factor * self.IQR))).any(axis=1)]
```

### 2. Nested Cross-Validation
```python
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.svm import SVC
import numpy as np

# Inner loop for hyperparameter tuning
p_grid = {"C": [1, 10, 100], "gamma": [0.01, 0.1]}
svm = SVC(kernel="rbf")
inner_cv = KFold(n_splits=4, shuffle=True, random_state=1)
clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv)

# Outer loop for model evaluation
outer_cv = KFold(n_splits=4, shuffle=True, random_state=1)
nested_score = cross_val_score(clf, X_iris, y_iris, cv=outer_cv)

print(f"Nested CV Score: {nested_score.mean():.3f} +/- {nested_score.std():.3f}")
```

### 3. Pipeline with ColumnTransformer
```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_features = ['age', 'fare']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['embarked', 'sex', 'pclass']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression())])
```

---

## Questions asked in Google interview
- How would you implement a custom loss function in Scikit-Learn?
- Explain how to handle data leakage in cross-validation
- Write code to implement nested cross-validation with hyperparameter tuning
- How would you optimize a model for minimal inference latency?
- Explain the bias-variance tradeoff with specific examples
- How would you implement model calibration for probability estimates?
- Write code to implement stratified sampling for imbalanced multi-class
- How would you handle concept drift in production ML systems?
- Explain how to implement feature importance with SHAP values
- How would you optimize memory for large sparse datasets?

## Questions asked in Amazon interview
- Write code to implement a complete ML pipeline for customer churn
- How would you handle high-cardinality categorical features?
- Explain the difference between different cross-validation strategies
- Write code to implement time series cross-validation
- How would you implement model monitoring in production?
- Explain how to handle missing data in production systems
- Write code to implement custom scoring functions
- How would you implement A/B testing for model comparison?
- Explain how to optimize hyperparameters efficiently
- How would you handle data leakage in feature engineering?

## Questions asked in Meta interview
- Write code to implement user engagement prediction pipeline
- How would you implement multi-label classification for content tagging?
- Explain how to handle extremely imbalanced datasets
- Write code to implement custom transformers for text features
- How would you implement feature selection for high-dimensional data?
- Explain how to implement model interpretability
- Write code to implement online learning with partial_fit
- How would you implement model calibration?
- Explain how to prevent overfitting in ensemble models
- How would you implement multivariate predictions?

## Questions asked in Microsoft interview
- Explain the Scikit-Learn estimator API design principles
- Write code to implement custom estimators extending BaseEstimator
- How would you implement regularization selection?
- Explain the differences between solver options in LogisticRegression
- Write code to implement feature engineering pipelines
- How would you optimize model training time?
- Explain how to implement model persistence correctly
- Write code to implement cross-validation with custom folds
- How would you handle numerical stability issues?
- Explain how to implement reproducible ML experiments

## Questions asked in Netflix interview
- Write code to implement recommendation feature engineering
- How would you implement content classification at scale?
- Explain how to handle user behavior data for ML
- Write code to implement streaming quality prediction
- How would you implement real-time inference optimization?
- Explain how to implement model monitoring and retraining
- Write code to implement cohort-based model evaluation
- How would you handle seasonality in user data?
- Explain how to implement A/B testing for ML models
- How would you implement customer lifetime value prediction?

## Questions asked in Apple interview
- Write code to implement privacy-preserving ML pipelines
- How would you implement on-device ML model optimization?
- Explain how to handle sensor data for ML
- Write code to implement quality control classification
- How would you implement model quantization for deployment?
- Explain best practices for production ML systems
- Write code to implement automated model retraining
- How would you handle data versioning?
- Explain how to implement cross-platform model deployment
- How would you implement model security?

---

## Additional Resources

- [Official Scikit-Learn Documentation](https://scikit-learn.org/stable/)
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
- [Introduction to Machine Learning with Python](https://www.oreilly.com/library/view/introduction-to-machine/9781449369880/)
- [Scikit-Learn Course by Andreas Mueller](https://github.com/amueller/ml-workshop-1-of-4)
