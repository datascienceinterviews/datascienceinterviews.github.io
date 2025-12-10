
---
title: Scikit-Learn Interview Questions
description: 100+ Scikit-Learn interview questions for cracking Machine Learning, Data Science, and ML Engineer interviews
---

# Scikit-Learn Interview Questions

<!-- [TOC] -->

This document provides a curated list of Scikit-Learn interview questions commonly asked in technical interviews for Machine Learning Engineer, Data Scientist, and AI/ML roles. It covers fundamental concepts to advanced machine learning techniques, model evaluation, and production deployment.

This is updated frequently but right now this is the most exhaustive list of type of questions being asked.

---

## Premium Interview Questions

### Explain the Scikit-Learn Estimator API - Google, Amazon Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `API Design`, `Core Concepts` | **Asked by:** Google, Amazon, Meta, Netflix

??? success "View Answer"

    **The Estimator Interface:**
    
    Scikit-Learn uses a consistent API across all algorithms:
    
    ```python
    from sklearn.ensemble import RandomForestClassifier
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)      # Learn from data
    predictions = model.predict(X_test)  # Make predictions
    accuracy = model.score(X_test, y_test)  # Evaluate
    ```
    
    **Three Types:** Estimator (`fit()`), Predictor (`fit()`, `predict()`), Transformer (`fit()`, `transform()`).
    
    Learned attributes end with underscore: `model.feature_importances_`, `model.coef_`.

    !!! tip "Interviewer's Insight"
        Knows fit/predict/transform pattern and underscore convention for learned attributes.

---

### How to Create an Sklearn Pipeline? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Pipeline`, `Best Practices` | **Asked by:** Google, Amazon, Meta, Netflix

??? success "View Answer"

    ```python
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier())
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    ```
    
    **Benefits:** Prevents data leakage, simplifies code, easy to deploy.

    !!! tip "Interviewer's Insight"
        Uses Pipeline to prevent data leakage and knows ColumnTransformer for mixed types.

---

### Explain Cross-Validation Strategies - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Model Evaluation` | **Asked by:** Google, Amazon, Meta, Netflix

??? success "View Answer"

    ```python
    from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit
    
    # KFold: General, StratifiedKFold: Imbalanced, GroupKFold: Grouped data, TimeSeriesSplit: Time series
    scores = cross_val_score(model, X, y, cv=StratifiedKFold(5), scoring='f1')
    ```
    
    **Choose based on:** Imbalanced classes → StratifiedKFold, Groups → GroupKFold, Time series → TimeSeriesSplit.

    !!! tip "Interviewer's Insight"
        Chooses appropriate CV for data type and knows data leakage risks.

---

### How to Handle Class Imbalance? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Imbalanced Data` | **Asked by:** Google, Amazon, Meta, Netflix

??? success "View Answer"

    ```python
    # Class weights
    model = RandomForestClassifier(class_weight='balanced')
    
    # SMOTE (from imblearn)
    from imblearn.over_sampling import SMOTE
    X_resampled, y_resampled = SMOTE().fit_resample(X, y)
    ```
    
    **Metrics:** Use precision, recall, F1, ROC-AUC instead of accuracy.

    !!! tip "Interviewer's Insight"
        Uses class_weight parameter, knows SMOTE, uses appropriate metrics.

---

### Explain GridSearchCV vs RandomizedSearchCV - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Hyperparameter Tuning` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    ```python
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    
    # GridSearchCV: Exhaustive (slow)
    grid = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    
    # RandomizedSearchCV: Sampled (faster for large spaces)
    random = RandomizedSearchCV(model, param_dist, n_iter=50, cv=5)
    ```
    
    Grid for small spaces, Random for large continuous spaces.

    !!! tip "Interviewer's Insight"
        Uses RandomizedSearchCV for large spaces and scipy distributions.

---

### How to Create a Custom Transformer? - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Custom Transformers` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    ```python
    from sklearn.base import BaseEstimator, TransformerMixin
    
    class LogTransformer(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            return np.log1p(X)
    ```
    
    Inherit from BaseEstimator and TransformerMixin. Return `self` in `fit()`.

    !!! tip "Interviewer's Insight"
        Inherits from correct base classes and implements get_feature_names_out.

---

### Explain Feature Scaling Methods - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Preprocessing` | **Asked by:** Most Tech Companies

??? success "View Answer"

    | Scaler | Formula | Use Case |
    |--------|---------|----------|
    | StandardScaler | $(x - \mu) / \sigma$ | Most algorithms |
    | MinMaxScaler | $(x - min) / (max - min)$ | Neural networks |
    | RobustScaler | Uses median/IQR | Data with outliers |
    
    **Important:** Fit on train, transform test to avoid data leakage.

    !!! tip "Interviewer's Insight"
        Knows data leakage prevention and chooses appropriate scaler.

---

### How to Evaluate Classification Models? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Metrics` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    ```python
    from sklearn.metrics import classification_report, roc_auc_score
    
    print(classification_report(y_test, predictions))
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    ```
    
    | Metric | Use When |
    |--------|----------|
    | Precision | Minimize false positives |
    | Recall | Minimize false negatives |
    | F1 | Balance precision/recall |

    !!! tip "Interviewer's Insight"
        Chooses metrics based on business context.

---

### Explain Ridge vs Lasso vs ElasticNet - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Regularization` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    | Method | Penalty | Effect |
    |--------|---------|--------|
    | Ridge (L2) | $\sum w^2$ | Shrinks weights |
    | Lasso (L1) | $\sum |w|$ | Feature selection (sparse) |
    | ElasticNet | Both | Combines benefits |
    
    Use RidgeCV/LassoCV for automatic alpha selection.

    !!! tip "Interviewer's Insight"
        Explains L1 sparsity property and uses CV versions.

---

### How to Implement Feature Selection? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Feature Selection` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    ```python
    from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel
    
    # Filter: SelectKBest(k=10)
    # Wrapper: RFE(model, n_features_to_select=10)
    # Embedded: SelectFromModel(RandomForestClassifier())
    ```
    
    Filter is fast, Wrapper is thorough but slow, Embedded uses model importance.

    !!! tip "Interviewer's Insight"
        Knows filter/wrapper/embedded distinction and computational tradeoffs.

---

### How to Save and Load Models? - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Deployment` | **Asked by:** Most Tech Companies

??? success "View Answer"

    ```python
    import joblib
    
    # Save
    joblib.dump(pipeline, 'model.joblib')
    
    # Load
    loaded_model = joblib.load('model.joblib')
    ```
    
    Use joblib for efficiency. Save version info with model for compatibility.

    !!! tip "Interviewer's Insight"
        Uses joblib, saves metadata, knows security concerns with pickle.

---

### Explain Random Forest Feature Importance - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Interpretability` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    ```python
    # Built-in (MDI) - biased to high-cardinality
    importances = model.feature_importances_
    
    # Permutation (more reliable)
    from sklearn.inspection import permutation_importance
    result = permutation_importance(model, X_test, y_test, n_repeats=10)
    ```
    
    Permutation importance is unbiased and computed on test data.

    !!! tip "Interviewer's Insight"
        Knows MDI bias and uses permutation importance for reliability.

---

### How to Use VotingClassifier? - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Ensemble` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    ```python
    from sklearn.ensemble import VotingClassifier
    
    voting = VotingClassifier([
        ('rf', RandomForestClassifier()),
        ('lr', LogisticRegression()),
        ('svc', SVC(probability=True))
    ], voting='soft')  # 'hard' for majority vote
    ```
    
    Soft voting averages probabilities, hard voting uses majority.

    !!! tip "Interviewer's Insight"
        Knows soft vs hard voting and stacking vs voting differences.

---

### How to Detect Overfitting? - Most Tech Companies Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Model Selection` | **Asked by:** Most Tech Companies

??? success "View Answer"

    ```python
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, val_scores = learning_curve(model, X, y, cv=5)
    # High train, low val = overfit
    # Low train, low val = underfit
    ```
    
    Solutions: More data, regularization, simpler model, cross-validation.

    !!! tip "Interviewer's Insight"
        Uses learning curves and knows multiple mitigation strategies.

---

### How to Handle Missing Values? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Imputation` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    ```python
    from sklearn.impute import SimpleImputer, KNNImputer
    
    # SimpleImputer: mean, median, most_frequent, constant
    imputer = SimpleImputer(strategy='median', add_indicator=True)
    
    # KNNImputer: uses nearest neighbors
    imputer = KNNImputer(n_neighbors=5)
    ```

    !!! tip "Interviewer's Insight"
        Uses appropriate strategy and add_indicator for missingness patterns.

---

### How to Debug a Failing Model? - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Debugging` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Checklist:**
    1. Check data quality (nulls, shapes, distributions)
    2. Check for data leakage
    3. Compare to baseline (DummyClassifier)
    4. Analyze learning curves
    5. Error analysis on misclassified samples
    
    ```python
    from sklearn.dummy import DummyClassifier
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train, y_train)
    print(f"Baseline: {dummy.score(X_test, y_test)}")
    ```

    !!! tip "Interviewer's Insight"
        Uses systematic approach and compares to dummy baseline.

---

### Explain probability calibration - Google, Netflix Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Calibration` | **Asked by:** Google, Netflix, Stripe

??? success "View Answer"

    ```python
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve
    
    # Calibrate model
    calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=5)
    
    # Check calibration
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
    ```
    
    Naive Bayes and SVM typically need calibration. Logistic Regression is usually calibrated.

    !!! tip "Interviewer's Insight"
        Knows which models need calibration and uses calibration curves.

---

### How to use ColumnTransformer? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Preprocessing` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    ```python
    from sklearn.compose import ColumnTransformer
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['age', 'income']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['city'])
    ], remainder='passthrough')
    ```
    
    Handles different preprocessing for different column types.

    !!! tip "Interviewer's Insight"
        Uses handle_unknown='ignore' and remainder parameter correctly.

---

### How to implement multi-label classification? - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Multi-Label` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    ```python
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.preprocessing import MultiLabelBinarizer
    
    # Binarize labels
    mlb = MultiLabelBinarizer()
    y_binary = mlb.fit_transform(y_multilabel)
    
    # Wrapper classifier
    multi = MultiOutputClassifier(RandomForestClassifier())
    multi.fit(X, y_binary)
    ```
    
    Metrics: hamming_loss, f1_score(average='samples')

    !!! tip "Interviewer's Insight"
        Distinguishes multi-label from multi-class and uses appropriate metrics.

---

### How to use make_scorer? - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Custom Metrics` | **Asked by:** Google, Amazon

??? success "View Answer"

    ```python
    from sklearn.metrics import make_scorer, fbeta_score
    
    # Custom scorer
    f2_scorer = make_scorer(fbeta_score, beta=2)
    
    # Business metric
    def profit_metric(y_true, y_pred):
        return (y_true == y_pred).sum() * 100
    
    profit_scorer = make_scorer(profit_metric, greater_is_better=True)
    
    # Use in GridSearchCV
    GridSearchCV(model, params, scoring=profit_scorer)
    ```

    !!! tip "Interviewer's Insight"
        Creates business-specific metrics and handles greater_is_better.

---

### How to perform polynomial regression? - Most Tech Companies Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Regression` | **Asked by:** Most Tech Companies

??? success "View Answer"

    ```python
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Pipeline
    poly_pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('reg', LinearRegression())
    ])
    ```

    !!! tip "Interviewer's Insight"
        Uses PolynomialFeatures in pipeline and knows include_bias parameter.

---

### How to compute learning curves? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Diagnostics` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    ```python
    from sklearn.model_selection import learning_curve
    import numpy as np
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring='accuracy'
    )
    
    # Plot train and val means to diagnose over/underfitting
    ```

    !!! tip "Interviewer's Insight"
        Uses learning curves for bias-variance diagnosis.

---

### How to use SMOTE for imbalanced data? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Imbalanced Data` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    ```python
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    
    # Resample
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    # Pipeline (use imblearn Pipeline!)
    pipeline = ImbPipeline([
        ('smote', SMOTE()),
        ('classifier', RandomForestClassifier())
    ])
    ```

    !!! tip "Interviewer's Insight"
        Uses imblearn Pipeline and applies SMOTE only on training data.

---

### How to perform stratified sampling? - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Data Splitting` | **Asked by:** Most Tech Companies

??? success "View Answer"

    ```python
    from sklearn.model_selection import train_test_split
    
    # Stratified split (maintains class proportions)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    ```

    !!! tip "Interviewer's Insight"
        Uses stratify parameter for imbalanced classification.

---

### How to tune hyperparameters with Optuna/HalvingGridSearch? - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Hyperparameter Tuning` | **Asked by:** Google, Amazon

??? success "View Answer"

    ```python
    from sklearn.experimental import enable_halving_search_cv
    from sklearn.model_selection import HalvingGridSearchCV
    
    # Successive halving (faster)
    halving = HalvingGridSearchCV(model, param_grid, cv=5, factor=2)
    
    # Optuna integration
    import optuna
    def objective(trial):
        params = {'n_estimators': trial.suggest_int('n_estimators', 50, 500)}
        model = RandomForestClassifier(**params)
        return cross_val_score(model, X, y, cv=5).mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    ```

    !!! tip "Interviewer's Insight"
        Knows HalvingGridSearchCV and Optuna for efficient search.
---

### How to implement SVM classification? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `SVM`, `Classification` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    ```python
    from sklearn.svm import SVC, LinearSVC
    
    # RBF kernel (non-linear)
    svc = SVC(kernel='rbf', C=1.0, gamma='scale')
    
    # Linear (faster for large datasets)
    linear_svc = LinearSVC(C=1.0, max_iter=1000)
    
    # For probabilities (slower)
    svc_proba = SVC(probability=True)
    ```
    
    **Kernels:** linear, poly, rbf, sigmoid. Use rbf for most problems.

    !!! tip "Interviewer's Insight"
        Uses LinearSVC for large datasets and knows kernel selection.

---

### How to implement K-Means clustering? - Most Tech Companies Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Clustering` | **Asked by:** Most Tech Companies

??? success "View Answer"

    ```python
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # Evaluate
    inertia = kmeans.inertia_  # Within-cluster sum of squares
    silhouette = silhouette_score(X, labels)  # [-1, 1]
    ```
    
    Use elbow method (inertia) or silhouette to choose k.

    !!! tip "Interviewer's Insight"
        Uses k-means++ initialization and knows evaluation metrics.

---

### How to implement PCA? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Dimensionality Reduction` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    ```python
    from sklearn.decomposition import PCA
    
    # Reduce to n components
    pca = PCA(n_components=50)
    X_reduced = pca.fit_transform(X)
    
    # Keep 95% variance
    pca = PCA(n_components=0.95)
    
    # Explained variance
    print(pca.explained_variance_ratio_.cumsum())
    ```

    !!! tip "Interviewer's Insight"
        Uses variance ratio for component selection and knows when to use PCA.

---

### How to implement Gradient Boosting? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Ensemble` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    ```python
    from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
    
    # Standard (slower)
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    
    # Histogram-based (faster, handles missing values)
    hgb = HistGradientBoostingClassifier()  # Native NA handling
    ```
    
    For large data, use HistGradientBoosting or XGBoost/LightGBM.

    !!! tip "Interviewer's Insight"
        Knows HistGradientBoosting advantages and when to use external libraries.

---

### How to implement Naive Bayes? - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Classification` | **Asked by:** Most Tech Companies

??? success "View Answer"

    ```python
    from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
    
    # GaussianNB: continuous features (assumes normal distribution)
    gnb = GaussianNB()
    
    # MultinomialNB: text/count data
    mnb = MultinomialNB()
    
    # BernoulliNB: binary features
    bnb = BernoulliNB()
    ```
    
    Fast, good baseline, works well for text classification.

    !!! tip "Interviewer's Insight"
        Chooses appropriate variant for data type.

---

### How to implement DBSCAN clustering? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Clustering` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    ```python
    from sklearn.cluster import DBSCAN
    
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(X)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    ```
    
    **Advantages:** Finds arbitrary shaped clusters, handles noise (-1 labels).

    !!! tip "Interviewer's Insight"
        Knows DBSCAN doesn't need k, handles outliers, and tunes eps.

---

### How to implement t-SNE? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Visualization` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    ```python
    from sklearn.manifold import TSNE
    
    # Reduce to 2D for visualization
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_embedded = tsne.fit_transform(X)
    
    # Note: fit_transform only, no separate transform!
    ```
    
    **Caution:** Slow, only for visualization, non-deterministic, no out-of-sample.

    !!! tip "Interviewer's Insight"
        Knows t-SNE limitations and uses UMAP for speed/quality.

---

### How to implement KNN? - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Classification`, `Regression` | **Asked by:** Most Tech Companies

??? success "View Answer"

    ```python
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')
    knn.fit(X_train, y_train)
    
    # For large datasets, use ball_tree or kd_tree
    knn = KNeighborsClassifier(algorithm='ball_tree')
    ```
    
    Scale features first! KNN is sensitive to feature scales.

    !!! tip "Interviewer's Insight"
        Scales features and knows algorithm options for large data.

---

### How to implement Isolation Forest? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Anomaly Detection` | **Asked by:** Google, Amazon, Netflix

??? success "View Answer"

    ```python
    from sklearn.ensemble import IsolationForest
    
    iso = IsolationForest(contamination=0.1, random_state=42)
    predictions = iso.fit_predict(X)  # -1 for anomalies, 1 for normal
    
    # Anomaly scores
    scores = iso.decision_function(X)  # Lower = more anomalous
    ```
    
    **Advantages:** No need for labels, works on high-dimensional data.

    !!! tip "Interviewer's Insight"
        Uses contamination parameter and understands isolation concept.

---

### How to implement Label Propagation? - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Semi-Supervised` | **Asked by:** Google, Amazon

??? success "View Answer"

    ```python
    from sklearn.semi_supervised import LabelPropagation, LabelSpreading
    
    # -1 indicates unlabeled samples
    y_train = np.array([0, 1, 1, -1, -1, -1, 0, -1])
    
    lp = LabelPropagation()
    lp.fit(X_train, y_train)
    predicted_labels = lp.transduction_
    ```
    
    Uses graph-based approach to propagate labels to unlabeled samples.

    !!! tip "Interviewer's Insight"
        Knows semi-supervised learning use case (few labels, many unlabeled).

---

### How to implement One-Class SVM? - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Anomaly Detection` | **Asked by:** Google, Amazon, Netflix

??? success "View Answer"

    ```python
    from sklearn.svm import OneClassSVM
    
    # Train on normal data only
    ocsvm = OneClassSVM(kernel='rbf', nu=0.1)
    ocsvm.fit(X_normal)
    
    # Predict
    predictions = ocsvm.predict(X_test)  # -1 for anomalies
    ```
    
    **nu:** Upper bound on fraction of outliers.

    !!! tip "Interviewer's Insight"
        Uses for novelty detection (trained on normal only).

---

### How to implement target encoding? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Feature Engineering` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    ```python
    from sklearn.preprocessing import TargetEncoder
    
    # Encode categorical with target mean
    encoder = TargetEncoder(smooth='auto')
    X_encoded = encoder.fit_transform(X[['category']], y)
    
    # Cross-fit to prevent leakage
    encoder = TargetEncoder(cv=5)
    ```
    
    **Caution:** Can cause leakage if not cross-fitted properly.

    !!! tip "Interviewer's Insight"
        Uses cross-validation to prevent target leakage.

---

### How to compute partial dependence plots? - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Interpretability` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    ```python
    from sklearn.inspection import PartialDependenceDisplay, partial_dependence
    
    # Compute
    features = [0, 1, (0, 1)]  # Feature indices
    pdp = PartialDependenceDisplay.from_estimator(model, X, features)
    
    # Or get raw values
    results = partial_dependence(model, X, features=[0])
    ```
    
    Shows marginal effect of feature on prediction.

    !!! tip "Interviewer's Insight"
        Uses for model explanation and understanding feature effects.

---

### How to implement stratified group split? - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Cross-Validation` | **Asked by:** Google, Amazon

??? success "View Answer"

    ```python
    from sklearn.model_selection import StratifiedGroupKFold
    
    # Stratified by y, no group leakage
    sgkf = StratifiedGroupKFold(n_splits=5)
    for train_idx, test_idx in sgkf.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
    ```
    
    Use when you have groups AND imbalanced classes.

    !!! tip "Interviewer's Insight"
        Knows when to combine stratification with grouping.

---

### How to implement validation curves? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Diagnostics` | **Asked by:** Google, Amazon

??? success "View Answer"

    ```python
    from sklearn.model_selection import validation_curve
    
    param_range = [1, 5, 10, 50, 100]
    train_scores, val_scores = validation_curve(
        model, X, y,
        param_name='n_estimators',
        param_range=param_range,
        cv=5
    )
    ```
    
    Shows how one hyperparameter affects train/val performance.

    !!! tip "Interviewer's Insight"
        Uses to check overfitting vs hyperparameter value.

---

### How to implement decision boundary visualization? - Most Tech Companies Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Visualization` | **Asked by:** Most Tech Companies

??? success "View Answer"

    ```python
    from sklearn.inspection import DecisionBoundaryDisplay
    
    # For 2D data
    DecisionBoundaryDisplay.from_estimator(
        model, X[:, :2], ax=ax,
        response_method='predict',
        cmap=plt.cm.RdYlBu
    )
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='black')
    ```

    !!! tip "Interviewer's Insight"
        Uses for model explanation in 2D feature space.

---

### How to implement neural network classifier? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Neural Networks` | **Asked by:** Google, Amazon

??? success "View Answer"

    ```python
    from sklearn.neural_network import MLPClassifier
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        max_iter=500,
        early_stopping=True
    )
    ```
    
    For serious deep learning, use PyTorch/TensorFlow instead.

    !!! tip "Interviewer's Insight"
        Knows sklearn MLP limitations vs deep learning frameworks.

---

### How to implement threshold tuning? - Google, Netflix Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Optimization` | **Asked by:** Google, Netflix, Stripe

??? success "View Answer"

    ```python
    from sklearn.metrics import precision_recall_curve
    
    # Get probabilities
    probas = model.predict_proba(X_test)[:, 1]
    
    # Find optimal threshold for F1
    precisions, recalls, thresholds = precision_recall_curve(y_test, probas)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Apply threshold
    predictions = (probas >= optimal_threshold).astype(int)
    ```

    !!! tip "Interviewer's Insight"
        Knows default 0.5 threshold is often suboptimal.

---

### How to implement cost-sensitive classification? - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Imbalanced Data` | **Asked by:** Google, Amazon

??? success "View Answer"

    ```python
    # Using sample_weight
    weights = np.where(y == 1, 10, 1)  # Weight positive class more
    model.fit(X, y, sample_weight=weights)
    
    # Using class_weight
    model = LogisticRegression(class_weight={0: 1, 1: 10})
    
    # Custom business loss
    def business_cost(y_true, y_pred):
        fp_cost = 10
        fn_cost = 100
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        return fp * fp_cost + fn * fn_cost
    ```

    !!! tip "Interviewer's Insight"
        Uses sample_weight for business-specific costs.

---

### How to implement LeaveOneOut CV? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Cross-Validation` | **Asked by:** Google, Amazon

??? success "View Answer"

    ```python
    from sklearn.model_selection import LeaveOneOut, cross_val_score
    
    loo = LeaveOneOut()
    scores = cross_val_score(model, X, y, cv=loo)
    
    # Computationally expensive! n folds for n samples
    # Use for small datasets only
    ```

    !!! tip "Interviewer's Insight"
        Knows LOO is for small datasets and its variance characteristics.

---

### How to implement confusion matrix visualization? - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Visualization` | **Asked by:** Most Tech Companies

??? success "View Answer"

    ```python
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(cm, display_labels=model.classes_)
    disp.plot(cmap='Blues')
    
    # Or directly
    ConfusionMatrixDisplay.from_predictions(y_test, predictions)
    ```

    !!! tip "Interviewer's Insight"
        Uses visualization for clear communication of results.

---

### How to implement precision-recall curves? - Google, Netflix Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Evaluation` | **Asked by:** Google, Netflix

??? success "View Answer"

    ```python
    from sklearn.metrics import PrecisionRecallDisplay
    
    # From estimator
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
    
    # From predictions
    PrecisionRecallDisplay.from_predictions(y_test, probas)
    
    # Average Precision
    from sklearn.metrics import average_precision_score
    ap = average_precision_score(y_test, probas)
    ```

    !!! tip "Interviewer's Insight"
        Uses PR curves for imbalanced data instead of ROC.

---

### How to implement model calibration check? - Google, Netflix Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Calibration` | **Asked by:** Google, Netflix

??? success "View Answer"

    ```python
    from sklearn.calibration import CalibrationDisplay
    
    # Compare calibration of multiple models
    fig, ax = plt.subplots()
    CalibrationDisplay.from_estimator(model1, X_test, y_test, ax=ax, name='RF')
    CalibrationDisplay.from_estimator(model2, X_test, y_test, ax=ax, name='LR')
    ```

    !!! tip "Interviewer's Insight"
        Compares calibration across models for probability quality.

---

### How to implement cross_validate for multiple metrics? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Evaluation` | **Asked by:** Google, Amazon

??? success "View Answer"

    ```python
    from sklearn.model_selection import cross_validate
    
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    results = cross_validate(model, X, y, cv=5, scoring=scoring, return_train_score=True)
    
    for metric in scoring:
        print(f"{metric}: {results[f'test_{metric}'].mean():.3f}")
    ```

    !!! tip "Interviewer's Insight"
        Evaluates multiple metrics in one call efficiently.

---

## Quick Reference: 100+ Interview Questions

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
