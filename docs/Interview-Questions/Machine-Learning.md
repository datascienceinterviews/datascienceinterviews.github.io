---
title: Machine Learning Interview Questions
description: A curated list of 100+ Machine Learning interview questions for cracking data science interviews at top tech companies
---

# Machine Learning Interview Questions

This comprehensive guide contains **100+ Machine Learning interview questions** commonly asked at top tech companies like Google, Amazon, Meta, Microsoft, and Netflix. Each premium question includes detailed explanations, code examples, and interviewer insights to help you ace your ML interviews.

---

## Premium Interview Questions

Master these frequently asked ML questions with detailed explanations, code examples, and insights into what interviewers really look for.

---

### What is the Bias-Variance Tradeoff in Machine Learning? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Model Evaluation`, `Generalization`, `Fundamentals` | **Asked by:** Google, Amazon, Meta, Microsoft

??? success "View Answer"

    **The Core Concept:**
    
    The bias-variance tradeoff is a fundamental concept that describes the tension between two sources of error in machine learning models:
    
    - **Bias**: Error from overly simplistic assumptions. High bias → underfitting.
    - **Variance**: Error from sensitivity to training data fluctuations. High variance → overfitting.
    
    **Mathematical Formulation:**
    
    For a model's expected prediction error:
    
    $$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$
    
    $$E[(y - \hat{f}(x))^2] = \text{Bias}[\hat{f}(x)]^2 + \text{Var}[\hat{f}(x)] + \sigma^2$$
    
    **Visual Understanding:**
    
    | Model Complexity | Bias | Variance | Result |
    |------------------|------|----------|--------|
    | Low (Linear) | High | Low | Underfitting |
    | Optimal | Balanced | Balanced | Good generalization |
    | High (Deep NN) | Low | High | Overfitting |
    
    **Practical Example:**
    
    ```python
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import PolynomialFeatures
    
    # High Bias Model (Underfitting)
    linear_model = LinearRegression()
    scores_linear = cross_val_score(linear_model, X, y, cv=5)
    print(f"Linear Model CV Score: {scores_linear.mean():.3f} (+/- {scores_linear.std():.3f})")
    
    # Balanced Model
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10)
    scores_rf = cross_val_score(rf_model, X, y, cv=5)
    print(f"Random Forest CV Score: {scores_rf.mean():.3f} (+/- {scores_rf.std():.3f})")
    
    # High Variance Model (Overfitting risk)
    rf_deep = RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_leaf=1)
    scores_deep = cross_val_score(rf_deep, X, y, cv=5)
    print(f"Deep RF CV Score: {scores_deep.mean():.3f} (+/- {scores_deep.std():.3f})")
    ```

    !!! tip "Interviewer's Insight"
        **What they're really testing:** Your ability to diagnose model performance issues and choose appropriate solutions.
        
        **Strong answer signals:**
        
        - Can draw the classic U-shaped curve from memory
        - Gives concrete examples: "Linear regression on non-linear data = high bias"
        - Mentions solutions: cross-validation, regularization, ensemble methods
        - Discusses real scenarios: "In production at scale, I often prefer slightly higher bias for stability"

---

### Explain L1 (Lasso) vs L2 (Ridge) Regularization - Amazon, Microsoft Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Regularization`, `Feature Selection`, `Overfitting` | **Asked by:** Amazon, Microsoft, Google, Netflix

??? success "View Answer"

    **Core Difference:**
    
    Both add a penalty term to the loss function to prevent overfitting, but with different effects:
    
    | Aspect | L1 (Lasso) | L2 (Ridge) |
    |--------|------------|------------|
    | Penalty | $\lambda \sum |w_i|$ | $\lambda \sum w_i^2$ |
    | Effect on weights | Drives weights to exactly 0 | Shrinks weights toward 0 |
    | Feature selection | Yes (sparse solutions) | No (keeps all features) |
    | Geometry | Diamond constraint | Circular constraint |
    | Best for | High-dimensional sparse data | Multicollinearity |
    
    **Mathematical Formulation:**
    
    $$\text{L1 Loss} = \text{MSE} + \lambda \sum_{i=1}^{n} |w_i|$$
    
    $$\text{L2 Loss} = \text{MSE} + \lambda \sum_{i=1}^{n} w_i^2$$
    
    **Why L1 Creates Sparsity (Geometric Intuition):**
    
    The L1 constraint region is a diamond shape. The optimal solution often occurs at corners where some weights = 0.
    
    ```python
    from sklearn.linear_model import Lasso, Ridge, ElasticNet
    from sklearn.datasets import make_regression
    import numpy as np
    
    # Generate data with some irrelevant features
    X, y = make_regression(n_samples=100, n_features=20, n_informative=5, noise=10)
    
    # L1 Regularization - Feature Selection
    lasso = Lasso(alpha=0.1)
    lasso.fit(X, y)
    print(f"L1 Non-zero coefficients: {np.sum(lasso.coef_ != 0)}/20")
    # Output: ~5 (identifies informative features)
    
    # L2 Regularization - All features kept
    ridge = Ridge(alpha=0.1)
    ridge.fit(X, y)
    print(f"L2 Non-zero coefficients: {np.sum(ridge.coef_ != 0)}/20")
    # Output: 20 (all features kept, but shrunk)
    
    # Elastic Net - Best of both worlds
    elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
    elastic.fit(X, y)
    print(f"Elastic Net Non-zero: {np.sum(elastic.coef_ != 0)}/20")
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Deep understanding of regularization mechanics, not just definitions.
        
        **Strong answer signals:**
        
        - Explains WHY L1 creates zeros (diamond geometry)
        - Knows when to use each: "L1 for feature selection, L2 for correlated features"
        - Mentions Elastic Net as hybrid solution
        - Can discuss tuning λ via cross-validation

---

### How Does Gradient Descent Work? - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Optimization`, `Deep Learning`, `Fundamentals` | **Asked by:** Google, Meta, Amazon, Apple

??? success "View Answer"

    **The Core Idea:**
    
    Gradient descent is an iterative optimization algorithm that finds the minimum of a function by repeatedly moving in the direction of steepest descent (negative gradient).
    
    **Update Rule:**
    
    $$w_{t+1} = w_t - \eta \cdot \nabla L(w_t)$$
    
    Where:
    - $w_t$ = current weights
    - $\eta$ = learning rate (step size)
    - $\nabla L(w_t)$ = gradient of loss function
    
    **Variants Comparison:**
    
    | Variant | Batch Size | Speed | Stability | Memory |
    |---------|------------|-------|-----------|--------|
    | Batch GD | All data | Slow | Very stable | High |
    | Stochastic GD | 1 sample | Fast | Noisy | Low |
    | Mini-batch GD | 32-512 | Balanced | Balanced | Medium |
    
    **Modern Optimizers:**
    
    ```python
    import torch.optim as optim
    
    # Standard SGD
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # SGD with Momentum (accelerates convergence)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Adam (adaptive learning rates per parameter)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    
    # AdamW (Adam with proper weight decay)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    ```
    
    **Adam's Magic Formula:**
    
    $$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
    $$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
    $$w_{t+1} = w_t - \eta \frac{m_t}{\sqrt{v_t} + \epsilon}$$

    !!! tip "Interviewer's Insight"
        **What they're testing:** Can you explain optimization intuitively AND mathematically?
        
        **Strong answer signals:**
        
        - Draws the loss landscape and shows how GD navigates it
        - Knows why learning rate matters (too high = diverge, too low = slow)
        - Can explain momentum: "Like a ball rolling downhill with inertia"
        - Knows Adam is often the default: "Adaptive LR + momentum, works well out-of-box"

---

### What is Cross-Validation and Why Is It Important? - Facebook, Amazon Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Model Evaluation`, `Validation`, `Overfitting` | **Asked by:** Meta, Amazon, Google, Netflix

??? success "View Answer"

    **The Problem It Solves:**
    
    A single train/test split can give misleading results due to random variation in how data is split. Cross-validation provides a more reliable estimate of model performance.
    
    **K-Fold Cross-Validation:**
    
    1. Split data into K equal folds
    2. For each fold i:
        - Train on all folds except i
        - Validate on fold i
    3. Average all K validation scores
    
    **Common Strategies:**
    
    | Strategy | K | Use Case |
    |----------|---|----------|
    | 5-Fold | 5 | Standard, good balance |
    | 10-Fold | 10 | More reliable, slower |
    | Leave-One-Out | N | Small datasets, expensive |
    | Stratified K-Fold | K | Imbalanced classification |
    | Time Series Split | K | Temporal data (no leakage) |
    
    ```python
    from sklearn.model_selection import (
        cross_val_score, KFold, StratifiedKFold, TimeSeriesSplit
    )
    from sklearn.ensemble import RandomForestClassifier
    
    # Standard K-Fold
    cv_scores = cross_val_score(
        RandomForestClassifier(),
        X, y,
        cv=5,
        scoring='accuracy'
    )
    print(f"CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
    
    # Stratified for imbalanced data
    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Time Series (prevents data leakage)
    tscv = TimeSeriesSplit(n_splits=5)
    for train_idx, test_idx in tscv.split(X):
        print(f"Train: {train_idx[:3]}..., Test: {test_idx[:3]}...")
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of model validation fundamentals.
        
        **Strong answer signals:**
        
        - Knows when to use stratified (imbalanced classes) vs regular
        - Immediately mentions TimeSeriesSplit for temporal data (data leakage awareness)
        - Can explain computational tradeoff: "10-fold is 2x slower but more reliable"
        - Mentions nested CV for hyperparameter tuning

---

### Explain Precision, Recall, and F1-Score - Google, Microsoft Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Classification Metrics`, `Model Evaluation`, `Imbalanced Data` | **Asked by:** Google, Microsoft, Amazon, Meta

??? success "View Answer"

    **Confusion Matrix Foundation:**
    
    |  | Predicted Positive | Predicted Negative |
    |--|--------------------|--------------------|
    | **Actual Positive** | TP (True Positive) | FN (False Negative) |
    | **Actual Negative** | FP (False Positive) | TN (True Negative) |
    
    **The Metrics:**
    
    $$\text{Precision} = \frac{TP}{TP + FP}$$
    
    *"Of all positive predictions, how many were correct?"*
    
    $$\text{Recall} = \frac{TP}{TP + FN}$$
    
    *"Of all actual positives, how many did we find?"*
    
    $$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
    
    *"Harmonic mean - penalizes extreme imbalances"*
    
    **When to Prioritize Which:**
    
    | Scenario | Priority | Why |
    |----------|----------|-----|
    | Spam detection | Precision | Don't want to lose important emails |
    | Cancer screening | Recall | Don't want to miss any cases |
    | Fraud detection | F1 or Recall | Balance matters, but missing fraud is costly |
    | Search ranking | Precision@K | Top results quality matters most |
    
    ```python
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        classification_report, precision_recall_curve
    )
    
    # All metrics at once
    print(classification_report(y_true, y_pred))
    
    # Adjust threshold for Precision-Recall tradeoff
    y_proba = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Find threshold for desired recall (e.g., 95%)
    target_recall = 0.95
    idx = np.argmin(np.abs(recalls - target_recall))
    optimal_threshold = thresholds[idx]
    print(f"Threshold for {target_recall} recall: {optimal_threshold:.3f}")
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Can you choose the right metric for the business problem?
        
        **Strong answer signals:**
        
        - Immediately asks: "What's the cost of false positives vs false negatives?"
        - Knows accuracy is misleading for imbalanced data
        - Can adjust classification threshold based on business needs
        - Mentions AUC-PR for highly imbalanced datasets

---

### What is a Decision Tree and How Does It Work? - Amazon, Facebook Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Tree Models`, `Interpretability`, `Classification` | **Asked by:** Amazon, Meta, Google, Microsoft

??? success "View Answer"

    **How Decision Trees Work:**
    
    Decision trees recursively split the data based on feature values to create pure (homogeneous) leaf nodes.
    
    **Splitting Criteria:**
    
    For Classification (Information Gain / Gini):
    
    $$\text{Gini} = 1 - \sum_{i=1}^{C} p_i^2$$
    
    $$\text{Entropy} = -\sum_{i=1}^{C} p_i \log_2(p_i)$$
    
    For Regression (Variance Reduction):
    
    $$\text{Variance} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \bar{y})^2$$
    
    **Pros and Cons:**
    
    | Pros | Cons |
    |------|------|
    | Interpretable (white-box) | Prone to overfitting |
    | No scaling needed | Unstable (small data changes → different tree) |
    | Handles non-linear relationships | Greedy, not globally optimal |
    | Feature importance built-in | Can't extrapolate beyond training range |
    
    ```python
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    import matplotlib.pyplot as plt
    
    # Create and train
    tree = DecisionTreeClassifier(
        max_depth=5,           # Prevent overfitting
        min_samples_split=20,  # Minimum samples to split
        min_samples_leaf=10,   # Minimum samples in leaf
        random_state=42
    )
    tree.fit(X_train, y_train)
    
    # Visualize the tree
    plt.figure(figsize=(20, 10))
    plot_tree(tree, feature_names=feature_names, 
              class_names=class_names, filled=True)
    plt.show()
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': tree.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importance.head(10))
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of interpretable ML and when to use simple models.
        
        **Strong answer signals:**
        
        - Knows trees are building blocks for Random Forest, XGBoost
        - Can explain pruning techniques (pre-pruning vs post-pruning)
        - Mentions when to use: "Interpretability required, e.g., credit decisioning"
        - Knows limitation: "Single trees overfit; ensembles solve this"

---

### Random Forest vs Gradient Boosting: When to Use Which? - Google, Netflix Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Ensemble Methods`, `XGBoost`, `Model Selection` | **Asked by:** Google, Netflix, Amazon, Meta

??? success "View Answer"

    **Fundamental Difference:**
    
    | Aspect | Random Forest | Gradient Boosting |
    |--------|---------------|-------------------|
    | Strategy | Bagging (parallel) | Boosting (sequential) |
    | Trees | Independent | Each fixes previous errors |
    | Bias-Variance | Reduces variance | Reduces bias |
    | Overfitting | Resistant | Can overfit if not tuned |
    | Training | Parallelizable, fast | Sequential, slower |
    | Tuning | Easy | Requires careful tuning |
    
    **When to Use Which:**
    
    | Scenario | Choice | Reason |
    |----------|--------|--------|
    | Quick baseline | Random Forest | Works well with default params |
    | Maximum accuracy | Gradient Boosting | Better with tuning |
    | Large dataset | Random Forest | Faster training |
    | Kaggle competition | XGBoost/LightGBM | State-of-art tabular |
    | Production (simplicity) | Random Forest | More robust, less tuning |
    
    ```python
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    
    # Random Forest - Quick and robust
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        n_jobs=-1  # Parallel training
    )
    
    # Gradient Boosting (sklearn) - Good baseline
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3
    )
    
    # XGBoost - Industry standard
    xgb = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss'
    )
    
    # LightGBM - Fastest, handles large data
    lgbm = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        feature_fraction=0.8
    )
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Practical model selection skills.
        
        **Strong answer signals:**
        
        - Explains bagging vs boosting conceptually
        - Knows XGBoost/LightGBM are gradient boosting implementations
        - Can discuss tradeoffs: "RF is easier to deploy, GB needs more tuning"
        - Mentions real experience: "In production, I often start with RF for baseline"

---

### What is Overfitting and How Do You Prevent It? - Amazon, Meta Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Generalization`, `Regularization`, `Model Evaluation` | **Asked by:** Amazon, Meta, Google, Apple, Netflix

??? success "View Answer"

    **Definition:**
    
    Overfitting occurs when a model learns the training data too well, including noise and outliers, and fails to generalize to new data.
    
    **Signs of Overfitting:**
    
    - High training accuracy, low test accuracy
    - Large gap between training and validation loss
    - Model complexity >> data complexity
    
    **Prevention Techniques:**
    
    | Technique | How It Helps |
    |-----------|--------------|
    | More data | Reduces variance |
    | Regularization (L1/L2) | Constrains model complexity |
    | Cross-validation | Better estimate of generalization |
    | Early stopping | Stops before overfitting |
    | Dropout | Prevents co-adaptation in NNs |
    | Data augmentation | Increases effective dataset size |
    | Ensemble methods | Averages out individual model errors |
    | Feature selection | Reduces irrelevant noise |
    
    ```python
    from sklearn.model_selection import learning_curve
    import matplotlib.pyplot as plt
    
    # Diagnose overfitting with learning curves
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring='accuracy'
    )
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores.mean(axis=1), label='Training')
    plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation')
    plt.xlabel('Training Size')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Learning Curve - Check for Overfitting')
    plt.show()
    
    # Early stopping example (XGBoost)
    from xgboost import XGBClassifier
    
    model = XGBClassifier(
        n_estimators=1000,
        early_stopping_rounds=50,  # Stop if no improvement
        eval_metric='logloss'
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    print(f"Best iteration: {model.best_iteration}")
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Core ML intuition and practical experience.
        
        **Strong answer signals:**
        
        - Can draw learning curves and interpret them
        - Mentions multiple techniques, not just one
        - Knows underfitting is the opposite problem
        - Gives real examples: "I use early stopping + regularization together"

---

### Explain Neural Networks and Backpropagation - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Deep Learning`, `Neural Networks`, `Optimization` | **Asked by:** Google, Meta, Amazon, Apple

??? success "View Answer"

    **Neural Network Architecture:**
    
    A neural network is a series of layers that transform input through weighted connections and non-linear activation functions:
    
    $$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$$
    $$a^{[l]} = g(z^{[l]})$$
    
    Where:
    - $W^{[l]}$ = weight matrix for layer $l$
    - $b^{[l]}$ = bias vector
    - $g$ = activation function (ReLU, sigmoid, etc.)
    
    **Backpropagation (Chain Rule):**
    
    $$\frac{\partial L}{\partial W^{[l]}} = \frac{\partial L}{\partial a^{[L]}} \cdot \frac{\partial a^{[L]}}{\partial z^{[L]}} \cdot ... \cdot \frac{\partial z^{[l]}}{\partial W^{[l]}}$$
    
    **Common Activation Functions:**
    
    | Function | Formula | Use Case |
    |----------|---------|----------|
    | ReLU | $\max(0, x)$ | Hidden layers (default) |
    | Sigmoid | $\frac{1}{1+e^{-x}}$ | Binary output |
    | Softmax | $\frac{e^{x_i}}{\sum e^{x_j}}$ | Multi-class output |
    | Tanh | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | Hidden layers (centered) |
    
    ```python
    import torch
    import torch.nn as nn
    
    class SimpleNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.layer1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.layer2 = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            x = self.layer1(x)
            x = self.relu(x)  # Non-linearity is crucial!
            x = self.layer2(x)
            return x
    
    # Training loop with backprop
    model = SimpleNN(784, 256, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        for X_batch, y_batch in dataloader:
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass (backpropagation)
            optimizer.zero_grad()  # Clear old gradients
            loss.backward()        # Compute gradients
            optimizer.step()       # Update weights
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Deep understanding of DL fundamentals.
        
        **Strong answer signals:**
        
        - Can explain why non-linearity is essential (stacked linear = just linear)
        - Knows vanishing gradient problem and solutions (ReLU, ResNets, LSTM)
        - Can derive simple backprop by hand (at least for 1-layer)
        - Mentions practical considerations: batch normalization, dropout

---

### What is Dropout and Why Does It Work? - Amazon, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Regularization`, `Deep Learning`, `Overfitting` | **Asked by:** Amazon, Meta, Google, Apple

??? success "View Answer"

    **How Dropout Works:**
    
    During training, randomly set a fraction $p$ of neuron outputs to zero:
    
    1. For each training batch:
        - Randomly select neurons to "drop" (output = 0)
        - Scale remaining outputs by $\frac{1}{1-p}$ to maintain expected value
    2. During inference:
        - Use all neurons (no dropout)
    
    **Why It Works (Multiple Perspectives):**
    
    | Perspective | Explanation |
    |-------------|-------------|
    | Ensemble | Training many sub-networks, averaging at test time |
    | Co-adaptation | Prevents neurons from relying on specific other neurons |
    | Regularization | Adds noise, similar to L2 regularization |
    | Bayesian | Approximates Bayesian inference (variational) |
    
    ```python
    import torch.nn as nn
    
    class DropoutNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 512)
            self.dropout1 = nn.Dropout(p=0.5)  # 50% dropout
            self.fc2 = nn.Linear(512, 256)
            self.dropout2 = nn.Dropout(p=0.3)  # 30% dropout
            self.fc3 = nn.Linear(256, 10)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout1(x)  # Applied during training
            x = F.relu(self.fc2(x))
            x = self.dropout2(x)
            x = self.fc3(x)
            return x
    
    # Important: model.eval() disables dropout for inference
    model.eval()
    with torch.no_grad():
        predictions = model(test_data)
    ```
    
    **Common Dropout Rates:**
    
    - Input layer: 0.2 (keep 80%)
    - Hidden layers: 0.5 (keep 50%)
    - After BatchNorm: Often not needed

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of regularization in deep learning.
        
        **Strong answer signals:**
        
        - Knows dropout is only active during training
        - Can explain the scaling factor ($\frac{1}{1-p}$)
        - Mentions alternatives: DropConnect, Spatial Dropout for CNNs
        - Knows practical tips: "Don't use after BatchNorm, less needed with modern architectures"

---

### What is Transfer Learning? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Deep Learning`, `Pretrained Models`, `Fine-tuning` | **Asked by:** Google, Amazon, Meta, Microsoft

??? success "View Answer"

    **The Core Idea:**
    
    Transfer learning leverages knowledge from a model trained on a large dataset (source task) to improve performance on a different but related task (target task).
    
    **Why It Works:**
    
    - Lower layers learn general features (edges, textures, word patterns)
    - Higher layers learn task-specific features
    - General features transfer well across tasks
    
    **Transfer Learning Strategies:**
    
    | Strategy | When to Use | How |
    |----------|-------------|-----|
    | Feature extraction | Small target dataset | Freeze pretrained layers, train new head |
    | Fine-tuning | Medium target dataset | Unfreeze some layers, train with low LR |
    | Full fine-tuning | Large target dataset | Unfreeze all, train end-to-end |
    
    ```python
    # Computer Vision (PyTorch)
    from torchvision import models
    
    # Load pretrained ResNet
    model = models.resnet50(pretrained=True)
    
    # Strategy 1: Feature Extraction (freeze backbone)
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace final layer for our task
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Strategy 2: Fine-tuning (unfreeze last block)
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    # NLP (Hugging Face Transformers)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    
    # Load pretrained BERT
    model = AutoModelForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2  # Binary classification
    )
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Fine-tune with lower learning rate for pretrained layers
    optimizer = AdamW([
        {'params': model.bert.parameters(), 'lr': 2e-5},     # Pretrained
        {'params': model.classifier.parameters(), 'lr': 1e-4}  # New head
    ])
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Practical deep learning experience.
        
        **Strong answer signals:**
        
        - Knows when to freeze vs fine-tune (data size matters)
        - Mentions learning rate strategies (lower LR for pretrained)
        - Can name popular pretrained models: ResNet, BERT, GPT
        - Discusses domain shift: "Fine-tune more when source/target domains differ"

---

### Explain ROC Curve and AUC Score - Microsoft, Netflix Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Classification Metrics`, `Model Evaluation`, `Binary Classification` | **Asked by:** Microsoft, Netflix, Google, Amazon

??? success "View Answer"

    **ROC Curve (Receiver Operating Characteristic):**
    
    Plots True Positive Rate vs False Positive Rate at various classification thresholds:
    
    $$TPR = \frac{TP}{TP + FN} = \text{Recall}$$
    
    $$FPR = \frac{FP}{FP + TN}$$
    
    **AUC (Area Under Curve):**
    
    - **AUC = 1.0**: Perfect classifier
    - **AUC = 0.5**: Random guessing (diagonal line)
    - **AUC < 0.5**: Worse than random (inverted predictions)
    
    **Interpretation:**
    
    AUC = Probability that a randomly chosen positive example ranks higher than a randomly chosen negative example.
    
    ```python
    from sklearn.metrics import roc_curve, auc, roc_auc_score
    import matplotlib.pyplot as plt
    
    # Get probabilities
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
    
    # Quick AUC calculation
    print(f"AUC Score: {roc_auc_score(y_test, y_proba):.3f}")
    
    # Find optimal threshold (Youden's J statistic)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal Threshold: {optimal_threshold:.3f}")
    ```
    
    **ROC-AUC vs PR-AUC:**
    
    | Metric | Best For | Why |
    |--------|----------|-----|
    | ROC-AUC | Balanced classes | Considers both classes equally |
    | PR-AUC | Imbalanced classes | Focuses on positive class performance |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of evaluation metrics beyond accuracy.
        
        **Strong answer signals:**
        
        - Knows ROC-AUC can be misleading for imbalanced data
        - Can interpret thresholds: "Moving along the curve = changing threshold"
        - Mentions practical application: "I use AUC for model comparison, threshold tuning for deployment"
        - Knows PR-AUC is better for highly imbalanced problems

---

### What is Dimensionality Reduction? Explain PCA - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Dimensionality Reduction`, `Feature Extraction`, `Unsupervised Learning` | **Asked by:** Google, Amazon, Meta, Microsoft

??? success "View Answer"

    **Why Reduce Dimensions:**
    
    - Curse of dimensionality (data becomes sparse)
    - Reduce computation time
    - Remove noise and redundant features
    - Enable visualization (2D/3D)
    
    **PCA (Principal Component Analysis):**
    
    Finds orthogonal directions (principal components) that maximize variance in the data.
    
    **Steps:**
    1. Center the data (subtract mean)
    2. Compute covariance matrix
    3. Find eigenvectors and eigenvalues
    4. Select top k eigenvectors
    5. Project data onto new basis
    
    $$\text{Maximize: } \sum_{i=1}^{k} \text{Var}(X \cdot w_i) = \sum_{i=1}^{k} \lambda_i$$
    
    ```python
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Step 1: Always scale before PCA!
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Step 2: Determine optimal number of components
    pca_full = PCA()
    pca_full.fit(X_scaled)
    
    # Plot explained variance
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    n_95 = np.argmax(cumsum >= 0.95) + 1
    print(f"Components for 95% variance: {n_95}")
    
    # Step 3: Apply PCA
    pca = PCA(n_components=n_95)
    X_reduced = pca.fit_transform(X_scaled)
    
    # Visualization (2D)
    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(X_scaled)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis')
    plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} var)')
    plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} var)')
    plt.show()
    ```
    
    **Alternative Methods:**
    
    | Method | Best For | Preserves |
    |--------|----------|-----------|
    | PCA | Linear relationships, variance | Global structure |
    | t-SNE | Visualization | Local structure |
    | UMAP | Large datasets, clustering | Local + global |
    | LDA | Classification | Class separability |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of unsupervised learning and feature engineering.
        
        **Strong answer signals:**
        
        - Knows to scale data before PCA (otherwise high-variance features dominate)
        - Can explain 95% variance retention heuristic
        - Mentions limitations: "PCA assumes linear relationships"
        - Knows alternatives: t-SNE for visualization, UMAP for clustering

---

### How Do You Handle Imbalanced Datasets? - Netflix, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Imbalanced Data`, `Classification`, `Sampling` | **Asked by:** Netflix, Meta, Amazon, Google

??? success "View Answer"

    **The Problem:**
    
    When one class dominates (e.g., 99% negative, 1% positive), models tend to predict the majority class and achieve high accuracy while missing the minority class entirely.
    
    **Solutions Toolkit:**
    
    | Technique | Category | When to Use |
    |-----------|----------|-------------|
    | Class weights | Cost-sensitive | Always try first |
    | SMOTE | Oversampling | Moderate imbalance |
    | Random undersampling | Undersampling | Large dataset |
    | Threshold tuning | Post-processing | Quick fix |
    | Focal Loss | Loss function | Deep learning |
    | Ensemble methods | Modeling | Severe imbalance |
    
    ```python
    from sklearn.utils.class_weight import compute_class_weight
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline
    
    # Method 1: Class Weights (built into most algorithms)
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    model = RandomForestClassifier(class_weight='balanced')
    
    # Method 2: SMOTE (Synthetic Minority Over-sampling)
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    # Method 3: Combined Sampling Pipeline
    pipeline = Pipeline([
        ('under', RandomUnderSampler(sampling_strategy=0.5)),
        ('over', SMOTE(sampling_strategy=1.0)),
    ])
    X_balanced, y_balanced = pipeline.fit_resample(X_train, y_train)
    
    # Method 4: Threshold Tuning
    y_proba = model.predict_proba(X_test)[:, 1]
    # Lower threshold to catch more positives
    y_pred_adjusted = (y_proba >= 0.3).astype(int)  # Instead of 0.5
    
    # Method 5: Focal Loss (PyTorch)
    class FocalLoss(nn.Module):
        def __init__(self, alpha=0.25, gamma=2):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
        
        def forward(self, inputs, targets):
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
            return focal_loss.mean()
    ```
    
    **Evaluation for Imbalanced Data:**
    
    - ❌ Accuracy (misleading)
    - ✅ Precision, Recall, F1
    - ✅ PR-AUC (better than ROC-AUC)
    - ✅ Confusion matrix

    !!! tip "Interviewer's Insight"
        **What they're testing:** Real-world ML problem-solving.
        
        **Strong answer signals:**
        
        - First asks: "How imbalanced? 90-10 is different from 99.9-0.1"
        - Knows class weights is usually the first approach
        - Warns about SMOTE pitfalls: "Can overfit to synthetic examples"
        - Mentions correct metrics: "Never use accuracy for imbalanced data"

---

### Explain K-Means Clustering - Amazon, Microsoft Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Clustering`, `Unsupervised Learning`, `K-Means` | **Asked by:** Amazon, Microsoft, Google, Meta

??? success "View Answer"

    **Algorithm Steps:**
    
    1. Initialize k centroids randomly
    2. Assign each point to nearest centroid
    3. Recalculate centroids as cluster means
    4. Repeat steps 2-3 until convergence
    
    **Objective Function (Inertia):**
    
    $$J = \sum_{i=1}^{n} \min_{j} ||x_i - \mu_j||^2$$
    
    Minimize within-cluster sum of squares.
    
    **Choosing K (Elbow Method):**
    
    ```python
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt
    
    # Elbow Method
    inertias = []
    silhouettes = []
    K_range = range(2, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X, kmeans.labels_))
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(K_range, inertias, 'bo-')
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    
    ax2.plot(K_range, silhouettes, 'ro-')
    ax2.set_xlabel('Number of Clusters (K)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Method')
    plt.show()
    
    # Final model
    optimal_k = 5  # From elbow analysis
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    ```
    
    **Limitations and Alternatives:**
    
    | Limitation | Better Alternative |
    |------------|-------------------|
    | Assumes spherical clusters | DBSCAN, GMM |
    | Sensitive to initialization | KMeans++ (default) |
    | Must specify K | DBSCAN (auto-detects) |
    | Sensitive to outliers | DBSCAN, Robust clustering |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Basic unsupervised learning understanding.
        
        **Strong answer signals:**
        
        - Knows K-means++ initialization (sklearn default)
        - Can explain limitations: "Assumes spherical, equal-size clusters"
        - Mentions silhouette score for validation
        - Knows when to use alternatives: "DBSCAN for arbitrary shapes"

---


### What Are Support Vector Machines (SVMs)? When Should You Use Them? - Google, Amazon, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Classification`, `Kernel Methods`, `Margin Maximization` | **Asked by:** Google, Amazon, Meta, Microsoft

??? success "View Answer"

    **What Are SVMs?**
    
    Support Vector Machines are supervised learning models that find the optimal hyperplane to separate classes with maximum margin.
    
    **Key Concepts:**
    
    | Concept | Meaning |
    |---------|---------|
    | Support Vectors | Data points closest to decision boundary |
    | Margin | Distance between boundary and nearest points |
    | Kernel Trick | Maps data to higher dimensions for non-linear separation |
    
    **Kernels:**
    
    ```python
    from sklearn.svm import SVC
    
    # Linear kernel - for linearly separable data
    svm_linear = SVC(kernel='linear', C=1.0)
    
    # RBF (Gaussian) - most common for non-linear
    svm_rbf = SVC(kernel='rbf', gamma='scale', C=1.0)
    
    # Polynomial kernel
    svm_poly = SVC(kernel='poly', degree=3, C=1.0)
    
    # Training
    svm_rbf.fit(X_train, y_train)
    predictions = svm_rbf.predict(X_test)
    ```
    
    **When to Use SVMs:**
    
    | Good for | Not good for |
    |----------|--------------|
    | High-dimensional data (text) | Very large datasets (slow) |
    | Clear margin of separation | Noisy data with overlapping classes |
    | Fewer samples than features | Multi-class (needs one-vs-one) |
    
    **Hyperparameters:**
    
    - **C (Regularization)**: Trade-off between margin and misclassification
    - **gamma**: Kernel coefficient - high = overfitting, low = underfitting

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of geometric intuition and kernel methods.
        
        **Strong answer signals:**
        
        - Explains margin maximization geometrically
        - Knows when to use different kernels
        - Mentions computational complexity O(n²) to O(n³)
        - Knows SVMs work well for text classification

---

### Explain Convolutional Neural Networks (CNNs) and Their Architecture - Google, Meta, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Deep Learning`, `Computer Vision`, `Neural Networks` | **Asked by:** Google, Meta, Amazon, Apple, NVIDIA

??? success "View Answer"

    **What Are CNNs?**
    
    CNNs are neural networks designed for processing structured grid data (images, time series) using convolutional layers that detect spatial patterns.
    
    **Core Components:**
    
    | Layer | Purpose |
    |-------|---------|
    | Convolutional | Extract features using learnable filters |
    | Pooling | Downsample, reduce computation, add translation invariance |
    | Fully Connected | Classification at the end |
    | Activation (ReLU) | Add non-linearity |
    
    **How Convolution Works:**
    
    ```python
    import torch.nn as nn
    
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            # Input: 3 channels (RGB), Output: 32 filters, 3x3 kernel
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)  # 2x2 pooling
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(64 * 8 * 8, 256)  # After 2 pools: 32→16→8
            self.fc2 = nn.Linear(256, 10)  # 10 classes
            
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))  # 32x32 → 16x16
            x = self.pool(F.relu(self.conv2(x)))  # 16x16 → 8x8
            x = x.view(-1, 64 * 8 * 8)  # Flatten
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    ```
    
    **Key CNN Architectures:**
    
    | Architecture | Year | Innovation |
    |--------------|------|------------|
    | LeNet | 1998 | First practical CNN |
    | AlexNet | 2012 | Deep CNNs, ReLU, Dropout |
    | VGG | 2014 | Small 3x3 filters, depth |
    | ResNet | 2015 | Skip connections (residual) |
    | EfficientNet | 2019 | Compound scaling |
    
    **Calculations:**
    
    Output size: $(W - K + 2P) / S + 1$
    
    Where: W = input, K = kernel, P = padding, S = stride

    !!! tip "Interviewer's Insight"
        **What they're testing:** Deep learning fundamentals and computer vision.
        
        **Strong answer signals:**
        
        - Can calculate output dimensions
        - Explains why pooling helps (translation invariance)
        - Knows ResNet skip connections solve vanishing gradients
        - Mentions transfer learning: "Use pretrained ImageNet models"

---

### What Are Recurrent Neural Networks (RNNs) and LSTMs? - Google, Amazon, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Deep Learning`, `Sequence Models`, `NLP` | **Asked by:** Google, Amazon, Meta, Microsoft

??? success "View Answer"

    **What Are RNNs?**
    
    RNNs process sequential data by maintaining hidden state that captures information from previous time steps.
    
    **The Problem: Vanishing Gradients**
    
    Standard RNNs struggle with long sequences because gradients vanish/explode during backpropagation through time.
    
    **LSTM Solution:**
    
    ```python
    import torch.nn as nn
    
    class LSTMModel(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, 
                               num_layers=2, dropout=0.5, 
                               batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional
            
        def forward(self, x):
            embedded = self.embedding(x)
            output, (hidden, cell) = self.lstm(embedded)
            # Concatenate final hidden states from both directions
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
            return self.fc(hidden)
    ```
    
    **LSTM Gates:**
    
    | Gate | Purpose |
    |------|---------|
    | Forget | Decide what to discard from cell state |
    | Input | Decide what new info to store |
    | Output | Decide what to output |
    
    **GRU vs LSTM:**
    
    | Aspect | LSTM | GRU |
    |--------|------|-----|
    | Gates | 3 (forget, input, output) | 2 (reset, update) |
    | Parameters | More | Fewer |
    | Performance | Better for longer sequences | Often comparable |
    
    **Modern Alternatives:**
    
    - **Transformers**: Now preferred for most NLP tasks
    - **1D CNNs**: Faster for some sequence tasks
    - **Attention mechanisms**: Can be added to RNNs

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of sequence modeling.
        
        **Strong answer signals:**
        
        - Explains vanishing gradient problem
        - Draws LSTM cell diagram with gates
        - Knows when to use bidirectional
        - Mentions: "Transformers have largely replaced LSTMs for NLP"

---

### What is Batch Normalization and Why Does It Help? - Google, Amazon, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Deep Learning`, `Training`, `Regularization` | **Asked by:** Google, Amazon, Meta, Microsoft, Apple

??? success "View Answer"

    **What is Batch Normalization?**
    
    Batch normalization normalizes layer inputs by re-centering and re-scaling, making training faster and more stable.
    
    **The Formula:**
    
    $$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
    $$y = \gamma \hat{x} + \beta$$
    
    Where $\gamma$ (scale) and $\beta$ (shift) are learnable parameters.
    
    **Benefits:**
    
    | Benefit | Explanation |
    |---------|-------------|
    | Faster training | Enables higher learning rates |
    | Regularization | Adds noise (mini-batch statistics) |
    | Reduces internal covariate shift | Stable distributions |
    | Less sensitive to initialization | Normalizes anyway |
    
    ```python
    import torch.nn as nn
    
    class CNNWithBatchNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(64)  # After conv, before activation
            self.relu = nn.ReLU()
            
            self.fc = nn.Linear(64 * 32 * 32, 10)
            self.bn_fc = nn.BatchNorm1d(10)  # For fully connected
            
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)  # Normalize
            x = self.relu(x)  # Then activate
            x = x.view(-1, 64 * 32 * 32)
            x = self.fc(x)
            return x
    
    # Training vs. inference mode matters!
    model.train()  # Uses batch statistics
    model.eval()   # Uses running averages
    ```
    
    **Layer Normalization (Alternative):**
    
    | BatchNorm | LayerNorm |
    |-----------|-----------|
    | Normalizes across batch | Normalizes across features |
    | Needs batch statistics | Works with batch size 1 |
    | Good for CNNs | Good for RNNs, Transformers |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of deep learning training dynamics.
        
        **Strong answer signals:**
        
        - Knows position: after linear/conv, before activation
        - Explains train vs eval mode difference
        - Mentions Layer Norm for Transformers
        - Knows it's less needed with skip connections (ResNet)

---

### What is XGBoost and How Does It Differ from Random Forest? - Amazon, Google, Microsoft Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Ensemble Methods`, `Boosting`, `Tabular Data` | **Asked by:** Amazon, Google, Microsoft, Netflix, Meta

??? success "View Answer"

    **XGBoost vs Random Forest:**
    
    | Aspect | Random Forest | XGBoost |
    |--------|---------------|---------|
    | Method | Bagging (parallel trees) | Boosting (sequential trees) |
    | Error Focus | Each tree is independent | Each tree fixes previous errors |
    | Overfitting | Resistant | Needs regularization |
    | Speed | Parallelizable | Optimized (GPU support) |
    | Interpretability | Feature importance | Feature importance + SHAP |
    
    **How XGBoost Works:**
    
    ```python
    import xgboost as xgb
    from sklearn.model_selection import cross_val_score
    
    # Basic XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # Feature importance
    importance = model.feature_importances_
    
    # Cross-validation
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    ```
    
    **Key Hyperparameters:**
    
    | Parameter | Effect |
    |-----------|--------|
    | n_estimators | Number of trees |
    | max_depth | Tree depth (prevent overfitting) |
    | learning_rate | Shrinkage (lower = more trees needed) |
    | subsample | Row sampling per tree |
    | colsample_bytree | Feature sampling per tree |
    | reg_alpha/lambda | L1/L2 regularization |
    
    **When to Use Which:**
    
    | Use Random Forest | Use XGBoost |
    |-------------------|-------------|
    | Quick baseline | Maximum accuracy |
    | Less tuning time | Tabular competitions |
    | Reduce overfitting | Handle missing values |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Practical ML knowledge for tabular data.
        
        **Strong answer signals:**
        
        - Explains bagging vs boosting difference
        - Knows key hyperparameters to tune
        - Mentions: "XGBoost handles missing values natively"
        - Knows alternatives: LightGBM (faster), CatBoost (categorical)

---

### Explain Attention Mechanisms and Transformers - Google, Meta, OpenAI Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Deep Learning`, `NLP`, `Transformers` | **Asked by:** Google, Meta, OpenAI, Microsoft, Amazon

??? success "View Answer"

    **What is Attention?**
    
    Attention allows models to focus on relevant parts of the input when producing output, replacing the need for recurrence.
    
    **Self-Attention Formula:**
    
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
    
    **Transformer Architecture:**
    
    ```python
    import torch
    import torch.nn as nn
    import math
    
    class SelfAttention(nn.Module):
        def __init__(self, embed_dim, num_heads):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
            
            self.q_linear = nn.Linear(embed_dim, embed_dim)
            self.k_linear = nn.Linear(embed_dim, embed_dim)
            self.v_linear = nn.Linear(embed_dim, embed_dim)
            self.out = nn.Linear(embed_dim, embed_dim)
            
        def forward(self, x, mask=None):
            batch_size = x.size(0)
            
            # Linear projections
            Q = self.q_linear(x)
            K = self.k_linear(x)
            V = self.v_linear(x)
            
            # Reshape for multi-head
            Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Scaled dot-product attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attention = torch.softmax(scores, dim=-1)
            out = torch.matmul(attention, V)
            
            # Concatenate heads
            out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
            return self.out(out)
    ```
    
    **Key Components:**
    
    | Component | Purpose |
    |-----------|---------|
    | Multi-Head Attention | Attend to different representation subspaces |
    | Position Encoding | Inject sequence order information |
    | Layer Normalization | Stabilize training |
    | Feed-Forward Network | Non-linear transformation |
    
    **Transformer Models:**
    
    | Model | Type | Use Case |
    |-------|------|----------|
    | BERT | Encoder-only | Classification, NER |
    | GPT | Decoder-only | Text generation |
    | T5 | Encoder-Decoder | Translation, summarization |
    | ViT | Vision | Image classification |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Modern deep learning architecture understanding.
        
        **Strong answer signals:**
        
        - Can explain Q, K, V analogy (query-key-value retrieval)
        - Knows why scaling by √d_k (prevent softmax saturation)
        - Understands positional encoding necessity
        - Mentions computational complexity: O(n²) for sequence length n

---

### What is Feature Engineering? Give Examples - Amazon, Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Data Preprocessing`, `Feature Engineering`, `ML Pipeline` | **Asked by:** Amazon, Google, Meta, Microsoft, Netflix

??? success "View Answer"

    **What is Feature Engineering?**
    
    Feature engineering is the process of creating, transforming, and selecting features to improve model performance.
    
    **Categories of Feature Engineering:**
    
    | Category | Examples |
    |----------|----------|
    | Creation | Domain-specific features, aggregations |
    | Transformation | Log, sqrt, polynomial features |
    | Encoding | One-hot, target encoding, embeddings |
    | Scaling | Standardization, normalization |
    | Selection | Filter, wrapper, embedded methods |
    
    ```python
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    
    # 1. Date/Time features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['hour'] = df['date'].dt.hour
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12)  # Cyclical
    
    # 2. Aggregation features
    df['user_total_purchases'] = df.groupby('user_id')['amount'].transform('sum')
    df['user_avg_purchase'] = df.groupby('user_id')['amount'].transform('mean')
    df['user_purchase_count'] = df.groupby('user_id')['amount'].transform('count')
    
    # 3. Text features
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    df['has_question'] = df['text'].str.contains(r'\?').astype(int)
    
    # 4. Interaction features
    df['price_per_sqft'] = df['price'] / df['sqft']
    df['bmi'] = df['weight'] / (df['height'] ** 2)
    
    # 5. Binning
    df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 100], 
                             labels=['child', 'young', 'middle', 'senior'])
    
    # 6. Target encoding (for categorical)
    target_means = df.groupby('category')['target'].mean()
    df['category_encoded'] = df['category'].map(target_means)
    
    # 7. Log transformation (for skewed data)
    df['log_income'] = np.log1p(df['income'])  # log1p handles zeros
    ```
    
    **Domain-Specific Examples:**
    
    | Domain | Feature Ideas |
    |--------|---------------|
    | E-commerce | Days since last purchase, cart abandonment rate |
    | Finance | Moving averages, volatility, ratios |
    | NLP | TF-IDF, n-grams, sentiment scores |
    | Healthcare | BMI, age groups, risk scores |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Practical data science skills.
        
        **Strong answer signals:**
        
        - Gives domain-specific examples
        - Knows cyclical encoding for time features
        - Mentions target encoding for high-cardinality categoricals
        - Warns about data leakage: "Always fit on train, transform on test"

---

### What is Model Interpretability? Explain SHAP and LIME - Google, Amazon, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Explainability`, `Model Interpretation`, `XAI` | **Asked by:** Google, Amazon, Meta, Microsoft

??? success "View Answer"

    **Why Interpretability Matters:**
    
    - Regulatory compliance (GDPR, healthcare)
    - Debug and improve models
    - Build trust with stakeholders
    - Detect bias and fairness issues
    
    **SHAP (SHapley Additive exPlanations):**
    
    Based on game theory - measures each feature's contribution to prediction.
    
    ```python
    import shap
    
    # Train model
    model = xgb.XGBClassifier().fit(X_train, y_train)
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Summary plot (global importance)
    shap.summary_plot(shap_values, X_test)
    
    # Force plot (single prediction)
    shap.force_plot(explainer.expected_value, 
                   shap_values[0], X_test.iloc[0])
    
    # Dependence plot (feature interaction)
    shap.dependence_plot("age", shap_values, X_test)
    ```
    
    **LIME (Local Interpretable Model-agnostic Explanations):**
    
    Creates local linear approximations around individual predictions.
    
    ```python
    from lime import lime_tabular
    
    explainer = lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns,
        class_names=['No', 'Yes'],
        mode='classification'
    )
    
    # Explain single prediction
    exp = explainer.explain_instance(
        X_test.iloc[0].values,
        model.predict_proba,
        num_features=10
    )
    exp.show_in_notebook()
    ```
    
    **Comparison:**
    
    | Aspect | SHAP | LIME |
    |--------|------|------|
    | Approach | Game theory (Shapley values) | Local linear models |
    | Consistency | Theoretically guaranteed | Approximate |
    | Speed | Slower | Faster |
    | Scope | Global + local | Local (per prediction) |
    
    **Other Methods:**
    
    - **Feature Importance**: Built-in for tree models
    - **Partial Dependence Plots**: Show marginal effect
    - **Permutation Importance**: Model-agnostic

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of responsible AI.
        
        **Strong answer signals:**
        
        - Knows difference between global vs local explanations
        - Can explain Shapley values intuitively
        - Mentions use cases: debugging, compliance, bias detection
        - Knows SHAP is theoretically grounded, LIME is approximate

---

### What is Hyperparameter Tuning? Explain Grid Search, Random Search, and Bayesian Optimization - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Model Optimization`, `Hyperparameter Tuning`, `AutoML` | **Asked by:** Amazon, Google, Microsoft, Meta

??? success "View Answer"

    **What Are Hyperparameters?**
    
    Hyperparameters are external configurations set before training (unlike learned parameters).
    
    **Tuning Methods:**
    
    | Method | Approach | Pros | Cons |
    |--------|----------|------|------|
    | Grid Search | Exhaustive search over parameter grid | Complete | Exponentially slow |
    | Random Search | Random sampling from distributions | Faster, finds good values | May miss optimal |
    | Bayesian | Probabilistic model of objective | Efficient, smart | More complex |
    
    ```python
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn.ensemble import RandomForestClassifier
    
    # Grid Search
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(
        RandomForestClassifier(),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    print(f"Best params: {grid_search.best_params_}")
    
    # Random Search (often better)
    from scipy.stats import randint, uniform
    
    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(3, 20),
        'min_samples_split': randint(2, 20)
    }
    
    random_search = RandomizedSearchCV(
        RandomForestClassifier(),
        param_dist,
        n_iter=50,  # Number of random combinations
        cv=5,
        random_state=42
    )
    random_search.fit(X_train, y_train)
    ```
    
    **Bayesian Optimization (Optuna):**
    
    ```python
    import optuna
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        }
        
        model = xgb.XGBClassifier(**params)
        score = cross_val_score(model, X, y, cv=5).mean()
        return score
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    
    print(f"Best params: {study.best_params}")
    ```
    
    **Key Insight:**
    
    Random Search is often better than Grid Search because it explores more values of important hyperparameters.

    !!! tip "Interviewer's Insight"
        **What they're testing:** Practical ML optimization skills.
        
        **Strong answer signals:**
        
        - Knows random search often beats grid search
        - Can explain why (more coverage of important params)
        - Mentions Optuna/Hyperopt for Bayesian optimization
        - Uses cross-validation to avoid tuning to test set

---

### What is Data Leakage? How Do You Prevent It? - Amazon, Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `ML Best Practices`, `Data Leakage`, `Validation` | **Asked by:** Amazon, Google, Meta, Microsoft

??? success "View Answer"

    **What is Data Leakage?**
    
    Data leakage occurs when information from outside the training set is used to create the model, causing overly optimistic validation scores that don't generalize.
    
    **Types of Leakage:**
    
    | Type | Example | Solution |
    |------|---------|----------|
    | Target Leakage | Using future data to predict past | Respect time ordering |
    | Train-Test Contamination | Scaling using full dataset stats | Fit on train only |
    | Feature Leakage | Feature derived from target | Domain knowledge review |
    
    **Common Examples:**
    
    ```python
    # ❌ WRONG: Preprocessing before split
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Sees all data!
    X_train, X_test = train_test_split(X_scaled, ...)
    
    # ✅ CORRECT: Preprocess after split
    X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit on train only
    X_test_scaled = scaler.transform(X_test)  # Transform with train params
    
    # ✅ BEST: Use Pipeline
    from sklearn.pipeline import Pipeline
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier())
    ])
    
    # Cross-validation respects the pipeline
    scores = cross_val_score(pipeline, X, y, cv=5)
    ```
    
    **Time Series Leakage:**
    
    ```python
    # ❌ WRONG: Random split for time series
    X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)
    
    # ✅ CORRECT: Temporal split
    train = df[df['date'] < '2024-01-01']
    test = df[df['date'] >= '2024-01-01']
    
    # Or use TimeSeriesSplit
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    ```
    
    **Subtle Leakage Examples:**
    
    - Customer ID that correlates with VIP status (target)
    - Hospital department that indicates diagnosis
    - Timestamp of transaction result recorded after outcome

    !!! tip "Interviewer's Insight"
        **What they're testing:** ML engineering rigor.
        
        **Strong answer signals:**
        
        - Immediately mentions fit_transform on train only
        - Uses sklearn Pipeline to avoid leakage
        - Knows time series requires temporal splits
        - Reviews features for target proxy patterns

---

### What is A/B Testing in the Context of ML Models? - Google, Netflix, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Experimentation`, `A/B Testing`, `Production ML` | **Asked by:** Google, Netflix, Meta, Amazon, Microsoft

??? success "View Answer"

    **Why A/B Test ML Models?**
    
    Offline metrics don't always correlate with business metrics. A/B testing validates that a new model improves real user outcomes.
    
    **A/B Testing Framework:**
    
    | Step | Description |
    |------|-------------|
    | 1. Hypothesis | New model improves metric X by Y% |
    | 2. Randomization | Users randomly assigned to control/treatment |
    | 3. Sample Size | Calculate required sample for statistical power |
    | 4. Run Experiment | Serve both models simultaneously |
    | 5. Analysis | Statistical significance test |
    
    **Sample Size Calculation:**
    
    ```python
    from scipy import stats
    
    def calculate_sample_size(baseline_rate, mde, alpha=0.05, power=0.8):
        """
        baseline_rate: Current conversion rate
        mde: Minimum detectable effect (relative change)
        """
        effect_size = baseline_rate * mde
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_power = stats.norm.ppf(power)
        
        p = baseline_rate
        p_hat = (p + (p + effect_size)) / 2
        
        n = (z_alpha * (2 * p_hat * (1 - p_hat))**0.5 + 
             z_power * (p * (1-p) + (p + effect_size) * (1 - (p + effect_size)))**0.5)**2 / effect_size**2
        
        return int(n)
    
    # Example: 5% baseline, detect 10% relative improvement
    n = calculate_sample_size(0.05, 0.10)
    print(f"Need {n} samples per group")
    ```
    
    **Statistical Significance:**
    
    ```python
    from scipy import stats
    
    def ab_test_significance(control_conversions, control_total,
                            treatment_conversions, treatment_total):
        control_rate = control_conversions / control_total
        treatment_rate = treatment_conversions / treatment_total
        
        # Two-proportion z-test
        pooled = (control_conversions + treatment_conversions) / (control_total + treatment_total)
        se = (pooled * (1 - pooled) * (1/control_total + 1/treatment_total)) ** 0.5
        z = (treatment_rate - control_rate) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return {
            'control_rate': control_rate,
            'treatment_rate': treatment_rate,
            'lift': (treatment_rate - control_rate) / control_rate,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    ```
    
    **ML-Specific Considerations:**
    
    - **Interleaving**: Show both models' results mixed together
    - **Multi-armed bandits**: Adaptive allocation to better variants
    - **Guardrail metrics**: Ensure no degradation in key metrics
    - **Novelty effects**: New models may show initial boost that fades

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of production ML and experimentation.
        
        **Strong answer signals:**
        
        - Knows offline vs online metrics difference
        - Can calculate sample size for desired power
        - Mentions guardrail metrics and novelty effects
        - Knows when to use bandits vs traditional A/B tests

---

### Explain Different Types of Recommendation Systems - Netflix, Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Recommendation Systems`, `Collaborative Filtering`, `Content-Based` | **Asked by:** Netflix, Amazon, Google, Meta, Spotify

??? success "View Answer"

    **Types of Recommendation Systems:**
    
    | Type | Approach | Pros | Cons |
    |------|----------|------|------|
    | Collaborative Filtering | User-item interactions | Discovers unexpected | Cold start problem |
    | Content-Based | Item features | No cold start for items | Limited novelty |
    | Hybrid | Combines both | Best of both | More complex |
    
    **Collaborative Filtering:**
    
    ```python
    # User-based: Find similar users
    from sklearn.metrics.pairwise import cosine_similarity
    
    user_similarity = cosine_similarity(user_item_matrix)
    
    # Item-based: Find similar items
    item_similarity = cosine_similarity(user_item_matrix.T)
    
    # Matrix Factorization (SVD)
    from scipy.sparse.linalg import svds
    
    U, sigma, Vt = svds(user_item_matrix, k=50)
    predicted_ratings = np.dot(np.dot(U, np.diag(sigma)), Vt)
    ```
    
    **Deep Learning Approach:**
    
    ```python
    import torch.nn as nn
    
    class NeuralCollaborativeFiltering(nn.Module):
        def __init__(self, num_users, num_items, embed_dim=32):
            super().__init__()
            self.user_embed = nn.Embedding(num_users, embed_dim)
            self.item_embed = nn.Embedding(num_items, embed_dim)
            
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            
        def forward(self, user_ids, item_ids):
            user_emb = self.user_embed(user_ids)
            item_emb = self.item_embed(item_ids)
            x = torch.cat([user_emb, item_emb], dim=-1)
            return self.mlp(x).squeeze()
    ```
    
    **Content-Based:**
    
    ```python
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Create item profiles from descriptions
    tfidf = TfidfVectorizer(stop_words='english')
    item_features = tfidf.fit_transform(item_descriptions)
    
    # Create user profile from liked items
    user_profile = item_features[liked_items].mean(axis=0)
    
    # Recommend similar items
    similarities = cosine_similarity(user_profile, item_features)
    ```
    
    **Evaluation Metrics:**
    
    | Metric | Measures |
    |--------|----------|
    | Precision@K | Relevant items in top K |
    | Recall@K | Coverage of relevant items |
    | NDCG | Ranking quality |
    | MAP | Mean average precision |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of personalization systems.
        
        **Strong answer signals:**
        
        - Explains cold start problem and solutions
        - Knows matrix factorization vs deep learning trade-offs
        - Mentions implicit vs explicit feedback
        - Discusses evaluation: "We use NDCG because ranking matters"

---

### What is Imbalanced Data? How Do You Handle It in Classification? - Amazon, Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Class Imbalance`, `Classification`, `Sampling` | **Asked by:** Amazon, Google, Meta, Netflix, Microsoft

??? success "View Answer"

    **What is Imbalanced Data?**
    
    When one class significantly outnumbers others (e.g., 99% negative, 1% positive). Common in fraud detection, medical diagnosis, anomaly detection.
    
    **Why It's a Problem:**
    
    - Model learns to predict majority class
    - Accuracy is misleading (99% accuracy by predicting all negative)
    - Minority class patterns not learned
    
    **Strategies:**
    
    | Level | Technique |
    |-------|-----------|
    | Data | Oversampling, undersampling, SMOTE |
    | Algorithm | Class weights, anomaly detection |
    | Evaluation | Use F1, PR-AUC, not accuracy |
    
    ```python
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    
    # SMOTE oversampling
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    # Combination: SMOTE + undersampling
    pipeline = ImbPipeline([
        ('over', SMOTE(sampling_strategy=0.3)),
        ('under', RandomUnderSampler(sampling_strategy=0.5)),
        ('model', RandomForestClassifier())
    ])
    
    # Class weights (no resampling needed)
    from sklearn.utils.class_weight import compute_class_weight
    
    weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    model = RandomForestClassifier(class_weight='balanced')
    
    # Or in XGBoost
    scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
    model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight)
    ```
    
    **Threshold Tuning:**
    
    ```python
    from sklearn.metrics import precision_recall_curve
    
    # Get probabilities
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Find optimal threshold for F1
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    
    # Use custom threshold
    y_pred = (y_proba >= optimal_threshold).astype(int)
    ```
    
    **Evaluation for Imbalanced:**
    
    | Use | Don't Use |
    |-----|-----------|
    | Precision-Recall AUC | Accuracy |
    | F1-Score | ROC-AUC (can be misleading) |
    | Confusion Matrix | Single metric alone |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Practical classification handling.
        
        **Strong answer signals:**
        
        - Never uses accuracy as primary metric
        - Knows SMOTE and when to use it
        - Suggests class weights as simpler alternative
        - Mentions threshold tuning on PR curve

---

### How Do You Deploy ML Models to Production? - Amazon, Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `MLOps`, `Deployment`, `Production ML` | **Asked by:** Amazon, Google, Meta, Microsoft, Netflix

??? success "View Answer"

    **Deployment Approaches:**
    
    | Approach | Use Case | Latency |
    |----------|----------|---------|
    | Batch | Periodic predictions, reports | High (okay) |
    | Real-time API | Interactive applications | Low (critical) |
    | Edge | Mobile, IoT, offline | Very low |
    | Streaming | Continuous data processing | Medium |
    
    **Real-time API with FastAPI:**
    
    ```python
    from fastapi import FastAPI
    import joblib
    import numpy as np
    
    app = FastAPI()
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
    
    @app.post("/predict")
    async def predict(features: list[float]):
        X = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)
        probability = model.predict_proba(X_scaled)
        
        return {
            "prediction": int(prediction[0]),
            "probability": float(probability[0].max())
        }
    ```
    
    **Docker Containerization:**
    
    ```dockerfile
    FROM python:3.10-slim
    
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    
    COPY model.joblib .
    COPY app.py .
    
    CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
    ```
    
    **MLOps Considerations:**
    
    | Component | Tools |
    |-----------|-------|
    | Model Registry | MLflow, Weights & Biases |
    | Serving | TensorFlow Serving, Triton |
    | Monitoring | Prometheus, Grafana |
    | Feature Store | Feast, Tecton |
    | Pipeline | Airflow, Kubeflow |
    
    **Monitoring:**
    
    ```python
    # Track prediction drift
    from evidently import Report
    from evidently.metrics import DataDriftPreset
    
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=train_df, current_data=production_df)
    report.save_html("drift_report.html")
    ```
    
    **Model Versioning:**
    
    ```python
    import mlflow
    
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Production ML engineering skills.
        
        **Strong answer signals:**
        
        - Knows batch vs real-time trade-offs
        - Mentions containerization (Docker)
        - Discusses monitoring for drift
        - Knows model versioning and rollback strategies

---


### What is Linear Regression? Explain Assumptions and Diagnostics - Google, Amazon Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Regression`, `Statistics`, `Fundamentals` | **Asked by:** Google, Amazon, Meta, Microsoft

??? success "View Answer"

    **What is Linear Regression?**
    
    Linear regression models the relationship between a dependent variable and one or more independent variables using a linear function.
    
    **The Formula:**
    
    $$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n + \epsilon$$
    
    ```python
    from sklearn.linear_model import LinearRegression
    import numpy as np
    
    # Simple linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_}")
    print(f"R² Score: {model.score(X_test, y_test)}")
    ```
    
    **Key Assumptions:**
    
    | Assumption | Check Method |
    |------------|--------------|
    | Linearity | Residual vs fitted plot |
    | Independence | Durbin-Watson test |
    | Homoscedasticity | Residual spread plot |
    | Normality | Q-Q plot of residuals |
    | No multicollinearity | VIF (Variance Inflation Factor) |
    
    **Diagnostics:**
    
    ```python
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    # Check multicollinearity
    vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print("VIF:", dict(zip(X.columns, vif)))  # VIF > 5 = problem
    
    # Residual analysis
    residuals = y_test - model.predict(X_test)
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Statistical foundation knowledge.
        
        **Strong answer signals:**
        
        - Lists assumptions without prompting
        - Knows how to check each assumption
        - Mentions VIF for multicollinearity
        - Knows OLS minimizes squared residuals

---

### What is Logistic Regression? When to Use It? - Google, Amazon, Meta Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Classification`, `Probability`, `Fundamentals` | **Asked by:** Google, Amazon, Meta, Microsoft

??? success "View Answer"

    **What is Logistic Regression?**
    
    Logistic regression is a linear model for binary classification that outputs probabilities using the sigmoid function.
    
    **The Sigmoid Function:**
    
    $$P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + ... + \beta_n x_n)}}$$
    
    ```python
    from sklearn.linear_model import LogisticRegression
    
    model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs')
    model.fit(X_train, y_train)
    
    # Probabilities
    probabilities = model.predict_proba(X_test)
    
    # Coefficients (log-odds)
    print("Coefficients:", model.coef_)
    
    # Odds ratio interpretation
    import numpy as np
    odds_ratios = np.exp(model.coef_)
    print("Odds Ratios:", odds_ratios)
    ```
    
    **Interpretation:**
    
    | Coefficient | Interpretation |
    |-------------|----------------|
    | Positive | Increases probability of class 1 |
    | Negative | Decreases probability of class 1 |
    | Odds Ratio > 1 | Feature increases odds |
    | Odds Ratio < 1 | Feature decreases odds |
    
    **When to Use:**
    
    | Use Logistic Regression | Don't Use |
    |-------------------------|-----------|
    | Binary classification | Complex non-linear relationships |
    | Need interpretability | Multi-class (use softmax) |
    | Baseline model | Very high dimensional |
    | Feature importance needed | |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of probabilistic classification.
        
        **Strong answer signals:**
        
        - Knows it's called "regression" but used for classification
        - Can interpret coefficients as log-odds
        - Mentions maximum likelihood estimation
        - Knows regularization prevents overfitting

---

### What is Naive Bayes? Why is it "Naive"? - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Classification`, `Probability`, `Text Classification` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **What is Naive Bayes?**
    
    Naive Bayes is a probabilistic classifier based on Bayes' theorem with the "naive" assumption of feature independence.
    
    **Bayes' Theorem:**
    
    $$P(C|X) = \frac{P(X|C) \cdot P(C)}{P(X)}$$
    
    **The Naive Assumption:**
    
    Features are conditionally independent given the class:
    $$P(x_1, x_2, ..., x_n|C) = P(x_1|C) \cdot P(x_2|C) \cdot ... \cdot P(x_n|C)$$
    
    ```python
    from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
    
    # For continuous features
    gnb = GaussianNB()
    
    # For text/count data (most common)
    mnb = MultinomialNB(alpha=1.0)  # alpha = Laplace smoothing
    
    # For binary features
    bnb = BernoulliNB()
    
    # Text classification example
    from sklearn.feature_extraction.text import CountVectorizer
    
    vectorizer = CountVectorizer()
    X_train_counts = vectorizer.fit_transform(train_texts)
    
    mnb.fit(X_train_counts, y_train)
    predictions = mnb.predict(vectorizer.transform(test_texts))
    ```
    
    **Types:**
    
    | Type | Use Case | Feature Type |
    |------|----------|--------------|
    | Gaussian | Continuous data | Real numbers |
    | Multinomial | Text, word counts | Counts |
    | Bernoulli | Binary features | 0/1 |
    
    **Why It Works Despite Being "Naive":**
    
    - Classification only needs relative probabilities
    - Works well with high-dimensional data
    - Very fast training and prediction

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of probabilistic reasoning.
        
        **Strong answer signals:**
        
        - Explains the independence assumption and why it's unrealistic
        - Knows it performs well for text classification
        - Mentions Laplace smoothing for zero probabilities
        - Compares to logistic regression: "Similar performance, faster"

---

### What is Feature Selection? Compare Filter, Wrapper, and Embedded Methods - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Feature Engineering`, `Model Optimization`, `Dimensionality` | **Asked by:** Amazon, Google, Meta, Microsoft

??? success "View Answer"

    **Why Feature Selection?**
    
    - Reduce overfitting
    - Improve accuracy
    - Reduce training time
    - Improve interpretability
    
    **Three Approaches:**
    
    | Method | How It Works | Speed | Accuracy |
    |--------|--------------|-------|----------|
    | Filter | Statistical tests, independent of model | Fast | Lower |
    | Wrapper | Evaluates subsets with model | Slow | Higher |
    | Embedded | Selection during training | Medium | High |
    
    **Filter Methods:**
    
    ```python
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
    
    # ANOVA F-test (for classification)
    selector = SelectKBest(f_classif, k=10)
    X_selected = selector.fit_transform(X, y)
    
    # Correlation-based
    correlation_matrix = X.corr()
    high_corr_features = correlation_matrix[abs(correlation_matrix) > 0.8]
    
    # Variance threshold
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=0.01)
    ```
    
    **Wrapper Methods:**
    
    ```python
    from sklearn.feature_selection import RFE, RFECV
    from sklearn.ensemble import RandomForestClassifier
    
    # Recursive Feature Elimination
    rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=10)
    rfe.fit(X, y)
    selected_features = X.columns[rfe.support_]
    
    # With cross-validation
    rfecv = RFECV(estimator=RandomForestClassifier(), cv=5)
    rfecv.fit(X, y)
    ```
    
    **Embedded Methods:**
    
    ```python
    # L1 regularization (Lasso)
    from sklearn.linear_model import LassoCV
    lasso = LassoCV(cv=5).fit(X, y)
    selected = X.columns[lasso.coef_ != 0]
    
    # Tree-based feature importance
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier().fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top_features = importances.nlargest(10).index
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Practical ML pipeline knowledge.
        
        **Strong answer signals:**
        
        - Knows trade-offs between methods
        - Uses filter for large datasets, wrapper for smaller
        - Mentions L1/Lasso as embedded selection
        - Warns about target leakage in feature selection

---

### What is Ensemble Learning? Explain Bagging, Boosting, and Stacking - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Ensemble Methods`, `Model Combination`, `Advanced` | **Asked by:** Google, Amazon, Meta, Netflix

??? success "View Answer"

    **What is Ensemble Learning?**
    
    Combining multiple models to produce better predictions than any single model.
    
    **Three Main Approaches:**
    
    | Method | How It Works | Reduces |
    |--------|--------------|---------|
    | Bagging | Parallel models on bootstrap samples | Variance |
    | Boosting | Sequential models fixing errors | Bias |
    | Stacking | Meta-model on base predictions | Both |
    
    **Bagging (Bootstrap Aggregating):**
    
    ```python
    from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
    
    # Random Forest is bagging + feature randomization
    rf = RandomForestClassifier(n_estimators=100, max_features='sqrt')
    
    # Generic bagging
    bagging = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=50,
        max_samples=0.8,
        bootstrap=True
    )
    ```
    
    **Boosting:**
    
    ```python
    from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
    import xgboost as xgb
    import lightgbm as lgb
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1)
    
    # LightGBM (faster)
    lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1)
    ```
    
    **Stacking:**
    
    ```python
    from sklearn.ensemble import StackingClassifier
    
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('xgb', xgb.XGBClassifier(n_estimators=100)),
        ('lgb', lgb.LGBMClassifier(n_estimators=100))
    ]
    
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=5
    )
    ```
    
    **Comparison:**
    
    | Aspect | Bagging | Boosting |
    |--------|---------|----------|
    | Training | Parallel | Sequential |
    | Goal | Reduce variance | Reduce bias |
    | Prone to overfitting | Less | More |
    | Example | Random Forest | XGBoost |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Advanced ML knowledge.
        
        **Strong answer signals:**
        
        - Explains variance vs bias reduction
        - Knows Random Forest = bagging + random features
        - Mentions early stopping for boosting overfitting
        - Can describe when to use each method

---

### How Do You Handle Missing Data? - Amazon, Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Data Preprocessing`, `Missing Data`, `Imputation` | **Asked by:** Amazon, Google, Meta, Microsoft

??? success "View Answer"

    **Types of Missing Data:**
    
    | Type | Description | Handling |
    |------|-------------|----------|
    | MCAR | Missing Completely at Random | Any method |
    | MAR | Missing at Random (depends on observed) | Model-based imputation |
    | MNAR | Missing Not at Random | Domain knowledge needed |
    
    **Basic Methods:**
    
    ```python
    import pandas as pd
    from sklearn.impute import SimpleImputer
    
    # Check missing
    print(df.isnull().sum())
    
    # Drop rows with missing
    df_clean = df.dropna()
    
    # Drop columns with > 50% missing
    df_clean = df.dropna(thresh=len(df) * 0.5, axis=1)
    
    # Simple imputation
    imputer = SimpleImputer(strategy='mean')  # or median, most_frequent
    X_imputed = imputer.fit_transform(X)
    ```
    
    **Advanced Imputation:**
    
    ```python
    from sklearn.impute import KNNImputer
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    
    # KNN Imputation
    knn_imputer = KNNImputer(n_neighbors=5)
    X_imputed = knn_imputer.fit_transform(X)
    
    # MICE (Multiple Imputation by Chained Equations)
    mice_imputer = IterativeImputer(max_iter=10, random_state=42)
    X_imputed = mice_imputer.fit_transform(X)
    ```
    
    **Indicator Variables:**
    
    ```python
    # Add missing indicator
    from sklearn.impute import SimpleImputer, MissingIndicator
    
    indicator = MissingIndicator()
    missing_flags = indicator.fit_transform(X)
    
    # Combine imputed data with indicators
    X_with_indicators = np.hstack([X_imputed, missing_flags])
    ```
    
    **Best Practices:**
    
    | Missing % | Recommendation |
    |-----------|----------------|
    | < 5% | Simple imputation |
    | 5-20% | Advanced imputation (KNN, MICE) |
    | > 20% | Consider dropping or domain knowledge |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Data quality handling skills.
        
        **Strong answer signals:**
        
        - Asks about missing mechanism (MCAR, MAR, MNAR)
        - Knows adding missing indicators can help
        - Uses IterativeImputer/MICE for complex cases
        - Warns: "Always impute after train/test split"

---

### What is Time Series Forecasting? Explain ARIMA and Its Components - Amazon, Google Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Time Series`, `Forecasting`, `ARIMA` | **Asked by:** Amazon, Google, Meta, Netflix

??? success "View Answer"

    **Time Series Components:**
    
    | Component | Description |
    |-----------|-------------|
    | Trend | Long-term increase/decrease |
    | Seasonality | Regular periodic patterns |
    | Cyclical | Non-fixed period fluctuations |
    | Noise | Random variation |
    
    **ARIMA (AutoRegressive Integrated Moving Average):**
    
    - **AR(p)**: AutoRegressive - uses past values
    - **I(d)**: Integrated - differencing for stationarity
    - **MA(q)**: Moving Average - uses past errors
    
    ```python
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    import pandas as pd
    
    # Check stationarity (ADF test)
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]}, p-value: {result[1]}")
    
    # Fit ARIMA
    model = ARIMA(series, order=(p, d, q))  # (AR, differencing, MA)
    fitted = model.fit()
    
    # Forecast
    forecast = fitted.forecast(steps=30)
    
    # Auto ARIMA
    from pmdarima import auto_arima
    auto_model = auto_arima(series, seasonal=True, m=12)  # m=12 for monthly
    ```
    
    **Choosing Parameters (p, d, q):**
    
    | Parameter | How to Choose |
    |-----------|---------------|
    | d | Number of differences for stationarity |
    | p | ACF cuts off, PACF decays |
    | q | PACF cuts off, ACF decays |
    
    **Modern Alternatives:**
    
    ```python
    # Prophet (Facebook)
    from prophet import Prophet
    model = Prophet(yearly_seasonality=True)
    model.fit(df)  # df with 'ds' and 'y' columns
    
    # Deep Learning
    # LSTM, Transformer models for complex patterns
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Time series understanding.
        
        **Strong answer signals:**
        
        - Checks stationarity first (ADF test)
        - Knows ACF/PACF for parameter selection
        - Mentions Prophet for quick results
        - Uses walk-forward validation, not random split

---

### What is Gradient Boosted Trees? How Does XGBoost Work? - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Boosting`, `XGBoost`, `Ensemble` | **Asked by:** Amazon, Google, Meta, Microsoft

??? success "View Answer"

    **How Gradient Boosting Works:**
    
    1. Fit initial model (e.g., mean)
    2. Calculate residuals (errors)
    3. Fit new tree to predict residuals
    4. Add new tree's predictions (with learning rate)
    5. Repeat
    
    **XGBoost Innovations:**
    
    | Feature | Benefit |
    |---------|---------|
    | Regularization | L1/L2 on leaf weights |
    | Sparsity awareness | Efficient missing value handling |
    | Weighted quantile sketch | Approximate tree learning |
    | Cache-aware access | 10x faster |
    | Block structure | Parallelization |
    
    ```python
    import xgboost as xgb
    
    # Basic model
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,  # L1
        reg_lambda=1.0,  # L2
        early_stopping_rounds=10
    )
    
    # Training with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    
    # Feature importance
    xgb.plot_importance(model)
    ```
    
    **LightGBM vs XGBoost:**
    
    | Aspect | XGBoost | LightGBM |
    |--------|---------|----------|
    | Tree growth | Level-wise | Leaf-wise |
    | Speed | Fast | Faster |
    | Memory | Higher | Lower |
    | Categorical | Needs encoding | Native support |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Practical tree ensemble knowledge.
        
        **Strong answer signals:**
        
        - Explains sequential fitting to residuals
        - Knows key hyperparameters (learning_rate, max_depth)
        - Uses early stopping to prevent overfitting
        - Compares XGBoost vs LightGBM trade-offs

---

### How Do You Evaluate Regression Models? - Amazon, Google Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Evaluation`, `Regression`, `Metrics` | **Asked by:** Amazon, Google, Meta, Microsoft

??? success "View Answer"

    **Common Regression Metrics:**
    
    | Metric | Formula | Interpretation |
    |--------|---------|----------------|
    | MAE | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ | Average error magnitude |
    | MSE | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Penalizes large errors |
    | RMSE | $\sqrt{MSE}$ | Same scale as target |
    | R² | $1 - \frac{SS_{res}}{SS_{tot}}$ | Variance explained |
    | MAPE | $\frac{100}{n}\sum|\frac{y_i - \hat{y}_i}{y_i}|$ | Percentage error |
    
    ```python
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # MAPE (handle zeros)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    ```
    
    **Choosing the Right Metric:**
    
    | Use Case | Best Metric |
    |----------|-------------|
    | Same units as target | MAE, RMSE |
    | Penalize large errors | RMSE, MSE |
    | Compare across scales | MAPE, R² |
    | Outlier-resistant | MAE |
    
    **Adjusted R²:**
    
    $$R^2_{adj} = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$$
    
    Penalizes adding features that don't improve fit.

    !!! tip "Interviewer's Insight"
        **What they're testing:** Evaluation metric knowledge.
        
        **Strong answer signals:**
        
        - Knows RMSE vs MAE trade-offs
        - Uses adjusted R² when comparing models
        - Mentions residual plots for diagnostics
        - Warns about MAPE with values near zero

---

### What is Dimensionality Reduction? Compare PCA and t-SNE - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Dimensionality Reduction`, `Visualization`, `PCA` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Why Reduce Dimensions?**
    
    - Combat curse of dimensionality
    - Reduce noise
    - Enable visualization (2D/3D)
    - Speed up training
    
    **PCA (Principal Component Analysis):**
    
    ```python
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Standardize first!
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit PCA
    pca = PCA(n_components=0.95)  # Keep 95% variance
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"Components: {pca.n_components_}")
    print(f"Explained variance: {pca.explained_variance_ratio_}")
    
    # Visualize variance explained
    import matplotlib.pyplot as plt
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Components')
    plt.ylabel('Cumulative Variance')
    ```
    
    **t-SNE (t-Distributed Stochastic Neighbor Embedding):**
    
    ```python
    from sklearn.manifold import TSNE
    
    # Usually for visualization only (2-3D)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis')
    ```
    
    **Comparison:**
    
    | Aspect | PCA | t-SNE |
    |--------|-----|-------|
    | Type | Linear | Non-linear |
    | Goal | Maximize variance | Preserve local structure |
    | Speed | Fast | Slow |
    | Deterministic | Yes | No |
    | Inverse transform | Yes | No |
    | Use case | Feature reduction | Visualization |
    
    **UMAP (Modern Alternative):**
    
    ```python
    import umap
    reducer = umap.UMAP(n_components=2)
    X_umap = reducer.fit_transform(X)
    # Faster than t-SNE, preserves global structure better
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of data representation.
        
        **Strong answer signals:**
        
        - Standardizes data before PCA
        - Knows PCA for features, t-SNE for visualization
        - Mentions perplexity tuning for t-SNE
        - Suggests UMAP as modern alternative

---

### What is Neural Network Optimization? Explain Adam and Learning Rate Schedules - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Deep Learning`, `Optimization`, `Training` | **Asked by:** Google, Meta, Amazon, Apple

??? success "View Answer"

    **Optimizers:**
    
    | Optimizer | Description | Use Case |
    |-----------|-------------|----------|
    | SGD | Basic gradient descent | Large-scale, convex |
    | Momentum | SGD with velocity | Faster convergence |
    | RMSprop | Adaptive learning rates | Non-stationary |
    | Adam | Momentum + RMSprop | Default choice |
    | AdamW | Adam + weight decay | Transformers |
    
    **Adam Optimizer:**
    
    $$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
    $$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
    $$\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$
    
    ```python
    import torch.optim as optim
    
    # Adam with default parameters
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    
    # AdamW for transformers
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    ```
    
    **Learning Rate Schedules:**
    
    ```python
    from torch.optim.lr_scheduler import (
        StepLR, ExponentialLR, CosineAnnealingLR, OneCycleLR
    )
    
    # Step decay
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Cosine annealing (popular)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)
    
    # One cycle (fast training)
    scheduler = OneCycleLR(optimizer, max_lr=0.01, epochs=10, steps_per_epoch=len(train_loader))
    
    # Training loop
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
        scheduler.step()
    ```
    
    **Learning Rate Finding:**
    
    ```python
    # Start low, increase exponentially, find where loss decreases fastest
    # Use lr_finder from pytorch-lightning or fastai
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Deep learning training expertise.
        
        **Strong answer signals:**
        
        - Uses Adam as default, knows when to use SGD
        - Implements learning rate scheduling
        - Knows warmup for transformers
        - Can explain momentum and adaptive learning rates

---

### What is Regularization? Compare L1, L2, Dropout, and Early Stopping - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Regularization`, `Overfitting`, `Training` | **Asked by:** Google, Amazon, Meta, Microsoft

??? success "View Answer"

    **Why Regularization?**
    
    Prevents overfitting by constraining model complexity.
    
    **Types of Regularization:**
    
    | Method | How It Works | Effect |
    |--------|--------------|--------|
    | L1 (Lasso) | Penalize sum of absolute weights | Sparse weights (feature selection) |
    | L2 (Ridge) | Penalize sum of squared weights | Small weights (prevents extreme values) |
    | Dropout | Randomly zero neurons during training | Ensemble effect |
    | Early Stopping | Stop when validation loss increases | Limits training time |
    
    **L1 vs L2:**
    
    ```python
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    
    # L2 regularization
    ridge = Ridge(alpha=1.0)
    
    # L1 regularization (sparse coefficients)
    lasso = Lasso(alpha=0.1)
    
    # Combination (Elastic Net)
    elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
    
    # In neural networks
    import torch.nn as nn
    
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.01)  # L2
    ```
    
    **Dropout:**
    
    ```python
    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 256)
            self.dropout = nn.Dropout(p=0.5)  # 50% dropout
            self.fc2 = nn.Linear(256, 10)
            
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout(x)  # Only during training
            x = self.fc2(x)
            return x
    ```
    
    **Early Stopping:**
    
    ```python
    # XGBoost
    model.fit(X_train, y_train, 
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=10)
    
    # PyTorch (manual)
    best_loss = float('inf')
    patience = 10
    counter = 0
    
    for epoch in range(epochs):
        val_loss = validate(model)
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            save_model(model)
        else:
            counter += 1
            if counter >= patience:
                break
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of overfitting prevention.
        
        **Strong answer signals:**
        
        - Knows L1 leads to sparsity (feature selection)
        - Uses dropout only during training
        - Implements early stopping with patience
        - Combines multiple regularization techniques

---

### What is the Curse of Dimensionality? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `High Dimensions`, `Feature Engineering`, `Theory` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **What is the Curse of Dimensionality?**
    
    As dimensions increase, data becomes increasingly sparse, making distance-based methods and density estimation unreliable.
    
    **Problems:**
    
    | Problem | Implication |
    |---------|-------------|
    | Data sparsity | Need exponentially more data |
    | Distance concentration | All points equidistant |
    | Computational cost | Memory and time explode |
    | Overfitting | More features = more noise |
    
    **Distance Concentration:**
    
    As dimensions → ∞, the ratio of nearest to farthest neighbor approaches 1:
    
    $$\lim_{d \to \infty} \frac{dist_{max} - dist_{min}}{dist_{min}} = 0$$
    
    ```python
    import numpy as np
    from sklearn.metrics.pairwise import euclidean_distances
    
    # Demonstrate distance concentration
    for d in [2, 10, 100, 1000]:
        X = np.random.randn(100, d)
        distances = euclidean_distances(X)
        ratio = (distances.max() - distances.min()) / distances.min()
        print(f"Dimensions: {d}, Max-Min Ratio: {ratio:.4f}")
    ```
    
    **Solutions:**
    
    ```python
    # 1. Dimensionality reduction
    from sklearn.decomposition import PCA
    X_reduced = PCA(n_components=50).fit_transform(X)
    
    # 2. Feature selection
    from sklearn.feature_selection import SelectKBest
    X_selected = SelectKBest(k=100).fit_transform(X, y)
    
    # 3. Use regularization
    from sklearn.linear_model import LassoCV
    model = LassoCV(cv=5).fit(X, y)
    
    # 4. Use tree-based models (less affected)
    from sklearn.ensemble import RandomForestClassifier
    ```
    
    **Rule of Thumb:**
    
    Need at least $5^d$ samples for $d$ dimensions to maintain data density.

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of high-dimensional data.
        
        **Strong answer signals:**
        
        - Explains why KNN fails in high dimensions
        - Knows distance metrics become meaningless
        - Suggests dimensionality reduction or regularization
        - Mentions: "Tree models are less affected"

---

### What is Cross-Entropy Loss? When to Use It? - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Loss Functions`, `Classification`, `Deep Learning` | **Asked by:** Google, Meta, Amazon, Microsoft

??? success "View Answer"

    **What is Cross-Entropy Loss?**
    
    Measures the distance between predicted probability distribution and true distribution.
    
    **Binary Cross-Entropy:**
    
    $$L = -\frac{1}{N}\sum_{i=1}^{N}[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$
    
    **Categorical Cross-Entropy:**
    
    $$L = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C}y_{i,c}\log(\hat{y}_{i,c})$$
    
    ```python
    import torch.nn as nn
    
    # Binary classification
    criterion = nn.BCELoss()  # With sigmoid output
    criterion = nn.BCEWithLogitsLoss()  # Raw logits (preferred)
    
    # Multi-class classification
    criterion = nn.CrossEntropyLoss()  # Raw logits (includes softmax)
    
    # Example
    logits = model(X)  # Shape: (batch_size, num_classes)
    loss = criterion(logits, targets)  # targets: (batch_size,) - class indices
    ```
    
    **Why Cross-Entropy?**
    
    | Loss | Gradient | Use |
    |------|----------|-----|
    | MSE | Small when wrong | Regression |
    | Cross-Entropy | Large when wrong | Classification |
    
    **Label Smoothing:**
    
    ```python
    # Prevents overconfident predictions
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Soft targets: Instead of [0, 1, 0]
    # Use: [0.05, 0.9, 0.05]
    ```
    
    **Focal Loss (Imbalanced Data):**
    
    ```python
    class FocalLoss(nn.Module):
        def __init__(self, gamma=2.0, alpha=0.25):
            super().__init__()
            self.gamma = gamma
            self.alpha = alpha
            
        def forward(self, inputs, targets):
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
            return focal_loss.mean()
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Loss function understanding.
        
        **Strong answer signals:**
        
        - Knows cross-entropy for probabilities, MSE for values
        - Uses BCEWithLogitsLoss for numerical stability
        - Mentions label smoothing for regularization
        - Knows focal loss for imbalanced data

---

### How Do You Handle Categorical Features? - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Feature Engineering`, `Encoding`, `Preprocessing` | **Asked by:** Amazon, Google, Meta, Microsoft

??? success "View Answer"

    **Encoding Methods:**
    
    | Method | Use Case | Cardinality |
    |--------|----------|-------------|
    | One-Hot | Tree models, low cardinality | < 10-15 |
    | Label Encoding | Tree models | Any |
    | Target Encoding | High cardinality | > 15 |
    | Frequency Encoding | When frequency matters | Any |
    | Embeddings | Deep learning | Very high |
    
    **One-Hot Encoding:**
    
    ```python
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder
    
    # Pandas
    df_encoded = pd.get_dummies(df, columns=['category'])
    
    # Scikit-learn
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df[['category']])
    ```
    
    **Target Encoding:**
    
    ```python
    from category_encoders import TargetEncoder
    
    encoder = TargetEncoder(smoothing=1.0)
    df['category_encoded'] = encoder.fit_transform(df['category'], df['target'])
    
    # Manual with smoothing
    global_mean = df['target'].mean()
    smoothing = 10
    
    agg = df.groupby('category')['target'].agg(['mean', 'count'])
    smoothed = (agg['count'] * agg['mean'] + smoothing * global_mean) / (agg['count'] + smoothing)
    df['category_encoded'] = df['category'].map(smoothed)
    ```
    
    **Embedding (Deep Learning):**
    
    ```python
    import torch.nn as nn
    
    class ModelWithEmbedding(nn.Module):
        def __init__(self, num_categories, embedding_dim):
            super().__init__()
            self.embedding = nn.Embedding(num_categories, embedding_dim)
            self.fc = nn.Linear(embedding_dim + n_numeric_features, 1)
            
        def forward(self, cat_features, num_features):
            cat_embedded = self.embedding(cat_features)
            x = torch.cat([cat_embedded, num_features], dim=1)
            return self.fc(x)
    ```
    
    **Best Practices:**
    
    | Model Type | Recommendation |
    |------------|----------------|
    | Linear | One-hot or target encoding |
    | Tree-based | Label or target encoding |
    | Neural Net | Embeddings |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Feature engineering skills.
        
        **Strong answer signals:**
        
        - Chooses encoding based on cardinality
        - Knows target encoding needs CV to avoid leakage
        - Uses embeddings for high cardinality in DL
        - Mentions CatBoost handles categoricals natively

---

### What is Model Calibration? - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Probability`, `Calibration`, `Evaluation` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **What is Calibration?**
    
    A model is well-calibrated if predicted probabilities match observed frequencies. If model says 70% probability, event should occur 70% of the time.
    
    **Why It Matters:**
    
    - Probability thresholding
    - Risk assessment
    - Decision making
    - Ensemble weighting
    
    **Checking Calibration:**
    
    ```python
    from sklearn.calibration import calibration_curve
    import matplotlib.pyplot as plt
    
    # Get probabilities
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, y_prob, n_bins=10
    )
    
    plt.plot(mean_predicted_value, fraction_of_positives, 's-')
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    ```
    
    **Calibration Methods:**
    
    ```python
    from sklearn.calibration import CalibratedClassifierCV
    
    # Platt scaling (logistic regression on probabilities)
    calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=5)
    
    # Isotonic regression (non-parametric)
    calibrated = CalibratedClassifierCV(model, method='isotonic', cv=5)
    
    calibrated.fit(X_train, y_train)
    calibrated_probs = calibrated.predict_proba(X_test)[:, 1]
    ```
    
    **Brier Score:**
    
    $$BS = \frac{1}{N}\sum_{i=1}^{N}(p_i - y_i)^2$$
    
    ```python
    from sklearn.metrics import brier_score_loss
    
    brier = brier_score_loss(y_test, y_prob)
    print(f"Brier Score: {brier:.4f}")  # Lower is better
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Advanced probability understanding.
        
        **Strong answer signals:**
        
        - Knows neural networks are often overconfident
        - Uses calibration curve for diagnosis
        - Chooses Platt (low data) vs isotonic (more data)
        - Mentions Brier score for evaluation

---

### What is Online Learning? - Amazon, Google Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Online Learning`, `Streaming`, `Production` | **Asked by:** Amazon, Google, Meta, Netflix

??? success "View Answer"

    **What is Online Learning?**
    
    Updating model incrementally as new data arrives, instead of retraining on entire dataset.
    
    **Use Cases:**
    
    | Use Case | Why Online |
    |----------|------------|
    | Streaming data | Too much to store |
    | Concept drift | Data distribution changes |
    | Real-time adaptation | Need immediate updates |
    | Resource constraints | Can't retrain frequently |
    
    **Scikit-learn partial_fit:**
    
    ```python
    from sklearn.linear_model import SGDClassifier
    
    model = SGDClassifier(loss='log_loss')  # Logistic regression
    
    # Initial training
    model.partial_fit(X_batch1, y_batch1, classes=[0, 1])
    
    # Incremental updates
    for X_batch, y_batch in stream:
        model.partial_fit(X_batch, y_batch)
    ```
    
    **Algorithms that Support Online Learning:**
    
    | Algorithm | Scikit-learn Class |
    |-----------|-------------------|
    | SGD Classifier | SGDClassifier |
    | SGD Regressor | SGDRegressor |
    | Naive Bayes | MultinomialNB |
    | Perceptron | Perceptron |
    | Mini-batch K-Means | MiniBatchKMeans |
    
    **River Library (Dedicated Online ML):**
    
    ```python
    from river import linear_model, preprocessing
    
    model = (
        preprocessing.StandardScaler() | 
        linear_model.LogisticRegression()
    )
    
    for x, y in stream:
        y_pred = model.predict_one(x)
        model.learn_one(x, y)
    ```
    
    **Handling Concept Drift:**
    
    - **Window-based**: Train on recent N samples
    - **Decay**: Weight recent samples more
    - **Drift detection**: Monitor performance, reset when needed

    !!! tip "Interviewer's Insight"
        **What they're testing:** Streaming/production ML knowledge.
        
        **Strong answer signals:**
        
        - Knows when to use online vs batch
        - Mentions concept drift
        - Uses partial_fit in scikit-learn
        - Knows decay/windowing strategies

---

### What is Semi-Supervised Learning? - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Semi-Supervised`, `Label Propagation`, `Learning Paradigms` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **What is Semi-Supervised Learning?**
    
    Uses both labeled and unlabeled data for training. Useful when labeling is expensive but data is abundant.
    
    **Approaches:**
    
    | Method | Description |
    |--------|-------------|
    | Self-training | Train, predict unlabeled, add confident predictions |
    | Co-training | Two models teach each other |
    | Label propagation | Spread labels through similarity graph |
    | Pseudo-labeling | Use model predictions as labels |
    
    **Self-Training:**
    
    ```python
    from sklearn.semi_supervised import SelfTrainingClassifier
    from sklearn.ensemble import RandomForestClassifier
    
    # -1 indicates unlabeled
    y_train_partial = y_train.copy()
    y_train_partial[unlabeled_mask] = -1
    
    model = SelfTrainingClassifier(
        RandomForestClassifier(),
        threshold=0.9,  # Confidence threshold
        max_iter=10
    )
    model.fit(X_train, y_train_partial)
    ```
    
    **Label Propagation:**
    
    ```python
    from sklearn.semi_supervised import LabelPropagation
    
    model = LabelPropagation(kernel='knn', n_neighbors=7)
    model.fit(X_train, y_train_partial)  # -1 for unlabeled
    
    # Get transduced labels
    transduced_labels = model.transduction_
    ```
    
    **Pseudo-Labeling (Deep Learning):**
    
    ```python
    # 1. Train on labeled data
    model.fit(X_labeled, y_labeled)
    
    # 2. Predict unlabeled with confidence
    probs = model.predict_proba(X_unlabeled)
    confident_mask = probs.max(axis=1) > 0.95
    pseudo_labels = probs.argmax(axis=1)[confident_mask]
    
    # 3. Add to training set
    X_combined = np.vstack([X_labeled, X_unlabeled[confident_mask]])
    y_combined = np.hstack([y_labeled, pseudo_labels])
    
    # 4. Retrain
    model.fit(X_combined, y_combined)
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Knowledge of learning paradigms.
        
        **Strong answer signals:**
        
        - Explains when it's useful (expensive labeling)
        - Knows confidence thresholding to avoid noise
        - Mentions transductive vs inductive
        - Compares to active learning

---

### What is Active Learning? - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Active Learning`, `Labeling`, `Human-in-the-Loop` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **What is Active Learning?**
    
    Model actively selects which samples to label, reducing labeling cost while maximizing performance.
    
    **Query Strategies:**
    
    | Strategy | How It Works |
    |----------|--------------|
    | Uncertainty Sampling | Select least confident predictions |
    | Query by Committee | Select where models disagree most |
    | Expected Model Change | Select that would change model most |
    | Diversity Sampling | Select diverse samples |
    
    **Uncertainty Sampling:**
    
    ```python
    from modAL.uncertainty import uncertainty_sampling
    from modAL.models import ActiveLearner
    from sklearn.ensemble import RandomForestClassifier
    
    # Initialize with few labeled samples
    learner = ActiveLearner(
        estimator=RandomForestClassifier(),
        query_strategy=uncertainty_sampling,
        X_training=X_initial,
        y_training=y_initial
    )
    
    # Active learning loop
    for _ in range(n_queries):
        query_idx, query_instance = learner.query(X_unlabeled)
        
        # Get label from oracle (human)
        label = get_label_from_human(query_instance)
        
        learner.teach(query_instance, label)
        
        # Remove from unlabeled pool
        X_unlabeled = np.delete(X_unlabeled, query_idx, axis=0)
    ```
    
    **Manual Implementation:**
    
    ```python
    # Uncertainty-based selection
    probs = model.predict_proba(X_unlabeled)
    
    # Least confident
    uncertainty = 1 - probs.max(axis=1)
    
    # Margin (difference between top 2)
    sorted_probs = np.sort(probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    
    # Entropy
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    
    # Select top uncertain samples
    query_indices = np.argsort(uncertainty)[-n_samples:]
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Efficient labeling strategies.
        
        **Strong answer signals:**
        
        - Knows different query strategies
        - Mentions exploration vs exploitation trade-off
        - Uses margin or entropy for uncertainty
        - Knows batch mode for efficiency

---

### What is AutoML? - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `AutoML`, `Automation`, `Model Selection` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **What is AutoML?**
    
    Automated Machine Learning - automating model selection, hyperparameter tuning, and feature engineering.
    
    **AutoML Components:**
    
    | Component | What It Automates |
    |-----------|-------------------|
    | Data preprocessing | Imputation, encoding, scaling |
    | Feature engineering | Transformations, interactions |
    | Model selection | Algorithm choice |
    | Hyperparameter tuning | Parameter optimization |
    | Ensembling | Combining models |
    
    **Popular AutoML Tools:**
    
    ```python
    # Auto-sklearn
    from autosklearn.classification import AutoSklearnClassifier
    
    automl = AutoSklearnClassifier(time_left_for_this_task=3600)
    automl.fit(X_train, y_train)
    
    # H2O AutoML
    import h2o
    from h2o.automl import H2OAutoML
    
    h2o.init()
    aml = H2OAutoML(max_runtime_secs=3600)
    aml.train(x=features, y=target, training_frame=train)
    
    # TPOT
    from tpot import TPOTClassifier
    
    tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2)
    tpot.fit(X_train, y_train)
    tpot.export('best_pipeline.py')
    ```
    
    **Google Cloud AutoML:**
    
    ```python
    # Vertex AI AutoML
    from google.cloud import aiplatform
    
    dataset = aiplatform.TabularDataset.create(
        display_name="my_dataset",
        gcs_source="gs://bucket/data.csv"
    )
    
    job = aiplatform.AutoMLTabularTrainingJob(
        display_name="my_model",
        optimization_prediction_type="classification"
    )
    
    model = job.run(dataset=dataset, target_column="label")
    ```
    
    **When to Use AutoML:**
    
    | Use AutoML | Don't Use |
    |------------|-----------|
    | Quick baseline | Need interpretability |
    | Limited ML expertise | Complex domain constraints |
    | Standard ML problems | Need custom architectures |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Awareness of automation tools.
        
        **Strong answer signals:**
        
        - Knows popular tools (auto-sklearn, H2O, TPOT)
        - Uses AutoML for baselines, then improves
        - Mentions computational cost
        - Knows when manual modeling is better

---

### What is Early Stopping and How Does It Work? - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Regularization`, `Training`, `Overfitting` | **Asked by:** Google, Meta, Amazon, Microsoft

??? success "View Answer"

    **Early Stopping:**
    
    Stop training when validation performance stops improving, preventing overfitting while maintaining optimal generalization.
    
    **Implementation:**
    
    ```python
    from sklearn.model_selection import train_test_split
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)
    
    # PyTorch example
    import torch
    import torch.nn as nn
    
    class EarlyStopping:
        def __init__(self, patience=5, min_delta=0):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_loss = None
            self.early_stop = False
            
        def __call__(self, val_loss):
            if self.best_loss is None:
                self.best_loss = val_loss
            elif val_loss > self.best_loss - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_loss = val_loss
                self.counter = 0
    
    # Training loop
    early_stopping = EarlyStopping(patience=10)
    
    for epoch in range(1000):
        # Train
        model.train()
        train_loss = train_one_epoch(model, train_loader)
        
        # Validate
        model.eval()
        val_loss = validate(model, val_loader)
        
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break
    ```
    
    **Keras Built-in:**
    
    ```python
    from tensorflow.keras.callbacks import EarlyStopping
    
    # Define callback
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        min_delta=0.001
    )
    
    # Train with early stopping
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=1000,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Visualize
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.axvline(x=len(history.history['loss']), color='r', 
                linestyle='--', label='Early Stop')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Early Stopping Visualization')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Over Time')
    
    plt.tight_layout()
    plt.show()
    ```
    
    **Key Parameters:**
    
    | Parameter | Meaning | Typical Value |
    |-----------|---------|---------------|
    | patience | Epochs to wait before stopping | 5-20 |
    | min_delta | Minimum improvement threshold | 0.001 |
    | restore_best_weights | Restore best model | True |
    | monitor | Metric to track | val_loss |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Practical training techniques.
        
        **Strong answer signals:**
        
        - "Stop when validation stops improving"
        - Mentions patience parameter
        - "Prevents overfitting naturally"
        - restore_best_weights is important
        - Use with validation set, not test

---

### Explain Learning Rate Scheduling - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Optimization`, `Training`, `Deep Learning` | **Asked by:** Google, Amazon, Meta, Microsoft

??? success "View Answer"

    **Learning Rate Scheduling:**
    
    Adjust learning rate during training to improve convergence and final performance.
    
    **Common Schedules:**
    
    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    
    epochs = 100
    initial_lr = 0.1
    
    # 1. Step Decay
    def step_decay(epoch, initial_lr=0.1, drop=0.5, epochs_drop=20):
        return initial_lr * (drop ** (epoch // epochs_drop))
    
    # 2. Exponential Decay
    def exponential_decay(epoch, initial_lr=0.1, decay_rate=0.96):
        return initial_lr * (decay_rate ** epoch)
    
    # 3. Cosine Annealing
    def cosine_annealing(epoch, initial_lr=0.1, T_max=100):
        return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / T_max))
    
    # 4. Linear Decay
    def linear_decay(epoch, initial_lr=0.1, total_epochs=100):
        return initial_lr * (1 - epoch / total_epochs)
    
    # 5. Warmup + Decay
    def warmup_cosine(epoch, initial_lr=0.1, warmup_epochs=10, total_epochs=100):
        if epoch < warmup_epochs:
            return initial_lr * (epoch / warmup_epochs)
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return initial_lr * 0.5 * (1 + np.cos(np.pi * progress))
    
    # Visualize schedules
    epochs_range = np.arange(epochs)
    
    plt.figure(figsize=(14, 8))
    
    schedules = {
        'Step Decay': [step_decay(e) for e in epochs_range],
        'Exponential Decay': [exponential_decay(e) for e in epochs_range],
        'Cosine Annealing': [cosine_annealing(e) for e in epochs_range],
        'Linear Decay': [linear_decay(e) for e in epochs_range],
        'Warmup + Cosine': [warmup_cosine(e) for e in epochs_range]
    }
    
    for name, lr_values in schedules.items():
        plt.plot(epochs_range, lr_values, linewidth=2, label=name)
    
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedules')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.show()
    ```
    
    **PyTorch Implementation:**
    
    ```python
    import torch
    import torch.optim as optim
    from torch.optim.lr_scheduler import *
    
    model = YourModel()
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    
    # 1. StepLR
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    # 2. ExponentialLR
    scheduler = ExponentialLR(optimizer, gamma=0.95)
    
    # 3. CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
    
    # 4. ReduceLROnPlateau (adaptive)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', 
                                  factor=0.1, patience=10)
    
    # 5. OneCycleLR (super-convergence)
    scheduler = OneCycleLR(optimizer, max_lr=0.1, 
                           steps_per_epoch=len(train_loader), 
                           epochs=100)
    
    # Training loop
    for epoch in range(num_epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            loss = compute_loss(model, batch)
            loss.backward()
            optimizer.step()
            
            # For OneCycleLR, step per batch
            if isinstance(scheduler, OneCycleLR):
                scheduler.step()
        
        # For others, step per epoch
        if not isinstance(scheduler, OneCycleLR):
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        print(f"Epoch {epoch}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    ```
    
    **Keras Implementation:**
    
    ```python
    from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
    
    # Custom schedule
    def lr_schedule(epoch):
        initial_lr = 0.1
        if epoch < 30:
            return initial_lr
        elif epoch < 60:
            return initial_lr * 0.1
        else:
            return initial_lr * 0.01
    
    scheduler = LearningRateScheduler(lr_schedule)
    
    # Or adaptive
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              callbacks=[scheduler])
    ```
    
    **When to Use Each:**
    
    | Schedule | Use Case |
    |----------|----------|
    | Step Decay | Simple, predictable decay |
    | Cosine | Smooth decay, good final performance |
    | ReduceLROnPlateau | Adaptive, no manual tuning |
    | OneCycleLR | Fast training, super-convergence |
    | Warmup | Large models, stabilize early training |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Advanced training strategies.
        
        **Strong answer signals:**
        
        - Lists multiple schedules
        - "Start high, decay slowly"
        - Mentions warmup for transformers
        - Cosine annealing popular in research
        - ReduceLROnPlateau for adaptive

---

### What is Data Leakage and How Do You Prevent It? - Most Tech Companies Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Data Preprocessing`, `Model Evaluation`, `Best Practices` | **Asked by:** Google, Amazon, Meta, Microsoft

??? success "View Answer"

    **Data Leakage:**
    
    Information from outside the training dataset is used to create the model, leading to overly optimistic performance.
    
    **Types of Leakage:**
    
    **1. Train-Test Contamination:**
    
    ```python
    # WRONG: Fit scaler on all data
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Uses test data!
    X_train, X_test = train_test_split(X_scaled)
    
    # CORRECT: Fit only on training data
    X_train, X_test = train_test_split(X)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Only transform
    ```
    
    **2. Target Leakage:**
    
    ```python
    # Example: Predicting loan default
    # WRONG: Using features known only after outcome
    features = [
        'credit_score',
        'income',
        'paid_back',  # ❌ This is the target!
        'num_missed_payments'  # ❌ Known only after default
    ]
    
    # CORRECT: Only use features available at prediction time
    features = [
        'credit_score',
        'income',
        'employment_length',
        'previous_defaults'
    ]
    ```
    
    **3. Temporal Leakage:**
    
    ```python
    # Time series: use future to predict past
    
    # WRONG: Random split
    X_train, X_test = train_test_split(time_series_data, shuffle=True)
    
    # CORRECT: Temporal split
    split_date = '2023-01-01'
    train_data = time_series_data[time_series_data['date'] < split_date]
    test_data = time_series_data[time_series_data['date'] >= split_date]
    
    # For time series CV
    from sklearn.model_selection import TimeSeriesSplit
    
    tscv = TimeSeriesSplit(n_splits=5)
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        # Train and evaluate
    ```
    
    **4. Feature Engineering Leakage:**
    
    ```python
    import pandas as pd
    import numpy as np
    
    # WRONG: Creating features using global statistics
    df['income_vs_mean'] = df['income'] - df['income'].mean()  # Uses test data mean!
    
    # CORRECT: Use training statistics only
    def create_features(train_df, test_df):
        # Compute statistics on training data
        train_mean = train_df['income'].mean()
        train_std = train_df['income'].std()
        
        # Apply to both
        train_df['income_normalized'] = (train_df['income'] - train_mean) / train_std
        test_df['income_normalized'] = (test_df['income'] - train_mean) / train_std
        
        return train_df, test_df
    ```
    
    **5. Group Leakage:**
    
    ```python
    # Multiple samples from same entity split across train/test
    
    # Example: Patient data with multiple visits
    # WRONG: Random split might put same patient in both sets
    
    # CORRECT: Split by patient ID
    from sklearn.model_selection import GroupShuffleSplit
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    
    for train_idx, test_idx in gss.split(X, y, groups=patient_ids):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    ```
    
    **Detection Strategies:**
    
    ```python
    # 1. Check for suspiciously high performance
    from sklearn.metrics import roc_auc_score
    
    auc = roc_auc_score(y_test, y_pred)
    if auc > 0.99:
        print("⚠️ Warning: Suspiciously high AUC. Check for leakage!")
    
    # 2. Feature importance analysis
    import shap
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Check if top features make logical sense
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': np.abs(shap_values).mean(0)
    }).sort_values('importance', ascending=False)
    
    print("Top features:")
    print(feature_importance.head(10))
    
    # 3. Compare train vs test performance
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    if train_score - test_score > 0.1:
        print("⚠️ Large train-test gap. Possible overfitting or leakage.")
    ```
    
    **Prevention Checklist:**
    
    | Check | Question |
    |-------|----------|
    | ✓ | Split data before any preprocessing? |
    | ✓ | Fit transformers only on training data? |
    | ✓ | Time-based split for temporal data? |
    | ✓ | Group-based split for related samples? |
    | ✓ | Features available at prediction time? |
    | ✓ | No target in features? |
    | ✓ | Reasonable performance (not too good)? |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Production ML awareness.
        
        **Strong answer signals:**
        
        - Concrete examples of leakage types
        - "Fit on train, transform on test"
        - Temporal leakage in time series
        - "Performance too good to be true"
        - Mentions pipeline safety

---

### Explain Calibration in Machine Learning - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Probability`, `Model Evaluation`, `Classification` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **Calibration:**
    
    A model is calibrated if P(y=1 | score=s) = s. That is, predicted probabilities match actual frequencies.
    
    **Why It Matters:**
    
    ```python
    # Uncalibrated model: says 90% confident, but only right 60% of time
    # Calibrated model: says 90% confident, and right 90% of time
    
    # Critical for:
    # - Medical diagnosis (need true probabilities)
    # - Risk assessment (financial, insurance)
    # - Decision-making under uncertainty
    ```
    
    **Measuring Calibration:**
    
    ```python
    from sklearn.calibration import calibration_curve
    import matplotlib.pyplot as plt
    
    # Get predicted probabilities
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    ```
    
    **Calibration Methods:**
    
    ```python
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    
    # Method 1: Platt Scaling (Logistic Regression)
    base_model = RandomForestClassifier(n_estimators=100)
    calibrated_platt = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
    calibrated_platt.fit(X_train, y_train)
    
    # Method 2: Isotonic Regression
    calibrated_isotonic = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
    calibrated_isotonic.fit(X_train, y_train)
    
    # Compare calibrations
    models = {
        'Uncalibrated RF': base_model.fit(X_train, y_train),
        'Platt Scaling': calibrated_platt,
        'Isotonic': calibrated_isotonic
    }
    
    plt.figure(figsize=(10, 6))
    
    for name, clf in models.items():
        y_prob = clf.predict_proba(X_test)[:, 1]
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=name)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    ```
    
    **Brier Score (Calibration Metric):**
    
    ```python
    from sklearn.metrics import brier_score_loss
    
    # Lower is better (0 = perfect)
    brier_uncal = brier_score_loss(y_test, 
                                    base_model.predict_proba(X_test)[:, 1])
    brier_platt = brier_score_loss(y_test, 
                                    calibrated_platt.predict_proba(X_test)[:, 1])
    brier_iso = brier_score_loss(y_test, 
                                  calibrated_isotonic.predict_proba(X_test)[:, 1])
    
    print(f"Brier Score:")
    print(f"  Uncalibrated: {brier_uncal:.4f}")
    print(f"  Platt Scaling: {brier_platt:.4f}")
    print(f"  Isotonic: {brier_iso:.4f}")
    ```
    
    **Which Models Need Calibration:**
    
    | Model | Naturally Calibrated? |
    |-------|----------------------|
    | Logistic Regression | ✓ Usually yes |
    | Naive Bayes | ✓ Usually yes |
    | Random Forest | ✗ No (overconfident) |
    | Gradient Boosting | ✗ No (overconfident) |
    | SVM | ✗ No (not probabilities) |
    | Neural Networks | △ Depends (often uncalibrated) |
    
    **Temperature Scaling (Neural Networks):**
    
    ```python
    import torch
    import torch.nn as nn
    
    class TemperatureScaling(nn.Module):
        def __init__(self):
            super().__init__()
            self.temperature = nn.Parameter(torch.ones(1))
        
        def forward(self, logits):
            return logits / self.temperature
        
        def fit(self, logits, labels):
            """Learn optimal temperature"""
            optimizer = torch.optim.LBFGS([self.temperature], lr=0.01)
            
            def eval():
                optimizer.zero_grad()
                loss = nn.CrossEntropyLoss()(self.forward(logits), labels)
                loss.backward()
                return loss
            
            optimizer.step(eval)
            return self.temperature.item()
    
    # Usage
    ts = TemperatureScaling()
    optimal_temp = ts.fit(val_logits, val_labels)
    
    # Apply to test set
    calibrated_probs = torch.softmax(test_logits / optimal_temp, dim=1)
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Beyond accuracy metrics.
        
        **Strong answer signals:**
        
        - "Predicted probabilities match reality"
        - Mentions calibration curve
        - Platt scaling, isotonic regression
        - "Random forests often uncalibrated"
        - Critical for decision-making

---

### What is Knowledge Distillation? - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Model Compression`, `Transfer Learning`, `Deep Learning` | **Asked by:** Google, Meta, Amazon, Microsoft

??? success "View Answer"

    **Knowledge Distillation:**
    
    Train a smaller "student" model to mimic a larger "teacher" model, transferring knowledge while reducing size/complexity.
    
    **Key Concept:**
    
    ```python
    # Teacher: Large, accurate model
    # Student: Small, fast model
    
    # Student learns from:
    # 1. Hard labels: y_true (ground truth)
    # 2. Soft labels: teacher's probability distribution
    
    # Soft labels contain more information:
    # Instead of [0, 1, 0] (hard)
    # Learn from [0.05, 0.9, 0.05] (soft) - reveals similarities
    ```
    
    **Implementation:**
    
    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class DistillationLoss(nn.Module):
        def __init__(self, temperature=3.0, alpha=0.5):
            super().__init__()
            self.temperature = temperature
            self.alpha = alpha  # Weight for distillation vs hard labels
        
        def forward(self, student_logits, teacher_logits, labels):
            # Soft targets from teacher
            soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
            soft_prob = F.log_softmax(student_logits / self.temperature, dim=1)
            
            # Distillation loss (KL divergence)
            distillation_loss = F.kl_div(
                soft_prob,
                soft_targets,
                reduction='batchmean'
            ) * (self.temperature ** 2)
            
            # Hard label loss
            student_loss = F.cross_entropy(student_logits, labels)
            
            # Combined loss
            total_loss = (
                self.alpha * distillation_loss + 
                (1 - self.alpha) * student_loss
            )
            
            return total_loss
    
    # Training
    teacher_model.eval()  # Teacher in eval mode
    student_model.train()
    
    distill_loss = DistillationLoss(temperature=3.0, alpha=0.7)
    
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            # Get teacher predictions
            with torch.no_grad():
                teacher_logits = teacher_model(images)
            
            # Get student predictions
            student_logits = student_model(images)
            
            # Compute distillation loss
            loss = distill_loss(student_logits, teacher_logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    ```
    
    **TensorFlow/Keras Implementation:**
    
    ```python
    import tensorflow as tf
    from tensorflow import keras
    
    class Distiller(keras.Model):
        def __init__(self, student, teacher):
            super().__init__()
            self.teacher = teacher
            self.student = student
        
        def compile(self, optimizer, metrics, student_loss_fn, 
                   distillation_loss_fn, alpha=0.1, temperature=3):
            super().compile(optimizer=optimizer, metrics=metrics)
            self.student_loss_fn = student_loss_fn
            self.distillation_loss_fn = distillation_loss_fn
            self.alpha = alpha
            self.temperature = temperature
        
        def train_step(self, data):
            x, y = data
            
            # Forward pass of teacher
            teacher_predictions = self.teacher(x, training=False)
            
            with tf.GradientTape() as tape:
                # Forward pass of student
                student_predictions = self.student(x, training=True)
                
                # Compute losses
                student_loss = self.student_loss_fn(y, student_predictions)
                
                distillation_loss = self.distillation_loss_fn(
                    tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                    tf.nn.softmax(student_predictions / self.temperature, axis=1),
                )
                
                loss = (
                    self.alpha * student_loss + 
                    (1 - self.alpha) * distillation_loss
                )
            
            # Compute gradients
            trainable_vars = self.student.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            
            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            
            # Update metrics
            self.compiled_metrics.update_state(y, student_predictions)
            
            return {m.name: m.result() for m in self.metrics}
    
    # Usage
    distiller = Distiller(student=student_model, teacher=teacher_model)
    distiller.compile(
        optimizer=keras.optimizers.Adam(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=keras.losses.KLDivergence(),
        alpha=0.1,
        temperature=3
    )
    
    distiller.fit(train_dataset, epochs=10, validation_data=val_dataset)
    ```
    
    **Why Temperature Matters:**
    
    ```python
    import numpy as np
    
    logits = np.array([2.0, 1.0, 0.1])
    
    # Low temperature (T=1): Peaked distribution
    probs_T1 = np.exp(logits / 1) / np.sum(np.exp(logits / 1))
    print(f"T=1: {probs_T1}")  # [0.66, 0.24, 0.10]
    
    # High temperature (T=5): Smooth distribution
    probs_T5 = np.exp(logits / 5) / np.sum(np.exp(logits / 5))
    print(f"T=5: {probs_T5}")  # [0.42, 0.34, 0.24]
    
    # Higher T → More information in "dark knowledge"
    ```
    
    **Results Comparison:**
    
    ```python
    # Evaluate models
    from sklearn.metrics import accuracy_score
    
    # Teacher (large model)
    teacher_preds = teacher_model.predict(X_test)
    teacher_acc = accuracy_score(y_test, teacher_preds.argmax(1))
    
    # Student without distillation
    student_scratch = train_from_scratch(student_model, X_train, y_train)
    scratch_acc = accuracy_score(y_test, student_scratch.predict(X_test).argmax(1))
    
    # Student with distillation
    student_distilled = train_with_distillation(student_model, teacher_model, X_train, y_train)
    distill_acc = accuracy_score(y_test, student_distilled.predict(X_test).argmax(1))
    
    print(f"Teacher accuracy: {teacher_acc:.3f}")
    print(f"Student (from scratch): {scratch_acc:.3f}")
    print(f"Student (distilled): {distill_acc:.3f}")
    
    # Typical: Distilled student >> Student from scratch
    # And much faster than teacher!
    ```
    
    **Benefits:**
    
    | Benefit | Description |
    |---------|-------------|
    | Model compression | 10-100x smaller |
    | Speed | 5-10x faster inference |
    | Better generalization | Learns from soft labels |
    | Knowledge transfer | Ensemble → Single model |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Advanced model optimization.
        
        **Strong answer signals:**
        
        - "Student learns from teacher's soft labels"
        - Temperature smooths distribution
        - "Dark knowledge" in wrong class probabilities
        - Combined loss: distillation + hard labels
        - Use case: Deploy smaller model to production

---

### Explain Ensemble Methods: Bagging vs Boosting - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Ensemble`, `Random Forest`, `Gradient Boosting` | **Asked by:** Amazon, Google, Meta, Microsoft

??? success "View Answer"

    **Ensemble Methods:**
    
    Combine multiple models to improve performance beyond individual models.
    
    **Bagging (Bootstrap Aggregating):**
    
    ```python
    # Train models in parallel on different random subsets
    # Average predictions (regression) or vote (classification)
    
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier
    
    # Base model
    base_model = DecisionTreeClassifier(max_depth=10)
    
    # Bagging ensemble
    bagging = BaggingClassifier(
        base_estimator=base_model,
        n_estimators=100,
        max_samples=0.8,  # Use 80% of data per model
        max_features=0.8,  # Use 80% of features per model
        bootstrap=True,    # Sample with replacement
        n_jobs=-1
    )
    
    bagging.fit(X_train, y_train)
    
    # Predictions
    predictions = bagging.predict(X_test)
    
    # Individual model predictions
    individual_preds = np.array([
        estimator.predict(X_test) 
        for estimator in bagging.estimators_
    ])
    
    # Variance across models (diversity)
    prediction_variance = individual_preds.var(axis=0).mean()
    print(f"Prediction diversity (variance): {prediction_variance:.3f}")
    ```
    
    **Boosting:**
    
    ```python
    # Train models sequentially, each correcting previous errors
    
    from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
    
    # AdaBoost: Reweight samples
    adaboost = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1),  # Weak learners
        n_estimators=100,
        learning_rate=1.0
    )
    
    adaboost.fit(X_train, y_train)
    
    # Gradient Boosting: Fit residuals
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8
    )
    
    gb.fit(X_train, y_train)
    ```
    
    **Key Differences:**
    
    | Aspect | Bagging | Boosting |
    |--------|---------|----------|
    | Training | Parallel | Sequential |
    | Sampling | Random with replacement | Weighted by errors |
    | Base models | Strong learners (deep trees) | Weak learners (stumps) |
    | Goal | Reduce variance | Reduce bias |
    | Overfitting risk | Low | Higher (if too many iterations) |
    | Example | Random Forest | AdaBoost, XGBoost |
    
    **Visual Comparison:**
    
    ```python
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    import matplotlib.pyplot as plt
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=2, n_informative=2,
                               n_redundant=0, n_clusters_per_class=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # Models
    models = {
        'Single Tree': DecisionTreeClassifier(max_depth=10),
        'Bagging (RF)': RandomForestClassifier(n_estimators=100, max_depth=10),
        'Boosting (GB)': GradientBoostingClassifier(n_estimators=100, max_depth=3)
    }
    
    # Train and evaluate
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import learning_curve
    
    plt.figure(figsize=(15, 5))
    
    for idx, (name, model) in enumerate(models.items(), 1):
        model.fit(X_train, y_train)
        
        # Learning curve
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        plt.subplot(1, 3, idx)
        plt.plot(train_sizes, train_scores.mean(axis=1), label='Training')
        plt.plot(train_sizes, test_scores.mean(axis=1), label='Validation')
        plt.xlabel('Training Size')
        plt.ylabel('Accuracy')
        plt.title(f'{name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    ```
    
    **Custom Bagging:**
    
    ```python
    # Implement bagging from scratch
    
    class SimpleBagging:
        def __init__(self, base_model, n_estimators=10):
            self.base_model = base_model
            self.n_estimators = n_estimators
            self.models = []
        
        def fit(self, X, y):
            n_samples = len(X)
            
            for _ in range(self.n_estimators):
                # Bootstrap sample
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_bootstrap = X[indices]
                y_bootstrap = y[indices]
                
                # Train model
                model = clone(self.base_model)
                model.fit(X_bootstrap, y_bootstrap)
                self.models.append(model)
        
        def predict(self, X):
            # Collect predictions
            predictions = np.array([model.predict(X) for model in self.models])
            
            # Majority vote
            return np.apply_along_axis(
                lambda x: np.bincount(x).argmax(),
                axis=0,
                arr=predictions
            )
    
    # Usage
    from sklearn.base import clone
    
    bagging = SimpleBagging(DecisionTreeClassifier(max_depth=5), n_estimators=50)
    bagging.fit(X_train, y_train)
    predictions = bagging.predict(X_test)
    ```
    
    **When to Use:**
    
    | Use Bagging When | Use Boosting When |
    |------------------|-------------------|
    | High variance models (deep trees) | High bias models (linear) |
    | Need fast parallel training | Can afford sequential training |
    | Want robustness | Want best performance |
    | Overfitting is main concern | Underfitting is main concern |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Ensemble understanding.
        
        **Strong answer signals:**
        
        - Bagging: parallel, reduce variance
        - Boosting: sequential, reduce bias
        - Random Forest = bagging + feature randomness
        - "Boosting can overfit, bagging rarely does"
        - Mentions diversity in ensemble

---

### What is Batch Normalization and Why Is It Effective? - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Deep Learning`, `Normalization`, `Training` | **Asked by:** Google, Meta, Amazon, Microsoft

??? success "View Answer"

    **Batch Normalization:**
    
    Normalize layer inputs during training to stabilize and accelerate learning.
    
    **Formula:**
    
    $$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
    $$y = \gamma \hat{x} + \beta$$
    
    Where:
    - μ_B, σ_B: Batch mean and variance
    - γ, β: Learnable parameters
    - ε: Small constant for numerical stability
    
    **Implementation:**
    
    ```python
    import torch
    import torch.nn as nn
    
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 256)
            self.bn1 = nn.BatchNorm1d(256)  # Add BN after linear
            
            self.fc2 = nn.Linear(256, 128)
            self.bn2 = nn.BatchNorm1d(128)
            
            self.fc3 = nn.Linear(128, 10)
        
        def forward(self, x):
            # Flatten
            x = x.view(x.size(0), -1)
            
            # Layer 1: Linear -> BN -> ReLU
            x = self.fc1(x)
            x = self.bn1(x)
            x = F.relu(x)
            
            # Layer 2: Linear -> BN -> ReLU
            x = self.fc2(x)
            x = self.bn2(x)
            x = F.relu(x)
            
            # Output layer
            x = self.fc3(x)
            return x
    
    # For CNNs
    class ConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
            self.bn1 = nn.BatchNorm2d(64)  # 2D for conv layers
            
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
            self.bn2 = nn.BatchNorm2d(128)
        
        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            return x
    ```
    
    **TensorFlow/Keras:**
    
    ```python
    from tensorflow.keras import layers, models
    
    model = models.Sequential([
        layers.Dense(256, input_shape=(784,)),
        layers.BatchNormalization(),  # Add BN
        layers.Activation('relu'),
        
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        
        layers.Dense(10, activation='softmax')
    ])
    ```
    
    **Why It Works:**
    
    ```python
    # 1. Reduces Internal Covariate Shift
    # - Layer inputs stay in similar range during training
    # - Prevents exploding/vanishing gradients
    
    # 2. Allows higher learning rates
    # - More stable training
    
    # 3. Acts as regularization
    # - Noise from batch statistics
    # - Can reduce need for dropout
    
    # 4. Makes network less sensitive to initialization
    ```
    
    **Training vs Inference:**
    
    ```python
    # Key difference: Running statistics
    
    # During TRAINING:
    # - Use batch mean and variance
    # - Update running mean/variance (momentum=0.1)
    
    # During INFERENCE:
    # - Use running mean and variance (population statistics)
    # - No batch dependency
    
    model.train()  # Uses batch statistics
    model.eval()   # Uses running statistics
    
    # PyTorch example
    bn = nn.BatchNorm1d(256, momentum=0.1)
    
    # Training mode
    bn.train()
    output_train = bn(input_batch)
    
    # Eval mode
    bn.eval()
    output_eval = bn(input_single)  # Can handle single sample
    ```
    
    **From Scratch:**
    
    ```python
    import numpy as np
    
    class BatchNorm:
        def __init__(self, num_features, momentum=0.1, eps=1e-5):
            self.gamma = np.ones(num_features)
            self.beta = np.zeros(num_features)
            self.momentum = momentum
            self.eps = eps
            
            # Running statistics
            self.running_mean = np.zeros(num_features)
            self.running_var = np.ones(num_features)
        
        def forward(self, x, training=True):
            if training:
                # Compute batch statistics
                batch_mean = x.mean(axis=0)
                batch_var = x.var(axis=0)
                
                # Normalize
                x_norm = (x - batch_mean) / np.sqrt(batch_var + self.eps)
                
                # Update running statistics
                self.running_mean = (
                    (1 - self.momentum) * self.running_mean + 
                    self.momentum * batch_mean
                )
                self.running_var = (
                    (1 - self.momentum) * self.running_var + 
                    self.momentum * batch_var
                )
            else:
                # Use running statistics
                x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            
            # Scale and shift
            out = self.gamma * x_norm + self.beta
            return out
    ```
    
    **Empirical Comparison:**
    
    ```python
    # Compare with and without BN
    
    from tensorflow.keras import callbacks
    
    # Without BN
    model_no_bn = models.Sequential([
        layers.Dense(256, activation='relu', input_shape=(784,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    # With BN
    model_with_bn = models.Sequential([
        layers.Dense(256, input_shape=(784,)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    # Train both
    history_no_bn = model_no_bn.fit(X_train, y_train, epochs=20, 
                                     validation_split=0.2, verbose=0)
    history_with_bn = model_with_bn.fit(X_train, y_train, epochs=20,
                                         validation_split=0.2, verbose=0)
    
    # Compare convergence
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history_no_bn.history['loss'], label='Without BN')
    plt.plot(history_with_bn.history['loss'], label='With BN')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.title('Training Convergence')
    
    plt.subplot(1, 2, 2)
    plt.plot(history_no_bn.history['val_accuracy'], label='Without BN')
    plt.plot(history_with_bn.history['val_accuracy'], label='With BN')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.title('Validation Performance')
    
    plt.tight_layout()
    plt.show()
    ```
    
    **Common Issues:**
    
    | Issue | Solution |
    |-------|----------|
    | Small batch size | Use Group Norm or Layer Norm |
    | RNNs/variable length | Use Layer Norm |
    | BN before/after activation? | Usually after linear, before activation |
    | Inference speed | Fuse BN into weights for deployment |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Deep learning training techniques.
        
        **Strong answer signals:**
        
        - Normalizes to mean=0, std=1 per batch
        - Learnable scale (γ) and shift (β)
        - Different behavior train vs eval
        - "Reduces internal covariate shift"
        - Allows higher learning rates

---

### Explain Residual Networks (ResNet) and Skip Connections - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Deep Learning`, `CNN`, `Architecture` | **Asked by:** Google, Meta, Amazon, Microsoft

??? success "View Answer"

    **Residual Networks (ResNet):**
    
    Use skip connections to enable training of very deep networks (100+ layers) by addressing vanishing gradient problem.
    
    **Key Innovation - Skip Connections:**
    
    $$y = F(x, \{W_i\}) + x$$
    
    Instead of learning H(x), learn residual F(x) = H(x) - x
    
    **Implementation:**
    
    ```python
    import torch
    import torch.nn as nn
    
    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            
            # Main path
            self.conv1 = nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=3, stride=stride, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            
            self.conv2 = nn.Conv2d(out_channels, out_channels,
                                   kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)
            
            # Skip connection (identity)
            self.skip = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                # Projection shortcut to match dimensions
                self.skip = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 
                             kernel_size=1, stride=stride),
                    nn.BatchNorm2d(out_channels)
                )
        
        def forward(self, x):
            # Main path
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            
            # Add skip connection
            out += self.skip(x)
            out = F.relu(out)
            
            return out
    
    # Build ResNet
    class ResNet(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            
            # Residual blocks
            self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
            self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
            self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
            self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)
            
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, num_classes)
        
        def _make_layer(self, in_channels, out_channels, num_blocks, stride):
            layers = []
            # First block may downsample
            layers.append(ResidualBlock(in_channels, out_channels, stride))
            # Rest maintain dimensions
            for _ in range(1, num_blocks):
                layers.append(ResidualBlock(out_channels, out_channels, 1))
            return nn.Sequential(*layers)
        
        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    ```
    
    **TensorFlow/Keras:**
    
    ```python
    from tensorflow.keras import layers, models
    
    def residual_block(x, filters, stride=1):
        # Main path
        y = layers.Conv2D(filters, 3, stride=stride, padding='same')(x)
        y = layers.BatchNormalization()(y)
        y = layers.Activation('relu')(y)
        
        y = layers.Conv2D(filters, 3, padding='same')(y)
        y = layers.BatchNormalization()(y)
        
        # Skip connection
        if stride != 1 or x.shape[-1] != filters:
            x = layers.Conv2D(filters, 1, stride=stride)(x)
            x = layers.BatchNormalization()(x)
        
        # Add skip
        out = layers.Add()([x, y])
        out = layers.Activation('relu')(out)
        
        return out
    
    # Build model
    inputs = layers.Input(shape=(224, 224, 3))
    x = layers.Conv2D(64, 7, stride=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(3, stride=2, padding='same')(x)
    
    # Add residual blocks
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)
    
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    ```
    
    **Why Skip Connections Work:**
    
    ```python
    # 1. Gradient Flow
    # Without skip: gradient must flow through many layers
    # With skip: gradient has direct path backward
    
    # 2. Identity Mapping
    # Easy to learn identity: just set F(x) = 0
    # Worst case: no degradation from adding layers
    
    # 3. Ensemble Effect
    # ResNet can be viewed as ensemble of many paths
    # Each path is a different depth network
    ```
    
    **Comparison:**
    
    ```python
    # Compare plain network vs ResNet
    
    class PlainNet(nn.Module):
        """Standard deep network without skip connections"""
        def __init__(self):
            super().__init__()
            layers = []
            in_ch = 64
            for i in range(20):  # 20 conv layers
                layers.extend([
                    nn.Conv2d(in_ch, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU()
                ])
            self.layers = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.layers(x)
    
    # Training comparison
    plain_net = PlainNet()
    resnet = ResNet()
    
    # Train both
    for epoch in range(10):
        # Plain network: gradients vanish, training stagnates
        # ResNet: gradients flow well, continues improving
        pass
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Modern architecture knowledge.
        
        **Strong answer signals:**
        
        - Skip connections: y = F(x) + x
        - Solves vanishing gradients
        - "Learn residual is easier than identity"
        - Enables 100+ layer networks
        - Won ImageNet 2015

---

### What is the Attention Mechanism? - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `NLP`, `Deep Learning`, `Transformers` | **Asked by:** Google, Meta, OpenAI, Amazon

??? success "View Answer"

    **Attention Mechanism:**
    
    Allows model to focus on relevant parts of input when producing output.
    
    **Core Formula:**
    
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
    
    Where:
    - Q: Query matrix
    - K: Key matrix  
    - V: Value matrix
    - d_k: Dimension of keys (for scaling)
    
    **Implementation:**
    
    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import math
    
    class ScaledDotProductAttention(nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, Q, K, V, mask=None):
            """
            Q: [batch_size, n_queries, d_k]
            K: [batch_size, n_keys, d_k]
            V: [batch_size, n_keys, d_v]
            """
            d_k = Q.size(-1)
            
            # Compute attention scores
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
            
            # Apply mask (optional, for padding)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            # Apply softmax
            attention_weights = F.softmax(scores, dim=-1)
            
            # Apply attention to values
            output = torch.matmul(attention_weights, V)
            
            return output, attention_weights
    
    # Example usage
    batch_size, seq_len, d_model = 2, 10, 512
    
    Q = torch.randn(batch_size, seq_len, d_model)
    K = torch.randn(batch_size, seq_len, d_model)
    V = torch.randn(batch_size, seq_len, d_model)
    
    attention = ScaledDotProductAttention()
    output, weights = attention(Q, K, V)
    
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    ```
    
    **Multi-Head Attention:**
    
    ```python
    class MultiHeadAttention(nn.Module):
        def __init__(self, d_model=512, num_heads=8):
            super().__init__()
            assert d_model % num_heads == 0
            
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads
            
            # Linear projections
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)
            
            self.attention = ScaledDotProductAttention()
        
        def forward(self, Q, K, V, mask=None):
            batch_size = Q.size(0)
            
            # Linear projections in batch
            Q = self.W_q(Q)  # [batch, seq_len, d_model]
            K = self.W_k(K)
            V = self.W_v(V)
            
            # Split into multiple heads
            Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            # Now: [batch, num_heads, seq_len, d_k]
            
            # Apply attention
            x, attention_weights = self.attention(Q, K, V, mask)
            
            # Concatenate heads
            x = x.transpose(1, 2).contiguous()
            x = x.view(batch_size, -1, self.d_model)
            
            # Final linear projection
            output = self.W_o(x)
            
            return output, attention_weights
    
    # Usage
    mha = MultiHeadAttention(d_model=512, num_heads=8)
    output, weights = mha(Q, K, V)
    ```
    
    **Types of Attention:**
    
    ```python
    # 1. Self-Attention (Q=K=V)
    class SelfAttention(nn.Module):
        def __init__(self, d_model, num_heads):
            super().__init__()
            self.mha = MultiHeadAttention(d_model, num_heads)
        
        def forward(self, x, mask=None):
            # Q, K, V are all the same input
            return self.mha(x, x, x, mask)
    
    # 2. Cross-Attention (Q≠K=V)
    class CrossAttention(nn.Module):
        def __init__(self, d_model, num_heads):
            super().__init__()
            self.mha = MultiHeadAttention(d_model, num_heads)
        
        def forward(self, query, key_value, mask=None):
            # Query from one source, Key/Value from another
            return self.mha(query, key_value, key_value, mask)
    
    # 3. Masked Self-Attention (for autoregressive models)
    def create_causal_mask(seq_len):
        """Prevent attending to future positions"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, False)
        return mask
    ```
    
    **Visualization:**
    
    ```python
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Simple attention example
    def visualize_attention(attention_weights, sentence):
        """
        attention_weights: [seq_len, seq_len]
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention_weights, 
                    xticklabels=sentence,
                    yticklabels=sentence,
                    cmap='viridis',
                    cbar=True)
        plt.xlabel('Key positions')
        plt.ylabel('Query positions')
        plt.title('Attention Weights')
        plt.show()
    
    # Example sentence
    sentence = "The cat sat on the mat".split()
    
    # Compute attention (simplified)
    seq_len = len(sentence)
    embeddings = torch.randn(1, seq_len, 64)
    
    attention = ScaledDotProductAttention()
    output, weights = attention(embeddings, embeddings, embeddings)
    
    visualize_attention(weights[0].detach().numpy(), sentence)
    ```
    
    **Why Attention Works:**
    
    | Benefit | Explanation |
    |---------|-------------|
    | Long-range dependencies | Can attend to any position |
    | Parallelizable | No sequential dependencies |
    | Interpretable | Attention weights show what model focuses on |
    | Flexible | Works for various sequence lengths |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Modern NLP/DL knowledge.
        
        **Strong answer signals:**
        
        - Query, Key, Value matrices
        - Softmax over Key-Query similarity
        - Multi-head for different representations
        - Scaled by sqrt(d_k) for stability
        - Transformer = multi-head attention + FFN

---

### Explain Feature Importance Methods - Most Tech Companies Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Interpretability`, `Feature Engineering`, `Model Evaluation` | **Asked by:** Google, Amazon, Meta, Microsoft

??? success "View Answer"

    **Feature Importance Methods:**
    
    Quantify the contribution of each feature to model predictions.
    
    **1. Tree-Based Importance (Gini/Gain):**
    
    ```python
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Train model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Get feature importances
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(10))
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'][:10], 
             feature_importance['importance'][:10])
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importances')
    plt.gca().invert_yaxis()
    plt.show()
    ```
    
    **2. Permutation Importance:**
    
    ```python
    from sklearn.inspection import permutation_importance
    
    # More reliable than built-in importances
    # Measures drop in performance when feature is shuffled
    
    perm_importance = permutation_importance(
        rf, X_test, y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )
    
    perm_imp_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    # Plot with error bars
    plt.figure(figsize=(10, 6))
    top_features = perm_imp_df.head(10)
    plt.barh(top_features['feature'], top_features['importance_mean'],
             xerr=top_features['importance_std'])
    plt.xlabel('Permutation Importance')
    plt.title('Top 10 Features (Permutation Importance)')
    plt.gca().invert_yaxis()
    plt.show()
    ```
    
    **3. SHAP (SHapley Additive exPlanations):**
    
    ```python
    import shap
    
    # Tree model
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)
    
    # Summary plot (global importance)
    shap.summary_plot(shap_values[1], X_test, plot_type="bar")
    
    # Detailed summary (shows distributions)
    shap.summary_plot(shap_values[1], X_test)
    
    # Individual prediction explanation
    shap.force_plot(explainer.expected_value[1], 
                    shap_values[1][0], 
                    X_test.iloc[0])
    
    # Dependence plot (feature interaction)
    shap.dependence_plot("age", shap_values[1], X_test)
    ```
    
    **4. LIME (Local Interpretable Model-agnostic Explanations):**
    
    ```python
    from lime import lime_tabular
    
    # Create explainer
    explainer = lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns,
        class_names=['class_0', 'class_1'],
        mode='classification'
    )
    
    # Explain a prediction
    exp = explainer.explain_instance(
        X_test.iloc[0].values,
        rf.predict_proba,
        num_features=10
    )
    
    # Show explanation
    exp.show_in_notebook()
    
    # As matplotlib
    exp.as_pyplot_figure()
    plt.tight_layout()
    plt.show()
    ```
    
    **5. Coefficient-Based (Linear Models):**
    
    ```python
    from sklearn.linear_model import LogisticRegression
    
    # Train logistic regression
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train_scaled, y_train)
    
    # Get coefficients
    coef_df = pd.DataFrame({
        'feature': X_train.columns,
        'coefficient': lr.coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    colors = ['red' if c < 0 else 'green' for c in coef_df['coefficient'][:10]]
    plt.barh(coef_df['feature'][:10], coef_df['coefficient'][:10], color=colors)
    plt.xlabel('Coefficient')
    plt.title('Feature Coefficients (Logistic Regression)')
    plt.gca().invert_yaxis()
    plt.show()
    ```
    
    **6. Partial Dependence Plots:**
    
    ```python
    from sklearn.inspection import partial_dependence, PartialDependenceDisplay
    
    # Show how predictions change with feature value
    features = ['age', 'income', ('age', 'income')]
    
    display = PartialDependenceDisplay.from_estimator(
        rf, X_train, features,
        kind='both',  # Shows both average and individual
        grid_resolution=50
    )
    
    plt.tight_layout()
    plt.show()
    ```
    
    **Comparison:**
    
    | Method | Speed | Global/Local | Model-Agnostic | Interaction |
    |--------|-------|--------------|----------------|-------------|
    | Tree importance | Fast | Global | No | No |
    | Permutation | Medium | Global | Yes | No |
    | SHAP | Slow | Both | Yes (TreeSHAP fast) | Yes |
    | LIME | Medium | Local | Yes | Limited |
    | Coefficients | Fast | Global | No (linear only) | No |
    | PDP | Medium | Global | Yes | Yes (2D) |
    
    **Custom Implementation:**
    
    ```python
    def drop_column_importance(model, X, y, metric):
        """
        Measure importance by training without each feature
        """
        baseline = metric(y, model.predict(X))
        importances = {}
        
        for col in X.columns:
            # Drop column
            X_reduced = X.drop(columns=[col])
            
            # Retrain model
            model_reduced = clone(model)
            model_reduced.fit(X_reduced, y)
            
            # Evaluate
            score = metric(y, model_reduced.predict(X_reduced))
            
            # Importance = performance drop
            importances[col] = baseline - score
        
        return pd.Series(importances).sort_values(ascending=False)
    
    # Usage
    from sklearn.metrics import accuracy_score
    from sklearn.base import clone
    
    importances = drop_column_importance(rf, X_test, y_test, accuracy_score)
    print(importances)
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Model interpretation skills.
        
        **Strong answer signals:**
        
        - Multiple methods (tree, permutation, SHAP)
        - "Permutation more reliable than Gini"
        - SHAP for individual predictions
        - "Linear coefs only for linear models"
        - Mentions computational cost

---

### What is Online Learning? - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Online Learning`, `Streaming`, `Model Updates` | **Asked by:** Google, Meta, Amazon, Netflix

??? success "View Answer"

    **Online Learning:**
    
    Train models incrementally as new data arrives, without retraining from scratch.
    
    **Batch vs Online:**
    
    ```python
    # Batch Learning
    model.fit(X_all, y_all)  # Train on all data once
    
    # Online Learning
    for X_batch, y_batch in data_stream:
        model.partial_fit(X_batch, y_batch)  # Update incrementally
    ```
    
    **Scikit-Learn Online Learning:**
    
    ```python
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    # Models supporting partial_fit
    model = SGDClassifier(loss='log', random_state=42)
    scaler = StandardScaler()
    
    # Simulate data stream
    batch_size = 100
    n_batches = 50
    
    # Initialize on first batch
    X_batch, y_batch = generate_batch(batch_size)
    scaler.partial_fit(X_batch)
    X_scaled = scaler.transform(X_batch)
    model.partial_fit(X_scaled, y_batch, classes=np.unique(y_batch))
    
    # Online updates
    accuracies = []
    for i in range(1, n_batches):
        # Get new data
        X_batch, y_batch = generate_batch(batch_size)
        
        # Update scaler and transform
        scaler.partial_fit(X_batch)
        X_scaled = scaler.transform(X_batch)
        
        # Evaluate before update
        accuracy = model.score(X_scaled, y_batch)
        accuracies.append(accuracy)
        
        # Update model
        model.partial_fit(X_scaled, y_batch)
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(accuracies)
    plt.xlabel('Batch')
    plt.ylabel('Accuracy')
    plt.title('Online Learning Performance')
    plt.grid(True, alpha=0.3)
    plt.show()
    ```
    
    **Models Supporting Online Learning:**
    
    ```python
    from sklearn.linear_model import SGDClassifier, SGDRegressor, PassiveAggressiveClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.cluster import MiniBatchKMeans
    
    # Classification
    online_classifiers = {
        'SGD': SGDClassifier(),
        'Passive-Aggressive': PassiveAggressiveClassifier(),
        'Naive Bayes': MultinomialNB()
    }
    
    # Regression
    online_regressors = {
        'SGD': SGDRegressor()
    }
    
    # Clustering
    online_clustering = {
        'MiniBatch K-Means': MiniBatchKMeans(n_clusters=5)
    }
    ```
    
    **Custom Online Model:**
    
    ```python
    class OnlineLogisticRegression:
        """Simple online logistic regression"""
        
        def __init__(self, n_features, learning_rate=0.01):
            self.weights = np.zeros(n_features)
            self.bias = 0
            self.learning_rate = learning_rate
        
        def sigmoid(self, z):
            return 1 / (1 + np.exp(-z))
        
        def predict_proba(self, X):
            z = np.dot(X, self.weights) + self.bias
            return self.sigmoid(z)
        
        def predict(self, X):
            return (self.predict_proba(X) >= 0.5).astype(int)
        
        def partial_fit(self, X, y):
            """Update weights with one batch"""
            n_samples = len(X)
            
            # Forward pass
            y_pred = self.predict_proba(X)
            
            # Compute gradients
            error = y_pred - y
            grad_w = np.dot(X.T, error) / n_samples
            grad_b = np.mean(error)
            
            # Update weights (gradient descent)
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b
            
            return self
    
    # Usage
    model = OnlineLogisticRegression(n_features=10)
    
    for X_batch, y_batch in data_stream:
        model.partial_fit(X_batch, y_batch)
    ```
    
    **Concept Drift Handling:**
    
    ```python
    from river import drift
    import numpy as np
    
    # Detect when data distribution changes
    class DriftDetector:
        def __init__(self, window_size=100):
            self.window_size = window_size
            self.reference_window = []
            self.current_window = []
        
        def update(self, error):
            """Add new error observation"""
            self.current_window.append(error)
            
            if len(self.current_window) > self.window_size:
                self.current_window.pop(0)
            
            # Check for drift
            if len(self.reference_window) == self.window_size:
                drift_detected = self.detect_drift()
                
                if drift_detected:
                    # Reset reference
                    self.reference_window = self.current_window.copy()
                    return True
            else:
                self.reference_window.append(error)
            
            return False
        
        def detect_drift(self):
            """Compare distributions using KS test"""
            from scipy.stats import ks_2samp
            
            stat, p_value = ks_2samp(self.reference_window, 
                                      self.current_window)
            
            return p_value < 0.05  # Significant difference
    
    # Usage
    detector = DriftDetector(window_size=100)
    model = SGDClassifier()
    
    for X_batch, y_batch in data_stream:
        # Predict
        y_pred = model.predict(X_batch)
        
        # Check for errors
        errors = (y_pred != y_batch).astype(int)
        
        for error in errors:
            drift = detector.update(error)
            
            if drift:
                print("Concept drift detected! Retraining...")
                # Could reset model or adjust learning rate
                model = SGDClassifier()
        
        # Update model
        model.partial_fit(X_batch, y_batch)
    ```
    
    **Evaluation Strategies:**
    
    ```python
    # 1. Prequential Evaluation (Test-then-Train)
    def prequential_evaluation(model, data_stream):
        """Evaluate then update"""
        scores = []
        
        for X_batch, y_batch in data_stream:
            # Test on new data
            score = model.score(X_batch, y_batch)
            scores.append(score)
            
            # Then train
            model.partial_fit(X_batch, y_batch)
        
        return scores
    
    # 2. Sliding Window
    def sliding_window_eval(model, data_stream, window_size=1000):
        """Evaluate on recent window"""
        window_X = []
        window_y = []
        scores = []
        
        for X_batch, y_batch in data_stream:
            # Update model
            model.partial_fit(X_batch, y_batch)
            
            # Add to window
            window_X.append(X_batch)
            window_y.append(y_batch)
            
            # Maintain window size
            if len(window_X) * len(X_batch) > window_size:
                window_X.pop(0)
                window_y.pop(0)
            
            # Evaluate on window
            if len(window_X) > 0:
                X_window = np.vstack(window_X)
                y_window = np.concatenate(window_y)
                score = model.score(X_window, y_window)
                scores.append(score)
        
        return scores
    ```
    
    **Use Cases:**
    
    | Application | Reason |
    |-------------|--------|
    | Recommendation systems | User preferences change |
    | Fraud detection | New fraud patterns emerge |
    | Stock prediction | Market conditions evolve |
    | Ad click prediction | User behavior shifts |
    | IoT sensor data | Continuous streaming |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Real-time ML systems.
        
        **Strong answer signals:**
        
        - partial_fit() for incremental updates
        - "Don't retrain from scratch"
        - Concept drift handling
        - Prequential evaluation
        - SGD-based models work well
        - Trade-off: speed vs accuracy

---

### Explain Hyperparameter Tuning Methods - Most Tech Companies Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Hyperparameters`, `Optimization`, `Model Selection` | **Asked by:** Google, Amazon, Meta, Microsoft, Apple

??? success "View Answer"

    **Hyperparameter Tuning:**
    
    Finding optimal hyperparameter values to maximize model performance.
    
    **1. Grid Search:**
    
    ```python
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    
    # Grid search
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_:.4f}")
    
    # Use best model
    best_model = grid_search.best_estimator_
    ```
    
    **2. Random Search:**
    
    ```python
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint, uniform
    
    # Define distributions
    param_distributions = {
        'n_estimators': randint(50, 500),
        'max_depth': [None] + list(randint(5, 50).rvs(10)),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': uniform(0.1, 0.9),  # Fraction
        'bootstrap': [True, False]
    }
    
    # Random search (more efficient)
    random_search = RandomizedSearchCV(
        rf, param_distributions,
        n_iter=100,  # Number of random combinations
        cv=5,
        scoring='f1',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"Best parameters: {random_search.best_params_}")
    ```
    
    **3. Bayesian Optimization:**
    
    ```python
    from skopt import BayesSearchCV
    from skopt.space import Real, Categorical, Integer
    
    # Define search space
    search_spaces = {
        'n_estimators': Integer(50, 500),
        'max_depth': Integer(5, 50),
        'min_samples_split': Integer(2, 20),
        'min_samples_leaf': Integer(1, 10),
        'max_features': Real(0.1, 1.0),
        'bootstrap': Categorical([True, False])
    }
    
    # Bayesian optimization (most efficient)
    bayes_search = BayesSearchCV(
        rf, search_spaces,
        n_iter=50,  # Fewer iterations needed
        cv=5,
        scoring='f1',
        n_jobs=-1,
        random_state=42
    )
    
    bayes_search.fit(X_train, y_train)
    
    print(f"Best parameters: {bayes_search.best_params_}")
    ```
    
    **4. Optuna (Modern Bayesian):**
    
    ```python
    import optuna
    from sklearn.model_selection import cross_val_score
    
    def objective(trial):
        """Objective function for Optuna"""
        
        # Suggest hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_float('max_features', 0.1, 1.0),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
        }
        
        # Train and evaluate
        model = RandomForestClassifier(**params, random_state=42)
        score = cross_val_score(model, X_train, y_train, 
                                cv=5, scoring='f1', n_jobs=-1).mean()
        
        return score
    
    # Create study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    
    print(f"Best parameters: {study.best_params}")
    print(f"Best score: {study.best_value:.4f}")
    
    # Visualize optimization
    from optuna.visualization import plot_optimization_history, plot_param_importances
    
    plot_optimization_history(study)
    plot_param_importances(study)
    ```
    
    **5. Halving Search (Successive Halving):**
    
    ```python
    from sklearn.experimental import enable_halving_search_cv
    from sklearn.model_selection import HalvingRandomSearchCV
    
    # Efficiently discard bad candidates early
    halving_search = HalvingRandomSearchCV(
        rf, param_distributions,
        factor=3,  # Reduce candidates by 1/3 each iteration
        resource='n_samples',
        max_resources='auto',
        cv=5,
        scoring='f1',
        n_jobs=-1,
        random_state=42
    )
    
    halving_search.fit(X_train, y_train)
    ```
    
    **Comparison:**
    
    ```python
    import time
    import pandas as pd
    
    methods = {
        'Grid Search': grid_search,
        'Random Search': random_search,
        'Bayesian Optimization': bayes_search
    }
    
    results = []
    
    for name, search in methods.items():
        start_time = time.time()
        search.fit(X_train, y_train)
        elapsed = time.time() - start_time
        
        results.append({
            'Method': name,
            'Best Score': search.best_score_,
            'Time (s)': elapsed,
            'Iterations': len(search.cv_results_['params'])
        })
    
    results_df = pd.DataFrame(results)
    print(results_df)
    ```
    
    | Method | Pros | Cons | When to Use |
    |--------|------|------|-------------|
    | Grid Search | Exhaustive, reproducible | Exponentially slow | Small parameter space |
    | Random Search | Fast, good for high-dim | May miss optimal | Large parameter space |
    | Bayesian Optimization | Most efficient, learns | Complex setup | Production, limited budget |
    | Successive Halving | Very fast | May discard good late bloomers | Quick iteration |
    
    **Manual Tuning Tips:**
    
    ```python
    # Start with defaults, tune one at a time
    
    # 1. Learning rate (most important for GBMs)
    for lr in [0.001, 0.01, 0.1, 1.0]:
        model = GradientBoostingClassifier(learning_rate=lr)
        score = cross_val_score(model, X, y, cv=5).mean()
        print(f"LR={lr}: {score:.4f}")
    
    # 2. Model complexity (depth, num trees)
    # 3. Regularization (min_samples, alpha)
    # 4. Data sampling (max_features, subsample)
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Practical ML optimization.
        
        **Strong answer signals:**
        
        - "Grid search exhaustive but slow"
        - "Random search better for high-dim"
        - "Bayesian most efficient"
        - Mentions cross-validation
        - Knows which hyperparameters matter most
        - "Start simple, tune iteratively"

---

### What is Model Compression? - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Model Compression`, `Deployment`, `Optimization` | **Asked by:** Google, Meta, Apple, Amazon

??? success "View Answer"

    **Model Compression:**
    
    Reducing model size and computational cost while maintaining performance.
    
    **1. Quantization:**
    
    ```python
    import torch
    import torch.quantization
    
    # Original model (32-bit floats)
    model = MyModel()
    model.eval()
    
    # Post-Training Static Quantization (8-bit integers)
    model_quantized = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},  # Layers to quantize
        dtype=torch.qint8
    )
    
    # Compare sizes
    def get_model_size(model):
        torch.save(model.state_dict(), "temp.pth")
        size = os.path.getsize("temp.pth") / 1e6  # MB
        os.remove("temp.pth")
        return size
    
    original_size = get_model_size(model)
    quantized_size = get_model_size(model_quantized)
    
    print(f"Original: {original_size:.2f} MB")
    print(f"Quantized: {quantized_size:.2f} MB")
    print(f"Compression: {original_size/quantized_size:.2f}x")
    
    # Quantization-Aware Training (better accuracy)
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    model_prepared = torch.quantization.prepare_qat(model)
    
    # Train with quantization simulation
    for epoch in range(num_epochs):
        train_one_epoch(model_prepared, train_loader)
    
    # Convert to quantized model
    model_quantized = torch.quantization.convert(model_prepared)
    ```
    
    **2. Pruning:**
    
    ```python
    import torch.nn.utils.prune as prune
    
    # Unstructured pruning (remove individual weights)
    model = MyModel()
    
    # Prune 30% of weights in linear layer
    prune.l1_unstructured(
        module=model.fc1,
        name='weight',
        amount=0.3
    )
    
    # Prune multiple layers
    parameters_to_prune = (
        (model.fc1, 'weight'),
        (model.fc2, 'weight'),
        (model.fc3, 'weight')
    )
    
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.3  # 30% of all weights
    )
    
    # Make pruning permanent
    for module, name in parameters_to_prune:
        prune.remove(module, name)
    
    # Structured pruning (remove entire filters)
    prune.ln_structured(
        module=model.conv1,
        name='weight',
        amount=0.3,
        n=2,  # L2 norm
        dim=0  # Prune output channels
    )
    ```
    
    **3. Knowledge Distillation:**
    
    ```python
    # Already covered in detail in previous question
    # Large teacher -> Small student
    
    class DistillationLoss(nn.Module):
        def __init__(self, temperature=3.0, alpha=0.5):
            super().__init__()
            self.temperature = temperature
            self.alpha = alpha
            self.kl_div = nn.KLDivLoss(reduction='batchmean')
            self.ce_loss = nn.CrossEntropyLoss()
        
        def forward(self, student_logits, teacher_logits, labels):
            # Soft targets from teacher
            soft_loss = self.kl_div(
                F.log_softmax(student_logits / self.temperature, dim=1),
                F.softmax(teacher_logits / self.temperature, dim=1)
            ) * (self.temperature ** 2)
            
            # Hard targets
            hard_loss = self.ce_loss(student_logits, labels)
            
            return self.alpha * soft_loss + (1 - self.alpha) * hard_loss
    ```
    
    **4. Low-Rank Factorization:**
    
    ```python
    import torch
    import torch.nn as nn
    
    class LowRankLinear(nn.Module):
        """Decompose weight matrix W = U @ V"""
        
        def __init__(self, in_features, out_features, rank):
            super().__init__()
            
            # W (out x in) ≈ U (out x rank) @ V (rank x in)
            self.U = nn.Parameter(torch.randn(out_features, rank))
            self.V = nn.Parameter(torch.randn(rank, in_features))
            self.bias = nn.Parameter(torch.zeros(out_features))
        
        def forward(self, x):
            # x @ V^T @ U^T + b
            return F.linear(F.linear(x, self.V), self.U, self.bias)
    
    # Replace linear layer
    def replace_with_low_rank(module, rank):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                in_features = child.in_features
                out_features = child.out_features
                
                # Replace
                setattr(module, name, 
                       LowRankLinear(in_features, out_features, rank))
            else:
                replace_with_low_rank(child, rank)
    
    model = MyModel()
    replace_with_low_rank(model, rank=50)
    
    # Compression ratio
    original_params = in_features * out_features
    compressed_params = rank * (in_features + out_features)
    print(f"Compression: {original_params / compressed_params:.2f}x")
    ```
    
    **5. Neural Architecture Search (Compact Models):**
    
    ```python
    # Find efficient architectures (e.g., MobileNet, EfficientNet)
    
    # MobileNet: Depthwise Separable Convolutions
    class DepthwiseSeparableConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            
            # Depthwise: one filter per input channel
            self.depthwise = nn.Conv2d(
                in_channels, in_channels,
                kernel_size=3, padding=1,
                groups=in_channels  # Key: groups = in_channels
            )
            
            # Pointwise: 1x1 conv to combine
            self.pointwise = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=1
            )
        
        def forward(self, x):
            x = self.depthwise(x)
            x = self.pointwise(x)
            return x
    
    # Parameters comparison
    # Standard conv: k*k*in*out
    # Depthwise separable: k*k*in + in*out
    # Compression: ~8-9x for 3x3 kernels
    ```
    
    **Comprehensive Compression Pipeline:**
    
    ```python
    def compress_model(model, X_train, y_train, X_val, y_val):
        """Apply multiple compression techniques"""
        
        # 1. Pruning
        print("Step 1: Pruning...")
        parameters_to_prune = [(module, 'weight') 
                               for module in model.modules() 
                               if isinstance(module, nn.Linear)]
        
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=0.3
        )
        
        # Fine-tune after pruning
        train(model, X_train, y_train, epochs=5)
        
        # 2. Quantization
        print("Step 2: Quantization...")
        model.eval()
        model_quantized = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        
        # 3. Evaluate
        original_acc = evaluate(model, X_val, y_val)
        compressed_acc = evaluate(model_quantized, X_val, y_val)
        
        original_size = get_model_size(model)
        compressed_size = get_model_size(model_quantized)
        
        print(f"\nResults:")
        print(f"Accuracy: {original_acc:.4f} -> {compressed_acc:.4f}")
        print(f"Size: {original_size:.2f} MB -> {compressed_size:.2f} MB")
        print(f"Compression: {original_size/compressed_size:.2f}x")
        
        return model_quantized
    ```
    
    **Comparison:**
    
    | Technique | Compression | Accuracy Loss | Speed Up |
    |-----------|-------------|---------------|----------|
    | Quantization (INT8) | 4x | <1% | 2-4x |
    | Pruning (50%) | 2x | <2% | 1.5-2x |
    | Knowledge Distillation | 10-100x | 2-5% | 10-100x |
    | Low-Rank | 2-5x | 1-3% | 1.5-3x |
    | Combined | 10-50x | 3-8% | 5-20x |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Deployment optimization.
        
        **Strong answer signals:**
        
        - Multiple techniques (quantization, pruning, distillation)
        - "INT8 quantization: 4x smaller"
        - "Pruning: remove redundant weights"
        - "Distillation: transfer knowledge to small model"
        - Mentions accuracy-size trade-off
        - "Combine techniques for best results"

---

### Explain Metrics for Imbalanced Classification - Most Tech Companies Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Metrics`, `Imbalanced Data`, `Evaluation` | **Asked by:** Google, Amazon, Meta, Microsoft, Netflix

??? success "View Answer"

    **Metrics for Imbalanced Data:**
    
    Standard accuracy misleads when classes are imbalanced (e.g., 99% negative, 1% positive).
    
    **Problem with Accuracy:**
    
    ```python
    # Dataset: 990 negative, 10 positive samples
    y_true = [0]*990 + [1]*10
    
    # Dummy classifier: always predict negative
    y_pred = [0]*1000
    
    accuracy = (y_true == y_pred).sum() / len(y_true)
    print(f"Accuracy: {accuracy:.1%}")  # 99%!
    
    # But it catches 0% of positive class!
    ```
    
    **Better Metrics:**
    
    **1. Confusion Matrix Metrics:**
    
    ```python
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Visualize
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Extract metrics
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    ```
    
    **2. F1, F-beta Scores:**
    
    ```python
    from sklearn.metrics import f1_score, fbeta_score, precision_recall_fscore_support
    
    # F1: Harmonic mean of precision and recall
    f1 = f1_score(y_true, y_pred)
    
    # F-beta: Weight recall more (beta>1) or precision more (beta<1)
    f_half = fbeta_score(y_true, y_pred, beta=0.5)  # Emphasize precision
    f2 = fbeta_score(y_true, y_pred, beta=2.0)      # Emphasize recall
    
    print(f"F1: {f1:.4f}")
    print(f"F0.5 (precision-focused): {f_half:.4f}")
    print(f"F2 (recall-focused): {f2:.4f}")
    
    # When to use:
    # - F1: Balanced importance
    # - F0.5: False positives costly (spam detection)
    # - F2: False negatives costly (disease detection)
    ```
    
    **3. ROC-AUC:**
    
    ```python
    from sklearn.metrics import roc_curve, roc_auc_score, auc
    
    # Need probability scores
    y_scores = model.predict_proba(X_test)[:, 1]
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Interpretation:
    # AUC = 1.0: Perfect
    # AUC = 0.5: Random
    # AUC < 0.5: Worse than random (flip predictions!)
    ```
    
    **4. Precision-Recall AUC (Better for Imbalanced):**
    
    ```python
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    # PR curve more informative than ROC for imbalanced data
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR (AUC = {pr_auc:.3f})')
    
    # Baseline: proportion of positive class
    baseline = y_true.sum() / len(y_true)
    plt.axhline(baseline, color='k', linestyle='--', 
                label=f'Baseline ({baseline:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    ```
    
    **5. Matthews Correlation Coefficient (MCC):**
    
    ```python
    from sklearn.metrics import matthews_corrcoef
    
    # Single metric for imbalanced data
    # Range: -1 (worst) to +1 (best), 0 = random
    mcc = matthews_corrcoef(y_true, y_pred)
    
    print(f"MCC: {mcc:.4f}")
    
    # Formula:
    # MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    ```
    
    **6. Cohen's Kappa:**
    
    ```python
    from sklearn.metrics import cohen_kappa_score
    
    # Agreement beyond chance
    kappa = cohen_kappa_score(y_true, y_pred)
    
    print(f"Kappa: {kappa:.4f}")
    
    # Interpretation:
    # < 0: Worse than random
    # 0-0.2: Slight agreement
    # 0.2-0.4: Fair
    # 0.4-0.6: Moderate
    # 0.6-0.8: Substantial
    # 0.8-1.0: Almost perfect
    ```
    
    **7. Balanced Accuracy:**
    
    ```python
    from sklearn.metrics import balanced_accuracy_score
    
    # Average of recall for each class
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    
    # Avoids being misled by imbalance
    ```
    
    **Comprehensive Evaluation:**
    
    ```python
    def evaluate_imbalanced(y_true, y_pred, y_scores=None):
        """Complete evaluation for imbalanced classification"""
        
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score, average_precision_score,
            matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score
        )
        
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'F1': f1_score(y_true, y_pred),
            'MCC': matthews_corrcoef(y_true, y_pred),
            'Kappa': cohen_kappa_score(y_true, y_pred)
        }
        
        if y_scores is not None:
            metrics['ROC-AUC'] = roc_auc_score(y_true, y_scores)
            metrics['PR-AUC'] = average_precision_score(y_true, y_scores)
        
        # Print table
        print("Evaluation Metrics:")
        print("-" * 40)
        for metric, value in metrics.items():
            print(f"{metric:20s}: {value:.4f}")
        
        return metrics
    
    # Usage
    metrics = evaluate_imbalanced(y_true, y_pred, y_scores)
    ```
    
    **Metric Selection Guide:**
    
    | Use Case | Recommended Metrics | Reason |
    |----------|---------------------|--------|
    | Medical diagnosis | Recall, F2, PR-AUC | Minimize false negatives |
    | Spam detection | Precision, F0.5 | Minimize false positives |
    | Fraud detection | PR-AUC, MCC | Extremely imbalanced |
    | General imbalanced | F1, Balanced Accuracy, MCC | Balanced view |
    | Ranking | ROC-AUC, PR-AUC | Threshold-independent |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of evaluation metrics.
        
        **Strong answer signals:**
        
        - "Accuracy misleading for imbalanced data"
        - Precision vs Recall trade-off
        - "PR-AUC better than ROC-AUC for imbalanced"
        - F-beta for different priorities
        - MCC: single balanced metric
        - "Choose metric based on business cost"

---

### What is Multi-Task Learning? - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Multi-Task Learning`, `Transfer Learning`, `Neural Networks` | **Asked by:** Google, Meta, Amazon, Microsoft

??? success "View Answer"

    **Multi-Task Learning (MTL):**
    
    Train a single model on multiple related tasks simultaneously to improve generalization.
    
    **Key Idea:**
    
    - Shared representations help all tasks
    - "What is learned for one task can help other tasks"
    
    **Architecture:**
    
    ```python
    import torch
    import torch.nn as nn
    
    class MultiTaskModel(nn.Module):
        """Multi-task learning with shared encoder"""
        
        def __init__(self, input_dim, num_tasks):
            super().__init__()
            
            # Shared encoder (learns general representations)
            self.shared_encoder = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            
            # Task-specific heads
            self.task_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)  # Binary classification
                )
                for _ in range(num_tasks)
            ])
        
        def forward(self, x):
            # Shared encoding
            shared_features = self.shared_encoder(x)
            
            # Task-specific predictions
            outputs = [head(shared_features) for head in self.task_heads]
            
            return outputs
    
    # Training
    model = MultiTaskModel(input_dim=100, num_tasks=3)
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(num_epochs):
        for X_batch, y_tasks in train_loader:
            # y_tasks: list of targets for each task
            
            # Forward
            outputs = model(X_batch)
            
            # Compute loss for each task
            losses = []
            for task_idx in range(len(outputs)):
                loss = F.binary_cross_entropy_with_logits(
                    outputs[task_idx].squeeze(),
                    y_tasks[task_idx].float()
                )
                losses.append(loss)
            
            # Combined loss (simple average)
            total_loss = sum(losses) / len(losses)
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    ```
    
    **Hard Parameter Sharing:**
    
    ```python
    class HardSharing(nn.Module):
        """Most common MTL architecture"""
        
        def __init__(self, input_dim):
            super().__init__()
            
            # Fully shared layers
            self.shared = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            
            # Task 1: Classification
            self.classifier = nn.Linear(128, 10)
            
            # Task 2: Regression
            self.regressor = nn.Linear(128, 1)
            
            # Task 3: Another classification
            self.aux_classifier = nn.Linear(128, 5)
        
        def forward(self, x):
            shared_features = self.shared(x)
            
            return {
                'classification': self.classifier(shared_features),
                'regression': self.regressor(shared_features),
                'auxiliary': self.aux_classifier(shared_features)
            }
    ```
    
    **Soft Parameter Sharing:**
    
    ```python
    class SoftSharing(nn.Module):
        """Each task has own parameters, but regularized to be similar"""
        
        def __init__(self, input_dim, num_tasks):
            super().__init__()
            
            # Separate encoders for each task
            self.task_encoders = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU()
                )
                for _ in range(num_tasks)
            ])
            
            # Task-specific heads
            self.task_heads = nn.ModuleList([
                nn.Linear(128, 1) for _ in range(num_tasks)
            ])
        
        def forward(self, x):
            outputs = []
            for encoder, head in zip(self.task_encoders, self.task_heads):
                features = encoder(x)
                output = head(features)
                outputs.append(output)
            return outputs
        
        def l2_regularization(self):
            """Encourage parameters to be similar across tasks"""
            reg_loss = 0
            num_tasks = len(self.task_encoders)
            
            for i in range(num_tasks):
                for j in range(i+1, num_tasks):
                    for p1, p2 in zip(self.task_encoders[i].parameters(),
                                     self.task_encoders[j].parameters()):
                        reg_loss += ((p1 - p2) ** 2).sum()
            
            return reg_loss
    
    # Training with regularization
    for X_batch, y_tasks in train_loader:
        outputs = model(X_batch)
        
        # Task losses
        task_losses = [criterion(out, target) 
                       for out, target in zip(outputs, y_tasks)]
        total_loss = sum(task_losses)
        
        # Add regularization
        reg_loss = model.l2_regularization()
        total_loss += 0.01 * reg_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    ```
    
    **Task Weighting Strategies:**
    
    ```python
    # 1. Uniform weighting
    total_loss = sum(losses) / len(losses)
    
    # 2. Manual weights
    task_weights = [1.0, 0.5, 2.0]  # Prioritize task 3
    total_loss = sum(w * l for w, l in zip(task_weights, losses))
    
    # 3. Uncertainty weighting (learned)
    class UncertaintyWeighting(nn.Module):
        """Learn task weights based on homoscedastic uncertainty"""
        
        def __init__(self, num_tasks):
            super().__init__()
            # Log variance for each task
            self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        
        def forward(self, losses):
            """
            Loss_weighted = Loss / (2*sigma^2) + log(sigma)
            sigma = exp(log_var / 2)
            """
            weighted_losses = []
            for i, loss in enumerate(losses):
                precision = torch.exp(-self.log_vars[i])
                weighted_loss = precision * loss + self.log_vars[i]
                weighted_losses.append(weighted_loss)
            
            return sum(weighted_losses)
    
    # Usage
    uncertainty_module = UncertaintyWeighting(num_tasks=3)
    total_loss = uncertainty_module(losses)
    
    # 4. Gradient normalization
    def grad_norm_loss(losses, shared_params):
        """Balance gradients from different tasks"""
        
        # Compute gradient norms for each task
        grad_norms = []
        for loss in losses:
            grads = torch.autograd.grad(loss, shared_params, 
                                        retain_graph=True, 
                                        create_graph=True)
            grad_norm = sum((g ** 2).sum() for g in grads).sqrt()
            grad_norms.append(grad_norm)
        
        # Normalize
        mean_norm = sum(grad_norms) / len(grad_norms)
        weights = [mean_norm / (gn + 1e-8) for gn in grad_norms]
        
        return sum(w * l for w, l in zip(weights, losses))
    ```
    
    **Real Example: NLP Multi-Task:**
    
    ```python
    class NLPMultiTask(nn.Module):
        """BERT-style multi-task model"""
        
        def __init__(self, vocab_size, embedding_dim=768):
            super().__init__()
            
            # Shared BERT-like encoder
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8),
                num_layers=6
            )
            
            # Task 1: Named Entity Recognition (token-level)
            self.ner_head = nn.Linear(embedding_dim, 10)  # 10 NER tags
            
            # Task 2: Sentiment Analysis (sequence-level)
            self.sentiment_head = nn.Sequential(
                nn.Linear(embedding_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 3)  # Positive/Negative/Neutral
            )
            
            # Task 3: Next Sentence Prediction
            self.nsp_head = nn.Linear(embedding_dim, 2)
        
        def forward(self, input_ids):
            # Shared encoding
            embeddings = self.embedding(input_ids)
            encoded = self.transformer(embeddings)
            
            # Task-specific outputs
            ner_logits = self.ner_head(encoded)  # All tokens
            
            # Use [CLS] token for sequence tasks
            cls_representation = encoded[:, 0, :]
            sentiment_logits = self.sentiment_head(cls_representation)
            nsp_logits = self.nsp_head(cls_representation)
            
            return {
                'ner': ner_logits,
                'sentiment': sentiment_logits,
                'nsp': nsp_logits
            }
    ```
    
    **Benefits:**
    
    | Benefit | Explanation |
    |---------|-------------|
    | Better generalization | Shared representations regularize |
    | Faster learning | Transfer knowledge between tasks |
    | Data efficiency | Leverage data from all tasks |
    | Implicit regularization | Prevents overfitting to single task |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Advanced ML architecture knowledge.
        
        **Strong answer signals:**
        
        - Hard vs soft parameter sharing
        - "Shared encoder + task-specific heads"
        - Task weighting strategies
        - "Helps when tasks related"
        - Examples: BERT (NER, sentiment, NLI)
        - "Can hurt if tasks very different"

---

| 13 | k-Nearest Neighbors (k-NN) | [Towards Data Science](https://towardsdatascience.com/k-nearest-neighbors-knn-algorithm-for-machine-learning-e883219c8f26) | Google, Amazon, Facebook | Easy | Instance-based Learning |
| 14 | Dimensionality Reduction: PCA | [Towards Data Science](https://towardsdatascience.com/a-one-stop-shop-for-principal-component-analysis-5582fb7e0a9c) | Google, Amazon, Microsoft | Medium | Dimensionality Reduction |
| 15 | Handling Missing Data | [Machine Learning Mastery](https://machinelearningmastery.com/handle-missing-data-python/) | Google, Amazon, Facebook | Easy | Data Preprocessing |
| 16 | Parametric vs Non-Parametric Models | [Towards Data Science](https://towardsdatascience.com/parametric-vs-non-parametric-models-825d1a0f5c2c) | Google, Amazon | Medium | Model Types |
| 17 | Neural Networks: Basics | [Towards Data Science](https://towardsdatascience.com/a-beginners-guide-to-neural-networks-2cf4c3f9c9d0) | Google, Facebook, Amazon | Medium | Deep Learning |
| 18 | Convolutional Neural Networks (CNNs) | [Towards Data Science](https://towardsdatascience.com/a-guide-to-convolutional-neural-networks-for-computer-vision-2bda48ea1e50) | Google, Facebook, Amazon | Hard | Deep Learning, Computer Vision |
| 19 | Recurrent Neural Networks (RNNs) and LSTMs | [Towards Data Science](https://towardsdatascience.com/recurrent-neural-networks-for-language-modeling-396f1d1659f2) | Google, Amazon, Facebook | Hard | Deep Learning, Sequence Models |
| 20 | Reinforcement Learning Basics | [Towards Data Science](https://towardsdatascience.com/introduction-to-reinforcement-learning-6346f7f8c1ef) | Google, Amazon, Facebook | Hard | Reinforcement Learning |
| 21 | Hyperparameter Tuning | [Machine Learning Mastery](https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/) | Google, Amazon, Microsoft | Medium | Model Optimization |
| 22 | Feature Engineering | [Towards Data Science](https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114) | Google, Amazon, Facebook | Medium | Data Preprocessing |
| 23 | ROC Curve and AUC | [Towards Data Science](https://towardsdatascience.com/roc-curve-and-auc-using-python-and-scikit-learn-42da0fa0d0d) | Google, Amazon, Microsoft | Medium | Model Evaluation |
| 24 | Regression Evaluation Metrics | [Scikit-Learn](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics) | Google, Amazon, Facebook | Medium | Model Evaluation, Regression |
| 25 | Curse of Dimensionality | [Machine Learning Mastery](https://machinelearningmastery.com/curse-of-dimensionality/) | Google, Amazon, Facebook | Hard | Data Preprocessing |
| 26 | Logistic Regression | [Towards Data Science](https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc) | Google, Amazon, Facebook | Easy | Classification, Regression |
| 27 | Linear Regression | [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2016/02/complete-tutorial-learn-data-science-scratch/) | Google, Amazon, Facebook | Easy | Regression |
| 28 | Loss Functions in ML | [Towards Data Science](https://towardsdatascience.com/common-loss-functions-in-machine-learning-3b7af9f8bf2b) | Google, Amazon, Microsoft | Medium | Optimization, Model Evaluation |
| 29 | Gradient Descent Variants | [Machine Learning Mastery](https://machinelearningmastery.com/difference-between-batch-and-stochastic-gradient-descent/) | Google, Amazon, Facebook | Medium | Optimization |
| 30 | Data Normalization and Standardization | [Machine Learning Mastery](https://machinelearningmastery.com/normalize-standardize-machine-learning-data/) | Google, Amazon, Facebook | Easy | Data Preprocessing |
| 31 | k-Means Clustering | [Towards Data Science](https://towardsdatascience.com/introduction-to-k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a) | Google, Amazon, Facebook | Medium | Clustering |
| 32 | Other Clustering Techniques | [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/) | Google, Amazon, Facebook | Medium | Clustering |
| 33 | Anomaly Detection | [Towards Data Science](https://towardsdatascience.com/anomaly-detection-techniques-in-python-50f650c75aaf) | Google, Amazon, Facebook | Hard | Outlier Detection |
| 34 | Learning Rate in Optimization | [Machine Learning Mastery](https://machinelearningmastery.com/learning-rate-for-deep-learning-neural-networks/) | Google, Amazon, Microsoft | Medium | Optimization |
| 35 | Deep Learning vs. Traditional ML | [IBM Cloud Learn](https://www.ibm.com/cloud/learn/deep-learning) | Google, Amazon, Facebook | Medium | Deep Learning, ML Basics |
| 36 | Dropout in Neural Networks | [Towards Data Science](https://towardsdatascience.com/understanding-dropout-in-neural-networks-3c5da7a57f86) | Google, Amazon, Facebook | Medium | Deep Learning, Regularization |
| 37 | Backpropagation | [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2017/05/implementation-neural-network-scratch-python/) | Google, Amazon, Facebook | Hard | Deep Learning, Neural Networks |
| 38 | Role of Activation Functions | [Machine Learning Mastery](https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/) | Google, Amazon, Facebook | Medium | Neural Networks |
| 39 | Word Embeddings and Their Use | [Towards Data Science](https://towardsdatascience.com/word-embeddings-6cb7d87c0f64) | Google, Amazon, Facebook | Medium | NLP, Deep Learning |
| 40 | Transfer Learning | [Machine Learning Mastery](https://machinelearningmastery.com/transfer-learning-for-deep-learning/) | Google, Amazon, Facebook | Medium | Deep Learning, Model Reuse |
| 41 | Bayesian Optimization for Hyperparameters | [Towards Data Science](https://towardsdatascience.com/bayesian-optimization-explained-4f6c2e60731d) | Google, Amazon, Microsoft | Hard | Hyperparameter Tuning, Optimization |
| 42 | Model Interpretability: SHAP and LIME | [Towards Data Science](https://towardsdatascience.com/interpreting-machine-learning-models-using-shap-values-df04dc62fbd4) | Google, Amazon, Facebook | Hard | Model Interpretability, Explainability |
| 43 | Ensemble Methods: Stacking and Blending | [Machine Learning Mastery](https://machinelearningmastery.com/ensemble-learning-stacking/) | Google, Amazon, Microsoft | Hard | Ensemble Methods |
| 44 | Gradient Boosting Machines (GBM) Basics | [Towards Data Science](https://towardsdatascience.com/understanding-gradient-boosting-machines-9be756fe76ab) | Google, Amazon, Facebook | Medium | Ensemble, Boosting |
| 45 | Extreme Gradient Boosting (XGBoost) Overview | [Towards Data Science](https://towardsdatascience.com/xgboost-optimized-gradient-boosting-e3d7b32d27b1) | Google, Amazon, Facebook | Medium | Ensemble, Boosting |
| 46 | LightGBM vs XGBoost Comparison | [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2019/06/lightgbm-vs-xgboost/) | Google, Amazon | Medium | Ensemble, Boosting |
| 47 | CatBoost: Handling Categorical Features | [Towards Data Science](https://towardsdatascience.com/catboost-for-beginners-d68638b78982) | Google, Amazon, Facebook | Medium | Ensemble, Categorical Data |
| 48 | Time Series Forecasting with ARIMA | [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2018/02/time-series-forecasting-methods/) | Google, Amazon, Facebook | Hard | Time Series, Forecasting |
| 49 | Time Series Forecasting with LSTM | [Towards Data Science](https://towardsdatascience.com/time-series-forecasting-using-lstm-3c6a39bfae39) | Google, Amazon, Facebook | Hard | Time Series, Deep Learning |
| 50 | Robust Scaling Techniques | [Towards Data Science](https://towardsdatascience.com/robust-scaling-why-when-and-how-3f2a67f1b0a3) | Google, Amazon, Facebook | Medium | Data Preprocessing |
| 51 | Data Imputation Techniques in ML | [Machine Learning Mastery](https://machinelearningmastery.com/handle-missing-data-python/) | Google, Amazon, Facebook | Medium | Data Preprocessing |
| 52 | Handling Imbalanced Datasets: SMOTE and Others | [Towards Data Science](https://towardsdatascience.com/smote-oversampling-for-imbalanced-classification-6c2046f13447) | Google, Amazon, Facebook | Hard | Data Preprocessing, Classification |
| 53 | Bias in Machine Learning: Fairness and Ethics | [Towards Data Science](https://towardsdatascience.com/fairness-in-machine-learning-6e21a5c4d5db) | Google, Amazon, Facebook | Hard | Ethics, Fairness |
| 54 | Model Deployment: From Prototype to Production | [Towards Data Science](https://towardsdatascience.com/deploying-machine-learning-models-3f6e41013240) | Google, Amazon, Facebook | Medium | Deployment |
| 55 | Online Learning Algorithms | [Towards Data Science](https://towardsdatascience.com/online-learning-algorithms-40bf6c6d19de) | Google, Amazon, Microsoft | Hard | Online Learning |
| 56 | Concept Drift in Machine Learning | [Towards Data Science](https://towardsdatascience.com/concept-drift-in-machine-learning-6b97e0f3f42d) | Google, Amazon, Facebook | Hard | Model Maintenance |
| 57 | Transfer Learning in NLP: BERT, GPT | [Towards Data Science](https://towardsdatascience.com/transfer-learning-in-nlp-9f26f10b2b96) | Google, Amazon, Facebook | Hard | NLP, Deep Learning |
| 58 | Natural Language Processing: Text Preprocessing | [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2020/07/text-preprocessing-techniques-in-python/) | Google, Amazon, Facebook | Easy | NLP, Data Preprocessing |
| 59 | Text Vectorization: TF-IDF vs Word2Vec | [Towards Data Science](https://towardsdatascience.com/text-vectorization-methods-6fd1d1a74a66) | Google, Amazon, Facebook | Medium | NLP, Feature Extraction |
| 60 | Transformer Architecture and Self-Attention | [Towards Data Science](https://towardsdatascience.com/transformers-141e32e69591) | Google, Amazon, Facebook | Hard | NLP, Deep Learning |
| 61 | Understanding BERT for NLP Tasks | [Towards Data Science](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270) | Google, Amazon, Facebook | Hard | NLP, Deep Learning |
| 62 | Understanding GPT Models | [Towards Data Science](https://towardsdatascience.com/what-is-gpt-3-and-why-is-it-so-important-95b9acb9d0a3) | Google, Amazon, Facebook | Hard | NLP, Deep Learning |
| 63 | Data Augmentation Techniques in ML | [Towards Data Science](https://towardsdatascience.com/data-augmentation-for-deep-learning-8e2f37e59a1b) | Google, Amazon, Facebook | Medium | Data Preprocessing |
| 64 | Adversarial Machine Learning: Attack and Defense | [Towards Data Science](https://towardsdatascience.com/adversarial-attacks-on-machine-learning-models-8a91b4a6a9a3) | Google, Amazon, Facebook | Hard | Security, ML |
| 65 | Explainable AI (XAI) in Practice | [Towards Data Science](https://towardsdatascience.com/explainable-ai-a-survey-of-methods-4d9e35597b0c) | Google, Amazon, Facebook | Hard | Model Interpretability |
| 66 | Federated Learning: Concepts and Challenges | [Towards Data Science](https://towardsdatascience.com/federated-learning-explained-d9e99d16ef57) | Google, Amazon, Facebook | Hard | Distributed Learning |
| 67 | Multi-Task Learning in Neural Networks | [Towards Data Science](https://towardsdatascience.com/multi-task-learning-for-neural-networks-6e4e2fcb5d3a) | Google, Amazon, Facebook | Hard | Deep Learning, Multi-Task |
| 68 | Metric Learning and Siamese Networks | [Towards Data Science](https://towardsdatascience.com/siamese-networks-for-one-shot-learning-60b2c8c9b71) | Google, Amazon, Facebook | Hard | Deep Learning, Metric Learning |
| 69 | Deep Reinforcement Learning: DQN Overview | [Towards Data Science](https://towardsdatascience.com/deep-q-learning-dqn-1b5f8bb83d11) | Google, Amazon, Facebook | Hard | Reinforcement Learning, Deep Learning |
| 70 | Policy Gradient Methods in Reinforcement Learning | [Towards Data Science](https://towardsdatascience.com/policy-gradient-methods-in-reinforcement-learning-713f77dceb79) | Google, Amazon, Facebook | Hard | Reinforcement Learning |
| 71 | Actor-Critic Methods in RL | [Towards Data Science](https://towardsdatascience.com/actor-critic-methods-in-reinforcement-learning-49cfa6403a5e) | Google, Amazon, Facebook | Hard | Reinforcement Learning |
| 72 | Monte Carlo Methods in Machine Learning | [Towards Data Science](https://towardsdatascience.com/monte-carlo-methods-in-machine-learning-8f7f0e9ad0e9) | Google, Amazon, Facebook | Medium | Optimization, Probabilistic Methods |
| 73 | Expectation-Maximization Algorithm | [Towards Data Science](https://towardsdatascience.com/expectation-maximization-algorithm-for-gaussian-mixture-models-ef96d0e98729) | Google, Amazon, Facebook | Hard | Clustering, Probabilistic Models |
| 74 | Gaussian Mixture Models (GMM) | [Towards Data Science](https://towardsdatascience.com/gaussian-mixture-models-in-python-6b85679b5a4) | Google, Amazon, Facebook | Medium | Clustering, Probabilistic Models |
| 75 | Bayesian Inference in ML | [Towards Data Science](https://towardsdatascience.com/introduction-to-bayesian-inference-7f72a56c97c) | Google, Amazon, Facebook | Hard | Bayesian Methods |
| 76 | Markov Chain Monte Carlo (MCMC) Methods | [Towards Data Science](https://towardsdatascience.com/markov-chain-monte-carlo-methods-a-tutorial-d3e4a14c6a1f) | Google, Amazon, Facebook | Hard | Bayesian Methods, Probabilistic Models |
| 77 | Variational Autoencoders (VAEs) | [Towards Data Science](https://towardsdatascience.com/variational-autoencoders-explained-8f7f0e9ad0e9) | Google, Amazon, Facebook | Hard | Deep Learning, Generative Models |
| 78 | Generative Adversarial Networks (GANs) | [Towards Data Science](https://towardsdatascience.com/generative-adversarial-networks-explained-34472718707a) | Google, Amazon, Facebook | Hard | Deep Learning, Generative Models |
| 79 | Conditional GANs for Data Generation | [Towards Data Science](https://towardsdatascience.com/conditional-gans-explained-9f2b30d3e5e3) | Google, Amazon, Facebook | Hard | Deep Learning, Generative Models |
| 80 | Sequence-to-Sequence Models in NLP | [Towards Data Science](https://towardsdatascience.com/sequence-to-sequence-models-for-machine-translation-873b51b65f0f) | Google, Amazon, Facebook | Hard | NLP, Deep Learning |
| 81 | Attention Mechanisms in Seq2Seq Models | [Towards Data Science](https://towardsdatascience.com/attention-mechanisms-in-deep-learning-a-tutorial-3d9b62f341d) | Google, Amazon, Facebook | Hard | NLP, Deep Learning |
| 82 | Capsule Networks: An Introduction | [Towards Data Science](https://towardsdatascience.com/capsule-networks-an-introduction-4d2b2a7dbd5) | Google, Amazon, Facebook | Hard | Deep Learning, Neural Networks |
| 83 | Self-Supervised Learning in Deep Learning | [Towards Data Science](https://towardsdatascience.com/self-supervised-learning-explained-7e0e4a2f8b8) | Google, Amazon, Facebook | Hard | Deep Learning, Unsupervised Learning |
| 84 | Zero-Shot and Few-Shot Learning | [Towards Data Science](https://towardsdatascience.com/zero-shot-learning-in-deep-learning-8f3e8c8e9a2b) | Google, Amazon, Facebook | Hard | Deep Learning, Transfer Learning |
| 85 | Meta-Learning: Learning to Learn | [Towards Data Science](https://towardsdatascience.com/meta-learning-what-is-it-and-why-it-matters-7a1a1e9d9e3) | Google, Amazon, Facebook | Hard | Deep Learning, Optimization |
| 86 | Hyperparameter Sensitivity Analysis | [Towards Data Science](https://towardsdatascience.com/hyperparameter-sensitivity-analysis-123456789) | Google, Amazon, Facebook | Medium | Hyperparameter Tuning |
| 87 | High-Dimensional Feature Selection Techniques | [Towards Data Science](https://towardsdatascience.com/feature-selection-methods-abc123) | Google, Amazon, Facebook | Hard | Feature Engineering, Dimensionality Reduction |
| 88 | Multi-Label Classification Techniques | [Towards Data Science](https://towardsdatascience.com/multi-label-classification-methods-456def) | Google, Amazon, Facebook | Hard | Classification, Multi-Output |
| 89 | Ordinal Regression in Machine Learning | [Towards Data Science](https://towardsdatascience.com/ordinal-regression-explained-789ghi) | Google, Amazon, Facebook | Medium | Regression, Classification |
| 90 | Survival Analysis in ML | [Towards Data Science](https://towardsdatascience.com/survival-analysis-in-machine-learning-abc789) | Google, Amazon, Facebook | Hard | Statistics, ML |
| 91 | Semi-Supervised Learning Methods | [Towards Data Science](https://towardsdatascience.com/semi-supervised-learning-101-123abc) | Google, Amazon, Facebook | Hard | Unsupervised Learning, ML Basics |
| 92 | Unsupervised Feature Learning | [Towards Data Science](https://towardsdatascience.com/unsupervised-feature-learning-abc456) | Google, Amazon, Facebook | Medium | Unsupervised Learning, Feature Extraction |
| 93 | Clustering Evaluation Metrics: Silhouette, Davies-Bouldin | [Towards Data Science](https://towardsdatascience.com/clustering-evaluation-metrics-789jkl) | Google, Amazon, Facebook | Medium | Clustering, Evaluation |
| 94 | Dimensionality Reduction: t-SNE and UMAP | [Towards Data Science](https://towardsdatascience.com/t-sne-and-umap-789mno) | Google, Amazon, Facebook | Medium | Dimensionality Reduction |
| 95 | Probabilistic Graphical Models: Bayesian Networks | [Towards Data Science](https://towardsdatascience.com/bayesian-networks-123jkl) | Google, Amazon, Facebook | Hard | Probabilistic Models, Graphical Models |
| 96 | Hidden Markov Models (HMMs) in ML | [Towards Data Science](https://towardsdatascience.com/hidden-markov-models-101-456mno) | Google, Amazon, Facebook | Hard | Probabilistic Models, Sequence Modeling |
| 97 | Recommender Systems: Collaborative Filtering | [Towards Data Science](https://towardsdatascience.com/collaborative-filtering-789pqr) | Google, Amazon, Facebook | Medium | Recommender Systems |
| 98 | Recommender Systems: Content-Based Filtering | [Towards Data Science](https://towardsdatascience.com/content-based-recommender-systems-123stu) | Google, Amazon, Facebook | Medium | Recommender Systems |
| 99 | Anomaly Detection in Time Series Data | [Towards Data Science](https://towardsdatascience.com/anomaly-detection-in-time-series-data-456vwx) | Google, Amazon, Facebook | Hard | Time Series, Anomaly Detection |
| 100 | Optimization Algorithms Beyond Gradient Descent (Adam, RMSProp, etc.) | [Towards Data Science](https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6f8eb4c5b0e4) | Google, Amazon, Facebook | Medium | Optimization, Deep Learning |

---

## Questions asked in Google interview
- Bias-Variance Tradeoff  
- Cross-Validation  
- Overfitting and Underfitting  
- Gradient Descent  
- Neural Networks: Basics  
- Convolutional Neural Networks (CNNs)  
- Recurrent Neural Networks (RNNs) and LSTMs  
- Reinforcement Learning Basics  
- Hyperparameter Tuning  
- Transfer Learning  

## Questions asked in Facebook interview
- Bias-Variance Tradeoff  
- Cross-Validation  
- Overfitting and Underfitting  
- Neural Networks: Basics  
- Convolutional Neural Networks (CNNs)  
- Recurrent Neural Networks (RNNs) and LSTMs  
- Support Vector Machines (SVM)  
- k-Nearest Neighbors (k-NN)  
- Feature Engineering  
- Dropout in Neural Networks  
- Backpropagation  

## Questions asked in Amazon interview
- Bias-Variance Tradeoff  
- Regularization Techniques (L1, L2)  
- Cross-Validation  
- Overfitting and Underfitting  
- Decision Trees  
- Ensemble Learning: Bagging and Boosting  
- Random Forest  
- Support Vector Machines (SVM)  
- Neural Networks: Basics  
- Hyperparameter Tuning  
- ROC Curve and AUC  
- Logistic Regression  
- Data Normalization and Standardization  
- k-Means Clustering  

## Questions asked in Microsoft interview
- Regularization Techniques (L1, L2)  
- Gradient Descent  
- Convolutional Neural Networks (CNNs)  
- Recurrent Neural Networks (RNNs) and LSTMs  
- Support Vector Machines (SVM)  
- Hyperparameter Tuning  
- ROC Curve and AUC  
- Loss Functions in ML  
- Learning Rate in Optimization  
- Bayesian Optimization for Hyperparameters  

## Questions asked in Uber interview
- Reinforcement Learning Basics  
- Anomaly Detection  
- Gradient Descent Variants  
- Model Deployment: From Prototype to Production  

## Questions asked in Swiggy interview
- Handling Missing Data  
- Data Imputation Techniques in ML  
- Feature Engineering  
- Model Interpretability: SHAP and LIME  

## Questions asked in Flipkart interview
- Ensemble Methods: Stacking and Blending  
- Time Series Forecasting with ARIMA  
- Time Series Forecasting with LSTM  
- Model Deployment: From Prototype to Production  

## Questions asked in Ola interview
- Time Series Forecasting with LSTM  
- Data Normalization and Standardization  
- Recurrent Neural Networks (RNNs) and LSTMs  

## Questions asked in Paytm interview
- Model Deployment: From Prototype to Production  
- Online Learning Algorithms  
- Handling Imbalanced Datasets: SMOTE and Others  

## Questions asked in OYO interview
- Data Preprocessing Techniques  
- Ensemble Learning: Bagging and Boosting  
- Regularization Techniques (L1, L2)  

## Questions asked in WhatsApp interview
- Neural Networks: Basics  
- Convolutional Neural Networks (CNNs)  
- Recurrent Neural Networks (RNNs) and LSTMs  
- Dropout in Neural Networks  

---

### Explain Few-Shot and Zero-Shot Learning - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Meta-Learning`, `Transfer Learning`, `Few-Shot` | **Asked by:** Google, Meta, OpenAI, Amazon

??? success "View Answer"

    **Few-Shot Learning:**
    
    Learn from very few examples (1-shot, 5-shot, etc.) per class.
    
    **Zero-Shot Learning:**
    
    Classify classes never seen during training using semantic information.
    
    **1. Prototypical Networks (Few-Shot):**
    
    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class PrototypicalNetwork(nn.Module):
        """Learn to classify from few examples"""
        
        def __init__(self, input_dim, hidden_dim=64):
            super().__init__()
            
            # Embedding network
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        def forward(self, support_set, support_labels, query_set):
            """
            support_set: [n_support, input_dim]
            support_labels: [n_support]
            query_set: [n_query, input_dim]
            """
            
            # Embed support and query
            support_embeddings = self.encoder(support_set)
            query_embeddings = self.encoder(query_set)
            
            # Compute class prototypes (mean of support examples per class)
            classes = torch.unique(support_labels)
            prototypes = []
            
            for c in classes:
                class_mask = (support_labels == c)
                class_embeddings = support_embeddings[class_mask]
                prototype = class_embeddings.mean(dim=0)
                prototypes.append(prototype)
            
            prototypes = torch.stack(prototypes)  # [n_classes, hidden_dim]
            
            # Classify query by distance to prototypes
            distances = torch.cdist(query_embeddings, prototypes)  # Euclidean
            logits = -distances  # Closer = higher score
            
            return logits
    
    # Few-shot episode
    def create_episode(dataset, n_way=5, k_shot=5, n_query=15):
        """Create N-way K-shot episode"""
        
        # Sample N classes
        classes = np.random.choice(len(dataset.classes), n_way, replace=False)
        
        support_set, support_labels = [], []
        query_set, query_labels = [], []
        
        for idx, cls in enumerate(classes):
            # Get examples from this class
            class_samples = dataset.get_class_samples(cls)
            
            # Sample K for support, rest for query
            samples = np.random.choice(class_samples, k_shot + n_query, replace=False)
            
            support_set.append(samples[:k_shot])
            support_labels.extend([idx] * k_shot)
            
            query_set.append(samples[k_shot:])
            query_labels.extend([idx] * n_query)
        
        return (torch.cat(support_set), torch.tensor(support_labels),
                torch.cat(query_set), torch.tensor(query_labels))
    
    # Training
    model = PrototypicalNetwork(input_dim=784, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters())
    
    for episode in range(10000):
        support_x, support_y, query_x, query_y = create_episode(train_dataset)
        
        logits = model(support_x, support_y, query_x)
        loss = F.cross_entropy(logits, query_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    ```
    
    **2. Matching Networks:**
    
    ```python
    class MatchingNetwork(nn.Module):
        """Attention-based few-shot learning"""
        
        def __init__(self, input_dim, hidden_dim=64):
            super().__init__()
            
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        def forward(self, support_set, support_labels, query_set):
            # Embed all samples
            support_embeddings = self.encoder(support_set)
            query_embeddings = self.encoder(query_set)
            
            # Compute attention weights (cosine similarity)
            attention = F.cosine_similarity(
                query_embeddings.unsqueeze(1),  # [n_query, 1, hidden_dim]
                support_embeddings.unsqueeze(0),  # [1, n_support, hidden_dim]
                dim=2
            )  # [n_query, n_support]
            
            attention = F.softmax(attention, dim=1)
            
            # Predict as weighted combination of support labels
            n_classes = support_labels.max() + 1
            support_one_hot = F.one_hot(support_labels, n_classes).float()
            
            predictions = torch.matmul(attention, support_one_hot)
            
            return predictions
    ```
    
    **3. MAML (Model-Agnostic Meta-Learning):**
    
    ```python
    import higher  # pip install higher
    
    class MAML:
        """Learn initialization that adapts quickly"""
        
        def __init__(self, model, inner_lr=0.01, outer_lr=0.001):
            self.model = model
            self.inner_lr = inner_lr
            self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)
        
        def inner_loop(self, support_x, support_y, n_steps=5):
            """Fast adaptation on support set"""
            
            # Create differentiable optimizer
            with higher.innerloop_ctx(self.model, 
                                     torch.optim.SGD(self.model.parameters(), 
                                                    lr=self.inner_lr)) as (fmodel, diffopt):
                
                # Inner loop updates
                for _ in range(n_steps):
                    logits = fmodel(support_x)
                    loss = F.cross_entropy(logits, support_y)
                    diffopt.step(loss)
                
                return fmodel
        
        def meta_update(self, task_batch):
            """Outer loop: update initialization"""
            
            meta_loss = 0
            
            for support_x, support_y, query_x, query_y in task_batch:
                # Fast adaptation
                adapted_model = self.inner_loop(support_x, support_y)
                
                # Evaluate on query set
                logits = adapted_model(query_x)
                loss = F.cross_entropy(logits, query_y)
                meta_loss += loss
            
            # Meta-optimization
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            self.meta_optimizer.step()
            
            return meta_loss.item() / len(task_batch)
    
    # Usage
    base_model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 5)  # N-way
    )
    
    maml = MAML(base_model)
    
    for iteration in range(10000):
        task_batch = [create_episode(train_dataset) for _ in range(32)]
        loss = maml.meta_update(task_batch)
    ```
    
    **4. Zero-Shot Learning (Attribute-Based):**
    
    ```python
    class ZeroShotClassifier(nn.Module):
        """Classify unseen classes using attributes"""
        
        def __init__(self, image_dim, attribute_dim):
            super().__init__()
            
            # Map images to attribute space
            self.image_encoder = nn.Sequential(
                nn.Linear(image_dim, 512),
                nn.ReLU(),
                nn.Linear(512, attribute_dim)
            )
        
        def forward(self, images, class_attributes):
            """
            images: [batch_size, image_dim]
            class_attributes: [n_classes, attribute_dim]
                e.g., [has_fur, has_wings, is_large, ...]
            """
            
            # Embed images
            image_embeddings = self.image_encoder(images)
            
            # Compute similarity to each class
            similarities = F.cosine_similarity(
                image_embeddings.unsqueeze(1),  # [batch, 1, attr_dim]
                class_attributes.unsqueeze(0),  # [1, n_classes, attr_dim]
                dim=2
            )
            
            return similarities
    
    # Example: Animal classification
    # Seen classes: dog, cat, bird
    # Unseen class: zebra
    
    class_attributes = {
        'dog': [1, 0, 0, 1, 0],  # has_fur, has_wings, has_stripes, is_mammal, can_fly
        'cat': [1, 0, 0, 1, 0],
        'bird': [0, 1, 0, 0, 1],
        'zebra': [1, 0, 1, 1, 0]  # Unseen during training
    }
    
    # Train on dog, cat, bird
    # Test on zebra using its attributes
    ```
    
    **5. Siamese Networks:**
    
    ```python
    class SiameseNetwork(nn.Module):
        """Learn similarity metric"""
        
        def __init__(self, input_dim):
            super().__init__()
            
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64)
            )
        
        def forward_one(self, x):
            return self.encoder(x)
        
        def forward(self, x1, x2):
            """Compute embeddings for pair"""
            emb1 = self.forward_one(x1)
            emb2 = self.forward_one(x2)
            return emb1, emb2
    
    # Contrastive Loss
    class ContrastiveLoss(nn.Module):
        def __init__(self, margin=1.0):
            super().__init__()
            self.margin = margin
        
        def forward(self, emb1, emb2, label):
            """
            label: 1 if same class, 0 if different
            """
            distance = F.pairwise_distance(emb1, emb2)
            
            # Similar pairs: minimize distance
            # Dissimilar pairs: maximize distance (up to margin)
            loss = label * distance.pow(2) + \
                   (1 - label) * F.relu(self.margin - distance).pow(2)
            
            return loss.mean()
    
    # For few-shot: compare query to support examples
    def predict_few_shot(model, support_set, support_labels, query):
        """Predict by finding nearest neighbor in support"""
        
        query_emb = model.forward_one(query)
        
        distances = []
        for support_sample in support_set:
            support_emb = model.forward_one(support_sample)
            dist = F.pairwise_distance(query_emb, support_emb)
            distances.append(dist)
        
        nearest_idx = torch.argmin(torch.stack(distances))
        return support_labels[nearest_idx]
    ```
    
    **Comparison:**
    
    | Method | Type | Key Idea | Pros | Cons |
    |--------|------|----------|------|------|
    | Prototypical | Few-shot | Class prototypes | Simple, effective | Assumes clustered classes |
    | Matching | Few-shot | Attention over support | Flexible | More complex |
    | MAML | Meta-learning | Learn initialization | General | Slow, memory-intensive |
    | Attribute-based | Zero-shot | Semantic attributes | True zero-shot | Needs attribute annotations |
    | Siamese | Metric learning | Learn similarity | Versatile | Requires pairs |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Advanced learning paradigms.
        
        **Strong answer signals:**
        
        - "Few-shot: learn from K examples"
        - "Zero-shot: unseen classes via attributes"
        - Prototypical networks: class centroids
        - MAML: meta-learning initialization
        - "Metric learning: learn similarity function"
        - Use cases: rare diseases, new products, low-resource languages

---

### What are Optimizers in Neural Networks? - Most Tech Companies Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Optimization`, `Neural Networks`, `Training` | **Asked by:** Google, Amazon, Meta, Microsoft, Apple

??? success "View Answer"

    **Optimizers:**
    
    Algorithms that update model weights to minimize loss function.
    
    **1. Stochastic Gradient Descent (SGD):**
    
    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    # Basic SGD
    model = MyModel()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # With momentum (accelerate in consistent direction)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # With Nesterov momentum (look ahead)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    
    # Training loop
    for X_batch, y_batch in train_loader:
        # Forward
        output = model(X_batch)
        loss = criterion(output, y_batch)
        
        # Backward
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights
    ```
    
    **2. Adam (Adaptive Moment Estimation):**
    
    ```python
    # Most popular optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    
    # Custom implementation
    class AdamOptimizer:
        def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
            self.params = list(params)
            self.lr = lr
            self.beta1 = beta1
            self.beta2 = beta2
            self.eps = eps
            
            # Initialize moment estimates
            self.m = [torch.zeros_like(p) for p in self.params]
            self.v = [torch.zeros_like(p) for p in self.params]
            self.t = 0
        
        def step(self):
            self.t += 1
            
            for i, param in enumerate(self.params):
                if param.grad is None:
                    continue
                
                grad = param.grad
                
                # Update biased first moment
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                
                # Update biased second moment
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
                
                # Bias correction
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                
                # Update parameters
                param.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
    ```
    
    **3. RMSprop:**
    
    ```python
    # Good for RNNs
    optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-8)
    
    # Implementation
    class RMSpropOptimizer:
        def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8):
            self.params = list(params)
            self.lr = lr
            self.alpha = alpha
            self.eps = eps
            
            # Running average of squared gradients
            self.v = [torch.zeros_like(p) for p in self.params]
        
        def step(self):
            for i, param in enumerate(self.params):
                if param.grad is None:
                    continue
                
                grad = param.grad
                
                # Update running average
                self.v[i] = self.alpha * self.v[i] + (1 - self.alpha) * (grad ** 2)
                
                # Update parameters
                param.data -= self.lr * grad / (torch.sqrt(self.v[i]) + self.eps)
    ```
    
    **4. AdamW (Adam with Weight Decay):**
    
    ```python
    # Better regularization than Adam
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Difference from Adam:
    # Adam: weight_decay is added to gradient
    # AdamW: weight_decay is applied directly to weights
    ```
    
    **5. AdaGrad:**
    
    ```python
    # Adapts learning rate per parameter
    optimizer = optim.Adagrad(model.parameters(), lr=0.01)
    
    # Good for: sparse data, different feature scales
    # Issue: learning rate decays too aggressively
    ```
    
    **6. Adadelta:**
    
    ```python
    # Extension of Adagrad (doesn't decay to zero)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0, rho=0.9)
    ```
    
    **Comparison:**
    
    ```python
    import matplotlib.pyplot as plt
    
    # Compare optimizers on same problem
    optimizers = {
        'SGD': optim.SGD(model1.parameters(), lr=0.01),
        'SGD+Momentum': optim.SGD(model2.parameters(), lr=0.01, momentum=0.9),
        'RMSprop': optim.RMSprop(model3.parameters(), lr=0.001),
        'Adam': optim.Adam(model4.parameters(), lr=0.001),
        'AdamW': optim.AdamW(model5.parameters(), lr=0.001, weight_decay=0.01)
    }
    
    histories = {name: [] for name in optimizers.keys()}
    
    for name, optimizer in optimizers.items():
        model = models[name]
        for epoch in range(50):
            loss = train_one_epoch(model, optimizer, train_loader)
            histories[name].append(loss)
    
    # Plot
    plt.figure(figsize=(12, 6))
    for name, losses in histories.items():
        plt.plot(losses, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Optimizer Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.show()
    ```
    
    **Learning Rate Scheduling:**
    
    ```python
    # Combine optimizer with learning rate scheduler
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Step decay
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Cosine annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    # Reduce on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.1, patience=10)
    
    # One cycle
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, 
                                             steps_per_epoch=len(train_loader), 
                                             epochs=50)
    
    # Training loop with scheduler
    for epoch in range(50):
        for X_batch, y_batch in train_loader:
            loss = train_step(model, optimizer, X_batch, y_batch)
            
            # For OneCycleLR: step every batch
            if isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
                scheduler.step()
        
        # For others: step every epoch
        if not isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                val_loss = evaluate(model, val_loader)
                scheduler.step(val_loss)
            else:
                scheduler.step()
    ```
    
    | Optimizer | Pros | Cons | When to Use |
    |-----------|------|------|-------------|
    | SGD | Simple, proven | Slow, needs tuning | Baseline, small models |
    | SGD+Momentum | Faster convergence | Still needs tuning | CNNs, proven recipes |
    | RMSprop | Adaptive LR | Can be unstable | RNNs |
    | Adam | Fast, adaptive, popular | Can overfit | Default choice |
    | AdamW | Better regularization | Slightly slower | Large models, Transformers |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Training knowledge.
        
        **Strong answer signals:**
        
        - "SGD: simple but slow"
        - "Adam: adaptive learning rates"
        - "Momentum: accelerate consistent directions"
        - "AdamW: decouple weight decay"
        - "RMSprop: good for RNNs"
        - Mentions learning rate scheduling
        - "Adam default, but SGD+momentum for vision"

---

### Explain Class Imbalance Handling Techniques - Most Tech Companies Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Imbalanced Data`, `Sampling`, `Loss Functions` | **Asked by:** Google, Amazon, Meta, Microsoft, Netflix

??? success "View Answer"

    **Class Imbalance:**
    
    When one class has significantly more samples than others (e.g., fraud: 0.1% positive).
    
    **1. Resampling:**
    
    ```python
    from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks
    from imblearn.combine import SMOTETomek
    
    # Original imbalanced data
    print(f"Original: {Counter(y_train)}")
    # {0: 9900, 1: 100}
    
    # 1a. Random Over-sampling
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    print(f"Over-sampled: {Counter(y_resampled)}")
    
    # 1b. SMOTE (Synthetic Minority Over-sampling)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    # Custom SMOTE implementation
    def simple_smote(X, y, k=5):
        """Generate synthetic samples"""
        from sklearn.neighbors import NearestNeighbors
        
        minority_class = 1
        X_minority = X[y == minority_class]
        
        # Find K nearest neighbors
        nn = NearestNeighbors(n_neighbors=k+1)
        nn.fit(X_minority)
        
        synthetic_samples = []
        
        for sample in X_minority:
            # Get neighbors
            neighbors = nn.kneighbors([sample], return_distance=False)[0][1:]
            
            # Create synthetic samples
            for neighbor_idx in neighbors:
                # Random point between sample and neighbor
                alpha = np.random.random()
                synthetic = sample + alpha * (X_minority[neighbor_idx] - sample)
                synthetic_samples.append(synthetic)
        
        return np.vstack([X, synthetic_samples])
    
    # 1c. Under-sampling
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    
    # 1d. Combined (SMOTETomek)
    smt = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smt.fit_resample(X_train, y_train)
    ```
    
    **2. Class Weights:**
    
    ```python
    from sklearn.utils.class_weight import compute_class_weight
    
    # Compute weights inversely proportional to class frequency
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights_dict = dict(zip(classes, class_weights))
    
    print(f"Class weights: {class_weights_dict}")
    # {0: 0.505, 1: 50.5}  # Minority class weighted 100x more
    
    # Scikit-learn
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(class_weight='balanced')
    model.fit(X_train, y_train)
    
    # PyTorch
    class_weights_tensor = torch.tensor([0.505, 50.5], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Training
    for X_batch, y_batch in train_loader:
        output = model(X_batch)
        loss = criterion(output, y_batch)  # Weighted loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # TensorFlow/Keras
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    model.fit(X_train, y_train, class_weight=class_weights_dict)
    ```
    
    **3. Focal Loss:**
    
    ```python
    import torch.nn.functional as F
    
    class FocalLoss(nn.Module):
        """Focus on hard examples"""
        
        def __init__(self, alpha=0.25, gamma=2.0):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
        
        def forward(self, inputs, targets):
            """
            inputs: [batch_size, num_classes]
            targets: [batch_size]
            """
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            
            # Get probabilities
            p = torch.exp(-ce_loss)
            
            # Focal term: (1-p)^gamma
            focal_weight = (1 - p) ** self.gamma
            
            # Alpha term (class balancing)
            alpha_t = self.alpha * (targets == 1).float() + (1 - self.alpha) * (targets == 0).float()
            
            loss = alpha_t * focal_weight * ce_loss
            
            return loss.mean()
    
    # Usage
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    for X_batch, y_batch in train_loader:
        output = model(X_batch)
        loss = criterion(output, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    ```
    
    **4. Ensemble Methods:**
    
    ```python
    from sklearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier, EasyEnsembleClassifier
    
    # Balanced Random Forest (under-sample each tree)
    brf = BalancedRandomForestClassifier(n_estimators=100)
    brf.fit(X_train, y_train)
    
    # Balanced Bagging
    from sklearn.tree import DecisionTreeClassifier
    bbc = BalancedBaggingClassifier(
        base_estimator=DecisionTreeClassifier(),
        n_estimators=10,
        random_state=42
    )
    bbc.fit(X_train, y_train)
    
    # Easy Ensemble (multiple balanced subsets)
    eec = EasyEnsembleClassifier(n_estimators=10)
    eec.fit(X_train, y_train)
    ```
    
    **5. Threshold Adjustment:**
    
    ```python
    from sklearn.metrics import precision_recall_curve
    
    # Train model normally
    model.fit(X_train, y_train)
    
    # Get probabilities
    y_scores = model.predict_proba(X_val)[:, 1]
    
    # Find optimal threshold
    precision, recall, thresholds = precision_recall_curve(y_val, y_scores)
    
    # Optimize F1-score
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"Optimal threshold: {optimal_threshold:.3f} (default: 0.5)")
    
    # Predict with optimal threshold
    y_pred = (y_scores >= optimal_threshold).astype(int)
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision[:-1], label='Precision')
    plt.plot(thresholds, recall[:-1], label='Recall')
    plt.plot(thresholds, f1_scores[:-1], label='F1-score')
    plt.axvline(optimal_threshold, color='r', linestyle='--', label='Optimal')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Threshold Selection')
    plt.grid(True, alpha=0.3)
    plt.show()
    ```
    
    **6. Anomaly Detection Approach:**
    
    ```python
    # When extreme imbalance (e.g., 0.01% positive)
    # Treat as anomaly/outlier detection
    
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    
    # Train only on majority class
    X_normal = X_train[y_train == 0]
    
    # Isolation Forest
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    iso_forest.fit(X_normal)
    
    # Predict: -1 for anomaly (minority), 1 for normal
    y_pred = iso_forest.predict(X_test)
    y_pred_binary = (y_pred == -1).astype(int)
    ```
    
    **Comparison:**
    
    | Technique | Pros | Cons | When to Use |
    |-----------|------|------|-------------|
    | SMOTE | Creates synthetic samples | Can create noise | Moderate imbalance (1:10) |
    | Class Weights | Simple, fast | May not help much | Any imbalance |
    | Focal Loss | Focus on hard examples | Needs tuning | Deep learning |
    | Under-sampling | Fast training | Loses information | Lots of data |
    | Threshold Tuning | No retraining needed | Requires validation set | After training |
    | Anomaly Detection | Principled approach | Different paradigm | Extreme imbalance (1:1000+) |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Practical problem-solving.
        
        **Strong answer signals:**
        
        - Multiple techniques (sampling, weighting, loss)
        - "SMOTE: synthetic samples"
        - "Class weights: penalize errors differently"
        - "Focal loss: focus on hard examples"
        - "Adjust threshold after training"
        - "Anomaly detection for extreme cases"
        - Mentions evaluation metrics (PR-AUC, not accuracy)

---

### Explainable AI (XAI) - Google, Amazon, Microsoft Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Model Interpretability`, `SHAP`, `LIME`, `Trust`, `Compliance` | **Asked by:** Google, Amazon, Microsoft, Meta

**Question:** What is Explainable AI (XAI) and why is it important? Explain different techniques for making machine learning models interpretable.

??? success "View Answer"

    **Explainable AI (XAI)** refers to methods and techniques that make machine learning model predictions understandable to humans. It's crucial for trust, debugging, compliance, and ethical AI deployment.

    **Why XAI Matters:**
    
    1. **Trust**: Stakeholders need to understand model decisions
    2. **Debugging**: Identify model errors and biases
    3. **Compliance**: Regulations (GDPR, etc.) require explanations
    4. **Fairness**: Detect and mitigate discrimination
    5. **Safety**: Critical in healthcare, finance, autonomous systems

    **Main XAI Techniques:**

    **1. SHAP (SHapley Additive exPlanations)**
    
    - Based on game theory (Shapley values)
    - Provides consistent feature attributions
    - Works for any model type
    
    ```python
    import shap
    import xgboost as xgb
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    
    # Load data
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = xgb.XGBClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Feature importance
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    
    # Single prediction explanation
    instance_idx = 0
    shap.force_plot(
        explainer.expected_value,
        shap_values[instance_idx],
        X_test.iloc[instance_idx],
        matplotlib=True,
        show=False
    )
    plt.savefig('shap_force.png')
    
    # Waterfall plot for one prediction
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[instance_idx],
            base_values=explainer.expected_value,
            data=X_test.iloc[instance_idx],
            feature_names=X_test.columns.tolist()
        ),
        show=False
    )
    plt.savefig('shap_waterfall.png')
    ```

    **2. LIME (Local Interpretable Model-agnostic Explanations)**
    
    - Explains individual predictions
    - Creates local linear approximations
    - Works for any black-box model
    
    ```python
    from lime import lime_tabular
    import numpy as np
    
    # LIME explainer
    explainer = lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=['Malignant', 'Benign'],
        mode='classification'
    )
    
    # Explain a prediction
    instance_idx = 0
    exp = explainer.explain_instance(
        X_test.iloc[instance_idx].values,
        model.predict_proba,
        num_features=10
    )
    
    # Show explanation
    print("Prediction:", model.predict([X_test.iloc[instance_idx]])[0])
    print("\nFeature contributions:")
    for feature, weight in exp.as_list():
        print(f"{feature}: {weight:.4f}")
    
    # Visualize
    fig = exp.as_pyplot_figure()
    plt.tight_layout()
    plt.savefig('lime_explanation.png')
    ```

    **3. Permutation Feature Importance**
    
    - Measures feature importance by shuffling
    - Model-agnostic
    
    ```python
    from sklearn.inspection import permutation_importance
    
    # Calculate permutation importance
    perm_importance = permutation_importance(
        model, X_test, y_test,
        n_repeats=10,
        random_state=42,
        scoring='accuracy'
    )
    
    # Sort and display
    importance_df = pd.DataFrame({
        'feature': X_test.columns,
        'importance': perm_importance.importances_mean,
        'std': perm_importance.importances_std
    }).sort_values('importance', ascending=False)
    
    print(importance_df.head(10))
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(
        importance_df.head(10)['feature'],
        importance_df.head(10)['importance']
    )
    plt.xlabel('Permutation Importance')
    plt.title('Top 10 Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('permutation_importance.png')
    ```

    **4. Partial Dependence Plots (PDP)**
    
    - Shows marginal effect of features
    - Visualizes feature-target relationships
    
    ```python
    from sklearn.inspection import partial_dependence, PartialDependenceDisplay
    
    # Select features to analyze
    features_to_plot = [0, 1, 2, (0, 1)]  # Individual and interaction
    
    # Create PDP
    fig, ax = plt.subplots(figsize=(12, 4))
    display = PartialDependenceDisplay.from_estimator(
        model,
        X_train,
        features_to_plot,
        feature_names=X_train.columns,
        ax=ax
    )
    plt.tight_layout()
    plt.savefig('partial_dependence.png')
    ```

    **5. Integrated Gradients (for Neural Networks)**
    
    ```python
    import torch
    import torch.nn as nn
    
    class SimpleNN(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            return self.fc(x)
    
    def integrated_gradients(model, input_tensor, baseline=None, steps=50):
        """Calculate integrated gradients."""
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, steps).unsqueeze(1)
        interpolated = baseline + alphas * (input_tensor - baseline)
        interpolated.requires_grad = True
        
        # Calculate gradients
        gradients = []
        for interp in interpolated:
            output = model(interp.unsqueeze(0))
            output.backward()
            gradients.append(interp.grad.clone())
            interp.grad.zero_()
        
        # Average gradients and scale
        avg_gradients = torch.stack(gradients).mean(dim=0)
        integrated_grads = (input_tensor - baseline) * avg_gradients
        
        return integrated_grads
    
    # Example usage
    model = SimpleNN(input_dim=30)
    input_sample = torch.randn(30)
    attributions = integrated_gradients(model, input_sample)
    
    print("Feature attributions:")
    for i, attr in enumerate(attributions):
        print(f"Feature {i}: {attr.item():.4f}")
    ```

    **Comparison of XAI Methods:**

    | Method | Scope | Model-Agnostic | Consistency | Speed |
    |--------|-------|----------------|-------------|-------|
    | SHAP | Local/Global | Yes | High | Slow |
    | LIME | Local | Yes | Medium | Medium |
    | Permutation | Global | Yes | High | Slow |
    | PDP | Global | Yes | High | Medium |
    | Integrated Gradients | Local | No (NN only) | High | Fast |

    **When to Use Each:**
    
    - **SHAP**: Best for comprehensive, theoretically-grounded explanations
    - **LIME**: Quick local explanations for any model
    - **Permutation**: Global feature importance without assumptions
    - **PDP**: Understanding feature effects and interactions
    - **Integrated Gradients**: Deep learning models

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        **Strong answer signals:**
        
        - Multiple XAI techniques (SHAP, LIME, permutation)
        - "SHAP: game theory, Shapley values"
        - "LIME: local linear approximations"
        - "Model-agnostic vs model-specific"
        - Practical considerations (speed vs accuracy)
        - "Compliance and trust requirements"
        - Trade-offs between methods
        - Real-world deployment challenges

---

### Curriculum Learning - DeepMind, OpenAI, Meta AI Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Training Strategy`, `Deep Learning`, `Sample Ordering` | **Asked by:** DeepMind, OpenAI, Meta AI, Google Research

**Question:** What is curriculum learning and how can it improve model training? Provide examples and implementation strategies.

??? success "View Answer"

    **Curriculum Learning** is a training strategy where a model learns from examples organized from easy to hard, mimicking how humans learn. It can accelerate training, improve generalization, and enable learning of complex tasks.

    **Key Concepts:**
    
    1. **Difficulty Scoring**: Quantify example difficulty
    2. **Ordering Strategy**: Sequence examples by difficulty
    3. **Pacing**: Gradually increase task complexity
    4. **Self-Paced**: Let model determine its own curriculum

    **Implementation Example:**

    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    from typing import List, Tuple
    
    class CurriculumDataset(Dataset):
        """Dataset with curriculum learning support."""
        
        def __init__(self, X, y, difficulty_scores=None):
            self.X = torch.FloatTensor(X)
            self.y = torch.LongTensor(y)
            
            # Calculate difficulty if not provided
            if difficulty_scores is None:
                self.difficulty = self._calculate_difficulty()
            else:
                self.difficulty = difficulty_scores
            
            # Sort by difficulty (easy first)
            sorted_indices = np.argsort(self.difficulty)
            self.X = self.X[sorted_indices]
            self.y = self.y[sorted_indices]
            self.difficulty = self.difficulty[sorted_indices]
        
        def _calculate_difficulty(self):
            """Estimate difficulty (e.g., distance to decision boundary)."""
            # Simple heuristic: variance in features
            return torch.var(self.X, dim=1).numpy()
        
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx], self.difficulty[idx]
    
    class CurriculumTrainer:
        """Trainer with curriculum learning strategies."""
        
        def __init__(self, model, device='cpu'):
            self.model = model.to(device)
            self.device = device
        
        def train_vanilla_curriculum(
            self,
            dataset,
            epochs=10,
            batch_size=32,
            lr=0.001
        ):
            """Simple curriculum: easy to hard."""
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
            
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False  # Keep difficulty order
            )
            
            history = []
            
            for epoch in range(epochs):
                self.model.train()
                total_loss = 0
                correct = 0
                total = 0
                
                for X_batch, y_batch, _ in loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += y_batch.size(0)
                    correct += predicted.eq(y_batch).sum().item()
                
                acc = 100.0 * correct / total
                avg_loss = total_loss / len(loader)
                history.append({'epoch': epoch, 'loss': avg_loss, 'acc': acc})
                
                print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={acc:.2f}%")
            
            return history
        
        def train_self_paced_curriculum(
            self,
            dataset,
            epochs=10,
            batch_size=32,
            lr=0.001,
            lambda_init=0.1,
            lambda_growth=1.1
        ):
            """Self-paced learning: model chooses examples."""
            criterion = nn.CrossEntropyLoss(reduction='none')
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
            
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True
            )
            
            lambda_param = lambda_init
            history = []
            
            for epoch in range(epochs):
                self.model.train()
                total_loss = 0
                correct = 0
                total = 0
                selected_count = 0
                
                for X_batch, y_batch, _ in loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(X_batch)
                    losses = criterion(outputs, y_batch)
                    
                    # Self-paced weighting: select easy examples
                    weights = (losses <= lambda_param).float()
                    selected_count += weights.sum().item()
                    
                    # Weighted loss
                    loss = (weights * losses).mean()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += y_batch.size(0)
                    correct += predicted.eq(y_batch).sum().item()
                
                # Increase lambda to include harder examples
                lambda_param *= lambda_growth
                
                acc = 100.0 * correct / total
                avg_loss = total_loss / len(loader)
                selection_rate = 100.0 * selected_count / total
                
                history.append({
                    'epoch': epoch,
                    'loss': avg_loss,
                    'acc': acc,
                    'selection_rate': selection_rate,
                    'lambda': lambda_param
                })
                
                print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={acc:.2f}%, "
                      f"Selected={selection_rate:.1f}%, λ={lambda_param:.3f}")
            
            return history
        
        def train_with_difficulty_stages(
            self,
            dataset,
            stages=3,
            epochs_per_stage=5,
            batch_size=32,
            lr=0.001
        ):
            """Train in stages with increasing difficulty."""
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
            
            # Divide dataset into difficulty stages
            dataset_size = len(dataset)
            stage_size = dataset_size // stages
            
            history = []
            
            for stage in range(stages):
                start_idx = 0  # Always start from beginning (easy)
                end_idx = (stage + 1) * stage_size
                
                # Create subset for this stage
                indices = list(range(min(end_idx, dataset_size)))
                subset = torch.utils.data.Subset(dataset, indices)
                loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
                
                print(f"\n=== Stage {stage+1}/{stages}: Training on "
                      f"{len(subset)} examples ===")
                
                for epoch in range(epochs_per_stage):
                    self.model.train()
                    total_loss = 0
                    correct = 0
                    total = 0
                    
                    for X_batch, y_batch, _ in loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        
                        optimizer.zero_grad()
                        outputs = self.model(X_batch)
                        loss = criterion(outputs, y_batch)
                        loss.backward()
                        optimizer.step()
                        
                        total_loss += loss.item()
                        _, predicted = outputs.max(1)
                        total += y_batch.size(0)
                        correct += predicted.eq(y_batch).sum().item()
                    
                    acc = 100.0 * correct / total
                    avg_loss = total_loss / len(loader)
                    
                    history.append({
                        'stage': stage,
                        'epoch': epoch,
                        'loss': avg_loss,
                        'acc': acc
                    })
                    
                    print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={acc:.2f}%")
            
            return history
    
    # Example: Compare curriculum vs random training
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_classes=3,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Simple model
    class SimpleClassifier(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, num_classes)
            )
        
        def forward(self, x):
            return self.fc(x)
    
    # Create curriculum dataset
    curriculum_dataset = CurriculumDataset(X_train, y_train)
    
    # Train with curriculum
    model_curriculum = SimpleClassifier(input_dim=20, num_classes=3)
    trainer = CurriculumTrainer(model_curriculum)
    
    print("Training with Vanilla Curriculum:")
    history_curriculum = trainer.train_vanilla_curriculum(
        curriculum_dataset,
        epochs=10,
        batch_size=32
    )
    
    print("\n" + "="*50)
    print("Training with Self-Paced Curriculum:")
    model_self_paced = SimpleClassifier(input_dim=20, num_classes=3)
    trainer_sp = CurriculumTrainer(model_self_paced)
    history_sp = trainer_sp.train_self_paced_curriculum(
        curriculum_dataset,
        epochs=10,
        batch_size=32
    )
    
    print("\n" + "="*50)
    print("Training with Staged Curriculum:")
    model_staged = SimpleClassifier(input_dim=20, num_classes=3)
    trainer_staged = CurriculumTrainer(model_staged)
    history_staged = trainer_staged.train_with_difficulty_stages(
        curriculum_dataset,
        stages=3,
        epochs_per_stage=4,
        batch_size=32
    )
    ```

    **Curriculum Strategies:**

    | Strategy | Description | Best For |
    |----------|-------------|----------|
    | Vanilla | Easy→Hard ordering | Stable, predictable training |
    | Self-Paced | Model selects examples | Noisy data, robust learning |
    | Staged | Train in phases | Complex multi-task learning |
    | Transfer | Pre-train→Fine-tune | Domain adaptation |

    **Benefits:**
    
    - **Faster Convergence**: Reach good solutions quicker
    - **Better Generalization**: Avoid local minima
    - **Stability**: Smoother training dynamics
    - **Sample Efficiency**: Learn more from less data

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        **Strong answer signals:**
        
        - "Easy-to-hard ordering, like human learning"
        - "Difficulty scoring mechanisms"
        - "Self-paced: model chooses curriculum"
        - "Staged training for complex tasks"
        - "Faster convergence, better generalization"
        - Implementation strategies
        - Trade-offs and challenges
        - Real-world applications (robotics, NLP)

---

### Active Learning - Google Research, Snorkel AI, Scale AI Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Data Labeling`, `Sample Selection`, `Uncertainty`, `Human-in-the-Loop` | **Asked by:** Google Research, Snorkel AI, Scale AI, Microsoft Research

**Question:** Explain active learning and how it reduces labeling costs. What are different query strategies for selecting samples to label?

??? success "View Answer"

    **Active Learning** is a machine learning paradigm where the model iteratively selects the most informative unlabeled samples for human annotation, minimizing labeling costs while maximizing model performance.

    **Core Concept:**
    
    Instead of randomly labeling data, actively select samples that provide maximum information gain for the model.

    **Active Learning Cycle:**
    
    1. Train model on labeled data
    2. Select most informative unlabeled samples
    3. Get human labels for selected samples
    4. Add to training set
    5. Retrain and repeat

    **Query Strategies:**

    ```python
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from typing import List, Tuple
    import matplotlib.pyplot as plt
    
    class ActiveLearner:
        """Active learning framework with multiple query strategies."""
        
        def __init__(self, model, X_labeled, y_labeled, X_unlabeled):
            self.model = model
            self.X_labeled = X_labeled
            self.y_labeled = y_labeled
            self.X_unlabeled = X_unlabeled
            self.history = []
        
        def uncertainty_sampling(self, n_samples=10):
            """Select samples with highest prediction uncertainty."""
            self.model.fit(self.X_labeled, self.y_labeled)
            probs = self.model.predict_proba(self.X_unlabeled)
            
            # Least confident: 1 - max(prob)
            uncertainty = 1 - np.max(probs, axis=1)
            
            # Select top uncertain samples
            selected_indices = np.argsort(uncertainty)[-n_samples:]
            
            return selected_indices, uncertainty
        
        def margin_sampling(self, n_samples=10):
            """Select samples with smallest margin between top 2 classes."""
            self.model.fit(self.X_labeled, self.y_labeled)
            probs = self.model.predict_proba(self.X_unlabeled)
            
            # Sort probabilities
            sorted_probs = np.sort(probs, axis=1)
            
            # Margin: difference between top 2
            margin = sorted_probs[:, -1] - sorted_probs[:, -2]
            
            # Select smallest margins (most uncertain)
            selected_indices = np.argsort(margin)[:n_samples]
            
            return selected_indices, margin
        
        def entropy_sampling(self, n_samples=10):
            """Select samples with highest entropy."""
            self.model.fit(self.X_labeled, self.y_labeled)
            probs = self.model.predict_proba(self.X_unlabeled)
            
            # Calculate entropy
            epsilon = 1e-10  # Avoid log(0)
            entropy = -np.sum(probs * np.log(probs + epsilon), axis=1)
            
            # Select highest entropy
            selected_indices = np.argsort(entropy)[-n_samples:]
            
            return selected_indices, entropy
        
        def query_by_committee(self, n_samples=10, n_committee=5):
            """Use committee of models to measure disagreement."""
            # Train committee of models with bootstrap
            committee_predictions = []
            
            for _ in range(n_committee):
                # Bootstrap sample
                indices = np.random.choice(
                    len(self.X_labeled),
                    size=len(self.X_labeled),
                    replace=True
                )
                X_boot = self.X_labeled[indices]
                y_boot = self.y_labeled[indices]
                
                # Train committee member
                model_copy = RandomForestClassifier(
                    n_estimators=10,
                    random_state=np.random.randint(1000)
                )
                model_copy.fit(X_boot, y_boot)
                preds = model_copy.predict(self.X_unlabeled)
                committee_predictions.append(preds)
            
            # Calculate vote entropy (disagreement)
            committee_predictions = np.array(committee_predictions)
            disagreement = []
            
            for i in range(len(self.X_unlabeled)):
                votes = committee_predictions[:, i]
                unique, counts = np.unique(votes, return_counts=True)
                vote_dist = counts / len(votes)
                entropy = -np.sum(vote_dist * np.log(vote_dist + 1e-10))
                disagreement.append(entropy)
            
            disagreement = np.array(disagreement)
            selected_indices = np.argsort(disagreement)[-n_samples:]
            
            return selected_indices, disagreement
        
        def expected_model_change(self, n_samples=10):
            """Select samples that would change model most."""
            self.model.fit(self.X_labeled, self.y_labeled)
            
            # Get current model parameters (for linear models)
            if hasattr(self.model, 'feature_importances_'):
                current_importances = self.model.feature_importances_
            else:
                current_importances = None
            
            changes = []
            
            for i in range(len(self.X_unlabeled)):
                # Simulate adding this sample with predicted label
                pred_label = self.model.predict([self.X_unlabeled[i]])[0]
                
                X_temp = np.vstack([self.X_labeled, self.X_unlabeled[i]])
                y_temp = np.append(self.y_labeled, pred_label)
                
                # Retrain
                temp_model = RandomForestClassifier(n_estimators=10, random_state=42)
                temp_model.fit(X_temp, y_temp)
                
                # Calculate parameter change
                if current_importances is not None:
                    new_importances = temp_model.feature_importances_
                    change = np.linalg.norm(new_importances - current_importances)
                else:
                    change = np.random.random()  # Fallback
                
                changes.append(change)
            
            changes = np.array(changes)
            selected_indices = np.argsort(changes)[-n_samples:]
            
            return selected_indices, changes
        
        def diversity_sampling(self, n_samples=10):
            """Select diverse samples using clustering."""
            from sklearn.cluster import KMeans
            
            # Cluster unlabeled data
            kmeans = KMeans(n_clusters=n_samples, random_state=42)
            kmeans.fit(self.X_unlabeled)
            
            # Select samples closest to cluster centers
            selected_indices = []
            for center in kmeans.cluster_centers_:
                distances = np.linalg.norm(self.X_unlabeled - center, axis=1)
                closest_idx = np.argmin(distances)
                if closest_idx not in selected_indices:
                    selected_indices.append(closest_idx)
            
            selected_indices = np.array(selected_indices[:n_samples])
            
            return selected_indices, np.zeros(len(self.X_unlabeled))
        
        def label_samples(self, indices, oracle_labels):
            """Add labeled samples to training set."""
            self.X_labeled = np.vstack([self.X_labeled, self.X_unlabeled[indices]])
            self.y_labeled = np.append(self.y_labeled, oracle_labels)
            self.X_unlabeled = np.delete(self.X_unlabeled, indices, axis=0)
        
        def evaluate(self, X_test, y_test):
            """Evaluate current model."""
            self.model.fit(self.X_labeled, self.y_labeled)
            y_pred = self.model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            return acc
    
    # Compare active learning strategies
    def compare_strategies(X, y, X_test, y_test, n_iterations=10, samples_per_iteration=10):
        """Compare different active learning strategies."""
        
        # Start with small labeled set
        initial_size = 50
        X_labeled = X[:initial_size]
        y_labeled = y[:initial_size]
        X_unlabeled = X[initial_size:]
        y_unlabeled = y[initial_size:]  # Oracle labels (hidden)
        
        strategies = {
            'Random': None,
            'Uncertainty': 'uncertainty_sampling',
            'Margin': 'margin_sampling',
            'Entropy': 'entropy_sampling',
            'Committee': 'query_by_committee'
        }
        
        results = {name: [] for name in strategies.keys()}
        
        for strategy_name, strategy_method in strategies.items():
            print(f"\nRunning {strategy_name} strategy...")
            
            # Reset data
            X_lab = X_labeled.copy()
            y_lab = y_labeled.copy()
            X_unlab = X_unlabeled.copy()
            y_unlab = y_unlabeled.copy()
            
            learner = ActiveLearner(
                RandomForestClassifier(n_estimators=50, random_state=42),
                X_lab, y_lab, X_unlab
            )
            
            for iteration in range(n_iterations):
                # Evaluate
                acc = learner.evaluate(X_test, y_test)
                results[strategy_name].append(acc)
                print(f"  Iteration {iteration+1}: Accuracy = {acc:.4f}, "
                      f"Labeled = {len(learner.y_labeled)}")
                
                if len(learner.X_unlabeled) < samples_per_iteration:
                    break
                
                # Select samples
                if strategy_name == 'Random':
                    indices = np.random.choice(
                        len(learner.X_unlabeled),
                        size=samples_per_iteration,
                        replace=False
                    )
                else:
                    method = getattr(learner, strategy_method)
                    indices, _ = method(n_samples=samples_per_iteration)
                
                # Get oracle labels
                oracle_labels = y_unlab[indices]
                y_unlab = np.delete(y_unlab, indices)
                
                # Update learner
                learner.label_samples(indices, oracle_labels)
        
        return results
    
    # Generate data and compare
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_classes=3,
        random_state=42
    )
    
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    results = compare_strategies(
        X_pool, y_pool,
        X_test, y_test,
        n_iterations=10,
        samples_per_iteration=20
    )
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for strategy, accuracies in results.items():
        labeled_sizes = [50 + i*20 for i in range(len(accuracies))]
        plt.plot(labeled_sizes, accuracies, marker='o', label=strategy)
    
    plt.xlabel('Number of Labeled Samples')
    plt.ylabel('Test Accuracy')
    plt.title('Active Learning Strategy Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('active_learning_comparison.png')
    plt.show()
    ```

    **Query Strategy Comparison:**

    | Strategy | Principle | Pros | Cons |
    |----------|-----------|------|------|
    | Uncertainty | Max prediction uncertainty | Simple, effective | Can select outliers |
    | Margin | Min margin between classes | Robust to noise | Only for classifiers |
    | Entropy | Max entropy | Theoretically grounded | Computationally expensive |
    | QBC | Committee disagreement | Reduces overfitting | Requires multiple models |
    | Expected Change | Max parameter change | Direct optimization | Very slow |
    | Diversity | Cover feature space | Good coverage | May miss informative samples |

    **Real-World Applications:**
    
    - **Medical imaging**: Label only diagnostically uncertain cases
    - **NLP**: Annotate ambiguous text samples
    - **Autonomous driving**: Label edge cases and rare scenarios
    - **Fraud detection**: Investigate suspicious transactions

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        **Strong answer signals:**
        
        - "Iterative: select → label → retrain"
        - "Minimize labeling costs"
        - Multiple query strategies (uncertainty, margin, entropy, QBC)
        - "Uncertainty sampling: most uncertain predictions"
        - "Query by committee: model disagreement"
        - "Cold start problem: initial labeled set"
        - "Stopping criteria: performance plateau"
        - Real-world cost-benefit analysis
        - Implementation challenges (outliers, class imbalance)

---

### Meta-Learning (Learning to Learn) - DeepMind, OpenAI Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Meta-Learning`, `MAML`, `Few-Shot`, `Transfer Learning` | **Asked by:** DeepMind, OpenAI, Meta AI, Google Research

**Question:** What is meta-learning and how does it differ from traditional transfer learning? Explain MAML (Model-Agnostic Meta-Learning) and its advantages.

??? success "View Answer"

    **Meta-Learning** (learning to learn) trains models to quickly adapt to new tasks with minimal data by learning a good initialization or learning strategy across multiple related tasks.

    **Key Difference from Transfer Learning:**
    
    - **Transfer Learning**: Learn features from task A, apply to task B
    - **Meta-Learning**: Learn *how to learn* across many tasks, rapidly adapt to new tasks

    **MAML (Model-Agnostic Meta-Learning):**
    
    Finds model parameters that are easily adaptable to new tasks with just a few gradient steps.

    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from typing import List, Tuple
    import matplotlib.pyplot as plt
    
    class MAMLModel(nn.Module):
        """Simple neural network for MAML."""
        
        def __init__(self, input_dim=1, hidden_dim=40, output_dim=1):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)
        
        def clone(self):
            """Create a copy with same architecture."""
            cloned = MAMLModel(
                input_dim=self.fc1.in_features,
                hidden_dim=self.fc1.out_features,
                output_dim=self.fc3.out_features
            )
            cloned.load_state_dict(self.state_dict())
            return cloned
    
    class MAML:
        """Model-Agnostic Meta-Learning implementation."""
        
        def __init__(
            self,
            model,
            meta_lr=0.001,
            inner_lr=0.01,
            inner_steps=5
        ):
            self.model = model
            self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
            self.inner_lr = inner_lr
            self.inner_steps = inner_steps
        
        def inner_loop(self, task_data, task_labels):
            """Adapt model to a specific task (inner loop)."""
            # Clone model for task-specific adaptation
            adapted_model = self.model.clone()
            
            # Task-specific optimizer
            task_optimizer = torch.optim.SGD(
                adapted_model.parameters(),
                lr=self.inner_lr
            )
            
            # Adapt on support set
            for _ in range(self.inner_steps):
                task_optimizer.zero_grad()
                predictions = adapted_model(task_data)
                loss = F.mse_loss(predictions, task_labels)
                loss.backward()
                task_optimizer.step()
            
            return adapted_model
        
        def meta_train_step(self, tasks):
            """
            Meta-training step across multiple tasks.
            
            Args:
                tasks: List of (support_x, support_y, query_x, query_y) tuples
            """
            self.meta_optimizer.zero_grad()
            
            meta_loss = 0
            
            for support_x, support_y, query_x, query_y in tasks:
                # Inner loop: adapt to task
                adapted_model = self.inner_loop(support_x, support_y)
                
                # Outer loop: evaluate on query set
                query_pred = adapted_model(query_x)
                task_loss = F.mse_loss(query_pred, query_y)
                meta_loss += task_loss
            
            # Meta-update
            meta_loss /= len(tasks)
            meta_loss.backward()
            self.meta_optimizer.step()
            
            return meta_loss.item()
        
        def adapt_to_new_task(self, support_x, support_y):
            """Quickly adapt to a new task."""
            return self.inner_loop(support_x, support_y)
    
    # Example: Sine wave regression tasks
    def generate_sine_task(amplitude=None, phase=None, n_samples=10):
        """Generate a sine wave regression task."""
        if amplitude is None:
            amplitude = np.random.uniform(0.1, 5.0)
        if phase is None:
            phase = np.random.uniform(0, np.pi)
        
        x = np.random.uniform(-5, 5, n_samples)
        y = amplitude * np.sin(x + phase)
        
        x = torch.FloatTensor(x).unsqueeze(1)
        y = torch.FloatTensor(y).unsqueeze(1)
        
        return x, y, amplitude, phase
    
    def train_maml():
        """Train MAML on sine wave tasks."""
        # Initialize model and MAML
        model = MAMLModel(input_dim=1, hidden_dim=40, output_dim=1)
        maml = MAML(
            model,
            meta_lr=0.001,
            inner_lr=0.01,
            inner_steps=5
        )
        
        n_meta_iterations = 1000
        tasks_per_iteration = 5
        
        meta_losses = []
        
        print("Meta-Training MAML...")
        for iteration in range(n_meta_iterations):
            # Sample batch of tasks
            tasks = []
            for _ in range(tasks_per_iteration):
                # Support set (for adaptation)
                support_x, support_y, _, _ = generate_sine_task(n_samples=10)
                # Query set (for meta-update)
                query_x, query_y, _, _ = generate_sine_task(n_samples=10)
                tasks.append((support_x, support_y, query_x, query_y))
            
            # Meta-train step
            meta_loss = maml.meta_train_step(tasks)
            meta_losses.append(meta_loss)
            
            if (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration+1}: Meta-loss = {meta_loss:.4f}")
        
        return maml, meta_losses
    
    def compare_maml_vs_scratch():
        """Compare MAML adaptation vs training from scratch."""
        # Train MAML
        maml, _ = train_maml()
        
        # Test on a new task
        test_amplitude, test_phase = 2.0, np.pi/4
        support_x, support_y, _, _ = generate_sine_task(
            amplitude=test_amplitude,
            phase=test_phase,
            n_samples=10
        )
        
        # Adapt MAML model
        adapted_model = maml.adapt_to_new_task(support_x, support_y)
        
        # Train model from scratch
        scratch_model = MAMLModel(input_dim=1, hidden_dim=40, output_dim=1)
        scratch_optimizer = torch.optim.Adam(scratch_model.parameters(), lr=0.01)
        
        for _ in range(100):  # More steps than MAML inner loop
            scratch_optimizer.zero_grad()
            pred = scratch_model(support_x)
            loss = F.mse_loss(pred, support_y)
            loss.backward()
            scratch_optimizer.step()
        
        # Evaluate both
        test_x = torch.linspace(-5, 5, 100).unsqueeze(1)
        test_y = test_amplitude * np.sin(test_x.numpy() + test_phase)
        
        with torch.no_grad():
            maml_pred = adapted_model(test_x).numpy()
            scratch_pred = scratch_model(test_x).numpy()
        
        # Plot comparison
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(support_x.numpy(), support_y.numpy(), 
                   c='red', s=100, label='Support Set', zorder=5)
        plt.plot(test_x.numpy(), test_y, 'k--', label='True Function', linewidth=2)
        plt.plot(test_x.numpy(), maml_pred, 'b-', label='MAML (5 steps)', linewidth=2)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('MAML: Fast Adaptation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.scatter(support_x.numpy(), support_y.numpy(), 
                   c='red', s=100, label='Support Set', zorder=5)
        plt.plot(test_x.numpy(), test_y, 'k--', label='True Function', linewidth=2)
        plt.plot(test_x.numpy(), scratch_pred, 'g-', label='From Scratch (100 steps)', linewidth=2)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Training from Scratch')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('maml_comparison.png')
        plt.show()
        
        # Calculate MSE
        maml_mse = np.mean((maml_pred - test_y)**2)
        scratch_mse = np.mean((scratch_pred - test_y)**2)
        
        print(f"\nMAML MSE (after 5 steps): {maml_mse:.4f}")
        print(f"From Scratch MSE (after 100 steps): {scratch_mse:.4f}")
    
    # Run comparison
    compare_maml_vs_scratch()
    
    # Prototypical Networks (alternative meta-learning approach)
    class PrototypicalNetwork:
        """Prototypical networks for few-shot classification."""
        
        def __init__(self, embedding_dim=64):
            self.encoder = nn.Sequential(
                nn.Linear(784, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, embedding_dim)
            )
        
        def compute_prototypes(self, support_embeddings, support_labels):
            """Compute class prototypes (mean of support embeddings)."""
            unique_labels = torch.unique(support_labels)
            prototypes = []
            
            for label in unique_labels:
                class_embeddings = support_embeddings[support_labels == label]
                prototype = class_embeddings.mean(dim=0)
                prototypes.append(prototype)
            
            return torch.stack(prototypes)
        
        def classify(self, query_embeddings, prototypes):
            """Classify based on nearest prototype."""
            # Euclidean distance to prototypes
            distances = torch.cdist(query_embeddings, prototypes)
            # Negative distance as logits
            return -distances
    ```

    **Meta-Learning Approaches:**

    | Method | Key Idea | Pros | Cons |
    |--------|----------|------|------|
    | MAML | Good initialization | Fast adaptation | Expensive meta-training |
    | Prototypical | Prototype-based classification | Simple, effective | Limited to classification |
    | Meta-SGD | Learn learning rates | Flexible | More parameters |
    | Reptile | First-order MAML | Computationally cheaper | Slightly worse performance |

    **Applications:**
    
    - **Few-shot learning**: Learn from 1-5 examples
    - **Robotics**: Quick adaptation to new tasks
    - **Drug discovery**: Transfer across molecular tasks
    - **Personalization**: Adapt to individual users

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        **Strong answer signals:**
        
        - "Learning to learn across tasks"
        - "MAML: good initialization for fast adaptation"
        - "Inner loop: task adaptation, outer loop: meta-update"
        - "Few-shot learning: learn from 1-5 examples"
        - vs. "Transfer learning: feature reuse"
        - "Prototypical networks: prototype-based"
        - Applications (robotics, personalization)
        - "Expensive meta-training, fast adaptation"

---

### Continual/Lifelong Learning - DeepMind, Meta AI Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Catastrophic Forgetting`, `Continual Learning`, `EWC`, `Replay` | **Asked by:** DeepMind, Meta AI, Google Research, Microsoft Research

**Question:** What is catastrophic forgetting and how can we enable continual learning? Explain strategies like Elastic Weight Consolidation (EWC) and experience replay.

??? success "View Answer"

    **Catastrophic Forgetting** occurs when a neural network forgets previously learned tasks upon learning new ones. **Continual/Lifelong Learning** aims to learn sequential tasks without forgetting.

    **The Problem:**
    
    Standard neural networks trained on Task B will overwrite weights learned for Task A, losing performance on A.

    **Solutions:**

    **1. Elastic Weight Consolidation (EWC)**
    
    Protects important weights for old tasks using Fisher Information Matrix.

    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np
    import matplotlib.pyplot as plt
    
    class EWC:
        """Elastic Weight Consolidation for continual learning."""
        
        def __init__(self, model, dataset, device='cpu'):
            self.model = model
            self.device = device
            self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
            self.fisher = self._compute_fisher(dataset)
        
        def _compute_fisher(self, dataset):
            """Compute Fisher Information Matrix diagonal."""
            fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
            
            self.model.eval()
            loader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.model.zero_grad()
                
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                
                # Accumulate squared gradients
                for n, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        fisher[n] += p.grad.pow(2)
            
            # Normalize
            n_samples = len(dataset)
            for n in fisher:
                fisher[n] /= n_samples
            
            return fisher
        
        def penalty(self):
            """Calculate EWC penalty term."""
            loss = 0
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    loss += (self.fisher[n] * (p - self.params[n]).pow(2)).sum()
            return loss
    
    class ContinualLearner:
        """Framework for continual learning experiments."""
        
        def __init__(self, model, device='cpu'):
            self.model = model.to(device)
            self.device = device
            self.ewc_list = []
        
        def train_task(
            self,
            train_loader,
            epochs=10,
            lr=0.001,
            ewc_lambda=5000,
            use_ewc=True
        ):
            """Train on a new task with optional EWC."""
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            
            self.model.train()
            
            for epoch in range(epochs):
                total_loss = 0
                correct = 0
                total = 0
                
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    
                    # Standard loss
                    loss = criterion(outputs, labels)
                    
                    # Add EWC penalty for previous tasks
                    if use_ewc:
                        for ewc in self.ewc_list:
                            loss += ewc_lambda * ewc.penalty()
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                
                acc = 100.0 * correct / total
                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Acc={acc:.2f}%")
        
        def add_ewc_constraint(self, dataset):
            """Add EWC constraint for current task."""
            ewc = EWC(self.model, dataset, self.device)
            self.ewc_list.append(ewc)
        
        def evaluate(self, test_loader):
            """Evaluate on test set."""
            self.model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            return 100.0 * correct / total
    
    # 2. Experience Replay
    class ReplayBuffer:
        """Store and replay past experiences."""
        
        def __init__(self, capacity=1000):
            self.capacity = capacity
            self.buffer = []
        
        def add_task_samples(self, dataset, samples_per_task=100):
            """Add samples from current task to buffer."""
            indices = np.random.choice(len(dataset), 
                                      min(samples_per_task, len(dataset)), 
                                      replace=False)
            
            for idx in indices:
                if len(self.buffer) >= self.capacity:
                    # Remove oldest samples
                    self.buffer.pop(0)
                self.buffer.append(dataset[idx])
        
        def get_replay_loader(self, batch_size=32):
            """Get data loader for replay samples."""
            if not self.buffer:
                return None
            
            X = torch.stack([x for x, _ in self.buffer])
            y = torch.tensor([y for _, y in self.buffer])
            dataset = TensorDataset(X, y)
            return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Progressive Neural Networks
    class ProgressiveNN(nn.Module):
        """Progressive Neural Networks: add new columns for new tasks."""
        
        def __init__(self, input_dim, hidden_dim, output_dims):
            super().__init__()
            self.columns = nn.ModuleList()
            self.adapters = nn.ModuleList()
            
            # First column
            self.add_column(input_dim, hidden_dim, output_dims[0])
        
        def add_column(self, input_dim, hidden_dim, output_dim):
            """Add a new column for a new task."""
            column = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
            self.columns.append(column)
            
            # Add lateral connections from previous columns
            if len(self.columns) > 1:
                adapters = nn.ModuleList([
                    nn.Linear(hidden_dim, hidden_dim) 
                    for _ in range(len(self.columns) - 1)
                ])
                self.adapters.append(adapters)
        
        def forward(self, x, task_id=-1):
            """Forward pass through column for task_id."""
            if task_id == -1:
                task_id = len(self.columns) - 1
            
            # Compute activations from previous columns
            prev_activations = []
            for i in range(task_id):
                with torch.no_grad():  # Freeze previous columns
                    h = self.columns[i][:-1](x)  # All but last layer
                    prev_activations.append(h)
            
            # Current column with lateral connections
            h = x
            for i, layer in enumerate(self.columns[task_id][:-1]):
                h = layer(h)
                
                # Add lateral connections
                if task_id > 0 and i > 0:
                    for j, prev_h in enumerate(prev_activations):
                        h = h + self.adapters[task_id-1][j](prev_h)
            
            # Final output layer
            return self.columns[task_id][-1](h)
    
    # Comparison experiment
    def compare_continual_methods():
        """Compare different continual learning strategies."""
        from sklearn.datasets import make_classification
        
        # Create 3 different tasks
        tasks = []
        for seed in [42, 43, 44]:
            X, y = make_classification(
                n_samples=500,
                n_features=20,
                n_informative=15,
                n_classes=3,
                random_state=seed
            )
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.LongTensor(y)
            dataset = TensorDataset(X_tensor, y_tensor)
            tasks.append(dataset)
        
        # Simple model
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Sequential(
                    nn.Linear(20, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 3)
                )
            
            def forward(self, x):
                return self.fc(x)
        
        methods = {
            'Naive (No Protection)': {'use_ewc': False, 'use_replay': False},
            'EWC': {'use_ewc': True, 'use_replay': False},
            'Experience Replay': {'use_ewc': False, 'use_replay': True},
            'EWC + Replay': {'use_ewc': True, 'use_replay': True}
        }
        
        results = {method: [] for method in methods}
        
        for method_name, config in methods.items():
            print(f"\n{'='*60}")
            print(f"Method: {method_name}")
            print('='*60)
            
            model = SimpleNet()
            learner = ContinualLearner(model)
            replay_buffer = ReplayBuffer(capacity=500) if config['use_replay'] else None
            
            task_accuracies = []
            
            for task_id, task_dataset in enumerate(tasks):
                print(f"\nTraining on Task {task_id+1}...")
                
                # Prepare data loader
                train_loader = DataLoader(task_dataset, batch_size=32, shuffle=True)
                
                # Add replay samples
                if replay_buffer and replay_buffer.buffer:
                    # Mix current task with replay
                    replay_loader = replay_buffer.get_replay_loader()
                    # For simplicity, train on task then replay
                    learner.train_task(train_loader, epochs=10, 
                                      use_ewc=config['use_ewc'])
                    learner.train_task(replay_loader, epochs=5, 
                                      use_ewc=config['use_ewc'])
                else:
                    learner.train_task(train_loader, epochs=10, 
                                      use_ewc=config['use_ewc'])
                
                # Add EWC constraint
                if config['use_ewc']:
                    learner.add_ewc_constraint(task_dataset)
                
                # Add to replay buffer
                if replay_buffer:
                    replay_buffer.add_task_samples(task_dataset, samples_per_task=100)
                
                # Evaluate on all previous tasks
                print(f"\nEvaluation after Task {task_id+1}:")
                for eval_task_id, eval_dataset in enumerate(tasks[:task_id+1]):
                    eval_loader = DataLoader(eval_dataset, batch_size=32)
                    acc = learner.evaluate(eval_loader)
                    print(f"  Task {eval_task_id+1} Accuracy: {acc:.2f}%")
                    
                    if len(task_accuracies) <= eval_task_id:
                        task_accuracies.append([])
                    task_accuracies[eval_task_id].append(acc)
            
            results[method_name] = task_accuracies
        
        # Plot forgetting
        plt.figure(figsize=(12, 5))
        
        for i, (method_name, task_accs) in enumerate(results.items()):
            plt.subplot(1, 2, 1)
            for task_id, accs in enumerate(task_accs):
                plt.plot(range(task_id, len(accs)+task_id), accs, 
                        marker='o', label=f'{method_name} - Task {task_id+1}')
        
        plt.xlabel('After Training Task')
        plt.ylabel('Accuracy (%)')
        plt.title('Task Performance Over Time')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Average forgetting
        plt.subplot(1, 2, 2)
        avg_forgetting = []
        for method_name, task_accs in results.items():
            # Calculate average forgetting
            forgetting = []
            for task_id, accs in enumerate(task_accs[:-1]):
                forgetting.append(accs[task_id] - accs[-1])  # Initial - Final
            avg_f = np.mean(forgetting) if forgetting else 0
            avg_forgetting.append(avg_f)
        
        plt.bar(range(len(methods)), avg_forgetting)
        plt.xticks(range(len(methods)), methods.keys(), rotation=45, ha='right')
        plt.ylabel('Average Forgetting (%)')
        plt.title('Catastrophic Forgetting Comparison')
        plt.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('continual_learning_comparison.png')
        plt.show()
    
    # Run comparison
    compare_continual_methods()
    ```

    **Continual Learning Strategies:**

    | Method | Key Idea | Pros | Cons |
    |--------|----------|------|------|
    | EWC | Protect important weights | No memory overhead | Hyperparameter sensitive |
    | Experience Replay | Store past examples | Simple, effective | Memory overhead |
    | Progressive NN | New network per task | No forgetting | Network grows unbounded |
    | PackNet | Prune and freeze | Compact | Requires pruning |
    | GEM | Constrained optimization | Strong guarantees | Computationally expensive |

    **Applications:**
    
    - **Robotics**: Learn new skills without forgetting old ones
    - **Personalization**: Adapt to user preferences over time
    - **Edge AI**: Update models without retraining from scratch

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        **Strong answer signals:**
        
        - "Catastrophic forgetting: forgetting old tasks"
        - "EWC: protect important weights using Fisher information"
        - "Experience replay: store and replay past samples"
        - "Progressive networks: add columns for new tasks"
        - "Trade-off: memory vs computation vs forgetting"
        - "Stability-plasticity dilemma"
        - Real-world constraints (memory, compute)
        - Evaluation metrics (average accuracy, forgetting)

---

### Graph Neural Networks (GNNs) - DeepMind, Meta AI, Twitter Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Graph Learning`, `GCN`, `Message Passing`, `Node Embeddings` | **Asked by:** DeepMind, Meta AI, Twitter, Pinterest, Uber

**Question:** What are Graph Neural Networks and how do they work? Explain message passing and different GNN architectures (GCN, GAT, GraphSAGE).

??? success "View Answer"

    **Graph Neural Networks (GNNs)** learn representations for graph-structured data by iteratively aggregating information from neighbors through message passing.

    **Key Concepts:**
    
    - **Nodes**: Entities (users, molecules, papers)
    - **Edges**: Relationships (friendships, bonds, citations)
    - **Features**: Node/edge attributes
    - **Message Passing**: Nodes exchange and aggregate neighbor information

    **Core GNN Operation:**
    
    $$h_v^{(k+1)} = \text{UPDATE}(h_v^{(k)}, \text{AGGREGATE}(\{h_u^{(k)} : u \in \mathcal{N}(v)\}))$$

    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt
    from typing import List, Tuple
    
    # 1. Graph Convolutional Network (GCN)
    class GCNLayer(nn.Module):
        """Graph Convolutional Layer."""
        
        def __init__(self, in_features, out_features):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)
        
        def forward(self, X, adj):
            """
            Args:
                X: Node features (N x in_features)
                adj: Adjacency matrix (N x N)
            """
            # Normalize adjacency: D^{-1/2} A D^{-1/2}
            D = torch.diag(adj.sum(dim=1).pow(-0.5))
            adj_norm = D @ adj @ D
            
            # Aggregate neighbors and transform
            return F.relu(self.linear(adj_norm @ X))
    
    class GCN(nn.Module):
        """Multi-layer GCN."""
        
        def __init__(self, in_features, hidden_dim, out_features, num_layers=2):
            super().__init__()
            self.layers = nn.ModuleList()
            
            # Input layer
            self.layers.append(GCNLayer(in_features, hidden_dim))
            
            # Hidden layers
            for _ in range(num_layers - 2):
                self.layers.append(GCNLayer(hidden_dim, hidden_dim))
            
            # Output layer
            self.layers.append(GCNLayer(hidden_dim, out_features))
        
        def forward(self, X, adj):
            for layer in self.layers[:-1]:
                X = layer(X, adj)
                X = F.dropout(X, p=0.5, training=self.training)
            return self.layers[-1](X, adj)
    
    # 2. Graph Attention Network (GAT)
    class GATLayer(nn.Module):
        """Graph Attention Layer with multi-head attention."""
        
        def __init__(self, in_features, out_features, num_heads=8, dropout=0.6):
            super().__init__()
            self.num_heads = num_heads
            self.out_features = out_features
            
            # Linear transformations
            self.W = nn.Parameter(torch.FloatTensor(in_features, out_features * num_heads))
            self.a = nn.Parameter(torch.FloatTensor(2 * out_features, 1))
            
            self.dropout = dropout
            self.leaky_relu = nn.LeakyReLU(0.2)
            
            self.reset_parameters()
        
        def reset_parameters(self):
            nn.init.xavier_uniform_(self.W)
            nn.init.xavier_uniform_(self.a)
        
        def forward(self, X, adj):
            """
            Args:
                X: Node features (N x in_features)
                adj: Adjacency matrix (N x N)
            """
            N = X.size(0)
            
            # Linear transformation
            H = X @ self.W  # (N x out_features*num_heads)
            H = H.view(N, self.num_heads, self.out_features)  # (N x heads x out)
            
            # Attention mechanism
            # Concatenate for all pairs
            a_input = torch.cat([
                H.repeat(1, N, 1).view(N * N, self.num_heads, self.out_features),
                H.repeat(N, 1, 1)
            ], dim=2).view(N, N, self.num_heads, 2 * self.out_features)
            
            # Compute attention scores
            e = self.leaky_relu((a_input @ self.a.view(2 * self.out_features, 1)).squeeze(-1))
            
            # Mask non-neighbors
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj.unsqueeze(-1) > 0, e, zero_vec)
            
            # Softmax
            attention = F.softmax(attention, dim=1)
            attention = F.dropout(attention, self.dropout, training=self.training)
            
            # Weighted sum
            H_prime = torch.matmul(attention.transpose(1, 2), H.unsqueeze(1).repeat(1, N, 1, 1))
            H_prime = H_prime.mean(dim=2)  # Average over heads
            
            return F.elu(H_prime)
    
    # 3. GraphSAGE
    class GraphSAGELayer(nn.Module):
        """GraphSAGE layer with sampling."""
        
        def __init__(self, in_features, out_features, aggregator='mean'):
            super().__init__()
            self.aggregator = aggregator
            
            # Separate transforms for self and neighbors
            self.W_self = nn.Linear(in_features, out_features)
            self.W_neigh = nn.Linear(in_features, out_features)
        
        def forward(self, X, adj, sample_size=10):
            """
            Args:
                X: Node features (N x in_features)
                adj: Adjacency matrix (N x N)
                sample_size: Number of neighbors to sample
            """
            N = X.size(0)
            
            # Sample neighbors
            neighbor_features = []
            for i in range(N):
                neighbors = torch.where(adj[i] > 0)[0]
                
                if len(neighbors) > sample_size:
                    # Sample
                    sampled = neighbors[torch.randperm(len(neighbors))[:sample_size]]
                else:
                    sampled = neighbors
                
                if len(sampled) > 0:
                    if self.aggregator == 'mean':
                        neigh_feat = X[sampled].mean(dim=0)
                    elif self.aggregator == 'max':
                        neigh_feat = X[sampled].max(dim=0)[0]
                    elif self.aggregator == 'lstm':
                        # LSTM aggregator (simplified)
                        neigh_feat = X[sampled].mean(dim=0)
                    else:
                        neigh_feat = X[sampled].mean(dim=0)
                else:
                    neigh_feat = torch.zeros_like(X[0])
                
                neighbor_features.append(neigh_feat)
            
            neighbor_features = torch.stack(neighbor_features)
            
            # Combine self and neighbor features
            self_features = self.W_self(X)
            neigh_features = self.W_neigh(neighbor_features)
            
            output = self_features + neigh_features
            
            # L2 normalization
            output = F.normalize(output, p=2, dim=1)
            
            return F.relu(output)
    
    # Example: Node classification on Karate Club
    def node_classification_example():
        """Node classification on Karate Club graph."""
        # Load Karate Club graph
        G = nx.karate_club_graph()
        
        # Create features (degree, clustering coefficient, etc.)
        n_nodes = G.number_of_nodes()
        features = []
        for node in G.nodes():
            features.append([
                G.degree(node),
                nx.clustering(G, node),
                nx.closeness_centrality(G, node)
            ])
        
        X = torch.FloatTensor(features)
        
        # Adjacency matrix
        adj = nx.to_numpy_array(G)
        adj = torch.FloatTensor(adj + np.eye(n_nodes))  # Add self-loops
        
        # Labels (which karate club each node joined)
        labels = [G.nodes[node]['club'] == 'Mr. Hi' for node in G.nodes()]
        y = torch.LongTensor(labels)
        
        # Train/test split
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[torch.randperm(n_nodes)[:int(0.6 * n_nodes)]] = True
        test_mask = ~train_mask
        
        # Train GCN
        model = GCN(in_features=3, hidden_dim=16, out_features=2, num_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        
        print("Training GCN...")
        model.train()
        for epoch in range(200):
            optimizer.zero_grad()
            out = model(X, adj)
            loss = F.cross_entropy(out[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 50 == 0:
                model.eval()
                with torch.no_grad():
                    pred = model(X, adj).argmax(dim=1)
                    train_acc = (pred[train_mask] == y[train_mask]).float().mean()
                    test_acc = (pred[test_mask] == y[test_mask]).float().mean()
                print(f"Epoch {epoch+1}: Loss={loss:.4f}, "
                      f"Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
                model.train()
        
        # Visualize embeddings
        model.eval()
        with torch.no_grad():
            embeddings = model.layers[0](X, adj).numpy()
        
        # Plot
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, node_color=y.numpy(), cmap='coolwarm', 
               with_labels=True, node_size=500)
        plt.title("Original Graph with True Labels")
        
        plt.subplot(1, 2, 2)
        plt.scatter(embeddings[:, 0], embeddings[:, 1], c=y.numpy(), cmap='coolwarm', s=100)
        for i, (x, y_coord) in enumerate(embeddings):
            plt.annotate(str(i), (x, y_coord), fontsize=8)
        plt.xlabel("Embedding Dimension 1")
        plt.ylabel("Embedding Dimension 2")
        plt.title("GCN Node Embeddings")
        
        plt.tight_layout()
        plt.savefig('gnn_embeddings.png')
        plt.show()
    
    # Run example
    node_classification_example()
    
    # Link prediction example
    def link_prediction_example():
        """Link prediction using GNN embeddings."""
        # Create a graph
        G = nx.karate_club_graph()
        
        # Remove some edges for testing
        edges = list(G.edges())
        np.random.shuffle(edges)
        n_test = len(edges) // 5
        test_edges = edges[:n_test]
        train_edges = edges[n_test:]
        
        # Create negative samples
        non_edges = list(nx.non_edges(G))
        np.random.shuffle(non_edges)
        neg_test_edges = non_edges[:n_test]
        
        print(f"Train edges: {len(train_edges)}")
        print(f"Test edges: {len(test_edges)}")
        print(f"Negative test edges: {len(neg_test_edges)}")
        
        # Train GNN on remaining graph
        G_train = nx.Graph()
        G_train.add_nodes_from(G.nodes())
        G_train.add_edges_from(train_edges)
        
        # ... (train GNN and predict links)
    
    link_prediction_example()
    ```

    **GNN Architecture Comparison:**

    | Architecture | Aggregation | Attention | Sampling | Best For |
    |--------------|-------------|-----------|----------|----------|
    | GCN | Mean | No | No | Transductive learning |
    | GAT | Weighted | Yes | No | Varying neighbor importance |
    | GraphSAGE | Mean/Max/LSTM | No | Yes | Inductive learning, large graphs |
    | GIN | Sum | No | No | Graph-level tasks |

    **Applications:**
    
    - **Social Networks**: Friend recommendation, influence prediction
    - **Molecules**: Property prediction, drug discovery
    - **Recommendation**: User-item graphs
    - **Knowledge Graphs**: Link prediction, reasoning
    - **Traffic**: Traffic flow prediction

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        **Strong answer signals:**
        
        - "Message passing: aggregate neighbor information"
        - "GCN: spectral convolution on graphs"
        - "GAT: attention-weighted aggregation"
        - "GraphSAGE: sampling for scalability"
        - "Inductive vs transductive learning"
        - "Over-smoothing problem in deep GNNs"
        - Applications (social networks, molecules)
        - "Node, edge, and graph-level tasks"

---

### Reinforcement Learning Basics - DeepMind, OpenAI, Tesla Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `RL`, `Q-Learning`, `Policy Gradient`, `Markov Decision Process` | **Asked by:** DeepMind, OpenAI, Tesla, Cruise, Waymo

**Question:** Explain the basics of Reinforcement Learning. What are the differences between value-based (Q-Learning) and policy-based (Policy Gradient) methods?

??? success "View Answer"

    **Reinforcement Learning (RL)** is learning optimal behavior through trial and error by interacting with an environment to maximize cumulative reward.

    **Core Components:**
    
    - **Agent**: Learner/decision maker
    - **Environment**: World the agent interacts with
    - **State (s)**: Current situation
    - **Action (a)**: Decision made by agent
    - **Reward (r)**: Feedback signal
    - **Policy (π)**: Strategy mapping states to actions

    **Markov Decision Process (MDP):**
    
    $$(S, A, P, R, \gamma)$$
    
    - S: State space
    - A: Action space
    - P: Transition probabilities
    - R: Reward function
    - γ: Discount factor

    **Q-Learning (Value-Based):**

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict
    
    class QLearning:
        """Q-Learning algorithm for discrete state/action spaces."""
        
        def __init__(
            self,
            n_states,
            n_actions,
            learning_rate=0.1,
            discount_factor=0.99,
            epsilon=0.1
        ):
            self.Q = np.zeros((n_states, n_actions))
            self.lr = learning_rate
            self.gamma = discount_factor
            self.epsilon = epsilon
        
        def select_action(self, state):
            """Epsilon-greedy action selection."""
            if np.random.random() < self.epsilon:
                return np.random.randint(self.Q.shape[1])  # Explore
            return np.argmax(self.Q[state])  # Exploit
        
        def update(self, state, action, reward, next_state, done):
            """Q-Learning update rule."""
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.max(self.Q[next_state])
            
            # TD error
            td_error = target - self.Q[state, action]
            
            # Update Q-value
            self.Q[state, action] += self.lr * td_error
            
            return td_error
    
    # Simple Grid World environment
    class GridWorld:
        """Simple grid world for RL."""
        
        def __init__(self, size=5):
            self.size = size
            self.n_states = size * size
            self.n_actions = 4  # Up, Down, Left, Right
            self.goal = (size-1, size-1)
            self.reset()
        
        def reset(self):
            self.agent_pos = (0, 0)
            return self._get_state()
        
        def _get_state(self):
            return self.agent_pos[0] * self.size + self.agent_pos[1]
        
        def step(self, action):
            """Execute action and return (next_state, reward, done)."""
            row, col = self.agent_pos
            
            # Actions: 0=Up, 1=Down, 2=Left, 3=Right
            if action == 0:  # Up
                row = max(0, row - 1)
            elif action == 1:  # Down
                row = min(self.size - 1, row + 1)
            elif action == 2:  # Left
                col = max(0, col - 1)
            elif action == 3:  # Right
                col = min(self.size - 1, col + 1)
            
            self.agent_pos = (row, col)
            
            # Reward
            if self.agent_pos == self.goal:
                reward = 1.0
                done = True
            else:
                reward = -0.01  # Small penalty for each step
                done = False
            
            return self._get_state(), reward, done
        
        def render(self, Q=None):
            """Visualize grid and policy."""
            grid = np.zeros((self.size, self.size))
            grid[self.agent_pos] = 0.5
            grid[self.goal] = 1.0
            
            plt.figure(figsize=(8, 8))
            plt.imshow(grid, cmap='viridis')
            
            # Draw policy arrows
            if Q is not None:
                arrow_map = {0: '↑', 1: '↓', 2: '←', 3: '→'}
                for i in range(self.size):
                    for j in range(self.size):
                        state = i * self.size + j
                        best_action = np.argmax(Q[state])
                        plt.text(j, i, arrow_map[best_action], 
                               ha='center', va='center', fontsize=20)
            
            plt.xticks([])
            plt.yticks([])
            plt.title("Grid World")
            plt.tight_layout()
            plt.savefig('gridworld_policy.png')
    
    # Train Q-Learning
    def train_q_learning(n_episodes=500):
        env = GridWorld(size=5)
        agent = QLearning(
            n_states=env.n_states,
            n_actions=env.n_actions,
            learning_rate=0.1,
            discount_factor=0.99,
            epsilon=0.1
        )
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            
            while steps < 100:  # Max steps per episode
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
                agent.update(state, action, reward, next_state, done)
                
                total_reward += reward
                steps += 1
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_length = np.mean(episode_lengths[-100:])
                print(f"Episode {episode+1}: Avg Reward={avg_reward:.3f}, "
                      f"Avg Length={avg_length:.1f}")
        
        return agent, episode_rewards, episode_lengths
    
    # Policy Gradient (REINFORCE)
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    class PolicyNetwork(nn.Module):
        """Neural network for policy."""
        
        def __init__(self, state_dim, action_dim, hidden_dim=128):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Softmax(dim=-1)
            )
        
        def forward(self, state):
            return self.fc(state)
    
    class PolicyGradient:
        """REINFORCE algorithm."""
        
        def __init__(self, state_dim, action_dim, learning_rate=0.001):
            self.policy = PolicyNetwork(state_dim, action_dim)
            self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        def select_action(self, state):
            """Sample action from policy."""
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = self.policy(state_tensor)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            return action.item(), log_prob
        
        def update(self, log_probs, rewards, gamma=0.99):
            """Update policy using REINFORCE."""
            # Calculate discounted returns
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)
            
            returns = torch.FloatTensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Policy gradient loss
            policy_loss = []
            for log_prob, G in zip(log_probs, returns):
                policy_loss.append(-log_prob * G)
            
            # Update
            self.optimizer.zero_grad()
            loss = torch.stack(policy_loss).sum()
            loss.backward()
            self.optimizer.step()
            
            return loss.item()
    
    # Compare Q-Learning vs Policy Gradient
    print("Training Q-Learning...")
    q_agent, q_rewards, q_lengths = train_q_learning(n_episodes=500)
    
    # Visualize learned policy
    env = GridWorld(size=5)
    env.render(Q=q_agent.Q)
    
    # Plot learning curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    window = 20
    q_rewards_smooth = np.convolve(q_rewards, np.ones(window)/window, mode='valid')
    plt.plot(q_rewards_smooth, label='Q-Learning')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    q_lengths_smooth = np.convolve(q_lengths, np.ones(window)/window, mode='valid')
    plt.plot(q_lengths_smooth, label='Q-Learning')
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    plt.title('Episode Length Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rl_learning_curves.png')
    plt.show()
    ```

    **Q-Learning vs Policy Gradient:**

    | Aspect | Q-Learning (Value-Based) | Policy Gradient |
    |--------|-------------------------|-----------------|
    | **Learns** | Q-values (state-action values) | Policy directly |
    | **Action Selection** | Argmax over Q-values | Sample from distribution |
    | **Continuous Actions** | Difficult | Natural |
    | **Convergence** | Can diverge with function approx | More stable |
    | **Sample Efficiency** | More efficient | Less efficient |
    | **Stochastic Policies** | Difficult | Natural |

    **Key RL Algorithms:**
    
    - **Value-Based**: Q-Learning, DQN, Double DQN
    - **Policy-Based**: REINFORCE, PPO, TRPO
    - **Actor-Critic**: A3C, SAC, TD3 (combines both)

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        **Strong answer signals:**
        
        - "MDP: states, actions, rewards, transitions"
        - "Q-Learning: learn Q(s,a), act greedily"
        - "Policy Gradient: learn π(a|s) directly"
        - "Exploration vs exploitation trade-off"
        - "Q-Learning: off-policy, sample efficient"
        - "Policy Gradient: on-policy, handles continuous"
        - "Actor-Critic: combines both approaches"
        - Applications (robotics, games, recommendation)

---

### Variational Autoencoders (VAEs) - DeepMind, OpenAI Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Generative Models`, `Latent Variables`, `VAE`, `ELBO` | **Asked by:** DeepMind, OpenAI, Meta AI, Stability AI

**Question:** Explain Variational Autoencoders (VAEs). How do they differ from regular autoencoders? What is the reparameterization trick and why is it needed?

??? success "View Answer"

    **Variational Autoencoders (VAEs)** are generative models that learn a probabilistic mapping between data and a latent space, enabling generation of new samples.

    **Key Differences from Regular Autoencoders:**

    | Regular Autoencoder | Variational Autoencoder |
    |---------------------|------------------------|
    | Deterministic encoding | Probabilistic encoding |
    | Learns point in latent space | Learns distribution in latent space |
    | Can't generate new samples reliably | Can generate new samples |
    | Reconstruction loss only | Reconstruction + KL divergence loss |

    **VAE Architecture:**
    
    - **Encoder**: Maps x → (μ, σ) representing q(z|x)
    - **Latent Space**: Sample z ~ N(μ, σ²)
    - **Decoder**: Maps z → x̂ representing p(x|z)

    **Loss Function (ELBO):**
    
    $$\mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))$$
    
    - First term: Reconstruction loss
    - Second term: KL divergence (regularization)

    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import matplotlib.pyplot as plt
    import numpy as np
    
    class VAE(nn.Module):
        """Variational Autoencoder."""
        
        def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
            super().__init__()
            
            # Encoder
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc_mu = nn.Linear(hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
            
            # Decoder
            self.fc3 = nn.Linear(latent_dim, hidden_dim)
            self.fc4 = nn.Linear(hidden_dim, input_dim)
        
        def encode(self, x):
            """Encode input to latent distribution parameters."""
            h = F.relu(self.fc1(x))
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar
        
        def reparameterize(self, mu, logvar):
            """
            Reparameterization trick: z = μ + σ * ε where ε ~ N(0,1)
            This allows backpropagation through sampling.
            """
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return z
        
        def decode(self, z):
            """Decode latent variable to reconstruction."""
            h = F.relu(self.fc3(z))
            return torch.sigmoid(self.fc4(h))
        
        def forward(self, x):
            """Full forward pass."""
            mu, logvar = self.encode(x.view(-1, 784))
            z = self.reparameterize(mu, logvar)
            recon_x = self.decode(z)
            return recon_x, mu, logvar
    
    def vae_loss(recon_x, x, mu, logvar):
        """
        VAE loss = Reconstruction loss + KL divergence.
        
        KL(q(z|x) || p(z)) = -0.5 * sum(1 + log(σ²) - μ² - σ²)
        """
        # Reconstruction loss (binary cross-entropy)
        BCE = F.binary_cross_entropy(
            recon_x,
            x.view(-1, 784),
            reduction='sum'
        )
        
        # KL divergence
        # KL(N(μ, σ²) || N(0, 1))
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return BCE + KLD, BCE, KLD
    
    # Train VAE on MNIST
    def train_vae():
        """Train VAE on MNIST dataset."""
        # Data
        transform = transforms.ToTensor()
        train_dataset = datasets.MNIST(
            './data',
            train=True,
            download=True,
            transform=transform
        )
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        
        # Model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = VAE(latent_dim=20).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # Training
        model.train()
        train_losses = []
        recon_losses = []
        kl_losses = []
        
        n_epochs = 10
        
        print("Training VAE...")
        for epoch in range(n_epochs):
            epoch_loss = 0
            epoch_recon = 0
            epoch_kl = 0
            
            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.to(device)
                
                optimizer.zero_grad()
                recon_batch, mu, logvar = model(data)
                loss, recon, kl = vae_loss(recon_batch, data, mu, logvar)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_recon += recon.item()
                epoch_kl += kl.item()
            
            avg_loss = epoch_loss / len(train_dataset)
            avg_recon = epoch_recon / len(train_dataset)
            avg_kl = epoch_kl / len(train_dataset)
            
            train_losses.append(avg_loss)
            recon_losses.append(avg_recon)
            kl_losses.append(avg_kl)
            
            print(f"Epoch {epoch+1}/{n_epochs}: "
                  f"Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, KL={avg_kl:.4f}")
        
        return model, train_losses, recon_losses, kl_losses
    
    # Train model
    model, train_losses, recon_losses, kl_losses = train_vae()
    
    # Visualize results
    def visualize_vae(model, n_samples=10):
        """Visualize VAE reconstructions and generations."""
        device = next(model.parameters()).device
        model.eval()
        
        # Load test data
        test_dataset = datasets.MNIST(
            './data',
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )
        
        # Reconstructions
        fig, axes = plt.subplots(2, n_samples, figsize=(15, 3))
        
        for i in range(n_samples):
            # Original
            img, _ = test_dataset[i]
            axes[0, i].imshow(img.squeeze(), cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original', fontsize=10)
            
            # Reconstruction
            with torch.no_grad():
                img_tensor = img.to(device)
                recon, _, _ = model(img_tensor.unsqueeze(0))
                recon_img = recon.view(28, 28).cpu()
            
            axes[1, i].imshow(recon_img, cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstructed', fontsize=10)
        
        plt.suptitle('VAE Reconstructions')
        plt.tight_layout()
        plt.savefig('vae_reconstructions.png')
        plt.show()
        
        # Generate new samples
        fig, axes = plt.subplots(2, n_samples, figsize=(15, 3))
        
        with torch.no_grad():
            # Sample from prior N(0, 1)
            z = torch.randn(n_samples, 20).to(device)
            generated = model.decode(z).view(n_samples, 28, 28).cpu()
            
            for i in range(n_samples):
                axes[0, i].imshow(generated[i], cmap='gray')
                axes[0, i].axis('off')
            
            # Interpolation between two samples
            z1 = torch.randn(1, 20).to(device)
            z2 = torch.randn(1, 20).to(device)
            
            for i in range(n_samples):
                alpha = i / (n_samples - 1)
                z_interp = (1 - alpha) * z1 + alpha * z2
                img_interp = model.decode(z_interp).view(28, 28).cpu()
                axes[1, i].imshow(img_interp, cmap='gray')
                axes[1, i].axis('off')
        
        axes[0, 0].set_title('Random Samples', fontsize=10)
        axes[1, 0].set_title('Interpolation', fontsize=10)
        plt.tight_layout()
        plt.savefig('vae_generations.png')
        plt.show()
        
        # Latent space visualization (2D)
        if model.fc_mu.out_features == 2:
            # Encode test set
            test_loader = DataLoader(test_dataset, batch_size=100)
            latents = []
            labels = []
            
            with torch.no_grad():
                for data, label in test_loader:
                    data = data.to(device)
                    mu, _ = model.encode(data.view(-1, 784))
                    latents.append(mu.cpu())
                    labels.append(label)
            
            latents = torch.cat(latents).numpy()
            labels = torch.cat(labels).numpy()
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(
                latents[:, 0],
                latents[:, 1],
                c=labels,
                cmap='tab10',
                alpha=0.5
            )
            plt.colorbar(scatter)
            plt.xlabel('Latent Dimension 1')
            plt.ylabel('Latent Dimension 2')
            plt.title('VAE Latent Space (colored by digit)')
            plt.tight_layout()
            plt.savefig('vae_latent_space.png')
            plt.show()
    
    visualize_vae(model)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(recon_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss')
    plt.title('Reconstruction Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(kl_losses)
    plt.xlabel('Epoch')
    plt.ylabel('KL Divergence')
    plt.title('KL Divergence')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vae_training_curves.png')
    plt.show()
    ```

    **The Reparameterization Trick:**
    
    **Problem**: Can't backpropagate through sampling operation z ~ N(μ, σ²)
    
    **Solution**: Reparameterize as z = μ + σ × ε where ε ~ N(0, 1)
    
    - Randomness moved to ε (no parameters)
    - Gradients flow through μ and σ
    - Enables end-to-end training

    **Applications:**
    
    - **Image Generation**: Generate realistic images
    - **Data Augmentation**: Synthetic training data
    - **Anomaly Detection**: Detect out-of-distribution samples
    - **Representation Learning**: Learn meaningful embeddings
    - **Drug Discovery**: Generate novel molecules

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        **Strong answer signals:**
        
        - "Probabilistic encoder: outputs μ and σ"
        - "Reparameterization trick: z = μ + σε"
        - "ELBO: reconstruction + KL divergence"
        - "KL term: regularizes latent space"
        - vs. "Regular AE: deterministic, can't generate"
        - "Continuous latent space enables interpolation"
        - "β-VAE: weighted KL for disentanglement"
        - Applications and limitations

---

### Generative Adversarial Networks (GANs) - NVIDIA, OpenAI, Stability AI Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Generative Models`, `GANs`, `Adversarial Training`, `Mode Collapse` | **Asked by:** NVIDIA, OpenAI, Stability AI, Meta AI, DeepMind

**Question:** Explain how Generative Adversarial Networks (GANs) work. What are common training challenges like mode collapse, and how can they be addressed?

??? success "View Answer"

    **Generative Adversarial Networks (GANs)** consist of two neural networks—a Generator and a Discriminator—that compete in a game-theoretic framework to generate realistic data.

    **Architecture:**
    
    - **Generator (G)**: Learns to create fake samples from noise
    - **Discriminator (D)**: Learns to distinguish real from fake
    - **Training**: Minimax game between G and D

    **Objective:**
    
    $$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Generator Network
    class Generator(nn.Module):
        """Generator network for GAN."""
        
        def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
            super().__init__()
            self.img_shape = img_shape
            
            def block(in_feat, out_feat, normalize=True):
                layers = [nn.Linear(in_feat, out_feat)]
                if normalize:
                    layers.append(nn.BatchNorm1d(out_feat, 0.8))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                return layers
            
            self.model = nn.Sequential(
                *block(latent_dim, 128, normalize=False),
                *block(128, 256),
                *block(256, 512),
                *block(512, 1024),
                nn.Linear(1024, int(np.prod(img_shape))),
                nn.Tanh()
            )
        
        def forward(self, z):
            img = self.model(z)
            img = img.view(img.size(0), *self.img_shape)
            return img
    
    # Discriminator Network
    class Discriminator(nn.Module):
        """Discriminator network for GAN."""
        
        def __init__(self, img_shape=(1, 28, 28)):
            super().__init__()
            
            self.model = nn.Sequential(
                nn.Linear(int(np.prod(img_shape)), 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
        
        def forward(self, img):
            img_flat = img.view(img.size(0), -1)
            validity = self.model(img_flat)
            return validity
    
    # GAN Trainer
    class GANTrainer:
        """Trainer for standard GAN."""
        
        def __init__(
            self,
            generator,
            discriminator,
            latent_dim=100,
            lr=0.0002,
            b1=0.5,
            b2=0.999,
            device='cpu'
        ):
            self.generator = generator.to(device)
            self.discriminator = discriminator.to(device)
            self.latent_dim = latent_dim
            self.device = device
            
            # Optimizers
            self.optimizer_G = optim.Adam(
                generator.parameters(),
                lr=lr,
                betas=(b1, b2)
            )
            self.optimizer_D = optim.Adam(
                discriminator.parameters(),
                lr=lr,
                betas=(b1, b2)
            )
            
            # Loss
            self.adversarial_loss = nn.BCELoss()
        
        def train_step(self, real_imgs):
            """Single training step."""
            batch_size = real_imgs.size(0)
            
            # Adversarial ground truths
            valid = torch.ones(batch_size, 1, device=self.device)
            fake = torch.zeros(batch_size, 1, device=self.device)
            
            real_imgs = real_imgs.to(self.device)
            
            # ---------------------
            # Train Generator
            # ---------------------
            self.optimizer_G.zero_grad()
            
            # Sample noise
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            
            # Generate fake images
            gen_imgs = self.generator(z)
            
            # Generator loss (fool discriminator)
            g_loss = self.adversarial_loss(self.discriminator(gen_imgs), valid)
            
            g_loss.backward()
            self.optimizer_G.step()
            
            # ---------------------
            # Train Discriminator
            # ---------------------
            self.optimizer_D.zero_grad()
            
            # Real images
            real_loss = self.adversarial_loss(self.discriminator(real_imgs), valid)
            
            # Fake images
            fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
            
            # Total discriminator loss
            d_loss = (real_loss + fake_loss) / 2
            
            d_loss.backward()
            self.optimizer_D.step()
            
            return g_loss.item(), d_loss.item(), gen_imgs
    
    # Wasserstein GAN (addresses training stability)
    class WassersteinGAN(GANTrainer):
        """WGAN with gradient penalty (WGAN-GP)."""
        
        def __init__(self, generator, discriminator, latent_dim=100, **kwargs):
            super().__init__(generator, discriminator, latent_dim, **kwargs)
            self.lambda_gp = 10  # Gradient penalty coefficient
        
        def compute_gradient_penalty(self, real_samples, fake_samples):
            """Calculate gradient penalty for WGAN-GP."""
            batch_size = real_samples.size(0)
            
            # Random weight term for interpolation
            alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
            
            # Interpolated samples
            interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
            
            d_interpolates = self.discriminator(interpolates)
            
            fake = torch.ones(batch_size, 1, device=self.device)
            
            # Get gradients
            gradients = torch.autograd.grad(
                outputs=d_interpolates,
                inputs=interpolates,
                grad_outputs=fake,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            
            gradients = gradients.view(batch_size, -1)
            
            # Calculate penalty
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            
            return gradient_penalty
        
        def train_step(self, real_imgs):
            """WGAN-GP training step."""
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(self.device)
            
            # ---------------------
            # Train Discriminator (Critic)
            # ---------------------
            self.optimizer_D.zero_grad()
            
            # Sample noise
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            
            # Generate fake images
            gen_imgs = self.generator(z)
            
            # Discriminator outputs
            real_validity = self.discriminator(real_imgs)
            fake_validity = self.discriminator(gen_imgs.detach())
            
            # Gradient penalty
            gradient_penalty = self.compute_gradient_penalty(real_imgs, gen_imgs.detach())
            
            # Wasserstein loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.lambda_gp * gradient_penalty
            
            d_loss.backward()
            self.optimizer_D.step()
            
            # ---------------------
            # Train Generator
            # ---------------------
            self.optimizer_G.zero_grad()
            
            # Generate new fake images
            gen_imgs = self.generator(z)
            fake_validity = self.discriminator(gen_imgs)
            
            g_loss = -torch.mean(fake_validity)
            
            g_loss.backward()
            self.optimizer_G.step()
            
            return g_loss.item(), d_loss.item(), gen_imgs
    
    # Train GAN
    def train_gan(gan_type='standard', n_epochs=50):
        """Train GAN on MNIST."""
        # Data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        train_dataset = datasets.MNIST(
            './data',
            train=True,
            download=True,
            transform=transform
        )
        dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        # Models
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        generator = Generator(latent_dim=100)
        discriminator = Discriminator()
        
        # Trainer
        if gan_type == 'wgan':
            trainer = WassersteinGAN(generator, discriminator, device=device)
        else:
            trainer = GANTrainer(generator, discriminator, device=device)
        
        # Training
        g_losses = []
        d_losses = []
        
        print(f"Training {gan_type.upper()}...")
        for epoch in range(n_epochs):
            epoch_g_loss = 0
            epoch_d_loss = 0
            
            for i, (imgs, _) in enumerate(dataloader):
                g_loss, d_loss, gen_imgs = trainer.train_step(imgs)
                
                epoch_g_loss += g_loss
                epoch_d_loss += d_loss
            
            avg_g_loss = epoch_g_loss / len(dataloader)
            avg_d_loss = epoch_d_loss / len(dataloader)
            
            g_losses.append(avg_g_loss)
            d_losses.append(avg_d_loss)
            
            print(f"Epoch {epoch+1}/{n_epochs}: G_loss={avg_g_loss:.4f}, D_loss={avg_d_loss:.4f}")
            
            # Save generated images
            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    z = torch.randn(25, 100, device=device)
                    gen_imgs = generator(z)
                    
                    fig, axes = plt.subplots(5, 5, figsize=(8, 8))
                    for idx, ax in enumerate(axes.flat):
                        img = gen_imgs[idx].cpu().squeeze()
                        ax.imshow(img, cmap='gray')
                        ax.axis('off')
                    
                    plt.suptitle(f'{gan_type.upper()} - Epoch {epoch+1}')
                    plt.tight_layout()
                    plt.savefig(f'{gan_type}_epoch_{epoch+1}.png')
                    plt.close()
        
        return generator, g_losses, d_losses
    
    # Train both types
    print("=" * 60)
    gen_standard, g_losses_std, d_losses_std = train_gan('standard', n_epochs=50)
    
    print("\n" + "=" * 60)
    gen_wgan, g_losses_wgan, d_losses_wgan = train_gan('wgan', n_epochs=50)
    
    # Compare training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(g_losses_std, label='Standard GAN - Generator')
    plt.plot(d_losses_std, label='Standard GAN - Discriminator')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Standard GAN Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(g_losses_wgan, label='WGAN-GP - Generator')
    plt.plot(d_losses_wgan, label='WGAN-GP - Critic')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('WGAN-GP Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gan_training_comparison.png')
    plt.show()
    ```

    **Common GAN Challenges & Solutions:**

    | Challenge | Description | Solutions |
    |-----------|-------------|-----------|
    | **Mode Collapse** | G produces limited variety | Minibatch discrimination, unrolled GAN |
    | **Training Instability** | Oscillating losses | WGAN, Spectral normalization |
    | **Vanishing Gradients** | D too strong, G can't learn | Feature matching, label smoothing |
    | **Hyperparameter Sensitivity** | Hard to tune | Progressive growing, StyleGAN |

    **GAN Variants:**
    
    - **DCGAN**: Use convolutional layers
    - **WGAN/WGAN-GP**: Wasserstein distance + gradient penalty
    - **StyleGAN**: Style-based generation
    - **CycleGAN**: Unpaired image-to-image translation
    - **Pix2Pix**: Paired image-to-image translation
    - **ProGAN**: Progressive growing

    **Applications:**
    
    - **Image Generation**: Realistic faces, artwork
    - **Super Resolution**: Enhance image quality
    - **Style Transfer**: Artistic style application
    - **Data Augmentation**: Generate training data
    - **Anomaly Detection**: Identify unusual patterns

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        **Strong answer signals:**
        
        - "Adversarial training: generator vs discriminator"
        - "Minimax game: G fools D, D distinguishes real/fake"
        - "Mode collapse: limited output diversity"
        - "WGAN: Wasserstein distance for stability"
        - "Gradient penalty enforces Lipschitz constraint"
        - "Training tricks: label smoothing, feature matching"
        - vs. "VAE: explicit likelihood, GAN: implicit"
        - Applications and recent advances (StyleGAN)

---

### Transformer Architecture - Google, OpenAI, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Transformers`, `Self-Attention`, `BERT`, `GPT` | **Asked by:** Google, OpenAI, Meta, Anthropic, Cohere

**Question:** Explain the Transformer architecture. How does self-attention work? What are the key differences between BERT and GPT?

??? success "View Answer"

    **Transformers** revolutionized NLP by replacing recurrence with self-attention mechanisms, enabling parallel processing and better long-range dependencies.

    **Key Components:**
    
    1. **Self-Attention**: Compute relevance between all positions
    2. **Multi-Head Attention**: Multiple attention patterns
    3. **Position Encoding**: Inject sequence order information
    4. **Feed-Forward Networks**: Process attended representations
    5. **Layer Normalization & Residual Connections**: Training stability

    **Self-Attention Mechanism:**
    
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

    - Q (Query), K (Key), V (Value) from input embeddings
    - Scaled dot-product attention
    - Output: weighted sum of values

    **BERT vs GPT:**

    | Aspect | BERT | GPT |
    |--------|------|-----|
    | **Architecture** | Encoder-only | Decoder-only |
    | **Training** | Masked Language Modeling | Causal Language Modeling |
    | **Attention** | Bidirectional | Unidirectional (causal) |
    | **Best For** | Understanding tasks | Generation tasks |
    | **Examples** | Classification, NER, QA | Text generation, completion |

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "Self-attention: compute relationships between all tokens"
        - "Multi-head: multiple attention patterns"
        - "Positional encoding: inject order information"
        - "BERT: bidirectional, masked LM"
        - "GPT: autoregressive, causal LM"
        - "Transformers: parallelizable, better than RNNs"
        - Applications in NLP, CV (ViT), multimodal

---

### Fine-Tuning vs Transfer Learning - Google, Meta, OpenAI Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Transfer Learning`, `Fine-Tuning`, `Feature Extraction`, `Domain Adaptation` | **Asked by:** Google, Meta, OpenAI, Microsoft, Amazon

**Question:** What's the difference between fine-tuning and transfer learning? When would you use each approach? Explain domain adaptation.

??? success "View Answer"

    **Transfer Learning**: Using knowledge from a source task to improve learning on a target task.

    **Fine-Tuning**: Continuing training of a pre-trained model on new data, updating some or all weights.

    **Strategies:**

    | Approach | When to Use | Pros | Cons |
    |----------|-------------|------|------|
    | **Feature Extraction** | Small target dataset, similar domains | Fast, prevents overfitting | Limited adaptation |
    | **Fine-Tune Last Layers** | Medium dataset, related domains | Good balance | Need to choose layers |
    | **Fine-Tune All Layers** | Large dataset, different domains | Maximum adaptation | Risk overfitting, slow |

    **Domain Adaptation**: Transfer when source and target have different distributions.
    
    - **Supervised**: Labels in both domains
    - **Unsupervised**: Labels only in source
    - **Semi-supervised**: Few labels in target

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "Transfer learning: reuse learned features"
        - "Fine-tuning: continue training with lower LR"
        - "Feature extraction: freeze base, train head"
        - "Domain shift: source ≠ target distribution"
        - "Few-shot: adapt with minimal examples"
        - Practical considerations (dataset size, compute)

---

### Handling Missing Data - Google, Amazon, Microsoft Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Data Preprocessing`, `Imputation`, `Missing Values` | **Asked by:** Google, Amazon, Microsoft, Meta, Apple

**Question:** What are different strategies for handling missing data? When would you use each approach?

??? success "View Answer"

    **Types of Missing Data:**
    
    1. **MCAR (Missing Completely At Random)**: No pattern
    2. **MAR (Missing At Random)**: Related to observed data
    3. **MNAR (Missing Not At Random)**: Related to missing value itself

    **Strategies:**

    | Method | Use Case | Pros | Cons |
    |--------|----------|------|------|
    | **Deletion** | MCAR, <5% missing | Simple, no bias if MCAR | Loses information |
    | **Mean/Median** | Numerical, MCAR | Fast, preserves size | Reduces variance |
    | **Mode** | Categorical | Simple | May create bias |
    | **Forward/Backward Fill** | Time series | Preserves trends | Not for cross-sectional |
    | **Interpolation** | Time series, ordered | Smooth estimates | Assumes continuity |
    | **KNN Imputation** | Complex patterns | Captures relationships | Slow, sensitive to K |
    | **Model-Based** | MAR, complex | Most accurate | Computationally expensive |
    | **Multiple Imputation** | Uncertainty quantification | Accounts for uncertainty | Complex, slow |

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - Understanding of MCAR/MAR/MNAR
        - Multiple imputation strategies
        - "Check missingness pattern first"
        - "Mean for MCAR numerical"
        - "KNN/model-based for complex patterns"
        - "Consider creating 'is_missing' indicator"
        - Impact on downstream models

---

### Feature Selection Techniques - Google, Amazon, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Feature Selection`, `Dimensionality Reduction`, `Model Interpretability` | **Asked by:** Google, Amazon, Meta, Airbnb, LinkedIn

**Question:** Compare different feature selection methods: filter, wrapper, and embedded methods. When would you use each?

??? success "View Answer"

    **Feature Selection Methods:**

    **1. Filter Methods** (Independent of model):
    - Correlation coefficients
    - Chi-square test
    - Information gain
    - Variance threshold

    **2. Wrapper Methods** (Model-dependent):
    - Forward selection
    - Backward elimination
    - Recursive Feature Elimination (RFE)

    **3. Embedded Methods** (During training):
    - Lasso (L1 regularization)
    - Ridge (L2 regularization)
    - Tree-based feature importance
    - Elastic Net

    **Comparison:**

    | Method | Speed | Accuracy | Model-Agnostic |
    |--------|-------|----------|----------------|
    | **Filter** | Fast | Moderate | Yes |
    | **Wrapper** | Slow | High | No |
    | **Embedded** | Medium | High | No |

    **When to Use:**
    - **Filter**: Quick exploration, many features
    - **Wrapper**: Small feature sets, need optimal subset
    - **Embedded**: During model training, automatic

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "Filter: univariate, fast, model-agnostic"
        - "Wrapper: search subsets, slow, accurate"
        - "Embedded: during training, efficient"
        - "Lasso for automatic selection"
        - "RFE with cross-validation"
        - "Consider domain knowledge"
        - Trade-offs (speed vs accuracy)

---

### Time Series Forecasting - Uber, Airbnb, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Time Series`, `ARIMA`, `LSTM`, `Seasonality` | **Asked by:** Uber, Airbnb, Amazon, Lyft, DoorDash

**Question:** Explain approaches to time series forecasting. Compare statistical methods (ARIMA) with deep learning (LSTM). How do you handle seasonality?

??? success "View Answer"

    **Time Series Components:**
    
    1. **Trend**: Long-term direction
    2. **Seasonality**: Regular patterns
    3. **Cyclic**: Non-fixed frequency patterns
    4. **Residual**: Random noise

    **Methods:**

    **Statistical:**
    - **ARIMA**: AutoRegressive Integrated Moving Average
    - **SARIMA**: Seasonal ARIMA
    - **Prophet**: Facebook's additive model
    - **Exponential Smoothing**: Weighted averages

    **Deep Learning:**
    - **LSTM/GRU**: Capture long-term dependencies
    - **Temporal Convolutional Networks**: Dilated convolutions
    - **Transformer**: Attention for time series

    **Comparison:**

    | Aspect | ARIMA | LSTM |
    |--------|-------|------|
    | **Interpretability** | High | Low |
    | **Data Required** | Small | Large |
    | **Seasonality** | SARIMA extension | Learns automatically |
    | **Multiple Variables** | VAR extension | Native support |
    | **Non-linearity** | Limited | Excellent |

    **Handling Seasonality:**
    - Decomposition (additive/multiplicative)
    - Differencing
    - Seasonal dummy variables
    - Fourier features
    - SARIMA
    - Let deep learning learn it

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "ARIMA: linear, interpretable, small data"
        - "LSTM: non-linear, large data, multivariate"
        - "Stationarity check (ADF test)"
        - "Seasonality: decomposition, differencing"
        - "Walk-forward validation for evaluation"
        - "Exogenous variables for forecasting"
        - Prophet for business time series

---

### A/B Testing in ML - Meta, Uber, Netflix, Airbnb Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `A/B Testing`, `Experimentation`, `Statistical Significance`, `ML Systems` | **Asked by:** Meta, Uber, Netflix, Airbnb, Booking.com

**Question:** How do you A/B test machine learning models in production? What metrics would you track? How do you handle seasonality and confounding variables?

??? success "View Answer"

    **A/B Testing ML Models:**

    **Setup:**
    1. **Control**: Current model
    2. **Treatment**: New model
    3. **Random assignment**: Users → groups
    4. **Measure**: Business + ML metrics

    **Key Considerations:**

    **Metrics:**
    - **Business**: Revenue, engagement, retention
    - **ML**: Accuracy, latency, throughput
    - **User Experience**: CTR, time on site, conversion

    **Challenges:**
    
    1. **Sample Size**: Power analysis for detection
    2. **Duration**: Account for day-of-week, seasonality
    3. **Network Effects**: User interactions
    4. **Multiple Testing**: Bonferroni correction
    5. **Novelty Effect**: Users try new things initially

    **Statistical Tests:**
    - T-test: Continuous metrics
    - Chi-square: Categorical metrics
    - Mann-Whitney U: Non-parametric
    - Bootstrap: Confidence intervals

    **Advanced Techniques:**
    - **Multi-armed Bandits**: Dynamic allocation
    - **Sequential Testing**: Early stopping
    - **Stratification**: Control for confounders
    - **CUPED**: Variance reduction using pre-experiment data

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "Random assignment for causal inference"
        - "Business metrics + ML metrics"
        - "Statistical power and sample size"
        - "Run 1-2 weeks to capture seasonality"
        - "Check A/A test first (sanity check)"
        - "Guard rails: latency, error rates"
        - "Novelty effect and long-term impact"
        - Multi-armed bandits for exploration

---

### Model Monitoring & Drift Detection - Uber, Netflix, Airbnb Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `ML Ops`, `Model Monitoring`, `Data Drift`, `Concept Drift` | **Asked by:** Uber, Netflix, Airbnb, DoorDash, Instacart

**Question:** How do you monitor ML models in production? Explain data drift vs concept drift. What metrics and techniques would you use for drift detection?

??? success "View Answer"

    **Types of Drift:**

    **1. Data Drift (Covariate Shift):**
    - Input distribution changes: P(X) changes
    - Features evolve over time
    - Example: User demographics shift

    **2. Concept Drift:**
    - Relationship changes: P(Y|X) changes
    - Target definition evolves
    - Example: User preferences change

    **3. Label Drift:**
    - Output distribution changes: P(Y) changes
    - Class balance shifts

    **Detection Methods:**

    | Method | Type | Use Case |
    |--------|------|----------|
    | **KL Divergence** | Statistical | Distribution comparison |
    | **KS Test** | Statistical | Two-sample test |
    | **PSI (Population Stability Index)** | Statistical | Feature drift |
    | **Performance Monitoring** | Model-based | Concept drift |
    | **Feature Distribution** | Statistical | Data drift |

    **Monitoring Metrics:**
    
    **Model Performance:**
    - Accuracy, precision, recall
    - AUC, F1 score
    - Prediction distribution

    **Data Quality:**
    - Missing values
    - Out-of-range values
    - New categorical values
    - Feature correlations

    **System Metrics:**
    - Latency (p50, p95, p99)
    - Throughput (requests/sec)
    - Error rates
    - Resource utilization

    **Response Strategies:**
    1. **Retrain**: On recent data
    2. **Online Learning**: Continuous updates
    3. **Ensemble**: Combine old + new models
    4. **Rollback**: Revert to previous version
    5. **Alert**: Human intervention

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "Data drift: P(X) changes"
        - "Concept drift: P(Y|X) changes"
        - "KS test, PSI for detection"
        - "Monitor both performance and data"
        - "Retrain triggers: performance drop, time-based"
        - "Shadow mode: test new model safely"
        - "Logging: predictions, features, outcomes"
        - Feedback loop and continuous improvement

---

### Neural Architecture Search (NAS) - Google, DeepMind Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `AutoML`, `NAS`, `Hyperparameter Optimization`, `Meta-Learning` | **Asked by:** Google, DeepMind, Microsoft Research, Meta AI

**Question:** What is Neural Architecture Search (NAS)? Explain different NAS methods and their trade-offs.

??? success "View Answer"

    **Neural Architecture Search (NAS)**: Automated process of designing optimal neural network architectures.

    **Components:**
    
    1. **Search Space**: Possible architectures
    2. **Search Strategy**: How to explore space
    3. **Performance Estimation**: Evaluate candidates

    **NAS Methods:**

    **1. Reinforcement Learning-based:**
    - Controller RNN generates architectures
    - Train child network, use accuracy as reward
    - Very expensive (thousands of GPUs)

    **2. Evolutionary:**
    - Population of architectures
    - Mutation and crossover
    - Natural selection based on performance

    **3. Gradient-based (DARTS):**
    - Continuous relaxation of search space
    - Differentiate w.r.t. architecture
    - Much faster than RL/EA

    **4. One-Shot:**
    - Train super-network once
    - Sample sub-networks for evaluation
    - Very efficient

    **Comparison:**

    | Method | Speed | Quality | Cost |
    |--------|-------|---------|------|
    | **RL-based** | Slow | High | Very High |
    | **Evolutionary** | Slow | High | High |
    | **DARTS** | Fast | Good | Low |
    | **One-Shot** | Very Fast | Moderate | Very Low |

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "NAS: automate architecture design"
        - "Search space, strategy, evaluation"
        - "RL-based: expensive, high quality"
        - "DARTS: gradient-based, efficient"
        - "Transfer NAS: search once, use everywhere"
        - "Hardware-aware NAS: optimize for deployment"
        - Trade-offs (cost vs performance)

---

### Federated Learning - Google, Apple, Microsoft Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Distributed ML`, `Privacy`, `Federated Learning`, `Edge Computing` | **Asked by:** Google, Apple, Microsoft, NVIDIA, Meta

**Question:** Explain Federated Learning. How does it preserve privacy? What are the challenges and how do you address them?

??? success "View Answer"

    **Federated Learning**: Train models across decentralized devices without collecting raw data centrally.

    **Process:**
    
    1. Server sends model to clients
    2. Clients train locally on private data
    3. Clients send updates (not data) to server
    4. Server aggregates updates
    5. Repeat

    **Key Algorithm: Federated Averaging (FedAvg)**
    
    - Clients perform multiple SGD steps
    - Server averages client models
    - Reduces communication rounds

    **Privacy Preservation:**
    - **Data stays local**: Never leaves device
    - **Differential Privacy**: Add noise to updates
    - **Secure Aggregation**: Encrypted aggregation
    - **Homomorphic Encryption**: Compute on encrypted data

    **Challenges:**

    | Challenge | Description | Solution |
    |-----------|-------------|----------|
    | **Non-IID Data** | Heterogeneous distributions | FedProx, personalization |
    | **Communication Cost** | Slow networks | Compression, quantization |
    | **Systems Heterogeneity** | Different devices | Asynchronous FL |
    | **Privacy Leakage** | Model inversion | Differential privacy, secure aggregation |
    | **Stragglers** | Slow devices | Asynchronous updates, timeout |

    **Applications:**
    - **Mobile Keyboards**: Gboard, Apple Keyboard
    - **Healthcare**: Hospital collaboration without sharing patient data
    - **Finance**: Cross-bank fraud detection
    - **IoT**: Edge device learning

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "Decentralized: data stays on device"
        - "FedAvg: average model updates"
        - "Non-IID data challenge"
        - "Differential privacy for protection"
        - "Communication efficiency critical"
        - "Personalization: global + local models"
        - Trade-offs (privacy vs accuracy)

---

### Model Compression - Google, NVIDIA, Apple Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Model Compression`, `Pruning`, `Quantization`, `Distillation`, `Mobile ML` | **Asked by:** Google, NVIDIA, Apple, Qualcomm, Meta

**Question:** Explain different model compression techniques. How do you deploy large models on resource-constrained devices?

??? success "View Answer"

    **Model Compression Techniques:**

    **1. Quantization:**
    - Reduce precision (FP32 → INT8)
    - 4x smaller, 4x faster
    - Post-training or quantization-aware training

    **2. Pruning:**
    - Remove unnecessary weights/neurons
    - Magnitude-based, gradient-based
    - Structured vs unstructured

    **3. Knowledge Distillation:**
    - Teacher (large) → Student (small)
    - Student learns from teacher's soft labels
    - Retains performance with fewer parameters

    **4. Low-Rank Factorization:**
    - Decompose weight matrices
    - Reduce parameters
    - SVD, Tucker decomposition

    **Comparison:**

    | Technique | Size Reduction | Speed Up | Accuracy Loss |
    |-----------|----------------|----------|---------------|
    | **Quantization** | 4x | 2-4x | Minimal |
    | **Pruning** | 2-10x | 2-3x | Low-Medium |
    | **Distillation** | Variable | Variable | Low |
    | **Low-Rank** | 2-5x | 2-3x | Medium |

    **Combined Approach:**
    - Distillation + Quantization + Pruning
    - Can achieve 10-100x compression
    - With <1% accuracy loss

    **Deployment Strategies:**
    - **TensorFlow Lite**: Mobile/embedded
    - **ONNX Runtime**: Cross-platform
    - **TensorRT**: NVIDIA GPUs
    - **Core ML**: Apple devices

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "Quantization: INT8 for 4x compression"
        - "Pruning: remove low-magnitude weights"
        - "Distillation: student learns from teacher"
        - "Combined techniques for best results"
        - "Hardware-aware: target device constraints"
        - "Quantization-aware training beats post-training"
        - Trade-offs (size vs accuracy vs latency)

---

### Causal Inference in ML - LinkedIn, Airbnb, Uber Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Causal Inference`, `Treatment Effects`, `Confounding`, `Counterfactuals` | **Asked by:** LinkedIn, Airbnb, Uber, Microsoft, Meta

**Question:** Explain the difference between correlation and causation. How do you estimate causal effects in observational data? What are confounders?

??? success "View Answer"

    **Causation vs Correlation:**
    
    - **Correlation**: X and Y move together
    - **Causation**: X causes Y (interventional)

    **Causal Framework:**
    
    **Key Concepts:**
    
    1. **Treatment**: Intervention (e.g., ad exposure)
    2. **Outcome**: Effect (e.g., purchase)
    3. **Confounder**: Affects both treatment & outcome
    4. **Counterfactual**: What would have happened?

    **Estimation Methods:**

    **1. Randomized Controlled Trials (RCTs):**
    - Gold standard
    - Random assignment eliminates confounding
    - Not always feasible

    **2. Propensity Score Matching:**
    - Match treated/control with similar propensity
    - Balance observed confounders
    - Doesn't handle unobserved confounders

    **3. Instrumental Variables:**
    - Use instrument correlated with treatment
    - Not directly affecting outcome
    - Handles unobserved confounding

    **4. Difference-in-Differences:**
    - Compare before/after treatment
    - Across treated/control groups
    - Parallel trends assumption

    **5. Regression Discontinuity:**
    - Exploit cutoff for treatment assignment
    - Local randomization at threshold

    **Causal ML:**
    - **Uplift Modeling**: Predict treatment effect
    - **Meta-Learners**: T-learner, S-learner, X-learner
    - **CATE**: Conditional Average Treatment Effect
    - **Double ML**: Debiased machine learning

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "Correlation ≠ Causation"
        - "Confounders: affect both treatment and outcome"
        - "RCT: randomization eliminates confounding"
        - "Propensity scores: match similar units"
        - "Counterfactuals: what if treatment not given"
        - "Uplift modeling: predict treatment effect"
        - Applications (marketing, policy, healthcare)

---

### Recommendation Systems - Netflix, Spotify, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Recommender Systems`, `Collaborative Filtering`, `Matrix Factorization`, `Two-Tower` | **Asked by:** Netflix, Spotify, Amazon, YouTube, Pinterest

**Question:** Explain different recommendation system approaches. Compare collaborative filtering, content-based, and hybrid methods. How do you handle cold start?

??? success "View Answer"

    **Recommendation Approaches:**

    **1. Collaborative Filtering:**
    - **User-based**: Similar users like similar items
    - **Item-based**: Similar items liked by same users
    - **Matrix Factorization**: Latent factors (SVD, ALS)

    **2. Content-Based:**
    - Recommend based on item features
    - User profile from past interactions
    - No cold start for new users

    **3. Hybrid:**
    - Combine collaborative + content-based
    - Ensemble or integrated models

    **4. Deep Learning:**
    - **Two-Tower**: User/item embeddings
    - **Neural Collaborative Filtering**: Deep CF
    - **Sequence Models**: RNN, Transformer for sessions

    **Comparison:**

    | Method | Cold Start | Diversity | Scalability |
    |--------|------------|-----------|-------------|
    | **Collaborative** | Poor | Good | Medium |
    | **Content-Based** | Good | Poor | Good |
    | **Hybrid** | Good | Good | Medium |
    | **Deep Learning** | Medium | Good | Depends |

    **Cold Start Solutions:**
    
    **New Users:**
    - Onboarding questionnaire
    - Popular items
    - Demographic-based

    **New Items:**
    - Content-based features
    - Transfer from similar items
    - Explore-exploit (bandits)

    **Metrics:**
    - **Accuracy**: RMSE, MAE
    - **Ranking**: Precision@K, Recall@K, NDCG
    - **Business**: CTR, engagement, diversity
    - **Coverage**: % items recommended

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "Collaborative filtering: user-item patterns"
        - "Matrix factorization: latent factors"
        - "Content-based: item features"
        - "Cold start: no history for new users/items"
        - "Two-tower models: user/item embeddings"
        - "Explore-exploit for new items"
        - "Metrics: accuracy + diversity + coverage"
        - Production challenges (scalability, freshness)

---

### Imbalanced Learning - Stripe, PayPal, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Imbalanced Data`, `Class Imbalance`, `Sampling`, `Cost-Sensitive` | **Asked by:** Stripe, PayPal, Meta, Amazon, Google

**Question:** How do you handle highly imbalanced datasets (e.g., fraud detection)? Compare different resampling and algorithmic approaches.

??? success "View Answer"

    **Problem**: When one class significantly outnumbers others (e.g., 99.9% non-fraud, 0.1% fraud).

    **Challenges:**
    - Models biased toward majority class
    - Poor recall on minority class
    - Accuracy misleading (99% by predicting all negative)

    **Solutions:**

    **1. Resampling:**
    - **Oversampling**: Duplicate minority (SMOTE)
    - **Undersampling**: Remove majority (Random, Tomek links)
    - **Hybrid**: Combine both (SMOTE + ENN)

    **2. Algorithmic:**
    - **Class Weights**: Penalize errors differently
    - **Threshold Tuning**: Adjust decision boundary
    - **Ensemble**: Balanced bagging/boosting
    - **Anomaly Detection**: Treat as outlier detection

    **3. Evaluation Metrics:**
    - **Precision-Recall**: Better than accuracy
    - **F1-Score**: Harmonic mean
    - **AUC-ROC**: Threshold-independent
    - **PR-AUC**: Better for imbalanced
    - **Matthews Correlation Coefficient**: Balanced measure

    **Comparison:**

    | Approach | Pros | Cons |
    |----------|------|------|
    | **SMOTE** | Creates synthetic samples | May create noise |
    | **Undersampling** | Fast, reduces majority | Loses information |
    | **Class Weights** | No data modification | Hyperparameter tuning |
    | **Anomaly Detection** | Unsupervised | Needs normal data |

    **Best Practices:**
    - Use stratified splitting
    - Focus on PR-AUC over accuracy
    - Combine multiple techniques
    - Cost-sensitive learning (business cost of errors)

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "Class imbalance: majority drowns minority"
        - "SMOTE: synthetic minority samples"
        - "Class weights: penalize errors differently"
        - "Accuracy misleading, use PR-AUC"
        - "Threshold tuning: optimize for business metric"
        - "Anomaly detection for extreme imbalance"
        - Real-world context (fraud 1:1000, rare disease 1:10000)

---

### Embedding Techniques - Google, Meta, LinkedIn Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Embeddings`, `Word2Vec`, `Entity Embeddings`, `Representation Learning` | **Asked by:** Google, Meta, LinkedIn, Pinterest, Twitter

**Question:** Explain different embedding techniques. How do Word2Vec, GloVe, and contextual embeddings (BERT) differ? When would you use entity embeddings for categorical variables?

??? success "View Answer"

    **Embeddings**: Dense vector representations that capture semantic relationships.

    **Word Embeddings:**

    **1. Word2Vec:**
    - **Skip-gram**: Predict context from word
    - **CBOW**: Predict word from context
    - Static embeddings (one vector per word)

    **2. GloVe:**
    - Global word co-occurrence statistics
    - Matrix factorization approach
    - Static embeddings

    **3. Contextual (BERT, GPT):**
    - Different vectors for same word in different contexts
    - "bank" (river) vs "bank" (financial)
    - Captures polysemy

    **Comparison:**

    | Method | Contextual | Training | Use Case |
    |--------|------------|----------|----------|
    | **Word2Vec** | No | Local context | Fast, lightweight |
    | **GloVe** | No | Global stats | Good for similarity |
    | **BERT** | Yes | Transformer | State-of-the-art |

    **Entity Embeddings:**
    - For categorical variables (user_id, product_id)
    - Learn dense representations
    - Capture relationships (similar users, substitute products)
    - Better than one-hot encoding for high cardinality

    **Benefits:**
    - Dimensionality reduction
    - Similarity computation
    - Transfer learning
    - Visualization

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "Embeddings: dense vector representations"
        - "Word2Vec: predict context or word"
        - "BERT: contextual, different vectors per context"
        - "Entity embeddings: for categorical features"
        - "Cosine similarity for semantic search"
        - "Pre-trained vs task-specific embeddings"
        - Applications (search, recommendations, clustering)

---

### Bias and Fairness in ML - Google, Microsoft, LinkedIn Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `ML Ethics`, `Bias`, `Fairness`, `Responsible AI` | **Asked by:** Google, Microsoft, LinkedIn, Meta, IBM

**Question:** What are different types of bias in ML systems? How do you measure and mitigate bias? Explain fairness metrics.

??? success "View Answer"

    **Types of Bias:**

    **1. Data Bias:**
    - **Historical**: Past discrimination in training data
    - **Sampling**: Non-representative samples
    - **Label**: Biased human annotations

    **2. Algorithmic Bias:**
    - **Representation**: Model amplifies data bias
    - **Aggregation**: One model for heterogeneous groups
    - **Evaluation**: Biased metrics

    **3. Deployment Bias:**
    - **Feedback Loops**: Predictions affect future data
    - **User Interaction**: Different groups use system differently

    **Fairness Definitions:**

    | Metric | Description | When to Use |
    |--------|-------------|-------------|
    | **Demographic Parity** | Equal positive rate across groups | Equal opportunity contexts |
    | **Equal Opportunity** | Equal TPR across groups | Lending, hiring |
    | **Equalized Odds** | Equal TPR and FPR | Criminal justice |
    | **Predictive Parity** | Equal precision across groups | Resource allocation |

    **Note**: Cannot satisfy all fairness criteria simultaneously (impossibility theorems).

    **Mitigation Strategies:**

    **Pre-processing:**
    - Collect representative data
    - Balance datasets
    - Remove sensitive attributes (with care)

    **In-processing:**
    - Fairness constraints during training
    - Adversarial debiasing
    - Fair representation learning

    **Post-processing:**
    - Adjust decision thresholds per group
    - Calibration
    - Reject option

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "Bias types: data, algorithmic, deployment"
        - "Fairness metrics: demographic parity, equal opportunity"
        - "Trade-offs: fairness vs accuracy"
        - "Can't satisfy all fairness definitions"
        - "Mitigation: pre/in/post-processing"
        - "Feedback loops amplify bias"
        - "Transparency and explainability"
        - Real-world consequences

---

### Multi-Task Learning - Google DeepMind, Meta AI Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Multi-Task Learning`, `Transfer Learning`, `Hard/Soft Sharing` | **Asked by:** Google DeepMind, Meta AI, Microsoft Research, NVIDIA

**Question:** What is multi-task learning? Explain hard vs soft parameter sharing. When does MTL help vs hurt?

??? success "View Answer"

    **Multi-Task Learning (MTL)**: Train one model on multiple related tasks simultaneously.

    **Benefits:**
    - **Regularization**: Shared representations prevent overfitting
    - **Data Efficiency**: Learn from multiple signals
    - **Transfer**: Knowledge transfer across tasks
    - **Faster Learning**: Auxiliary tasks help main task

    **Parameter Sharing:**

    **Hard Sharing:**
    - Shared hidden layers
    - Task-specific output layers
    - Most common approach

    **Soft Sharing:**
    - Separate models per task
    - Encourage similarity via regularization
    - More flexible but complex

    **When MTL Helps:**
    - Related tasks (sentiment, topic, intent)
    - Limited data per task
    - Shared underlying structure
    - Auxiliary tasks provide useful signal

    **When MTL Hurts:**
    - Unrelated tasks (negative transfer)
    - One task dominates training
    - Tasks require different representations
    - Task conflicts

    **Challenges:**
    - **Task Balancing**: Equal contribution to loss
    - **Negative Transfer**: Tasks hurt each other
    - **Architecture Design**: How much to share?
    - **Optimization**: Different convergence rates

    **Solutions:**
    - **Task Weighting**: Learn task weights
    - **Gradients**: GradNorm, PCGrad (project conflicting gradients)
    - **Architecture Search**: Learn sharing structure
    - **Uncertainty Weighting**: Weight by task uncertainty

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "MTL: train multiple tasks together"
        - "Hard sharing: shared layers"
        - "Soft sharing: separate models, regularized"
        - "Benefits: regularization, data efficiency"
        - "Negative transfer: tasks hurt each other"
        - "Task weighting: balance contributions"
        - "Auxiliary tasks can improve main task"
        - Applications (NLP: NER + POS + parsing)

---

### Adversarial Robustness - Google, OpenAI, DeepMind Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Adversarial Examples`, `Robustness`, `Security`, `Adversarial Training` | **Asked by:** Google, OpenAI, DeepMind, Microsoft, Meta

**Question:** What are adversarial examples? How do you make models robust to adversarial attacks? Explain adversarial training.

??? success "View Answer"

    **Adversarial Examples**: Inputs with small perturbations that cause misclassification.

    $$x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}(x, y))$$

    **Types of Attacks:**

    **White-Box** (Full model access):
    - **FGSM**: Fast Gradient Sign Method
    - **PGD**: Projected Gradient Descent
    - **C&W**: Carlini & Wagner attack

    **Black-Box** (No model access):
    - **Transfer attacks**: Use substitute model
    - **Query-based**: Test inputs iteratively

    **Defense Strategies:**

    **1. Adversarial Training:**
    - Include adversarial examples in training
    - Most effective but expensive

    **2. Defensive Distillation:**
    - Train student on soft labels from teacher
    - Smooths decision boundaries

    **3. Input Transformations:**
    - Compression, denoising
    - Can be circumvented

    **4. Detection:**
    - Identify adversarial inputs
    - Reject or handle specially

    **5. Certified Defense:**
    - Mathematical guarantees of robustness
    - Randomized smoothing

    **Trade-offs:**
    - Robust accuracy vs standard accuracy
    - Computational cost
    - Robustness vs interpretability

    **Applications:**
    - **Autonomous Vehicles**: Stop sign attacks
    - **Face Recognition**: Evade detection
    - **Malware Detection**: Adversarial malware
    - **Medical Imaging**: Misdiagnosis

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "Adversarial examples: imperceptible perturbations"
        - "FGSM: gradient-based attack"
        - "Adversarial training: train on adversarial examples"
        - "Robust accuracy vs standard accuracy trade-off"
        - "White-box vs black-box attacks"
        - "Certified robustness: mathematical guarantees"
        - Security implications in production

---

### Self-Supervised Learning - Meta AI, Google Research Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Self-Supervised`, `Contrastive Learning`, `Pre-training`, `SimCLR` | **Asked by:** Meta AI, Google Research, DeepMind, OpenAI

**Question:** What is self-supervised learning? Explain contrastive learning methods like SimCLR. How does it differ from supervised and unsupervised learning?

??? success "View Answer"

    **Self-Supervised Learning**: Learn representations from unlabeled data by creating pretext tasks.

    **Key Idea**: Generate supervision signal from data itself.

    **Approaches:**

    **1. Contrastive Learning:**
    - Pull similar samples together
    - Push dissimilar samples apart
    - **SimCLR, MoCo, CLIP**

    **2. Predictive:**
    - Predict missing parts
    - **BERT (masked LM), GPT (next token)**

    **3. Generative:**
    - Reconstruct input
    - **Autoencoders, MAE**

    **SimCLR (Simple Contrastive Learning):**
    
    1. Augment same image twice (positive pair)
    2. Encode to embeddings
    3. Maximize agreement between positive pairs
    4. Minimize agreement with negative pairs

    **Loss (InfoNCE):**
    
    $$\mathcal{L} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{k \neq i} \exp(\text{sim}(z_i, z_k)/\tau)}$$

    **Comparison:**

    | Paradigm | Labels | Pre-training | Fine-tuning |
    |----------|--------|--------------|-------------|
    | **Supervised** | Required | With labels | Optional |
    | **Unsupervised** | None | Clustering, PCA | N/A |
    | **Self-Supervised** | None | Pretext tasks | On downstream |

    **Benefits:**
    - Leverage unlabeled data (abundant)
    - Better transfer learning
    - Robust representations
    - Less label dependence

    **Applications:**
    - **Computer Vision**: ImageNet pre-training
    - **NLP**: BERT, GPT pre-training
    - **Multimodal**: CLIP (image-text)
    - **Speech**: wav2vec, HuBERT

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "Self-supervised: create tasks from data"
        - "Contrastive: similar together, dissimilar apart"
        - "SimCLR: augmentation + contrastive loss"
        - "Leverage massive unlabeled data"
        - "Pre-train then fine-tune paradigm"
        - vs. "Supervised: needs labels"
        - "BERT: masked language modeling"
        - "CLIP: align images and text"

---

### Few-Shot Learning - Meta AI, DeepMind, OpenAI Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Few-Shot`, `Meta-Learning`, `Prototypical Networks`, `In-Context Learning` | **Asked by:** Meta AI, DeepMind, OpenAI, Google Research

**Question:** What is few-shot learning? Compare different approaches: metric learning, meta-learning, and large language model in-context learning.

??? success "View Answer"

    **Few-Shot Learning**: Learn new concepts from very few examples (1-5).

    **Problem**: Traditional ML needs 100s-1000s of examples per class.

    **Approaches:**

    **1. Metric Learning:**
    - Learn similarity metric
    - Classify based on distance to prototypes
    - **Siamese Networks, Prototypical Networks**

    **2. Meta-Learning:**
    - Learn to learn across tasks
    - **MAML, Reptile**
    - Good initialization for fast adaptation

    **3. Data Augmentation:**
    - Generate synthetic examples
    - Hallucination from base classes

    **4. LLM In-Context Learning:**
    - Provide examples in prompt
    - No parameter updates
    - **GPT-3, GPT-4**

    **Comparison:**

    | Method | Training | Adaptation | Examples Needed |
    |--------|----------|------------|-----------------|
    | **Prototypical** | Episode-based | Compute prototypes | 1-5 per class |
    | **MAML** | Meta-train | Few gradient steps | 5-10 per task |
    | **In-Context** | Pre-training | Add to prompt | 0-5 per class |

    **Evaluation:**
    - **N-way K-shot**: N classes, K examples each
    - **Support Set**: Training examples
    - **Query Set**: Test examples

    **Applications:**
    - **Drug Discovery**: Predict properties of new molecules
    - **Robotics**: Adapt to new objects quickly
    - **Personalization**: User-specific models
    - **Rare Disease**: Classify with few patients

    **Interview Insights:**
    
    !!! tip "Interviewer's Insight"
        
        - "Few-shot: learn from 1-5 examples"
        - "Prototypical: classify by distance to prototypes"
        - "MAML: meta-learn good initialization"
        - "In-context: LLMs with prompt examples"
        - "N-way K-shot evaluation"
        - "Support set vs query set"
        - "Transfer from base classes"
        - Applications in low-data scenarios

---

## Questions asked in Slack interview
- Bias-Variance Tradeoff  
- Cross-Validation  
- Feature Engineering  
- Transfer Learning  

## Questions asked in Airbnb interview
- Bias-Variance Tradeoff  
- Hyperparameter Tuning  
- Transfer Learning  
- Model Interpretability: SHAP and LIME  

---

*Note:* The practice links are curated from reputable sources such as Machine Learning Mastery, Towards Data Science, Analytics Vidhya, and Scikit-learn. You can update/contribute to these lists or add new ones as more resources become available.

---
