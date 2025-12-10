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

## Quick Reference: 100 Interview Questions

| Sno | Question Title | Practice Links | Companies Asking | Difficulty | Topics |
|-----|----------------|----------------|------------------|------------|--------|
| 1 | Bias-Variance Tradeoff | [Machine Learning Mastery](https://machinelearningmastery.com/gentle-introduction-to-the-bias-variance-trade-off/) | Google, Facebook, Amazon | Medium | Model Evaluation, Generalization |
| 2 | Regularization Techniques (L1, L2) | [Machine Learning Mastery](https://machinelearningmastery.com/introduction-to-regularization-to-reduce-overfitting-and-improve-generalization/) | Google, Amazon, Microsoft | Medium | Overfitting, Generalization |
| 3 | Cross-Validation | [Scikit-Learn Cross Validation](https://scikit-learn.org/stable/modules/cross_validation.html) | Google, Facebook, Amazon | Easy | Model Evaluation |
| 4 | Overfitting and Underfitting | [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2017/09/overfitting-underfitting-machine-learning/) | Google, Amazon, Facebook | Easy | Model Evaluation |
| 5 | Gradient Descent | [Towards Data Science](https://towardsdatascience.com/gradient-descent-101-402f4b3a33f3) | Google, Amazon, Microsoft | Medium | Optimization |
| 6 | Supervised vs Unsupervised Learning | [IBM Cloud Learn](https://www.ibm.com/cloud/learn/supervised-vs-unsupervised-learning) | Google, Facebook, Amazon | Easy | ML Basics |
| 7 | Classification vs Regression | [Towards Data Science](https://towardsdatascience.com/classification-vs-regression-a-tutorial-4a2d123b9288) | Google, Amazon, Facebook | Easy | ML Basics |
| 8 | Evaluation Metrics: Precision, Recall, F1-score | [Towards Data Science](https://towardsdatascience.com/accuracy-precision-recall-and-f1-score-5f728d4a57f0) | Google, Amazon, Microsoft | Medium | Model Evaluation |
| 9 | Decision Trees | [Machine Learning Mastery](https://machinelearningmastery.com/decision-trees-in-machine-learning/) | Google, Amazon, Facebook | Medium | Tree-based Models |
| 10 | Ensemble Learning: Bagging and Boosting | [Towards Data Science](https://towardsdatascience.com/ensemble-learning-bagging-and-boosting-9e17ce8b0b64) | Google, Amazon, Microsoft | Medium | Ensemble Methods |
| 11 | Random Forest | [Towards Data Science](https://towardsdatascience.com/the-random-forest-algorithm-d457d499ffcd) | Google, Amazon, Facebook | Medium | Ensemble, Decision Trees |
| 12 | Support Vector Machines (SVM) | [Machine Learning Mastery](https://machinelearningmastery.com/support-vector-machines-for-machine-learning/) | Google, Facebook, Amazon | Hard | Classification, Kernel Methods |
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
