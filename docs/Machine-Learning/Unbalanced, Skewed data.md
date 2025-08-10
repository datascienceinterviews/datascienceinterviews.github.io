---
title: Unbalanced and Skewed Data
description: Comprehensive guide to handling Unbalanced and Skewed Data with sampling techniques, evaluation metrics, and practical solutions.
comments: true
---

# =Ø Unbalanced and Skewed Data

Unbalanced and Skewed Data are common challenges in machine learning where the distribution of classes or feature values is highly imbalanced, leading to biased models that favor majority classes or specific value ranges.

**Resources:** [Imbalanced-learn Library](https://imbalanced-learn.org/stable/) | [SMOTE Paper](https://arxiv.org/abs/1106.1813) | [Cost-Sensitive Learning Survey](https://link.springer.com/article/10.1007/s10994-013-5425-6)

##  Summary

**Unbalanced Data** (Class Imbalance) occurs when the distribution of target classes is significantly unequal. **Skewed Data** refers to non-normal distributions in features where most values are concentrated at one end of the range.

**Key Characteristics:**

**Unbalanced Data:**
- Minority classes have significantly fewer samples than majority classes
- Common in fraud detection, medical diagnosis, rare event prediction
- Standard algorithms tend to favor majority class
- Accuracy can be misleading as a performance metric

**Skewed Data:**
- Feature distributions are asymmetric (left-skewed or right-skewed)
- Mean and median differ significantly
- Can cause issues with algorithms assuming normal distributions
- May contain outliers that affect model performance

**Applications:**
- Fraud detection (few fraud cases vs. many normal transactions)
- Medical diagnosis (rare diseases vs. healthy patients)
- Email spam detection (spam vs. legitimate emails)
- Quality control (defective vs. normal products)
- Customer churn prediction (churned vs. retained customers)
- Anomaly detection in cybersecurity

**Related Concepts:**
- **Sampling Techniques**: Methods to balance class distributions
- **Cost-Sensitive Learning**: Assigning different costs to classification errors
- **Ensemble Methods**: Combining multiple models to improve minority class performance
- **Evaluation Metrics**: Precision, Recall, F1-score, AUC-ROC for imbalanced datasets

## >à Intuition

### How Imbalanced Data Affects Learning

Imagine training a model to detect rare diseases where only 1% of patients have the disease. A naive model could achieve 99% accuracy by always predicting "no disease" - but this would be completely useless for actually identifying sick patients. The model learns to favor the majority class because:

1. **Training Bias**: More examples of majority class dominate the learning process
2. **Decision Boundary**: Gets pushed toward minority class regions
3. **Loss Function**: Optimizes for overall accuracy, not class-specific performance
4. **Gradient Updates**: Majority class errors have more influence on weight updates

### Mathematical Foundation

#### 1. Class Imbalance Ratio

For binary classification with classes 0 and 1:
$$\text{Imbalance Ratio} = \frac{\text{Number of samples in minority class}}{\text{Number of samples in majority class}}$$

Severe imbalance: IR < 0.1, Moderate imbalance: 0.1 d IR d 0.5

#### 2. Skewness Measure

For a distribution with values $x_1, x_2, ..., x_n$:
$$\text{Skewness} = \frac{E[(X - \mu)^3]}{\sigma^3} = \frac{\frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^3}{s^3}$$

Where:
- Skewness > 0: Right-skewed (long tail on right)
- Skewness < 0: Left-skewed (long tail on left)  
- Skewness H 0: Approximately symmetric

#### 3. Cost-Sensitive Learning

Modify the loss function to penalize minority class errors more heavily:
$$\text{Cost-Sensitive Loss} = \sum_{i=1}^{n} C(y_i) \cdot L(y_i, \hat{y_i})$$

Where $C(y_i)$ is the cost matrix assigning higher costs to minority class misclassifications.

#### 4. SMOTE Algorithm

Synthetic Minority Oversampling Technique creates new minority samples:
$$x_{new} = x_i + \lambda \cdot (x_{neighbor} - x_i)$$

Where $\lambda \in [0,1]$ is a random number and $x_{neighbor}$ is a randomly chosen k-nearest neighbor.

## =" Implementation using Libraries

### Scikit-learn and Imbalanced-learn Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           precision_recall_curve, roc_auc_score, roc_curve,
                           precision_score, recall_score, f1_score)
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Generate imbalanced dataset
def create_imbalanced_dataset(n_samples=10000, weights=[0.99, 0.01], random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=1,
        weights=weights,
        random_state=random_state
    )
    return X, y

# Create dataset
X, y = create_imbalanced_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    stratify=y, random_state=42)

print("Original Dataset Distribution:")
print(f"Total samples: {len(y)}")
print(f"Class distribution: {Counter(y)}")
print(f"Imbalance ratio: {Counter(y)[1] / Counter(y)[0]:.3f}")

# Helper function for evaluation
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Comprehensive model evaluation for imbalanced datasets"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    print(f"\n{model_name} Results:")
    print("-" * 50)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    if y_proba is not None:
        auc_score = roc_auc_score(y_test, y_proba)
        print(f"AUC-ROC Score: {auc_score:.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")
    
    return {
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None
    }

# 1. Baseline model without handling imbalance
print("="*60)
print("1. BASELINE MODEL (No Imbalance Handling)")
print("="*60)

baseline_model = LogisticRegression(random_state=42)
baseline_results = evaluate_model(baseline_model, X_train, X_test, y_train, y_test, 
                                "Baseline Logistic Regression")
```

### Sampling Techniques

```python
print("\n" + "="*60)
print("2. SAMPLING TECHNIQUES")
print("="*60)

# Dictionary to store all resampling techniques
sampling_techniques = {
    'Random Over-sampling': RandomOverSampler(random_state=42),
    'SMOTE': SMOTE(random_state=42),
    'ADASYN': ADASYN(random_state=42),
    'Random Under-sampling': RandomUnderSampler(random_state=42),
    'Edited Nearest Neighbours': EditedNearestNeighbours(),
    'SMOTE + ENN': SMOTEENN(random_state=42),
    'SMOTE + Tomek': SMOTETomek(random_state=42)
}

sampling_results = {}

for name, sampler in sampling_techniques.items():
    print(f"\n{name}:")
    print("-" * 40)
    
    try:
        # Apply sampling
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        
        print(f"Original distribution: {Counter(y_train)}")
        print(f"Resampled distribution: {Counter(y_resampled)}")
        print(f"Resampling ratio: {len(y_resampled) / len(y_train):.2f}")
        
        # Train and evaluate model
        model = LogisticRegression(random_state=42)
        results = evaluate_model(model, X_resampled, X_test, y_resampled, y_test, name)
        sampling_results[name] = results
        
    except Exception as e:
        print(f"Error with {name}: {str(e)}")
        sampling_results[name] = {'precision': 0, 'recall': 0, 'f1': 0, 'auc': 0}
```

### Cost-Sensitive Learning

```python
print("\n" + "="*60)
print("3. COST-SENSITIVE LEARNING")
print("="*60)

# Calculate class weights
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

print(f"Computed class weights: {class_weight_dict}")

# Models with different class weights
cost_sensitive_models = {
    'Balanced Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42),
    'Balanced Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
    'Balanced RF (imblearn)': BalancedRandomForestClassifier(random_state=42)
}

cost_sensitive_results = {}

for name, model in cost_sensitive_models.items():
    results = evaluate_model(model, X_train, X_test, y_train, y_test, name)
    cost_sensitive_results[name] = results
```

### Advanced Ensemble Methods

```python
print("\n" + "="*60)
print("4. ENSEMBLE METHODS FOR IMBALANCED DATA")
print("="*60)

from imblearn.ensemble import BalancedBaggingClassifier, EasyEnsembleClassifier, RUSBoostClassifier

ensemble_models = {
    'Balanced Bagging': BalancedBaggingClassifier(random_state=42),
    'Easy Ensemble': EasyEnsembleClassifier(random_state=42),
    'RUSBoost': RUSBoostClassifier(random_state=42)
}

ensemble_results = {}

for name, model in ensemble_models.items():
    results = evaluate_model(model, X_train, X_test, y_train, y_test, name)
    ensemble_results[name] = results
```

### Comprehensive Results Comparison

```python
print("\n" + "="*60)
print("5. COMPREHENSIVE RESULTS COMPARISON")
print("="*60)

# Combine all results
all_results = {
    'Baseline': baseline_results,
    **sampling_results,
    **cost_sensitive_results,
    **ensemble_results
}

# Create comparison DataFrame
results_df = pd.DataFrame(all_results).T
results_df = results_df.round(3)

print("Performance Comparison:")
print(results_df.sort_values('f1', ascending=False))

# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

metrics = ['precision', 'recall', 'f1', 'auc']
titles = ['Precision', 'Recall', 'F1-Score', 'AUC-ROC']

for i, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[i//2, i%2]
    
    # Filter out None values for AUC
    plot_data = results_df[metric].dropna()
    
    plot_data.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title(f'{title} Comparison')
    ax.set_ylabel(title)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Find best performing models
print("\nBest Models by Metric:")
print("-" * 30)
for metric in ['precision', 'recall', 'f1', 'auc']:
    if metric in results_df.columns:
        best_model = results_df[metric].idxmax()
        best_score = results_df[metric].max()
        print(f"{metric.upper():10s}: {best_model} ({best_score:.3f})")
```

## ™ From Scratch Implementation

### SMOTE Implementation from Scratch

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

class SMOTEFromScratch:
    """
    Synthetic Minority Oversampling Technique (SMOTE) implementation from scratch
    """
    
    def __init__(self, k_neighbors=5, random_state=42):
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        np.random.seed(random_state)
    
    def fit_resample(self, X, y):
        """
        Apply SMOTE to balance the dataset
        """
        # Separate majority and minority classes
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError("SMOTE currently supports only binary classification")
        
        class_counts = [(cls, np.sum(y == cls)) for cls in unique_classes]
        class_counts.sort(key=lambda x: x[1])  # Sort by count
        
        minority_class, minority_count = class_counts[0]
        majority_class, majority_count = class_counts[1]
        
        print(f"Original distribution - Minority class {minority_class}: {minority_count}, "
              f"Majority class {majority_class}: {majority_count}")
        
        # Extract minority class samples
        minority_mask = (y == minority_class)
        X_minority = X[minority_mask]
        
        # Calculate number of synthetic samples needed
        n_synthetic = majority_count - minority_count
        
        # Generate synthetic samples
        X_synthetic = self._generate_synthetic_samples(X_minority, n_synthetic)
        y_synthetic = np.full(n_synthetic, minority_class)
        
        # Combine original and synthetic data
        X_resampled = np.vstack([X, X_synthetic])
        y_resampled = np.hstack([y, y_synthetic])
        
        print(f"Generated {n_synthetic} synthetic samples")
        print(f"New distribution - Class {minority_class}: {majority_count}, "
              f"Class {majority_class}: {majority_count}")
        
        return X_resampled, y_resampled
    
    def _generate_synthetic_samples(self, X_minority, n_synthetic):
        """
        Generate synthetic samples using SMOTE algorithm
        """
        n_minority = X_minority.shape[0]
        n_features = X_minority.shape[1]
        
        # Find k-nearest neighbors for each minority sample
        nn = NearestNeighbors(n_neighbors=self.k_neighbors + 1)  # +1 to exclude self
        nn.fit(X_minority)
        
        synthetic_samples = []
        
        for _ in range(n_synthetic):
            # Randomly select a minority sample
            random_idx = np.random.randint(0, n_minority)
            sample = X_minority[random_idx]
            
            # Find k-nearest neighbors
            distances, indices = nn.kneighbors([sample])
            neighbor_indices = indices[0][1:]  # Exclude the sample itself
            
            # Randomly select one of the k-nearest neighbors
            neighbor_idx = np.random.choice(neighbor_indices)
            neighbor = X_minority[neighbor_idx]
            
            # Generate synthetic sample along the line between sample and neighbor
            lambda_val = np.random.random()  # Random value between 0 and 1
            synthetic_sample = sample + lambda_val * (neighbor - sample)
            
            synthetic_samples.append(synthetic_sample)
        
        return np.array(synthetic_samples)
    
    def plot_samples(self, X_original, y_original, X_resampled, y_resampled, 
                    feature_idx=[0, 1]):
        """
        Visualize original vs resampled data (for 2D visualization)
        """
        if X_original.shape[1] < 2:
            print("Need at least 2 features for 2D visualization")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot original data
        for class_val in np.unique(y_original):
            mask = y_original == class_val
            ax1.scatter(X_original[mask, feature_idx[0]], 
                       X_original[mask, feature_idx[1]], 
                       label=f'Class {class_val}', alpha=0.7)
        ax1.set_title('Original Data')
        ax1.set_xlabel(f'Feature {feature_idx[0]}')
        ax1.set_ylabel(f'Feature {feature_idx[1]}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot resampled data
        for class_val in np.unique(y_resampled):
            mask = y_resampled == class_val
            ax2.scatter(X_resampled[mask, feature_idx[0]], 
                       X_resampled[mask, feature_idx[1]], 
                       label=f'Class {class_val}', alpha=0.7)
        ax2.set_title('After SMOTE')
        ax2.set_xlabel(f'Feature {feature_idx[0]}')
        ax2.set_ylabel(f'Feature {feature_idx[1]}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Custom Cost-Sensitive Logistic Regression
class CostSensitiveLogisticRegression:
    """
    Logistic Regression with custom cost-sensitive learning
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, cost_ratio=1.0):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.cost_ratio = cost_ratio  # Cost ratio for minority class
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def _sigmoid(self, z):
        """Sigmoid activation function with numerical stability"""
        z = np.clip(z, -250, 250)  # Prevent overflow
        return 1 / (1 + np.exp(-z))
    
    def _compute_cost(self, y_true, y_pred):
        """Compute cost-sensitive logistic loss"""
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Standard logistic loss
        standard_loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        # Apply cost weighting
        cost_weights = np.where(y_true == 1, self.cost_ratio, 1.0)
        weighted_loss = cost_weights * standard_loss
        
        return np.mean(weighted_loss)
    
    def fit(self, X, y):
        """Train the cost-sensitive logistic regression model"""
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.max_iterations):
            # Forward pass
            z = X @ self.weights + self.bias
            y_pred = self._sigmoid(z)
            
            # Compute cost
            cost = self._compute_cost(y, y_pred)
            self.cost_history.append(cost)
            
            # Compute gradients with cost weighting
            cost_weights = np.where(y == 1, self.cost_ratio, 1.0)
            weighted_errors = cost_weights * (y_pred - y)
            
            dw = (1/n_samples) * X.T @ weighted_errors
            db = (1/n_samples) * np.sum(weighted_errors)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Print progress
            if i % 100 == 0:
                print(f"Iteration {i}, Cost: {cost:.4f}")
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        z = X @ self.weights + self.bias
        return self._sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """Make binary predictions"""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def plot_cost_history(self):
        """Plot the cost function over iterations"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.title('Cost-Sensitive Logistic Regression - Training Cost')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.grid(True, alpha=0.3)
        plt.show()

# Example usage of custom implementations
if __name__ == "__main__":
    print("\n" + "="*60)
    print("CUSTOM IMPLEMENTATIONS EXAMPLE")
    print("="*60)
    
    # Generate imbalanced dataset for testing
    from sklearn.datasets import make_classification
    X_demo, y_demo = make_classification(n_samples=1000, n_features=2, 
                                        n_redundant=0, n_informative=2,
                                        n_clusters_per_class=1, 
                                        weights=[0.9, 0.1], random_state=42)
    
    X_train_demo, X_test_demo, y_train_demo, y_test_demo = train_test_split(
        X_demo, y_demo, test_size=0.2, stratify=y_demo, random_state=42
    )
    
    print(f"Demo dataset - Training distribution: {Counter(y_train_demo)}")
    
    # Test custom SMOTE
    print("\n1. Testing Custom SMOTE Implementation:")
    print("-" * 40)
    
    smote_custom = SMOTEFromScratch(k_neighbors=5, random_state=42)
    X_smote, y_smote = smote_custom.fit_resample(X_train_demo, y_train_demo)
    
    # Visualize SMOTE results
    smote_custom.plot_samples(X_train_demo, y_train_demo, X_smote, y_smote)
    
    # Test custom cost-sensitive logistic regression
    print("\n2. Testing Custom Cost-Sensitive Logistic Regression:")
    print("-" * 50)
    
    # Calculate appropriate cost ratio
    minority_count = np.sum(y_train_demo == 1)
    majority_count = np.sum(y_train_demo == 0)
    cost_ratio = majority_count / minority_count
    
    print(f"Calculated cost ratio: {cost_ratio:.2f}")
    
    # Train custom model
    custom_model = CostSensitiveLogisticRegression(
        learning_rate=0.1, 
        max_iterations=1000, 
        cost_ratio=cost_ratio
    )
    
    custom_model.fit(X_train_demo, y_train_demo)
    
    # Make predictions
    y_pred_custom = custom_model.predict(X_test_demo)
    y_proba_custom = custom_model.predict_proba(X_test_demo)
    
    # Evaluate custom model
    print("\nCustom Model Results:")
    print("Classification Report:")
    from sklearn.metrics import classification_report, roc_auc_score
    print(classification_report(y_test_demo, y_pred_custom))
    print(f"AUC-ROC: {roc_auc_score(y_test_demo, y_proba_custom):.3f}")
    
    # Plot cost history
    custom_model.plot_cost_history()
```

##   Assumptions and Limitations

### Assumptions for Imbalanced Data Techniques

**SMOTE Assumptions:**
- Minority class samples form meaningful clusters
- Linear interpolation between samples creates realistic examples
- Local neighborhood structure is preserved
- Features are continuous (not categorical)

**Cost-Sensitive Learning Assumptions:**
- Misclassification costs can be accurately estimated
- Cost ratios remain constant across different regions of feature space
- Business/domain costs can be translated to algorithmic costs

**Undersampling Assumptions:**
- Removed majority samples are truly redundant
- Information loss is acceptable for balance
- Remaining samples are representative of the full distribution

### Limitations and Challenges

**SMOTE Limitations:**
- **Overgeneralization**: May create synthetic samples in inappropriate regions
- **Curse of dimensionality**: Less effective in high-dimensional spaces
- **Categorical features**: Not directly applicable to categorical variables
- **Noise amplification**: May amplify noise in minority class data

**Cost-Sensitive Limitations:**
- **Cost estimation**: Difficult to determine appropriate cost ratios
- **Class overlap**: May not work well when classes have significant overlap
- **Imbalanced validation**: Standard cross-validation may not be appropriate

**General Limitations:**
- **Evaluation challenges**: Standard metrics can be misleading
- **Model selection**: Need specialized techniques for hyperparameter tuning
- **Real-world deployment**: Performance may degrade in production
- **Temporal drift**: Class distributions may change over time

### Comparison with Alternative Approaches

**Sampling vs. Algorithmic Solutions:**
- **Sampling**: Modifies data distribution, works with any algorithm
- **Algorithmic**: Modifies algorithm behavior, preserves original data

**Ensemble vs. Single Model:**
- **Ensemble**: More robust but complex and harder to interpret
- **Single Model**: Simpler but may be more sensitive to imbalance

**Threshold Moving vs. Data Modification:**
- **Threshold**: Simple post-processing approach
- **Data Modification**: Changes training process but may introduce artifacts

## =¡ Interview Questions

??? question "1. What is the difference between imbalanced and skewed data? How do you detect each?"
    **Answer:**
    - **Imbalanced Data**: Unequal class distribution in target variable
      - Detection: Check class counts, calculate imbalance ratio
      - Example: 95% normal transactions, 5% fraud
    - **Skewed Data**: Non-normal distribution in features
      - Detection: Histogram analysis, skewness coefficient
      - Example: Income distribution (most people earn moderate amounts, few earn very high)
    - **Key Differences**:
      - Imbalanced affects target variable, skewed affects features
      - Different solutions: sampling for imbalance, transformation for skewness
      - Different evaluation challenges
    - **Detection Methods**:
      - Imbalanced: `Counter(y)`, class distribution plots
      - Skewed: `scipy.stats.skew()`, Q-Q plots, histograms

??? question "2. Why is accuracy a poor metric for imbalanced datasets? What metrics should you use instead?"
    **Answer:**
    - **Why Accuracy Fails**:
      - Can achieve high accuracy by always predicting majority class
      - Example: 99% accuracy on 99:1 dataset by predicting majority class
      - Doesn't reflect performance on minority class
    - **Better Metrics**:
      - **Precision**: TP/(TP+FP) - How many predicted positives are actually positive
      - **Recall/Sensitivity**: TP/(TP+FN) - How many actual positives were found
      - **F1-Score**: Harmonic mean of precision and recall
      - **AUC-ROC**: Area under ROC curve, threshold-independent
      - **AUC-PR**: Area under Precision-Recall curve, better for severe imbalance
    - **Confusion Matrix Analysis**:
      - Focus on True Positives and False Negatives for minority class
      - Consider business cost of different error types
    - **Stratified Evaluation**:
      - Use stratified cross-validation
      - Report per-class metrics separately

??? question "3. Explain SMOTE algorithm. What are its advantages and disadvantages?"
    **Answer:**
    - **SMOTE Algorithm**:
      - Finds k-nearest neighbors of minority samples
      - Creates synthetic samples along lines between samples and neighbors
      - Formula: new_sample = sample + » × (neighbor - sample), »  [0,1]
    - **Advantages**:
      - Increases minority class size without exact duplication
      - Considers local neighborhood structure
      - Works well with continuous features
      - Reduces overfitting compared to simple oversampling
    - **Disadvantages**:
      - Can create synthetic samples in majority class regions
      - Assumes linear relationships between features
      - Doesn't work well with categorical features
      - May amplify noise in the data
      - Can lead to overgeneralization
    - **Variants**:
      - **Borderline-SMOTE**: Focuses on borderline samples
      - **ADASYN**: Adaptive density-based approach
      - **SMOTE-ENN/Tomek**: Combines oversampling with undersampling

??? question "4. When would you use undersampling vs oversampling? What are the tradeoffs?"
    **Answer:**
    - **Undersampling When**:
      - Large dataset with sufficient majority samples
      - Computational resources are limited
      - Majority class has redundant/noisy samples
      - Training time is a major constraint
    - **Oversampling When**:
      - Small dataset where information loss is critical
      - Minority class is very small
      - Sufficient computational resources available
      - Want to preserve all original information
    - **Tradeoffs**:
      - **Undersampling**: Faster training, information loss, potential underfitting
      - **Oversampling**: Preserves information, longer training, potential overfitting
    - **Hybrid Approaches**:
      - SMOTE + Tomek: Oversample then clean borderline samples
      - SMOTE + ENN: Oversample then remove noisy samples
    - **Decision Framework**:
      - Consider data size, computational budget, domain expertise
      - Try both approaches and compare validation performance

??? question "5. How do you evaluate a model on imbalanced data? What cross-validation strategy should you use?"
    **Answer:**
    - **Evaluation Strategy**:
      - **Stratified Cross-Validation**: Maintains class distribution in each fold
      - **Multiple Metrics**: Use precision, recall, F1, AUC-ROC, AUC-PR
      - **Business Metrics**: Consider actual costs of different error types
      - **Threshold Analysis**: Plot precision-recall curves, find optimal threshold
    - **Cross-Validation Considerations**:
      - Always use stratified splits to maintain class balance
      - Consider time-series splits for temporal data
      - Be careful with very small minority classes (may have zero samples in some folds)
    - **Reporting Guidelines**:
      - Report confidence intervals for metrics
      - Show confusion matrices for each fold
      - Analyze per-class performance separately
      - Consider statistical significance tests
    - **Validation Pitfalls**:
      - Don't use random splits without stratification
      - Don't rely solely on accuracy
      - Don't ignore class-specific performance

??? question "6. What is cost-sensitive learning and how does it help with imbalanced data?"
    **Answer:**
    - **Cost-Sensitive Learning**:
      - Assigns different costs to different types of misclassifications
      - Modifies loss function to penalize minority class errors more heavily
      - Can be applied at algorithm level or through class weights
    - **Implementation Methods**:
      - **Class Weights**: Multiply loss by class-specific weights
      - **Cost Matrix**: Define explicit costs for each error type
      - **Threshold Adjustment**: Move decision boundary based on costs
    - **Benefits**:
      - Directly incorporates business/domain costs
      - Can be applied to most algorithms
      - More principled than arbitrary sampling
    - **Challenges**:
      - Difficult to estimate appropriate costs
      - May not work well with overlapping classes
      - Requires domain expertise for cost determination
    - **Example**: In medical diagnosis, false negative (missing disease) might cost 100x more than false positive (unnecessary test)

??? question "7. How would you handle a dataset with 99.9% majority class and 0.1% minority class?"
    **Answer:**
    - **Severe Imbalance Strategies**:
      - **Anomaly Detection**: Treat as one-class problem instead of classification
      - **Ensemble Methods**: Use specialized ensemble techniques (BalancedBagging)
      - **Threshold Optimization**: Move decision boundary toward minority class
      - **Cost-Sensitive Learning**: High penalty for minority class errors
    - **Sampling Approaches**:
      - **Conservative Oversampling**: Moderate SMOTE to avoid overfitting
      - **Informed Undersampling**: Remove only clearly redundant majority samples
      - **Hybrid Methods**: Combine multiple techniques carefully
    - **Evaluation Considerations**:
      - Focus on AUC-PR rather than AUC-ROC
      - Use stratified sampling with large number of folds
      - Consider using bootstrap validation
    - **Alternative Formulations**:
      - Treat as ranking problem
      - Use one-class SVM or isolation forest
      - Consider active learning to find more minority examples
    - **Business Considerations**:
      - Understand the cost of false negatives vs false positives
      - Consider if problem needs to be solved as classification or detection

??? question "8. What are the challenges in deploying models trained on imbalanced data in production?"
    **Answer:**
    - **Distribution Drift**:
      - Class distributions may change over time
      - Need monitoring systems to detect drift
      - May require model retraining or threshold adjustment
    - **Performance Degradation**:
      - Synthetic samples may not reflect real-world complexity
      - Overfitted models may perform poorly on new data
      - Need robust validation strategies
    - **Threshold Selection**:
      - Optimal threshold may change in production
      - Need business-driven threshold selection
      - Consider implementing dynamic thresholds
    - **Monitoring Challenges**:
      - Standard accuracy metrics are misleading
      - Need to monitor precision, recall separately
      - Set up alerts for minority class performance drops
    - **Mitigation Strategies**:
      - Implement A/B testing for model updates
      - Use ensemble models for robustness
      - Maintain feedback loops for continuous learning
      - Regular model retraining with fresh data

??? question "9. How do you choose between different sampling techniques (SMOTE, ADASYN, Random sampling, etc.)?"
    **Answer:**
    - **Selection Criteria**:
      - **Data characteristics**: Size, dimensionality, noise level
      - **Computational constraints**: Training time, memory requirements
      - **Domain knowledge**: Understanding of feature relationships
    - **Technique Guidelines**:
      - **Random Oversampling**: Quick baseline, risk of overfitting
      - **SMOTE**: Good for continuous features, assumes linear relationships
      - **ADASYN**: Better for varying density distributions
      - **Borderline-SMOTE**: When minority class has clear boundaries
      - **Random Undersampling**: Large datasets, computational constraints
    - **Experimental Approach**:
      - Try multiple techniques with cross-validation
      - Compare using appropriate metrics (F1, AUC-PR)
      - Consider ensemble of different sampling approaches
    - **Validation Strategy**:
      - Use stratified CV to compare techniques
      - Test on holdout set with original distribution
      - Consider robustness across different random seeds
    - **Practical Considerations**:
      - Implementation complexity and maintenance
      - Interpretability requirements
      - Integration with existing ML pipelines

??? question "10. Describe a real-world scenario where you had to deal with severely imbalanced data and explain your approach."
    **Answer:**
    *This question expects a detailed walkthrough of a practical solution. Here's an example framework:*
    
    - **Problem Description**:
      - "Fraud detection with 0.1% fraud rate in credit card transactions"
      - Business impact: $1M loss per undetected fraud, $50 cost per false alarm
    - **Initial Analysis**:
      - Analyzed data distribution, feature importance
      - Identified temporal patterns, seasonal effects
      - Explored feature engineering opportunities
    - **Solution Approach**:
      - **Phase 1**: Established baseline with cost-sensitive logistic regression
      - **Phase 2**: Applied SMOTE with careful validation
      - **Phase 3**: Implemented ensemble with threshold optimization
    - **Evaluation Strategy**:
      - Time-based validation splits (no data leakage)
      - Focused on precision-recall curves
      - Business-driven threshold selection ($50 vs $1M cost)
    - **Production Considerations**:
      - Real-time scoring requirements
      - Model monitoring and drift detection
      - A/B testing framework for improvements
    - **Results and Learning**:
      - Achieved 95% recall at 2% precision (acceptable business tradeoff)
      - Learned importance of domain expertise in feature engineering
      - Ongoing monitoring revealed seasonal drift patterns

## >à Examples

### Real-World Example: Credit Fraud Detection

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import seaborn as sns

# Simulate credit fraud dataset
def create_fraud_dataset():
    """Create a realistic fraud detection dataset"""
    np.random.seed(42)
    
    # Create imbalanced dataset
    X, y = make_classification(
        n_samples=50000,
        n_features=30,
        n_informative=20,
        n_redundant=5,
        n_clusters_per_class=2,
        weights=[0.999, 0.001],  # 0.1% fraud rate
        random_state=42
    )
    
    # Create meaningful feature names
    feature_names = [
        'transaction_amount', 'account_age_days', 'num_transactions_day',
        'avg_transaction_amount', 'time_since_last_transaction', 'merchant_risk_score',
        'geographic_risk', 'device_risk_score', 'velocity_1hr', 'velocity_24hr'
    ] + [f'feature_{i}' for i in range(10, 30)]
    
    return pd.DataFrame(X, columns=feature_names), y

# Create the fraud dataset
print("Creating Fraud Detection Dataset...")
X_fraud, y_fraud = create_fraud_dataset()

print(f"Dataset shape: {X_fraud.shape}")
print(f"Fraud cases: {np.sum(y_fraud)} ({np.sum(y_fraud)/len(y_fraud)*100:.3f}%)")
print(f"Normal cases: {np.sum(y_fraud == 0)} ({np.sum(y_fraud == 0)/len(y_fraud)*100:.3f}%)")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_fraud, y_fraud, test_size=0.2, stratify=y_fraud, random_state=42
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFraud Detection: Comprehensive Analysis")
print("="*60)

# 1. Baseline Model
print("\n1. BASELINE MODEL (No Imbalance Handling)")
print("-"*50)

baseline_model = LogisticRegression(random_state=42)
baseline_model.fit(X_train_scaled, y_train)

y_pred_baseline = baseline_model.predict(X_test_scaled)
y_proba_baseline = baseline_model.predict_proba(X_test_scaled)[:, 1]

print("Baseline Results:")
print(classification_report(y_test, y_pred_baseline))

# Calculate business metrics
def calculate_business_metrics(y_true, y_pred, cost_fn=1000000, cost_fp=50):
    """Calculate business-relevant metrics for fraud detection"""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    total_cost = fn * cost_fn + fp * cost_fp  # Cost of false negatives + false positives
    savings = tp * cost_fn  # Money saved by catching fraud
    net_savings = savings - total_cost
    
    return {
        'total_cost': total_cost,
        'savings': savings,
        'net_savings': net_savings,
        'cost_per_transaction': total_cost / len(y_true)
    }

baseline_business = calculate_business_metrics(y_test, y_pred_baseline)
print(f"\nBaseline Business Metrics:")
print(f"Total Cost: ${baseline_business['total_cost']:,.2f}")
print(f"Savings: ${baseline_business['savings']:,.2f}")
print(f"Net Savings: ${baseline_business['net_savings']:,.2f}")

# 2. SMOTE Approach
print("\n2. SMOTE APPROACH")
print("-"*30)

smote_pipeline = ImbPipeline([
    ('sampling', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(random_state=42))
])

smote_pipeline.fit(X_train_scaled, y_train)
y_pred_smote = smote_pipeline.predict(X_test_scaled)
y_proba_smote = smote_pipeline.predict_proba(X_test_scaled)[:, 1]

print("SMOTE Results:")
print(classification_report(y_test, y_pred_smote))

smote_business = calculate_business_metrics(y_test, y_pred_smote)
print(f"\nSMOTE Business Metrics:")
print(f"Total Cost: ${smote_business['total_cost']:,.2f}")
print(f"Savings: ${smote_business['savings']:,.2f}")
print(f"Net Savings: ${smote_business['net_savings']:,.2f}")

# 3. Cost-Sensitive Approach
print("\n3. COST-SENSITIVE APPROACH")
print("-"*35)

# Calculate class weights based on business costs
fraud_cases = np.sum(y_train == 1)
normal_cases = np.sum(y_train == 0)
cost_ratio = (cost_fn / cost_fp) * (normal_cases / fraud_cases)

cost_sensitive_model = LogisticRegression(
    class_weight={0: 1, 1: cost_ratio}, 
    random_state=42
)
cost_sensitive_model.fit(X_train_scaled, y_train)

y_pred_cost = cost_sensitive_model.predict(X_test_scaled)
y_proba_cost = cost_sensitive_model.predict_proba(X_test_scaled)[:, 1]

print("Cost-Sensitive Results:")
print(classification_report(y_test, y_pred_cost))

cost_business = calculate_business_metrics(y_test, y_pred_cost)
print(f"\nCost-Sensitive Business Metrics:")
print(f"Total Cost: ${cost_business['total_cost']:,.2f}")
print(f"Savings: ${cost_business['savings']:,.2f}")
print(f"Net Savings: ${cost_business['net_savings']:,.2f}")

# 4. Threshold Optimization
print("\n4. THRESHOLD OPTIMIZATION")
print("-"*30)

def find_optimal_threshold(y_true, y_proba, cost_fn=1000000, cost_fp=50):
    """Find optimal threshold based on business costs"""
    thresholds = np.arange(0.01, 1.0, 0.01)
    best_threshold = 0.5
    best_net_savings = float('-inf')
    
    results = []
    
    for threshold in thresholds:
        y_pred_thresh = (y_proba >= threshold).astype(int)
        business_metrics = calculate_business_metrics(y_true, y_pred_thresh, cost_fn, cost_fp)
        
        results.append({
            'threshold': threshold,
            'net_savings': business_metrics['net_savings'],
            'total_cost': business_metrics['total_cost']
        })
        
        if business_metrics['net_savings'] > best_net_savings:
            best_net_savings = business_metrics['net_savings']
            best_threshold = threshold
    
    return best_threshold, best_net_savings, pd.DataFrame(results)

# Find optimal threshold for cost-sensitive model
optimal_threshold, optimal_savings, threshold_results = find_optimal_threshold(
    y_test, y_proba_cost
)

print(f"Optimal Threshold: {optimal_threshold:.3f}")
print(f"Optimal Net Savings: ${optimal_savings:,.2f}")

# Apply optimal threshold
y_pred_optimal = (y_proba_cost >= optimal_threshold).astype(int)
print("\nOptimal Threshold Results:")
print(classification_report(y_test, y_pred_optimal))

# 5. Comprehensive Visualization
print("\n5. COMPREHENSIVE VISUALIZATION")
print("-"*35)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Class Distribution
ax1 = axes[0, 0]
class_counts = [np.sum(y_fraud == 0), np.sum(y_fraud == 1)]
ax1.bar(['Normal', 'Fraud'], class_counts, color=['lightblue', 'red'], alpha=0.7)
ax1.set_title('Original Class Distribution')
ax1.set_ylabel('Number of Samples')
ax1.set_yscale('log')  # Log scale to show both classes clearly

# Plot 2: ROC Curves
ax2 = axes[0, 1]
models = {
    'Baseline': y_proba_baseline,
    'SMOTE': y_proba_smote,
    'Cost-Sensitive': y_proba_cost
}

for name, y_proba in models.items():
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    ax2.plot(fpr, tpr, label=f'{name} (AUC={auc_score:.3f})')

ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curves Comparison')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Precision-Recall Curves
ax3 = axes[0, 2]
for name, y_proba in models.items():
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    auc_pr = auc(recall, precision)
    ax3.plot(recall, precision, label=f'{name} (AUC-PR={auc_pr:.3f})')

ax3.set_xlabel('Recall')
ax3.set_ylabel('Precision')
ax3.set_title('Precision-Recall Curves')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Business Metrics Comparison
ax4 = axes[1, 0]
methods = ['Baseline', 'SMOTE', 'Cost-Sensitive', 'Optimal Threshold']
net_savings = [
    baseline_business['net_savings'],
    smote_business['net_savings'], 
    cost_business['net_savings'],
    optimal_savings
]

colors = ['red' if x < 0 else 'green' for x in net_savings]
bars = ax4.bar(methods, net_savings, color=colors, alpha=0.7)
ax4.set_title('Net Savings Comparison')
ax4.set_ylabel('Net Savings ($)')
ax4.tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar, value in zip(bars, net_savings):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'${value:,.0f}', ha='center', va='bottom' if value > 0 else 'top')

# Plot 5: Threshold Analysis
ax5 = axes[1, 1]
ax5.plot(threshold_results['threshold'], threshold_results['net_savings'])
ax5.axvline(x=optimal_threshold, color='red', linestyle='--', 
           label=f'Optimal: {optimal_threshold:.3f}')
ax5.set_xlabel('Classification Threshold')
ax5.set_ylabel('Net Savings ($)')
ax5.set_title('Threshold vs Net Savings')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Confusion Matrix for Optimal Model
ax6 = axes[1, 2]
cm_optimal = confusion_matrix(y_test, y_pred_optimal)
sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Blues', ax=ax6)
ax6.set_title('Optimal Model Confusion Matrix')
ax6.set_xlabel('Predicted')
ax6.set_ylabel('Actual')

plt.tight_layout()
plt.show()

# Final Summary
print("\n" + "="*60)
print("FINAL SUMMARY - FRAUD DETECTION CASE STUDY")
print("="*60)

summary_data = {
    'Method': ['Baseline', 'SMOTE', 'Cost-Sensitive', 'Optimal Threshold'],
    'Precision': [
        precision_score(y_test, y_pred_baseline),
        precision_score(y_test, y_pred_smote),
        precision_score(y_test, y_pred_cost),
        precision_score(y_test, y_pred_optimal)
    ],
    'Recall': [
        recall_score(y_test, y_pred_baseline),
        recall_score(y_test, y_pred_smote),
        recall_score(y_test, y_pred_cost),
        recall_score(y_test, y_pred_optimal)
    ],
    'F1-Score': [
        f1_score(y_test, y_pred_baseline),
        f1_score(y_test, y_pred_smote),
        f1_score(y_test, y_pred_cost),
        f1_score(y_test, y_pred_optimal)
    ],
    'Net Savings': net_savings
}

summary_df = pd.DataFrame(summary_data)
print("\nPerformance Summary:")
print(summary_df.round(3).to_string(index=False))

print(f"\nKey Insights:")
print(f"" Baseline model had high precision but very low recall")
print(f"" SMOTE improved recall but at the cost of precision")
print(f"" Cost-sensitive approach balanced precision and recall better")
print(f"" Optimal threshold maximized business value")
print(f"" Best approach achieved ${optimal_savings:,.0f} net savings")

print(f"\nBusiness Recommendations:")
print(f"" Deploy cost-sensitive model with optimal threshold ({optimal_threshold:.3f})")
print(f"" Monitor precision-recall tradeoff in production")
print(f"" Implement real-time threshold adjustment based on costs")
print(f"" Regular model retraining as fraud patterns evolve")
```

**Key Takeaways from the Example:**
- **Business Context Matters**: The optimal approach depends on the actual costs of different error types
- **Multiple Techniques**: Often combining approaches (cost-sensitive + threshold optimization) works best
- **Evaluation is Critical**: Standard metrics can be misleading; business metrics are essential
- **Threshold Optimization**: Can significantly improve business outcomes without changing the model

## =Ú References

1. **Books:**
   - [Learning from Imbalanced Data Sets - Alberto Fernández](https://link.springer.com/book/10.1007/978-3-319-98074-4)
   - [Imbalanced Learning: Foundations, Algorithms, and Applications - He & Ma](https://ieeexplore.ieee.org/book/6267335)
   - [The Elements of Statistical Learning - Hastie, Tibshirani, Friedman](https://web.stanford.edu/~hastie/ElemStatLearn/)

2. **Research Papers:**
   - [SMOTE: Synthetic Minority Over-sampling Technique - Chawla et al.](https://arxiv.org/abs/1106.1813)
   - [Learning from Imbalanced Data - He & Garcia](https://ieeexplore.ieee.org/document/5128907)
   - [Cost-Sensitive Learning - Elkan](https://www.semanticscholar.org/paper/The-Foundations-of-Cost-Sensitive-Learning-Elkan/aa5b7bb1e214b7dfa2daa6280aedee8a25b9a264)

3. **Libraries and Documentation:**
   - [Imbalanced-learn Documentation](https://imbalanced-learn.org/stable/)
   - [Scikit-learn Imbalanced Datasets](https://scikit-learn.org/stable/modules/learning_with_multiclass_multioutput.html#multiclass-and-multilabel-classification)
   - [XGBoost Imbalanced Classification](https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html)

4. **Online Resources:**
   - [Google's Rules of Machine Learning - Dealing with Imbalanced Data](https://developers.google.com/machine-learning/guides/rules-of-ml/)
   - [Towards Data Science - Imbalanced Data Articles](https://towardsdatascience.com/tagged/imbalanced-data)
   - [Kaggle Learn - Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning)

5. **Datasets for Practice:**
   - [Credit Card Fraud Detection - Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
   - [Adult Income Dataset - UCI](https://archive.ics.uci.edu/ml/datasets/adult)
   - [Mammographic Mass Dataset - UCI](https://archive.ics.uci.edu/ml/datasets/Mammographic+Mass)