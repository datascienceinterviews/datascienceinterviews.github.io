---
title: Confusion Matrix
description: Comprehensive guide to Confusion Matrix with implementation, interpretation, and evaluation metrics.
comments: true
---

# 📘 Confusion Matrix

A confusion matrix is a table used to evaluate the performance of classification models by showing the actual vs predicted classifications in a structured format.

**Resources:** [Scikit-learn Confusion Matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) | [Wikipedia Confusion Matrix](https://en.wikipedia.org/wiki/Confusion_matrix)

## ✍️ Summary

A confusion matrix is a fundamental tool in machine learning for evaluating the performance of classification algorithms. It provides a detailed breakdown of correct and incorrect predictions for each class, enabling comprehensive analysis of model performance.

**Key characteristics:**
- **Visual representation**: Clear tabular format showing prediction accuracy
- **Multi-class support**: Works with binary and multi-class classification
- **Metric foundation**: Basis for calculating precision, recall, F1-score, etc.
- **Error analysis**: Helps identify which classes are being confused

**Applications:**
- Model evaluation and comparison
- Error analysis and debugging
- Performance reporting
- Threshold optimization
- Medical diagnosis validation
- Quality control systems

The matrix is typically organized with:
- **Rows**: Actual (true) class labels
- **Columns**: Predicted class labels
- **Diagonal**: Correct predictions
- **Off-diagonal**: Misclassifications

## 🧠 Intuition

### Mathematical Foundation

For a binary classification problem, the confusion matrix is a 2×2 table:

```
                Predicted
                0    1
Actual    0    TN   FP
          1    FN   TP
```

Where:
- **TP (True Positive)**: Correctly predicted positive cases
- **TN (True Negative)**: Correctly predicted negative cases  
- **FP (False Positive)**: Incorrectly predicted as positive (Type I error)
- **FN (False Negative)**: Incorrectly predicted as negative (Type II error)

### Derived Metrics

From the confusion matrix, we can calculate several important metrics:

**Accuracy**: Overall correctness
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Precision**: How many selected items are relevant
$$\text{Precision} = \frac{TP}{TP + FP}$$

**Recall (Sensitivity)**: How many relevant items are selected
$$\text{Recall} = \frac{TP}{TP + FN}$$

**Specificity**: True negative rate
$$\text{Specificity} = \frac{TN}{TN + FP}$$

**F1-Score**: Harmonic mean of precision and recall
$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### Multi-class Extension

For multi-class problems with $n$ classes, the matrix becomes $n \times n$:

$$C_{i,j} = \text{number of observations known to be in group } i \text{ and predicted to be in group } j$$

## 🔢 Implementation using Libraries

### Using Scikit-learn

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Generate sample dataset
X, y = make_classification(n_samples=1000, n_features=4, n_classes=3, 
                          n_redundant=0, n_informative=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\nMetrics:")
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")

# Detailed classification report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))
```

### Visualization with Seaborn

```python
# Create a more detailed visualization
def plot_confusion_matrix(cm, class_names=None, title='Confusion Matrix'):
    """
    Plot confusion matrix with annotations and percentages
    """
    plt.figure(figsize=(8, 6))
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotations with both counts and percentages
    annotations = []
    for i in range(cm.shape[0]):
        row_annotations = []
        for j in range(cm.shape[1]):
            row_annotations.append(f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)')
        annotations.append(row_annotations)
    
    # Plot heatmap
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

# Plot the confusion matrix
class_names = ['Class 0', 'Class 1', 'Class 2']
plot_confusion_matrix(cm, class_names, 'Random Forest Confusion Matrix')
```

### Binary Classification Example

```python
# Binary classification example
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# Generate binary classification data
X_binary, y_binary = make_classification(n_samples=500, n_features=2, 
                                        n_redundant=0, n_informative=2,
                                        n_classes=2, random_state=42)
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_binary, y_binary, test_size=0.3, random_state=42)

# Train logistic regression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_b, y_train_b)
y_pred_b = log_reg.predict(X_test_b)

# Binary confusion matrix
cm_binary = confusion_matrix(y_test_b, y_pred_b)
print("Binary Confusion Matrix:")
print(cm_binary)

# Extract values
tn, fp, fn, tp = cm_binary.ravel()
print(f"\nTrue Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")

# Calculate metrics manually
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nManually Calculated Metrics:")
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"Specificity: {specificity:.3f}")
print(f"F1-Score: {f1:.3f}")
```

## ⚙️ From Scratch Implementation

```python
import numpy as np
from collections import Counter

class ConfusionMatrix:
    """
    From-scratch implementation of Confusion Matrix with metric calculations
    """
    
    def __init__(self):
        self.matrix = None
        self.classes = None
        self.n_classes = None
        
    def fit(self, y_true, y_pred):
        """
        Create confusion matrix from true and predicted labels
        
        Parameters:
        y_true: array-like, true class labels
        y_pred: array-like, predicted class labels
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Get unique classes
        self.classes = np.unique(np.concatenate([y_true, y_pred]))
        self.n_classes = len(self.classes)
        
        # Create mapping from class to index
        class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Initialize matrix
        self.matrix = np.zeros((self.n_classes, self.n_classes), dtype=int)
        
        # Fill matrix
        for true_label, pred_label in zip(y_true, y_pred):
            true_idx = class_to_idx[true_label]
            pred_idx = class_to_idx[pred_label]
            self.matrix[true_idx, pred_idx] += 1
        
        return self
    
    def get_matrix(self):
        """Return the confusion matrix"""
        if self.matrix is None:
            raise ValueError("Matrix not computed. Call fit() first.")
        return self.matrix
    
    def accuracy(self):
        """Calculate overall accuracy"""
        if self.matrix is None:
            raise ValueError("Matrix not computed. Call fit() first.")
        
        correct = np.trace(self.matrix)  # Sum of diagonal
        total = np.sum(self.matrix)
        return correct / total if total > 0 else 0
    
    def precision(self, average='macro'):
        """
        Calculate precision for each class or average
        
        Parameters:
        average: str, 'macro', 'micro', 'weighted', or None
        """
        if self.matrix is None:
            raise ValueError("Matrix not computed. Call fit() first.")
        
        # Per-class precision
        precisions = []
        for i in range(self.n_classes):
            true_positives = self.matrix[i, i]
            predicted_positives = np.sum(self.matrix[:, i])
            
            if predicted_positives == 0:
                precision = 0.0
            else:
                precision = true_positives / predicted_positives
            
            precisions.append(precision)
        
        precisions = np.array(precisions)
        
        if average is None:
            return precisions
        elif average == 'macro':
            return np.mean(precisions)
        elif average == 'micro':
            total_tp = np.trace(self.matrix)
            total_pred_pos = np.sum(self.matrix)
            return total_tp / total_pred_pos if total_pred_pos > 0 else 0
        elif average == 'weighted':
            support = np.sum(self.matrix, axis=1)
            return np.average(precisions, weights=support)
        else:
            raise ValueError("Invalid average type")
    
    def recall(self, average='macro'):
        """
        Calculate recall for each class or average
        
        Parameters:
        average: str, 'macro', 'micro', 'weighted', or None
        """
        if self.matrix is None:
            raise ValueError("Matrix not computed. Call fit() first.")
        
        # Per-class recall
        recalls = []
        for i in range(self.n_classes):
            true_positives = self.matrix[i, i]
            actual_positives = np.sum(self.matrix[i, :])
            
            if actual_positives == 0:
                recall = 0.0
            else:
                recall = true_positives / actual_positives
                
            recalls.append(recall)
        
        recalls = np.array(recalls)
        
        if average is None:
            return recalls
        elif average == 'macro':
            return np.mean(recalls)
        elif average == 'micro':
            total_tp = np.trace(self.matrix)
            total_actual_pos = np.sum(self.matrix)
            return total_tp / total_actual_pos if total_actual_pos > 0 else 0
        elif average == 'weighted':
            support = np.sum(self.matrix, axis=1)
            return np.average(recalls, weights=support)
        else:
            raise ValueError("Invalid average type")
    
    def f1_score(self, average='macro'):
        """Calculate F1-score"""
        precision = self.precision(average=average)
        recall = self.recall(average=average)
        
        if isinstance(precision, np.ndarray):
            # Per-class F1 scores
            f1_scores = 2 * (precision * recall) / (precision + recall)
            f1_scores = np.nan_to_num(f1_scores)  # Handle division by zero
            return f1_scores
        else:
            # Average F1 score
            if (precision + recall) == 0:
                return 0.0
            return 2 * (precision * recall) / (precision + recall)
    
    def classification_report(self):
        """Generate a detailed classification report"""
        if self.matrix is None:
            raise ValueError("Matrix not computed. Call fit() first.")
        
        precisions = self.precision(average=None)
        recalls = self.recall(average=None)
        f1_scores = self.f1_score(average=None)
        support = np.sum(self.matrix, axis=1)
        
        print("Classification Report:")
        print("-" * 60)
        print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 60)
        
        for i, cls in enumerate(self.classes):
            print(f"{cls:<10} {precisions[i]:<12.3f} {recalls[i]:<12.3f} "
                  f"{f1_scores[i]:<12.3f} {support[i]:<10}")
        
        print("-" * 60)
        print(f"{'Accuracy':<10} {'':<12} {'':<12} {self.accuracy():<12.3f} {np.sum(support):<10}")
        print(f"{'Macro Avg':<10} {self.precision('macro'):<12.3f} "
              f"{self.recall('macro'):<12.3f} {self.f1_score('macro'):<12.3f} {np.sum(support):<10}")
        print(f"{'Weighted':<10} {self.precision('weighted'):<12.3f} "
              f"{self.recall('weighted'):<12.3f} {self.f1_score('weighted'):<12.3f} {np.sum(support):<10}")

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_samples = 300
    
    # Create synthetic predictions vs true labels
    y_true = np.random.choice([0, 1, 2], size=n_samples, p=[0.4, 0.35, 0.25])
    
    # Create predictions with some errors
    y_pred = y_true.copy()
    error_indices = np.random.choice(n_samples, size=int(0.2 * n_samples), replace=False)
    y_pred[error_indices] = np.random.choice([0, 1, 2], size=len(error_indices))
    
    # Create confusion matrix
    cm = ConfusionMatrix()
    cm.fit(y_true, y_pred)
    
    print("Confusion Matrix:")
    print(cm.get_matrix())
    print(f"\nAccuracy: {cm.accuracy():.3f}")
    print(f"Macro Precision: {cm.precision('macro'):.3f}")
    print(f"Macro Recall: {cm.recall('macro'):.3f}")
    print(f"Macro F1-Score: {cm.f1_score('macro'):.3f}")
    
    print("\n" + "="*60)
    cm.classification_report()
```

## ⚠️ Assumptions and Limitations

### Assumptions

1. **Ground Truth Availability**: Requires true labels for evaluation
2. **Consistent Labeling**: True and predicted labels must use the same class encoding
3. **Complete Predictions**: Every sample must have both true and predicted labels
4. **Class Balance Consideration**: Some metrics are sensitive to class imbalance

### Limitations

1. **Information Loss**: 
   - Doesn't show prediction confidence/probability
   - No information about feature importance

2. **Class Imbalance Sensitivity**:
   - Accuracy can be misleading with imbalanced datasets
   - May need to focus on per-class metrics

3. **Multi-label Limitations**:
   - Standard confusion matrix doesn't handle multi-label classification well
   - Each label needs separate evaluation

4. **Threshold Independence**:
   - Doesn't show how performance varies with different classification thresholds
   - May need ROC curves for threshold analysis

### Comparison with Other Evaluation Methods

| Method | Pros | Cons |
|--------|------|------|
| **Confusion Matrix** | Detailed breakdown, interpretable | Static, no confidence info |
| **ROC Curve** | Threshold analysis, AUC metric | Only for binary/one-vs-rest |
| **PR Curve** | Better for imbalanced data | More complex to interpret |
| **Cross-validation** | Robust performance estimate | Computationally expensive |

### When to Use Alternatives

- **Highly Imbalanced Data**: Use precision-recall curves
- **Probability Calibration**: Use reliability diagrams
- **Cost-Sensitive Applications**: Use cost matrices
- **Ranking Problems**: Use ranking metrics (NDCG, MAP)

## 💡 Interview Questions

??? question "**Q1: What is a confusion matrix and what does each cell represent?**"
    
    **Answer:** A confusion matrix is a table used to evaluate classification model performance. For binary classification:
    - **True Positives (TP)**: Correctly predicted positive cases
    - **True Negatives (TN)**: Correctly predicted negative cases
    - **False Positives (FP)**: Incorrectly predicted as positive (Type I error)
    - **False Negatives (FN)**: Incorrectly predicted as negative (Type II error)
    
    The diagonal represents correct predictions, while off-diagonal elements represent errors.

??? question "**Q2: How do you calculate precision and recall from a confusion matrix?**"
    
    **Answer:** From a binary confusion matrix:
    - **Precision = TP / (TP + FP)** - "Of all positive predictions, how many were correct?"
    - **Recall = TP / (TP + FN)** - "Of all actual positives, how many did we find?"
    
    For multi-class: Calculate per-class metrics and then average (macro, micro, or weighted).

??? question "**Q3: What's the difference between macro, micro, and weighted averaging?**"
    
    **Answer:**
    - **Macro Average**: Simple average of per-class metrics (treats all classes equally)
    - **Micro Average**: Calculate metrics globally by counting total TP, FP, FN
    - **Weighted Average**: Average of per-class metrics weighted by class support
    
    Micro average is better for imbalanced datasets, macro average for balanced datasets.

??? question "**Q4: When would accuracy be a poor metric to use?**"
    
    **Answer:** Accuracy is poor when:
    - **Class Imbalance**: 95% accuracy on a 95%-5% dataset might just predict majority class
    - **Cost-Sensitive Applications**: False negatives in medical diagnosis are more costly
    - **Multi-label Problems**: Partial correctness isn't captured
    - **Different Error Costs**: When different types of errors have different consequences

??? question "**Q5: How do you interpret a confusion matrix for multi-class classification?**"
    
    **Answer:** In an n×n matrix for n classes:
    - **Diagonal elements**: Correct predictions for each class
    - **Row sums**: Total actual instances of each class
    - **Column sums**: Total predicted instances of each class
    - **Off-diagonal**: Shows which classes are confused with each other
    
    Look for patterns: Are specific classes consistently confused?

??? question "**Q6: What is the relationship between specificity and false positive rate?**"
    
    **Answer:** 
    - **Specificity = TN / (TN + FP)** (True Negative Rate)
    - **False Positive Rate = FP / (TN + FP)**
    - **Relationship**: Specificity + FPR = 1
    
    High specificity means low false positive rate. This is important in applications where false alarms are costly.

??? question "**Q7: How would you handle a confusion matrix with very small numbers?**"
    
    **Answer:** When dealing with small sample sizes:
    - Use **confidence intervals** for metrics
    - Consider **bootstrapping** for robust estimates
    - Be cautious of **overfitting** to small test sets
    - Use **cross-validation** for better estimates
    - Consider **Bayesian approaches** with priors

??? question "**Q8: Can you explain the trade-off between precision and recall?**"
    
    **Answer:** There's typically an inverse relationship:
    - **Higher Precision**: Fewer false positives, but might miss true positives (lower recall)
    - **Higher Recall**: Catch more true positives, but might include false positives (lower precision)
    
    **F1-score** balances both. The optimal balance depends on the application's cost of false positives vs false negatives.

??? question "**Q9: How do you create a normalized confusion matrix and why is it useful?**"
    
    **Answer:** Normalize by dividing each row by its sum:
    ```python
    normalized_cm = cm / cm.sum(axis=1)[:, np.newaxis]
    ```
    
    **Benefits:**
    - Shows proportions instead of absolute counts
    - Better for comparing across different datasets
    - Easier to identify per-class performance patterns
    - Less affected by class imbalance in visualization

??? question "**Q10: What additional information would you want beyond a confusion matrix?**"
    
    **Answer:**
    - **Prediction probabilities**: For threshold tuning
    - **Feature importance**: To understand model decisions
    - **ROC/PR curves**: For threshold-dependent analysis
    - **Cost matrix**: For business-specific error costs
    - **Learning curves**: To check for overfitting
    - **Per-sample analysis**: To identify difficult cases

## 🧠 Examples

### Medical Diagnosis Example

```python
# Simulate medical diagnosis scenario
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Simulate a medical test for disease diagnosis
# True condition: 0 = Healthy, 1 = Disease
# Test result: 0 = Negative, 1 = Positive

np.random.seed(42)

# Create realistic medical scenario
# Disease prevalence: 5% (realistic for many conditions)
n_patients = 1000
disease_prevalence = 0.05

# Generate true conditions
y_true = np.random.choice([0, 1], size=n_patients, 
                         p=[1-disease_prevalence, disease_prevalence])

# Simulate test with known sensitivity and specificity
sensitivity = 0.95  # True positive rate
specificity = 0.90  # True negative rate

y_pred = []
for true_condition in y_true:
    if true_condition == 1:  # Patient has disease
        # Test positive with probability = sensitivity
        prediction = np.random.choice([0, 1], p=[1-sensitivity, sensitivity])
    else:  # Patient is healthy
        # Test negative with probability = specificity
        prediction = np.random.choice([0, 1], p=[specificity, 1-specificity])
    y_pred.append(prediction)

y_pred = np.array(y_pred)

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Medical Test Confusion Matrix:")
print("                Predicted")
print("               Neg  Pos")
print(f"Actual   Neg   {cm[0,0]:3d}  {cm[0,1]:3d}")
print(f"         Pos   {cm[1,0]:3d}  {cm[1,1]:3d}")

# Calculate important medical metrics
tn, fp, fn, tp = cm.ravel()

sensitivity_calc = tp / (tp + fn)
specificity_calc = tn / (tn + fp)
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value

print(f"\nMedical Test Performance:")
print(f"Sensitivity (True Positive Rate): {sensitivity_calc:.3f}")
print(f"Specificity (True Negative Rate): {specificity_calc:.3f}")
print(f"Positive Predictive Value (Precision): {ppv:.3f}")
print(f"Negative Predictive Value: {npv:.3f}")

# Visualize with medical terminology
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Raw confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_title('Medical Test Confusion Matrix')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')
ax1.set_xticklabels(['Negative', 'Positive'])
ax1.set_yticklabels(['Healthy', 'Disease'])

# Normalized confusion matrix
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues', ax=ax2)
ax2.set_title('Normalized Confusion Matrix (Percentages)')
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')
ax2.set_xticklabels(['Negative', 'Positive'])
ax2.set_yticklabels(['Healthy', 'Disease'])

plt.tight_layout()
plt.show()

# Interpretation
print(f"\nInterpretation:")
print(f"• Out of {tn + fp} healthy patients, {tn} were correctly identified (Specificity: {specificity_calc:.1%})")
print(f"• Out of {tp + fn} disease patients, {tp} were correctly identified (Sensitivity: {sensitivity_calc:.1%})")
print(f"• Out of {tp + fp} positive tests, {tp} were true positives (PPV: {ppv:.1%})")
print(f"• Out of {tn + fn} negative tests, {tn} were true negatives (NPV: {npv:.1%})")
```

### E-commerce Recommendation Example

```python
# E-commerce recommendation system evaluation
# Predict whether user will purchase recommended items

# Simulate user behavior data
np.random.seed(123)
n_recommendations = 2000

# Features that might affect purchase (simplified)
user_engagement = np.random.beta(2, 5, n_recommendations)  # 0-1 engagement score
item_popularity = np.random.beta(1.5, 3, n_recommendations)  # 0-1 popularity score
price_sensitivity = np.random.normal(0.5, 0.2, n_recommendations)  # Price factor

# True purchase probability (complex relationship)
purchase_prob = (0.4 * user_engagement + 
                0.3 * item_popularity + 
                0.3 * (1 - price_sensitivity))
purchase_prob = np.clip(purchase_prob, 0.1, 0.9)

# Generate true purchases
y_true_ecommerce = np.random.binomial(1, purchase_prob)

# Simulate recommendation algorithm predictions (with some errors)
pred_prob = purchase_prob + np.random.normal(0, 0.15, n_recommendations)
pred_prob = np.clip(pred_prob, 0, 1)

# Convert probabilities to binary predictions using threshold
threshold = 0.5
y_pred_ecommerce = (pred_prob > threshold).astype(int)

# Create confusion matrix
cm_ecommerce = confusion_matrix(y_true_ecommerce, y_pred_ecommerce)

print("E-commerce Recommendation Confusion Matrix:")
print("                    Predicted")
print("                No Purchase  Purchase")
print(f"Actual No Purchase    {cm_ecommerce[0,0]:4d}      {cm_ecommerce[0,1]:4d}")
print(f"       Purchase       {cm_ecommerce[1,0]:4d}      {cm_ecommerce[1,1]:4d}")

# Business metrics
tn, fp, fn, tp = cm_ecommerce.ravel()

# Business interpretation
conversion_rate = (tp + fn) / (tn + fp + fn + tp)
predicted_conversion = (tp + fp) / (tn + fp + fn + tp)
precision_purchase = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_purchase = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"\nBusiness Metrics:")
print(f"Overall Conversion Rate: {conversion_rate:.1%}")
print(f"Predicted Conversion Rate: {predicted_conversion:.1%}")
print(f"Recommendation Precision: {precision_purchase:.1%} (of recommended items, how many were purchased)")
print(f"Purchase Recall: {recall_purchase:.1%} (of actual purchases, how many were recommended)")

# Cost analysis (hypothetical)
revenue_per_purchase = 50  # $50 average order value
cost_per_recommendation = 0.1  # $0.10 cost to show recommendation

total_revenue = tp * revenue_per_purchase
total_cost = (tp + fp) * cost_per_recommendation
net_profit = total_revenue - total_cost
roi = (net_profit / total_cost) * 100 if total_cost > 0 else 0

print(f"\nCost Analysis:")
print(f"Total Revenue from TP: ${total_revenue:.2f}")
print(f"Total Recommendation Cost: ${total_cost:.2f}")
print(f"Net Profit: ${net_profit:.2f}")
print(f"ROI: {roi:.1f}%")

# Show impact of different thresholds
thresholds = np.arange(0.1, 0.9, 0.1)
results = []

for thresh in thresholds:
    y_pred_thresh = (pred_prob > thresh).astype(int)
    cm_thresh = confusion_matrix(y_true_ecommerce, y_pred_thresh)
    tn_t, fp_t, fn_t, tp_t = cm_thresh.ravel()
    
    precision_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
    recall_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
    
    revenue_t = tp_t * revenue_per_purchase
    cost_t = (tp_t + fp_t) * cost_per_recommendation
    profit_t = revenue_t - cost_t
    
    results.append({
        'threshold': thresh,
        'precision': precision_t,
        'recall': recall_t,
        'profit': profit_t,
        'recommendations': tp_t + fp_t
    })

# Find optimal threshold
optimal_thresh = max(results, key=lambda x: x['profit'])
print(f"\nOptimal Threshold Analysis:")
print(f"Best threshold for profit: {optimal_thresh['threshold']:.1f}")
print(f"Precision at optimal: {optimal_thresh['precision']:.1%}")
print(f"Recall at optimal: {optimal_thresh['recall']:.1%}")
print(f"Profit at optimal: ${optimal_thresh['profit']:.2f}")
print(f"Total recommendations: {optimal_thresh['recommendations']}")
```

## 📚 References

1. **Documentation:**
   - [Scikit-learn Confusion Matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
   - [Scikit-learn Classification Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)

2. **Books:**
   - "Pattern Recognition and Machine Learning" by Christopher Bishop
   - "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman
   - "Hands-On Machine Learning" by Aurélien Géron

3. **Research Papers:**
   - "A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection" - Kohavi (1995)
   - "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets" - Saito & Rehmsmeier (2015)

4. **Online Resources:**
   - [Wikipedia: Confusion Matrix](https://en.wikipedia.org/wiki/Confusion_matrix)
   - [Google ML Crash Course: Classification](https://developers.google.com/machine-learning/crash-course/classification)
   - [Towards Data Science: Confusion Matrix Articles](https://towardsdatascience.com/tagged/confusion-matrix)

5. **Video Tutorials:**
   - [StatQuest: Confusion Matrix](https://www.youtube.com/watch?v=Kdsp6soqA7o)
   - [3Blue1Brown: Neural Networks Series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
