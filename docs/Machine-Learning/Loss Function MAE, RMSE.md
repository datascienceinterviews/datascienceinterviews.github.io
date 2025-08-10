---
title: Loss Functions - MAE, RMSE
description: Comprehensive guide to Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) loss functions with mathematical intuition, implementations, and interview questions.
comments: true
---

# 📊 Loss Functions - MAE, RMSE

Loss functions quantify the difference between predicted and actual values, serving as the foundation for training machine learning models through optimization algorithms.

**Resources:** [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html) | [Deep Learning Book - Chapter 5](https://www.deeplearningbook.org/contents/ml.html)

## 
 Summary

Loss functions are mathematical functions that measure the discrepancy between predicted values and true values in machine learning models. MAE and RMSE are two fundamental regression loss functions:

**Mean Absolute Error (MAE):**
- Measures the average magnitude of errors in predictions
- Less sensitive to outliers
- Provides uniform penalty for all errors
- Also known as L1 loss

**Root Mean Square Error (RMSE):**
- Measures the square root of the average squared differences
- More sensitive to outliers due to squaring
- Penalizes larger errors more heavily
- Related to L2 loss (MSE)

**Applications:**
- Regression model evaluation
- Neural network training objectives
- Time series forecasting assessment
- Computer vision tasks
- Financial modeling
- Performance benchmarking

**When to use which:**
- **MAE**: When all errors are equally important and outliers should not dominate
- **RMSE**: When larger errors are more problematic and should be penalized heavily

## 🧠 Intuition

### Mathematical Foundation

#### Mean Absolute Error (MAE)

**Definition:**
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

Where:
- $n$ is the number of samples
- $y_i$ is the true value
- $\hat{y}_i$ is the predicted value
- $|·|$ denotes absolute value

**Properties:**
- **Linear penalty**: Each unit of error contributes equally
- **Robust to outliers**: Outliers don't disproportionately affect the loss
- **Non-differentiable at zero**: Gradient-based optimization can be challenging
- **Interpretable**: Same units as the target variable

#### Root Mean Square Error (RMSE)

**Definition:**
$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

**Relation to Mean Square Error (MSE):**
$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
$$\text{RMSE} = \sqrt{\text{MSE}}$$

**Properties:**
- **Quadratic penalty**: Larger errors are penalized exponentially more
- **Sensitive to outliers**: Large errors dominate the loss function
- **Differentiable everywhere**: Smooth optimization landscape
- **Interpretable units**: Same units as the target variable (unlike MSE)

### Geometric Interpretation

**MAE (L1 Loss):**
- Forms diamond-shaped contours in parameter space
- Encourages sparse solutions
- Equal penalty regardless of error magnitude

**RMSE/MSE (L2 Loss):**
- Forms circular contours in parameter space
- Smooth gradients everywhere
- Increasing penalty with error magnitude

### Gradient Analysis

**MAE Gradient:**
$$\frac{\partial \text{MAE}}{\partial \hat{y}_i} = \frac{1}{n} \cdot \text{sign}(y_i - \hat{y}_i)$$

**MSE Gradient:**
$$\frac{\partial \text{MSE}}{\partial \hat{y}_i} = \frac{2}{n} (y_i - \hat{y}_i)$$

The MAE gradient is constant (±1/n), while MSE gradient is proportional to the error magnitude.

## 💻 Implementation using Libraries

### Using Scikit-learn

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import warnings
warnings.filterwarnings('ignore')

# Generate sample regression data
X, y = make_regression(n_samples=1000, n_features=5, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate MAE and RMSE
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Square Error (MSE): {mse:.4f}")
print(f"Root Mean Square Error (RMSE): {rmse:.4f}")

# Demonstrate outlier sensitivity
# Add some outliers to predictions
y_pred_with_outliers = y_pred.copy()
y_pred_with_outliers[:5] += 100  # Add large errors to first 5 predictions

mae_outliers = mean_absolute_error(y_test, y_pred_with_outliers)
rmse_outliers = np.sqrt(mean_squared_error(y_test, y_pred_with_outliers))

print(f"\nWith Outliers:")
print(f"MAE: {mae:.4f} -> {mae_outliers:.4f} (increase: {mae_outliers/mae:.2f}x)")
print(f"RMSE: {rmse:.4f} -> {rmse_outliers:.4f} (increase: {rmse_outliers/rmse:.2f}x)")
```

### Using TensorFlow/Keras

```python
import tensorflow as tf
from tensorflow import keras

# Custom loss functions
def mae_loss(y_true, y_pred):
    """Mean Absolute Error loss function"""
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def rmse_loss(y_true, y_pred):
    """Root Mean Square Error loss function"""
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

# Example neural network with different loss functions
def create_model(loss_fn):
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(5,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss=loss_fn, metrics=[mae_loss, rmse_loss])
    return model

# Train models with different loss functions
mae_model = create_model(mae_loss)
rmse_model = create_model('mse')  # MSE is equivalent to RMSE for optimization

print("MAE Model:")
mae_history = mae_model.fit(X_train, y_train, epochs=50, batch_size=32, 
                           validation_split=0.2, verbose=0)

print("RMSE Model:")
rmse_history = rmse_model.fit(X_train, y_train, epochs=50, batch_size=32, 
                             validation_split=0.2, verbose=0)
```

### Plotting Loss Functions

```python
# Visualize how MAE and RMSE behave with different error magnitudes
errors = np.linspace(-5, 5, 100)
mae_values = np.abs(errors)
mse_values = errors**2

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(errors, mae_values, label='MAE = |error|', linewidth=2)
plt.plot(errors, mse_values, label='MSE = error²', linewidth=2)
plt.xlabel('Prediction Error')
plt.ylabel('Loss Value')
plt.title('MAE vs MSE Loss Functions')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(errors[errors != 0], 1 * np.ones_like(errors[errors != 0]), 
         label='MAE Gradient = ±1', linewidth=2)
plt.plot(errors, 2 * errors, label='MSE Gradient = 2×error', linewidth=2)
plt.xlabel('Prediction Error')
plt.ylabel('Gradient')
plt.title('Gradient Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## 🔧 From Scratch Implementation

### Pure Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

class LossFunctions:
    """
    Implementation of MAE and RMSE loss functions from scratch
    """
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MAE value
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have same length")
        
        errors = y_true - y_pred
        absolute_errors = np.abs(errors)
        mae = np.mean(absolute_errors)
        
        return mae
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Square Error
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MSE value
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have same length")
        
        errors = y_true - y_pred
        squared_errors = errors ** 2
        mse = np.mean(squared_errors)
        
        return mse
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Root Mean Square Error
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            RMSE value
        """
        mse_value = LossFunctions.mse(y_true, y_pred)
        rmse = np.sqrt(mse_value)
        
        return rmse
    
    @staticmethod
    def mae_gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate gradient of MAE with respect to predictions
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Gradient array
        """
        errors = y_true - y_pred
        gradients = np.zeros_like(errors)
        
        gradients[errors > 0] = -1  # If error is positive, gradient is -1
        gradients[errors < 0] = 1   # If error is negative, gradient is +1
        gradients[errors == 0] = 0  # If error is zero, gradient is 0
        
        return gradients / len(y_true)
    
    @staticmethod
    def mse_gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate gradient of MSE with respect to predictions
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Gradient array
        """
        errors = y_true - y_pred
        gradients = -2 * errors  # Derivative of (y_true - y_pred)²
        
        return gradients / len(y_true)

# Demonstration of custom implementation
def demonstrate_loss_functions():
    """Demonstrate the custom loss function implementation"""
    # Create sample data
    np.random.seed(42)
    y_true = np.random.normal(50, 10, 100)
    y_pred = y_true + np.random.normal(0, 5, 100)  # Add some noise
    
    # Calculate losses using our implementation
    loss_calc = LossFunctions()
    
    mae_value = loss_calc.mae(y_true, y_pred)
    mse_value = loss_calc.mse(y_true, y_pred)
    rmse_value = loss_calc.rmse(y_true, y_pred)
    
    print("Custom Implementation Results:")
    print(f"MAE: {mae_value:.4f}")
    print(f"MSE: {mse_value:.4f}")
    print(f"RMSE: {rmse_value:.4f}")
    
    # Compare with sklearn
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    sklearn_mae = mean_absolute_error(y_true, y_pred)
    sklearn_mse = mean_squared_error(y_true, y_pred)
    sklearn_rmse = np.sqrt(sklearn_mse)
    
    print("\nSklearn Results:")
    print(f"MAE: {sklearn_mae:.4f}")
    print(f"MSE: {sklearn_mse:.4f}")
    print(f"RMSE: {sklearn_rmse:.4f}")
    
    print(f"\nDifferences (should be ~0):")
    print(f"MAE diff: {abs(mae_value - sklearn_mae):.10f}")
    print(f"MSE diff: {abs(mse_value - sklearn_mse):.10f}")
    print(f"RMSE diff: {abs(rmse_value - sklearn_rmse):.10f}")
    
    # Demonstrate gradient calculation
    mae_grad = loss_calc.mae_gradient(y_true, y_pred)
    mse_grad = loss_calc.mse_gradient(y_true, y_pred)
    
    print(f"\nGradient Statistics:")
    print(f"MAE gradient mean: {np.mean(mae_grad):.6f}")
    print(f"MSE gradient mean: {np.mean(mse_grad):.6f}")
    print(f"MAE gradient std: {np.std(mae_grad):.6f}")
    print(f"MSE gradient std: {np.std(mse_grad):.6f}")

# Run demonstration
demonstrate_loss_functions()
```

### Robust Loss Function Implementation

```python
class RobustLossFunctions:
    """
    Enhanced implementation with robust error handling and additional metrics
    """
    
    def __init__(self):
        self.history = []
    
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Calculate comprehensive error metrics
        
        Returns:
            Dictionary containing all metrics
        """
        errors = y_true - y_pred
        abs_errors = np.abs(errors)
        squared_errors = errors ** 2
        
        metrics = {
            'mae': np.mean(abs_errors),
            'mse': np.mean(squared_errors),
            'rmse': np.sqrt(np.mean(squared_errors)),
            'median_ae': np.median(abs_errors),
            'max_error': np.max(abs_errors),
            'mean_error': np.mean(errors),  # Bias
            'std_error': np.std(errors),
            'r2_score': self._r2_score(y_true, y_pred)
        }
        
        self.history.append(metrics)
        return metrics
    
    def _r2_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R² score"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2
    
    def compare_with_baseline(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            baseline_pred: np.ndarray) -> dict:
        """
        Compare model performance with a baseline
        
        Args:
            y_true: True values
            y_pred: Model predictions
            baseline_pred: Baseline predictions (e.g., mean prediction)
            
        Returns:
            Comparison metrics
        """
        model_metrics = self.calculate_all_metrics(y_true, y_pred)
        baseline_metrics = self.calculate_all_metrics(y_true, baseline_pred)
        
        comparison = {}
        for metric in model_metrics:
            if metric in ['r2_score']:  # Higher is better
                improvement = model_metrics[metric] - baseline_metrics[metric]
            else:  # Lower is better
                improvement = baseline_metrics[metric] - model_metrics[metric]
            
            comparison[f'{metric}_improvement'] = improvement
            comparison[f'{metric}_improvement_pct'] = (improvement / baseline_metrics[metric]) * 100
        
        return comparison
```

## ⚠️ Assumptions and Limitations

### Assumptions

**MAE:**
- All prediction errors are equally important
- Outliers should not dominate the loss function
- The cost of over-prediction equals the cost of under-prediction
- Linear penalty structure is appropriate for the problem

**RMSE:**
- Larger errors are more problematic and should be penalized heavily
- The relationship between error magnitude and penalty should be quadratic
- Gaussian error distribution is assumed (for probabilistic interpretation)
- Differentiability of loss function is required for optimization

### Limitations

**MAE Limitations:**
- **Non-differentiable at zero**: Makes gradient-based optimization challenging
- **Equal weighting**: May not reflect real-world cost structures where large errors are disproportionately costly
- **Slower convergence**: Constant gradients can lead to slower optimization
- **Less sensitive to small improvements**: May not distinguish between models with similar performance

**RMSE Limitations:**
- **Outlier sensitivity**: Few extreme values can dominate the loss
- **Unit dependency**: Values are affected by the scale of the target variable
- **Overemphasis on large errors**: May ignore many small errors
- **Not robust**: Performance degrades significantly with outliers

### Comparison with Other Loss Functions

| Loss Function | Outlier Sensitivity | Differentiability | Interpretability | Use Case |
|---------------|-------------------|-------------------|------------------|-----------|
| **MAE** | Low | No (at 0) | High | Robust regression |
| **RMSE/MSE** | High | Yes | Medium | Standard regression |
| **Huber** | Medium | Yes | Medium | Robust with smoothness |
| **Quantile** | Variable | Yes | High | Risk-aware prediction |

### When to Use Which

**Use MAE when:**
- Outliers are present and shouldn't dominate
- All errors are equally costly
- You need robust estimates
- Interpretability is crucial
- Working with heavy-tailed distributions

**Use RMSE when:**
- Large errors are more problematic
- You have clean data without extreme outliers
- Need smooth gradients for optimization
- Working with Gaussian-like distributions
- Standard benchmarking is required

## ❓ Interview Questions

??? question "**Q1: What is the main difference between MAE and RMSE in terms of outlier sensitivity?**"

    **Answer:** 
    
    MAE (L1 loss) is less sensitive to outliers because it uses absolute values, giving equal weight to all errors. Each error contributes linearly to the total loss.
    
    RMSE (L2 loss) is highly sensitive to outliers because it squares the errors before averaging. This means large errors are penalized exponentially more than small errors. A single large outlier can dominate the entire loss value.
    
    **Mathematical example:**
    - Normal errors: [1, 1, 1, 1, 1]  MAE = 1, RMSE = 1
    - With outlier: [1, 1, 1, 1, 10]  MAE = 2.8, RMSE = 4.6
    
    The outlier causes RMSE to increase by 4.6x while MAE only increases by 2.8x.

??? question "**Q2: Why is MAE non-differentiable at zero and how does this affect optimization?**"

    **Answer:**
    
    MAE uses the absolute value function |x|, which has a sharp corner at x=0. At this point, the left derivative is -1 and the right derivative is +1, making the function non-differentiable.
    
    **Impact on optimization:**
    - Gradient-based optimizers (SGD, Adam) struggle near zero error
    - Can cause oscillations around the optimal solution
    - May require smaller learning rates
    - Subgradient methods or smoothed versions (like Huber loss) are often used instead
    
    **Solutions:**
    - Use subgradient descent
    - Implement Huber loss (smooth approximation)
    - Use specialized optimizers designed for non-smooth functions

??? question "**Q3: In what scenarios would you choose MAE over RMSE for model evaluation?**"

    **Answer:**
    
    **Choose MAE when:**
    1. **Outliers are present:** MAE provides more robust evaluation
    2. **Equal error costs:** All prediction errors have the same business impact
    3. **Heavy-tailed distributions:** Data doesn't follow normal distribution
    4. **Interpretability matters:** MAE is in the same units as the target variable
    5. **Median-based predictions:** MAE aligns with median-based models
    
    **Real-world examples:**
    - **Sales forecasting:** Missing by $100 or $1000 might have similar operational impact
    - **Medical dosage:** All dosage errors are equally concerning
    - **Robust regression:** When data contains measurement errors or anomalies

??? question "**Q4: How do MAE and RMSE relate to different types of statistical estimators?**"

    **Answer:**
    
    **MAE (L1 loss):**
    - Corresponds to **median** estimator
    - Minimizing MAE gives the median of the target distribution
    - Robust to outliers, represents the "typical" error
    - Related to Laplace distribution assumption
    
    **RMSE/MSE (L2 loss):**
    - Corresponds to **mean** estimator  
    - Minimizing MSE gives the mean of the target distribution
    - Optimal under Gaussian noise assumption
    - Related to maximum likelihood estimation for normal distribution
    
    **Practical implication:** If your target variable is skewed or has outliers, MAE might give more representative results than RMSE.

??? question "**Q5: How do you implement a custom loss function that combines MAE and RMSE?**"

    **Answer:**
    
    **Huber Loss** is a common combination that transitions from MAE to MSE:
    
    ```python
    def huber_loss(y_true, y_pred, delta=1.0):
        errors = y_true - y_pred
        abs_errors = np.abs(errors)
        
        # Use MSE for small errors, MAE for large errors
        mask = abs_errors <= delta
        
        loss = np.where(mask, 
                       0.5 * errors**2,  # MSE region
                       delta * abs_errors - 0.5 * delta**2)  # MAE region
        
        return np.mean(loss)
    ```
    
    **Other combinations:**
    - **Weighted combination:** `± * MAE + (1-±) * MSE`
    - **Log-cosh loss:** `log(cosh(y_pred - y_true))`
    - **Quantile loss:** For asymmetric penalty structures

??? question "**Q6: What is the relationship between RMSE and standard deviation?**"

    **Answer:**
    
    RMSE and standard deviation are closely related but serve different purposes:
    
    **Similarities:**
    - Both use squared differences
    - Both are in the same units as the original data
    - Both penalize large deviations more heavily
    
    **Key differences:**
    - **RMSE:** Measures prediction error (predicted vs actual)
    - **Standard deviation:** Measures variability around the mean
    - **RMSE formula:** `(£(y_actual - y_pred)²/n)`
    - **Std dev formula:** `(£(x - ¼)²/n)`
    
    **Interpretation:** RMSE can be thought of as the "standard deviation of prediction errors."

??? question "**Q7: How do you handle the scale dependency of MAE and RMSE?**"

    **Answer:**
    
    Both MAE and RMSE are affected by the scale of the target variable, making cross-dataset comparison difficult.
    
    **Solutions:**
    
    1. **Normalized RMSE (NRMSE):**
       ```python
       nrmse = rmse / (y_max - y_min)  # or / y_mean
       ```
    
    2. **Mean Absolute Percentage Error (MAPE):**
       ```python
       mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
       ```
    
    3. **Coefficient of Variation of RMSE:**
       ```python
       cv_rmse = rmse / np.mean(y_true)
       ```
    
    4. **Standardized errors:**
       ```python
       standardized_mae = mae / np.std(y_true)
       ```

??? question "**Q8: What are the computational complexities of MAE and RMSE?**"

    **Answer:**
    
    **Time Complexity:**
    - **MAE:** O(n) - linear scan through errors, absolute value operation
    - **RMSE:** O(n) - linear scan through errors, square and square root operations
    
    **Space Complexity:**
    - Both: O(1) additional space (can compute incrementally)
    - Or O(n) if storing all errors for analysis
    
    **Computational considerations:**
    - **RMSE** requires more expensive operations (squaring, square root)
    - **MAE** operations are simpler but may require specialized handling for gradients
    - Both can be computed incrementally for streaming data
    - Vectorized implementations (NumPy, GPU) make the difference negligible

??? question "**Q9: How do MAE and RMSE behave differently during model training?**"

    **Answer:**
    
    **Convergence patterns:**
    - **RMSE-based training:** Smooth convergence, large errors get immediate attention
    - **MAE-based training:** Can have slower convergence due to constant gradients
    
    **Learning dynamics:**
    - **RMSE:** Model focuses on reducing largest errors first
    - **MAE:** Model treats all errors equally, more balanced learning
    
    **Practical implications:**
    ```python
    # RMSE training might show:
    # Epoch 1: Large errors dominate loss
    # Epoch 50: Focuses on medium errors  
    # Epoch 100: Fine-tuning small errors
    
    # MAE training might show:
    # More consistent error reduction across all samples
    # Less dramatic early improvements
    # Better final performance on median metrics
    ```

??? question "**Q10: What are some advanced variants and alternatives to standard MAE and RMSE?**"

    **Answer:**
    
    **Advanced variants:**
    
    1. **Weighted MAE/RMSE:** Different weights for different samples
       ```python
       weighted_mae = np.mean(weights * np.abs(y_true - y_pred))
       ```
    
    2. **Trimmed MAE/RMSE:** Remove extreme values before calculation
    
    3. **Quantile Loss:** Asymmetric loss function
       ```python
       def quantile_loss(y_true, y_pred, quantile=0.5):
           errors = y_true - y_pred
           return np.mean(np.maximum(quantile * errors, (quantile - 1) * errors))
       ```
    
    4. **Log-cosh Loss:** Smooth approximation of MAE
       ```python
       def log_cosh_loss(y_true, y_pred):
           return np.mean(np.log(np.cosh(y_pred - y_true)))
       ```
    
    5. **Huber Loss:** Combines MAE and MSE benefits
    
    6. **Fair Loss:** Less sensitive to outliers than MSE, smoother than MAE

## 💡 Examples

### Real-world Example: House Price Prediction

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load California housing dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Add some artificial outliers to demonstrate difference
np.random.seed(42)
outlier_indices = np.random.choice(len(y), size=50, replace=False)
y_with_outliers = y.copy()
y_with_outliers[outlier_indices] *= 5  # Make some house prices 5x higher

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_with_outliers, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train different models
lr_model = LinearRegression()
huber_model = HuberRegressor(epsilon=1.5)

lr_model.fit(X_train_scaled, y_train)
huber_model.fit(X_train_scaled, y_train)

# Make predictions
lr_pred = lr_model.predict(X_test_scaled)
huber_pred = huber_model.predict(X_test_scaled)

# Calculate metrics
def calculate_metrics(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"\n{model_name} Results:")
    print(f"MAE: ${mae:.3f}k")
    print(f"MSE: ${mse:.3f}k²")
    print(f"RMSE: ${rmse:.3f}k")
    
    return mae, mse, rmse

# Compare models
lr_metrics = calculate_metrics(y_test, lr_pred, "Linear Regression")
huber_metrics = calculate_metrics(y_test, huber_pred, "Huber Regression")

# Visualization
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(y_test, lr_pred, alpha=0.6, label='Linear Regression')
plt.scatter(y_test, huber_pred, alpha=0.6, label='Huber Regression')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('True House Price ($100k)')
plt.ylabel('Predicted House Price ($100k)')
plt.title('Predictions vs True Values')
plt.legend()

plt.subplot(1, 3, 2)
lr_errors = y_test - lr_pred
huber_errors = y_test - huber_pred
plt.hist(lr_errors, bins=50, alpha=0.7, label='Linear Regression', density=True)
plt.hist(huber_errors, bins=50, alpha=0.7, label='Huber Regression', density=True)
plt.xlabel('Prediction Error ($100k)')
plt.ylabel('Density')
plt.title('Error Distribution')
plt.legend()

plt.subplot(1, 3, 3)
models = ['Linear Regression', 'Huber Regression']
mae_values = [lr_metrics[0], huber_metrics[0]]
rmse_values = [lr_metrics[2], huber_metrics[2]]

x = np.arange(len(models))
width = 0.35

plt.bar(x - width/2, mae_values, width, label='MAE', alpha=0.8)
plt.bar(x + width/2, rmse_values, width, label='RMSE', alpha=0.8)

plt.xlabel('Models')
plt.ylabel('Error ($100k)')
plt.title('MAE vs RMSE Comparison')
plt.xticks(x, models)
plt.legend()

plt.tight_layout()
plt.show()

# Outlier analysis
outlier_mask = np.abs(y_test - lr_pred) > np.percentile(np.abs(y_test - lr_pred), 95)
print(f"\nOutlier Analysis (top 5% errors):")
print(f"Number of outlier predictions: {np.sum(outlier_mask)}")
print(f"MAE on outliers: ${mean_absolute_error(y_test[outlier_mask], lr_pred[outlier_mask]):.3f}k")
print(f"MAE on non-outliers: ${mean_absolute_error(y_test[~outlier_mask], lr_pred[~outlier_mask]):.3f}k")
```

### Time Series Forecasting Example

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Generate synthetic time series data
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=365, freq='D')

# Create trend + seasonality + noise
trend = np.linspace(100, 120, 365)
seasonality = 10 * np.sin(2 * np.pi * np.arange(365) / 365.25 * 4)  # Quarterly pattern
noise = np.random.normal(0, 2, 365)
outliers = np.zeros(365)
outliers[100] = 20  # Add a large outlier
outliers[200] = -15  # Add a negative outlier

time_series = trend + seasonality + noise + outliers

# Simple moving average prediction
window = 30
predictions = []
actuals = []

for i in range(window, len(time_series)):
    # Predict using simple moving average
    pred = np.mean(time_series[i-window:i])
    predictions.append(pred)
    actuals.append(time_series[i])

predictions = np.array(predictions)
actuals = np.array(actuals)

# Calculate rolling metrics
window_size = 30
rolling_mae = []
rolling_rmse = []

for i in range(window_size, len(predictions)):
    window_actuals = actuals[i-window_size:i]
    window_preds = predictions[i-window_size:i]
    
    mae = mean_absolute_error(window_actuals, window_preds)
    rmse = np.sqrt(mean_squared_error(window_actuals, window_preds))
    
    rolling_mae.append(mae)
    rolling_rmse.append(rmse)

# Visualization
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(dates[window:], actuals, label='Actual', alpha=0.7)
plt.plot(dates[window:], predictions, label='Predicted (MA)', alpha=0.7)
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Time Series Forecasting')
plt.legend()

plt.subplot(2, 2, 2)
errors = actuals - predictions
plt.plot(dates[window:], errors, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Prediction Error')
plt.title('Prediction Errors Over Time')

plt.subplot(2, 2, 3)
plt.plot(dates[window+window_size:], rolling_mae, label='Rolling MAE', linewidth=2)
plt.plot(dates[window+window_size:], rolling_rmse, label='Rolling RMSE', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Error Metric')
plt.title('Rolling Error Metrics')
plt.legend()

plt.subplot(2, 2, 4)
plt.hist(errors, bins=30, alpha=0.7, density=True)
plt.axvline(x=np.mean(errors), color='r', linestyle='--', label=f'Mean: {np.mean(errors):.2f}')
plt.axvline(x=np.median(errors), color='g', linestyle='--', label=f'Median: {np.median(errors):.2f}')
plt.xlabel('Prediction Error')
plt.ylabel('Density')
plt.title('Error Distribution')
plt.legend()

plt.tight_layout()
plt.show()

# Summary statistics
print("Time Series Forecasting Results:")
print(f"Overall MAE: {mean_absolute_error(actuals, predictions):.3f}")
print(f"Overall RMSE: {np.sqrt(mean_squared_error(actuals, predictions)):.3f}")
print(f"Mean Error (Bias): {np.mean(errors):.3f}")
print(f"Std Error: {np.std(errors):.3f}")
print(f"Max Error: {np.max(np.abs(errors)):.3f}")
print(f"Median Absolute Error: {np.median(np.abs(errors)):.3f}")
```

## 📚 References

**Books:**
- [The Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/) - Hastie, Tibshirani, Friedman
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) - Christopher Bishop
- [Deep Learning](https://www.deeplearningbook.org/) - Ian Goodfellow, Yoshua Bengio, Aaron Courville

**Academic Papers:**
- [Huber Loss Function](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-35/issue-1/Robust-Estimation-of-a-Location-Parameter/10.1214/aoms/1177703732.full) - Peter Huber (1964)
- [Quantile Regression](https://www.jstor.org/stable/1913643) - Roger Koenker and Gilbert Bassett (1978)

**Online Resources:**
- [Scikit-learn Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [TensorFlow Loss Functions](https://www.tensorflow.org/api_docs/python/tf/keras/losses)
- [PyTorch Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions)
- [Loss Functions for Regression](https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0)

**Tutorials and Blogs:**
- [Understanding Different Loss Functions](https://towardsdatascience.com/understanding-different-loss-functions-for-neural-networks-dd1ed0274718)
- [MAE vs RMSE vs MAPE](https://medium.com/analytics-vidhya/mae-mse-rmse-coefficient-of-determination-adjusted-r-squared-which-metric-is-better-cd0326a5697e)
- [Robust Loss Functions for Deep Learning](https://towardsdatascience.com/robust-loss-functions-for-deep-learning-a7d2937b7e71)