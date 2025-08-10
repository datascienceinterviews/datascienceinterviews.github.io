---
title: Linear Regression
description: Comprehensive guide to Linear Regression with mathematical intuition, implementations, and interview questions.
comments: true
---

# 📘 Linear Regression

Linear Regression is a fundamental supervised learning algorithm that models the linear relationship between a dependent variable and one or more independent variables by finding the best-fitting straight line through the data points.

**Resources:** [Scikit-learn Linear Regression](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares) | [Stanford CS229 Notes](http://cs229.stanford.edu/notes/cs229-notes1.pdf) | [ISL Chapter 3](https://www.statlearning.com/)

## ✍️ Summary

Linear Regression is the simplest and most widely used regression technique that assumes a linear relationship between input features and the target variable. It aims to find the best line (or hyperplane in higher dimensions) that minimizes the sum of squared differences between actual and predicted values.

**Key characteristics:**
- **Simplicity**: Easy to understand and implement
- **Interpretability**: Coefficients have clear meaning
- **Fast**: Quick to train and predict
- **Baseline**: Often used as a starting point for regression problems
- **Probabilistic**: Provides confidence intervals and statistical tests

**Applications:**
- Predicting house prices based on features
- Sales forecasting from marketing spend
- Risk assessment in finance
- Medical diagnosis and treatment effects
- Economics and business analytics
- Scientific research and hypothesis testing

**Types:**
- **Simple Linear Regression**: One independent variable
- **Multiple Linear Regression**: Multiple independent variables
- **Polynomial Regression**: Non-linear relationships using polynomial features
- **Regularized Regression**: Ridge, Lasso, and Elastic Net

## 🧠 Intuition

### How Linear Regression Works

Imagine you're trying to predict house prices based on their size. Linear regression finds the straight line that best fits through all the data points, minimizing the overall prediction error. This line can then be used to predict prices for new houses.

### Mathematical Foundation

#### 1. Simple Linear Regression

For one feature, the model is:
$$y = \beta_0 + \beta_1 x + \epsilon$$

Where:
- $y$ is the dependent variable (target)
- $x$ is the independent variable (feature)
- $\beta_0$ is the intercept (y-intercept)
- $\beta_1$ is the slope (coefficient)
- $\epsilon$ is the error term

#### 2. Multiple Linear Regression

For multiple features:
$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n + \epsilon$$

In matrix form:
$$\mathbf{y} = \mathbf{X\beta} + \boldsymbol{\epsilon}$$

Where:
- $\mathbf{y}$ is the target vector $(n \times 1)$
- $\mathbf{X}$ is the feature matrix $(n \times p)$ with bias column
- $\boldsymbol{\beta}$ is the coefficient vector $(p \times 1)$
- $\boldsymbol{\epsilon}$ is the error vector $(n \times 1)$

#### 3. Cost Function (Mean Squared Error)

$$J(\boldsymbol{\beta}) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\boldsymbol{\beta}}(\mathbf{x}^{(i)}) - y^{(i)})^2$$

Or in matrix form:
$$J(\boldsymbol{\beta}) = \frac{1}{2m} (\mathbf{X\beta} - \mathbf{y})^T(\mathbf{X\beta} - \mathbf{y})$$

#### 4. Normal Equation (Closed-form Solution)

The optimal coefficients can be found analytically:
$$\boldsymbol{\beta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

#### 5. Gradient Descent (Iterative Solution)

Update rule for coefficients:
$$\beta_j := \beta_j - \alpha \frac{\partial}{\partial \beta_j} J(\boldsymbol{\beta})$$

The gradient is:
$$\frac{\partial J}{\partial \boldsymbol{\beta}} = \frac{1}{m} \mathbf{X}^T(\mathbf{X\beta} - \mathbf{y})$$

### Key Assumptions

1. **Linearity**: Relationship between features and target is linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance of errors
4. **Normality**: Errors are normally distributed
5. **No multicollinearity**: Features are not highly correlated

## 🔢 Implementation using Libraries

### Using Scikit-learn

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Generate sample data
X, y = make_regression(
    n_samples=1000,
    n_features=1,
    noise=20,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions
y_pred = lr_model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.3f}")
print(f"Intercept: {lr_model.intercept_:.2f}")
print(f"Coefficient: {lr_model.coef_[0]:.2f}")

# Visualize results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(X_test, y_test, alpha=0.6, label='Actual')
plt.scatter(X_test, y_pred, alpha=0.6, label='Predicted')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Linear Regression Fit')
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Predictions vs Actual (R² = {r2:.3f})')

plt.subplot(1, 3, 3)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')

plt.tight_layout()
plt.show()

# Multiple Linear Regression Example
# Load Boston housing dataset
boston = load_boston()
X_multi, y_multi = boston.data, boston.target
feature_names = boston.feature_names

# Split data
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42
)

# Scale features for better interpretation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_multi)
X_test_scaled = scaler.transform(X_test_multi)

# Train model
lr_multi = LinearRegression()
lr_multi.fit(X_train_scaled, y_train_multi)

# Predictions
y_pred_multi = lr_multi.predict(X_test_scaled)

# Metrics
r2_multi = r2_score(y_test_multi, y_pred_multi)
rmse_multi = np.sqrt(mean_squared_error(y_test_multi, y_pred_multi))

print(f"\nMultiple Linear Regression Results:")
print(f"R² Score: {r2_multi:.3f}")
print(f"RMSE: {rmse_multi:.2f}")

# Feature coefficients analysis
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': lr_multi.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nFeature Coefficients (scaled):")
print(coef_df)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(coef_df['Feature'], coef_df['Coefficient'])
plt.xlabel('Coefficient Value')
plt.title('Linear Regression Coefficients')
plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
plt.show()
```

### Using StatsModels for Statistical Analysis

```python
import statsmodels.api as sm
from scipy import stats

# Prepare data with intercept
X_with_intercept = sm.add_constant(X_train_multi)
X_test_with_intercept = sm.add_constant(X_test_multi)

# Fit OLS model
ols_model = sm.OLS(y_train_multi, X_with_intercept).fit()

# Print comprehensive summary
print("OLS Regression Results:")
print(ols_model.summary())

# Predictions with confidence intervals
predictions = ols_model.get_prediction(X_test_with_intercept)
pred_summary = predictions.summary_frame(alpha=0.05)

print("\nPredictions with Confidence Intervals (first 5):")
print(pred_summary.head())

# Statistical tests
print(f"\nModel Statistics:")
print(f"F-statistic: {ols_model.fvalue:.2f}")
print(f"F-statistic p-value: {ols_model.f_pvalue:.2e}")
print(f"AIC: {ols_model.aic:.2f}")
print(f"BIC: {ols_model.bic:.2f}")

# Residual analysis
residuals = ols_model.resid
fitted_values = ols_model.fittedvalues

# Diagnostic plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Q-Q plot
stats.probplot(residuals, dist="norm", plot=axes[0,0])
axes[0,0].set_title("Q-Q Plot")

# Residuals vs Fitted
axes[0,1].scatter(fitted_values, residuals, alpha=0.6)
axes[0,1].axhline(y=0, color='r', linestyle='--')
axes[0,1].set_xlabel('Fitted Values')
axes[0,1].set_ylabel('Residuals')
axes[0,1].set_title('Residuals vs Fitted')

# Histogram of residuals
axes[1,0].hist(residuals, bins=20, alpha=0.7)
axes[1,0].set_xlabel('Residuals')
axes[1,0].set_ylabel('Frequency')
axes[1,0].set_title('Histogram of Residuals')

# Scale-Location plot
standardized_residuals = np.sqrt(np.abs(residuals / np.std(residuals)))
axes[1,1].scatter(fitted_values, standardized_residuals, alpha=0.6)
axes[1,1].set_xlabel('Fitted Values')
axes[1,1].set_ylabel('√|Standardized Residuals|')
axes[1,1].set_title('Scale-Location Plot')

plt.tight_layout()
plt.show()
```

## ⚙️ From Scratch Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

class LinearRegressionScratch:
    """
    Linear Regression implementation from scratch using both
    Normal Equation and Gradient Descent methods.
    """
    
    def __init__(self, method='normal_equation', learning_rate=0.01, n_iterations=1000):
        """
        Initialize Linear Regression.
        
        Parameters:
        -----------
        method : str, 'normal_equation' or 'gradient_descent'
        learning_rate : float, learning rate for gradient descent
        n_iterations : int, number of iterations for gradient descent
        """
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.coefficients = None
        self.intercept = None
        self.cost_history = []
        
    def add_intercept(self, X):
        """Add bias column to the feature matrix."""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def fit(self, X, y):
        """
        Fit linear regression model.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
        y : array-like, shape = [n_samples]
        """
        # Ensure y is a column vector
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            
        # Add intercept term
        X_with_intercept = self.add_intercept(X)
        
        if self.method == 'normal_equation':
            self._fit_normal_equation(X_with_intercept, y)
        elif self.method == 'gradient_descent':
            self._fit_gradient_descent(X_with_intercept, y)
        else:
            raise ValueError("Method must be 'normal_equation' or 'gradient_descent'")
    
    def _fit_normal_equation(self, X, y):
        """Fit using normal equation: β = (X^T X)^(-1) X^T y"""
        try:
            # Normal equation
            XtX = np.dot(X.T, X)
            XtX_inv = np.linalg.inv(XtX)
            Xty = np.dot(X.T, y)
            theta = np.dot(XtX_inv, Xty)
            
            self.intercept = theta[0, 0]
            self.coefficients = theta[1:].flatten()
            
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if matrix is singular
            theta = np.dot(np.linalg.pinv(X), y)
            self.intercept = theta[0, 0]
            self.coefficients = theta[1:].flatten()
    
    def _fit_gradient_descent(self, X, y):
        """Fit using gradient descent."""
        m, n = X.shape
        
        # Initialize parameters
        theta = np.zeros((n, 1))
        
        for i in range(self.n_iterations):
            # Forward pass
            predictions = np.dot(X, theta)
            
            # Compute cost
            cost = self._compute_cost(predictions, y)
            self.cost_history.append(cost)
            
            # Compute gradients
            gradients = (1/m) * np.dot(X.T, (predictions - y))
            
            # Update parameters
            theta = theta - self.learning_rate * gradients
        
        self.intercept = theta[0, 0]
        self.coefficients = theta[1:].flatten()
    
    def _compute_cost(self, predictions, y):
        """Compute mean squared error cost."""
        m = y.shape[0]
        cost = (1/(2*m)) * np.sum(np.power(predictions - y, 2))
        return cost
    
    def predict(self, X):
        """
        Make predictions using the fitted model.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
        
        Returns:
        --------
        predictions : array-like, shape = [n_samples]
        """
        return np.dot(X, self.coefficients) + self.intercept
    
    def score(self, X, y):
        """Calculate R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def get_params(self):
        """Get model parameters."""
        return {
            'intercept': self.intercept,
            'coefficients': self.coefficients,
            'cost_history': self.cost_history
        }

# Example usage and comparison
if __name__ == "__main__":
    # Generate sample data
    X, y = make_regression(n_samples=1000, n_features=3, noise=10, random_state=42)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Our implementation - Normal Equation
    lr_normal = LinearRegressionScratch(method='normal_equation')
    lr_normal.fit(X_train, y_train)
    y_pred_normal = lr_normal.predict(X_test)
    r2_normal = lr_normal.score(X_test, y_test)
    
    # Our implementation - Gradient Descent
    lr_gd = LinearRegressionScratch(method='gradient_descent', learning_rate=0.01, n_iterations=1000)
    lr_gd.fit(X_train, y_train)
    y_pred_gd = lr_gd.predict(X_test)
    r2_gd = lr_gd.score(X_test, y_test)
    
    # Sklearn for comparison
    from sklearn.linear_model import LinearRegression
    sklearn_lr = LinearRegression()
    sklearn_lr.fit(X_train, y_train)
    y_pred_sklearn = sklearn_lr.predict(X_test)
    r2_sklearn = sklearn_lr.score(X_test, y_test)
    
    # Compare results
    print("Comparison of Implementations:")
    print(f"Normal Equation R²: {r2_normal:.6f}")
    print(f"Gradient Descent R²: {r2_gd:.6f}")
    print(f"Sklearn R²: {r2_sklearn:.6f}")
    
    print(f"\nIntercept comparison:")
    print(f"Normal Equation: {lr_normal.intercept:.6f}")
    print(f"Gradient Descent: {lr_gd.intercept:.6f}")
    print(f"Sklearn: {sklearn_lr.intercept_:.6f}")
    
    print(f"\nCoefficients comparison:")
    print(f"Normal Equation: {lr_normal.coefficients}")
    print(f"Gradient Descent: {lr_gd.coefficients}")
    print(f"Sklearn: {sklearn_lr.coef_}")
    
    # Plot cost history for gradient descent
    if lr_gd.cost_history:
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(lr_gd.cost_history)
        plt.title('Cost Function During Training')
        plt.xlabel('Iterations')
        plt.ylabel('Cost (MSE)')
        
        plt.subplot(1, 2, 2)
        plt.scatter(y_test, y_pred_normal, alpha=0.6, label='Normal Equation')
        plt.scatter(y_test, y_pred_gd, alpha=0.6, label='Gradient Descent')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predictions Comparison')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

# Polynomial Regression from scratch
class PolynomialRegressionScratch:
    """Polynomial Regression using our Linear Regression implementation."""
    
    def __init__(self, degree=2, method='normal_equation'):
        self.degree = degree
        self.linear_regression = LinearRegressionScratch(method=method)
        
    def _create_polynomial_features(self, X):
        """Create polynomial features up to specified degree."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        n_samples, n_features = X.shape
        
        # Start with original features
        poly_features = X.copy()
        
        # Add polynomial terms
        for deg in range(2, self.degree + 1):
            for feature_idx in range(n_features):
                poly_feature = np.power(X[:, feature_idx], deg).reshape(-1, 1)
                poly_features = np.concatenate([poly_features, poly_feature], axis=1)
        
        return poly_features
    
    def fit(self, X, y):
        """Fit polynomial regression."""
        X_poly = self._create_polynomial_features(X)
        self.linear_regression.fit(X_poly, y)
        
    def predict(self, X):
        """Make predictions."""
        X_poly = self._create_polynomial_features(X)
        return self.linear_regression.predict(X_poly)
    
    def score(self, X, y):
        """Calculate R² score."""
        X_poly = self._create_polynomial_features(X)
        return self.linear_regression.score(X_poly, y)

# Test polynomial regression
if __name__ == "__main__":
    # Generate non-linear data
    np.random.seed(42)
    X_poly = np.linspace(-2, 2, 100).reshape(-1, 1)
    y_poly = 0.5 * X_poly.ravel()**3 - 2 * X_poly.ravel()**2 + X_poly.ravel() + np.random.normal(0, 0.5, 100)
    
    # Fit polynomial regression
    poly_reg = PolynomialRegressionScratch(degree=3)
    poly_reg.fit(X_poly, y_poly)
    
    # Predictions
    X_plot = np.linspace(-2, 2, 300).reshape(-1, 1)
    y_pred_poly = poly_reg.predict(X_plot)
    
    # Plot results
    plt.figure(figsize=(8, 6))
    plt.scatter(X_poly, y_poly, alpha=0.6, label='Data')
    plt.plot(X_plot, y_pred_poly, color='red', linewidth=2, label='Polynomial Fit (degree=3)')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Polynomial Regression from Scratch')
    plt.legend()
    plt.show()
    
    r2_poly = poly_reg.score(X_poly, y_poly)
    print(f"Polynomial Regression R²: {r2_poly:.3f}")
```

## ⚠️ Assumptions and Limitations

### Assumptions

1. **Linearity**
   - Relationship between features and target is linear
   - **Check**: Scatter plots, residual plots
   - **Violation**: Use polynomial features or non-linear models

2. **Independence**
   - Observations are independent of each other
   - **Check**: Domain knowledge, autocorrelation tests
   - **Violation**: Use time series models or clustered standard errors

3. **Homoscedasticity**
   - Constant variance of residuals across all levels of features
   - **Check**: Residuals vs fitted values plot
   - **Violation**: Use weighted least squares or transform target variable

4. **Normality of Residuals**
   - Residuals should be normally distributed
   - **Check**: Q-Q plots, Shapiro-Wilk test
   - **Violation**: Transform variables or use robust regression

5. **No Multicollinearity**
   - Features should not be highly correlated
   - **Check**: Correlation matrix, VIF (Variance Inflation Factor)
   - **Violation**: Remove features, use regularization (Ridge/Lasso)

### Limitations

#### 1. **Linear Relationship Only**
- Cannot capture non-linear patterns without feature engineering
- **Solution**: Polynomial features, interaction terms, or non-linear models

#### 2. **Sensitive to Outliers**
- Outliers can significantly affect the regression line
- **Solution**: Robust regression, outlier detection and removal

#### 3. **Multicollinearity Issues**
- High correlation between features causes unstable coefficients
- **Solution**: Feature selection, regularization techniques

#### 4. **Overfitting with Many Features**
- Can overfit when number of features approaches number of samples
- **Solution**: Regularization (Ridge, Lasso), feature selection

#### 5. **Assumes Linear Relationship**
- May perform poorly on complex, non-linear datasets
- **Alternative**: Polynomial regression, kernel methods, tree-based models

### When to Use vs Avoid

**Use Linear Regression when:**
- Relationship appears linear
- Interpretability is important
- Need quick baseline model
- Small to medium datasets
- Features are not highly correlated
- Statistical inference is needed

**Avoid Linear Regression when:**
- Clear non-linear relationships exist
- Many irrelevant features present
- High multicollinearity among features
- Outliers are prevalent and cannot be removed
- Need high prediction accuracy over interpretability

## 💡 Interview Questions

??? question "1. Explain the difference between correlation and causation in the context of linear regression."

    **Answer:**
    - **Correlation**: Statistical relationship between variables; high correlation doesn't imply causation
    - **Causation**: One variable directly influences another
    - **In regression**: A significant coefficient shows correlation but not necessarily causation
    - **Example**: Ice cream sales and drowning deaths are correlated (both increase in summer) but ice cream doesn't cause drowning
    - **Establishing causation**: Requires randomized controlled experiments, domain expertise, and careful study design
    - **Confounding variables**: Can create spurious correlations that disappear when controlled for

??? question "2. What is the difference between R² and adjusted R²? When should you use each?"

    **Answer:**
    - **R²**: Proportion of variance in dependent variable explained by independent variables
      - Formula: $R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$
      - Always increases with more features
    - **Adjusted R²**: Penalizes for number of features
      - Formula: $R^2_{adj} = 1 - \frac{(1-R^2)(n-1)}{n-k-1}$
      - Can decrease if adding irrelevant features
    - **Use R²**: When comparing models with same number of features
    - **Use Adjusted R²**: When comparing models with different numbers of features
    - **Better metric**: Adjusted R² prevents overfitting by penalizing model complexity

??? question "3. How do you handle multicollinearity in linear regression?"

    **Answer:**
    **Detection methods:**
    - Correlation matrix (threshold > 0.8)
    - VIF (Variance Inflation Factor) > 5 or 10
    - Condition number > 30

    **Solutions:**
    1. **Remove highly correlated features**: Drop one from each correlated pair
    2. **Principal Component Analysis (PCA)**: Transform to orthogonal components
    3. **Ridge Regression**: L2 regularization reduces impact of multicollinearity
    4. **Feature combination**: Create new features by combining correlated ones
    5. **Domain knowledge**: Remove features that don't make business sense
    6. **Regularization**: Lasso can automatically select relevant features

??? question "4. Explain the normal equation vs gradient descent for linear regression. When would you use each?"

    **Answer:**
    **Normal Equation: $\beta = (X^TX)^{-1}X^Ty$**
    - **Advantages**: Exact solution, no hyperparameters, no iterations needed
    - **Disadvantages**: O(n³) complexity for matrix inversion, doesn't work if $X^TX$ is singular
    - **Use when**: Small datasets (n < 10,000), need exact solution

    **Gradient Descent:**
    - **Advantages**: Works with large datasets, O(kn²) per iteration, more memory efficient
    - **Disadvantages**: Requires hyperparameter tuning, may not converge, approximate solution
    - **Use when**: Large datasets (n > 10,000), online learning needed

    **Practical rule**: Normal equation for small datasets, gradient descent for large ones

??? question "5. What are the key assumptions of linear regression and how do you test them?"

    **Answer:**
    1. **Linearity**: 
       - Test: Scatter plots of features vs target, residual plots
       - Violation: Add polynomial terms or use non-linear models

    2. **Independence**:
       - Test: Domain knowledge, Durbin-Watson test for autocorrelation
       - Violation: Use time series models or account for clustering

    3. **Homoscedasticity**:
       - Test: Residuals vs fitted plot, Breusch-Pagan test
       - Violation: Use weighted least squares or log transformation

    4. **Normality of residuals**:
       - Test: Q-Q plots, Shapiro-Wilk test, histogram of residuals
       - Violation: Transform variables or use robust regression

    5. **No multicollinearity**:
       - Test: VIF > 5, correlation matrix
       - Violation: Remove features, use regularization

??? question "6. How do you interpret the coefficients in linear regression?"

    **Answer:**
    **For continuous variables:**
    - Coefficient represents change in target for one unit change in feature, holding other features constant
    - Example: If coefficient for 'years of experience' is 5000, each additional year increases salary by $5000

    **For categorical variables (dummy coded):**
    - Coefficient represents difference from reference category
    - Example: If 'gender_male' coefficient is 3000, males earn $3000 more than females (reference)

    **Important considerations:**
    - **Scale matters**: Larger-scale features have smaller coefficients
    - **Standardization**: Standardized coefficients allow comparison of feature importance
    - **Interaction effects**: Coefficients change meaning with interaction terms
    - **Confidence intervals**: Provide uncertainty estimates around coefficients

??? question "7. What is the bias-variance tradeoff in linear regression?"

    **Answer:**
    **Bias**: Error from overly simplistic assumptions
    - **High bias**: Model consistently misses relevant patterns
    - **In linear regression**: Assuming linear relationship when it's non-linear

    **Variance**: Error from sensitivity to small fluctuations in training set
    - **High variance**: Model changes significantly with different training data
    - **In linear regression**: Overfitting with too many features relative to data

    **Tradeoff**: 
    - Simple models (fewer features): High bias, low variance
    - Complex models (many features): Low bias, high variance
    - **Optimal point**: Minimizes total error = bias² + variance + irreducible error

    **Solutions**:
    - Cross-validation to find optimal complexity
    - Regularization (Ridge/Lasso) to balance bias-variance
    - More training data reduces variance

??? question "8. Compare Ridge, Lasso, and Elastic Net regression."

    **Answer:**

    | Aspect | Ridge (L2) | Lasso (L1) | Elastic Net |
    |--------|------------|------------|-------------|
    | **Penalty** | $\lambda \sum \beta_i^2$ | $\lambda \sum |\beta_i|$ | $\alpha \lambda \sum |\beta_i| + (1-\alpha) \lambda \sum \beta_i^2$ |
    | **Feature Selection** | No (shrinks to near 0) | Yes (shrinks to exactly 0) | Yes (selective) |
    | **Multicollinearity** | Handles well | Arbitrary selection | Handles well |
    | **Sparse Solutions** | No | Yes | Yes |
    | **Groups of correlated features** | Includes all | Picks one arbitrarily | Tends to include/exclude together |

    **When to use:**
    - **Ridge**: Multicollinearity, want to keep all features
    - **Lasso**: Feature selection, want sparse model
    - **Elastic Net**: Best of both, handles grouped variables well

??? question "9. How do you handle categorical variables in linear regression?"

    **Answer:**
    **Encoding methods:**

    1. **One-Hot Encoding (Dummy Variables)**:
       - Create binary columns for each category
       - Drop one category to avoid multicollinearity (dummy variable trap)
       - Example: Color {Red, Blue, Green} → Color_Red, Color_Blue (Green is reference)

    2. **Effect Coding**:
       - Similar to dummy coding but reference category coded as -1
       - Coefficients represent deviation from overall mean

    3. **Ordinal Encoding**:
       - For ordered categories (Low, Medium, High → 1, 2, 3)
       - Assumes linear relationship between categories

    **Considerations:**
    - **Reference category**: Choose meaningful baseline for interpretation
    - **High cardinality**: Use target encoding or frequency encoding
    - **Interaction effects**: May need to include interactions with other features
    - **Regularization**: Helps when many categories create many dummy variables

??? question "10. What evaluation metrics would you use for regression problems and why?"

    **Answer:**
    **Common metrics:**

    1. **Mean Absolute Error (MAE)**:
       - $MAE = \frac{1}{n}\sum|y_i - \hat{y}_i|$
       - **Pros**: Easy to interpret, robust to outliers
       - **Cons**: Not differentiable at zero

    2. **Mean Squared Error (MSE)**:
       - $MSE = \frac{1}{n}\sum(y_i - \hat{y}_i)^2$
       - **Pros**: Differentiable, penalizes large errors more
       - **Cons**: Sensitive to outliers, units are squared

    3. **Root Mean Squared Error (RMSE)**:
       - $RMSE = \sqrt{MSE}$
       - **Pros**: Same units as target, interpretable
       - **Cons**: Still sensitive to outliers

    4. **R² Score**:
       - $R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$
       - **Pros**: Scale-independent, easy to interpret (% variance explained)
       - **Cons**: Can be misleading with non-linear relationships

    **Choose based on:**
    - **Business context**: What type of errors are most costly?
    - **Outliers**: Use MAE if outliers present, RMSE if not
    - **Interpretability**: R² for general performance, RMSE for same-unit comparison

## 🧠 Examples

### Real-world Example: Sales Prediction

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# Create synthetic sales data
np.random.seed(42)
n_samples = 500

# Generate features
advertising_spend = np.random.normal(50, 20, n_samples)  # in thousands
temperature = np.random.normal(70, 15, n_samples)  # Fahrenheit
is_weekend = np.random.binomial(1, 0.3, n_samples)  # 30% weekends
season = np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'], n_samples)
competitor_price = np.random.normal(25, 5, n_samples)

# Create realistic sales relationship
base_sales = 100
sales = (base_sales + 
         2.5 * advertising_spend +  # Strong positive effect
         0.5 * temperature +        # Weather effect
         15 * is_weekend +          # Weekend boost
         -1.2 * competitor_price +  # Competition effect
         np.random.normal(0, 10, n_samples))  # Random noise

# Add seasonal effects
season_effects = {'Spring': 10, 'Summer': 20, 'Fall': 5, 'Winter': -15}
sales += np.array([season_effects[s] for s in season])

# Create DataFrame
sales_data = pd.DataFrame({
    'advertising_spend': advertising_spend,
    'temperature': temperature,
    'is_weekend': is_weekend,
    'season': season,
    'competitor_price': competitor_price,
    'sales': sales
})

print("Sales Dataset:")
print(sales_data.head())
print(f"\nDataset shape: {sales_data.shape}")
print("\nBasic statistics:")
print(sales_data.describe())

# One-hot encode categorical variables
sales_encoded = pd.get_dummies(sales_data, columns=['season'], prefix='season')

# Prepare features and target
X = sales_encoded.drop('sales', axis=1)
y = sales_encoded['sales']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions
y_pred = lr_model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"RMSE: ${rmse:.2f}")
print(f"R² Score: {r2:.3f}")

# Analyze coefficients
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr_model.coef_,
    'Abs_Coefficient': np.abs(lr_model.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

print(f"\nFeature Importance (Coefficients):")
print(coefficients)

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Actual vs Predicted
axes[0,0].scatter(y_test, y_pred, alpha=0.6)
axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0,0].set_xlabel('Actual Sales')
axes[0,0].set_ylabel('Predicted Sales')
axes[0,0].set_title(f'Predictions vs Actual (R² = {r2:.3f})')

# 2. Residual plot
residuals = y_test - y_pred
axes[0,1].scatter(y_pred, residuals, alpha=0.6)
axes[0,1].axhline(y=0, color='r', linestyle='--')
axes[0,1].set_xlabel('Predicted Sales')
axes[0,1].set_ylabel('Residuals')
axes[0,1].set_title('Residual Plot')

# 3. Feature importance
top_features = coefficients.head(8)
axes[1,0].barh(range(len(top_features)), top_features['Coefficient'])
axes[1,0].set_yticks(range(len(top_features)))
axes[1,0].set_yticklabels(top_features['Feature'])
axes[1,0].set_xlabel('Coefficient Value')
axes[1,0].set_title('Feature Coefficients')
axes[1,0].axvline(x=0, color='k', linestyle='--', alpha=0.5)

# 4. Sales vs Advertising relationship
axes[1,1].scatter(sales_data['advertising_spend'], sales_data['sales'], alpha=0.6)
axes[1,1].set_xlabel('Advertising Spend ($000)')
axes[1,1].set_ylabel('Sales')
axes[1,1].set_title('Sales vs Advertising Spend')

# Add trend line
z = np.polyfit(sales_data['advertising_spend'], sales_data['sales'], 1)
p = np.poly1d(z)
axes[1,1].plot(sales_data['advertising_spend'], p(sales_data['advertising_spend']), "r--", alpha=0.8)

plt.tight_layout()
plt.show()

# Business insights
print(f"\n=== Business Insights ===")
print(f"1. Every $1K in advertising spend increases sales by ${lr_model.coef_[0]:.2f}")
print(f"2. Weekend sales are ${lr_model.coef_[2]:.2f} higher than weekdays")
print(f"3. Each degree temperature increase adds ${coefficients[coefficients['Feature']=='temperature']['Coefficient'].iloc[0]:.2f} to sales")

# Prediction example
print(f"\n=== Sales Prediction Example ===")
example_data = np.array([[60, 75, 1, 20, 0, 0, 1, 0]])  # Summer weekend with high advertising
example_pred = lr_model.predict(example_data)[0]
print(f"Predicted sales for summer weekend with $60K advertising: ${example_pred:.2f}")

# Feature correlation analysis
correlation_matrix = X.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()
```

### Example: Medical Diagnosis - Drug Dosage Prediction

```python
# Generate synthetic medical data for drug dosage prediction
np.random.seed(123)
n_patients = 300

# Patient characteristics
age = np.random.normal(55, 15, n_patients)
weight = np.random.normal(70, 12, n_patients)  # kg
height = np.random.normal(170, 10, n_patients)  # cm
gender = np.random.binomial(1, 0.5, n_patients)  # 0=Female, 1=Male
kidney_function = np.random.normal(90, 20, n_patients)  # GFR
liver_function = np.random.normal(80, 15, n_patients)  # ALT levels

# Calculate BMI
bmi = weight / ((height/100)**2)

# Generate optimal dosage based on medical relationships
optimal_dosage = (
    5 +                           # Base dosage
    0.1 * age +                   # Age factor
    0.3 * weight +                # Weight-based dosing
    2 * gender +                  # Gender difference
    0.05 * kidney_function +      # Kidney clearance
    -0.02 * liver_function +      # Liver metabolism
    0.2 * bmi +                   # Body mass effect
    np.random.normal(0, 2, n_patients)  # Individual variation
)

# Ensure dosage is positive and reasonable
optimal_dosage = np.clip(optimal_dosage, 1, 50)

# Create medical dataset
medical_data = pd.DataFrame({
    'age': age,
    'weight': weight,
    'height': height,
    'gender': gender,
    'kidney_function': kidney_function,
    'liver_function': liver_function,
    'bmi': bmi,
    'optimal_dosage': optimal_dosage
})

print("Medical Dataset for Drug Dosage Prediction:")
print(medical_data.head())
print(f"\nDataset shape: {medical_data.shape}")

# Prepare data
X_med = medical_data.drop('optimal_dosage', axis=1)
y_med = medical_data['optimal_dosage']

# Split data
X_train_med, X_test_med, y_train_med, y_test_med = train_test_split(
    X_med, y_med, test_size=0.2, random_state=42
)

# Scale features for better interpretation
scaler_med = StandardScaler()
X_train_scaled_med = scaler_med.fit_transform(X_train_med)
X_test_scaled_med = scaler_med.transform(X_test_med)

# Train model
lr_med = LinearRegression()
lr_med.fit(X_train_scaled_med, y_train_med)

# Predictions
y_pred_med = lr_med.predict(X_test_scaled_med)

# Metrics
r2_med = r2_score(y_test_med, y_pred_med)
rmse_med = np.sqrt(mean_squared_error(y_test_med, y_pred_med))

print(f"\nMedical Model Performance:")
print(f"R² Score: {r2_med:.3f}")
print(f"RMSE: {rmse_med:.2f} mg")

# Feature importance analysis
coef_med = pd.DataFrame({
    'Feature': X_med.columns,
    'Coefficient': lr_med.coef_,
    'Abs_Coefficient': np.abs(lr_med.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

print(f"\nFeature Importance in Dosage Prediction:")
print(coef_med)

# Safety analysis - prediction intervals
from scipy import stats
residuals_med = y_test_med - y_pred_med
residual_std = np.std(residuals_med)

# 95% prediction intervals
prediction_interval = 1.96 * residual_std
print(f"\n95% Prediction Interval: ±{prediction_interval:.2f} mg")

# Clinical interpretation
print(f"\n=== Clinical Insights ===")
print(f"1. Model explains {r2_med*100:.1f}% of dosage variation")
print(f"2. Average prediction error: {rmse_med:.2f} mg")
print(f"3. Most important factors: {', '.join(coef_med.head(3)['Feature'].tolist())}")

# Visualize medical relationships
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Age vs Dosage
axes[0,0].scatter(medical_data['age'], medical_data['optimal_dosage'], alpha=0.6)
axes[0,0].set_xlabel('Age (years)')
axes[0,0].set_ylabel('Optimal Dosage (mg)')
axes[0,0].set_title('Dosage vs Age')

# Weight vs Dosage
axes[0,1].scatter(medical_data['weight'], medical_data['optimal_dosage'], alpha=0.6)
axes[0,1].set_xlabel('Weight (kg)')
axes[0,1].set_ylabel('Optimal Dosage (mg)')
axes[0,1].set_title('Dosage vs Weight')

# Gender differences
gender_data = medical_data.groupby('gender')['optimal_dosage'].agg(['mean', 'std'])
axes[0,2].bar(['Female', 'Male'], gender_data['mean'], 
              yerr=gender_data['std'], alpha=0.7, capsize=5)
axes[0,2].set_ylabel('Average Dosage (mg)')
axes[0,2].set_title('Dosage by Gender')

# Kidney function vs Dosage
axes[1,0].scatter(medical_data['kidney_function'], medical_data['optimal_dosage'], alpha=0.6)
axes[1,0].set_xlabel('Kidney Function (GFR)')
axes[1,0].set_ylabel('Optimal Dosage (mg)')
axes[1,0].set_title('Dosage vs Kidney Function')

# Predictions vs Actual
axes[1,1].scatter(y_test_med, y_pred_med, alpha=0.6)
axes[1,1].plot([y_test_med.min(), y_test_med.max()], 
               [y_test_med.min(), y_test_med.max()], 'r--', lw=2)
axes[1,1].set_xlabel('Actual Dosage (mg)')
axes[1,1].set_ylabel('Predicted Dosage (mg)')
axes[1,1].set_title(f'Medical Model Predictions (R² = {r2_med:.3f})')

# Feature importance
axes[1,2].barh(range(len(coef_med)), coef_med['Coefficient'])
axes[1,2].set_yticks(range(len(coef_med)))
axes[1,2].set_yticklabels(coef_med['Feature'])
axes[1,2].set_xlabel('Coefficient (Standardized)')
axes[1,2].set_title('Feature Importance')
axes[1,2].axvline(x=0, color='k', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# Dosage recommendation system
def predict_dosage(age, weight, height, gender, kidney_func, liver_func):
    """Predict optimal drug dosage for a patient."""
    bmi = weight / ((height/100)**2)
    
    patient_data = np.array([[age, weight, height, gender, kidney_func, liver_func, bmi]])
    patient_scaled = scaler_med.transform(patient_data)
    predicted_dosage = lr_med.predict(patient_scaled)[0]
    
    # Add safety bounds
    predicted_dosage = np.clip(predicted_dosage, 1, 50)
    
    return predicted_dosage, prediction_interval

# Example patient
example_age, example_weight, example_height = 65, 75, 175
example_gender, example_kidney, example_liver = 1, 85, 75

predicted_dose, interval = predict_dosage(
    example_age, example_weight, example_height,
    example_gender, example_kidney, example_liver
)

print(f"\n=== Dosage Recommendation ===")
print(f"Patient: {example_age}yr old, {example_weight}kg, {'Male' if example_gender else 'Female'}")
print(f"Recommended dosage: {predicted_dose:.1f} mg")
print(f"95% confidence interval: {predicted_dose-interval:.1f} - {predicted_dose+interval:.1f} mg")
```

## 📚 References

### Books
1. **"An Introduction to Statistical Learning"** by James, Witten, Hastie, and Tibshirani - Chapter 3
2. **"The Elements of Statistical Learning"** by Hastie, Tibshirani, and Friedman - Chapter 3
3. **"Hands-On Machine Learning"** by Aurélien Géron - Chapter 4
4. **"Pattern Recognition and Machine Learning"** by Christopher Bishop - Chapter 3

### Papers and Articles
1. **[Linear Regression (Wikipedia)](https://en.wikipedia.org/wiki/Linear_regression)** - Comprehensive overview
2. **[Ordinary Least Squares](https://en.wikipedia.org/wiki/Ordinary_least_squares)** - Mathematical foundation
3. **[The Gauss-Markov Theorem](https://en.wikipedia.org/wiki/Gauss%E2%80%93Markov_theorem)** - Theoretical properties

### Online Resources
1. **[Scikit-learn Linear Regression Documentation](https://scikit-learn.org/stable/modules/linear_model.html)**
2. **[StatsModels OLS Documentation](https://www.statsmodels.org/stable/regression.html)**
3. **[Khan Academy: Linear Regression](https://www.khanacademy.org/math/statistics-probability/describing-relationships-quantitative-data)**
4. **[Coursera Machine Learning Course](https://www.coursera.org/learn/machine-learning)** by Andrew Ng

### Interactive Tutorials
1. **[Linear Regression Interactive Visualization](https://seeing-theory.brown.edu/regression-analysis/index.html)**
2. **[Regression Analysis Explained Visually](http://setosa.io/ev/ordinary-least-squares-regression/)**
3. **[Kaggle Learn: Introduction to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning)**

### Video Resources
1. **[StatQuest: Linear Regression](https://www.youtube.com/watch?v=nk2CQITm_eo)**
2. **[3Blue1Brown: Linear Algebra Essence](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)**
3. **[MIT OpenCourseWare: Statistics](https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/)**

### Practical Applications
1. **[Real Estate Price Prediction](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)**
2. **[Medical Research Applications](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3049417/)**
3. **[Business Analytics Case Studies](https://hbr.org/topic/analytics)**
