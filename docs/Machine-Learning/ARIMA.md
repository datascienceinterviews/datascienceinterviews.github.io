---
title: ARIMA (AutoRegressive Integrated Moving Average)
description: Comprehensive guide to ARIMA time series forecasting model with implementation, intuition, and interview questions.
comments: true
---

# 📘 ARIMA (AutoRegressive Integrated Moving Average)

ARIMA is a powerful time series forecasting method that combines autoregression, differencing, and moving averages to model and predict sequential data patterns.

**Resources:** [Statsmodels ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html) | [Time Series Analysis Book](https://otexts.com/fpp2/arima.html)

## ✍️ Summary

ARIMA (AutoRegressive Integrated Moving Average) is a statistical model used for analyzing and forecasting time series data. It's particularly effective for data that shows patterns over time but may not be stationary. ARIMA combines three components:

- **AR (AutoRegressive)**: Uses the relationship between an observation and lagged observations
- **I (Integrated)**: Uses differencing to make the time series stationary
- **MA (Moving Average)**: Uses the dependency between an observation and residual errors from lagged observations

ARIMA is widely used in:
- Stock price forecasting
- Sales prediction
- Economic indicators analysis
- Weather forecasting
- Demand planning

The model is denoted as ARIMA(p,d,q) where:
- p: number of lag observations (AR terms)
- d: degree of differencing (I terms)
- q: size of moving average window (MA terms)

## 🧠 Intuition

### Mathematical Foundation

The ARIMA model can be understood through its three components:

#### 1. AutoRegressive (AR) Component
The AR(p) model predicts future values based on past values:

$$X_t = c + \phi_1 X_{t-1} + \phi_2 X_{t-2} + ... + \phi_p X_{t-p} + \epsilon_t$$

Where:
- $X_t$ is the value at time t
- $c$ is a constant
- $\phi_i$ are the autoregressive parameters
- $\epsilon_t$ is white noise

#### 2. Integrated (I) Component
The I(d) component makes the series stationary by differencing:

$$\nabla^d X_t = (1-L)^d X_t$$

Where:
- $L$ is the lag operator
- $d$ is the degree of differencing
- First difference: $\nabla X_t = X_t - X_{t-1}$
- Second difference: $\nabla^2 X_t = \nabla X_t - \nabla X_{t-1}$

#### 3. Moving Average (MA) Component
The MA(q) model uses past forecast errors:

$$X_t = \mu + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + ... + \theta_q \epsilon_{t-q}$$

Where:
- $\mu$ is the mean
- $\theta_i$ are the moving average parameters
- $\epsilon_t$ are error terms

#### Complete ARIMA Model
Combining all components, ARIMA(p,d,q) is:

$$(1 - \phi_1 L - ... - \phi_p L^p)(1-L)^d X_t = (1 + \theta_1 L + ... + \theta_q L^q)\epsilon_t$$

### Intuitive Understanding

Think of ARIMA as answering three questions:
1. **AR**: How much do past values influence future values?
2. **I**: How many times do we need to difference the data to remove trends?
3. **MA**: How much do past prediction errors affect current predictions?

## 🔢 Implementation using Libraries

### Using Statsmodels

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

# Generate sample time series data
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=200, freq='D')
trend = np.linspace(100, 120, 200)
seasonal = 10 * np.sin(2 * np.pi * np.arange(200) / 30)
noise = np.random.normal(0, 2, 200)
ts = trend + seasonal + noise

# Create time series
data = pd.Series(ts, index=dates)

# Step 1: Check stationarity
def check_stationarity(timeseries, title):
    # Perform Augmented Dickey-Fuller test
    result = adfuller(timeseries)
    print(f'Results of Dickey-Fuller Test for {title}:')
    print(f'ADF Statistic: {result[0]:.6f}')
    print(f'p-value: {result[1]:.6f}')
    print(f'Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.3f}')
    
    if result[1] <= 0.05:
        print("Data is stationary")
    else:
        print("Data is non-stationary")
    print("-" * 50)

check_stationarity(data, "Original Series")

# Step 2: Make series stationary if needed
data_diff = data.diff().dropna()
check_stationarity(data_diff, "First Differenced Series")

# Step 3: Determine ARIMA parameters using ACF and PACF plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Original series
axes[0,0].plot(data)
axes[0,0].set_title('Original Time Series')
axes[0,0].set_xlabel('Date')
axes[0,0].set_ylabel('Value')

# Differenced series
axes[0,1].plot(data_diff)
axes[0,1].set_title('First Differenced Series')
axes[0,1].set_xlabel('Date')
axes[0,1].set_ylabel('Differenced Value')

# ACF and PACF plots
plot_acf(data_diff, ax=axes[1,0], lags=20, title='ACF of Differenced Series')
plot_pacf(data_diff, ax=axes[1,1], lags=20, title='PACF of Differenced Series')

plt.tight_layout()
plt.show()

# Step 4: Fit ARIMA model
# Let's try ARIMA(2,1,2) based on the plots
model = ARIMA(data, order=(2, 1, 2))
fitted_model = model.fit()

# Print model summary
print(fitted_model.summary())

# Step 5: Make predictions
n_periods = 30
forecast = fitted_model.forecast(steps=n_periods)
forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), 
                               periods=n_periods, freq='D')

# Get confidence intervals
forecast_ci = fitted_model.get_forecast(steps=n_periods).conf_int()

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(data.index, data, label='Original Data', color='blue')
plt.plot(forecast_index, forecast, label='Forecast', color='red', linestyle='--')
plt.fill_between(forecast_index, 
                 forecast_ci.iloc[:, 0], 
                 forecast_ci.iloc[:, 1], 
                 color='red', alpha=0.2, label='Confidence Interval')
plt.legend()
plt.title('ARIMA Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True)
plt.show()

# Step 6: Model diagnostics
fitted_model.plot_diagnostics(figsize=(15, 8))
plt.tight_layout()
plt.show()
```

### Auto ARIMA for Parameter Selection

```python
from pmdarima import auto_arima

# Automatically find best ARIMA parameters
auto_model = auto_arima(data, 
                        start_p=0, start_q=0,
                        max_p=5, max_q=5,
                        seasonal=False,
                        stepwise=True,
                        suppress_warnings=True,
                        error_action='ignore')

print(f"Best ARIMA model: {auto_model.order}")
print(auto_model.summary())

# Forecast with auto ARIMA
auto_forecast = auto_model.predict(n_periods=30)
print(f"Next 30 predictions: {auto_forecast}")
```

## ⚙️ From Scratch Implementation

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

class ARIMAFromScratch:
    def __init__(self, p=1, d=1, q=1):
        """
        ARIMA model implementation from scratch
        
        Parameters:
        p (int): Order of autoregression
        d (int): Degree of differencing
        q (int): Order of moving average
        """
        self.p = p
        self.d = d
        self.q = q
        self.params = None
        self.fitted_values = None
        self.residuals = None
        
    def difference(self, series, d):
        """Apply differencing to make series stationary"""
        diff_series = series.copy()
        for _ in range(d):
            diff_series = np.diff(diff_series)
        return diff_series
    
    def inverse_difference(self, diff_series, original_series, d):
        """Reverse the differencing operation"""
        result = diff_series.copy()
        for _ in range(d):
            # Add back the last value from previous level
            cumsum_result = np.cumsum(result)
            # Add the last original value before differencing
            result = cumsum_result + original_series[-(d-_)]
        return result
    
    def ar_component(self, data, ar_params):
        """Calculate AR component"""
        ar_component = np.zeros(len(data))
        for i in range(self.p, len(data)):
            for j in range(self.p):
                ar_component[i] += ar_params[j] * data[i-j-1]
        return ar_component
    
    def ma_component(self, residuals, ma_params):
        """Calculate MA component"""
        ma_component = np.zeros(len(residuals))
        for i in range(self.q, len(residuals)):
            for j in range(self.q):
                ma_component[i] += ma_params[j] * residuals[i-j-1]
        return ma_component
    
    def likelihood(self, params, data):
        """Calculate negative log-likelihood for optimization"""
        try:
            # Split parameters
            ar_params = params[:self.p] if self.p > 0 else []
            ma_params = params[self.p:self.p + self.q] if self.q > 0 else []
            sigma = params[-1] if len(params) > self.p + self.q else 1.0
            
            n = len(data)
            errors = np.zeros(n)
            predictions = np.zeros(n)
            
            # Initialize predictions and errors
            for i in range(max(self.p, self.q), n):
                # AR component
                ar_pred = 0
                if self.p > 0:
                    for j in range(self.p):
                        ar_pred += ar_params[j] * data[i-j-1]
                
                # MA component
                ma_pred = 0
                if self.q > 0:
                    for j in range(self.q):
                        ma_pred += ma_params[j] * errors[i-j-1]
                
                predictions[i] = ar_pred + ma_pred
                errors[i] = data[i] - predictions[i]
            
            # Calculate log-likelihood
            valid_errors = errors[max(self.p, self.q):]
            log_likelihood = -0.5 * len(valid_errors) * np.log(2 * np.pi * sigma**2)
            log_likelihood -= 0.5 * np.sum(valid_errors**2) / (sigma**2)
            
            return -log_likelihood  # Return negative for minimization
            
        except:
            return np.inf
    
    def fit(self, data):
        """Fit ARIMA model to data"""
        # Apply differencing
        if self.d > 0:
            diff_data = self.difference(data, self.d)
        else:
            diff_data = data
        
        # Initial parameter guesses
        initial_params = []
        
        # AR parameters (between -1 and 1)
        initial_params.extend([0.1] * self.p)
        
        # MA parameters (between -1 and 1)
        initial_params.extend([0.1] * self.q)
        
        # Sigma (positive)
        initial_params.append(1.0)
        
        # Bounds for parameters
        bounds = []
        bounds.extend([(-0.99, 0.99)] * self.p)  # AR params
        bounds.extend([(-0.99, 0.99)] * self.q)  # MA params
        bounds.append((0.01, None))  # Sigma
        
        # Optimize parameters
        try:
            result = minimize(self.likelihood, initial_params, args=(diff_data,),
                            method='L-BFGS-B', bounds=bounds)
            
            if result.success:
                self.params = result.x
                
                # Calculate fitted values and residuals
                self._calculate_fitted_values(data)
                
                return self
            else:
                raise ValueError("Optimization failed")
                
        except Exception as e:
            print(f"Error during fitting: {e}")
            return None
    
    def _calculate_fitted_values(self, data):
        """Calculate fitted values and residuals"""
        if self.params is None:
            return
        
        # Apply differencing
        if self.d > 0:
            diff_data = self.difference(data, self.d)
        else:
            diff_data = data
        
        # Get parameters
        ar_params = self.params[:self.p] if self.p > 0 else []
        ma_params = self.params[self.p:self.p + self.q] if self.q > 0 else []
        
        n = len(diff_data)
        fitted = np.zeros(n)
        errors = np.zeros(n)
        
        # Calculate fitted values
        for i in range(max(self.p, self.q), n):
            # AR component
            ar_pred = 0
            if self.p > 0:
                for j in range(self.p):
                    ar_pred += ar_params[j] * diff_data[i-j-1]
            
            # MA component
            ma_pred = 0
            if self.q > 0:
                for j in range(self.q):
                    ma_pred += ma_params[j] * errors[i-j-1]
            
            fitted[i] = ar_pred + ma_pred
            errors[i] = diff_data[i] - fitted[i]
        
        self.fitted_values = fitted
        self.residuals = errors
    
    def predict(self, steps):
        """Make predictions for future time steps"""
        if self.params is None:
            raise ValueError("Model must be fitted before prediction")
        
        ar_params = self.params[:self.p] if self.p > 0 else []
        ma_params = self.params[self.p:self.p + self.q] if self.q > 0 else []
        
        predictions = []
        
        # Use last values from fitted data for prediction
        last_values = self.fitted_values[-self.p:] if self.p > 0 else []
        last_errors = self.residuals[-self.q:] if self.q > 0 else []
        
        for step in range(steps):
            # AR component
            ar_pred = 0
            if self.p > 0 and len(last_values) >= self.p:
                for j in range(self.p):
                    ar_pred += ar_params[j] * last_values[-(j+1)]
            
            # MA component (assumes future errors are 0)
            ma_pred = 0
            if self.q > 0 and len(last_errors) >= self.q and step == 0:
                for j in range(self.q):
                    ma_pred += ma_params[j] * last_errors[-(j+1)]
            
            pred = ar_pred + ma_pred
            predictions.append(pred)
            
            # Update last values for next prediction
            if self.p > 0:
                last_values = np.append(last_values[1:], pred)
            if self.q > 0:
                last_errors = np.append(last_errors[1:], 0)  # Assume future errors are 0
        
        return np.array(predictions)
    
    def summary(self):
        """Print model summary"""
        if self.params is None:
            print("Model not fitted")
            return
        
        print(f"ARIMA({self.p}, {self.d}, {self.q}) Model Summary")
        print("=" * 50)
        
        if self.p > 0:
            print("AR Parameters:")
            for i, param in enumerate(self.params[:self.p]):
                print(f"  AR({i+1}): {param:.6f}")
        
        if self.q > 0:
            print("MA Parameters:")
            for i, param in enumerate(self.params[self.p:self.p + self.q]):
                print(f"  MA({i+1}): {param:.6f}")
        
        print(f"Sigma: {self.params[-1]:.6f}")

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n = 100
    true_ar = [0.7, -0.2]
    true_ma = [0.3]
    
    # Generate ARIMA(2,1,1) data
    data = []
    errors = np.random.normal(0, 1, n + 10)
    
    for t in range(2, n):
        if t < 3:
            value = errors[t]
        else:
            ar_component = true_ar[0] * (data[t-1] - data[t-2]) + true_ar[1] * (data[t-2] - data[t-3]) if t > 2 else 0
            ma_component = true_ma[0] * errors[t-1] if t > 0 else 0
            value = ar_component + ma_component + errors[t]
            if t > 0:
                value += data[t-1]  # Add integration
        data.append(value)
    
    data = np.array(data)
    
    # Fit custom ARIMA model
    model = ARIMAFromScratch(p=2, d=1, q=1)
    fitted_model = model.fit(data)
    
    if fitted_model:
        fitted_model.summary()
        
        # Make predictions
        predictions = fitted_model.predict(10)
        print(f"\nNext 10 predictions: {predictions}")
```

## ⚠️ Assumptions and Limitations

### Assumptions

1. **Stationarity**: After differencing, the series should be stationary (constant mean and variance)
2. **Linear relationships**: ARIMA assumes linear dependencies between observations
3. **Normal residuals**: Error terms should be normally distributed with zero mean
4. **Homoscedasticity**: Constant variance of residuals over time
5. **No autocorrelation in residuals**: Residuals should be independent

### Limitations

1. **Linear models only**: Cannot capture non-linear patterns
2. **Requires sufficient data**: Needs adequate historical data for reliable forecasting
3. **Parameter selection complexity**: Choosing optimal (p,d,q) can be challenging
4. **Poor with structural breaks**: Struggles when underlying data patterns change
5. **No exogenous variables**: Basic ARIMA doesn't include external predictors
6. **Computational intensity**: Parameter estimation can be slow for large datasets

### Comparison with Other Models

| Model | Strengths | Weaknesses |
|-------|-----------|------------|
| **ARIMA** | Good for linear trends, well-established theory | Limited to linear relationships |
| **LSTM** | Captures complex patterns, handles non-linearity | Requires large datasets, black box |
| **Prophet** | Handles seasonality well, robust to outliers | Less flexible than ARIMA |
| **Exponential Smoothing** | Simple, fast computation | Limited complexity modeling |

## 💡 Interview Questions

??? question "1. What does each parameter in ARIMA(p,d,q) represent?"

    **Answer:** 
    - **p (AR order)**: Number of lagged observations used in the autoregressive component. Determines how many past values influence the current prediction.
    - **d (Differencing degree)**: Number of times the series is differenced to achieve stationarity. Usually 0, 1, or 2.
    - **q (MA order)**: Size of the moving average window, representing how many past forecast errors are used in the prediction.
    
    Example: ARIMA(2,1,1) uses 2 lagged values, applies 1 level of differencing, and includes 1 past error term.

??? question "2. How do you determine the optimal ARIMA parameters?"

    **Answer:**
    Several methods can be used:
    
    **Manual approach:**
    - Use ACF and PACF plots to identify parameters
    - ACF helps determine q (cuts off after q lags for MA processes)  
    - PACF helps determine p (cuts off after p lags for AR processes)
    - Use ADF test to determine d (degree of differencing needed for stationarity)
    
    **Automatic approach:**
    - Information criteria (AIC, BIC) - lower values indicate better fit
    - Grid search over parameter combinations
    - Auto ARIMA algorithms (like `pmdarima.auto_arima()`)
    - Cross-validation for out-of-sample performance

??? question "3. What is the difference between MA and AR components?"

    **Answer:**
    
    **Autoregressive (AR):**
    - Uses past values of the series itself for prediction
    - Formula: $X_t = c + \phi_1 X_{t-1} + \phi_2 X_{t-2} + ... + \epsilon_t$
    - Captures the "memory" of the time series
    
    **Moving Average (MA):**
    - Uses past forecast errors (residuals) for prediction  
    - Formula: $X_t = \mu + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + ...$
    - Captures short-term irregularities and shocks

??? question "4. How do you check if your ARIMA model is good?"

    **Answer:**
    
    **Residual Analysis:**
    - Residuals should be white noise (no patterns)
    - Ljung-Box test for autocorrelation in residuals
    - Jarque-Bera test for normality of residuals
    - Plot ACF/PACF of residuals (should show no significant lags)
    
    **Performance Metrics:**
    - AIC/BIC for model comparison
    - MAPE, RMSE, MAE for forecast accuracy
    - Out-of-sample validation
    
    **Visual Inspection:**
    - Q-Q plots for normality
    - Residual plots over time
    - Fitted vs actual values plot

??? question "5. What makes a time series stationary and why is it important for ARIMA?"

    **Answer:**
    
    **Stationary Series Properties:**
    - Constant mean over time
    - Constant variance over time  
    - Covariance between periods depends only on lag, not time
    
    **Importance for ARIMA:**
    - ARIMA models assume stationarity for reliable parameter estimation
    - Non-stationary data can lead to spurious relationships
    - Differencing (I component) is used to achieve stationarity
    
    **Tests for Stationarity:**
    - Augmented Dickey-Fuller (ADF) test
    - KPSS test
    - Visual inspection of plots

??? question "6. Explain the concept of differencing in ARIMA"

    **Answer:**
    
    **Differencing** transforms a non-stationary series into stationary by computing differences between consecutive observations.
    
    **Types:**
    - **First differencing**: $\nabla X_t = X_t - X_{t-1}$
    - **Second differencing**: $\nabla^2 X_t = \nabla X_t - \nabla X_{t-1}$
    - **Seasonal differencing**: $\nabla_s X_t = X_t - X_{t-s}$
    
    **Effects:**
    - Removes trends (first differencing)
    - Removes curvature (second differencing)
    - Usually d=1 is sufficient, rarely need d>2
    - Over-differencing can introduce unnecessary complexity

??? question "7. How would you handle seasonality in ARIMA?"

    **Answer:**
    
    **SARIMA (Seasonal ARIMA):**
    - Extends ARIMA to SARIMA(p,d,q)(P,D,Q)s
    - Additional seasonal parameters P, D, Q for period s
    - Formula includes both non-seasonal and seasonal components
    
    **Implementation:**
    ```python
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    model = SARIMAX(data, order=(p,d,q), seasonal_order=(P,D,Q,s))
    ```
    
    **Alternative approaches:**
    - Seasonal decomposition before applying ARIMA
    - Use of external regressors for seasonal patterns
    - Prophet or other specialized seasonal models

??? question "8. What are the limitations of ARIMA and when would you choose alternatives?"

    **Answer:**
    
    **ARIMA Limitations:**
    - Only captures linear relationships
    - Requires stationary data
    - Sensitive to outliers
    - Cannot handle multiple seasonal patterns
    - No external variables incorporation
    
    **When to choose alternatives:**
    - **Non-linear patterns**: Use LSTM, Neural Networks
    - **Multiple seasonalities**: Use Prophet, TBATS
    - **External predictors**: Use ARIMAX, VAR models  
    - **Regime changes**: Use structural break models
    - **High-frequency data**: Use GARCH for volatility modeling
    - **Small datasets**: Use Exponential Smoothing

## 🧠 Examples

### Example 1: Stock Price Forecasting

```python
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Download stock data
stock = yf.download("AAPL", start="2020-01-01", end="2023-01-01")
prices = stock['Close']

# Fit ARIMA model
model = ARIMA(prices, order=(1,1,1))
fitted_model = model.fit()

# Forecast next 30 days
forecast = fitted_model.forecast(steps=30)
conf_int = fitted_model.get_forecast(steps=30).conf_int()

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(prices.index[-60:], prices.values[-60:], label='Actual', color='blue')
forecast_dates = pd.date_range(start=prices.index[-1], periods=31, freq='D')[1:]
plt.plot(forecast_dates, forecast, label='Forecast', color='red', linestyle='--')
plt.fill_between(forecast_dates, conf_int.iloc[:, 0], conf_int.iloc[:, 1], 
                 color='red', alpha=0.2)
plt.legend()
plt.title('AAPL Stock Price Forecast using ARIMA')
plt.show()

print("Forecast Summary:")
print(f"Last actual price: ${prices.iloc[-1]:.2f}")
print(f"30-day forecast mean: ${forecast.mean():.2f}")
print(f"Expected return: {((forecast.mean()/prices.iloc[-1])-1)*100:.2f}%")
```

### Example 2: Sales Forecasting with Seasonality

```python
# Generate seasonal sales data
np.random.seed(42)
dates = pd.date_range('2018-01-01', periods=365*3, freq='D')
trend = np.linspace(1000, 1500, len(dates))
seasonal = 200 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
weekly = 50 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
noise = np.random.normal(0, 50, len(dates))
sales = trend + seasonal + weekly + noise

sales_ts = pd.Series(sales, index=dates)

# Apply seasonal ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Fit SARIMA model
sarima_model = SARIMAX(sales_ts, 
                       order=(1, 1, 1), 
                       seasonal_order=(1, 1, 1, 365))
sarima_fitted = sarima_model.fit(disp=False)

# Forecast next quarter
forecast_days = 90
forecast = sarima_fitted.forecast(steps=forecast_days)
conf_int = sarima_fitted.get_forecast(steps=forecast_days).conf_int()

# Performance metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

# In-sample predictions for evaluation
in_sample_pred = sarima_fitted.fittedvalues
mae = mean_absolute_error(sales_ts, in_sample_pred)
rmse = np.sqrt(mean_squared_error(sales_ts, in_sample_pred))

print(f"Model Performance:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"Mean Sales: {sales_ts.mean():.2f}")
print(f"MAPE: {(mae/sales_ts.mean()*100):.2f}%")

# Plot seasonal forecast
plt.figure(figsize=(15, 8))
plt.plot(sales_ts.index[-180:], sales_ts.values[-180:], 
         label='Historical Sales', color='blue')
forecast_dates = pd.date_range(start=sales_ts.index[-1], 
                               periods=forecast_days+1, freq='D')[1:]
plt.plot(forecast_dates, forecast, 
         label='SARIMA Forecast', color='red', linestyle='--')
plt.fill_between(forecast_dates, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                 color='red', alpha=0.2, label='Confidence Interval')
plt.legend()
plt.title('Seasonal Sales Forecasting with SARIMA')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True)
plt.show()
```

## 📚 References

- **Books:**
  - [Forecasting: Principles and Practice](https://otexts.com/fpp2/) by Rob Hyndman & George Athanasopoulos
  - [Time Series Analysis](https://www.amazon.com/Time-Series-Analysis-Forecasting-Control/dp/1118675029) by Box, Jenkins, Reinsel & Ljung
  - [Introduction to Time Series and Forecasting](https://www.springer.com/gp/book/9783319298528) by Brockwell & Davis

- **Documentation:**
  - [Statsmodels ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)
  - [pmdarima (Auto ARIMA)](https://alkaline-ml.com/pmdarima/)
  - [Scikit-learn Time Series](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model)

- **Tutorials:**
  - [ARIMA Model Complete Guide](https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/)
  - [Time Series Forecasting Guide](https://www.kaggle.com/code/prashant111/complete-guide-on-time-series-analysis-in-python)

- **Research Papers:**
  - Box, G. E. P., & Jenkins, G. M. (1970). Time series analysis: Forecasting and control
  - Akaike, H. (1974). A new look at the statistical model identification

- **Online Courses:**
  - [Time Series Analysis on Coursera](https://www.coursera.org/learn/practical-time-series-analysis)
  - [Forecasting Using R on DataCamp](https://www.datacamp.com/courses/forecasting-using-r)