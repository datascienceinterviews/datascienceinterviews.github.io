---
title: Normal Distribution
description: Comprehensive guide to Normal Distribution with mathematical foundations, properties, applications in machine learning, and statistical inference.
comments: true
---

🔧#🔧 🔧📊🔧 🔧N🔧o🔧r🔧m🔧a🔧l🔧 🔧D🔧i🔧s🔧t🔧r🔧i🔧b🔧u🔧t🔧i🔧o🔧n🔧

The Normal Distribution (also called Gaussian Distribution) is the most important continuous probability distribution in statistics and machine learning, characterized by its symmetric bell-shaped curve and defined by two parameters: mean and standard deviation.

**Resources:** [SciPy Stats Documentation](https://docs.scipy.org/doc/scipy/reference/stats.html) | [Khan Academy Statistics](https://www.khanacademy.org/math/statistics-probability) | [Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/)

🔧#🔧#🔧 🔧
 Summary

The Normal Distribution is a continuous probability distribution that is symmetric around its mean, with the shape determined by its standard deviation. It's fundamental to statistics and machine learning due to the Central Limit Theorem and its mathematical properties.

**Key Characteristics:**
- **Bell-shaped curve**: Symmetric around the mean
- **Unimodal**: Single peak at the mean
- **Asymptotic**: Tails approach zero but never reach it
- **Defined by two parameters**: Mean (¼) and standard deviation (Ã)
- **68-95-99.7 rule**: Empirical rule for data spread

**Standard Normal Distribution:**
- Mean (¼) = 0
- Standard deviation (Ã) = 1
- Used for standardization and z-scores

**Applications in Machine Learning:**
- **Assumption in algorithms**: Linear regression, Naive Bayes, LDA
- **Initialization**: Weight initialization in neural networks
- **Regularization**: Gaussian priors in Bayesian methods
- **Noise modeling**: Gaussian noise assumptions
- **Feature engineering**: Normalization and standardization
- **Hypothesis testing**: Statistical significance testing
- **Confidence intervals**: Uncertainty quantification

**Real-world Examples:**
- Heights and weights of populations
- Measurement errors in experiments
- Financial returns (approximately)
- IQ scores
- Blood pressure measurements
- Test scores and grades

🔧#🔧#🔧 🔧🧠🔧 🔧I🔧n🔧t🔧u🔧i🔧t🔧i🔧o🔧n🔧

🔧#🔧#🔧#🔧 🔧M🔧a🔧t🔧h🔧e🔧m🔧a🔧t🔧i🔧c🔧a🔧l🔧 🔧F🔧o🔧u🔧n🔧d🔧a🔧t🔧i🔧o🔧n🔧

🔧#🔧#🔧#🔧#🔧 🔧P🔧r🔧o🔧b🔧a🔧b🔧i🔧l🔧i🔧t🔧y🔧 🔧D🔧e🔧n🔧s🔧i🔧t🔧y🔧 🔧F🔧u🔧n🔧c🔧t🔧i🔧o🔧n🔧 🔧(🔧P🔧D🔧F🔧)🔧

The Normal Distribution is defined by its PDF:

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

Where:
- $\mu$ is the mean (location parameter)
- $\sigma$ is the standard deviation (scale parameter)  
- $\sigma^2$ is the variance
- $e \approx 2.718$ (Euler's number)
- $\pi \approx 3.14159$

🔧#🔧#🔧#🔧#🔧 🔧S🔧t🔧a🔧n🔧d🔧a🔧r🔧d🔧 🔧N🔧o🔧r🔧m🔧a🔧l🔧 🔧D🔧i🔧s🔧t🔧r🔧i🔧b🔧u🔧t🔧i🔧o🔧n🔧

When $\mu = 0$ and $\sigma = 1$:

$$\phi(z) = \frac{1}{\sqrt{2\pi}} e^{-\frac{z^2}{2}}$$

🔧#🔧#🔧#🔧#🔧 🔧C🔧u🔧m🔧u🔧l🔧a🔧t🔧i🔧v🔧e🔧 🔧D🔧i🔧s🔧t🔧r🔧i🔧b🔧u🔧t🔧i🔧o🔧n🔧 🔧F🔧u🔧n🔧c🔧t🔧i🔧o🔧n🔧 🔧(🔧C🔧D🔧F🔧)🔧

$$F(x) = P(X \leq x) = \int_{-\infty}^{x} \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(t-\mu)^2}{2\sigma^2}} dt$$

For standard normal: $\Phi(z) = \int_{-\infty}^{z} \phi(t) dt$

🔧#🔧#🔧#🔧#🔧 🔧Z🔧-🔧S🔧c🔧o🔧r🔧e🔧 🔧T🔧r🔧a🔧n🔧s🔧f🔧o🔧r🔧m🔧a🔧t🔧i🔧o🔧n🔧

Convert any normal distribution to standard normal:

$$Z = \frac{X - \mu}{\sigma}$$

Where $Z \sim N(0, 1)$

🔧#🔧#🔧#🔧 🔧K🔧e🔧y🔧 🔧P🔧r🔧o🔧p🔧e🔧r🔧t🔧i🔧e🔧s🔧

🔧#🔧#🔧#🔧#🔧 🔧M🔧o🔧m🔧e🔧n🔧t🔧 🔧P🔧r🔧o🔧p🔧e🔧r🔧t🔧i🔧e🔧s🔧

**Mean (First Moment):**
$$E[X] = \mu$$

**Variance (Second Central Moment):**
$$\text{Var}(X) = E[(X-\mu)^2] = \sigma^2$$

**Skewness (Third Standardized Moment):**
$$\text{Skewness} = E\left[\left(\frac{X-\mu}{\sigma}\right)^3\right] = 0$$

**Kurtosis (Fourth Standardized Moment):**
$$\text{Kurtosis} = E\left[\left(\frac{X-\mu}{\sigma}\right)^4\right] = 3$$

🔧#🔧#🔧#🔧#🔧 🔧T🔧h🔧e🔧 🔧6🔧8🔧-🔧9🔧5🔧-🔧9🔧9🔧.🔧7🔧 🔧R🔧u🔧l🔧e🔧 🔧(🔧E🔧m🔧p🔧i🔧r🔧i🔧c🔧a🔧l🔧 🔧R🔧u🔧l🔧e🔧)🔧

For any normal distribution:
- **68%** of data falls within 1 standard deviation: $P(\mu - \sigma \leq X \leq \mu + \sigma) = 0.68$
- **95%** of data falls within 2 standard deviations: $P(\mu - 2\sigma \leq X \leq \mu + 2\sigma) = 0.95$
- **99.7%** of data falls within 3 standard deviations: $P(\mu - 3\sigma \leq X \leq \mu + 3\sigma) = 0.997$

🔧#🔧#🔧#🔧#🔧 🔧L🔧i🔧n🔧e🔧a🔧r🔧 🔧C🔧o🔧m🔧b🔧i🔧n🔧a🔧t🔧i🔧o🔧n🔧s🔧

If $X \sim N(\mu_1, \sigma_1^2)$ and $Y \sim N(\mu_2, \sigma_2^2)$ are independent:

$$aX + bY \sim N(a\mu_1 + b\mu_2, a^2\sigma_1^2 + b^2\sigma_2^2)$$

🔧#🔧#🔧#🔧#🔧 🔧C🔧e🔧n🔧t🔧r🔧a🔧l🔧 🔧L🔧i🔧m🔧i🔧t🔧 🔧T🔧h🔧e🔧o🔧r🔧e🔧m🔧

For any population with mean $\mu$ and finite variance $\sigma^2$, the sampling distribution of the sample mean approaches normal as sample size increases:

$$\bar{X}_n \sim N\left(\mu, \frac{\sigma^2}{n}\right) \text{ as } n \to \infty$$

🔧#🔧#🔧#🔧 🔧M🔧a🔧x🔧i🔧m🔧u🔧m🔧 🔧L🔧i🔧k🔧e🔧l🔧i🔧h🔧o🔧o🔧d🔧 🔧E🔧s🔧t🔧i🔧m🔧a🔧t🔧i🔧o🔧n🔧

Given observations $x_1, x_2, ..., x_n$ from $N(\mu, \sigma^2)$:

**Log-likelihood:**
$$\ell(\mu, \sigma^2) = -\frac{n}{2}\ln(2\pi) - \frac{n}{2}\ln(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2$$

**MLE Estimators:**
$$\hat{\mu} = \frac{1}{n}\sum_{i=1}^{n} x_i = \bar{x}$$

$$\hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^{n} (x_i - \bar{x})^2$$

🔧#🔧#🔧 🔧=🔧"🔧 🔧I🔧m🔧p🔧l🔧e🔧m🔧e🔧n🔧t🔧a🔧t🔧i🔧o🔧n🔧 🔧u🔧s🔧i🔧n🔧g🔧 🔧L🔧i🔧b🔧r🔧a🔧r🔧i🔧e🔧s🔧

🔧#🔧#🔧#🔧 🔧U🔧s🔧i🔧n🔧g🔧 🔧N🔧u🔧m🔧P🔧y🔧 🔧a🔧n🔧d🔧 🔧S🔧c🔧i🔧P🔧y🔧

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

🔧#🔧 🔧S🔧e🔧t🔧 🔧s🔧t🔧y🔧l🔧e🔧 🔧f🔧o🔧r🔧 🔧b🔧e🔧t🔧t🔧e🔧r🔧 🔧p🔧l🔧o🔧t🔧s🔧
plt.style.use('seaborn-v0_8')
np.random.seed(42)

🔧#🔧 🔧G🔧e🔧n🔧e🔧r🔧a🔧t🔧e🔧 🔧n🔧o🔧r🔧m🔧a🔧l🔧 🔧d🔧i🔧s🔧t🔧r🔧i🔧b🔧u🔧t🔧i🔧o🔧n🔧 🔧s🔧a🔧m🔧p🔧l🔧e🔧s🔧
def generate_normal_samples(mu=0, sigma=1, size=1000):
    """
    Generate samples from normal distribution
    
    Args:
        mu: Mean parameter
        sigma: Standard deviation parameter
        size: Number of samples
        
    Returns:
        Array of samples
    """
    return np.random.normal(mu, sigma, size)

🔧#🔧 🔧B🔧a🔧s🔧i🔧c🔧 🔧n🔧o🔧r🔧m🔧a🔧l🔧 🔧d🔧i🔧s🔧t🔧r🔧i🔧b🔧u🔧t🔧i🔧o🔧n🔧 🔧o🔧p🔧e🔧r🔧a🔧t🔧i🔧o🔧n🔧s🔧
mu, sigma = 5, 2
samples = generate_normal_samples(mu, sigma, 10000)

print(f"True parameters: ¼={mu}, Ã={sigma}")
print(f"Sample statistics: ¼={np.mean(samples):.3f}, Ã={np.std(samples, ddof=1):.3f}")
print(f"Sample size: {len(samples)}")

🔧#🔧 🔧U🔧s🔧i🔧n🔧g🔧 🔧s🔧c🔧i🔧p🔧y🔧.🔧s🔧t🔧a🔧t🔧s🔧 🔧f🔧o🔧r🔧 🔧n🔧o🔧r🔧m🔧a🔧l🔧 🔧d🔧i🔧s🔧t🔧r🔧i🔧b🔧u🔧t🔧i🔧o🔧n🔧
normal_dist = stats.norm(loc=mu, scale=sigma)

🔧#🔧 🔧C🔧a🔧l🔧c🔧u🔧l🔧a🔧t🔧e🔧 🔧p🔧r🔧o🔧b🔧a🔧b🔧i🔧l🔧i🔧t🔧i🔧e🔧s🔧 🔧a🔧n🔧d🔧 🔧q🔧u🔧a🔧n🔧t🔧i🔧l🔧e🔧s🔧
x_values = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
pdf_values = normal_dist.pdf(x_values)
cdf_values = normal_dist.cdf(x_values)

print(f"\nProbability calculations:")
print(f"P(X d 7) = {normal_dist.cdf(7):.4f}")
print(f"P(X e 3) = {1 - normal_dist.cdf(3):.4f}")
print(f"P(3 d X d 7) = {normal_dist.cdf(7) - normal_dist.cdf(3):.4f}")

🔧#🔧 🔧Q🔧u🔧a🔧n🔧t🔧i🔧l🔧e🔧s🔧 🔧(🔧i🔧n🔧v🔧e🔧r🔧s🔧e🔧 🔧C🔧D🔧F🔧)🔧
print(f"\nQuantiles:")
print(f"25th percentile: {normal_dist.ppf(0.25):.3f}")
print(f"50th percentile (median): {normal_dist.ppf(0.5):.3f}")
print(f"75th percentile: {normal_dist.ppf(0.75):.3f}")
print(f"95th percentile: {normal_dist.ppf(0.95):.3f}")

🔧#🔧 🔧E🔧m🔧p🔧i🔧r🔧i🔧c🔧a🔧l🔧 🔧r🔧u🔧l🔧e🔧 🔧v🔧e🔧r🔧i🔧f🔧i🔧c🔧a🔧t🔧i🔧o🔧n🔧
within_1_sigma = np.sum(np.abs(samples - mu) <= sigma) / len(samples)
within_2_sigma = np.sum(np.abs(samples - mu) <= 2*sigma) / len(samples)
within_3_sigma = np.sum(np.abs(samples - mu) <= 3*sigma) / len(samples)

print(f"\nEmpirical Rule Verification:")
print(f"Within 1Ã: {within_1_sigma:.3f} (expected: 0.683)")
print(f"Within 2Ã: {within_2_sigma:.3f} (expected: 0.954)")
print(f"Within 3Ã: {within_3_sigma:.3f} (expected: 0.997)")

🔧#🔧 🔧V🔧i🔧s🔧u🔧a🔧l🔧i🔧z🔧a🔧t🔧i🔧o🔧n🔧
plt.figure(figsize=(15, 12))

🔧#🔧 🔧P🔧D🔧F🔧 🔧a🔧n🔧d🔧 🔧h🔧i🔧s🔧t🔧o🔧g🔧r🔧a🔧m🔧
plt.subplot(3, 2, 1)
plt.hist(samples, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
plt.plot(x_values, pdf_values, 'r-', linewidth=2, label=f'PDF: N({mu}, {sigma}²)')
plt.axvline(mu, color='red', linestyle='--', alpha=0.8, label=f'Mean = {mu}')
plt.axvline(mu + sigma, color='orange', linestyle='--', alpha=0.8, label=f'¼ + Ã')
plt.axvline(mu - sigma, color='orange', linestyle='--', alpha=0.8, label=f'¼ - Ã')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Normal Distribution PDF with Histogram')
plt.legend()
plt.grid(True, alpha=0.3)

🔧#🔧 🔧C🔧D🔧F🔧
plt.subplot(3, 2, 2)
plt.plot(x_values, cdf_values, 'b-', linewidth=2, label='CDF')
plt.axhline(0.5, color='red', linestyle='--', alpha=0.8, label='P = 0.5')
plt.axvline(mu, color='red', linestyle='--', alpha=0.8, label=f'Median = {mu}')
plt.xlabel('Value')
plt.ylabel('Cumulative Probability')
plt.title('Normal Distribution CDF')
plt.legend()
plt.grid(True, alpha=0.3)

🔧#🔧 🔧Q🔧-🔧Q🔧 🔧p🔧l🔧o🔧t🔧
plt.subplot(3, 2, 3)
stats.probplot(samples, dist="norm", plot=plt)
plt.title('Q-Q Plot: Testing Normality')
plt.grid(True, alpha=0.3)

🔧#🔧 🔧E🔧m🔧p🔧i🔧r🔧i🔧c🔧a🔧l🔧 🔧r🔧u🔧l🔧e🔧 🔧v🔧i🔧s🔧u🔧a🔧l🔧i🔧z🔧a🔧t🔧i🔧o🔧n🔧
plt.subplot(3, 2, 4)
x_emp = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
y_emp = stats.norm.pdf(x_emp, mu, sigma)
plt.plot(x_emp, y_emp, 'k-', linewidth=2, label='PDF')

🔧#🔧 🔧S🔧h🔧a🔧d🔧e🔧 🔧r🔧e🔧g🔧i🔧o🔧n🔧s🔧
plt.fill_between(x_emp, 0, y_emp, where=((x_emp >= mu-sigma) & (x_emp <= mu+sigma)), 
                alpha=0.3, color='blue', label='68% (1Ã)')
plt.fill_between(x_emp, 0, y_emp, where=((x_emp >= mu-2*sigma) & (x_emp <= mu+2*sigma)), 
                alpha=0.2, color='green', label='95% (2Ã)')
plt.fill_between(x_emp, 0, y_emp, where=((x_emp >= mu-3*sigma) & (x_emp <= mu+3*sigma)), 
                alpha=0.1, color='red', label='99.7% (3Ã)')

plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Empirical Rule (68-95-99.7)')
plt.legend()
plt.grid(True, alpha=0.3)

🔧#🔧 🔧S🔧t🔧a🔧n🔧d🔧a🔧r🔧d🔧 🔧n🔧o🔧r🔧m🔧a🔧l🔧 🔧c🔧o🔧m🔧p🔧a🔧r🔧i🔧s🔧o🔧n🔧
plt.subplot(3, 2, 5)
z_scores = (samples - mu) / sigma
plt.hist(z_scores, bins=50, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
x_std = np.linspace(-4, 4, 1000)
y_std = stats.norm.pdf(x_std, 0, 1)
plt.plot(x_std, y_std, 'r-', linewidth=2, label='Standard Normal N(0,1)')
plt.xlabel('Z-score')
plt.ylabel('Density')
plt.title('Standardized Data vs Standard Normal')
plt.legend()
plt.grid(True, alpha=0.3)

🔧#🔧 🔧M🔧u🔧l🔧t🔧i🔧p🔧l🔧e🔧 🔧n🔧o🔧r🔧m🔧a🔧l🔧 🔧d🔧i🔧s🔧t🔧r🔧i🔧b🔧u🔧t🔧i🔧o🔧n🔧s🔧 🔧c🔧o🔧m🔧p🔧a🔧r🔧i🔧s🔧o🔧n🔧
plt.subplot(3, 2, 6)
params = [(0, 1), (0, 2), (2, 1), (-1, 0.5)]
colors = ['blue', 'red', 'green', 'orange']
x_comp = np.linspace(-6, 6, 1000)

for (m, s), color in zip(params, colors):
    y_comp = stats.norm.pdf(x_comp, m, s)
    plt.plot(x_comp, y_comp, color=color, linewidth=2, 
            label=f'N({m}, {s}²)')

plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Comparison of Different Normal Distributions')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

🔧#🔧#🔧#🔧 🔧S🔧t🔧a🔧t🔧i🔧s🔧t🔧i🔧c🔧a🔧l🔧 🔧T🔧e🔧s🔧t🔧s🔧 🔧f🔧o🔧r🔧 🔧N🔧o🔧r🔧m🔧a🔧l🔧i🔧t🔧y🔧

```python
from scipy.stats import shapiro, normaltest, anderson, kstest
from sklearn.datasets import make_regression

def test_normality(data, alpha=0.05):
    """
    Perform multiple tests for normality
    
    Args:
        data: Array of data to test
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    results = {}
    
    # Shapiro-Wilk test
    shapiro_stat, shapiro_p = shapiro(data)
    results['Shapiro-Wilk'] = {
        'statistic': shapiro_stat,
        'p_value': shapiro_p,
        'is_normal': shapiro_p > alpha,
        'interpretation': 'Normal' if shapiro_p > alpha else 'Not Normal'
    }
    
    # D'Agostino-Pearson test
    dp_stat, dp_p = normaltest(data)
    results["D'Agostino-Pearson"] = {
        'statistic': dp_stat,
        'p_value': dp_p,
        'is_normal': dp_p > alpha,
        'interpretation': 'Normal' if dp_p > alpha else 'Not Normal'
    }
    
    # Anderson-Darling test
    ad_result = anderson(data, dist='norm')
    # Use critical value for 5% significance level
    critical_value = ad_result.critical_values[2]  # 5% level
    results['Anderson-Darling'] = {
        'statistic': ad_result.statistic,
        'critical_value': critical_value,
        'is_normal': ad_result.statistic < critical_value,
        'interpretation': 'Normal' if ad_result.statistic < critical_value else 'Not Normal'
    }
    
    # Kolmogorov-Smirnov test
    # First estimate parameters
    mu_est, sigma_est = np.mean(data), np.std(data, ddof=1)
    ks_stat, ks_p = kstest(data, lambda x: stats.norm.cdf(x, mu_est, sigma_est))
    results['Kolmogorov-Smirnov'] = {
        'statistic': ks_stat,
        'p_value': ks_p,
        'is_normal': ks_p > alpha,
        'interpretation': 'Normal' if ks_p > alpha else 'Not Normal'
    }
    
    return results

🔧#🔧 🔧T🔧e🔧s🔧t🔧 🔧w🔧i🔧t🔧h🔧 🔧d🔧i🔧f🔧f🔧e🔧r🔧e🔧n🔧t🔧 🔧t🔧y🔧p🔧e🔧s🔧 🔧o🔧f🔧 🔧d🔧a🔧t🔧a🔧
datasets = {
    'Normal Data': np.random.normal(0, 1, 1000),
    'Uniform Data': np.random.uniform(-2, 2, 1000),
    'Exponential Data': np.random.exponential(1, 1000),
    'Mixed Normal': np.concatenate([np.random.normal(-2, 1, 500), 
                                   np.random.normal(2, 1, 500)])
}

print("Normality Test Results:")
print("=" * 80)

for name, data in datasets.items():
    print(f"\n{name}:")
    print(f"Mean: {np.mean(data):.3f}, Std: {np.std(data, ddof=1):.3f}")
    
    results = test_normality(data)
    
    for test_name, result in results.items():
        if 'p_value' in result:
            print(f"{test_name:20}: p={result['p_value']:.4f}, {result['interpretation']}")
        else:
            print(f"{test_name:20}: stat={result['statistic']:.4f}, {result['interpretation']}")

🔧#🔧 🔧V🔧i🔧s🔧u🔧a🔧l🔧i🔧z🔧a🔧t🔧i🔧o🔧n🔧 🔧o🔧f🔧 🔧d🔧i🔧f🔧f🔧e🔧r🔧e🔧n🔧t🔧 🔧d🔧a🔧t🔧a🔧 🔧t🔧y🔧p🔧e🔧s🔧
plt.figure(figsize=(16, 10))

for i, (name, data) in enumerate(datasets.items()):
    # Histogram
    plt.subplot(2, 4, i+1)
    plt.hist(data, bins=50, density=True, alpha=0.7, color=f'C{i}', edgecolor='black')
    
    # Fit normal distribution
    mu_fit, sigma_fit = stats.norm.fit(data)
    x_fit = np.linspace(data.min(), data.max(), 100)
    y_fit = stats.norm.pdf(x_fit, mu_fit, sigma_fit)
    plt.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Fitted Normal')
    
    plt.title(f'{name}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    
    # Q-Q plot
    plt.subplot(2, 4, i+5)
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(f'{name} - Q-Q Plot')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

🔧#🔧#🔧#🔧 🔧N🔧o🔧r🔧m🔧a🔧l🔧 🔧D🔧i🔧s🔧t🔧r🔧i🔧b🔧u🔧t🔧i🔧o🔧n🔧 🔧i🔧n🔧 🔧M🔧a🔧c🔧h🔧i🔧n🔧e🔧 🔧L🔧e🔧a🔧r🔧n🔧i🔧n🔧g🔧 🔧C🔧o🔧n🔧t🔧e🔧x🔧t🔧

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

🔧#🔧 🔧D🔧e🔧m🔧o🔧n🔧s🔧t🔧r🔧a🔧t🔧e🔧 🔧n🔧o🔧r🔧m🔧a🔧l🔧 🔧d🔧i🔧s🔧t🔧r🔧i🔧b🔧u🔧t🔧i🔧o🔧n🔧 🔧a🔧s🔧s🔧u🔧m🔧p🔧t🔧i🔧o🔧n🔧s🔧 🔧i🔧n🔧 🔧M🔧L🔧
def demonstrate_ml_normality():
    """
    Show how normal distribution is used in machine learning
    """
    
    # Generate dataset
    X, y = make_classification(n_samples=1000, n_features=4, n_redundant=0, 
                              n_informative=4, n_clusters_per_class=1, 
                              random_state=42)
    
    feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]
    
    print("Dataset Analysis:")
    print(f"Dataset shape: {X.shape}")
    print(f"Classes: {np.unique(y)}")
    
    # Analyze feature distributions
    plt.figure(figsize=(16, 12))
    
    for i in range(X.shape[1]):
        # Histogram with normal overlay
        plt.subplot(3, 4, i+1)
        plt.hist(X[:, i], bins=30, density=True, alpha=0.7, color=f'C{i}')
        
        # Fit and plot normal distribution
        mu, sigma = stats.norm.fit(X[:, i])
        x_range = np.linspace(X[:, i].min(), X[:, i].max(), 100)
        plt.plot(x_range, stats.norm.pdf(x_range, mu, sigma), 'r-', linewidth=2)
        plt.title(f'{feature_names[i]} Distribution')
        plt.xlabel('Value')
        plt.ylabel('Density')
        
        # Q-Q plot
        plt.subplot(3, 4, i+5)
        stats.probplot(X[:, i], dist="norm", plot=plt)
        plt.title(f'{feature_names[i]} Q-Q Plot')
        
        # Test normality
        _, p_value = shapiro(X[:, i])
        print(f"{feature_names[i]} Shapiro-Wilk p-value: {p_value:.4f}")
    
    # Before and after standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Plot comparison
    for i in range(2):  # Just first 2 features for space
        plt.subplot(3, 4, i+9)
        plt.hist(X_scaled[:, i], bins=30, density=True, alpha=0.7, color='green')
        x_std = np.linspace(-4, 4, 100)
        plt.plot(x_std, stats.norm.pdf(x_std, 0, 1), 'r-', linewidth=2)
        plt.title(f'{feature_names[i]} After Standardization')
        plt.xlabel('Standardized Value')
        plt.ylabel('Density')
    
    plt.tight_layout()
    plt.show()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled = train_test_split(X_scaled, test_size=0.2, random_state=42)
    
    # Compare models with and without standardization
    models = {
        'Logistic Regression (Original)': LogisticRegression(random_state=42),
        'Logistic Regression (Scaled)': LogisticRegression(random_state=42),
        'Gaussian Naive Bayes (Original)': GaussianNB(),
        'Gaussian Naive Bayes (Scaled)': GaussianNB()
    }
    
    results = {}
    
    # Train and evaluate models
    for i, (name, model) in enumerate(models.items()):
        if 'Original' in name:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y_test)
        results[name] = accuracy
        
        print(f"\n{name}:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    
    return results

🔧#🔧 🔧R🔧u🔧n🔧 🔧d🔧e🔧m🔧o🔧n🔧s🔧t🔧r🔧a🔧t🔧i🔧o🔧n🔧
ml_results = demonstrate_ml_normality()

🔧#🔧 🔧C🔧o🔧m🔧p🔧a🔧r🔧e🔧 🔧r🔧e🔧s🔧u🔧l🔧t🔧s🔧
plt.figure(figsize=(10, 6))
names = list(ml_results.keys())
accuracies = list(ml_results.values())
colors = ['blue', 'lightblue', 'red', 'lightcoral']

bars = plt.bar(range(len(names)), accuracies, color=colors, alpha=0.7, edgecolor='black')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Performance: Original vs Standardized Features')
plt.xticks(range(len(names)), [name.replace(' (Original)', '\n(Original)').replace(' (Scaled)', '\n(Scaled)') 
                               for name in names], rotation=0)
plt.ylim(0, 1)

🔧#🔧 🔧A🔧d🔧d🔧 🔧v🔧a🔧l🔧u🔧e🔧 🔧l🔧a🔧b🔧e🔧l🔧s🔧 🔧o🔧n🔧 🔧b🔧a🔧r🔧s🔧
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()
```

🔧#🔧#🔧 🔧🔧🔧 🔧F🔧r🔧o🔧m🔧 🔧S🔧c🔧r🔧a🔧t🔧c🔧h🔧 🔧I🔧m🔧p🔧l🔧e🔧m🔧e🔧n🔧t🔧a🔧t🔧i🔧o🔧n🔧

🔧#🔧#🔧#🔧 🔧C🔧o🔧m🔧p🔧l🔧e🔧t🔧e🔧 🔧N🔧o🔧r🔧m🔧a🔧l🔧 🔧D🔧i🔧s🔧t🔧r🔧i🔧b🔧u🔧t🔧i🔧o🔧n🔧 🔧I🔧m🔧p🔧l🔧e🔧m🔧e🔧n🔧t🔧a🔧t🔧i🔧o🔧n🔧

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List, Tuple

class NormalDistribution:
    """
    Complete implementation of Normal Distribution from scratch
    """
    
    def __init__(self, mu: float = 0, sigma: float = 1):
        """
        Initialize Normal Distribution
        
        Args:
            mu: Mean parameter
            sigma: Standard deviation parameter (must be positive)
        """
        if sigma <= 0:
            raise ValueError("Standard deviation must be positive")
        
        self.mu = mu
        self.sigma = sigma
        self.variance = sigma ** 2
    
    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Probability Density Function
        
        Args:
            x: Value(s) to evaluate
            
        Returns:
            PDF value(s)
        """
        x = np.asarray(x)
        coefficient = 1 / (self.sigma * np.sqrt(2 * np.pi))
        exponent = -0.5 * ((x - self.mu) / self.sigma) ** 2
        return coefficient * np.exp(exponent)
    
    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Cumulative Distribution Function using error function approximation
        
        Args:
            x: Value(s) to evaluate
            
        Returns:
            CDF value(s)
        """
        x = np.asarray(x)
        z = (x - self.mu) / (self.sigma * np.sqrt(2))
        return 0.5 * (1 + self._erf(z))
    
    def _erf(self, z: np.ndarray) -> np.ndarray:
        """
        Error function approximation using Abramowitz and Stegun formula
        
        Args:
            z: Input values
            
        Returns:
            Error function values
        """
        # Constants for approximation
        a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
        p = 0.3275911
        
        # Save the sign of z
        sign = np.sign(z)
        z = np.abs(z)
        
        # A&S formula 7.1.26
        t = 1 / (1 + p * z)
        y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-z * z)
        
        return sign * y
    
    def ppf(self, p: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Percent Point Function (inverse CDF) using Beasley-Springer-Moro algorithm
        
        Args:
            p: Probability values (0 < p < 1)
            
        Returns:
            Quantile values
        """
        p = np.asarray(p)
        
        if np.any(p <= 0) or np.any(p >= 1):
            raise ValueError("Probabilities must be between 0 and 1")
        
        # Convert to standard normal quantiles first
        z = self._standard_normal_ppf(p)
        
        # Transform to desired distribution
        return self.mu + self.sigma * z
    
    def _standard_normal_ppf(self, p: np.ndarray) -> np.ndarray:
        """
        Standard normal PPF using Beasley-Springer-Moro algorithm
        """
        # Constants
        a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
        
        b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01]
        
        c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
        
        d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00]
        
        p_low = 0.02425
        p_high = 1 - p_low
        
        result = np.zeros_like(p)
        
        # Low region
        mask_low = p < p_low
        if np.any(mask_low):
            q = np.sqrt(-2 * np.log(p[mask_low]))
            result[mask_low] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                              ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        
        # Central region
        mask_central = (p >= p_low) & (p <= p_high)
        if np.any(mask_central):
            q = p[mask_central] - 0.5
            r = q * q
            result[mask_central] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
                                  (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        
        # High region
        mask_high = p > p_high
        if np.any(mask_high):
            q = np.sqrt(-2 * np.log(1 - p[mask_high]))
            result[mask_high] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        
        return result
    
    def sample(self, size: int = 1) -> Union[float, np.ndarray]:
        """
        Generate random samples using Box-Muller transformation
        
        Args:
            size: Number of samples to generate
            
        Returns:
            Random samples
        """
        # Box-Muller transformation
        if size % 2 == 1:
            size += 1
            trim = True
        else:
            trim = False
        
        # Generate uniform random numbers
        u1 = np.random.uniform(0, 1, size // 2)
        u2 = np.random.uniform(0, 1, size // 2)
        
        # Box-Muller transformation
        r = np.sqrt(-2 * np.log(u1))
        theta = 2 * np.pi * u2
        
        z1 = r * np.cos(theta)
        z2 = r * np.sin(theta)
        
        # Combine and transform to desired distribution
        samples = np.concatenate([z1, z2])
        samples = self.mu + self.sigma * samples
        
        if trim:
            samples = samples[:-1]
        
        return samples[0] if len(samples) == 1 else samples
    
    def fit(self, data: np.ndarray) -> 'NormalDistribution':
        """
        Fit normal distribution to data using Maximum Likelihood Estimation
        
        Args:
            data: Sample data
            
        Returns:
            Fitted NormalDistribution object
        """
        data = np.asarray(data)
        mu_mle = np.mean(data)
        sigma_mle = np.std(data, ddof=0)  # MLE uses population std
        
        return NormalDistribution(mu_mle, sigma_mle)
    
    def log_likelihood(self, data: np.ndarray) -> float:
        """
        Calculate log-likelihood of data
        
        Args:
            data: Sample data
            
        Returns:
            Log-likelihood value
        """
        data = np.asarray(data)
        n = len(data)
        
        ll = (-n/2) * np.log(2 * np.pi) - n * np.log(self.sigma) - \
             np.sum((data - self.mu)**2) / (2 * self.sigma**2)
        
        return ll
    
    def __str__(self) -> str:
        return f"Normal(¼={self.mu:.3f}, Ã={self.sigma:.3f})"
    
    def __repr__(self) -> str:
        return f"NormalDistribution(mu={self.mu}, sigma={self.sigma})"

🔧#🔧 🔧D🔧e🔧m🔧o🔧n🔧s🔧t🔧r🔧a🔧t🔧i🔧o🔧n🔧 🔧o🔧f🔧 🔧c🔧u🔧s🔧t🔧o🔧m🔧 🔧i🔧m🔧p🔧l🔧e🔧m🔧e🔧n🔧t🔧a🔧t🔧i🔧o🔧n🔧
def demo_custom_normal():
    """Demonstrate custom normal distribution implementation"""
    
    print("Custom Normal Distribution Implementation Demo")
    print("=" * 50)
    
    # Create distribution
    norm = NormalDistribution(mu=2, sigma=1.5)
    print(f"Distribution: {norm}")
    
    # Generate samples
    samples = norm.sample(10000)
    print(f"\nGenerated {len(samples)} samples")
    print(f"Sample mean: {np.mean(samples):.3f} (expected: {norm.mu})")
    print(f"Sample std: {np.std(samples, ddof=1):.3f} (expected: {norm.sigma})")
    
    # Test PDF, CDF, PPF
    test_values = np.array([-1, 0, 1, 2, 3, 4, 5])
    print(f"\nFunction evaluations:")
    print("Value\tPDF\t\tCDF\t\tPPF(CDF)")
    
    for val in test_values:
        pdf_val = norm.pdf(val)
        cdf_val = norm.cdf(val)
        ppf_val = norm.ppf(cdf_val) if 0 < cdf_val < 1 else np.nan
        print(f"{val:.1f}\t{pdf_val:.6f}\t{cdf_val:.6f}\t{ppf_val:.3f}")
    
    # Compare with scipy
    import scipy.stats as stats
    scipy_norm = stats.norm(norm.mu, norm.sigma)
    
    print(f"\nComparison with SciPy (first 5 test values):")
    print("Value\tCustom PDF\tSciPy PDF\tDiff PDF\tCustom CDF\tSciPy CDF\tDiff CDF")
    
    for val in test_values[:5]:
        custom_pdf = norm.pdf(val)
        scipy_pdf = scipy_norm.pdf(val)
        custom_cdf = norm.cdf(val)
        scipy_cdf = scipy_norm.cdf(val)
        
        print(f"{val:.1f}\t{custom_pdf:.6f}\t{scipy_pdf:.6f}\t{abs(custom_pdf-scipy_pdf):.2e}\t"
              f"{custom_cdf:.6f}\t{scipy_cdf:.6f}\t{abs(custom_cdf-scipy_cdf):.2e}")
    
    # Fit to data
    fitted_norm = NormalDistribution().fit(samples)
    print(f"\nFitted distribution: {fitted_norm}")
    print(f"Log-likelihood: {fitted_norm.log_likelihood(samples):.2f}")
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # PDF comparison
    x = np.linspace(norm.mu - 4*norm.sigma, norm.mu + 4*norm.sigma, 1000)
    custom_pdf = norm.pdf(x)
    scipy_pdf = scipy_norm.pdf(x)
    
    plt.subplot(2, 3, 1)
    plt.plot(x, custom_pdf, 'b-', linewidth=2, label='Custom Implementation')
    plt.plot(x, scipy_pdf, 'r--', linewidth=2, label='SciPy', alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('PDF')
    plt.title('PDF Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # CDF comparison
    custom_cdf = norm.cdf(x)
    scipy_cdf = scipy_norm.cdf(x)
    
    plt.subplot(2, 3, 2)
    plt.plot(x, custom_cdf, 'b-', linewidth=2, label='Custom Implementation')
    plt.plot(x, scipy_cdf, 'r--', linewidth=2, label='SciPy', alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('CDF')
    plt.title('CDF Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Sample histogram with fitted PDF
    plt.subplot(2, 3, 3)
    plt.hist(samples, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    plt.plot(x, norm.pdf(x), 'r-', linewidth=2, label='Original PDF')
    plt.plot(x, fitted_norm.pdf(x), 'g--', linewidth=2, label='Fitted PDF')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Samples with Original and Fitted PDF')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # PPF comparison
    p_values = np.linspace(0.01, 0.99, 100)
    custom_ppf = norm.ppf(p_values)
    scipy_ppf = scipy_norm.ppf(p_values)
    
    plt.subplot(2, 3, 4)
    plt.plot(p_values, custom_ppf, 'b-', linewidth=2, label='Custom Implementation')
    plt.plot(p_values, scipy_ppf, 'r--', linewidth=2, label='SciPy', alpha=0.7)
    plt.xlabel('Probability')
    plt.ylabel('Quantile')
    plt.title('PPF (Quantile Function) Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Error analysis
    pdf_errors = np.abs(custom_pdf - scipy_pdf)
    cdf_errors = np.abs(custom_cdf - scipy_cdf)
    
    plt.subplot(2, 3, 5)
    plt.semilogy(x, pdf_errors, 'b-', linewidth=2, label='PDF Error')
    plt.semilogy(x, cdf_errors, 'r-', linewidth=2, label='CDF Error')
    plt.xlabel('Value')
    plt.ylabel('Absolute Error (log scale)')
    plt.title('Implementation Error Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Box-Muller samples vs normal samples
    plt.subplot(2, 3, 6)
    box_muller_samples = norm.sample(1000)
    scipy_samples = scipy_norm.rvs(1000, random_state=42)
    
    plt.hist(box_muller_samples, bins=30, alpha=0.5, label='Box-Muller', density=True)
    plt.hist(scipy_samples, bins=30, alpha=0.5, label='SciPy', density=True)
    plt.plot(x, norm.pdf(x), 'k-', linewidth=2, label='True PDF')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Sample Generation Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return norm, samples

🔧#🔧 🔧R🔧u🔧n🔧 🔧d🔧e🔧m🔧o🔧n🔧s🔧t🔧r🔧a🔧t🔧i🔧o🔧n🔧
custom_norm, custom_samples = demo_custom_normal()
```

🔧#🔧#🔧#🔧 🔧A🔧d🔧v🔧a🔧n🔧c🔧e🔧d🔧 🔧S🔧t🔧a🔧t🔧i🔧s🔧t🔧i🔧c🔧a🔧l🔧 🔧M🔧e🔧t🔧h🔧o🔧d🔧s🔧

```python
class AdvancedNormalAnalysis:
    """
    Advanced methods for normal distribution analysis
    """
    
    @staticmethod
    def confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for the mean
        
        Args:
            data: Sample data
            confidence: Confidence level (0 < confidence < 1)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        n = len(data)
        mean = np.mean(data)
        std_err = np.std(data, ddof=1) / np.sqrt(n)
        
        # For large n, use normal distribution
        if n >= 30:
            from scipy import stats
            z_score = stats.norm.ppf((1 + confidence) / 2)
            margin_error = z_score * std_err
        else:
            # For small n, use t-distribution
            from scipy import stats
            t_score = stats.t.ppf((1 + confidence) / 2, n-1)
            margin_error = t_score * std_err
        
        return (mean - margin_error, mean + margin_error)
    
    @staticmethod
    def hypothesis_test_mean(data: np.ndarray, null_mean: float = 0, 
                           alternative: str = 'two-sided', alpha: float = 0.05) -> dict:
        """
        One-sample t-test for the mean
        
        Args:
            data: Sample data
            null_mean: Hypothesized population mean
            alternative: 'two-sided', 'greater', or 'less'
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        from scipy import stats
        
        n = len(data)
        sample_mean = np.mean(data)
        sample_std = np.std(data, ddof=1)
        
        # Test statistic
        t_stat = (sample_mean - null_mean) / (sample_std / np.sqrt(n))
        
        # P-value calculation
        if alternative == 'two-sided':
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n-1))
        elif alternative == 'greater':
            p_value = 1 - stats.t.cdf(t_stat, n-1)
        elif alternative == 'less':
            p_value = stats.t.cdf(t_stat, n-1)
        else:
            raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
        
        # Critical value
        if alternative == 'two-sided':
            critical_value = stats.t.ppf(1 - alpha/2, n-1)
        else:
            critical_value = stats.t.ppf(1 - alpha, n-1)
        
        reject_null = p_value < alpha
        
        return {
            'sample_mean': sample_mean,
            'null_mean': null_mean,
            't_statistic': t_stat,
            'p_value': p_value,
            'critical_value': critical_value,
            'reject_null': reject_null,
            'conclusion': f"{'Reject' if reject_null else 'Fail to reject'} the null hypothesis",
            'alpha': alpha,
            'alternative': alternative
        }
    
    @staticmethod
    def power_analysis(effect_size: float, alpha: float = 0.05, 
                      power: float = 0.8, alternative: str = 'two-sided') -> int:
        """
        Calculate required sample size for given power
        
        Args:
            effect_size: Cohen's d (standardized effect size)
            alpha: Significance level
            power: Desired statistical power
            alternative: Type of test
            
        Returns:
            Required sample size
        """
        from scipy import stats
        
        # Z-scores for alpha and power
        if alternative == 'two-sided':
            z_alpha = stats.norm.ppf(1 - alpha/2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)
        
        z_power = stats.norm.ppf(power)
        
        # Sample size calculation
        if alternative == 'two-sided':
            n = ((z_alpha + z_power) / effect_size) ** 2
        else:
            n = ((z_alpha + z_power) / effect_size) ** 2
        
        return int(np.ceil(n))
    
    @staticmethod
    def transformation_analysis(data: np.ndarray) -> dict:
        """
        Analyze data and suggest transformations to achieve normality
        
        Args:
            data: Sample data
            
        Returns:
            Dictionary with transformation analysis
        """
        from scipy import stats
        
        results = {
            'original': {
                'data': data,
                'shapiro_p': stats.shapiro(data)[1],
                'skewness': stats.skew(data),
                'kurtosis': stats.kurtosis(data)
            }
        }
        
        # Log transformation (for positive data)
        if np.all(data > 0):
            log_data = np.log(data)
            results['log'] = {
                'data': log_data,
                'shapiro_p': stats.shapiro(log_data)[1],
                'skewness': stats.skew(log_data),
                'kurtosis': stats.kurtosis(log_data)
            }
        
        # Square root transformation (for non-negative data)
        if np.all(data >= 0):
            sqrt_data = np.sqrt(data)
            results['sqrt'] = {
                'data': sqrt_data,
                'shapiro_p': stats.shapiro(sqrt_data)[1],
                'skewness': stats.skew(sqrt_data),
                'kurtosis': stats.kurtosis(sqrt_data)
            }
        
        # Box-Cox transformation
        if np.all(data > 0):
            try:
                boxcox_data, lambda_param = stats.boxcox(data)
                results['boxcox'] = {
                    'data': boxcox_data,
                    'lambda': lambda_param,
                    'shapiro_p': stats.shapiro(boxcox_data)[1],
                    'skewness': stats.skew(boxcox_data),
                    'kurtosis': stats.kurtosis(boxcox_data)
                }
            except:
                pass
        
        # Yeo-Johnson transformation (can handle negative values)
        try:
            yeojohnson_data, lambda_param = stats.yeojohnson(data)
            results['yeojohnson'] = {
                'data': yeojohnson_data,
                'lambda': lambda_param,
                'shapiro_p': stats.shapiro(yeojohnson_data)[1],
                'skewness': stats.skew(yeojohnson_data),
                'kurtosis': stats.kurtosis(yeojohnson_data)
            }
        except:
            pass
        
        # Find best transformation
        best_transform = max(results.keys(), 
                           key=lambda k: results[k]['shapiro_p'])
        results['best_transformation'] = best_transform
        
        return results

🔧#🔧 🔧D🔧e🔧m🔧o🔧n🔧s🔧t🔧r🔧a🔧t🔧i🔧o🔧n🔧 🔧o🔧f🔧 🔧a🔧d🔧v🔧a🔧n🔧c🔧e🔧d🔧 🔧m🔧e🔧t🔧h🔧o🔧d🔧s🔧
def demo_advanced_analysis():
    """Demonstrate advanced normal distribution analysis"""
    
    # Generate different types of data
    np.random.seed(42)
    
    datasets = {
        'Normal Data': np.random.normal(10, 2, 100),
        'Skewed Data': np.random.exponential(1, 100),
        'Heavy-tailed Data': np.random.standard_t(3, 100),
        'Bimodal Data': np.concatenate([np.random.normal(5, 1, 50), 
                                       np.random.normal(15, 1, 50)])
    }
    
    analyzer = AdvancedNormalAnalysis()
    
    print("Advanced Normal Distribution Analysis")
    print("=" * 60)
    
    for name, data in datasets.items():
        print(f"\n{name}:")
        print(f"Mean: {np.mean(data):.3f}, Std: {np.std(data, ddof=1):.3f}")
        
        # Confidence interval
        ci = analyzer.confidence_interval(data, 0.95)
        print(f"95% CI for mean: ({ci[0]:.3f}, {ci[1]:.3f})")
        
        # Hypothesis test (test if mean = 10)
        test_result = analyzer.hypothesis_test_mean(data, null_mean=10)
        print(f"T-test (H0: ¼=10): t={test_result['t_statistic']:.3f}, "
              f"p={test_result['p_value']:.4f}, {test_result['conclusion']}")
        
        # Power analysis
        effect_size = abs(np.mean(data) - 10) / np.std(data, ddof=1)
        required_n = analyzer.power_analysis(effect_size, power=0.8)
        print(f"Required sample size for 80% power: {required_n}")
        
        # Transformation analysis
        transforms = analyzer.transformation_analysis(data)
        print(f"Best transformation: {transforms['best_transformation']} "
              f"(Shapiro p-value: {transforms[transforms['best_transformation']]['shapiro_p']:.4f})")
    
    # Visualization of transformations
    plt.figure(figsize=(16, 12))
    
    for i, (name, data) in enumerate(datasets.items()):
        transforms = analyzer.transformation_analysis(data)
        
        # Original data
        plt.subplot(4, 4, i*4 + 1)
        plt.hist(data, bins=20, density=True, alpha=0.7, color=f'C{i}')
        plt.title(f'{name}\n(Original)')
        plt.xlabel('Value')
        plt.ylabel('Density')
        
        # Best transformation
        best_transform = transforms['best_transformation']
        if best_transform != 'original':
            best_data = transforms[best_transform]['data']
            
            plt.subplot(4, 4, i*4 + 2)
            plt.hist(best_data, bins=20, density=True, alpha=0.7, color=f'C{i}')
            plt.title(f'{name}\n({best_transform.title()})')
            plt.xlabel('Transformed Value')
            plt.ylabel('Density')
            
            # Q-Q plots
            plt.subplot(4, 4, i*4 + 3)
            stats.probplot(data, dist="norm", plot=plt)
            plt.title('Original Q-Q Plot')
            
            plt.subplot(4, 4, i*4 + 4)
            stats.probplot(best_data, dist="norm", plot=plt)
            plt.title(f'{best_transform.title()} Q-Q Plot')
        else:
            plt.subplot(4, 4, i*4 + 2)
            plt.text(0.5, 0.5, 'No transformation\nneeded', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.axis('off')
            
            plt.subplot(4, 4, i*4 + 3)
            stats.probplot(data, dist="norm", plot=plt)
            plt.title('Q-Q Plot')
            
            plt.subplot(4, 4, i*4 + 4)
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()

🔧#🔧 🔧R🔧u🔧n🔧 🔧a🔧d🔧v🔧a🔧n🔧c🔧e🔧d🔧 🔧a🔧n🔧a🔧l🔧y🔧s🔧i🔧s🔧 🔧d🔧e🔧m🔧o🔧n🔧s🔧t🔧r🔧a🔧t🔧i🔧o🔧n🔧
demo_advanced_analysis()
```

🔧#🔧#🔧 🔧 🔧🔧 🔧A🔧s🔧s🔧u🔧m🔧p🔧t🔧i🔧o🔧n🔧s🔧 🔧a🔧n🔧d🔧 🔧L🔧i🔧m🔧i🔧t🔧a🔧t🔧i🔧o🔧n🔧s🔧

🔧#🔧#🔧#🔧 🔧A🔧s🔧s🔧u🔧m🔧p🔧t🔧i🔧o🔧n🔧s🔧

**Mathematical Assumptions:**
- **Continuous data**: Variables are measured on a continuous scale
- **Independence**: Observations are independent of each other  
- **Infinite support**: Theoretically, values can range from - to +
- **Symmetry**: Distribution is perfectly symmetric around the mean
- **Single mode**: Only one peak in the distribution

**Statistical Assumptions in ML:**
- **IID samples**: Data points are independently and identically distributed
- **Stationarity**: Distribution parameters don't change over time
- **Linearity**: Linear relationships between variables (in linear models)
- **Homoscedasticity**: Constant variance across all levels of independent variables
- **No outliers**: Extreme values don't significantly affect the distribution

🔧#🔧#🔧#🔧 🔧L🔧i🔧m🔧i🔧t🔧a🔧t🔧i🔧o🔧n🔧s🔧

**Theoretical Limitations:**
- **Infinite tails**: Assigns non-zero probability to extreme values (can be unrealistic)
- **Symmetry assumption**: Real-world data often shows skewness
- **Single modality**: Cannot model multimodal distributions
- **Parameter sensitivity**: Small changes in ¼ or Ã can significantly affect probabilities
- **Curse of dimensionality**: In high dimensions, most data lies far from the center

**Practical Limitations:**
- **Finite data**: Real datasets have finite ranges, unlike theoretical normal distribution
- **Measurement precision**: Discrete measurements approximate continuous distributions
- **Outlier sensitivity**: Sample statistics heavily influenced by extreme values
- **Model assumptions**: Many statistical tests assume normality but real data may not follow this
- **Transformation needs**: Data often requires preprocessing to achieve normality

🔧#🔧#🔧#🔧 🔧W🔧h🔧e🔧n🔧 🔧N🔧o🔧r🔧m🔧a🔧l🔧i🔧t🔧y🔧 🔧A🔧s🔧s🔧u🔧m🔧p🔧t🔧i🔧o🔧n🔧s🔧 🔧F🔧a🔧i🔧l🔧

**Common Violations:**

| Issue | Description | Detection | Solutions |
|-------|-------------|-----------|-----------|
| **Skewness** | Asymmetric distribution | Histogram, skewness statistic | Log transform, Box-Cox |
| **Heavy tails** | More extreme values than expected | Kurtosis, Q-Q plots | Robust methods, t-distribution |
| **Multimodality** | Multiple peaks | Histogram, density plots | Mixture models, clustering |
| **Discrete data** | Integer or categorical values | Data inspection | Poisson, binomial models |
| **Bounded data** | Limited range (e.g., percentages) | Domain knowledge | Beta distribution, logit transform |

**Diagnostic Tools:**
- **Visual**: Histograms, Q-Q plots, box plots
- **Statistical**: Shapiro-Wilk, Anderson-Darling, Kolmogorov-Smirnov tests
- **Descriptive**: Skewness, kurtosis, range checks

🔧#🔧#🔧#🔧 🔧R🔧o🔧b🔧u🔧s🔧t🔧 🔧A🔧l🔧t🔧e🔧r🔧n🔧a🔧t🔧i🔧v🔧e🔧s🔧

**When to Use Alternatives:**

1. **t-Distribution**: For heavy-tailed data or small samples
2. **Log-normal**: For positively skewed data
3. **Gamma/Exponential**: For non-negative, skewed data  
4. **Beta**: For bounded data (0,1)
5. **Mixture Models**: For multimodal data
6. **Non-parametric Methods**: When no distributional assumptions can be made

🔧#🔧#🔧 🔧=🔧¡🔧 🔧I🔧n🔧t🔧e🔧r🔧v🔧i🔧e🔧w🔧 🔧Q🔧u🔧e🔧s🔧t🔧i🔧o🔧n🔧s🔧

??? question "**Q1: Explain the difference between normal distribution and standard normal distribution.**"

    **Answer:**
    
    **Normal Distribution:**
    - General form: $N(\mu, \sigma^2)$
    - Can have any mean $\mu$ and standard deviation $\sigma > 0$
    - PDF: $f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$
    
    **Standard Normal Distribution:**
    - Special case: $N(0, 1)$
    - Mean = 0, Standard deviation = 1
    - PDF: $\phi(z) = \frac{1}{\sqrt{2\pi}} e^{-\frac{z^2}{2}}$
    - Also called z-distribution
    
    **Relationship:**
    Any normal distribution can be converted to standard normal using z-score transformation:
    $$Z = \frac{X - \mu}{\sigma}$$
    
    **Why it's important:**
    - Standardizes comparisons across different scales
    - Simplifies probability calculations
    - Used in hypothesis testing and confidence intervals
    - Tables and software often reference standard normal

??? question "**Q2: What is the Central Limit Theorem and why is it important for the normal distribution?**"

    **Answer:**
    
    **Central Limit Theorem (CLT) states:**
    For any population with mean $\mu$ and finite variance $\sigma^2$, the distribution of sample means approaches normal as sample size increases:
    
    $$\bar{X}_n \sim N\left(\mu, \frac{\sigma^2}{n}\right) \text{ as } n \to \infty$$
    
    **Key Points:**
    - Original population can have ANY distribution
    - Sample size typically needs to be e30 for good approximation
    - Larger samples  better normal approximation
    - Standard error decreases as $\frac{\sigma}{\sqrt{n}}$
    
    **Importance:**
    1. **Statistical Inference**: Enables confidence intervals and hypothesis tests
    2. **Machine Learning**: Justifies normality assumptions in many algorithms
    3. **Quality Control**: Control charts based on sample means
    4. **A/B Testing**: Comparison of group means
    
    **Example:**
    Even if individual heights are not perfectly normal, the average height of groups of 30+ people will be approximately normal.

??? question "**Q3: How do you test if data follows a normal distribution?**"

    **Answer:**
    
    **Visual Methods:**
    1. **Histogram**: Should show bell-shaped curve
    2. **Q-Q Plot**: Points should lie on straight line
    3. **Box Plot**: Should be symmetric with few outliers
    
    **Statistical Tests:**
    
    1. **Shapiro-Wilk Test** (best for n < 50):
       - H: Data is normally distributed
       - Most powerful test for normality
       - `scipy.stats.shapiro(data)`
    
    2. **Anderson-Darling Test**:
       - More sensitive to tail deviations
       - `scipy.stats.anderson(data, dist='norm')`
    
    3. **Kolmogorov-Smirnov Test**:
       - Tests against fitted normal distribution
       - Less powerful than others
    
    4. **D'Agostino-Pearson Test**:
       - Based on skewness and kurtosis
       - `scipy.stats.normaltest(data)`
    
    **Rule of Thumb:**
    - Use multiple methods together
    - Visual inspection is crucial
    - Tests may reject normality for large samples due to minor deviations
    - Consider practical significance, not just statistical significance

??? question "**Q4: What are the parameters of normal distribution and how do they affect the shape?**"

    **Answer:**
    
    **Parameters:**
    
    1. **Mean (¼)** - Location parameter:
       - Determines center of distribution
       - Peak of bell curve occurs at ¼
       - Shifting ¼ moves entire curve left/right
       - Range: $-\infty < \mu < \infty$
    
    2. **Standard Deviation (Ã)** - Scale parameter:
       - Determines spread/width of distribution
       - Controls how dispersed values are around mean
       - Larger Ã  wider, flatter curve
       - Smaller Ã  narrower, taller curve
       - Range: $Ã > 0$
    
    **Shape Effects:**
    ```
    ¼ = 0, Ã = 1: Standard normal (tall, narrow)
    ¼ = 0, Ã = 2: Same center, wider spread
    ¼ = 5, Ã = 1: Shifted right, same spread
    ¼ = 5, Ã = 2: Shifted right, wider spread
    ```
    
    **Mathematical Properties:**
    - Mode = Median = Mean = ¼
    - Inflection points at ¼ ± Ã
    - 68% of data within ¼ ± Ã
    - 95% of data within ¼ ± 2Ã
    - 99.7% of data within ¼ ± 3Ã

??? question "**Q5: Explain the 68-95-99.7 rule (Empirical Rule) and its applications.**"

    **Answer:**
    
    **The Empirical Rule states:**
    For any normal distribution:
    - **68%** of values lie within 1 standard deviation: $P(\mu - \sigma \leq X \leq \mu + \sigma) = 0.6827$
    - **95%** of values lie within 2 standard deviations: $P(\mu - 2\sigma \leq X \leq \mu + 2\sigma) = 0.9545$
    - **99.7%** of values lie within 3 standard deviations: $P(\mu - 3\sigma \leq X \leq \mu + 3\sigma) = 0.9973$
    
    **Applications:**
    
    1. **Quality Control**:
       - Products outside 3Ã limits considered defective
       - Six Sigma methodology aims for 6Ã quality
    
    2. **Outlier Detection**:
       - Values beyond 2Ã or 3Ã flagged as outliers
       - Z-scores > 3 are rare (0.3% probability)
    
    3. **Risk Assessment**:
       - Financial returns: VaR calculations
       - Insurance: Claim amount predictions
    
    4. **Educational Testing**:
       - Standardized test scores (SAT, GRE)
       - Grade curving and percentile ranks
    
    5. **Medical Diagnostics**:
       - Normal ranges for lab values
       - Growth charts for children
    
    **Example:**
    If IQ scores are N(100, 15):
    - 68% of people have IQ between 85-115
    - 95% have IQ between 70-130  
    - 99.7% have IQ between 55-145

??? question "**Q6: How is normal distribution used in machine learning algorithms?**"

    **Answer:**
    
    **Direct Usage:**
    
    1. **Naive Bayes Classifier**:
       - Assumes features follow normal distribution
       - Uses Gaussian likelihood: $P(x_i|y) = N(\mu_{i,y}, \sigma_{i,y}^2)$
    
    2. **Linear Regression**:
       - Assumes residuals are normally distributed
       - Enables confidence intervals and hypothesis tests
    
    3. **Discriminant Analysis (LDA/QDA)**:
       - Assumes classes have multivariate normal distributions
       - Decision boundaries based on Gaussian densities
    
    **Indirect Usage:**
    
    1. **Weight Initialization**:
       - Neural networks: Xavier/He initialization
       - Random weights from normal distribution
    
    2. **Regularization**:
       - Gaussian priors in Bayesian methods
       - L2 regularization equivalent to normal prior
    
    3. **Feature Engineering**:
       - Box-Cox transformation to achieve normality
       - StandardScaler assumes normal-like distribution
    
    4. **Uncertainty Quantification**:
       - Bayesian neural networks
       - Gaussian processes
    
    5. **Generative Models**:
       - VAE latent space often assumed normal
       - Normalizing flows
    
    **Why Normal Distribution is Preferred:**
    - Mathematical tractability
    - Central Limit Theorem justification
    - Maximum entropy for given mean and variance
    - Conjugate priors in Bayesian inference

??? question "**Q7: What is the relationship between normal distribution and maximum likelihood estimation?**"

    **Answer:**
    
    **MLE for Normal Distribution:**
    
    Given samples $x_1, x_2, ..., x_n$ from $N(\mu, \sigma^2)$:
    
    **Likelihood Function:**
    $$L(\mu, \sigma^2) = \prod_{i=1}^{n} \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x_i-\mu)^2}{2\sigma^2}}$$
    
    **Log-Likelihood:**
    $$\ell(\mu, \sigma^2) = -\frac{n}{2}\ln(2\pi) - n\ln(\sigma) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i-\mu)^2$$
    
    **MLE Estimators:**
    Taking derivatives and setting to zero:
    
    $$\hat{\mu}_{MLE} = \frac{1}{n}\sum_{i=1}^{n} x_i = \bar{x}$$
    
    $$\hat{\sigma}^2_{MLE} = \frac{1}{n}\sum_{i=1}^{n} (x_i - \bar{x})^2$$
    
    **Key Properties:**
    - Sample mean is unbiased: $E[\hat{\mu}] = \mu$
    - MLE variance estimator is biased: $E[\hat{\sigma}^2_{MLE}] = \frac{n-1}{n}\sigma^2$
    - Unbiased estimator uses $n-1$ in denominator
    - Both estimators are consistent and efficient
    
    **Connection to Machine Learning:**
    - Linear regression with Gaussian noise assumption
    - Foundation for many statistical tests
    - Basis for confidence intervals

??? question "**Q8: How do you handle non-normal data in machine learning?**"

    **Answer:**
    
    **Detection Methods:**
    - Visual inspection (histograms, Q-Q plots)
    - Statistical tests (Shapiro-Wilk, Anderson-Darling)
    - Skewness and kurtosis analysis
    
    **Transformation Techniques:**
    
    1. **Log Transformation**: For right-skewed data
       - $y = \log(x)$ (requires $x > 0$)
       - Reduces positive skewness
    
    2. **Square Root**: For count data or mild skewness
       - $y = \sqrt{x}$ (requires $x \geq 0$)
    
    3. **Box-Cox Transformation**: For positive data
       - $y = \frac{x^{\lambda} - 1}{\lambda}$ (if $\lambda \neq 0$)
       - $y = \log(x)$ (if $\lambda = 0$)
       - Automatically finds optimal »
    
    4. **Yeo-Johnson**: Handles negative values
       - Extension of Box-Cox for all real numbers
    
    **Alternative Approaches:**
    
    1. **Robust Methods**:
       - Use median instead of mean
       - Robust regression (Huber loss)
       - Trimmed statistics
    
    2. **Non-parametric Methods**:
       - Random Forest, SVM
       - k-NN, Decision Trees
       - No distributional assumptions
    
    3. **Different Distributions**:
       - Poisson for count data
       - Binomial for binary outcomes  
       - Exponential for survival times
    
    4. **Ensemble Methods**:
       - Bootstrap aggregating
       - Less sensitive to individual distributions

??? question "**Q9: What is standardization and why is it important when features follow normal distributions?**"

    **Answer:**
    
    **Standardization (Z-score normalization):**
    Transform features to have mean=0 and std=1:
    
    $$z = \frac{x - \mu}{\sigma}$$
    
    **Why Important for Normal Data:**
    
    1. **Scale Independence**:
       - Features with different units become comparable
       - Example: Age (0-100) vs Income ($0-$100,000)
    
    2. **Algorithm Performance**:
       - Distance-based algorithms (k-NN, SVM, k-means)
       - Gradient descent convergence
       - Neural network training stability
    
    3. **Mathematical Properties**:
       - Preserves normal distribution shape
       - Standardized normal has known properties
       - Enables use of z-tables and standard formulas
    
    **When Normal Assumption Helps:**
    - 68-95-99.7 rule applies after standardization
    - Outlier detection using z-scores
    - Statistical tests and confidence intervals
    - Feature importance comparison
    
    **Implementation:**
    ```python
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ```
    
    **Alternative: Min-Max Normalization**:
    - Scales to [0,1] range
    - Better when distribution is not normal
    - Preserves original distribution shape

??? question "**Q10: Explain the concept of multivariate normal distribution and its applications.**"

    **Answer:**
    
    **Multivariate Normal Distribution:**
    Extension of normal distribution to multiple variables:
    
    $$\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$$
    
    **PDF:**
    $$f(\mathbf{x}) = \frac{1}{(2\pi)^{k/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$
    
    Where:
    - $\mathbf{x} = [x_1, x_2, ..., x_k]^T$ is k-dimensional vector
    - $\boldsymbol{\mu}$ is mean vector
    - $\boldsymbol{\Sigma}$ is covariance matrix
    
    **Key Properties:**
    - Marginal distributions are normal
    - Linear combinations are normal
    - Conditional distributions are normal
    - Zero correlation implies independence
    
    **Applications in ML:**
    
    1. **Gaussian Mixture Models (GMM)**:
       - Clustering with probabilistic assignments
       - Each cluster is a multivariate normal
    
    2. **Principal Component Analysis (PCA)**:
       - Assumes data follows multivariate normal
       - Finds principal directions of variation
    
    3. **Linear Discriminant Analysis (LDA)**:
       - Classes assumed multivariate normal
       - Same covariance matrix across classes
    
    4. **Gaussian Processes**:
       - Function values follow multivariate normal
       - Used in Bayesian optimization
    
    5. **Kalman Filters**:
       - State estimation in time series
       - System and observation noise assumed normal
    
    **Covariance Matrix Interpretation:**
    - Diagonal: Individual variable variances
    - Off-diagonal: Correlations between variables
    - Eigenvalues: Principal component variances
    - Eigenvectors: Principal directions

🔧#🔧#🔧 🔧🧠🔧 🔧E🔧x🔧a🔧m🔧p🔧l🔧e🔧s🔧

🔧#🔧#🔧#🔧 🔧R🔧e🔧a🔧l🔧-🔧w🔧o🔧r🔧l🔧d🔧 🔧E🔧x🔧a🔧m🔧p🔧l🔧e🔧:🔧 🔧Q🔧u🔧a🔧l🔧i🔧t🔧y🔧 🔧C🔧o🔧n🔧t🔧r🔧o🔧l🔧 🔧i🔧n🔧 🔧M🔧a🔧n🔧u🔧f🔧a🔧c🔧t🔧u🔧r🔧i🔧n🔧g🔧

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
import seaborn as sns

🔧#🔧 🔧M🔧a🔧n🔧u🔧f🔧a🔧c🔧t🔧u🔧r🔧i🔧n🔧g🔧 🔧Q🔧u🔧a🔧l🔧i🔧t🔧y🔧 🔧C🔧o🔧n🔧t🔧r🔧o🔧l🔧 🔧E🔧x🔧a🔧m🔧p🔧l🔧e🔧
np.random.seed(42)

def simulate_manufacturing_data():
    """
    Simulate manufacturing data with normal distribution assumptions
    """
    # Simulate production data over 30 days
    n_days = 30
    samples_per_day = 50
    
    # Target specifications
    target_length = 100.0  # mm
    tolerance = ±2.0  # mm (so acceptable range is 98-102 mm)
    process_std = 0.8  # mm (process standard deviation)
    
    # Simulate daily production
    production_data = []
    
    for day in range(1, n_days + 1):
        # Daily mean might drift slightly (process variation)
        daily_mean = target_length + np.random.normal(0, 0.2)
        
        # Generate samples for the day
        daily_samples = np.random.normal(daily_mean, process_std, samples_per_day)
        
        for sample in daily_samples:
            production_data.append({
                'day': day,
                'length': sample,
                'within_spec': 98 <= sample <= 102,
                'defect_type': 'none' if 98 <= sample <= 102 else ('short' if sample < 98 else 'long')
            })
    
    return pd.DataFrame(production_data)

🔧#🔧 🔧G🔧e🔧n🔧e🔧r🔧a🔧t🔧e🔧 🔧m🔧a🔧n🔧u🔧f🔧a🔧c🔧t🔧u🔧r🔧i🔧n🔧g🔧 🔧d🔧a🔧t🔧a🔧
manufacturing_df = simulate_manufacturing_data()

print("Manufacturing Quality Control Analysis")
print("=" * 50)
print(f"Total samples: {len(manufacturing_df)}")
print(f"Target length: 100.0 mm ± 2.0 mm")
print(f"Overall mean: {manufacturing_df['length'].mean():.3f} mm")
print(f"Overall std: {manufacturing_df['length'].std():.3f} mm")
print(f"Defect rate: {(~manufacturing_df['within_spec']).mean():.1%}")

🔧#🔧 🔧D🔧a🔧i🔧l🔧y🔧 🔧s🔧t🔧a🔧t🔧i🔧s🔧t🔧i🔧c🔧s🔧
daily_stats = manufacturing_df.groupby('day')['length'].agg(['mean', 'std', 'count'])
daily_defect_rate = manufacturing_df.groupby('day')['within_spec'].apply(lambda x: (~x).mean())

🔧#🔧 🔧C🔧o🔧n🔧t🔧r🔧o🔧l🔧 🔧c🔧h🔧a🔧r🔧t🔧 🔧a🔧n🔧a🔧l🔧y🔧s🔧i🔧s🔧 🔧u🔧s🔧i🔧n🔧g🔧 🔧n🔧o🔧r🔧m🔧a🔧l🔧 🔧d🔧i🔧s🔧t🔧r🔧i🔧b🔧u🔧t🔧i🔧o🔧n🔧
overall_mean = manufacturing_df['length'].mean()
overall_std = manufacturing_df['length'].std()

🔧#🔧 🔧C🔧o🔧n🔧t🔧r🔧o🔧l🔧 🔧l🔧i🔧m🔧i🔧t🔧s🔧 🔧(🔧3🔧-🔧s🔧i🔧g🔧m🔧a🔧 🔧r🔧u🔧l🔧e🔧)🔧
ucl = overall_mean + 3 * overall_std  # Upper Control Limit
lcl = overall_mean - 3 * overall_std  # Lower Control Limit
usl = 102  # Upper Specification Limit
lsl = 98   # Lower Specification Limit

print(f"\nControl Chart Limits:")
print(f"Upper Control Limit (UCL): {ucl:.3f} mm")
print(f"Lower Control Limit (LCL): {lcl:.3f} mm")
print(f"Upper Specification Limit (USL): {usl:.1f} mm")
print(f"Lower Specification Limit (LSL): {lsl:.1f} mm")

🔧#🔧 🔧P🔧r🔧o🔧c🔧e🔧s🔧s🔧 🔧c🔧a🔧p🔧a🔧b🔧i🔧l🔧i🔧t🔧y🔧 🔧a🔧n🔧a🔧l🔧y🔧s🔧i🔧s🔧
cp = (usl - lsl) / (6 * overall_std)  # Process capability
cpk = min((usl - overall_mean)/(3 * overall_std), 
          (overall_mean - lsl)/(3 * overall_std))  # Process capability index

print(f"\nProcess Capability:")
print(f"Cp (potential capability): {cp:.3f}")
print(f"Cpk (actual capability): {cpk:.3f}")
print(f"Process capability interpretation:")
if cpk >= 1.33:
    print("  Excellent process (< 63 PPM defects)")
elif cpk >= 1.0:
    print("  Adequate process (< 2,700 PPM defects)")
else:
    print("  Poor process (> 2,700 PPM defects)")

🔧#🔧 🔧C🔧a🔧l🔧c🔧u🔧l🔧a🔧t🔧e🔧 🔧t🔧h🔧e🔧o🔧r🔧e🔧t🔧i🔧c🔧a🔧l🔧 🔧d🔧e🔧f🔧e🔧c🔧t🔧 🔧r🔧a🔧t🔧e🔧s🔧 🔧u🔧s🔧i🔧n🔧g🔧 🔧n🔧o🔧r🔧m🔧a🔧l🔧 🔧d🔧i🔧s🔧t🔧r🔧i🔧b🔧u🔧t🔧i🔧o🔧n🔧
z_usl = (usl - overall_mean) / overall_std
z_lsl = (lsl - overall_mean) / overall_std

prob_exceed_usl = 1 - stats.norm.cdf(z_usl)
prob_below_lsl = stats.norm.cdf(z_lsl)
theoretical_defect_rate = prob_exceed_usl + prob_below_lsl

print(f"\nDefect Rate Analysis:")
print(f"Observed defect rate: {(~manufacturing_df['within_spec']).mean():.1%}")
print(f"Theoretical defect rate (based on normal): {theoretical_defect_rate:.1%}")
print(f"Theoretical PPM: {theoretical_defect_rate * 1e6:.0f}")

🔧#🔧 🔧V🔧i🔧s🔧u🔧a🔧l🔧i🔧z🔧a🔧t🔧i🔧o🔧n🔧
plt.figure(figsize=(20, 15))

🔧#🔧 🔧1🔧.🔧 🔧O🔧v🔧e🔧r🔧a🔧l🔧l🔧 🔧d🔧i🔧s🔧t🔧r🔧i🔧b🔧u🔧t🔧i🔧o🔧n🔧 🔧w🔧i🔧t🔧h🔧 🔧s🔧p🔧e🔧c🔧i🔧f🔧i🔧c🔧a🔧t🔧i🔧o🔧n🔧s🔧
plt.subplot(3, 4, 1)
plt.hist(manufacturing_df['length'], bins=50, density=True, alpha=0.7, 
         color='lightblue', edgecolor='black', label='Observed Data')

🔧#🔧 🔧F🔧i🔧t🔧 🔧n🔧o🔧r🔧m🔧a🔧l🔧 🔧d🔧i🔧s🔧t🔧r🔧i🔧b🔧u🔧t🔧i🔧o🔧n🔧
mu_fit, sigma_fit = stats.norm.fit(manufacturing_df['length'])
x = np.linspace(manufacturing_df['length'].min() - 1, manufacturing_df['length'].max() + 1, 1000)
plt.plot(x, stats.norm.pdf(x, mu_fit, sigma_fit), 'r-', linewidth=2, label='Fitted Normal')

🔧#🔧 🔧A🔧d🔧d🔧 🔧s🔧p🔧e🔧c🔧i🔧f🔧i🔧c🔧a🔧t🔧i🔧o🔧n🔧 🔧l🔧i🔧m🔧i🔧t🔧s🔧
plt.axvline(lsl, color='red', linestyle='--', linewidth=2, label='Spec Limits')
plt.axvline(usl, color='red', linestyle='--', linewidth=2)
plt.axvline(overall_mean, color='green', linestyle='-', linewidth=2, label='Process Mean')

plt.xlabel('Length (mm)')
plt.ylabel('Density')
plt.title('Overall Distribution vs Specifications')
plt.legend()
plt.grid(True, alpha=0.3)

🔧#🔧 🔧2🔧.🔧 🔧C🔧o🔧n🔧t🔧r🔧o🔧l🔧 🔧c🔧h🔧a🔧r🔧t🔧 🔧f🔧o🔧r🔧 🔧d🔧a🔧i🔧l🔧y🔧 🔧m🔧e🔧a🔧n🔧s🔧
plt.subplot(3, 4, 2)
plt.plot(daily_stats.index, daily_stats['mean'], 'bo-', markersize=4)
plt.axhline(overall_mean, color='green', linestyle='-', label='Grand Mean')
plt.axhline(overall_mean + 3*overall_std/np.sqrt(50), color='red', linestyle='--', label='±3Ã limits')
plt.axhline(overall_mean - 3*overall_std/np.sqrt(50), color='red', linestyle='--')
plt.xlabel('Day')
plt.ylabel('Daily Mean Length (mm)')
plt.title('X-bar Control Chart')
plt.legend()
plt.grid(True, alpha=0.3)

🔧#🔧 🔧3🔧.🔧 🔧C🔧o🔧n🔧t🔧r🔧o🔧l🔧 🔧c🔧h🔧a🔧r🔧t🔧 🔧f🔧o🔧r🔧 🔧d🔧a🔧i🔧l🔧y🔧 🔧r🔧a🔧n🔧g🔧e🔧s🔧
plt.subplot(3, 4, 3)
plt.plot(daily_stats.index, daily_stats['std'], 'go-', markersize=4)
plt.axhline(overall_std, color='blue', linestyle='-', label='Grand Std')
plt.xlabel('Day')
plt.ylabel('Daily Std (mm)')
plt.title('S Control Chart')
plt.legend()
plt.grid(True, alpha=0.3)

🔧#🔧 🔧4🔧.🔧 🔧D🔧a🔧i🔧l🔧y🔧 🔧d🔧e🔧f🔧e🔧c🔧t🔧 🔧r🔧a🔧t🔧e🔧s🔧
plt.subplot(3, 4, 4)
plt.bar(daily_defect_rate.index, daily_defect_rate.values * 100, alpha=0.7, color='orange')
plt.axhline(theoretical_defect_rate * 100, color='red', linestyle='--', 
           label=f'Expected: {theoretical_defect_rate:.1%}')
plt.xlabel('Day')
plt.ylabel('Defect Rate (%)')
plt.title('Daily Defect Rates')
plt.legend()
plt.grid(True, alpha=0.3)

🔧#🔧 🔧5🔧.🔧 🔧Q🔧-🔧Q🔧 🔧p🔧l🔧o🔧t🔧 🔧t🔧o🔧 🔧v🔧e🔧r🔧i🔧f🔧y🔧 🔧n🔧o🔧r🔧m🔧a🔧l🔧i🔧t🔧y🔧
plt.subplot(3, 4, 5)
stats.probplot(manufacturing_df['length'], dist="norm", plot=plt)
plt.title('Q-Q Plot: Normality Check')
plt.grid(True, alpha=0.3)

🔧#🔧 🔧6🔧.🔧 🔧I🔧n🔧d🔧i🔧v🔧i🔧d🔧u🔧a🔧l🔧 🔧m🔧e🔧a🔧s🔧u🔧r🔧e🔧m🔧e🔧n🔧t🔧s🔧 🔧c🔧o🔧n🔧t🔧r🔧o🔧l🔧 🔧c🔧h🔧a🔧r🔧t🔧
plt.subplot(3, 4, 6)
sample_subset = manufacturing_df.head(100)  # First 100 samples
plt.plot(range(len(sample_subset)), sample_subset['length'], 'b-', alpha=0.6)
plt.axhline(overall_mean, color='green', linestyle='-', label='Mean')
plt.axhline(ucl, color='red', linestyle='--', label='Control Limits')
plt.axhline(lcl, color='red', linestyle='--')
plt.axhline(usl, color='orange', linestyle=':', label='Spec Limits')
plt.axhline(lsl, color='orange', linestyle=':')
plt.xlabel('Sample Number')
plt.ylabel('Length (mm)')
plt.title('Individual Measurements (First 100)')
plt.legend()
plt.grid(True, alpha=0.3)

🔧#🔧 🔧7🔧.🔧 🔧P🔧r🔧o🔧c🔧e🔧s🔧s🔧 🔧c🔧a🔧p🔧a🔧b🔧i🔧l🔧i🔧t🔧y🔧 🔧v🔧i🔧s🔧u🔧a🔧l🔧i🔧z🔧a🔧t🔧i🔧o🔧n🔧
plt.subplot(3, 4, 7)
x_cap = np.linspace(94, 106, 1000)
y_cap = stats.norm.pdf(x_cap, overall_mean, overall_std)
plt.plot(x_cap, y_cap, 'b-', linewidth=2, label='Process Distribution')
plt.fill_between(x_cap, 0, y_cap, where=((x_cap >= lsl) & (x_cap <= usl)), 
                alpha=0.3, color='green', label='Within Spec')
plt.fill_between(x_cap, 0, y_cap, where=(x_cap < lsl), 
                alpha=0.3, color='red', label='Below LSL')
plt.fill_between(x_cap, 0, y_cap, where=(x_cap > usl), 
                alpha=0.3, color='red', label='Above USL')
plt.axvline(lsl, color='red', linestyle='--', linewidth=2)
plt.axvline(usl, color='red', linestyle='--', linewidth=2)
plt.xlabel('Length (mm)')
plt.ylabel('Density')
plt.title(f'Process Capability (Cpk = {cpk:.3f})')
plt.legend()
plt.grid(True, alpha=0.3)

🔧#🔧 🔧8🔧.🔧 🔧H🔧i🔧s🔧t🔧o🔧g🔧r🔧a🔧m🔧 🔧b🔧y🔧 🔧d🔧e🔧f🔧e🔧c🔧t🔧 🔧t🔧y🔧p🔧e🔧
plt.subplot(3, 4, 8)
defect_counts = manufacturing_df['defect_type'].value_counts()
colors = {'none': 'green', 'short': 'red', 'long': 'orange'}
bars = plt.bar(defect_counts.index, defect_counts.values, 
               color=[colors[x] for x in defect_counts.index])
plt.xlabel('Defect Type')
plt.ylabel('Count')
plt.title('Distribution by Defect Type')
for bar, count in zip(bars, defect_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
             str(count), ha='center', va='bottom')
plt.grid(True, alpha=0.3)

🔧#🔧 🔧9🔧.🔧 🔧M🔧o🔧v🔧i🔧n🔧g🔧 🔧a🔧v🔧e🔧r🔧a🔧g🔧e🔧 🔧t🔧r🔧e🔧n🔧d🔧
plt.subplot(3, 4, 9)
manufacturing_df['moving_avg'] = manufacturing_df['length'].rolling(window=20, center=True).mean()
plt.plot(range(len(manufacturing_df)), manufacturing_df['length'], 'b-', alpha=0.3, label='Individual')
plt.plot(range(len(manufacturing_df)), manufacturing_df['moving_avg'], 'r-', linewidth=2, label='Moving Avg (20)')
plt.axhline(target_length, color='green', linestyle='--', label='Target')
plt.xlabel('Sample Number')
plt.ylabel('Length (mm)')
plt.title('Process Trend Analysis')
plt.legend()
plt.grid(True, alpha=0.3)

🔧#🔧 🔧1🔧0🔧.🔧 🔧P🔧r🔧o🔧b🔧a🔧b🔧i🔧l🔧i🔧t🔧y🔧 🔧c🔧a🔧l🔧c🔧u🔧l🔧a🔧t🔧i🔧o🔧n🔧s🔧
plt.subplot(3, 4, 10)
🔧#🔧 🔧C🔧r🔧e🔧a🔧t🔧e🔧 🔧p🔧r🔧o🔧b🔧a🔧b🔧i🔧l🔧i🔧t🔧y🔧 🔧d🔧e🔧n🔧s🔧i🔧t🔧y🔧 🔧a🔧r🔧e🔧a🔧s🔧
x_prob = np.linspace(96, 104, 1000)
y_prob = stats.norm.pdf(x_prob, overall_mean, overall_std)

plt.plot(x_prob, y_prob, 'b-', linewidth=2)

🔧#🔧 🔧S🔧h🔧a🔧d🔧e🔧 🔧d🔧i🔧f🔧f🔧e🔧r🔧e🔧n🔧t🔧 🔧p🔧r🔧o🔧b🔧a🔧b🔧i🔧l🔧i🔧t🔧y🔧 🔧r🔧e🔧g🔧i🔧o🔧n🔧s🔧
plt.fill_between(x_prob, 0, y_prob, where=(x_prob <= overall_mean - overall_std), 
                alpha=0.3, color='lightcoral', label='< ¼-Ã (16%)')
plt.fill_between(x_prob, 0, y_prob, where=((x_prob > overall_mean - overall_std) & 
                (x_cap <= overall_mean + overall_std)), 
                alpha=0.3, color='lightgreen', label='¼±Ã (68%)')
plt.fill_between(x_prob, 0, y_prob, where=(x_prob > overall_mean + overall_std), 
                alpha=0.3, color='lightcoral', label='> ¼+Ã (16%)')

plt.xlabel('Length (mm)')
plt.ylabel('Density')
plt.title('Probability Regions (Empirical Rule)')
plt.legend()
plt.grid(True, alpha=0.3)

🔧#🔧 🔧1🔧1🔧-🔧1🔧2🔧:🔧 🔧A🔧d🔧d🔧i🔧t🔧i🔧o🔧n🔧a🔧l🔧 🔧a🔧n🔧a🔧l🔧y🔧s🔧e🔧s🔧
plt.subplot(3, 4, 11)
🔧#🔧 🔧B🔧o🔧x🔧 🔧p🔧l🔧o🔧t🔧 🔧b🔧y🔧 🔧d🔧a🔧y🔧 🔧(🔧s🔧a🔧m🔧p🔧l🔧e🔧 🔧o🔧f🔧 🔧d🔧a🔧y🔧s🔧)🔧
sample_days = [1, 10, 20, 30]
box_data = [manufacturing_df[manufacturing_df['day'] == day]['length'].values 
           for day in sample_days]
plt.boxplot(box_data, labels=[f'Day {d}' for d in sample_days])
plt.ylabel('Length (mm)')
plt.title('Distribution by Selected Days')
plt.grid(True, alpha=0.3)

plt.subplot(3, 4, 12)
🔧#🔧 🔧C🔧u🔧m🔧u🔧l🔧a🔧t🔧i🔧v🔧e🔧 🔧d🔧e🔧f🔧e🔧c🔧t🔧 🔧r🔧a🔧t🔧e🔧
manufacturing_df['cumulative_defects'] = (~manufacturing_df['within_spec']).cumsum()
manufacturing_df['cumulative_rate'] = manufacturing_df['cumulative_defects'] / np.arange(1, len(manufacturing_df) + 1)
plt.plot(range(len(manufacturing_df)), manufacturing_df['cumulative_rate'] * 100, 'b-')
plt.axhline(theoretical_defect_rate * 100, color='red', linestyle='--', label='Expected Rate')
plt.xlabel('Sample Number')
plt.ylabel('Cumulative Defect Rate (%)')
plt.title('Cumulative Defect Rate Trend')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

🔧#🔧 🔧S🔧t🔧a🔧t🔧i🔧s🔧t🔧i🔧c🔧a🔧l🔧 🔧s🔧u🔧m🔧m🔧a🔧r🔧y🔧
print(f"\nStatistical Analysis Summary:")
print(f"Shapiro-Wilk normality test: p = {stats.shapiro(manufacturing_df['length'])[1]:.6f}")
if stats.shapiro(manufacturing_df['length'])[1] > 0.05:
    print("   Data appears to follow normal distribution (p > 0.05)")
else:
    print("   Data may not be perfectly normal (p d 0.05)")

🔧#🔧 🔧R🔧e🔧c🔧o🔧m🔧m🔧e🔧n🔧d🔧a🔧t🔧i🔧o🔧n🔧
print(f"\nRecommendations:")
if cpk >= 1.33:
    print(" Process is operating excellently")
elif cpk >= 1.0:
    print("  Process is adequate but could be improved")
    print("  - Consider reducing process variation")
    print("  - Check for process centering")
else:
    print("L Process needs immediate attention")
    print("  - High defect rate detected")
    print("  - Consider process adjustment or tighter control")

return manufacturing_df

🔧#🔧 🔧R🔧u🔧n🔧 🔧t🔧h🔧e🔧 🔧m🔧a🔧n🔧u🔧f🔧a🔧c🔧t🔧u🔧r🔧i🔧n🔧g🔧 🔧a🔧n🔧a🔧l🔧y🔧s🔧i🔧s🔧
manufacturing_data = simulate_manufacturing_data()
```

🔧#🔧#🔧#🔧 🔧F🔧i🔧n🔧a🔧n🔧c🔧i🔧a🔧l🔧 🔧R🔧i🔧s🔧k🔧 🔧A🔧n🔧a🔧l🔧y🔧s🔧i🔧s🔧 🔧E🔧x🔧a🔧m🔧p🔧l🔧e🔧

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from datetime import datetime, timedelta

🔧#🔧 🔧F🔧i🔧n🔧a🔧n🔧c🔧i🔧a🔧l🔧 🔧R🔧i🔧s🔧k🔧 🔧A🔧n🔧a🔧l🔧y🔧s🔧i🔧s🔧 🔧u🔧s🔧i🔧n🔧g🔧 🔧N🔧o🔧r🔧m🔧a🔧l🔧 🔧D🔧i🔧s🔧t🔧r🔧i🔧b🔧u🔧t🔧i🔧o🔧n🔧
def financial_risk_analysis():
    """
    Analyze financial portfolio using normal distribution assumptions
    """
    np.random.seed(42)
    
    # Simulate daily returns for different assets
    n_days = 252  # One trading year
    assets = ['Stock_A', 'Stock_B', 'Bond', 'Commodity']
    
    # Define asset characteristics (annual returns and volatility)
    asset_params = {
        'Stock_A': {'annual_return': 0.12, 'volatility': 0.20},
        'Stock_B': {'annual_return': 0.08, 'volatility': 0.15},
        'Bond': {'annual_return': 0.04, 'volatility': 0.05},
        'Commodity': {'annual_return': 0.06, 'volatility': 0.25}
    }
    
    # Convert to daily parameters
    daily_returns = {}
    for asset, params in asset_params.items():
        daily_mean = params['annual_return'] / 252
        daily_std = params['volatility'] / np.sqrt(252)
        daily_returns[asset] = np.random.normal(daily_mean, daily_std, n_days)
    
    # Create DataFrame
    returns_df = pd.DataFrame(daily_returns)
    
    # Portfolio weights
    weights = np.array([0.4, 0.3, 0.2, 0.1])  # 40% Stock_A, 30% Stock_B, 20% Bond, 10% Commodity
    
    # Calculate portfolio returns
    returns_df['Portfolio'] = np.dot(returns_df[assets], weights)
    
    print("Financial Portfolio Risk Analysis")
    print("=" * 50)
    print("Asset Allocation:")
    for asset, weight in zip(assets, weights):
        print(f"  {asset}: {weight:.1%}")
    
    print(f"\nPortfolio Statistics (Annualized):")
    portfolio_annual_return = returns_df['Portfolio'].mean() * 252
    portfolio_annual_volatility = returns_df['Portfolio'].std() * np.sqrt(252)
    sharpe_ratio = portfolio_annual_return / portfolio_annual_volatility
    
    print(f"Expected Return: {portfolio_annual_return:.2%}")
    print(f"Volatility (Risk): {portfolio_annual_volatility:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
    
    # Value at Risk (VaR) calculations using normal distribution
    confidence_levels = [0.95, 0.99, 0.999]
    
    print(f"\nValue at Risk (VaR) Analysis:")
    print("Confidence Level | 1-Day VaR | 10-Day VaR | Monthly VaR")
    print("-" * 55)
    
    portfolio_mean = returns_df['Portfolio'].mean()
    portfolio_std = returns_df['Portfolio'].std()
    
    for conf in confidence_levels:
        # Z-score for confidence level
        z_score = stats.norm.ppf(1 - conf)
        
        # VaR calculations (negative because it's a loss)
        var_1day = -(portfolio_mean + z_score * portfolio_std)
        var_10day = -(portfolio_mean * 10 + z_score * portfolio_std * np.sqrt(10))
        var_monthly = -(portfolio_mean * 21 + z_score * portfolio_std * np.sqrt(21))
        
        print(f"{conf:.1%}            | {var_1day:.2%}     | {var_10day:.2%}      | {var_monthly:.2%}")
    
    # Monte Carlo simulation for portfolio value
    initial_value = 1000000  # $1M initial portfolio
    
    # Simulate portfolio path
    cumulative_returns = (1 + returns_df['Portfolio']).cumprod()
    portfolio_values = initial_value * cumulative_returns
    
    # Calculate maximum drawdown
    running_max = portfolio_values.cummax()
    drawdown = (portfolio_values - running_max) / running_max
    max_drawdown = drawdown.min()
    
    print(f"\nPortfolio Performance:")
    print(f"Initial Value: ${initial_value:,.0f}")
    print(f"Final Value: ${portfolio_values.iloc[-1]:,.0f}")
    print(f"Total Return: {(portfolio_values.iloc[-1]/initial_value - 1):.2%}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    
    # Risk metrics
    downside_returns = returns_df['Portfolio'][returns_df['Portfolio'] < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252)
    
    print(f"Downside Deviation: {downside_deviation:.2%}")
    print(f"Probability of Loss (daily): {(returns_df['Portfolio'] < 0).mean():.1%}")
    
    # Normal distribution tests for each asset
    print(f"\nNormality Tests (Shapiro-Wilk p-values):")
    for asset in assets + ['Portfolio']:
        _, p_value = stats.shapiro(returns_df[asset])
        status = "Normal" if p_value > 0.05 else "Non-normal"
        print(f"  {asset}: p = {p_value:.4f} ({status})")
    
    # Visualization
    plt.figure(figsize=(20, 16))
    
    # 1. Portfolio value over time
    plt.subplot(4, 3, 1)
    plt.plot(portfolio_values, linewidth=2, color='blue')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Trading Day')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, alpha=0.3)
    
    # 2. Daily returns distribution
    plt.subplot(4, 3, 2)
    plt.hist(returns_df['Portfolio'], bins=50, density=True, alpha=0.7, color='lightblue', edgecolor='black')
    
    # Fit normal distribution
    mu, sigma = stats.norm.fit(returns_df['Portfolio'])
    x = np.linspace(returns_df['Portfolio'].min(), returns_df['Portfolio'].max(), 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Fitted Normal')
    
    # Mark VaR levels
    var_95 = stats.norm.ppf(0.05, mu, sigma)
    var_99 = stats.norm.ppf(0.01, mu, sigma)
    plt.axvline(var_95, color='orange', linestyle='--', label='95% VaR')
    plt.axvline(var_99, color='red', linestyle='--', label='99% VaR')
    
    plt.title('Portfolio Daily Returns Distribution')
    plt.xlabel('Daily Return')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Q-Q plot for portfolio returns
    plt.subplot(4, 3, 3)
    stats.probplot(returns_df['Portfolio'], dist="norm", plot=plt)
    plt.title('Portfolio Returns Q-Q Plot')
    plt.grid(True, alpha=0.3)
    
    # 4. Individual asset returns
    plt.subplot(4, 3, 4)
    for i, asset in enumerate(assets):
        plt.plot(np.cumsum(returns_df[asset]), label=asset, linewidth=2)
    plt.title('Cumulative Returns by Asset')
    plt.xlabel('Trading Day')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Correlation matrix
    plt.subplot(4, 3, 5)
    corr_matrix = returns_df[assets].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": .5})
    plt.title('Asset Correlation Matrix')
    
    # 6. Rolling volatility
    plt.subplot(4, 3, 6)
    rolling_vol = returns_df['Portfolio'].rolling(window=21).std() * np.sqrt(252)
    plt.plot(rolling_vol, linewidth=2, color='purple')
    plt.axhline(portfolio_annual_volatility, color='red', linestyle='--', 
               label=f'Average: {portfolio_annual_volatility:.1%}')
    plt.title('Rolling 21-Day Volatility (Annualized)')
    plt.xlabel('Trading Day')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. Drawdown chart
    plt.subplot(4, 3, 7)
    plt.fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.3)
    plt.plot(drawdown, color='red', linewidth=2)
    plt.title('Portfolio Drawdown')
    plt.xlabel('Trading Day')
    plt.ylabel('Drawdown')
    plt.grid(True, alpha=0.3)
    
    # 8. VaR backtest
    plt.subplot(4, 3, 8)
    var_95_threshold = -(portfolio_mean + stats.norm.ppf(0.05) * portfolio_std)
    breaches = returns_df['Portfolio'] < -var_95_threshold
    
    plt.plot(returns_df['Portfolio'], alpha=0.7, color='blue', linewidth=1)
    plt.axhline(-var_95_threshold, color='red', linestyle='--', label='95% VaR Threshold')
    plt.scatter(range(len(returns_df)), returns_df['Portfolio'], 
               c=breaches, cmap='RdYlGn', s=10, alpha=0.7)
    
    breach_rate = breaches.mean()
    plt.title(f'VaR Backtesting (Breach Rate: {breach_rate:.1%})')
    plt.xlabel('Trading Day')
    plt.ylabel('Daily Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 9. Risk-return scatter
    plt.subplot(4, 3, 9)
    annual_returns = returns_df[assets].mean() * 252
    annual_volatilities = returns_df[assets].std() * np.sqrt(252)
    
    plt.scatter(annual_volatilities, annual_returns, s=100, alpha=0.7)
    for i, asset in enumerate(assets):
        plt.annotate(asset, (annual_volatilities[i], annual_returns[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    # Add portfolio point
    plt.scatter(portfolio_annual_volatility, portfolio_annual_return, 
               s=200, color='red', marker='*', label='Portfolio')
    
    plt.title('Risk-Return Profile')
    plt.xlabel('Volatility (Risk)')
    plt.ylabel('Expected Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 10. Monte Carlo simulation of future paths
    plt.subplot(4, 3, 10)
    n_simulations = 100
    days_forward = 252  # 1 year forward
    
    future_paths = []
    current_value = portfolio_values.iloc[-1]
    
    for _ in range(n_simulations):
        future_returns = np.random.normal(portfolio_mean, portfolio_std, days_forward)
        future_values = current_value * np.cumprod(1 + future_returns)
        future_paths.append(future_values)
        plt.plot(range(days_forward), future_values, alpha=0.1, color='blue')
    
    # Plot percentiles
    future_paths = np.array(future_paths)
    percentiles = [5, 25, 50, 75, 95]
    colors = ['red', 'orange', 'green', 'orange', 'red']
    
    for p, color in zip(percentiles, colors):
        plt.plot(range(days_forward), np.percentile(future_paths, p, axis=0), 
                color=color, linewidth=2, label=f'{p}th percentile')
    
    plt.title('Monte Carlo Future Price Simulation')
    plt.xlabel('Days Forward')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 11. Stress testing
    plt.subplot(4, 3, 11)
    stress_scenarios = {
        'Normal': (0, 1),
        'Mild Stress': (-0.02, 1.5),
        'Severe Stress': (-0.05, 2.0),
        'Extreme Stress': (-0.10, 3.0)
    }
    
    scenario_results = []
    scenario_names = []
    
    for name, (shock_mean, vol_multiplier) in stress_scenarios.items():
        stressed_returns = np.random.normal(
            portfolio_mean + shock_mean, 
            portfolio_std * vol_multiplier, 
            1000
        )
        var_99_stressed = -np.percentile(stressed_returns, 1)
        scenario_results.append(var_99_stressed)
        scenario_names.append(name)
    
    bars = plt.bar(scenario_names, scenario_results, color=['green', 'yellow', 'orange', 'red'])
    plt.title('Stress Testing - 99% VaR')
    plt.ylabel('VaR (1%)')
    plt.xticks(rotation=45)
    
    # Add value labels
    for bar, val in zip(bars, scenario_results):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{val:.2%}', ha='center', va='bottom')
    plt.grid(True, alpha=0.3)
    
    # 12. Expected shortfall (Conditional VaR)
    plt.subplot(4, 3, 12)
    confidence_levels_es = np.linspace(0.90, 0.999, 50)
    var_values = []
    es_values = []
    
    for conf in confidence_levels_es:
        var_threshold = -stats.norm.ppf(1-conf, portfolio_mean, portfolio_std)
        var_values.append(var_threshold)
        
        # Expected Shortfall (average of losses beyond VaR)
        tail_returns = returns_df['Portfolio'][returns_df['Portfolio'] <= -var_threshold]
        es = -tail_returns.mean() if len(tail_returns) > 0 else var_threshold
        es_values.append(es)
    
    plt.plot(confidence_levels_es, var_values, label='VaR', linewidth=2)
    plt.plot(confidence_levels_es, es_values, label='Expected Shortfall', linewidth=2)
    plt.title('VaR vs Expected Shortfall')
    plt.xlabel('Confidence Level')
    plt.ylabel('Risk Measure')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return returns_df, portfolio_values

🔧#🔧 🔧R🔧u🔧n🔧 🔧t🔧h🔧e🔧 🔧f🔧i🔧n🔧a🔧n🔧c🔧i🔧a🔧l🔧 🔧a🔧n🔧a🔧l🔧y🔧s🔧i🔧s🔧
returns_data, portfolio_data = financial_risk_analysis()
```

🔧#🔧#🔧 🔧=🔧Ú🔧 🔧R🔧e🔧f🔧e🔧r🔧e🔧n🔧c🔧e🔧s🔧

**Foundational Texts:**
- [Statistical Inference](https://web.stanford.edu/~hastie/CASI/) - Casella & Berger
- [Introduction to Mathematical Statistics](https://www.amazon.com/Introduction-Mathematical-Statistics-Robert-Hogg/dp/0321795431) - Hogg, McKean, Craig
- [Probability and Statistics](https://www.amazon.com/Probability-Statistics-Engineers-Scientists-3rd/dp/0132047675) - Walpole, Myers, Myers, Ye

**Machine Learning Applications:**
- [The Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/) - Hastie, Tibshirani, Friedman
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) - Christopher Bishop
- [Machine Learning: A Probabilistic Perspective](https://probml.github.io/pml-book/) - Kevin Murphy

**Classical Papers:**
- [Central Limit Theorem](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-40/issue-6/An-Invariance-Principle-for-the-Law-of-the-Iterated-Logarithm/10.1214/aoms/1177697380.full) - Strassen (1964)
- [Maximum Likelihood Estimation](https://www.jstor.org/stable/2334654) - Fisher (1922)
- [Box-Cox Transformations](https://www.jstor.org/stable/2984418) - Box & Cox (1964)

**Statistical Computing:**
- [SciPy Statistics Documentation](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Statsmodels Documentation](https://www.statsmodels.org/stable/index.html)
- [NumPy Random Documentation](https://numpy.org/doc/stable/reference/random/index.html)

**Quality Control Applications:**
- [Introduction to Statistical Quality Control](https://www.amazon.com/Introduction-Statistical-Quality-Control-Montgomery/dp/1118146816) - Douglas Montgomery
- [Statistical Process Control](https://www.amazon.com/Statistical-Process-Control-John-Oakland/dp/0750669241) - John Oakland

**Financial Applications:**
- [Risk Management and Financial Institutions](https://www.amazon.com/Risk-Management-Financial-Institutions-Wiley/dp/1118955943) - John Hull
- [Options, Futures, and Other Derivatives](https://www.amazon.com/Options-Futures-Other-Derivatives-10th/dp/013447208X) - John Hull
- [Value at Risk: The New Benchmark for Managing Financial Risk](https://www.amazon.com/Value-Risk-Benchmark-Managing-Financial/dp/0071464956) - Philippe Jorion

**Online Resources:**
- [Khan Academy Statistics](https://www.khanacademy.org/math/statistics-probability)
- [StatQuest YouTube Channel](https://www.youtube.com/c/joshstarmer)
- [MIT OpenCourseWare - Probability and Statistics](https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/)
- [Stanford CS229 Machine Learning Notes](http://cs229.stanford.edu/notes/)

**Specialized Topics:**
- [Multivariate Normal Distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)
- [Gaussian Processes](http://www.gaussianprocess.org/gpml/)
- [Normalizing Flows](https://arxiv.org/abs/1505.05770)
- [Central Limit Theorem Variations](https://projecteuclid.org/journals/annals-of-probability)