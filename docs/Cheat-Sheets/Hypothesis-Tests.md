---
title: Hypothesis Tests in Python (Cheat Sheet)
description: A cheat sheet for the most popular statistical hypothesis tests for a machine learning project with examples using the Python API.
---

# Hypothesis Tests in Python

[TOC]

A [**statistical hypothesis test**](https://en.wikipedia.org/wiki/Statistical_hypothesis_testing) is a method of [statistical inference](https://en.wikipedia.org/wiki/Statistical_inference "Statistical inference") used to decide whether the data at hand sufficiently support a particular hypothesis. Hypothesis testing allows us to make probabilistic statements about population parameters.

??? note "**Hypothesis Tests Mindmap** - Visual Overview"

    ![Hypothesis Tests Mindmap](../assets/img/Hypothesis-Tests-mindmap-1.png)

**Few Notes:**

- When it comes to assumptions such as the expected distribution of data or sample size, the results of a given test are likely to degrade gracefully rather than become immediately unusable if an assumption is violated.
- Generally, data samples need to be representative of the domain and large enough to expose their distribution to analysis.
- In some cases, the data can be corrected to meet the assumptions, such as correcting a nearly normal distribution to be normal by removing outliers, or using a correction to the degrees of freedom in a statistical test when samples have differing variance, to name two examples.


## Normality Tests

This section lists statistical tests that you can use to check if your data has a Gaussian distribution.

[Gaussian distribution (also known as normal distribution)](https://en.wikipedia.org/wiki/Normal_distribution) is a bell-shaped curve.

### Shapiro-Wilk Test

Tests whether a data sample has a Gaussian distribution/Normal distribution.

- **Assumptions**

    - Observations in each sample are independent and identically distributed (iid).

- **Interpretation**

    - H0: the sample has a Gaussian distribution.
    - H1: the sample does not have a Gaussian distribution.

- **Python Code**

    ```python
    # Example of the Shapiro-Wilk Normality Test
    from scipy.stats import shapiro
    data = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
    stat, p = shapiro(data)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably Gaussian')
    else:
        print('Probably not Gaussian')
    ```

- **Sources**

    -   [scipy.stats.shapiro](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html)
    -   [Shapiro-Wilk test on Wikipedia](https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test)

### D’Agostino’s K^2 Test

Tests whether a data sample has a Gaussian distribution/Normal distribution.

- **Assumptions**

    - Observations in each sample are independent and identically distributed (iid).

- **Interpretation**

    - H0: the sample has a Gaussian distribution.
    - H1: the sample does not have a Gaussian distribution.

- **Python Code**

    ```python
    # Example of the D'Agostino's K^2 Normality Test
    from scipy.stats import normaltest
    data = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
    stat, p = normaltest(data)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably Gaussian')
    else:
        print('Probably not Gaussian')
    ```

- **Sources**

    -   [scipy.stats.normaltest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html)
    -   [D'Agostino's K-squared test on Wikipedia](https://en.wikipedia.org/wiki/D%27Agostino%27s_K-squared_test)

### Anderson-Darling Test

Tests whether a data sample has a Gaussian distribution/Normal distribution.

- **Assumptions**

    - Observations in each sample are independent and identically distributed (iid).

- **Interpretation**

    - H0: the sample has a Gaussian distribution.
    - H1: the sample does not have a Gaussian distribution.

- **Python Code**

    ```python
    # Example of the Anderson-Darling Normality Test
    from scipy.stats import anderson
    data = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
    result = anderson(data)
    print('stat=%.3f' % (result.statistic))
    for i in range(len(result.critical_values)):
        sl, cv = result.significance_level[i], result.critical_values[i]
        if result.statistic < cv:
            print('Probably Gaussian at the %.1f%% level' % (sl))
        else:
            print('Probably not Gaussian at the %.1f%% level' % (sl))
    ```

- **Sources**

    -   [scipy.stats.anderson](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson.html)
    -   [Anderson-Darling test on Wikipedia](https://en.wikipedia.org/wiki/Anderson%E2%80%93Darling_test)

## Correlation Tests

This section lists statistical tests that you can use to check if two samples are related.

### Pearson’s Correlation Coefficient

Tests whether two samples have a linear relationship.

- **Assumptions**

    -   Observations in each sample are independent and identically distributed (iid).
    -   Observations in each sample are normally distributed.
    -   Observations in each sample have the same variance.

- **Interpretation**

    -   H0: the two samples are independent.
    -   H1: there is a dependency between the samples.

- **Python Code**

    ```python
    # Example of the Pearson's Correlation test
    from scipy.stats import pearsonr
    data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
    data2 = [0.353, 3.517, 0.125, -7.545, -0.555, -1.536, 3.350, -1.578, -3.537, -1.579]
    stat, p = pearsonr(data1, data2)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably independent')
    else:
        print('Probably dependent')
    ```

- **Sources**

    -   [scipy.stats.pearsonr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html)
    -   [Pearson's correlation coefficient on Wikipedia](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)

### Spearman’s Rank Correlation

Tests whether two samples have a monotonic relationship.

- **Assumptions**

    -   Observations in each sample are independent and identically distributed (iid).
    -   Observations in each sample can be ranked.

- **Interpretation**

    -   H0: the two samples are independent.
    -   H1: there is a dependency between the samples.

- **Python Code**

    ```python
    # Example of the Spearman's Rank Correlation Test
    from scipy.stats import spearmanr
    data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
    data2 = [0.353, 3.517, 0.125, -7.545, -0.555, -1.536, 3.350, -1.578, -3.537, -1.579]
    stat, p = spearmanr(data1, data2)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably independent')
    else:
        print('Probably dependent')
    ```

- **Sources**

    -   [scipy.stats.spearmanr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html)
    -   [Spearman's rank correlation coefficient on Wikipedia](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient)

### Kendall’s Rank Correlation

Tests whether two samples have a monotonic relationship.

- **Assumptions**

    -   Observations in each sample are independent and identically distributed (iid).
    -   Observations in each sample can be ranked.

- **Interpretation**

    -   H0: the two samples are independent.
    -   H1: there is a dependency between the samples.

- **Python Code**

    ```python
    # Example of the Kendall's Rank Correlation Test
    from scipy.stats import kendalltau
    data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
    data2 = [0.353, 3.517, 0.125, -7.545, -0.555, -1.536, 3.350, -1.578, -3.537, -1.579]
    stat, p = kendalltau(data1, data2)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably independent')
    else:
        print('Probably dependent')
    ```

- **Sources**

    -   [scipy.stats.kendalltau](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html)
    -   [Kendall rank correlation coefficient on Wikipedia](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient)

### Chi-Squared Test

Tests whether two categorical variables are related or independent.

- **Assumptions**

    -   Observations used in the calculation of the contingency table are independent.
    -   25 or more examples in each cell of the contingency table.

- **Interpretation**

    -   H0: the two samples are independent.
    -   H1: there is a dependency between the samples.

- **Python Code**

    ```python
    # Example of the Chi-Squared Test
    from scipy.stats import chi2_contingency
    table = [[10, 20, 30],[6,  9,  17]]
    stat, p, dof, expected = chi2_contingency(table)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably independent')
    else:
        print('Probably dependent')
    ```

- **Sources**

    -   [scipy.stats.chi2_contingency](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html)
    -   [Chi-Squared test on Wikipedia](https://en.wikipedia.org/wiki/Chi-squared_test)

## Stationary Tests

This section lists statistical tests that you can use to check if a time series is stationary or not.

### Augmented Dickey-Fuller Unit Root Test

Tests whether a time series has a unit root, e.g. has a trend or more generally is autoregressive.

- **Assumptions**

    -   Observations in are temporally ordered.

- **Interpretation**

    -   H0: a unit root is present (series is non-stationary).
    -   H1: a unit root is not present (series is stationary).

- **Python Code**

    ```python
    # Example of the Augmented Dickey-Fuller unit root test
    from statsmodels.tsa.stattools import adfuller
    data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    stat, p, lags, obs, crit, t = adfuller(data)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably not Stationary')
    else:
        print('Probably Stationary')
    ```

- **Sources**

    -   [statsmodels.tsa.stattools.adfuller API](https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html).
    -   [Augmented Dickey--Fuller test, Wikipedia](https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test).

### Kwiatkowski-Phillips-Schmidt-Shin

Tests whether a time series is trend stationary or not.

- **Assumptions**

    -   Observations in are temporally ordered.

- **Interpretation**

    -   H0: the time series is trend-stationary.
    -   H1: the time series is not trend-stationary.

- **Python Code**

    ```python
    # Example of the Kwiatkowski-Phillips-Schmidt-Shin test
    from statsmodels.tsa.stattools import kpss
    data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    stat, p, lags, crit = kpss(data)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably Stationary')
    else:
        print('Probably not Stationary')
    ```

- **Sources**

    -   [statsmodels.tsa.stattools.kpss API](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html#statsmodels.tsa.stattools.kpss).
    -   [KPSS test, Wikipedia](https://en.wikipedia.org/wiki/KPSS_test).

## Parametric Statistical Hypothesis Tests

This section lists statistical tests that you can use to compare data samples.

### Student’s t-test

Tests whether the means of two independent samples are significantly different.

- **Assumptions**

    -   Observations in each sample are independent and identically distributed (iid).
    -   Observations in each sample are normally distributed.
    -   Observations in each sample have the same variance.

- **Interpretation**

    -   H0: the means of the samples are equal.
    -   H1: the means of the samples are unequal.

- **Python Code**

    ```python
    # Example of the Student's t-test
    from scipy.stats import ttest_ind
    data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
    data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
    stat, p = ttest_ind(data1, data2)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')
    ```

- **Sources**

    -   [scipy.stats.ttest_ind](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)
    -   [Student's t-test on Wikipedia](https://en.wikipedia.org/wiki/Student%27s_t-test)

### Paired Student’s t-test

Tests whether the means of two independent samples are significantly different.

- **Assumptions**

    -   Observations in each sample are independent and identically distributed (iid).
    -   Observations in each sample are normally distributed.
    -   Observations in each sample have the same variance.
    -   Observations across each sample are paired.

- **Interpretation**

    -   H0: the means of the samples are equal.
    -   H1: the means of the samples are unequal.

- **Python Code**

    ```python
    # Example of the Paired Student's t-test
    from scipy.stats import ttest_rel
    data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
    data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
    stat, p = ttest_rel(data1, data2)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')
    ```

- **Sources**

    -   [scipy.stats.ttest_rel](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html)
    -   [Student's t-test on Wikipedia](https://en.wikipedia.org/wiki/Student%27s_t-test)

### Analysis of Variance Test (ANOVA)

Tests whether the means of two or more independent samples are significantly different.

- **Assumptions**

    -   Observations in each sample are independent and identically distributed (iid).
    -   Observations in each sample are normally distributed.
    -   Observations in each sample have the same variance.

- **Interpretation**

    -   H0: the means of the samples are equal.
    -   H1: the means of the samples are unequal.

- **Python Code**

    ```python
    # Example of the Analysis of Variance Test
    from scipy.stats import f_oneway
    data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
    data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
    data3 = [-0.208, 0.696, 0.928, -1.148, -0.213, 0.229, 0.137, 0.269, -0.870, -1.204]
    stat, p = f_oneway(data1, data2, data3)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')
    ```

- **Sources**

    -   [scipy.stats.f_oneway](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html)
    -   [Analysis of variance on Wikipedia](https://en.wikipedia.org/wiki/Analysis_of_variance)

### Repeated Measures ANOVA Test

Tests whether the means of two or more paired samples are significantly different.

- **Assumptions**

    -   Observations in each sample are independent and identically distributed (iid).
    -   Observations in each sample are normally distributed.
    -   Observations in each sample have the same variance.
    -   Observations across each sample are paired.

- **Interpretation**

    -   H0: the means of the samples are equal.
    -   H1: one or more of the means of the samples are unequal.

- **Python Code**

    ```python
    # Currently not supported in Python. :(
    ```

- **Sources**

    -   [Analysis of variance on Wikipedia](https://en.wikipedia.org/wiki/Analysis_of_variance)

## Nonparametric Statistical Hypothesis Tests

In Non-Parametric tests, we don't make any assumption about the parameters for the given population or the population we are studying. In fact, these tests don't depend on the population. Hence, there is no fixed set of parameters is available, and also there is no distribution (normal distribution, etc.)

### Mann-Whitney U Test

Tests whether the distributions of two independent samples are equal or not.

- **Assumptions**

    -   Observations in each sample are independent and identically distributed (iid).
    -   Observations in each sample can be ranked.

- **Interpretation**

    -   H0: the distributions of both samples are equal.
    -   H1: the distributions of both samples are not equal.

- **Python Code**

    ```python
    # Example of the Mann-Whitney U Test
    from scipy.stats import mannwhitneyu
    data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
    data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
    stat, p = mannwhitneyu(data1, data2)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')
    ```

- **Sources**

    -   [scipy.stats.mannwhitneyu](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html)
    -   [Mann-Whitney U test on Wikipedia](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test)

### Wilcoxon Signed-Rank Test

Tests whether the distributions of two paired samples are equal or not.

- **Assumptions**

    -   Observations in each sample are independent and identically distributed (iid).
    -   Observations in each sample can be ranked.
    -   Observations across each sample are paired.

- **Interpretation**

    -   H0: the distributions of both samples are equal.
    -   H1: the distributions of both samples are not equal.

- **Python Code**

    ```python
    # Example of the Wilcoxon Signed-Rank Test
    from scipy.stats import wilcoxon
    data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
    data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
    stat, p = wilcoxon(data1, data2)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')
    ```

- **Sources**

    -   [scipy.stats.wilcoxon](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html)
    -   [Wilcoxon signed-rank test on Wikipedia](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test)

### Kruskal-Wallis H Test

Tests whether the distributions of two or more independent samples are equal or not.

- **Assumptions**

    -   Observations in each sample are independent and identically distributed (iid).
    -   Observations in each sample can be ranked.

- **Interpretation**

    -   H0: the distributions of all samples are equal.
    -   H1: the distributions of one or more samples are not equal.

- **Python Code**

    ```python
    # Example of the Kruskal-Wallis H Test
    from scipy.stats import kruskal
    data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
    data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
    stat, p = kruskal(data1, data2)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')
    ```

- **Sources**

    -   [scipy.stats.kruskal](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html)
    -   [Kruskal-Wallis one-way analysis of variance on Wikipedia](https://en.wikipedia.org/wiki/Kruskal%E2%80%93Wallis_one-way_analysis_of_variance)

### Friedman Test

Tests whether the distributions of two or more paired samples are equal or not.

- **Assumptions**

    -   Observations in each sample are independent and identically distributed (iid).
    -   Observations in each sample can be ranked.
    -   Observations across each sample are paired.

- **Interpretation**

    -   H0: the distributions of all samples are equal.
    -   H1: the distributions of one or more samples are not equal.

- **Python Code**

    ```python
    # Example of the Friedman Test
    from scipy.stats import friedmanchisquare
    data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
    data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
    data3 = [-0.208, 0.696, 0.928, -1.148, -0.213, 0.229, 0.137, 0.269, -0.870, -1.204]
    stat, p = friedmanchisquare(data1, data2, data3)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')
    ```

- **Sources**

    -   [scipy.stats.friedmanchisquare](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html)
    -   [Friedman test on Wikipedia](https://en.wikipedia.org/wiki/Friedman_test)

## Equality of variance test

Test is used to assess the equality of variance between two different samples.

### Levene's test

Levene’s test is used to assess the equality of variance between two or more different samples.

- **Assumptions**

    -   The samples from the populations under consideration are independent.
    -   The populations under consideration are approximately normally distributed.

- **Interpretation**

    -   H0: All the samples variances are equal
    -   H1: At least one variance is different from the rest

- **Python Code**

    ```python
    # Example of the Levene's test
    from scipy.stats import levene
    a = [8.88, 9.12, 9.04, 8.98, 9.00, 9.08, 9.01, 8.85, 9.06, 8.99]
    b = [8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05]
    c = [8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98]
    stat, p = levene(a, b, c)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably the same variances')
    else:
        print('Probably at least one variance is different from the rest')
    ```

- **Sources**

    -   [scipy.stats.levene](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levene.html)
    -   [Levene's test on Wikipedia](https://en.wikipedia.org/wiki/Levene's_test)

---

Source: https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/
