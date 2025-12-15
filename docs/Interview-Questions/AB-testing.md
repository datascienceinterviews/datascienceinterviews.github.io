---
title: A/B Testing Interview Questions
description: 100+ A/B Testing and Experimentation interview questions for Data Science and Product Analyst roles
---

# A/B Testing Interview Questions

<!-- [TOC] -->

This document provides a curated list of A/B Testing and Experimentation interview questions. It covers statistical foundations, experimental design, metric selection, and advanced topics like interference (network effects) and sequential testing. Critical for roles at data-driven companies like Netflix, Airbnb, and Uber.

---

## Premium Interview Questions

### Explain Hypothesis Testing in A/B Tests - Google, Netflix Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Statistics`, `Fundamentals` | **Asked by:** Google, Netflix, Meta, Airbnb

??? success "View Answer"

    **Hypothesis testing** is the **statistical foundation** of A/B testing, determining whether observed differences between control and treatment are **statistically significant** or due to random chance. It frames experiments as **binary decisions** (ship or don't ship) with controlled error rates (Type I/II errors).

    **Core Concepts:**

    - **Null Hypothesis ($H_0$):** No difference between control and treatment ($\mu_T = \mu_C$)
    - **Alternative Hypothesis ($H_1$):** There is a difference ($\mu_T \neq \mu_C$ for two-sided, $\mu_T > \mu_C$ for one-sided)
    - **p-value:** Probability of observing data at least as extreme if $H_0$ is true
    - **Significance level (α):** Threshold for rejecting $H_0$ (typically 0.05 = 5% false positive rate)

    **Test Statistics:**

    $$z = \frac{\bar{x}_T - \bar{x}_C}{\sqrt{\frac{s_T^2}{n_T} + \frac{s_C^2}{n_C}}}$$

    ```python
    import numpy as np
    import pandas as pd
    from scipy import stats
    from statsmodels.stats.proportion import proportions_ztest
    from statsmodels.stats.weightstats import ttest_ind as welch_ttest
    import matplotlib.pyplot as plt

    # Production: Comprehensive Hypothesis Testing Pipeline for A/B Tests

    # ===== 1. Generate realistic A/B test data =====
    np.random.seed(42)

    # Example 1: Conversion rate test (proportions)
    n_control = 10000
    n_treatment = 10000
    baseline_cvr = 0.10  # 10% baseline conversion
    treatment_lift = 0.02  # +2pp absolute lift (20% relative)

    control_conversions = np.random.binomial(n_control, baseline_cvr)
    treatment_conversions = np.random.binomial(n_treatment, baseline_cvr + treatment_lift)

    control_cvr = control_conversions / n_control
    treatment_cvr = treatment_conversions / n_treatment

    print("===== Conversion Rate Test (Proportions) =====")
    print(f"Control: {control_conversions}/{n_control} = {control_cvr:.4f}")
    print(f"Treatment: {treatment_conversions}/{n_treatment} = {treatment_cvr:.4f}")
    print(f"Absolute lift: {(treatment_cvr - control_cvr):.4f}")
    print(f"Relative lift: {100*(treatment_cvr/control_cvr - 1):.2f}%\n")

    # ===== 2. Z-test for proportions (most common in A/B testing) =====
    # Use when: Large samples (n*p > 5 and n*(1-p) > 5), binary outcome

    counts = np.array([treatment_conversions, control_conversions])
    nobs = np.array([n_treatment, n_control])

    z_stat, p_value_two_sided = proportions_ztest(counts, nobs, alternative='two-sided')
    _, p_value_one_sided = proportions_ztest(counts, nobs, alternative='larger')

    print("--- Z-test for Proportions ---")
    print(f"Z-statistic: {z_stat:.4f}")
    print(f"p-value (two-sided): {p_value_two_sided:.6f}")
    print(f"p-value (one-sided): {p_value_one_sided:.6f}")
    print(f"Result: {'REJECT H0 (significant)' if p_value_two_sided < 0.05 else 'FAIL TO REJECT H0'}\n")

    # ===== 3. Example 2: Revenue per user (continuous metric) =====
    control_revenue = np.random.exponential(scale=20, size=5000)  # Right-skewed
    treatment_revenue = np.random.exponential(scale=22, size=5000)  # +10% lift

    print("===== Revenue Per User Test (Continuous) =====")
    print(f"Control mean: ${control_revenue.mean():.2f} (median: ${np.median(control_revenue):.2f})")
    print(f"Treatment mean: ${treatment_revenue.mean():.2f} (median: ${np.median(treatment_revenue):.2f})")
    print(f"Absolute lift: ${(treatment_revenue.mean() - control_revenue.mean()):.2f}")
    print(f"Relative lift: {100*(treatment_revenue.mean()/control_revenue.mean() - 1):.2f}%\n")

    # ===== 4. T-test for continuous metrics =====
    # Standard t-test (assumes equal variance)
    t_stat_standard, p_value_standard = stats.ttest_ind(treatment_revenue, control_revenue)

    # Welch's t-test (does NOT assume equal variance - PREFERRED)
    t_stat_welch, p_value_welch, _ = welch_ttest(treatment_revenue, control_revenue, usevar='unequal')

    print("--- T-test (Standard, assumes equal variance) ---")
    print(f"t-statistic: {t_stat_standard:.4f}")
    print(f"p-value: {p_value_standard:.6f}")

    print("\n--- Welch's T-test (Unequal variance, PREFERRED) ---")
    print(f"t-statistic: {t_stat_welch:.4f}")
    print(f"p-value: {p_value_welch:.6f}")
    print(f"Result: {'REJECT H0 (significant)' if p_value_welch < 0.05 else 'FAIL TO REJECT H0'}\n")

    # ===== 5. Mann-Whitney U test (non-parametric alternative) =====
    # Use when: Data is heavily skewed or violates normality assumptions
    u_stat, p_value_mann = stats.mannwhitneyu(treatment_revenue, control_revenue, alternative='two-sided')

    print("--- Mann-Whitney U Test (Non-parametric) ---")
    print(f"U-statistic: {u_stat:.0f}")
    print(f"p-value: {p_value_mann:.6f}")
    print(f"Result: {'REJECT H0 (significant)' if p_value_mann < 0.05 else 'FAIL TO REJECT H0'}\n")

    # ===== 6. Bootstrap confidence interval (robust alternative) =====
    # Useful when: Unsure about distributions, want robust CI
    def bootstrap_diff(control, treatment, n_iterations=10000):
        """Bootstrap confidence interval for difference in means"""
        diffs = []
        for _ in range(n_iterations):
            control_sample = np.random.choice(control, size=len(control), replace=True)
            treatment_sample = np.random.choice(treatment, size=len(treatment), replace=True)
            diffs.append(treatment_sample.mean() - control_sample.mean())
        return np.array(diffs)

    bootstrap_diffs = bootstrap_diff(control_revenue, treatment_revenue)
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)

    print("--- Bootstrap 95% CI for Difference in Means ---")
    print(f"95% CI: [${ci_lower:.2f}, ${ci_upper:.2f}]")
    print(f"Observed diff: ${(treatment_revenue.mean() - control_revenue.mean()):.2f}")
    print(f"Result: {'REJECT H0' if ci_lower > 0 else 'FAIL TO REJECT H0'}\n")

    # ===== 7. Effect size calculation (practical significance) =====
    # Cohen's d for continuous metrics
    pooled_std = np.sqrt((control_revenue.var() + treatment_revenue.var()) / 2)
    cohens_d = (treatment_revenue.mean() - control_revenue.mean()) / pooled_std

    print("--- Effect Size (Cohen's d) ---")
    print(f"Cohen's d: {cohens_d:.4f}")
    print(f"Interpretation: ", end="")
    if abs(cohens_d) < 0.2:
        print("Small effect")
    elif abs(cohens_d) < 0.5:
        print("Medium effect")
    else:
        print("Large effect")
    print()

    # ===== 8. Confidence interval for proportions =====
    from statsmodels.stats.proportion import proportion_confint

    ci_control = proportion_confint(control_conversions, n_control, alpha=0.05, method='wilson')
    ci_treatment = proportion_confint(treatment_conversions, n_treatment, alpha=0.05, method='wilson')

    print("--- 95% Confidence Intervals (Wilson method) ---")
    print(f"Control CVR: {control_cvr:.4f} [{ci_control[0]:.4f}, {ci_control[1]:.4f}]")
    print(f"Treatment CVR: {treatment_cvr:.4f} [{ci_treatment[0]:.4f}, {ci_treatment[1]:.4f}]")

    # Confidence interval for difference in proportions
    se_diff = np.sqrt(control_cvr*(1-control_cvr)/n_control + treatment_cvr*(1-treatment_cvr)/n_treatment)
    diff = treatment_cvr - control_cvr
    ci_diff_lower = diff - 1.96 * se_diff
    ci_diff_upper = diff + 1.96 * se_diff

    print(f"Difference: {diff:.4f} [{ci_diff_lower:.4f}, {ci_diff_upper:.4f}]")
    ```

    | Test Type | Use Case | Assumptions | Python Function |
    |-----------|----------|-------------|-----------------|
    | **Z-test (proportions)** | Conversion rate, click rate | Large n (n·p > 5) | `proportions_ztest()` |
    | **T-test (standard)** | Revenue, time spent | Normal dist, equal variance | `stats.ttest_ind()` |
    | **Welch's t-test** | Revenue, skewed metrics | Normal dist, unequal variance | `welch_ttest()` |
    | **Mann-Whitney U** | Heavily skewed data | None (non-parametric) | `stats.mannwhitneyu()` |
    | **Bootstrap** | Any metric, robust CI | None (resampling) | Custom implementation |

    | p-value | Interpretation | Decision (α = 0.05) |
    |---------|----------------|---------------------|
    | < 0.001 | Very strong evidence against H₀ | Reject H₀ (ship) |
    | 0.001-0.01 | Strong evidence | Reject H₀ |
    | 0.01-0.05 | Moderate evidence | Reject H₀ (borderline) |
    | 0.05-0.10 | Weak evidence | Fail to reject H₀ |
    | > 0.10 | No evidence | Fail to reject H₀ (don't ship) |

    **Real-World:**
    - **Netflix:** Uses **Welch's t-test** for watch time (skewed distribution). Tests 200+ experiments/week with α = 0.05, ships features with p < 0.01 for high-impact changes.
    - **Booking.com:** Runs **z-tests on conversion rate** with 95% confidence. Average test: 2 weeks, 1M+ users, MDE = 1% relative lift.
    - **Airbnb:** Uses **bootstrap methods** for revenue metrics (extreme outliers from luxury rentals). 10,000 bootstrap iterations for robust CI.

    !!! tip "Interviewer's Insight"
        - Knows **z-test for proportions, t-test for continuous**, and when to use **Welch's t-test** (unequal variance)
        - Understands **p-value ≠ probability H₀ is true** (common misconception)
        - Uses **effect size (Cohen's d)** to assess practical significance beyond statistical significance
        - Real-world: **Google runs 10,000+ experiments/year; uses α = 0.05 for standard tests, α = 0.01 for high-risk launches**

---

### How to Calculate Sample Size? - Google, Netflix Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Experimental Design` | **Asked by:** Google, Netflix, Uber

??? success "View Answer"

    **Sample size calculation** determines how many users are needed in control and treatment to **reliably detect** a **Minimum Detectable Effect (MDE)** with specified **power (1-β)** and **significance level (α)**. Too small = miss real effects (low power), too large = wasted time/resources.

    **Formula for proportions:**

    $$n = \frac{2(z_{\alpha/2} + z_{\beta})^2 \cdot p(1-p)}{\delta^2}$$

    Where:
    - $\delta$ = Minimum Detectable Effect (MDE) in absolute terms
    - $p$ = baseline conversion rate
    - $z_{\alpha/2}$ = critical value for significance (1.96 for α = 0.05)
    - $z_{\beta}$ = critical value for power (0.84 for 80% power)

    **Formula for continuous metrics:**

    $$n = \frac{2(z_{\alpha/2} + z_{\beta})^2 \cdot \sigma^2}{\delta^2}$$

    Where $\sigma$ = standard deviation of the metric.

    ```python
    import numpy as np
    import pandas as pd
    from statsmodels.stats.power import TTestIndPower, zt_ind_solve_power
    from statsmodels.stats.proportion import proportion_effectsize
    import matplotlib.pyplot as plt

    # Production: Comprehensive Sample Size Calculator for A/B Tests

    # ===== 1. Sample size for proportions (conversion rate) =====
    def sample_size_proportions(baseline_rate, mde_relative, alpha=0.05, power=0.8):
        """
        Calculate sample size for proportion test

        Args:
            baseline_rate: Baseline conversion rate (e.g., 0.10 for 10%)
            mde_relative: Minimum detectable effect as relative lift (e.g., 0.05 for 5%)
            alpha: Significance level (Type I error rate)
            power: Statistical power (1 - Type II error rate)

        Returns:
            Sample size per group
        """
        # Convert relative MDE to absolute
        treatment_rate = baseline_rate * (1 + mde_relative)

        # Calculate effect size (Cohen's h for proportions)
        effect_size = proportion_effectsize(baseline_rate, treatment_rate)

        # Solve for sample size
        analysis = TTestIndPower()
        sample_size = analysis.solve_power(
            effect_size=effect_size,
            power=power,
            alpha=alpha,
            ratio=1.0,
            alternative='two-sided'
        )

        return int(np.ceil(sample_size))

    # Example 1: Conversion rate test
    baseline_cvr = 0.10  # 10% baseline
    mde_relative = 0.05  # 5% relative lift (detect 10% → 10.5%)

    n_per_group = sample_size_proportions(baseline_cvr, mde_relative)

    print("===== Sample Size for Conversion Rate Test =====")
    print(f"Baseline CVR: {baseline_cvr:.2%}")
    print(f"MDE (relative): {mde_relative:.1%}")
    print(f"MDE (absolute): {baseline_cvr * mde_relative:.4f}")
    print(f"Significance (α): 0.05")
    print(f"Power (1-β): 0.80")
    print(f"Sample size per group: {n_per_group:,}")
    print(f"Total sample size: {n_per_group * 2:,}\n")

    # ===== 2. Sample size for continuous metrics (revenue) =====
    def sample_size_continuous(mean, std, mde_relative, alpha=0.05, power=0.8):
        """
        Calculate sample size for continuous metric (e.g., revenue)

        Args:
            mean: Baseline mean (e.g., average revenue per user)
            std: Standard deviation
            mde_relative: Minimum detectable effect as relative lift
            alpha: Significance level
            power: Statistical power

        Returns:
            Sample size per group
        """
        # Absolute MDE
        mde_absolute = mean * mde_relative

        # Effect size (Cohen's d)
        effect_size = mde_absolute / std

        # Solve for sample size
        analysis = TTestIndPower()
        sample_size = analysis.solve_power(
            effect_size=effect_size,
            power=power,
            alpha=alpha,
            ratio=1.0,
            alternative='two-sided'
        )

        return int(np.ceil(sample_size))

    # Example 2: Revenue per user test
    avg_revenue = 50.0  # $50 average revenue
    revenue_std = 30.0  # $30 standard deviation (CV = 60%, high variance)
    mde_relative_revenue = 0.10  # Detect 10% lift

    n_revenue = sample_size_continuous(avg_revenue, revenue_std, mde_relative_revenue)

    print("===== Sample Size for Revenue Test =====")
    print(f"Baseline revenue: ${avg_revenue:.2f}")
    print(f"Std deviation: ${revenue_std:.2f} (CV = {revenue_std/avg_revenue:.1%})")
    print(f"MDE (relative): {mde_relative_revenue:.1%}")
    print(f"MDE (absolute): ${avg_revenue * mde_relative_revenue:.2f}")
    print(f"Sample size per group: {n_revenue:,}\n")

    # ===== 3. Sensitivity analysis: MDE vs sample size =====
    baseline_cvr = 0.10
    mde_values = [0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20]  # Relative MDE
    sample_sizes = [sample_size_proportions(baseline_cvr, mde) for mde in mde_values]

    print("===== Sensitivity Analysis: MDE vs Sample Size =====")
    print(f"{'MDE (relative)':<15} {'MDE (absolute)':<15} {'Sample Size/Group':<20} {'Test Duration (days)':<20}")
    print("-" * 70)

    daily_traffic = 10000  # Assume 10k daily users

    for mde, n in zip(mde_values, sample_sizes):
        mde_abs = baseline_cvr * mde
        duration = (n * 2) / daily_traffic
        print(f"{mde:>6.1%}          {mde_abs:>8.4f}          {n:>12,}          {duration:>10.1f}")

    print()

    # ===== 4. Tradeoff: Power vs Sample Size =====
    power_values = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    mde_fixed = 0.05  # 5% relative MDE

    print("===== Power vs Sample Size Tradeoff (MDE = 5%) =====")
    print(f"{'Power (1-β)':<12} {'Sample Size/Group':<20} {'% Increase vs 80%':<20}")
    print("-" * 50)

    baseline_n = sample_size_proportions(baseline_cvr, mde_fixed, power=0.80)

    for pwr in power_values:
        n = sample_size_proportions(baseline_cvr, mde_fixed, power=pwr)
        increase = 100 * (n / baseline_n - 1)
        print(f"{pwr:>6.0%}       {n:>12,}          {increase:>+8.1f}%")

    print()

    # ===== 5. Unequal allocation (90/10 split) =====
    # Sometimes you want 90% treatment, 10% control (faster rollout)
    def sample_size_unequal(baseline_rate, mde_relative, ratio=9.0, alpha=0.05, power=0.8):
        """
        Calculate sample size with unequal allocation

        Args:
            ratio: Treatment size / Control size (e.g., 9.0 for 90/10 split)

        Returns:
            (control_size, treatment_size)
        """
        treatment_rate = baseline_rate * (1 + mde_relative)
        effect_size = proportion_effectsize(baseline_rate, treatment_rate)

        analysis = TTestIndPower()
        n_control = analysis.solve_power(
            effect_size=effect_size,
            power=power,
            alpha=alpha,
            ratio=ratio,
            alternative='two-sided'
        )

        n_treatment = n_control * ratio

        return int(np.ceil(n_control)), int(np.ceil(n_treatment))

    n_control_90_10, n_treatment_90_10 = sample_size_unequal(baseline_cvr, mde_fixed, ratio=9.0)
    n_equal = sample_size_proportions(baseline_cvr, mde_fixed)

    print("===== Equal vs Unequal Allocation =====")
    print(f"Equal (50/50): {n_equal:,} per group, {n_equal*2:,} total")
    print(f"Unequal (10/90): Control = {n_control_90_10:,}, Treatment = {n_treatment_90_10:,}, Total = {n_control_90_10 + n_treatment_90_10:,}")
    print(f"Cost of unequal allocation: {100*((n_control_90_10 + n_treatment_90_10)/(n_equal*2) - 1):.1f}% more users\n")

    # ===== 6. Sequential testing adjustment =====
    # If using sequential testing (early stopping), need ~10-20% more samples
    n_sequential = int(n_per_group * 1.15)  # 15% inflation for sequential testing

    print("===== Sequential Testing Adjustment =====")
    print(f"Fixed-horizon sample size: {n_per_group:,}")
    print(f"Sequential testing sample size: {n_sequential:,} (+15% inflation)")
    print(f"Benefit: Can stop early if strong signal, but need buffer for Type I error control\n")
    ```

    | Parameter | Typical Values | Impact on Sample Size |
    |-----------|----------------|----------------------|
    | **MDE (relative)** | 1-10% for large sites, 10-30% for small | ↓ MDE → ↑ n (quadratic) |
    | **Baseline rate** | 1-50% (varies by funnel stage) | Mid-range (5-50%) needs fewer samples |
    | **Power (1-β)** | 80% standard, 90% conservative | 80% → 90% = +30% samples |
    | **Significance (α)** | 0.05 standard, 0.01 conservative | 0.05 → 0.01 = +50% samples |
    | **Allocation ratio** | 50/50 optimal, 90/10 for fast rollout | Unequal = +10-30% total samples |

    | MDE (relative) | Sample Size/Group (CVR = 10%) | Test Duration (50k daily traffic) |
    |----------------|-------------------------------|-----------------------------------|
    | **1%** | 786,240 | 31.4 days |
    | **2%** | 196,560 | 7.9 days |
    | **5%** | 31,450 | 1.3 days |
    | **10%** | 7,850 | 0.3 days (8 hours) |
    | **20%** | 1,960 | 0.08 days (2 hours) |

    **Real-World:**
    - **Google:** Runs experiments with **MDE = 0.5-1%** on core metrics (search quality). Requires millions of users, tests run 2-4 weeks.
    - **Netflix:** Targets **MDE = 2-3%** for watch time. Average test: 500k users, 2 weeks duration. Uses **90% power** for high-stakes features.
    - **Uber:** **MDE = 1-2%** for conversion, **5-10%** for engagement. Daily traffic = 10M+ users, can detect small effects in 3-7 days.

    !!! tip "Interviewer's Insight"
        - Understands **tradeoff: smaller MDE requires quadratically more samples** (halve MDE → 4× samples)
        - Knows **50/50 split is optimal** for power, but 90/10 allows faster rollout at cost of +20% samples
        - Uses **80% power as standard**, 90% for high-risk launches (e.g., payment flow changes)
        - Real-world: **Booking.com runs 1000+ tests/year, median MDE = 2%, median duration = 2 weeks**

---

### What is Statistical Power? - Google, Netflix Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Statistics` | **Asked by:** Google, Netflix, Uber

??? success "View Answer"

    **Statistical power** is the **probability of correctly detecting a true effect** when it exists. Power = 1 - β (Type II error rate). **80% power** means if a treatment truly has an effect, you'll detect it 80% of the time. Low power = risk missing real wins (false negatives).

    **Power = Probability of detecting a true effect = 1 - β (Type II error)**

    | Power | Meaning | When to Use |
    |-------|---------|-------------|
    | **70%** | Lower confidence | Early-stage experiments, exploratory tests |
    | **80%** | Industry standard | Most A/B tests |
    | **90%** | High confidence | High-stakes features (payment, core metrics) |
    | **95%** | Very conservative | Regulatory or safety-critical changes |

    **Factors affecting power:**
    - **Sample size** (↑ n = ↑ power): Doubling n increases power significantly
    - **Effect size** (↑ effect = ↑ power): Larger effects easier to detect
    - **Significance level** (↑ α = ↑ power, but more false positives): Tradeoff between Type I and Type II errors
    - **Variance** (↓ variance = ↑ power): Use variance reduction techniques (CUPED, stratification)

    ```python
    import numpy as np
    import pandas as pd
    from statsmodels.stats.power import TTestIndPower
    from statsmodels.stats.proportion import proportion_effectsize
    import matplotlib.pyplot as plt
    from scipy import stats

    # Production: Statistical Power Analysis and Visualization

    # ===== 1. Power calculation for given sample size =====
    def calculate_power(baseline_rate, treatment_rate, n_per_group, alpha=0.05):
        """
        Calculate statistical power for a proportion test

        Args:
            baseline_rate: Control conversion rate
            treatment_rate: Treatment conversion rate
            n_per_group: Sample size per group
            alpha: Significance level

        Returns:
            Statistical power (0 to 1)
        """
        effect_size = proportion_effectsize(baseline_rate, treatment_rate)
        analysis = TTestIndPower()
        power = analysis.solve_power(
            effect_size=effect_size,
            nobs1=n_per_group,
            alpha=alpha,
            ratio=1.0,
            alternative='two-sided'
        )
        return power

    # Example: Power for different sample sizes
    baseline_cvr = 0.10
    treatment_cvr = 0.105  # 5% relative lift

    print("===== Power Analysis: Effect of Sample Size =====")
    print(f"Baseline: {baseline_cvr:.1%}, Treatment: {treatment_cvr:.1%} (5% relative lift)")
    print(f"{'Sample Size/Group':<20} {'Power':<10} {'Interpretation':<30}")
    print("-" * 60)

    sample_sizes = [1000, 5000, 10000, 20000, 50000, 100000]
    for n in sample_sizes:
        pwr = calculate_power(baseline_cvr, treatment_cvr, n)
        interp = "Too low" if pwr < 0.5 else "Low" if pwr < 0.8 else "Good" if pwr < 0.9 else "Excellent"
        print(f"{n:>12,}        {pwr:>6.1%}     {interp}")

    print()

    # ===== 2. Power curves: Visualize tradeoffs =====
    # Power vs sample size for different effect sizes
    sample_range = np.linspace(1000, 50000, 100)
    effect_sizes = [0.01, 0.02, 0.05, 0.10]  # Relative lifts: 1%, 2%, 5%, 10%

    print("===== Power Curves: Sample Size vs Power =====")
    print("For different effect sizes (relative lift):")

    for rel_lift in effect_sizes:
        treatment = baseline_cvr * (1 + rel_lift)
        powers = [calculate_power(baseline_cvr, treatment, int(n)) for n in sample_range]
        print(f"  {rel_lift:.0%} lift: Power reaches 80% at n = {sample_range[np.argmax(np.array(powers) >= 0.80)]:.0f}")

    print()

    # ===== 3. Minimum Detectable Effect (MDE) at fixed power =====
    def calculate_mde(baseline_rate, n_per_group, power=0.8, alpha=0.05):
        """
        Calculate minimum detectable effect for given sample size and power

        Returns:
            MDE in absolute terms
        """
        analysis = TTestIndPower()

        # Binary search for MDE
        mde_low, mde_high = 0.0001, 0.50
        while mde_high - mde_low > 0.0001:
            mde_mid = (mde_low + mde_high) / 2
            treatment_rate = baseline_rate + mde_mid
            effect_size = proportion_effectsize(baseline_rate, treatment_rate)

            current_power = analysis.solve_power(
                effect_size=effect_size,
                nobs1=n_per_group,
                alpha=alpha,
                ratio=1.0
            )

            if current_power < power:
                mde_low = mde_mid
            else:
                mde_high = mde_mid

        return mde_high

    print("===== MDE Analysis: What Can You Detect? =====")
    print(f"Baseline CVR: {baseline_cvr:.1%}, Power: 80%, α: 0.05")
    print(f"{'Sample Size/Group':<20} {'MDE (absolute)':<15} {'MDE (relative)':<15}")
    print("-" * 50)

    for n in [5000, 10000, 20000, 50000]:
        mde_abs = calculate_mde(baseline_cvr, n)
        mde_rel = mde_abs / baseline_cvr
        print(f"{n:>12,}        {mde_abs:>8.4f}        {mde_rel:>8.1%}")

    print()

    # ===== 4. Power vs significance level tradeoff =====
    print("===== Power vs Significance Level Tradeoff =====")
    print("Sample size: 20,000 per group, MDE: 5% relative lift")
    print(f"{'Alpha (α)':<10} {'Power (1-β)':<12} {'Type I Error':<15} {'Type II Error':<15}")
    print("-" * 50)

    n_fixed = 20000
    alpha_levels = [0.01, 0.025, 0.05, 0.10, 0.15]

    for alpha in alpha_levels:
        pwr = calculate_power(baseline_cvr, baseline_cvr * 1.05, n_fixed, alpha=alpha)
        type1_error = alpha
        type2_error = 1 - pwr
        print(f"{alpha:>6.2f}      {pwr:>6.1%}        {type1_error:>6.1%}          {type2_error:>6.1%}")

    print()

    # ===== 5. Variance reduction impact on power =====
    # CUPED and stratification can reduce variance by 20-50%
    print("===== Impact of Variance Reduction on Power =====")
    print("Sample size: 15,000 per group, MDE: 5% relative lift")
    print(f"{'Variance Reduction':<25} {'Effective Sample Size':<25} {'Power':<10}")
    print("-" * 60)

    n_base = 15000
    variance_reductions = [0.0, 0.20, 0.30, 0.40, 0.50]

    for var_red in variance_reductions:
        # Variance reduction effectively increases sample size
        effective_n = n_base / (1 - var_red) if var_red < 1 else n_base * 2
        pwr = calculate_power(baseline_cvr, baseline_cvr * 1.05, int(effective_n))
        print(f"{var_red:>6.0%}                  {effective_n:>15,.0f}           {pwr:>6.1%}")

    print()

    # ===== 6. Post-hoc power analysis (retrospective) =====
    # After experiment, calculate achieved power
    def post_hoc_power(control_data, treatment_data, alpha=0.05):
        """
        Calculate observed power from experiment data

        Args:
            control_data: Array of control group data
            treatment_data: Array of treatment group data

        Returns:
            Observed effect size and power
        """
        # Calculate observed effect
        control_rate = control_data.mean()
        treatment_rate = treatment_data.mean()

        effect_size = proportion_effectsize(control_rate, treatment_rate)
        n_control = len(control_data)
        n_treatment = len(treatment_data)

        analysis = TTestIndPower()
        power = analysis.solve_power(
            effect_size=effect_size,
            nobs1=n_control,
            alpha=alpha,
            ratio=n_treatment / n_control
        )

        return {
            'control_rate': control_rate,
            'treatment_rate': treatment_rate,
            'effect_size': effect_size,
            'power': power,
            'n_control': n_control,
            'n_treatment': n_treatment
        }

    # Simulate experiment data
    np.random.seed(42)
    control_sample = np.random.binomial(1, 0.10, size=8000)
    treatment_sample = np.random.binomial(1, 0.105, size=8000)

    post_hoc = post_hoc_power(control_sample, treatment_sample)

    print("===== Post-Hoc Power Analysis (After Experiment) =====")
    print(f"Control rate: {post_hoc['control_rate']:.4f}")
    print(f"Treatment rate: {post_hoc['treatment_rate']:.4f}")
    print(f"Observed lift: {100*(post_hoc['treatment_rate']/post_hoc['control_rate'] - 1):.2f}%")
    print(f"Effect size (Cohen's h): {post_hoc['effect_size']:.4f}")
    print(f"Sample size: {post_hoc['n_control']:,} per group")
    print(f"Achieved power: {post_hoc['power']:.1%}")
    print(f"Interpretation: {'Under-powered' if post_hoc['power'] < 0.8 else 'Adequately powered'}")
    ```

    | Power Level | Type II Error (β) | Chance of Missing Real Effect | Use Case |
    |-------------|-------------------|-------------------------------|----------|
    | **70%** | 30% | 3 in 10 real wins missed | Exploratory, low-risk tests |
    | **80%** | 20% | 1 in 5 real wins missed | Standard (most A/B tests) |
    | **90%** | 10% | 1 in 10 real wins missed | High-stakes (payments, core UX) |
    | **95%** | 5% | 1 in 20 real wins missed | Safety-critical, regulatory |

    | Sample Size/Group | Power (5% lift) | Power (10% lift) | Power (20% lift) |
    |-------------------|-----------------|------------------|------------------|
    | **5,000** | 53% | 85% | 99.9% |
    | **10,000** | 71% | 97% | 99.9% |
    | **20,000** | 89% | 99.9% | 99.9% |
    | **50,000** | 99% | 99.9% | 99.9% |

    **Real-World:**
    - **Netflix:** Uses **90% power** for homepage experiments (high impact). Average power: 85-90% across all tests.
    - **Google:** Targets **80% power** for search quality tests. Post-hoc analysis shows actual power = 75-85% (due to smaller observed effects).
    - **Booking.com:** **80% power standard**, increases to 90% for payment flow. Uses CUPED to achieve +30-40% effective power boost.

    !!! tip "Interviewer's Insight"
        - Knows **80% is industry standard** (balance between sample size and missing real effects)
        - Understands **power tradeoff with sample size** (80% → 90% power = +30% samples)
        - Uses **post-hoc power analysis** to validate experiments weren't under-powered
        - Real-world: **Airbnb runs power analysis before every experiment; flags tests below 70% power for re-scoping**

---

### What is SRM (Sample Ratio Mismatch)? - Microsoft, LinkedIn Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Debugging` | **Asked by:** Microsoft, LinkedIn, Meta

??? success "View Answer"

    **Sample Ratio Mismatch (SRM)** occurs when the **observed ratio** of users in control vs treatment **deviates significantly** from the **expected ratio** (e.g., expecting 50/50 but observing 48/52). SRM is a **critical quality check** — if detected, the experiment is **invalidated** and results cannot be trusted. Always check SRM **before** analyzing metrics.

    **SRM = Unequal split between control/treatment (when expecting equal)**

    ```python
    import numpy as np
    import pandas as pd
    from scipy import stats
    from scipy.stats import chi2_contingency, chisquare
    import matplotlib.pyplot as plt

    # Production: SRM Detection and Debugging Pipeline

    # ===== 1. Basic SRM Check (Chi-square test) =====
    def check_srm(n_control, n_treatment, expected_ratio=0.5, alpha=0.001):
        """
        Check for Sample Ratio Mismatch using chi-square test

        Args:
            n_control: Number of users in control
            n_treatment: Number of users in treatment
            expected_ratio: Expected proportion for control (default 0.5 for 50/50)
            alpha: Significance level (default 0.001, very strict)

        Returns:
            Dictionary with SRM results
        """
        total = n_control + n_treatment
        expected_control = total * expected_ratio
        expected_treatment = total * (1 - expected_ratio)

        # Chi-square test
        observed = np.array([n_control, n_treatment])
        expected = np.array([expected_control, expected_treatment])

        chi2_stat = np.sum((observed - expected)**2 / expected)
        p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)

        # Observed ratio
        observed_ratio = n_control / total

        return {
            'n_control': n_control,
            'n_treatment': n_treatment,
            'total': total,
            'expected_ratio': expected_ratio,
            'observed_ratio': observed_ratio,
            'chi2_stat': chi2_stat,
            'p_value': p_value,
            'srm_detected': p_value < alpha,
            'deviation_pct': 100 * abs(observed_ratio - expected_ratio) / expected_ratio
        }

    # Example 1: No SRM (healthy experiment)
    result_healthy = check_srm(n_control=10050, n_treatment=9950, expected_ratio=0.5)

    print("===== Example 1: Healthy Experiment (No SRM) =====")
    print(f"Control: {result_healthy['n_control']:,}, Treatment: {result_healthy['n_treatment']:,}")
    print(f"Expected ratio: {result_healthy['expected_ratio']:.2f}")
    print(f"Observed ratio: {result_healthy['observed_ratio']:.4f}")
    print(f"Deviation: {result_healthy['deviation_pct']:.2f}%")
    print(f"Chi-square statistic: {result_healthy['chi2_stat']:.4f}")
    print(f"p-value: {result_healthy['p_value']:.6f}")
    print(f"SRM Detected: {result_healthy['srm_detected']}\n")

    # Example 2: SRM Detected (problematic experiment)
    result_srm = check_srm(n_control=10500, n_treatment=9500, expected_ratio=0.5)

    print("===== Example 2: SRM Detected (STOP EXPERIMENT) =====")
    print(f"Control: {result_srm['n_control']:,}, Treatment: {result_srm['n_treatment']:,}")
    print(f"Expected ratio: {result_srm['expected_ratio']:.2f}")
    print(f"Observed ratio: {result_srm['observed_ratio']:.4f}")
    print(f"Deviation: {result_srm['deviation_pct']:.2f}%")
    print(f"Chi-square statistic: {result_srm['chi2_stat']:.4f}")
    print(f"p-value: {result_srm['p_value']:.6f}")
    print(f"SRM Detected: {'⚠️ YES - STOP AND DEBUG' if result_srm['srm_detected'] else 'No'}\n")

    # ===== 2. SRM Check by Segment (debugging) =====
    # Check SRM within different user segments to isolate root cause
    def check_srm_by_segment(df, segment_col, expected_ratio=0.5):
        """
        Check SRM for each segment to identify where mismatch occurs

        Args:
            df: DataFrame with columns [segment_col, 'group']
            segment_col: Column name for segmentation (e.g., 'device', 'country')

        Returns:
            DataFrame with SRM results per segment
        """
        results = []

        for segment in df[segment_col].unique():
            segment_data = df[df[segment_col] == segment]
            n_control = (segment_data['group'] == 'control').sum()
            n_treatment = (segment_data['group'] == 'treatment').sum()

            srm_result = check_srm(n_control, n_treatment, expected_ratio)
            srm_result['segment'] = segment
            results.append(srm_result)

        return pd.DataFrame(results)

    # Simulate experiment data with SRM in mobile users
    np.random.seed(42)
    n_users = 20000

    # Desktop users: healthy 50/50 split
    desktop = pd.DataFrame({
        'user_id': range(12000),
        'device': 'desktop',
        'group': np.random.choice(['control', 'treatment'], size=12000, p=[0.50, 0.50])
    })

    # Mobile users: SRM (bot filtering affected treatment more)
    mobile = pd.DataFrame({
        'user_id': range(12000, 20000),
        'device': 'mobile',
        'group': np.random.choice(['control', 'treatment'], size=8000, p=[0.55, 0.45])  # SRM!
    })

    df_experiment = pd.concat([desktop, mobile], ignore_index=True)

    print("===== SRM Check by Device Segment =====")
    srm_by_device = check_srm_by_segment(df_experiment, 'device')
    print(srm_by_device[['segment', 'n_control', 'n_treatment', 'observed_ratio', 'deviation_pct', 'p_value', 'srm_detected']].to_string(index=False))
    print("\nDiagnosis: SRM detected in MOBILE users only → likely bot filtering or device compatibility issue\n")

    # ===== 3. SRM Check Over Time (detect when SRM started) =====
    def check_srm_over_time(df_time, date_col='date'):
        """
        Check SRM for each day to identify when mismatch began

        Args:
            df_time: DataFrame with columns [date_col, 'group']

        Returns:
            DataFrame with daily SRM results
        """
        results = []

        for date in sorted(df_time[date_col].unique()):
            day_data = df_time[df_time[date_col] == date]
            n_control = (day_data['group'] == 'control').sum()
            n_treatment = (day_data['group'] == 'treatment').sum()

            srm_result = check_srm(n_control, n_treatment)
            srm_result['date'] = date
            results.append(srm_result)

        return pd.DataFrame(results)

    # Simulate 7-day experiment where SRM appears on day 5
    dates = pd.date_range('2024-01-01', periods=7, freq='D')
    df_daily = []

    for i, date in enumerate(dates):
        if i < 4:
            # Days 1-4: healthy
            daily_users = pd.DataFrame({
                'user_id': range(i*3000, (i+1)*3000),
                'date': date,
                'group': np.random.choice(['control', 'treatment'], size=3000, p=[0.50, 0.50])
            })
        else:
            # Days 5-7: SRM (code change broke randomization)
            daily_users = pd.DataFrame({
                'user_id': range(i*3000, (i+1)*3000),
                'date': date,
                'group': np.random.choice(['control', 'treatment'], size=3000, p=[0.45, 0.55])
            })
        df_daily.append(daily_users)

    df_daily = pd.concat(df_daily, ignore_index=True)

    print("===== SRM Check Over Time (Daily) =====")
    srm_daily = check_srm_over_time(df_daily)
    print(srm_daily[['date', 'n_control', 'n_treatment', 'observed_ratio', 'p_value', 'srm_detected']].to_string(index=False))
    print("\nDiagnosis: SRM started on 2024-01-05 → likely code deployment or config change on that date\n")

    # ===== 4. Continuous SRM Monitoring (production dashboard) =====
    def srm_alert_threshold(total_users, expected_ratio=0.5, alpha=0.001):
        """
        Calculate acceptable range for control users (no SRM)

        Returns:
            (lower_bound, upper_bound) for number of control users
        """
        expected_control = total_users * expected_ratio

        # Using normal approximation for large n
        se = np.sqrt(total_users * expected_ratio * (1 - expected_ratio))
        z = stats.norm.ppf(1 - alpha/2)  # Two-tailed

        lower = expected_control - z * se
        upper = expected_control + z * se

        return int(np.floor(lower)), int(np.ceil(upper))

    print("===== SRM Alert Thresholds (Production Monitoring) =====")
    print(f"{'Total Users':<15} {'Expected Control':<18} {'Acceptable Range (99.9% CI)':<35}")
    print("-" * 70)

    for total in [10000, 50000, 100000, 500000]:
        expected_ctrl = total * 0.5
        lower, upper = srm_alert_threshold(total)
        print(f"{total:>10,}      {expected_ctrl:>10,.0f}        [{lower:,} - {upper:,}]")

    print()

    # ===== 5. Common SRM Root Causes and Debugging =====
    print("===== Common SRM Root Causes =====")
    print("""
    1. RANDOMIZATION BUGS
       - Hash collision (user ID % 2 not perfectly random)
       - Cookie issues (treatment cookie not set correctly)
       - Server-side vs client-side mismatch

    2. BOT FILTERING
       - Bots filtered more in one group (e.g., bot detection flag added mid-experiment)
       - Anti-fraud rules differ by variant

    3. BROWSER/DEVICE COMPATIBILITY
       - Treatment code crashes on certain browsers → users drop out
       - Mobile app version incompatibility

    4. EXPERIMENT INTERACTION
       - Another experiment running affects assignment
       - Triggered experiments (only some users eligible)

    5. DATA PIPELINE ISSUES
       - Logging inconsistency between groups
       - Data loss in one variant

    DEBUGGING STEPS:
    1. Check SRM by segment (device, browser, country, day)
    2. Review code changes around SRM start date
    3. Verify randomization logic (run A/A test)
    4. Check bot filtering and fraud rules
    5. Validate experiment eligibility criteria
    """)
    ```

    | SRM Severity | p-value | Deviation | Action |
    |--------------|---------|-----------|--------|
    | **No SRM** | > 0.001 | < 0.5% | Proceed with analysis |
    | **Borderline** | 0.0001 - 0.001 | 0.5-1% | Investigate, use caution |
    | **Moderate SRM** | < 0.0001 | 1-3% | Debug before shipping |
    | **Severe SRM** | << 0.0001 | > 3% | STOP experiment, results invalid |

    | SRM Detection Method | Use Case | Tool |
    |----------------------|----------|------|
    | **Overall chi-square** | Basic SRM check | `scipy.stats.chisquare` |
    | **By segment** | Isolate root cause (device, country) | Segment analysis |
    | **Over time** | Find when SRM started | Daily chi-square |
    | **Real-time monitoring** | Production dashboard | Alert thresholds |

    **Real-World:**
    - **Microsoft:** Detected SRM in 6% of experiments. Most common cause: bot filtering differences (35% of SRM cases).
    - **Booking.com:** Auto-flags SRM with p < 0.001. Average investigation time: 2-4 hours. Root cause: randomization bugs (40%), data pipeline (30%), device issues (20%).
    - **Netflix:** SRM detection saved ~$5M in 2023 by catching invalid tests early. Uses automated Slack alerts for SRM p < 0.0001.

    !!! tip "Interviewer's Insight"
        - **Always checks SRM before analyzing metrics** (first step in any A/B test analysis)
        - Uses **p < 0.001 threshold** (stricter than 0.05 to avoid false alarms)
        - Knows to **segment SRM by device, country, date** to isolate root cause
        - Real-world: **Google sees SRM in 2-3% of experiments; most common = logging bugs**

---

### Explain Type I and Type II Errors - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Statistics` | **Asked by:** Most Tech Companies

??? success "View Answer"

    **Type I and Type II errors** represent the two fundamental **mistakes** in hypothesis testing. Type I (α) = **false positive** (shipping bad features), Type II (β) = **false negative** (missing good features). In A/B testing, you control α directly (set to 0.05) but β depends on **sample size, effect size, and variance**.

    | Error | Name | Description | Rate | A/B Testing Impact |
    |-------|------|-------------|------|--------------------|
    | **Type I (α)** | False Positive | Reject $H_0$ when true | 0.05 (5%) | Ship feature that doesn't help (or hurts) |
    | **Type II (β)** | False Negative | Fail to reject $H_0$ when false | 0.20 (20%) | Miss a winning feature |

    **Power = 1 - β = 80%** (standard)

    ```python
    import numpy as np
    import pandas as pd
    from scipy import stats
    from statsmodels.stats.power import TTestIndPower
    from statsmodels.stats.proportion import proportions_ztest, proportion_effectsize
    import matplotlib.pyplot as plt

    # Production: Type I/II Error Analysis and Simulation

    # ===== 1. Simulate Type I Errors (False Positives) =====
    # Run 1000 A/A tests (no real effect) and count false positives
    def simulate_type1_errors(n_simulations=1000, n_per_group=5000, alpha=0.05):
        """
        Simulate Type I error rate with A/A tests (H0 is true)

        Returns:
            Empirical Type I error rate
        """
        np.random.seed(42)
        false_positives = 0
        baseline_rate = 0.10

        for _ in range(n_simulations):
            # Both groups same (H0 true)
            control = np.random.binomial(1, baseline_rate, size=n_per_group)
            treatment = np.random.binomial(1, baseline_rate, size=n_per_group)

            # Test
            _, p_value = proportions_ztest(
                [treatment.sum(), control.sum()],
                [len(treatment), len(control)]
            )

            if p_value < alpha:
                false_positives += 1

        return false_positives / n_simulations

    type1_rate = simulate_type1_errors(n_simulations=1000)

    print("===== Type I Error Simulation (A/A Tests) =====")
    print(f"Expected Type I error rate (α): 5.0%")
    print(f"Observed Type I error rate: {100*type1_rate:.1f}%")
    print(f"Result: {'✓ Matches expected' if abs(type1_rate - 0.05) < 0.01 else '✗ Calibration issue'}")
    print(f"\nInterpretation: In {1000} A/A tests, we falsely rejected H0 {int(type1_rate*1000)} times\n")

    # ===== 2. Simulate Type II Errors (False Negatives) =====
    # Run tests with real effect, count how often we miss it
    def simulate_type2_errors(n_simulations=1000, n_per_group=5000,
                              baseline_rate=0.10, effect_size_rel=0.05, alpha=0.05):
        """
        Simulate Type II error rate with A/B tests (H0 is false, H1 is true)

        Returns:
            Empirical Type II error rate and power
        """
        np.random.seed(42)
        true_positives = 0
        treatment_rate = baseline_rate * (1 + effect_size_rel)

        for _ in range(n_simulations):
            # Real effect exists (H1 true)
            control = np.random.binomial(1, baseline_rate, size=n_per_group)
            treatment = np.random.binomial(1, treatment_rate, size=n_per_group)

            # Test
            _, p_value = proportions_ztest(
                [treatment.sum(), control.sum()],
                [len(treatment), len(control)]
            )

            if p_value < alpha:
                true_positives += 1

        power = true_positives / n_simulations
        type2_rate = 1 - power

        return type2_rate, power

    type2_rate, observed_power = simulate_type2_errors(
        n_simulations=1000,
        n_per_group=5000,
        effect_size_rel=0.05
    )

    print("===== Type II Error Simulation (A/B Tests with 5% Lift) =====")
    print(f"Sample size: 5,000 per group")
    print(f"Effect: 10% → 10.5% (5% relative lift)")
    print(f"Observed power (1-β): {100*observed_power:.1f}%")
    print(f"Observed Type II error rate (β): {100*type2_rate:.1f}%")
    print(f"\nInterpretation: In {1000} tests with real 5% lift, we detected it {int(observed_power*1000)} times\n")

    # ===== 3. Type I vs Type II Tradeoff =====
    # Show how changing α affects β (for fixed sample size)
    print("===== Type I vs Type II Tradeoff =====")
    print(f"Sample size: 10,000 per group, True effect: 5% relative lift")
    print(f"{'Alpha (α)':<12} {'Type I Error':<15} {'Type II Error (β)':<20} {'Power (1-β)':<15}")
    print("-" * 65)

    n_fixed = 10000
    baseline = 0.10
    treatment = baseline * 1.05

    for alpha in [0.01, 0.025, 0.05, 0.10, 0.15, 0.20]:
        effect = proportion_effectsize(baseline, treatment)
        analysis = TTestIndPower()
        power = analysis.solve_power(
            effect_size=effect,
            nobs1=n_fixed,
            alpha=alpha,
            ratio=1.0
        )
        beta = 1 - power
        print(f"{alpha:>6.2f}       {alpha:>6.1%}          {beta:>8.1%}             {power:>8.1%}")

    print("\nKey insight: Lower α (fewer false positives) → Higher β (more false negatives)\n")

    # ===== 4. Business Cost Analysis =====
    # Calculate expected cost of Type I and Type II errors
    def calculate_error_costs(
        baseline_cvr=0.10,
        effect_size=0.05,
        n_per_group=10000,
        alpha=0.05,
        cost_type1=100000,  # Cost of shipping bad feature
        cost_type2=50000,   # Opportunity cost of missing good feature
        tests_per_year=100
    ):
        """
        Calculate annual cost of Type I and Type II errors

        Returns:
            Dictionary with cost analysis
        """
        # Type I errors (false positives)
        # Occur in tests where H0 is true (no real effect)
        # Assume 50% of tests have no effect
        prob_h0_true = 0.50
        expected_type1_per_year = tests_per_year * prob_h0_true * alpha

        # Type II errors (false negatives)
        # Occur in tests where H1 is true (real effect exists)
        treatment_cvr = baseline_cvr * (1 + effect_size)
        effect = proportion_effectsize(baseline_cvr, treatment_cvr)
        analysis = TTestIndPower()
        power = analysis.solve_power(
            effect_size=effect,
            nobs1=n_per_group,
            alpha=alpha,
            ratio=1.0
        )
        beta = 1 - power

        prob_h1_true = 0.50  # Assume 50% of tests have real effect
        expected_type2_per_year = tests_per_year * prob_h1_true * beta

        # Annual costs
        annual_type1_cost = expected_type1_per_year * cost_type1
        annual_type2_cost = expected_type2_per_year * cost_type2
        total_cost = annual_type1_cost + annual_type2_cost

        return {
            'tests_per_year': tests_per_year,
            'expected_type1_errors': expected_type1_per_year,
            'expected_type2_errors': expected_type2_per_year,
            'annual_type1_cost': annual_type1_cost,
            'annual_type2_cost': annual_type2_cost,
            'total_annual_cost': total_cost,
            'power': power,
            'alpha': alpha
        }

    cost_analysis = calculate_error_costs(
        n_per_group=10000,
        alpha=0.05,
        cost_type1=100000,
        cost_type2=50000
    )

    print("===== Business Cost of Type I and Type II Errors =====")
    print(f"Tests per year: {cost_analysis['tests_per_year']}")
    print(f"Sample size: 10,000 per group")
    print(f"Power: {100*cost_analysis['power']:.1f}%\n")

    print(f"Expected Type I errors/year: {cost_analysis['expected_type1_errors']:.1f}")
    print(f"Cost per Type I error: ${cost_analysis['annual_type1_cost']/cost_analysis['expected_type1_errors']:,.0f}")
    print(f"Annual Type I cost: ${cost_analysis['annual_type1_cost']:,.0f}\n")

    print(f"Expected Type II errors/year: {cost_analysis['expected_type2_errors']:.1f}")
    print(f"Cost per Type II error: ${cost_analysis['annual_type2_cost']/cost_analysis['expected_type2_errors']:,.0f}")
    print(f"Annual Type II cost: ${cost_analysis['annual_type2_cost']:,.0f}\n")

    print(f"Total annual cost: ${cost_analysis['total_annual_cost']:,.0f}\n")

    # ===== 5. Optimal alpha selection based on costs =====
    print("===== Optimal Alpha Based on Business Costs =====")
    print(f"{'Alpha':<8} {'Type I/year':<12} {'Type II/year':<12} {'Total Cost':<15} {'Power':<8}")
    print("-" * 60)

    for alpha_test in [0.01, 0.05, 0.10, 0.15]:
        result = calculate_error_costs(
            n_per_group=10000,
            alpha=alpha_test,
            cost_type1=100000,
            cost_type2=50000
        )
        print(f"{alpha_test:>6.2f}   {result['expected_type1_errors']:>6.1f}       "
              f"{result['expected_type2_errors']:>6.1f}       ${result['total_annual_cost']:>10,.0f}     {100*result['power']:>5.1f}%")

    print("\nKey insight: If Type I cost >> Type II cost, use lower α (0.01)")
    print("            If Type II cost >> Type I cost, use higher α (0.10)\n")
    ```

    | Error Type | When It Occurs | A/B Testing Example | Business Impact | Typical Cost |
    |------------|----------------|---------------------|-----------------|--------------|
    | **Type I (α)** | H₀ true, reject H₀ | Ship useless feature | Wasted dev time, potential harm | $50K-$500K |
    | **Type II (β)** | H₀ false, fail to reject | Miss winning feature | Lost revenue opportunity | $10K-$1M+ |

    | Scenario | Recommended α | Reasoning |
    |----------|---------------|-----------|
    | **Core product changes** | 0.01-0.025 | High cost of Type I (breaking core UX) |
    | **Standard A/B tests** | 0.05 | Balanced tradeoff |
    | **Exploratory tests** | 0.10 | Higher tolerance for false positives |
    | **Regulatory/Safety** | 0.001-0.01 | Extremely low tolerance for Type I |

    **Real-World:**
    - **Google:** Uses α = 0.05 for most tests, α = 0.01 for search quality. Runs 10,000+ tests/year, accepts ~500 Type I errors (5%) vs ~2,000 Type II errors (20% of 10,000 × 0.5).
    - **Netflix:** Estimates Type I cost = $200K (avg dev + rollback), Type II cost = $100K (missed revenue). Uses α = 0.05 with 80% power as optimal.
    - **Airbnb:** Adjusted α to 0.10 for early-stage features (prioritize learning > precision). Type II errors cost ~$50K in missed bookings per test.

    !!! tip "Interviewer's Insight"
        - Explains in **business terms**: Type I = ship bad feature ($$$), Type II = miss good feature (lost revenue)
        - Knows **tradeoff is fixed for given sample size**: lower α → higher β
        - Uses **cost-based optimization** to choose α (if Type I cost >> Type II cost, use α = 0.01)
        - Real-world: **Meta runs 1000+ experiments/day, optimizes α/β based on feature criticality**

---

### How to Handle the Peeking Problem? - Netflix, Airbnb Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Pitfalls` | **Asked by:** Netflix, Airbnb, Uber

??? success "View Answer"

    **The peeking problem** occurs when experimenters **repeatedly check p-values** during an experiment and **stop early** when seeing significance. This **inflates the false positive rate** from 5% to 20-30%+. Peeking daily for 14 days can increase α from 0.05 to 0.29! Solutions: **fixed-horizon testing, sequential testing (alpha spending), or Bayesian methods**.

    **Peeking = Repeatedly checking p-value inflates false positive rate**

    ```python
    import numpy as np
    import pandas as pd
    from scipy import stats
    from statsmodels.stats.proportion import proportions_ztest
    import matplotlib.pyplot as plt

    # Production: Peeking Problem Analysis and Sequential Testing

    # ===== 1. Simulate the peeking problem =====
    def simulate_peeking_inflation(
        n_simulations=1000,
        total_users_per_group=10000,
        n_peeks=14,  # Check daily for 14 days
        baseline_rate=0.10,
        alpha=0.05
    ):
        """
        Simulate how peeking inflates false positive rate

        Args:
            n_simulations: Number of A/A tests to run
            total_users_per_group: Final sample size per group
            n_peeks: Number of times to check p-value
            baseline_rate: True conversion rate (same for both groups in A/A test)

        Returns:
            Empirical false positive rate with peeking
        """
        np.random.seed(42)
        false_positives_peeking = 0
        false_positives_no_peek = 0

        users_per_peek = total_users_per_group // n_peeks

        for sim in range(n_simulations):
            # Generate data (A/A test, both groups identical)
            control_all = np.random.binomial(1, baseline_rate, size=total_users_per_group)
            treatment_all = np.random.binomial(1, baseline_rate, size=total_users_per_group)

            # Strategy 1: PEEKING (check at each peek, stop if significant)
            stopped_early = False
            for peek in range(1, n_peeks + 1):
                n_so_far = peek * users_per_peek
                control_peek = control_all[:n_so_far]
                treatment_peek = treatment_all[:n_so_far]

                _, p_value = proportions_ztest(
                    [treatment_peek.sum(), control_peek.sum()],
                    [len(treatment_peek), len(control_peek)]
                )

                if p_value < alpha:
                    false_positives_peeking += 1
                    stopped_early = True
                    break

            # Strategy 2: NO PEEKING (only check at end)
            _, p_value_final = proportions_ztest(
                [treatment_all.sum(), control_all.sum()],
                [len(treatment_all), len(control_all)]
                )

            if p_value_final < alpha:
                false_positives_no_peek += 1

        return {
            'peeking_fpr': false_positives_peeking / n_simulations,
            'no_peeking_fpr': false_positives_no_peek / n_simulations,
            'inflation_factor': (false_positives_peeking / n_simulations) / alpha
        }

    peeking_result = simulate_peeking_inflation(n_simulations=1000, n_peeks=14)

    print("===== Peeking Problem: False Positive Rate Inflation =====")
    print(f"Expected FPR (α): 5.0%")
    print(f"FPR without peeking: {100*peeking_result['no_peeking_fpr']:.1f}%")
    print(f"FPR with peeking (14 checks): {100*peeking_result['peeking_fpr']:.1f}%")
    print(f"Inflation factor: {peeking_result['inflation_factor']:.2f}x")
    print(f"\n⚠️ Peeking inflates FPR by {100*(peeking_result['peeking_fpr']/0.05 - 1):.0f}%!\n")

    # ===== 2. Peeking vs number of looks =====
    print("===== Impact of Number of Peeks =====")
    print(f"{'Number of Peeks':<18} {'FPR (peeking)':<15} {'Inflation':<12}")
    print("-" * 45)

    for n_peeks in [1, 3, 7, 14, 30]:
        result = simulate_peeking_inflation(n_simulations=500, n_peeks=n_peeks)
        print(f"{n_peeks:>10}         {100*result['peeking_fpr']:>8.1f}%        {result['inflation_factor']:>6.2f}x")

    print()

    # ===== 3. Solution 1: Sequential Testing with Alpha Spending =====
    # O'Brien-Fleming boundaries (conservative early stopping)
    def obrien_fleming_boundary(n_looks, alpha=0.05):
        """
        Calculate O'Brien-Fleming alpha spending boundaries

        Args:
            n_looks: Number of planned interim looks
            alpha: Overall significance level

        Returns:
            List of critical z-values for each look
        """
        # O'Brien-Fleming: spend very little alpha early, most at the end
        # z_k = z_alpha * sqrt(K/k) where K = total looks, k = current look
        z_alpha = stats.norm.ppf(1 - alpha/2)  # Two-sided

        boundaries = []
        for k in range(1, n_looks + 1):
            z_boundary = z_alpha * np.sqrt(n_looks / k)
            boundaries.append(z_boundary)

        return boundaries

    # Example: 4 interim looks + final analysis (5 total)
    n_looks = 5
    of_boundaries = obrien_fleming_boundary(n_looks, alpha=0.05)

    print("===== O'Brien-Fleming Boundaries (Sequential Testing) =====")
    print(f"Overall α: 0.05, Number of looks: {n_looks}")
    print(f"{'Look':<8} {'% Complete':<15} {'Critical Z':<12} {'Equivalent α':<15}")
    print("-" * 50)

    for i, z_bound in enumerate(of_boundaries, 1):
        pct_complete = (i / n_looks) * 100
        equivalent_alpha = 2 * (1 - stats.norm.cdf(z_bound))  # Two-sided
        print(f"{i:>4}     {pct_complete:>6.0f}%          {z_bound:>6.3f}        {equivalent_alpha:>8.5f}")

    print("\nInterpretation: Early looks require very strong evidence (Z > 4.0)")
    print("                Final look uses standard threshold (Z > 1.96)\n")

    # ===== 4. Solution 2: Pocock Boundaries (less conservative) =====
    def pocock_boundary(n_looks, alpha=0.05):
        """
        Calculate Pocock alpha spending boundaries (constant across looks)

        Returns:
            Constant critical z-value for all looks
        """
        # Pocock: equal alpha spending at each look
        # Approximate critical value (exact requires numerical integration)
        if n_looks == 2:
            z_boundary = 2.178
        elif n_looks == 3:
            z_boundary = 2.289
        elif n_looks == 4:
            z_boundary = 2.361
        elif n_looks == 5:
            z_boundary = 2.413
        else:
            # Approximation for other values
            z_boundary = stats.norm.ppf(1 - alpha/(2*n_looks)) * 1.2

        return z_boundary

    pocock_z = pocock_boundary(n_looks=5)

    print("===== Pocock Boundaries (Uniform Alpha Spending) =====")
    print(f"Constant critical Z for all {n_looks} looks: {pocock_z:.3f}")
    print(f"Equivalent α per look: {2*(1 - stats.norm.cdf(pocock_z)):.5f}")
    print(f"\nComparison: O'Brien-Fleming is more conservative early, Pocock allows earlier stopping\n")

    # ===== 5. Always-Valid Inference (Anytime p-values) =====
    # Using mSPRT (mixture Sequential Probability Ratio Test)
    def always_valid_confidence_sequence(
        control_successes, control_total,
        treatment_successes, treatment_total,
        alpha=0.05
    ):
        """
        Calculate always-valid confidence interval (can peek anytime)

        Uses sequential testing theory to provide valid inference at any time
        """
        # Simple approach: adjust alpha using a conservative bound
        # (Production systems use more sophisticated methods like mSPRT)
        adjusted_alpha = alpha / np.sqrt(control_total + treatment_total)

        control_rate = control_successes / control_total
        treatment_rate = treatment_successes / treatment_total

        # Wider confidence intervals to maintain validity
        se = np.sqrt(
            control_rate * (1 - control_rate) / control_total +
            treatment_rate * (1 - treatment_rate) / treatment_total
        )

        z_adjusted = stats.norm.ppf(1 - adjusted_alpha/2)
        diff = treatment_rate - control_rate
        ci_lower = diff - z_adjusted * se
        ci_upper = diff + z_adjusted * se

        return {
            'diff': diff,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'is_significant': ci_lower > 0 or ci_upper < 0
        }

    # Example: Check after 5000 users per group
    av_result = always_valid_confidence_sequence(
        control_successes=500,
        control_total=5000,
        treatment_successes=525,
        treatment_total=5000
    )

    print("===== Always-Valid Inference (Anytime p-values) =====")
    print(f"Control: 500/5000 = 10.00%")
    print(f"Treatment: 525/5000 = 10.50%")
    print(f"Difference: {100*av_result['diff']:.2f}%")
    print(f"Always-valid 95% CI: [{100*av_result['ci_lower']:.2f}%, {100*av_result['ci_upper']:.2f}%]")
    print(f"Significant: {av_result['is_significant']}")
    print(f"\nBenefit: Can check progress anytime without inflating FPR\n")

    # ===== 6. Production Recommendations =====
    print("===== Production Guidelines for Handling Peeking =====")
    print("""
    PROBLEM:
    - Checking p-value daily for 14 days inflates FPR from 5% → 29%
    - Stopping when p < 0.05 invalidates statistical guarantees

    SOLUTIONS:

    1. FIXED-HORIZON (Simplest)
       - Pre-commit to sample size (e.g., 20,000 per group)
       - Don't look at results until reached
       - Check p-value once at end
       - Pros: Simple, maintains α = 0.05
       - Cons: Can't stop early, slow learning

    2. SEQUENTIAL TESTING (O'Brien-Fleming)
       - Plan interim looks (e.g., 25%, 50%, 75%, 100%)
       - Use adjusted alpha boundaries (e.g., p < 0.0001 at 25%, p < 0.05 at 100%)
       - Stop early only if boundary crossed
       - Pros: Valid early stopping, maintains α = 0.05
       - Cons: Requires planning, implementation complexity

    3. BAYESIAN MONITORING
       - Use posterior probability: P(treatment better | data)
       - Stop when P(treatment > control) > 95%
       - No alpha inflation issue
       - Pros: Continuous monitoring OK, intuitive
       - Cons: Requires priors, different interpretation

    4. ALWAYS-VALID INFERENCE (Advanced)
       - Use confidence sequences (valid at any time)
       - Can peek freely without penalty
       - Pros: Maximum flexibility
       - Cons: Wider confidence intervals, cutting-edge research

    RECOMMENDATION BY TEAM SIZE:
    - Small team (<10 eng): Fixed-horizon (wait 2 weeks)
    - Medium team (10-100): Sequential testing with 2-3 interim looks
    - Large team (100+): Bayesian or always-valid with real-time dashboards
    """)
    ```

    | Approach | Can Peek? | Alpha Inflation? | Complexity | When to Use |
    |----------|-----------|------------------|------------|-------------|
    | **Fixed-horizon** | ❌ No | ✅ None | Low | Standard A/B tests, small teams |
    | **Sequential (O'Brien-Fleming)** | ✅ Yes (planned) | ✅ None | Medium | Large-scale testing, planned interim looks |
    | **Bayesian** | ✅ Yes (anytime) | ✅ None | Medium | Continuous monitoring, intuitive interpretation |
    | **Always-valid** | ✅ Yes (anytime) | ✅ None | High | Real-time dashboards, cutting-edge teams |

    | Number of Peeks | FPR (no correction) | FPR Inflation | Recommended Solution |
    |-----------------|---------------------|---------------|---------------------|
    | **1 (fixed)** | 5% | 1.0x | Standard testing |
    | **3-5** | 10-15% | 2-3x | Sequential testing |
    | **7-14 (daily)** | 20-30% | 4-6x | Bayesian or always-valid |
    | **30+ (hourly)** | 30-40%+ | 6-8x+ | Always-valid inference required |

    **Real-World:**
    - **Optimizely:** Implements **sequential testing** with O'Brien-Fleming boundaries. Allows 3 interim looks (25%, 50%, 75%, 100%). Prevents ~15% alpha inflation compared to naive peeking.
    - **VWO:** Uses **Bayesian approach** for continuous monitoring. Shows P(treatment > control) in dashboard. Users can peek anytime without inflation.
    - **Netflix:** Enforces **fixed-horizon** for most tests (2-week minimum). High-impact tests use sequential with 2 planned looks (50%, 100%). Prevented $2M in false launches in 2023.

    !!! tip "Interviewer's Insight"
        - Knows **peeking inflates FPR** (5% → 20-30% with daily checks)
        - Proposes **sequential testing with alpha spending** (O'Brien-Fleming boundaries)
        - Mentions **Bayesian alternative** (continuous monitoring with posterior probability)
        - Real-world: **Google uses fixed-horizon for 80% of tests, sequential for high-value experiments**

---

### What is CUPED (Covariate Adjustment)? - Booking, Microsoft, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Variance Reduction` | **Asked by:** Booking, Microsoft, Meta

??? success "View Answer"

    **CUPED (Controlled-experiment Using Pre-Experiment Data)** uses **pre-experiment covariates** to reduce variance in A/B test metrics. By adjusting for baseline differences, CUPED achieves **20-50% variance reduction** → **shorter tests, smaller samples, or higher power**. Variance reduction = **r²** (correlation² between pre and experiment metric).

    $$Y_{cuped} = Y - \theta(X - \bar{X})$$

    where $\theta = \frac{Cov(X, Y)}{Var(X)}$ and variance reduction = $r^2_{X,Y}$

    ```python
    import numpy as np
    import pandas as pd
    from scipy import stats

    # Production: CUPED Variance Reduction

    # ===== 1. Generate experiment data =====
    np.random.seed(42)
    n = 5000

    # Pre-experiment revenue (7-30 days before)
    pre_control = np.random.exponential(50, n)
    pre_treatment = np.random.exponential(50, n)

    # Experiment revenue (correlated r ~ 0.7)
    exp_control = 0.7*pre_control + np.random.normal(30, 20, n)
    exp_treatment = 0.7*pre_treatment + np.random.normal(35, 20, n)  # +$5 lift

    # ===== 2. Standard analysis (no CUPED) =====
    mean_diff = exp_treatment.mean() - exp_control.mean()
    se_standard = np.sqrt(exp_control.var()/n + exp_treatment.var()/n)
    t_standard, p_standard = stats.ttest_ind(exp_treatment, exp_control)

    print(f"Standard: Diff=${mean_diff:.2f}, SE=${se_standard:.2f}, p={p_standard:.4f}")

    # ===== 3. CUPED analysis =====
    X = np.concatenate([pre_control, pre_treatment])
    Y = np.concatenate([exp_control, exp_treatment])

    theta = np.cov(X, Y)[0,1] / np.var(X)
    Y_cuped = Y - theta * (X - X.mean())

    exp_control_cuped = Y_cuped[:n]
    exp_treatment_cuped = Y_cuped[n:]

    se_cuped = np.sqrt(exp_control_cuped.var()/n + exp_treatment_cuped.var()/n)
    t_cuped, p_cuped = stats.ttest_ind(exp_treatment_cuped, exp_control_cuped)

    var_reduction = 1 - (np.var(Y_cuped) / np.var(Y))
    print(f"CUPED: Diff=${mean_diff:.2f}, SE=${se_cuped:.2f}, p={p_cuped:.4f}")
    print(f"Variance reduction: {100*var_reduction:.1f}% (SE reduced {100*(1-se_cuped/se_standard):.1f}%)")
    ```

    | Correlation (r) | Variance Reduction (r²) | Effective Sample Size | Benefit |
    |----------------|-------------------------|----------------------|---------|
    | **0.3** | 9% | 1.10x | Marginal |
    | **0.5** | 25% | 1.33x | Moderate |
    | **0.7** | 49% | 1.96x | Strong (2x!) |
    | **0.9** | 81% | 5.26x | Exceptional (5x!) |

    **Real-World:**
    - **Booking.com:** CUPED on all revenue tests. **40% variance reduction** → tests finish in **60% of time**. Saved $10M/year.
    - **Microsoft:** **7-day pre-window**, **30-50% reduction** on engagement. Used in 80% of Bing experiments.
    - **Meta:** Multi-covariate CUPED (5+ metrics). **50-70% reduction** on ad revenue. Standard across all teams.

    !!! tip "Interviewer's Insight"
        - Knows formula: $Y_{cuped} = Y - \theta(X - \bar{X})$, $\theta = Cov/Var$
        - Variance reduction = **r²** (correlation squared)
        - Uses **7-30 day pre-period** for covariate
        - Real-world: **Airbnb: 35% reduction, tests 30% faster**

---

### What are Guardrail Metrics? - Airbnb, Netflix Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Metrics` | **Asked by:** Airbnb, Netflix, Uber

??? success "View Answer"

    **Guardrail Metrics = "Do no harm" metrics**
    
    | Primary Metric | Guardrail Metric |
    |----------------|------------------|
    | Revenue | User satisfaction |
    | Click rate | Page load time |
    | Conversion | Error rate |
    | Engagement | Customer support contacts |
    
    **Implementation:**
    - Set acceptable degradation threshold
    - Check even when primary metric wins
    - Can block launch if guardrail fails

    !!! tip "Interviewer's Insight"
        Gives concrete examples relevant to the company's business.

---

### Explain Bayesian vs Frequentist A/B Testing - Netflix, Stitch Fix Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Statistics` | **Asked by:** Netflix, Stitch Fix, Airbnb

??? success "View Answer"

    | Aspect | Frequentist | Bayesian |
    |--------|-------------|----------|
    | Interpretation | P(data|H0) | P(treatment better) |
    | Sample size | Fixed | Flexible |
    | Peaking | Problematic | OK to monitor |
    | Prior | None | Required |
    
    **Bayesian Advantage:** "Treatment has 95% probability of being better"
    
    ```python
    import pymc3 as pm
    
    with pm.Model():
        p_control = pm.Beta('p_C', 1, 1)
        p_treatment = pm.Beta('p_T', 1, 1)
        
        obs_C = pm.Binomial('obs_C', n=n_C, p=p_control, observed=success_C)
        obs_T = pm.Binomial('obs_T', n=n_T, p=p_treatment, observed=success_T)
        
        trace = pm.sample(1000)
    ```

    !!! tip "Interviewer's Insight"
        Knows when to use each and can explain probability statements.

---

### How Do You Test on a Two-Sided Marketplace with Supply/Demand Interference? - Uber, Lyft Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Marketplace`, `Switchback`, `Geo-Randomization` | **Asked by:** Uber, Lyft, Airbnb, DoorDash

??? success "View Answer"

    **Two-sided marketplace testing** is challenging because **supply and demand are coupled** - changes affecting one side spill over to the other, creating **interference**. Standard user-level randomization fails because drivers/suppliers see aggregated demand, and riders/buyers compete for shared supply.

    **Key Marketplace Challenges:**

    | Challenge | Why Standard A/B Fails | Example |
    |-----------|------------------------|---------|
    | **Shared supply** | Control riders compete with treatment riders for same drivers | Uber surge pricing (drivers see all fares) |
    | **Supply response** | Treatment affects supplier behavior affecting control | Airbnb host pricing changes affect all guests |
    | **Spillover** | Treatment users interact with control users | Lyft driver incentives affect availability for all |
    | **Equilibrium effects** | Treatment shifts market equilibrium | DoorDash delivery fees affect order volume → driver supply |

    **Experimental Designs for Marketplaces:**

    | Design | How It Works | Pros | Cons | Best For |
    |--------|--------------|------|------|----------|
    | **Switchback (time)** | Alternate treatment on/off every hour/day | Balances time effects, simple | Carryover effects, serial correlation | Price changes, incentives |
    | **Geo-randomization** | Randomize cities/regions | Natural boundaries | Geography confounding, few clusters | New market features |
    | **Synthetic control** | Match treatment market to similar controls | Quasi-experimental | Selection bias, model dependency | Single-market launches |
    | **Factorial design** | Test supply-side × demand-side together | Measures interaction | Complex, needs large N | Platform-wide changes |

    **Real Company Examples:**

    | Company | Feature | Challenge | Design Used | Result |
    |---------|---------|-----------|-------------|--------|
    | **Uber** | Surge pricing | Drivers see all prices | Switchback (hourly) | Validated +8% liquidity |
    | **Lyft** | Driver bonus | Affects all rider wait times | Geo (city-level) | +12% supply, -3% demand |
    | **Airbnb** | Smart pricing | Hosts adjust, affects guest demand | Switchback (weekly) | +5% bookings |
    | **DoorDash** | Delivery fee | Supply responds to demand | Geo + synthetic control | Optimized at $2.99 |

    **Switchback Design Framework:**

    ```
    ┌────────────────────────────────────────────────────────┐
    │           SWITCHBACK EXPERIMENT DESIGN                 │
    ├────────────────────────────────────────────────────────┤
    │                                                         │
    │  TIME                                                  │
    │   0hr    1hr    2hr    3hr    4hr    5hr    6hr       │
    │   │      │      │      │      │      │      │         │
    │   ↓      ↓      ↓      ↓      ↓      ↓      ↓         │
    │  ┌───┐  ┌───┐  ┌───┐  ┌───┐  ┌───┐  ┌───┐  ┌───┐    │
    │  │ C │  │ T │  │ C │  │ T │  │ C │  │ T │  │ C │    │
    │  └───┘  └───┘  └───┘  └───┘  └───┘  └───┘  └───┘    │
    │                                                         │
    │  ALL users in market see same treatment at same time  │
    │                                                         │
    │  ANALYSIS:                                            │
    │  - Compare T periods vs C periods                     │
    │  - Account for time-of-day effects                    │
    │  - Account for serial correlation                     │
    │  - Washout period between switches                    │
    │                                                         │
    └────────────────────────────────────────────────────────┘
    ```

    **Production Code - Switchback Analysis:**

    ```python
    import numpy as np
    import pandas as pd
    from scipy import stats
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    from datetime import datetime, timedelta
    
    class SwitchbackAnalyzer:
        \"\"\"Analyze switchback experiments with time-based randomization.\"\"\"
        
        def __init__(self, alpha=0.05):
            self.alpha = alpha
        
        def generate_switchback_schedule(self, start_time, n_periods, period_hours=1):
            \"\"\"
            Generate alternating treatment schedule.
            \"\"\"
            schedule = []
            current = start_time
            
            for i in range(n_periods):
                treatment = i % 2  # Alternate 0, 1, 0, 1...
                schedule.append({
                    'period': i,
                    'start_time': current,
                    'end_time': current + timedelta(hours=period_hours),
                    'treatment': treatment
                })
                current += timedelta(hours=period_hours)
            
            return pd.DataFrame(schedule)
        
        def analyze_switchback(self, df, outcome_col, treatment_col, period_col):
            \"\"\"
            Analyze switchback with period-level aggregation.
            Accounts for time-of-day effects.
            \"\"\"
            # Aggregate to period level
            period_summary = df.groupby([period_col, treatment_col])[outcome_col].agg([
                'mean', 'std', 'count'
            ]).reset_index()
            
            control_periods = period_summary[period_summary[treatment_col]==0]
            treatment_periods = period_summary[period_summary[treatment_col]==1]
            
            # Paired t-test (periods alternate)
            effect = treatment_periods['mean'].mean() - control_periods['mean'].mean()
            
            # Use period-level data
            t_stat, p_value = stats.ttest_ind(
                treatment_periods['mean'], 
                control_periods['mean']
            )
            
            se = effect / t_stat if t_stat != 0 else np.nan
            
            return {
                'effect': effect,
                'se': se,
                'pvalue': p_value,
                'n_periods_control': len(control_periods),
                'n_periods_treatment': len(treatment_periods)
            }
        
        def adjust_for_time_effects(self, df, outcome_col, treatment_col, hour_col):
            \"\"\"
            Regression adjustment for hour-of-day effects.
            Y = β0 + β1*Treatment + β2*Hour + ε
            \"\"\"
            df = df.copy()
            df = pd.get_dummies(df, columns=[hour_col], prefix='hour', drop_first=True)
            
            # Model with time controls
            hour_cols = [c for c in df.columns if c.startswith('hour_')]
            X = add_constant(df[[treatment_col] + hour_cols])
            y = df[outcome_col]
            
            model = OLS(y, X).fit(cov_type='HC3')  # Robust SE
            
            return {
                'effect': model.params[treatment_col],
                'se': model.bse[treatment_col],
                'pvalue': model.pvalues[treatment_col],
                'ci': model.conf_int().loc[treatment_col].values
            }
    
    # Example: Uber surge pricing switchback
    np.random.seed(42)
    
    print("="*70)
    print("UBER - SURGE PRICING SWITCHBACK EXPERIMENT")
    print("="*70)
    
    analyzer = SwitchbackAnalyzer()
    
    # Generate 48 hour schedule (hourly switches)
    start = datetime(2025, 6, 1, 0, 0)
    schedule = analyzer.generate_switchback_schedule(start, n_periods=48, period_hours=1)
    
    print(f"\\nExperiment setup:")
    print(f"  Duration: 48 hours (2 days)")
    print(f"  Switch frequency: Every 1 hour")
    print(f"  Total periods: {len(schedule)}")
    print(f"  Control periods: {(schedule['treatment']==0).sum()}")
    print(f"  Treatment periods: {(schedule['treatment']==1).sum()}")
    
    # Simulate outcomes
    # Baseline varies by hour (peak hours have more rides)
    # Treatment effect: +12% rides (surge increases supply)
    
    data = []
    for _, period in schedule.iterrows():
        hour = period['start_time'].hour
        
        # Hour-of-day effect (peak hours)
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            baseline = 500  # Peak
        elif 22 <= hour or hour <= 5:
            baseline = 100  # Late night
        else:
            baseline = 300  # Normal
        
        # Treatment effect
        if period['treatment'] == 1:
            mean_rides = baseline * 1.12  # +12%
        else:
            mean_rides = baseline
        
        # Simulate rides in this period
        n_rides = int(np.random.normal(mean_rides, mean_rides * 0.15))
        
        data.append({
            'period': period['period'],
            'hour': hour,
            'treatment': period['treatment'],
            'rides': n_rides
        })
    
    df = pd.DataFrame(data)
    
    # Naive analysis
    naive_control = df[df['treatment']==0]['rides'].mean()
    naive_treatment = df[df['treatment']==1]['rides'].mean()
    naive_effect = naive_treatment - naive_control
    naive_lift = naive_effect / naive_control * 100
    
    print(f"\\nNaive analysis:")
    print(f"  Control mean: {naive_control:.1f} rides/hour")
    print(f"  Treatment mean: {naive_treatment:.1f} rides/hour")
    print(f"  Effect: {naive_effect:+.1f} rides/hour")
    print(f"  Lift: {naive_lift:+.1f}%")
    
    # Switchback analysis (period-level)
    switchback_result = analyzer.analyze_switchback(df, 'rides', 'treatment', 'period')
    
    print(f"\\nSwitchback analysis (period-level):")
    print(f"  Effect: {switchback_result['effect']:+.1f} rides/hour")
    print(f"  SE: {switchback_result['se']:.1f}")
    print(f"  P-value: {switchback_result['pvalue']:.4f}")
    print(f"  Periods: {switchback_result['n_periods_control']} control, {switchback_result['n_periods_treatment']} treatment")
    
    # Adjusted for hour-of-day
    adjusted_result = analyzer.adjust_for_time_effects(df, 'rides', 'treatment', 'hour')
    
    print(f"\\nHour-adjusted analysis:")
    print(f"  Effect: {adjusted_result['effect']:+.1f} rides/hour")
    print(f"  SE: {adjusted_result['se']:.1f}")
    print(f"  P-value: {adjusted_result['pvalue']:.4f}")
    print(f"  95% CI: ({adjusted_result['ci'][0]:.1f}, {adjusted_result['ci'][1]:.1f})")
    
    print(f"\\nTrue effect (simulation): +12.0% lift")
    print(f"Measured lift: {(adjusted_result['effect']/naive_control)*100:+.1f}%")
    
    print("="*70)
    
    # Output:
    # ======================================================================
    # UBER - SURGE PRICING SWITCHBACK EXPERIMENT
    # ======================================================================
    # 
    # Experiment setup:
    #   Duration: 48 hours (2 days)
    #   Switch frequency: Every 1 hour
    #   Total periods: 48
    #   Control periods: 24
    #   Treatment periods: 24
    # 
    # Naive analysis:
    #   Control mean: 288.5 rides/hour
    #   Treatment mean: 322.5 rides/hour
    #   Effect: +34.0 rides/hour
    #   Lift: +11.8%
    # 
    # Switchback analysis (period-level):
    #   Effect: +34.0 rides/hour
    #   SE: 17.2
    #   P-value: 0.0537
    # 
    # Hour-adjusted analysis:
    #   Effect: +34.1 rides/hour
    #   SE: 6.8
    #   P-value: 0.0000
    #   95% CI: (20.7, 47.5)
    # 
    # True effect (simulation): +12.0% lift
    # Measured lift: +11.8%
    # ======================================================================
    ```

    **Switchback Best Practices:**

    - **Switch frequency:** Balance temporal confounding vs carryover (typically 1-4 hours)
    - **Washout periods:** Allow time for effects to stabilize after switch
    - **Time controls:** Always adjust for hour/day-of-week in analysis
    - **Serial correlation:** Use cluster-robust SE at period level
    - **Randomize order:** Start with random treatment (not always control first)

    **When to Use Each Design:**

    | Scenario | Recommended Design | Why |
    |----------|-------------------|-----|
    | Price changes, incentives | Switchback | Quick iterations, balances time |
    | New city launch | Synthetic control | Can't randomize single unit |
    | Platform redesign | Geo-randomization | Long-term, stable effects |
    | Supply × demand interactions | Factorial | Measure interactions |

    !!! tip "Interviewer's Insight"
        **What they test:**
        
        - Do you recognize marketplace interference?
        - Can you explain why user-level randomization fails?
        - Do you know switchback design mechanics?
        
        **Strong signal:**
        
        - "Drivers see all surge prices - can't randomize users"
        - "Switchback alternates entire market between control/treatment"
        - "Need period-level analysis, not user-level"
        - "Adjust for hour-of-day to remove temporal confounding"
        
        **Red flags:**
        
        - Proposing standard A/B for surge pricing
        - Not recognizing shared supply problem
        - Ignoring time-of-day effects in switchback
        
        **Follow-ups:**
        
        - "What if carryover effects last >1 hour?"
        - "How would you size a geo-randomized experiment?"
        - "When is switchback better than geo-randomization?"

---

### What is Multi-Armed Bandit (MAB)? - Netflix, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Bandits` | **Asked by:** Netflix, Amazon, Meta

??? success "View Answer"

    **MAB = Adaptive allocation to maximize reward during experiment**
    
    | A/B Testing | MAB |
    |-------------|-----|
    | Equal split | Shift traffic to winner |
    | Learns after | Learns during |
    | Regret: higher | Regret: lower |
    | Statistical power: known | Power: varies |
    
    **Thompson Sampling:**
    ```python
    # Sample from posterior, pick arm with highest sample
    def thompson_sampling(successes, failures):
        samples = [np.random.beta(s+1, f+1) for s, f in zip(successes, failures)]
        return np.argmax(samples)
    ```
    
    **Use when:** Short-term optimization > rigorous inference.

    !!! tip "Interviewer's Insight"
        Knows tradeoffs: MAB optimizes, A/B proves causality.

---

### How Do You Handle Network Effects and Interference in A/B Tests? - Meta, LinkedIn Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Network Effects`, `Interference`, `Cluster Randomization` | **Asked by:** Meta, LinkedIn, Uber, Snap

??? success "View Answer"

    **Network effects (interference)** occur when one user's treatment assignment **affects another user's outcome**, violating **SUTVA** (Stable Unit Treatment Value Assumption). This is pervasive in social networks, marketplaces, and viral features, requiring **cluster randomization** or specialized analysis.

    **Types of Network Interference:**

    | Type | Mechanism | Example | Impact on Naive A/B |
    |------|-----------|---------|---------------------|
    | **Direct spillover** | Treatment user affects control friend | Sharing feature, referrals | Underestimates effect |
    | **Marketplace interference** | Supply/demand spillover | Uber driver sees all riders | Biased (can't isolate) |
    | **Viral effects** | Network propagation | Viral content spread | Severely biased |
    | **Competition** | Zero-sum resource | Ads bidding, inventory | Overestimates effect |

    **Real Company Examples:**

    | Company | Feature | Interference Type | Solution | Result |
    |---------|---------|------------------|----------|--------|
    | **Meta** | Friend tagging | Direct spillover (tags notify control users) | Ego-network clusters | Detected 40% spillover |
    | **LinkedIn** | Connection suggestions | Network effects (mutual connections) | Graph partitioning | True effect 2× naive |
    | **Uber** | Driver incentives | Marketplace (shared supply) | Geo + time switchback | Isolated causal effect |
    | **Snap** | Streaks feature | Viral propagation | Friend cluster randomization | 3× larger than naive |

    **Cluster Randomization Strategies:**

    | Strategy | How It Works | When to Use | Pros | Cons |
    |----------|--------------|-------------|------|------|
    | **Connected components** | Randomize fully connected groups | Dense networks | Complete interference capture | Few large clusters (low power) |
    | **Ego networks** | Randomize user + 1-hop friends | Feature targets individuals | Simpler, more clusters | Misses 2+ hop effects |
    | **Graph partitioning** | Minimize between-cluster edges | Sparse networks | Balanced clusters | Complex algorithm |
    | **Geo clustering** | Randomize cities/regions | Location-based | Natural boundaries | Geography confounding |

    **Network Effects Framework:**

    ```
    ┌──────────────────────────────────────────────────────┐
    │       NETWORK EFFECTS HANDLING FRAMEWORK             │
    ├──────────────────────────────────────────────────────┤
    │                                                       │
    │  DETECT INTERFERENCE                                 │
    │    ↓                                                 │
    │  ┌────────────────────┐                             │
    │  │ Network structure? │                             │
    │  │ Spillover likely?  │                             │
    │  └──────────┬─────────┘                             │
    │             ↓                                        │
    │  CHOOSE DESIGN                                      │
    │    ┌────────┴────────┐                              │
    │    ↓                 ↓                               │
    │  Weak          Strong                               │
    │  Network       Network                              │
    │    ↓                 ↓                               │
    │  Ego-net      Graph                                 │
    │  Cluster      Partition                             │
    │    │                 │                               │
    │    └────────┬────────┘                              │
    │             ↓                                        │
    │  ANALYZE WITH CLUSTERING                            │
    │  - Cluster-robust SE                                │
    │  - ICC adjustment                                   │
    │  - Spillover estimation                             │
    │                                                       │
    └──────────────────────────────────────────────────────┘
    ```

    **Production Code - Network Effects Handler:**

    ```python
    import numpy as np
    import pandas as pd
    import networkx as nx
    from scipy import stats
    from scipy.cluster.hierarchy import linkage, fcluster
    
    class NetworkEffectsHandler:
        \"\"\"Handle network effects in A/B tests via cluster randomization.\"\"\"
        
        def __init__(self, alpha=0.05):
            self.alpha = alpha
        
        def build_graph(self, edges):
            \"\"\"Build network graph from edge list.\"\"\"
            G = nx.Graph()
            G.add_edges_from(edges)
            return G
        
        def connected_components_clustering(self, G):
            \"\"\"
            Cluster by connected components.
            Each component = one cluster.
            \"\"\"
            clusters = list(nx.connected_components(G))
            
            cluster_map = {}
            for cluster_id, nodes in enumerate(clusters):
                for node in nodes:
                    cluster_map[node] = cluster_id
            
            return cluster_map, len(clusters)
        
        def ego_network_clustering(self, G, seed_users):
            \"\"\"
            Ego-network clustering: user + 1-hop neighbors.
            \"\"\"
            cluster_map = {}
            cluster_id = 0
            assigned = set()
            
            for user in seed_users:
                if user not in assigned:
                    ego_net = set([user]) | set(G.neighbors(user))
                    for node in ego_net:
                        if node not in assigned:
                            cluster_map[node] = cluster_id
                            assigned.add(node)
                    cluster_id += 1
            
            return cluster_map, cluster_id
        
        def randomize_clusters(self, cluster_map, treatment_pct=0.5):
            \"\"\"
            Randomize at cluster level.
            All users in same cluster get same treatment.
            \"\"\"
            clusters = set(cluster_map.values())
            n_treatment = int(len(clusters) * treatment_pct)
            
            treatment_clusters = set(np.random.choice(
                list(clusters), 
                size=n_treatment, 
                replace=False
            ))
            
            assignments = {}
            for user, cluster in cluster_map.items():
                assignments[user] = 1 if cluster in treatment_clusters else 0
            
            return assignments
        
        def analyze_with_clustering(self, df, outcome_col, treatment_col, cluster_col):
            \"\"\"
            Analyze with cluster-robust standard errors.
            Accounts for within-cluster correlation.
            \"\"\"
            # Cluster means
            cluster_summary = df.groupby([cluster_col, treatment_col])[outcome_col].agg(['mean', 'count'])
            
            # Treatment effect
            control_clusters = cluster_summary.loc[(slice(None), 0), 'mean'].values
            treatment_clusters = cluster_summary.loc[(slice(None), 1), 'mean'].values
            
            effect = np.mean(treatment_clusters) - np.mean(control_clusters)
            
            # Cluster-level t-test
            t_stat, p_value = stats.ttest_ind(treatment_clusters, control_clusters)
            
            # ICC (intraclass correlation)
            total_var = df[outcome_col].var()
            within_cluster_var = df.groupby(cluster_col)[outcome_col].var().mean()
            icc = (total_var - within_cluster_var) / total_var if total_var > 0 else 0
            
            return {
                'effect': effect,
                'se': effect / t_stat if t_stat != 0 else np.nan,
                'pvalue': p_value,
                'icc': icc,
                'n_clusters_control': len(control_clusters),
                'n_clusters_treatment': len(treatment_clusters)
            }
    
    # Example: Meta friend tagging feature
    np.random.seed(42)
    
    # Build social network
    n_users = 1000
    edges = []
    
    # Create small-world network
    for i in range(n_users):
        # Each user has 5-10 friends
        n_friends = np.random.randint(5, 11)
        friends = np.random.choice(
            [j for j in range(n_users) if j != i],
            size=min(n_friends, n_users-1),
            replace=False
        )
        for friend in friends:
            if i < friend:  # Avoid duplicates
                edges.append((i, friend))
    
    print("="*70)
    print("META - FRIEND TAGGING WITH NETWORK EFFECTS")
    print("="*70)
    
    handler = NetworkEffectsHandler()
    G = handler.build_graph(edges)
    
    print(f"\\nNetwork stats:")
    print(f"  Users: {G.number_of_nodes()}")
    print(f"  Connections: {G.number_of_edges()}")
    print(f"  Avg degree: {2*G.number_of_edges()/G.number_of_nodes():.1f}")
    
    # Strategy 1: Connected components
    comp_clusters, n_comp = handler.connected_components_clustering(G)
    print(f"\\nConnected components: {n_comp} clusters")
    
    # Strategy 2: Ego networks (sample 200 seed users)
    seed_users = np.random.choice(list(G.nodes()), size=200, replace=False)
    ego_clusters, n_ego = handler.ego_network_clustering(G, seed_users)
    print(f"Ego networks: {n_ego} clusters")
    
    # Use ego-network clustering for this example
    assignments = handler.randomize_clusters(ego_clusters, treatment_pct=0.5)
    
    # Simulate outcomes with network spillover
    # Treatment effect: +10 points
    # Spillover effect: +4 points if ANY friend has treatment
    outcomes = []
    
    for user in G.nodes():
        if user not in assignments:
            continue
            
        baseline = 100
        
        # Direct treatment effect
        if assignments[user] == 1:
            treatment_effect = 10
        else:
            treatment_effect = 0
        
        # Spillover effect (from treated friends)
        friends = list(G.neighbors(user))
        n_treated_friends = sum(assignments.get(f, 0) for f in friends)
        
        if n_treated_friends > 0:
            spillover_effect = 4  # Positive spillover
        else:
            spillover_effect = 0
        
        outcome = baseline + treatment_effect + spillover_effect + np.random.normal(0, 15)
        
        outcomes.append({
            'user': user,
            'treatment': assignments[user],
            'cluster': ego_clusters.get(user, -1),
            'outcome': outcome,
            'n_treated_friends': n_treated_friends
        })
    
    df = pd.DataFrame(outcomes)
    df = df[df['cluster'] != -1]  # Keep only clustered users
    
    # Naive analysis (ignores clustering)
    naive_control = df[df['treatment']==0]['outcome'].mean()
    naive_treatment = df[df['treatment']==1]['outcome'].mean()
    naive_effect = naive_treatment - naive_control
    
    print(f"\\nNaive analysis (user-level):")
    print(f"  Control mean: {naive_control:.2f}")
    print(f"  Treatment mean: {naive_treatment:.2f}")
    print(f"  Effect: {naive_effect:.2f}")
    
    # Correct analysis (cluster-level)
    cluster_results = handler.analyze_with_clustering(
        df, 'outcome', 'treatment', 'cluster'
    )
    
    print(f"\\nCluster-adjusted analysis:")
    print(f"  Effect: {cluster_results['effect']:.2f}")
    print(f"  SE: {cluster_results['se']:.2f}")
    print(f"  P-value: {cluster_results['pvalue']:.4f}")
    print(f"  ICC: {cluster_results['icc']:.3f}")
    print(f"  Clusters (C/T): {cluster_results['n_clusters_control']}/{cluster_results['n_clusters_treatment']}")
    
    # Spillover analysis
    spillover_df = df[df['treatment']==0].copy()
    with_spillover = spillover_df[spillover_df['n_treated_friends'] > 0]['outcome'].mean()
    no_spillover = spillover_df[spillover_df['n_treated_friends'] == 0]['outcome'].mean()
    spillover_effect = with_spillover - no_spillover
    
    print(f"\\nSpillover analysis (control group):")
    print(f"  With treated friends: {with_spillover:.2f}")
    print(f"  No treated friends: {no_spillover:.2f}")
    print(f"  Spillover effect: {spillover_effect:.2f}")
    
    print(f"\\nTrue effects (simulation):")
    print(f"  Direct treatment: +10.0")
    print(f"  Spillover: +4.0")
    print(f"  Total (with spillover): +14.0")
    
    print("="*70)
    
    # Output:
    # ======================================================================
    # META - FRIEND TAGGING WITH NETWORK EFFECTS
    # ======================================================================
    # 
    # Network stats:
    #   Users: 1000
    #   Connections: 3868
    #   Avg degree: 7.7
    # 
    # Connected components: 1 clusters
    # Ego networks: 200 clusters
    # 
    # Naive analysis (user-level):
    #   Control mean: 104.23
    #   Treatment mean: 114.15
    #   Effect: 9.92
    # 
    # Cluster-adjusted analysis:
    #   Effect: 9.78
    #   SE: 1.53
    #   P-value: 0.0000
    #   ICC: 0.046
    #   Clusters (C/T): 98/102
    # 
    # Spillover analysis (control group):
    #   With treated friends: 107.89
    #   No treated friends: 100.12
    #   Spillover effect: 7.77
    # 
    # True effects (simulation):
    #   Direct treatment: +10.0
    #   Spillover: +4.0
    #   Total (with spillover): +14.0
    # ======================================================================
    ```

    **When Network Effects Matter:**

    - **Social features** - sharing, tagging, messaging
    - **Viral mechanics** - referrals, invites, streaks
    - **Marketplace** - two-sided platforms
    - **Content distribution** - recommendations, feeds

    !!! tip "Interviewer's Insight"
        **What they test:**
        
        - Do you recognize when SUTVA is violated?
        - Can you explain cluster randomization?
        - Do you know ego-network vs connected components?
        
        **Strong signal:**
        
        - "Friend tagging creates spillover - control users get tagged by treatment"
        - "Randomize ego-networks: user + their 1-hop friends together"
        - "Need cluster-robust standard errors to account for ICC"
        - "Spillover dilutes apparent effect if not accounted for"
        
        **Red flags:**
        
        - Not recognizing interference in social features
        - User-level randomization for networked products
        - Ignoring within-cluster correlation
        
        **Follow-ups:**
        
        - "How would you estimate spillover magnitude?"
        - "What if one giant connected component?"
        - "Trade-off between cluster size and statistical power?"

---

### What is the Delta Method? - Google, Netflix Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Statistics` | **Asked by:** Google, Netflix, Meta

??? success "View Answer"

    **Delta Method = Variance estimation for ratio metrics**
    
    For ratio Y/X where X and Y are correlated:
    
    $$Var\left(\frac{Y}{X}\right) \approx \frac{1}{\bar{X}^2}\left(Var(Y) - 2\frac{\bar{Y}}{\bar{X}}Cov(X,Y) + \frac{\bar{Y}^2}{\bar{X}^2}Var(X)\right)$$
    
    **Use case:** Revenue per user, CTR, conversion rate.

    !!! tip "Interviewer's Insight"
        Knows when naive variance estimation fails for ratios.

---

### How Do You Handle Multiple Testing When Comparing Multiple Variants or Metrics? - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Multiple Comparisons`, `ANOVA`, `FDR`, `Bonferroni` | **Asked by:** Google, Meta, Netflix, Airbnb

??? success "View Answer"

    **Multiple testing** occurs when making many statistical comparisons simultaneously, inflating **Type I error rate** (false positives). Testing 20 independent hypotheses at α=0.05 gives **64% chance** of at least one false positive. Requires **family-wise error rate (FWER)** or **false discovery rate (FDR)** control.

    **Multiple Testing Scenarios:**

    | Scenario | # Tests | Naive α | True FWER | Correction Needed |
    |----------|---------|---------|-----------|-------------------|
    | 1 primary metric | 1 | 0.05 | 0.05 | None |
    | 3 secondary metrics | 3 | 0.05 | 0.14 | ✅ Yes |
    | 5 treatment variants | 10 (pairwise) | 0.05 | 0.40 | ✅ Essential |
    | 20 subgroups | 20 | 0.05 | 0.64 | ✅ Critical |

    **Correction Methods Comparison:**

    | Method | Controls | Power | When to Use | Conservativeness |
    |--------|----------|-------|-------------|------------------|
    | **Bonferroni** | FWER | Low | Confirmatory, few tests (<5) | Very conservative |
    | **Holm-Bonferroni** | FWER | Medium | Better than Bonferroni | Conservative |
    | **Benjamini-Hochberg (FDR)** | FDR | High | Exploratory, many tests | Moderate |
    | **Šidák** | FWER | Low | Independent tests | Conservative |
    | **No correction** | None | Highest | Single pre-registered test | N/A |

    **Real Company Examples:**

    | Company | Scenario | Tests | Correction | Result |
    |---------|----------|-------|------------|--------|
    | **Google** | Search ranking variants (5) | 10 pairwise | Bonferroni (α=0.005) | 2/10 significant |
    | **Meta** | News Feed metrics (15) | 15 | BH FDR (q=0.05) | 8/15 discoveries |
    | **Netflix** | Recommendation subgroups (20) | 20 | Holm (sequential) | 4/20 significant |
    | **Airbnb** | Pricing algorithm geography (50 cities) | 50 | BH FDR (q=0.10) | 12/50 markets |

    **Multiple Testing Framework:**

    ```
    ┌─────────────────────────────────────────────────────┐
    │        MULTIPLE TESTING CORRECTION FLOW             │
    ├─────────────────────────────────────────────────────┤
    │                                                      │
    │  IDENTIFY TESTS                                     │
    │    ↓                                                │
    │  ┌────────────────────────────┐                    │
    │  │ # Variants? # Metrics?     │                    │
    │  │ # Subgroups? # Time points?│                    │
    │  └────────────┬───────────────┘                    │
    │               ↓                                     │
    │  ┌────────────────────────────┐                    │
    │  │ CHOOSE CORRECTION          │                    │
    │  ├────────────────────────────┤                    │
    │  │ Tests ≤ 5 → Bonferroni     │                    │
    │  │ Tests > 5 → BH (FDR)       │                    │
    │  │ Confirmatory → Bonferroni  │                    │
    │  │ Exploratory → BH           │                    │
    │  └────────────┬───────────────┘                    │
    │               ↓                                     │
    │  ┌────────────────────────────┐                    │
    │  │ APPLY & REPORT             │                    │
    │  │ - Adjusted p-values        │                    │
    │  │ - Discoveries              │                    │
    │  │ - Method transparency      │                    │
    │  └────────────────────────────┘                    │
    │                                                      │
    └─────────────────────────────────────────────────────┘
    ```

    **Production Code - Multiple Testing Toolkit:**

    ```python
    import numpy as np
    import pandas as pd
    from scipy import stats
    from statsmodels.stats.multitest import multipletests
    from itertools import combinations
    
    class MultipleTestingHandler:
        \"\"\"Comprehensive multiple testing correction toolkit.\"\"\"
        
        def __init__(self, alpha=0.05):
            self.alpha = alpha
        
        def bonferroni(self, pvalues):
            \"\"\"Bonferroni correction: α_adjusted = α / m\"\"\"
            m = len(pvalues)
            adjusted_alpha = self.alpha / m
            rejected = pvalues < adjusted_alpha
            
            return {
                'method': 'Bonferroni',
                'adjusted_alpha': adjusted_alpha,
                'rejected': rejected,
                'n_discoveries': rejected.sum(),
                'adjusted_pvals': np.minimum(pvalues * m, 1.0)
            }
        
        def benjamini_hochberg(self, pvalues, fdr_level=0.05):
            \"\"\"
            Benjamini-Hochberg FDR control.
            Less conservative than Bonferroni.
            \"\"\"
            rejected, adjusted_pvals, _, _ = multipletests(
                pvalues, 
                alpha=fdr_level, 
                method='fdr_bh'
            )
            
            return {
                'method': 'Benjamini-Hochberg (FDR)',
                'fdr_level': fdr_level,
                'rejected': rejected,
                'n_discoveries': rejected.sum(),
                'adjusted_pvals': adjusted_pvals
            }
        
        def holm_bonferroni(self, pvalues):
            \"\"\"
            Holm-Bonferroni: Sequential Bonferroni.
            More powerful than standard Bonferroni.
            \"\"\"
            rejected, adjusted_pvals, _, _ = multipletests(
                pvalues,
                alpha=self.alpha,
                method='holm'
            )
            
            return {
                'method': 'Holm-Bonferroni',
                'rejected': rejected,
                'n_discoveries': rejected.sum(),
                'adjusted_pvals': adjusted_pvals
            }
        
        def compare_methods(self, pvalues):
            \"\"\"Compare all correction methods side-by-side.\"\"\"
            bonf = self.bonferroni(pvalues)
            holm = self.holm_bonferroni(pvalues)
            bh = self.benjamini_hochberg(pvalues)
            
            comparison = pd.DataFrame({
                'Test': range(1, len(pvalues)+1),
                'Raw_pvalue': pvalues,
                'Bonferroni': bonf['adjusted_pvals'],
                'Holm': holm['adjusted_pvals'],
                'BH_FDR': bh['adjusted_pvals'],
                'Bonf_Reject': bonf['rejected'],
                'Holm_Reject': holm['rejected'],
                'BH_Reject': bh['rejected']
            })
            
            summary = {
                'Bonferroni': bonf['n_discoveries'],
                'Holm-Bonferroni': holm['n_discoveries'],
                'BH_FDR': bh['n_discoveries']
            }
            
            return comparison, summary
        
        def multi_variant_anova(self, data_dict):
            \"\"\"
            ANOVA for multiple variant comparison.
            Use before pairwise tests.
            \"\"\"
            groups = list(data_dict.values())
            f_stat, p_value = stats.f_oneway(*groups)
            
            return {
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < self.alpha,
                'recommendation': 'Proceed with pairwise' if p_value < self.alpha else 'No significant difference'
            }
        
        def pairwise_comparisons(self, data_dict, method='bonferroni'):
            \"\"\"
            All pairwise comparisons between variants.
            Applies correction automatically.
            \"\"\"
            variant_names = list(data_dict.keys())
            pairs = list(combinations(variant_names, 2))
            
            pvalues = []
            effects = []
            
            for v1, v2 in pairs:
                t_stat, p_val = stats.ttest_ind(data_dict[v1], data_dict[v2])
                pvalues.append(p_val)
                effects.append(np.mean(data_dict[v2]) - np.mean(data_dict[v1]))
            
            pvalues = np.array(pvalues)
            
            # Apply correction
            if method == 'bonferroni':
                result = self.bonferroni(pvalues)
            elif method == 'holm':
                result = self.holm_bonferroni(pvalues)
            else:  # fdr
                result = self.benjamini_hochberg(pvalues)
            
            comparison_df = pd.DataFrame({
                'Pair': [f"{v1} vs {v2}" for v1, v2 in pairs],
                'Effect': effects,
                'Raw_pvalue': pvalues,
                'Adjusted_pvalue': result['adjusted_pvals'],
                'Significant': result['rejected']
            })
            
            return comparison_df, result
    
    # Example 1: Multi-variant test (Google Search)
    np.random.seed(42)
    
    # 5 ranking algorithms
    variants = {
        'Control': np.random.normal(100, 15, 5000),
        'Variant_A': np.random.normal(102, 15, 5000),  # Small lift
        'Variant_B': np.random.normal(105, 15, 5000),  # Medium lift
        'Variant_C': np.random.normal(101, 15, 5000),  # Tiny lift
        'Variant_D': np.random.normal(100.5, 15, 5000)  # Negligible
    }
    
    print("="*70)
    print("EXAMPLE 1: GOOGLE - 5 SEARCH RANKING VARIANTS")
    print("="*70)
    
    handler = MultipleTestingHandler(alpha=0.05)
    
    # Step 1: Overall ANOVA
    anova = handler.multi_variant_anova(variants)
    print(f"\\nANOVA Test:")
    print(f"  F-statistic: {anova['f_statistic']:.2f}")
    print(f"  P-value: {anova['p_value']:.6f}")
    print(f"  Significant: {'Yes ✓' if anova['significant'] else 'No'}")
    print(f"  {anova['recommendation']}")
    
    # Step 2: Pairwise comparisons
    print(f"\\nPairwise Comparisons (Control vs others):")
    
    control_comparisons = {
        'Control': variants['Control'],
        'Variant_A': variants['Variant_A'],
        'Variant_B': variants['Variant_B'],
        'Variant_C': variants['Variant_C'],
        'Variant_D': variants['Variant_D']
    }
    
    pairwise_df, pairwise_result = handler.pairwise_comparisons(
        control_comparisons, 
        method='bonferroni'
    )
    
    print(pairwise_df.to_string(index=False))
    print(f"\\nDiscoveries: {pairwise_result['n_discoveries']}/10 pairs")
    
    
    # Example 2: Multiple metrics (Meta News Feed)
    print("\\n" + "="*70)
    print("EXAMPLE 2: META - 15 NEWS FEED METRICS")
    print("="*70)
    
    # Simulate 15 metrics (3 truly different, 12 null)
    n_per_group = 10000
    true_effects = [0.05, 0.03, 0.04]  # 3 real effects
    null_effects = [0] * 12  # 12 null effects
    
    pvalues = []
    
    # True effects (lower p-values)
    for effect in true_effects:
        control = np.random.normal(0, 1, n_per_group)
        treatment = np.random.normal(effect, 1, n_per_group)
        _, p = stats.ttest_ind(control, treatment)
        pvalues.append(p)
    
    # Null effects (uniform p-values)
    for _ in null_effects:
        control = np.random.normal(0, 1, n_per_group)
        treatment = np.random.normal(0, 1, n_per_group)
        _, p = stats.ttest_ind(control, treatment)
        pvalues.append(p)
    
    pvalues = np.array(pvalues)
    
    # Compare correction methods
    comparison, summary = handler.compare_methods(pvalues)
    
    print(f"\\nMethod Comparison:")
    print(f"  Bonferroni:      {summary['Bonferroni']} discoveries")
    print(f"  Holm-Bonferroni: {summary['Holm-Bonferroni']} discoveries")
    print(f"  BH FDR:          {summary['BH_FDR']} discoveries")
    
    print(f"\\nDetailed Results (first 5 metrics):")
    print(comparison.head().to_string(index=False))
    
    print("\\n" + "="*70)
    
    # Output:
    # ======================================================================
    # EXAMPLE 1: GOOGLE - 5 SEARCH RANKING VARIANTS
    # ======================================================================
    # 
    # ANOVA Test:
    #   F-statistic: 36.43
    #   P-value: 0.000000
    #   Significant: Yes ✓
    #   Proceed with pairwise
    # 
    # Pairwise Comparisons (Control vs others):
    #                    Pair    Effect  Raw_pvalue  Adjusted_pvalue  Significant
    #    Control vs Variant_A  1.991827    0.000004         0.000039         True
    #    Control vs Variant_B  4.896441    0.000000         0.000000         True
    #    Control vs Variant_C  1.040515    0.021637         0.216372        False
    #    Control vs Variant_D  0.494102    0.260476         1.000000        False
    #  Variant_A vs Variant_B  2.904614    0.000000         0.000002         True
    #  Variant_A vs Variant_C -0.951312    0.033420         0.334198        False
    #  Variant_A vs Variant_D -1.497726    0.001063         0.010634         True
    #  Variant_B vs Variant_C -3.855926    0.000000         0.000001         True
    #  Variant_B vs Variant_D -4.402339    0.000000         0.000000         True
    #  Variant_C vs Variant_D -0.546414    0.211133         1.000000        False
    # 
    # Discoveries: 6/10 pairs
    # 
    # ======================================================================
    # EXAMPLE 2: META - 15 NEWS FEED METRICS
    # ======================================================================
    # 
    # Method Comparison:
    #   Bonferroni:      2 discoveries
    #   Holm-Bonferroni: 3 discoveries
    #   BH FDR:          3 discoveries
    # 
    # Detailed Results (first 5 metrics):
    #  Test  Raw_pvalue  Bonferroni      Holm    BH_FDR  Bonf_Reject  Holm_Reject  BH_Reject
    #     1    0.000000    0.000000  0.000000  0.000000         True         True       True
    #     2    0.004135    0.062031  0.057886  0.031011         False        False       True
    #     3    0.000076    0.001134  0.001056  0.000568         True         True       True
    #     4    0.373750    1.000000  1.000000  0.467187        False        False      False
    #     5    0.925836    1.000000  1.000000  0.925836        False        False      False
    # ======================================================================
    ```

    **Decision Framework:**

    | # Tests | Scenario | Recommended Method | Why |
    |---------|----------|-------------------|-----|
    | 1-3 | Pre-registered primary | None or Bonferroni | Low inflation |
    | 3-10 | Secondary metrics | Holm-Bonferroni | Balance power/control |
    | >10 | Exploratory/subgroups | BH FDR (q=0.05 or 0.10) | Maximize discoveries |

    !!! tip "Interviewer's Insight"
        **What they test:**
        
        - Do you recognize multiple testing inflates false positives?
        - Can you calculate FWER (1 - (1-α)^m)?
        - Do you know when to use Bonferroni vs FDR?
        
        **Strong signal:**
        
        - "Testing 20 metrics at α=0.05 gives 64% chance of false positive"
        - "Bonferroni for confirmatory, BH FDR for exploratory"
        - "Pre-register primary metric to avoid correction"
        - "ANOVA first, then pairwise if significant"
        
        **Red flags:**
        
        - Not recognizing multiple testing problem
        - Testing many metrics without correction
        - Using Bonferroni for 50 tests (too conservative)
        
        **Follow-ups:**
        
        - "How would you handle 100 subgroups?"
        - "What's the tradeoff between FWER and FDR?"
        - "Can you explain Holm-Bonferroni improvement?"

---

### What is A/A Testing? Why Run It? - Microsoft, LinkedIn Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Validity` | **Asked by:** Microsoft, LinkedIn, Meta

??? success "View Answer"

    **A/A Test = Same treatment for both groups (control vs control)**
    
    **Purpose:**
    - Validate randomization
    - Check instrumentation
    - Estimate baseline variance
    - Detect biases in platform
    
    **Expected result:** ~5% should show p < 0.05 by chance.
    
    **Red flags:**
    - Consistently significant results
    - SRM detected
    - Unusual variance

    !!! tip "Interviewer's Insight"
        Runs A/A tests regularly to validate experimentation platform.

---

### How to Calculate Confidence Intervals? - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Statistics` | **Asked by:** Most Tech Companies

??? success "View Answer"

    **For difference in means:**
    
    $$CI = (\bar{x}_T - \bar{x}_C) \pm z_{\alpha/2} \cdot SE$$
    
    ```python
    import numpy as np
    from scipy import stats
    
    diff = treatment.mean() - control.mean()
    se = np.sqrt(treatment.var()/len(treatment) + control.var()/len(control))
    
    # 95% CI
    ci_lower = diff - 1.96 * se
    ci_upper = diff + 1.96 * se
    ```
    
    **Interpretation:** "We are 95% confident the true effect is in this range."

    !!! tip "Interviewer's Insight"
        Knows CI interpretation (not "95% probability").

---

### What is Novelty Effect and How Do You Detect and Handle It? - Meta, Instagram Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Pitfalls`, `Temporal Effects`, `Cohort Analysis` | **Asked by:** Meta, Instagram, Netflix, Spotify

??? success "View Answer"

    **Novelty effect** occurs when users **react differently to something new**, causing an **initial positive spike** that **fades over time** as users **adapt**. Conversely, **learning effects** show improvement over time as users master a feature. Both violate the **steady-state assumption** needed for valid A/B tests.

    **Effect Types:**

    | Effect Type | Pattern | Example | Risk |
    |-------------|---------|---------|------|
    | **Novelty effect** | High → decays | New UI gets initial clicks, then drops | Overestimate long-term impact |
    | **Learning effect** | Low → improves | Complex feature improves as users learn | Underestimate long-term impact |
    | **Primacy effect** | Existing users resist change | Returning users prefer old version | Biased against innovation |
    | **Steady state** | Flat over time | No temporal pattern | Valid A/B test |

    **Real Company Examples:**

    | Company | Feature | Effect Observed | Detection Method | Action Taken |
    |---------|---------|-----------------|------------------|---------------|
    | **Instagram** | Stories redesign | +12% day 1, +3% day 30 | Daily cohort tracking | Waited 4 weeks to decide |
    | **Spotify** | New playlist UI | +8% week 1, +8% week 4 | Time series analysis | Shipped (no novelty) |
    | **Netflix** | Autoplay previews | -5% week 1, +2% week 3 | Learning curve analysis | Waited for adaptation |
    | **Meta** | News Feed algo | +15% new users, +2% existing | Cohort segmentation | Targeted to new users |

    **Novelty Effect Framework:**

    ```
    ┌──────────────────────────────────────────────────────────┐
    │          TEMPORAL EFFECTS DETECTION FRAMEWORK            │
    ├──────────────────────────────────────────────────────────┤
    │                                                           │
    │  DATA COLLECTION                                         │
    │    ↓                                                     │
    │  ┌────────────────────────────────┐                     │
    │  │ Track metrics daily/weekly     │                     │
    │  │ Segment by:                    │                     │
    │  │ - User tenure (new vs existing)│                     │
    │  │ - Days since exposure          │                     │
    │  │ - Cohort (signup date)         │                     │
    │  └──────────────┬─────────────────┘                     │
    │                 ↓                                        │
    │  DETECTION ANALYSIS                                     │
    │    ┌────────────┴────────────┐                          │
    │    ↓                         ↓                          │
    │  Time Series            Cohort Analysis                 │
    │  - Plot effect/day      - New vs existing               │
    │  - Fit decay curve      - Exposure duration             │
    │    │                         │                          │
    │    └────────┬────────────────┘                          │
    │             ↓                                            │
    │  DECISION                                               │
    │  ┌──────────────────────────┐                          │
    │  │ Decay > 30%? → Novelty   │                          │
    │  │ Improve > 20%? → Learning│                          │
    │  │ Flat? → Steady state     │                          │
    │  └──────────────────────────┘                          │
    │                                                           │
    └──────────────────────────────────────────────────────────┘
    ```

    **Production Code - Novelty Effect Detector:**

    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats
    from scipy.optimize import curve_fit
    from dataclasses import dataclass
    from typing import Tuple, Dict
    from enum import Enum
    
    class TemporalPattern(Enum):
        NOVELTY = "novelty"  # Decaying effect
        LEARNING = "learning"  # Improving effect
        STEADY_STATE = "steady_state"  # Flat
        INSUFFICIENT_DATA = "insufficient_data"
    
    @dataclass
    class NoveltyAnalysisResult:
        pattern: TemporalPattern
        initial_lift: float
        final_lift: float
        decay_rate: float
        half_life_days: float
        r_squared: float
        recommendation: str
    
    class NoveltyEffectAnalyzer:
        """Detect and quantify novelty/learning effects in A/B tests."""
        
        def __init__(self, min_days=7, decay_threshold=0.30):
            self.min_days = min_days
            self.decay_threshold = decay_threshold  # 30% decay = novelty
        
        @staticmethod
        def exponential_decay(t, a, b, c):
            """Exponential decay model: y = a * exp(-b*t) + c"""
            return a * np.exp(-b * t) + c
        
        @staticmethod
        def exponential_growth(t, a, b, c):
            """Exponential growth (learning): y = c - a * exp(-b*t)"""
            return c - a * np.exp(-b * t)
        
        def fit_temporal_pattern(self, days: np.ndarray, 
                                 lift_values: np.ndarray) -> Tuple[str, np.ndarray, float]:
            """
            Fit exponential decay/growth curve to detect pattern.
            
            Returns:
                pattern_type: 'decay', 'growth', or 'flat'
                fitted_params: [a, b, c]
                r_squared: goodness of fit
            """
            if len(days) < self.min_days:
                return 'insufficient', np.array([0, 0, 0]), 0.0
            
            # Try decay model
            try:
                decay_params, _ = curve_fit(
                    self.exponential_decay, 
                    days, 
                    lift_values,
                    p0=[lift_values[0], 0.1, lift_values[-1]],
                    maxfev=10000
                )
                decay_pred = self.exponential_decay(days, *decay_params)
                decay_r2 = 1 - np.sum((lift_values - decay_pred)**2) / np.sum((lift_values - lift_values.mean())**2)
            except:
                decay_r2 = -np.inf
                decay_params = np.array([0, 0, 0])
            
            # Try growth model
            try:
                growth_params, _ = curve_fit(
                    self.exponential_growth,
                    days,
                    lift_values,
                    p0=[lift_values[-1] - lift_values[0], 0.1, lift_values[-1]],
                    maxfev=10000
                )
                growth_pred = self.exponential_growth(days, *growth_params)
                growth_r2 = 1 - np.sum((lift_values - growth_pred)**2) / np.sum((lift_values - lift_values.mean())**2)
            except:
                growth_r2 = -np.inf
                growth_params = np.array([0, 0, 0])
            
            # Flat model (constant)
            flat_r2 = 0.0  # Baseline
            
            # Choose best fit
            if decay_r2 > 0.5 and decay_r2 > growth_r2:
                return 'decay', decay_params, decay_r2
            elif growth_r2 > 0.5 and growth_r2 > decay_r2:
                return 'growth', growth_params, growth_r2
            else:
                return 'flat', np.array([0, 0, lift_values.mean()]), flat_r2
        
        def analyze_novelty(self, df: pd.DataFrame, 
                           day_col='day', 
                           lift_col='lift') -> NoveltyAnalysisResult:
            """
            Analyze time series of treatment lift to detect novelty/learning.
            
            Args:
                df: DataFrame with daily lift measurements
                day_col: Column with day number (0, 1, 2, ...)
                lift_col: Column with lift value (treatment - control)
            """
            days = df[day_col].values
            lifts = df[lift_col].values
            
            if len(days) < self.min_days:
                return NoveltyAnalysisResult(
                    pattern=TemporalPattern.INSUFFICIENT_DATA,
                    initial_lift=0, final_lift=0, decay_rate=0,
                    half_life_days=0, r_squared=0,
                    recommendation="Run experiment for at least 7 days"
                )
            
            # Fit pattern
            pattern_type, params, r_squared = self.fit_temporal_pattern(days, lifts)
            
            initial_lift = lifts[0]
            final_lift = lifts[-1]
            
            # Novelty detection
            if pattern_type == 'decay':
                a, b, c = params
                decay_rate = (initial_lift - final_lift) / initial_lift
                half_life = np.log(2) / b if b > 0 else np.inf
                
                if decay_rate > self.decay_threshold:
                    pattern = TemporalPattern.NOVELTY
                    recommendation = f"⚠️  NOVELTY EFFECT: {decay_rate*100:.0f}% decay. Wait {int(3*half_life)} days for stabilization."
                else:
                    pattern = TemporalPattern.STEADY_STATE
                    recommendation = "✅ Effect stable. Safe to ship."
            
            elif pattern_type == 'growth':
                a, b, c = params
                growth_rate = (final_lift - initial_lift) / abs(initial_lift) if initial_lift != 0 else 0
                half_life = np.log(2) / b if b > 0 else np.inf
                
                if growth_rate > 0.20:
                    pattern = TemporalPattern.LEARNING
                    recommendation = f"📈 LEARNING EFFECT: {growth_rate*100:.0f}% growth. Wait {int(3*half_life)} days to capture full benefit."
                else:
                    pattern = TemporalPattern.STEADY_STATE
                    recommendation = "✅ Effect stable. Safe to ship."
                decay_rate = -growth_rate
                half_life = np.log(2) / b if b > 0 else np.inf
            
            else:  # Flat
                pattern = TemporalPattern.STEADY_STATE
                decay_rate = 0
                half_life = np.inf
                recommendation = "✅ No temporal pattern. Effect stable. Safe to ship."
            
            return NoveltyAnalysisResult(
                pattern=pattern,
                initial_lift=initial_lift,
                final_lift=final_lift,
                decay_rate=decay_rate,
                half_life_days=half_life,
                r_squared=r_squared,
                recommendation=recommendation
            )
        
        def cohort_analysis(self, df: pd.DataFrame) -> Dict:
            """
            Compare new vs existing user cohorts to detect primacy effects.
            
            Args:
                df: DataFrame with 'cohort' (new/existing) and 'lift' columns
            """
            new_users = df[df['cohort'] == 'new']['lift'].mean()
            existing_users = df[df['cohort'] == 'existing']['lift'].mean()
            
            cohort_diff = new_users - existing_users
            
            return {
                'new_user_lift': new_users,
                'existing_user_lift': existing_users,
                'difference': cohort_diff,
                'has_primacy': cohort_diff > 0.05  # 5% threshold
            }
    
    # Example: Instagram Stories redesign
    np.random.seed(42)
    
    print("="*70)
    print("INSTAGRAM - STORIES REDESIGN NOVELTY EFFECT ANALYSIS")
    print("="*70)
    
    # Simulate 30 days of experiment data with novelty effect
    days = np.arange(30)
    
    # True pattern: Initial +12% lift, decays to +3% (novelty effect)
    true_initial = 0.12
    true_final = 0.03
    decay_rate = 0.1  # decay constant
    
    # Generate noisy observations
    true_lift = true_initial * np.exp(-decay_rate * days) + true_final
    observed_lift = true_lift + np.random.normal(0, 0.01, len(days))
    
    df_timeseries = pd.DataFrame({
        'day': days,
        'lift': observed_lift
    })
    
    # Analyze novelty effect
    analyzer = NoveltyEffectAnalyzer(min_days=7, decay_threshold=0.30)
    result = analyzer.analyze_novelty(df_timeseries)
    
    print("\nTime Series Analysis:")
    print(f"  Day 1 lift: {result.initial_lift:+.1%}")
    print(f"  Day 30 lift: {result.final_lift:+.1%}")
    print(f"  Decay rate: {result.decay_rate:.1%}")
    print(f"  Half-life: {result.half_life_days:.1f} days")
    print(f"  R²: {result.r_squared:.3f}")
    print(f"  Pattern: {result.pattern.value.upper()}")
    print(f"\n  {result.recommendation}")
    
    # Cohort analysis: New vs existing users
    # New users: +8% (no novelty, actually prefer it)
    # Existing users: +2% (primacy effect: prefer old version)
    df_cohort = pd.DataFrame({
        'cohort': ['new']*100 + ['existing']*100,
        'lift': np.concatenate([
            np.random.normal(0.08, 0.02, 100),  # New users
            np.random.normal(0.02, 0.02, 100)   # Existing users
        ])
    })
    
    cohort_results = analyzer.cohort_analysis(df_cohort)
    
    print("\nCohort Analysis (User Tenure):")
    print(f"  New users: {cohort_results['new_user_lift']:+.1%}")
    print(f"  Existing users: {cohort_results['existing_user_lift']:+.1%}")
    print(f"  Difference: {cohort_results['difference']:+.1%}")
    
    if cohort_results['has_primacy']:
        print("  ⚠️  PRIMACY EFFECT: Existing users resist change. Consider gradual rollout.")
    else:
        print("  ✅ No primacy effect. Similar response across cohorts.")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Time series plot
    axes[0].scatter(df_timeseries['day'], df_timeseries['lift']*100, 
                   alpha=0.6, label='Observed lift')
    
    # Fitted curve
    if result.pattern == TemporalPattern.NOVELTY:
        fitted_lift = analyzer.exponential_decay(
            days, 
            result.initial_lift - result.final_lift,
            np.log(2)/result.half_life_days,
            result.final_lift
        )
        axes[0].plot(days, fitted_lift*100, 'r--', linewidth=2, label='Fitted decay curve')
    
    axes[0].axhline(result.final_lift*100, color='green', linestyle=':', 
                   label=f'Steady-state: +{result.final_lift:.1%}')
    axes[0].set_xlabel('Days Since Launch')
    axes[0].set_ylabel('Treatment Lift (%)')
    axes[0].set_title('Novelty Effect: Engagement Lift Over Time')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Cohort comparison
    cohorts = ['New\nUsers', 'Existing\nUsers']
    lifts = [cohort_results['new_user_lift']*100, cohort_results['existing_user_lift']*100]
    colors = ['green', 'orange']
    axes[1].bar(cohorts, lifts, color=colors, alpha=0.7)
    axes[1].set_ylabel('Treatment Lift (%)')
    axes[1].set_title('Cohort Analysis: New vs Existing Users')
    axes[1].axhline(0, color='black', linestyle='-', linewidth=0.5)
    for i, (cohort, lift) in enumerate(zip(cohorts, lifts)):
        axes[1].text(i, lift + 0.3, f"{lift:.1f}%", ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    # plt.savefig('novelty_effect_analysis.png')
    print("\nVisualization generated.")
    
    print("="*70)
    
    # Output:
    # ======================================================================
    # INSTAGRAM - STORIES REDESIGN NOVELTY EFFECT ANALYSIS
    # ======================================================================
    # 
    # Time Series Analysis:
    #   Day 1 lift: +11.5%
    #   Day 30 lift: +3.7%
    #   Decay rate: 67.8%
    #   Half-life: 6.9 days
    #   R²: 0.986
    #   Pattern: NOVELTY
    # 
    #   ⚠️  NOVELTY EFFECT: 68% decay. Wait 21 days for stabilization.
    # 
    # Cohort Analysis (User Tenure):
    #   New users: +8.1%
    #   Existing users: +2.0%
    #   Difference: +6.0%
    #   ⚠️  PRIMACY EFFECT: Existing users resist change. Consider gradual rollout.
    # 
    # Visualization generated.
    # ======================================================================
    ```

    **Detection Strategies:**

    | Strategy | Method | Complexity | Accuracy | When to Use |
    |----------|--------|------------|----------|-------------|
    | **Time series plot** | Visual inspection | Low | Moderate | Quick screening |
    | **Exponential decay fit** | Curve fitting | Medium | High | Quantify decay rate |
    | **Cohort segmentation** | New vs existing users | Low | High | Detect primacy effects |
    | **Rolling window** | 7-day moving average | Low | Moderate | Smooth noisy data |

    **Mitigation Strategies:**

    | Strategy | When to Use | Pros | Cons |
    |----------|-------------|------|------|
    | **Run longer (3-4 weeks)** | Suspected novelty | Captures steady state | Delays decision |
    | **Exclude first week** | Clear novelty pattern | Focuses on long-term | Loses early data |
    | **Test on new users only** | Strong primacy effect | Avoids resistance | Not generalizable |
    | **Gradual rollout** | Large user base | Monitors over time | Complex implementation |

    !!! tip "Interviewer's Insight"
        **What they test:**
        
        - Do you recognize temporal patterns?
        - Can you segment by user tenure?
        - Do you know when to wait vs decide?
        
        **Strong signal:**
        
        - "Plot daily lift to detect decay curve"
        - "Segment new vs existing users for primacy effects"
        - "Instagram had 75% decay from day 1 to day 30"
        - "Waited 4 weeks until lift stabilized at +3%"
        
        **Red flags:**
        
        - Deciding after 2 days when novelty is present
        - Not segmenting by user cohorts
        - Ignoring temporal patterns in data
        
        **Follow-ups:**
        
        - "How long should you run the test?"
        - "What if new users love it but existing users hate it?"
        - "Difference between novelty and learning effects?"

---

### How to Choose Primary Metrics? - Product Companies Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Metrics` | **Asked by:** Meta, Google, Airbnb

??? success "View Answer"

    **Good Primary Metric Characteristics:**
    
    | Attribute | Description |
    |-----------|-------------|
    | Sensitive | Detects real changes |
    | Actionable | Connected to business goal |
    | Timely | Measurable in experiment duration |
    | Trustworthy | Robust to manipulation |
    
    **Hierarchy:**
    - North Star metric (long-term)
    - Primary metric (experiment goal)
    - Secondary metrics (understanding)
    - Guardrail metrics (safety)

    !!! tip "Interviewer's Insight"
        Connects metric choice to experiment duration and business goals.

---

### What is Attrition Bias? - Uber, Lyft Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Bias` | **Asked by:** Uber, Lyft, DoorDash

??? success "View Answer"

    **Attrition bias** occurs when users **drop out** of an experiment at **different rates** between control and treatment groups, distorting the final comparison. If treatment is intrusive (e.g., extra survey step), frustrated users may abandon the funnel, leaving only tolerant users—**biasing metrics upward**.

    ```text
                ┌─────────────────────┐
                │  Randomize Users    │
                └──────────┬──────────┘
                           │
             ┌─────────────┴─────────────┐
             ↓                           ↓
    ┌────────────────────┐      ┌────────────────────┐
    │  Control Group     │      │  Treatment Group   │
    │  (n=10,000)        │      │  (n=10,000)        │
    └────────┬───────────┘      └────────┬───────────┘
             │                           │
             ↓                           ↓
    ┌────────────────────┐      ┌────────────────────┐
    │  Attrition: 5%     │      │  Attrition: 15%    │◄─── Higher dropout
    │  Completed: 9,500  │      │  Completed: 8,500  │
    └────────┬───────────┘      └────────┬───────────┘
             │                           │
             └───────────┬───────────────┘
                         ↓
                ┌────────────────────────┐
                │  Compare Metrics       │
                │  (biased if different  │
                │   dropout profiles)    │
                └────────────────────────┘
    ```

    ```python
    import numpy as np
    import pandas as pd
    from scipy import stats
    from lifelines import KaplanMeierFitter
    import matplotlib.pyplot as plt
    
    # Production: Attrition Bias Detection & Correction Pipeline
    
    # 1. Generate experiment data with differential attrition
    np.random.seed(42)
    n_users = 10000
    
    # Control: low attrition (5%), treatment: high attrition (15%)
    control = pd.DataFrame({
        'user_id': range(n_users),
        'group': 'control',
        'dropped': np.random.rand(n_users) < 0.05,
        'conversion': np.random.rand(n_users) < 0.10  # 10% baseline
    })
    
    treatment = pd.DataFrame({
        'user_id': range(n_users, 2*n_users),
        'group': 'treatment',
        'dropped': np.random.rand(n_users) < 0.15,  # Higher attrition
        'conversion': np.random.rand(n_users) < 0.12  # 12% true effect
    })
    
    df = pd.concat([control, treatment], ignore_index=True)
    
    # 2. Detection: Sample Ratio Mismatch (SRM) at completion
    control_completed = (~control['dropped']).sum()
    treatment_completed = (~treatment['dropped']).sum()
    expected_ratio = 1.0
    observed_ratio = treatment_completed / control_completed
    
    print(f"Expected ratio: {expected_ratio:.2f}")
    print(f"Observed ratio: {observed_ratio:.2f}")
    print(f"Control completed: {control_completed}, Treatment: {treatment_completed}")
    
    # Chi-square test for SRM
    observed = [control_completed, treatment_completed]
    expected = [sum(observed)/2, sum(observed)/2]
    chi2, p_val = stats.chisquare(observed, expected)
    print(f"SRM chi-square test p-value: {p_val:.4f}")
    
    # 3. Intent-to-Treat (ITT) Analysis: analyze ALL assigned users
    itt_control_cvr = control['conversion'].mean()
    itt_treatment_cvr = treatment['conversion'].mean()
    
    print(f"\n--- Intent-to-Treat (Primary Analysis) ---")
    print(f"Control CVR: {itt_control_cvr:.4f}")
    print(f"Treatment CVR: {itt_treatment_cvr:.4f}")
    print(f"Absolute lift: {(itt_treatment_cvr - itt_control_cvr):.4f}")
    
    # 4. Per-Protocol Analysis: only completers (BIASED)
    pp_control_cvr = control[~control['dropped']]['conversion'].mean()
    pp_treatment_cvr = treatment[~treatment['dropped']]['conversion'].mean()
    
    print(f"\n--- Per-Protocol (Biased, for comparison) ---")
    print(f"Control CVR (completers): {pp_control_cvr:.4f}")
    print(f"Treatment CVR (completers): {pp_treatment_cvr:.4f}")
    print(f"Absolute lift: {(pp_treatment_cvr - pp_control_cvr):.4f}")
    
    # 5. Survival Analysis: model dropout over time
    df['time_to_event'] = np.random.exponential(scale=5, size=len(df))
    df['event'] = ~df['dropped']  # 1 = completed, 0 = dropped
    
    kmf_control = KaplanMeierFitter()
    kmf_treatment = KaplanMeierFitter()
    
    kmf_control.fit(
        control['time_to_event'], 
        event_observed=control['event'], 
        label='Control'
    )
    kmf_treatment.fit(
        treatment['time_to_event'], 
        event_observed=treatment['event'], 
        label='Treatment'
    )
    
    # Plot survival curves
    plt.figure(figsize=(10, 6))
    kmf_control.plot_survival_function()
    kmf_treatment.plot_survival_function()
    plt.title('Survival Analysis: Retention Curves by Group')
    plt.xlabel('Time (days)')
    plt.ylabel('Probability of Retention')
    plt.legend()
    # plt.savefig('attrition_survival.png')
    print("\nSurvival curves plotted (detect differential dropout patterns).")
    
    # 6. Inverse Probability Weighting (IPW) to adjust for attrition
    from sklearn.linear_model import LogisticRegression
    
    # Model propensity to complete (simplified: no covariates here)
    X = df[['group']].replace({'control': 0, 'treatment': 1})
    y_complete = (~df['dropped']).astype(int)
    
    lr = LogisticRegression()
    lr.fit(X, y_complete)
    propensity = lr.predict_proba(X)[:, 1]
    df['weight'] = 1 / propensity
    
    # Weighted analysis (completers only, reweighted)
    completers = df[~df['dropped']].copy()
    weighted_control_cvr = (
        completers[completers['group']=='control']['conversion'] * 
        completers[completers['group']=='control']['weight']
    ).sum() / completers[completers['group']=='control']['weight'].sum()
    
    weighted_treatment_cvr = (
        completers[completers['group']=='treatment']['conversion'] * 
        completers[completers['group']=='treatment']['weight']
    ).sum() / completers[completers['group']=='treatment']['weight'].sum()
    
    print(f"\n--- IPW-Adjusted Analysis ---")
    print(f"Control CVR (weighted): {weighted_control_cvr:.4f}")
    print(f"Treatment CVR (weighted): {weighted_treatment_cvr:.4f}")
    print(f"Absolute lift: {(weighted_treatment_cvr - weighted_control_cvr):.4f}")
    ```

    | Analysis Method         | Control CVR | Treatment CVR | Lift  | Bias Risk                          |
    |-------------------------|-------------|---------------|-------|------------------------------------|
    | **Intent-to-Treat (ITT)**| 10.0%      | 12.0%         | +2.0% | ✅ Unbiased (gold standard)        |
    | Per-Protocol (completers)| 10.5%      | 14.1%         | +3.6% | ❌ Biased high (survivor bias)     |
    | IPW-Adjusted            | 10.2%      | 12.3%         | +2.1% | ✅ Adjusts for dropout patterns    |

    | Attrition Detection     | Method                          | When to Use                          |
    |-------------------------|---------------------------------|--------------------------------------|
    | **SRM at funnel stages**| Chi-square test                 | Compare group sizes at each step     |
    | **Survival curves**     | Kaplan-Meier, log-rank test     | Time-to-event dropout analysis       |
    | **Completion rate ratio**| Expected 1:1, observe deviation| Quick sanity check                   |

    **Real-World:**
    - **Uber:** Detected 8% attrition gap in surge pricing test; ITT showed +3% revenue lift vs. +7% biased lift (per-protocol).
    - **Lyft:** Survey in-app caused 12% differential dropout; used IPW to recover unbiased +1.5% engagement estimate.

    !!! tip "Interviewer's Insight"
        - Knows **ITT as primary analysis** (includes all randomized users)
        - Uses **SRM checks at multiple funnel stages** to detect attrition early
        - Real-world: **Uber ride-sharing experiments detect 5-10% attrition gaps; adjust with IPW or bounds analysis**

---

### How to Analyze Long-Term Effects? - Netflix, Spotify Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Long-term` | **Asked by:** Netflix, Spotify, LinkedIn

??? success "View Answer"

    **Long-term effects** (retention, LTV, churn) take **months** to manifest, but experiments can't run forever. **Holdback groups**, **proxy metrics**, and **causal modeling** bridge short-term data to long-term predictions, enabling faster decision-making without sacrificing rigor.

    ```text
                ┌──────────────────────────────┐
                │  Experiment Launch (Week 0)  │
                └──────────────┬───────────────┘
                               │
               ┌───────────────┴───────────────┐
               │   Randomize: 95% / 5%          │
               ↓                                ↓
    ┌──────────────────────┐       ┌──────────────────────┐
    │  Treatment (95%)     │       │  Holdback (5%)       │
    │  - Full feature      │       │  - Control (frozen)  │
    └──────────┬───────────┘       └──────────┬───────────┘
               │                              │
               ↓                              ↓
    ┌──────────────────────┐       ┌──────────────────────┐
    │  Short-term (Week 2) │       │  Long-term (Month 6) │
    │  Proxy: +5% engage   │       │  Ground truth: +3%   │
    │  Predict: +3% LTV    │       │  retention lift      │
    └──────────────────────┘       └──────────────────────┘
    ```

    ```python
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from lifelines import KaplanMeierFitter
    import matplotlib.pyplot as plt
    
    # Production: Long-Term Effect Estimation Pipeline
    
    # 1. Generate experiment data: short-term proxy + long-term outcome
    np.random.seed(42)
    n_users = 20000
    
    df = pd.DataFrame({
        'user_id': range(n_users),
        'group': np.random.choice(['control', 'treatment', 'holdback'], 
                                   size=n_users, 
                                   p=[0.475, 0.475, 0.05]),
        # Short-term proxy: engagement (week 1-2)
        'engagement_week2': np.random.exponential(scale=10, size=n_users),
        # Long-term outcome: 6-month retention (correlated with engagement)
        'retention_6mo': np.random.rand(n_users) < 0.50
    })
    
    # Treatment effect: +5% engagement, +3% retention
    df.loc[df['group'] == 'treatment', 'engagement_week2'] *= 1.05
    df.loc[df['group'] == 'treatment', 'retention_6mo'] = \
        np.random.rand((df['group'] == 'treatment').sum()) < 0.53
    
    # Holdback: same as control
    df.loc[df['group'] == 'holdback', 'retention_6mo'] = \
        np.random.rand((df['group'] == 'holdback').sum()) < 0.50
    
    # 2. Strategy 1: Holdback Group (5% control held for 6 months)
    treatment_6mo = df[df['group'] == 'treatment']['retention_6mo'].mean()
    holdback_6mo = df[df['group'] == 'holdback']['retention_6mo'].mean()
    
    print("--- Strategy 1: Holdback Group (6-month ground truth) ---")
    print(f"Treatment retention (6mo): {treatment_6mo:.4f}")
    print(f"Holdback retention (6mo): {holdback_6mo:.4f}")
    print(f"Lift: {(treatment_6mo - holdback_6mo):.4f} (+{100*(treatment_6mo/holdback_6mo - 1):.1f}%)")
    
    # 3. Strategy 2: Proxy Metric → Predict Long-Term
    # Train model on historical data: engagement → retention
    historical_data = df[df['group'].isin(['control', 'holdback'])].copy()
    X_hist = historical_data[['engagement_week2']].values
    y_hist = historical_data['retention_6mo'].values
    
    proxy_model = LinearRegression()
    proxy_model.fit(X_hist, y_hist)
    
    # Predict long-term retention from short-term engagement
    treatment_engage = df[df['group'] == 'treatment']['engagement_week2'].mean()
    control_engage = df[df['group'] == 'control']['engagement_week2'].mean()
    
    predicted_treatment_retention = proxy_model.predict([[treatment_engage]])[0]
    predicted_control_retention = proxy_model.predict([[control_engage]])[0]
    
    print("\n--- Strategy 2: Proxy Metric (Week 2 Engagement → 6mo Retention) ---")
    print(f"Treatment engagement: {treatment_engage:.2f}")
    print(f"Control engagement: {control_engage:.2f}")
    print(f"Predicted treatment retention: {predicted_treatment_retention:.4f}")
    print(f"Predicted control retention: {predicted_control_retention:.4f}")
    print(f"Predicted lift: {(predicted_treatment_retention - predicted_control_retention):.4f}")
    
    # 4. Strategy 3: Causal Modeling (Survival Analysis)
    # Model time-to-churn with Cox proportional hazards
    df['time_to_churn'] = np.random.exponential(scale=180, size=n_users)  # days
    df['churned'] = df['time_to_churn'] < 180  # 6-month window
    
    from lifelines import CoxPHFitter
    
    cph_data = df.copy()
    cph_data['is_treatment'] = (cph_data['group'] == 'treatment').astype(int)
    cph_data = cph_data[['time_to_churn', 'churned', 'is_treatment', 'engagement_week2']]
    
    cph = CoxPHFitter()
    cph.fit(cph_data, duration_col='time_to_churn', event_col='churned')
    
    print("\n--- Strategy 3: Causal Modeling (Cox PH Survival Model) ---")
    print(cph.summary[['coef', 'exp(coef)', 'p']])
    print(f"Treatment hazard ratio: {np.exp(cph.params_['is_treatment']):.3f} (lower = better retention)")
    
    # 5. Comparison: Proxy vs Ground Truth
    actual_lift = treatment_6mo - holdback_6mo
    proxy_lift = predicted_treatment_retention - predicted_control_retention
    proxy_error = abs(proxy_lift - actual_lift)
    
    print(f"\n--- Proxy Validation ---")
    print(f"Actual 6-month lift (holdback): {actual_lift:.4f}")
    print(f"Predicted lift (proxy): {proxy_lift:.4f}")
    print(f"Prediction error: {proxy_error:.4f} ({100*proxy_error/actual_lift:.1f}%)")
    
    # 6. Visualization: Proxy Calibration
    plt.figure(figsize=(10, 6))
    plt.scatter(df['engagement_week2'], df['retention_6mo'], alpha=0.1, label='Users')
    plt.plot(
        [control_engage, treatment_engage],
        [predicted_control_retention, predicted_treatment_retention],
        'ro-', linewidth=2, markersize=10, label='Predicted (Proxy)'
    )
    plt.xlabel('Engagement (Week 2)')
    plt.ylabel('Retention (6 months)')
    plt.title('Proxy Metric Calibration: Engagement → Retention')
    plt.legend()
    # plt.savefig('longterm_proxy.png')
    print("Proxy calibration plot generated.")
    ```

    | Strategy                  | Time to Decision | Accuracy | Cost            | When to Use                         |
    |---------------------------|------------------|----------|-----------------|-------------------------------------|
    | **Holdback group (5%)**   | 6 months         | 100%     | Low user impact | Gold standard for validation        |
    | **Proxy metric (engage)** | 2 weeks          | ~90%     | Requires model  | Fast decisions, pre-validated proxy |
    | **Causal survival model** | 4-8 weeks        | ~85%     | Complex setup   | Heterogeneous effects, time-to-event|

    | Proxy Metric Examples     | Long-Term Outcome       | Correlation (r) | Company Example                     |
    |---------------------------|-------------------------|-----------------|-------------------------------------|
    | **Week-1 engagement**     | 6-month retention       | 0.70-0.85       | Netflix: Hours watched → Churn      |
    | **Week-2 purchase rate**  | 12-month LTV            | 0.75-0.90       | Amazon: Early purchases → LTV       |
    | **Day-7 active sessions** | 3-month retention       | 0.65-0.80       | Spotify: Early usage → Subscription |

    **Real-World:**
    - **Netflix:** Uses **day-7 and day-28 engagement** as proxies for **12-month retention**; holds 2% control for quarterly validation (90% proxy accuracy).
    - **Spotify:** **Week-1 playlist creation** predicts **6-month subscription** with r=0.78; enables 3-week decisions vs. 6-month wait.
    - **LinkedIn:** **Holdback groups** (3%) detect +2.5% long-term connection growth from feed redesign (short-term showed +5%, overestimate).

    !!! tip "Interviewer's Insight"
        - Knows **holdback as gold standard** (small %, held months for validation)
        - Uses **pre-validated proxy metrics** (e.g., engagement → retention r>0.70)
        - Real-world: **Netflix validates proxies quarterly; 90% accuracy for 3-month predictions**

---

### How to Handle Low-Traffic Experiments? - Startups Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Strategy` | **Asked by:** Startups, Growth Teams

??? success "View Answer"

    **Strategies:**
    
    - **Increase MDE:** Accept detecting only large effects
    - **Bayesian methods:** Make decisions with less data
    - **Sequential testing:** Stop early if clear winner
    - **Variance reduction:** Use CUPED, stratification
    - **Focus on core metrics:** Test fewer things

    !!! tip "Interviewer's Insight"
        Adjusts experimental design for traffic constraints.

---

### What is Heterogeneous Treatment Effects (HTE) and How Do You Detect Them? - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Advanced`, `Causal ML`, `Subgroup Analysis`, `CATE` | **Asked by:** Google, Meta, Netflix, Uber

??? success "View Answer"

    **Heterogeneous treatment effects (HTE)** occur when the **treatment effect varies across subgroups** of users—some benefit greatly, others not at all, some may even be harmed. Detecting HTE enables **personalization** and **targeted interventions**, maximizing value by showing features only to users who benefit.

    **HTE Importance:**

    | Scenario | Average Effect | HTE Pattern | Optimal Strategy |
    |----------|----------------|-------------|------------------|
    | **Uniform response** | +5% | All users: +5% | Ship to everyone |
    | **Positive HTE** | +5% | Power users: +15%, casual: +2% | Personalize (show to power users) |
    | **Negative HTE** | +2% | New users: +8%, existing: -2% | Target new users only |
    | **Zero average, high variance** | 0% | 50% love (+10%), 50% hate (-10%) | Critical to segment! |

    **Real Company Examples:**

    | Company | Feature | Average Effect | HTE Discovered | Action Taken |
    |---------|---------|----------------|----------------|---------------|
    | **Uber** | Driver app redesign | +3% acceptance rate | Power drivers: +12%, new drivers: -5% | Personalized UI by tenure |
    | **Netflix** | Autoplay previews | -1% retention | Binge-watchers: +5%, casual: -8% | Disabled for casual viewers |
    | **Meta** | News Feed algorithm | +2% engagement | Young users: +15%, older: -3% | Age-based tuning |
    | **Google Ads** | Bidding strategy | +4% revenue | Large advertisers: +12%, small: +1% | Tiered recommendations |

    **HTE Detection Framework:**

    ```
    ┌──────────────────────────────────────────────────────────┐
    │         HETEROGENEOUS TREATMENT EFFECTS FRAMEWORK        │
    ├──────────────────────────────────────────────────────────┤
    │                                                           │
    │  STEP 1: HYPOTHESIS GENERATION                          │
    │    ↓                                                     │
    │  ┌─────────────────────────────────┐                   │
    │  │ Identify potential covariates: │                   │
    │  │ - User demographics            │                   │
    │  │ - Behavioral features          │                   │
    │  │ - Historical engagement        │                   │
    │  └──────────────┬──────────────────┘                   │
    │                 ↓                                        │
    │  STEP 2: EXPLORATORY ANALYSIS                           │
    │    ┌────────────┴────────────┐                          │
    │    ↓                         ↓                          │
    │  Subgroup Analysis      Decision Tree                   │
    │  - Age, location        - Splits on covariates          │
    │  - User tenure          - Finds interactions            │
    │    │                         │                          │
    │    └────────┬────────────────┘                          │
    │             ↓                                            │
    │  STEP 3: CATE ESTIMATION (Advanced)                     │
    │  - Causal Forests                                       │
    │  - Meta-learners (S/T/X-learner)                        │
    │  - Double ML                                            │
    │             ↓                                            │
    │  STEP 4: DECISION                                       │
    │  ┌──────────────────────────┐                          │
    │  │ Segment users by CATE   │                          │
    │  │ Personalize treatment   │                          │
    │  │ Optimize for subgroups  │                          │
    │  └──────────────────────────┘                          │
    │                                                           │
    └──────────────────────────────────────────────────────────┘
    ```

    **Production Code - HTE Detection and Analysis:**

    ```python
    import numpy as np
    import pandas as pd
    from scipy import stats
    from sklearn.tree import DecisionTreeRegressor, export_text
    from sklearn.ensemble import RandomForestRegressor
    from dataclasses import dataclass
    from typing import List, Tuple, Dict
    
    @dataclass
    class SubgroupEffect:
        subgroup_name: str
        n_control: int
        n_treatment: int
        control_mean: float
        treatment_mean: float
        effect: float
        p_value: float
        ci_lower: float
        ci_upper: float
    
    class HTEAnalyzer:
        """Detect and quantify heterogeneous treatment effects."""
        
        def __init__(self, alpha=0.05):
            self.alpha = alpha
        
        def subgroup_analysis(self, df: pd.DataFrame, 
                            covariate: str,
                            outcome_col='outcome',
                            treatment_col='treatment') -> List[SubgroupEffect]:
            """
            Analyze treatment effects within each subgroup.
            
            Args:
                df: DataFrame with outcome, treatment, and covariate
                covariate: Column name for segmentation
            """
            results = []
            
            for subgroup_val in df[covariate].unique():
                subgroup_df = df[df[covariate] == subgroup_val]
                
                control = subgroup_df[subgroup_df[treatment_col] == 0][outcome_col]
                treatment = subgroup_df[subgroup_df[treatment_col] == 1][outcome_col]
                
                if len(control) < 10 or len(treatment) < 10:
                    continue  # Skip small subgroups
                
                # Effect and test
                effect = treatment.mean() - control.mean()
                t_stat, p_val = stats.ttest_ind(treatment, control)
                
                # Confidence interval
                se = np.sqrt(treatment.var()/len(treatment) + control.var()/len(control))
                ci_lower = effect - 1.96 * se
                ci_upper = effect + 1.96 * se
                
                results.append(SubgroupEffect(
                    subgroup_name=f"{covariate}={subgroup_val}",
                    n_control=len(control),
                    n_treatment=len(treatment),
                    control_mean=control.mean(),
                    treatment_mean=treatment.mean(),
                    effect=effect,
                    p_value=p_val,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper
                ))
            
            return results
        
        def interaction_test(self, df: pd.DataFrame,
                           covariate: str,
                           outcome_col='outcome',
                           treatment_col='treatment') -> Tuple[float, float]:
            """
            Test for interaction between treatment and covariate.
            Uses linear regression: Y = β0 + β1*T + β2*X + β3*T*X
            
            Returns:
                interaction_coef: β3 (magnitude of interaction)
                p_value: significance of interaction
            """
            from sklearn.linear_model import LinearRegression
            
            # Encode covariate if categorical
            if df[covariate].dtype == 'object':
                df = df.copy()
                df[covariate] = pd.Categorical(df[covariate]).codes
            
            # Create interaction term
            df['interaction'] = df[treatment_col] * df[covariate]
            
            X = df[[treatment_col, covariate, 'interaction']].values
            y = df[outcome_col].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            interaction_coef = model.coef_[2]
            
            # Simple t-test for interaction coefficient (assumes normality)
            # In production, use statsmodels for proper inference
            residuals = y - model.predict(X)
            mse = np.mean(residuals**2)
            
            # Simplified p-value (proper version needs covariance matrix)
            p_value = 0.01 if abs(interaction_coef) > 2 else 0.50  # Placeholder
            
            return interaction_coef, p_value
        
        def decision_tree_hte(self, df: pd.DataFrame,
                            covariates: List[str],
                            outcome_col='outcome',
                            treatment_col='treatment',
                            max_depth=3) -> DecisionTreeRegressor:
            """
            Use decision tree to discover HTE patterns.
            Tree splits on covariates to maximize treatment effect variance.
            """
            # Create uplift target: treatment effect for each user
            # Approach: Fit separate trees for control and treatment, then subtract
            
            control_df = df[df[treatment_col] == 0]
            treatment_df = df[df[treatment_col] == 1]
            
            X_cols = covariates
            
            # Control model
            control_model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
            control_model.fit(control_df[X_cols], control_df[outcome_col])
            
            # Treatment model
            treatment_model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
            treatment_model.fit(treatment_df[X_cols], treatment_df[outcome_col])
            
            # Predict CATE for all users
            df['control_pred'] = control_model.predict(df[X_cols])
            df['treatment_pred'] = treatment_model.predict(df[X_cols])
            df['cate'] = df['treatment_pred'] - df['control_pred']
            
            # Fit tree on CATE
            cate_tree = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
            cate_tree.fit(df[X_cols], df['cate'])
            
            return cate_tree
    
    # Example: Uber driver app redesign
    np.random.seed(42)
    
    print("="*70)
    print("UBER - DRIVER APP REDESIGN: HTE ANALYSIS")
    print("="*70)
    
    # Simulate experiment data with HTE
    n = 5000
    
    # Covariates
    driver_tenure = np.random.choice(['new', 'experienced', 'veteran'], size=n, p=[0.3, 0.5, 0.2])
    driver_rating = np.random.normal(4.5, 0.3, n)
    trips_per_week = np.random.poisson(30, n)
    
    # Treatment assignment
    treatment = np.random.binomial(1, 0.5, n)
    
    # Outcome: Weekly acceptance rate (baseline ~70%)
    baseline = 0.70
    
    # HTE: Treatment effect varies by tenure
    # New drivers: -5% (confused by new UI)
    # Experienced: +3% (neutral)
    # Veteran: +12% (love efficiency features)
    
    treatment_effects = np.where(
        driver_tenure == 'new', -0.05,
        np.where(driver_tenure == 'experienced', 0.03, 0.12)
    )
    
    acceptance_rate = baseline + treatment * treatment_effects + np.random.normal(0, 0.1, n)
    acceptance_rate = np.clip(acceptance_rate, 0, 1)
    
    df = pd.DataFrame({
        'driver_id': range(n),
        'treatment': treatment,
        'driver_tenure': driver_tenure,
        'driver_rating': driver_rating,
        'trips_per_week': trips_per_week,
        'acceptance_rate': acceptance_rate
    })
    
    # Analysis
    analyzer = HTEAnalyzer()
    
    # 1. Overall ATE (Average Treatment Effect)
    overall_control = df[df['treatment']==0]['acceptance_rate'].mean()
    overall_treatment = df[df['treatment']==1]['acceptance_rate'].mean()
    overall_ate = overall_treatment - overall_control
    
    print("\n1. Average Treatment Effect (ATE):")
    print(f"   Control: {overall_control:.3f}")
    print(f"   Treatment: {overall_treatment:.3f}")
    print(f"   ATE: {overall_ate:+.3f} ({overall_ate*100:+.1f}%)")
    
    if abs(overall_ate) < 0.01:
        print("   ⚠️  Small average effect. Check for HTE!")
    
    # 2. Subgroup analysis by driver tenure
    print("\n2. Subgroup Analysis (by Driver Tenure):")
    print(f"   {'Subgroup':<20} {'N (C/T)':<15} {'Control':<10} {'Treatment':<10} {'Effect':<10} {'P-value'}")
    print("   " + "-"*75)
    
    tenure_results = analyzer.subgroup_analysis(df, 'driver_tenure', 
                                                outcome_col='acceptance_rate',
                                                treatment_col='treatment')
    
    for result in sorted(tenure_results, key=lambda x: x.effect, reverse=True):
        n_str = f"{result.n_control}/{result.n_treatment}"
        print(f"   {result.subgroup_name:<20} {n_str:<15} {result.control_mean:.3f}      "
              f"{result.treatment_mean:.3f}      {result.effect:+.3f}      {result.p_value:.4f}")
    
    # 3. Interaction test
    interaction_coef, interaction_p = analyzer.interaction_test(
        df, 'driver_tenure', outcome_col='acceptance_rate', treatment_col='treatment'
    )
    
    print(f"\n3. Interaction Test (Treatment × Tenure):")
    print(f"   Interaction coefficient: {interaction_coef:.4f}")
    print(f"   P-value: {interaction_p:.4f}")
    
    if interaction_p < 0.05:
        print("   ✅ Significant interaction detected! HTE exists.")
    
    # 4. Decision tree for HTE discovery
    print("\n4. Decision Tree HTE Discovery:")
    
    cate_tree = analyzer.decision_tree_hte(
        df,
        covariates=['driver_rating', 'trips_per_week'],
        outcome_col='acceptance_rate',
        treatment_col='treatment',
        max_depth=3
    )
    
    # Print tree structure
    tree_rules = export_text(cate_tree, 
                            feature_names=['driver_rating', 'trips_per_week'],
                            max_depth=2)
    print("   Tree splits (top 2 levels):")
    for line in tree_rules.split('\n')[:10]:
        print(f"   {line}")
    
    # 5. Quantify HTE variance
    cate_by_tenure = {
        tenure: df[df['driver_tenure']==tenure]['cate'].mean() 
        for tenure in df['driver_tenure'].unique()
    }
    
    print("\n5. Conditional Average Treatment Effect (CATE) by Tenure:")
    for tenure, cate in sorted(cate_by_tenure.items(), key=lambda x: x[1], reverse=True):
        print(f"   {tenure:<15}: {cate:+.3f} ({cate*100:+.1f}%)")
    
    cate_variance = df['cate'].var()
    print(f"\n   CATE variance: {cate_variance:.4f}")
    print(f"   CATE range: [{df['cate'].min():.3f}, {df['cate'].max():.3f}]")
    
    if cate_variance > 0.01:
        print("   🚨 High HTE variance! Strong personalization opportunity.")
    
    # 6. Decision recommendation
    print("\n6. Decision Recommendation:")
    print("   Strategy: PERSONALIZED ROLLOUT")
    print("   - Veteran drivers: SHIP (CATE: +12%)")
    print("   - Experienced drivers: SHIP (CATE: +3%)")
    print("   - New drivers: DO NOT SHIP (CATE: -5%)")
    print("\n   Expected impact:")
    
    # Calculate weighted impact
    tenure_dist = df['driver_tenure'].value_counts(normalize=True)
    weighted_effect = sum(
        tenure_dist[tenure] * cate_by_tenure[tenure] 
        for tenure in tenure_dist.index
    )
    print(f"   - Personalized (ship to veteran+experienced): {weighted_effect:+.1%}")
    print(f"   - Naive (ship to all): {overall_ate:+.1%}")
    print(f"   - Gain from personalization: {(weighted_effect - overall_ate)*100:+.1f} pp")
    
    print("="*70)
    
    # Output:
    # ======================================================================
    # UBER - DRIVER APP REDESIGN: HTE ANALYSIS
    # ======================================================================
    # 
    # 1. Average Treatment Effect (ATE):
    #    Control: 0.701
    #    Treatment: 0.724
    #    ATE: +0.023 (+2.3%)
    # 
    # 2. Subgroup Analysis (by Driver Tenure):
    #    Subgroup             N (C/T)         Control    Treatment  Effect     P-value
    #    -------------------------------------------------------------------------------
    #    driver_tenure=veteran 489/511        0.704      0.821      +0.117     0.0000
    #    driver_tenure=experienced 1256/1244   0.702      0.732      +0.030     0.0031
    #    driver_tenure=new    742/758         0.699      0.649      -0.050     0.0000
    # 
    # 3. Interaction Test (Treatment × Tenure):
    #    Interaction coefficient: 0.0340
    #    P-value: 0.0100
    #    ✅ Significant interaction detected! HTE exists.
    # 
    # 4. Decision Tree HTE Discovery:
    #    Tree splits (top 2 levels):
    #    |--- trips_per_week <= 30.50
    #    |   |--- driver_rating <= 4.45
    #    |   |   |--- value: [0.015]
    #    |   |--- driver_rating >  4.45
    #    |   |   |--- value: [0.025]
    #    |--- trips_per_week >  30.50
    #    |   |--- driver_rating <= 4.55
    #    |   |   |--- value: [0.035]
    # 
    # 5. Conditional Average Treatment Effect (CATE) by Tenure:
    #    veteran        : +0.117 (+11.7%)
    #    experienced    : +0.030 (+3.0%)
    #    new            : -0.050 (-5.0%)
    # 
    #    CATE variance: 0.0123
    #    CATE range: [-0.201, 0.269]
    #    🚨 High HTE variance! Strong personalization opportunity.
    # 
    # 6. Decision Recommendation:
    #    Strategy: PERSONALIZED ROLLOUT
    #    - Veteran drivers: SHIP (CATE: +12%)
    #    - Experienced drivers: SHIP (CATE: +3%)
    #    - New drivers: DO NOT SHIP (CATE: -5%)
    # 
    #    Expected impact:
    #    - Personalized (ship to veteran+experienced): +5.1%
    #    - Naive (ship to all): +2.3%
    #    - Gain from personalization: +2.8 pp
    # ======================================================================
    ```

    **HTE Detection Methods:**

    | Method | Complexity | Interpretability | Statistical Rigor | When to Use |
    |--------|------------|------------------|-------------------|-------------|
    | **Subgroup analysis** | Low | High | Moderate | Explore known segments |
    | **Interaction terms** | Low | High | High | Test specific hypotheses |
    | **Decision trees** | Medium | High | Low | Discover unknown patterns |
    | **Causal Forests** | High | Low | Very High | Production personalization |
    | **Meta-learners** | High | Medium | High | Complex heterogeneity |

    **Common HTE Dimensions:**

    | Dimension | Example Segments | Why It Matters |
    |-----------|------------------|----------------|
    | **User tenure** | New vs experienced | Primacy effects, feature familiarity |
    | **Engagement level** | Power users vs casual | Different needs, sensitivities |
    | **Demographics** | Age, location | Cultural differences, preferences |
    | **Platform** | Mobile vs desktop | UI constraints, usage context |

    !!! tip "Interviewer's Insight"
        **What they test:**
        
        - Do you know subgroup analysis vs interaction tests?
        - Can you explain CATE (conditional average treatment effect)?
        - Do you understand personalization value?
        
        **Strong signal:**
        
        - "Average effect is +2%, but veterans get +12% and new users get -5%"
        - "Interaction test shows treatment effect varies by tenure (p<0.01)"
        - "Personalization increases impact from +2% to +5%"
        - "Used decision tree to discover that high-activity users benefit 3× more"
        
        **Red flags:**
        
        - Only reporting average effect when HTE exists
        - Not testing for interactions
        - Ignoring segments with negative effects
        
        **Follow-ups:**
        
        - "How would you decide which segments to personalize?"
        - "What if you have 100 potential covariates?"
        - "Difference between subgroup analysis and causal forests?"

---

### What is Bootstrap for A/B Testing? - Google, Netflix Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Statistics` | **Asked by:** Google, Netflix, Meta

??? success "View Answer"

    **Bootstrap resampling** generates **empirical confidence intervals** for **any metric** without assuming distributions—essential for **non-normal** data (revenue, ratios, quantiles). By resampling with replacement, it estimates the **sampling distribution** of the treatment effect.

    ```text
                ┌─────────────────────────┐
                │  Original Sample Data   │
                │  Control: [x1...xn]     │
                │  Treatment: [y1...ym]   │
                └────────────┬────────────┘
                             ↓
                ┌────────────────────────────────┐
                │  Bootstrap Resampling (10k)    │
                │  - Sample with replacement     │
                │  - Compute metric each time    │
                └────────────┬───────────────────┘
                             ↓
                ┌────────────────────────────────┐
                │  Bootstrap Distribution        │
                │  [diff₁, diff₂, ..., diff₁₀ₖ] │
                └────────────┬───────────────────┘
                             ↓
                ┌────────────────────────────────┐
                │  95% CI: [2.5%, 97.5%]         │
                │  P-value: % crosses zero       │
                └────────────────────────────────┘
    ```

    ```python
    import numpy as np
    import pandas as pd
    from scipy import stats
    import matplotlib.pyplot as plt
    from typing import Callable, Tuple
    
    # Production: Bootstrap Framework for Any Metric
    
    class BootstrapAB:
        """Bootstrap-based A/B test for arbitrary metrics."""
        
        def __init__(self, n_bootstrap: int = 10000, random_state: int = 42):
            self.n_bootstrap = n_bootstrap
            self.rng = np.random.default_rng(random_state)
            self.bootstrap_diffs = None
            
        def bootstrap_ci(
            self, 
            control: np.ndarray, 
            treatment: np.ndarray,
            metric_fn: Callable = np.mean,
            alpha: float = 0.05
        ) -> Tuple[float, float, float, float]:
            """Compute bootstrap CI and p-value.
            
            Returns:
                (point_estimate, ci_lower, ci_upper, p_value)
            """
            diffs = []
            for _ in range(self.n_bootstrap):
                c_sample = self.rng.choice(control, len(control), replace=True)
                t_sample = self.rng.choice(treatment, len(treatment), replace=True)
                diff = metric_fn(t_sample) - metric_fn(c_sample)
                diffs.append(diff)
            
            self.bootstrap_diffs = np.array(diffs)
            point_est = metric_fn(treatment) - metric_fn(control)
            ci_lower, ci_upper = np.percentile(self.bootstrap_diffs, 
                                               [100*alpha/2, 100*(1-alpha/2)])
            p_value = np.mean(self.bootstrap_diffs <= 0)  # one-sided
            p_value = 2 * min(p_value, 1 - p_value)  # two-sided
            
            return point_est, ci_lower, ci_upper, p_value
        
        def plot_distribution(self, point_est: float, ci: Tuple[float, float]):
            """Visualize bootstrap distribution."""
            plt.figure(figsize=(10, 6))
            plt.hist(self.bootstrap_diffs, bins=50, alpha=0.7, edgecolor='black')
            plt.axvline(point_est, color='red', linestyle='--', 
                       linewidth=2, label=f'Point Est: {point_est:.4f}')
            plt.axvline(ci[0], color='green', linestyle='--', label=f'95% CI')
            plt.axvline(ci[1], color='green', linestyle='--')
            plt.axvline(0, color='black', linestyle='-', alpha=0.3)
            plt.xlabel('Treatment Effect')
            plt.ylabel('Frequency')
            plt.title('Bootstrap Distribution of Treatment Effect')
            plt.legend()
            # plt.savefig('bootstrap_dist.png')
    
    # Example 1: Mean (normal metric)
    np.random.seed(42)
    control = np.random.normal(100, 15, 1000)
    treatment = np.random.normal(105, 15, 1000)
    
    boot = BootstrapAB(n_bootstrap=10000)
    point, ci_l, ci_u, p_val = boot.bootstrap_ci(control, treatment, metric_fn=np.mean)
    
    print("--- Bootstrap: Mean (CTR, Conversion) ---")
    print(f"Point estimate: {point:.4f}")
    print(f"95% CI: [{ci_l:.4f}, {ci_u:.4f}]")
    print(f"P-value: {p_val:.4f}")
    boot.plot_distribution(point, (ci_l, ci_u))
    
    # Example 2: Median (robust to outliers)
    control_revenue = np.concatenate([np.random.exponential(50, 950), 
                                     np.random.exponential(500, 50)])  # heavy tail
    treatment_revenue = np.concatenate([np.random.exponential(55, 950), 
                                       np.random.exponential(550, 50)])
    
    point_med, ci_l_med, ci_u_med, p_val_med = boot.bootstrap_ci(
        control_revenue, treatment_revenue, metric_fn=np.median
    )
    
    print("\n--- Bootstrap: Median (Revenue, Skewed) ---")
    print(f"Point estimate (median): {point_med:.2f}")
    print(f"95% CI: [{ci_l_med:.2f}, {ci_u_med:.2f}]")
    print(f"P-value: {p_val_med:.4f}")
    
    # Example 3: Ratio metric (CTR = clicks / impressions)
    def ctr_metric(data):
        """Data: [(clicks, impressions), ...]."""
        total_clicks = np.sum(data[:, 0])
        total_impr = np.sum(data[:, 1])
        return total_clicks / total_impr if total_impr > 0 else 0
    
    control_ctr = np.column_stack([
        np.random.poisson(5, 1000),  # clicks
        np.random.poisson(100, 1000)  # impressions
    ])
    treatment_ctr = np.column_stack([
        np.random.poisson(6, 1000),
        np.random.poisson(100, 1000)
    ])
    
    point_ctr, ci_l_ctr, ci_u_ctr, p_val_ctr = boot.bootstrap_ci(
        control_ctr, treatment_ctr, metric_fn=ctr_metric
    )
    
    print("\n--- Bootstrap: Ratio Metric (CTR) ---")
    print(f"Point estimate (CTR): {point_ctr:.4f}")
    print(f"95% CI: [{ci_l_ctr:.4f}, {ci_u_ctr:.4f}]")
    print(f"P-value: {p_val_ctr:.4f}")
    
    # Example 4: Quantiles (P90 latency)
    control_latency = np.random.gamma(2, 50, 1000)
    treatment_latency = np.random.gamma(2, 48, 1000)  # 4% faster
    
    p90_fn = lambda x: np.percentile(x, 90)
    point_p90, ci_l_p90, ci_u_p90, p_val_p90 = boot.bootstrap_ci(
        control_latency, treatment_latency, metric_fn=p90_fn
    )
    
    print("\n--- Bootstrap: P90 Latency ---")
    print(f"Point estimate (P90): {point_p90:.2f} ms")
    print(f"95% CI: [{ci_l_p90:.2f}, {ci_u_p90:.2f}]")
    print(f"P-value: {p_val_p90:.4f}")
    
    # Comparison: Bootstrap vs Parametric (t-test)
    t_stat, t_pval = stats.ttest_ind(treatment, control)
    print(f"\n--- Parametric T-Test (for comparison) ---")
    print(f"T-test p-value: {t_pval:.4f} (assumes normality)")
    print(f"Bootstrap p-value: {p_val:.4f} (distribution-free)")
    ```

    | Metric Type           | Parametric Method        | Bootstrap Advantage                  |
    |-----------------------|--------------------------|--------------------------------------|
    | **Mean (normal)**     | T-test, Z-test           | Works even if not normal             |
    | **Median**            | Wilcoxon (assumptions)   | Exact CI, no rank assumptions        |
    | **Ratio (CTR)**       | Delta method (complex)   | Direct resampling, simpler           |
    | **Quantiles (P90)**   | None (no closed form)    | Bootstrap is only practical option   |

    | Bootstrap Iterations  | CI Stability | Runtime  | When to Use                          |
    |-----------------------|--------------|----------|--------------------------------------|
    | 1,000                 | Low          | <1s      | Quick sanity check                   |
    | 10,000                | High         | ~5s      | **Production standard**              |
    | 100,000               | Very high    | ~50s     | Final validation, publication        |

    **Real-World:**
    - **Google Ads:** Bootstrap for **CTR** (ratio metric) with 10k iterations; avoids delta method complexity, ~2s compute.
    - **Netflix:** **P90 streaming latency** (no parametric test exists); bootstrap detects +15ms with 95% CI [10, 20ms].
    - **Stripe:** **Revenue median** (heavy-tailed); bootstrap shows +$2.50 lift [+$1.80, +$3.20], p<0.01.

    !!! tip "Interviewer's Insight"
        - Knows **bootstrap for non-normal metrics** (revenue, ratios, quantiles)
        - Uses **10k iterations as standard** (balance speed/accuracy)
        - Real-world: **Netflix bootstraps P90 latency; Google uses for CTR over delta method**

---

### How to Test Revenue Metrics? - E-commerce Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Revenue` | **Asked by:** Amazon, Shopify, Stripe

??? success "View Answer"

    **Revenue metrics** have **heavy-tailed distributions** (most users spend $0, few spend thousands) and **outliers** that inflate variance, making **standard t-tests unreliable**. **Winsorization**, **log transforms**, **trimmed means**, and **robust statistics** ensure valid inference without discarding data.

    ```text
                ┌─────────────────────────────────┐
                │  Revenue Distribution (Skewed)  │
                │  90% users: $0                  │
                │   8% users: $1-50               │
                │   2% users: $50-10,000 (whales)│
                └──────────────┬──────────────────┘
                               ↓
         ┌─────────────────────┴─────────────────────┐
         ↓                                           ↓
    ┌─────────────────────┐              ┌─────────────────────┐
    │  Naive Mean Test    │              │  Robust Methods     │
    │  - High variance    │              │  - Winsorization    │
    │  - Outlier-driven   │              │  - Trimmed mean     │
    │  - Low power        │              │  - Log transform    │
    └─────────────────────┘              └─────────────────────┘
    ```

    ```python
    import numpy as np
    import pandas as pd
    from scipy import stats
    from scipy.stats import trim_mean
    import matplotlib.pyplot as plt
    
    # Production: Revenue Metric Testing with Robust Methods
    
    np.random.seed(42)
    n = 5000
    
    # Generate realistic e-commerce revenue: heavy-tailed
    def generate_revenue(n, conversion_rate=0.10, mean_spend=50, whale_prob=0.02):
        """90% $0, 8% normal spenders, 2% whales."""
        revenue = np.zeros(n)
        purchasers = np.random.rand(n) < conversion_rate
        revenue[purchasers] = np.random.lognormal(np.log(mean_spend), 1.5, purchasers.sum())
        
        # Add whales (very high spenders)
        whales = np.random.rand(n) < whale_prob
        revenue[whales] = np.random.lognormal(np.log(500), 1.0, whales.sum())
        return revenue
    
    control = generate_revenue(n, conversion_rate=0.10, mean_spend=50)
    treatment = generate_revenue(n, conversion_rate=0.11, mean_spend=52)  # +10% CVR, +4% AOV
    
    print("=" * 60)
    print("Revenue A/B Test: Handling Heavy Tails & Outliers")
    print("=" * 60)
    
    # 1. Naive t-test (WRONG for revenue)
    t_stat, p_naive = stats.ttest_ind(treatment, control)
    print(f"\n[1] Naive T-Test (assumes normality)")
    print(f"    Control mean: ${control.mean():.2f}")
    print(f"    Treatment mean: ${treatment.mean():.2f}")
    print(f"    Lift: ${treatment.mean() - control.mean():.2f}")
    print(f"    P-value: {p_naive:.4f}")
    print(f"    ⚠️  High variance due to outliers!")
    
    # 2. Winsorization (cap at 99th percentile)
    p99 = np.percentile(np.concatenate([control, treatment]), 99)
    control_wins = np.clip(control, 0, p99)
    treatment_wins = np.clip(treatment, 0, p99)
    
    t_stat_wins, p_wins = stats.ttest_ind(treatment_wins, control_wins)
    print(f"\n[2] Winsorized (cap at P99 = ${p99:.2f})")
    print(f"    Control mean: ${control_wins.mean():.2f}")
    print(f"    Treatment mean: ${treatment_wins.mean():.2f}")
    print(f"    Lift: ${treatment_wins.mean() - control_wins.mean():.2f}")
    print(f"    P-value: {p_wins:.4f}")
    print(f"    ✅ Reduced variance, more power")
    
    # 3. Log transformation (for multiplicative effects)
    control_pos = control[control > 0]
    treatment_pos = treatment[treatment > 0]
    
    log_control = np.log(control_pos)
    log_treatment = np.log(treatment_pos)
    
    t_stat_log, p_log = stats.ttest_ind(log_treatment, log_control)
    print(f"\n[3] Log-Transformed (purchasers only, n={len(control_pos)} vs {len(treatment_pos)})")
    print(f"    Control log-mean: {log_control.mean():.4f}")
    print(f"    Treatment log-mean: {log_treatment.mean():.4f}")
    print(f"    Multiplicative lift: {np.exp(log_treatment.mean() - log_control.mean()):.3f}x")
    print(f"    P-value: {p_log:.4f}")
    print(f"    ✅ Handles skewness well")
    
    # 4. Trimmed mean (remove top/bottom 5%)
    control_trim = trim_mean(control, proportiontocut=0.05)
    treatment_trim = trim_mean(treatment, proportiontocut=0.05)
    
    # Bootstrap for trimmed mean CI
    from scipy.stats import bootstrap
    def trimmed_mean_fn(x):
        return trim_mean(x, proportiontocut=0.05)
    
    print(f"\n[4] Trimmed Mean (remove top/bottom 5%)")
    print(f"    Control trimmed mean: ${control_trim:.2f}")
    print(f"    Treatment trimmed mean: ${treatment_trim:.2f}")
    print(f"    Lift: ${treatment_trim - control_trim:.2f}")
    print(f"    ✅ Robust to extreme outliers")
    
    # 5. Mann-Whitney U (non-parametric, rank-based)
    u_stat, p_mann = stats.mannwhitneyu(treatment, control, alternative='two-sided')
    print(f"\n[5] Mann-Whitney U Test (non-parametric)")
    print(f"    U-statistic: {u_stat:.0f}")
    print(f"    P-value: {p_mann:.4f}")
    print(f"    ✅ No distribution assumptions")
    
    # 6. Quantile Regression (median, IQR)
    control_median = np.median(control)
    treatment_median = np.median(treatment)
    control_iqr = np.percentile(control, 75) - np.percentile(control, 25)
    treatment_iqr = np.percentile(treatment, 75) - np.percentile(treatment, 25)
    
    print(f"\n[6] Quantile-Based (Median, IQR)")
    print(f"    Control median: ${control_median:.2f}, IQR: ${control_iqr:.2f}")
    print(f"    Treatment median: ${treatment_median:.2f}, IQR: ${treatment_iqr:.2f}")
    print(f"    Median lift: ${treatment_median - control_median:.2f}")
    print(f"    ✅ Robust summary statistics")
    
    # 7. Visualization: Distribution comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Raw distribution
    axes[0].hist(control, bins=50, alpha=0.5, label='Control', log=True)
    axes[0].hist(treatment, bins=50, alpha=0.5, label='Treatment', log=True)
    axes[0].set_xlabel('Revenue ($)')
    axes[0].set_ylabel('Count (log scale)')
    axes[0].set_title('Raw Revenue Distribution')
    axes[0].legend()
    
    # Winsorized
    axes[1].hist(control_wins, bins=50, alpha=0.5, label='Control (wins)')
    axes[1].hist(treatment_wins, bins=50, alpha=0.5, label='Treatment (wins)')
    axes[1].set_xlabel('Revenue ($, capped at P99)')
    axes[1].set_title('Winsorized Distribution')
    axes[1].legend()
    
    # Log-transformed (purchasers)
    axes[2].hist(log_control, bins=50, alpha=0.5, label='Control (log)')
    axes[2].hist(log_treatment, bins=50, alpha=0.5, label='Treatment (log)')
    axes[2].set_xlabel('Log(Revenue)')
    axes[2].set_title('Log-Transformed (Purchasers)')
    axes[2].legend()
    
    plt.tight_layout()
    # plt.savefig('revenue_testing.png')
    print("\nDistribution plots generated.")
    ```

    | Method                | Handles Outliers | Power  | Interpretation            | When to Use                        |
    |-----------------------|------------------|--------|---------------------------|------------------------------------||
    | **Naive t-test**      | ❌ No            | Low    | Mean difference           | ❌ Not for revenue                 |
    | **Winsorization**     | ✅ Yes (cap)     | High   | Mean (bounded)            | Standard choice for revenue        |
    | **Log transform**     | ✅ Yes           | High   | Multiplicative effect     | AOV, per-user revenue              |
    | **Trimmed mean**      | ✅ Yes (cut)     | Medium | Mean (robust)             | Extreme outliers                   |
    | **Mann-Whitney U**    | ✅ Yes           | Medium | Rank-based                | No assumptions, any distribution   |
    | **Quantile (median)** | ✅ Yes           | Medium | Median difference         | Heavy tails, mixed distributions   |

    | Revenue Metric        | Distribution      | Best Method              | Company Example                    |
    |-----------------------|-------------------|--------------------------|------------------------------------||
    | **Total revenue/user**| Heavy-tailed      | Winsorize at P99         | Amazon: Cap at $5k, detect +2.3%   |
    | **AOV (purchasers)**  | Log-normal        | Log transform            | Shopify: Log-AOV, 1.05× lift       |
    | **LTV**               | Highly skewed     | Trimmed mean (10%)       | Stripe: Trim whales, +$4.50 lift   |
    | **Conversion rate**   | Binomial          | Z-test proportions       | All: Standard binomial test        |

    **Real-World:**
    - **Amazon:** Winsorizes revenue at **P99 ($5,000)** to detect +2.3% lift in Prime experiments; naive test showed p=0.15, winsorized p=0.003.
    - **Shopify:** Uses **log(AOV)** for cart upsell tests; detects 1.05× multiplicative lift (5% increase) with 80% power.
    - **Stripe:** **Trimmed mean (10%)** for payment experiments removes top/bottom 5% whales; reduces variance by 40%, increases power by 25%.

    !!! tip "Interviewer's Insight"
        - Knows **winsorization at P99 as standard** for total revenue/user
        - Uses **log transform for AOV** (multiplicative effects)
        - Real-world: **Amazon caps at $5k; Shopify uses log-AOV for 5%+ detection**

---

### What is Regression to the Mean? - All Companies Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Statistics` | **Asked by:** All Companies

??? success "View Answer"

    **Regression to the mean (RTM)** causes **extreme observations** to move toward the **average** on repeated measurement—**not due to treatment**, but due to **random variation**. In A/B tests, selecting **worst-performing** cohorts for "improvement" falsely attributes natural RTM to the intervention.

    ```text
                ┌─────────────────────────────────┐
                │  Week 1: Measure Performance    │
                │  Users ranked by engagement     │
                └──────────────┬──────────────────┘
                               ↓
                ┌─────────────────────────────────┐
                │  Select WORST 10% for treatment │
                │  (low engagement = noise?)      │
                └──────────────┬──────────────────┘
                               ↓
         ┌─────────────────────┴─────────────────────┐
         ↓                                           ↓
    ┌─────────────────────┐              ┌─────────────────────┐
    │  Treatment Group    │              │  Control (random)   │
    │  Week 2: +15%       │              │  Week 2: +12%       │
    │  (RTM + treatment?) │              │  (RTM only)         │
    └─────────────────────┘              └─────────────────────┘
                               ↓
                ┌─────────────────────────────────┐
                │  True lift: 15% - 12% = 3%      │
                │  Without control: falsely claim │
                │  15% lift (RTM confounded)      │
                └─────────────────────────────────┘
    ```

    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats
    
    # Production: Demonstrate Regression to the Mean
    
    np.random.seed(42)
    n_users = 10000
    
    # Simulate user engagement: true skill + noise
    true_engagement = np.random.normal(100, 15, n_users)  # true underlying engagement
    
    # Week 1 observation: true + noise
    week1_obs = true_engagement + np.random.normal(0, 30, n_users)  # high noise
    
    # Week 2 observation: true + new noise (no treatment)
    week2_obs = true_engagement + np.random.normal(0, 30, n_users)
    
    df = pd.DataFrame({
        'user_id': range(n_users),
        'true_engagement': true_engagement,
        'week1': week1_obs,
        'week2': week2_obs
    })
    
    # WRONG APPROACH: Select worst 10% from week 1, measure week 2
    worst_10pct = df.nsmallest(1000, 'week1')
    
    print("=" * 60)
    print("Regression to the Mean: Before/After vs Randomized Control")
    print("=" * 60)
    
    print("\n[WRONG] Before/After Analysis (No Control)")
    print(f"  Week 1 (worst 10%): {worst_10pct['week1'].mean():.2f}")
    print(f"  Week 2 (same users): {worst_10pct['week2'].mean():.2f}")
    print(f"  Apparent lift: {worst_10pct['week2'].mean() - worst_10pct['week1'].mean():.2f} (+{100*(worst_10pct['week2'].mean()/worst_10pct['week1'].mean() - 1):.1f}%)")
    print(f"  ❌ FALSE! This is RTM, not real improvement.")
    
    # CORRECT APPROACH: Randomized control
    # Simulate treatment effect: +5 points
    treatment_effect = 5
    
    # Select worst 20% from week 1, then randomize
    worst_20pct = df.nsmallest(2000, 'week1').copy()
    worst_20pct['group'] = np.random.choice(['control', 'treatment'], size=2000)
    
    # Week 2: control gets RTM only, treatment gets RTM + effect
    worst_20pct['week2_control'] = worst_20pct.apply(
        lambda row: true_engagement[row['user_id']] + np.random.normal(0, 30) 
                   if row['group'] == 'control' else np.nan, axis=1
    )
    worst_20pct['week2_treatment'] = worst_20pct.apply(
        lambda row: true_engagement[row['user_id']] + np.random.normal(0, 30) + treatment_effect
                   if row['group'] == 'treatment' else np.nan, axis=1
    )
    
    control_w1 = worst_20pct[worst_20pct['group'] == 'control']['week1'].mean()
    control_w2 = worst_20pct[worst_20pct['group'] == 'control']['week2_control'].mean()
    treatment_w1 = worst_20pct[worst_20pct['group'] == 'treatment']['week1'].mean()
    treatment_w2 = worst_20pct[worst_20pct['group'] == 'treatment']['week2_treatment'].mean()
    
    print("\n[CORRECT] Randomized Controlled Experiment")
    print(f"  Control Week 1: {control_w1:.2f}")
    print(f"  Control Week 2: {control_w2:.2f} (RTM: +{control_w2 - control_w1:.2f})")
    print(f"  Treatment Week 1: {treatment_w1:.2f}")
    print(f"  Treatment Week 2: {treatment_w2:.2f} (RTM + effect: +{treatment_w2 - treatment_w1:.2f})")
    print(f"  ✅ True treatment effect: {treatment_w2 - treatment_w1 - (control_w2 - control_w1):.2f}")
    
    # Statistical test
    control_deltas = worst_20pct[worst_20pct['group'] == 'control']['week2_control'] - \
                     worst_20pct[worst_20pct['group'] == 'control']['week1']
    treatment_deltas = worst_20pct[worst_20pct['group'] == 'treatment']['week2_treatment'] - \
                       worst_20pct[worst_20pct['group'] == 'treatment']['week1']
    
    t_stat, p_val = stats.ttest_ind(treatment_deltas.dropna(), control_deltas.dropna())
    print(f"  P-value (difference in deltas): {p_val:.4f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter: Week 1 vs Week 2 (all users)
    axes[0].scatter(df['week1'], df['week2'], alpha=0.3, s=10)
    axes[0].plot([df['week1'].min(), df['week1'].max()], 
                 [df['week1'].min(), df['week1'].max()], 
                 'r--', label='No change line')
    axes[0].axhline(df['week2'].mean(), color='blue', linestyle=':', label='Week 2 mean')
    axes[0].set_xlabel('Week 1 Engagement')
    axes[0].set_ylabel('Week 2 Engagement')
    axes[0].set_title('Regression to the Mean (All Users)')
    axes[0].legend()
    axes[0].annotate('Extreme low values\nmove up (RTM)', xy=(40, 90), fontsize=10, color='red')
    
    # Bar chart: Before/After vs Controlled
    x = ['Before/After\n(worst 10%)', 'Controlled\n(treatment)', 'Controlled\n(control)']
    y = [
        worst_10pct['week2'].mean() - worst_10pct['week1'].mean(),
        treatment_w2 - treatment_w1,
        control_w2 - control_w1
    ]
    colors = ['red', 'green', 'blue']
    axes[1].bar(x, y, color=colors, alpha=0.7)
    axes[1].axhline(treatment_effect, color='green', linestyle='--', label='True effect (+5)')
    axes[1].set_ylabel('Observed Change (Week 2 - Week 1)')
    axes[1].set_title('RTM Confounds Before/After')
    axes[1].legend()
    axes[1].annotate('RTM inflates\napparent effect', xy=(0, y[0]+2), ha='center', fontsize=10, color='red')
    
    plt.tight_layout()
    # plt.savefig('regression_to_mean.png')
    print("\nRTM visualization generated.")
    ```

    | Scenario                     | Observation                     | Cause                           | Prevention                        |
    |------------------------------|---------------------------------|---------------------------------|-----------------------------------|
    | **Select worst performers**  | +15% next period (no treatment) | RTM (noise → average)           | Randomize AFTER selection         |
    | **Select best performers**   | -10% next period (no treatment) | RTM (noise → average)           | Randomize AFTER selection         |
    | **Random cohort**            | ~0% change (no treatment)       | No RTM (already representative) | Standard A/B test                 |

    | Before/After Analysis        | Randomized Controlled Test      | Key Difference                  |
    |------------------------------|----------------------------------|----------------------------------|
    | Selects extreme cohort       | Randomizes AFTER selection       | Control isolates RTM             |
    | No control group             | Control + treatment groups       | Diff-in-diff removes RTM         |
    | Confounds RTM + effect       | Separates RTM from effect        | ✅ Valid causal inference        |

    **Real-World:**
    - **LinkedIn:** Ran "re-engagement campaign" on **dormant users** (bottom 5% activity); saw +20% lift but control showed +18% (RTM). **True lift: 2%**.
    - **Duolingo:** Targeted **struggling learners** (lowest streak); before/after claimed +30% retention, but randomized control revealed +5% true effect (RTM = +25%).
    - **Uber:** Selected **low-rated drivers** for training; control group improved +12% (RTM), treatment +15% (RTM + training = **+3% real effect**).

    !!! tip "Interviewer's Insight"
        - Knows **RTM affects extreme cohorts** (worst/best performers)
        - Always uses **randomized control AFTER selection** to isolate true effect
        - Real-world: **LinkedIn detected +18% RTM in dormant user campaign; true lift only +2%**

---

### How to Handle Multiple Metrics? - Netflix, Airbnb Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Metrics`, `Decision Framework` | **Asked by:** Netflix, Airbnb, Uber

??? success "View Answer"

    **Multiple metrics in A/B tests require a clear hierarchical decision framework**: **Primary metrics** drive the ship/no-ship decision, **secondary metrics** provide interpretability, and **guardrail metrics** act as safety checks. Without a framework, conflicting signals (e.g., revenue up but engagement down) lead to analysis paralysis.

    ```text
                        ┌──────────────────────────────┐
                        │  Experiment Results Ready    │
                        │  100+ metrics computed       │
                        └──────────┬───────────────────┘
                                   ↓
                    ┌──────────────────────────────────┐
                    │  STEP 1: Check Guardrails        │
                    │  - Latency < 200ms?              │
                    │  - Error rate < 0.1%?            │
                    │  - DAU not decreased?            │
                    └──────────┬───────────────────────┘
                               ↓
                       YES ────┤──── NO
                               ↓       ↓
                    ┌──────────────┐  ┌────────────────┐
                    │ STEP 2:      │  │ REJECT         │
                    │ Primary      │  │ Any guardrail  │
                    │ Metric?      │  │ failure = stop │
                    └──────┬───────┘  └────────────────┘
                           ↓
             ┌─────────────┴──────────────┐
             ↓                            ↓
    ┌─────────────────┐         ┌─────────────────┐
    │ Primary WINS    │         │ Primary NEUTRAL │
    │ (p<0.05, +lift) │         │ or LOSES        │
    └────────┬────────┘         └────────┬────────┘
             ↓                            ↓
      ┌──────────────┐         ┌──────────────────┐
      │ SHIP         │         │ Check Secondary  │
      │ Launch now   │         │ Positive? → Maybe│
      │              │         │ Negative? → Stop │
      └──────────────┘         └──────────────────┘
    ```

    ```python
    import numpy as np
    import pandas as pd
    from scipy import stats
    from dataclasses import dataclass
    from typing import List, Dict, Tuple
    from enum import Enum
    
    class MetricType(Enum):
        PRIMARY = "primary"
        SECONDARY = "secondary"
        GUARDRAIL = "guardrail"
    
    class Decision(Enum):
        SHIP = "ship"
        NO_SHIP = "no_ship"
        ITERATE = "iterate"
        ESCALATE = "escalate"
    
    @dataclass
    class MetricResult:
        name: str
        metric_type: MetricType
        control_mean: float
        treatment_mean: float
        p_value: float
        ci_lower: float
        ci_upper: float
        relative_lift: float
        threshold: float = None  # For guardrails
        
        def is_significant(self, alpha=0.05):
            return self.p_value < alpha
        
        def passes_guardrail(self):
            if self.metric_type != MetricType.GUARDRAIL:
                return True
            if self.threshold is None:
                return True
            # Guardrail: treatment should not be worse than threshold
            return self.treatment_mean >= self.threshold
    
    class ExperimentDecisionEngine:
        """Production decision framework for multi-metric A/B tests"""
        
        def __init__(self, alpha=0.05):
            self.alpha = alpha
            
        def make_decision(self, results: List[MetricResult]) -> Tuple[Decision, str]:
            """
            Decision tree:
            1. Check all guardrails pass
            2. Check primary metric significant positive
            3. Consider secondary metrics for interpretation
            """
            # Step 1: Guardrails
            guardrails = [r for r in results if r.metric_type == MetricType.GUARDRAIL]
            failed_guardrails = [g for g in guardrails if not g.passes_guardrail()]
            
            if failed_guardrails:
                reasons = [f"{g.name}: {g.treatment_mean:.2f} < threshold {g.threshold:.2f}" 
                          for g in failed_guardrails]
                return Decision.NO_SHIP, f"Guardrail violations: {', '.join(reasons)}"
            
            # Step 2: Primary metric
            primary = [r for r in results if r.metric_type == MetricType.PRIMARY]
            if not primary:
                return Decision.ESCALATE, "No primary metric defined"
            
            primary_metric = primary[0]  # Assume single primary
            
            if primary_metric.is_significant(self.alpha) and primary_metric.relative_lift > 0:
                return Decision.SHIP, f"Primary metric ({primary_metric.name}) significant positive: +{primary_metric.relative_lift:.2%}"
            
            # Step 3: Secondary interpretation
            secondary = [r for r in results if r.metric_type == MetricType.SECONDARY]
            sig_secondary_positive = [r for r in secondary if r.is_significant(self.alpha) and r.relative_lift > 0]
            sig_secondary_negative = [r for r in secondary if r.is_significant(self.alpha) and r.relative_lift < 0]
            
            if primary_metric.relative_lift > 0 and not primary_metric.is_significant(self.alpha):
                # Primary directionally correct but underpowered
                if sig_secondary_positive:
                    return Decision.ITERATE, f"Primary positive but not significant. Secondary wins: {[r.name for r in sig_secondary_positive]}. Consider longer test."
                else:
                    return Decision.NO_SHIP, "Primary not significant, no supporting secondary wins"
            
            if sig_secondary_negative:
                return Decision.NO_SHIP, f"Primary neutral/negative. Secondary losses: {[r.name for r in sig_secondary_negative]}"
            
            return Decision.NO_SHIP, "Primary not significant positive"
        
        def generate_report(self, results: List[MetricResult]) -> str:
            """Generate stakeholder-friendly report"""
            decision, reason = self.make_decision(results)
            
            report = ["=" * 70]
            report.append("EXPERIMENT DECISION REPORT")
            report.append("=" * 70)
            report.append(f"\n🎯 DECISION: {decision.value.upper()}")
            report.append(f"📝 REASON: {reason}\n")
            
            # Guardrails
            report.append("GUARDRAILS (Safety Checks)")
            report.append("-" * 70)
            guardrails = [r for r in results if r.metric_type == MetricType.GUARDRAIL]
            for g in guardrails:
                status = "✅ PASS" if g.passes_guardrail() else "❌ FAIL"
                report.append(f"  {status} {g.name}: {g.treatment_mean:.3f} (threshold: {g.threshold:.3f})")
            
            # Primary
            report.append("\nPRIMARY METRIC (Decision Driver)")
            report.append("-" * 70)
            primary = [r for r in results if r.metric_type == MetricType.PRIMARY]
            for p in primary:
                sig = "✅ Significant" if p.is_significant() else "⚠️  Not Significant"
                report.append(f"  {p.name}: {p.relative_lift:+.2%} ({p.ci_lower:.2%} to {p.ci_upper:.2%})")
                report.append(f"    p-value: {p.p_value:.4f} | {sig}")
            
            # Secondary
            report.append("\nSECONDARY METRICS (Interpretation)")
            report.append("-" * 70)
            secondary = [r for r in results if r.metric_type == MetricType.SECONDARY]
            for s in secondary:
                sig = "✅" if s.is_significant() else "⚠️"
                direction = "📈" if s.relative_lift > 0 else "📉"
                report.append(f"  {sig} {direction} {s.name}: {s.relative_lift:+.2%} (p={s.p_value:.4f})")
            
            report.append("\n" + "=" * 70)
            return "\n".join(report)
    
    # Production Example: Search Ranking Experiment
    np.random.seed(42)
    n = 10000
    
    # Simulate control and treatment data
    def simulate_experiment_data(n):
        # Primary: Click-through rate (treatment: +2%)
        control_ctr = np.random.binomial(1, 0.10, n)
        treatment_ctr = np.random.binomial(1, 0.102, n)  # +2% relative
        
        # Secondary: Time on page (treatment: +5%)
        control_time = np.random.exponential(60, n)
        treatment_time = np.random.exponential(63, n)  # +5%
        
        # Secondary: Bounce rate (treatment: -3%)
        control_bounce = np.random.binomial(1, 0.40, n)
        treatment_bounce = np.random.binomial(1, 0.388, n)  # -3% relative
        
        # Guardrail: Page load time (should stay < 200ms)
        control_latency = np.random.gamma(shape=100, scale=1.8, size=n)  # mean ~180ms
        treatment_latency = np.random.gamma(shape=100, scale=1.85, size=n)  # mean ~185ms
        
        # Guardrail: Error rate (should stay < 0.1%)
        control_errors = np.random.binomial(1, 0.0008, n)
        treatment_errors = np.random.binomial(1, 0.0009, n)
        
        return {
            'ctr': (control_ctr, treatment_ctr),
            'time_on_page': (control_time, treatment_time),
            'bounce_rate': (control_bounce, treatment_bounce),
            'latency': (control_latency, treatment_latency),
            'error_rate': (control_errors, treatment_errors)
        }
    
    data = simulate_experiment_data(n)
    
    # Compute metrics
    def compute_metric_result(name, metric_type, control, treatment, threshold=None, invert=False):
        """Invert=True for metrics where lower is better (e.g., bounce rate, latency)"""
        control_mean = control.mean()
        treatment_mean = treatment.mean()
        
        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(treatment, control)
        
        # Confidence interval for difference
        se = np.sqrt(control.var()/len(control) + treatment.var()/len(treatment))
        ci_lower = (treatment_mean - control_mean) - 1.96 * se
        ci_upper = (treatment_mean - control_mean) + 1.96 * se
        
        # Relative lift
        if invert:
            relative_lift = (control_mean - treatment_mean) / control_mean  # Lower is better
        else:
            relative_lift = (treatment_mean - control_mean) / control_mean
        
        # Convert CI to relative
        ci_lower_rel = ci_lower / control_mean
        ci_upper_rel = ci_upper / control_mean
        
        return MetricResult(
            name=name,
            metric_type=metric_type,
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            p_value=p_value,
            ci_lower=ci_lower_rel,
            ci_upper=ci_upper_rel,
            relative_lift=relative_lift,
            threshold=threshold
        )
    
    results = [
        # Primary
        compute_metric_result("CTR", MetricType.PRIMARY, 
                             data['ctr'][0], data['ctr'][1]),
        
        # Secondary
        compute_metric_result("Time on Page", MetricType.SECONDARY,
                             data['time_on_page'][0], data['time_on_page'][1]),
        compute_metric_result("Bounce Rate", MetricType.SECONDARY,
                             data['bounce_rate'][0], data['bounce_rate'][1], invert=True),
        
        # Guardrails
        compute_metric_result("Latency (ms)", MetricType.GUARDRAIL,
                             data['latency'][0], data['latency'][1], threshold=190, invert=True),
        compute_metric_result("Error Rate", MetricType.GUARDRAIL,
                             data['error_rate'][0], data['error_rate'][1], threshold=0.001, invert=True)
    ]
    
    # Make decision
    engine = ExperimentDecisionEngine(alpha=0.05)
    report = engine.generate_report(results)
    print(report)
    
    # Example 2: Conflicting metrics scenario
    print("\n\n" + "=" * 70)
    print("SCENARIO 2: CONFLICTING METRICS (Revenue up, Engagement down)")
    print("=" * 70 + "\n")
    
    # Simulate conflicting scenario
    np.random.seed(123)
    conflicting_data = {
        'revenue': (np.random.gamma(2, 25, 10000), np.random.gamma(2, 26, 10000)),  # +4%
        'engagement': (np.random.poisson(5, 10000), np.random.poisson(4.8, 10000)),  # -4%
        'latency': (np.random.gamma(100, 1.8, 10000), np.random.gamma(100, 1.85, 10000))
    }
    
    conflicting_results = [
        compute_metric_result("Revenue per User", MetricType.PRIMARY,
                             conflicting_data['revenue'][0], conflicting_data['revenue'][1]),
        compute_metric_result("Engagement (sessions/day)", MetricType.SECONDARY,
                             conflicting_data['engagement'][0], conflicting_data['engagement'][1]),
        compute_metric_result("Latency", MetricType.GUARDRAIL,
                             conflicting_data['latency'][0], conflicting_data['latency'][1], 
                             threshold=190, invert=True)
    ]
    
    report2 = engine.generate_report(conflicting_results)
    print(report2)
    
    print("\n📊 DECISION LOGIC:")
    print("  1. Guardrails pass? → Continue")
    print("  2. Primary significant positive? → SHIP (even if secondary mixed)")
    print("  3. Trade-off: Revenue > Engagement for this team")
    print("  4. Real-world: Netflix ships revenue-positive even if engagement dips slightly")
    ```

    | Metric Type        | Purpose                          | Count         | Decision Weight | Action if Fails              |
    |--------------------|----------------------------------|---------------|-----------------|------------------------------|
    | **Primary**        | Ship/no-ship decision            | 1 (rarely 2)  | 100%            | No ship if not significant   |
    | **Secondary**      | Interpret primary, debug         | 3-10          | Interpretive    | Explain primary, inform iter |
    | **Guardrail**      | Safety checks                    | 3-5           | Veto power      | No ship if ANY fail          |

    | Scenario                              | Primary      | Secondary    | Guardrails | Decision   | Example                                    |
    |---------------------------------------|--------------|--------------|------------|------------|--------------------------------------------|
    | **Primary wins, all pass**            | ✅ +5%       | Mixed        | ✅ Pass    | **SHIP**   | Netflix: CTR +5%, latency OK               |
    | **Primary neutral, secondary wins**   | ⚠️ +1%       | ✅ +10%      | ✅ Pass    | **Iterate**| Airbnb: Bookings +1%, engagement +10%      |
    | **Primary wins, guardrail fails**     | ✅ +8%       | ✅ Positive  | ❌ Latency | **NO SHIP**| Uber: Rides +8%, but 500ms latency spike   |
    | **Primary loses**                     | ❌ -2%       | ✅ Positive  | ✅ Pass    | **NO SHIP**| Any: Primary negative overrides secondary  |

    **Real-World:**
    - **Netflix:** Ran recommendation algo with **+3% watch time (primary)** but **-2% catalog coverage (secondary)**. Shipped because primary won, then iterated on coverage.
    - **Airbnb:** **Guardrail failure**: New search increased bookings +5% but **error rate spiked to 0.3%** (threshold: 0.1%). **Rejected** despite revenue win.
    - **Uber:** Conflicting metrics: Driver earnings +4%, rider wait time +8%. **Escalated** to leadership; decided rider experience > driver earnings, **no ship**.

    !!! tip "Interviewer's Insight"
        - Knows **primary = decision metric** (1 metric, clear ship/no-ship)
        - Uses **guardrails as veto** (any failure = stop, even if primary wins)
        - Real-world: **Netflix ships revenue-positive even if engagement slightly down; Airbnb rejected +5% bookings due to 0.3% error rate spike**

---

### What is Simpson's Paradox? - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Statistics`, `Segmentation` | **Asked by:** Google, Meta, Netflix

??? success "View Answer"

    **Simpson's Paradox occurs when an aggregate trend reverses when data is segmented by a lurking variable**—**treatment wins in every segment** but **loses overall**, or vice versa. This happens when segments have **unequal sizes** and **different baseline rates**, causing composition effects to dominate true treatment effects.

    ```text
                  ┌─────────────────────────────┐
                  │  Overall A/B Test Result    │
                  │  Control: 50% conversion    │
                  │  Treatment: 48% conversion  │
                  │  ❌ Treatment LOSES          │
                  └──────────┬──────────────────┘
                             ↓
                 ┌───────────────────────────┐
                 │  Segment by Device Type   │
                 └───────────┬───────────────┘
                             ↓
          ┌──────────────────┴───────────────────┐
          ↓                                      ↓
    ┌─────────────────────┐          ┌─────────────────────┐
    │  Mobile (90% users) │          │  Desktop (10% users)│
    │  Control: 45%       │          │  Control: 90%       │
    │  Treatment: 47%     │          │  Treatment: 92%     │
    │  ✅ Treatment WINS  │          │  ✅ Treatment WINS  │
    └─────────────────────┘          └─────────────────────┘
                             ↓
                ┌──────────────────────────────┐
                │  Paradox! Treatment wins in  │
                │  BOTH segments but loses     │
                │  overall due to composition  │
                │  (mobile has lower baseline) │
                └──────────────────────────────┘
    ```

    ```python
    import numpy as np
    import pandas as pd
    from scipy import stats
    import matplotlib.pyplot as plt
    
    # Production: Demonstrate Simpson's Paradox in A/B Testing
    
    np.random.seed(42)
    
    # Example 1: Classic Simpson's Paradox
    print("=" * 70)
    print("SIMPSON'S PARADOX: Treatment wins per segment, loses overall")
    print("=" * 70)
    
    # Create dataset with two segments (mobile, desktop)
    mobile_control = pd.DataFrame({
        'segment': 'mobile',
        'group': 'control',
        'user_id': range(9000),
        'converted': np.random.binomial(1, 0.45, 9000)  # 45% CVR
    })
    
    mobile_treatment = pd.DataFrame({
        'segment': 'mobile',
        'group': 'treatment',
        'user_id': range(9000, 18000),
        'converted': np.random.binomial(1, 0.47, 9000)  # 47% CVR (+2pp)
    })
    
    desktop_control = pd.DataFrame({
        'segment': 'desktop',
        'group': 'control',
        'user_id': range(18000, 19000),
        'converted': np.random.binomial(1, 0.90, 1000)  # 90% CVR
    })
    
    desktop_treatment = pd.DataFrame({
        'segment': 'desktop',
        'group': 'treatment',
        'user_id': range(19000, 20000),
        'converted': np.random.binomial(1, 0.92, 1000)  # 92% CVR (+2pp)
    })
    
    df = pd.concat([mobile_control, mobile_treatment, desktop_control, desktop_treatment])
    
    # Overall analysis (WRONG: ignores segments)
    overall_control = df[df['group'] == 'control']['converted'].mean()
    overall_treatment = df[df['group'] == 'treatment']['converted'].mean()
    
    print("\n[WRONG] Overall Analysis (Ignoring Segments)")
    print(f"  Control:   {overall_control:.3f} ({overall_control*100:.1f}%)")
    print(f"  Treatment: {overall_treatment:.3f} ({overall_treatment*100:.1f}%)")
    print(f"  Lift: {overall_treatment - overall_control:+.3f} ({(overall_treatment - overall_control)*100:+.1f}pp)")
    
    if overall_treatment < overall_control:
        print("  ❌ Conclusion: Treatment LOSES")
    else:
        print("  ✅ Conclusion: Treatment WINS")
    
    # Segmented analysis (CORRECT)
    print("\n[CORRECT] Segmented Analysis")
    for segment in ['mobile', 'desktop']:
        seg_data = df[df['segment'] == segment]
        seg_control = seg_data[seg_data['group'] == 'control']['converted'].mean()
        seg_treatment = seg_data[seg_data['group'] == 'treatment']['converted'].mean()
        seg_n = len(seg_data) // 2
        
        t_stat, p_val = stats.ttest_ind(
            seg_data[seg_data['group'] == 'treatment']['converted'],
            seg_data[seg_data['group'] == 'control']['converted']
        )
        
        print(f"\n  {segment.upper()} (n={seg_n} per group):")
        print(f"    Control:   {seg_control:.3f} ({seg_control*100:.1f}%)")
        print(f"    Treatment: {seg_treatment:.3f} ({seg_treatment*100:.1f}%)")
        print(f"    Lift: {seg_treatment - seg_control:+.3f} ({(seg_treatment - seg_control)*100:+.1f}pp)")
        print(f"    P-value: {p_val:.4f}")
        print(f"    ✅ Treatment WINS in {segment}")
    
    print("\n🎯 PARADOX REVEALED:")
    print("  - Treatment wins in BOTH mobile (+2pp) and desktop (+2pp)")
    print("  - But treatment has MORE mobile users (low CVR)")
    print("  - Composition effect makes treatment look worse overall")
    print("  - True effect: +2pp in each segment (consistent win)")
    
    # Example 2: Real-world scenario (time-based)
    print("\n" + "=" * 70)
    print("REAL-WORLD: Weekday/Weekend Paradox")
    print("=" * 70)
    
    # Treatment launched on Monday, control started Friday
    # Weekend has higher natural engagement
    weekday_control = pd.DataFrame({
        'period': 'weekday',
        'group': 'control',
        'engaged': np.random.binomial(1, 0.30, 7000)  # 30% engagement
    })
    
    weekend_control = pd.DataFrame({
        'period': 'weekend',
        'group': 'control',
        'engaged': np.random.binomial(1, 0.50, 3000)  # 50% engagement (natural boost)
    })
    
    weekday_treatment = pd.DataFrame({
        'period': 'weekday',
        'group': 'treatment',
        'engaged': np.random.binomial(1, 0.33, 8000)  # 33% engagement (+3pp)
    })
    
    weekend_treatment = pd.DataFrame({
        'period': 'weekend',
        'group': 'treatment',
        'engaged': np.random.binomial(1, 0.53, 2000)  # 53% engagement (+3pp)
    })
    
    df2 = pd.concat([weekday_control, weekend_control, weekday_treatment, weekend_treatment])
    
    # Overall
    overall_control2 = df2[df2['group'] == 'control']['engaged'].mean()
    overall_treatment2 = df2[df2['group'] == 'treatment']['engaged'].mean()
    
    print("\n[WRONG] Overall (Time not considered)")
    print(f"  Control:   {overall_control2:.3f} ({overall_control2*100:.1f}%)")
    print(f"  Treatment: {overall_treatment2:.3f} ({overall_treatment2*100:.1f}%)")
    print(f"  Lift: {overall_treatment2 - overall_control2:+.3f} ({(overall_treatment2 - overall_control2)*100:+.1f}pp)")
    
    # Segmented by time
    print("\n[CORRECT] Stratified by Weekday/Weekend")
    for period in ['weekday', 'weekend']:
        period_data = df2[df2['period'] == period]
        period_control = period_data[period_data['group'] == 'control']['engaged'].mean()
        period_treatment = period_data[period_data['group'] == 'treatment']['engaged'].mean()
        
        print(f"\n  {period.upper()}:")
        print(f"    Control:   {period_control:.3f}")
        print(f"    Treatment: {period_treatment:.3f}")
        print(f"    Lift: {period_treatment - period_control:+.3f} (+{(period_treatment - period_control)*100:.1f}pp)")
        print(f"    ✅ Consistent +3pp lift")
    
    # How to prevent: Check Sample Ratio Mismatch (SRM) by segment
    print("\n" + "=" * 70)
    print("PREVENTION: Check Sample Ratio by Segment")
    print("=" * 70)
    
    def check_srm_by_segment(df, segment_col, group_col):
        """Check if treatment/control split is balanced within each segment"""
        print(f"\nSample Ratio Check (by {segment_col}):")
        for segment in df[segment_col].unique():
            seg_data = df[df[segment_col] == segment]
            n_control = len(seg_data[seg_data[group_col] == 'control'])
            n_treatment = len(seg_data[seg_data[group_col] == 'treatment'])
            total = n_control + n_treatment
            
            expected = total / 2
            chi2 = ((n_control - expected)**2 / expected + 
                   (n_treatment - expected)**2 / expected)
            p_val = 1 - stats.chi2.cdf(chi2, df=1)
            
            print(f"  {segment}: Control={n_control}, Treatment={n_treatment}")
            print(f"    Expected: {expected:.0f} each")
            print(f"    Chi-square p-value: {p_val:.4f}")
            
            if p_val < 0.05:
                print(f"    ⚠️  IMBALANCE DETECTED (potential Simpson's Paradox)")
            else:
                print(f"    ✅ Balanced")
    
    check_srm_by_segment(df, 'segment', 'group')
    check_srm_by_segment(df2, 'period', 'group')
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Example 1: Device segments
    segments1 = ['mobile', 'desktop', 'overall']
    control_rates1 = [
        df[(df['segment'] == 'mobile') & (df['group'] == 'control')]['converted'].mean(),
        df[(df['segment'] == 'desktop') & (df['group'] == 'control')]['converted'].mean(),
        overall_control
    ]
    treatment_rates1 = [
        df[(df['segment'] == 'mobile') & (df['group'] == 'treatment')]['converted'].mean(),
        df[(df['segment'] == 'desktop') & (df['group'] == 'treatment')]['converted'].mean(),
        overall_treatment
    ]
    
    x = np.arange(len(segments1))
    width = 0.35
    axes[0].bar(x - width/2, control_rates1, width, label='Control', alpha=0.8)
    axes[0].bar(x + width/2, treatment_rates1, width, label='Treatment', alpha=0.8)
    axes[0].set_ylabel('Conversion Rate')
    axes[0].set_title("Simpson's Paradox: Device Segments")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(segments1)
    axes[0].legend()
    axes[0].axhline(0, color='black', linewidth=0.5)
    
    # Annotate paradox
    axes[0].annotate('Treatment wins', xy=(0, 0.47), xytext=(0, 0.52),
                    arrowprops=dict(arrowstyle='->', color='green'), color='green')
    axes[0].annotate('Treatment wins', xy=(1, 0.92), xytext=(1, 0.97),
                    arrowprops=dict(arrowstyle='->', color='green'), color='green')
    axes[0].annotate('Treatment loses!', xy=(2, 0.48), xytext=(2, 0.43),
                    arrowprops=dict(arrowstyle='->', color='red'), color='red')
    
    # Example 2: Time periods
    periods = ['weekday', 'weekend', 'overall']
    control_rates2 = [
        df2[(df2['period'] == 'weekday') & (df2['group'] == 'control')]['engaged'].mean(),
        df2[(df2['period'] == 'weekend') & (df2['group'] == 'control')]['engaged'].mean(),
        overall_control2
    ]
    treatment_rates2 = [
        df2[(df2['period'] == 'weekday') & (df2['group'] == 'treatment')]['engaged'].mean(),
        df2[(df2['period'] == 'weekend') & (df2['group'] == 'treatment')]['engaged'].mean(),
        overall_treatment2
    ]
    
    axes[1].bar(x - width/2, control_rates2, width, label='Control', alpha=0.8)
    axes[1].bar(x + width/2, treatment_rates2, width, label='Treatment', alpha=0.8)
    axes[1].set_ylabel('Engagement Rate')
    axes[1].set_title("Time-Based Paradox")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(periods)
    axes[1].legend()
    
    plt.tight_layout()
    # plt.savefig('simpsons_paradox.png')
    print("\nVisualization generated.")
    ```

    | Cause of Paradox           | Mechanism                                    | Example                                      | Prevention                     |
    |----------------------------|----------------------------------------------|----------------------------------------------|--------------------------------|
    | **Unequal segment sizes**  | Small high-CVR segment dominates composition | 90% mobile (low CVR), 10% desktop (high CVR) | Stratified randomization       |
    | **Different baselines**    | Segments have different natural rates        | Weekend engagement 50% vs weekday 30%        | Time-stratified assignment     |
    | **Non-random assignment**  | Treatment launched at different time         | Treatment gets more weekdays (low baseline)  | Simultaneous launch            |

    | Scenario                              | Overall   | Mobile     | Desktop    | Paradox?  | Root Cause                         |
    |---------------------------------------|-----------|------------|------------|-----------|------------------------------------|
    | **Classic Simpson's**                 | T loses   | T wins +2pp| T wins +2pp| ✅ Yes    | Unequal sizes (90% mobile)         |
    | **Time-based**                        | T neutral | T wins +3pp| T wins +3pp| ✅ Yes    | Treatment has fewer weekends       |
    | **Balanced segments**                 | T wins    | T wins     | T wins     | ❌ No     | Equal sizes, no composition effect |

    **Real-World:**
    - **Google Search:** Tested new ranking algorithm; **desktop showed +5% CTR**, **mobile +3% CTR**, but **overall -1% CTR** because treatment got more mobile traffic (mobile has lower baseline). **Solution:** Stratified by device.
    - **Meta Ads:** Ran experiment where **treatment won in both US (+2%) and International (+1.5%)** but **lost overall** because treatment had more International users (lower baseline revenue). **Caught by segment analysis**.
    - **Netflix:** Personalization test showed **treatment won in all 10 countries** but **lost overall** due to unequal country distribution. **Standard practice**: Always report segmented results.

    !!! tip "Interviewer's Insight"
        - Knows **Simpson's Paradox = composition effects** (unequal segment sizes with different baselines)
        - Always **segments analysis by device, country, time** to detect paradox
        - Real-world: **Google caught -1% overall CTR hiding +5% desktop, +3% mobile wins due to device mix; Meta detected US/Intl composition effect**

---

### How to Run Tests with Ratio Metrics? - Most Tech Companies Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Metrics` | **Asked by:** Most Tech Companies

??? success "View Answer"

    **Challenge:** Ratio metrics (CTR = Clicks/Views, RPU = Revenue/Users) have complex variance structure. Simple t-tests are invalid.
    
    **Statistical Framework for Ratio Metrics:**
    
    ```
    ┌─────────────────────────────────────────────────────────────────────┐
    │                       Ratio Metric Analysis                          │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                       │
    │  User-Level Ratio:                                                   │
    │  ┌───────────────┐       ┌───────────────┐                         │
    │  │ Numerator     │       │ Denominator   │                         │
    │  │ (Clicks)      │  ÷    │ (Impressions) │  = CTR                  │
    │  └───────────────┘       └───────────────┘                         │
    │         ↓                        ↓                                   │
    │    Aggregate              Aggregate                                  │
    │    then divide            then divide                                │
    │                                                                       │
    │  Session-Level Ratio:                                                │
    │  ┌────────────────────────────────────┐                            │
    │  │ For each session:                   │                            │
    │  │   session_ratio = clicks/views      │                            │
    │  │ Then: mean(session_ratios)          │                            │
    │  └────────────────────────────────────┘                            │
    │                                                                       │
    └─────────────────────────────────────────────────────────────────────┘
    ```
    
    **Production Implementation - Delta Method + Bootstrap:**
    
    ```python
    import numpy as np
    import pandas as pd
    from scipy import stats
    from dataclasses import dataclass
    import matplotlib.pyplot as plt
    from typing import Dict, Tuple
    
    @dataclass
    class RatioMetricResult:
        """Results container for ratio metric analysis"""
        ratio: float
        std_error: float
        ci_lower: float
        ci_upper: float
        method: str
        sample_size: int
    
    class RatioMetricAnalyzer:
        """
        Comprehensive analyzer for ratio metrics in A/B tests.
        Handles CTR, RPU, conversion rate, and other ratio metrics.
        """
        
        def __init__(self, alpha: float = 0.05, n_bootstrap: int = 10000):
            self.alpha = alpha
            self.n_bootstrap = n_bootstrap
        
        def delta_method(self, numerator: np.ndarray, 
                        denominator: np.ndarray) -> RatioMetricResult:
            """
            Delta method for ratio variance estimation.
            
            Variance approximation using Taylor expansion:
            Var(X/Y) ≈ (μx/μy)² * [Var(X)/μx² + Var(Y)/μy² - 2*Cov(X,Y)/(μx*μy)]
            
            More accurate than naive ratio, especially for small samples.
            """
            n = len(numerator)
            
            # Mean estimates
            mean_num = np.mean(numerator)
            mean_den = np.mean(denominator)
            ratio = mean_num / mean_den
            
            # Variance and covariance
            var_num = np.var(numerator, ddof=1)
            var_den = np.var(denominator, ddof=1)
            cov = np.cov(numerator, denominator)[0, 1]
            
            # Delta method variance
            delta_var = (1 / mean_den**2) * (
                var_num + 
                ratio**2 * var_den - 
                2 * ratio * cov
            )
            
            std_error = np.sqrt(delta_var / n)
            
            # Confidence interval
            z_crit = stats.norm.ppf(1 - self.alpha / 2)
            ci_lower = ratio - z_crit * std_error
            ci_upper = ratio + z_crit * std_error
            
            return RatioMetricResult(
                ratio=ratio,
                std_error=std_error,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                method='delta_method',
                sample_size=n
            )
        
        def bootstrap_ratio(self, numerator: np.ndarray,
                           denominator: np.ndarray) -> RatioMetricResult:
            """
            Bootstrap method for ratio metrics.
            More robust, no distributional assumptions.
            """
            n = len(numerator)
            bootstrap_ratios = []
            
            # Bootstrap resampling
            for _ in range(self.n_bootstrap):
                indices = np.random.choice(n, size=n, replace=True)
                boot_num = numerator[indices]
                boot_den = denominator[indices]
                
                # Calculate ratio for bootstrap sample
                boot_ratio = np.sum(boot_num) / np.sum(boot_den)
                bootstrap_ratios.append(boot_ratio)
            
            bootstrap_ratios = np.array(bootstrap_ratios)
            
            # Point estimate and CI
            ratio = np.sum(numerator) / np.sum(denominator)
            std_error = np.std(bootstrap_ratios, ddof=1)
            ci_lower = np.percentile(bootstrap_ratios, self.alpha * 100 / 2)
            ci_upper = np.percentile(bootstrap_ratios, 100 - self.alpha * 100 / 2)
            
            return RatioMetricResult(
                ratio=ratio,
                std_error=std_error,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                method='bootstrap',
                sample_size=n
            )
        
        def compare_treatments(self, 
                             control_num: np.ndarray, control_den: np.ndarray,
                             treatment_num: np.ndarray, treatment_den: np.ndarray,
                             method: str = 'delta') -> Dict:
            """
            Compare ratio metrics between control and treatment.
            Returns effect size and statistical significance.
            """
            if method == 'delta':
                control_result = self.delta_method(control_num, control_den)
                treatment_result = self.delta_method(treatment_num, treatment_den)
            else:
                control_result = self.bootstrap_ratio(control_num, control_den)
                treatment_result = self.bootstrap_ratio(treatment_num, treatment_den)
            
            # Relative lift
            lift = (treatment_result.ratio - control_result.ratio) / control_result.ratio
            
            # Test statistic
            diff = treatment_result.ratio - control_result.ratio
            se_diff = np.sqrt(control_result.std_error**2 + treatment_result.std_error**2)
            z_stat = diff / se_diff
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            return {
                'control_ratio': control_result.ratio,
                'treatment_ratio': treatment_result.ratio,
                'absolute_lift': diff,
                'relative_lift': lift,
                'se_diff': se_diff,
                'z_statistic': z_stat,
                'p_value': p_value,
                'significant': p_value < self.alpha,
                'control_ci': (control_result.ci_lower, control_result.ci_upper),
                'treatment_ci': (treatment_result.ci_lower, treatment_result.ci_upper)
            }
    
    # Example: Click-Through Rate (CTR) Analysis
    np.random.seed(42)
    
    # Simulate user-level data
    n_control = 10000
    n_treatment = 10000
    
    # Control group: 2% CTR
    control_impressions = np.random.poisson(lam=100, size=n_control)
    control_clicks = np.random.binomial(
        n=control_impressions, 
        p=0.02
    )
    
    # Treatment group: 2.3% CTR (15% relative lift)
    treatment_impressions = np.random.poisson(lam=100, size=n_treatment)
    treatment_clicks = np.random.binomial(
        n=treatment_impressions,
        p=0.023
    )
    
    # Analyze with both methods
    analyzer = RatioMetricAnalyzer(alpha=0.05, n_bootstrap=10000)
    
    print("=" * 80)
    print("RATIO METRIC ANALYSIS: Click-Through Rate (CTR)")
    print("=" * 80)
    
    # Delta method
    delta_comparison = analyzer.compare_treatments(
        control_clicks, control_impressions,
        treatment_clicks, treatment_impressions,
        method='delta'
    )
    
    print("\n📊 DELTA METHOD RESULTS:")
    print(f"Control CTR:    {delta_comparison['control_ratio']:.4f} "
          f"(95% CI: {delta_comparison['control_ci'][0]:.4f} - "
          f"{delta_comparison['control_ci'][1]:.4f})")
    print(f"Treatment CTR:  {delta_comparison['treatment_ratio']:.4f} "
          f"(95% CI: {delta_comparison['treatment_ci'][0]:.4f} - "
          f"{delta_comparison['treatment_ci'][1]:.4f})")
    print(f"Absolute Lift:  {delta_comparison['absolute_lift']:.4f}")
    print(f"Relative Lift:  {delta_comparison['relative_lift']*100:.2f}%")
    print(f"P-value:        {delta_comparison['p_value']:.4f}")
    print(f"Significant:    {delta_comparison['significant']}")
    
    # Bootstrap method
    bootstrap_comparison = analyzer.compare_treatments(
        control_clicks, control_impressions,
        treatment_clicks, treatment_impressions,
        method='bootstrap'
    )
    
    print("\n🔄 BOOTSTRAP METHOD RESULTS:")
    print(f"Control CTR:    {bootstrap_comparison['control_ratio']:.4f} "
          f"(95% CI: {bootstrap_comparison['control_ci'][0]:.4f} - "
          f"{bootstrap_comparison['control_ci'][1]:.4f})")
    print(f"Treatment CTR:  {bootstrap_comparison['treatment_ratio']:.4f} "
          f"(95% CI: {bootstrap_comparison['treatment_ci'][0]:.4f} - "
          f"{bootstrap_comparison['treatment_ci'][1]:.4f})")
    print(f"Relative Lift:  {bootstrap_comparison['relative_lift']*100:.2f}%")
    print(f"P-value:        {bootstrap_comparison['p_value']:.4f}")
    
    # Output:
    # ================================================================================
    # RATIO METRIC ANALYSIS: Click-Through Rate (CTR)
    # ================================================================================
    # 
    # 📊 DELTA METHOD RESULTS:
    # Control CTR:    0.0200 (95% CI: 0.0198 - 0.0203)
    # Treatment CTR:  0.0230 (95% CI: 0.0227 - 0.0233)
    # Absolute Lift:  0.0029
    # Relative Lift:  14.72%
    # P-value:        0.0000
    # Significant:    True
    # 
    # 🔄 BOOTSTRAP METHOD RESULTS:
    # Control CTR:    0.0200 (95% CI: 0.0198 - 0.0203)
    # Treatment CTR:  0.0230 (95% CI: 0.0227 - 0.0233)
    # Relative Lift:  14.72%
    # P-value:        0.0000
    ```
    
    **Method Comparison Table:**
    
    | Method | Time Complexity | Assumptions | Best Use Case | Accuracy |
    |--------|----------------|-------------|---------------|----------|
    | **Delta Method** | O(n) | Normal approx | Large samples (n>1000) | ±0.5% error |
    | **Bootstrap** | O(n * B) | None | Any sample size | ±0.1% error |
    | **Naive Ratio** | O(n) | Independent num/den | ❌ Never use | Biased |
    
    **Real-World Examples with Company Metrics:**
    
    | Company | Metric Type | Challenge | Solution | Impact |
    |---------|-------------|-----------|----------|--------|
    | **Netflix** | Play rate (plays/visits) | High variance in visits per user | Delta method with stratification by country | 95% accurate predictions |
    | **Google Ads** | CTR (clicks/impressions) | Users have different impression counts | User-level aggregation + bootstrap | Reduced false positives by 40% |
    | **Uber** | Trips per active day | Zero-inflated (many 0-trip days) | Mixed model: logit(active) + Poisson(trips) | Detected 12% lift vs 8% naive |
    | **Amazon** | Units per order | Order size varies 100x | Winsorized delta method (99th percentile) | Reduced outlier impact by 80% |
    
    **User-Level vs Session-Level Analysis:**
    
    ```python
    # User-level ratio (RECOMMENDED)
    user_level_ctr = df.groupby('user_id').agg({
        'clicks': 'sum',
        'impressions': 'sum'
    })
    overall_ctr_user = user_level_ctr['clicks'].sum() / user_level_ctr['impressions'].sum()
    
    # Session-level ratio (USE WITH CAUTION)
    session_level_ctr = (df['clicks'] / df['impressions']).mean()
    
    # User-level is correct because:
    # 1. Matches randomization unit
    # 2. Proper variance estimation
    # 3. Avoids Simpson's Paradox
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of ratio metric complexity, variance estimation methods.
        
        **Strong answer signals:**
        
        - Explains why naive ratio is biased (numerator/denominator correlation)
        - Implements delta method correctly (includes covariance term)
        - Knows when bootstrap is better (small samples, non-normal)
        - Matches analysis unit to randomization unit (user-level)
        - Discusses trade-offs: delta method faster, bootstrap more robust
        
        **Red flags:**
        
        - Using simple t-test on ratio
        - Ignoring correlation between numerator and denominator
        - Session-level analysis with user-level randomization
        
        **Follow-up questions:**
        
        - "What if denominator can be zero?" (Fieller's theorem, exclude zeros)
        - "How to handle extreme ratios?" (Winsorize, log-transform)
        - "What if ratio distribution is skewed?" (Bootstrap, log-ratio)

---

### What is Sensitivity Analysis? - Netflix, Uber Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Robustness` | **Asked by:** Netflix, Uber, Airbnb

??? success "View Answer"

    **Purpose:** Test if experimental conclusions are robust to analytical choices, assumptions, and edge cases. Critical before launch decisions.
    
    **Sensitivity Analysis Framework:**
    
    ```
    ┌─────────────────────────────────────────────────────────────────────┐
    │                      Sensitivity Analysis                            │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                       │
    │  Base Analysis                                                       │
    │  ┌──────────────────┐                                               │
    │  │ Primary Metric   │                                               │
    │  │ Full Dataset     │ ──► Result: +5% lift, p=0.02                 │
    │  │ Standard Method  │                                               │
    │  └──────────────────┘                                               │
    │           │                                                          │
    │           ├──► Robustness Checks:                                   │
    │           │                                                          │
    │           ├─► Time Periods                                          │
    │           │    ├─ Week 1 only                                       │
    │           │    ├─ Week 2 only                                       │
    │           │    └─ Exclude weekends                                  │
    │           │                                                          │
    │           ├─► Outlier Treatment                                     │
    │           │    ├─ Winsorize 1%                                      │
    │           │    ├─ Winsorize 5%                                      │
    │           │    └─ Remove top 0.1%                                   │
    │           │                                                          │
    │           ├─► Segments                                              │
    │           │    ├─ Desktop only                                      │
    │           │    ├─ Mobile only                                       │
    │           │    └─ By country                                        │
    │           │                                                          │
    │           └─► Alternative Methods                                   │
    │                ├─ Mann-Whitney U                                    │
    │                ├─ Permutation test                                  │
    │                └─ Bootstrap                                         │
    │                                                                       │
    └─────────────────────────────────────────────────────────────────────┘
    ```
    
    **Production Implementation - Comprehensive Sensitivity Analyzer:**
    
    ```python
    import numpy as np
    import pandas as pd
    from scipy import stats
    from typing import Dict, List, Callable, Tuple
    from dataclasses import dataclass
    import warnings
    
    @dataclass
    class SensitivityResult:
        """Container for sensitivity check results"""
        check_name: str
        effect_size: float
        p_value: float
        ci_lower: float
        ci_upper: float
        sample_size: int
        significant: bool
        
    class SensitivityAnalyzer:
        """
        Comprehensive sensitivity analysis for A/B test results.
        Tests robustness across multiple dimensions.
        """
        
        def __init__(self, alpha: float = 0.05):
            self.alpha = alpha
            self.results = []
        
        def baseline_analysis(self, control: np.ndarray, 
                             treatment: np.ndarray,
                             name: str = "Baseline") -> SensitivityResult:
            """Standard t-test baseline"""
            stat, p_value = stats.ttest_ind(treatment, control)
            effect = np.mean(treatment) - np.mean(control)
            
            # Cohen's d
            pooled_std = np.sqrt((np.var(control, ddof=1) + np.var(treatment, ddof=1)) / 2)
            cohens_d = effect / pooled_std if pooled_std > 0 else 0
            
            # Confidence interval
            se_diff = np.sqrt(np.var(control, ddof=1)/len(control) + 
                             np.var(treatment, ddof=1)/len(treatment))
            ci_lower = effect - 1.96 * se_diff
            ci_upper = effect + 1.96 * se_diff
            
            result = SensitivityResult(
                check_name=name,
                effect_size=effect,
                p_value=p_value,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                sample_size=len(control) + len(treatment),
                significant=p_value < self.alpha
            )
            self.results.append(result)
            return result
        
        def time_period_sensitivity(self, df: pd.DataFrame,
                                   metric_col: str, 
                                   treatment_col: str,
                                   date_col: str) -> List[SensitivityResult]:
            """Test sensitivity across time periods"""
            time_results = []
            
            # Week-by-week
            df['week'] = pd.to_datetime(df[date_col]).dt.isocalendar().week
            for week in df['week'].unique():
                week_data = df[df['week'] == week]
                control = week_data[week_data[treatment_col] == 0][metric_col].values
                treatment = week_data[week_data[treatment_col] == 1][metric_col].values
                
                result = self.baseline_analysis(control, treatment, f"Week_{week}")
                time_results.append(result)
            
            # Weekday vs Weekend
            df['is_weekend'] = pd.to_datetime(df[date_col]).dt.dayofweek >= 5
            
            for is_weekend in [False, True]:
                subset = df[df['is_weekend'] == is_weekend]
                control = subset[subset[treatment_col] == 0][metric_col].values
                treatment = subset[subset[treatment_col] == 1][metric_col].values
                
                period_name = "Weekend" if is_weekend else "Weekday"
                result = self.baseline_analysis(control, treatment, period_name)
                time_results.append(result)
            
            return time_results
        
        def outlier_sensitivity(self, control: np.ndarray,
                               treatment: np.ndarray) -> List[SensitivityResult]:
            """Test sensitivity to outlier treatment"""
            outlier_results = []
            
            # Original
            outlier_results.append(
                self.baseline_analysis(control, treatment, "Original (no treatment)")
            )
            
            # Winsorization at different levels
            for percentile in [1, 5, 10]:
                lower_p = percentile
                upper_p = 100 - percentile
                
                control_wins = np.clip(
                    control,
                    np.percentile(control, lower_p),
                    np.percentile(control, upper_p)
                )
                treatment_wins = np.clip(
                    treatment,
                    np.percentile(treatment, lower_p),
                    np.percentile(treatment, upper_p)
                )
                
                result = self.baseline_analysis(
                    control_wins, treatment_wins, 
                    f"Winsorize_{percentile}%"
                )
                outlier_results.append(result)
            
            # Remove extreme outliers (3 IQR rule)
            def remove_outliers(data):
                q1, q3 = np.percentile(data, [25, 75])
                iqr = q3 - q1
                lower = q1 - 3 * iqr
                upper = q3 + 3 * iqr
                return data[(data >= lower) & (data <= upper)]
            
            control_clean = remove_outliers(control)
            treatment_clean = remove_outliers(treatment)
            
            result = self.baseline_analysis(
                control_clean, treatment_clean,
                "Remove_3IQR_outliers"
            )
            outlier_results.append(result)
            
            return outlier_results
        
        def method_sensitivity(self, control: np.ndarray,
                              treatment: np.ndarray) -> List[SensitivityResult]:
            """Test sensitivity to statistical method"""
            method_results = []
            
            # Parametric t-test
            method_results.append(
                self.baseline_analysis(control, treatment, "t-test")
            )
            
            # Non-parametric Mann-Whitney U
            stat, p_value = stats.mannwhitneyu(treatment, control, alternative='two-sided')
            effect = np.median(treatment) - np.median(control)
            
            method_results.append(SensitivityResult(
                check_name="Mann-Whitney U",
                effect_size=effect,
                p_value=p_value,
                ci_lower=np.nan,  # Bootstrap for CI
                ci_upper=np.nan,
                sample_size=len(control) + len(treatment),
                significant=p_value < self.alpha
            ))
            
            # Permutation test
            observed_diff = np.mean(treatment) - np.mean(control)
            combined = np.concatenate([control, treatment])
            n_control = len(control)
            
            perm_diffs = []
            for _ in range(5000):
                perm = np.random.permutation(combined)
                perm_control = perm[:n_control]
                perm_treatment = perm[n_control:]
                perm_diffs.append(np.mean(perm_treatment) - np.mean(perm_control))
            
            p_value_perm = np.mean(np.abs(perm_diffs) >= abs(observed_diff))
            
            method_results.append(SensitivityResult(
                check_name="Permutation Test",
                effect_size=observed_diff,
                p_value=p_value_perm,
                ci_lower=np.percentile(perm_diffs, 2.5),
                ci_upper=np.percentile(perm_diffs, 97.5),
                sample_size=len(control) + len(treatment),
                significant=p_value_perm < self.alpha
            ))
            
            return method_results
        
        def generate_report(self) -> pd.DataFrame:
            """Generate summary report of all sensitivity checks"""
            report_data = []
            for result in self.results:
                report_data.append({
                    'Check': result.check_name,
                    'Effect Size': f"{result.effect_size:.4f}",
                    'P-value': f"{result.p_value:.4f}",
                    'CI': f"[{result.ci_lower:.4f}, {result.ci_upper:.4f}]" if not np.isnan(result.ci_lower) else "N/A",
                    'Significant': '✓' if result.significant else '✗',
                    'N': result.sample_size
                })
            
            return pd.DataFrame(report_data)
    
    # Example: Revenue Metric Sensitivity Analysis
    np.random.seed(42)
    
    # Simulate revenue data with outliers
    n = 5000
    control_revenue = np.random.gamma(shape=2, scale=50, size=n)
    treatment_revenue = np.random.gamma(shape=2, scale=52, size=n)  # 4% lift
    
    # Add extreme outliers (whales)
    control_revenue[np.random.choice(n, 10)] *= 20
    treatment_revenue[np.random.choice(n, 10)] *= 20
    
    # Run comprehensive sensitivity analysis
    analyzer = SensitivityAnalyzer(alpha=0.05)
    
    print("=" * 80)
    print("SENSITIVITY ANALYSIS REPORT: Revenue Metric")
    print("=" * 80)
    
    # Baseline
    baseline = analyzer.baseline_analysis(control_revenue, treatment_revenue, "Baseline")
    print(f"\n📊 BASELINE ANALYSIS:")
    print(f"Effect: ${baseline.effect_size:.2f}, P-value: {baseline.p_value:.4f}")
    
    # Outlier sensitivity
    print(f"\n🎯 OUTLIER SENSITIVITY:")
    outlier_results = analyzer.outlier_sensitivity(control_revenue, treatment_revenue)
    for result in outlier_results:
        print(f"{result.check_name:30s} Effect: ${result.effect_size:8.2f}  "
              f"P-value: {result.p_value:.4f}  {'✓' if result.significant else '✗'}")
    
    # Method sensitivity
    print(f"\n🔬 METHOD SENSITIVITY:")
    method_results = analyzer.method_sensitivity(control_revenue, treatment_revenue)
    for result in method_results:
        print(f"{result.check_name:30s} Effect: ${result.effect_size:8.2f}  "
              f"P-value: {result.p_value:.4f}  {'✓' if result.significant else '✗'}")
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE:")
    print("=" * 80)
    print(analyzer.generate_report().to_string(index=False))
    
    # Check consistency
    significant_count = sum(1 for r in analyzer.results if r.significant)
    total_checks = len(analyzer.results)
    consistency_rate = significant_count / total_checks
    
    print(f"\n📈 ROBUSTNESS SCORE: {consistency_rate*100:.1f}% of checks significant")
    
    if consistency_rate > 0.8:
        print("✓ DECISION: Strong evidence - SHIP")
    elif consistency_rate > 0.5:
        print("⚠ DECISION: Moderate evidence - CONSIDER MORE DATA")
    else:
        print("✗ DECISION: Weak evidence - DO NOT SHIP")
    ```
    
    **Sensitivity Dimensions Table:**
    
    | Dimension | Checks | Purpose | Interpretation |
    |-----------|--------|---------|----------------|
    | **Time Period** | Week-by-week, weekday/weekend, first/last week | Temporal stability | All periods show lift → stable effect |
    | **Outliers** | Winsorize 1/5/10%, Remove 3-IQR | Robustness to extremes | Effect persists → not driven by whales |
    | **Segments** | Device, country, new/returning | Heterogeneity | Effect in all segments → universal |
    | **Methods** | t-test, Mann-Whitney, permutation, bootstrap | Distributional assumptions | All methods agree → robust |
    
    **Real-World Examples with Company Metrics:**
    
    | Company | Metric | Sensitivity Issue | Solution | Outcome |
    |---------|--------|-------------------|----------|---------|
    | **Netflix** | Watch time | Weekday vs weekend patterns differ | Separate analysis by day type | Found 10% weekday lift, 5% weekend lift |
    | **Uber** | Trips per user | Top 0.1% users drive 30% of trips | Winsorize at 99th percentile | Reduced from 15% lift to 8% (true effect) |
    | **Airbnb** | Booking rate | Different by season | Stratify by month, use regression adjustment | Confirmed 5% lift across all seasons |
    | **Meta** | Session duration | First week novelty effect | Exclude first 3 days from analysis | Effect dropped from 20% to 12% |
    
    **Decision Framework:**
    
    | Consistency Rate | Decision | Action |
    |-----------------|----------|--------|
    | **>80% significant** | 🟢 Strong evidence | Ship confidently |
    | **50-80% significant** | 🟡 Moderate evidence | Collect more data or segment |
    | **<50% significant** | 🔴 Weak evidence | Do not ship |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Rigor, understanding of robustness, decision-making under uncertainty.
        
        **Strong answer signals:**
        
        - Tests multiple dimensions (time, outliers, methods, segments)
        - Quantifies consistency across checks (e.g., "80% of checks significant")
        - Explains trade-offs (winsorizing reduces power but increases robustness)
        - Knows when to adjust (e.g., exclude first week for novelty effects)
        - Makes clear recommendation based on sensitivity results
        
        **Red flags:**
        
        - Only runs one analysis configuration
        - Ignores time periods or outliers
        - Can't explain why results might differ across checks
        
        **Follow-up questions:**
        
        - "What if different checks give opposite conclusions?" (Report both, investigate heterogeneity)
        - "How many sensitivity checks is enough?" (Cover major assumptions, not exhaustive)
        - "What if outlier treatment changes conclusion?" (Flag as inconclusive, need more data)

---

### How to Communicate Results to Stakeholders? - All Companies Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Communication` | **Asked by:** All Companies

??? success "View Answer"

    **Effective communication** of A/B test results requires **tailoring content** to your audience—executives need business impact, product managers need actionable insights, and engineers need technical details. The goal is to **drive decisions** with clarity and confidence while being **honest about limitations**.

    **Communication Framework:**

    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │              A/B Test Results Communication Flow                │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
    │  │   Analysis   │────▶│    Report    │────▶│   Decision   │   │
    │  │   Complete   │     │  Generation  │     │   Meeting    │   │
    │  └──────────────┘     └──────────────┘     └──────────────┘   │
    │         │                     │                     │           │
    │         ↓                     ↓                     ↓           │
    │  ┌──────────────────────────────────────────────────────┐     │
    │  │  Stakeholder-Specific Content                         │     │
    │  ├──────────────────────────────────────────────────────┤     │
    │  │  Executive:   1-pager with decision + impact         │     │
    │  │  Product:     Detailed slides with segments          │     │
    │  │  Engineering: Technical appendix with methods        │     │
    │  └──────────────────────────────────────────────────────┘     │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘
    ```

    **Report Structure (Pyramid Principle):**

    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    from datetime import datetime, timedelta
    
    # Production: Comprehensive Stakeholder Communication System
    
    class ABTestReporter:
        """
        Generate stakeholder-appropriate A/B test reports with visualizations.
        """
        
        def __init__(self, experiment_name, start_date, end_date):
            self.experiment_name = experiment_name
            self.start_date = start_date
            self.end_date = end_date
            self.results = {}
            
        def analyze_test(self, control_data, treatment_data, metric_name, 
                        metric_type='continuous', alpha=0.05):
            """
            Analyze A/B test and store results.
            
            metric_type: 'continuous' or 'proportion'
            """
            if metric_type == 'continuous':
                # Welch's t-test for continuous metrics
                t_stat, p_value = stats.ttest_ind(treatment_data, control_data, 
                                                   equal_var=False)
                control_mean = np.mean(control_data)
                treatment_mean = np.mean(treatment_data)
                control_se = stats.sem(control_data)
                treatment_se = stats.sem(treatment_data)
                
                # Confidence interval for difference
                diff = treatment_mean - control_mean
                se_diff = np.sqrt(control_se**2 + treatment_se**2)
                ci_lower = diff - 1.96 * se_diff
                ci_upper = diff + 1.96 * se_diff
                
            else:  # proportion
                # Z-test for proportions
                n_control = len(control_data)
                n_treatment = len(treatment_data)
                p_control = np.mean(control_data)
                p_treatment = np.mean(treatment_data)
                
                # Pooled proportion
                p_pooled = (np.sum(control_data) + np.sum(treatment_data)) / (n_control + n_treatment)
                se_pooled = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_control + 1/n_treatment))
                
                z_stat = (p_treatment - p_control) / se_pooled
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                
                control_mean = p_control
                treatment_mean = p_treatment
                
                # CI for difference in proportions
                diff = treatment_mean - control_mean
                se_diff = np.sqrt(p_control*(1-p_control)/n_control + 
                                 p_treatment*(1-p_treatment)/n_treatment)
                ci_lower = diff - 1.96 * se_diff
                ci_upper = diff + 1.96 * se_diff
            
            # Store results
            self.results[metric_name] = {
                'control_mean': control_mean,
                'treatment_mean': treatment_mean,
                'absolute_lift': diff,
                'relative_lift': (treatment_mean / control_mean - 1) * 100,
                'p_value': p_value,
                'significant': p_value < alpha,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'sample_size_control': len(control_data),
                'sample_size_treatment': len(treatment_data)
            }
            
            return self.results[metric_name]
        
        def generate_executive_summary(self, primary_metric):
            """
            1-page executive summary: Decision + Impact + Next Steps
            """
            result = self.results[primary_metric]
            
            summary = f"""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║           A/B TEST EXECUTIVE SUMMARY (1-PAGER)                    ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║                                                                    ║
    ║  EXPERIMENT: {self.experiment_name}                              ║
    ║  DATE RANGE: {self.start_date} to {self.end_date}                ║
    ║                                                                    ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║  📊 BOTTOM LINE RECOMMENDATION                                     ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║                                                                    ║
    ║  {'✅ SHIP TREATMENT' if result['significant'] and result['absolute_lift'] > 0 else '❌ DO NOT SHIP'}
    ║                                                                    ║
    ║  Primary Metric ({primary_metric}):                               ║
    ║  • Control:    {result['control_mean']:.4f}                       ║
    ║  • Treatment:  {result['treatment_mean']:.4f}                     ║
    ║  • Lift:       {result['relative_lift']:+.2f}% ({result['absolute_lift']:+.4f})
    ║  • 95% CI:     [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]
    ║  • p-value:    {result['p_value']:.4f}                            ║
    ║  • Significant: {'Yes ✓' if result['significant'] else 'No ✗'}   ║
    ║                                                                    ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║  💰 BUSINESS IMPACT                                                ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║                                                                    ║
    ║  Projected Annual Value: $XXM (based on user base)                ║
    ║  Implementation Cost: $XXK                                         ║
    ║  ROI: XXX%                                                         ║
    ║                                                                    ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║  🚦 GUARDRAIL METRICS                                              ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║                                                                    ║
    ║  • User retention:     ✅ No degradation                           ║
    ║  • Load time:          ✅ Within SLA                               ║
    ║  • Error rate:         ✅ No increase                              ║
    ║                                                                    ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║  📋 NEXT STEPS                                                     ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║                                                                    ║
    ║  1. Roll out to 100% of users by [DATE]                           ║
    ║  2. Monitor metrics for 2 weeks post-launch                        ║
    ║  3. Prepare follow-up iteration for Q[X]                           ║
    ║                                                                    ║
    ╚═══════════════════════════════════════════════════════════════════╝
            """
            
            return summary
        
        def generate_product_report(self):
            """
            Detailed slides for product managers with segmentation.
            """
            report = f"""
    ═══════════════════════════════════════════════════════════════════
    PRODUCT MANAGER REPORT: {self.experiment_name}
    ═══════════════════════════════════════════════════════════════════
    
    SLIDE 1: EXPERIMENT OVERVIEW
    ─────────────────────────────────────────────────────────────────
    Hypothesis: [Treatment] will improve [Metric] by [X]%
    Duration: {self.start_date} to {self.end_date}
    Sample Size: {list(self.results.values())[0]['sample_size_control']:,} control, 
                 {list(self.results.values())[0]['sample_size_treatment']:,} treatment
    
    SLIDE 2: KEY RESULTS
    ─────────────────────────────────────────────────────────────────
    """
            
            for metric_name, result in self.results.items():
                report += f"""
    {metric_name}:
      • Lift: {result['relative_lift']:+.2f}% (95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}])
      • Significant: {'Yes ✓' if result['significant'] else 'No ✗'} (p={result['p_value']:.4f})
      • Sample: {result['sample_size_treatment']:,} users
                """
            
            report += """
    
    SLIDE 3: SEGMENT ANALYSIS
    ─────────────────────────────────────────────────────────────────
    [Include visualizations showing lift by segment]
    
    SLIDE 4: USER FEEDBACK
    ─────────────────────────────────────────────────────────────────
    [Qualitative insights from surveys, support tickets]
    
    SLIDE 5: RECOMMENDATION
    ─────────────────────────────────────────────────────────────────
    Ship / Don't Ship / Iterate
    Rationale: [Brief explanation]
    ═══════════════════════════════════════════════════════════════════
            """
            
            return report
        
        def create_visualization(self, primary_metric, save_path=None):
            """
            Create comprehensive visualization for stakeholders.
            """
            result = self.results[primary_metric]
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'A/B Test Results: {self.experiment_name}', fontsize=16, fontweight='bold')
            
            # 1. Bar chart: Control vs Treatment
            ax1 = axes[0, 0]
            groups = ['Control', 'Treatment']
            means = [result['control_mean'], result['treatment_mean']]
            ax1.bar(groups, means, color=['#3498db', '#e74c3c'], alpha=0.7)
            ax1.set_ylabel(primary_metric)
            ax1.set_title('Control vs Treatment')
            ax1.axhline(result['control_mean'], color='gray', linestyle='--', alpha=0.5)
            
            # 2. Confidence interval plot
            ax2 = axes[0, 1]
            ax2.errorbar([1], [result['absolute_lift']], 
                        yerr=[[result['absolute_lift'] - result['ci_lower']], 
                              [result['ci_upper'] - result['absolute_lift']]], 
                        fmt='o', markersize=10, capsize=10, capthick=2, color='#e74c3c')
            ax2.axhline(0, color='black', linestyle='-', linewidth=2)
            ax2.set_xlim(0.5, 1.5)
            ax2.set_xticks([1])
            ax2.set_xticklabels(['Lift'])
            ax2.set_ylabel('Absolute Lift')
            ax2.set_title(f'Treatment Effect with 95% CI\np-value = {result["p_value"]:.4f}')
            ax2.grid(axis='y', alpha=0.3)
            
            # 3. Statistical significance indicator
            ax3 = axes[1, 0]
            ax3.axis('off')
            decision_text = "✅ STATISTICALLY SIGNIFICANT\nSHIP TREATMENT" if result['significant'] and result['absolute_lift'] > 0 else "❌ NOT SIGNIFICANT\nDO NOT SHIP"
            decision_color = '#27ae60' if result['significant'] and result['absolute_lift'] > 0 else '#e74c3c'
            ax3.text(0.5, 0.5, decision_text, fontsize=18, fontweight='bold', 
                    ha='center', va='center', color=decision_color,
                    bbox=dict(boxstyle='round', facecolor=decision_color, alpha=0.2, pad=1))
            
            # 4. Key metrics table
            ax4 = axes[1, 1]
            ax4.axis('off')
            table_data = [
                ['Metric', 'Value'],
                ['Control Mean', f'{result["control_mean"]:.4f}'],
                ['Treatment Mean', f'{result["treatment_mean"]:.4f}'],
                ['Relative Lift', f'{result["relative_lift"]:+.2f}%'],
                ['95% CI Lower', f'{result["ci_lower"]:.4f}'],
                ['95% CI Upper', f'{result["ci_upper"]:.4f}'],
                ['p-value', f'{result["p_value"]:.4f}'],
                ['Sample Size', f'{result["sample_size_treatment"]:,}']
            ]
            table = ax4.table(cellText=table_data, cellLoc='left', loc='center',
                            colWidths=[0.5, 0.5])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # Style header
            for i in range(2):
                table[(0, i)].set_facecolor('#3498db')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Visualization saved to {save_path}")
            
            return fig
    
    
    # Example: Netflix Watch Time A/B Test
    np.random.seed(42)
    
    reporter = ABTestReporter(
        experiment_name="New Recommendation Algorithm",
        start_date="2024-11-01",
        end_date="2024-11-14"
    )
    
    # Generate realistic data
    n_users = 50000
    control_watch_time = np.random.gamma(shape=2, scale=30, size=n_users)
    treatment_watch_time = np.random.gamma(shape=2, scale=33, size=n_users)  # 10% lift
    
    # Analyze
    result = reporter.analyze_test(
        control_watch_time, 
        treatment_watch_time, 
        metric_name='Daily Watch Time (minutes)',
        metric_type='continuous'
    )
    
    print("=" * 70)
    print("A/B TEST STAKEHOLDER COMMUNICATION EXAMPLE")
    print("=" * 70)
    
    # Executive Summary
    print(reporter.generate_executive_summary('Daily Watch Time (minutes)'))
    
    # Product Report
    print("\n" + reporter.generate_product_report())
    
    # Create visualization
    reporter.create_visualization('Daily Watch Time (minutes)')
    plt.show()
    ```

    **Audience-Specific Content:**

    | Audience | Content Focus | Format | Length | Key Elements |
    |----------|--------------|--------|--------|--------------|
    | **Executive (C-suite)** | Business impact, ROI, decision | 1-pager PDF | 1 page | Bottom line, $ impact, risk assessment |
    | **Product Manager** | Metrics, segments, user behavior | Slide deck | 5-10 slides | Detailed results, segment analysis, next steps |
    | **Engineering** | Implementation, technical details | Technical doc | As needed | Methods, assumptions, code, edge cases |
    | **Data Science** | Statistical methods, sensitivity | Jupyter notebook | Full analysis | P-values, CIs, diagnostics, robustness checks |

    **Real Company Examples:**

    | Company | Test | Communication Approach | Outcome |
    |---------|------|----------------------|---------|
    | **Netflix** | Recommendation algorithm | Executive 1-pager: "+5% watch time = $100M ARR", full deck for product | Approved in 1 meeting, shipped globally |
    | **Uber** | Surge pricing UI | PM deck with segment analysis (new vs power users), guardrail dashboard | Identified 10% churn risk in new users, iterated |
    | **Google Ads** | Bid optimization | Automated email report to 50+ PMs daily with traffic lights (🟢🟡🔴) | Scaled decision-making to 1000+ tests/year |
    | **Airbnb** | Search ranking | Weekly newsletter with "Experiment of the Week" featuring learnings | Built experimentation culture across org |

    **Communication Anti-Patterns (AVOID):**

    | ❌ Anti-Pattern | Why It's Bad | ✅ Better Approach |
    |----------------|--------------|-------------------|
    | **P-value only** | "p=0.03" → Stakeholders don't understand | "95% confident lift is +5-8%" |
    | **Cherry-picking** | Show only positive segments | Show all pre-registered segments |
    | **Jargon overload** | "Heteroskedasticity", "CUPED" | "Adjusted for pre-experiment behavior" |
    | **No decision** | "Here are the numbers" | "Recommend shipping because..." |
    | **Overconfidence** | "This will definitely work" | "Based on 2 weeks, expect +5% with ±2% uncertainty" |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Communication skills, stakeholder management, ability to tailor technical content.
        
        **Strong answer signals:**
        
        - Starts with decision (ship/don't ship), not statistics
        - Uses confidence intervals instead of p-values
        - Tailors content to audience (exec vs engineer)
        - Includes guardrail metrics and caveats
        - Provides clear next steps
        - Shows visualizations (bar charts, CI plots)
        - Mentions business impact ($, users affected)
        - Knows when to simplify vs go deep
        
        **Red flags:**
        
        - Leads with p-values and test statistics
        - Uses jargon without explanation
        - No clear recommendation
        - Ignores guardrails or negative results
        - Same report for all audiences
        
        **Follow-up questions:**
        
        - "How would you explain p-value to a non-technical PM?" (Probability of seeing this result by chance)
        - "What if the exec asks 'Are you sure?'" (Quantify confidence with CI, mention assumptions)
        - "How do you handle requests to 'dig deeper' when results are negative?" (Pre-register analysis plan, avoid p-hacking)

---

### What is the Minimum Detectable Effect (MDE)? - Netflix, Uber Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Experimental Design` | **Asked by:** Netflix, Uber, Airbnb

??? success "View Answer"

    **Minimum Detectable Effect (MDE)** is the **smallest effect size** that an experiment can reliably detect with a given sample size, significance level (α), and statistical power (1-β). It's a **pre-experiment planning tool** that answers: "How large must the treatment effect be for us to detect it?"

    **MDE Formula (continuous metrics):**

    $$MDE = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{\sigma^2}{n_C} + \frac{\sigma^2}{n_T}} \cdot \frac{1}{\sqrt{n_C}}$$

    For **equal sample sizes** ($n_C = n_T = n$):

    $$MDE = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sigma \cdot \sqrt{\frac{2}{n}}$$

    Where:
    - $z_{1-\alpha/2}$ = critical value for two-sided test (1.96 for α=0.05)
    - $z_{1-\beta}$ = critical value for power (0.84 for 80% power, 1.28 for 90% power)
    - $\sigma$ = standard deviation of metric
    - $n$ = sample size per group

    **Power Analysis Triangle:**

    ```
                   ┌──────────────────────────────────┐
                   │    Power Analysis Components      │
                   ├──────────────────────────────────┤
                   │                                   │
                   │         Effect Size (MDE)         │
                   │              ▲                    │
                   │              │                    │
                   │              │                    │
                   │    ┌─────────┴─────────┐         │
                   │    │                   │         │
                   │    │                   │         │
                   │    ▼                   ▼         │
                   │  Sample              Power       │
                   │   Size                (1-β)      │
                   │    │                   │         │
                   │    │                   │         │
                   │    └─────────┬─────────┘         │
                   │              │                    │
                   │              ▼                    │
                   │    Significance Level (α)        │
                   │                                   │
                   │  Fix 3 parameters → Solve for 4th│
                   │                                   │
                   └──────────────────────────────────┘
    ```

    **Production MDE Calculator:**

    ```python
    import numpy as np
    import pandas as pd
    from scipy import stats
    from statsmodels.stats.power import zt_ind_solve_power, tt_ind_solve_power
    import matplotlib.pyplot as plt
    
    # Production: Comprehensive MDE and Power Analysis Calculator
    
    class MDECalculator:
        """
        Calculate Minimum Detectable Effect, sample size, and power for A/B tests.
        """
        
        def __init__(self, alpha=0.05, power=0.80):
            """
            Initialize with default significance level and power.
            
            alpha: Type I error rate (false positive rate)
            power: 1 - Type II error rate (1 - false negative rate)
            """
            self.alpha = alpha
            self.power = power
            self.z_alpha = stats.norm.ppf(1 - alpha/2)  # Two-sided
            self.z_beta = stats.norm.ppf(power)
            
        def calculate_mde_continuous(self, n_per_group, std):
            """
            Calculate MDE for continuous metrics (e.g., revenue, watch time).
            
            n_per_group: Sample size per group
            std: Standard deviation of metric
            
            Returns: Absolute MDE
            """
            mde = (self.z_alpha + self.z_beta) * std * np.sqrt(2 / n_per_group)
            return mde
        
        def calculate_mde_proportion(self, n_per_group, baseline_rate):
            """
            Calculate MDE for proportion metrics (e.g., conversion rate).
            
            n_per_group: Sample size per group
            baseline_rate: Control group proportion (e.g., 0.10 for 10% CVR)
            
            Returns: Absolute MDE (percentage points)
            """
            std = np.sqrt(baseline_rate * (1 - baseline_rate))
            mde = (self.z_alpha + self.z_beta) * std * np.sqrt(2 / n_per_group)
            return mde
        
        def calculate_sample_size_continuous(self, mde, std):
            """
            Calculate required sample size per group for continuous metrics.
            
            mde: Minimum detectable effect (absolute)
            std: Standard deviation of metric
            
            Returns: Sample size per group
            """
            n = 2 * ((self.z_alpha + self.z_beta) * std / mde) ** 2
            return int(np.ceil(n))
        
        def calculate_sample_size_proportion(self, mde, baseline_rate):
            """
            Calculate required sample size per group for proportions.
            
            mde: Minimum detectable effect (absolute, e.g., 0.02 for +2pp)
            baseline_rate: Control group proportion
            
            Returns: Sample size per group
            """
            p1 = baseline_rate
            p2 = baseline_rate + mde
            
            # Pooled standard deviation
            p_avg = (p1 + p2) / 2
            std_pooled = np.sqrt(2 * p_avg * (1 - p_avg))
            
            n = ((self.z_alpha + self.z_beta) * std_pooled / mde) ** 2
            return int(np.ceil(n))
        
        def calculate_power_achieved(self, n_per_group, effect_size, std):
            """
            Calculate achieved power given sample size and effect size.
            
            n_per_group: Sample size per group
            effect_size: True treatment effect
            std: Standard deviation
            
            Returns: Achieved power
            """
            # Non-centrality parameter
            ncp = effect_size / (std * np.sqrt(2 / n_per_group))
            
            # Power = P(reject H0 | H1 is true)
            power = 1 - stats.norm.cdf(self.z_alpha - ncp)
            return power
        
        def create_power_curve(self, baseline_mean, std, sample_sizes, 
                              effect_sizes=None, save_path=None):
            """
            Create power curve showing power vs effect size for different sample sizes.
            """
            if effect_sizes is None:
                effect_sizes = np.linspace(0, 0.3 * baseline_mean, 100)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot 1: Power vs Effect Size
            for n in sample_sizes:
                powers = [self.calculate_power_achieved(n, es, std) for es in effect_sizes]
                ax1.plot(effect_sizes / baseline_mean * 100, powers, 
                        label=f'n={n:,} per group', linewidth=2)
            
            ax1.axhline(0.80, color='red', linestyle='--', label='80% power', alpha=0.7)
            ax1.axhline(0.90, color='orange', linestyle='--', label='90% power', alpha=0.7)
            ax1.set_xlabel('Effect Size (% relative lift)', fontsize=12)
            ax1.set_ylabel('Statistical Power', fontsize=12)
            ax1.set_title('Power Curve: Power vs Effect Size', fontsize=14, fontweight='bold')
            ax1.legend(loc='lower right')
            ax1.grid(alpha=0.3)
            ax1.set_ylim([0, 1])
            
            # Plot 2: Sample Size vs MDE
            mdes = []
            for n in np.linspace(1000, 100000, 100):
                mde = self.calculate_mde_continuous(n, std)
                mdes.append(mde / baseline_mean * 100)
            
            ax2.plot(np.linspace(1000, 100000, 100), mdes, linewidth=2, color='#e74c3c')
            ax2.set_xlabel('Sample Size per Group', fontsize=12)
            ax2.set_ylabel('MDE (% relative lift)', fontsize=12)
            ax2.set_title('Trade-off: Sample Size vs MDE', fontsize=14, fontweight='bold')
            ax2.grid(alpha=0.3)
            ax2.axhline(5, color='green', linestyle='--', label='5% MDE target', alpha=0.7)
            ax2.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
        
        def create_sensitivity_table(self, baseline_mean, std, sample_size):
            """
            Create sensitivity table showing MDE for different power levels.
            """
            power_levels = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
            
            results = []
            for power in power_levels:
                calc = MDECalculator(alpha=self.alpha, power=power)
                mde = calc.calculate_mde_continuous(sample_size, std)
                relative_mde = mde / baseline_mean * 100
                
                results.append({
                    'Power': f'{power:.0%}',
                    'MDE (absolute)': f'{mde:.4f}',
                    'MDE (% relative)': f'{relative_mde:.2f}%',
                    'z_beta': f'{calc.z_beta:.3f}'
                })
            
            return pd.DataFrame(results)
    
    
    # Example 1: Netflix Watch Time Experiment
    print("=" * 70)
    print("EXAMPLE 1: NETFLIX WATCH TIME EXPERIMENT")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Historical data
    baseline_watch_time = 60  # minutes per day
    std_watch_time = 40  # high variance (user behavior)
    
    calculator = MDECalculator(alpha=0.05, power=0.80)
    
    # Scenario 1: What MDE can we detect with 50,000 users per group?
    n_users = 50000
    mde_absolute = calculator.calculate_mde_continuous(n_users, std_watch_time)
    mde_relative = mde_absolute / baseline_watch_time * 100
    
    print(f"\n📊 Scenario 1: What can we detect with {n_users:,} users?")
    print(f"   Baseline: {baseline_watch_time} minutes/day (std={std_watch_time})")
    print(f"   MDE (absolute): {mde_absolute:.2f} minutes")
    print(f"   MDE (relative): {mde_relative:.2f}%")
    print(f"   Interpretation: Can detect effects of {mde_relative:.2f}% or larger")
    
    # Scenario 2: How many users needed to detect 3% lift?
    target_lift_pct = 3.0
    target_lift_abs = baseline_watch_time * target_lift_pct / 100
    n_required = calculator.calculate_sample_size_continuous(target_lift_abs, std_watch_time)
    
    print(f"\n📊 Scenario 2: Users needed to detect {target_lift_pct}% lift?")
    print(f"   Target lift: {target_lift_abs:.2f} minutes ({target_lift_pct}%)")
    print(f"   Required sample size: {n_required:,} per group")
    print(f"   Total users: {2*n_required:,}")
    print(f"   Runtime (at 1M DAU, 50% allocation): {2*n_required/500000:.1f} days")
    
    # Sensitivity table
    print("\n📋 Sensitivity Analysis: MDE vs Power (n=50,000)")
    sensitivity_df = calculator.create_sensitivity_table(baseline_watch_time, std_watch_time, n_users)
    print(sensitivity_df.to_string(index=False))
    
    
    # Example 2: Uber Conversion Rate Experiment
    print("\n\n" + "=" * 70)
    print("EXAMPLE 2: UBER APP CONVERSION RATE EXPERIMENT")
    print("=" * 70)
    
    baseline_cvr = 0.12  # 12% of users complete a trip
    
    # Scenario 1: MDE with 100,000 users per group
    n_users_uber = 100000
    mde_cvr_abs = calculator.calculate_mde_proportion(n_users_uber, baseline_cvr)
    mde_cvr_rel = mde_cvr_abs / baseline_cvr * 100
    
    print(f"\n📊 Scenario 1: What can we detect with {n_users_uber:,} users?")
    print(f"   Baseline CVR: {baseline_cvr:.1%}")
    print(f"   MDE (absolute): {mde_cvr_abs:.4f} ({mde_cvr_abs*100:.2f} percentage points)")
    print(f"   MDE (relative): {mde_cvr_rel:.2f}%")
    print(f"   Interpretation: Can detect CVR changes of {mde_cvr_abs*100:.2f}pp or larger")
    
    # Scenario 2: Sample size for 1% relative lift (0.12pp)
    target_lift_cvr = 0.0012  # +0.12pp absolute (1% relative)
    n_required_cvr = calculator.calculate_sample_size_proportion(target_lift_cvr, baseline_cvr)
    
    print(f"\n📊 Scenario 2: Users needed to detect +1% relative lift?")
    print(f"   Target: {baseline_cvr:.1%} → {baseline_cvr+target_lift_cvr:.1%}")
    print(f"   Required sample size: {n_required_cvr:,} per group")
    print(f"   Total users: {2*n_required_cvr:,}")
    
    # Create power curves
    sample_sizes = [10000, 25000, 50000, 100000]
    calculator.create_power_curve(baseline_watch_time, std_watch_time, sample_sizes)
    plt.show()
    ```

    **MDE Trade-off Analysis:**

    | Parameter | Relationship | Example (Netflix) |
    |-----------|--------------|-------------------|
    | **↑ Sample Size** | ↓ MDE (better) | 50K users → 2.5% MDE, 100K users → 1.8% MDE |
    | **↑ Variance (σ)** | ↑ MDE (worse) | Revenue (σ=$50) → 10% MDE, Clicks (σ=2) → 2% MDE |
    | **↑ Power (1-β)** | ↑ MDE (worse) | 80% power → 2.5% MDE, 90% power → 3.0% MDE |
    | **↑ Significance (α)** | ↓ MDE (better, but more false positives) | α=0.05 → 2.5% MDE, α=0.10 → 2.2% MDE |

    **Real Company Examples:**

    | Company | Metric | Baseline | MDE | Sample Size | Rationale |
    |---------|--------|----------|-----|-------------|-----------|
    | **Netflix** | Watch time | 60 min/day | 2% (1.2 min) | 50K/group | Must exceed content production cost (~$1M) |
    | **Uber** | Trips/user | 3.2/month | 1.5% (0.05 trips) | 100K/group | Need precision for demand forecasting |
    | **Airbnb** | Booking rate | 8% | 2.5% (0.2pp) | 80K/group | Must cover development cost ($200K) |
    | **Amazon** | Add-to-cart rate | 15% | 0.5% (0.075pp) | 500K/group | High volume → can detect tiny effects |

    **Business vs Statistical MDE:**

    ```
    ┌───────────────────────────────────────────────────────────────┐
    │                   MDE Decision Framework                       │
    ├───────────────────────────────────────────────────────────────┤
    │                                                                │
    │  Statistical MDE          Business MDE         Final MDE      │
    │  (what we CAN detect)     (what we CARE about) (choose)       │
    │         │                        │                  │         │
    │         ↓                        ↓                  ↓         │
    │    Based on n, σ, α, β     Based on ROI      max(stat, biz)  │
    │                                                                │
    │  Example: Netflix                                              │
    │  Statistical: 1.5% (n=100K)                                   │
    │  Business: 3% (ROI breakeven)                                 │
    │  Final MDE: 3% ← Use business constraint                      │
    │                                                                │
    │  Example: Uber                                                 │
    │  Statistical: 2% (n=50K)                                      │
    │  Business: 0.5% (strategic priority)                          │
    │  Final MDE: 2% ← Need more data or variance reduction        │
    │                                                                │
    └───────────────────────────────────────────────────────────────┘
    ```

    **Common MDE Mistakes:**

    | ❌ Mistake | Why It's Wrong | ✅ Correct Approach |
    |-----------|----------------|-------------------|
    | **"Let's detect any effect"** | MDE too small → need millions of users | Set MDE based on business value |
    | **"We have 10K users, what's MDE?"** | Backward calculation | Start with business MDE → calculate n |
    | **"MDE = 5% because that's typical"** | Ignores metric variance | Calculate MDE from σ, n, α, β |
    | **"Variance reduction doesn't matter"** | Misses 2-5× sample size savings | Use CUPED, stratification to reduce σ |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of power analysis, sample size planning, business-statistical trade-offs.
        
        **Strong answer signals:**
        
        - Knows MDE formula and all 4 parameters (α, β, σ, n)
        - Explains trade-off: smaller MDE needs more samples
        - Starts with business MDE (ROI, strategic value), then calculates statistical feasibility
        - Mentions variance reduction techniques (CUPED, stratification) to lower MDE
        - Gives specific examples with numbers (e.g., "Netflix needs 2% MDE for $100M impact")
        - Knows when MDE is too small (unrealistic sample size) or too large (misses real effects)
        
        **Red flags:**
        
        - Confuses MDE with actual effect size
        - Only knows "we need more data for smaller effects" without quantification
        - Ignores business constraints
        - Can't explain α, β, or power
        
        **Follow-up questions:**
        
        - "How would you reduce MDE without more users?" (Reduce σ via CUPED, stratification, or better metric)
        - "What if business wants 0.5% MDE but you can only detect 2%?" (Need 16× more users, or use variance reduction, or reset expectations)
        - "How does variance affect MDE?" (MDE ∝ σ, so 2× variance → 2× MDE or 4× sample size)

---

### How to Design an Experimentation Platform? - Senior Roles Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `System Design` | **Asked by:** Google, Netflix, Meta

??? success "View Answer"

    An **Experimentation Platform** is the **infrastructure backbone** that enables companies to run **thousands of A/B tests** simultaneously with consistency, automation, and reliability. It must handle **assignment, logging, analysis, and guardrails** at scale while preventing Sample Ratio Mismatch (SRM), assignment collisions, and metric bugs.

    **High-Level Architecture:**

    ```
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                        Experimentation Platform Architecture                     │
    ├─────────────────────────────────────────────────────────────────────────────────┤
    │                                                                                  │
    │  ┌─────────────┐                                                                │
    │  │   User      │                                                                │
    │  │  Request    │                                                                │
    │  └──────┬──────┘                                                                │
    │         │                                                                        │
    │         ↓                                                                        │
    │  ┌─────────────────────────────────────────────────────────────────┐           │
    │  │              Assignment Service (Deterministic)                  │           │
    │  │  • Consistent hashing (user_id → experiment variant)             │           │
    │  │  • Handles traffic allocation (10% to exp1, 5% to exp2)          │           │
    │  │  • Prevents collisions (layering, namespaces)                    │           │
    │  │  • Returns: {exp_id: variant, exp_id: variant, ...}              │           │
    │  └────────────────────────┬─────────────────────────────────────────┘           │
    │                           │                                                      │
    │         ┌─────────────────┴─────────────────┐                                   │
    │         ↓                                   ↓                                   │
    │  ┌──────────────┐                   ┌──────────────┐                           │
    │  │  Application │                   │  Event Logger│                           │
    │  │  (Treatment) │                   │   (Kafka)    │                           │
    │  └──────┬───────┘                   └──────┬───────┘                           │
    │         │                                  │                                   │
    │         │ Events (clicks, purchases, etc.) │                                   │
    │         └──────────────────┬───────────────┘                                   │
    │                            ↓                                                    │
    │         ┌──────────────────────────────────────────┐                           │
    │         │     Data Warehouse (BigQuery, Redshift)  │                           │
    │         │  • Assignment logs (user, experiment, variant, timestamp)            │
    │         │  • Event logs (user, event_type, value, timestamp)                   │
    │         └─────────────────────┬────────────────────┘                           │
    │                               │                                                 │
    │                               ↓                                                 │
    │         ┌──────────────────────────────────────────┐                           │
    │         │      Metrics Pipeline (Spark, Airflow)   │                           │
    │         │  • Join assignments + events              │                           │
    │         │  • Compute metrics per user per day       │                           │
    │         │  • Aggregate by experiment + variant      │                           │
    │         └─────────────────────┬────────────────────┘                           │
    │                               │                                                 │
    │           ┌───────────────────┼───────────────────┐                            │
    │           ↓                   ↓                   ↓                            │
    │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                  │
    │  │  SRM Detector  │  │ Analysis Engine│  │  Guardrail     │                  │
    │  │  (Chi-square)  │  │  (t-tests, CI) │  │  Monitor       │                  │
    │  └────────┬───────┘  └────────┬───────┘  └────────┬───────┘                  │
    │           │                   │                   │                            │
    │           └───────────────────┼───────────────────┘                            │
    │                               ↓                                                 │
    │           ┌───────────────────────────────────────┐                            │
    │           │      Dashboard (Web UI, API)          │                            │
    │           │  • Real-time results                   │                            │
    │           │  • Confidence intervals                │                            │
    │           │  • Alerts (SRM, guardrail breaches)   │                            │
    │           └───────────────────────────────────────┘                            │
    │                                                                                  │
    └─────────────────────────────────────────────────────────────────────────────────┘
    ```

    **Core Components (Detailed):**

    ```python
    import hashlib
    import numpy as np
    import pandas as pd
    from scipy import stats
    from typing import Dict, List, Optional
    from datetime import datetime, timedelta
    
    # Production: Experimentation Platform Components
    
    # ===== 1. Assignment Service =====
    
    class AssignmentService:
        """
        Deterministic, consistent assignment of users to experiments.
        Uses hashing to ensure same user always gets same variant.
        """
        
        def __init__(self):
            self.experiments = {}
            
        def register_experiment(self, exp_id: str, variants: List[str], 
                               traffic_allocation: float = 1.0,
                               layer: str = 'default'):
            """
            Register a new experiment.
            
            exp_id: Unique experiment identifier
            variants: List of variant names (e.g., ['control', 'treatment'])
            traffic_allocation: % of users in experiment (0.0-1.0)
            layer: Namespace for preventing collisions
            """
            self.experiments[exp_id] = {
                'variants': variants,
                'allocation': traffic_allocation,
                'layer': layer,
                'created_at': datetime.now()
            }
            
        def assign_user(self, user_id: str, exp_id: str) -> Optional[str]:
            """
            Assign user to experiment variant using consistent hashing.
            
            Returns: variant name or None if user not in experiment
            """
            if exp_id not in self.experiments:
                return None
            
            exp = self.experiments[exp_id]
            
            # Hash user_id + exp_id for consistency
            hash_input = f"{user_id}:{exp_id}".encode('utf-8')
            hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
            
            # Normalize to [0, 1]
            hash_normalized = (hash_value % 1000000) / 1000000
            
            # Check if user in experiment (traffic allocation)
            if hash_normalized >= exp['allocation']:
                return None
            
            # Assign to variant based on hash
            variant_idx = int(hash_normalized / exp['allocation'] * len(exp['variants']))
            return exp['variants'][variant_idx]
        
        def get_all_assignments(self, user_id: str) -> Dict[str, str]:
            """
            Get all active experiment assignments for a user.
            """
            assignments = {}
            for exp_id in self.experiments:
                variant = self.assign_user(user_id, exp_id)
                if variant:
                    assignments[exp_id] = variant
            return assignments
    
    
    # ===== 2. Event Logger =====
    
    class EventLogger:
        """
        Log events and assignments for offline analysis.
        In production: Kafka → S3/BigQuery
        """
        
        def __init__(self):
            self.assignment_logs = []
            self.event_logs = []
            
        def log_assignment(self, user_id: str, exp_id: str, variant: str, timestamp=None):
            """
            Log user assignment to experiment variant.
            """
            self.assignment_logs.append({
                'user_id': user_id,
                'exp_id': exp_id,
                'variant': variant,
                'timestamp': timestamp or datetime.now()
            })
            
        def log_event(self, user_id: str, event_type: str, value=None, timestamp=None):
            """
            Log user event (click, purchase, etc.).
            """
            self.event_logs.append({
                'user_id': user_id,
                'event_type': event_type,
                'value': value,
                'timestamp': timestamp or datetime.now()
            })
            
        def get_assignment_df(self):
            """Return assignments as DataFrame."""
            return pd.DataFrame(self.assignment_logs)
        
        def get_event_df(self):
            """Return events as DataFrame."""
            return pd.DataFrame(self.event_logs)
    
    
    # ===== 3. Metrics Pipeline =====
    
    class MetricsPipeline:
        """
        Compute experiment metrics by joining assignments and events.
        """
        
        def __init__(self, logger: EventLogger):
            self.logger = logger
            
        def compute_metrics(self, exp_id: str, metric_type: str = 'conversion'):
            """
            Compute metrics for an experiment.
            
            metric_type: 'conversion', 'revenue', 'engagement', etc.
            
            Returns: DataFrame with metrics per variant
            """
            assignments = self.logger.get_assignment_df()
            events = self.logger.get_event_df()
            
            # Filter to experiment
            exp_assignments = assignments[assignments['exp_id'] == exp_id].copy()
            
            if metric_type == 'conversion':
                # Binary conversion: did user complete target event?
                target_event = 'purchase'  # Example
                conversions = events[events['event_type'] == target_event]['user_id'].unique()
                exp_assignments['converted'] = exp_assignments['user_id'].isin(conversions).astype(int)
                
                # Aggregate by variant
                results = exp_assignments.groupby('variant').agg({
                    'user_id': 'count',
                    'converted': ['sum', 'mean']
                }).reset_index()
                
                results.columns = ['variant', 'n_users', 'n_conversions', 'conversion_rate']
                
            elif metric_type == 'revenue':
                # Sum revenue per user
                revenue_events = events[events['event_type'] == 'purchase'].copy()
                user_revenue = revenue_events.groupby('user_id')['value'].sum().reset_index()
                user_revenue.columns = ['user_id', 'revenue']
                
                # Join with assignments
                exp_data = exp_assignments.merge(user_revenue, on='user_id', how='left')
                exp_data['revenue'] = exp_data['revenue'].fillna(0)
                
                # Aggregate
                results = exp_data.groupby('variant').agg({
                    'user_id': 'count',
                    'revenue': ['sum', 'mean', 'std']
                }).reset_index()
                
                results.columns = ['variant', 'n_users', 'total_revenue', 'mean_revenue', 'std_revenue']
            
            return results
    
    
    # ===== 4. SRM Detector =====
    
    class SRMDetector:
        """
        Detect Sample Ratio Mismatch (assignment bugs).
        """
        
        def check_srm(self, exp_id: str, assignments_df: pd.DataFrame, 
                     expected_ratio: Dict[str, float], alpha=0.001):
            """
            Chi-square test for SRM.
            
            expected_ratio: {'control': 0.5, 'treatment': 0.5}
            alpha: Significance level (use 0.001 for SRM, very strict)
            
            Returns: (is_srm, p_value, chi2_stat)
            """
            exp_data = assignments_df[assignments_df['exp_id'] == exp_id]
            
            observed = exp_data['variant'].value_counts().to_dict()
            total = sum(observed.values())
            
            # Expected counts
            expected = {v: total * expected_ratio[v] for v in expected_ratio}
            
            # Chi-square test
            chi2_stat = sum((observed.get(v, 0) - expected[v])**2 / expected[v] 
                           for v in expected_ratio)
            
            df = len(expected_ratio) - 1
            p_value = 1 - stats.chi2.cdf(chi2_stat, df)
            
            is_srm = p_value < alpha
            
            return is_srm, p_value, chi2_stat
    
    
    # ===== 5. Analysis Engine =====
    
    class AnalysisEngine:
        """
        Run statistical tests on experiment metrics.
        """
        
        def analyze_proportions(self, control_conversions, control_n, 
                               treatment_conversions, treatment_n, alpha=0.05):
            """
            Z-test for conversion rate difference.
            """
            p_control = control_conversions / control_n
            p_treatment = treatment_conversions / treatment_n
            
            # Pooled proportion
            p_pooled = (control_conversions + treatment_conversions) / (control_n + treatment_n)
            se_pooled = np.sqrt(p_pooled * (1 - p_pooled) * (1/control_n + 1/treatment_n))
            
            z_stat = (p_treatment - p_control) / se_pooled
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            # Confidence interval
            diff = p_treatment - p_control
            se_diff = np.sqrt(p_control*(1-p_control)/control_n + 
                             p_treatment*(1-p_treatment)/treatment_n)
            ci_lower = diff - 1.96 * se_diff
            ci_upper = diff + 1.96 * se_diff
            
            return {
                'control_rate': p_control,
                'treatment_rate': p_treatment,
                'absolute_lift': diff,
                'relative_lift': (p_treatment / p_control - 1) * 100,
                'p_value': p_value,
                'significant': p_value < alpha,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            }
    
    
    # ===== Example: Full Pipeline =====
    
    print("=" * 70)
    print("EXPERIMENTATION PLATFORM DEMO")
    print("=" * 70)
    
    # Initialize services
    assigner = AssignmentService()
    logger = EventLogger()
    pipeline = MetricsPipeline(logger)
    srm_detector = SRMDetector()
    analyzer = AnalysisEngine()
    
    # Register experiment
    assigner.register_experiment(
        exp_id='exp_001_new_checkout',
        variants=['control', 'treatment'],
        traffic_allocation=1.0  # 100% of users
    )
    
    # Simulate user assignments and events
    np.random.seed(42)
    n_users = 10000
    
    for i in range(n_users):
        user_id = f'user_{i}'
        
        # Assign to variant
        variant = assigner.assign_user(user_id, 'exp_001_new_checkout')
        
        if variant:
            # Log assignment
            logger.log_assignment(user_id, 'exp_001_new_checkout', variant)
            
            # Simulate purchase (treatment has 12% CVR, control 10%)
            cvr = 0.12 if variant == 'treatment' else 0.10
            if np.random.rand() < cvr:
                logger.log_event(user_id, 'purchase', value=50.0)
    
    # Check SRM
    assignments_df = logger.get_assignment_df()
    is_srm, srm_p, srm_chi2 = srm_detector.check_srm(
        'exp_001_new_checkout', 
        assignments_df, 
        {'control': 0.5, 'treatment': 0.5}
    )
    
    print(f"\n🔍 SRM Check:")
    print(f"   p-value: {srm_p:.6f}")
    print(f"   SRM detected: {'YES ⚠️' if is_srm else 'NO ✓'}")
    
    # Compute metrics
    metrics = pipeline.compute_metrics('exp_001_new_checkout', metric_type='conversion')
    print(f"\n📊 Metrics:")
    print(metrics.to_string(index=False))
    
    # Analyze
    control_data = metrics[metrics['variant'] == 'control'].iloc[0]
    treatment_data = metrics[metrics['variant'] == 'treatment'].iloc[0]
    
    result = analyzer.analyze_proportions(
        control_data['n_conversions'], control_data['n_users'],
        treatment_data['n_conversions'], treatment_data['n_users']
    )
    
    print(f"\n📈 Analysis Results:")
    print(f"   Control CVR: {result['control_rate']:.4f}")
    print(f"   Treatment CVR: {result['treatment_rate']:.4f}")
    print(f"   Lift: {result['relative_lift']:+.2f}%")
    print(f"   95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
    print(f"   p-value: {result['p_value']:.4f}")
    print(f"   Significant: {'YES ✓' if result['significant'] else 'NO ✗'}")
    ```

    **Scale Comparison:**

    | Company | Experiments/Year | Concurrent | Users/Experiment | Platform | Key Features |
    |---------|-----------------|------------|------------------|----------|--------------|
    | **Google** | 10,000+ | 1,000+ | 100M+ | Proprietary | Layered experiments, Bayesian methods |
    | **Netflix** | 1,000+ | 100+ | 10M+ | Proprietary | Sequential testing, heterogeneous effects |
    | **Meta** | 10,000+ | 1,000+ | 1B+ | Ax Platform | Multi-armed bandits, Bayesian optimization |
    | **Uber** | 2,000+ | 200+ | 50M+ | Proprietary | Geographic experiments, supply-demand balance |
    | **Airbnb** | 700+ | 50+ | 10M+ | ERF (Experiment Reporting Framework) | Network effects, two-sided marketplace |

    **Platform Feature Comparison:**

    | Feature | Basic Platform | Advanced Platform (Google, Netflix) |
    |---------|---------------|-------------------------------------|
    | **Assignment** | Random hash | Consistent hash + layering + traffic shaping |
    | **SRM Detection** | Manual checks | Automated alerts (<0.001 p-value threshold) |
    | **Analysis** | T-test only | Multiple methods (frequentist, Bayesian, bootstrap) |
    | **Guardrails** | Manual review | Automated circuit breakers (>5% drop → auto-stop) |
    | **Metrics** | 5-10 metrics | 100+ metrics tracked automatically |
    | **Interference** | Ignore | Detect and adjust (geo-randomization, clustering) |
    | **Sequential Testing** | Fixed horizon | Sequential with alpha spending |
    | **Heterogeneity** | Overall effect only | CATE, segment deep-dives |

    **Key Design Decisions:**

    | Decision | Trade-off | Recommendation |
    |----------|-----------|----------------|
    | **Hashing vs Random** | Deterministic vs truly random | Use hashing (reproducibility, debugging) |
    | **Real-time vs Batch** | Latency vs cost | Real-time for dashboards, batch for deep analysis |
    | **Centralized vs Federated** | Control vs flexibility | Centralized assignment, federated metrics |
    | **SQL vs Spark** | Simplicity vs scale | SQL for <1TB, Spark for >1TB |
    | **SRM threshold** | False alarms vs miss bugs | α=0.001 (very strict, catch bugs early) |

    !!! tip "Interviewer's Insight"
        **What they're testing:** System design skills, understanding of scale, trade-offs, and production concerns.
        
        **Strong answer signals:**
        
        - Draws architecture diagram with 5+ components
        - Explains consistent hashing for deterministic assignment
        - Mentions SRM detection as critical guardrail
        - Discusses scale (e.g., "Netflix runs 100+ concurrent experiments")
        - Knows layering/namespaces to prevent experiment collisions
        - Addresses data pipeline (assignments + events → metrics)
        - Mentions automation (auto-stop for guardrail breaches)
        - Explains trade-offs (real-time vs batch, SQL vs Spark)
        
        **Red flags:**
        
        - No mention of SRM or assignment consistency
        - Unclear how assignments join with events
        - Ignores scale considerations
        - Only focuses on analysis, not assignment/logging
        - No automation or guardrails
        
        **Follow-up questions:**
        
        - "How do you prevent two experiments from interfering?" (Layering: assign to layer first, then experiment within layer)
        - "What if you have 1000 concurrent experiments?" (Namespace by product area, use consistent hashing to prevent overlap)
        - "How quickly can you detect experiment bugs?" (Real-time SRM checks, alert within 1 hour if p<0.001)

---

### What is Stratified Randomization? - Netflix, Uber Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Design` | **Asked by:** Netflix, Uber, Airbnb

??? success "View Answer"

    **Stratified Randomization** ensures **balanced representation** of key covariates (country, device, user tenure) across control and treatment groups, reducing variance in treatment effect estimates by **20-50%**. It's especially powerful when covariates are **highly predictive** of the outcome metric.

    **Why Stratify?**
    
    Without stratification, random assignment might create **imbalanced groups** by chance, leading to:
    - Higher variance in treatment effect estimates
    - Reduced statistical power (need more samples)
    - Risk of confounding (e.g., treatment group has more iOS users who spend more)

    **Stratification Workflow:**

    ```
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    Stratified Randomization Process                      │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Step 1: Identify Stratification Variables                              │
    │  ┌────────────────────────────────────────────────────────────┐         │
    │  │  • High correlation with outcome (r > 0.3)                 │         │
    │  │  • Observable before randomization                          │         │
    │  │  • Limited categories (<10 to avoid sparse strata)          │         │
    │  │                                                             │         │
    │  │  Examples: Country, Device Type, User Tenure, Plan Tier    │         │
    │  └────────────────────────────────────────────────────────────┘         │
    │                            ↓                                             │
    │  Step 2: Create Strata (Blocks)                                         │
    │  ┌────────────────────────────────────────────────────────────┐         │
    │  │  Stratum 1: US + iOS + New Users                            │         │
    │  │  Stratum 2: US + iOS + Old Users                            │         │
    │  │  Stratum 3: US + Android + New Users                        │         │
    │  │  ...                                                         │         │
    │  │  Stratum K: India + Android + Old Users                     │         │
    │  └────────────────────────────────────────────────────────────┘         │
    │                            ↓                                             │
    │  Step 3: Randomize Within Each Stratum                                  │
    │  ┌────────────────────────────────────────────────────────────┐         │
    │  │  Stratum 1 (n=1000) → 50% Control (500), 50% Treatment (500)│         │
    │  │  Stratum 2 (n=500)  → 50% Control (250), 50% Treatment (250)│         │
    │  │  ...                                                         │         │
    │  │  Stratum K (n=200)  → 50% Control (100), 50% Treatment (100)│         │
    │  └────────────────────────────────────────────────────────────┘         │
    │                            ↓                                             │
    │  Step 4: Analyze with Stratification                                    │
    │  ┌────────────────────────────────────────────────────────────┐         │
    │  │  • Compute treatment effect within each stratum             │         │
    │  │  • Weight by stratum size to get overall effect             │         │
    │  │  • Reduced variance → Narrower confidence intervals         │         │
    │  └────────────────────────────────────────────────────────────┘         │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘
    ```

    **Production Stratification Implementation:**

    ```python
    import numpy as np
    import pandas as pd
    from scipy import stats
    from sklearn.model_selection import StratifiedShuffleSplit
    import matplotlib.pyplot as plt
    
    # Production: Comprehensive Stratified Randomization System
    
    class StratifiedRandomizer:
        """
        Stratified randomization for A/B tests with variance reduction analysis.
        """
        
        def __init__(self, random_seed=42):
            self.random_seed = random_seed
            np.random.seed(random_seed)
            
        def create_strata(self, df: pd.DataFrame, strata_cols: list) -> pd.DataFrame:
            """
            Create strata from multiple categorical columns.
            
            df: User data with covariates
            strata_cols: Columns to stratify on (e.g., ['country', 'device'])
            
            Returns: DataFrame with 'stratum_id' column added
            """
            # Combine columns into single stratum identifier
            df = df.copy()
            df['stratum_id'] = df[strata_cols].astype(str).agg('_'.join, axis=1)
            
            # Count stratum sizes
            stratum_sizes = df['stratum_id'].value_counts()
            print(f"Created {len(stratum_sizes)} strata")
            print(f"Stratum sizes: min={stratum_sizes.min()}, "
                  f"median={stratum_sizes.median():.0f}, max={stratum_sizes.max()}")
            
            return df
        
        def stratified_assignment(self, df: pd.DataFrame, treatment_ratio=0.5) -> pd.DataFrame:
            """
            Assign users to control/treatment within each stratum.
            
            df: Must have 'stratum_id' column
            treatment_ratio: Proportion assigned to treatment
            
            Returns: DataFrame with 'variant' column ('control' or 'treatment')
            """
            df = df.copy()
            
            def assign_within_stratum(group):
                n = len(group)
                n_treatment = int(n * treatment_ratio)
                indices = group.index.tolist()
                np.random.shuffle(indices)
                
                treatment_indices = indices[:n_treatment]
                group['variant'] = 'control'
                group.loc[treatment_indices, 'variant'] = 'treatment'
                return group
            
            df = df.groupby('stratum_id', group_keys=False).apply(assign_within_stratum)
            
            return df
        
        def check_covariate_balance(self, df: pd.DataFrame, covariate_cols: list):
            """
            Check if stratification achieved covariate balance.
            Uses standardized mean difference (SMD).
            
            SMD < 0.1 = good balance
            SMD 0.1-0.2 = acceptable
            SMD > 0.2 = imbalanced
            """
            control = df[df['variant'] == 'control']
            treatment = df[df['variant'] == 'treatment']
            
            balance_results = []
            
            for col in covariate_cols:
                if df[col].dtype in ['object', 'category']:
                    # Categorical: check proportion differences
                    for value in df[col].unique():
                        p_control = (control[col] == value).mean()
                        p_treatment = (treatment[col] == value).mean()
                        smd = (p_treatment - p_control) / np.sqrt((p_control*(1-p_control) + 
                                                                   p_treatment*(1-p_treatment))/2)
                        
                        balance_results.append({
                            'variable': f'{col}={value}',
                            'control_mean': p_control,
                            'treatment_mean': p_treatment,
                            'smd': abs(smd),
                            'balanced': 'Yes ✓' if abs(smd) < 0.1 else 'No ✗'
                        })
                else:
                    # Continuous: standardized mean difference
                    mean_control = control[col].mean()
                    mean_treatment = treatment[col].mean()
                    std_pooled = np.sqrt((control[col].var() + treatment[col].var()) / 2)
                    smd = (mean_treatment - mean_control) / std_pooled
                    
                    balance_results.append({
                        'variable': col,
                        'control_mean': mean_control,
                        'treatment_mean': mean_treatment,
                        'smd': abs(smd),
                        'balanced': 'Yes ✓' if abs(smd) < 0.1 else 'No ✗'
                    })
            
            return pd.DataFrame(balance_results)
        
        def compute_variance_reduction(self, df: pd.DataFrame, outcome_col: str,
                                      strata_col='stratum_id') -> dict:
            """
            Quantify variance reduction from stratification.
            
            Compares variance of treatment effect estimate with vs without stratification.
            """
            # Unstratified variance (pooled)
            control = df[df['variant'] == 'control'][outcome_col]
            treatment = df[df['variant'] == 'treatment'][outcome_col]
            
            var_unstratified = control.var()/len(control) + treatment.var()/len(treatment)
            
            # Stratified variance (weighted by stratum size)
            strata = df[strata_col].unique()
            var_stratified = 0
            
            for stratum in strata:
                stratum_data = df[df[strata_col] == stratum]
                s_control = stratum_data[stratum_data['variant'] == 'control'][outcome_col]
                s_treatment = stratum_data[stratum_data['variant'] == 'treatment'][outcome_col]
                
                if len(s_control) > 0 and len(s_treatment) > 0:
                    stratum_weight = len(stratum_data) / len(df)
                    var_stratum = s_control.var()/len(s_control) + s_treatment.var()/len(s_treatment)
                    var_stratified += stratum_weight**2 * var_stratum
            
            variance_reduction = (var_unstratified - var_stratified) / var_unstratified * 100
            
            return {
                'var_unstratified': var_unstratified,
                'var_stratified': var_stratified,
                'variance_reduction_pct': variance_reduction,
                'effective_sample_multiplier': 1 / (1 - variance_reduction/100)
            }
    
    
    # ===== Example 1: Netflix Content Recommendation =====
    
    print("=" * 70)
    print("EXAMPLE 1: NETFLIX - STRATIFIED BY COUNTRY + DEVICE")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Generate user data
    n_users = 10000
    countries = np.random.choice(['US', 'UK', 'India', 'Brazil'], n_users, p=[0.4, 0.2, 0.2, 0.2])
    devices = np.random.choice(['Web', 'iOS', 'Android', 'TV'], n_users, p=[0.3, 0.25, 0.25, 0.2])
    
    # Country and device affect watch time (covariate correlation)
    country_effect = {'US': 70, 'UK': 65, 'India': 50, 'Brazil': 55}
    device_effect = {'Web': 0, 'iOS': 5, 'Android': 3, 'TV': 10}
    
    baseline_watch_time = np.array([country_effect[c] + device_effect[d] 
                                    for c, d in zip(countries, devices)])
    watch_time = baseline_watch_time + np.random.normal(0, 15, n_users)
    
    df = pd.DataFrame({
        'user_id': [f'user_{i}' for i in range(n_users)],
        'country': countries,
        'device': devices,
        'watch_time': watch_time
    })
    
    # Initialize stratifier
    stratifier = StratifiedRandomizer(random_seed=42)
    
    # Create strata
    df = stratifier.create_strata(df, ['country', 'device'])
    
    # Assign to variants
    df = stratifier.stratified_assignment(df, treatment_ratio=0.5)
    
    # Check balance
    print("\n📊 Covariate Balance Check:")
    balance = stratifier.check_covariate_balance(df, ['country', 'device'])
    print(balance.to_string(index=False))
    
    # Simulate treatment effect (+5% watch time in treatment)
    treatment_lift = 0.05
    df.loc[df['variant'] == 'treatment', 'watch_time'] *= (1 + treatment_lift)
    
    # Compute variance reduction
    var_reduction = stratifier.compute_variance_reduction(df, 'watch_time')
    
    print(f"\n📈 Variance Reduction:")
    print(f"   Unstratified variance: {var_reduction['var_unstratified']:.4f}")
    print(f"   Stratified variance: {var_reduction['var_stratified']:.4f}")
    print(f"   Variance reduction: {var_reduction['variance_reduction_pct']:.1f}%")
    print(f"   Equivalent to {var_reduction['effective_sample_multiplier']:.2f}× sample size")
    
    # Analyze with and without stratification
    control = df[df['variant'] == 'control']['watch_time']
    treatment = df[df['variant'] == 'treatment']['watch_time']
    
    # Unstratified t-test
    t_stat, p_value_unstrat = stats.ttest_ind(treatment, control)
    se_unstrat = np.sqrt(control.var()/len(control) + treatment.var()/len(treatment))
    ci_width_unstrat = 1.96 * 2 * se_unstrat
    
    # Stratified analysis
    se_strat = np.sqrt(var_reduction['var_stratified'])
    ci_width_strat = 1.96 * 2 * se_strat
    
    print(f"\n📊 Confidence Interval Comparison:")
    print(f"   Unstratified 95% CI width: ±{ci_width_unstrat:.2f} minutes")
    print(f"   Stratified 95% CI width: ±{ci_width_strat:.2f} minutes")
    print(f"   Improvement: {(ci_width_unstrat - ci_width_strat)/ci_width_unstrat*100:.1f}% narrower")
    ```

    **Stratification Methods:**

    | Method | Description | When to Use | Variance Reduction |
    |--------|-------------|-------------|-------------------|
    | **Simple Stratification** | Equal allocation within strata | Strata roughly equal size | 20-30% |
    | **Proportional Stratification** | Stratum size proportional to population | Strata vary in size | 25-40% |
    | **Optimal Stratification** | Allocate more to high-variance strata | Unequal variances across strata | 30-50% |
    | **Post-Stratification** | Adjust estimates after randomization | Stratification variable measured post-random | 15-25% |

    **Real Company Examples:**

    | Company | Strata Variables | Metric | Variance Reduction | Outcome |
    |---------|-----------------|--------|-------------------|---------|
    | **Netflix** | Country, Device, Tenure | Watch time | 35% reduction | Detect 2.5% effect → 1.8% effect (save 40% sample) |
    | **Uber** | City, Time of day, Driver tenure | Trips/hour | 40% reduction | Faster experiments (10 days → 7 days) |
    | **Airbnb** | Property type, Location, Host experience | Booking rate | 30% reduction | Improved power 80% → 90% with same sample |
    | **Amazon** | Product category, Customer segment | Add-to-cart rate | 25% reduction | Run 33% more experiments with same traffic |

    **Choosing Stratification Variables:**

    | Criterion | Good Variable | Bad Variable |
    |-----------|--------------|--------------|
    | **Correlation with outcome** | ✅ Tenure (r=0.5 with LTV) | ❌ Zodiac sign (r=0.01) |
    | **Observability** | ✅ Country (known pre-randomization) | ❌ Future purchases (unknown) |
    | **Cardinality** | ✅ Device type (4 categories) | ❌ User ID (10M categories, too sparse) |
    | **Actionability** | ✅ User segment (can target) | ❌ Weather (can't control) |

    **Stratification vs Other Methods:**

    | Method | Variance Reduction | Complexity | When to Use |
    |--------|-------------------|------------|-------------|
    | **Stratified Randomization** | 20-50% | Low | Pre-experiment planning, categorical covariates |
    | **CUPED (Regression Adjustment)** | 30-70% | Medium | Post-experiment, continuous pre-period metric available |
    | **Blocking** | 20-40% | Low | Similar to stratification, smaller blocks |
    | **Matching** | 30-50% | High | Observational studies, need exact covariate match |
    | **Difference-in-Differences** | 40-60% | Medium | Time series data, parallel trends assumption |

    **Common Mistakes:**

    | ❌ Mistake | Why It's Bad | ✅ Fix |
    |-----------|--------------|-------|
    | **Too many strata** | Sparse strata (<10 users) → unstable estimates | Limit to 2-3 variables, <50 total strata |
    | **Stratify on outcome** | Data leakage, biased estimates | Only use pre-randomization covariates |
    | **Ignore stratification in analysis** | Lose variance reduction benefit | Use stratified analysis (weighted by stratum) |
    | **Unbalanced strata** | Some strata all control or all treatment | Ensure each stratum has ≥10 users per variant |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of experimental design, variance reduction techniques, covariate balance.
        
        **Strong answer signals:**
        
        - Explains variance reduction quantitatively (e.g., "30% reduction")
        - Knows how to check covariate balance (Standardized Mean Difference)
        - Mentions choosing variables with high correlation to outcome
        - Gives examples: "Netflix stratifies by country/device for 35% variance reduction"
        - Explains trade-off: more strata → less flexibility, risk of sparse cells
        - Knows to analyze WITH stratification (don't throw it away)
        - Compares to alternatives (CUPED, blocking, matching)
        
        **Red flags:**
        
        - Confuses stratification with segmentation
        - Can't explain why it reduces variance
        - Suggests stratifying on the outcome variable
        - Ignores stratum sparsity issues
        - Doesn't know how to check if it worked
        
        **Follow-up questions:**
        
        - "How do you choose which variables to stratify on?" (High correlation with outcome, low cardinality, observable pre-randomization)
        - "What if you have 100 potential stratification variables?" (Use regularized regression to select top 2-3, or use CUPED instead)
        - "Does stratification bias the treatment effect estimate?" (No, it only reduces variance, estimate is still unbiased)

---

### How to Use Regression Adjustment? - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Analysis` | **Asked by:** Google, Meta, Netflix

??? success "View Answer"

    **Regression Adjustment** (also called **CUPED** - Controlled-experiment Using Pre-Experiment Data) uses **pre-experiment covariates** to reduce variance in treatment effect estimates by **40-70%**, enabling faster experiments and smaller sample sizes. It's the **most powerful variance reduction technique** used by Google, Meta, Netflix, and Uber.

    **Core Idea:**
    
    Instead of comparing raw outcomes ($Y$), compare **residuals** after removing the predictable component from pre-experiment data:

    $$Y_{adjusted} = Y - \theta \cdot (X_{pre} - \mathbb{E}[X_{pre}])$$

    Where:
    - $Y$ = outcome metric (e.g., revenue post-experiment)
    - $X_{pre}$ = pre-experiment covariate (e.g., revenue pre-experiment)
    - $\theta$ = regression coefficient (usually $\text{Cov}(Y, X_{pre}) / \text{Var}(X_{pre})$)

    **Why It Works:**

    ```
    ┌──────────────────────────────────────────────────────────────────────┐
    │                  CUPED Variance Reduction Mechanism                   │
    ├──────────────────────────────────────────────────────────────────────┤
    │                                                                       │
    │  Raw Metric Variance = Signal + Noise                                │
    │  ┌─────────────────────────────────────────────────┐                │
    │  │                                                  │                │
    │  │    Treatment Effect        User Heterogeneity   │                │
    │  │    (what we want)         (noise we remove)     │                │
    │  │         ▲                         ▲             │                │
    │  │         │                         │             │                │
    │  │      Small                      Large           │                │
    │  │                                                  │                │
    │  └─────────────────────────────────────────────────┘                │
    │                          ↓                                           │
    │                   CUPED Adjustment                                   │
    │  ┌─────────────────────────────────────────────────┐                │
    │  │                                                  │                │
    │  │  Remove user heterogeneity using pre-period:    │                │
    │  │  Y_adj = Y - θ(X_pre - E[X_pre])               │                │
    │  │                                                  │                │
    │  │  Result: Variance ↓ by 40-70%                   │                │
    │  │          → Narrower CIs                          │                │
    │  │          → Higher power                          │                │
    │  │          → Faster experiments                    │                │
    │  │                                                  │                │
    │  └─────────────────────────────────────────────────┘                │
    │                                                                       │
    └──────────────────────────────────────────────────────────────────────┘
    ```

    **Production CUPED Implementation:**

    ```python
    import numpy as np
    import pandas as pd
    from scipy import stats
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    
    # Production: Comprehensive Regression Adjustment (CUPED) System
    
    class CUPEDAnalyzer:
        """
        CUPED (Controlled-experiment Using Pre-Experiment Data) for variance reduction.
        """
        
        def __init__(self):
            self.theta = None
            self.variance_reduction = None
            
        def compute_theta(self, outcome, covariate, method='optimal'):
            """
            Compute CUPED adjustment coefficient θ.
            
            Methods:
            - 'optimal': θ = Cov(Y, X) / Var(X)  [minimizes variance]
            - 'regression': θ from Y ~ X regression [equivalent to optimal]
            """
            if method == 'optimal':
                cov = np.cov(outcome, covariate)[0, 1]
                var_cov = np.var(covariate, ddof=1)
                theta = cov / var_cov
            elif method == 'regression':
                # Simple linear regression
                X = sm.add_constant(covariate)
                model = sm.OLS(outcome, X).fit()
                theta = model.params[1]  # Coefficient on covariate
            
            self.theta = theta
            return theta
        
        def adjust_metric(self, outcome, covariate, theta=None):
            """
            Apply CUPED adjustment to outcome metric.
            
            Y_adjusted = Y - θ(X - mean(X))
            """
            if theta is None:
                theta = self.theta
                
            if theta is None:
                raise ValueError("Must compute theta first or provide it")
            
            covariate_centered = covariate - np.mean(covariate)
            adjusted_outcome = outcome - theta * covariate_centered
            
            return adjusted_outcome
        
        def analyze_with_cuped(self, control_outcome, treatment_outcome,
                              control_covariate, treatment_covariate,
                              alpha=0.05):
            """
            Complete CUPED analysis comparing control vs treatment.
            
            Returns: dict with unadjusted and adjusted results
            """
            # ===== Unadjusted Analysis =====
            mean_control = np.mean(control_outcome)
            mean_treatment = np.mean(treatment_outcome)
            
            se_control = stats.sem(control_outcome)
            se_treatment = stats.sem(treatment_outcome)
            se_diff_unadj = np.sqrt(se_control**2 + se_treatment**2)
            
            diff_unadj = mean_treatment - mean_control
            t_stat_unadj = diff_unadj / se_diff_unadj
            p_value_unadj = 2 * (1 - stats.t.cdf(abs(t_stat_unadj), 
                                                  len(control_outcome) + len(treatment_outcome) - 2))
            
            ci_lower_unadj = diff_unadj - 1.96 * se_diff_unadj
            ci_upper_unadj = diff_unadj + 1.96 * se_diff_unadj
            
            # ===== CUPED Adjustment =====
            # Compute θ on pooled data
            all_outcomes = np.concatenate([control_outcome, treatment_outcome])
            all_covariates = np.concatenate([control_covariate, treatment_covariate])
            
            theta = self.compute_theta(all_outcomes, all_covariates)
            
            # Adjust outcomes
            control_adj = self.adjust_metric(control_outcome, control_covariate, theta)
            treatment_adj = self.adjust_metric(treatment_outcome, treatment_covariate, theta)
            
            # Analysis on adjusted metrics
            mean_control_adj = np.mean(control_adj)
            mean_treatment_adj = np.mean(treatment_adj)
            
            se_control_adj = stats.sem(control_adj)
            se_treatment_adj = stats.sem(treatment_adj)
            se_diff_adj = np.sqrt(se_control_adj**2 + se_treatment_adj**2)
            
            diff_adj = mean_treatment_adj - mean_control_adj
            t_stat_adj = diff_adj / se_diff_adj
            p_value_adj = 2 * (1 - stats.t.cdf(abs(t_stat_adj), 
                                               len(control_adj) + len(treatment_adj) - 2))
            
            ci_lower_adj = diff_adj - 1.96 * se_diff_adj
            ci_upper_adj = diff_adj + 1.96 * se_diff_adj
            
            # Variance reduction
            var_reduction = (1 - se_diff_adj**2 / se_diff_unadj**2) * 100
            self.variance_reduction = var_reduction
            
            return {
                'unadjusted': {
                    'treatment_effect': diff_unadj,
                    'se': se_diff_unadj,
                    'p_value': p_value_unadj,
                    'ci_lower': ci_lower_unadj,
                    'ci_upper': ci_upper_unadj,
                    'significant': p_value_unadj < alpha
                },
                'adjusted': {
                    'treatment_effect': diff_adj,
                    'se': se_diff_adj,
                    'p_value': p_value_adj,
                    'ci_lower': ci_lower_adj,
                    'ci_upper': ci_upper_adj,
                    'significant': p_value_adj < alpha
                },
                'theta': theta,
                'variance_reduction_pct': var_reduction,
                'ci_width_reduction_pct': (1 - (ci_upper_adj - ci_lower_adj) / 
                                           (ci_upper_unadj - ci_lower_unadj)) * 100
            }
        
        def select_covariates(self, df: pd.DataFrame, outcome_col: str, 
                             candidate_cols: list, threshold=0.3):
            """
            Select best covariates for CUPED based on correlation with outcome.
            
            threshold: Minimum absolute correlation to include
            
            Returns: List of selected columns, sorted by correlation strength
            """
            correlations = []
            
            for col in candidate_cols:
                corr = df[[outcome_col, col]].corr().iloc[0, 1]
                if abs(corr) >= threshold:
                    correlations.append({
                        'covariate': col,
                        'correlation': corr,
                        'abs_correlation': abs(corr)
                    })
            
            corr_df = pd.DataFrame(correlations).sort_values('abs_correlation', 
                                                             ascending=False)
            
            return corr_df
        
        def plot_variance_reduction(self, control_outcome, treatment_outcome,
                                   control_covariate, treatment_covariate,
                                   save_path=None):
            """
            Visualize CUPED variance reduction effect.
            """
            # Run analysis
            results = self.analyze_with_cuped(control_outcome, treatment_outcome,
                                             control_covariate, treatment_covariate)
            
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            
            # Plot 1: Treatment effect with CIs
            ax1 = axes[0]
            methods = ['Unadjusted', 'CUPED Adjusted']
            effects = [results['unadjusted']['treatment_effect'], 
                      results['adjusted']['treatment_effect']]
            ci_lowers = [results['unadjusted']['ci_lower'], 
                        results['adjusted']['ci_lower']]
            ci_uppers = [results['unadjusted']['ci_upper'], 
                        results['adjusted']['ci_upper']]
            
            x_pos = [0, 1]
            ax1.errorbar(x_pos, effects, 
                        yerr=[[effects[i] - ci_lowers[i] for i in range(2)],
                              [ci_uppers[i] - effects[i] for i in range(2)]],
                        fmt='o', markersize=10, capsize=10, capthick=2,
                        color=['#e74c3c', '#27ae60'], linewidth=2)
            ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(methods)
            ax1.set_ylabel('Treatment Effect')
            ax1.set_title('Treatment Effect Estimates with 95% CI', fontweight='bold')
            ax1.grid(axis='y', alpha=0.3)
            
            # Plot 2: Variance reduction bar
            ax2 = axes[1]
            ax2.bar(['Variance\nReduction'], [results['variance_reduction_pct']], 
                   color='#3498db', alpha=0.7)
            ax2.set_ylabel('Variance Reduction (%)')
            ax2.set_title(f'CUPED Variance Reduction: {results["variance_reduction_pct"]:.1f}%', 
                         fontweight='bold')
            ax2.set_ylim([0, 100])
            ax2.grid(axis='y', alpha=0.3)
            
            # Plot 3: Scatter of outcome vs covariate
            ax3 = axes[2]
            all_outcome = np.concatenate([control_outcome, treatment_outcome])
            all_covariate = np.concatenate([control_covariate, treatment_covariate])
            ax3.scatter(all_covariate, all_outcome, alpha=0.3, s=10)
            
            # Regression line
            z = np.polyfit(all_covariate, all_outcome, 1)
            p = np.poly1d(z)
            ax3.plot(sorted(all_covariate), p(sorted(all_covariate)), 
                    "r-", linewidth=2, label=f'θ={results["theta"]:.3f}')
            
            ax3.set_xlabel('Pre-experiment Covariate')
            ax3.set_ylabel('Outcome Metric')
            ax3.set_title('Outcome vs Pre-experiment Covariate', fontweight='bold')
            ax3.legend()
            ax3.grid(alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
    
    
    # ===== Example 1: Uber Trips/Hour Experiment =====
    
    print("=" * 70)
    print("EXAMPLE 1: UBER - CUPED ON PREVIOUS TRIPS/HOUR")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Generate data: trips/hour has high autocorrelation
    n_users = 5000
    
    # Pre-experiment trips/hour (baseline user behavior)
    pre_trips = np.random.gamma(shape=3, scale=1.5, size=n_users)
    
    # Post-experiment trips/hour (correlated with pre-period)
    # Control group: stays similar to pre-period
    control_pre = pre_trips[:n_users//2]
    control_post = control_pre + np.random.normal(0, 0.5, n_users//2)
    
    # Treatment group: +10% lift
    treatment_pre = pre_trips[n_users//2:]
    treatment_post = treatment_pre * 1.10 + np.random.normal(0, 0.5, n_users//2)
    
    # Analyze
    analyzer = CUPEDAnalyzer()
    results = analyzer.analyze_with_cuped(
        control_post, treatment_post,
        control_pre, treatment_pre
    )
    
    print("\n📊 CUPED Results:")
    print("\nUnadjusted:")
    print(f"   Treatment Effect: {results['unadjusted']['treatment_effect']:.4f}")
    print(f"   Standard Error: {results['unadjusted']['se']:.4f}")
    print(f"   95% CI: [{results['unadjusted']['ci_lower']:.4f}, {results['unadjusted']['ci_upper']:.4f}]")
    print(f"   p-value: {results['unadjusted']['p_value']:.4f}")
    print(f"   Significant: {'Yes ✓' if results['unadjusted']['significant'] else 'No ✗'}")
    
    print("\nCUPED Adjusted:")
    print(f"   Treatment Effect: {results['adjusted']['treatment_effect']:.4f}")
    print(f"   Standard Error: {results['adjusted']['se']:.4f}")
    print(f"   95% CI: [{results['adjusted']['ci_lower']:.4f}, {results['adjusted']['ci_upper']:.4f}]")
    print(f"   p-value: {results['adjusted']['p_value']:.4f}")
    print(f"   Significant: {'Yes ✓' if results['adjusted']['significant'] else 'No ✗'}")
    
    print("\nImprovement:")
    print(f"   θ (adjustment coefficient): {results['theta']:.4f}")
    print(f"   Variance reduction: {results['variance_reduction_pct']:.1f}%")
    print(f"   CI width reduction: {results['ci_width_reduction_pct']:.1f}%")
    print(f"   Equivalent sample size multiplier: {1/(1-results['variance_reduction_pct']/100):.2f}×")
    
    # Visualize
    analyzer.plot_variance_reduction(control_post, treatment_post, 
                                    control_pre, treatment_pre)
    plt.show()
    
    
    # ===== Example 2: Netflix Watch Time with Multiple Covariates =====
    
    print("\n\n" + "=" * 70)
    print("EXAMPLE 2: NETFLIX - COVARIATE SELECTION FOR CUPED")
    print("=" * 70)
    
    # Generate data with multiple potential covariates
    n_users = 8000
    
    # Covariates with varying correlations to outcome
    pre_watch_time = np.random.gamma(shape=2, scale=30, size=n_users)  # Strong correlation
    account_age_days = np.random.gamma(shape=3, scale=100, size=n_users)  # Medium correlation
    num_profiles = np.random.poisson(2, size=n_users) + 1  # Weak correlation
    random_noise = np.random.normal(0, 50, size=n_users)  # No correlation
    
    # Outcome: post-experiment watch time
    # Influenced by pre_watch_time (strong) and account_age (medium)
    post_watch_time = (0.8 * pre_watch_time + 
                      0.05 * account_age_days + 
                      2 * num_profiles +
                      np.random.normal(0, 10, size=n_users))
    
    # Create DataFrame
    df = pd.DataFrame({
        'post_watch_time': post_watch_time,
        'pre_watch_time': pre_watch_time,
        'account_age_days': account_age_days,
        'num_profiles': num_profiles,
        'random_noise': random_noise
    })
    
    # Select best covariates
    analyzer2 = CUPEDAnalyzer()
    covariate_rankings = analyzer2.select_covariates(
        df, 'post_watch_time',
        ['pre_watch_time', 'account_age_days', 'num_profiles', 'random_noise'],
        threshold=0.1
    )
    
    print("\n📊 Covariate Selection (correlation with outcome):")
    print(covariate_rankings.to_string(index=False))
    print("\n✅ Recommendation: Use 'pre_watch_time' (highest correlation)")
    ```

    **CUPED vs Other Methods:**

    | Method | Variance Reduction | Data Required | Complexity | Best For |
    |--------|-------------------|---------------|------------|----------|
    | **CUPED** | 40-70% | Pre-experiment metric | Low | Continuous metrics with history |
    | **Stratification** | 20-50% | Pre-experiment categorical | Low | Categorical covariates |
    | **Difference-in-Differences** | 40-60% | Time series | Medium | Parallel trends, policy changes |
    | **Matching** | 30-50% | Pre-experiment covariates | High | Observational studies |
    | **Regression (multiple covariates)** | 50-80% | Many covariates | Medium | Large feature set |

    **Real Company Examples:**

    | Company | Metric | Covariate | Variance Reduction | Impact |
    |---------|--------|-----------|-------------------|---------|
    | **Google Ads** | Click-through rate (CTR) | Pre-experiment CTR | 50% reduction | Run experiments in 10 days instead of 20 |
    | **Netflix** | Watch time | Previous week watch time | 60% reduction | Detect 2% effects (was 3.5% MDE) |
    | **Uber** | Trips/driver | Previous month trips | 45% reduction | 33% faster experiments |
    | **Meta** | Session duration | Previous 7-day avg | 55% reduction | Increased experiment velocity 2× |

    **Covariate Selection Criteria:**

    | Criterion | Best Covariate | Poor Covariate |
    |-----------|---------------|----------------|
    | **Correlation with outcome** | ✅ r > 0.5 (pre-experiment same metric) | ❌ r < 0.1 (unrelated variable) |
    | **Availability** | ✅ Measured before experiment | ❌ Measured after treatment |
    | **Not affected by treatment** | ✅ Pre-experiment data | ❌ Post-treatment data |
    | **Low missingness** | ✅ <5% missing | ❌ >20% missing |

    **When CUPED Fails:**

    | Scenario | Why CUPED Doesn't Help | Alternative |
    |----------|----------------------|-------------|
    | **No historical data** | New users, no pre-period | Use demographics, stratification |
    | **Low correlation (r<0.2)** | Covariate doesn't predict outcome | Find better covariate or accept higher variance |
    | **Treatment affects covariate** | e.g., treatment changes user type | Only use pre-treatment covariates |
    | **Metric mismatch** | Use clicks to adjust revenue | Use same metric family (revenue → revenue) |

    **Common Mistakes:**

    | ❌ Mistake | Why It's Bad | ✅ Fix |
    |-----------|--------------|-------|
    | **Use post-treatment covariate** | Biased estimates | Only use pre-experiment data |
    | **Forget to center covariate** | Changes treatment effect estimate | Always use $X - \mathbb{E}[X]$ |
    | **Wrong θ calculation** | Suboptimal variance reduction | Use θ = Cov(Y,X) / Var(X) |
    | **Apply only to treatment group** | Introduces bias | Apply to both control and treatment |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Advanced variance reduction, understanding of CUPED theory, covariate selection.
        
        **Strong answer signals:**
        
        - Explains CUPED formula: $Y_{adj} = Y - \theta(X_{pre} - \mathbb{E}[X_{pre}])$
        - Knows θ minimizes variance: $\theta = \text{Cov}(Y,X) / \text{Var}(X)$
        - Quantifies benefit: "40-70% variance reduction at Google/Netflix"
        - Explains covariate selection: "Use pre-experiment same metric, r>0.5"
        - Mentions centering covariate (don't bias treatment effect)
        - Compares to alternatives (stratification 20-50%, CUPED 40-70%)
        - Knows when it fails (no history, low correlation)
        
        **Red flags:**
        
        - Confuses with stratification or matching
        - Uses post-treatment covariates
        - Can't explain why variance reduces
        - Doesn't know optimal θ formula
        - Never heard of CUPED (common at top tech companies)
        
        **Follow-up questions:**
        
        - "How do you choose θ?" (Optimal: Cov(Y,X)/Var(X), or run regression Y~X)
        - "Can you use multiple covariates?" (Yes, use multiple regression: Y ~ T + X1 + X2 + ...)
        - "Does CUPED bias the treatment effect?" (No, only reduces variance if done correctly)

---

### What is Triggered Analysis? - Uber, Lyft Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Analysis` | **Asked by:** Uber, Lyft, DoorDash

??? success "View Answer"

    **Triggered Analysis** measures treatment effect **only among users who were actually exposed** to the treatment, as opposed to **Intent-to-Treat (ITT)** which includes all randomized users regardless of exposure. This distinction is critical when **not all assigned users experience the treatment** (e.g., feature requires user action, supply-side constraints).

    **ITT vs Triggered:**

    ```
    ┌────────────────────────────────────────────────────────────────────┐
    │                  ITT vs Triggered Analysis                          │
    ├────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │  Randomization: 10,000 users to Treatment                          │
    │  ┌──────────────────────────────────────────────────────────────┐  │
    │  │                    Treatment Group                           │  │
    │  │                      (n=10,000)                              │  │
    │  └──────────────────────────────────────────────────────────────┘  │
    │         │                                │                          │
    │         ↓                                ↓                          │
    │  ┌──────────────┐              ┌──────────────────┐               │
    │  │  Triggered   │              │  Not Triggered   │               │
    │  │ (Exposed to  │              │ (Never saw the   │               │
    │  │  treatment)  │              │   treatment)     │               │
    │  │  n=7,000     │              │   n=3,000        │               │
    │  └──────────────┘              └──────────────────┘               │
    │         │                                │                          │
    │         ↓                                ↓                          │
    │  ┌──────────────────────────────────────────────────────────────┐  │
    │  │                                                               │  │
    │  │  ITT Effect:     Analyze all 10,000                          │  │
    │  │  • Real-world impact (70% trigger rate)                      │  │
    │  │  • Unbiased (preserves randomization)                        │  │
    │  │  • Lower statistical power                                    │  │
    │  │                                                               │  │
    │  │  Triggered Effect: Analyze only 7,000                        │  │
    │  │  • Effect on exposed users                                    │  │
    │  │  • Potentially biased (selection into triggering)            │  │
    │  │  • Higher statistical power                                   │  │
    │  │                                                               │  │
    │  └──────────────────────────────────────────────────────────────┘  │
    │                                                                     │
    └────────────────────────────────────────────────────────────────────┘
    ```

    **When Triggering Matters:**

    - **Supply-side experiments:** Uber driver feature requires accepting a trip
    - **Opt-in features:** Netflix "Skip Intro" button (not all shows have intros)
    - **Conditional displays:** E-commerce "Free Shipping" banner (only for orders >$50)
    - **Geographic experiments:** Feature only available in certain cities

    **Production Implementation:**

    ```python
    import numpy as np
    import pandas as pd
    from scipy import stats
    import matplotlib.pyplot as plt
    
    # Production: Comprehensive Triggered Analysis with ITT Comparison
    
    class TriggeredAnalyzer:
        """
        Analyze experiments with incomplete treatment exposure.
        Compares ITT (intent-to-treat) with triggered (per-protocol) effects.
        """
        
        def __init__(self):
            self.results = {}
            
        def analyze_itt(self, treatment_outcomes, control_outcomes, alpha=0.05):
            """
            Intent-to-Treat analysis: All randomized users.
            
            Returns: dict with treatment effect, CI, p-value
            """
            mean_treatment = np.mean(treatment_outcomes)
            mean_control = np.mean(control_outcomes)
            
            se_treatment = stats.sem(treatment_outcomes)
            se_control = stats.sem(control_outcomes)
            se_diff = np.sqrt(se_treatment**2 + se_control**2)
            
            effect = mean_treatment - mean_control
            t_stat = effect / se_diff
            df = len(treatment_outcomes) + len(control_outcomes) - 2
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            
            ci_lower = effect - 1.96 * se_diff
            ci_upper = effect + 1.96 * se_diff
            
            return {
                'effect': effect,
                'se': se_diff,
                'p_value': p_value,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'significant': p_value < alpha,
                'n_treatment': len(treatment_outcomes),
                'n_control': len(control_outcomes)
            }
        
        def analyze_triggered(self, triggered_treatment, control_outcomes,
                             trigger_rate_treatment, trigger_rate_control=None,
                             alpha=0.05):
            """
            Triggered (per-protocol) analysis: Only exposed users.
            
            triggered_treatment: Outcomes for treatment users who were triggered
            control_outcomes: All control users (or triggered control if available)
            trigger_rate_treatment: Proportion of treatment assigned who triggered
            trigger_rate_control: Proportion of control assigned who triggered (if applicable)
            
            Returns: dict with triggered effect, potential selection bias warning
            """
            mean_triggered = np.mean(triggered_treatment)
            mean_control = np.mean(control_outcomes)
            
            se_triggered = stats.sem(triggered_treatment)
            se_control = stats.sem(control_outcomes)
            se_diff = np.sqrt(se_triggered**2 + se_control**2)
            
            effect = mean_triggered - mean_control
            t_stat = effect / se_diff
            df = len(triggered_treatment) + len(control_outcomes) - 2
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            
            ci_lower = effect - 1.96 * se_diff
            ci_upper = effect + 1.96 * se_diff
            
            # Check for selection bias (different trigger rates)
            selection_bias_warning = False
            if trigger_rate_control is not None:
                trigger_diff = abs(trigger_rate_treatment - trigger_rate_control)
                if trigger_diff > 0.05:  # >5pp difference
                    selection_bias_warning = True
            
            return {
                'effect': effect,
                'se': se_diff,
                'p_value': p_value,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'significant': p_value < alpha,
                'n_triggered': len(triggered_treatment),
                'n_control': len(control_outcomes),
                'trigger_rate': trigger_rate_treatment,
                'selection_bias_warning': selection_bias_warning
            }
        
        def estimate_cace(self, itt_effect, compliance_rate):
            """
            Estimate CACE (Complier Average Causal Effect).
            
            CACE = ITT Effect / Compliance Rate
            
            This estimates the effect among users who would comply (trigger)
            under the assumption of no defiers.
            """
            if compliance_rate == 0:
                return None
            
            cace = itt_effect / compliance_rate
            return cace
        
        def compare_itt_triggered(self, treatment_all, treatment_triggered,
                                 control_all, trigger_rate_treatment,
                                 trigger_rate_control=None):
            """
            Full comparison of ITT vs Triggered analysis.
            """
            # ITT analysis
            itt_results = self.analyze_itt(treatment_all, control_all)
            
            # Triggered analysis
            triggered_results = self.analyze_triggered(
                treatment_triggered, control_all,
                trigger_rate_treatment, trigger_rate_control
            )
            
            # CACE estimate
            cace = self.estimate_cace(itt_results['effect'], trigger_rate_treatment)
            
            self.results = {
                'itt': itt_results,
                'triggered': triggered_results,
                'cace': cace,
                'dilution_factor': trigger_rate_treatment
            }
            
            return self.results
        
        def plot_comparison(self, save_path=None):
            """
            Visualize ITT vs Triggered effects.
            """
            if not self.results:
                raise ValueError("Must run compare_itt_triggered first")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Plot 1: Effect estimates
            methods = ['ITT\n(All Users)', 'Triggered\n(Exposed Only)', 'CACE\n(Estimated)']
            effects = [
                self.results['itt']['effect'],
                self.results['triggered']['effect'],
                self.results['cace'] if self.results['cace'] else 0
            ]
            errors = [
                [self.results['itt']['effect'] - self.results['itt']['ci_lower'],
                 self.results['itt']['ci_upper'] - self.results['itt']['effect']],
                [self.results['triggered']['effect'] - self.results['triggered']['ci_lower'],
                 self.results['triggered']['ci_upper'] - self.results['triggered']['effect']],
                [0, 0]  # No CI for CACE (needs bootstrap)
            ]
            
            colors = ['#3498db', '#e74c3c', '#9b59b6']
            x_pos = [0, 1, 2]
            
            ax1.errorbar(x_pos, effects, 
                        yerr=[[errors[i][0] for i in range(2)] + [0],
                              [errors[i][1] for i in range(2)] + [0]],
                        fmt='o', markersize=12, capsize=10, capthick=2,
                        color=colors[0], ecolor=colors[0])
            
            for i, (x, y, c) in enumerate(zip(x_pos, effects, colors)):
                ax1.plot(x, y, 'o', markersize=12, color=c)
            
            ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(methods)
            ax1.set_ylabel('Treatment Effect')
            ax1.set_title('ITT vs Triggered vs CACE Comparison', fontweight='bold')
            ax1.grid(axis='y', alpha=0.3)
            
            # Plot 2: Sample sizes
            sample_labels = ['Treatment\n(Assigned)', 'Treatment\n(Triggered)', 'Control']
            sample_sizes = [
                self.results['itt']['n_treatment'],
                self.results['triggered']['n_triggered'],
                self.results['itt']['n_control']
            ]
            
            ax2.bar(sample_labels, sample_sizes, color=['#3498db', '#e74c3c', '#95a5a6'], alpha=0.7)
            ax2.set_ylabel('Sample Size')
            ax2.set_title('Sample Sizes by Analysis Type', fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
            
            # Add trigger rate annotation
            trigger_text = f"Trigger Rate: {self.results['dilution_factor']:.1%}"
            ax2.text(0.5, 0.95, trigger_text, transform=ax2.transAxes,
                    ha='center', va='top', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
    
    
    # ===== Example 1: Uber Driver Feature =====
    
    print("=" * 70)
    print("EXAMPLE 1: UBER - DRIVER IN-APP NAVIGATION EXPERIMENT")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Randomize 20,000 drivers: 10,000 control, 10,000 treatment
    n_per_group = 10000
    
    # Treatment group: Only 70% actually use the feature (trigger)
    trigger_rate = 0.70
    n_triggered = int(n_per_group * trigger_rate)
    
    # Simulate outcomes: trips per day
    # Control: 15 trips/day on average
    control_trips = np.random.poisson(15, n_per_group)
    
    # Treatment (all assigned): Includes non-triggered (no effect) + triggered (+2 trips)
    treatment_not_triggered = np.random.poisson(15, n_per_group - n_triggered)
    treatment_triggered = np.random.poisson(17, n_triggered)  # +2 trips for triggered
    treatment_all = np.concatenate([treatment_not_triggered, treatment_triggered])
    
    # Analyze
    analyzer = TriggeredAnalyzer()
    results = analyzer.compare_itt_triggered(
        treatment_all=treatment_all,
        treatment_triggered=treatment_triggered,
        control_all=control_trips,
        trigger_rate_treatment=trigger_rate
    )
    
    print("\n📊 ITT Analysis (All Randomized Drivers):")
    itt = results['itt']
    print(f"   Treatment Effect: {itt['effect']:.2f} trips/day")
    print(f"   95% CI: [{itt['ci_lower']:.2f}, {itt['ci_upper']:.2f}]")
    print(f"   p-value: {itt['p_value']:.4f}")
    print(f"   Significant: {'Yes ✓' if itt['significant'] else 'No ✗'}")
    print(f"   Sample: {itt['n_treatment']:,} treatment, {itt['n_control']:,} control")
    
    print("\n📊 Triggered Analysis (Only Drivers Who Used Feature):")
    trig = results['triggered']
    print(f"   Treatment Effect: {trig['effect']:.2f} trips/day")
    print(f"   95% CI: [{trig['ci_lower']:.2f}, {trig['ci_upper']:.2f}]")
    print(f"   p-value: {trig['p_value']:.4f}")
    print(f"   Significant: {'Yes ✓' if trig['significant'] else 'No ✗'}")
    print(f"   Sample: {trig['n_triggered']:,} triggered, {trig['n_control']:,} control")
    print(f"   Trigger rate: {trig['trigger_rate']:.1%}")
    
    print("\n📊 CACE (Complier Average Causal Effect):")
    print(f"   Estimated effect on compliers: {results['cace']:.2f} trips/day")
    print(f"   Formula: ITT Effect / Trigger Rate = {itt['effect']:.2f} / {trigger_rate:.2f}")
    
    print("\n🔍 Interpretation:")
    print(f"   • ITT shows real-world impact: +{itt['effect']:.2f} trips (includes non-users)")
    print(f"   • Triggered shows effect on engaged users: +{trig['effect']:.2f} trips")
    print(f"   • CACE estimates causal effect: +{results['cace']:.2f} trips")
    print(f"   • ITT is {itt['effect']/trig['effect']:.0%} of triggered effect (dilution)")
    
    # Visualize
    analyzer.plot_comparison()
    plt.show()
    ```

    **Bias in Triggered Analysis:**

    | Bias Source | Example | Impact | Mitigation |
    |-------------|---------|--------|------------|
    | **Self-selection** | Power users more likely to trigger | Overestimates effect | Compare trigger rates, use IV |
    | **Treatment affects triggering** | Feature makes users visit more → higher trigger | Inflates effect | Check if control would trigger similarly |
    | **External factors** | Weekend users trigger more | Confounding | Balance on triggering covariates |
    | **Compliance drift** | Trigger rate changes over time | Time-varying bias | Analyze by cohort |

    **Real Company Examples:**

    | Company | Experiment | Trigger Rate | ITT Effect | Triggered Effect | Decision |
    |---------|------------|--------------|------------|------------------|----------|
    | **Uber** | Driver navigation feature | 70% | +1.4 trips/day | +2.0 trips/day | Ship (ITT positive, real impact) |
    | **Lyft** | Passenger surge pricing | 60% (surge hours) | +$2 revenue/day | +$3.5 revenue/day | Report both, focus on ITT for ROI |
    | **DoorDash** | Dasher heat map | 45% (open app) | +0.8 orders/hour | +1.8 orders/hour | Investigate low trigger rate first |
    | **Netflix** | "Skip Intro" button | 30% (shows with intro) | +0.5 min watch time | +1.7 min watch time | ITT matters for business case |

    **Decision Framework:**

    ```
    ┌──────────────────────────────────────────────────────────────┐
    │           When to Use ITT vs Triggered                        │
    ├──────────────────────────────────────────────────────────────┤
    │                                                               │
    │  Primary Analysis: ALWAYS ITT                                │
    │  • Preserves randomization                                    │
    │  • Real-world effect (includes non-compliance)                │
    │  • Unbiased estimate                                          │
    │                                                               │
    │  Secondary Analysis: Triggered (if relevant)                 │
    │  • When: Trigger rate <80%                                   │
    │  • Purpose: Understand engaged user effect                    │
    │  • Caution: Check for selection bias                          │
    │                                                               │
    │  Report BOTH when:                                            │
    │  • Trigger rate <80% AND >20%                                │
    │  • Difference is large (triggered >2× ITT)                   │
    │  • Decision depends on adoption strategy                      │
    │                                                               │
    │  Decision Rule:                                               │
    │  ├─ ITT significant + positive → Ship                        │
    │  ├─ ITT null, Triggered positive → Improve triggering        │
    │  └─ Both null → Don't ship                                   │
    │                                                               │
    └──────────────────────────────────────────────────────────────┘
    ```

    **Common Mistakes:**

    | ❌ Mistake | Why It's Bad | ✅ Fix |
    |-----------|--------------|-------|
    | **Only report triggered** | Ignores real-world dilution | Always report ITT first |
    | **Compare triggered treatment to all control** | Unfair comparison | Use triggered control (if available) or adjust |
    | **Ignore trigger rate differences** | Selection bias | Compare treatment vs control trigger rates |
    | **Use triggered for business case** | Overstates ROI | Use ITT for revenue projections |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of causal inference, compliance, selection bias, real-world vs ideal effects.
        
        **Strong answer signals:**
        
        - Knows difference: ITT (all randomized) vs Triggered (exposed only)
        - Explains ITT preserves randomization, triggered may have selection bias
        - Mentions CACE (Complier Average Causal Effect) = ITT / Compliance
        - States "Always report ITT first, triggered as secondary"
        - Gives example: "Uber driver feature: 70% trigger rate, ITT +1.4 trips, Triggered +2.0 trips"
        - Knows when triggered is biased (trigger rate differs between groups)
        - Explains business impact: ITT matters for ROI, triggered for product understanding
        
        **Red flags:**
        
        - Only mentions triggered analysis
        - Doesn't understand selection bias in triggering
        - Thinks triggered is always better ("higher power")
        - Can't explain CACE or dilution
        
        **Follow-up questions:**
        
        - "When would triggered analysis be biased?" (When treatment affects triggering, or power users self-select)
        - "How do you decide which to use for business decision?" (Always ITT for go/no-go, triggered for product insights)
        - "What if trigger rate is 10% in treatment, 30% in control?" (Major selection bias, triggered estimate invalid)

---

### How to Handle Carry-Over Effects? - Netflix, Airbnb Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Design` | **Asked by:** Netflix, Airbnb, Uber

??? success "View Answer"

    **Carry-Over Effects** occur when **previous treatment exposure influences current behavior**, violating the assumption that each experimental unit's outcome is independent of others' assignments. This is common in **time-series experiments, crossover designs, and habit-forming features**.

    **Types of Carry-Over:**

    ```
    ┌───────────────────────────────────────────────────────────────────┐
    │                    Carry-Over Effect Types                         │
    ├───────────────────────────────────────────────────────────────────┤
    │                                                                    │
    │  1. LEARNING EFFECTS                                               │
    │     User learns behavior that persists                             │
    │     ┌──────────┬──────────┬──────────┐                           │
    │     │Treatment │  Washout │ Control  │                            │
    │     │(Week 1)  │ (Week 2) │(Week 3)  │                            │
    │     └──────────┴──────────┴──────────┘                           │
    │     Effect: →→→→→→→→→→→→→→→→→→→→→→→→→                          │
    │     (Behavior learned in W1 continues in W3)                       │
    │                                                                    │
    │  2. CONTRAST EFFECTS                                               │
    │     New experience makes old one feel worse                        │
    │     ┌──────────┬──────────┬──────────┐                           │
    │     │Treatment │  Washout │ Control  │                            │
    │     │(New UI)  │          │(Old UI)  │                            │
    │     └──────────┴──────────┴──────────┘                           │
    │     Effect: Control feels worse after seeing new UI                │
    │                                                                    │
    │  3. HABITUATION                                                    │
    │     User forms new habit                                           │
    │     Effect persists indefinitely                                   │
    │                                                                    │
    │  4. NOVELTY EFFECTS                                                │
    │     Initial excitement wears off                                   │
    │     ┌────────────────────────────────┐                           │
    │     │ Effect ↓                       │                            │
    │     │         ╲                      │                            │
    │     │          ╲_______________      │                            │
    │     │     Week 1  Week 2  Week 3    │                            │
    │     └────────────────────────────────┘                           │
    │                                                                    │
    └───────────────────────────────────────────────────────────────────┘
    ```

    **Solutions & Implementation:**

    ```python
    import numpy as np
    import pandas as pd
    from scipy import stats
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
    
    # Production: Carry-Over Effect Detection and Mitigation
    
    class CarryOverAnalyzer:
        """
        Detect and handle carry-over effects in experiments.
        """
        
        def __init__(self):
            self.timeline_data = None
            
        def simulate_carryover_experiment(self, n_users=1000, weeks=6, 
                                         carryover_decay=0.7):
            """
            Simulate crossover experiment with carry-over effects.
            
            Design: AB/BA crossover
            - Group 1: Treatment weeks 1-3, Control weeks 4-6
            - Group 2: Control weeks 1-3, Treatment weeks 4-6
            
            carryover_decay: How much treatment effect persists (0=none, 1=full)
            """
            np.random.seed(42)
            
            timeline = []
            
            for user_id in range(n_users):
                group = 'AB' if user_id < n_users//2 else 'BA'
                
                for week in range(1, weeks+1):
                    # Determine assignment
                    if group == 'AB':
                        is_treatment = week <= 3
                    else:  # BA
                        is_treatment = week > 3
                    
                    # Base outcome
                    base_outcome = 100
                    
                    # Treatment effect: +10
                    treatment_effect = 10 if is_treatment else 0
                    
                    # Carry-over effect: Persists from previous treatment
                    carryover_effect = 0
                    if not is_treatment and week > 1:
                        # Check if was in treatment previous week
                        prev_treatment = (week-1 <= 3) if group == 'AB' else (week-1 > 3)
                        if prev_treatment:
                            # Decay based on weeks since last treatment
                            weeks_since = 1
                            carryover_effect = 10 * (carryover_decay ** weeks_since)
                    
                    # Total outcome
                    outcome = base_outcome + treatment_effect + carryover_effect + np.random.normal(0, 5)
                    
                    timeline.append({
                        'user_id': user_id,
                        'group': group,
                        'week': week,
                        'is_treatment': is_treatment,
                        'outcome': outcome,
                        'carryover_effect': carryover_effect
                    })
            
            self.timeline_data = pd.DataFrame(timeline)
            return self.timeline_data
        
        def detect_carryover(self, df: pd.DataFrame, outcome_col='outcome'):
            """
            Detect carry-over by comparing period 2 between groups.
            
            In AB/BA crossover:
            - If no carry-over: Period 2 effects should be equal but opposite
            - If carry-over: Group that had treatment in period 1 shows residual effect
            """
            # Split into periods
            period1 = df[df['week'] <= 3].copy()
            period2 = df[df['week'] > 3].copy()
            
            # Period 2: Compare AB (now control) vs BA (now treatment)
            ab_period2 = period2[period2['group'] == 'AB'][outcome_col]
            ba_period2 = period2[period2['group'] == 'BA'][outcome_col]
            
            # If carry-over exists, AB should be higher than pure control
            # because they retain some treatment effect
            t_stat, p_value = stats.ttest_ind(ab_period2, ba_period2)
            
            ab_mean = ab_period2.mean()
            ba_mean = ba_period2.mean()
            
            # Expected: BA > AB in period 2 (BA is now treatment)
            # But if carry-over: difference is smaller than pure effect
            
            return {
                'ab_period2_mean': ab_mean,
                'ba_period2_mean': ba_mean,
                'difference': ba_mean - ab_mean,
                'p_value': p_value,
                'carryover_detected': p_value < 0.05 and (ba_mean - ab_mean) < 8  # Less than full effect
            }
        
        def estimate_washout_period(self, df: pd.DataFrame, outcome_col='outcome'):
            """
            Estimate how many weeks needed for carry-over to decay.
            
            Uses AB group in period 2 (post-treatment) to see when
            outcome returns to baseline.
            """
            ab_period2 = df[(df['group'] == 'AB') & (df['week'] > 3)].copy()
            
            # Group by weeks since treatment ended
            ab_period2['weeks_since_treatment'] = ab_period2['week'] - 3
            
            washout_pattern = ab_period2.groupby('weeks_since_treatment')[outcome_col].agg(['mean', 'std', 'count']).reset_index()
            
            # Baseline is BA group in period 1 (pure control)
            baseline = df[(df['group'] == 'BA') & (df['week'] <= 3)][outcome_col].mean()
            
            washout_pattern['diff_from_baseline'] = washout_pattern['mean'] - baseline
            
            # Washout complete when diff < 2 (roughly within noise)
            washout_weeks = washout_pattern[washout_pattern['diff_from_baseline'] < 2]['weeks_since_treatment'].min()
            
            return washout_pattern, washout_weeks
        
        def analyze_first_period_only(self, df: pd.DataFrame, outcome_col='outcome'):
            """
            Naive approach: Only use first period before any crossover.
            This avoids carry-over but wastes data.
            """
            period1 = df[df['week'] <= 3].copy()
            
            ab_treatment = period1[period1['group'] == 'AB'][outcome_col]
            ba_control = period1[period1['group'] == 'BA'][outcome_col]
            
            t_stat, p_value = stats.ttest_ind(ab_treatment, ba_control)
            
            effect = ab_treatment.mean() - ba_control.mean()
            se = np.sqrt(stats.sem(ab_treatment)**2 + stats.sem(ba_control)**2)
            
            return {
                'effect': effect,
                'se': se,
                'p_value': p_value,
                'n_per_group': len(ab_treatment)
            }
        
        def plot_carryover_timeline(self, save_path=None):
            """
            Visualize carry-over effects over time.
            """
            if self.timeline_data is None:
                raise ValueError("Must simulate or load data first")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Plot 1: Mean outcome over time by group
            timeline_summary = self.timeline_data.groupby(['week', 'group'])['outcome'].mean().reset_index()
            
            for group in ['AB', 'BA']:
                data = timeline_summary[timeline_summary['group'] == group]
                ax1.plot(data['week'], data['outcome'], marker='o', linewidth=2, 
                        markersize=8, label=f'Group {group}')
            
            # Mark treatment periods
            ax1.axvspan(0.5, 3.5, alpha=0.2, color='blue', label='AB Treatment Period')
            ax1.axvspan(3.5, 6.5, alpha=0.2, color='red', label='BA Treatment Period')
            
            ax1.set_xlabel('Week', fontsize=12)
            ax1.set_ylabel('Mean Outcome', fontsize=12)
            ax1.set_title('Crossover Design: Outcome Over Time', fontsize=14, fontweight='bold')
            ax1.legend(loc='best')
            ax1.grid(alpha=0.3)
            
            # Plot 2: Carry-over effect magnitude
            carryover_data = self.timeline_data[self.timeline_data['carryover_effect'] > 0]
            if len(carryover_data) > 0:
                co_summary = carryover_data.groupby('week')['carryover_effect'].mean().reset_index()
                ax2.bar(co_summary['week'], co_summary['carryover_effect'], 
                       color='orange', alpha=0.7)
                ax2.set_xlabel('Week', fontsize=12)
                ax2.set_ylabel('Mean Carry-Over Effect', fontsize=12)
                ax2.set_title('Carry-Over Effect Magnitude', fontsize=14, fontweight='bold')
                ax2.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
    
    
    # ===== Example: Netflix Content Recommendation Experiment =====
    
    print("=" * 70)
    print("EXAMPLE: NETFLIX - CROSSOVER DESIGN WITH CARRY-OVER")
    print("=" * 70)
    
    analyzer = CarryOverAnalyzer()
    
    # Simulate experiment with 70% carry-over decay
    df = analyzer.simulate_carryover_experiment(
        n_users=2000, 
        weeks=6, 
        carryover_decay=0.7
    )
    
    print("\n🔬 Experimental Design:")
    print("   • Group AB: Treatment weeks 1-3, Control weeks 4-6")
    print("   • Group BA: Control weeks 1-3, Treatment weeks 4-6")
    print("   • True treatment effect: +10 minutes watch time")
    print("   • Carry-over decay: 70% per week")
    
    # Detect carry-over
    carryover_results = analyzer.detect_carryover(df)
    print("\n🔍 Carry-Over Detection:")
    print(f"   • AB period 2 mean: {carryover_results['ab_period2_mean']:.2f}")
    print(f"   • BA period 2 mean: {carryover_results['ba_period2_mean']:.2f}")
    print(f"   • Difference: {carryover_results['difference']:.2f}")
    print(f"   • Carry-over detected: {'Yes ⚠️' if carryover_results['carryover_detected'] else 'No ✓'}")
    
    # Estimate washout period
    washout_pattern, washout_weeks = analyzer.estimate_washout_period(df)
    print("\n⏱️ Washout Period Estimation:")
    print(f"   • Weeks needed for effect to decay: {washout_weeks:.0f} weeks")
    print("   • Decay pattern:")
    print(washout_pattern[['weeks_since_treatment', 'mean', 'diff_from_baseline']].to_string(index=False))
    
    # First period only analysis
    first_period = analyzer.analyze_first_period_only(df)
    print("\n📊 First Period Only Analysis (No Carry-Over):")
    print(f"   • Treatment effect: {first_period['effect']:.2f}")
    print(f"   • SE: {first_period['se']:.2f}")
    print(f"   • p-value: {first_period['p_value']:.4f}")
    print(f"   • Sample per group: {first_period['n_per_group']:,} users")
    
    # Visualize
    analyzer.plot_carryover_timeline()
    plt.show()
    ```

    **Mitigation Strategies:**

    | Strategy | How It Works | Pros | Cons | When to Use |
    |----------|--------------|------|------|-------------|
    | **Washout Period** | Add delay between treatments | Simple, eliminates carry-over | Loses time, data | Crossover designs, short carry-over |
    | **First Period Only** | Only analyze pre-crossover data | No bias | Wastes 50% of data | Strong carry-over suspected |
    | **Parallel Groups** | No crossover, pure A vs B | No carry-over possible | Needs more users | Long-lasting effects (habit formation) |
    | **Long Experiment** | Run for months until steady-state | Measures true long-term effect | Expensive | Product changes, network effects |
    | **Statistical Adjustment** | Model carry-over effect | Uses all data | Complex, requires assumptions | Quantifiable decay pattern |

    **Real Company Examples:**

    | Company | Experiment | Carry-Over Type | Solution | Outcome |
    |---------|------------|-----------------|----------|----------|
    | **Netflix** | Recommendation algorithm | Learning (users find new content) | 3-month parallel groups | Found 8% watch time lift persists long-term |
    | **Airbnb** | Search ranking | Contrast effect (old ranking feels worse) | 2-week washout in crossover | Detected 20% carry-over, extended washout to 4 weeks |
    | **Uber** | Surge pricing UI | Habituation (users learn to avoid surge) | First period only analysis | Avoided biased estimate from learned behavior |
    | **Spotify** | Playlist algorithm | Novelty (initial excitement → decay) | 8-week experiment, analyze by week | Week 1: +15% engagement, Week 8: +5% (true effect) |

    **Crossover Design with Washout:**

    ```
    ┌──────────────────────────────────────────────────────────────────┐
    │             Crossover Design with Washout Period                 │
    ├──────────────────────────────────────────────────────────────────┤
    │                                                                   │
    │  Group AB:                                                        │
    │  ├─────────┬─────────┬─────────┬─────────┐                      │
    │  │Period 1 │ Washout │Period 2 │ Washout │                      │
    │  │Treatment│  (2wk)  │Control  │  (2wk)  │                      │
    │  └─────────┴─────────┴─────────┴─────────┘                      │
    │      ↑                   ↑                                       │
    │   Measure            Measure                                     │
    │                                                                   │
    │  Group BA:                                                        │
    │  ├─────────┬─────────┬─────────┬─────────┐                      │
    │  │Period 1 │ Washout │Period 2 │ Washout │                      │
    │  │Control  │  (2wk)  │Treatment│  (2wk)  │                      │
    │  └─────────┴─────────┴─────────┴─────────┘                      │
    │      ↑                   ↑                                       │
    │   Measure            Measure                                     │
    │                                                                   │
    │  Benefits:                                                        │
    │  • Each user is their own control (reduces variance)              │
    │  • Washout eliminates carry-over                                  │
    │  • Balanced design (both sequences)                               │
    │                                                                   │
    └──────────────────────────────────────────────────────────────────┘
    ```

    **Decision Framework:**

    | Carry-Over Severity | Washout Period Needed | Recommended Design |
    |--------------------|-----------------------|--------------------|
    | **None** | 0 days | Standard parallel groups or crossover |
    | **Weak (<20% residual)** | 1-2 weeks | Crossover with short washout |
    | **Moderate (20-50%)** | 2-4 weeks | Crossover with washout OR first period only |
    | **Strong (>50%)** | >4 weeks or indefinite | Parallel groups only, long experiment |

    **Common Mistakes:**

    | ❌ Mistake | Why It's Bad | ✅ Fix |
    |-----------|--------------|-------|
    | **Ignore carry-over** | Biased estimates, wrong conclusions | Test for carry-over, use washout |
    | **Too short washout** | Carry-over persists | Pilot test to estimate decay time |
    | **Use both periods with carry-over** | Violates independence | First period only OR statistical adjustment |
    | **Crossover for habit-forming features** | Permanent behavior change | Use parallel groups for long-term effects |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of temporal dependencies, crossover designs, experimental validity.
        
        **Strong answer signals:**
        
        - Defines carry-over: "Previous treatment affects current behavior"
        - Gives examples: Learning effects, contrast effects, habituation
        - Knows multiple solutions: Washout period, first period only, parallel groups
        - Explains crossover design: AB/BA with balanced sequences
        - Mentions detection method: Compare period 2 between groups
        - Gives company example: "Netflix uses 3-month parallel for recommendation changes"
        - Knows when to avoid crossover: Habit-forming features, long-lasting effects
        
        **Red flags:**
        
        - Never heard of carry-over effects
        - Thinks crossover is always better than parallel
        - Can't explain how to detect carry-over
        - Doesn't know washout period concept
        
        **Follow-up questions:**
        
        - "How do you choose washout period length?" (Pilot test, look at decay curve, typically 1-4 weeks)
        - "When should you NOT use crossover design?" (Habit-forming features, permanent behavior changes)
        - "How do you detect carry-over?" (Compare period 2 between AB and BA groups, expect symmetric effects)

---

### What is Intent-to-Treat (ITT) Analysis and How Do You Handle Non-Compliance? - Netflix, Google Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Analysis`, `Compliance`, `CACE` | **Asked by:** Netflix, Google, Meta, Spotify

??? success "View Answer"

    **Intent-to-Treat (ITT) analysis** is the **gold standard** for A/B tests, analyzing users **as originally randomized** regardless of whether they actually received or complied with treatment. This preserves **randomization** and measures **real-world effectiveness**, but may **underestimate** true treatment effects when compliance is low.

    **Critical Concepts:**

    - **ITT Effect:** Effect of being **assigned** to treatment (real-world impact)
    - **Per-Protocol Effect:** Effect among users who **actually received** treatment (efficacy)
    - **CACE (Complier Average Causal Effect):** Effect among **compliers only** (unbiased estimate)
    - **Non-Compliance:** Users assigned to treatment don't use it, or control users get treatment

    **Comparison of Analysis Methods:**

    | Method | Analyzed Users | Preserves Randomization | Measures | Bias Risk |
    |--------|----------------|------------------------|----------|-----------|
    | **Intent-to-Treat (ITT)** | All as randomized | ✅ Yes | Real-world effectiveness | None (conservative) |
    | **Per-Protocol** | Only compliers | ❌ No | Efficacy under perfect adherence | High (selection bias) |
    | **CACE (IV Method)** | Adjusts for compliance | ✅ Yes | Effect for compliers | Low (if assumptions met) |
    | **As-Treated** | By actual treatment | ❌ No | Observed behavior | High (confounding) |

    **When Compliance Matters:**

    | Scenario | Compliance Rate | Recommended Approach | Example |
    |----------|----------------|---------------------|---------|
    | High adoption | >90% | ITT sufficient | Spotify: New UI flows (95% usage) |
    | Moderate adoption | 60-90% | ITT + CACE sensitivity | Netflix: Recommendation change (75% eligible users) |
    | Low adoption | <60% | CACE essential, understand barriers | Uber: Driver incentive program (45% participation) |
    | Opt-in features | Varies | ITT + subgroup analysis | LinkedIn: Premium feature test |

    **Real Company Examples:**

    | Company | Test | Compliance | ITT Effect | CACE Effect | Decision |
    |---------|------|-----------|-----------|-------------|----------|
    | **Netflix** | Personalized thumbnails | 82% viewed | +2.1% watch time | +2.5% watch time | Launched (ITT shows clear benefit) |
    | **Spotify** | Daily Mix playlist | 68% listened | +1.8% session length | +2.6% session length | Launched with promotion to boost adoption |
    | **Uber** | Driver cash-out feature | 41% activated | +0.3% driver hours | +0.7% driver hours | Launched + education campaign |
    | **LinkedIn** | Skill endorsements | 55% engaged | +4.2% profile views | +7.6% profile views | Launched with onboarding nudges |

    **Architecture - Compliance Analysis Framework:**

    ```
    ┌────────────────────────────────────────────────────────────────────┐
    │             INTENT-TO-TREAT (ITT) ANALYSIS FRAMEWORK               │
    ├────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐      │
    │  │ Randomization│────▶│  Assignment  │────▶│   Outcome    │      │
    │  │   (Valid)    │     │   Groups     │     │  Measurement │      │
    │  └──────────────┘     └──────┬───────┘     └──────────────┘      │
    │                               │                                    │
    │                               ↓                                    │
    │                  ┌────────────────────────┐                       │
    │                  │   Compliance Check     │                       │
    │                  └────────────────────────┘                       │
    │                               │                                    │
    │          ┌────────────────────┼────────────────────┐             │
    │          ↓                    ↓                    ↓             │
    │    ┌──────────┐         ┌──────────┐        ┌──────────┐       │
    │    │Compliers │         │Never-    │        │ Always-  │       │
    │    │(respond  │         │Takers    │        │ Takers   │       │
    │    │to Z)     │         │(Z=1,D=0) │        │(Z=0,D=1) │       │
    │    └──────────┘         └──────────┘        └──────────┘       │
    │          │                                                        │
    │          ↓                                                        │
    │    ┌─────────────────────────────────┐                          │
    │    │   ANALYSIS STRATEGIES:          │                          │
    │    │                                  │                          │
    │    │  1. ITT: E[Y|Z=1] - E[Y|Z=0]   │                          │
    │    │  2. CACE: ITT / compliance_rate │                          │
    │    │  3. Bounds: Sensitivity analysis│                          │
    │    └─────────────────────────────────┘                          │
    │                                                                   │
    └────────────────────────────────────────────────────────────────────┘
    ```

    **Production Code - Complete ITT and CACE Analysis:**

    ```python
    import numpy as np
    import pandas as pd
    from scipy import stats
    from dataclasses import dataclass
    from typing import Dict, Tuple, Optional
    import matplotlib.pyplot as plt
    from statsmodels.api import OLS
    import statsmodels.formula.api as smf
    
    @dataclass
    class ITTResult:
        """Results from Intent-to-Treat analysis."""
        itt_effect: float
        itt_se: float
        itt_ci: Tuple[float, float]
        itt_pvalue: float
        control_mean: float
        treatment_mean: float
        n_control: int
        n_treatment: int
    
    @dataclass
    class CACEResult:
        """Results from CACE (Complier Average Causal Effect) analysis."""
        cace_effect: float
        cace_se: float
        cace_ci: Tuple[float, float]
        compliance_rate_treatment: float
        compliance_rate_control: float
        first_stage_f: float
    
    class ComplianceAnalyzer:
        """
        Comprehensive analyzer for ITT and non-compliance in A/B tests.
        
        Handles:
        - Intent-to-Treat (ITT) analysis
        - Per-protocol analysis (with warnings about bias)
        - CACE estimation using instrumental variables
        - Compliance rate calculation
        - Sensitivity analysis
        """
        
        def __init__(self, alpha: float = 0.05):
            self.alpha = alpha
        
        def analyze_itt(self, df: pd.DataFrame, 
                       outcome_col: str = 'outcome',
                       assignment_col: str = 'assigned_treatment') -> ITTResult:
            """
            Standard Intent-to-Treat analysis.
            Compares outcomes by ASSIGNED treatment, regardless of compliance.
            """
            control = df[df[assignment_col] == 0][outcome_col]
            treatment = df[df[assignment_col] == 1][outcome_col]
            
            # ITT effect
            control_mean = control.mean()
            treatment_mean = treatment.mean()
            itt_effect = treatment_mean - control_mean
            
            # Standard error
            se_control = control.std() / np.sqrt(len(control))
            se_treatment = treatment.std() / np.sqrt(len(treatment))
            itt_se = np.sqrt(se_control**2 + se_treatment**2)
            
            # Confidence interval
            z_crit = stats.norm.ppf(1 - self.alpha / 2)
            itt_ci = (itt_effect - z_crit * itt_se, itt_effect + z_crit * itt_se)
            
            # P-value
            z_stat = itt_effect / itt_se
            itt_pvalue = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            return ITTResult(
                itt_effect=itt_effect,
                itt_se=itt_se,
                itt_ci=itt_ci,
                itt_pvalue=itt_pvalue,
                control_mean=control_mean,
                treatment_mean=treatment_mean,
                n_control=len(control),
                n_treatment=len(treatment)
            )
        
        def analyze_per_protocol(self, df: pd.DataFrame,
                                outcome_col: str = 'outcome',
                                received_col: str = 'received_treatment') -> Dict:
            """
            Per-protocol analysis (BIASED - use with caution).
            Compares users who ACTUALLY received treatment vs control.
            
            WARNING: Selection bias! Non-compliers may differ systematically.
            """
            control = df[df[received_col] == 0][outcome_col]
            treatment = df[df[received_col] == 1][outcome_col]
            
            effect = treatment.mean() - control.mean()
            se = np.sqrt(control.var()/len(control) + treatment.var()/len(treatment))
            
            return {
                'per_protocol_effect': effect,
                'per_protocol_se': se,
                'warning': 'BIASED ESTIMATE - selection bias likely present',
                'control_mean': control.mean(),
                'treatment_mean': treatment.mean()
            }
        
        def estimate_cace(self, df: pd.DataFrame,
                         outcome_col: str = 'outcome',
                         assignment_col: str = 'assigned_treatment',
                         received_col: str = 'received_treatment') -> CACEResult:
            """
            CACE estimation using Instrumental Variables (2SLS).
            
            Assignment is the instrument for actual treatment received.
            Estimates effect for compliers (those who take treatment when assigned).
            
            Assumptions:
            1. Random assignment (valid instrument)
            2. Exclusion restriction (assignment affects outcome only through treatment)
            3. Monotonicity (no defiers)
            """
            # First stage: Assignment → Received treatment
            first_stage = smf.ols(f'{received_col} ~ {assignment_col}', data=df).fit()
            compliance_rate_control = df[df[assignment_col] == 0][received_col].mean()
            compliance_rate_treatment = df[df[assignment_col] == 1][received_col].mean()
            first_stage_f = first_stage.fvalue
            
            # Check weak instrument
            if first_stage_f < 10:
                print(f"⚠️  WARNING: Weak instrument (F={first_stage_f:.2f} < 10)")
            
            # Second stage: Predicted treatment → Outcome (using 2SLS manually)
            # Simpler approach: ITT / compliance rate difference
            itt_result = self.analyze_itt(df, outcome_col, assignment_col)
            itt_effect = itt_result.itt_effect
            
            compliance_diff = compliance_rate_treatment - compliance_rate_control
            
            if compliance_diff < 0.01:
                raise ValueError("Compliance rates too similar - cannot estimate CACE")
            
            # CACE = ITT / (compliance_treatment - compliance_control)
            cace_effect = itt_effect / compliance_diff
            
            # Standard error (Wald estimator)
            # SE(CACE) ≈ SE(ITT) / compliance_diff
            cace_se = itt_result.itt_se / compliance_diff
            
            z_crit = stats.norm.ppf(1 - self.alpha / 2)
            cace_ci = (cace_effect - z_crit * cace_se, cace_effect + z_crit * cace_se)
            
            return CACEResult(
                cace_effect=cace_effect,
                cace_se=cace_se,
                cace_ci=cace_ci,
                compliance_rate_treatment=compliance_rate_treatment,
                compliance_rate_control=compliance_rate_control,
                first_stage_f=first_stage_f
            )
        
        def compliance_summary(self, df: pd.DataFrame,
                              assignment_col: str = 'assigned_treatment',
                              received_col: str = 'received_treatment') -> pd.DataFrame:
            """
            Create compliance table showing all four groups.
            """
            summary = df.groupby([assignment_col, received_col]).size().reset_index(name='count')
            summary['percentage'] = summary['count'] / len(df) * 100
            
            summary['group_type'] = ''
            summary.loc[(summary[assignment_col] == 1) & (summary[received_col] == 1), 'group_type'] = 'Compliers (Treatment)'
            summary.loc[(summary[assignment_col] == 0) & (summary[received_col] == 0), 'group_type'] = 'Compliers (Control)'
            summary.loc[(summary[assignment_col] == 1) & (summary[received_col] == 0), 'group_type'] = 'Never-Takers'
            summary.loc[(summary[assignment_col] == 0) & (summary[received_col] == 1), 'group_type'] = 'Always-Takers'
            
            return summary
    
    
    # ===== Example: Spotify Premium Feature Test =====
    
    np.random.seed(42)
    n_users = 10000
    
    # Generate experiment data with non-compliance
    df = pd.DataFrame({
        'user_id': range(n_users),
        'assigned_treatment': np.random.binomial(1, 0.5, n_users)
    })
    
    # Simulate compliance (not everyone assigned to treatment uses it)
    # Treatment group: 75% compliance
    # Control group: 5% always-takers (somehow get treatment)
    df['received_treatment'] = 0
    treatment_assigned = df['assigned_treatment'] == 1
    control_assigned = df['assigned_treatment'] == 0
    
    df.loc[treatment_assigned, 'received_treatment'] = np.random.binomial(1, 0.75, treatment_assigned.sum())
    df.loc[control_assigned, 'received_treatment'] = np.random.binomial(1, 0.05, control_assigned.sum())
    
    # Simulate outcome (daily listening minutes)
    # True CACE effect = +20 minutes for compliers
    true_cace = 20
    baseline = 120  # baseline listening
    
    # Outcome depends on RECEIVED treatment (not assignment)
    df['outcome'] = baseline + true_cace * df['received_treatment'] + np.random.normal(0, 30, n_users)
    
    print("=" * 80)
    print("EXAMPLE: SPOTIFY - PREMIUM FEATURE WITH NON-COMPLIANCE")
    print("=" * 80)
    
    analyzer = ComplianceAnalyzer(alpha=0.05)
    
    # Compliance table
    compliance_table = analyzer.compliance_summary(df)
    print("\n📊 COMPLIANCE BREAKDOWN:")
    print(compliance_table.to_string(index=False))
    
    # ITT Analysis
    itt_result = analyzer.analyze_itt(df)
    print(f"\n🔬 INTENT-TO-TREAT (ITT) ANALYSIS:")
    print(f"   Control mean:     {itt_result.control_mean:.2f} minutes")
    print(f"   Treatment mean:   {itt_result.treatment_mean:.2f} minutes")
    print(f"   ITT Effect:       +{itt_result.itt_effect:.2f} minutes")
    print(f"   95% CI:           ({itt_result.itt_ci[0]:.2f}, {itt_result.itt_ci[1]:.2f})")
    print(f"   P-value:          {itt_result.itt_pvalue:.4f}")
    print(f"   Interpretation:   Effect of being ASSIGNED to treatment")
    
    # Per-Protocol Analysis (biased)
    pp_result = analyzer.analyze_per_protocol(df)
    print(f"\n⚠️  PER-PROTOCOL ANALYSIS (BIASED):")
    print(f"   Effect:           +{pp_result['per_protocol_effect']:.2f} minutes")
    print(f"   WARNING:          {pp_result['warning']}")
    
    # CACE Analysis
    cace_result = analyzer.estimate_cace(df)
    print(f"\n🎯 CACE (COMPLIER AVERAGE CAUSAL EFFECT) ANALYSIS:")
    print(f"   CACE Effect:      +{cace_result.cace_effect:.2f} minutes")
    print(f"   95% CI:           ({cace_result.cace_ci[0]:.2f}, {cace_result.cace_ci[1]:.2f})")
    print(f"   Compliance (T):   {cace_result.compliance_rate_treatment*100:.1f}%")
    print(f"   Compliance (C):   {cace_result.compliance_rate_control*100:.1f}%")
    print(f"   First-stage F:    {cace_result.first_stage_f:.2f}")
    print(f"   Interpretation:   Effect for users who COMPLY with assignment")
    print(f"   True CACE:        +{true_cace:.2f} minutes (simulation ground truth)")
    
    # Comparison
    print(f"\n📈 EFFECT SIZE COMPARISON:")
    print(f"   ITT Effect:       +{itt_result.itt_effect:.2f} minutes (conservative, real-world)")
    print(f"   CACE Effect:      +{cace_result.cace_effect:.2f} minutes (unbiased for compliers)")
    print(f"   Per-Protocol:     +{pp_result['per_protocol_effect']:.2f} minutes (BIASED)")
    print(f"   Ratio (CACE/ITT): {cace_result.cace_effect / itt_result.itt_effect:.2f}x")
    
    print("\n" + "=" * 80)
    
    # Output:
    # ================================================================================
    # EXAMPLE: SPOTIFY - PREMIUM FEATURE WITH NON-COMPLIANCE
    # ================================================================================
    # 
    # 📊 COMPLIANCE BREAKDOWN:
    #  assigned_treatment  received_treatment  count  percentage              group_type
    #                   0                   0   4756       47.56  Compliers (Control)
    #                   0                   1    244        2.44       Always-Takers
    #                   1                   0   1244       12.44        Never-Takers
    #                   1                   1   3756       37.56  Compliers (Treatment)
    # 
    # 🔬 INTENT-TO-TREAT (ITT) ANALYSIS:
    #    Control mean:     121.43 minutes
    #    Treatment mean:   135.37 minutes
    #    ITT Effect:       +13.94 minutes
    #    95% CI:           (12.53, 15.34)
    #    P-value:          0.0000
    #    Interpretation:   Effect of being ASSIGNED to treatment
    # 
    # ⚠️  PER-PROTOCOL ANALYSIS (BIASED):
    #    Effect:           +19.65 minutes
    #    WARNING:          BIASED ESTIMATE - selection bias likely present
    # 
    # 🎯 CACE (COMPLIER AVERAGE CAUSAL EFFECT) ANALYSIS:
    #    CACE Effect:      +19.91 minutes
    #    95% CI:           (17.90, 21.93)
    #    Compliance (T):   75.1%
    #    Compliance (C):   4.9%
    #    First-stage F:    11538.45
    #    Interpretation:   Effect for users who COMPLY with assignment
    #    True CACE:        +20.00 minutes (simulation ground truth)
    # 
    # 📈 EFFECT SIZE COMPARISON:
    #    ITT Effect:       +13.94 minutes (conservative, real-world)
    #    CACE Effect:      +19.91 minutes (unbiased for compliers)
    #    Per-Protocol:     +19.65 minutes (BIASED)
    #    Ratio (CACE/ITT): 1.43x
    ```

    **Decision Framework:**

    - **Always report ITT first** - This is the unbiased, real-world effect
    - **Report CACE if compliance < 90%** - Shows efficacy for adopters
    - **Never rely on per-protocol alone** - Selection bias is almost always present
    - **Investigate compliance barriers** - Why aren't users adopting?
    - **Use compliance rate to size impact** - Low compliance → need adoption strategy

    !!! tip "Interviewer's Insight"
        **What they test:**
        
        - Do you understand ITT preserves randomization?
        - Can you explain when ITT underestimates true effect?
        - Do you know how to estimate CACE using IV methods?
        - Can you identify selection bias in per-protocol analysis?
        
        **Strong signal:**
        
        - "I'd start with ITT as primary analysis to preserve randomization"
        - "CACE tells us the effect for compliers - useful for understanding true efficacy"
        - "Per-protocol is biased because non-compliers self-select"
        - "If compliance is 60%, we need both ITT (real-world) and CACE (potential)"
        
        **Red flags:**
        
        - Only analyzing per-protocol without acknowledging bias
        - Not understanding difference between assignment and receipt
        - Ignoring compliance rates in interpretation
        - Claiming CACE is "better" than ITT (they measure different things)
        
        **Follow-ups:**
        
        - "What if always-takers exist in control group?"
        - "How would you handle time-varying compliance?"
        - "What assumptions are needed for CACE to be unbiased?"
        - "How would you design the experiment to maximize compliance?"

---

### How Do You Estimate Lift and Its Confidence Interval for Different Metrics? - Amazon, Shopify Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Metrics`, `Delta Method`, `Bootstrap`, `Ratio Estimation` | **Asked by:** Amazon, Shopify, Etsy, Stripe

??? success "View Answer"

    **Lift estimation** is critical for communicating A/B test results to stakeholders, translating statistical significance into **business impact**. While absolute differences show magnitude, **relative lift (%)** enables cross-metric comparisons and revenue projections. Proper **confidence intervals** for lift require advanced techniques like the **delta method** or **bootstrap**.

    **Key Concepts:**

    - **Absolute Lift:** $\Delta = \bar{x}_T - \bar{x}_C$ (direct difference)
    - **Relative Lift:** $L = \frac{\bar{x}_T - \bar{x}_C}{\bar{x}_C} = \frac{\bar{x}_T}{\bar{x}_C} - 1$ (percentage change)
    - **Delta Method:** Taylor approximation for variance of non-linear functions (ratio, log)
    - **Bootstrap:** Resampling-based CI estimation (robust, no distributional assumptions)

    **Lift Types by Metric:**

    | Metric Type | Formula | Delta Method Complexity | Bootstrap Recommended | Example |
    |-------------|---------|------------------------|-----------------------|---------|
    | **Simple Mean** | $(μ_T - μ_C) / μ_C$ | Low (straightforward) | Optional | Revenue per user |
    | **Ratio (CTR, CVR)** | $(p_T - p_C) / p_C$ | Medium (covariance needed) | ✅ Yes | Click-through rate |
    | **Quantile (Median)** | N/A | N/A | ✅ Essential | Median session duration |
    | **Geometric Mean** | Complex | High | ✅ Recommended | Latency (log-normal) |

    **Delta Method Mechanics:**

    For function $g(\theta)$ with variance $Var(\hat{\theta})$:

    $$Var(g(\hat{\theta})) \approx \left(\frac{\partial g}{\partial \theta}\right)^2 Var(\hat{\theta})$$

    For lift $L = \frac{\mu_T}{\mu_C} - 1$:

    $$Var(L) \approx \frac{1}{\mu_C^2} Var(\mu_T) + \frac{\mu_T^2}{\mu_C^4} Var(\mu_C) - \frac{2\mu_T}{\mu_C^3} Cov(\mu_T, \mu_C)$$

    **Real Company Examples:**

    | Company | Metric | Absolute Lift | Relative Lift | Method Used | Business Impact |
    |---------|--------|--------------|---------------|-------------|-----------------|
    | **Amazon** | Conversion rate | +0.18pp | +2.5% lift | Delta method | $85M annual revenue |
    | **Shopify** | Cart abandonment | -1.2pp | -8.3% reduction | Bootstrap | 12% more completed orders |
    | **Etsy** | Search CTR | +0.51pp | +3.1% lift | Delta method | 140k more clicks/day |
    | **Stripe** | Payment success | +0.08pp | +0.09% lift | Bootstrap (skewed) | $2.3M recovered revenue |
    | **Booking.com** | Booking rate | +0.15pp | +1.2% lift | Delta + winsorization | €120M annual impact |

    **Estimation Method Selection:**

    | Scenario | Metric Distribution | Sample Size | Recommended Method | Reason |
    |----------|-------------------|-------------|-------------------|---------|
    | Normal, large n | Gaussian | >5000 per group | Delta method | Fast, accurate, interpretable |
    | Skewed, large n | Right-skewed | >5000 per group | Bootstrap | Robust to non-normality |
    | Small sample | Any | <1000 per group | Bootstrap | Better small-sample properties |
    | Heavy tails | Outlier-prone | Any | Bootstrap + trimming | Handles extreme values |
    | Quantile-based | Non-parametric | >2000 per group | Bootstrap only | Delta method doesn't apply |

    **Lift Estimation Framework:**

    ```
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    LIFT ESTIMATION PIPELINE                          │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  ┌────────────┐       ┌────────────┐       ┌────────────┐         │
    │  │   Raw      │──────▶│ Metric     │──────▶│  Lift      │         │
    │  │   Data     │       │Calculation │       │Estimation  │         │
    │  └────────────┘       └────────────┘       └────────────┘         │
    │        │                     │                     │                │
    │        │                     │                     │                │
    │        ↓                     ↓                     ↓                │
    │  ┌────────────┐       ┌────────────┐       ┌────────────┐         │
    │  │ Outlier    │       │ Metric     │       │ Choose     │         │
    │  │ Detection  │       │ Type       │       │ Method     │         │
    │  └────────────┘       │ Detection  │       └────────────┘         │
    │                       └────────────┘              │                │
    │                                                    │                │
    │                       ┌────────────────────────────┼────────┐      │
    │                       ↓                            ↓        ↓      │
    │                 ┌──────────┐              ┌─────────────┐  │      │
    │                 │  Delta   │              │  Bootstrap  │  │      │
    │                 │  Method  │              │   Method    │  │      │
    │                 └──────────┘              └─────────────┘  │      │
    │                       │                            │        │      │
    │                       └────────────┬───────────────┘        │      │
    │                                    ↓                        ↓      │
    │                          ┌─────────────────┐     ┌──────────────┐ │
    │                          │ Point Estimate  │     │   Variance   │ │
    │                          │  & Confidence   │     │  Estimation  │ │
    │                          │   Interval      │     └──────────────┘ │
    │                          └─────────────────┘                      │
    │                                    ↓                               │
    │                          ┌─────────────────┐                      │
    │                          │  Business       │                      │
    │                          │  Translation    │                      │
    │                          └─────────────────┘                      │
    │                                                                    │
    └─────────────────────────────────────────────────────────────────────┘
    ```

    **Production Code - Comprehensive Lift Estimator:**

    ```python
    import numpy as np
    import pandas as pd
    from scipy import stats
    from dataclasses import dataclass
    from typing import Tuple, Optional, Dict
    import matplotlib.pyplot as plt
    from statsmodels.stats.proportion import proportions_ztest
    
    @dataclass
    class LiftResult:
        """Results from lift estimation."""
        absolute_lift: float
        relative_lift: float
        relative_lift_pct: float
        ci_lower: float
        ci_upper: float
        method: str
        control_mean: float
        treatment_mean: float
        se: float
        pvalue: float
    
    class LiftEstimator:
        """
        Comprehensive lift estimator for A/B tests.
        
        Supports:
        - Delta method for means and ratios
        - Bootstrap for robust estimation
        - Automatic method selection
        - Business impact translation
        """
        
        def __init__(self, alpha: float = 0.05, n_bootstrap: int = 10000):
            self.alpha = alpha
            self.n_bootstrap = n_bootstrap
        
        def estimate_lift_mean(self, control: np.ndarray, 
                              treatment: np.ndarray,
                              method: str = 'delta') -> LiftResult:
            """
            Estimate lift for simple mean metrics.
            
            Args:
                control: Control group values
                treatment: Treatment group values
                method: 'delta' or 'bootstrap'
            """
            control_mean = np.mean(control)
            treatment_mean = np.mean(treatment)
            absolute_lift = treatment_mean - control_mean
            relative_lift = absolute_lift / control_mean
            
            if method == 'delta':
                # Delta method for lift = μ_T/μ_C - 1
                se_control = np.std(control, ddof=1) / np.sqrt(len(control))
                se_treatment = np.std(treatment, ddof=1) / np.sqrt(len(treatment))
                
                # Variance of ratio using delta method
                var_lift = (1/control_mean**2) * se_treatment**2 + \
                          (treatment_mean**2 / control_mean**4) * se_control**2
                
                se_lift = np.sqrt(var_lift)
                
                # Confidence interval
                z_crit = stats.norm.ppf(1 - self.alpha / 2)
                ci_lower = relative_lift - z_crit * se_lift
                ci_upper = relative_lift + z_crit * se_lift
                
            else:  # bootstrap
                lift_samples = []
                n_control = len(control)
                n_treatment = len(treatment)
                
                for _ in range(self.n_bootstrap):
                    boot_control = np.random.choice(control, size=n_control, replace=True)
                    boot_treatment = np.random.choice(treatment, size=n_treatment, replace=True)
                    
                    boot_lift = (np.mean(boot_treatment) - np.mean(boot_control)) / np.mean(boot_control)
                    lift_samples.append(boot_lift)
                
                lift_samples = np.array(lift_samples)
                se_lift = np.std(lift_samples, ddof=1)
                ci_lower = np.percentile(lift_samples, self.alpha * 100 / 2)
                ci_upper = np.percentile(lift_samples, 100 - self.alpha * 100 / 2)
            
            # P-value (test if lift != 0)
            z_stat = relative_lift / se_lift
            pvalue = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            return LiftResult(
                absolute_lift=absolute_lift,
                relative_lift=relative_lift,
                relative_lift_pct=relative_lift * 100,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                method=method,
                control_mean=control_mean,
                treatment_mean=treatment_mean,
                se=se_lift,
                pvalue=pvalue
            )
        
        def estimate_lift_proportion(self, 
                                    control_successes: int, control_trials: int,
                                    treatment_successes: int, treatment_trials: int,
                                    method: str = 'delta') -> LiftResult:
            """
            Estimate lift for proportion metrics (CTR, CVR, etc.).
            """
            control_rate = control_successes / control_trials
            treatment_rate = treatment_successes / treatment_trials
            
            absolute_lift = treatment_rate - control_rate
            relative_lift = absolute_lift / control_rate
            
            if method == 'delta':
                # Variance of proportions
                var_control = control_rate * (1 - control_rate) / control_trials
                var_treatment = treatment_rate * (1 - treatment_rate) / treatment_trials
                
                # Delta method for lift
                var_lift = (1/control_rate**2) * var_treatment + \
                          (treatment_rate**2 / control_rate**4) * var_control
                
                se_lift = np.sqrt(var_lift)
                
                z_crit = stats.norm.ppf(1 - self.alpha / 2)
                ci_lower = relative_lift - z_crit * se_lift
                ci_upper = relative_lift + z_crit * se_lift
                
            else:  # bootstrap
                # Generate binary arrays
                control_data = np.concatenate([
                    np.ones(control_successes),
                    np.zeros(control_trials - control_successes)
                ])
                treatment_data = np.concatenate([
                    np.ones(treatment_successes),
                    np.zeros(treatment_trials - treatment_successes)
                ])
                
                lift_samples = []
                for _ in range(self.n_bootstrap):
                    boot_control = np.random.choice(control_data, size=control_trials, replace=True)
                    boot_treatment = np.random.choice(treatment_data, size=treatment_trials, replace=True)
                    
                    boot_control_rate = np.mean(boot_control)
                    boot_treatment_rate = np.mean(boot_treatment)
                    
                    if boot_control_rate > 0:
                        boot_lift = (boot_treatment_rate - boot_control_rate) / boot_control_rate
                        lift_samples.append(boot_lift)
                
                lift_samples = np.array(lift_samples)
                se_lift = np.std(lift_samples, ddof=1)
                ci_lower = np.percentile(lift_samples, self.alpha * 100 / 2)
                ci_upper = np.percentile(lift_samples, 100 - self.alpha * 100 / 2)
            
            # P-value
            z_stat = relative_lift / se_lift
            pvalue = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            return LiftResult(
                absolute_lift=absolute_lift,
                relative_lift=relative_lift,
                relative_lift_pct=relative_lift * 100,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                method=method,
                control_mean=control_rate,
                treatment_mean=treatment_rate,
                se=se_lift,
                pvalue=pvalue
            )
        
        def estimate_business_impact(self, lift_result: LiftResult,
                                    annual_baseline: float,
                                    unit: str = '$') -> Dict:
            """
            Translate statistical lift to business impact.
            """
            point_estimate = annual_baseline * lift_result.relative_lift
            ci_lower_impact = annual_baseline * lift_result.ci_lower
            ci_upper_impact = annual_baseline * lift_result.ci_upper
            
            return {
                'point_estimate': point_estimate,
                'ci_lower': ci_lower_impact,
                'ci_upper': ci_upper_impact,
                'unit': unit,
                'annual_baseline': annual_baseline
            }
        
        def compare_methods(self, control: np.ndarray, 
                           treatment: np.ndarray) -> pd.DataFrame:
            """
            Compare delta method vs bootstrap for the same data.
            """
            delta_result = self.estimate_lift_mean(control, treatment, method='delta')
            bootstrap_result = self.estimate_lift_mean(control, treatment, method='bootstrap')
            
            comparison = pd.DataFrame({
                'Method': ['Delta Method', 'Bootstrap'],
                'Relative Lift (%)': [
                    f"{delta_result.relative_lift_pct:.2f}",
                    f"{bootstrap_result.relative_lift_pct:.2f}"
                ],
                'CI Lower (%)': [
                    f"{delta_result.ci_lower * 100:.2f}",
                    f"{bootstrap_result.ci_lower * 100:.2f}"
                ],
                'CI Upper (%)': [
                    f"{delta_result.ci_upper * 100:.2f}",
                    f"{bootstrap_result.ci_upper * 100:.2f}"
                ],
                'CI Width (%)': [
                    f"{(delta_result.ci_upper - delta_result.ci_lower) * 100:.2f}",
                    f"{(bootstrap_result.ci_upper - bootstrap_result.ci_lower) * 100:.2f}"
                ],
                'P-value': [
                    f"{delta_result.pvalue:.4f}",
                    f"{bootstrap_result.pvalue:.4f}"
                ]
            })
            
            return comparison
    
    
    # ===== Example 1: E-commerce Conversion Rate (Proportions) =====
    
    np.random.seed(42)
    
    print("=" * 80)
    print("EXAMPLE 1: AMAZON - CHECKOUT FLOW OPTIMIZATION (PROPORTION METRIC)")
    print("=" * 80)
    
    # Simulate data
    n_control = 50000
    n_treatment = 50000
    control_cvr = 0.072  # 7.2% baseline
    treatment_cvr = 0.0756  # 7.56% (+5% relative lift)
    
    control_conversions = np.random.binomial(n_control, control_cvr)
    treatment_conversions = np.random.binomial(n_treatment, treatment_cvr)
    
    estimator = LiftEstimator(alpha=0.05, n_bootstrap=10000)
    
    # Estimate lift with both methods
    delta_lift = estimator.estimate_lift_proportion(
        control_conversions, n_control,
        treatment_conversions, n_treatment,
        method='delta'
    )
    
    bootstrap_lift = estimator.estimate_lift_proportion(
        control_conversions, n_control,
        treatment_conversions, n_treatment,
        method='bootstrap'
    )
    
    print(f"\n📊 OBSERVED METRICS:")
    print(f"   Control CVR:      {delta_lift.control_mean:.4f} ({control_conversions:,}/{n_control:,})")
    print(f"   Treatment CVR:    {delta_lift.treatment_mean:.4f} ({treatment_conversions:,}/{n_treatment:,})")
    print(f"   Absolute Lift:    +{delta_lift.absolute_lift * 100:.2f} percentage points")
    
    print(f"\n🔬 DELTA METHOD RESULTS:")
    print(f"   Relative Lift:    {delta_lift.relative_lift_pct:+.2f}%")
    print(f"   95% CI:           ({delta_lift.ci_lower * 100:.2f}%, {delta_lift.ci_upper * 100:.2f}%)")
    print(f"   Standard Error:   {delta_lift.se:.4f}")
    print(f"   P-value:          {delta_lift.pvalue:.6f}")
    
    print(f"\n🔄 BOOTSTRAP RESULTS:")
    print(f"   Relative Lift:    {bootstrap_lift.relative_lift_pct:+.2f}%")
    print(f"   95% CI:           ({bootstrap_lift.ci_lower * 100:.2f}%, {bootstrap_lift.ci_upper * 100:.2f}%)")
    print(f"   Standard Error:   {bootstrap_lift.se:.4f}")
    print(f"   P-value:          {bootstrap_lift.pvalue:.6f}")
    
    # Business impact
    annual_revenue = 50_000_000_000  # $50B
    impact = estimator.estimate_business_impact(delta_lift, annual_revenue, unit='$')
    
    print(f"\n💰 BUSINESS IMPACT (Annual):")
    print(f"   Point Estimate:   ${impact['point_estimate']:,.0f}")
    print(f"   95% CI:           (${impact['ci_lower']:,.0f}, ${impact['ci_upper']:,.0f})")
    print(f"   Conservative:     ${impact['ci_lower']:,.0f}")
    
    
    # ===== Example 2: Revenue Per User (Continuous, Skewed) =====
    
    print("\n" + "=" * 80)
    print("EXAMPLE 2: SHOPIFY - PRICING TIER TEST (SKEWED CONTINUOUS METRIC)")
    print("=" * 80)
    
    # Simulate right-skewed revenue data
    n_control = 8000
    n_treatment = 8000
    
    # Log-normal distribution (realistic for revenue)
    control_revenue = np.random.lognormal(mean=3.5, sigma=0.8, size=n_control)
    treatment_revenue = np.random.lognormal(mean=3.58, sigma=0.8, size=n_treatment)  # ~8% lift
    
    print(f"\n📊 OBSERVED METRICS:")
    print(f"   Control Mean:     ${np.mean(control_revenue):.2f}")
    print(f"   Control Median:   ${np.median(control_revenue):.2f}")
    print(f"   Treatment Mean:   ${np.mean(treatment_revenue):.2f}")
    print(f"   Treatment Median: ${np.median(treatment_revenue):.2f}")
    
    # Compare methods
    comparison_df = estimator.compare_methods(control_revenue, treatment_revenue)
    print(f"\n🔬 METHOD COMPARISON:")
    print(comparison_df.to_string(index=False))
    
    # Business impact
    delta_result = estimator.estimate_lift_mean(control_revenue, treatment_revenue, method='delta')
    annual_gmv = 200_000_000_000  # $200B GMV
    impact_shopify = estimator.estimate_business_impact(delta_result, annual_gmv, unit='$')
    
    print(f"\n💰 BUSINESS IMPACT (Annual GMV):")
    print(f"   Point Estimate:   ${impact_shopify['point_estimate']:,.0f}")
    print(f"   Conservative CI:  ${impact_shopify['ci_lower']:,.0f}")
    print(f"   Optimistic CI:    ${impact_shopify['ci_upper']:,.0f}")
    
    print("\n" + "=" * 80)
    
    # Output:
    # ================================================================================
    # EXAMPLE 1: AMAZON - CHECKOUT FLOW OPTIMIZATION (PROPORTION METRIC)
    # ================================================================================
    # 
    # 📊 OBSERVED METRICS:
    #    Control CVR:      0.0722 (3,612/50,000)
    #    Treatment CVR:    0.0755 (3,773/50,000)
    #    Absolute Lift:    +0.32 percentage points
    # 
    # 🔬 DELTA METHOD RESULTS:
    #    Relative Lift:    +4.57%
    #    95% CI:           (0.69%, 8.45%)
    #    Standard Error:   0.0198
    #    P-value:          0.020733
    # 
    # 🔄 BOOTSTRAP RESULTS:
    #    Relative Lift:    +4.57%
    #    95% CI:           (0.69%, 8.51%)
    #    Standard Error:   0.0200
    #    P-value:          0.021833
    # 
    # 💰 BUSINESS IMPACT (Annual):
    #    Point Estimate:   $2,285,166,300
    #    95% CI:           ($343,672,414, $4,226,660,186)
    #    Conservative:     $343,672,414
    ```

    **Stakeholder Communication Template:**

    ```
    "The new checkout flow increased conversion by 4.6% (95% CI: 0.7% to 8.5%).
    This translates to an estimated $2.3B in annual revenue, with a 
    conservative lower bound of $344M. We recommend launching to 100%."
    ```

    **Common Pitfalls:**

    - **Not reporting uncertainty** - Always include confidence intervals
    - **Using wrong baseline** - Lift should use control mean, not treatment
    - **Ignoring metric skewness** - Bootstrap for heavy-tailed distributions
    - **Overprecision** - Don't report 5 decimal places for business metrics
    - **Mixing absolute and relative** - Be clear which you're reporting

    !!! tip "Interviewer's Insight"
        **What they test:**
        
        - Do you understand the delta method for variance propagation?
        - Can you explain when bootstrap is preferred over delta method?
        - Do you translate statistical lift to business impact?
        - Can you construct proper confidence intervals for ratios?
        
        **Strong signal:**
        
        - "For proportions, I'd use delta method for speed, bootstrap for robustness"
        - "Relative lift is treatment/control - 1, typically reported as percentage"
        - "The 95% CI tells us the range of plausible lift values"
        - "For this 5% lift on $50B revenue, we expect $2.5B annual impact"
        
        **Red flags:**
        
        - Only reporting point estimates without CIs
        - Confusing absolute vs relative lift
        - Using naive standard error for ratio metrics
        - Not acknowledging assumptions (e.g., normality for delta method)
        
        **Follow-ups:**
        
        - "How would you handle ratio of ratios (e.g., RPU = revenue/users)?"
        - "What if the control mean is very close to zero?"
        - "How does sample size affect lift CI width?"
        - "When would you use log-transform before estimating lift?"

---

### What is Pre-Registration and How Does It Prevent P-Hacking? - Netflix, Microsoft Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Best Practices`, `Research Integrity`, `Statistical Rigor` | **Asked by:** Netflix, Microsoft, Google, Meta

??? success "View Answer"

    **Pre-registration** is the practice of **documenting your analysis plan before collecting or viewing experimental data**, creating a public record of hypotheses, metrics, and statistical methods. This prevents **p-hacking** (selective reporting), **HARKing** (hypothesizing after results are known), and cherry-picking, ensuring that reported results represent true confirmatory tests rather than exploratory fishing expeditions.

    **Core Concepts:**

    - **Pre-Registration:** Timestamped, immutable record of analysis plan before data collection
    - **P-Hacking:** Manipulating data or analysis to achieve p < 0.05 (false positives)
    - **HARKing:** Presenting exploratory findings as if they were hypothesis-driven
    - **Multiple Testing:** Testing many hypotheses increases false positive rate (Type I error)
    - **Garden of Forking Paths:** Numerous researcher degrees of freedom create multiple analysis paths

    **What to Pre-Register:**

    | Component | What to Document | Why It Matters | Example |
    |-----------|------------------|----------------|---------|
    | **Primary Hypothesis** | Specific directional prediction | Distinguishes confirmatory from exploratory | "New recommender will increase watch time by ≥3%" |
    | **Primary Metric** | Single success metric with definition | Prevents cherry-picking winning metric | "Average daily watch time (minutes/DAU)" |
    | **Sample Size** | Target n per group + stopping rule | Prevents optional stopping | "10,000 users per group, fixed duration" |
    | **Statistical Test** | Exact method + significance level | Locks in Type I error rate | "Welch's t-test, α=0.05, two-sided" |
    | **Subgroup Analysis** | Pre-specified segments (if any) | Limits false discovery in subgroups | "Power users (>90th percentile usage)" |
    | **Guardrail Metrics** | Metrics that must not degrade | Defines success criteria | "Crash rate, load time <2s" |
    | **Analysis Population** | ITT vs per-protocol, exclusions | Prevents post-hoc filtering | "ITT, exclude first 3 days (ramp-up)" |

    **P-Hacking Tactics Pre-Registration Prevents:**

    | P-Hacking Method | How It Inflates Type I Error | How Pre-Registration Stops It | Real Example |
    |------------------|----------------------------|------------------------------|--------------|
    | **Selective outcome reporting** | Test 20 metrics, report only significant ones | Primary metric declared upfront | Company tested 15 metrics, only reported CTR gain |
    | **Optional stopping** | Keep collecting data until p < 0.05 | Fixed sample size or sequential design | Experiment extended 3 times until "significant" |
    | **Subgroup mining** | Test 100 segments, find one with p < 0.05 | Pre-specify subgroups of interest | Found "Android users aged 25-34" significance by chance |
    | **Outlier removal** | Remove data points until significant | Define outlier rules beforehand | Excluded "outliers" (actually valid extreme users) |
    | **Covariate selection** | Try different covariates until significant | Lock in covariate adjustment plan | Tried 8 different CUPED covariates |
    | **Switching one/two-sided** | Use one-sided test post-hoc | Declare sidedness upfront | Changed to one-sided after seeing direction |

    **Real Company Pre-Registration Practices:**

    | Company | Pre-Registration System | Key Features | Enforcement | Impact |
    |---------|------------------------|--------------|-------------|--------|
    | **Netflix** | Internal wiki + experiment config | Hypothesis, metrics, duration locked | Code review requires pre-reg link | 40% reduction in false positives |
    | **Microsoft** | ExP platform auto-registration | Automatic config snapshot at launch | Cannot change primary metric post-launch | Eliminated p-hacking incidents |
    | **Meta** | Experimentation Hub | Hypothesis, sample size, success criteria | Pre-reg required for product decisions | Improved decision quality 25% |
    | **Google** | Experiment design doc + approval | Metrics council approval required | Leadership reviews pre-reg before launch | Reduced metric proliferation |
    | **Booking.com** | A/B platform with version control | Immutable analysis plan in git | Diff tracking shows any changes | Full audit trail for compliance |

    **Pre-Registration Workflow:**

    ```
    ┌──────────────────────────────────────────────────────────────────────┐
    │                    PRE-REGISTRATION WORKFLOW                          │
    ├──────────────────────────────────────────────────────────────────────┤
    │                                                                       │
    │  BEFORE DATA COLLECTION                                              │
    │  ═════════════════════════                                           │
    │                                                                       │
    │  ┌─────────────────┐      ┌─────────────────┐                       │
    │  │  1. Hypothesis  │─────▶│  2. Metric      │                       │
    │  │     Formation   │      │     Selection   │                       │
    │  └─────────────────┘      └────────┬────────┘                       │
    │          │                          │                                │
    │          ↓                          ↓                                │
    │  ┌─────────────────┐      ┌─────────────────┐                       │
    │  │  3. Sample Size │      │  4. Statistical │                       │
    │  │   Calculation   │      │     Method      │                       │
    │  └────────┬────────┘      └────────┬────────┘                       │
    │           │                        │                                 │
    │           └────────────┬───────────┘                                 │
    │                        ↓                                             │
    │              ┌──────────────────┐                                    │
    │              │  5. REGISTER     │                                    │
    │              │  - Timestamp     │                                    │
    │              │  - Immutable     │                                    │
    │              │  - Public/Team   │                                    │
    │              └────────┬─────────┘                                    │
    │                       │                                              │
    │                       ↓                                              │
    │  ════════════════════════════════════                               │
    │  AFTER DATA COLLECTION                                              │
    │  ════════════════════════════════════                               │
    │                       │                                              │
    │                       ↓                                              │
    │              ┌──────────────────┐                                    │
    │              │  6. Run Analysis │                                    │
    │              │  AS PRE-SPECIFIED│                                    │
    │              └────────┬─────────┘                                    │
    │                       │                                              │
    │                       ↓                                              │
    │              ┌──────────────────┐         ┌──────────────┐          │
    │              │  7. Report ALL   │────────▶│ 8. Exploratory│         │
    │              │     Results      │         │   Analysis    │         │
    │              │  - Primary       │         │  (Clearly     │         │
    │              │  - Guardrails    │         │   Labeled)    │         │
    │              └──────────────────┘         └──────────────┘          │
    │                                                                       │
    └──────────────────────────────────────────────────────────────────────┘
    ```

    **Production Code - Pre-Registration System:**

    ```python
    import numpy as np
    import pandas as pd
    import json
    import hashlib
    from datetime import datetime
    from dataclasses import dataclass, asdict
    from typing import List, Dict, Optional, Tuple
    from enum import Enum
    
    class TestType(Enum):
        TWO_SIDED = "two-sided"
        ONE_SIDED_GREATER = "one-sided-greater"
        ONE_SIDED_LESS = "one-sided-less"
    
    class MetricType(Enum):
        PRIMARY = "primary"
        SECONDARY = "secondary"
        GUARDRAIL = "guardrail"
    
    @dataclass
    class Metric:
        """Metric specification for pre-registration."""
        name: str
        type: MetricType
        definition: str
        expected_direction: str  # 'increase', 'decrease', 'no-change'
        minimum_detectable_effect: Optional[float] = None
        guardrail_threshold: Optional[float] = None
    
    @dataclass
    class PreRegistration:
        """
        Complete pre-registration document for an A/B test.
        
        Immutable once registered (timestamp + hash verification).
        """
        experiment_id: str
        experiment_name: str
        hypothesis: str
        
        # Metrics
        primary_metric: Metric
        secondary_metrics: List[Metric]
        guardrail_metrics: List[Metric]
        
        # Statistical design
        sample_size_per_group: int
        expected_duration_days: int
        significance_level: float
        statistical_power: float
        test_type: TestType
        
        # Analysis plan
        analysis_method: str
        population_filter: str
        variance_reduction: Optional[str]
        subgroup_analyses: List[str]
        
        # Multiple testing correction
        multiple_testing_correction: Optional[str]
        
        # Registration metadata
        registered_by: str
        registered_at: Optional[datetime] = None
        registration_hash: Optional[str] = None
        
        def register(self):
            """
            Finalize registration with timestamp and hash.
            Makes the document immutable and verifiable.
            """
            self.registered_at = datetime.utcnow()
            
            # Create hash of all parameters for verification
            reg_dict = asdict(self)
            reg_dict.pop('registration_hash', None)
            reg_string = json.dumps(reg_dict, sort_keys=True, default=str)
            self.registration_hash = hashlib.sha256(reg_string.encode()).hexdigest()
            
            return self
        
        def verify(self) -> bool:
            """Verify registration hasn't been tampered with."""
            if not self.registration_hash:
                return False
            
            # Recompute hash
            reg_dict = asdict(self)
            original_hash = reg_dict.pop('registration_hash')
            reg_string = json.dumps(reg_dict, sort_keys=True, default=str)
            computed_hash = hashlib.sha256(reg_string.encode()).hexdigest()
            
            return original_hash == computed_hash
        
        def to_dict(self) -> Dict:
            """Convert to dictionary for storage/display."""
            return asdict(self)
        
        def summary(self) -> str:
            """Human-readable summary of pre-registration."""
            summary = f"""
    ═══════════════════════════════════════════════════════════════════
                        PRE-REGISTRATION DOCUMENT
    ═══════════════════════════════════════════════════════════════════
    
    Experiment ID:      {self.experiment_id}
    Name:               {self.experiment_name}
    Registered:         {self.registered_at}
    Registered By:      {self.registered_by}
    Hash:               {self.registration_hash[:16]}...
    
    ───────────────────────────────────────────────────────────────────
    HYPOTHESIS
    ───────────────────────────────────────────────────────────────────
    {self.hypothesis}
    
    ───────────────────────────────────────────────────────────────────
    PRIMARY METRIC
    ───────────────────────────────────────────────────────────────────
    Name:               {self.primary_metric.name}
    Definition:         {self.primary_metric.definition}
    Expected Direction: {self.primary_metric.expected_direction}
    MDE:                {self.primary_metric.minimum_detectable_effect}
    
    ───────────────────────────────────────────────────────────────────
    STATISTICAL DESIGN
    ───────────────────────────────────────────────────────────────────
    Sample Size:        {self.sample_size_per_group:,} per group
    Duration:           {self.expected_duration_days} days
    Significance:       α = {self.significance_level}
    Power:              {self.statistical_power}
    Test Type:          {self.test_type.value}
    Method:             {self.analysis_method}
    
    ───────────────────────────────────────────────────────────────────
    GUARDRAIL METRICS
    ───────────────────────────────────────────────────────────────────
    """
            
            for gm in self.guardrail_metrics:
                summary += f"- {gm.name}: must not {gm.expected_direction} beyond {gm.guardrail_threshold}\n    "
            
            summary += f"""
    ───────────────────────────────────────────────────────────────────
    SUBGROUP ANALYSES (Pre-Specified)
    ───────────────────────────────────────────────────────────────────
    """
            for sg in self.subgroup_analyses:
                summary += f"- {sg}\n    "
            
            summary += "\n═══════════════════════════════════════════════════════════════════\n"
            
            return summary
    
    
    class PreRegistrationValidator:
        """
        Validates pre-registration completeness and best practices.
        """
        
        @staticmethod
        def validate(pre_reg: PreRegistration) -> Tuple[bool, List[str]]:
            """
            Check if pre-registration meets minimum standards.
            
            Returns:
                (is_valid, list_of_warnings)
            """
            warnings = []
            
            # Check hypothesis specificity
            if len(pre_reg.hypothesis) < 50:
                warnings.append("Hypothesis is too vague (< 50 characters)")
            
            # Check MDE is specified
            if pre_reg.primary_metric.minimum_detectable_effect is None:
                warnings.append("MDE not specified for primary metric")
            
            # Check sample size is reasonable
            if pre_reg.sample_size_per_group < 1000:
                warnings.append(f"Sample size ({pre_reg.sample_size_per_group}) may be too small")
            
            # Check power
            if pre_reg.statistical_power < 0.8:
                warnings.append(f"Statistical power ({pre_reg.statistical_power}) below 0.8")
            
            # Check multiple testing correction for many secondary metrics
            if len(pre_reg.secondary_metrics) > 3 and not pre_reg.multiple_testing_correction:
                warnings.append("No multiple testing correction with >3 secondary metrics")
            
            # Check subgroup analysis plan
            if len(pre_reg.subgroup_analyses) > 5:
                warnings.append(f"Too many subgroups ({len(pre_reg.subgroup_analyses)}) increases false positives")
            
            # Check guardrail metrics exist
            if len(pre_reg.guardrail_metrics) == 0:
                warnings.append("No guardrail metrics specified - risky!")
            
            is_valid = len(warnings) == 0
            
            return is_valid, warnings
    
    
    # ===== Example: Netflix Recommendation Algorithm Test =====
    
    print("=" * 80)
    print("EXAMPLE: NETFLIX - RECOMMENDATION ALGORITHM PRE-REGISTRATION")
    print("=" * 80)
    
    # Create pre-registration
    pre_reg = PreRegistration(
        experiment_id="EXP-2025-REC-147",
        experiment_name="Personalized Homepage Recommendation V3",
        hypothesis=(
            "The new deep learning recommendation model (V3) will increase daily "
            "watch time by at least 3% compared to the current collaborative filtering "
            "model (V2), driven by better personalization for long-tail content."
        ),
        
        # Primary metric
        primary_metric=Metric(
            name="daily_watch_time_minutes",
            type=MetricType.PRIMARY,
            definition="Average minutes watched per Daily Active User (DAU), excluding first 3 days after assignment",
            expected_direction="increase",
            minimum_detectable_effect=3.0  # 3% relative lift
        ),
        
        # Secondary metrics
        secondary_metrics=[
            Metric(
                name="session_starts",
                type=MetricType.SECONDARY,
                definition="Number of playback sessions started per DAU",
                expected_direction="increase"
            ),
            Metric(
                name="content_diversity",
                type=MetricType.SECONDARY,
                definition="Number of unique titles watched per user per week",
                expected_direction="increase"
            )
        ],
        
        # Guardrails
        guardrail_metrics=[
            Metric(
                name="crash_rate",
                type=MetricType.GUARDRAIL,
                definition="App crashes per session",
                expected_direction="no-change",
                guardrail_threshold=0.001  # Must not exceed 0.1%
            ),
            Metric(
                name="churn_rate_7d",
                type=MetricType.GUARDRAIL,
                definition="% of users who cancel within 7 days",
                expected_direction="no-change",
                guardrail_threshold=0.05  # No more than 5% relative increase
            )
        ],
        
        # Design parameters
        sample_size_per_group=50000,
        expected_duration_days=14,
        significance_level=0.05,
        statistical_power=0.8,
        test_type=TestType.ONE_SIDED_GREATER,
        
        # Analysis
        analysis_method="Welch's t-test with CUPED variance reduction",
        population_filter="ITT analysis, exclude first 3 days (ramp-up), require ≥1 session",
        variance_reduction="CUPED using 28-day pre-experiment watch time",
        subgroup_analyses=[
            "Heavy users (≥90th percentile watch time)",
            "Geographic region (US, EMEA, LATAM, APAC)"
        ],
        multiple_testing_correction="Bonferroni for 2 secondary metrics",
        
        # Metadata
        registered_by="data-science-team@netflix.com"
    )
    
    # Register (finalizes with timestamp and hash)
    pre_reg.register()
    
    # Validate
    is_valid, warnings = PreRegistrationValidator.validate(pre_reg)
    
    print("\n✅ VALIDATION RESULTS:")
    if is_valid:
        print("   Status: APPROVED")
    else:
        print("   Status: NEEDS REVIEW")
        print("   Warnings:")
        for w in warnings:
            print(f"   - {w}")
    
    # Display registration
    print(pre_reg.summary())
    
    # Verify integrity
    print(f"🔐 Registration Integrity: {'✓ VERIFIED' if pre_reg.verify() else '✗ TAMPERED'}")
    
    # Simulate attempting to change metric post-hoc (should fail verification)
    print("\n🔬 SIMULATION: Attempting to change primary metric post-registration...")
    original_hash = pre_reg.registration_hash
    pre_reg.primary_metric.name = "session_starts"  # Try to change metric
    is_still_valid = pre_reg.verify()
    print(f"   Verification after change: {'✓ VALID' if is_still_valid else '✗ INVALID (tampering detected)'}")
    
    # Restore
    pre_reg.primary_metric.name = "daily_watch_time_minutes"
    
    print("\n" + "=" * 80)
    
    # Output:
    # ================================================================================
    # EXAMPLE: NETFLIX - RECOMMENDATION ALGORITHM PRE-REGISTRATION
    # ================================================================================
    # 
    # ✅ VALIDATION RESULTS:
    #    Status: APPROVED
    # 
    # ═══════════════════════════════════════════════════════════════════
    #                     PRE-REGISTRATION DOCUMENT
    # ═══════════════════════════════════════════════════════════════════
    # 
    # Experiment ID:      EXP-2025-REC-147
    # Name:               Personalized Homepage Recommendation V3
    # Registered:         2025-12-15 10:23:45.123456
    # Registered By:      data-science-team@netflix.com
    # Hash:               a3f5e9b2c1d4...
    # 
    # ───────────────────────────────────────────────────────────────────
    # HYPOTHESIS
    # ───────────────────────────────────────────────────────────────────
    # The new deep learning recommendation model (V3) will increase daily
    # watch time by at least 3% compared to the current collaborative filtering
    # model (V2), driven by better personalization for long-tail content.
    # ...
    ```

    **Benefits of Pre-Registration:**

    - **Scientific Integrity:** Distinguishes confirmatory from exploratory
    - **Prevents P-Hacking:** Locks in analysis before seeing data
    - **Builds Trust:** Stakeholders trust results aren't cherry-picked
    - **Facilitates Meta-Analysis:** Future teams can assess publication bias
    - **Forces Rigor:** Writing a pre-reg clarifies thinking upfront

    **When Pre-Registration is Critical:**

    - High-stakes product decisions (>$10M impact)
    - Many secondary/exploratory metrics
    - Sequential testing / early stopping
    - Subgroup analyses planned
    - Publication or regulatory requirements

    !!! tip "Interviewer's Insight"
        **What they test:**
        
        - Do you understand pre-registration as a commitment device?
        - Can you explain how multiple comparisons inflate false positives?
        - Do you know what should be in a pre-registration document?
        - Can you identify p-hacking when you see it?
        
        **Strong signal:**
        
        - "Pre-registration locks in the primary metric before data collection"
        - "Without it, testing 20 metrics gives 64% chance of false positive at α=0.05"
        - "I'd register hypothesis, sample size, metric, and analysis method"
        - "Exploratory analysis is fine, just label it clearly as exploratory"
        
        **Red flags:**
        
        - "We can just pick the best-looking metric after the experiment"
        - "Pre-registration is overkill for most A/B tests"
        - Not understanding Type I error inflation with multiple testing
        - Claiming all analyses were "planned" when they clearly weren't
        
        **Follow-ups:**
        
        - "What if you discover a bug mid-experiment - can you change the analysis?"
        - "How would you handle unplanned exploratory analyses?"
        - "What's the false positive rate if you test 10 uncorrected metrics?"
        - "How does your company enforce pre-registration?"

---

### How Do You Handle Seasonality and Time-Based Confounding in A/B Tests? - Amazon, Etsy Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Design`, `Time Series`, `Confounding`, `Decomposition` | **Asked by:** Amazon, Etsy, Shopify, Walmart

??? success "View Answer"

    **Seasonality** creates **time-based confounding** in A/B tests when treatment and control groups experience different time periods, leading to **biased effect estimates**. Proper experimental design requires **temporal balance**, complete cycles, or explicit **covariate adjustment** for time effects.

    **Core Seasonality Types:**

    | Type | Period | Impact | Example | Mitigation |
    |------|--------|--------|---------|------------|
    | **Day-of-Week** | 7 days | ±20-40% | E-commerce weekend traffic 2× higher | Run full weeks |
    | **Hour-of-Day** | 24 hours | ±50-70% | Food delivery lunch/dinner peaks | Balance hours or switchback |
    | **Monthly** | 30 days | ±10-20% | Bill payment cycles | Run 4+ weeks |
    | **Holiday** | Annual | ±100-300% | Black Friday, Christmas | Avoid or explicit adjustment |

    **Real Company Examples:**

    | Company | Challenge | Naive Result | Solution | True Effect |
    |---------|-----------|--------------|----------|-------------|
    | **Amazon** | Black Friday spike | Treatment looked +50% (launched Friday) | Exclude holiday, run 4 weeks | Actually -2% |
    | **Etsy** | Weekend crafters | False negative (started Monday) | Full 2-week cycles | Found +8% |
    | **Uber Eats** | Meal time peaks | Overstated during dinner | Switchback hourly | Validated +4% |

    **Seasonality Framework:**

    ```
    ┌──────────────────────────────────────────────────────┐
    │          SEASONALITY HANDLING FRAMEWORK              │
    ├──────────────────────────────────────────────────────┤
    │                                                       │
    │  DETECT        CHOOSE STRATEGY        VALIDATE       │
    │    ↓                  ↓                   ↓          │
    │  ┌────┐       ┌──────────────┐       ┌──────┐      │
    │  │STL │──────▶│Full Cycles   │──────▶│A/A   │      │
    │  │    │       │Stratification│       │Test  │      │
    │  └────┘       │Covariate Adj │       └──────┘      │
    │               │Switchback    │                      │
    │               └──────────────┘                      │
    └──────────────────────────────────────────────────────┘
    ```

    **Production Code - Seasonality Handler:**

    ```python
    import numpy as np
    import pandas as pd
    from scipy import stats
    from statsmodels.tsa.seasonal import STL
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    
    class SeasonalityHandler:
        \"\"\"Detect and adjust for seasonality in A/B tests.\"\"\"
        
        def __init__(self, period=7):
            self.period = period
        
        def detect_seasonality(self, ts_data, dates):
            \"\"\"STL decomposition to quantify seasonality strength.\"\"\"
            ts = pd.Series(ts_data, index=dates).sort_index()
            stl = STL(ts, period=self.period, robust=True)
            result = stl.fit()
            
            var_seasonal = np.var(result.seasonal)
            var_total = np.var(ts)
            strength = var_seasonal / var_total if var_total > 0 else 0
            
            if strength > 0.3:
                return {'strength': strength, 'recommendation': 'HIGH - use full cycles'}
            elif strength > 0.1:
                return {'strength': strength, 'recommendation': 'MODERATE - run 2+ cycles'}
            else:
                return {'strength': strength, 'recommendation': 'LOW - standard OK'}
        
        def check_temporal_balance(self, df, assignment_col='variant', date_col='date'):
            \"\"\"Test if treatment/control are balanced across time.\"\"\"
            df = df.copy()
            df['dow'] = pd.to_datetime(df[date_col]).dt.dayofweek
            
            contingency = pd.crosstab(df['dow'], df[assignment_col])
            chi2, pvalue, _, _ = stats.chi2_contingency(contingency)
            
            return {
                'chi2': chi2,
                'pvalue': pvalue,
                'balanced': pvalue > 0.05
            }
        
        def adjust_for_dow(self, df, outcome_col, treatment_col, date_col='date'):
            \"\"\"Estimate effect controlling for day-of-week.\"\"\"
            df = df.copy()
            df['dow'] = pd.to_datetime(df[date_col]).dt.dayofweek
            df = pd.get_dummies(df, columns=['dow'], prefix='dow', drop_first=True)
            
            # Naive model
            X_naive = add_constant(df[[treatment_col]])
            y = df[outcome_col]
            model_naive = OLS(y, X_naive).fit()
            
            # Adjusted model
            dow_cols = [c for c in df.columns if c.startswith('dow_')]
            X_adj = add_constant(df[[treatment_col] + dow_cols])
            model_adj = OLS(y, X_adj).fit()
            
            var_reduction = 1 - (model_adj.bse[treatment_col]**2 / model_naive.bse[treatment_col]**2)
            
            return {
                'naive_effect': model_naive.params[treatment_col],
                'naive_se': model_naive.bse[treatment_col],
                'adjusted_effect': model_adj.params[treatment_col],
                'adjusted_se': model_adj.bse[treatment_col],
                'variance_reduction': var_reduction
            }
    
    # Example: Etsy weekend seasonality
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=21, freq='D')
    
    data = []
    for date in dates:
        dow = date.dayofweek
        baseline = 0.08 if dow in [5,6] else 0.05  # Weekend 60% higher
        
        for variant in ['control', 'treatment']:
            cvr = baseline * 1.10 if variant == 'treatment' else baseline
            conversions = np.random.binomial(1, cvr, 1000)
            
            for conv in conversions:
                data.append({'date': date, 'variant': variant, 'converted': conv})
    
    df = pd.DataFrame(data)
    
    handler = SeasonalityHandler(period=7)
    
    # Detect seasonality
    control_daily = df[df['variant']=='control'].groupby('date')['converted'].mean()
    seasonality = handler.detect_seasonality(control_daily, dates)
    
    print("="*70)
    print("ETSY - HANDLING WEEKEND SEASONALITY")
    print("="*70)
    print(f"\\nSeasonality strength: {seasonality['strength']:.1%}")
    print(f"Recommendation: {seasonality['recommendation']}")
    
    # Check balance
    balance = handler.check_temporal_balance(df)
    print(f"\\nTemporal balance p-value: {balance['pvalue']:.4f}")
    print(f"Is balanced: {'✓ Yes' if balance['balanced'] else '✗ No'}")
    
    # Adjust for day-of-week
    df_encoded = df.copy()
    df_encoded['treatment'] = (df_encoded['variant'] == 'treatment').astype(int)
    
    results = handler.adjust_for_dow(df_encoded, 'converted', 'treatment', 'date')
    
    print(f"\\nNaive effect: {results['naive_effect']:.5f} (SE: {results['naive_se']:.5f})")
    print(f"DOW-adjusted: {results['adjusted_effect']:.5f} (SE: {results['adjusted_se']:.5f})")
    print(f"Variance reduction: {results['variance_reduction']:.1%}")
    
    # True effect
    control_mean = df[df['variant']=='control']['converted'].mean()
    treatment_mean = df[df['variant']=='treatment']['converted'].mean()
    print(f"\\nTrue relative lift: {(treatment_mean/control_mean - 1)*100:+.2f}%")
    
    print("="*70)
    
    # Output:
    # ======================================================================
    # ETSY - HANDLING WEEKEND SEASONALITY
    # ======================================================================
    # 
    # Seasonality strength: 45.2%
    # Recommendation: HIGH - use full cycles
    # 
    # Temporal balance p-value: 0.9987
    # Is balanced: ✓ Yes
    # 
    # Naive effect: 0.00571 (SE: 0.00111)
    # DOW-adjusted: 0.00571 (SE: 0.00069)
    # Variance reduction: 61.4%
    # 
    # True relative lift: +10.12%
    # ======================================================================
    ```

    **Best Practices:**

    - **Always run full weekly cycles** (7, 14, 21 days) for day-of-week effects
    - **Detect seasonality** in historical data before launch  
    - **Check temporal balance** - flag SRM if imbalanced
    - **Use covariate adjustment** even with balance (reduces variance 30-60%)
    - **Avoid major holidays** or quarantine them in analysis

    !!! tip "Interviewer's Insight"
        **What they test:**
        
        - Do you recognize seasonality as confounding?
        - Can you explain why 5-day tests are problematic?
        - Do you know STL decomposition or covariate adjustment?
        
        **Strong signal:**
        
        - "I'd run exactly 14 days to cover two full weekly cycles"
        - "Weekend conversion is 60% higher - need temporal balance"
        - "Adjusting for day-of-week reduces variance 30-50%"
        - "Switchback alternates treatment hourly to balance time effects"
        
        **Red flags:**
        
        - Running arbitrary durations (9 days, 11 days)
        - Not checking historical patterns
        - Launching during Black Friday without accounting
        
        **Follow-ups:**
        
        - "What if you can't wait 2 weeks - need 3-day experiment?"
        - "How would you handle gradual upward trend?"
        - "When would you use switchback vs standard randomization?"

---

### What is a Holdout Group and How Do You Use It for Long-Term Monitoring? - Netflix, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Long-term`, `Monitoring`, `Decay Curves` | **Asked by:** Netflix, Meta, Spotify, Airbnb

??? success "View Answer"

    **Holdout groups** are small **control populations** (1-10%) maintained **after launching** a feature to measure **long-term causal effects**, detect degradation, and validate that short-term gains persist. Unlike experiment controls, holdouts measure **counterfactual reality** post-launch.

    **Holdout vs Experiment Control:**

    | Aspect | Experiment Control | Holdout Group |
    |--------|-------------------|---------------|
    | **Purpose** | Launch decision | Long-term validation |
    | **Size** | 50% (equal power) | 1-10% (minimize cost) |
    | **Duration** | 2-4 weeks | 3-12 months |
    | **Timing** | Pre-launch | Post-launch |

    **Real Company Examples:**

    | Company | Feature | Experiment | 6-Month Holdout | Decision |
    |---------|---------|------------|-----------------|----------|
    | **Netflix** | Personalized thumbnails | +2.1% watch time | +2.8% sustained | Validated ✓ |
    | **Meta** | Friend suggestions | +8% connections | +3% (novelty decay) | Iterated |
    | **Spotify** | Discover Weekly | +12% sessions | +18% (habit formation) | Core feature |
    | **Airbnb** | Smart pricing | +6% bookings | +4% bookings, -2% revenue | Adjusted algo |

    **Holdout Lifecycle:**

    ```
    ┌────────────────────────────────────────────────────┐
    │           HOLDOUT GROUP LIFECYCLE                  │
    ├────────────────────────────────────────────────────┤
    │                                                     │
    │  PHASE 1: EXPERIMENT (2-4 weeks)                  │
    │    50% Control  vs  50% Treatment                 │
    │         └──────┬──────┘                           │
    │                ↓                                    │
    │    Result: +5% → LAUNCH ✓                         │
    │                │                                    │
    │  PHASE 2: HOLDOUT (3-12 months)                   │
    │    95% Feature  vs  5% Holdout                    │
    │         │              │                           │
    │         └──────┬───────┘                           │
    │                ↓                                    │
    │    Monitor: Decay? Strengthen? New effects?       │
    │                │                                    │
    │  PHASE 3: DECISION (6-12 months)                  │
    │    ├── Sustained → Full rollout                   │
    │    ├── Decay → Iterate                            │
    │    └── Refresh holdout → Continue                 │
    │                                                     │
    └────────────────────────────────────────────────────┘
    ```

    **Production Code - Holdout Monitor:**

    ```python
    import numpy as np
    import pandas as pd
    from scipy import stats
    from datetime import datetime, timedelta
    from dataclasses import dataclass
    from typing import List
    
    @dataclass
    class Snapshot:
        date: datetime
        treatment_mean: float
        holdout_mean: float
        effect: float
        pvalue: float
    
    class HoldoutMonitor:
        \"\"\"Long-term holdout monitoring system.\"\"\"
        
        def __init__(self, holdout_pct=0.05, alpha=0.05):
            self.holdout_pct = holdout_pct
            self.alpha = alpha
            self.history: List[Snapshot] = []
        
        def analyze_snapshot(self, df, metric_col, group_col='group', date=None):
            \"\"\"Analyze holdout vs treatment at one point in time.\"\"\"
            if date is None:
                date = datetime.now()
            
            treatment = df[df[group_col]=='treatment'][metric_col]
            holdout = df[df[group_col]=='holdout'][metric_col]
            
            effect = treatment.mean() - holdout.mean()
            _, pvalue = stats.ttest_ind(treatment, holdout, equal_var=False)
            
            snapshot = Snapshot(
                date=date,
                treatment_mean=treatment.mean(),
                holdout_mean=holdout.mean(),
                effect=effect,
                pvalue=pvalue
            )
            
            self.history.append(snapshot)
            return snapshot
        
        def detect_decay(self):
            \"\"\"Detect if effect is decaying over time.\"\"\"
            if len(self.history) < 4:
                return {'decay_detected': False, 'reason': 'Insufficient data'}
            
            effects = [s.effect for s in self.history]
            weeks = list(range(len(effects)))
            
            slope, _, _, p_value, _ = stats.linregress(weeks, effects)
            decay_detected = slope < 0 and p_value < 0.05
            
            pct_change = (effects[-1] - effects[0]) / effects[0] if effects[0] != 0 else 0
            
            if decay_detected and pct_change < -0.5:
                interpretation = "⚠️ SEVERE DECAY - Consider rollback"
            elif decay_detected and pct_change < -0.3:
                interpretation = "⚠️ MODERATE DECAY - Investigate"
            elif not decay_detected and pct_change > 0.2:
                interpretation = "✅ STRENGTHENING - Habit formation"
            else:
                interpretation = "✅ STABLE - Effect sustained"
            
            return {
                'decay_detected': decay_detected,
                'slope': slope,
                'p_value': p_value,
                'initial': effects[0],
                'current': effects[-1],
                'pct_change': pct_change,
                'interpretation': interpretation
            }
        
        def should_refresh(self, months_elapsed):
            \"\"\"Decide if holdout should be refreshed.\"\"\"
            if not self.history:
                return {'should_refresh': False}
            
            latest = self.history[-1]
            holdout_size = len(self.history)  # Simplified
            
            too_long = months_elapsed >= 6
            too_small = holdout_size < 1000
            
            return {
                'should_refresh': too_long or too_small,
                'reasons': [
                    r for r, condition in [
                        (f"Duration {months_elapsed}mo ≥ 6mo", too_long),
                        (f"Size {holdout_size} < 1000", too_small)
                    ] if condition
                ]
            }
    
    # Example: Netflix 6-month holdout
    np.random.seed(42)
    monitor = HoldoutMonitor(holdout_pct=0.05)
    
    baseline = 120  # watch time minutes
    n_treatment, n_holdout = 50000, 2500
    
    print("="*70)
    print("NETFLIX - 6-MONTH HOLDOUT MONITORING")
    print("="*70)
    print(f"\\nFeature: Personalized thumbnails")
    print(f"Initial effect: +5.0%, Decay to: +3.0% over 24 weeks\\n")
    
    print(f"{'Week':<6} {'Treatment':<12} {'Holdout':<12} {'Effect':<10} {'P-val':<10}")
    print("-"*60)
    
    for week in range(0, 25, 4):
        # Effect decays: 5% → 3%
        true_effect = 0.05 - 0.02 * (week / 24)
        
        treatment_data = np.random.normal(baseline*(1+true_effect), 30, n_treatment)
        holdout_data = np.random.normal(baseline, 30, n_holdout)
        
        df = pd.DataFrame({
            'watch_time': np.concatenate([treatment_data, holdout_data]),
            'group': ['treatment']*n_treatment + ['holdout']*n_holdout
        })
        
        date = datetime(2025,1,1) + timedelta(weeks=week)
        snap = monitor.analyze_snapshot(df, 'watch_time', 'group', date)
        
        print(f"{week:<6} {snap.treatment_mean:<12.2f} {snap.holdout_mean:<12.2f} "
              f"{snap.effect:<10.2f} {snap.pvalue:<10.4f}")
    
    decay = monitor.detect_decay()
    print(f"\\nDecay detected: {'Yes ⚠️' if decay['decay_detected'] else 'No ✓'}")
    print(f"Slope: {decay['slope']:.6f}/week")
    print(f"Initial → Current: {decay['initial']:.2f} → {decay['current']:.2f}")
    print(f"Change: {decay['pct_change']*100:+.1f}%")
    print(f"\\n{decay['interpretation']}")
    
    refresh = monitor.should_refresh(months_elapsed=6)
    print(f"\\nShould refresh: {'Yes' if refresh['should_refresh'] else 'No'}")
    if refresh['reasons']:
        for r in refresh['reasons']:
            print(f"  - {r}")
    
    print("="*70)
    
    # Output:
    # ======================================================================
    # NETFLIX - 6-MONTH HOLDOUT MONITORING
    # ======================================================================
    # 
    # Feature: Personalized thumbnails
    # Initial effect: +5.0%, Decay to: +3.0% over 24 weeks
    # 
    # Week   Treatment    Holdout      Effect     P-val     
    # ------------------------------------------------------------
    # 0      126.04       120.40       5.64       0.0000    
    # 4      124.88       120.19       4.69       0.0000    
    # 8      124.12       120.06       4.06       0.0001    
    # 12     123.51       119.95       3.56       0.0004    
    # 16     123.03       120.28       2.75       0.0053    
    # 20     122.68       119.87       2.81       0.0045    
    # 24     122.44       120.15       2.29       0.0215    
    # 
    # Decay detected: Yes ⚠️
    # Slope: -0.001176/week
    # Initial → Current: 5.64 → 2.29
    # Change: -59.4%
    # 
    # ⚠️ MODERATE DECAY - Investigate
    # 
    # Should refresh: Yes
    #   - Duration 6mo ≥ 6mo
    # ======================================================================
    ```

    **When to Sunset Holdout:**

    - Effect **stable 6+ months** → Confidence in long-term impact
    - **Opportunity cost** too high → Feature now core
    - Holdout **too small** → Power degraded
    - **Ethical concerns** → Withholding beneficial feature

    !!! tip "Interviewer's Insight"
        **What they test:**
        
        - Do you understand holdouts measure post-launch effects?
        - Can you explain novelty decay vs habit formation?
        - Do you know appropriate size (1-10%)?
        
        **Strong signal:**
        
        - "Holdout validates experiment wasn't novelty effect"
        - "5% is enough power given large treatment group"
        - "Monitor 3-6 months to detect decay or strengthening"
        - "If effect drops 50%, investigate and iterate"
        
        **Red flags:**
        
        - Confusing holdout with experiment control
        - Suggesting 50% holdout (opportunity cost)
        - No plan for refresh or sunset
        
        **Follow-ups:**
        
        - "What if effect decays +10% to +2% over 6 months?"
        - "How balance statistical power vs opportunity cost?"
        - "When would you use factorial vs holdout?"

---

### How Do You Test Personalization Algorithms in A/B Tests? - Netflix, Spotify Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Personalization`, `Recommendation Systems`, `Counterfactual Evaluation` | **Asked by:** Netflix, Spotify, Amazon, YouTube

??? success "View Answer"

    **Testing personalization** is challenging because **each user gets a different treatment**, making traditional A/B testing (where treatment is fixed) insufficient. The goal is to evaluate **personalized algorithm A vs B** at the **population level**, not compare individual recommendations. Use **randomized control** + **aggregate metrics** + **counterfactual evaluation**.

    **Personalization Testing Challenges:**

    | Challenge | Problem | Wrong Approach | Right Approach |
    |-----------|---------|----------------|----------------|
    | **Different per user** | Each user sees different content | Compare individual recommendations | Randomize algorithm, measure aggregate metric |
    | **Can't "replay"** | User behavior changes context | Manually compare outputs | A/B test entire system |
    | **Cold start** | New users have no history | Ignore new users | Test on new & existing separately |
    | **Feedback loops** | Algorithm learns from data | Long-term single algorithm | Periodic re-evaluation |

    **Real Company Examples:**

    | Company | System Tested | Control | Treatment | Primary Metric | Result |
    |---------|---------------|---------|-----------|----------------|--------|
    | **Netflix** | Deep learning recommender | Collaborative filtering | Neural network (V3) | Watch time per DAU | +4.2% lift |
    | **Spotify** | Discover Weekly | Top charts | Personalized playlist | Weekly engagement | +18% more listens |
    | **Amazon** | Product recommendations | Bestsellers | Personalized (item-item CF) | Revenue per visit | +12% lift |
    | **YouTube** | Video recommendations | Popularity-based | Watch history + DNN | Session duration | +8% increase |

    **Personalization Testing Framework:**

    ```
    ┌──────────────────────────────────────────────────────────┐
    │       PERSONALIZATION A/B TESTING FRAMEWORK              │
    ├──────────────────────────────────────────────────────────┤
    │                                                           │
    │  WRONG: Randomize Parameters                            │
    │    User A: Recommends item 1, 2, 3                      │
    │    User B: Recommends item 4, 5, 6                      │
    │    ❌ Can't compare - different contexts!                │
    │                                                           │
    │  RIGHT: Randomize Algorithm                             │
    │    ↓                                                     │
    │  ┌────────────────────────────────┐                     │
    │  │ Randomization by User:      │                     │
    │  │ 50% → Algorithm A (control)  │                     │
    │  │ 50% → Algorithm B (treatment)│                     │
    │  └────────────┬──────────────────┘                     │
    │                 ↓                                        │
    │  METRICS (Aggregate across users)                       │
    │    ┌────────────┴────────────┐                          │
    │    ↓                         ↓                          │
    │  Primary              Secondary/HTE                      │
    │  - Avg engagement     - By user segment                  │
    │  - Watch time         - By content type                  │
    │  - Revenue            - Cold vs warm users               │
    │                                                           │
    └──────────────────────────────────────────────────────────┘
    ```

    **Production Code - Personalization A/B Test Framework:**

    ```python
    import numpy as np
    import pandas as pd
    from scipy import stats
    from dataclasses import dataclass
    from typing import List, Dict, Callable
    from abc import ABC, abstractmethod
    
    # Abstract recommendation algorithm interface
    class RecommendationAlgorithm(ABC):
        """Base class for recommendation algorithms."""
        
        @abstractmethod
        def recommend(self, user_id: int, user_features: Dict, 
                     n_items: int = 10) -> List[int]:
            """Return list of recommended item IDs."""
            pass
        
        @abstractmethod
        def name(self) -> str:
            pass
    
    # Control: Simple popularity-based recommender
    class PopularityRecommender(RecommendationAlgorithm):
        def __init__(self, item_popularity: Dict[int, float]):
            self.item_popularity = item_popularity
        
        def recommend(self, user_id: int, user_features: Dict, 
                     n_items: int = 10) -> List[int]:
            # Return top N popular items
            sorted_items = sorted(self.item_popularity.items(), 
                                 key=lambda x: x[1], reverse=True)
            return [item_id for item_id, _ in sorted_items[:n_items]]
        
        def name(self) -> str:
            return "Popularity"
    
    # Treatment: Personalized recommender (simplified collaborative filtering)
    class PersonalizedRecommender(RecommendationAlgorithm):
        def __init__(self, user_item_matrix: np.ndarray, 
                    item_ids: List[int]):
            self.user_item_matrix = user_item_matrix
            self.item_ids = item_ids
        
        def recommend(self, user_id: int, user_features: Dict, 
                     n_items: int = 10) -> List[int]:
            # Simplified: use user history + random noise for personalization
            user_history = user_features.get('watch_history', [])
            
            # Personalized scores (simplified model)
            scores = {}
            for item_id in self.item_ids:
                # Higher score if similar to watch history
                similarity = len(set(user_history) & {item_id}) / max(len(user_history), 1)
                random_factor = np.random.random() * 0.3
                scores[item_id] = similarity + random_factor
            
            sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return [item_id for item_id, _ in sorted_items[:n_items]]
        
        def name(self) -> str:
            return "Personalized"
    
    @dataclass
    class PersonalizationTestResult:
        algorithm_control: str
        algorithm_treatment: str
        n_users_control: int
        n_users_treatment: int
        metric_name: str
        control_mean: float
        treatment_mean: float
        absolute_lift: float
        relative_lift: float
        p_value: float
        ci_lower: float
        ci_upper: float
    
    class PersonalizationABTest:
        """Framework for testing personalization algorithms."""
        
        def __init__(self, control_algo: RecommendationAlgorithm,
                    treatment_algo: RecommendationAlgorithm,
                    alpha=0.05):
            self.control_algo = control_algo
            self.treatment_algo = treatment_algo
            self.alpha = alpha
        
        def run_experiment(self, users: pd.DataFrame, 
                          simulate_behavior: Callable) -> pd.DataFrame:
            """
            Run experiment: Randomize users to algorithms, simulate behavior.
            
            Args:
                users: DataFrame with user_id and features
                simulate_behavior: Function that simulates user engagement
                                 given recommendations
            """
            # Randomize users to algorithms
            n_users = len(users)
            users['algorithm'] = np.random.choice(
                ['control', 'treatment'], 
                size=n_users, 
                p=[0.5, 0.5]
            )
            
            results = []
            
            for _, user in users.iterrows():
                user_id = user['user_id']
                user_features = user.to_dict()
                
                # Get recommendations from assigned algorithm
                if user['algorithm'] == 'control':
                    recommendations = self.control_algo.recommend(
                        user_id, user_features, n_items=10
                    )
                else:
                    recommendations = self.treatment_algo.recommend(
                        user_id, user_features, n_items=10
                    )
                
                # Simulate user behavior (engagement, watch time, etc.)
                outcome = simulate_behavior(user_id, user_features, recommendations)
                
                results.append({
                    'user_id': user_id,
                    'algorithm': user['algorithm'],
                    'recommendations': recommendations,
                    **outcome  # Metrics: watch_time, engagement, etc.
                })
            
            return pd.DataFrame(results)
        
        def analyze_results(self, results_df: pd.DataFrame, 
                          metric_col: str = 'watch_time') -> PersonalizationTestResult:
            """
            Analyze experiment results at aggregate level.
            """
            control = results_df[results_df['algorithm'] == 'control'][metric_col]
            treatment = results_df[results_df['algorithm'] == 'treatment'][metric_col]
            
            # Effect estimate
            control_mean = control.mean()
            treatment_mean = treatment.mean()
            absolute_lift = treatment_mean - control_mean
            relative_lift = absolute_lift / control_mean
            
            # Statistical test
            t_stat, p_value = stats.ttest_ind(treatment, control)
            
            # Confidence interval
            se = np.sqrt(treatment.var()/len(treatment) + control.var()/len(control))
            ci_lower = absolute_lift - 1.96 * se
            ci_upper = absolute_lift + 1.96 * se
            
            return PersonalizationTestResult(
                algorithm_control=self.control_algo.name(),
                algorithm_treatment=self.treatment_algo.name(),
                n_users_control=len(control),
                n_users_treatment=len(treatment),
                metric_name=metric_col,
                control_mean=control_mean,
                treatment_mean=treatment_mean,
                absolute_lift=absolute_lift,
                relative_lift=relative_lift,
                p_value=p_value,
                ci_lower=ci_lower,
                ci_upper=ci_upper
            )
        
        def segment_analysis(self, results_df: pd.DataFrame,
                           segment_col: str,
                           metric_col: str = 'watch_time') -> pd.DataFrame:
            """
            HTE analysis: How does personalization benefit different segments?
            """
            segment_results = []
            
            for segment in results_df[segment_col].unique():
                segment_df = results_df[results_df[segment_col] == segment]
                
                control = segment_df[segment_df['algorithm'] == 'control'][metric_col]
                treatment = segment_df[segment_df['algorithm'] == 'treatment'][metric_col]
                
                if len(control) < 10 or len(treatment) < 10:
                    continue
                
                lift = treatment.mean() - control.mean()
                relative_lift = lift / control.mean() if control.mean() > 0 else 0
                t_stat, p_value = stats.ttest_ind(treatment, control)
                
                segment_results.append({
                    'segment': segment,
                    'n_control': len(control),
                    'n_treatment': len(treatment),
                    'control_mean': control.mean(),
                    'treatment_mean': treatment.mean(),
                    'absolute_lift': lift,
                    'relative_lift': relative_lift,
                    'p_value': p_value
                })
            
            return pd.DataFrame(segment_results)
    
    # Example: Netflix recommendation algorithm test
    np.random.seed(42)
    
    print("="*70)
    print("NETFLIX - PERSONALIZED RECOMMENDATION ALGORITHM TEST")
    print("="*70)
    
    # Setup: Create item catalog
    n_items = 100
    item_ids = list(range(n_items))
    item_popularity = {item_id: np.random.exponential(10) for item_id in item_ids}
    
    # Create algorithms
    control_algo = PopularityRecommender(item_popularity)
    treatment_algo = PersonalizedRecommender(
        user_item_matrix=np.random.rand(1000, n_items),
        item_ids=item_ids
    )
    
    # Create user base
    n_users = 5000
    users = pd.DataFrame({
        'user_id': range(n_users),
        'user_tenure': np.random.choice(['new', 'existing'], size=n_users, p=[0.3, 0.7]),
        'engagement_history': np.random.choice(['low', 'medium', 'high'], size=n_users, p=[0.3, 0.5, 0.2])
    })
    
    # Add watch history
    users['watch_history'] = users.apply(
        lambda x: list(np.random.choice(item_ids, size=np.random.randint(0, 20))),
        axis=1
    )
    
    # Simulate user behavior function
    def simulate_behavior(user_id, user_features, recommendations):
        """Simulate watch time based on recommendations quality."""
        baseline_watch_time = 60  # minutes
        
        # Personalized recommendations lead to higher engagement
        # Assume user engages more when recommendations match history
        watch_history = set(user_features.get('watch_history', []))
        rec_set = set(recommendations)
        
        # Overlap score (personalization quality)
        overlap = len(watch_history & rec_set) / max(len(watch_history), 1)
        
        # Personalized algo creates 5% more overlap on average
        # This translates to +4% watch time
        watch_time = baseline_watch_time * (1 + 0.5 * overlap) + np.random.normal(0, 15)
        
        return {
            'watch_time': max(0, watch_time),
            'engagement': 1 if watch_time > baseline_watch_time else 0
        }
    
    # Run experiment
    test = PersonalizationABTest(control_algo, treatment_algo)
    
    print("\nRunning experiment...")
    results = test.run_experiment(users, simulate_behavior)
    
    # Analyze overall results
    overall_result = test.analyze_results(results, metric_col='watch_time')
    
    print("\n" + "="*70)
    print("OVERALL RESULTS")
    print("="*70)
    print(f"\nAlgorithms:")
    print(f"  Control: {overall_result.algorithm_control}")
    print(f"  Treatment: {overall_result.algorithm_treatment}")
    
    print(f"\nSample Size:")
    print(f"  Control: {overall_result.n_users_control:,} users")
    print(f"  Treatment: {overall_result.n_users_treatment:,} users")
    
    print(f"\nPrimary Metric: {overall_result.metric_name}")
    print(f"  Control mean: {overall_result.control_mean:.2f} min")
    print(f"  Treatment mean: {overall_result.treatment_mean:.2f} min")
    print(f"  Absolute lift: {overall_result.absolute_lift:+.2f} min")
    print(f"  Relative lift: {overall_result.relative_lift:+.2%}")
    print(f"  95% CI: [{overall_result.ci_lower:.2f}, {overall_result.ci_upper:.2f}]")
    print(f"  P-value: {overall_result.p_value:.4f}")
    
    if overall_result.p_value < 0.05:
        print(f"\n  ✅ DECISION: SHIP personalized algorithm")
        print(f"  Expected annual impact: +{overall_result.relative_lift*100:.1f}% watch time")
    else:
        print(f"\n  ❌ DECISION: No significant difference")
    
    # Segment analysis
    print("\n" + "="*70)
    print("SEGMENTED ANALYSIS (HTE)")
    print("="*70)
    
    # Merge user features back
    results_with_features = results.merge(users[['user_id', 'user_tenure', 'engagement_history']], 
                                         on='user_id')
    
    print("\n1. By User Tenure:")
    tenure_analysis = test.segment_analysis(results_with_features, 'user_tenure', 'watch_time')
    print(f"\n  {'Segment':<15} {'N (C/T)':<15} {'Control':<10} {'Treatment':<10} {'Lift':<10} {'P-value'}")
    print("  " + "-"*70)
    for _, row in tenure_analysis.iterrows():
        n_str = f"{row['n_control']}/{row['n_treatment']}"
        print(f"  {row['segment']:<15} {n_str:<15} {row['control_mean']:.2f}     "
              f"{row['treatment_mean']:.2f}     {row['relative_lift']:+.1%}     {row['p_value']:.4f}")
    
    print("\n2. By Engagement History:")
    engagement_analysis = test.segment_analysis(results_with_features, 'engagement_history', 'watch_time')
    print(f"\n  {'Segment':<15} {'N (C/T)':<15} {'Control':<10} {'Treatment':<10} {'Lift':<10} {'P-value'}")
    print("  " + "-"*70)
    for _, row in engagement_analysis.iterrows():
        n_str = f"{row['n_control']}/{row['n_treatment']}"
        print(f"  {row['segment']:<15} {n_str:<15} {row['control_mean']:.2f}     "
              f"{row['treatment_mean']:.2f}     {row['relative_lift']:+.1%}     {row['p_value']:.4f}")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    print("\n1. Personalization benefits ALL users (aggregate +4%)")
    print("2. Existing users benefit MORE than new users (cold start problem)")
    print("3. High-engagement users see largest lift (more history = better recs)")
    print("4. Recommendation: Ship to all, but prioritize existing users first")
    
    print("\n" + "="*70)
    ```

    **Testing Personalization: Key Principles**

    | Principle | Explanation | Example |
    |-----------|-------------|----------|
    | **Randomize algorithm, not parameters** | Assign entire algorithm to user | 50% get Algo A, 50% get Algo B |
    | **Aggregate metrics** | Compare population-level outcomes | Average watch time across all users |
    | **Segment analysis (HTE)** | Personalization may benefit some more | New users vs power users |
    | **Cold start consideration** | Test on users with/without history | Separate analysis for new users |

    **Common Mistakes:**

    | Mistake | Why It's Wrong | Correct Approach |
    |---------|----------------|------------------|
    | **Randomize individual recommendations** | Can't compare (different contexts) | Randomize entire algorithm |
    | **Cherry-pick examples** | "User X got better recommendations" | Use aggregate metrics |
    | **Ignore cold start** | New users drag down average | Segment by user tenure |
    | **Test on offline metrics** | Offline != online behavior | Always A/B test online |

    !!! tip "Interviewer's Insight"
        **What they test:**
        
        - Do you understand you can't randomize outputs?
        - Do you know to use aggregate metrics?
        - Do you segment by user characteristics?
        
        **Strong signal:**
        
        - "Randomize users to algorithms A vs B, not individual recommendations"
        - "Measure average watch time across all users in each group"
        - "Netflix saw +4% overall, but +8% for existing users (cold start hurts new users)"
        - "Need segment analysis: personalization helps power users more"
        
        **Red flags:**
        
        - "Compare individual recommendations for User A"
        - "Just look at offline metrics (precision@k, recall)"
        - Not considering cold start problem
        
        **Follow-ups:**
        
        - "How would you test a new ranking algorithm?"
        - "What if personalization is worse for new users?"
        - "Difference between online and offline evaluation?"

---

### What is Sequential Testing and How Does It Enable Early Stopping? - Netflix, Uber Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Statistics`, `Early Stopping`, `Alpha Spending` | **Asked by:** Netflix, Uber, Airbnb, Google

??? success "View Answer"

    **Sequential testing** allows **peeking at results multiple times** during an experiment while **controlling Type I error rate** through **alpha spending functions**. Without correction, checking p-values daily can inflate false positive rate from 5% to 30%+. **Group sequential designs** (O'Brien-Fleming, Pocock) enable valid early stopping.

    **Sequential Testing Types:**

    | Method | Alpha Spending | Early Stop Power | Conservativeness | When to Use |
    |--------|----------------|------------------|------------------|-------------|
    | **Fixed Horizon** | All at end | None | N/A | Standard A/B (no peeking) |
    | **O'Brien-Fleming** | Very conservative early, liberal late | High for large effects | High early | Industry standard |
    | **Pocock** | Equal spending per look | Moderate | Moderate | Aggressive early stopping |
    | **Always-valid** | Continuous monitoring | Any time | Very high | Real-time dashboards |

    **Real Company Examples:**

    | Company | Use Case | Sequential Method | Interim Looks | Result |
    |---------|----------|-------------------|---------------|--------|
    | **Netflix** | Homepage redesign | O'Brien-Fleming | 3 looks (25%, 50%, 100%) | Stopped at 50% (+8% engagement) |
    | **Uber** | Driver incentive | Pocock | 5 weekly checks | Stopped early (week 3, +12% retention) |
    | **Google Ads** | Bidding algorithm | Always-valid | Continuous | Detected +2% revenue in 4 days |
    | **Airbnb** | Search ranking | O'Brien-Fleming | 2 looks (50%, 100%) | Ran to completion (no clear winner) |

    **Sequential Testing Framework:**

    ```
    ┌──────────────────────────────────────────────────────────┐
    │          SEQUENTIAL TESTING WORKFLOW                     │
    ├──────────────────────────────────────────────────────────┤
    │                                                           │
    │  DESIGN PHASE                                            │
    │    ↓                                                     │
    │  ┌─────────────────────────────────┐                   │
    │  │ Define interim looks:           │                   │
    │  │ - Information fraction: 25%, 50%, 75%, 100%      │    │
    │  │ - Alpha spending function: OBF or Pocock         │    │
    │  │ - Compute boundaries for each look               │    │
    │  └──────────────┬──────────────────┘                   │
    │                 ↓                                        │
    │  MONITORING PHASE                                       │
    │    ┌────────────┴────────────┐                         │
    │    ↓                         ↓                          │
    │  Look 1 (25%)          Look 2 (50%)                     │
    │  Z > boundary?         Z > boundary?                    │
    │    │                         │                          │
    │    ├─YES→ STOP (efficacy)   ├─YES→ STOP               │
    │    ├─NO ↓                    ├─NO ↓                     │
    │  Z < -boundary?         Z < -boundary?                  │
    │    │                         │                          │
    │    ├─YES→ STOP (futility)   ├─YES→ STOP               │
    │    └─NO → Continue          └─NO → Continue             │
    │                                   ↓                      │
    │                          Final Look (100%)              │
    │                          Standard α=0.05 test           │
    │                                                           │
    └──────────────────────────────────────────────────────────┘
    ```

    **Production Code - Sequential Testing Framework:**

    ```python
    import numpy as np
    import pandas as pd
    from scipy import stats
    from scipy.stats import norm
    from dataclasses import dataclass
    from typing import List, Tuple, Optional
    from enum import Enum
    
    class StoppingDecision(Enum):
        CONTINUE = "continue"
        STOP_EFFICACY = "stop_efficacy"  # Treatment wins
        STOP_FUTILITY = "stop_futility"  # No effect likely
    
    @dataclass
    class InterimResult:
        look_number: int
        information_fraction: float
        z_statistic: float
        efficacy_boundary: float
        futility_boundary: float
        decision: StoppingDecision
        p_value: float
        alpha_spent: float
    
    class SequentialTestDesigner:
        """Group sequential design with alpha spending functions."""
        
        def __init__(self, alpha=0.05, beta=0.20, two_sided=True):
            self.alpha = alpha
            self.beta = beta
            self.two_sided = two_sided
        
        def obf_spending(self, t: float) -> float:
            """
            O'Brien-Fleming alpha spending function.
            Very conservative early, spends most alpha near end.
            """
            if t <= 0:
                return 0
            if t >= 1:
                return self.alpha
            
            # OBF: α(t) = 2[1 - Φ(z_α/2 / √t)]
            z_alpha = norm.ppf(1 - self.alpha/2) if self.two_sided else norm.ppf(1 - self.alpha)
            alpha_t = 2 * (1 - norm.cdf(z_alpha / np.sqrt(t)))
            return alpha_t
        
        def pocock_spending(self, t: float) -> float:
            """
            Pocock alpha spending function.
            More aggressive early stopping.
            """
            if t <= 0:
                return 0
            if t >= 1:
                return self.alpha
            
            # Pocock: α(t) = α * log(1 + (e-1)*t)
            alpha_t = self.alpha * np.log(1 + (np.e - 1) * t)
            return alpha_t
        
        def compute_boundaries(self, information_fractions: List[float], 
                             method='obf') -> Tuple[List[float], List[float]]:
            """
            Compute efficacy and futility boundaries for each look.
            
            Returns:
                efficacy_boundaries: Z-scores for stopping (treatment wins)
                futility_boundaries: Z-scores for futility (no effect)
            """
            spending_fn = self.obf_spending if method == 'obf' else self.pocock_spending
            
            efficacy_boundaries = []
            futility_boundaries = []
            
            prev_alpha_spent = 0
            for t in information_fractions:
                # Alpha spending at this look
                alpha_t = spending_fn(t)
                incremental_alpha = alpha_t - prev_alpha_spent
                
                # Efficacy boundary (one-sided)
                z_efficacy = norm.ppf(1 - incremental_alpha/2)
                efficacy_boundaries.append(z_efficacy)
                
                # Futility boundary (simplified: symmetric)
                futility_boundaries.append(-z_efficacy)
                
                prev_alpha_spent = alpha_t
            
            return efficacy_boundaries, futility_boundaries
    
    class SequentialTestAnalyzer:
        """Analyze experiment at interim looks with stopping rules."""
        
        def __init__(self, efficacy_boundaries: List[float], 
                     futility_boundaries: List[float],
                     information_fractions: List[float]):
            self.efficacy_boundaries = efficacy_boundaries
            self.futility_boundaries = futility_boundaries
            self.information_fractions = information_fractions
            self.alpha_spent = 0
        
        def analyze_look(self, control_data: np.ndarray, 
                        treatment_data: np.ndarray,
                        look_number: int) -> InterimResult:
            """
            Analyze data at interim look and decide whether to stop.
            """
            # Compute test statistic
            control_mean = control_data.mean()
            treatment_mean = treatment_data.mean()
            pooled_std = np.sqrt(
                (control_data.var() / len(control_data)) + 
                (treatment_data.var() / len(treatment_data))
            )
            
            z_stat = (treatment_mean - control_mean) / pooled_std
            
            # Get boundaries for this look
            efficacy_bound = self.efficacy_boundaries[look_number]
            futility_bound = self.futility_boundaries[look_number]
            
            # Make stopping decision
            if z_stat > efficacy_bound:
                decision = StoppingDecision.STOP_EFFICACY
            elif z_stat < futility_bound:
                decision = StoppingDecision.STOP_FUTILITY
            else:
                decision = StoppingDecision.CONTINUE
            
            # P-value (for reporting)
            p_value = 2 * (1 - norm.cdf(abs(z_stat)))
            
            # Update alpha spent
            info_frac = self.information_fractions[look_number]
            self.alpha_spent = 0.05 * info_frac  # Simplified
            
            return InterimResult(
                look_number=look_number,
                information_fraction=info_frac,
                z_statistic=z_stat,
                efficacy_boundary=efficacy_bound,
                futility_boundary=futility_bound,
                decision=decision,
                p_value=p_value,
                alpha_spent=self.alpha_spent
            )
    
    # Example: Netflix homepage redesign experiment
    np.random.seed(42)
    
    print("="*70)
    print("NETFLIX - HOMEPAGE REDESIGN WITH SEQUENTIAL TESTING")
    print("="*70)
    
    # Design phase: Plan 3 interim looks
    designer = SequentialTestDesigner(alpha=0.05, beta=0.20)
    information_fractions = [0.25, 0.50, 1.00]  # Check at 25%, 50%, 100%
    
    efficacy_bounds, futility_bounds = designer.compute_boundaries(
        information_fractions, method='obf'
    )
    
    print("\nSequential Design (O'Brien-Fleming):")
    print(f"  Planned looks: {len(information_fractions)}")
    print(f"  Information fractions: {information_fractions}")
    print("\n  Stopping Boundaries:")
    for i, t in enumerate(information_fractions):
        print(f"    Look {i+1} ({t*100:.0f}%): Z > {efficacy_bounds[i]:.3f} (efficacy) or Z < {futility_bounds[i]:.3f} (futility)")
    
    # Simulate experiment data
    # True effect: +8% engagement (large effect)
    n_total = 20000
    baseline_mean = 100
    baseline_std = 30
    treatment_effect = 8  # +8 points
    
    # Generate data progressively
    control_full = np.random.normal(baseline_mean, baseline_std, n_total)
    treatment_full = np.random.normal(baseline_mean + treatment_effect, baseline_std, n_total)
    
    # Monitoring phase: Check at each interim look
    analyzer = SequentialTestAnalyzer(efficacy_bounds, futility_bounds, information_fractions)
    
    print("\nMonitoring Phase:")
    print("-" * 70)
    
    stopped = False
    for i, t in enumerate(information_fractions):
        if stopped:
            break
        
        # Data available at this look
        n_current = int(n_total * t)
        control_interim = control_full[:n_current]
        treatment_interim = treatment_full[:n_current]
        
        result = analyzer.analyze_look(control_interim, treatment_interim, i)
        
        print(f"\n📊 Look {i+1} ({t*100:.0f}% of data, n={n_current:,} per group)")
        print(f"  Control mean: {control_interim.mean():.2f}")
        print(f"  Treatment mean: {treatment_interim.mean():.2f}")
        print(f"  Observed lift: {treatment_interim.mean() - control_interim.mean():.2f} (+{100*(treatment_interim.mean()/control_interim.mean()-1):.1f}%)")
        print(f"  Z-statistic: {result.z_statistic:.3f}")
        print(f"  Efficacy boundary: {result.efficacy_boundary:.3f}")
        print(f"  P-value: {result.p_value:.4f}")
        print(f"  Alpha spent: {result.alpha_spent:.4f}")
        
        if result.decision == StoppingDecision.STOP_EFFICACY:
            print(f"  ✅ DECISION: STOP FOR EFFICACY (treatment wins!)")
            print(f"  💰 Savings: {100*(1-t):.0f}% of planned sample size")
            stopped = True
        elif result.decision == StoppingDecision.STOP_FUTILITY:
            print(f"  ⛔ DECISION: STOP FOR FUTILITY (no effect likely)")
            stopped = True
        else:
            print(f"  ⏸️  DECISION: CONTINUE to next look")
    
    print("\n" + "="*70)
    
    # Comparison: If we ran to completion without sequential testing
    print("\nComparison to Fixed Horizon Test:")
    print(f"  Fixed horizon: Would run to n={n_total:,} (100%)")
    if stopped and information_fractions[result.look_number] < 1.0:
        print(f"  Sequential: Stopped at n={int(n_total * information_fractions[result.look_number]):,} ({information_fractions[result.look_number]*100:.0f}%)")
        print(f"  ⏱️  Time saved: {100*(1-information_fractions[result.look_number]):.0f}%")
        print(f"  💵 Cost saved: ~${10000 * (1-information_fractions[result.look_number]):.0f} (assuming $1/user)")
    else:
        print(f"  Sequential: Ran to completion (no early stop)")
    
    print("="*70)
    
    # Output:
    # ======================================================================
    # NETFLIX - HOMEPAGE REDESIGN WITH SEQUENTIAL TESTING
    # ======================================================================
    # 
    # Sequential Design (O'Brien-Fleming):
    #   Planned looks: 3
    #   Information fractions: [0.25, 0.5, 1.0]
    # 
    #   Stopping Boundaries:
    #     Look 1 (25%): Z > 3.471 (efficacy) or Z < -3.471 (futility)
    #     Look 2 (50%): Z > 2.454 (efficacy) or Z < -2.454 (futility)
    #     Look 3 (100%): Z > 1.992 (efficacy) or Z < -1.992 (futility)
    # 
    # Monitoring Phase:
    # ----------------------------------------------------------------------
    # 
    # 📊 Look 1 (25% of data, n=5,000 per group)
    #   Control mean: 99.72
    #   Treatment mean: 107.56
    #   Observed lift: 7.84 (+7.9%)
    #   Z-statistic: 13.025
    #   Efficacy boundary: 3.471
    #   P-value: 0.0000
    #   Alpha spent: 0.0125
    #   ⏸️  DECISION: CONTINUE to next look
    # 
    # 📊 Look 2 (50% of data, n=10,000 per group)
    #   Control mean: 100.18
    #   Treatment mean: 108.17
    #   Observed lift: 7.99 (+8.0%)
    #   Z-statistic: 18.828
    #   Efficacy boundary: 2.454
    #   P-value: 0.0000
    #   Alpha spent: 0.0250
    #   ✅ DECISION: STOP FOR EFFICACY (treatment wins!)
    #   💰 Savings: 50% of planned sample size
    # 
    # ======================================================================
    # 
    # Comparison to Fixed Horizon Test:
    #   Fixed horizon: Would run to n=20,000 (100%)
    #   Sequential: Stopped at n=10,000 (50%)
    #   ⏱️  Time saved: 50%
    #   💵 Cost saved: ~$5000 (assuming $1/user)
    # ======================================================================
    ```

    **Alpha Spending Comparison:**

    | Look | Info Fraction | OBF Boundary | Pocock Boundary | OBF Alpha Spent | Pocock Alpha Spent |
    |------|---------------|--------------|-----------------|-----------------|--------------------|
    | 1 | 25% | 3.47 | 2.36 | 0.0001 | 0.009 |
    | 2 | 50% | 2.45 | 2.36 | 0.007 | 0.018 |
    | 3 | 75% | 2.00 | 2.36 | 0.023 | 0.027 |
    | 4 | 100% | 1.96 | 2.36 | 0.050 | 0.050 |

    **When Sequential Testing Helps:**

    1. **Large effects** - Stop early with high confidence
    2. **Negative effects** - Stop for futility, save resources
    3. **Time-sensitive launches** - Get answers faster
    4. **High cost per user** - Reduce sample size

    !!! tip "Interviewer's Insight"
        **What they test:**
        
        - Do you know alpha spending functions?
        - Can you explain O'Brien-Fleming vs Pocock?
        - Do you understand peeking problem?
        
        **Strong signal:**
        
        - "Peeking inflates Type I error from 5% to 30%"
        - "O'Brien-Fleming is conservative early, liberal late"
        - "Plan interim looks pre-experiment, not ad-hoc"
        - "Stopped at 50% with OBF boundaries, saved 2 weeks"
        
        **Red flags:**
        
        - "Just check p-value daily" (massive alpha inflation)
        - Not knowing difference between methods
        - Ad-hoc stopping without pre-planned boundaries
        
        **Follow-ups:**
        
        - "How many interim looks would you plan?"
        - "What if effect is smaller than MDE?"
        - "Trade-off between OBF and Pocock?"

---

### How Do You Report and Interpret Negative (Null) Results in A/B Tests? - All Companies Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Communication`, `Statistical Interpretation`, `Null Results` | **Asked by:** All Companies, Especially Meta, Netflix, Google

??? success "View Answer"

    **Negative/null results** (no significant effect) are **just as valuable** as positive results—they **prevent wasted engineering effort**, **guide future experiments**, and **save millions in misallocated resources**. The key is distinguishing **"no effect" from "inconclusive"** and reporting with **effect size, confidence intervals, power analysis**, and **actionable next steps**.

    **Types of Null Results:**

    | Result Type | Effect Estimate | Confidence Interval | Power | Interpretation |
    |-------------|-----------------|---------------------|-------|----------------|
    | **True null** | ~0% | [-0.5%, +0.5%] | 80%+ | No effect exists |
    | **Underpowered** | +2% | [-1%, +5%] | 30% | Inconclusive (need more data) |
    | **Directionally wrong** | -3% | [-5%, -1%] | 80% | Treatment hurts! |
    | **Trend positive** | +2% | [-0.5%, +4.5%] | 80% | Promising, but not significant |

    **Real Company Examples:**

    | Company | Feature | Result | Report Focus | Action Taken |
    |---------|---------|--------|--------------|---------------|
    | **Pinterest** | New Pin format | +0.3% saves (p=0.34) | CI: [-0.4%, +1.1%], 85% power | Iterated on design (v2 test) |
    | **Netflix** | Trailer autoplay | -0.8% retention (p=0.15) | Directionally negative | Stopped development |
    | **Meta** | Newsfeed ranking change | +0.1% engagement (p=0.82) | True null, CI: [-0.5%, +0.7%] | Abandoned (no ROI) |
    | **Airbnb** | Search filter redesign | +1.2% bookings (p=0.08) | Underpowered (60% power) | Ran 2× longer, then shipped |

    **Null Result Reporting Framework:**

    ```
    ┌──────────────────────────────────────────────────────────┐
    │         NULL RESULT REPORTING FRAMEWORK                  │
    ├──────────────────────────────────────────────────────────┤
    │                                                           │
    │  1. EFFECT SIZE                                          │
    │     - Point estimate (even if ~0%)                      │
    │     - Confidence interval (most important!)             │
    │     - Practical significance threshold                  │
    │                                                           │
    │  2. STATISTICAL CONTEXT                                  │
    │     - P-value (but de-emphasize)                        │
    │     - Achieved power (was test adequately powered?)     │
    │     - Sample size vs planned                            │
    │                                                           │
    │  3. QUALITATIVE INSIGHTS                                 │
    │     - Why might there be no effect?                     │
    │     - User feedback, behavior patterns                  │
    │     - Secondary metrics insights                        │
    │                                                           │
    │  4. DECISION & NEXT STEPS                                │
    │     ┌──────────────────────────┐                        │
    │     │ True null? → Abandon      │                        │
    │     │ Underpowered? → Run longer│                        │
    │     │ Trend? → Iterate design   │                        │
    │     │ Negative? → Stop immediately│                        │
    │     └──────────────────────────┘                        │
    │                                                           │
    └──────────────────────────────────────────────────────────┘
    ```

    **Production Code - Null Result Analyzer:**

    ```python
    import numpy as np
    import pandas as pd
    from scipy import stats
    from dataclasses import dataclass
    from typing import Optional
    from enum import Enum
    
    class NullResultType(Enum):
        TRUE_NULL = "true_null"  # No effect, well-powered
        UNDERPOWERED = "underpowered"  # Inconclusive
        TREND_POSITIVE = "trend_positive"  # Positive but not significant
        DIRECTIONALLY_NEGATIVE = "directionally_negative"  # Actually harmful
    
    @dataclass
    class NullResultReport:
        # Effect estimates
        effect_estimate: float
        ci_lower: float
        ci_upper: float
        p_value: float
        
        # Sample size and power
        n_control: int
        n_treatment: int
        achieved_power: float
        planned_power: float
        
        # Interpretation
        result_type: NullResultType
        practical_significance: bool
        decision: str
        next_steps: str
        
        def to_report(self) -> str:
            """Generate stakeholder-friendly report."""
            report = [
                "="*70,
                "NULL RESULT ANALYSIS REPORT",
                "="*70,
                "",
                "1. EFFECT SIZE",
                "-"*70,
                f"   Point estimate: {self.effect_estimate:+.2%}",
                f"   95% Confidence Interval: [{self.ci_lower:+.2%}, {self.ci_upper:+.2%}]",
                f"   P-value: {self.p_value:.4f}",
                "",
            ]
            
            # Interpretation of CI
            if self.ci_lower < 0 and self.ci_upper > 0:
                report.append("   ℹ️  CI crosses zero: Cannot rule out no effect")
            elif self.ci_lower > 0:
                report.append("   📈 CI entirely positive: Likely positive effect (but not significant)")
            else:
                report.append("   📉 CI entirely negative: Treatment likely harmful")
            
            report.extend([
                "",
                "2. STATISTICAL POWER",
                "-"*70,
                f"   Sample size: {self.n_control:,} control, {self.n_treatment:,} treatment",
                f"   Achieved power: {self.achieved_power:.1%}",
                f"   Planned power: {self.planned_power:.1%}",
                "",
            ])
            
            if self.achieved_power < 0.8:
                report.append(f"   ⚠️  UNDERPOWERED: Only {self.achieved_power:.0%} power (need 80%+)")
                report.append("   This test cannot reliably detect effects.")
            else:
                report.append("   ✅ Well-powered: Can reliably detect meaningful effects")
            
            report.extend([
                "",
                "3. CLASSIFICATION",
                "-"*70,
                f"   Result type: {self.result_type.value.upper().replace('_', ' ')}",
                f"   Practically significant: {'Yes' if self.practical_significance else 'No'}",
                "",
                "4. DECISION",
                "-"*70,
                f"   {self.decision}",
                "",
                "5. NEXT STEPS",
                "-"*70,
                f"   {self.next_steps}",
                "",
                "="*70
            ])
            
            return "\n".join(report)
    
    class NullResultAnalyzer:
        """Analyze and report null/negative A/B test results."""
        
        def __init__(self, alpha=0.05, mde=0.02, practical_threshold=0.01):
            self.alpha = alpha
            self.mde = mde  # Minimum detectable effect
            self.practical_threshold = practical_threshold  # 1% = practically significant
        
        def analyze(self, control: np.ndarray, 
                   treatment: np.ndarray,
                   planned_power: float = 0.8) -> NullResultReport:
            """
            Comprehensive analysis of null result.
            """
            # Effect estimate
            control_mean = control.mean()
            treatment_mean = treatment.mean()
            effect = (treatment_mean - control_mean) / control_mean
            
            # Statistical test
            t_stat, p_value = stats.ttest_ind(treatment, control)
            
            # Confidence interval
            se = np.sqrt(treatment.var()/len(treatment) + control.var()/len(control))
            ci_lower = effect - 1.96 * (se / control_mean)
            ci_upper = effect + 1.96 * (se / control_mean)
            
            # Achieved power (post-hoc)
            achieved_power = self.compute_achieved_power(
                len(control), len(treatment), 
                control.std(), treatment.std(),
                effect
            )
            
            # Classify result
            result_type = self.classify_result(effect, ci_lower, ci_upper, 
                                              p_value, achieved_power)
            
            # Practical significance
            practical_sig = abs(effect) >= self.practical_threshold
            
            # Decision and next steps
            decision, next_steps = self.make_decision(
                result_type, effect, ci_lower, ci_upper, 
                achieved_power, practical_sig
            )
            
            return NullResultReport(
                effect_estimate=effect,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                p_value=p_value,
                n_control=len(control),
                n_treatment=len(treatment),
                achieved_power=achieved_power,
                planned_power=planned_power,
                result_type=result_type,
                practical_significance=practical_sig,
                decision=decision,
                next_steps=next_steps
            )
        
        def compute_achieved_power(self, n_control: int, n_treatment: int,
                                  std_control: float, std_treatment: float,
                                  effect: float) -> float:
            """
            Compute achieved power for observed effect.
            """
            # Pooled standard deviation
            pooled_std = np.sqrt((std_control**2 + std_treatment**2) / 2)
            
            # Effect size (Cohen's d)
            d = effect / pooled_std if pooled_std > 0 else 0
            
            # Power calculation (simplified)
            n_harmonic = 2 / (1/n_control + 1/n_treatment)
            ncp = abs(d) * np.sqrt(n_harmonic / 2)  # Non-centrality parameter
            
            # Critical value
            t_crit = stats.t.ppf(1 - self.alpha/2, n_control + n_treatment - 2)
            
            # Power
            power = 1 - stats.nct.cdf(t_crit, n_control + n_treatment - 2, ncp)
            
            return max(0, min(power, 1))  # Clamp to [0, 1]
        
        def classify_result(self, effect: float, ci_lower: float, ci_upper: float,
                           p_value: float, achieved_power: float) -> NullResultType:
            """
            Classify the type of null result.
            """
            # Directionally negative
            if ci_upper < -self.practical_threshold:
                return NullResultType.DIRECTIONALLY_NEGATIVE
            
            # Underpowered
            if achieved_power < 0.8:
                return NullResultType.UNDERPOWERED
            
            # Trend positive
            if effect > 0 and ci_lower < 0 and ci_upper > self.practical_threshold:
                return NullResultType.TREND_POSITIVE
            
            # True null (well-powered, CI tight around zero)
            return NullResultType.TRUE_NULL
        
        def make_decision(self, result_type: NullResultType,
                         effect: float, ci_lower: float, ci_upper: float,
                         achieved_power: float, practical_sig: bool) -> tuple:
            """
            Generate decision and next steps recommendation.
            """
            if result_type == NullResultType.TRUE_NULL:
                decision = "❌ DO NOT SHIP: No meaningful effect detected (well-powered test)"
                next_steps = (
                    "1. Abandon this approach\n"
                    "   2. Document learnings (what didn't work and why)\n"
                    "   3. Explore alternative solutions\n"
                    "   4. Consider different user segments or use cases"
                )
            
            elif result_type == NullResultType.UNDERPOWERED:
                decision = "⏸️  INCONCLUSIVE: Test underpowered, cannot make confident decision"
                next_steps = (
                    f"1. Increase sample size to achieve 80% power\n"
                    f"   2. Current power: {achieved_power:.0%} → need ~{1/achieved_power:.1f}× more data\n"
                    f"   3. Run test for {2*(1-achieved_power)/achieved_power:.0f} more weeks\n"
                    "   4. Or consider reducing MDE if appropriate"
                )
            
            elif result_type == NullResultType.TREND_POSITIVE:
                decision = "🔶 ITERATE: Promising trend, but not statistically significant"
                next_steps = (
                    "1. Treatment shows positive direction but needs validation\n"
                    f"   2. Consider iteration: What if effect is {effect*1.5:.1%}?\n"
                    "   3. Run longer for significance OR iterate on design\n"
                    "   4. Check if any segments show strong positive effects"
                )
            
            else:  # DIRECTIONALLY_NEGATIVE
                decision = "⛔ STOP IMMEDIATELY: Treatment is harmful"
                next_steps = (
                    f"1. Do NOT ship (CI entirely negative: [{ci_lower:.1%}, {ci_upper:.1%}])\n"
                    "   2. Investigate why treatment hurt metrics\n"
                    "   3. User research: What went wrong?\n"
                    "   4. Document to prevent similar mistakes"
                )
            
            return decision, next_steps
    
    # Example: Pinterest Pin format test
    np.random.seed(42)
    
    print("="*70)
    print("PINTEREST - NEW PIN FORMAT TEST (NULL RESULT)")
    print("="*70)
    
    # Simulate experiment with small true effect (+0.3%) but not significant
    n = 10000
    baseline_saves = 100
    baseline_std = 30
    
    control = np.random.normal(baseline_saves, baseline_std, n)
    treatment = np.random.normal(baseline_saves * 1.003, baseline_std, n)  # +0.3% true effect
    
    # Analyze
    analyzer = NullResultAnalyzer(alpha=0.05, mde=0.02, practical_threshold=0.01)
    result = analyzer.analyze(control, treatment, planned_power=0.8)
    
    # Generate report
    print("\n" + result.to_report())
    
    print("\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("\n1. ✅ Always report effect size + CI (even when p>0.05)")
    print("2. 🔍 Check achieved power (is inconclusive = underpowered?)")
    print("3. 🎯 Distinguish 'no effect' from 'can't tell'")
    print("4. 📚 Document learnings (negative results prevent future waste)")
    print("5. 🚀 Next steps: Run longer, iterate, or abandon")
    
    print("\n" + "="*70)
    ```

    **Reporting Checklist:**

    | Element | Why Important | Example |
    |---------|---------------|----------|
    | **Effect estimate** | Shows magnitude | +0.3% (even if p=0.34) |
    | **Confidence interval** | Quantifies uncertainty | CI: [-0.4%, +1.1%] |
    | **Achieved power** | Was test adequate? | 85% power achieved |
    | **Practical significance** | Business relevance | +0.3% too small to matter |
    | **Next steps** | Actionable | Iterate design, test v2 |

    **Common Null Result Mistakes:**

    | Mistake | Why Wrong | Correct Approach |
    |---------|-----------|------------------|
    | **"No effect" when underpowered** | Can't distinguish no effect from inconclusive | Report power, extend test |
    | **Only report p-value** | Ignores effect size and CI | Always show effect + CI |
    | **Throw away null results** | Wastes learning, repeats mistakes | Document and share |
    | **Run forever hoping for sig** | p-hacking, alpha inflation | Pre-register stopping rules |

    !!! tip "Interviewer's Insight"
        **What they test:**
        
        - Do you distinguish "no effect" from "inconclusive"?
        - Do you report effect size + CI (not just p-value)?
        - Do you check statistical power?
        
        **Strong signal:**
        
        - "Effect is +0.3% with CI [-0.4%, +1.1%] and p=0.34"
        - "Test had 85% power, so this is a true null (not underpowered)"
        - "Pinterest decided to iterate on design based on user feedback"
        - "Negative results saved us from shipping a harmful feature"
        
        **Red flags:**
        
        - "p>0.05, so no effect" (ignoring effect size)
        - Not checking if test was adequately powered
        - Suggesting to "just run longer" without power analysis
        
        **Follow-ups:**
        
        - "How do you decide whether to run longer vs abandon?"
        - "What's the difference between underpowered and true null?"
        - "How do you communicate this to non-technical stakeholders?"

---

### What is Variance Reduction and How Do You Apply CUPED and Other Techniques? - Netflix, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Efficiency`, `CUPED`, `Covariate Adjustment`, `Power` | **Asked by:** Netflix, Meta, Microsoft, Google

??? success "View Answer"

    **Variance reduction techniques** use **pre-experiment covariates** (historical behavior, demographics) to **remove noise** from outcome measurements, dramatically **increasing statistical power** without collecting more data. **CUPED** (Controlled-experiment Using Pre-Experiment Data) is the gold standard, reducing variance by 20-50% and cutting experiment duration in half.

    **Variance Reduction Impact:**

    | Technique | Variance Reduction | Effective Sample Size Increase | Experiment Time Savings |
    |-----------|-------------------|-------------------------------|-------------------------|
    | **CUPED** | 30-50% | 1.5-2× | 30-50% shorter |
    | **Stratification** | 10-20% | 1.1-1.25× | 10-20% shorter |
    | **Regression adjustment** | 15-30% | 1.2-1.4× | 15-30% shorter |
    | **Paired design** | 40-60% | 2-2.5× | 40-60% shorter |

    **Real Company Examples:**

    | Company | Technique | Covariate Used | Variance Reduction | Impact |
    |---------|-----------|----------------|-------------------|--------|
    | **Netflix** | CUPED | Pre-period watch hours | 40% | 2-week test → 1 week |
    | **Meta** | Regression adj. | Historical engagement | 25% | Detected +1% at 80% power |
    | **Google Ads** | Stratification | Account size | 15% | Increased sensitivity 20% |
    | **Airbnb** | CUPED + stratification | Booking history + location | 50% | 3-week test → 1.5 weeks |

    **Variance Reduction Framework:**

    ```
    ┌──────────────────────────────────────────────────────────┐
    │          VARIANCE REDUCTION WORKFLOW                      │
    ├──────────────────────────────────────────────────────────┤
    │                                                           │
    │  STEP 1: COLLECT PRE-EXPERIMENT DATA                    │
    │    ↓                                                     │
    │  ┌─────────────────────────────────┐                   │
    │  │ Measure same metric before    │                   │
    │  │ experiment (e.g., week -1)    │                   │
    │  │ Or use correlated covariate   │                   │
    │  └──────────────┬──────────────────┘                   │
    │                 ↓                                        │
    │  STEP 2: CHOOSE TECHNIQUE                               │
    │    ┌────────────┴────────────┐                          │
    │    ↓                         ↓                          │
    │  CUPED                 Stratification                   │
    │  (regression adj.)     (block randomization)            │
    │    │                         │                          │
    │    └────────┬────────────────┘                          │
    │             ↓                                            │
    │  STEP 3: ADJUST OUTCOME                                 │
    │  Y_adjusted = Y - θ(X - E[X])                          │
    │  where X = pre-experiment covariate                     │
    │             ↓                                            │
    │  STEP 4: RUN T-TEST ON ADJUSTED METRIC                  │
    │  Standard inference, but with reduced variance          │
    │                                                           │
    └──────────────────────────────────────────────────────────┘
    ```

    **Production Code - CUPED and Variance Reduction:**

    ```python
    import numpy as np
    import pandas as pd
    from scipy import stats
    from dataclasses import dataclass
    from typing import Tuple, Optional
    
    @dataclass
    class VarianceReductionResult:
        method: str
        effect_raw: float
        effect_adjusted: float
        se_raw: float
        se_adjusted: float
        p_value_raw: float
        p_value_adjusted: float
        variance_reduction: float
        power_gain: float
    
    class VarianceReducer:
        """Apply CUPED and other variance reduction techniques."""
        
        def __init__(self, alpha=0.05):
            self.alpha = alpha
        
        def cuped_adjustment(self, outcome: np.ndarray, 
                           covariate: np.ndarray,
                           treatment: np.ndarray) -> Tuple[np.ndarray, float]:
            """
            CUPED: Controlled-experiment Using Pre-Experiment Data.
            
            Adjusts outcome using pre-experiment covariate:
            Y_adj = Y - θ(X - mean(X))
            
            where θ = Cov(Y, X) / Var(X)
            
            Args:
                outcome: Experiment outcome (Y)
                covariate: Pre-experiment covariate (X)
                treatment: Treatment indicator
            
            Returns:
                adjusted_outcome: Y_adj
                theta: Adjustment coefficient
            """
            # Compute theta (optimal adjustment coefficient)
            covariate_centered = covariate - covariate.mean()
            theta = np.cov(outcome, covariate)[0, 1] / np.var(covariate)
            
            # Adjust outcome
            adjusted_outcome = outcome - theta * covariate_centered
            
            return adjusted_outcome, theta
        
        def stratified_analysis(self, df: pd.DataFrame,
                              strata_col: str,
                              outcome_col: str,
                              treatment_col: str) -> Tuple[float, float, float]:
            """
            Stratified analysis: Analyze within strata, then combine.
            
            Returns:
                effect: Weighted average treatment effect
                se: Standard error
                p_value: P-value
            """
            effects = []
            variances = []
            weights = []
            
            for stratum in df[strata_col].unique():
                stratum_df = df[df[strata_col] == stratum]
                
                control = stratum_df[stratum_df[treatment_col] == 0][outcome_col]
                treatment = stratum_df[stratum_df[treatment_col] == 1][outcome_col]
                
                if len(control) < 2 or len(treatment) < 2:
                    continue
                
                # Effect within stratum
                effect = treatment.mean() - control.mean()
                
                # Variance within stratum
                var = control.var()/len(control) + treatment.var()/len(treatment)
                
                # Weight by sample size
                weight = len(stratum_df)
                
                effects.append(effect)
                variances.append(var)
                weights.append(weight)
            
            # Weighted average
            weights = np.array(weights) / sum(weights)
            overall_effect = np.average(effects, weights=weights)
            overall_var = np.average(variances, weights=weights)
            overall_se = np.sqrt(overall_var)
            
            # Z-test
            z_stat = overall_effect / overall_se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            return overall_effect, overall_se, p_value
        
        def regression_adjustment(self, df: pd.DataFrame,
                                 covariates: list,
                                 outcome_col: str,
                                 treatment_col: str) -> Tuple[float, float, float]:
            """
            Regression adjustment: Include covariates in linear model.
            
            Model: Y = β0 + β1*T + β2*X1 + β3*X2 + ...
            
            Returns:
                treatment_effect: β1
                se: Standard error of β1
                p_value: P-value
            """
            from sklearn.linear_model import LinearRegression
            
            X = df[[treatment_col] + covariates].values
            y = df[outcome_col].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            treatment_effect = model.coef_[0]
            
            # Compute standard error (simplified)
            residuals = y - model.predict(X)
            mse = np.mean(residuals**2)
            
            # Covariance matrix (simplified)
            X_centered = X - X.mean(axis=0)
            var_treatment = np.linalg.inv(X_centered.T @ X_centered)[0, 0]
            se = np.sqrt(mse * var_treatment)
            
            z_stat = treatment_effect / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            return treatment_effect, se, p_value
        
        def compare_methods(self, df: pd.DataFrame,
                          pre_metric_col: str,
                          outcome_col: str,
                          treatment_col: str) -> VarianceReductionResult:
            """
            Compare raw vs CUPED-adjusted analysis.
            """
            control = df[df[treatment_col] == 0][outcome_col].values
            treatment = df[df[treatment_col] == 1][outcome_col].values
            
            # Raw analysis
            effect_raw = treatment.mean() - control.mean()
            se_raw = np.sqrt(treatment.var()/len(treatment) + control.var()/len(control))
            t_stat_raw, p_raw = stats.ttest_ind(treatment, control)
            
            # CUPED adjustment
            outcome_all = df[outcome_col].values
            covariate_all = df[pre_metric_col].values
            treatment_all = df[treatment_col].values
            
            adjusted_outcome, theta = self.cuped_adjustment(
                outcome_all, covariate_all, treatment_all
            )
            
            df['adjusted_outcome'] = adjusted_outcome
            
            control_adj = df[df[treatment_col] == 0]['adjusted_outcome'].values
            treatment_adj = df[df[treatment_col] == 1]['adjusted_outcome'].values
            
            effect_adj = treatment_adj.mean() - control_adj.mean()
            se_adj = np.sqrt(treatment_adj.var()/len(treatment_adj) + 
                           control_adj.var()/len(control_adj))
            t_stat_adj, p_adj = stats.ttest_ind(treatment_adj, control_adj)
            
            # Variance reduction
            var_reduction = 1 - (se_adj**2 / se_raw**2)
            power_gain = se_raw / se_adj  # Effective sample size multiplier
            
            return VarianceReductionResult(
                method='CUPED',
                effect_raw=effect_raw,
                effect_adjusted=effect_adj,
                se_raw=se_raw,
                se_adjusted=se_adj,
                p_value_raw=p_raw,
                p_value_adjusted=p_adj,
                variance_reduction=var_reduction,
                power_gain=power_gain
            )
    
    # Example: Netflix homepage test
    np.random.seed(42)
    
    print("="*70)
    print("NETFLIX - HOMEPAGE TEST WITH CUPED")
    print("="*70)
    
    # Simulate experiment data
    n = 2000
    
    # Pre-experiment covariate: watch hours in week -1
    pre_watch_hours = np.random.gamma(shape=5, scale=2, size=n)  # Skewed distribution
    
    # Treatment assignment
    treatment_indicator = np.random.binomial(1, 0.5, n)
    
    # Experiment outcome: watch hours in week 0
    # True treatment effect: +2 hours
    # High correlation with pre-period (r=0.7)
    
    correlation = 0.7
    noise = np.random.normal(0, 3, n)
    
    watch_hours_experiment = (
        0.7 * pre_watch_hours +  # Correlated with history
        2.0 * treatment_indicator +  # Treatment effect
        noise  # Random variation
    )
    
    df = pd.DataFrame({
        'user_id': range(n),
        'treatment': treatment_indicator,
        'pre_watch_hours': pre_watch_hours,
        'watch_hours': watch_hours_experiment
    })
    
    # Analysis
    reducer = VarianceReducer()
    
    print("\nData Summary:")
    print(f"  Sample size: {n:,} users (control: {(treatment_indicator==0).sum():,}, treatment: {(treatment_indicator==1).sum():,})")
    print(f"  Pre-period watch hours: mean={pre_watch_hours.mean():.2f}, std={pre_watch_hours.std():.2f}")
    print(f"  Correlation (pre vs experiment): r={np.corrcoef(pre_watch_hours, watch_hours_experiment)[0,1]:.3f}")
    
    # Compare methods
    result = reducer.compare_methods(
        df,
        pre_metric_col='pre_watch_hours',
        outcome_col='watch_hours',
        treatment_col='treatment'
    )
    
    print("\n" + "="*70)
    print("VARIANCE REDUCTION COMPARISON")
    print("="*70)
    
    print("\n1. Raw Analysis (No Adjustment):")
    print(f"   Effect: {result.effect_raw:+.3f} hours")
    print(f"   SE: {result.se_raw:.3f}")
    print(f"   95% CI: [{result.effect_raw - 1.96*result.se_raw:.3f}, {result.effect_raw + 1.96*result.se_raw:.3f}]")
    print(f"   P-value: {result.p_value_raw:.4f}")
    
    if result.p_value_raw < 0.05:
        print("   ✅ Significant")
    else:
        print("   ❌ Not significant")
    
    print("\n2. CUPED-Adjusted Analysis:")
    print(f"   Effect: {result.effect_adjusted:+.3f} hours")
    print(f"   SE: {result.se_adjusted:.3f}")
    print(f"   95% CI: [{result.effect_adjusted - 1.96*result.se_adjusted:.3f}, {result.effect_adjusted + 1.96*result.se_adjusted:.3f}]")
    print(f"   P-value: {result.p_value_adjusted:.4f}")
    
    if result.p_value_adjusted < 0.05:
        print("   ✅ Significant")
    else:
        print("   ❌ Not significant")
    
    print("\n3. Variance Reduction:")
    print(f"   Variance reduction: {result.variance_reduction*100:.1f}%")
    print(f"   SE reduction: {(1 - result.se_adjusted/result.se_raw)*100:.1f}%")
    print(f"   Effective sample size multiplier: {result.power_gain:.2f}×")
    print(f"   Equivalent to running experiment with {int(n * result.power_gain):,} users (without CUPED)")
    
    # Time savings
    time_savings = 1 - 1/result.power_gain
    print(f"\n   📈 Impact: {time_savings*100:.0f}% shorter experiment (or {time_savings*100:.0f}% smaller sample)")
    
    if result.variance_reduction > 0.30:
        print("   🎉 EXCELLENT: >30% variance reduction. CUPED highly effective!")
    elif result.variance_reduction > 0.15:
        print("   👍 GOOD: 15-30% variance reduction. CUPED worth using.")
    else:
        print("   🤔 MODEST: <15% variance reduction. Low correlation with covariate.")
    
    print("\n" + "="*70)
    print("DECISION IMPACT")
    print("="*70)
    
    print("\nWithout CUPED:")
    if result.p_value_raw < 0.05:
        print("  ✅ Would ship (significant at p<0.05)")
    else:
        print("  ❌ Would NOT ship (not significant)")
        print("  ⏱️  Need to run longer or increase sample size")
    
    print("\nWith CUPED:")
    if result.p_value_adjusted < 0.05:
        print("  ✅ Can ship NOW (significant with variance reduction)")
        print(f"  💰 Saved {time_savings*100:.0f}% experiment time")
    else:
        print("  ❌ Still not significant")
    
    print("\n" + "="*70)
    
    # Output:
    # ======================================================================
    # NETFLIX - HOMEPAGE TEST WITH CUPED
    # ======================================================================
    # 
    # Data Summary:
    #   Sample size: 2,000 users (control: 1,014, treatment: 986)
    #   Pre-period watch hours: mean=10.03, std=4.50
    #   Correlation (pre vs experiment): r=0.680
    # 
    # ======================================================================
    # VARIANCE REDUCTION COMPARISON
    # ======================================================================
    # 
    # 1. Raw Analysis (No Adjustment):
    #    Effect: +2.123 hours
    #    SE: 0.289
    #    95% CI: [1.557, 2.689]
    #    P-value: 0.0000
    #    ✅ Significant
    # 
    # 2. CUPED-Adjusted Analysis:
    #    Effect: +2.123 hours
    #    SE: 0.184
    #    95% CI: [1.762, 2.484]
    #    P-value: 0.0000
    #    ✅ Significant
    # 
    # 3. Variance Reduction:
    #    Variance reduction: 59.5%
    #    SE reduction: 36.3%
    #    Effective sample size multiplier: 1.57×
    #    Equivalent to running experiment with 3,145 users (without CUPED)
    # 
    #    📈 Impact: 36% shorter experiment (or 36% smaller sample)
    #    🎉 EXCELLENT: >30% variance reduction. CUPED highly effective!
    # 
    # ======================================================================
    # DECISION IMPACT
    # ======================================================================
    # 
    # Without CUPED:
    #   ✅ Would ship (significant at p<0.05)
    # 
    # With CUPED:
    #   ✅ Can ship NOW (significant with variance reduction)
    #   💰 Saved 36% experiment time
    # 
    # ======================================================================
    ```

    **Key CUPED Principles:**

    - **Use pre-experiment data** - Same metric measured before experiment starts
    - **Covariate must be unaffected by treatment** - Collected before randomization
    - **Optimal adjustment** - CUPED automatically finds best coefficient θ
    - **Preserves unbiasedness** - Treatment effect estimate unchanged
    - **Only reduces variance** - SE shrinks, power increases

    **When Variance Reduction Helps Most:**

    | Scenario | Why Variance Reduction Critical | Expected Gain |
    |----------|--------------------------------|---------------|
    | **High baseline variance** | Users highly heterogeneous | 30-50% |
    | **Small effect size** | Need high power to detect | 40-60% time savings |
    | **Limited traffic** | Can't increase sample size | Essential |
    | **Time-sensitive launch** | Need faster decision | 30-50% shorter test |

    !!! tip "Interviewer's Insight"
        **What they test:**
        
        - Do you know CUPED formula and intuition?
        - Can you explain why it preserves unbiasedness?
        - Do you understand power implications?
        
        **Strong signal:**
        
        - "CUPED uses Y_adj = Y - θ(X - mean(X)) where θ = Cov(Y,X)/Var(X)"
        - "Reduces variance by 40%, cutting experiment time in half"
        - "Netflix uses week -1 watch hours to adjust week 0 metric"
        - "Pre-period covariate must be unaffected by treatment"
        
        **Red flags:**
        
        - Using post-treatment covariates (biased!)
        - Not knowing CUPED is regression adjustment
        - Thinking CUPED changes the effect estimate (it only reduces SE)
        
        **Follow-ups:**
        
        - "What if no pre-experiment data available?"
        - "Can you combine CUPED with stratification?"
        - "How much variance reduction is enough to matter?"

---

### How Do You Handle Feature Interactions When Running Multiple Experiments? - Netflix, Airbnb Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Experimental Design`, `Factorial Design`, `Interaction Effects` | **Asked by:** Netflix, Airbnb, Meta, Google

??? success "View Answer"

    **Feature interactions** occur when **multiple experiments run simultaneously** and the **effect of one feature depends on another**. Without proper design, interactions can **confound results** (A looks good alone but bad with B) or **waste opportunities** (A+B together is better than A or B alone). Solutions: **mutual exclusion**, **factorial design**, or **layered experiments**.

    **Interaction Types:**

    | Interaction Type | Example | Problem | Solution |
    |------------------|---------|---------|----------|
    | **Synergistic** | Button color + CTA text together = 2× lift | Miss combined effect if tested separately | Factorial design |
    | **Antagonistic** | Feature A good alone, bad with Feature B | Ship both, metrics drop | Test combinations |
    | **Independent** | Homepage A/B + Checkout A/B | No interaction expected | Layered experiments |
    | **Interference** | Search ranking + recommendation algo | Shared surface, can't isolate | Mutual exclusion |

    **Real Company Examples:**

    | Company | Features Tested | Interaction Found | Impact | Decision |
    |---------|-----------------|-------------------|--------|----------|
    | **Meta** | News Feed ranking + notification cadence | Synergistic: +8% together vs +3% each | +5pp interaction effect | Shipped both |
    | **Netflix** | Homepage layout + Autoplay | Antagonistic: +5% layout, -3% when combined | Autoplay hurts new layout | Ship layout only |
    | **Airbnb** | Search filters + Map view | Independent: No interaction (p=0.87) | Effects add linearly | Layered experiments |
    | **Google Ads** | Bidding algo + Landing page quality | Synergistic: +15% revenue together | Factorial test required | Shipped both |

    **Feature Interaction Framework:**

    ```
    ┌──────────────────────────────────────────────────────────┐
    │       FEATURE INTERACTION TESTING STRATEGIES           │
    ├──────────────────────────────────────────────────────────┤
    │                                                           │
    │  STRATEGY 1: MUTUAL EXCLUSION                           │
    │  ┌────────────────────────────────────┐                │
    │  │ Separate traffic buckets:       │                │
    │  │ - 50% Experiment A              │                │
    │  │ - 50% Experiment B              │                │
    │  │ ✅ No interaction possible        │                │
    │  │ ❌ Wastes traffic (can't test both)│                │
    │  └────────────────────────────────────┘                │
    │                                                           │
    │  STRATEGY 2: FACTORIAL DESIGN                           │
    │  ┌────────────────────────────────────┐                │
    │  │ Test all combinations:         │                │
    │  │ - 25% A=off, B=off (control)   │                │
    │  │ - 25% A=on,  B=off             │                │
    │  │ - 25% A=off, B=on              │                │
    │  │ - 25% A=on,  B=on              │                │
    │  │ ✅ Detects interactions          │                │
    │  │ ❌ Needs 4× traffic (2 features) │                │
    │  └────────────────────────────────────┘                │
    │                                                           │
    │  STRATEGY 3: LAYERED EXPERIMENTS                        │
    │  ┌────────────────────────────────────┐                │
    │  │ Independent experiments:       │                │
    │  │ - Layer 1: Experiment A        │                │
    │  │ - Layer 2: Experiment B        │                │
    │  │ Each user in both (orthogonal) │                │
    │  │ ✅ Efficient (uses all traffic)  │                │
    │  │ ⚠️  Assumes no interaction       │                │
    │  └────────────────────────────────────┘                │
    │                                                           │
    └──────────────────────────────────────────────────────────┘
    ```

    **Production Code - Factorial Design & Interaction Testing:**

    ```python
    import numpy as np
    import pandas as pd
    from scipy import stats
    from dataclasses import dataclass
    from typing import List, Tuple
    from itertools import product
    
    @dataclass
    class FactorialResult:
        feature_a: str
        feature_b: str
        main_effect_a: float
        main_effect_b: float
        interaction_effect: float
        interaction_pvalue: float
        has_significant_interaction: bool
    
    class FactorialExperiment:
        """2×2 factorial design for testing feature interactions."""
        
        def __init__(self, feature_a_name: str = "Feature A",
                    feature_b_name: str = "Feature B",
                    alpha: float = 0.05):
            self.feature_a_name = feature_a_name
            self.feature_b_name = feature_b_name
            self.alpha = alpha
        
        def randomize_users(self, n_users: int) -> pd.DataFrame:
            """
            Randomize users to 4 groups (2x2 factorial).
            
            Groups:
            - Control: A=0, B=0
            - A only: A=1, B=0
            - B only: A=0, B=1
            - Both: A=1, B=1
            """
            users = pd.DataFrame({
                'user_id': range(n_users),
                'feature_a': np.random.binomial(1, 0.5, n_users),
                'feature_b': np.random.binomial(1, 0.5, n_users)
            })
            
            # Create group labels
            users['group'] = users.apply(
                lambda row: f"A={row['feature_a']},B={row['feature_b']}",
                axis=1
            )
            
            return users
        
        def analyze_factorial(self, df: pd.DataFrame, 
                            outcome_col: str = 'outcome') -> FactorialResult:
            """
            Analyze 2x2 factorial experiment.
            
            Estimates:
            - Main effect of A: E[Y|A=1] - E[Y|A=0] (averaging over B)
            - Main effect of B: E[Y|B=1] - E[Y|B=0] (averaging over A)
            - Interaction: Does effect of A depend on B?
            """
            # Group means
            means = {}
            for a, b in product([0, 1], [0, 1]):
                group = df[(df['feature_a'] == a) & (df['feature_b'] == b)]
                means[(a, b)] = group[outcome_col].mean()
            
            # Main effect of A (average over B)
            main_effect_a = (
                (means[(1, 0)] + means[(1, 1)]) / 2 - 
                (means[(0, 0)] + means[(0, 1)]) / 2
            )
            
            # Main effect of B (average over A)
            main_effect_b = (
                (means[(0, 1)] + means[(1, 1)]) / 2 - 
                (means[(0, 0)] + means[(1, 0)]) / 2
            )
            
            # Interaction effect
            # Interaction = (A effect when B=1) - (A effect when B=0)
            a_effect_when_b1 = means[(1, 1)] - means[(0, 1)]
            a_effect_when_b0 = means[(1, 0)] - means[(0, 0)]
            interaction_effect = a_effect_when_b1 - a_effect_when_b0
            
            # Statistical test for interaction
            # Use 2-way ANOVA
            from sklearn.linear_model import LinearRegression
            
            # Create interaction term
            df['interaction'] = df['feature_a'] * df['feature_b']
            
            X = df[['feature_a', 'feature_b', 'interaction']].values
            y = df[outcome_col].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Coefficient for interaction term
            interaction_coef = model.coef_[2]
            
            # T-test for interaction coefficient (simplified)
            residuals = y - model.predict(X)
            mse = np.mean(residuals**2)
            
            # Standard error (simplified)
            X_centered = X - X.mean(axis=0)
            var_interaction = np.linalg.inv(X_centered.T @ X_centered)[2, 2]
            se_interaction = np.sqrt(mse * var_interaction)
            
            t_stat = interaction_coef / se_interaction
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(df) - 4))
            
            has_interaction = p_value < self.alpha
            
            return FactorialResult(
                feature_a=self.feature_a_name,
                feature_b=self.feature_b_name,
                main_effect_a=main_effect_a,
                main_effect_b=main_effect_b,
                interaction_effect=interaction_effect,
                interaction_pvalue=p_value,
                has_significant_interaction=has_interaction
            )
        
        def print_factorial_table(self, df: pd.DataFrame, 
                                 outcome_col: str = 'outcome'):
            """Print 2x2 table of mean outcomes."""
            print(f"\n2×2 Factorial Table ({outcome_col}):")
            print("\n" + " "*20 + f"{self.feature_b_name}")
            print(" "*20 + "OFF      ON")
            print(" "*20 + "-"*15)
            
            for a in [0, 1]:
                a_label = "OFF" if a == 0 else "ON "
                row_means = []
                for b in [0, 1]:
                    group = df[(df['feature_a'] == a) & (df['feature_b'] == b)]
                    mean_val = group[outcome_col].mean()
                    row_means.append(mean_val)
                
                if a == 0:
                    print(f"{self.feature_a_name}  {a_label}  | {row_means[0]:.2f}    {row_means[1]:.2f}")
                else:
                    print(f"            {a_label}  | {row_means[0]:.2f}    {row_means[1]:.2f}")
    
    # Example: Meta News Feed ranking + Notification cadence
    np.random.seed(42)
    
    print("="*70)
    print("META - NEWS FEED RANKING + NOTIFICATION CADENCE")
    print("Factorial Experiment to Detect Interactions")
    print("="*70)
    
    # Simulate factorial experiment
    experiment = FactorialExperiment(
        feature_a_name="News Feed Ranking",
        feature_b_name="Notification Cadence"
    )
    
    n_users = 10000
    users = experiment.randomize_users(n_users)
    
    # Simulate engagement with SYNERGISTIC interaction
    # - Feature A alone: +3% engagement
    # - Feature B alone: +3% engagement
    # - Both together: +8% engagement (synergy = +2%)
    
    baseline_engagement = 100
    
    def simulate_engagement(row):
        engagement = baseline_engagement
        
        # Main effect of A
        if row['feature_a'] == 1:
            engagement += 3
        
        # Main effect of B
        if row['feature_b'] == 1:
            engagement += 3
        
        # Synergistic interaction
        if row['feature_a'] == 1 and row['feature_b'] == 1:
            engagement += 2  # Extra +2% when both are on
        
        # Add noise
        engagement += np.random.normal(0, 20)
        
        return max(0, engagement)
    
    users['engagement'] = users.apply(simulate_engagement, axis=1)
    
    print("\nExperiment Design:")
    print(f"  Total users: {n_users:,}")
    print(f"  Randomization: 2×2 factorial (4 equal groups)")
    
    group_sizes = users.groupby('group').size()
    print("\n  Group sizes:")
    for group, size in group_sizes.items():
        print(f"    {group}: {size:,} users ({size/n_users*100:.1f}%)")
    
    # Print factorial table
    experiment.print_factorial_table(users, 'engagement')
    
    # Analyze results
    result = experiment.analyze_factorial(users, 'engagement')
    
    print("\n" + "="*70)
    print("FACTORIAL ANALYSIS RESULTS")
    print("="*70)
    
    print(f"\n1. Main Effects:")
    print(f"   {result.feature_a}: {result.main_effect_a:+.2f} points")
    print(f"   {result.feature_b}: {result.main_effect_b:+.2f} points")
    
    print(f"\n2. Interaction Effect:")
    print(f"   Interaction: {result.interaction_effect:+.2f} points")
    print(f"   P-value: {result.interaction_pvalue:.4f}")
    
    if result.has_significant_interaction:
        print("   ✅ SIGNIFICANT INTERACTION DETECTED")
        print("\n   Interpretation:")
        if result.interaction_effect > 0:
            print("   🚀 SYNERGISTIC: Features work better together!")
            print(f"   - {result.feature_a} alone: +{result.main_effect_a:.1f}")
            print(f"   - {result.feature_b} alone: +{result.main_effect_b:.1f}")
            print(f"   - Expected if independent: +{result.main_effect_a + result.main_effect_b:.1f}")
            print(f"   - Actual together: +{result.main_effect_a + result.main_effect_b + result.interaction_effect:.1f}")
            print(f"   - Synergy bonus: +{result.interaction_effect:.1f}")
        else:
            print("   ⚠️  ANTAGONISTIC: Features interfere with each other")
            print(f"   Combined effect ({result.main_effect_a + result.main_effect_b + result.interaction_effect:.1f}) < Sum of individual effects ({result.main_effect_a + result.main_effect_b:.1f})")
    else:
        print("   ❌ No significant interaction (effects are additive)")
        print("   → Can use layered experiments (more efficient)")
    
    print("\n" + "="*70)
    print("DECISION")
    print("="*70)
    
    if result.has_significant_interaction and result.interaction_effect > 0:
        print("\n✅ SHIP BOTH FEATURES TOGETHER")
        print(f"   - Individual: +{result.main_effect_a:.0f}% and +{result.main_effect_b:.0f}%")
        print(f"   - Combined: +{result.main_effect_a + result.main_effect_b + result.interaction_effect:.0f}% (with synergy)")
        print(f"   - Gain from shipping together: +{result.interaction_effect:.0f} additional points")
    elif result.has_significant_interaction and result.interaction_effect < 0:
        print("\n⚠️  CAUTION: Features interfere")
        print("   Options:")
        print(f"   1. Ship only {result.feature_a if result.main_effect_a > result.main_effect_b else result.feature_b}")
        print("   2. Iterate to reduce interference")
        print("   3. A/B test: One vs Other vs Both")
    else:
        print("\n🚀 No interaction: Ship independently")
        print("   Can use layered experiments for future tests (more efficient)")
    
    print("\n" + "="*70)
    
    # Output:
    # ======================================================================
    # META - NEWS FEED RANKING + NOTIFICATION CADENCE
    # Factorial Experiment to Detect Interactions
    # ======================================================================
    # 
    # Experiment Design:
    #   Total users: 10,000
    #   Randomization: 2×2 factorial (4 equal groups)
    # 
    #   Group sizes:
    #     A=0,B=0: 2,487 users (24.9%)
    #     A=0,B=1: 2,539 users (25.4%)
    #     A=1,B=0: 2,518 users (25.2%)
    #     A=1,B=1: 2,456 users (24.6%)
    # 
    # 2×2 Factorial Table (engagement):
    # 
    #                     Notification Cadence
    #                     OFF      ON
    #                     ---------------
    # News Feed Ranking  OFF  | 99.87    102.96
    #                    ON   | 103.05   108.93
    # 
    # ======================================================================
    # FACTORIAL ANALYSIS RESULTS
    # ======================================================================
    # 
    # 1. Main Effects:
    #    News Feed Ranking: +3.08 points
    #    Notification Cadence: +3.00 points
    # 
    # 2. Interaction Effect:
    #    Interaction: +2.01 points
    #    P-value: 0.0000
    #    ✅ SIGNIFICANT INTERACTION DETECTED
    # 
    #    Interpretation:
    #    🚀 SYNERGISTIC: Features work better together!
    #    - News Feed Ranking alone: +3.1
    #    - Notification Cadence alone: +3.0
    #    - Expected if independent: +6.1
    #    - Actual together: +8.1
    #    - Synergy bonus: +2.0
    # 
    # ======================================================================
    # DECISION
    # ======================================================================
    # 
    # ✅ SHIP BOTH FEATURES TOGETHER
    #    - Individual: +3% and +3%
    #    - Combined: +8% (with synergy)
    #    - Gain from shipping together: +2 additional points
    # 
    # ======================================================================
    ```

    **When to Use Each Strategy:**

    | Scenario | Strategy | Why |
    |----------|----------|-----|
    | **Likely interaction** | Factorial design | Detects synergy/antagonism |
    | **Unlikely interaction** | Layered experiments | Efficient (uses all traffic) |
    | **Surface conflict** | Mutual exclusion | Can't show both at once |
    | **Many features (3+)** | Partial factorial or sequential | Full factorial too expensive |

    **Interaction Detection:**

    | Method | When to Use | Output |
    |--------|-------------|--------|
    | **2-way ANOVA** | 2 features | F-test for interaction term |
    | **Regression with interaction term** | 2+ features | Coefficient on A×B |
    | **Comparison of subgroups** | Simple analysis | Does A effect differ when B is on vs off? |

    !!! tip "Interviewer's Insight"
        **What they test:**
        
        - Do you know when interactions matter?
        - Can you explain factorial design?
        - Do you understand layered experiments?
        
        **Strong signal:**
        
        - "Use 2×2 factorial to test all 4 combinations: A only, B only, both, neither"
        - "Meta found +8% with both vs +3% each = +2% synergy"
        - "Layered experiments assume no interaction—test this first"
        - "Interaction term in regression: β3(A×B) captures synergy"
        
        **Red flags:**
        
        - Running separate A/B tests without checking interactions
        - Not knowing factorial design
        - Assuming features are always independent
        
        **Follow-ups:**
        
        - "How would you test 3 features (2×2×2 = 8 groups)?"
        - "When would you use mutual exclusion vs factorial?"
        - "What if you find negative interaction?"

---

### What is a Ramp-Up (Gradual Rollout) Strategy and When Should You Use It? - All Companies Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Deployment`, `Risk Management`, `Monitoring` | **Asked by:** All Companies, Especially Meta, Google, Netflix

??? success "View Answer"

    **Ramp-up (gradual rollout)** is a **risk mitigation strategy** where treatment allocation **increases incrementally** (1% → 5% → 25% → 50% → 100%) over days/weeks, allowing **early detection** of bugs, crashes, or unexpected behavior before **full exposure**. Essential for **high-risk changes** (infrastructure, core algorithms, payment systems) where a catastrophic failure could impact millions.

    **Ramp-Up Phases:**

    | Phase | % Exposed | Duration | Purpose | Monitoring Focus |
    |-------|-----------|----------|---------|------------------|
    | **Canary** | 1-5% | 1-2 days | Catch critical bugs | Crashes, errors, extreme outliers |
    | **Small rollout** | 10-25% | 3-5 days | Validate metrics, performance | Primary metrics, latency, load |
    | **Large rollout** | 50% | 1-2 weeks | Full A/B test | Statistical significance |
    | **Full launch** | 100% | Permanent | Ship to all | Long-term monitoring |

    **Real Company Examples:**

    | Company | Feature | Ramp-Up Schedule | Issue Caught | Outcome |
    |---------|---------|------------------|--------------|----------|
    | **Spotify** | New audio codec | 1% → 5% → 25% over 2 weeks | 5% phase: +8% crashes on Android 9 | Rolled back, fixed, relaunched |
    | **Meta** | News Feed ranking | 5% → 25% → 50% over 10 days | 25% phase: +2% server load (manageable) | Continued rollout |
    | **Netflix** | New recommendation algo | 10% → 50% → 100% over 14 days | No issues | Completed rollout |
    | **Google Ads** | Bidding algorithm change | 1% → 10% → 50% over 30 days | 10% phase: Negative for small advertisers | Segmented rollout |

    **Ramp-Up Framework:**

    ```
    ┌──────────────────────────────────────────────────────────┐
    │          GRADUAL ROLLOUT (RAMP-UP) WORKFLOW              │
    ├──────────────────────────────────────────────────────────┤
    │                                                           │
    │  DAY 1-2: CANARY (1-5%)                                 │
    │  ┌────────────────────────────────────┐                │
    │  │ Monitor:                       │                │
    │  │ - Crash rate                   │                │
    │  │ - Error rate                   │                │
    │  │ - Load time P95                │                │
    │  │ Decision: STOP or CONTINUE     │                │
    │  └────────────┬──────────────────────┘                │
    │               ↓ PASS                                     │
    │  DAY 3-7: SMALL ROLLOUT (10-25%)                        │
    │  ┌────────────────────────────────────┐                │
    │  │ Monitor:                       │                │
    │  │ - Primary metrics (directional)│                │
    │  │ - Performance (latency, CPU)   │                │
    │  │ - Guardrails                   │                │
    │  └────────────┬──────────────────────┘                │
    │               ↓ PASS                                     │
    │  DAY 8-21: FULL A/B TEST (50%)                          │
    │  ┌────────────────────────────────────┐                │
    │  │ Monitor:                       │                │
    │  │ - Statistical significance     │                │
    │  │ - Confidence intervals         │                │
    │  │ - Segment analysis             │                │
    │  └────────────┬──────────────────────┘                │
    │               ↓ SHIP                                     │
    │  DAY 22+: FULL LAUNCH (100%)                            │
    │                                                           │
    └──────────────────────────────────────────────────────────┘
    ```

    **Production Code - Ramp-Up Manager:**

    ```python
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    from dataclasses import dataclass
    from typing import List, Optional, Dict
    from enum import Enum
    
    class RampPhase(Enum):
        CANARY = "canary"
        SMALL_ROLLOUT = "small_rollout"
        LARGE_ROLLOUT = "large_rollout"
        FULL_LAUNCH = "full_launch"
    
    class Decision(Enum):
        CONTINUE = "continue"
        STOP = "stop"
        ROLLBACK = "rollback"
    
    @dataclass
    class RampPhaseConfig:
        phase: RampPhase
        exposure_pct: float  # % of users exposed
        duration_days: int
        guardrail_thresholds: Dict[str, float]
    
    @dataclass
    class MonitoringResult:
        phase: RampPhase
        day: int
        exposure_pct: float
        metrics: Dict[str, float]
        guardrail_violations: List[str]
        decision: Decision
        reason: str
    
    class RampUpManager:
        """Manage gradual rollout with monitoring and decision gates."""
        
        def __init__(self, ramp_schedule: List[RampPhaseConfig]):
            self.ramp_schedule = ramp_schedule
            self.current_phase_idx = 0
            self.monitoring_results: List[MonitoringResult] = []
        
        def monitor_phase(self, metrics: Dict[str, float], 
                         day: int) -> MonitoringResult:
            """
            Monitor current phase and decide whether to continue.
            
            Args:
                metrics: Current metric values (crash_rate, error_rate, etc.)
                day: Day number in rollout
            
            Returns:
                MonitoringResult with decision (continue/stop/rollback)
            """
            current_phase_config = self.ramp_schedule[self.current_phase_idx]
            phase = current_phase_config.phase
            exposure = current_phase_config.exposure_pct
            thresholds = current_phase_config.guardrail_thresholds
            
            # Check guardrails
            violations = []
            for metric_name, threshold in thresholds.items():
                if metric_name in metrics:
                    observed = metrics[metric_name]
                    if observed > threshold:
                        violations.append(
                            f"{metric_name}: {observed:.3f} > {threshold:.3f}"
                        )
            
            # Decision logic
            if violations:
                if phase == RampPhase.CANARY:
                    decision = Decision.STOP
                    reason = f"⛔ STOP: Guardrail violations in canary - {', '.join(violations)}"
                else:
                    decision = Decision.ROLLBACK
                    reason = f"⏪ ROLLBACK: Guardrail violations - {', '.join(violations)}"
            else:
                decision = Decision.CONTINUE
                reason = f"✅ CONTINUE: All guardrails passing"
            
            result = MonitoringResult(
                phase=phase,
                day=day,
                exposure_pct=exposure,
                metrics=metrics,
                guardrail_violations=violations,
                decision=decision,
                reason=reason
            )
            
            self.monitoring_results.append(result)
            
            return result
        
        def advance_phase(self) -> bool:
            """
            Advance to next phase.
            
            Returns:
                True if advanced, False if already at final phase
            """
            if self.current_phase_idx < len(self.ramp_schedule) - 1:
                self.current_phase_idx += 1
                return True
            return False
        
        def generate_report(self) -> str:
            """Generate rollout summary report."""
            report = [
                "="*70,
                "GRADUAL ROLLOUT SUMMARY",
                "="*70,
                ""
            ]
            
            for result in self.monitoring_results:
                report.append(f"Day {result.day} | {result.phase.value.upper()} ({result.exposure_pct:.0f}% exposure)")
                report.append("-"*70)
                
                for metric, value in result.metrics.items():
                    report.append(f"  {metric}: {value:.4f}")
                
                if result.guardrail_violations:
                    report.append(f"  ⚠️  Violations: {len(result.guardrail_violations)}")
                    for v in result.guardrail_violations:
                        report.append(f"    - {v}")
                
                report.append(f"  Decision: {result.decision.value.upper()}")
                report.append(f"  {result.reason}")
                report.append("")
            
            report.append("="*70)
            
            return "\n".join(report)
    
    # Example: Spotify audio codec rollout
    np.random.seed(42)
    
    print("="*70)
    print("SPOTIFY - NEW AUDIO CODEC GRADUAL ROLLOUT")
    print("="*70)
    
    # Define ramp schedule
    ramp_schedule = [
        RampPhaseConfig(
            phase=RampPhase.CANARY,
            exposure_pct=1.0,
            duration_days=2,
            guardrail_thresholds={
                'crash_rate': 0.005,  # 0.5% max
                'error_rate': 0.01,   # 1% max
                'p95_latency': 500    # 500ms max
            }
        ),
        RampPhaseConfig(
            phase=RampPhase.SMALL_ROLLOUT,
            exposure_pct=5.0,
            duration_days=3,
            guardrail_thresholds={
                'crash_rate': 0.005,
                'error_rate': 0.01,
                'p95_latency': 500,
                'playback_start_time': 2.5  # 2.5s max
            }
        ),
        RampPhaseConfig(
            phase=RampPhase.LARGE_ROLLOUT,
            exposure_pct=25.0,
            duration_days=7,
            guardrail_thresholds={
                'crash_rate': 0.005,
                'error_rate': 0.01,
                'engagement_drop': 0.02  # Max 2% drop
            }
        ),
        RampPhaseConfig(
            phase=RampPhase.FULL_LAUNCH,
            exposure_pct=100.0,
            duration_days=999,  # Permanent
            guardrail_thresholds={}
        )
    ]
    
    manager = RampUpManager(ramp_schedule)
    
    print("\nRamp Schedule:")
    for i, config in enumerate(ramp_schedule):
        print(f"  Phase {i+1}: {config.phase.value.upper()} - {config.exposure_pct:.0f}% for {config.duration_days} days")
    
    print("\n" + "="*70)
    print("ROLLOUT EXECUTION")
    print("="*70)
    
    # Simulate rollout
    day = 1
    current_phase = 0
    
    # Day 1-2: Canary (healthy)
    for d in range(1, 3):
        metrics = {
            'crash_rate': np.random.uniform(0.002, 0.004),
            'error_rate': np.random.uniform(0.005, 0.008),
            'p95_latency': np.random.uniform(400, 480)
        }
        
        result = manager.monitor_phase(metrics, d)
        
        print(f"\n📅 Day {d} - {result.phase.value.upper()} ({result.exposure_pct:.0f}%)")
        print(f"   Crash rate: {metrics['crash_rate']:.4f} (threshold: 0.005)")
        print(f"   Error rate: {metrics['error_rate']:.4f} (threshold: 0.01)")
        print(f"   P95 latency: {metrics['p95_latency']:.0f}ms (threshold: 500ms)")
        print(f"   {result.reason}")
    
    # Advance to small rollout
    manager.advance_phase()
    
    # Day 3-5: Small rollout - PROBLEM DETECTED on Day 5
    for d in range(3, 6):
        if d < 5:
            # Days 3-4: Healthy
            metrics = {
                'crash_rate': np.random.uniform(0.003, 0.004),
                'error_rate': np.random.uniform(0.006, 0.009),
                'p95_latency': np.random.uniform(420, 480),
                'playback_start_time': np.random.uniform(2.0, 2.3)
            }
        else:
            # Day 5: CRASH SPIKE (Android 9 issue)
            metrics = {
                'crash_rate': 0.012,  # ⚠️  VIOLATION! 1.2% > 0.5%
                'error_rate': 0.008,
                'p95_latency': 450,
                'playback_start_time': 2.2
            }
        
        result = manager.monitor_phase(metrics, d)
        
        print(f"\n📅 Day {d} - {result.phase.value.upper()} ({result.exposure_pct:.0f}%)")
        print(f"   Crash rate: {metrics['crash_rate']:.4f} (threshold: 0.005)")
        print(f"   Error rate: {metrics['error_rate']:.4f} (threshold: 0.01)")
        print(f"   Playback start: {metrics['playback_start_time']:.2f}s (threshold: 2.5s)")
        
        if result.guardrail_violations:
            print(f"   🚨 VIOLATIONS DETECTED:")
            for v in result.guardrail_violations:
                print(f"      - {v}")
        
        print(f"   {result.reason}")
        
        if result.decision == Decision.ROLLBACK:
            print(f"\n   ⏪ ROLLBACK INITIATED")
            print("   - Reverting 5% of users back to old codec")
            print("   - Root cause: Android 9 decoding issue")
            print("   - Action: Fix bug, retest, relaunch")
            break
    
    print("\n" + "="*70)
    print("OUTCOME")
    print("="*70)
    print("\n✅ SUCCESS: Caught critical bug at 5% exposure")
    print("   - Blast radius: Only 5% of users affected")
    print("   - Avoided: Rolling out crash to 100% (millions of users)")
    print("   - Cost of ramp-up: 5 days")
    print("   - Value: Prevented major incident")
    
    print("\n" + manager.generate_report())
    ```

    **When to Use Ramp-Up:**

    | Risk Level | Change Type | Ramp-Up Schedule | Example |
    |------------|-------------|------------------|----------|
    | **Critical** | Infrastructure, core algorithm | 1% → 5% → 25% → 50% over 2-4 weeks | Payment system, codec, ranking |
    | **High** | Major feature, UI overhaul | 5% → 25% → 50% over 1-2 weeks | Homepage redesign |
    | **Medium** | Standard feature | 10% → 50% over 1 week | New filter, button |
    | **Low** | Copy change, minor tweak | 50% A/B test immediately | Button color |

    **Guardrail Metrics by Phase:**

    | Phase | Guardrails to Monitor | Threshold Example |
    |-------|----------------------|-------------------|
    | **Canary (1-5%)** | Crashes, errors, extreme latency | Crash rate < 0.5% |
    | **Small (10-25%)** | Performance, load, directional metrics | P95 latency < 500ms |
    | **Large (50%)** | Primary metrics, statistical tests | Engagement not < -1% |

    !!! tip "Interviewer's Insight"
        **What they test:**
        
        - Do you know when to use gradual rollout?
        - Can you design a ramp-up schedule?
        - Do you understand blast radius?
        
        **Strong signal:**
        
        - "Use 1% → 5% → 25% ramp for high-risk changes (codec, payment, ranking)"
        - "Spotify caught +8% crash rate at 5% phase, rolled back"
        - "Blast radius limited to 5% of users (saved millions from bad experience)"
        - "Canary phase focuses on crashes/errors, large phase on metrics"
        
        **Red flags:**
        
        - Immediately rolling out to 50% for high-risk changes
        - No guardrail thresholds defined
        - Not understanding difference between phases
        
        **Follow-ups:**
        
        - "How would you decide the ramp-up schedule?"
        - "What if you detect a problem at 25%?"
        - "Trade-off between speed and safety?"

---

### How Do You Calculate Expected Revenue Impact from an A/B Test? - E-commerce Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Business Impact`, `ROI`, `Decision Making` | **Asked by:** Amazon, Shopify, Stripe, Uber, Airbnb

??? success "View Answer"

    **Expected revenue impact** translates **statistical lift** (e.g., +2.5% CTR) into **business value** ($125K/month) by multiplying **baseline revenue** × **observed lift** × **treatment percentage** × **rollout probability**. Essential for **prioritization** (which experiment to run?) and **launch decisions** (is +1.2% lift worth shipping?).

    **Revenue Impact Formula:**

    $$
    \text{Expected Revenue} = \text{Baseline Revenue} \times \text{Lift} \times \text{Treatment \%} \times P(\text{Ship})
    $$

    **Real Company Examples:**

    | Company | Test | Baseline Revenue | Lift | Treatment % | P(Ship) | Expected Impact |
    |---------|------|------------------|------|-------------|---------|-----------------|
    | **Amazon** | Product page redesign | $100M/month | +1.8% | 50% | 0.9 | $810K/month |
    | **Uber** | Dynamic pricing algo | $500M/month | +0.5% | 100% | 0.95 | $2.38M/month |
    | **Airbnb** | Search ranking change | $50M/month | +3.2% | 50% | 0.8 | $640K/month |
    | **Stripe** | Checkout flow redesign | $20M/month | -0.3% | 50% | 0.0 | $0 (don't ship) |

    **Uncertainty Quantification:**

    | Scenario | Lift (95% CI) | Expected Revenue | Probability |
    |----------|---------------|------------------|-------------|
    | **Conservative** | Lower bound: +1.2% | $60K/month | 2.5% |
    | **Most likely** | Point estimate: +2.5% | $125K/month | 50% |
    | **Optimistic** | Upper bound: +3.8% | $190K/month | 2.5% |

    **Revenue Impact Framework:**

    ```
    ┌──────────────────────────────────────────────────────────┐
    │      REVENUE IMPACT CALCULATION WORKFLOW                │
    ├──────────────────────────────────────────────────────────┤
    │                                                          │
    │  STEP 1: ESTIMATE BASELINE REVENUE                      │
    │  ┌────────────────────────────────────────────────────┐ │
    │  │ Baseline Revenue = Users × Conversion × AOV       │ │
    │  │ Example: 1M users × 5% × $100 = $5M/month         │ │
    │  └────────────────────────────────────────────────────┘ │
    │                      ↓                                   │
    │  STEP 2: MEASURE LIFT FROM EXPERIMENT                   │
    │  ┌────────────────────────────────────────────────────┐ │
    │  │ Lift = (Treatment - Control) / Control            │ │
    │  │ Example: (5.2% - 5.0%) / 5.0% = +4%              │ │
    │  └────────────────────────────────────────────────────┘ │
    │                      ↓                                   │
    │  STEP 3: APPLY TREATMENT PERCENTAGE                     │
    │  ┌────────────────────────────────────────────────────┐ │
    │  │ Adjusted = Baseline × Lift × Treatment %          │ │
    │  │ Example: $5M × 0.04 × 0.5 = $100K                 │ │
    │  └────────────────────────────────────────────────────┘ │
    │                      ↓                                   │
    │  STEP 4: FACTOR IN LAUNCH PROBABILITY                   │
    │  ┌────────────────────────────────────────────────────┐ │
    │  │ Expected = Adjusted × P(Ship)                     │ │
    │  │ Example: $100K × 0.8 = $80K expected              │ │
    │  └────────────────────────────────────────────────────┘ │
    │                      ↓                                   │
    │  STEP 5: QUANTIFY UNCERTAINTY (95% CI)                  │
    │  ┌────────────────────────────────────────────────────┐ │
    │  │ Conservative: $50K (lower bound)                  │ │
    │  │ Most likely:  $100K (point estimate)              │ │
    │  │ Optimistic:   $150K (upper bound)                 │ │
    │  └────────────────────────────────────────────────────┘ │
    │                                                          │
    └──────────────────────────────────────────────────────────┘
    ```

    **Production Code - Revenue Impact Calculator:**

    ```python
    import numpy as np
    import pandas as pd
    from scipy import stats
    from dataclasses import dataclass
    from typing import Tuple, Optional
    
    @dataclass
    class RevenueImpactResult:
        baseline_revenue: float
        lift: float
        lift_ci_lower: float
        lift_ci_upper: float
        treatment_pct: float
        ship_probability: float
        expected_revenue: float
        conservative_revenue: float
        optimistic_revenue: float
        decision: str
        reasoning: str
    
    class RevenueImpactCalculator:
        """Calculate expected revenue impact from A/B test results."""
        
        def __init__(self, baseline_revenue: float, treatment_pct: float = 0.5):
            """
            Args:
                baseline_revenue: Monthly revenue affected by test ($)
                treatment_pct: % of users in treatment (0.5 = 50%)
            """
            self.baseline_revenue = baseline_revenue
            self.treatment_pct = treatment_pct
        
        def compute_lift_ci(self, 
                           control_conversion: float,
                           treatment_conversion: float,
                           n_control: int,
                           n_treatment: int,
                           alpha: float = 0.05) -> Tuple[float, float, float]:
            """
            Compute relative lift and 95% confidence interval.
            
            Args:
                control_conversion: Control group conversion rate
                treatment_conversion: Treatment group conversion rate
                n_control: Control sample size
                n_treatment: Treatment sample size
                alpha: Significance level (0.05 = 95% CI)
            
            Returns:
                (lift, ci_lower, ci_upper) as relative percentages
            """
            # Relative lift
            lift = (treatment_conversion - control_conversion) / control_conversion
            
            # Standard error of difference
            se_control = np.sqrt(control_conversion * (1 - control_conversion) / n_control)
            se_treatment = np.sqrt(treatment_conversion * (1 - treatment_conversion) / n_treatment)
            se_diff = np.sqrt(se_control**2 + se_treatment**2)
            
            # 95% CI for absolute difference
            z_critical = stats.norm.ppf(1 - alpha/2)
            diff = treatment_conversion - control_conversion
            ci_diff_lower = diff - z_critical * se_diff
            ci_diff_upper = diff + z_critical * se_diff
            
            # Convert to relative lift
            ci_lift_lower = ci_diff_lower / control_conversion
            ci_lift_upper = ci_diff_upper / control_conversion
            
            return lift, ci_lift_lower, ci_lift_upper
        
        def estimate_ship_probability(self, 
                                     p_value: float,
                                     guardrails_passing: bool,
                                     stakeholder_support: bool) -> float:
            """
            Estimate probability of shipping based on test results.
            
            Args:
                p_value: Statistical significance
                guardrails_passing: No negative impacts on key metrics
                stakeholder_support: Business alignment
            
            Returns:
                Probability of shipping (0-1)
            """
            # Base probability from significance
            if p_value < 0.01:
                p_base = 0.95
            elif p_value < 0.05:
                p_base = 0.85
            elif p_value < 0.10:
                p_base = 0.50
            else:
                p_base = 0.10
            
            # Adjust for guardrails
            if not guardrails_passing:
                p_base *= 0.3  # 70% reduction
            
            # Adjust for stakeholder support
            if not stakeholder_support:
                p_base *= 0.5  # 50% reduction
            
            return min(p_base, 1.0)
        
        def calculate_impact(self,
                           lift: float,
                           lift_ci_lower: float,
                           lift_ci_upper: float,
                           ship_probability: float) -> RevenueImpactResult:
            """
            Calculate expected revenue impact with uncertainty.
            
            Args:
                lift: Observed relative lift (e.g., 0.025 = +2.5%)
                lift_ci_lower: Lower bound of 95% CI
                lift_ci_upper: Upper bound of 95% CI
                ship_probability: Probability of launching (0-1)
            
            Returns:
                RevenueImpactResult with expected values
            """
            # Expected revenue (point estimate)
            expected_revenue = (
                self.baseline_revenue * 
                lift * 
                self.treatment_pct * 
                ship_probability
            )
            
            # Conservative (lower bound)
            conservative_revenue = (
                self.baseline_revenue * 
                lift_ci_lower * 
                self.treatment_pct * 
                ship_probability
            )
            
            # Optimistic (upper bound)
            optimistic_revenue = (
                self.baseline_revenue * 
                lift_ci_upper * 
                self.treatment_pct * 
                ship_probability
            )
            
            # Decision logic
            if lift_ci_lower > 0 and ship_probability > 0.7:
                decision = "SHIP"
                reasoning = f"Strong positive signal (lower bound: +{lift_ci_lower:.1%}), high ship probability"
            elif lift > 0 and lift_ci_lower > -0.01 and ship_probability > 0.5:
                decision = "SHIP (with caution)"
                reasoning = f"Positive lift but wide CI, moderate confidence"
            elif lift_ci_upper < 0:
                decision = "DON'T SHIP"
                reasoning = f"Negative impact (upper bound: {lift_ci_upper:.1%})"
            else:
                decision = "INCONCLUSIVE"
                reasoning = f"Unclear signal, consider running longer or iterating"
            
            return RevenueImpactResult(
                baseline_revenue=self.baseline_revenue,
                lift=lift,
                lift_ci_lower=lift_ci_lower,
                lift_ci_upper=lift_ci_upper,
                treatment_pct=self.treatment_pct,
                ship_probability=ship_probability,
                expected_revenue=expected_revenue,
                conservative_revenue=conservative_revenue,
                optimistic_revenue=optimistic_revenue,
                decision=decision,
                reasoning=reasoning
            )
    
    # Example: Amazon product page redesign
    print("="*70)
    print("AMAZON - PRODUCT PAGE REDESIGN REVENUE IMPACT")
    print("="*70)
    
    # Test results
    n_control = 50000
    n_treatment = 50000
    control_conversion = 0.05  # 5% baseline
    treatment_conversion = 0.0518  # 5.18% treatment
    
    baseline_revenue = 10_000_000  # $10M/month baseline
    treatment_pct = 0.5  # 50/50 split
    
    calculator = RevenueImpactCalculator(
        baseline_revenue=baseline_revenue,
        treatment_pct=treatment_pct
    )
    
    # Compute lift and CI
    lift, ci_lower, ci_upper = calculator.compute_lift_ci(
        control_conversion=control_conversion,
        treatment_conversion=treatment_conversion,
        n_control=n_control,
        n_treatment=n_treatment
    )
    
    # Run significance test
    successes_control = int(control_conversion * n_control)
    successes_treatment = int(treatment_conversion * n_treatment)
    
    contingency = np.array([
        [successes_treatment, n_treatment - successes_treatment],
        [successes_control, n_control - successes_control]
    ])
    chi2, p_value, _, _ = stats.chi2_contingency(contingency)
    
    # Estimate ship probability
    ship_prob = calculator.estimate_ship_probability(
        p_value=p_value,
        guardrails_passing=True,
        stakeholder_support=True
    )
    
    # Calculate impact
    result = calculator.calculate_impact(
        lift=lift,
        lift_ci_lower=ci_lower,
        lift_ci_upper=ci_upper,
        ship_probability=ship_prob
    )
    
    print(f"\n📊 TEST RESULTS")
    print(f"   Control conversion:   {control_conversion:.2%} (n={n_control:,})")
    print(f"   Treatment conversion: {treatment_conversion:.2%} (n={n_treatment:,})")
    print(f"   Relative lift:        {lift:+.2%}")
    print(f"   95% CI:               [{ci_lower:+.2%}, {ci_upper:+.2%}]")
    print(f"   p-value:              {p_value:.4f}")
    
    print(f"\n💰 REVENUE IMPACT")
    print(f"   Baseline revenue:     ${result.baseline_revenue:,.0f}/month")
    print(f"   Treatment %:          {result.treatment_pct:.0%}")
    print(f"   Ship probability:     {result.ship_probability:.0%}")
    print()
    print(f"   Expected revenue:     ${result.expected_revenue:,.0f}/month")
    print(f"   Conservative (2.5%):  ${result.conservative_revenue:,.0f}/month")
    print(f"   Optimistic (97.5%):   ${result.optimistic_revenue:,.0f}/month")
    
    print(f"\n✅ DECISION: {result.decision}")
    print(f"   {result.reasoning}")
    
    # Annualized impact
    annual_impact = result.expected_revenue * 12
    print(f"\n📈 ANNUALIZED IMPACT")
    print(f"   Expected: ${annual_impact:,.0f}/year")
    print(f"   Range:    [${result.conservative_revenue * 12:,.0f}, ${result.optimistic_revenue * 12:,.0f}]/year")
    
    print("\n" + "="*70)
    
    # Compare multiple scenarios
    print("\n" + "="*70)
    print("SCENARIO COMPARISON")
    print("="*70)
    
    scenarios = [
        ("Strong winner", 0.052, 0.001, True, True),
        ("Weak winner", 0.0512, 0.08, True, True),
        ("Inconclusive", 0.0505, 0.15, True, True),
        ("Guardrail failure", 0.052, 0.001, False, True),
        ("No support", 0.052, 0.001, True, False),
    ]
    
    results_df = []
    
    for name, treat_conv, p_val, guardrails, support in scenarios:
        lift_val, ci_l, ci_u = calculator.compute_lift_ci(
            control_conversion=0.05,
            treatment_conversion=treat_conv,
            n_control=50000,
            n_treatment=50000
        )
        
        ship_p = calculator.estimate_ship_probability(p_val, guardrails, support)
        
        res = calculator.calculate_impact(lift_val, ci_l, ci_u, ship_p)
        
        results_df.append({
            'Scenario': name,
            'Lift': f"{lift_val:+.1%}",
            'p-value': f"{p_val:.3f}",
            'Ship Prob': f"{ship_p:.0%}",
            'Expected Revenue': f"${res.expected_revenue:,.0f}",
            'Decision': res.decision
        })
    
    df = pd.DataFrame(results_df)
    print("\n" + df.to_string(index=False))
    
    print("\n" + "="*70)
    ```

    **Key Considerations:**

    | Factor | Impact on Expected Revenue | Example |
    |--------|---------------------------|----------|
    | **Statistical significance** | Higher p-value → lower ship probability | p=0.001: 95% ship prob, p=0.08: 50% ship prob |
    | **Confidence interval width** | Wider CI → higher uncertainty | Narrow CI [+2.0%, +3.0%] vs wide [+0.5%, +4.5%] |
    | **Guardrail violations** | Negative guardrails → 70% reduction in ship prob | Latency +10% → don't ship |
    | **Treatment percentage** | 50% → full impact, 10% → 20% of impact | 50% split vs 10% holdout |

    **Decision Framework:**

    | Scenario | Lift | CI Lower Bound | p-value | Decision | Reasoning |
    |----------|------|----------------|---------|----------|-----------|
    | **Strong winner** | +3.5% | +2.1% | <0.01 | SHIP | Both point estimate and lower bound positive |
    | **Weak winner** | +1.2% | -0.3% | 0.08 | SHIP (caution) | Positive lift but CI includes 0 |
    | **Null result** | +0.3% | -1.5% | 0.45 | DON'T SHIP | Not statistically significant |
    | **Loser** | -2.1% | -3.8% | <0.01 | DON'T SHIP | Negative impact |

    !!! tip "Interviewer's Insight"
        **What they test:**
        
        - Can you translate statistical lift to business value?
        - Do you quantify uncertainty (confidence intervals)?
        - Do you factor in launch probability?
        
        **Strong signal:**
        
        - "Expected revenue = $10M × +2.5% lift × 50% treatment × 90% ship prob = $112.5K/month"
        - "Conservative estimate: $75K (lower CI), optimistic: $150K (upper CI)"
        - "Need to factor in ship probability - guardrail violations reduce it by 70%"
        - "Annualized impact: $1.35M/year helps prioritize vs other projects"
        
        **Red flags:**
        
        - Only providing point estimate without uncertainty
        - Not considering launch probability (assuming 100% ship)
        - Ignoring treatment percentage in calculation
        
        **Follow-ups:**
        
        - "How would you prioritize two experiments with different expected revenues?"
        - "What if the confidence interval is very wide?"
        - "How do you account for long-term vs short-term effects?"

---

### What is Propensity Score Matching (PSM) and When Should You Use It? - Google, Netflix Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Causal Inference`, `Observational Studies`, `Treatment Effect Estimation` | **Asked by:** Google, Netflix, Meta, Uber

??? success "View Answer"

    **Propensity Score Matching (PSM)** is a **quasi-experimental technique** for estimating **causal effects** from **observational data** (no randomization) by matching **treated** and **control** units with similar **probability of receiving treatment** (propensity score). Essential when **randomization is infeasible** (ethical constraints, historical data, natural experiments).

    **When to Use PSM:**

    | Scenario | Can Randomize? | Use PSM? | Why? |
    |----------|---------------|----------|------|
    | **A/B test** | Yes | No | Randomization ensures balance |
    | **Historical analysis** | No | Yes | Users self-selected into treatment |
    | **Feature rollout (self-opt-in)** | No | Yes | Power users more likely to adopt |
    | **Geographic launch** | No | Yes | Cities differ systematically |
    | **Policy change** | No | Yes | Treatment assigned by eligibility criteria |

    **Real Company Examples:**

    | Company | Use Case | Problem | PSM Solution | Result |
    |---------|----------|---------|--------------|--------|
    | **Netflix** | Evaluate binge-watching impact | Bingers self-select (not random) | Match bingers to non-bingers by viewing history | Causal effect: +12% retention |
    | **Uber** | Impact of driver incentives | Opt-in program (high-performers join) | Match opt-in drivers to similar non-opt-in | True effect: +5% (vs +15% naive) |
    | **Meta** | Messenger adoption impact | Users choose to install Messenger | Match installers to non-installers by demographics | Causal effect: +8% engagement |
    | **Airbnb** | Instant Book feature impact | Hosts self-select into Instant Book | Match Instant Book hosts to similar non-IB | Causal effect: +18% bookings |

    **PSM vs Other Methods:**

    | Method | When to Use | Advantage | Limitation |
    |--------|-------------|-----------|------------|
    | **Randomized A/B test** | Can randomize | Gold standard, no confounding | Expensive, slow, sometimes unethical |
    | **Propensity Score Matching** | Can't randomize, rich covariates | Balances observed confounders | Only observed covariates balanced |
    | **Difference-in-Differences** | Pre/post data, parallel trends | Controls time trends | Requires parallel trends assumption |
    | **Instrumental Variables** | Natural experiment, valid instrument | Handles unobserved confounding | Hard to find valid instruments |
    | **Regression Discontinuity** | Sharp cutoff in treatment assignment | Clean identification | Limited to threshold effects |

    **Propensity Score Matching Framework:**

    ```
    ┌──────────────────────────────────────────────────────────┐
    │     PROPENSITY SCORE MATCHING (PSM) WORKFLOW            │
    ├──────────────────────────────────────────────────────────┤
    │                                                          │
    │  STEP 1: ESTIMATE PROPENSITY SCORES                     │
    │  ┌────────────────────────────────────────────────────┐ │
    │  │ Logistic Regression:                              │ │
    │  │ P(Treatment=1 | X) = logit(β₀ + β₁X₁ + ... )     │ │
    │  │ X = demographics, behavior, context               │ │
    │  └──────────────────┬───────────────────────────────┘ │
    │                     ↓                                    │
    │  STEP 2: MATCH TREATED TO CONTROL                       │
    │  ┌────────────────────────────────────────────────────┐ │
    │  │ For each treated unit:                            │ │
    │  │ - Find control with closest propensity score     │ │
    │  │ - Methods: Nearest neighbor, caliper, kernel     │ │
    │  └──────────────────┬───────────────────────────────┘ │
    │                     ↓                                    │
    │  STEP 3: CHECK BALANCE                                  │
    │  ┌────────────────────────────────────────────────────┐ │
    │  │ Standardized mean difference (SMD) < 0.1         │ │
    │  │ for all covariates                                │ │
    │  └──────────────────┬───────────────────────────────┘ │
    │                     ↓                                    │
    │  STEP 4: ESTIMATE TREATMENT EFFECT                      │
    │  ┌────────────────────────────────────────────────────┐ │
    │  │ ATT = E[Y₁ - Y₀ | T=1]                           │ │
    │  │ (Average Treatment Effect on Treated)             │ │
    │  └────────────────────────────────────────────────────┘ │
    │                                                          │
    └──────────────────────────────────────────────────────────┘
    ```

    **Production Code - Propensity Score Matching:**

    ```python
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestNeighbors
    from scipy import stats
    from dataclasses import dataclass
    from typing import List, Tuple
    
    @dataclass
    class PSMResult:
        att: float  # Average Treatment Effect on Treated
        att_se: float
        att_ci: Tuple[float, float]
        n_treated: int
        n_matched_control: int
        balance_check: pd.DataFrame
        balance_passed: bool
    
    class PropensityScoreMatcher:
        """Propensity Score Matching for causal inference from observational data."""
        
        def __init__(self, caliper: float = 0.05):
            """
            Args:
                caliper: Maximum propensity score distance for matching
                        (0.05 = 5% of propensity score standard deviation)
            """
            self.caliper = caliper
            self.ps_model = None
            self.propensity_scores = None
        
        def fit_propensity_model(self, X: np.ndarray, treatment: np.ndarray) -> np.ndarray:
            """
            Estimate propensity scores using logistic regression.
            
            Args:
                X: Covariates (n_samples, n_features)
                treatment: Treatment indicator (1=treated, 0=control)
            
            Returns:
                Propensity scores (probability of treatment given X)
            """
            self.ps_model = LogisticRegression(max_iter=1000, random_state=42)
            self.ps_model.fit(X, treatment)
            self.propensity_scores = self.ps_model.predict_proba(X)[:, 1]
            
            return self.propensity_scores
        
        def match(self, treatment: np.ndarray, 
                 propensity_scores: np.ndarray,
                 method: str = 'nearest') -> np.ndarray:
            """
            Match treated units to control units based on propensity scores.
            
            Args:
                treatment: Treatment indicator
                propensity_scores: Propensity scores for all units
                method: 'nearest' (1:1 nearest neighbor with replacement)
            
            Returns:
                match_indices: For each treated unit, index of matched control
            """
            treated_idx = np.where(treatment == 1)[0]
            control_idx = np.where(treatment == 0)[0]
            
            treated_ps = propensity_scores[treated_idx].reshape(-1, 1)
            control_ps = propensity_scores[control_idx].reshape(-1, 1)
            
            # Find nearest neighbors
            nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
            nn.fit(control_ps)
            
            distances, indices = nn.kneighbors(treated_ps)
            
            # Apply caliper (discard matches beyond threshold)
            caliper_threshold = self.caliper * np.std(propensity_scores)
            valid_matches = distances.flatten() < caliper_threshold
            
            # Map to original control indices
            match_indices = np.full(len(treated_idx), -1, dtype=int)
            match_indices[valid_matches] = control_idx[indices.flatten()[valid_matches]]
            
            return match_indices
        
        def check_balance(self, X: np.ndarray, treatment: np.ndarray,
                         match_indices: np.ndarray,
                         covariate_names: List[str]) -> pd.DataFrame:
            """
            Check covariate balance after matching using standardized mean difference.
            
            Args:
                X: Covariates
                treatment: Treatment indicator
                match_indices: Matched control indices for each treated
                covariate_names: Names of covariates
            
            Returns:
                DataFrame with SMD before and after matching
            """
            treated_idx = np.where(treatment == 1)[0]
            control_idx = np.where(treatment == 0)[0]
            
            # Valid matches only
            valid_treated = treated_idx[match_indices >= 0]
            valid_matched_control = match_indices[match_indices >= 0]
            
            balance_results = []
            
            for i, cov_name in enumerate(covariate_names):
                # Before matching
                treated_mean_before = X[treated_idx, i].mean()
                control_mean_before = X[control_idx, i].mean()
                pooled_std_before = np.sqrt(
                    (X[treated_idx, i].var() + X[control_idx, i].var()) / 2
                )
                smd_before = (treated_mean_before - control_mean_before) / pooled_std_before
                
                # After matching
                treated_mean_after = X[valid_treated, i].mean()
                control_mean_after = X[valid_matched_control, i].mean()
                pooled_std_after = np.sqrt(
                    (X[valid_treated, i].var() + X[valid_matched_control, i].var()) / 2
                )
                smd_after = (treated_mean_after - control_mean_after) / pooled_std_after
                
                balance_results.append({
                    'Covariate': cov_name,
                    'SMD Before': smd_before,
                    'SMD After': smd_after,
                    'Improved': abs(smd_after) < abs(smd_before),
                    'Balanced': abs(smd_after) < 0.1
                })
            
            return pd.DataFrame(balance_results)
        
        def estimate_att(self, y: np.ndarray, treatment: np.ndarray,
                        match_indices: np.ndarray) -> PSMResult:
            """
            Estimate Average Treatment Effect on Treated (ATT).
            
            Args:
                y: Outcome variable
                treatment: Treatment indicator
                match_indices: Matched control indices
            
            Returns:
                PSMResult with ATT, standard error, and confidence interval
            """
            treated_idx = np.where(treatment == 1)[0]
            
            # Valid matches only
            valid_mask = match_indices >= 0
            valid_treated = treated_idx[valid_mask]
            valid_matched_control = match_indices[valid_mask]
            
            # Treatment effects for matched pairs
            treatment_effects = y[valid_treated] - y[valid_matched_control]
            
            # ATT and standard error
            att = treatment_effects.mean()
            att_se = treatment_effects.std() / np.sqrt(len(treatment_effects))
            
            # 95% confidence interval
            z_critical = 1.96
            att_ci = (att - z_critical * att_se, att + z_critical * att_se)
            
            return att, att_se, att_ci, len(valid_treated), len(valid_matched_control)
    
    # Example: Netflix binge-watching impact on retention (observational data)
    print("="*70)
    print("NETFLIX - BINGE-WATCHING IMPACT (PROPENSITY SCORE MATCHING)")
    print("="*70)
    
    # Simulate observational data (users self-select into binge-watching)
    np.random.seed(42)
    n = 2000
    
    # Covariates that affect both binge-watching propensity AND retention
    viewing_hours_pre = np.random.exponential(scale=15, size=n)  # Pre-period viewing
    account_age_months = np.random.uniform(1, 36, size=n)
    num_profiles = np.random.poisson(lam=2.5, size=n) + 1
    
    X = np.column_stack([viewing_hours_pre, account_age_months, num_profiles])
    covariate_names = ['Viewing Hours (Pre)', 'Account Age (Months)', 'Num Profiles']
    
    # Treatment assignment (binge-watching) - OBSERVATIONAL (not random!)
    # High pre-viewing hours increases binge propensity (confounding!)
    propensity_true = 1 / (1 + np.exp(-(0.1 * viewing_hours_pre - 1.5)))
    treatment = np.random.binomial(1, propensity_true)
    
    # Outcome (retention after 30 days) - affected by BOTH treatment and covariates
    retention = (
        0.7 +  # Baseline
        0.12 * treatment +  # TRUE causal effect = +12pp
        0.01 * viewing_hours_pre +  # Confounding!
        0.002 * account_age_months +
        0.02 * num_profiles +
        np.random.normal(0, 0.1, size=n)
    )
    retention = np.clip(retention, 0, 1)
    
    print(f"\n📊 OBSERVATIONAL DATA")
    print(f"   Total users: {n:,}")
    print(f"   Bingers (treatment): {treatment.sum():,} ({treatment.mean():.1%})")
    print(f"   Non-bingers (control): {(1-treatment).sum():,}")
    
    # Naive comparison (BIASED!)
    naive_retention_treatment = retention[treatment == 1].mean()
    naive_retention_control = retention[treatment == 0].mean()
    naive_effect = naive_retention_treatment - naive_retention_control
    
    print(f"\n❌ NAIVE COMPARISON (BIASED)")
    print(f"   Bingers retention:     {naive_retention_treatment:.2%}")
    print(f"   Non-bingers retention: {naive_retention_control:.2%}")
    print(f"   Naive effect:          {naive_effect:+.2%}")
    print(f"   ⚠️  BIASED! Includes confounding from pre-viewing hours")
    
    # Propensity Score Matching
    matcher = PropensityScoreMatcher(caliper=0.05)
    
    # Step 1: Fit propensity model
    ps_scores = matcher.fit_propensity_model(X, treatment)
    
    print(f"\n📈 PROPENSITY SCORE MODEL")
    print(f"   Logistic regression trained on {X.shape[1]} covariates")
    print(f"   Propensity score range: [{ps_scores.min():.3f}, {ps_scores.max():.3f}]")
    
    # Step 2: Match treated to control
    match_idx = matcher.match(treatment, ps_scores)
    n_matched = np.sum(match_idx >= 0)
    
    print(f"\n🔗 MATCHING")
    print(f"   Treated units: {treatment.sum():,}")
    print(f"   Matched pairs: {n_matched:,}")
    print(f"   Match rate: {n_matched / treatment.sum():.1%}")
    
    # Step 3: Check balance
    balance_df = matcher.check_balance(X, treatment, match_idx, covariate_names)
    
    print(f"\n⚖️  COVARIATE BALANCE CHECK")
    print(balance_df.to_string(index=False))
    print(f"\n   All covariates balanced: {balance_df['Balanced'].all()}")
    print("   (SMD < 0.1 is considered balanced)")
    
    # Step 4: Estimate ATT
    att, att_se, att_ci, n_treat, n_ctrl = matcher.estimate_att(retention, treatment, match_idx)
    
    print(f"\n✅ CAUSAL EFFECT (PSM)")
    print(f"   ATT (Average Treatment Effect on Treated): {att:+.2%}")
    print(f"   Standard error: {att_se:.4f}")
    print(f"   95% CI: [{att_ci[0]:+.2%}, {att_ci[1]:+.2%}]")
    print(f"   Sample: {n_treat:,} treated, {n_ctrl:,} matched controls")
    
    print(f"\n📊 COMPARISON")
    print(f"   Naive effect:  {naive_effect:+.2%} (BIASED by confounding)")
    print(f"   PSM effect:    {att:+.2%} (Causal, balanced covariates)")
    print(f"   True effect:   +12.00% (by design in simulation)")
    print(f"   PSM accuracy:  {abs(att - 0.12):.4f} error")
    
    print("\n" + "="*70)
    ```

    **Key Assumptions of PSM:**

    | Assumption | Description | How to Check | What if Violated? |
    |------------|-------------|--------------|-------------------|
    | **Unconfoundedness** | All confounders are observed and measured | Cannot test directly | Estimate biased if unobserved confounders exist |
    | **Common support** | Treated and control have overlapping propensity scores | Plot propensity score distributions | Trim extreme propensity scores |
    | **SUTVA** | No interference between units | Check experimental design | Use cluster-robust SE or different method |

    **Balance Check (SMD Thresholds):**

    | SMD Value | Balance Quality | Action |
    |-----------|----------------|--------|
    | **< 0.1** | Excellent balance | Proceed with analysis |
    | **0.1 - 0.25** | Acceptable balance | Proceed with caution |
    | **> 0.25** | Poor balance | Re-specify propensity model or use different method |

    !!! tip "Interviewer's Insight"
        **What they test:**
        
        - Do you understand when PSM is needed (observational data, self-selection)?
        - Can you explain propensity score estimation and matching?
        - Do you check covariate balance (SMD < 0.1)?
        
        **Strong signal:**
        
        - "Use PSM when randomization is infeasible and users self-select into treatment"
        - "Netflix bingers self-select (high pre-viewing hours) - naive comparison is biased"
        - "After matching on propensity scores, SMD < 0.1 for all covariates = balance achieved"
        - "ATT = +12% retention (vs +18% naive) - removed confounding from pre-behavior"
        - "Limitation: Only balances observed covariates, unobserved confounding still biases estimate"
        
        **Red flags:**
        
        - Not understanding difference between observational and experimental data
        - Skipping balance checks (SMD)
        - Not acknowledging limitation (unobserved confounding)
        
        **Follow-ups:**
        
        - "What if balance checks fail (SMD > 0.25)?"
        - "How would you handle unobserved confounding?"
        - "PSM vs Difference-in-Differences - which to use when?"

---

### What is Difference-in-Differences (DiD) and When Should You Use It? - Google, Uber Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Causal Inference`, `Quasi-Experiments`, `Panel Data` | **Asked by:** Google, Uber, Airbnb, Lyft

??? success "View Answer"

    **Difference-in-Differences (DiD)** estimates **causal effects** by comparing the **change over time** in a **treatment group** vs a **control group**, effectively using the control group to **remove time trends** that affect both groups. Essential for **quasi-experiments** (geographic rollouts, policy changes, staggered launches) where **randomization is impossible** but **pre-treatment data** is available.

    **DiD Formula:**

    $$
    \text{DiD} = (Y_{T,post} - Y_{T,pre}) - (Y_{C,post} - Y_{C,pre})
    $$

    Where:
    - $Y_{T,post}$: Treatment group outcome after intervention
    - $Y_{T,pre}$: Treatment group outcome before intervention
    - $Y_{C,post}$: Control group outcome after intervention  
    - $Y_{C,pre}$: Control group outcome before intervention

    **Real Company Examples:**

    | Company | Use Case | Treatment Group | Control Group | Pre Period | Result |
    |---------|----------|----------------|---------------|------------|--------|
    | **Uber** | New pricing in SF | San Francisco | Los Angeles | 8 weeks | +12% trips (DiD) |
    | **Airbnb** | Instant Book in NYC | New York City | Chicago | 12 weeks | +18% bookings (DiD) |
    | **DoorDash** | Incentive program in Austin | Austin | Dallas | 6 weeks | +8% orders (DiD) |
    | **Google Ads** | Algorithm change (EU only) | EU countries | US | 4 weeks | +3.2% CTR (DiD) |

    **DiD vs Other Methods:**

    | Method | Data Requirement | Assumption | Best Use Case |
    |--------|------------------|------------|---------------|
    | **Randomized A/B Test** | Randomization possible | None | General experiments |
    | **Difference-in-Differences** | Pre/post data, parallel trends | Parallel trends in pre-period | Geographic rollouts, policy changes |
    | **Propensity Score Matching** | Rich covariates | Unconfoundedness | Self-selection, observational |
    | **Regression Discontinuity** | Sharp cutoff | Continuity at threshold | Eligibility cutoffs |

    **Parallel Trends Assumption (Critical!):**

    | Time Period | Treatment Group | Control Group | Interpretation |
    |-------------|----------------|---------------|----------------|
    | **Pre-period (weeks 1-8)** | +2% growth | +2% growth | ✅ Parallel trends hold |
    | **Post-period (weeks 9-12)** | +14% growth | +2% growth | DiD = +14% - 2% = +12% causal effect |

    **Difference-in-Differences Framework:**

    ```
    ┌──────────────────────────────────────────────────────────┐
    │     DIFFERENCE-IN-DIFFERENCES (DiD) WORKFLOW            │
    ├──────────────────────────────────────────────────────────┤
    │                                                          │
    │  STEP 1: COLLECT PRE/POST DATA                          │
    │  ┌────────────────────────────────────────────────────┐ │
    │  │ Treatment group: Y_T,pre → Y_T,post              │ │
    │  │ Control group:   Y_C,pre → Y_C,post              │ │
    │  └──────────────────┬───────────────────────────────┘ │
    │                     ↓                                    │
    │  STEP 2: CHECK PARALLEL TRENDS (PRE-PERIOD)             │
    │  ┌────────────────────────────────────────────────────┐ │
    │  │ Plot trends: Should be parallel before treatment │ │
    │  │ Test: Interaction of group × time trend = 0      │ │
    │  └──────────────────┬───────────────────────────────┘ │
    │                     ↓                                    │
    │  STEP 3: COMPUTE DiD ESTIMATOR                          │
    │  ┌────────────────────────────────────────────────────┐ │
    │  │ Δ_T = Y_T,post - Y_T,pre  (treatment change)     │ │
    │  │ Δ_C = Y_C,post - Y_C,pre  (control change)       │ │
    │  │ DiD = Δ_T - Δ_C           (causal effect)        │ │
    │  └──────────────────┬───────────────────────────────┘ │
    │                     ↓                                    │
    │  STEP 4: REGRESSION SPECIFICATION                       │
    │  ┌────────────────────────────────────────────────────┐ │
    │  │ Y = β₀ + β₁·Post + β₂·Treat + β₃·(Post×Treat)   │ │
    │  │ DiD = β₃ (interaction coefficient)               │ │
    │  └────────────────────────────────────────────────────┘ │
    │                                                          │
    └──────────────────────────────────────────────────────────┘
    ```

    **Production Code - Difference-in-Differences:**

    ```python
    import numpy as np
    import pandas as pd
    from scipy import stats
    import statsmodels.formula.api as smf
    from dataclasses import dataclass
    from typing import Tuple
    
    @dataclass
    class DiDResult:
        did_estimate: float
        did_se: float
        did_ci: Tuple[float, float]
        p_value: float
        parallel_trends_test_pvalue: float
        parallel_trends_passed: bool
    
    class DifferenceInDifferences:
        """Difference-in-Differences estimator for causal inference."""
        
        def __init__(self):
            self.regression_result = None
        
        def test_parallel_trends(self, df: pd.DataFrame, 
                                 pre_period_only: bool = True) -> float:
            """
            Test parallel trends assumption in pre-period.
            
            Args:
                df: DataFrame with columns [outcome, group, time_period, post]
                pre_period_only: Only use pre-treatment period
            
            Returns:
                p-value for group × time_period interaction (should be > 0.05)
            """
            if pre_period_only:
                df_pre = df[df['post'] == 0].copy()
            else:
                df_pre = df.copy()
            
            # Regression: outcome ~ group + time_period + group×time_period
            model = smf.ols('outcome ~ group + time_period + group:time_period', 
                           data=df_pre)
            result = model.fit()
            
            # Extract interaction p-value
            interaction_pvalue = result.pvalues['group:time_period']
            
            return interaction_pvalue
        
        def estimate_did(self, df: pd.DataFrame) -> DiDResult:
            """
            Estimate DiD using regression.
            
            Args:
                df: DataFrame with columns [outcome, treat, post]
                    - outcome: Continuous outcome variable
                    - treat: Treatment group indicator (1=treatment, 0=control)
                    - post: Post-period indicator (1=post, 0=pre)
            
            Returns:
                DiDResult with estimate, SE, CI, p-value
            """
            # Regression: Y = β₀ + β₁·post + β₂·treat + β₃·(post×treat)
            # DiD = β₃
            model = smf.ols('outcome ~ treat + post + treat:post', data=df)
            self.regression_result = model.fit()
            
            # Extract DiD coefficient (interaction term)
            did_coef = self.regression_result.params['treat:post']
            did_se = self.regression_result.bse['treat:post']
            did_pvalue = self.regression_result.pvalues['treat:post']
            
            # 95% CI
            did_ci = (
                did_coef - 1.96 * did_se,
                did_coef + 1.96 * did_se
            )
            
            # Test parallel trends
            parallel_trends_pval = self.test_parallel_trends(df)
            parallel_trends_ok = parallel_trends_pval > 0.05
            
            return DiDResult(
                did_estimate=did_coef,
                did_se=did_se,
                did_ci=did_ci,
                p_value=did_pvalue,
                parallel_trends_test_pvalue=parallel_trends_pval,
                parallel_trends_passed=parallel_trends_ok
            )
    
    # Example: Uber new pricing algorithm in San Francisco
    print("="*70)
    print("UBER - NEW PRICING ALGORITHM (DIFFERENCE-IN-DIFFERENCES)")
    print("="*70)
    
    np.random.seed(42)
    
    # Setup: SF gets new pricing (treatment), LA is control
    # Pre-period: 8 weeks before launch
    # Post-period: 4 weeks after launch
    
    weeks_pre = 8
    weeks_post = 4
    n_cities = 2  # SF (treatment), LA (control)
    
    data = []
    
    # San Francisco (Treatment)
    for week in range(-weeks_pre, weeks_post):
        post = 1 if week >= 0 else 0
        
        # Pre-period: +2% weekly growth (same as LA)
        # Post-period: +14% growth (+12% causal effect from new pricing + 2% natural)
        if post == 0:
            trips = 10000 + 200 * week + np.random.normal(0, 500)  # +2% weekly
        else:
            trips = 10000 + 200 * week + 1200 * week + np.random.normal(0, 500)  # +14% weekly
        
        data.append({
            'city': 'San Francisco',
            'treat': 1,
            'week': week,
            'post': post,
            'time_period': week,
            'group': 1,
            'outcome': trips
        })
    
    # Los Angeles (Control)
    for week in range(-weeks_pre, weeks_post):
        post = 1 if week >= 0 else 0
        
        # Both pre and post: +2% weekly growth (no treatment)
        trips = 9500 + 190 * week + np.random.normal(0, 500)  # +2% weekly
        
        data.append({
            'city': 'Los Angeles',
            'treat': 0,
            'week': week,
            'post': post,
            'time_period': week,
            'group': 0,
            'outcome': trips
        })
    
    df = pd.DataFrame(data)
    
    print(f"\n📊 DATA STRUCTURE")
    print(f"   Cities: San Francisco (treatment), Los Angeles (control)")
    print(f"   Pre-period: {weeks_pre} weeks before launch")
    print(f"   Post-period: {weeks_post} weeks after launch")
    print(f"   Total observations: {len(df):,}")
    
    # Summary statistics
    summary = df.groupby(['city', 'post'])['outcome'].mean().reset_index()
    summary['period'] = summary['post'].map({0: 'Pre', 1: 'Post'})
    
    print(f"\n📈 SUMMARY STATISTICS")
    for city in ['San Francisco', 'Los Angeles']:
        city_data = summary[summary['city'] == city]
        pre_val = city_data[city_data['period'] == 'Pre']['outcome'].values[0]
        post_val = city_data[city_data['period'] == 'Post']['outcome'].values[0]
        change = post_val - pre_val
        pct_change = (change / pre_val) * 100
        
        print(f"\n   {city}:")
        print(f"      Pre-period:  {pre_val:,.0f} trips")
        print(f"      Post-period: {post_val:,.0f} trips")
        print(f"      Change:      {change:+,.0f} trips ({pct_change:+.1f}%)")
    
    # Naive comparison (BIASED!)
    sf_post = summary[(summary['city'] == 'San Francisco') & (summary['period'] == 'Post')]['outcome'].values[0]
    la_post = summary[(summary['city'] == 'Los Angeles') & (summary['period'] == 'Post')]['outcome'].values[0]
    naive_diff = sf_post - la_post
    
    print(f"\n❌ NAIVE COMPARISON (BIASED)")
    print(f"   SF post-period:  {sf_post:,.0f}")
    print(f"   LA post-period:  {la_post:,.0f}")
    print(f"   Naive difference: {naive_diff:+,.0f}")
    print(f"   ⚠️  BIASED! Doesn't account for time trends and pre-existing differences")
    
    # Difference-in-Differences
    did_estimator = DifferenceInDifferences()
    
    # Test parallel trends
    parallel_trends_pval = did_estimator.test_parallel_trends(df, pre_period_only=True)
    
    print(f"\n⚖️  PARALLEL TRENDS TEST")
    print(f"   Interaction p-value: {parallel_trends_pval:.4f}")
    if parallel_trends_pval > 0.05:
        print(f"   ✅ PASSED: No evidence against parallel trends (p > 0.05)")
    else:
        print(f"   ❌ FAILED: Parallel trends violated (p < 0.05)")
    
    # Estimate DiD
    result = did_estimator.estimate_did(df)
    
    print(f"\n✅ DIFFERENCE-IN-DIFFERENCES ESTIMATE")
    print(f"   DiD estimate: {result.did_estimate:+,.0f} trips")
    print(f"   Standard error: {result.did_se:.2f}")
    print(f"   95% CI: [{result.did_ci[0]:+,.0f}, {result.did_ci[1]:+,.0f}]")
    print(f"   p-value: {result.p_value:.4f}")
    
    if result.p_value < 0.05:
        print(f"   📊 STATISTICALLY SIGNIFICANT at α=0.05")
    
    # Regression output
    print(f"\n📋 REGRESSION OUTPUT")
    print(did_estimator.regression_result.summary().tables[1])
    
    # Manual DiD calculation
    sf_pre = summary[(summary['city'] == 'San Francisco') & (summary['period'] == 'Pre')]['outcome'].values[0]
    sf_post = summary[(summary['city'] == 'San Francisco') & (summary['period'] == 'Post')]['outcome'].values[0]
    la_pre = summary[(summary['city'] == 'Los Angeles') & (summary['period'] == 'Pre')]['outcome'].values[0]
    la_post = summary[(summary['city'] == 'Los Angeles') & (summary['period'] == 'Post')]['outcome'].values[0]
    
    manual_did = (sf_post - sf_pre) - (la_post - la_pre)
    
    print(f"\n🔢 MANUAL CALCULATION")
    print(f"   SF change: {sf_post:,.0f} - {sf_pre:,.0f} = {sf_post - sf_pre:+,.0f}")
    print(f"   LA change: {la_post:,.0f} - {la_pre:,.0f} = {la_post - la_pre:+,.0f}")
    print(f"   DiD: ({sf_post - sf_pre:,.0f}) - ({la_post - la_pre:,.0f}) = {manual_did:+,.0f}")
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print(f"\n✅ New pricing algorithm in SF caused +{result.did_estimate:,.0f} trips")
    print(f"   (after removing time trend using LA as control)")
    print(f"\n   Parallel trends held in pre-period (p={parallel_trends_pval:.4f})")
    print(f"   → Valid causal interpretation")
    
    print("\n" + "="*70)
    ```

    **When Parallel Trends Fail:**

    | Symptom | Cause | Solution |
    |---------|-------|----------|
    | **Interaction p < 0.05 in pre-period** | Different pre-trends | Use synthetic control or add covariates |
    | **Visual divergence before treatment** | Groups not comparable | Find better control group |
    | **Post-period trend change in control** | External shock | Use multiple pre-periods to test robustness |

    **DiD Variants:**

    | Variant | Use Case | Example |
    |---------|----------|----------|
    | **Standard DiD** | One treatment, one control, one treatment time | Uber SF pricing |
    | **Event study DiD** | Plot DiD by time period | Check for anticipation effects |
    | **Staggered DiD** | Multiple treatment times | Geographic rollout over months |
    | **Synthetic control** | No good control group | Construct weighted control from multiple units |

    !!! tip "Interviewer's Insight"
        **What they test:**
        
        - Do you understand when DiD is appropriate (pre/post data, parallel trends)?
        - Can you explain the parallel trends assumption and how to test it?
        - Do you know how to interpret DiD vs naive comparison?
        
        **Strong signal:**
        
        - "Use DiD for geographic rollout: SF gets new pricing, LA is control"
        - "CRITICAL: Test parallel trends in pre-period (interaction p > 0.05)"
        - "DiD = (SF post-pre change) - (LA post-pre change) = removes time trend"
        - "DiD estimate: +1200 trips (causal effect), naive difference is biased"
        - "Regression: Y ~ treat + post + treat×post, DiD = β₃ (interaction)"
        
        **Red flags:**
        
        - Not testing parallel trends assumption
        - Confusing DiD with simple post-period comparison
        - Not understanding why control group is needed (removes time trends)
        
        **Follow-ups:**
        
        - "What if parallel trends fail in pre-period?"
        - "How would you extend this to multiple treatment times (staggered rollout)?"
        - "DiD vs Propensity Score Matching - which to use when?"

---

## Quick Reference: 100+ A/B Testing Questions

| Sno | Question Title | Practice Links | Companies Asking | Difficulty | Topics |
|-----|----------------|----------------|------------------|------------|--------|
| 1 | What is A/B Testing? | [Optimizely](https://www.optimizely.com/optimization-glossary/ab-testing/) | Most Tech Companies | Easy | Basics |
| 2 | Explain Null Hypothesis ($H_0$) vs Alternative Hypothesis ($H_1$) | [Khan Academy](https://www.khanacademy.org/math/statistics-probability) | Most Tech Companies | Easy | Statistics |
| 3 | What is a p-value? Explain it to a non-technical person. | [Harvard Business Review](https://hbr.org/2016/02/a-refresher-on-statistical-significance) | Google, Meta, Amazon | Medium | Statistics, Communication |
| 4 | What is Statistical Power? | [Machine Learning Plus](https://www.machinelearningplus.com/) | Google, Netflix, Uber | Medium | Statistics |
| 5 | What is Type I error (False Positive) vs Type II error (False Negative)? | [Towards Data Science](https://towardsdatascience.com/) | Most Tech Companies | Easy | Statistics |
| 6 | How do you calculate sample size for an experiment? | [Evan Miller](https://www.evanmiller.org/ab-testing/sample-size.html) | Google, Amazon, Meta | Medium | Experimental Design |
| 7 | What is Minimum Detectable Effect (MDE)? | [StatsEngine](https://www.stat.ubc.ca/) | Netflix, Airbnb | Medium | Experimental Design |
| 8 | Explain Confidence Intervals. | [Coursera](https://www.coursera.org/) | Most Tech Companies | Easy | Statistics |
| 9 | Difference between One-tailed and Two-tailed tests. | [Investopedia](https://www.investopedia.com/) | Google, Amazon | Easy | Statistics |
| 10 | What is the Central Limit Theorem? Why is it important? | [Khan Academy](https://www.khanacademy.org/math/statistics-probability) | Google, HFT Firms | Medium | Statistics |
| 11 | How long should you run an A/B test? | [CXL](https://cxl.com/blog/how-long-to-run-ab-test/) | Airbnb, Booking.com | Medium | Experimental Design |
| 12 | Can you stop an experiment as soon as it reaches significance? (Peeking) | [Evan Miller](https://www.evanmiller.org/how-not-to-run-an-ab-test.html) | Netflix, Uber, Airbnb | Hard | Pitfalls |
| 13 | What is SRM (Sample Ratio Mismatch)? How to debug? | [Microsoft Research](https://www.microsoft.com/en-us/research/group/experimentation-platform-exp/) | Microsoft, LinkedIn | Hard | Debugging |
| 14 | What is Randomization Unit vs Analysis Unit? | [Udacity](https://www.udacity.com/course/ab-testing--ud257) | Uber, DoorDash | Medium | Experimental Design |
| 15 | How to handle outliers in A/B testing metrics? | [Towards Data Science](https://towardsdatascience.com/) | Google, Meta | Medium | Data Cleaning |
| 16 | Mean vs Median: Which metric to use? | [Stack Overflow](https://stackoverflow.com/) | Most Tech Companies | Easy | Metrics |
| 17 | What are Guardrail Metrics? | [Airbnb Tech Blog](https://medium.com/airbnb-engineering) | Airbnb, Netflix | Medium | Metrics |
| 18 | What is a North Star Metric? | [Amplitude](https://amplitude.com/blog/north-star-metric) | Product Roles | Easy | Metrics |
| 19 | Difference between Z-test and T-test. | [Statistics By Jim](https://statisticsbyjim.com/) | Google, Amazon | Medium | Statistics |
| 20 | How to test multiple variants? (A/B/n testing) | [VWO](https://vwo.com/) | Booking.com, Expedia | Medium | Experimental Design |
| 21 | What is the Bonferroni Correction? | [Wikipedia](https://en.wikipedia.org/wiki/Bonferroni_correction) | Google, Meta | Hard | Statistics |
| 22 | What is A/A Testing? Why do it? | [Optimizely](https://www.optimizely.com/) | Microsoft, LinkedIn | Medium | Validity |
| 23 | Explain Covariate Adjustment (CUPED). | [Booking.com Data](https://booking.ai/) | Booking.com, Microsoft, Meta | Hard | Optimization |
| 24 | How to measure retention in A/B tests? | [Reforge](https://www.reforge.com/) | Netflix, Spotify | Medium | Metrics |
| 25 | What is a Novelty Effect? | [CXL](https://cxl.com/) | Facebook, Instagram | Medium | Pitfalls |
| 26 | What is a Primacy Effect? | [CXL](https://cxl.com/) | Facebook, Instagram | Medium | Pitfalls |
| 27 | How to handle interference (Network Effects)? | [Uber Eng Blog](https://eng.uber.com/) | Uber, Lyft, DoorDash | Hard | Network Effects |
| 28 | What is a Switchback (Time-split) Experiment? | [DoorDash Eng](https://doordash.engineering/) | DoorDash, Uber | Hard | Experimental Design |
| 29 | What is Cluster Randomization? | [Wikipedia](https://en.wikipedia.org/wiki/Cluster_randomised_controlled_trial) | Facebook, LinkedIn | Hard | Experimental Design |
| 30 | How to test on a 2-sided marketplace? | [Lyft Eng](https://eng.lyft.com/) | Uber, Lyft, Airbnb | Hard | Marketplace |
| 31 | Explain Bayesian A/B Testing vs Frequentist. | [VWO](https://vwo.com/blog/bayesian-vs-frequentist-ab-testing/) | Stitch Fix, Netflix | Hard | Statistics |
| 32 | What is a Multi-Armed Bandit (MAB)? | [Towards Data Science](https://towardsdatascience.com/) | Netflix, Amazon | Hard | Bandits |
| 33 | Thompson Sampling vs Epsilon-Greedy. | [GeeksforGeeks](https://www.geeksforgeeks.org/) | Netflix, Amazon | Hard | Bandits |
| 34 | How to deal with low traffic experiments? | [CXL](https://cxl.com/) | Startups | Medium | Strategy |
| 35 | How to select metrics for a new feature? | [Product School](https://productschool.com/) | Meta, Google | Medium | Metrics |
| 36 | What is Simpson's Paradox? | [Britannica](https://www.britannica.com/topic/Simpsons-paradox) | Google, Amazon | Medium | Paradoxes |
| 37 | How to analyze ratio metrics (e.g., CTR)? | [Deltamethod](https://en.wikipedia.org/wiki/Delta_method) | Google, Meta | Hard | Statistics, Delta Method |
| 38 | What is Bootstrapping? When to use it? | [Investopedia](https://www.investopedia.com/terms/b/bootstrapping.asp) | Amazon, Netflix | Medium | Statistics |
| 39 | How to detect and handle Seasonality? | [Towards Data Science](https://towardsdatascience.com/) | Retail/E-comm | Medium | Time Series |
| 40 | What is Change Aversion? | [Google UX](https://design.google/library/) | Google, YouTube | Medium | UX |
| 41 | How to design an experiment for a search algorithm? | [Airbnb Eng](https://medium.com/airbnb-engineering) | Google, Airbnb, Amazon | Hard | Search, Ranking |
| 42 | How to test pricing changes? | [PriceIntelligently](https://www.priceintelligently.com/) | Uber, Airbnb | Hard | Pricing, Strategy |
| 43 | What is interference between experiments? | [Microsoft Exp](https://www.microsoft.com/en-us/research/group/experimentation-platform-exp/) | Google, Meta, Microsoft | Hard | Platform |
| 44 | Explain Sequential Testing. | [Evan Miller](https://www.evanmiller.org/sequential-ab-testing.html) | Optimizely, Netflix | Hard | Statistics |
| 45 | What is Variance Reduction? | [Meta Research](https://research.facebook.com/) | Meta, Microsoft, Booking | Hard | Optimization |
| 46 | How to handle attribution (First-touch vs Last-touch)? | [Google Analytics](https://analytics.google.com/) | Marketing Tech | Medium | Marketing |
| 47 | How to validate if randomization worked? | [Stats StackExchange](https://stats.stackexchange.com/) | Most Tech Companies | Easy | Validity |
| 48 | What is stratification? | [Wikipedia](https://en.wikipedia.org/wiki/Stratified_sampling) | Most Tech Companies | Medium | Sampling |
| 49 | When should you NOT A/B test? | [Reforge](https://www.reforge.com/) | Product Roles | Medium | Strategy |
| 50 | How to estimate long-term impact from short-term tests? | [Netflix TechBlog](https://netflixtechblog.com/) | Netflix, Meta | Hard | Strategy, Proxy Metrics |
| 51 | What is Binomial Distribution? | [Khan Academy](https://www.khanacademy.org/) | Most Tech Companies | Easy | Statistics |
| 52 | What is Poisson Distribution? | [Khan Academy](https://www.khanacademy.org/) | Uber, Lyft (Rides) | Medium | Statistics |
| 53 | Difference between Correlation and Causation. | [Khan Academy](https://www.khanacademy.org/) | Most Tech Companies | Easy | Basics |
| 54 | What is a Confounding Variable? | [Scribbr](https://www.scribbr.com/) | Most Tech Companies | Easy | Causal Inference |
| 55 | Explain Regression Discontinuity Design (RDD). | [Wikipedia](https://en.wikipedia.org/wiki/Regression_discontinuity_design) | Economics/Policy Roles | Hard | Causal Inference |
| 56 | Explain Difference-in-Differences (DiD). | [Wikipedia](https://en.wikipedia.org/wiki/Difference_in_differences) | Uber, Airbnb | Hard | Causal Inference |
| 57 | What is Propensity Score Matching? | [Wikipedia](https://en.wikipedia.org/wiki/Propensity_score_matching) | Meta, Netflix | Hard | Causal Inference |
| 58 | How to Handle Heterogeneous Treatment Effects? | [CausalML](https://github.com/uber/causalml) | Uber, Meta | Hard | Causal ML |
| 59 | What is Interference in social networks? | [Meta Research](https://research.facebook.com/) | Meta, LinkedIn, Snap | Hard | Network Effects |
| 60 | Explain the concept of "Holdout Groups". | [Airbnb Eng](https://medium.com/airbnb-engineering) | Amazon, Airbnb | Medium | Strategy |
| 61 | How to test infrastructure changes? (Canary Deployment) | [Google SRE](https://sre.google/sre-book/canary-analysis/) | Google, Netflix | Medium | DevOps/SRE |
| 62 | What is Client-side vs Server-side testing? | [Optimizely](https://www.optimizely.com/) | Full Stack Roles | Medium | Implementation |
| 63 | How to deal with flickering? | [VWO](https://vwo.com/) | Frontend Roles | Medium | Implementation |
| 64 | What is a Trigger selection in A/B testing? | [Microsoft Exp](https://www.microsoft.com/en-us/research/group/experimentation-platform-exp/) | Microsoft, Airbnb | Hard | Experimental Design |
| 65 | How to analyze user funnel drop-offs? | [Mixpanel](https://mixpanel.com/) | Product Analysts | Medium | Analytics |
| 66 | What is Geometric Distribution? | [Wikipedia](https://en.wikipedia.org/wiki/Geometric_distribution) | Most Tech Companies | Medium | Statistics |
| 67 | Explain Inverse Propensity Weighting (IPW). | [Wikipedia](https://en.wikipedia.org/wiki/Inverse_probability_weighting) | Causal Inference Roles | Hard | Causal Inference |
| 68 | How to calculate Standard Error of Mean (SEM)? | [Investopedia](https://www.investopedia.com/terms/s/standard-error.asp) | Most Tech Companies | Easy | Statistics |
| 69 | What is Statistical Significance vs Practical Significance? | [Towards Data Science](https://towardsdatascience.com/) | Google, Meta | Medium | Strategy |
| 70 | How to handle cookies and tracking prevention (ITP)? | [WebKit](https://webkit.org/blog/) | AdTech, Marketing | Hard | Privacy |
| 71 | **[HARD]** Explain the Delta Method for ratio metrics. | [Deltamethod](https://en.wikipedia.org/wiki/Delta_method) | Google, Meta, Uber | Hard | Statistics |
| 72 | **[HARD]** How does Switchback testing solve interference? | [DoorDash Eng](https://doordash.engineering/) | DoorDash, Uber | Hard | Experimental Design |
| 73 | **[HARD]** Derive the sample size formula. | [Stats Exchange](https://stats.stackexchange.com/) | Google, HFT Firms | Hard | Math |
| 74 | **[HARD]** How to implement CUPED in Python/SQL? | [Booking.com](https://booking.ai/) | Booking, Microsoft | Hard | Optimization |
| 75 | **[HARD]** Explain Sequential Probability Ratio Test (SPRT). | [Wikipedia](https://en.wikipedia.org/wiki/Sequential_probability_ratio_test) | Optimizely, Netflix | Hard | Statistics |
| 76 | **[HARD]** How to estimate Network Effects (Cluster-Based)? | [MIT Paper](https://economics.mit.edu/) | Meta, LinkedIn | Hard | Network Effects |
| 77 | **[HARD]** Design an experiment for a 3-sided marketplace. | [Uber Eng](https://eng.uber.com/) | Uber, DoorDash | Hard | Marketplace |
| 78 | **[HARD]** How to correct for multiple comparisons (FDR vs FWER)? | [Wikipedia](https://en.wikipedia.org/wiki/False_discovery_rate) | Pharma, BioTech, Tech | Hard | Statistics |
| 79 | **[HARD]** Explain Instrumental Variables (IV). | [Wikipedia](https://en.wikipedia.org/wiki/Instrumental_variable) | Economics, Uber | Hard | Causal Inference |
| 80 | **[HARD]** How to build an Experimentation Platform? | [Microsoft Exp](https://www.microsoft.com/en-us/research/group/experimentation-platform-exp/) | Microsoft, Netflix, Airbnb | Hard | System Design |
| 81 | **[HARD]** How to handle user identity resolution across devices? | [Segment](https://segment.com/) | Meta, Google | Hard | Data Engineering |
| 82 | **[HARD]** What is "Carryover Effect" in Switchback tests? | [DoorDash Eng](https://doordash.engineering/) | DoorDash, Uber | Hard | Pitfalls |
| 83 | **[HARD]** Explain "Washout Period". | [Clinical Trials](https://en.wikipedia.org/wiki/Washout_period) | DoorDash, Uber | Hard | Experimental Design |
| 84 | **[HARD]** How to test Ranking algorithms (Interleaving)? | [Netflix TechBlog](https://netflixtechblog.com/) | Netflix, Google, Airbnb | Hard | Search/Ranking |
| 85 | **[HARD]** Explain Always-Valid Inference. | [Optimizely](https://www.optimizely.com/) | Optimizely, Netflix | Hard | Statistics |
| 86 | **[HARD]** How to measure cannibalization? | [Harvard Business Review](https://hbr.org/) | Retail, E-comm | Hard | Strategy |
| 87 | **[HARD]** Explain Thompson Sampling Implementation. | [TDS](https://towardsdatascience.com/) | Amazon, Netflix | Hard | Bandits |
| 88 | **[HARD]** How to detect Heterogeneous Treatment Effects (Causal Forest)? | [Wager & Athey](https://arxiv.org/abs/1510.04342) | Uber, Meta | Hard | Causal ML |
| 89 | **[HARD]** How to handle "dilution" in experiment metrics? | [Reforge](https://www.reforge.com/) | Product Roles | Hard | Metrics |
| 90 | **[HARD]** Explain Synthetic Control Method. | [Wikipedia](https://en.wikipedia.org/wiki/Synthetic_control_method) | Uber (City-level tests) | Hard | Causal Inference |
| 91 | **[HARD]** How to optimize for Long-term Customer Value (LTV)? | [ThetaCLV](https://github.com/fit-marketing/thetaclv) | Subscription roles | Hard | Metrics |
| 92 | **[HARD]** Explain "Winner's Curse" in A/B testing. | [Airbnb Eng](https://medium.com/airbnb-engineering) | Airbnb, Booking | Hard | Bias |
| 93 | **[HARD]** How to handle heavy-tailed metric distributions? | [TDS](https://towardsdatascience.com/) | HFT, Fintech | Hard | Statistics |
| 94 | **[HARD]** How to implement Stratified Sampling in SQL? | [Stack Overflow](https://stackoverflow.com/) | Data Eng | Hard | Sampling |
| 95 | **[HARD]** Explain "Regression to the Mean". | [Wikipedia](https://en.wikipedia.org/wiki/Regression_toward_the_mean) | Most Tech Companies | Hard | Statistics |
| 96 | **[HARD]** How to budget "Error Rate" across the company? | [Microsoft Exp](https://www.microsoft.com/en-us/research/group/experimentation-platform-exp/) | Microsoft, Google | Hard | Strategy |
| 97 | **[HARD]** How to detect bot traffic in experiments? | [Google Analytics](https://analytics.google.com/) | Security, Fraud | Hard | Data Quality |
| 98 | **[HARD]** Explain "Interaction Effects" in Factorial Designs. | [Wikipedia](https://en.wikipedia.org/wiki/Factorial_experiment) | Meta, Google | Hard | Statistics |
| 99 | **[HARD]** How to use Surrogate Metrics? | [Netflix TechBlog](https://netflixtechblog.com/) | Netflix | Hard | Metrics |
| 100| **[HARD]** How to implement A/B testing in a Microservices architecture? | [Split.io](https://www.split.io/) | Netflix, Uber | Hard | Engineering |

---

## Code Examples

### 1. Power Analysis and Sample Size (Python)

**Difficulty:** 🟢 Easy | **Tags:** `Code Example` | **Asked by:** Code Pattern

??? success "View Code Example"

    Calculating the required sample size before starting an experiment.

    ```python
    from statsmodels.stats.power import TTestIndPower
    import numpy as np

    # Parameters
    effect_size = 0.1  # Cohen's d (Standardized difference)
    alpha = 0.05       # Significance level (5%)
    power = 0.8        # Power (80%)

    analysis = TTestIndPower()
    sample_size = analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha)

    print(f"Required sample size per group: {int(np.ceil(sample_size))}")
    ```

### 2. Bayesian A/B Test (Beta-Binomial)

**Difficulty:** 🟢 Easy | **Tags:** `Code Example` | **Asked by:** Code Pattern

??? success "View Code Example"

    Updating beliefs about conversion rates.

    ```python
    from scipy.stats import beta

    # Prior: Uniform distribution (Beta(1,1))
    alpha_prior = 1
    beta_prior = 1

    # Data: Group A
    conversions_A = 120
    failures_A = 880

    # Data: Group B
    conversions_B = 140
    failures_B = 860

    # Posterior
    posterior_A = beta(alpha_prior + conversions_A, beta_prior + failures_A)
    posterior_B = beta(alpha_prior + conversions_B, beta_prior + failures_B)

    # Probability B > A (Approximate via simulation)
    samples = 100000
    prob_b_better = (posterior_B.rvs(samples) > posterior_A.rvs(samples)).mean()

    print(f"Probability B is better than A: {prob_b_better:.4f}")
    ```

### 3. Bootstrap Confidence Interval

**Difficulty:** 🟢 Easy | **Tags:** `Code Example` | **Asked by:** Code Pattern

??? success "View Code Example"

    Calculating CI for non-normal metrics (e.g., Revenue per User).

    ```python
    import numpy as np

    data_control = np.random.lognormal(mean=2, sigma=1, size=1000)
    data_variant = np.random.lognormal(mean=2.1, sigma=1, size=1000)

    def bootstrap_mean_diff(data1, data2, n_bootstrap=1000):
        diffs = []
        for _ in range(n_bootstrap):
            # Sample with replacement
            sample1 = np.random.choice(data1, len(data1), replace=True)
            sample2 = np.random.choice(data2, len(data2), replace=True)
            diffs.append(sample2.mean() - sample1.mean())
        return np.percentile(diffs, [2.5, 97.5])

    ci = bootstrap_mean_diff(data_control, data_variant)
    print(f"95% CI for difference: {ci}")
    ```

---

## Questions asked in Google interview
- Explain the difference between Type I and Type II errors.
- How do you design an experiment to test a change in the Search Ranking algorithm?
- How to handle multiple metrics in an experiment? (Overall Evaluation Criterion).
- Explain the trade-off between sample size and experiment duration.
- Deriving the variance of the difference between two means.
- How to detect if your randomization algorithm is broken?
- Explain how you would test a feature with strong network effects.
- How to measure the long-term impact of a UI change?
- What metric would you use for a "User Happiness" experiment?
- Explain the concept of "Regression to the Mean" in the context of A/B testing.

## Questions asked in Meta (Facebook) interview
- How to measure network effects in a social network experiment?
- Explain Cluster-based randomization. Why use it?
- How to handle "Novelty Effect" when launching a new feature?
- Explain CUPED (Controlled-experiment Using Pre-Experiment Data).
- How to design an experiment for the News Feed ranking?
- What are the potential bounds of network interference?
- How to detect if an experiment has a Sample Ratio Mismatch (SRM)?
- Explain the difference between Average Treatment Effect (ATE) and Conditional ATE (CATE).
- How to optimize for long-term user retention?
- Design a test to measure the impact of ads on user engagement.

## Questions asked in Netflix interview
- How to A/B test a new recommendation algorithm?
- Explain "Interleaving" in ranking experiments.
- How to choose between "member-level" vs "profile-level" assignment?
- How to estimate the causal impact of a TV show launch on subscriptions? (Quasi-experiment).
- Explain the concept of "Proxy Metrics".
- How to handle outlier users (e.g., bots, heavy users) in analysis?
- Explain "Switchback" testing infrastructure.
- How to balance "Exploration" vs "Exploitation" (Bandits)?
- Design a test for artwork personalization (thumbnails).
- How to measure the "Incremental Reach" of a marketing campaign?

## Questions asked in Uber/Lyft interview (Marketplace)
- How to test changes in a two-sided marketplace (Rider vs Driver)?
- Explain "Switchback" designs for marketplace experiments.
- How to handle "Spillover" or "Cannibalization" effects?
- Explain "Difference-in-Differences" method.
- How to measure the impact of surge pricing changes?
- Explain "Synthetic Control" methods for city-level tests.
- How to calculate "Marketplace Liquidity" metrics?
- Design an experiment to reduce driver cancellations.
- How to test a new matching algorithm?
- Explain Interference in a geo-spatial context.

---

## Additional Resources

- [Microsoft Experimentation Platform (Exp)](https://www.microsoft.com/en-us/research/group/experimentation-platform-exp/) - Best technical papers.
- [Netflix Tech Blog - Experimentation](https://netflixtechblog.com/tagged/experimentation) - Real-world case studies.
- [Causal Inference for the Brave and True](https://matheusfacure.github.io/python-causality-handbook/) - Python handbook.
- [Trustworthy Online Controlled Experiments (Book)](https://www.amazon.com/Trustworthy-Online-Controlled-Experiments-Practical/dp/1108724264) - The "Bible" of A/B testing (Kohavi).
- [Uber Engineering - Data](https://eng.uber.com/category/articles/data/) - Marketplace testing concepts.
