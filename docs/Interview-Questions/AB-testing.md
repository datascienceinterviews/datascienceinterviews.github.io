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

    **Core Concepts:**
    
    - **Null Hypothesis ($H_0$):** No difference between control and treatment
    - **Alternative Hypothesis ($H_1$):** There is a difference
    
    **Test Statistics:**
    
    $$z = \frac{\bar{x}_T - \bar{x}_C}{\sqrt{\frac{s_T^2}{n_T} + \frac{s_C^2}{n_C}}}$$
    
    ```python
    from scipy import stats
    
    # Two-sample t-test
    t_stat, p_value = stats.ttest_ind(control, treatment)
    
    # Z-test for proportions
    from statsmodels.stats.proportion import proportions_ztest
    z_stat, p_value = proportions_ztest([success_T, success_C], [n_T, n_C])
    ```

    !!! tip "Interviewer's Insight"
        Knows p-value interpretation and difference between z-test and t-test.

---

### How to Calculate Sample Size? - Google, Netflix Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Experimental Design` | **Asked by:** Google, Netflix, Uber

??? success "View Answer"

    **Formula for proportions:**
    
    $$n = \frac{2(z_{\alpha/2} + z_{\beta})^2 \cdot p(1-p)}{\delta^2}$$
    
    Where $\delta$ is the Minimum Detectable Effect (MDE).
    
    ```python
    from statsmodels.stats.power import TTestIndPower, proportion_effectsize
    
    # For proportions
    effect_size = proportion_effectsize(0.10, 0.12)  # baseline, expected
    power_analysis = TTestIndPower()
    sample_size = power_analysis.solve_power(
        effect_size=effect_size,
        power=0.8,
        alpha=0.05,
        ratio=1.0
    )
    ```
    
    **Key Factors:** MDE, baseline rate, significance (α), power (1-β).

    !!! tip "Interviewer's Insight"
        Understands tradeoffs between MDE, sample size, and test duration.

---

### What is Statistical Power? - Google, Netflix Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Statistics` | **Asked by:** Google, Netflix, Uber

??? success "View Answer"

    **Power = Probability of detecting a true effect = 1 - β (Type II error)**
    
    | Power | Meaning |
    |-------|---------|
    | 80% | Standard in industry |
    | 90% | High confidence needed |
    | 95% | Very conservative |
    
    **Factors affecting power:**
    - Sample size (↑ size = ↑ power)
    - Effect size (↑ effect = ↑ power)
    - Significance level (↑ α = ↑ power)
    - Variance (↓ variance = ↑ power)

    !!! tip "Interviewer's Insight"
        Knows 80% is standard and how to increase power.

---

### What is SRM (Sample Ratio Mismatch)? - Microsoft, LinkedIn Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Debugging` | **Asked by:** Microsoft, LinkedIn, Meta

??? success "View Answer"

    **SRM = Unequal split between control/treatment (when expecting equal)**
    
    ```python
    from scipy.stats import chi2_contingency
    
    observed = [n_control, n_treatment]
    expected = [total/2, total/2]
    
    chi2 = sum((o - e)**2 / e for o, e in zip(observed, expected))
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    
    if p_value < 0.001:  # Significant SRM
        print("STOP: Debug before analyzing results!")
    ```
    
    **Common Causes:**
    - Randomization bugs
    - Bot filtering differences
    - Browser/device incompatibility
    - Experiment interaction

    !!! tip "Interviewer's Insight"
        Always checks SRM before analyzing results and knows debugging steps.

---

### Explain Type I and Type II Errors - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Statistics` | **Asked by:** Most Tech Companies

??? success "View Answer"

    | Error | Name | Description | Rate |
    |-------|------|-------------|------|
    | Type I | False Positive | Reject $H_0$ when true | α (usually 0.05) |
    | Type II | False Negative | Fail to reject $H_0$ when false | β (usually 0.2) |
    
    **A/B Testing Context:**
    - Type I: Ship a feature that doesn't help (or hurts)
    - Type II: Miss a winning feature
    
    **Trade-off:** Lower α means higher β (for fixed sample size).

    !!! tip "Interviewer's Insight"
        Explains in business terms (shipping bad feature vs missing good one).

---

### How to Handle the Peeking Problem? - Netflix, Airbnb Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Pitfalls` | **Asked by:** Netflix, Airbnb, Uber

??? success "View Answer"

    **Peeking = Repeatedly checking p-value inflates false positive rate**
    
    **Solutions:**
    
    1. **Fixed-horizon:** Don't look until sample size reached
    
    2. **Sequential testing:**
    ```python
    # O'Brien-Fleming boundaries (conservative)
    # Alpha spending function
    from statsmodels.stats.multitest import local_fdr
    
    # Or use always-valid p-values
    ```
    
    3. **Bayesian approach:** Continuous monitoring with updating beliefs
    
    **Impact:** Peeking daily can inflate α from 5% to 30%+!

    !!! tip "Interviewer's Insight"
        Knows multiple solutions: sequential testing, alpha-spending, Bayesian.

---

### What is CUPED (Covariate Adjustment)? - Booking, Microsoft, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Variance Reduction` | **Asked by:** Booking, Microsoft, Meta

??? success "View Answer"

    **CUPED = Controlled-experiment Using Pre-Experiment Data**
    
    Reduces variance by using pre-experiment covariates:
    
    $$Y_{cuped} = Y - \theta(X - \bar{X})$$
    
    where $\theta = \frac{Cov(X, Y)}{Var(X)}$
    
    ```python
    import numpy as np
    
    # X = pre-experiment metric, Y = experiment metric
    theta = np.cov(X, Y)[0, 1] / np.var(X)
    Y_cuped = Y - theta * (X - X.mean())
    
    # Variance reduction
    variance_reduction = 1 - np.corrcoef(X, Y)[0, 1]**2
    ```
    
    **Benefit:** 20-50% variance reduction → shorter experiments.

    !!! tip "Interviewer's Insight"
        Knows CUPED formula and practical variance reduction benefits.

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

### How to Test on a Two-Sided Marketplace? - Uber, Lyft, Airbnb Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Marketplace` | **Asked by:** Uber, Lyft, Airbnb

??? success "View Answer"

    **Challenge:** Buyers and sellers interact (interference/spillover)
    
    **Solutions:**
    
    1. **Switchback experiments:** Time-based randomization
    2. **Geo-randomization:** Randomize by city/region
    3. **Synthetic control:** Compare to similar markets
    
    ```python
    # Switchback: alternate treatment periods
    # Period 1: Control, Period 2: Treatment, Period 3: Control...
    
    # Analysis accounts for temporal correlation
    ```
    
    **Uber example:** Can't A/B test surge pricing normally (drivers see all prices).

    !!! tip "Interviewer's Insight"
        Knows interference problem and proposes appropriate design.

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

### How to Handle Network Effects (Interference)? - Meta, LinkedIn Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Network Effects` | **Asked by:** Meta, LinkedIn, Uber

??? success "View Answer"

    **Interference = Treatment affects control (or vice versa)**
    
    **Example:** Sharing feature affects friends in control group.
    
    **Solutions:**
    
    1. **Cluster randomization:** Randomize friend groups together
    2. **Ego-cluster:** Randomize user + their network
    3. **Graph cluster randomization**
    
    ```python
    # Cluster by connected components
    import networkx as nx
    
    G = nx.from_edgelist(friend_pairs)
    clusters = list(nx.connected_components(G))
    
    # Randomize clusters, not users
    ```

    !!! tip "Interviewer's Insight"
        Identifies interference scenarios and proposes cluster randomization.

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

### How to Handle Multiple Testing? - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Multiple Comparisons` | **Asked by:** Google, Meta, Netflix

??? success "View Answer"

    **Multiple testing inflates false positive rate**
    
    ```python
    from statsmodels.stats.multitest import multipletests
    
    # Bonferroni (conservative)
    corrected_alpha = 0.05 / num_tests
    
    # Benjamini-Hochberg (FDR control)
    rejected, corrected_pvals, _, _ = multipletests(pvalues, method='fdr_bh')
    
    # Holm-Bonferroni (less conservative)
    rejected, corrected_pvals, _, _ = multipletests(pvalues, method='holm')
    ```
    
    **Rule of thumb:** Use BH for exploratory, Bonferroni for confirmatory.

    !!! tip "Interviewer's Insight"
        Knows Bonferroni is too conservative and proposes FDR control.

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

### What is Novelty Effect and How to Handle It? - Meta, Instagram Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Pitfalls` | **Asked by:** Meta, Instagram, Netflix

??? success "View Answer"

    **Novelty Effect = Users react differently to something new**
    
    **Impact:** Initial positive effect fades over time.
    
    **Detection:**
    - Segment by user tenure
    - Plot effect over time
    - Compare new vs returning users
    
    **Solutions:**
    - Run longer experiments
    - Exclude initial period from analysis
    - Only test on new users

    !!! tip "Interviewer's Insight"
        Segments analysis by user tenure and run duration.

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
    1. North Star metric (long-term)
    2. Primary metric (experiment goal)
    3. Secondary metrics (understanding)
    4. Guardrail metrics (safety)

    !!! tip "Interviewer's Insight"
        Connects metric choice to experiment duration and business goals.

---

### What is Attrition Bias? - Uber, Lyft Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Bias` | **Asked by:** Uber, Lyft, DoorDash

??? success "View Answer"

    **Attrition = Different dropout rates between groups**
    
    **Example:** Treatment is so bad users leave before completing.
    
    **Detection:**
    - Compare completion rates
    - Check SRM at different funnel stages
    
    **Solutions:**
    - Intent-to-treat analysis (analyze all assigned)
    - Survivor analysis (adjust for dropout)

    !!! tip "Interviewer's Insight"
        Uses intent-to-treat as primary analysis.

---

### How to Analyze Long-Term Effects? - Netflix, Spotify Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Long-term` | **Asked by:** Netflix, Spotify, LinkedIn

??? success "View Answer"

    **Challenge:** Can't run experiments forever.
    
    **Solutions:**
    
    1. **Holdback groups:** Small control group held for months
    2. **Proxy metrics:** Leading indicators of long-term outcomes
    3. **Causal modeling:** Estimate long-term from short-term
    
    ```python
    # Holdback: 5% control, 95% treatment
    # Re-evaluate quarterly
    ```
    
    **Netflix example:** Use engagement to predict retention.

    !!! tip "Interviewer's Insight"
        Proposes holdback groups and proxy metric strategy.

---

### How to Handle Low-Traffic Experiments? - Startups Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Strategy` | **Asked by:** Startups, Growth Teams

??? success "View Answer"

    **Strategies:**
    
    1. **Increase MDE:** Accept detecting only large effects
    2. **Bayesian methods:** Make decisions with less data
    3. **Sequential testing:** Stop early if clear winner
    4. **Variance reduction:** Use CUPED, stratification
    5. **Focus on core metrics:** Test fewer things

    !!! tip "Interviewer's Insight"
        Adjusts experimental design for traffic constraints.

---

### What is Heterogeneous Treatment Effects (HTE)? - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Advanced` | **Asked by:** Google, Meta, Netflix

??? success "View Answer"

    **HTE = Treatment effect varies across subgroups**
    
    ```python
    # Causal Forest for HTE estimation
    from econml.dml import CausalForestDML
    
    cf = CausalForestDML()
    cf.fit(Y, T, X=covariates, W=controls)
    treatment_effects = cf.effect(X_test)
    ```
    
    **Use cases:**
    - Personalization
    - Understanding who benefits most
    - Targeting treatment

    !!! tip "Interviewer's Insight"
        Uses causal ML methods for HTE estimation.

---

### What is Bootstrap for A/B Testing? - Google, Netflix Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Statistics` | **Asked by:** Google, Netflix, Meta

??? success "View Answer"

    ```python
    import numpy as np
    
    def bootstrap_ci(control, treatment, n_bootstrap=10000):
        diffs = []
        for _ in range(n_bootstrap):
            c_sample = np.random.choice(control, len(control), replace=True)
            t_sample = np.random.choice(treatment, len(treatment), replace=True)
            diffs.append(t_sample.mean() - c_sample.mean())
        
        return np.percentile(diffs, [2.5, 97.5])
    ```
    
    **Advantages:** Non-parametric, works for any statistic.

    !!! tip "Interviewer's Insight"
        Uses bootstrap for non-standard metrics.

---

### How to Test Revenue Metrics? - E-commerce Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Revenue` | **Asked by:** Amazon, Shopify, Stripe

??? success "View Answer"

    **Challenges:**
    - Heavy-tailed distribution
    - Many zeros (non-purchasers)
    - Outliers (large purchases)
    
    **Solutions:**
    - Winsorization (cap at 99th percentile)
    - Log transformation
    - Use trimmed means
    - Delta method for ratio metrics

    !!! tip "Interviewer's Insight"
        Handles outliers and heavy tails appropriately.

---

### What is Regression to the Mean? - All Companies Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Statistics` | **Asked by:** All Companies

??? success "View Answer"

    **RTM = Extreme values tend to move toward average**
    
    **A/B Testing Impact:**
    - Selecting worst performers to "improve" = natural improvement
    - Can confuse with treatment effect
    
    **Prevention:**
    - Randomization
    - Control group comparison
    - Don't select based on outcome

    !!! tip "Interviewer's Insight"
        Recognizes RTM in before/after comparisons.

---

### How to Handle Multiple Metrics? - Netflix, Airbnb Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Metrics` | **Asked by:** Netflix, Airbnb, Uber

??? success "View Answer"

    **Prioritization:**
    1. Primary: Decision metric
    2. Secondary: Interpretation metrics
    3. Guardrails: Safety checks
    
    **Decision rules:**
    - Primary wins + guardrails OK → Ship
    - Primary neutral + secondary positive → Consider shipping
    - Any guardrail fails → Don't ship

    !!! tip "Interviewer's Insight"
        Has clear decision framework for conflicting metrics.

---

### What is Simpson's Paradox? - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Statistics` | **Asked by:** Google, Meta, Netflix

??? success "View Answer"

    **Simpson's Paradox = Trend reverses when data is aggregated**
    
    **Example:**
    - Mobile: Treatment wins
    - Desktop: Treatment wins
    - Combined: Control wins!
    
    **Cause:** Unequal group sizes across segments.
    
    **Prevention:** Stratification, always segment analysis.

    !!! tip "Interviewer's Insight"
        Always segments analysis and checks for paradox.

---

### How to Run Tests with Ratio Metrics? - Most Tech Companies Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Metrics` | **Asked by:** Most Tech Companies

??? success "View Answer"

    **User-level vs Session-level:**
    
    - **User-level ratio:** Total clicks / Total users
    - **Session-level ratio:** Σ(session clicks / session views)
    
    **Best practice:** Randomize at user level, analyze at user level.
    
    Use delta method or bootstrap for variance estimation.

    !!! tip "Interviewer's Insight"
        Matches analysis unit to randomization unit.

---

### What is Sensitivity Analysis? - Netflix, Uber Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Robustness` | **Asked by:** Netflix, Uber, Airbnb

??? success "View Answer"

    **Check if conclusions hold under different assumptions:**
    
    - Different time periods
    - Excluding outliers
    - Different segments
    - Alternative metrics
    
    **If results are robust:** High confidence in decision.

    !!! tip "Interviewer's Insight"
        Tests robustness before making launch decisions.

---

### How to Communicate Results to Stakeholders? - All Companies Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Communication` | **Asked by:** All Companies

??? success "View Answer"

    **Structure:**
    1. **Bottom line:** Ship or don't ship
    2. **Key results:** Primary metric + CI
    3. **Context:** Guardrails, segments, caveats
    4. **Recommendations:** Clear next steps
    
    **Avoid:** p-value jargon, overconfidence

    !!! tip "Interviewer's Insight"
        Leads with business impact, not statistics.

---

### What is the Minimum Detectable Effect (MDE)? - Netflix, Uber Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Experimental Design` | **Asked by:** Netflix, Uber, Airbnb

??? success "View Answer"

    **MDE = Smallest effect size you can reliably detect**
    
    $$MDE = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{2\sigma^2}{n}}$$
    
    **Trade-offs:**
    - Lower MDE → More samples needed
    - Higher MDE → Might miss real effects
    
    **Rule of thumb:** MDE should be meaningful for business.

    !!! tip "Interviewer's Insight"
        Sets MDE based on business value, not just statistics.

---

### How to Design an Experimentation Platform? - Senior Roles Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `System Design` | **Asked by:** Google, Netflix, Meta

??? success "View Answer"

    **Core Components:**
    
    1. **Assignment service:** Random, consistent assignment
    2. **Logging:** Events, assignments, metrics
    3. **Analysis pipeline:** Automated statistics
    4. **Dashboard:** Results, SRM checks
    5. **Guardrails:** Automated safety checks
    
    **Scale considerations:** Netflix runs 100+ experiments simultaneously.

    !!! tip "Interviewer's Insight"
        Thinks about scale, consistency, and automation.

---

### What is Stratified Randomization? - Netflix, Uber Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Design` | **Asked by:** Netflix, Uber, Airbnb

??? success "View Answer"

    **Stratify = Balance groups on important covariates**
    
    ```python
    # Stratify by country, device, etc.
    # Ensures equal distribution across strata
    
    from sklearn.model_selection import StratifiedShuffleSplit
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5)
    for control_idx, treatment_idx in sss.split(X, strata):
        control = users[control_idx]
        treatment = users[treatment_idx]
    ```
    
    **Reduces variance** in treatment effect estimates.

    !!! tip "Interviewer's Insight"
        Stratifies on key covariates for balanced groups.

---

### How to Use Regression Adjustment? - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Analysis` | **Asked by:** Google, Meta, Netflix

??? success "View Answer"

    **Include covariates in regression for variance reduction**
    
    ```python
    import statsmodels.api as sm
    
    # Simple: Y ~ Treatment
    # Adjusted: Y ~ Treatment + Covariates
    
    X = sm.add_constant(df[['treatment', 'covariate1', 'covariate2']])
    model = sm.OLS(df['outcome'], X).fit()
    
    # Treatment effect with smaller SE
    print(model.params['treatment'], model.pvalues['treatment'])
    ```

    !!! tip "Interviewer's Insight"
        Uses covariates for more precise estimates.

---

### What is Triggered Analysis? - Uber, Lyft Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Analysis` | **Asked by:** Uber, Lyft, DoorDash

??? success "View Answer"

    **Only analyze users who encountered the treatment**
    
    - ITT: All randomized users
    - Triggered: Only exposed users
    
    **Caution:** Can introduce selection bias if triggering differs.
    
    **Best practice:** Report both ITT and triggered analyses.

    !!! tip "Interviewer's Insight"
        Reports both ITT and triggered for complete picture.

---

### How to Handle Carry-Over Effects? - Netflix, Airbnb Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Design` | **Asked by:** Netflix, Airbnb, Uber

??? success "View Answer"

    **Carry-over = Previous treatment affects current behavior**
    
    **Solutions:**
    - Washout period between treatments
    - Only use first exposure
    - Crossover designs with randomized order
    - Long-running experiments

    !!! tip "Interviewer's Insight"
        Designs experiments with washout periods.

---

### What is Intent-to-Treat (ITT)? - All Companies Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Analysis` | **Asked by:** All Companies

??? success "View Answer"

    **ITT = Analyze all users as randomized**
    
    Even if users:
    - Didn't use the feature
    - Switched groups
    - Dropped out
    
    **Why:** Preserves randomization, real-world effect.
    
    **Alternative:** Per-protocol (analyze only compliant users).

    !!! tip "Interviewer's Insight"
        Uses ITT as primary analysis.

---

### How to Estimate Lift? - E-commerce Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Metrics` | **Asked by:** Amazon, Shopify, Etsy

??? success "View Answer"

    **Lift = Relative improvement**
    
    $$\text{Lift} = \frac{\bar{x}_T - \bar{x}_C}{\bar{x}_C} \times 100\%$$
    
    ```python
    lift = (treatment_mean - control_mean) / control_mean * 100
    
    # Confidence interval for lift uses delta method
    ```
    
    **Communicate:** "Treatment increased conversion by 5%"

    !!! tip "Interviewer's Insight"
        Reports both absolute and relative effects.

---

### What is Pre-Registration? - Research Companies Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Best Practices` | **Asked by:** Research Labs, Netflix

??? success "View Answer"

    **Document analysis plan before seeing results**
    
    Pre-register:
    - Hypothesis
    - Primary metric
    - Sample size
    - Analysis method
    - Success criteria
    
    **Prevents:** p-hacking, cherry-picking, HARKing.

    !!! tip "Interviewer's Insight"
        Pre-registers to prevent post-hoc rationalization.

---

### How to Handle Seasonality? - E-commerce Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Design` | **Asked by:** Amazon, Etsy, Shopify

??? success "View Answer"

    **Time-based confounds affect experiments**
    
    **Solutions:**
    - Run full weekly cycles
    - Randomize within time strata
    - Control for day-of-week effects
    - Avoid holiday periods
    - Use switchback designs

    !!! tip "Interviewer's Insight"
        Runs experiments for complete cycles.

---

### What is a Holdout Group? - Netflix, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Long-term` | **Asked by:** Netflix, Meta, Spotify

??? success "View Answer"

    **Small control group held for long-term measurement**
    
    ```
    Experiment: 50% control, 50% treatment
    After launch: 5% holdout, 95% launched feature
    ```
    
    **Purpose:** Measure long-term effects, detect degradation.
    
    **Duration:** Weeks to months.

    !!! tip "Interviewer's Insight"
        Uses holdouts for long-term monitoring.

---

### How to Test Personalization? - Netflix, Spotify Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Personalization` | **Asked by:** Netflix, Spotify, Amazon

??? success "View Answer"

    **Challenge:** Different treatment for different users.
    
    **Approach:**
    1. Randomize into control (old) vs treatment (personalized)
    2. Not: randomize personalization parameters
    
    **Metrics:** Aggregate effect + HTE analysis by segments.

    !!! tip "Interviewer's Insight"
        Tests personalization vs non-personalized, not between variants.

---

### What is Sequential Testing? - Netflix, Uber Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Statistics` | **Asked by:** Netflix, Uber, Airbnb

??? success "View Answer"

    **Allows peeking while controlling error rate**
    
    ```python
    # O'Brien-Fleming spending function
    # Pocock spending function
    # Always-valid inference
    
    # Example: Stop early if effect is very large
    # Continue if inconclusive
    # Stop for futility if effect is negligible
    ```

    !!! tip "Interviewer's Insight"
        Uses sequential testing for early stopping.

---

### How to Report Negative Results? - All Companies Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Communication` | **Asked by:** All Companies

??? success "View Answer"

    **No effect is still valuable information**
    
    **Report:**
    - Effect size (even if near zero)
    - Confidence interval
    - Statistical power achieved
    - What we learned
    - Next steps/iterations

    !!! tip "Interviewer's Insight"
        Frames negative results as learnings.

---

### What is Variance Reduction? - Netflix, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Efficiency` | **Asked by:** Netflix, Meta, Microsoft

??? success "View Answer"

    **Techniques to reduce required sample size:**
    
    | Method | Variance Reduction |
    |--------|-------------------|
    | CUPED | 20-50% |
    | Stratification | 5-20% |
    | Regression | 10-30% |
    | Paired design | Variable |
    
    **All reduce variance → shorter experiments.**

    !!! tip "Interviewer's Insight"
        Combines multiple variance reduction techniques.

---

### How to Handle Feature Interactions? - Netflix, Airbnb Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Design` | **Asked by:** Netflix, Airbnb, Meta

??? success "View Answer"

    **Multiple experiments running simultaneously**
    
    **Approaches:**
    - Mutual exclusion (separate traffic)
    - Factorial design (all combinations)
    - Layered experiments (non-interacting)
    
    **Test:** Check if interaction effect is significant.

    !!! tip "Interviewer's Insight"
        Uses layers for non-interacting experiments.

---

### What is a Ramp-Up Strategy? - All Companies Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Deployment` | **Asked by:** All Companies

??? success "View Answer"

    **Gradually increase treatment allocation**
    
    ```
    Day 1: 1% treatment
    Day 2: 5% treatment  
    Day 3: 10% treatment
    ...
    Final: 50% treatment
    ```
    
    **Purpose:** Catch bugs early, limit blast radius.

    !!! tip "Interviewer's Insight"
        Ramps up to detect problems early.

---

### How to Calculate Expected Revenue Impact? - E-commerce Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Business` | **Asked by:** Amazon, Shopify, Stripe

??? success "View Answer"

    ```python
    # Point estimate
    revenue_lift = (treatment_rev - control_rev) / control_rev
    expected_annual = revenue_lift * annual_revenue
    
    # With uncertainty
    ci_low, ci_high = bootstrap_ci(control_rev, treatment_rev)
    range_annual = (ci_low * annual_revenue, ci_high * annual_revenue)
    ```
    
    **Always include confidence intervals!**

    !!! tip "Interviewer's Insight"
        Translates statistical results to business value.

---

### What is Propensity Score Matching? - Google, Netflix Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Causal Inference` | **Asked by:** Google, Netflix, Meta

??? success "View Answer"

    **Used when randomization isn't possible (observational data)**
    
    ```python
    from sklearn.linear_model import LogisticRegression
    
    # Estimate propensity scores
    ps_model = LogisticRegression()
    ps_model.fit(X_covariates, treatment)
    propensity_scores = ps_model.predict_proba(X_covariates)[:, 1]
    
    # Match treated with similar control units
    # Then compare outcomes
    ```
    
    **Limitations:** Only balances observed covariates.

    !!! tip "Interviewer's Insight"
        Knows when to use PSM vs other causal methods.

---

### What is Difference-in-Differences? - Google, Uber Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Causal Inference` | **Asked by:** Google, Uber, Airbnb

??? success "View Answer"

    **Compare treatment vs control before and after intervention**
    
    $$\text{DiD} = (Y_{T,post} - Y_{T,pre}) - (Y_{C,post} - Y_{C,pre})$$
    
    **Assumption:** Parallel trends (groups would have similar trends without treatment).
    
    **Use case:** Policy changes, geographic rollouts.

    !!! tip "Interviewer's Insight"
        Checks parallel trends assumption with pre-period data.

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
