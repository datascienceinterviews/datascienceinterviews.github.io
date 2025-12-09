---
title: A/B Testing Interview Questions
description: 100+ A/B Testing and Experimentation interview questions for Data Science and Product Analyst roles
---

# A/B Testing Interview Questions

<!-- [TOC] -->

This document provides a curated list of A/B Testing and Experimentation interview questions. It covers statistical foundations, experimental design, metric selection, and advanced topics like interference (network effects) and sequential testing. Critical for roles at data-driven companies like Netflix, Airbnb, and Uber.

---

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
