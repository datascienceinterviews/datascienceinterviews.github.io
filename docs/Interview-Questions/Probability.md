---
title: Probability Interview Questions
description: A curated list of probability interview questions for data science and technical interviews
# hide:
#   - toc
---

# Probability Interview Questions

<!-- ![Total Questions](https://img.shields.io/badge/Total%20Questions-1-blue?style=flat&labelColor=black&color=blue)
![Unanswered Questions](https://img.shields.io/badge/Unanswered%20Questions-0-blue?style=flat&labelColor=black&color=yellow)
![Answered Questions](https://img.shields.io/badge/Answered%20Questions-1-blue?style=flat&labelColor=black&color=success) -->


This document provides a curated list of common probability interview questions frequently asked in technical interviews. It covers basic probability concepts, probability distributions, key theorems, and real-world applications. Use the practice links to explore detailed explanations and examples.

---

## Premium Interview Questions

### What is Bayes' Theorem? Explain with an Example - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Bayes`, `Conditional Probability`, `Inference` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Bayes' Theorem:**
    
    $$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$
    
    **Components:**
    
    | Term | Name | Meaning |
    |------|------|---------|
    | P(A\|B) | Posterior | Probability of A given B |
    | P(B\|A) | Likelihood | Probability of B given A |
    | P(A) | Prior | Initial probability of A |
    | P(B) | Evidence | Total probability of B |
    
    **Medical Test Example:**
    
    - Disease prevalence: P(Disease) = 1%
    - Test sensitivity: P(Positive|Disease) = 99%
    - Test specificity: P(Negative|No Disease) = 95%
    
    **What's P(Disease|Positive)?**
    
    ```python
    # Prior
    p_disease = 0.01
    p_no_disease = 0.99
    
    # Likelihood
    p_pos_given_disease = 0.99
    p_pos_given_no_disease = 0.05  # False positive rate
    
    # Evidence: P(Positive)
    p_positive = (p_pos_given_disease * p_disease + 
                  p_pos_given_no_disease * p_no_disease)
    # = 0.99 * 0.01 + 0.05 * 0.99 = 0.0099 + 0.0495 = 0.0594
    
    # Posterior
    p_disease_given_pos = (p_pos_given_disease * p_disease) / p_positive
    # = 0.0099 / 0.0594 ≈ 0.167 or 16.7%
    ```
    
    **Insight:** Even with 99% accurate test, only 16.7% chance of disease!

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of conditional probability.
        
        **Strong answer signals:**
        
        - Writes formula without hesitation
        - Explains base rate fallacy
        - Shows numerical calculation
        - Relates to real applications (spam, medical)

---

### Explain Conditional Probability vs Independence - Google, Meta Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Conditional Probability`, `Independence`, `Fundamentals` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **Conditional Probability:**
    
    Probability of A given B has occurred:
    
    $$P(A|B) = \frac{P(A \cap B)}{P(B)}$$
    
    **Independence:**
    
    Events A and B are independent if:
    
    $$P(A|B) = P(A) \quad \text{or equivalently} \quad P(A \cap B) = P(A) \cdot P(B)$$
    
    **Example - Card Drawing:**
    
    ```python
    # Drawing from a deck
    # A = First card is Hearts
    # B = Second card is Hearts
    
    # WITH replacement (independent):
    p_a = 13/52  # = 1/4
    p_b_given_a = 13/52  # Same, deck reset
    p_both = (13/52) * (13/52) = 1/16
    
    # WITHOUT replacement (dependent):
    p_a = 13/52
    p_b_given_a = 12/51  # One heart removed
    p_both = (13/52) * (12/51) ≈ 0.059
    ```
    
    **Key Differences:**
    
    | Independent | Dependent |
    |-------------|-----------|
    | P(A∩B) = P(A)·P(B) | P(A∩B) ≠ P(A)·P(B) |
    | Knowing B doesn't change P(A) | Knowing B changes P(A) |
    | Coin flips, dice rolls | Card draws w/o replacement |
    
    **Common Confusion:** Independent ≠ Mutually exclusive!
    
    - Mutually exclusive: P(A∩B) = 0 (can't both occur)
    - Independent: P(A∩B) = P(A)·P(B) (outcomes don't affect each other)

    !!! tip "Interviewer's Insight"
        **What they're testing:** Fundamental probability concepts.
        
        **Strong answer signals:**
        
        - Clearly distinguishes conditional from joint
        - Knows independence vs mutually exclusive
        - Uses correct notation
        - Provides intuitive examples

---

### What is the Law of Total Probability? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Total Probability`, `Partition`, `Bayes` | **Asked by:** Google, Amazon, Microsoft

??? success "View Answer"

    **Law of Total Probability:**
    
    If B₁, B₂, ..., Bₙ partition the sample space:
    
    $$P(A) = \sum_{i=1}^{n} P(A|B_i) \cdot P(B_i)$$
    
    **Intuition:** Break complex probability into simpler conditional pieces.
    
    **Example - Product Defects:**
    
    Three factories produce parts:
    - Factory A: 50% of parts, 2% defect rate
    - Factory B: 30% of parts, 3% defect rate  
    - Factory C: 20% of parts, 5% defect rate
    
    **What's P(Defective)?**
    
    ```python
    p_a, p_b, p_c = 0.5, 0.3, 0.2  # Factory proportions
    d_a, d_b, d_c = 0.02, 0.03, 0.05  # Defect rates
    
    p_defective = (d_a * p_a + d_b * p_b + d_c * p_c)
    # = 0.02*0.5 + 0.03*0.3 + 0.05*0.2
    # = 0.01 + 0.009 + 0.01
    # = 0.029 or 2.9%
    ```
    
    **Follow-up: Given defective, which factory? (Bayes)**
    
    ```python
    # P(Factory A | Defective)
    p_a_given_defective = (d_a * p_a) / p_defective
    # = 0.01 / 0.029 ≈ 0.345 or 34.5%
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Breaking down complex probabilities.
        
        **Strong answer signals:**
        
        - Knows it requires exhaustive, mutually exclusive partition
        - Uses as setup for Bayes' theorem
        - Can apply to real scenarios
        - Shows clear calculation

---

### Explain Expected Value and Its Properties - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Expected Value`, `Mean`, `Random Variables` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Expected Value (Mean):**
    
    $$E[X] = \sum_x x \cdot P(X=x) \quad \text{(discrete)}$$
    
    $$E[X] = \int_{-\infty}^{\infty} x \cdot f(x) \, dx \quad \text{(continuous)}$$
    
    **Key Properties:**
    
    | Property | Formula |
    |----------|---------|
    | Linearity | E[aX + b] = a·E[X] + b |
    | Sum | E[X + Y] = E[X] + E[Y] (always!) |
    | Product (independent) | E[XY] = E[X]·E[Y] |
    | Constant | E[c] = c |
    
    **Example - Dice:**
    
    ```python
    # Fair 6-sided die
    E_X = sum(x * (1/6) for x in range(1, 7))
    # = (1 + 2 + 3 + 4 + 5 + 6) / 6 = 21/6 = 3.5
    ```
    
    **Casino Example:**
    
    Bet $1, win $35 if dice shows 6, lose otherwise:
    
    ```python
    # X = profit
    p_win = 1/6
    p_lose = 5/6
    
    E_X = 35 * (1/6) + (-1) * (5/6)
    # = 35/6 - 5/6 = 30/6 = 5
    # Expected profit = $5 per game (very favorable!)
    
    # Real casino: win $5 (not $35)
    E_X = 5 * (1/6) + (-1) * (5/6) = 5/6 - 5/6 = 0
    # Fair game
    ```
    
    **Why Linearity Matters:**
    
    E[X₁ + X₂ + ... + Xₙ] = E[X₁] + E[X₂] + ... + E[Xₙ]
    
    Works even when Xᵢ are dependent!

    !!! tip "Interviewer's Insight"
        **What they're testing:** Foundation of probability calculations.
        
        **Strong answer signals:**
        
        - Knows linearity works without independence
        - Can calculate for discrete and continuous
        - Applies to decision-making problems
        - Distinguishes expected value from most likely value

---

### What is Variance? How is it Related to Standard Deviation? - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Variance`, `Standard Deviation`, `Spread` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **Variance:**
    
    Measures spread of distribution around mean:
    
    $$Var(X) = E[(X - \mu)^2] = E[X^2] - (E[X])^2$$
    
    **Standard Deviation:**
    
    $$\sigma = \sqrt{Var(X)}$$
    
    **Properties:**
    
    | Property | Formula |
    |----------|---------|
    | Variance of constant | Var(c) = 0 |
    | Scaling | Var(aX) = a²·Var(X) |
    | Shift | Var(X + b) = Var(X) |
    | Sum (independent) | Var(X + Y) = Var(X) + Var(Y) |
    | Sum (dependent) | Var(X + Y) = Var(X) + Var(Y) + 2·Cov(X,Y) |
    
    **Example:**
    
    ```python
    import numpy as np
    
    # Roll of fair die
    outcomes = [1, 2, 3, 4, 5, 6]
    probs = [1/6] * 6
    
    E_X = sum(x * p for x, p in zip(outcomes, probs))  # 3.5
    E_X2 = sum(x**2 * p for x, p in zip(outcomes, probs))  # 15.17
    
    variance = E_X2 - E_X**2  # 15.17 - 12.25 = 2.92
    std_dev = np.sqrt(variance)  # 1.71
    ```
    
    **Why Standard Deviation?**
    
    - Same units as original data (variance has squared units)
    - Interpretable: ~68% of data within 1 std dev (normal)
    - Used in confidence intervals, z-scores

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of spread/uncertainty.
        
        **Strong answer signals:**
        
        - Uses E[X²] - (E[X])² formula
        - Knows covariance term for dependent variables
        - Explains why σ has same units as X
        - Can compute by hand

---

### Explain the Central Limit Theorem - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `CLT`, `Normal Distribution`, `Sampling` | **Asked by:** Google, Amazon, Meta, Microsoft

??? success "View Answer"

    **Central Limit Theorem:**
    
    Sample means of any distribution approach normal as n → ∞:
    
    $$\bar{X}_n \xrightarrow{d} N\left(\mu, \frac{\sigma^2}{n}\right)$$
    
    **Key Points:**
    
    1. Works for ANY distribution (with finite variance)
    2. n ≥ 30 is usually "large enough"
    3. More skewed → need larger n
    
    **Why It Matters:**
    
    - Enables confidence intervals
    - Justifies z-tests and t-tests
    - A/B testing relies on CLT
    
    **Example:**
    
    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Exponential distribution (highly skewed)
    population = np.random.exponential(scale=1, size=100000)
    
    # Sample means (n=50)
    sample_means = [np.mean(np.random.choice(population, 50)) 
                    for _ in range(10000)]
    
    # Sample means are normal even though population is exponential!
    plt.hist(sample_means, bins=50, density=True)
    plt.title("Distribution of Sample Means (n=50)")
    ```
    
    **Standard Error:**
    
    $$SE = \frac{\sigma}{\sqrt{n}}$$
    
    As sample size increases, sampling distribution narrows.

    !!! tip "Interviewer's Insight"
        **What they're testing:** Core statistical foundation.
        
        **Strong answer signals:**
        
        - Knows it applies to means of any distribution
        - Can state conditions (finite variance)
        - Links to hypothesis testing
        - Explains standard error formula

---

### What is the Normal Distribution? State its Properties - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Normal`, `Gaussian`, `Continuous Distribution` | **Asked by:** Google, Amazon, Meta, Microsoft

??? success "View Answer"

    **Normal Distribution:**
    
    $$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$
    
    **Key Properties:**
    
    | Property | Value |
    |----------|-------|
    | Mean | μ |
    | Variance | σ² |
    | Skewness | 0 (symmetric) |
    | Kurtosis | 3 (standard) |
    | Mode = Median = Mean | μ |
    
    **Empirical Rule (68-95-99.7):**
    
    ```
    μ ± 1σ → 68.27% of data
    μ ± 2σ → 95.45% of data
    μ ± 3σ → 99.73% of data
    ```
    
    **Standard Normal (Z-score):**
    
    $$Z = \frac{X - \mu}{\sigma} \sim N(0, 1)$$
    
    **Sum of Normals:**
    
    If X ~ N(μ₁, σ₁²) and Y ~ N(μ₂, σ₂²) are independent:
    
    X + Y ~ N(μ₁ + μ₂, σ₁² + σ₂²)
    
    **Python:**
    
    ```python
    from scipy import stats
    
    # N(100, 15) - IQ distribution
    iq = stats.norm(loc=100, scale=15)
    
    # P(IQ > 130)?
    p_above_130 = 1 - iq.cdf(130)  # ≈ 0.0228 or 2.28%
    
    # What IQ is 95th percentile?
    iq_95 = iq.ppf(0.95)  # ≈ 124.7
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Most important distribution knowledge.
        
        **Strong answer signals:**
        
        - Knows 68-95-99.7 rule
        - Can standardize to Z-score
        - Knows sum of normals is normal
        - Uses scipy.stats for calculations

---

### Explain the Binomial Distribution - Amazon, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Binomial`, `Discrete`, `Bernoulli Trials` | **Asked by:** Amazon, Meta, Google

??? success "View Answer"

    **Binomial Distribution:**
    
    Number of successes in n independent Bernoulli trials:
    
    $$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$
    
    **Parameters:**
    
    - n = number of trials
    - p = probability of success per trial
    - k = number of successes
    
    **Formulas:**
    
    | Statistic | Formula |
    |-----------|---------|
    | Mean | E[X] = np |
    | Variance | Var(X) = np(1-p) |
    | Mode | floor((n+1)p) or floor((n+1)p)-1 |
    
    **Example - Quality Control:**
    
    10 items, 5% defect rate. P(exactly 2 defective)?
    
    ```python
    from scipy.stats import binom
    from math import comb
    
    n, p, k = 10, 0.05, 2
    
    # Manual calculation
    p_2 = comb(10, 2) * (0.05**2) * (0.95**8)
    # = 45 * 0.0025 * 0.6634 ≈ 0.0746
    
    # Using scipy
    p_2 = binom.pmf(k=2, n=10, p=0.05)
    
    # P(at least 1 defective)?
    p_at_least_1 = 1 - binom.pmf(k=0, n=10, p=0.05)
    # = 1 - 0.5987 ≈ 0.401
    ```
    
    **Normal Approximation (n large):**
    
    If np ≥ 5 and n(1-p) ≥ 5:
    
    X ~ N(np, np(1-p)) approximately

    !!! tip "Interviewer's Insight"
        **What they're testing:** Core discrete distribution.
        
        **Strong answer signals:**
        
        - States conditions (fixed n, independent, same p)
        - Knows mean = np without derivation
        - Uses complement for "at least" problems
        - Knows normal approximation conditions

---

### What is the Poisson Distribution? When to Use It? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Poisson`, `Discrete`, `Rare Events` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Poisson Distribution:**
    
    Models count of events in fixed interval (time, space):
    
    $$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$
    
    **Parameters:**
    
    - λ = rate (expected count per interval)
    - k = actual count (0, 1, 2, ...)
    
    **Special Property:**
    
    E[X] = Var(X) = λ
    
    **When to Use:**
    
    1. Events occur independently
    2. Rate is constant
    3. Events are "rare" (compared to opportunities)
    
    **Examples:**
    - Website visits per minute
    - Typos per page
    - Goals in a soccer game
    - Radioactive decays per second
    
    **Python:**
    
    ```python
    from scipy.stats import poisson
    
    # 4 customers per hour on average
    lambda_rate = 4
    
    # P(exactly 6 customers)?
    p_6 = poisson.pmf(k=6, mu=4)  # ≈ 0.104
    
    # P(at most 2 customers)?
    p_le_2 = poisson.cdf(k=2, mu=4)  # ≈ 0.238
    
    # P(more than 5)?
    p_gt_5 = 1 - poisson.cdf(k=5, mu=4)  # ≈ 0.215
    ```
    
    **Poisson as Binomial Limit:**
    
    When n → ∞, p → 0, np = λ:
    Binomial(n, p) → Poisson(λ)

    !!! tip "Interviewer's Insight"
        **What they're testing:** Count data modeling.
        
        **Strong answer signals:**
        
        - States E[X] = Var(X) = λ
        - Gives real-world examples
        - Knows Poisson-Binomial relationship
        - Uses for rate-based problems

---

### Explain the Exponential Distribution - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Exponential`, `Continuous`, `Waiting Time` | **Asked by:** Google, Amazon, Microsoft

??? success "View Answer"

    **Exponential Distribution:**
    
    Models time between Poisson events (waiting time):
    
    $$f(x) = \lambda e^{-\lambda x}, \quad x \geq 0$$
    
    **Parameters:**
    
    | Statistic | Formula |
    |-----------|---------|
    | Mean | 1/λ |
    | Variance | 1/λ² |
    | Median | ln(2)/λ |
    
    **Memoryless Property:**
    
    $$P(X > s + t | X > s) = P(X > t)$$
    
    "Past doesn't affect future" - unique to exponential!
    
    **Example:**
    
    Bus arrives every 10 minutes on average (λ = 0.1/min):
    
    ```python
    from scipy.stats import expon
    
    # λ = 0.1, scale = 1/λ = 10
    wait_time = expon(scale=10)
    
    # P(wait < 5 minutes)?
    p_lt_5 = wait_time.cdf(5)  # ≈ 0.393
    
    # P(wait > 15 minutes)?
    p_gt_15 = 1 - wait_time.cdf(15)  # ≈ 0.223
    
    # Mean wait time
    mean_wait = wait_time.mean()  # 10 minutes
    ```
    
    **Relationship with Poisson:**
    
    - If counts per time ~ Poisson(λ)
    - Then time between events ~ Exponential(λ)

    !!! tip "Interviewer's Insight"
        **What they're testing:** Continuous distribution for waiting.
        
        **Strong answer signals:**
        
        - Knows memoryless property and its implications
        - Connects to Poisson process
        - Can calculate probabilities
        - Gives practical examples

---

### What is the Geometric Distribution? - Amazon, Microsoft Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Geometric`, `Discrete`, `First Success` | **Asked by:** Amazon, Microsoft, Google

??? success "View Answer"

    **Geometric Distribution:**
    
    Number of trials until first success:
    
    $$P(X = k) = (1-p)^{k-1} p, \quad k = 1, 2, 3, ...$$
    
    **Formulas:**
    
    | Statistic | Formula |
    |-----------|---------|
    | Mean | 1/p |
    | Variance | (1-p)/p² |
    | Mode | 1 |
    
    **Memoryless (like Exponential):**
    
    P(X > m + n | X > m) = P(X > n)
    
    **Example - Interview Success:**
    
    30% chance of passing each interview:
    
    ```python
    from scipy.stats import geom
    
    p = 0.3
    
    # P(pass on exactly 3rd interview)?
    p_3rd = geom.pmf(k=3, p=0.3)
    # = (0.7)^2 * 0.3 = 0.147
    
    # Expected interviews until first pass?
    expected = 1 / 0.3  # ≈ 3.33 interviews
    
    # P(need more than 5 interviews)?
    p_gt_5 = 1 - geom.cdf(k=5, p=0.3)
    # = (0.7)^5 ≈ 0.168
    ```
    
    **Alternative Definition:**
    
    Some texts define as failures before first success (k = 0, 1, 2, ...)

    !!! tip "Interviewer's Insight"
        **What they're testing:** First success modeling.
        
        **Strong answer signals:**
        
        - Knows two common definitions
        - Calculates E[X] = 1/p intuitively
        - Connects to negative binomial
        - Uses memoryless property

---

### What is the Birthday Problem? Calculate the Probability - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Birthday Problem`, `Combinatorics`, `Probability Puzzle` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Birthday Problem:**
    
    What's the probability that in a group of n people, at least 2 share a birthday?
    
    **Approach - Complement:**
    
    P(at least 2 share) = 1 - P(all different birthdays)
    
    $$P(\text{all different}) = \frac{365}{365} \cdot \frac{364}{365} \cdot \frac{363}{365} \cdot ... \cdot \frac{365-n+1}{365}$$
    
    **Calculation:**
    
    ```python
    def birthday_probability(n):
        """P(at least 2 share birthday in group of n)"""
        p_all_different = 1.0
        for i in range(n):
            p_all_different *= (365 - i) / 365
        return 1 - p_all_different
    
    # Results:
    # n=23: 50.7% (famous result!)
    # n=50: 97.0%
    # n=70: 99.9%
    
    for n in [10, 23, 30, 50, 70]:
        print(f"n={n}: {birthday_probability(n):.1%}")
    ```
    
    **Why So Counter-Intuitive?**
    
    - We think: 23 people, 365 days → small chance
    - Reality: C(23,2) = 253 pairs to compare!
    
    **Generalized Version:**
    
    P(collision in hash table) follows same logic - birthday attack in cryptography.

    !!! tip "Interviewer's Insight"
        **What they're testing:** Complement probability, combinatorics.
        
        **Strong answer signals:**
        
        - Uses complement approach
        - Knows n=23 gives ~50%
        - Can generalize to other collision problems
        - Explains why intuition fails

---

### Explain the Monty Hall Problem - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Monty Hall`, `Conditional Probability`, `Puzzle` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **The Setup:**
    
    - 3 doors: 1 car, 2 goats
    - You pick a door (say Door 1)
    - Host (who knows what's behind each) opens another door showing a goat
    - Should you switch?
    
    **Answer: YES - Switch gives 2/3 chance!**
    
    **Intuition:**
    
    ```
    Initial pick: P(Car) = 1/3
    Other doors:  P(Car) = 2/3
    
    After host reveals goat:
    - Your door still has P = 1/3
    - Remaining door gets all 2/3
    ```
    
    **Simulation Proof:**
    
    ```python
    import random
    
    def monty_hall(switch, n_simulations=100000):
        wins = 0
        for _ in range(n_simulations):
            car = random.randint(0, 2)
            choice = random.randint(0, 2)
            
            # Host opens a goat door (not your choice, not car)
            goat_doors = [i for i in range(3) if i != choice and i != car]
            host_opens = random.choice(goat_doors)
            
            if switch:
                # Switch to remaining door
                choice = [i for i in range(3) if i != choice and i != host_opens][0]
            
            if choice == car:
                wins += 1
        
        return wins / n_simulations
    
    print(f"Stay:   {monty_hall(switch=False):.1%}")   # ~33.3%
    print(f"Switch: {monty_hall(switch=True):.1%}")    # ~66.7%
    ```
    
    **Key Insight:**
    
    Host's action is not random - he MUST reveal a goat. This transfers information.

    !!! tip "Interviewer's Insight"
        **What they're testing:** Conditional probability reasoning.
        
        **Strong answer signals:**
        
        - Gives correct answer (switch = 2/3)
        - Explains WHY (host's constraint)
        - Can simulate or prove mathematically
        - Addresses common misconceptions

---

### What is Covariance and Correlation? - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Covariance`, `Correlation`, `Dependency` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **Covariance:**
    
    Measures joint variability of two variables:
    
    $$Cov(X, Y) = E[(X - \mu_X)(Y - \mu_Y)] = E[XY] - E[X]E[Y]$$
    
    **Correlation (Pearson):**
    
    Standardized covariance, range [-1, 1]:
    
    $$\rho_{XY} = \frac{Cov(X, Y)}{\sigma_X \sigma_Y}$$
    
    **Interpretation:**
    
    | Value | Meaning |
    |-------|---------|
    | ρ = 1 | Perfect positive linear |
    | ρ = 0 | No linear relationship |
    | ρ = -1 | Perfect negative linear |
    
    **Important Properties:**
    
    ```python
    # Covariance
    Cov(X, X) = Var(X)
    Cov(X, Y) = Cov(Y, X)  # Symmetric
    Cov(aX + b, Y) = a·Cov(X, Y)
    
    # Correlation
    Corr(aX + b, Y) = sign(a) · Corr(X, Y)  # Unaffected by linear transform
    ```
    
    **Python Calculation:**
    
    ```python
    import numpy as np
    
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 5, 4, 5])
    
    cov_matrix = np.cov(x, y)
    cov_xy = cov_matrix[0, 1]  # Covariance
    
    corr_matrix = np.corrcoef(x, y)
    corr_xy = corr_matrix[0, 1]  # Correlation
    ```
    
    **Warning:**
    
    Correlation ≠ Causation
    Correlation = 0 does NOT mean independence!

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding relationship measures.
        
        **Strong answer signals:**
        
        - Knows correlation is dimensionless
        - States correlation measures LINEAR relationship only
        - Knows correlation = 0 ≠ independence
        - Can distinguish correlation from causation

---

### Explain the Law of Large Numbers - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `LLN`, `Convergence`, `Sampling` | **Asked by:** Google, Amazon, Microsoft

??? success "View Answer"

    **Law of Large Numbers:**
    
    Sample mean converges to population mean as sample size → ∞:
    
    $$\bar{X}_n \xrightarrow{p} \mu \quad \text{as} \quad n \to \infty$$
    
    **Two Forms:**
    
    | Weak LLN | Strong LLN |
    |----------|------------|
    | Convergence in probability | Almost sure convergence |
    | P(\|X̄ₙ - μ\| > ε) → 0 | P(X̄ₙ → μ) = 1 |
    
    **Intuition:**
    
    More samples → better estimate of true mean
    
    **Example:**
    
    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Fair coin: P(Heads) = 0.5
    np.random.seed(42)
    
    flips = np.random.binomial(1, 0.5, 10000)
    running_mean = np.cumsum(flips) / np.arange(1, 10001)
    
    plt.plot(running_mean)
    plt.axhline(y=0.5, color='r', linestyle='--', label='True Mean')
    plt.xlabel('Number of Flips')
    plt.ylabel('Running Mean')
    plt.title('Law of Large Numbers: Coin Flips')
    ```
    
    **Key Distinction from CLT:**
    
    | LLN | CLT |
    |-----|-----|
    | Sample mean → population mean | Sample mean distribution → Normal |
    | About convergence to a value | About shape of distribution |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Asymptotic behavior understanding.
        
        **Strong answer signals:**
        
        - Distinguishes LLN from CLT
        - Knows weak vs strong forms
        - Explains practical implications
        - Shows convergence concept

---

### What is a PDF vs PMF vs CDF? - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `PDF`, `PMF`, `CDF`, `Distributions` | **Asked by:** Google, Amazon, Meta, Microsoft

??? success "View Answer"

    **Probability Mass Function (PMF):**
    
    For discrete random variables:
    
    $$P(X = x) = p(x)$$
    
    Properties:
    - p(x) ≥ 0
    - Σp(x) = 1
    
    **Probability Density Function (PDF):**
    
    For continuous random variables:
    
    $$P(a < X < b) = \int_a^b f(x) dx$$
    
    Properties:
    - f(x) ≥ 0
    - ∫f(x)dx = 1
    - P(X = a) = 0 for any exact value!
    
    **Cumulative Distribution Function (CDF):**
    
    For both discrete and continuous:
    
    $$F(x) = P(X \leq x)$$
    
    Properties:
    - F(-∞) = 0, F(+∞) = 1
    - Monotonically non-decreasing
    - F'(x) = f(x) for continuous
    
    **Visual Comparison:**
    
    ```python
    from scipy import stats
    import numpy as np
    
    # Discrete: Binomial PMF and CDF
    x_discrete = np.arange(0, 11)
    pmf = stats.binom.pmf(x_discrete, n=10, p=0.5)
    cdf_discrete = stats.binom.cdf(x_discrete, n=10, p=0.5)
    
    # Continuous: Normal PDF and CDF
    x_continuous = np.linspace(-4, 4, 100)
    pdf = stats.norm.pdf(x_continuous)
    cdf_continuous = stats.norm.cdf(x_continuous)
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Distribution fundamentals.
        
        **Strong answer signals:**
        
        - Knows PDF ≠ probability (can exceed 1)
        - Uses CDF for probability calculations
        - Knows F'(x) = f(x) relationship
        - Distinguishes discrete from continuous

---

### What is a Confidence Interval? How to Interpret It? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Confidence Interval`, `Inference`, `Uncertainty` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Confidence Interval:**
    
    Range that likely contains true population parameter:
    
    $$CI = \text{estimate} \pm \text{margin of error}$$
    
    **For Mean (known σ):**
    
    $$CI = \bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$
    
    **For Mean (unknown σ):**
    
    $$CI = \bar{x} \pm t_{\alpha/2, n-1} \cdot \frac{s}{\sqrt{n}}$$
    
    **Common z-values:**
    
    | Confidence | z-value |
    |------------|---------|
    | 90% | 1.645 |
    | 95% | 1.96 |
    | 99% | 2.576 |
    
    **Correct Interpretation:**
    
    ✅ "95% of such intervals contain the true mean"
    ❌ "95% probability the true mean is in this interval"
    
    **Python:**
    
    ```python
    from scipy import stats
    import numpy as np
    
    data = [23, 25, 28, 22, 26, 27, 24, 29, 25, 26]
    
    # 95% CI for mean
    mean = np.mean(data)
    se = stats.sem(data)  # Standard error
    ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=se)
    
    print(f"95% CI: ({ci[0]:.2f}, {ci[1]:.2f})")
    ```
    
    **Width Factors:**
    
    - Higher confidence → wider CI
    - Larger n → narrower CI
    - More variability → wider CI

    !!! tip "Interviewer's Insight"
        **What they're testing:** Statistical inference understanding.
        
        **Strong answer signals:**
        
        - Correct frequentist interpretation
        - Knows t vs z distribution choice
        - Understands factors affecting width
        - Can calculate by hand

---

### Explain Hypothesis Testing: Null, Alternative, p-value - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Hypothesis Testing`, `p-value`, `Significance` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Hypothesis Testing Framework:**
    
    | Component | Description |
    |-----------|-------------|
    | H₀ (Null) | Default assumption (no effect) |
    | H₁ (Alternative) | What we want to prove |
    | α (Significance) | False positive threshold (usually 0.05) |
    | p-value | P(data \| H₀ true) |
    
    **Decision Rule:**
    
    - If p-value ≤ α: Reject H₀
    - If p-value > α: Fail to reject H₀
    
    **Types of Errors:**
    
    | Error | Description | Name |
    |-------|-------------|------|
    | Type I | Reject H₀ when true | False Positive |
    | Type II | Accept H₀ when false | False Negative |
    
    **Example - A/B Test:**
    
    ```python
    from scipy import stats
    
    # Control: 100 conversions out of 1000
    # Treatment: 120 conversions out of 1000
    
    control_conv = 100
    control_n = 1000
    treatment_conv = 120
    treatment_n = 1000
    
    # H₀: p1 = p2 (no difference)
    # H₁: p1 ≠ p2 (difference exists)
    
    # Two-proportion z-test
    from statsmodels.stats.proportion import proportions_ztest
    
    stat, pvalue = proportions_ztest(
        [control_conv, treatment_conv],
        [control_n, treatment_n]
    )
    
    print(f"p-value: {pvalue:.4f}")
    # If p < 0.05, reject H₀ → significant difference
    ```
    
    **p-value Misconceptions:**
    
    ❌ p-value = P(H₀ is true)
    ✅ p-value = P(observing this data or more extreme | H₀ true)

    !!! tip "Interviewer's Insight"
        **What they're testing:** Core statistical testing knowledge.
        
        **Strong answer signals:**
        
        - Correct p-value interpretation
        - Knows Type I vs Type II errors
        - Understands "fail to reject" vs "accept"
        - Can set up hypotheses correctly

---

### What is Power in Hypothesis Testing? - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Power`, `Type II Error`, `Sample Size` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **Power:**
    
    Probability of correctly rejecting H₀ when it's false:
    
    $$\text{Power} = 1 - \beta = P(\text{Reject } H_0 | H_0 \text{ is false})$$
    
    **Factors Affecting Power:**
    
    | Factor | Effect on Power |
    |--------|-----------------|
    | Effect size ↑ | Power ↑ |
    | Sample size ↑ | Power ↑ |
    | α ↑ | Power ↑ |
    | Variance ↓ | Power ↑ |
    
    **Typical Target: Power = 0.80**
    
    **Power Analysis - Sample Size Calculation:**
    
    ```python
    from statsmodels.stats.power import TTestIndPower
    
    # Parameters
    effect_size = 0.5  # Cohen's d (medium effect)
    alpha = 0.05
    power = 0.80
    
    # Calculate required sample size
    analysis = TTestIndPower()
    n = analysis.solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        ratio=1  # Equal group sizes
    )
    
    print(f"Required n per group: {n:.0f}")
    # ~64 per group for medium effect
    ```
    
    **Effect Size (Cohen's d):**
    
    $$d = \frac{\mu_1 - \mu_2}{\sigma}$$
    
    | d | Interpretation |
    |---|---------------|
    | 0.2 | Small |
    | 0.5 | Medium |
    | 0.8 | Large |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Experimental design knowledge.
        
        **Strong answer signals:**
        
        - Knows power = 1 - β
        - Can perform power analysis
        - Understands sample size trade-offs
        - Uses effect size appropriately

---

### Explain Permutations vs Combinations - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Combinatorics`, `Counting`, `Fundamentals` | **Asked by:** Google, Amazon, Meta, Microsoft

??? success "View Answer"

    **Permutations (Order Matters):**
    
    $$P(n, r) = \frac{n!}{(n-r)!}$$
    
    **Combinations (Order Doesn't Matter):**
    
    $$C(n, r) = \binom{n}{r} = \frac{n!}{r!(n-r)!}$$
    
    **Key Relationship:**
    
    $$P(n, r) = C(n, r) \cdot r!$$
    
    **Examples:**
    
    ```python
    from math import factorial, comb, perm
    
    # 5 people, select 3 for positions (President, VP, Secretary)
    # Order matters → Permutation
    positions = perm(5, 3)  # = 5 × 4 × 3 = 60
    
    # 5 people, select 3 for a committee
    # Order doesn't matter → Combination
    committee = comb(5, 3)  # = 10
    
    # Relationship
    assert perm(5, 3) == comb(5, 3) * factorial(3)
    # 60 = 10 × 6
    ```
    
    **With Repetition:**
    
    | Type | Formula |
    |------|---------|
    | Permutation with repetition | nʳ |
    | Combination with repetition | C(n+r-1, r) |
    
    ```python
    # 4-digit PIN (0-9): 10^4 = 10000
    # Choose 3 scoops from 5 flavors (repeats OK): C(5+3-1, 3) = C(7,3) = 35
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Basic counting principles.
        
        **Strong answer signals:**
        
        - Immediate recognition of order relevance
        - Knows formulas without derivation
        - Distinguishes with vs without replacement
        - Gives intuitive examples

---

### What is the Negative Binomial Distribution? - Amazon, Microsoft Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Negative Binomial`, `Discrete`, `Failures` | **Asked by:** Amazon, Microsoft, Google

??? success "View Answer"

    **Negative Binomial Distribution:**
    
    Number of trials until rth success:
    
    $$P(X = k) = \binom{k-1}{r-1} p^r (1-p)^{k-r}$$
    
    **Alternative: Number of failures before rth success (Y = X - r)**
    
    **Parameters:**
    
    | Statistic | Formula |
    |-----------|---------|
    | Mean | r/p |
    | Variance | r(1-p)/p² |
    
    **Special Case:**
    
    When r = 1: Negative Binomial → Geometric
    
    **Example - Quality Control:**
    
    Need 3 good widgets. P(good) = 0.8. Expected total inspections?
    
    ```python
    from scipy.stats import nbinom
    
    r, p = 3, 0.8
    
    # Expected trials until 3 successes
    expected_trials = r / p  # = 3 / 0.8 = 3.75
    
    # P(need exactly 5 trials)?
    # 5 trials, 3 successes, 2 failures
    p_5 = nbinom.pmf(k=2, n=3, p=0.8)  # k = failures
    # = C(4,2) * 0.8^3 * 0.2^2 = 0.0512
    
    # P(need at most 4 trials)?
    p_le_4 = nbinom.cdf(k=1, n=3, p=0.8)  # ≤1 failure
    ```
    
    **Applications:**
    
    - Number of sales calls until quota
    - Waiting for multiple events
    - Overdispersed count data

    !!! tip "Interviewer's Insight"
        **What they're testing:** Generalized geometric distribution.
        
        **Strong answer signals:**
        
        - Knows relationship to geometric
        - Handles both parameterizations
        - Calculates mean = r/p
        - Gives practical applications

---

### What is the Beta Distribution? - Amazon, Google Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Beta`, `Continuous`, `Bayesian` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **Beta Distribution:**
    
    Models probabilities (values in [0, 1]):
    
    $$f(x; \alpha, \beta) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}$$
    
    **Parameters:**
    
    | Statistic | Formula |
    |-----------|---------|
    | Mean | α / (α + β) |
    | Mode | (α-1) / (α+β-2) for α,β > 1 |
    | Variance | αβ / [(α+β)²(α+β+1)] |
    
    **Special Cases:**
    
    | α | β | Shape |
    |---|---|-------|
    | 1 | 1 | Uniform |
    | 0.5 | 0.5 | U-shaped |
    | 2 | 5 | Left-skewed |
    | 5 | 2 | Right-skewed |
    
    **Bayesian Application:**
    
    Prior for probability p, with binomial likelihood:
    
    ```python
    from scipy.stats import beta
    import numpy as np
    
    # Prior: Beta(2, 2) - slight preference for 0.5
    # Observed: 7 successes, 3 failures
    # Posterior: Beta(2+7, 2+3) = Beta(9, 5)
    
    prior_alpha, prior_beta = 2, 2
    successes, failures = 7, 3
    
    post_alpha = prior_alpha + successes
    post_beta = prior_beta + failures
    
    posterior = beta(post_alpha, post_beta)
    
    mean = posterior.mean()  # 9/14 ≈ 0.643
    ci = posterior.interval(0.95)  # 95% credible interval
    ```
    
    **Why Use Beta?**
    
    - Conjugate prior for binomial
    - Posterior is also Beta
    - Flexible shape for [0,1] data

    !!! tip "Interviewer's Insight"
        **What they're testing:** Bayesian statistics foundation.
        
        **Strong answer signals:**
        
        - Knows it models probabilities
        - Uses as prior in Bayesian inference
        - Understands conjugacy
        - Can update with observed data

---

### What is the Gamma Distribution? - Amazon, Google Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Gamma`, `Continuous`, `Waiting` | **Asked by:** Amazon, Google, Microsoft

??? success "View Answer"

    **Gamma Distribution:**
    
    Generalized exponential - time until kth event:
    
    $$f(x; k, \theta) = \frac{x^{k-1}e^{-x/\theta}}{\theta^k \Gamma(k)}$$
    
    **Parameters:**
    
    | Statistic | Formula |
    |-----------|---------|
    | Mean | kθ |
    | Variance | kθ² |
    | Mode | (k-1)θ for k ≥ 1 |
    
    **Special Cases:**
    
    | Distribution | Gamma Parameters |
    |--------------|------------------|
    | Exponential | k = 1 |
    | Chi-squared | k = ν/2, θ = 2 |
    | Erlang | k ∈ integers |
    
    **Application:**
    
    ```python
    from scipy.stats import gamma
    
    # Phone calls: avg 3 per hour (λ=3)
    # Time until 5th call?
    
    k = 5  # 5th event
    theta = 1/3  # Scale = 1/rate
    
    waiting = gamma(a=k, scale=theta)
    
    # Expected wait
    expected = waiting.mean()  # = 5 * (1/3) = 1.67 hours
    
    # P(wait > 2 hours)?
    p_gt_2 = 1 - waiting.cdf(2)
    ```
    
    **Relationship to Poisson:**
    
    - Poisson: count in fixed time
    - Gamma: time until kth count

    !!! tip "Interviewer's Insight"
        **What they're testing:** Advanced distribution knowledge.
        
        **Strong answer signals:**
        
        - Knows exponential is Gamma(1, θ)
        - Connects to Poisson process
        - Uses for waiting time problems
        - Knows chi-squared is special gamma

---


### What is a Markov Chain? - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Markov Chain`, `Stochastic Process`, `Probability` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Markov Chain:**
    
    Stochastic process with memoryless property:
    
    $$P(X_{n+1} = j | X_n = i, X_{n-1}, ..., X_0) = P(X_{n+1} = j | X_n = i)$$
    
    "Future depends only on present, not past"
    
    **Components:**
    
    - States: Finite or infinite set
    - Transition probabilities: P(i → j)
    - Transition matrix: P where Pᵢⱼ = P(i → j)
    
    **Example - Weather:**
    
    ```python
    import numpy as np
    
    # States: Sunny (0), Rainy (1)
    # P[i,j] = probability of going from i to j
    P = np.array([
        [0.8, 0.2],  # Sunny → Sunny=0.8, Rainy=0.2
        [0.4, 0.6]   # Rainy → Sunny=0.4, Rainy=0.6
    ])
    
    # After n steps from initial state
    def state_after_n_steps(P, initial, n):
        return np.linalg.matrix_power(P, n)[initial]
    
    # Stationary distribution (π = πP)
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    stationary = eigenvectors[:, 0].real
    stationary = stationary / stationary.sum()
    # [0.667, 0.333] - long-run: 66.7% sunny
    ```
    
    **Key Properties:**
    
    - Irreducible: Can reach any state from any other
    - Aperiodic: No fixed cycles
    - Ergodic: Irreducible + aperiodic → unique stationary dist

    !!! tip "Interviewer's Insight"
        **What they're testing:** Stochastic modeling knowledge.
        
        **Strong answer signals:**
        
        - States memoryless property clearly
        - Can write transition matrix
        - Knows stationary distribution concept
        - Gives PageRank as application

---

### What is Entropy in Information Theory? - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Entropy`, `Information Theory`, `Uncertainty` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **Shannon Entropy:**
    
    Measures uncertainty/information content:
    
    $$H(X) = -\sum_x P(x) \log_2 P(x)$$
    
    **Properties:**
    
    | Distribution | Entropy |
    |--------------|---------|
    | Uniform | Maximum (log₂n for n outcomes) |
    | Deterministic | 0 (no uncertainty) |
    | Binary (p=0.5) | 1 bit |
    
    **Example:**
    
    ```python
    import numpy as np
    
    def entropy(probs):
        """Calculate Shannon entropy in bits"""
        probs = np.array(probs)
        probs = probs[probs > 0]  # Avoid log(0)
        return -np.sum(probs * np.log2(probs))
    
    # Fair coin: maximum entropy
    fair_coin = entropy([0.5, 0.5])  # 1.0 bit
    
    # Biased coin
    biased = entropy([0.9, 0.1])  # 0.47 bits
    
    # Fair die
    fair_die = entropy([1/6] * 6)  # 2.58 bits
    ```
    
    **Cross-Entropy (ML Loss):**
    
    $$H(p, q) = -\sum_x p(x) \log q(x)$$
    
    **KL Divergence:**
    
    $$D_{KL}(p||q) = H(p, q) - H(p)$$

    !!! tip "Interviewer's Insight"
        **What they're testing:** Information theory fundamentals.
        
        **Strong answer signals:**
        
        - Knows entropy measures uncertainty
        - Uses log₂ for bits, ln for nats
        - Connects to ML cross-entropy loss
        - Understands maximum entropy principle

---

### What are Joint and Marginal Distributions? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Joint Distribution`, `Marginal`, `Multivariate` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Joint Distribution:**
    
    Probability distribution over multiple variables:
    
    $$P(X=x, Y=y) = P(X=x \cap Y=y)$$
    
    **Marginal Distribution:**
    
    Distribution of single variable from joint:
    
    $$P(X=x) = \sum_y P(X=x, Y=y)$$
    
    **Example:**
    
    ```python
    import numpy as np
    
    # Joint PMF of X and Y
    joint = np.array([
        [0.1, 0.2, 0.1],  # X=0
        [0.2, 0.2, 0.1],  # X=1
        [0.0, 0.05, 0.05] # X=2
    ])
    # Columns: Y=0, Y=1, Y=2
    
    # Marginal of X (sum over Y)
    marginal_x = joint.sum(axis=1)  # [0.4, 0.5, 0.1]
    
    # Marginal of Y (sum over X)
    marginal_y = joint.sum(axis=0)  # [0.3, 0.45, 0.25]
    
    # Conditional P(Y|X=1)
    conditional_y_given_x1 = joint[1] / marginal_x[1]
    # [0.4, 0.4, 0.2]
    ```
    
    **Independence Check:**
    
    X and Y independent iff:
    P(X=x, Y=y) = P(X=x) · P(Y=y) for all x, y

    !!! tip "Interviewer's Insight"
        **What they're testing:** Multivariate probability.
        
        **Strong answer signals:**
        
        - Knows marginalization = summing out
        - Can derive conditional from joint
        - Checks independence via product rule
        - Extends to continuous case

---

### What is the Chi-Squared Distribution and Test? - Amazon, Microsoft Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Chi-Squared`, `Hypothesis Testing`, `Categorical` | **Asked by:** Amazon, Microsoft, Google

??? success "View Answer"

    **Chi-Squared Distribution:**
    
    Sum of squared standard normals:
    
    $$\chi^2_k = Z_1^2 + Z_2^2 + ... + Z_k^2$$
    
    where Zᵢ ~ N(0,1) and k = degrees of freedom
    
    **Chi-Squared Test for Independence:**
    
    Tests if two categorical variables are independent:
    
    $$\chi^2 = \sum \frac{(O - E)^2}{E}$$
    
    - O = observed frequency
    - E = expected frequency (under independence)
    
    **Example:**
    
    ```python
    from scipy.stats import chi2_contingency
    import numpy as np
    
    # Observed: Gender vs Product Preference
    observed = np.array([
        [30, 10, 15],  # Male: A, B, C
        [20, 25, 10]   # Female: A, B, C
    ])
    
    chi2, p_value, dof, expected = chi2_contingency(observed)
    
    print(f"Chi-squared: {chi2:.2f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Degrees of freedom: {dof}")
    
    # If p < 0.05: Reject H₀ → Variables are dependent
    ```
    
    **Chi-Squared Goodness of Fit:**
    
    Tests if data follows expected distribution:
    
    ```python
    from scipy.stats import chisquare
    
    observed = [18, 22, 28, 32]  # Dice rolls
    expected = [25, 25, 25, 25]   # Fair die
    
    stat, p = chisquare(observed, expected)
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Categorical data analysis.
        
        **Strong answer signals:**
        
        - Knows χ² tests independence/goodness-of-fit
        - Calculates expected under null
        - Uses at least 5 per cell rule
        - Interprets p-value correctly

---

### What is the t-Distribution? When to Use It? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `t-Distribution`, `Small Samples`, `Inference` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **t-Distribution:**
    
    For inference when σ is unknown (uses sample s):
    
    $$t = \frac{\bar{X} - \mu}{s / \sqrt{n}}$$
    
    **Properties:**
    
    | Property | Value |
    |----------|-------|
    | Mean | 0 (for ν > 1) |
    | Variance | ν/(ν-2) for ν > 2 |
    | Shape | Bell-shaped, heavier tails than Normal |
    | DOF → ∞ | Converges to N(0,1) |
    
    **When to Use:**
    
    | Use t | Use z |
    |-------|-------|
    | σ unknown | σ known |
    | Small n (< 30) | Large n (n ≥ 30) |
    | Population ~normal | CLT applies |
    
    **Python:**
    
    ```python
    from scipy import stats
    
    # t-test: is population mean = 100?
    data = [102, 98, 105, 99, 103, 101, 97, 104]
    
    # One-sample t-test
    t_stat, p_value = stats.ttest_1samp(data, 100)
    
    # Critical value for 95% CI (df = n-1)
    t_crit = stats.t.ppf(0.975, df=len(data)-1)
    
    # Two-sample t-test
    group1 = [23, 25, 28, 22, 26]
    group2 = [19, 21, 24, 20, 22]
    t_stat, p_value = stats.ttest_ind(group1, group2)
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Small sample inference.
        
        **Strong answer signals:**
        
        - Knows to use t when σ unknown
        - States heavier tails than normal
        - Uses correct degrees of freedom
        - Knows t → z as n → ∞

---

### What is the Uniform Distribution? - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Uniform`, `Continuous`, `Random` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Continuous Uniform Distribution:**
    
    Equal probability over interval [a, b]:
    
    $$f(x) = \frac{1}{b-a}, \quad a \leq x \leq b$$
    
    **Properties:**
    
    | Statistic | Formula |
    |-----------|---------|
    | Mean | (a + b) / 2 |
    | Variance | (b - a)² / 12 |
    | CDF | (x - a) / (b - a) |
    
    **Discrete Uniform:**
    
    $$P(X = k) = \frac{1}{n}$$
    
    for k in {1, 2, ..., n}
    
    **Python:**
    
    ```python
    from scipy.stats import uniform
    import numpy as np
    
    # Uniform[0, 1]
    U = uniform(loc=0, scale=1)
    
    # Generate random samples
    samples = np.random.uniform(0, 1, 1000)
    
    # Uniform[2, 8]
    U = uniform(loc=2, scale=6)  # loc=a, scale=b-a
    U.mean()  # 5.0
    U.var()   # 3.0
    ```
    
    **Inverse Transform Sampling:**
    
    If U ~ Uniform(0,1), then F⁻¹(U) has distribution F:
    
    ```python
    # Generate exponential from uniform
    u = np.random.uniform(0, 1, 1000)
    exponential_samples = -np.log(1 - u)  # Inverse CDF of Exp(1)
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Basic distribution knowledge.
        
        **Strong answer signals:**
        
        - Knows mean = (a+b)/2
        - Uses for random number generation
        - Knows inverse transform method
        - Distinguishes continuous vs discrete

---

### Explain Sampling With vs Without Replacement - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Sampling`, `Replacement`, `Combinatorics` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **With Replacement:**
    
    - Each item can be selected multiple times
    - Trials are independent
    - Probabilities remain constant
    
    **Without Replacement:**
    
    - Each item selected at most once
    - Trials are dependent
    - Probabilities change after each selection
    
    **Example:**
    
    ```python
    import numpy as np
    
    population = [1, 2, 3, 4, 5]
    
    # With replacement - same item can appear multiple times
    with_rep = np.random.choice(population, size=3, replace=True)
    # Possible: [3, 3, 1]
    
    # Without replacement - unique items only
    without_rep = np.random.choice(population, size=3, replace=False)
    # Possible: [4, 1, 3] but never [3, 3, 1]
    ```
    
    **Probability Differences:**
    
    Drawing 2 red cards from deck:
    
    ```python
    # With replacement
    p_with = (26/52) * (26/52) = 0.25
    
    # Without replacement  
    p_without = (26/52) * (25/51) ≈ 0.245
    ```
    
    **When Each is Used:**
    
    | With Replacement | Without Replacement |
    |------------------|---------------------|
    | Bootstrap sampling | Survey sampling |
    | Dice rolling | Lottery |
    | Monte Carlo | Card dealing |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Sampling concepts.
        
        **Strong answer signals:**
        
        - Knows independence implications
        - Can calculate both scenarios
        - Mentions hypergeometric for without
        - Knows bootstrap uses with replacement

---

### What is the Hypergeometric Distribution? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Hypergeometric`, `Sampling`, `Without Replacement` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Hypergeometric Distribution:**
    
    Successes in n draws without replacement:
    
    $$P(X = k) = \frac{\binom{K}{k}\binom{N-K}{n-k}}{\binom{N}{n}}$$
    
    - N = population size
    - K = successes in population
    - n = sample size
    - k = successes in sample
    
    **Example - Quality Control:**
    
    Lot of 100 items, 10 defective. Sample 15 without replacement.
    
    ```python
    from scipy.stats import hypergeom
    
    N, K, n = 100, 10, 15
    
    # P(exactly 2 defective)?
    p_2 = hypergeom.pmf(k=2, M=N, n=K, N=n)
    
    # Expected defectives
    expected = n * K / N  # = 15 * 10/100 = 1.5
    
    # P(at least 1 defective)?
    p_at_least_1 = 1 - hypergeom.pmf(k=0, M=N, n=K, N=n)
    ```
    
    **Comparison with Binomial:**
    
    | Hypergeometric | Binomial |
    |----------------|----------|
    | Without replacement | With replacement |
    | p changes | p constant |
    | Var < np(1-p) | Var = np(1-p) |
    
    **For large N, hypergeometric ≈ binomial**

    !!! tip "Interviewer's Insight"
        **What they're testing:** Finite population sampling.
        
        **Strong answer signals:**
        
        - Knows formula intuitively
        - Compares to binomial
        - Uses for quality control problems
        - Knows approximation for large N

---

### What is the F-Distribution? - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `F-Distribution`, `ANOVA`, `Variance Comparison` | **Asked by:** Google, Amazon, Microsoft

??? success "View Answer"

    **F-Distribution:**
    
    Ratio of two chi-squared distributions:
    
    $$F = \frac{\chi^2_1 / d_1}{\chi^2_2 / d_2}$$
    
    **Use Cases:**
    
    1. ANOVA (compare group means)
    2. Comparing variances
    3. Regression overall significance
    
    **F-Test for Variance:**
    
    ```python
    from scipy import stats
    import numpy as np
    
    # Compare variances of two samples
    sample1 = [23, 25, 28, 22, 26, 27]
    sample2 = [19, 31, 24, 28, 20, 35]
    
    var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
    
    f_stat = var1 / var2
    df1, df2 = len(sample1) - 1, len(sample2) - 1
    
    p_value = 2 * min(
        stats.f.cdf(f_stat, df1, df2),
        1 - stats.f.cdf(f_stat, df1, df2)
    )
    ```
    
    **One-Way ANOVA:**
    
    ```python
    from scipy.stats import f_oneway
    
    group1 = [85, 90, 88, 92, 87]
    group2 = [78, 82, 80, 79, 81]
    group3 = [91, 95, 89, 94, 92]
    
    f_stat, p_value = f_oneway(group1, group2, group3)
    # If p < 0.05: At least one group mean differs
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Advanced statistical tests.
        
        **Strong answer signals:**
        
        - Knows F = ratio of variances
        - Uses for ANOVA and regression
        - Understands two df parameters
        - Can interpret F-stat and p-value

---

### How Do You Calculate Sample Size for A/B Tests? - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `A/B Testing`, `Sample Size`, `Power` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **Sample Size Formula (Two Proportions):**
    
    $$n = \frac{2(z_{\alpha/2} + z_{\beta})^2 \bar{p}(1-\bar{p})}{\delta^2}$$
    
    where:
    - δ = minimum detectable effect
    - p̄ = average proportion
    - α = significance level (usually 0.05)
    - 1-β = power (usually 0.80)
    
    **Python Calculation:**
    
    ```python
    from statsmodels.stats.power import NormalIndPower
    from statsmodels.stats.proportion import proportion_effectsize
    
    # Current conversion: 10%
    # Want to detect: 2% absolute lift (to 12%)
    p1, p2 = 0.10, 0.12
    
    # Effect size
    effect_size = proportion_effectsize(p1, p2)
    
    # Power analysis
    power_analysis = NormalIndPower()
    n_per_group = power_analysis.solve_power(
        effect_size=effect_size,
        alpha=0.05,
        power=0.80,
        ratio=1
    )
    
    print(f"Required per group: {n_per_group:.0f}")
    # ~3,600 per group for 2% lift detection
    ```
    
    **Rule of Thumb:**
    
    For 80% power, 5% significance:
    - 1% absolute lift: ~15,000 per group
    - 2% absolute lift: ~3,800 per group
    - 5% absolute lift: ~600 per group
    
    **Factors:**
    
    | Factor | Effect on n |
    |--------|-------------|
    | Smaller effect → | Larger n |
    | Higher power → | Larger n |
    | Lower α → | Larger n |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Experimental design skills.
        
        **Strong answer signals:**
        
        - Knows key inputs (effect, power, α)
        - Uses standard library for calculation
        - Understands trade-offs
        - Gives practical rule of thumb

---

### What is Bayesian vs Frequentist Probability? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Bayesian`, `Frequentist`, `Philosophy` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Frequentist:**
    
    - Probability = long-run frequency
    - Parameters are fixed (unknown constants)
    - Inference via sampling distribution
    - Uses p-values and confidence intervals
    
    **Bayesian:**
    
    - Probability = degree of belief
    - Parameters have distributions
    - Inference via Bayes' theorem
    - Uses posterior and credible intervals
    
    **Comparison:**
    
    | Aspect | Frequentist | Bayesian |
    |--------|-------------|----------|
    | Probability | Long-run frequency | Belief/uncertainty |
    | Parameters | Fixed | Random |
    | Prior info | Not used | Used explicitly |
    | Intervals | 95% CI: "95% of intervals contain true value" | 95% credible: "95% probability parameter in interval" |
    
    **Example:**
    
    ```python
    # Frequentist: p-value
    from scipy.stats import ttest_1samp
    data = [52, 48, 55, 49, 51]
    t_stat, p_value = ttest_1samp(data, 50)
    
    # Bayesian: posterior
    import pymc as pm
    with pm.Model():
        mu = pm.Normal('mu', mu=50, sigma=10)  # Prior
        obs = pm.Normal('obs', mu=mu, sigma=3, observed=data)
        trace = pm.sample(1000)
    # 95% credible interval from posterior
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Statistical philosophy understanding.
        
        **Strong answer signals:**
        
        - Explains both paradigms fairly
        - Knows interval interpretation difference
        - Mentions when each is preferred
        - Doesn't dogmatically favor one

---

### What is the Multiple Comparisons Problem? - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Multiple Testing`, `FWER`, `FDR` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **The Problem:**
    
    With many tests at α=0.05, false positives accumulate:
    
    P(at least 1 false positive) = 1 - (1-α)ⁿ
    
    - 20 tests: 64% chance of false positive
    - 100 tests: 99.4% chance!
    
    **Solutions:**
    
    **1. Bonferroni Correction (FWER):**
    
    Use α/n for each test:
    
    ```python
    n_tests = 20
    alpha = 0.05
    bonferroni_alpha = alpha / n_tests  # 0.0025
    ```
    
    Conservative but controls family-wise error rate.
    
    **2. Benjamini-Hochberg (FDR):**
    
    Controls false discovery rate:
    
    ```python
    from scipy.stats import false_discovery_control
    
    p_values = [0.001, 0.008, 0.012, 0.045, 0.060, 0.120]
    
    # Adjust p-values
    adjusted = false_discovery_control(p_values, method='bh')
    
    # Or manually:
    sorted_p = sorted(p_values)
    n = len(p_values)
    for i, p in enumerate(sorted_p):
        threshold = (i + 1) / n * alpha
        print(f"p={p:.3f}, threshold={threshold:.3f}")
    ```
    
    **When to Use:**
    
    | Method | Use Case |
    |--------|----------|
    | No correction | Single pre-specified test |
    | Bonferroni | Few tests, must avoid any FP |
    | BH | Many tests, some FP acceptable |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Rigorous testing knowledge.
        
        **Strong answer signals:**
        
        - Explains why it's a problem
        - Knows Bonferroni is conservative
        - Uses FDR for exploratory analysis
        - Applies to A/B testing scenarios

---

### What is Bootstrap Sampling? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Bootstrap`, `Resampling`, `Non-parametric` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Bootstrap:**
    
    Resampling with replacement to estimate sampling distribution.
    
    **Process:**
    
    1. Draw n samples with replacement from data
    2. Calculate statistic of interest
    3. Repeat B times (e.g., 10,000)
    4. Use distribution of statistics for inference
    
    **Example:**
    
    ```python
    import numpy as np
    
    data = [23, 25, 28, 22, 26, 27, 30, 24, 29, 25]
    n_bootstrap = 10000
    
    # Bootstrap confidence interval for mean
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    # 95% CI (percentile method)
    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)
    
    print(f"95% CI: ({ci_lower:.2f}, {ci_upper:.2f})")
    ```
    
    **Use Cases:**
    
    - Confidence intervals for any statistic
    - Estimating standard errors
    - When distribution unknown
    - Complex statistics (median, ratios)
    
    **Types:**
    
    | Method | Description |
    |--------|-------------|
    | Percentile | Use quantiles directly |
    | Basic | Reflect around estimate |
    | BCa | Bias-corrected accelerated |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Modern statistical methods.
        
        **Strong answer signals:**
        
        - Knows to resample WITH replacement
        - Uses for non-standard statistics
        - Knows different CI methods
        - Mentions computational cost

---

### Average score on a dice role of at most 3 times
**Difficulty:** 🔴 Hard | **Tags:** `Probability`, `Expected Value`, `Game Theory` | **Asked by:** Jane Street, Hudson River Trading, Citadel

??? question "Full Question" 

    Consider a fair 6-sided dice. 
    Your aim is to get the highest score you can, in at-most 3 roles.

    A score is defined as the number that appears on the face of the dice facing up after the role. 
    You can role at most 3 times but every time you role it is up to you to decide whether you want to role again.

    The last score will be counted as your final score.

    - Find the average score if you rolled the dice only once?
    - Find the average score that you can get with at most 3 roles?
    - If the dice is fair, why is the average score for at most 3 roles and 1 role not the same?

??? info "Hint 1"
    Find what is the expected score on single role

    And for cases when scores of single role < `expected score on single role` 
    is when you will go for next role

    Eg: if expected score of single role comes out to be 4.5, 
    you will only role next turn for 1,2,3,4 and not for 5,6

??? success "Answer"

    If you role a fair dice once you can get:

    | Score  | Probability  |
    |:-:|:-:|
    | 1 | 1/6 |
    | 2 | 1/6 |
    | 3 | 1/6 |
    | 4 | 1/6 |
    | 5 | 1/6 |
    | 6 | 1/6 |


    So your average score with one role is: 

    `sum of(score * scores's probability)` = (1+2+3+4+5+6)*(1/6) = (21/6) = 3.5

    __The average score if you rolled the dice only once is 3.5__

    For at most 3 roles, let's try back-tracking. Let's say just did your second role and you have to decide whether to do your 3rd role!

    We just found out if we role dice once on average we can expect score of 3.5. So we will only role the 3rd time if score on 2nd role is less than 3.5 i.e (1,2 or 3)

    Possibilities

    | 2nd role score  | Probability  | 3rd role score  | Probability  |
    |:-:|:-:|:-:|:-:|
    | 1 | 1/6 | 3.5 | 1/6 |
    | 2 | 1/6 | 3.5 | 1/6 |
    | 3 | 1/6 | 3.5 | 1/6 |
    | 4 | 1/6 | NA | We won't role|
    | 5 | 1/6 | NA | 3rd time if we|
    | 6 | 1/6 | NA | get score >3 on 2nd|

    So if we had 2 roles, average score would be:

    ```
    [We role again if current score is less than 3.4]
    (3.5)*(1/6) + (3.5)*(1/6) + (3.5)*(1/6) 
    +
    (4)*(1/6) + (5)*(1/6) + (6)*(1/6) [Decide not to role again]
    =
    1.75 + 2.5 = 4.25
    ```

    The average score if you rolled the dice twice is 4.25

    So now if we look from the perspective of first role. We will only role again if our score is less than 4.25 i.e 1,2,3 or 4

    Possibilities

    | 1st role score | Probability | 2nd role score (Exp) | Probability/Note |
    | :---: | :---: | :---: | :---: |
    | 1 | 1/6 | 4.25 | 1/6 |
    | 2 | 1/6 | 4.25 | 1/6 |
    | 3 | 1/6 | 4.25 | 1/6 |
    | 4 | 1/6 | 4.25 | 1/6 |
    | 5 | 1/6 | NA | We won't role again if we|
    | 6 | 1/6 | NA | get score >4.25 on 1st|

    So if we had 3 roles, average score would be:

    ```
    [We role again if current score is less than 4.25]
    (4.25)*(1/6) + (4.25)*(1/6) + (4.25)*(1/6) + (4.25)*(1/6) 
    +
    (5)*(1/6) + (6)*(1/6) [[Decide not to role again]
    =
    17/6 + 11/6 = 4.66
    ```
    __The average score if you rolled the dice only once is 4.66__

    The average score for at most 3 roles and 1 role is not the same because although the dice is fair the event of rolling the dice is no longer __independent__.
    The scores would have been the same if we rolled the dice 2nd and 3rd time without considering what we got in the last roll i.e. if the event of rolling the dice was independent.

    !!! tip "Interviewer's Insight"
        **What they're testing:** Optimal stopping and backward induction.


### Explain the Coupon Collector Problem - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Coupon Collector`, `Expected Value`, `Puzzle` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Problem:**
    
    How many items to collect before getting all n types?
    (Each type equally likely)
    
    **Expected Value:**
    
    $$E[T] = n \cdot H_n = n \cdot \sum_{i=1}^{n} \frac{1}{i}$$
    
    where Hₙ is the nth harmonic number.
    
    **Intuition:**
    
    After collecting k types, expected trials until new type = n/(n-k)
    
    **Example:**
    
    ```python
    import numpy as np
    
    def expected_trials(n):
        """Expected trials to collect all n types"""
        return n * sum(1/i for i in range(1, n+1))
    
    # 6 types (like Pokemon cards)
    print(f"E[trials]: {expected_trials(6):.2f}")  # ~14.7
    
    # Simulation
    def simulate_coupon_collector(n, simulations=10000):
        trials = []
        for _ in range(simulations):
            collected = set()
            count = 0
            while len(collected) < n:
                collected.add(np.random.randint(n))
                count += 1
            trials.append(count)
        return np.mean(trials)
    
    print(f"Simulated: {simulate_coupon_collector(6):.2f}")
    ```
    
    **Applications:**
    
    - A/B testing (all user segments)
    - Load testing (all code paths)
    - Collecting rare items

    !!! tip "Interviewer's Insight"
        **What they're testing:** Probability puzzle solving.
        
        **Strong answer signals:**
        
        - Uses linearity of expectation
        - Knows harmonic series result
        - Can simulate to verify
        - Applies to real scenarios

---

### What is Simpson's Paradox? - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Simpson's Paradox`, `Confounding`, `Causality` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **Simpson's Paradox:**
    
    Trend appears in subgroups but reverses when combined.
    
    **Classic Example - UC Berkeley Admissions:**
    
    | | Men Apply | Men Admit | Women Apply | Women Admit |
    |-|-----------|-----------|-------------|-------------|
    | Overall | 8,442 | 44% | 4,321 | 35% |
    
    Looks like discrimination against women!
    
    But by department:
    
    | Dept | Men Apply | Men % | Women Apply | Women % |
    |------|-----------|-------|-------------|---------|
    | A | 825 | 62% | 108 | 82% |
    | B | 560 | 63% | 25 | 68% |
    | C | 325 | 37% | 593 | 34% |
    
    Women had HIGHER rates in each department!
    
    **Cause:**
    
    Women applied more to competitive departments.
    
    ```python
    import pandas as pd
    
    # Weighted vs unweighted
    data = pd.DataFrame({
        'dept': ['A', 'A', 'B', 'B'],
        'gender': ['M', 'F', 'M', 'F'],
        'applications': [825, 108, 560, 25],
        'rate': [0.62, 0.82, 0.63, 0.68]
    })
    
    # Department is a confounding variable
    ```
    
    **Lesson:**
    
    Always consider lurking/confounding variables before drawing conclusions.

    !!! tip "Interviewer's Insight"
        **What they're testing:** Critical thinking about data.
        
        **Strong answer signals:**
        
        - Gives clear example
        - Identifies confounding variable
        - Knows when to aggregate vs stratify
        - Relates to A/B testing concerns

---

### What Are Quantiles and Percentiles? - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Quantiles`, `Percentiles`, `Descriptive` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Definitions:**
    
    - Quantile: Values dividing distribution into intervals
    - Percentile: Quantile expressed as percentage
    - P-th percentile: Value below which P% of data falls
    
    **Common Quantiles:**
    
    | Name | Divides Into |
    |------|--------------|
    | Median (Q2) | 2 equal parts |
    | Quartiles (Q1, Q2, Q3) | 4 equal parts |
    | Deciles | 10 equal parts |
    | Percentiles | 100 equal parts |
    
    **Calculation:**
    
    ```python
    import numpy as np
    
    data = [12, 15, 18, 20, 22, 25, 28, 30, 35, 40]
    
    # Percentiles
    p25 = np.percentile(data, 25)  # Q1
    p50 = np.percentile(data, 50)  # Median
    p75 = np.percentile(data, 75)  # Q3
    p90 = np.percentile(data, 90)  # 90th percentile
    
    # IQR (Interquartile Range)
    iqr = p75 - p25
    ```
    
    **Uses:**
    
    - Latency: "p99 response time < 100ms"
    - Salaries: "In top 10% earners"
    - Outlier detection: Beyond 1.5*IQR
    
    **Z-score to Percentile:**
    
    ```python
    from scipy.stats import norm
    
    # Z = 1.645 → 95th percentile
    norm.cdf(1.645)  # ≈ 0.95
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Basic statistical literacy.
        
        **Strong answer signals:**
        
        - Knows p50 = median
        - Uses for SLA metrics
        - Can convert z-scores to percentiles
        - Understands IQR for robustness

---

### What is the Difference Between Standard Deviation and Standard Error? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Standard Deviation`, `Standard Error`, `Sampling` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Standard Deviation (SD):**
    
    Measures spread of individual observations:
    
    $$SD = \sqrt{\frac{\sum(x_i - \bar{x})^2}{n-1}}$$
    
    **Standard Error (SE):**
    
    Measures uncertainty in sample mean:
    
    $$SE = \frac{SD}{\sqrt{n}}$$
    
    **Key Difference:**
    
    | SD | SE |
    |----|----|
    | Describes data spread | Describes estimate precision |
    | Doesn't depend on n (conceptually) | Decreases with larger n |
    | Used for z-scores | Used for confidence intervals |
    
    **Example:**
    
    ```python
    import numpy as np
    from scipy.stats import sem
    
    data = [23, 25, 28, 22, 26, 27, 24, 29, 25, 26]
    
    sd = np.std(data, ddof=1)  # Sample SD
    se = sem(data)  # Standard error of mean
    # or se = sd / np.sqrt(len(data))
    
    mean = np.mean(data)
    
    # 95% CI using SE
    ci = (mean - 1.96*se, mean + 1.96*se)
    
    print(f"SD: {sd:.2f}")   # ~2.21
    print(f"SE: {se:.2f}")   # ~0.70
    print(f"95% CI: {ci}")
    ```
    
    **Intuition:**
    
    - SD: "Typical distance of point from mean"
    - SE: "Typical error in our estimate of the mean"

    !!! tip "Interviewer's Insight"
        **What they're testing:** Sampling variability understanding.
        
        **Strong answer signals:**
        
        - Clearly distinguishes the two concepts
        - Knows SE = SD/√n
        - Uses SE for confidence intervals
        - Knows SE decreases with n

---

### What is Moment Generating Function? - Amazon, Microsoft Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `MGF`, `Moments`, `Advanced` | **Asked by:** Amazon, Microsoft, Google

??? success "View Answer"

    **Moment Generating Function (MGF):**
    
    $$M_X(t) = E[e^{tX}] = \sum_x e^{tx} P(X=x)$$
    
    **Why "Moment Generating"?**
    
    nth moment = nth derivative at t=0:
    
    $$E[X^n] = M_X^{(n)}(0)$$
    
    **Properties:**
    
    1. Uniquely determines distribution
    2. Sum of independent RVs: MGF = product of MGFs
    3. Linear transform: M_{aX+b}(t) = e^{bt} M_X(at)
    
    **Examples:**
    
    | Distribution | MGF |
    |--------------|-----|
    | Normal(μ,σ²) | exp(μt + σ²t²/2) |
    | Exponential(λ) | λ/(λ-t) for t < λ |
    | Poisson(λ) | exp(λ(eᵗ-1)) |
    | Binomial(n,p) | (1-p+peᵗ)ⁿ |
    
    **Deriving Moments:**
    
    ```python
    # For Exponential(λ=2): M(t) = 2/(2-t)
    # E[X] = M'(0) = 2/(2-0)² = 1/2
    # E[X²] = M''(0) = 4/(2-0)³ = 1/2
    # Var(X) = E[X²] - (E[X])² = 1/2 - 1/4 = 1/4
    ```
    
    **Application:**
    
    Proving CLT: MGF of sum → MGF of normal

    !!! tip "Interviewer's Insight"
        **What they're testing:** Advanced probability theory.
        
        **Strong answer signals:**
        
        - Knows moment derivation via derivatives
        - Uses for proving sum distributions
        - Knows MGF uniquely identifies distribution
        - Can derive simple moments

---

### What is the Waiting Time Paradox? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Waiting Time`, `Inspection Paradox`, `Counter-intuitive` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **The Paradox:**
    
    Average wait for a bus can exceed half the average interval!
    
    **Explanation:**
    
    You're more likely to arrive during a LONG interval than a short one.
    
    **Mathematical:**
    
    For Poisson arrivals (rate λ):
    - Average interval: 1/λ
    - Expected wait: 1/λ (same as full interval!)
    
    Due to memoryless property.
    
    **Example:**
    
    ```python
    import numpy as np
    
    # Buses every 10 minutes on average (Poisson)
    lambda_rate = 0.1  # per minute
    
    # Simulate arrivals
    n_buses = 10000
    intervals = np.random.exponential(1/lambda_rate, n_buses)
    
    # Arrive at random time within each interval
    random_fraction = np.random.uniform(0, 1, n_buses)
    wait_times = intervals * random_fraction
    
    avg_wait = np.mean(wait_times)
    avg_interval = np.mean(intervals)
    
    print(f"Avg interval: {avg_interval:.1f} min")
    print(f"Avg wait: {avg_wait:.1f} min")
    # Both approximately 10 minutes!
    ```
    
    **Real-World:**
    
    If buses are scheduled (not Poisson), wait ≈ interval/2.
    But with variability, wait increases due to "length-biased sampling."

    !!! tip "Interviewer's Insight"
        **What they're testing:** Counter-intuitive probability.
        
        **Strong answer signals:**
        
        - Explains length-biased sampling
        - Connects to memoryless property
        - Can simulate to demonstrate
        - Knows scheduled vs random arrivals differ

---

### How Do You Estimate Probability from Rare Events? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Rare Events`, `Estimation`, `Confidence` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **The Challenge:**
    
    0 events in n trials. Is probability really 0?
    
    **Rule of Three:**
    
    If 0 events in n trials, 95% confident p < 3/n
    
    ```python
    n = 1000  # trials
    events = 0  # observed
    
    # 95% upper bound
    upper_bound = 3 / n  # 0.003 or 0.3%
    ```
    
    **Bayesian Approach:**
    
    ```python
    from scipy.stats import beta
    
    # Prior: Beta(1, 1) = Uniform
    # Posterior: Beta(1 + k, 1 + n - k)
    
    n, k = 1000, 0
    posterior = beta(1 + k, 1 + n - k)
    
    # 95% credible interval
    ci = posterior.interval(0.95)
    print(f"95% CI: ({ci[0]:.5f}, {ci[1]:.5f})")
    # (0.0, 0.003)
    
    # With 3 events in 1000:
    posterior = beta(1 + 3, 1 + 1000 - 3)
    mean_estimate = posterior.mean()  # ≈ 0.004
    ```
    
    **Methods Comparison:**
    
    | Method | Estimate | CI |
    |--------|----------|----| 
    | MLE (k/n) | 0 | Undefined |
    | Rule of 3 | - | (0, 0.003) |
    | Bayesian | 0.001 | (0, 0.003) |
    | Wilson | 0.0002 | (0, 0.002) |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Practical estimation skills.
        
        **Strong answer signals:**
        
        - Knows Rule of Three for quick bounds
        - Uses Bayesian for proper intervals
        - Doesn't report 0 as point estimate
        - Mentions sample size requirements

---

## Quick Reference: 100+ Interview Questions

| Sno | Question Title                                                        | Practice Links                                                                                                                          | Companies Asking                     | Difficulty | Topics                                   |
|-----|-----------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------|------------|------------------------------------------|
| 1   | Basic Probability Concepts: Definitions of Sample Space, Event, Outcome | [Wikipedia: Probability](https://en.wikipedia.org/wiki/Probability)                                                                     | Google, Amazon, Microsoft            | Easy       | Fundamental Concepts                     |
| 2   | Conditional Probability and Independence                              | [Khan Academy: Conditional Probability](https://www.khanacademy.org/math/statistics-probability/probability-library)                      | Google, Facebook, Amazon             | Medium     | Conditional Probability, Independence    |
| 3   | Bayes’ Theorem: Statement and Application                               | [Wikipedia: Bayes' Theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem)                                                               | Google, Amazon, Microsoft            | Medium     | Bayesian Inference                       |
| 4   | Law of Total Probability                                                | [Wikipedia: Law of Total Probability](https://en.wikipedia.org/wiki/Law_of_total_probability)                                             | Google, Facebook                     | Medium     | Theoretical Probability                  |
| 5   | Expected Value and Variance                                             | [Khan Academy: Expected Value](https://www.khanacademy.org/math/statistics-probability/probability-library)                                | Google, Amazon, Facebook             | Medium     | Random Variables, Moments                |
| 6   | Probability Distributions: Discrete vs. Continuous                        | [Wikipedia: Probability Distribution](https://en.wikipedia.org/wiki/Probability_distribution)                                             | Google, Amazon, Microsoft            | Easy       | Distributions                            |
| 7   | Binomial Distribution: Definition and Applications                      | [Khan Academy: Binomial Distribution](https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library)             | Amazon, Facebook                     | Medium     | Discrete Distributions                   |
| 8   | Poisson Distribution: Characteristics and Uses                          | [Wikipedia: Poisson Distribution](https://en.wikipedia.org/wiki/Poisson_distribution)                                                     | Google, Amazon                       | Medium     | Discrete Distributions                   |
| 9   | Exponential Distribution: Properties and Applications                   | [Wikipedia: Exponential Distribution](https://en.wikipedia.org/wiki/Exponential_distribution)                                             | Google, Amazon                       | Medium     | Continuous Distributions                 |
| 10  | Normal Distribution and the Central Limit Theorem                        | [Khan Academy: Normal Distribution](https://www.khanacademy.org/math/statistics-probability/modeling-distributions-of-data)                | Google, Microsoft, Facebook          | Medium     | Continuous Distributions, CLT            |
| 11  | Law of Large Numbers                                                    | [Wikipedia: Law of Large Numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers)                                                     | Google, Amazon                       | Medium     | Statistical Convergence                  |
| 12  | Covariance and Correlation: Definitions and Differences                 | [Khan Academy: Covariance and Correlation](https://www.khanacademy.org/math/statistics-probability/describing-relationships-quantitatively) | Google, Facebook                     | Medium     | Statistics, Dependency                   |
| 13  | Moment Generating Functions (MGFs)                                      | [Wikipedia: Moment-generating function](https://en.wikipedia.org/wiki/Moment-generating_function)                                         | Amazon, Microsoft                    | Hard       | Random Variables, Advanced Concepts      |
| 14  | Markov Chains: Basics and Applications                                  | [Wikipedia: Markov chain](https://en.wikipedia.org/wiki/Markov_chain)                                                                     | Google, Amazon, Facebook             | Hard       | Stochastic Processes                     |
| 15  | Introduction to Stochastic Processes                                    | [Wikipedia: Stochastic process](https://en.wikipedia.org/wiki/Stochastic_process)                                                         | Google, Microsoft                    | Hard       | Advanced Probability                     |
| 16  | Difference Between Independent and Mutually Exclusive Events            | [Wikipedia: Independent events](https://en.wikipedia.org/wiki/Independence_(probability_theory))                                            | Google, Facebook                     | Easy       | Fundamental Concepts                     |
| 17  | Geometric Distribution: Concept and Use Cases                           | [Wikipedia: Geometric distribution](https://en.wikipedia.org/wiki/Geometric_distribution)                                                 | Amazon, Microsoft                    | Medium     | Discrete Distributions                   |
| 18  | Hypergeometric Distribution: When to Use It                             | [Wikipedia: Hypergeometric distribution](https://en.wikipedia.org/wiki/Hypergeometric_distribution)                                         | Google, Amazon                       | Medium     | Discrete Distributions                   |
| 19  | Confidence Intervals: Definition and Calculation                        | [Khan Academy: Confidence intervals](https://www.khanacademy.org/math/statistics-probability/confidence-intervals)                           | Microsoft, Facebook                  | Medium     | Inferential Statistics                   |
| 20  | Hypothesis Testing: p-values, Type I and Type II Errors                   | [Khan Academy: Hypothesis testing](https://www.khanacademy.org/math/statistics-probability/significance-tests)                              | Google, Amazon, Facebook             | Medium     | Inferential Statistics                   |
| 21  | Chi-Squared Test: Basics and Applications                               | [Wikipedia: Chi-squared test](https://en.wikipedia.org/wiki/Chi-squared_test)                                                             | Amazon, Microsoft                    | Medium     | Inferential Statistics                   |
| 22  | Permutations and Combinations                                            | [Khan Academy: Permutations and Combinations](https://www.khanacademy.org/math/statistics-probability/probability-library)                    | Google, Facebook                     | Easy       | Combinatorics                            |
| 23  | The Birthday Problem and Its Implications                               | [Wikipedia: Birthday problem](https://en.wikipedia.org/wiki/Birthday_problem)                                                             | Google, Amazon                       | Medium     | Probability Puzzles                      |
| 24  | The Monty Hall Problem                                                  | [Wikipedia: Monty Hall problem](https://en.wikipedia.org/wiki/Monty_Hall_problem)                                                         | Google, Facebook                     | Medium     | Probability Puzzles, Conditional Probability |
| 25  | Marginal vs. Conditional Probabilities                                  | [Khan Academy: Conditional Probability](https://www.khanacademy.org/math/statistics-probability/probability-library)                         | Google, Amazon                       | Medium     | Theoretical Concepts                     |
| 26  | Real-World Application of Bayes’ Theorem                                | [Towards Data Science: Bayes’ Theorem Applications](https://towardsdatascience.com/bayes-theorem-in-machine-learning-6a8b5e9ad0f3)          | Google, Amazon                       | Medium     | Bayesian Inference                       |
| 27  | Probability Mass Function (PMF) vs. Probability Density Function (PDF)   | [Wikipedia: Probability density function](https://en.wikipedia.org/wiki/Probability_density_function)                                       | Amazon, Facebook                     | Medium     | Distributions                            |
| 28  | Cumulative Distribution Function (CDF): Definition and Uses             | [Wikipedia: Cumulative distribution function](https://en.wikipedia.org/wiki/Cumulative_distribution_function)                              | Google, Microsoft                    | Medium     | Distributions                            |
| 29  | Determining Independence of Events                                      | [Khan Academy: Independent Events](https://www.khanacademy.org/math/statistics-probability/probability-library)                              | Google, Amazon                       | Easy       | Fundamental Concepts                     |
| 30  | Entropy in Information Theory                                           | [Wikipedia: Entropy (information theory)](https://en.wikipedia.org/wiki/Entropy_(information_theory))                                        | Google, Facebook                     | Hard       | Information Theory, Probability          |
| 31  | Joint Probability Distributions                                         | [Khan Academy: Joint Probability](https://www.khanacademy.org/math/statistics-probability/probability-library)                               | Microsoft, Amazon                    | Medium     | Multivariate Distributions               |
| 32  | Conditional Expectation                                                 | [Wikipedia: Conditional expectation](https://en.wikipedia.org/wiki/Conditional_expectation)                                                | Google, Facebook                     | Hard       | Advanced Concepts                        |
| 33  | Sampling Methods: With and Without Replacement                         | [Khan Academy: Sampling](https://www.khanacademy.org/math/statistics-probability)                                                          | Amazon, Microsoft                    | Easy       | Sampling, Combinatorics                   |
| 34  | Risk Modeling Using Probability                                         | [Investopedia: Risk Analysis](https://www.investopedia.com/terms/r/risk-analysis.asp)                                                      | Google, Amazon                       | Medium     | Applications, Finance                    |
| 35  | In-Depth: Central Limit Theorem and Its Importance                      | [Khan Academy: Central Limit Theorem](https://www.khanacademy.org/math/statistics-probability/modeling-distributions-of-data)               | Google, Microsoft                    | Medium     | Theoretical Concepts, Distributions      |
| 36  | Variance under Linear Transformations                                   | [Wikipedia: Variance](https://en.wikipedia.org/wiki/Variance)                                                                              | Amazon, Facebook                     | Hard       | Advanced Statistics                      |
| 37  | Quantiles: Definition and Interpretation                                | [Khan Academy: Percentiles](https://www.khanacademy.org/math/statistics-probability)                                                       | Google, Amazon                       | Medium     | Descriptive Statistics                   |
| 38  | Common Probability Puzzles and Brain Teasers                            | [Brilliant.org: Probability Puzzles](https://brilliant.org/wiki/probability/)                                                              | Google, Facebook                     | Medium     | Puzzles, Recreational Mathematics         |
| 39  | Real-World Applications of Probability in Data Science                  | [Towards Data Science](https://towardsdatascience.com/) *(Search for probability applications in DS)*                                        | Google, Amazon, Facebook             | Medium     | Applications, Data Science               |
| 40  | Advanced Topic: Introduction to Stochastic Calculus                      | [Wikipedia: Stochastic calculus](https://en.wikipedia.org/wiki/Stochastic_calculus)                                                       | Microsoft, Amazon                    | Hard       | Advanced Probability, Finance            |

---

## Questions asked in Google interview
- Bayes’ Theorem: Statement and Application  
- Conditional Probability and Independence  
- The Birthday Problem  
- The Monty Hall Problem  
- Normal Distribution and the Central Limit Theorem  
- Law of Large Numbers  

## Questions asked in Facebook interview
- Conditional Probability and Independence  
- Bayes’ Theorem  
- Chi-Squared Test  
- The Monty Hall Problem  
- Entropy in Information Theory  

## Questions asked in Amazon interview
- Basic Probability Concepts  
- Bayes’ Theorem  
- Expected Value and Variance  
- Binomial and Poisson Distributions  
- Permutations and Combinations  
- Real-World Applications of Bayes’ Theorem  

## Questions asked in Microsoft interview
- Bayes’ Theorem  
- Markov Chains  
- Stochastic Processes  
- Central Limit Theorem  
- Variance under Linear Transformations  

---



