# Portfolio Replication & Investment Cloning

## Module #3 - Investment Replica

**Author:** Raffaele Zenti (raffaele.zenti@wealthype.it)  
**Title:** Co-Founder, Chief AI Officer, Wealthype-AI SpA

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Motivations & Use Cases](#motivations--use-cases)
3. [Statistical & Mathematical Framework](#statistical--mathematical-framework)
4. [Linear Clones & Implementation](#linear-clones--implementation)
5. [Leveraging with Futures](#leveraging-with-futures)
6. [Data Overview](#data-overview)
7. [Practical Implementation Considerations](#practical-implementation-considerations)
8. [References](#references)
9. [Supplementary Notes](#supplementary-notes)

---

## Problem Statement

### Portfolio Replication in Short

**Challenge:** We want to track a given portfolio (the "target portfolio") **without knowing its holdings**.

**Key Constraints:**

- We do not know which securities the portfolio contains
- We do not know the weight of each security
- We can only observe reported returns
- The target portfolio is a complete **"black box"** to us

**Objective:** "Crack this black box" and understand its internal structure through statistical inference and signal processing.

---

## Motivations & Use Cases

### Motivation 1: Alternative Investment Clones

**Scenario:** You want exposure to alternative investments but without the associated drawbacks.

**Benefits:**

- Access similar financial risk and return profiles
- **Avoid high fees** typical of alternative investments (e.g., hedge funds)
- Improve liquidity (alternative funds often have liquidity restrictions)
- Hedge downturns in existing alternative fund positions
- Understand the risk factors driving alternative fund performance

---

### Motivation 2: Index Tracking

**Problem:** Direct index tracking faces practical challenges.

**Challenges Solved:**

- **Illiquid indices:** Very costly or impossible to replicate physically
- **Large indices:** Tracking requires billions in AuM
  - Many funds have < €100M or $100M
  - Physical replication requires billions
- **Solution:** Use a sparse portfolio of key liquid factors instead of all holdings

---

### Motivation 3: UCITS Clones

**Use Case:** Transform institutional alternative products into retail-friendly instruments.

**Scenario:**

- Company has an alternative fund (institutional only) with non-UCITS instruments
  - Private debt
  - Other alternative funds
  - Assets not MIFID-eligible

**Goal:** Create a retail UCITS vehicle (mutual fund) with similar financial characteristics but composed of liquid instruments (Futures, ETFs, etc.)

**Value Proposition:** Democratize access to alternative investment characteristics while maintaining regulatory compliance.

---

### Motivation 4: Risk Management

**Alternative to standard factor models:** Discover key liquid financial factors behind any investment, even if underlying assets are not disclosed.

**Use Cases:**

- Due diligence investigations
- Litigation support
- Scenario simulation (use a few factors instead of thousands of securities)

---

## Statistical & Mathematical Framework

### Statistical Perspective

**Problem Formulation:**
From **observed information** (actual performances), we want to extract information about:

- **Latent variables:** Underlying factors not directly observable
- **Functions and structural parameters:** Relationships between factors and returns

**Mathematical Approach:**

- Treat portfolio replication as a **signal processing problem**
- Use regression models on historical data
- Subject to portfolio constraints

---

## Linear Clones & Implementation

### Linear Model Foundation

**Key Assumption:** Portfolio return is a **linear combination** of asset returns.

**Mathematical Form:**

```
R_portfolio = β₁ × R₁ + β₂ × R₂ + ... + βₙ × Rₙ + ε
```

where:

- R_portfolio = target portfolio return
- Rᵢ = returns of underlying factors/futures
- βᵢ = portfolio weights (investment positions)
- ε = tracking error

### Construction Process

A linear clone is constructed by:

1. **Estimation Phase:** Estimate a regression model using time series of:
   - Index returns (or target portfolio returns)
   - Factor returns (underlying assets or futures)

2. **Implementation Phase:** Use resulting coefficient estimates as portfolio weights
   - These become actual investment positions in the underlying factors

3. **Optimization:** Apply portfolio constraints (e.g., LASSO regularization)

### Sparse Portfolio Tracking

**Concept:** Use a **small number of assets** to (approximately) replicate a portfolio or index.

**Benefits:**

- Reduced complexity
- Lower transaction costs
- Easier rebalancing
- Improved interpretability

**Regularization Techniques:**

- **LASSO (Least Absolute Shrinkage and Selection Operator):**
  - Shrinks small weights toward zero
  - Eliminates unimportant factors
  - Creates sparse solutions

- **Constraints & Post-Processing:**
  - Eliminate very small trades
  - Normalize weights (preserve proportions)
  - Use threshold-based rebalancing

---

## Leveraging with Futures

### Mechanics of Using Futures

**Illustrative Scenario:**

```
Capital:                    €100 million
Futures exposure:           €100 million (S&P 500)
Futures margin requirement: 3-12% of notional value

Capital allocation:
├─ Short-term collateral bonds: €90 million
├─ Futures margin (cash):       €10 million
└─ Total financial exposure:    €200 million (200% leverage)
```

**How It Works:**

1. Buy Futures contracts for desired exposure (e.g., €100M S&P 500)
2. Only pay margin requirement (3-12% of contract value)
3. Invest remaining capital in short-term collateral bonds
4. Maintain adequate margins for margin calls

### Regulatory Leverage Limits

**UCITS/MIFID Regulations allow:**

| Constraint                             | Limit               |
| -------------------------------------- | ------------------- |
| **Max Leverage (Gross Exposure)**      | 200%                |
| **Max Value-at-Risk (99% confidence)** | 20% of AuM (1M VaR) |

These constraints ensure prudent risk management while allowing enhanced returns.

---

## Data Overview

### Dataset Description

**Source:** Bloomberg Terminal  
**Period:** October 2007 – April 2021 (13+ years)  
**Frequency:** Weekly data  
**Coverage:** Financial crisis through COVID-19

### Data Components

#### 1. Global Hedge Fund Index

- **Index:** HFRX Global Hedge Fund Index
- **Purpose:** Target portfolio for replication

#### 2. Global Indices

- **BB Global Bond Aggregate:** Bond market exposure
- **MSCI World AC:** Global equity exposure (all countries)
- **MSCI World:** Global developed markets equity

#### 3. Key Futures Contracts

- Essential components for factor replication
- Liquid, low-cost instruments
- Enable leverage within regulatory limits

### Performance Overview (Rebased to Oct-07 = 1.0)

**Indices Timeline:** 2008–2020

- Recovered from 2008 financial crisis lows
- Showed resilience through various market cycles
- MSCI World exhibited higher volatility than bond indices

**Futures Contracts:** Higher volatility and leverage exposure

- Amplified gains during bull markets
- Increased losses during downturns
- Essential for tracking alternative fund characteristics

---

## Practical Implementation Considerations

### Overview

**Scope:** This module focuses on the **algorithm** (Machine Learning approach)

**Why:** This is a Machine Learning lab for Fintech—algorithm optimization is the primary focus

**Extended Considerations:** Going deeper into these areas improves real-world performance:

---

### Trading Costs

#### Cost Structure

- **Futures advantage:** Most liquid, least expensive instruments to trade
- **Assumed AuM:** >€50 million (decent-sized mutual fund)
- **Trading method:** Automated program trading
- **Direct trading costs:** 2–4 basis points (bps)
  - **Conversion:** 0.02% – 0.04% of notional value

#### Cost Implications

- Minimal impact on large positions
- Cumulative effect on frequent rebalancing
- Amortized across tracking period

---

### Rebalancing Strategy

**Frequency Options:**

- Weekly rebalancing
- Monthly rebalancing
- Quarterly rebalancing
- Event-based rebalancing

**Methods for Minimizing Trading:**

1. **Regularization (LASSO):**
   - Already built into ML model
   - Shrinks small weights toward zero
   - Reduces need for small trades

2. **Post-Processing:**
   - Eliminate trades below minimum threshold
   - Normalize weights to preserve proportions
   - Skip rebalancing if portfolio weight changes < threshold

3. **Turnover Penalty:**
   - Include in optimization function
   - Penalize large changes between rebalancing periods
   - Balances tracking accuracy vs. trading costs

**Formula Concept:**

```
Objective = TrackingError + λ × Turnover Penalty
```

---

### Rollover Effects

**Challenge:** Futures contracts expire—exposure must be maintained.

#### The Rollover Process

When a contract nears expiration:

1. **Close** the expiring contract (priced near spot)
2. **Open** a new contract in a further-out month
3. **Manage** timing and volume to minimize slippage

#### Contango vs. Backwardation

| Term              | Definition                 | Impact                                 |
| ----------------- | -------------------------- | -------------------------------------- |
| **Contango**      | Forward price > Spot price | Negative rollover (buy high, sell low) |
| **Backwardation** | Forward price < Spot price | Positive rollover (buy low, sell high) |

**Impact on Returns:**

- Over long periods, rollover effects accumulate
- Can create tracking error if not managed
- Varies by commodity and market conditions

**Example:**

- Gold in strong backwardation: Positive rollover yield
- Oil in contango: Negative rollover cost

---

## References

### Academic Papers

1. **Wu, L., Yang, Y., & Liu, H. (2014)**  
   "Nonnegative-lasso and application in index tracking"  
   _Computational Statistics & Data Analysis_, 70.  
   https://doi.org/10.1016/j.csda.2013.08.012

2. **Tibshirani, R. (1996)**  
   "Regression shrinkage and selection via the lasso"  
   _Journal of the Royal Statistical Society B_, 73(3), 273–282.  
   https://tibshirani.su.domains/ftp/lasso-retro.pdf

3. **Akansu, A.N., Kulkarni, S.R., & Malioutov, D.M. (2016)**  
   "Financial Signal Processing and Machine Learning"  
   _Wiley-IEEE Press_

4. **Roncalli, T. & Weisang, G. (2009)**  
   "Tracking Problems, Hedge Fund Replication and Alternative Beta"  
   _SSRN Electronic Journal_.  
   https://ssrn.com/abstract=1325190  
   https://dx.doi.org/10.2139/ssrn.1325190

---

## Supplementary Notes

### Key Insights & Implementation Notes

#### 1. The Black Box Problem

- **Root cause:** Information asymmetry between fund managers and investors
- **Economic driver:** Funds profit from confidentiality; investors require transparency
- **Solution:** Statistical inference can reveal factor exposure without disclosure
- **Validation:** Compare synthetic portfolio performance against actual fund returns

#### 2. Machine Learning's Role in Portfolio Replication

- **Beyond statistics:** ML techniques add:
  - Non-linear relationships detection (if needed)
  - Regularization for sparse solutions
  - Automated hyperparameter optimization
  - Robustness to regime changes

#### 3. Risk-Return Trade-off in Replication

- **Perfect replication:** Costs money (trading + complexity)
- **Approximate replication:** Lower costs but higher tracking error
- **Optimization:** Find the sweet spot where marginal cost equals marginal benefit

#### 4. Regulatory Considerations

- **UCITS framework:** Constrains leverage (max 200% gross, max 20% VaR)
- **MIFID II:** Requires transparent fund structures; alternative beta products solve this
- **Systemic risk:** Leverage limits prevent cascading failures across funds

#### 5. Data Quality & Governance

- **Weekly frequency:** Good for capturing trends; may miss intraday volatility
- **13+ year period:** Covers multiple market regimes (crisis, recovery, COVID)
- **Standardization:** Bloomberg data ensures consistency across indices/futures

#### 6. Practical Challenges Not Covered in Detail

- **Liquidity:** Futures liquidity varies by contract; front-month contracts are most liquid
- **Slippage:** Actual execution prices differ from theoretical prices
- **Basis risk:** Futures don't perfectly replicate spot markets
- **Model decay:** Performance deteriorates over time; periodic reoptimization needed

#### 7. Extension Ideas for Implementation

- **Cross-validation:** Use rolling window approach to validate out-of-sample performance
- **Stress testing:** Evaluate performance during extreme market conditions (2008, 2020)
- **Multi-period optimization:** Consider path dependency of costs over rebalancing cycles
- **Dynamic factors:** Allow factor weights to evolve over time (non-stationary models)
- **Risk attribution:** Decompose tracking error by source (model risk, execution, rollover)

#### 8. Competitive Advantages of This Approach

- **Cost efficiency:** Futures + sparse selection vs. owning hundreds of securities
- **Transparency:** Factor-based representation makes risk clear
- **Scalability:** Algorithm works for funds of any size (€10M to €10B+)
- **Flexibility:** Adapts to different asset classes (equity, fixed income, commodities, crypto)
- **Compliance:** UCITS-compliant clones unlock retail distribution channels
