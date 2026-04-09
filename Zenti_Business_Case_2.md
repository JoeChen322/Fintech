# Module #2: Needs-based Recommendation Systems, Collaborative Filtering, Next Best Actions (NBA)

## The Key to Financial Personalization

> "Good personalization can lift revenues by up to 15 percent and increase marketing ROI by up to 30 percent" - McKinsey & Company (2023)

---

## Why Needs & Goals Matter for Client Targeting

### The Fundamental Insight

After all, we are still monkeys: most of us don't really understand financial and insurance products (so we don't buy…)

**But we all have real needs** – and we understand them:
- **Protect ourselves** → Insurance
- **Buy things** → Payment tools & services
- **Save for future consumption** → Savings & investments
- **Anticipate future consumption or investments** → Borrowing

**Key Principle:** We tend to buy what we need. Thus, needs are a good starting point for recommending financial products and services in a personalized way.

### Legal & Regulatory Reasons

1. **MIFID/IDD Compliance:** Coherence between needs/goals and financial/insurance products sold to clients
   - Retail Investment Strategy (RIS) requires that products/portfolios must be in the customer's "Best Interest"
   - Products must have Value for Money

2. **Data Collection:** Key information about clients must be collected using MIFID/IDD questionnaires
   - A lot of detailed information is collected through MIFID/IDD questionnaires
   - This data can be crunched by algorithms
   - *Basically, you get a broad survey for free – that's the reason why MIFID/IDD questionnaires should be properly prepared*

**Motto:** "Make a virtue out of necessity"

---

## Financial Needs: The Theory

### Life-Cycle Needs Framework

Financial needs evolve over a person's lifetime:

```
Age stages and typical needs:
├── Young Adult
│   ├── Job protection insurance
│   ├── Health insurance
│   └── Mobility insurance
├── Building Family
│   ├── Home insurance
│   ├── Life insurance
│   ├── Investing plan for kids
│   └── Borrowing (buying a home)
├── Mid-Career
│   ├── Pension investing plan
│   ├── Capital investing
│   ├── Tax optimization
│   └── Health insurance for income
└── Mature/Retirement
    ├── Decumulation strategies
    ├── Succession plan
    └── Long-term care insurance
```

### The Reality of Financial & Insurance Needs

**Key Challenges:**
- Not everybody will start a family at 30
- Maybe at 72 not everybody is willing to plan their inheritance process; maybe they're getting ready for a marathon or sailing around the world
- What about if at 50 you have 2 divorces and 2 maintenance allowances?
- Maybe at 35 someone faces a big recession, gets fired, and cannot buy a home
- ...and more unpredictable life events

**Critical Insight:** Financial needs change over time following our random life events, and our random lives are not all equal.

---

## Estimating Financial Needs: The ML Approach

### The Mathematical Framework

Financial needs change overtime based on data:

```
Need(i, t) = f(client situation(t), context(t))
```

This follows a machine learning model:

```
Y = f(x₁, x₂, x₃, …)

Where:
- Y = responses = Need (present or absent)
- x₁, x₂, x₃, … = X = features = client situation, context
```

### Practical Example: Insurance Company with Wealth Management

**Protection dimension:**
- Me (personal protection)
- Home
- Children
- Others
- Pets

**Investments dimension:**
- Generic investment
- Income
- Retirement
- Buy a home
- Mobility (buy car/boat, etc.)
- Education
- Legacy
- Take a vacation

---

## The Supervised ML Problem

### Framing the Problem

- A client might have/not have a given financial need (or goal)
- Each client might have multiple needs
- Needs can be satisfied by financial products
- **Solution:** Teach an ML algorithm to recognize presence/absence of needs
- **Type:** Classification problem (or regression)

### The Real-World Challenge

> Welcome to the real world!

In a classification problem, we teach an algorithm to put labels (Y values). But:

- **Where are the labels?** Who is able to say "Client A has need Z"?
- **Needs are not observable!**
- We have our features (Xs) but we are not sure about the labels (Ys) → **We have a problem...**

---

## Solving the Label Problem

### Two Main Approaches

#### **Case 1: Explicit Labels**
A human being creates the Ys directly

```
Y = 1 if client(i) has need(j)
Y = 0 otherwise
```

**Characteristics:**
- Quite common in image recognition
- If the human-labeler is reliable → very good results
- **Problem:** Financial needs are not easy to spot (not like cats/dogs/pedestrians)
- **Cost:** Requires investment/insurance/banking experts → expensive

#### **Case 2: Implicit Labels**
Ys inferred from expert behavior

**Approach:**
- Learn from those who should know if a client has a given need: financial advisors
- If an advisor sells a financial product that satisfies a given need, then probably it was to satisfy that need
- Thus:
  ```
  Y = 1 if client(i) owns a product that satisfies need(j)
  Y = 0 otherwise
  ```

**Critical Issue:** If the human sells products that maximize their own profits, the algorithm will learn exactly that process
- ⚠️ **AI Ethics Concern:** This can perpetuate biased or self-interested behaviors

**Solutions:**
- Filter experts and their behaviors through "expert-picking"
- Use a priori information (Bayesian models)
- Combine different models (Bayesian Model Averaging)

---

## Model Architecture Choices

### One-vs-All Models (Binary Classification)

```
1 need ← 1 model
Examples:
- 10 needs ← 10 models
- 30 needs ← 30 models
```

**Advantages:**
- Simpler individual models
- Easier to interpret and maintain

### Multiclass Models (True Multiclass)

```
N needs ← 1 model
Examples:
- 10 needs ← 1 model
- 100 needs ← 1 model
```

**Characteristics:**
- Often more complex
- Might be less robust (see Occam's Razor principle)
- Can capture interactions between needs
- More difficult to interpret

---

## From Needs to Recommendation: Next Best Actions (NBA)

### Finding the Best Matching Between Client Profile and Product Profile

The goal is to match client profiles with appropriate product recommendations.

#### **1. Content-Based Filtering**
- Knowledge-based methods that rely mostly on domain-knowledge
- Uses characteristics of products and user preferences

#### **2. Case-Based Recommender Systems**
- Applies case-based reasoning (CBR)
- Solves the recommendation problem based on similar cases
- Leverages historical similar situations

#### **3. Collaborative Filtering**
Other approaches learning from similar situations/users

**Methods:**
- Singular Value Decomposition (SVD)
- Autoencoders
- Build recommender systems using latent variables

**Limitation:** Often doesn't go to the heart of the matter
- Unable to manage complex situations
- Particularly problematic in financial product recommendations
- Requires consideration of many factors, primarily regulatory constraints

---

## Summary

This module covers a comprehensive approach to:

1. **Understanding financial needs** as the foundation for personalization
2. **Leveraging regulatory requirements** (MIFID/IDD) as data sources
3. **Applying machine learning** to identify and predict client needs
4. **Addressing the label problem** through explicit and implicit approaches
5. **Recommending financial products** through various filtering techniques
6. **Balancing complexity and robustness** in model architecture

The key insight: **Financial personalization through needs-based recommendation systems can significantly improve client outcomes and business performance, but requires careful attention to data quality, expert behavior, and regulatory compliance.**

---

## Coding Session

The presentation includes hands-on coding exercises on Next Best Actions and recommendation systems.

---

## Submission Instructions

**Email to:** raffaele.zenti@wealthype.it

**Required:**
- Include the group number in the email subject
- Include a zipped file with:
  - Code
  - PowerPoint presentation
  - Other relevant materials
- Include the group name in the zipped file
