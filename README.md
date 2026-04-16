# Estimating Investment Needs - Recommendation System

## Project Overview

This is a **needs-based recommendation system** project for a wealth management company, developed as part of a PoliMI (Politecnico di Milano) course on Fintech.

### Objective

Build a machine learning system that:

1. **Estimates investment needs** for customers (binary classification)
2. **Recommends suitable financial products** based on customer profiles and risk tolerance

### Why This Matters

- **Personalization**: "Good personalization can lift revenues by up to 15 percent and increase marketing ROI by up to 30 percent" - McKinsey & Company
- **Regulatory Compliance**: MIFID/IDD requires coherence between customer needs and financial products sold
- **Next Best Action (NBA)**: Identifies optimal product recommendations for each client

---

## Dataset Structure

### Needs Dataset (Customer Data)

| Feature                | Description                                              |
| ---------------------- | -------------------------------------------------------- |
| ID                     | Customer ID                                              |
| Age                    | Age in years (mean: 55.25)                               |
| Gender                 | Female=1, Male=0                                         |
| FamilyMembers          | Number of family members (mean: 2.51)                    |
| FinancialEducation     | Normalized financial education level                     |
| RiskPropensity         | Normalized risk propensity from MIFID profile            |
| Income                 | Income in thousands of euros                             |
| Wealth                 | Sum of investments and cash accounts                     |
| IncomeInvestment       | **Target**: 1=High propensity for income investing       |
| AccumulationInvestment | **Target**: 1=High propensity for accumulation investing |

### Products Dataset

| Feature   | Description                 |
| --------- | --------------------------- |
| IDProduct | Product identifier          |
| Type      | 1=Accumulation, 0=Income    |
| Risk      | Normalized risk score [0,1] |

### Product Information

| IDProduct | Product description                                         |
| --------- | ----------------------------------------------------------- |
| 1         | Balanced Mutual Fund                                        |
| 2         | Income Conservative Unit-Linked (Life Insurance)            |
| 3         | Fixed Income Mutual Fund                                    |
| 4         | Balanced High Dividend Mutual Fund                          |
| 5         | Balanced Mutual Fund                                        |
| 6         | Defensive Flexible Allocation Unit-Linked (Life Insurance)  |
| 7         | Aggressive Flexible Allocation Unit-Linked (Life Insurance) |
| 8         | Balanced Flexible Allocation Unit-Linkled (Life Insurance)  |
| 9         | Cautious Allocation Segregated Account                      |
| 10        | Fixed Income Segregated Account                             |
| 11        | Total Return Aggressive Allocation Segregated Account       |

### Target Variables

- **IncomeInvestment**: For clients who want income (lump-sum investing, typically older clients)
- **AccumulationInvestment**: For clients who want to accumulate wealth (dollar-cost averaging, typically younger clients)

---

## Project Tasks (To-Do List)

### Enhanced EDA

- [x] **Outlier Detection**
  - Box plots for all numeric variables
  - Identify extreme values in Wealth, Income, Age

- [x] **Normality Testing**
  - Q-Q plots for continuous variables
  - Assess distribution assumptions

- [x] **Violin Plots**
  - Compare distributions by Gender groups
  - Compare distributions by FinancialEducation levels
  - Group-wise density curve analysis (peaks, valleys, tails)

- [x] **Demographic Distribution Analysis**
  - Distribution plots by age groups (young/mid-career/mature)
  - Wealth and Income distributions by demographic segments

- [x] **Dependence Analysis**
  - Joint plots for target vs explanatory variables
  - Scatter histogram plots
  - Candidate response-covariate relationships

### Feature Engineering

- [x] **Additional Features**
  - Age groups (young/mid-career/mature)
  - Life-cycle indicators
  - Income/Wealth percentile ranks

- [x] **Domain Knowledge Integration**
  - Financial advisor logic replication
  - Risk profiling improvements

### Model Enhancement

- [x] **Implement Voting Classifier** (soft/hard voting)
  - Combine SVM, XGBoost, MLP predictions
  - Use sklearn VotingClassifier

- [x] **Implement Stacking Classifier**
  - Meta-model learning optimal weight combination
  - Try logistic regression as meta-learner

- [x] **Try Random Forest**
  - Often performs well on financial data
  - Compare with XGBoost

- [x] **Try KNN**
  - Simple but effective for certain problems
  - May outperform complex models

- [ ] **Hyperparameter Tuning with AutoML**
  - We gonna use Optuna

### Advanced Recommender Systems

- [x] **SVD-based Recommender**
  - User-item interaction matrix
  - Latent factor analysis
  - sklearn TruncatedSVD

- [x] **Autoencoder Recommender**
  - Non-linear representation learning
  - PyTorch implementation
  - Compress user-item interactions

### Validation & Documentation

- [ ] **Model Ranking Rules**
  - Define primary metric
  - Consider the [MiFID/IDD](https://eur-lex.europa.eu/eli/dir/2014/65/oj/eng)
  - Under the regulatory framework of MiFID and IDD, "consumer protection" and the "suitability principle (selling the right product to the right person)" are absolute red lines
  -

- [ ] **Local Explanations**
  - Individual SHAP prediction explanations
  - LIME for local interpretability

- [ ] **Documentation**
  - PowerPoint presentation
  - Code documentation
  - Submission package

---

## File Organization

- `EDA.ipynb`
- `classification-ml-models.ipynb`: machine learning models of classification
- `classification-dl-models.ipynb`: deep learning models of classification
- `recommend-system.ipynb`

## Submission Requirements

- **Email**: raffaele.zenti@wealthype.it
- **Format**: Zipped file containing:
  - Code
  - PowerPoint presentation
  - Relevant materials
- **Subject**: Include group number

---
