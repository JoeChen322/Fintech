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

### Target Variables

- **IncomeInvestment**: For clients who want income (lump-sum investing, typically older clients)
- **AccumulationInvestment**: For clients who want to accumulate wealth (dollar-cost averaging, typically younger clients)

---

## Current Implementation (Jupyter Notebook)

### Completed Work

1. **Data Exploration**
   - Variable summary and statistics
   - Target variable distribution analysis
   - Wealth transformation analysis (log, power)
   - Correlation analysis and pair plots

2. **Feature Engineering**
   - Log transformation of Wealth and Income
   - Income/Wealth ratio feature
   - MinMaxScaler normalization

3. **Baseline Models**
   - SVM (Support Vector Machine)
   - Gaussian Naive Bayes
   - XGBoost

4. **Advanced Models**
   - Multi-Layer Perceptron (MLP) Neural Network
   - Batch Normalization
   - Learning Rate Scheduler (ReduceLROnPlateau)

5. **Explainable AI (XAI)**
   - XGBoost Feature Importance
   - SHAP Values Analysis

6. **Recommendation System**
   - Client identification (high accumulation propensity)
   - Risk-based product matching
   - Suitability analysis

### Model Performance (Test Set)

| Model   | Target       | Accuracy | Precision | Recall | F1    |
| ------- | ------------ | -------- | --------- | ------ | ----- |
| XGBoost | Income       | 76.3%    | 76.5%     | 55.2%  | 64.1% |
| XGBoost | Accumulation | 76.8%    | 78.3%     | 75.8%  | 77.0% |
| MLP     | Income       | 76.3%    | 78.4%     | 52.9%  | 63.1% |
| MLP     | Accumulation | 78.4%    | 84.9%     | 70.4%  | 77.0% |

---

## Project Tasks (To-Do List)

### Priority 1: Model Enhancement

- [ ] **Implement Voting Classifier** (soft/hard voting)
  - Combine SVM, XGBoost, MLP predictions
  - Use sklearn VotingClassifier

- [ ] **Implement Stacking Classifier**
  - Meta-model learning optimal weight combination
  - Try logistic regression as meta-learner

- [ ] **Try Random Forest**
  - Often performs well on financial data
  - Compare with XGBoost

- [ ] **Try KNN**
  - Simple but effective for certain problems
  - May outperform complex models

- [ ] **Hyperparameter Tuning with AutoML**
  - Optuna
  - Ray Tune
  - Keras Tuner

### Priority 2: Enhanced EDA (From HINTS)

- [ ] **Outlier Detection**
  - Box plots for all numeric variables
  - Identify extreme values in Wealth, Income, Age

- [ ] **Normality Testing**
  - Q-Q plots for continuous variables
  - Assess distribution assumptions

- [ ] **Violin Plots**
  - Compare distributions by Gender groups
  - Compare distributions by FinancialEducation levels
  - Group-wise density curve analysis (peaks, valleys, tails)

- [ ] **Demographic Distribution Analysis**
  - Distribution plots by age groups (young/mid-career/mature)
  - Wealth and Income distributions by demographic segments

- [ ] **Dependence Analysis**
  - Joint plots for target vs explanatory variables
  - Scatter histogram plots
  - Candidate response-covariate relationships

### Priority 3: Advanced Recommender Systems

- [x] **SVD-based Recommender**
  - User-item interaction matrix
  - Latent factor analysis
  - sklearn TruncatedSVD

- [x] **Autoencoder Recommender**
  - Non-linear representation learning
  - PyTorch implementation
  - Compress user-item interactions

### Priority 4: Feature Engineering

- [ ] **Additional Features**
  - Age groups (young/mid-career/mature)
  - Life-cycle indicators
  - Income/Wealth percentile ranks

- [ ] **Domain Knowledge Integration**
  - Financial advisor logic replication
  - Risk profiling improvements

### Priority 5: Validation & Documentation

- [ ] **Model Ranking Rules**
  - Define primary metric (e.g., Recall for identifying high-value prospects)
  - Define sufficiency thresholds
  - Confusion matrix analysis

- [ ] **Local Explanations**
  - Individual SHAP prediction explanations
  - LIME for local interpretability

- [ ] **Documentation**
  - PowerPoint presentation
  - Code documentation
  - Submission package

---

## Business Context

### Life-Cycle Needs Framework

```
Age Stages:
├── Young Adult: Job protection, health insurance, mobility
├── Building Family: Home, life insurance, kids investing
├── Mid-Career: Pension, capital investing, tax optimization
└── Mature/Retirement: Decumulation, succession, long-term care
```

### Regulatory Framework

- **MIFID/IDD Compliance**: Products must match customer needs
- **Best Interest**: Portfolios must be in customer's best interest
- **Value for Money**: Products must demonstrate value

---

## Submission Requirements

- **Email**: raffaele.zenti@wealthype.it
- **Format**: Zipped file containing:
  - Code
  - PowerPoint presentation
  - Relevant materials
- **Subject**: Include group number

---

## Key Insights from Analysis

### Feature Importance (XGBoost)

1. **Income Investment**: RiskPropensity, Age, Income_log, Wealth_log
2. **Accumulation Investment**: Age, Wealth_log, RiskPropensity, Income_log

### SHAP Findings

- Older customers → Higher IncomeInvestment propensity
- Higher RiskPropensity → Higher AccumulationInvestment propensity
- Wealth and Income strongly influence both needs

### Recommendations Coverage

- 74.28% of target clients received valid recommendations
- Most recommended products: IDs 1, 5, 9

---

## Libraries & Dependencies

```
pandas
numpy
scikit-learn
xgboost
torch
shap
matplotlib
seaborn
```

---

## Getting Started

1. Download dataset from Google Drive (`Dataset2_Needs.xls`)
2. Open `EstimatingNeedsPoliMI.ipynb` in Google Colab or Jupyter
3. Run cells sequentially
4. Follow TODO comments for additional tasks

---

## Author Notes

- This project serves as both an academic exercise and a practical business application
- The label problem is solved through revealed preference (advisor behavior)
- Model interpretation is crucial for regulatory compliance (MIFID/IDD)
