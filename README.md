# Credit Scoring Model

Predict an individual's creditworthiness from past financial data using classification algorithms, then turn the predicted default probability into a 300–850 credit score.

Built on the Kaggle [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) dataset (150,000 borrowers). The notebook can also generate a synthetic dataset if you just want to try it out.

## Overview

| | |
|---|---|
| **Objective** | Predict whether a borrower will become seriously delinquent (90+ days past due) within 2 years |
| **Target** | `default` (1 = serious delinquency, 0 = good) |
| **Models** | Logistic Regression, Decision Tree, Random Forest |
| **Metrics** | Precision, Recall, F1-Score, ROC-AUC |
| **Class balance** | Highly imbalanced (about 6.7% defaults), handled with `class_weight="balanced"` |

## What's inside

1. **Data loading and cleaning**: renames columns, fixes impossible ages, converts sentinel codes (96/98 late-payment counts) to missing values, and clips extreme outliers.
2. **Exploratory data analysis**: class balance, missing values, and the features most related to default.
3. **Feature engineering** from financial history:
   - `total_late` and `weighted_delinquency` (30–59, 60–89 and 90+ days late, weighted by severity)
   - `has_severe_delinquency`, `high_utilization`
   - `income_missing` (missing income is itself informative)
   - `log_income`, `income_per_dependent`, `est_debt_payment`, `credit_lines_per_year`
4. **Modelling**: scikit-learn pipelines (median imputation, scaling for Logistic Regression) so no test data leaks into preprocessing, with stratified 5-fold cross-validation.
5. **Evaluation**: classification reports, confusion matrices, ROC curves and precision-recall curves.
6. **Interpretability**: Random Forest feature importance, Logistic Regression coefficients, and a decision tree plot.
7. **Threshold tuning**: picks the decision cutoff that minimises business cost (a missed default costs more than a rejected good customer).
8. **Credit score**: maps default probability to a 300–850 scale and groups applicants into Poor / Fair / Good / Very Good / Excellent bands.

## Getting started

### 1. Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

### 2. Get the data

Download `cs-training.csv` from the [Give Me Some Credit competition page](https://www.kaggle.com/c/GiveMeSomeCredit/data) (a free Kaggle account is required) and place it next to the notebook.

### 3. Run the notebook

```bash
jupyter notebook credit_scoring.ipynb
```

## Results

Run the notebook to fill in your own numbers. Test-set results are printed in Section 5.

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | | | | |
| Decision Tree | | | | |
| Random Forest | | | | |

**A note on precision:** with only about 6.7% defaults, precision at the default 0.5 cutoff is low. ROC-AUC is the better measure of ranking quality here, and Section 8 shows how to choose a cutoff that fits real lending costs.

## Project structure

```
.
├── credit_scoring.ipynb   # full analysis: data, features, models, evaluation, scoring
├── cs-training.csv        # dataset (download separately, not included)
└── README.md
```

## Possible improvements

- Hyperparameter search with `GridSearchCV` or `RandomizedSearchCV`
- Gradient boosting (XGBoost, LightGBM)
- SMOTE or other resampling for the class imbalance
- Probability calibration
- SHAP explanations for individual decisions
- Fairness checks across age and other groups

## Data source

Give Me Some Credit, Kaggle competition dataset. Check the competition page for its terms of use before redistributing the data.
