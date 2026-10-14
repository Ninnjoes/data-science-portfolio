# Titanic: Machine Learning from Disaster

Personal solution and exploratory analysis for Kaggle's Titanic competition.

## Overview

The goal is to predict which passengers survived the Titanic shipwreck ($0 = \text{died}$, $1 = \text{survived}$) based on passenger characteristics like age, sex, ticket class, fare, and family relationships.

- **Train data:** 891 rows
- **Test data:** 418 rows
- **Evaluation metric:** Accuracy

## Findings & Methodology

### 1. Key Data Patterns
- **Gender & Class:** Female passengers had an overall survival rate of 74.2%, compared to 18.9% for males. 1st class women had a 96.8% survival rate, while 3rd class men had only 13.5%.
- **Children:** Boys with the title `Master` (median age 3.5) had a 57.5% survival rate, showing that age superseded gender during evacuation.

### 2. Feature Engineering
- **Title Extraction:** Extracted titles (`Mr`, `Mrs`, `Miss`, `Master`, `Rare`) from passenger names to infer marital status and age groups.
- **Conditional Imputation:** Imputed missing values in `Age` (177 missing) using the median age of each specific title rather than the global median, preventing young boys from being treated as 30-year-old adults.
- **Family Size:** Derived `FamilySize = SibSp + Parch + 1` and flagged solo travelers (`IsAlone`).
- **Fare Transformation:** Applied `log1p` on `Fare` to reduce right skewness.

### 3. Model Results (Stratified 5-Fold Cross-Validation)

| Model | Setup | 5-Fold CV Accuracy |
| :--- | :--- | :---: |
| **Heuristic Baseline** | Predict all women survive, men die | 78.68% |
| **Logistic Regression** | `StandardScaler` + `L2 Regularization` | 82.94% (±0.78%) |
| **Random Forest** | 150 trees, `max_depth=5`, `min_samples_split=4` | **83.28% (±0.74%)** |

Top features by importance: `Title_Mr` (23.8%), `Sex` (20.6%), `Pclass` (12.3%), `LogFare` (9.7%), and `Age` (6.4%).

## Project Files

- [`01_eda.py`](01_eda.py): Quick EDA script printing summary statistics and crosstabs.
- [`02_feature_engineering.py`](02_feature_engineering.py): Data cleaning, imputation, and feature creation logic.
- [`03_train_models.py`](03_train_models.py): Model training, cross-validation, and Kaggle `submission.csv` export.
- [`notebooks/titanic_solution_walkthrough.ipynb`](notebooks/titanic_solution_walkthrough.ipynb): Interactive notebook runnable locally or via Google Colab.
- [`submission.csv`](submission.csv): Final test set predictions.

## How to Run

```bash
# Run EDA
python 01_eda.py

# Train models and generate submission.csv
python 03_train_models.py
```
