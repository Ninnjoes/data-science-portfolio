# 🚢 Titanic Disaster: Survival Prediction & End-to-End ML Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF.svg?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/titanic)

An end-to-end data science portfolio project solving the classic **Kaggle Titanic: Machine Learning from Disaster** competition. This repository demonstrates exploratory data analysis (EDA), domain-driven feature engineering, cross-validation benchmarking, and production submission pipelines.

---

## 📌 Executive Summary

* **Target Variable:** Binary classification predicting passenger survival ($1 = \text{Survived}$, $0 = \text{Perished}$).
* **Dataset Size:** 891 training records and 418 testing records across 12 raw features.
* **Core Challenge:** Avoiding data leakage, handling high-cardinality missing values (Age, Cabin), and engineering discriminative signals from textual passenger metadata.
* **Key Result:** Advanced from a **78.68% heuristic baseline** to **83.28% (±0.74%) cross-validated accuracy** using a tuned Random Forest ensemble.

---

## 🔍 Key Insights from Exploratory Data Analysis (EDA)

1. **Gender Disparity:** Female passengers had a **74.20% survival rate**, compared to just **18.89%** for male passengers, confirming the strict enforcement of the *"Women and children first"* protocol.
2. **Socioeconomic Stratification:** 1st Class passengers had a **62.96% survival rate**, while 3rd Class passengers had only **24.24%**.
3. **Intersection Effect (Class $\times$ Sex):**
   * **1st Class Females:** **96.81%** survived.
   * **3rd Class Males:** Only **13.54%** survived.
4. **Age Dynamics:** Young boys (honorific title `Master`, median age 3.5) had a **57.50% survival rate** despite being male, proving that age was a critical modifier for evacuation priority.

---

## 🛠️ Feature Engineering & Smart Imputation

| Feature | Raw Representation | Engineered Representation | Rationale |
| :--- | :--- | :--- | :--- |
| **Title** | Full string (`Name`) | Categorical (`Mr`, `Mrs`, `Miss`, `Master`, `Rare`) | Captures social status, marital status, and age group. |
| **Age** | Continuous (`177 missing`) | Conditional Median by `Title` | Avoids biasing young boys (`Master`) with the adult global median (30 yrs). |
| **FamilySize** | `SibSp` + `Parch` | Discrete ($\text{SibSp} + \text{Parch} + 1$) | Quantifies travel group size. |
| **IsAlone** | Derived | Binary flag ($\text{FamilySize} == 1$) | Solo travelers faced distinct evacuation constraints. |
| **LogFare** | Continuous (Skewed) | $\log(1 + \text{Fare})$ | Stabilizes variance and normalizes heavy right-skewed ticket fares. |

---

## 📊 Model Benchmarking & Performance

Models were evaluated using **Stratified 5-Fold Cross-Validation** to preserve class balance across all evaluation splits:

| Model | Technique | 5-Fold CV Accuracy | Key Advantage |
| :--- | :--- | :---: | :--- |
| **Heuristic Baseline** | Rule: Predict Female = 1, Male = 0 | 78.68% | Benchmark threshold |
| **Logistic Regression** | `StandardScaler` + `L2 Regularization` | 82.94% (±0.78%) | High interpretability, linear baseline |
| **Random Forest (Winner)** | Ensemble of 150 Trees, `max_depth=5` | **83.28% (±0.74%)** | Captures non-linear feature interactions |

### Top Predictive Features (Random Forest Importance)
1. `Title_Mr` (23.8%)
2. `Sex_Code` (20.6%)
3. `Pclass` (12.3%)
4. `LogFare` (9.7%)
5. `Age` (6.4%)
6. `FamilySize` (6.1%)

---

## 📂 Repository Structure

```text
├── README.md                      # Project documentation and portfolio overview
├── requirements.txt               # Python package dependencies
├── .gitignore                     # Git tracking exclusions
├── notebooks/
│   └── titanic_solution_walkthrough.ipynb  # Interactive walkthrough (Colab-ready)
├── data/
│   ├── train.csv                  # Kaggle training dataset (891 rows)
│   └── test.csv                   # Kaggle test dataset (418 rows)
├── 01_eda.py                      # Exploratory Data Analysis script
├── 02_feature_engineering.py      # Data cleaning and feature extraction logic
├── 03_train_models.py             # Model training, evaluation, and submission generator
└── submission.csv                 # Generated Kaggle test predictions (418 rows)
```

---

## 🚀 How to Run

### Option A: Run Locally
```bash
# 1. Clone repository
git clone https://github.com/<your-username>/titanic-survival-prediction.git
cd titanic-survival-prediction

# 2. Setup virtual environment & install requirements
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Execute end-to-end pipeline
python 01_eda.py
python 03_train_models.py
```

### Option B: Run on Google Colab
Open [`notebooks/titanic_solution_walkthrough.ipynb`](notebooks/titanic_solution_walkthrough.ipynb) directly in [Google Colab](https://colab.research.google.com) to execute the complete pipeline in the cloud with zero setup.
