# 🧠 Data Science & Machine Learning Portfolio Hub

[![Author](https://img.shields.io/badge/Author-Thanakrit%20Nuansuwan-blue.svg)](https://github.com/Ninnjoes)
[![GitHub](https://img.shields.io/badge/GitHub-Ninnjoes-181717.svg?logo=github)](https://github.com/Ninnjoes)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competitions-20BEFF.svg?logo=kaggle&logoColor=white)](https://www.kaggle.com/)

Welcome to my central **Data Science, Machine Learning, and Analytics Portfolio**. This repository houses my hands-on solutions to competitive machine learning challenges (Kaggle), end-to-end predictive modeling pipelines, and business intelligence analyses.

---

## 👨‍💻 About Me

* **Name:** Thanakrit Nuansuwan
* **GitHub Profile:** [@Ninnjoes](https://github.com/Ninnjoes)
* **Focus Areas:** Exploratory Data Analysis (EDA), Statistical Modeling, Feature Engineering, and Applied Machine Learning.
* **Philosophy:** Prioritizing rigorous baselines, sound statistical validation (avoiding data leakage), and building transparent, interpretable models.

---

## 🛠️ Technical Competencies

| Domain | Tools & Techniques |
| :--- | :--- |
| **Programming & Computation** | Python (Pandas, NumPy, Scipy, Google Colab, Jupyter, VS Code) |
| **Data Analysis & Visualization** | Matplotlib, Seaborn, Pivot Tables, Statistical Distributions, Correlation Analysis |
| **Feature Engineering** | Conditional Imputation, Text Extraction (Regex), Categorical Encoding, Outlier Handling |
| **Machine Learning** | Scikit-Learn, Logistic Regression, Decision Trees, Random Forest, Cross-Validation |
| **Evaluation & Metrics** | Confusion Matrix, ROC-AUC, Precision/Recall, F1-Score, RMSE, Log-Loss |

---

## 📂 Projects & Competitions Directory

| # | Project Name | Domain / Problem Type | Key Techniques | Benchmark / Score | Quick Links |
| :-: | :--- | :--- | :--- | :-: | :---: |
| **01** | [**Titanic: Disaster Survival Prediction**](01-titanic-survival-prediction/) | Binary Classification | Title Extraction, Conditional Imputation, Stratified 5-Fold CV, Random Forest | **83.28% CV** *(Baseline: 78.68%)* | [📂 Folder](01-titanic-survival-prediction/) • [📓 Colab Notebook](01-titanic-survival-prediction/notebooks/titanic_solution_walkthrough.ipynb) |
| **02** | [**House Prices: Advanced Regression**](02-house-prices-regression/) | Regression / Continuous Target | High-dimensional EDA, Log-Transformations, Regularized Regression (Ridge/Lasso), XGBoost | *In Progress* | [📂 Folder](02-house-prices-regression/) |
| **03** | [**Customer Churn & Retention Analytics**](03-customer-churn-analytics/) | Business Analytics & Classification | Cohort Analysis, ROC-AUC optimization, Customer Lifetime Value (LTV) insights | *Planned* | [📂 Folder](03-customer-churn-analytics/) |

---

## 🌟 Featured Project: 01. Titanic Survival Prediction

An end-to-end classification system on historical passenger manifests from the RMS Titanic.

### Key Highlights:
1. **The Heuristic Baseline Rule:** Established that predicting *all women survive and all men perish* achieves a **78.68% baseline accuracy**. All machine learning models were held to this baseline standard.
2. **Domain Feature Engineering:** 
   - Extracted honorific titles (`Mr`, `Mrs`, `Miss`, `Master`).
   - Imputed missing ages conditionally (e.g., young boys with title `Master` have a median age of 3.5 years, preserving the critical "children first" protocol instead of distorting them with the adult 30-year global median).
   - Created interaction metrics: `FamilySize`, `IsAlone`, and `LogFare`.
3. **Model Evaluation (Stratified 5-Fold CV):**
   - **Logistic Regression:** 82.94% (±0.78%)
   - **Random Forest (150 estimators, max depth 5):** **83.28% (±0.74%)**
4. **Top Predictive Factors:** Social status & gender interaction (`Title_Mr` 23.8%, `Sex` 20.6%, `Pclass` 12.3%, `Fare` 9.7%).

👉 [**Read Full Titanic Technical Writeup →**](01-titanic-survival-prediction/README.md)

---

## 🚀 How to Run Locally

```bash
# 1. Clone this portfolio repository
git clone https://github.com/Ninnjoes/data-science-portfolio.git
cd data-science-portfolio

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Run any project pipeline (e.g., Project 01)
cd 01-titanic-survival-prediction
python 01_eda.py
python 03_train_models.py
```

---

*Continuous learning in progress. More projects added regularly!*
