# Data Science & Kaggle Projects

A collection of data analysis, feature engineering experiments, and machine learning models for Kaggle competitions and real-world datasets.

## Projects

### 1. Titanic - Machine Learning from Disaster
- **Folder:** [`01-titanic-survival-prediction/`](01-titanic-survival-prediction/)
- **Type:** Binary Classification
- **Tech Stack:** Python, Pandas, Scikit-Learn, Seaborn
- **Summary:** End-to-end classification pipeline predicting passenger survival. Explores demographic interactions (Sex, Pclass, Age), extracts title-based features to handle missing age values, and evaluates models using Stratified 5-Fold Cross Validation.
- **Results:**
  - Heuristic Baseline (Women survive, Men perish): 78.68%
  - Logistic Regression: 82.94% (±0.78%)
  - Random Forest (150 trees, depth 5): **83.28% (±0.74%)**

---

## Getting Started

Clone the repository and install requirements:

```bash
git clone https://github.com/Ninnjoes/data-science-portfolio.git
cd data-science-portfolio

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To run a specific project:

```bash
cd 01-titanic-survival-prediction
python 01_eda.py
python 03_train_models.py
```
