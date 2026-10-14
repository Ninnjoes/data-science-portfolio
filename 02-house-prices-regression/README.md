# 🏠 Project 02: House Prices - Advanced Regression Techniques

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-House%20Prices-20BEFF.svg?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow.svg)]()

Predict sales prices and practice advanced regression techniques, feature engineering with 79 explanatory variables, and ensembling.

---

## 📌 Problem Overview
* **Type:** Supervised Learning (Continuous Target / Regression)
* **Target:** `SalePrice` (\$)
* **Evaluation Metric:** Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price.
* **Key Challenges:**
  - High dimensional feature space (79 features describing almost every aspect of residential homes in Ames, Iowa).
  - Skewed target distribution requiring log-transformations.
  - Multicollinearity and complex missing data patterns across physical house characteristics.

---

## 🛠️ Planned Methodology
1. **Exploratory Data Analysis (EDA):** Identify correlation clusters (e.g. `OverallQual`, `GrLivArea`, `GarageCars`), detect bivariate outliers.
2. **Feature Engineering:** Total square footage synthesis, age-at-sale calculations, neighborhood price tier binning.
3. **Modeling:**
   - Regularized Linear Models (Ridge, Lasso, ElasticNet)
   - Gradient Boosting (LightGBM, XGBoost, CatBoost)
   - Stacking / Blending Ensembles

*(Work in progress - Stay tuned for updates!)*
