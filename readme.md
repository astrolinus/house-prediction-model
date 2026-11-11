# California Housing Price Prediction

A machine learning project that predicts median house values in California, comparing a baseline **Linear Regression** model, **Decision Tree Regressor**, and a **Random Forest Regressor**.

## Overview

This project uses the `housing.csv` dataset to predict California median home values. It cleans missing data, encodes categorical features, creates domain-specific ratio features, trains Linear Regression, Decision Tree, and Random Forest models, and evaluates their performance side by side.

## Workflow

1. **Load Data**: Reads `housing.csv`.
2. **Encode Categories**: One-hot encodes `ocean_proximity` using `pd.get_dummies(..., drop_first=True)`.
3. **Clean Missing Values**: Drops rows with missing values in `total_bedrooms`.
4. **Feature Engineering**: Calculates structural and density ratio features:
   - `rooms_per_houshold`: `total_rooms` / `households`
   - `bedrooms_per_rooms`: `total_bedrooms` / `total_rooms`
   - `people_per_houses`: `population` / `households`
5. **Split Dataset**: Separates features and target (`median_house_value`), splitting into 80% training and 20% testing sets (`random_state=404`).
6. **Train Models**: Fits three models on the training set:
   - `LinearRegression` (baseline)
   - `DecisionTreeRegressor` (tuned with `max_depth=10`, `min_samples_split=22`, `min_samples_leaf=10`, `random_state=42`)
   - `RandomForestRegressor` (ensemble using `random_state=42`, `n_jobs=-1`)
7. **Evaluate**: Computes and compares MAE, RMSE, and R² scores on the test set.

## Requirements

Install the dependencies using:

```bash
pip install -r requirements.txt
```

## How to Run

Run the script with:

```bash
python main.py
```

## Model Comparison & Results

```text
--- Linear Regression ---
Mean Absolute Error (MAE):      $48,680.38
Root Mean Squared Error (RMSE): $66,819.12
R-squared (R2) Score:           0.6600

--- Decision Tree Regressor ---
Mean Absolute Error (MAE):      $39,225.67
Root Mean Squared Error (RMSE): $57,919.06
R-squared (R2) Score:           0.7445

--- Random Forrest Regressor ---
Mean Absolute Error (MAE):      $32,031.82
Root Mean Squared Error (RMSE): $48,732.27
R-squared (R2) Score:           0.8191
```

### Observations
Tree-based models capture non-linear spatial and structural patterns much better than the linear baseline, with the ensemble **Random Forest Regressor** performing best overall:
- **Decision Tree Regressor**: Improved **R²** from **0.6600** to **0.7445**, reducing **RMSE** by **~$8,900** and **MAE** by **~$9,455** compared to Linear Regression.
- **Random Forest Regressor**: Achieved the strongest performance with **R² = 0.8191**, lowering **RMSE** to **$48,732.27** (~$18,087 below Linear Regression and ~$9,187 below Decision Tree) and **MAE** to **$32,031.82** (~$16,649 below Linear Regression).

