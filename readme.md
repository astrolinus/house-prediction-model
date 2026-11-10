# California Housing Price Prediction

A machine learning project that predicts median house values in California, comparing a baseline **Linear Regression** model with a tuned **Decision Tree Regressor**.

## Overview

This project uses the `housing.csv` dataset to predict California median home values. It cleans missing data, encodes categorical features, creates domain-specific ratio features, trains both Linear Regression and Decision Tree models, and evaluates their performance side by side.

## Workflow

1. **Load Data**: Reads `housing.csv`.
2. **Encode Categories**: One-hot encodes `ocean_proximity` using `pd.get_dummies(..., drop_first=True)`.
3. **Clean Missing Values**: Drops rows with missing values in `total_bedrooms`.
4. **Feature Engineering**: Calculates structural and density ratio features:
   - `rooms_per_houshold`: `total_rooms` / `households`
   - `bedrooms_per_rooms`: `total_bedrooms` / `total_rooms`
   - `people_per_houses`: `population` / `households`
5. **Split Dataset**: Separates features and target (`median_house_value`), splitting into 80% training and 20% testing sets (`random_state=404`).
6. **Train Models**: Fits both:
   - `LinearRegression` (baseline)
   - `DecisionTreeRegressor` (tuned with `max_depth=10`, `min_samples_split=22`, `min_samples_leaf=10`, `random_state=42`)
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

--- Decision Tree Regression ---
Mean Absolute Error (MAE):      $39,225.67
Root Mean Squared Error (RMSE): $57,919.06
R-squared (R2) Score:           0.7445
```

### Observations
The tuned **Decision Tree Regressor** captures non-linear spatial and structural patterns much better than the linear baseline:
- **R² Score** improved from **0.6600** to **0.7445** (~+8.5% variance explained).
- **RMSE** dropped by **~$8,900** (from $66,819.12 down to $57,919.06).
- **MAE** dropped by **~$9,450** (from $48,680.38 down to $39,225.67).

