# California Housing Price Prediction

A basic machine learning project that predicts median house values in California using Linear Regression.

## Overview

This project uses the `housing.csv` dataset to train a Linear Regression model. It cleans the data, converts categorical columns, trains the model on 80% of the data, and evaluates its predictions on the remaining 20%.

## Workflow

1. **Load Data**: Reads `housing.csv`.
2. **Encode Categories**: One-hot encodes the `ocean_proximity` column using `pd.get_dummies`.
3. **Clean Missing Values**: Drops rows where `total_bedrooms` is missing.
4. **Split Dataset**: Separates features and target (`median_house_value`), splitting into 80% training and 20% testing sets.
5. **Train Model**: Fits a `LinearRegression` model on the training data.
6. **Evaluate**: Evaluates predictions using MAE, RMSE, and R² score.

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

## Output

```text
--- Linear Regression ---
Mean Absolute Error (MAE):      $50,193.05
Root Mean Squared Error (RMSE): $69,047.61
R-squared (R2) Score:           0.6483

```
