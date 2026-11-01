import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the housing dataset from CSV file
df = pd.read_csv("housing.csv")

# Convert categorical column 'ocean_proximity' into one-hot encoded dummy variables
housing_data = pd.get_dummies(df, columns=["ocean_proximity"])

# Drop rows containing missing/null values in the 'total_bedrooms' column
housing_data = housing_data.dropna(subset=["total_bedrooms"])


# Separate independent features (X) and target variable (y)
X = housing_data.drop(columns=["median_house_value"])
y = housing_data["median_house_value"]

# Split the data into training (80%) and testing (20%) sets
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2)

# Initialize the Linear Regression model
model = LinearRegression()

# Train (fit) the model on the training data
model.fit(X_train , y_train)

# Generate predictions on the test dataset
y_pred = model.predict(X_test)

# Function to calculate and print key evaluation metrics (MAE, RMSE, R2)
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"--- {model_name} ---")
    print(f"Mean Absolute Error (MAE):      ${mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    print(f"R-squared (R2) Score:           {r2:.4f}\n")
    return mae, rmse, r2

# Evaluate model performance on the test set
lr_metric = evaluate_model(y_test , y_pred, "Linear Regression")

# Print the resulting metrics tuple
print(lr_metric)
