import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the housing dataset from CSV file
df = pd.read_csv("housing.csv")

# Convert categorical column 'ocean_proximity' into one-hot encoded dummy variables
housing_data = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

# Drop rows containing missing/null values in the 'total_bedrooms' column
housing_data = housing_data.dropna(subset=["total_bedrooms"])

# Feature engineering: structural and density ratios
housing_data["rooms_per_houshold"] = housing_data["total_rooms"]/housing_data["households"]
housing_data["bedrooms_per_rooms"] = housing_data["total_bedrooms"]/housing_data["total_rooms"]
housing_data["people_per_houses"] = housing_data["population"]/housing_data["households"]


# Separate independent features (X) and target variable (y)
X = housing_data.drop(columns=["median_house_value"])
y = housing_data["median_house_value"]

# Split the data into training (80%) and testing (20%) sets
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2, random_state=404)

# Initialize Linear Regression, tuned Decision Tree, and Random Forest regressors
linear_model = LinearRegression()

treeRegressor_model = DecisionTreeRegressor(
    max_depth=10,
    min_samples_split=22,
    min_samples_leaf=10,
    random_state=42)

randomForestModel = RandomForestRegressor(random_state=42,
                                          n_jobs=-1)

# Train all three models on the training data
linear_model.fit(X_train , y_train)
treeRegressor_model.fit(X_train , y_train)
randomForestModel.fit(X_train , y_train)

# Generate predictions for all three models on the test set
lr_y_pred = linear_model.predict(X_test)
decision_tree_y_pred = treeRegressor_model.predict(X_test)
random_forest_y_pred = randomForestModel.predict(X_test)

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

# Evaluate and compare all three models on the test set
lr_metric = evaluate_model(y_test , lr_y_pred, "Linear Regression")

decs_tree_metric = evaluate_model(y_test , decision_tree_y_pred, "Decision Tree Regressor")

randomForrest_metric = evaluate_model(y_test , random_forest_y_pred, "Random Forrest Regressor")




