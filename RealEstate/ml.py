import pandas as pd 
import numpy as np s
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Test df
df = pd.read_csv("data/condos.csv")
# Full df
# df = pd.read_csv("data/condos.csv").iloc[:, 4: ]
# # Copy coordinates in separate table and drop from df
# coordinates = df[["lat", "long"]].copy()
# df.drop(["lat", "long"], axis=1, inplace=True)

# Define features and target
y = df[['price']]
X = df.iloc[:, 2 :]

# Split data 
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=42)

# Train LR model
LR = LinearRegression()
LR.fit(X_train, y_train)

# Predict test values
predicted = LR.predict(X_test)

# R^2
LR.score(X_test, y_test)

# Predicted vs. actual
y_pred = pd.Series(predicted.flatten())
y_actual = y_test.reset_index(drop=True).astype("float")
# Merge into data frame
results = pd.concat([y_actual, y_pred], axis=1, ignore_index=True)
results.columns = ['Actual', 'Predicted']
# Save results
results.to_csv("data/linear_regression_results.csv")

# MSE
mse = mean_squared_error(y_true=y_actual, y_pred=y_pred)
np.sqrt(mse)
