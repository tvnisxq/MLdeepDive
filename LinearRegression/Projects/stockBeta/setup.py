import pandas as pd
import numpy as np
import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# 1. Load the data
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['PRICE'] = data.target # The target variable(median house value in )
print("Data Loaded Succesfully!")

# Save the data to CSV
df.to_csv('data/california_housing.csv', index=False)
print("Data saved to data/california_housing.csv")


# 2. The analyst's view: What matters?
# We check correlation to the target 'PRICE'
correlation_matrix = df.corr()
print("Top 3 drivers of price:")
# PRICE will be 1st entry, look at the next three entries
print(correlation_matrix['PRICE'].sort_values(ascending=False).head(4))

# 3. Split data: 80% training, 20% testing
X = df.drop('PRICE', axis=1)
y = df['PRICE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data succesfully split into Train & Test sets")

# NOW WE TRAIN TWO MODELS: Linear Regression and Random Forest
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

#-----------MODEL 1: The Baseline(Linear Regression)
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
print("Linear Regression Predictions\n",lr_preds)

#-----------MODEL 2: The Heavy Hitter(Random Forest)
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print("Random Forest Predictions\n", rf_preds)

# EVALUATION
# This is where we act like an advisor. WE use R� to measure
# how well the model explains variance in the data
# R� = 1 ---> Perfect(suspicious), R� ---> 0(Useless)

def evaluate(model_name, actual, predicted):
    mse = mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    print(f"---{model_name} Performance ---")
    print(f"MSE (Error): {mse:.4f}")
    print(f"R� (Accuracy): {r2:.4f}") # The higher, the better

evaluate("Linear Regression", y_test, lr_preds)
evaluate("Random Forest", y_test, rf_preds)


# VISUALIZATION
import matplotlib.pyplot as plt

# Plotting Linear Regression results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, lr_preds, alpha=0.5, color='red', label="Linear Regression")
plt.plot([0, 5], [0, 5],'--k', lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.legend()
plt.grid(True)
os.makedirs('visualizations', exist_ok=True)
plt.savefig('visualizations/lr_visuals.png')
plt.show()

# Plotting Random Forest results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, rf_preds, alpha=0.5, color='green', label="Random Forest")
plt.plot([0,5], [0,5], '--k', lw=2) # Diagonal line(perfect prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.legend()
plt.grid(True)
plt.savefig('visualizations/rf_visuals.png')
plt.show()
