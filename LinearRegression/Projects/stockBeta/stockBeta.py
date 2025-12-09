import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. GET DATA
# We need a timeframe that captures different market conditions.
# 1 year is standard
assets = ['TSLA', 'SPY']
data = yf.download(assets, start='2024-01-01', end='2025-01-01')


# 2. PREPARE DATA
# Prices don't matter in Regression; returns do.
returns = data.pct_change().dropna()

# Define X(market) and y(stock) 
# sklearn requires 2D arrays for X, hence the double brackets[['Close', 'SPY']]
X = returns['Close']['SPY'].values.reshape(-1, 1)
y = returns['Close']['TSLA'].values

# 3. Run Linear Regression
model = LinearRegression()
model.fit(X,y)

beta = model.coef_[0]
alpha = model.intercept_
r_squared = model.score(X, y)


# 4. PLOT RESULTS
# Create scatter plot with regression line
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5, label='Actual Returns')

y_pred = model.predict(X)

# Generate regression line
plt.plot(X, y_pred, color='red', linewidth=2, label="Regression Line(The Beta)")
plt.xlabel('SPY Returns', fontsize=12)
plt.ylabel('TSLA Returns', fontsize=12)
plt.title('TSLA vs SPY Returns - Linear Regression', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig('stockBeta_plot.png', dpi=100)
plt.show()

# OUTPUT:
print(f"Stock: TESLA V/s Benchmark: SPY")
print(f"-------------------------------")
print(f"Beta: {beta:.4f}")
print(f"Alpha: {alpha:.4f}")
print(f"R²: {r_squared:.4f}")
print(f"\nPlot saved as 'stockBeta_plot.png'")
