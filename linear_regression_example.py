import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('Housing.csv')

# Convert categorical columns to numeric
categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in categorical_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

# One-hot encode furnishingstatus
df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

# Features and target
X = df.drop('price', axis=1)
y = df['price']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Simple Linear Regression using 'area' feature
X_train_area = X_train[['area']]
X_test_area = X_test[['area']]

simple_lr = LinearRegression()
simple_lr.fit(X_train_area, y_train)

y_pred_simple = simple_lr.predict(X_test_area)

# Evaluate simple linear regression
mae_simple = mean_absolute_error(y_test, y_pred_simple)
mse_simple = mean_squared_error(y_test, y_pred_simple)
r2_simple = r2_score(y_test, y_pred_simple)

print("Simple Linear Regression (area):")
print(f"MAE: {mae_simple:.2f}")
print(f"MSE: {mse_simple:.2f}")
print(f"R^2: {r2_simple:.4f}")
print(f"Coefficient: {simple_lr.coef_[0]:.4f}")
print(f"Intercept: {simple_lr.intercept_:.4f}")

# Plot regression line for simple linear regression
plt.scatter(X_test_area, y_test, color='blue', label='Actual')
plt.plot(X_test_area, y_pred_simple, color='red', label='Predicted')
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Simple Linear Regression: Price vs Area')
plt.legend()
plt.show()

# Multiple Linear Regression using all features
multi_lr = LinearRegression()
multi_lr.fit(X_train, y_train)

y_pred_multi = multi_lr.predict(X_test)

# Evaluate multiple linear regression
mae_multi = mean_absolute_error(y_test, y_pred_multi)
mse_multi = mean_squared_error(y_test, y_pred_multi)
r2_multi = r2_score(y_test, y_pred_multi)

print("\nMultiple Linear Regression (all features):")
print(f"MAE: {mae_multi:.2f}")
print(f"MSE: {mse_multi:.2f}")
print(f"R^2: {r2_multi:.4f}")

# Print coefficients with feature names
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': multi_lr.coef_})
print("\nCoefficients:")
print(coefficients)
