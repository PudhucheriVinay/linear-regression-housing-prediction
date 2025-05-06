# 🏠 House Price Prediction using Linear Regression

## 📌 Task Objective
Implement and understand simple and multiple linear regression using Scikit-learn, Pandas, and Matplotlib. This project involves:
- Data preprocessing
- Training a regression model
- Evaluating model performance
- Visualizing results
- Interpreting regression metrics

## Features used in this project:
- `LotArea`
- `OverallQual`
- `YearBuilt`
- Target variable: `SalePrice`

## ⚙️ Tools & Libraries
- Python 3.x
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## 🚀 Project Steps

1. **Import and clean data**
   - Loaded CSV and removed missing values
2. **Feature selection**
   - Selected numerical predictors
3. **Train-test split**
   - 80% training / 20% testing
4. **Model training**
   - Used `LinearRegression()` from `sklearn.linear_model`
5. **Evaluation**
   - Calculated MAE, MSE, R² score
6. **Visualization**
   - Plotted regression line (for simple regression)

## Interview Questions and Answers:
## 1. What assumptions does linear regression make?

   Linearity: Relationship between features and target is linear.

   Independence: Observations are independent.

   Homoscedasticity: Constant variance of errors.

   Normality: Residuals are normally distributed.

   No multicollinearity: Predictors aren’t highly correlated.

## 2. How do you interpret the coefficients?

   Each coefficient represents the change in the target variable for a one-unit change in the predictor, assuming other predictors remain constant.

## 3. What is R² score and its significance?

   R² (coefficient of determination) shows the proportion of variance in the dependent variable predictable from the independent variables (0 to 1). Closer to 1 is better.

## 4. When would you prefer MSE over MAE?

   Prefer MSE when larger errors are more serious (it penalizes them more). Use MAE when all errors should be treated equally.

## 5. How do you detect multicollinearity?

   Use Variance Inflation Factor (VIF) or check correlation matrix. High VIF (>10) indicates multicollinearity.

## 6. What is the difference between simple and multiple regression?

   Simple regression uses one feature; multiple regression uses more than one.

## 7. Can linear regression be used for classification?

   No, it’s not appropriate. Use logistic regression for binary classification instead.

## 8. What happens if you violate regression assumptions?

   It leads to biased estimates, incorrect inferences, and poor predictive performance.
