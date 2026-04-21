# ============================================================
# Student Score Predictor using Linear Regression
# Author: Amandeep Kumar
# Description: Predicts student exam scores based on
#              number of study hours using Linear Regression
# ============================================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------
# 1. Dataset
# -------------------------------------------------------
data = {
    'study_hours': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5,
                    6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5,
                    1.2, 2.3, 3.8, 4.2, 5.7, 6.8, 7.3, 8.2, 9.1, 10.2],
    'score':       [25, 30, 35, 40, 47, 52, 58, 63, 68, 72,
                    75, 78, 82, 85, 88, 90, 92, 94, 96, 98,
                    28, 38, 54, 60, 70, 80, 83, 89, 93, 97]
}

df = pd.DataFrame(data)

print("=" * 55)
print("     STUDENT SCORE PREDICTOR - Amandeep Kumar")
print("=" * 55)

print(f"\nDataset Overview:")
print(f"Total students  : {len(df)}")
print(f"Avg Study Hours : {df['study_hours'].mean():.2f} hrs")
print(f"Avg Score       : {df['score'].mean():.2f} marks")
print(f"Min Score       : {df['score'].min()} | Max Score: {df['score'].max()}")

# -------------------------------------------------------
# 2. Features & Target
# -------------------------------------------------------
X = df[['study_hours']]
y = df['score']

# -------------------------------------------------------
# 3. Train/Test Split (80/20)
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------------
# 4. Train Linear Regression Model
# -------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

print(f"\n--- Model Parameters ---")
print(f"Slope (coefficient) : {model.coef_[0]:.4f}")
print(f"Intercept           : {model.intercept_:.4f}")
print(f"Equation            : Score = {model.coef_[0]:.2f} x Hours + {model.intercept_:.2f}")

# -------------------------------------------------------
# 5. Evaluate Model
# -------------------------------------------------------
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print(f"\n--- Model Evaluation ---")
print(f"Mean Absolute Error (MAE) : {mae:.2f}")
print(f"Root Mean Sq Error (RMSE) : {rmse:.2f}")
print(f"R² Score                  : {r2:.4f}  ({r2*100:.2f}% variance explained)")

# -------------------------------------------------------
# 6. Predictions Table
# -------------------------------------------------------
print(f"\n--- Actual vs Predicted (Test Set) ---")
print(f"{'Study Hours':<15} {'Actual Score':<15} {'Predicted Score':<15}")
print("-" * 45)
results = pd.DataFrame({'Hours': X_test['study_hours'], 'Actual': y_test, 'Predicted': np.round(y_pred, 1)})
for _, row in results.iterrows():
    print(f"{row['Hours']:<15} {row['Actual']:<15} {row['Predicted']:<15}")

# -------------------------------------------------------
# 7. Predict for Custom Input
# -------------------------------------------------------
def predict_score(hours):
    hours = np.array([[hours]])
    predicted = model.predict(hours)[0]
    predicted = max(0, min(100, predicted))  # Clamp between 0-100
    return round(predicted, 1)

print(f"\n--- Predict Score for Custom Study Hours ---")
test_hours = [2.0, 5.0, 7.5, 9.0, 11.0]
for h in test_hours:
    score = predict_score(h)
    print(f"Study Hours: {h} hrs  -->  Predicted Score: {score}/100")

print("\n" + "=" * 55)
