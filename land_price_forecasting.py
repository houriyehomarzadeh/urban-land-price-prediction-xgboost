#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Urban Land Price Prediction Using XGBoost

Machine learning pipeline for predicting urban land prices using
socioeconomic, demographic, housing, and land-use indicators.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from scipy.stats import pearsonr
from xgboost import XGBRegressor


# ------------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------------

housing = pd.read_excel("Merged_Final_MillionRial.xlsx")
housing.columns = housing.columns.str.strip()


# ------------------------------------------------------------------
# Define Target Variable and Features
# ------------------------------------------------------------------

target_col = "Average Land Price per Square Meter (MRials/m²)"

numeric_cols = [
    "year",
    "Average Household Income (MRials/month)",
    "Average Rent (MRials/month)",
    "Average apartment price(MRials/m²)",
    "Population Density (%)",
    "Household Size (%)",
    "Immigrant Inflow (%)",
    "Commercial Use (%)",
    "Medical Use (%)",
    "Green Space Use (%)",
    "Educational Use (%)",
    "Concrete Structure (%)",
    "Building Age (years)",
    "Repair Quality (%)",
    "Renovation Quality (%)"
]

cat_cols = ["Neighborhood"]


# ------------------------------------------------------------------
# Split Data into Training and Testing Sets
# ------------------------------------------------------------------

X = housing[numeric_cols + cat_cols]
y = housing[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ------------------------------------------------------------------
# Build Preprocessing Pipeline
# ------------------------------------------------------------------

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, numeric_cols),
    ("cat", cat_pipeline, cat_cols)
])


# ------------------------------------------------------------------
# Build XGBoost Regression Model
# ------------------------------------------------------------------

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("xgb", XGBRegressor(
        n_estimators=500,
        max_depth=10,
        learning_rate=0.05,
        random_state=42,
        objective="reg:squarederror"
    ))
])


# ------------------------------------------------------------------
# Train Model
# ------------------------------------------------------------------

model_pipeline.fit(X_train, y_train)


# ------------------------------------------------------------------
# Generate Predictions
# ------------------------------------------------------------------

y_pred = model_pipeline.predict(X_test)


# ------------------------------------------------------------------
# Model Evaluation
# ------------------------------------------------------------------

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

cv_rmse = -cross_val_score(
    model_pipeline,
    X,
    y,
    cv=5,
    scoring="neg_root_mean_squared_error"
)

corr, pvalue = pearsonr(y_test, y_pred)


# ------------------------------------------------------------------
# Print Results
# ------------------------------------------------------------------

print("\nUrban Land Price Prediction - Model Evaluation")
print("-" * 50)

print(f"RMSE                 : {rmse:.2f}")
print(f"MAE                  : {mae:.2f}")
print(f"R² Score             : {r2:.3f}")
print(f"Cross-Validation RMSE: {cv_rmse.mean():.2f} ± {cv_rmse.std():.2f}")
print(f"Pearson Correlation  : {corr:.3f}")
print(f"P-value              : {pvalue:.4f}")

