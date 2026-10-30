
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# 1. Load and clean data
df = pd.read_csv("Advertising.csv")
df.columns = df.columns.str.strip()

df = df.drop(columns=["Unnamed: 0"], errors="ignore")
df = df.drop_duplicates()

features = ["TV", "Radio", "Newspaper"]
target = "Sales"

df = df.dropna(subset=[target])
X = df[features]
y = df[target]

# 2. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 3. Define models
models = {
    "Linear Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LinearRegression())
    ]),
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            min_samples_leaf=2
        ))
    ])
}

# 4. Train and evaluate
results = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    cv_scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring="r2"
    )

    results.append({
        "Model": name,
        "R2": round(r2, 4),
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "CV Mean R2": round(cv_scores.mean(), 4)
    })

    trained_models[name] = model

results_df = pd.DataFrame(results)
print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

# 5. Select the model using test MAE for this demonstration
best_result = min(results, key=lambda r: r["MAE"])
best_name = best_result["Model"]
best_model = trained_models[best_name]

os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(features, "models/features.pkl")

results_df.to_csv("model_results.csv", index=False)

print(f"\nSelected model: {best_name}")
print("Model saved to models/best_model.pkl")
best_model = trained_models[best_name]

os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(features, "models/features.pkl")

results_df.to_csv("model_results.csv", index=False)

# 6. Advertising impact analysis
lr = trained_models["Linear Regression"]
coefficients = lr.named_steps["model"].coef_

print("\nLINEAR REGRESSION COEFFICIENTS")
for feature, coefficient in zip(features, coefficients):
    print(f"{feature}: {coefficient:.4f}")

print(f"\nSelected model: {best_name}")
print("Model saved to models/best_model.pkl")
