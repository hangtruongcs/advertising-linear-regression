---
name: ml-engineering
description: >
  Use this skill for machine learning model development, comparison, hyperparameter tuning,
  cross-validation, feature importance, overfitting analysis, and model persistence.
  Triggers: "train a model", "compare models", "hyperparameter tuning", "cross-validation",
  "feature importance", "overfitting", "regularisation", "grid search", "model selection",
  "sklearn pipeline", "learning curve", "bias-variance", "save model", "deploy model",
  "ridge vs lasso", "model performance", "beat baseline".
  For Advertising dataset: always start with a DummyRegressor baseline, then compare
  OLS / Ridge / Lasso / Random Forest, use 5-fold CV, and report standardised
  coefficients for interpretability.
---

# ML Engineering Skill
## Project: Advertising Sales Prediction · ISL Ch.3 Baseline → Extensions

---

## Phase Overview

```
Phase 1 → Problem framing & success criteria
Phase 2 → Baseline model (DummyRegressor)
Phase 3 → Model comparison (OLS, Ridge, Lasso, RF)
Phase 4 → Hyperparameter tuning (RidgeCV / LassoCV / GridSearchCV)
Phase 5 → Learning curve analysis (bias vs variance)
Phase 6 → Final evaluation (test set — touch ONCE)
Phase 7 → Feature importance
Phase 8 → Model persistence (pickle)
```

---

## Phase 1 — Problem Framing

| Item | Answer for this project |
|---|---|
| Task type | Regression |
| Target | Sales (thousands of units, continuous) |
| Primary metric | RMSE (same unit as target) |
| Secondary metric | R² (explained variance) |
| Baseline | Predict training mean (DummyRegressor) |
| Success threshold | RMSE < 2.0, R² > 0.85 |
| Data leakage risk | Scale after split; no target-derived features |

---

## Phase 2 — Baseline Model

```python
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

baseline = DummyRegressor(strategy='mean')
baseline.fit(X_train, y_train)
y_base = baseline.predict(X_test)

print("=== Baseline (predict mean) ===")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_base)):.4f}")
print(f"R²:   {r2_score(y_test, y_base):.4f}")     # will be ≈ 0
```

Every model in Phase 3 must beat this baseline.

---

## Phase 3 — Model Comparison (5-fold CV)

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd

# Use pipelines to prevent leakage during CV
models = {
    "OLS Linear":      Pipeline([('sc', StandardScaler()), ('m', LinearRegression())]),
    "Ridge (α=1)":     Pipeline([('sc', StandardScaler()), ('m', Ridge(alpha=1.0))]),
    "Lasso (α=0.1)":   Pipeline([('sc', StandardScaler()), ('m', Lasso(alpha=0.1))]),
    "Elastic Net":     Pipeline([('sc', StandardScaler()),
                                 ('m', ElasticNet(alpha=0.1, l1_ratio=0.5))]),
    "Random Forest":   RandomForestRegressor(n_estimators=100, random_state=42),
}

rows = []
for name, model in models.items():
    cv = cross_val_score(model, X_train, y_train,
                         cv=5, scoring='neg_root_mean_squared_error')
    rows.append({
        "Model":         name,
        "CV RMSE Mean":  round(-cv.mean(), 4),
        "CV RMSE ±":     round(cv.std(), 4),
    })

comparison = pd.DataFrame(rows).sort_values("CV RMSE Mean")
print(comparison.to_markdown(index=False))
```

---

## Phase 4 — Hyperparameter Tuning

### RidgeCV / LassoCV (fast, built-in)

```python
from sklearn.linear_model import RidgeCV, LassoCV

alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train_sc, y_train)
print(f"Ridge best α: {ridge_cv.alpha_}")

lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10_000)
lasso_cv.fit(X_train_sc, y_train)
print(f"Lasso best α: {lasso_cv.alpha_}")
print(f"Lasso non-zero coefs: {(lasso_cv.coef_ != 0).sum()}")
```

### GridSearchCV (Random Forest)

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = {
    'n_estimators':  [50, 100, 200],
    'max_depth':     [None, 5, 10],
    'min_samples_leaf': [1, 2, 5],
}

grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid, cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1, verbose=1
)
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
print(f"Best CV RMSE: {-grid.best_score_:.4f}")
best_rf = grid.best_estimator_
```

---

## Phase 5 — Learning Curve (Bias vs Variance)

```python
from sklearn.model_selection import learning_curve

train_sizes, train_sc, val_sc = learning_curve(
    best_model, X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='neg_root_mean_squared_error',
    n_jobs=-1
)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, -train_sc.mean(axis=1), 'o-', label='Train RMSE')
plt.plot(train_sizes, -val_sc.mean(axis=1),   's-', label='Val RMSE')
plt.fill_between(train_sizes,
    -train_sc.mean(axis=1) - train_sc.std(axis=1),
    -train_sc.mean(axis=1) + train_sc.std(axis=1), alpha=0.1)
plt.fill_between(train_sizes,
    -val_sc.mean(axis=1) - val_sc.std(axis=1),
    -val_sc.mean(axis=1) + val_sc.std(axis=1), alpha=0.1)
plt.xlabel("Training set size"); plt.ylabel("RMSE")
plt.title("Learning Curve — Bias/Variance Diagnosis")
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig("outputs/figures/learning_curve.png", dpi=150, bbox_inches='tight')
plt.close()
```

**Interpretation:**
| Pattern | Diagnosis | Fix |
|---|---|---|
| Both errors high | Underfitting (high bias) | More features, polynomial, less regularisation |
| Train low, val high | Overfitting (high variance) | More data, stronger regularisation |
| Both errors low & converge | Good fit | Done |

---

## Phase 6 — Final Test Set Evaluation

**Touch the test set only ONCE, at the very end.**

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_pred_final = best_model.predict(X_test)  # or X_test_sc for regularised

final = {
    "RMSE": round(float(np.sqrt(mean_squared_error(y_test, y_pred_final))), 4),
    "MAE":  round(float(mean_absolute_error(y_test, y_pred_final)), 4),
    "R²":   round(float(r2_score(y_test, y_pred_final)), 4),
}
print(pd.Series(final))

# Compare models in one table (for Results section Table 3)
model_results = pd.DataFrame([
    {"Model": "Baseline", "RMSE": baseline_rmse, "R²": 0.000},
    {"Model": "OLS (TV only)", "RMSE": rmse_tv, "R²": r2_tv},
    {"Model": "OLS (all 3)", "RMSE": rmse_full, "R²": r2_full},
    {"Model": "OLS + interaction", "RMSE": rmse_int, "R²": r2_int},
])
print(model_results.to_markdown(index=False))
```

---

## Phase 7 — Feature Importance

```python
# Linear model: standardised coefficients (comparable effect sizes)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

sc       = StandardScaler()
X_tr_sc  = sc.fit_transform(X_train)
lr_std   = LinearRegression().fit(X_tr_sc, y_train)

std_coefs = pd.DataFrame({
    "Feature":            X_train.columns,
    "Standardised β̂":   lr_std.coef_.round(4),
}).sort_values("Standardised β̂", key=abs, ascending=False)
print(std_coefs)

# Visualise
plt.figure(figsize=(6,4))
plt.barh(std_coefs["Feature"], std_coefs["Standardised β̂"])
plt.xlabel("Standardised coefficient"); plt.title("Feature Importance (Std β̂)")
plt.savefig("outputs/figures/feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()
```

---

## Phase 8 — Model Persistence

```python
import pickle, json, os

os.makedirs("outputs", exist_ok=True)

# Save pipeline (scaler + model together — prevents leakage on reload)
with open("outputs/model.pkl", "wb") as f:
    pickle.dump(best_pipeline, f)

# Save final metrics
with open("outputs/reports/metrics.json", "w") as f:
    json.dump(final, f, indent=2)

# --- Load and predict on new data ---
with open("outputs/model.pkl", "rb") as f:
    loaded = pickle.load(f)

new_market = pd.DataFrame({"TV": [200], "Radio": [30], "Newspaper": [10]})
pred = loaded.predict(new_market)[0]
print(f"Predicted sales: {pred:.2f} thousand units")
```

---

## ML Reporting Checklist (Results section)

- [ ] Baseline RMSE and R² reported.
- [ ] ≥3 models compared with CV RMSE mean ± std.
- [ ] Best hyperparameters stated (α, n_estimators, etc.).
- [ ] Learning curve figure included with interpretation.
- [ ] Final test set evaluation reported separately from CV.
- [ ] Standardised coefficient table for linear models.
- [ ] Model persistence code documented.
- [ ] All comparison numbers bolded in Table 3.
