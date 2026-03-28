---
name: statistical-analysis
description: >
  Use this skill for any statistical analysis task following the ISL framework:
  EDA, hypothesis testing, OLS regression (simple, multiple), assumption checking
  (LINE), coefficient interpretation, model comparison, or writing up results.
  Triggers: "run regression", "check assumptions", "interpret p-value", "R squared",
  "test normality", "VIF", "residual plots", "confidence intervals", "OLS",
  "statsmodels", "scipy stats", "F-test", "t-test", "Durbin-Watson", "Breusch-Pagan",
  "Shapiro-Wilk", "Cook's distance", "analyse the Advertising dataset", or any
  request to statistically analyse a CSV or dataset.
  For the Advertising Sales topic, follow ISL Ch.3 exactly: seven research questions,
  LINE assumptions, F-test for overall significance, t-tests per coefficient.
---

# Statistical Analysis Skill
## Framework: ISL Ch.3 · Dataset: Advertising Sales (Kaggle)

---

## Full Analysis Workflow

```
Step 1 → Load & validate data
Step 2 → EDA (descriptive stats + plots)
Step 3 → State hypotheses (H₀ / H₁ for each predictor)
Step 4 → Pre-fit assumption check (linearity & outliers)
Step 5 → Fit models (Simple LR → Multiple LR → Interaction)
Step 6 → Post-fit diagnostics (all LINE tests)
Step 7 → Report results (4-element rule)
Step 8 → Write up in paper language
```

---

## Step 1 — Load & Validate

```python
import pandas as pd
import numpy as np

# Kaggle dataset: yasserh/advertising-sales-dataset
df = pd.read_csv('data/raw/advertising.csv')

print(f"Shape: {df.shape}")          # expect (200, 4)
print(df.dtypes)
print(df.isnull().sum())             # expect all 0
print(df.describe().round(2))
```

Expected schema:
| Column | Type | Range |
|---|---|---|
| TV | float64 | 0.7 – 296.4 |
| Radio | float64 | 0.0 – 49.6 |
| Newspaper | float64 | 0.3 – 114.0 |
| Sales | float64 | 1.6 – 27.0 |

---

## Step 2 — EDA with Visualisation

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Pairplot — sales vs each predictor
sns.pairplot(df, x_vars=['TV','Radio','Newspaper'], y_vars=['Sales'],
             kind='reg', diag_kind=None, height=4)
plt.suptitle('Sales vs Advertising Spend', y=1.02)
plt.savefig('outputs/figures/pairplot.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. Correlation heatmap
plt.figure(figsize=(6,5))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True)
plt.title('Correlation Heatmap')
plt.savefig('outputs/figures/heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. Distribution histograms
df.hist(bins=20, figsize=(10,6), edgecolor='black')
plt.suptitle('Variable Distributions')
plt.tight_layout()
plt.savefig('outputs/figures/histograms.png', dpi=150, bbox_inches='tight')
plt.close()

# 4. Boxplots (outlier detection)
df.boxplot(figsize=(8,5))
plt.title('Boxplots — Outlier Check')
plt.savefig('outputs/figures/boxplots.png', dpi=150, bbox_inches='tight')
plt.close()

# 5. Simple regression scatter — ISL Figure 2.1 style
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ['TV','Radio','Newspaper']):
    ax.scatter(df[col], df['Sales'], alpha=0.4, s=15)
    m, b = np.polyfit(df[col], df['Sales'], 1)
    x_line = np.linspace(df[col].min(), df[col].max(), 100)
    ax.plot(x_line, m*x_line + b, color='blue', linewidth=2)
    ax.set_xlabel(f'{col} Budget ($K)'); ax.set_ylabel('Sales (K units)')
    ax.set_title(f'Sales vs {col}')
plt.tight_layout()
plt.savefig('outputs/figures/scatter_simple_lr.png', dpi=150, bbox_inches='tight')
plt.close()
```

**EDA required outputs (checklist):**
- [ ] Shape and null check printed
- [ ] Descriptive statistics table
- [ ] Pairplot with regression lines
- [ ] Correlation heatmap
- [ ] Histograms
- [ ] Boxplots

---

## Step 3 — Hypotheses (ISL 7 Questions)

State ALL before fitting any model:

```
Q1 — Overall relationship
H₀: β_TV = β_Radio = β_Newspaper = 0  (no predictor is related to sales)
H₁: At least one βⱼ ≠ 0
Test: F-statistic, α = 0.05

Q3 — Individual media
H₀: β_TV = 0        H₁: β_TV ≠ 0
H₀: β_Radio = 0     H₁: β_Radio ≠ 0
H₀: β_Newspaper = 0 H₁: β_Newspaper ≠ 0
Test: t-statistic per coefficient

Q7 — Synergy (interaction)
H₀: β_TV×Radio = 0  H₁: β_TV×Radio ≠ 0
```

---

## Step 4 — Pre-fit Checks

```python
# Linearity: scatter Y vs each X (already done in EDA — inspect for curves)
# Outliers: IQR rule
for col in ['TV','Radio','Newspaper','Sales']:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    print(f"{col}: {len(outliers)} outliers")
```

---

## Step 5 — Model Fitting

### 5a. Simple OLS (one predictor at a time)

```python
import statsmodels.api as sm

results_simple = {}
for col in ['TV','Radio','Newspaper']:
    X_s = sm.add_constant(df[[col]])
    m   = sm.OLS(df['Sales'], X_s).fit()
    results_simple[col] = {
        'beta_0': round(m.params['const'], 4),
        'beta_1': round(m.params[col], 4),
        'R2':     round(m.rsquared, 4),
        'p':      round(m.pvalues[col], 4)
    }

print(pd.DataFrame(results_simple).T)
```

### 5b. Multiple OLS (all 3 predictors)

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df[['TV','Radio','Newspaper']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# statsmodels — for inference
X_train_sm = sm.add_constant(X_train)
model_ols   = sm.OLS(y_train, X_train_sm).fit()
print(model_ols.summary())

# scikit-learn — for RMSE on test set
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lr = LinearRegression().fit(X_train, y_train)
y_pred   = lr.predict(X_test)
rmse     = np.sqrt(mean_squared_error(y_test, y_pred))
r2_test  = r2_score(y_test, y_pred)
print(f"Test RMSE = {rmse:.4f}  |  Test R² = {r2_test:.4f}")
```

### 5c. Interaction model (Q7 — synergy)

```python
df['TV_Radio'] = df['TV'] * df['Radio']
X_int    = df[['TV','Radio','Newspaper','TV_Radio']]
X_int_sm = sm.add_constant(X_int)
model_int = sm.OLS(y, X_int_sm).fit()
print(model_int.summary())
print(f"ΔR² from interaction: {model_int.rsquared - model_ols.rsquared:.4f}")
```

---

## Step 6 — Post-fit Diagnostics (LINE)

```python
import scipy.stats as stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os

os.makedirs('outputs/figures', exist_ok=True)

residuals = model_ols.resid
fitted    = model_ols.fittedvalues

# ── L: Linearity ── Residuals vs Fitted
plt.figure(figsize=(6,4))
plt.scatter(fitted, residuals, alpha=0.5, s=20)
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.xlabel('Fitted values'); plt.ylabel('Residuals')
plt.title('Residuals vs Fitted (Linearity Check)')
plt.savefig('outputs/figures/resid_vs_fitted.png', dpi=150, bbox_inches='tight')
plt.close()

# ── N: Normality ── Q-Q Plot
fig, ax = plt.subplots(figsize=(5,5))
stats.probplot(residuals, dist='norm', plot=ax)
ax.set_title('Normal Q-Q Plot')
plt.savefig('outputs/figures/qq_plot.png', dpi=150, bbox_inches='tight')
plt.close()

sw_stat, sw_p = stats.shapiro(residuals)
print(f"Shapiro-Wilk: W = {sw_stat:.4f}, p = {sw_p:.4f}")
# p > .05 → normality holds

# ── E: Equal Variance ── Scale-Location + Breusch-Pagan
plt.figure(figsize=(6,4))
plt.scatter(fitted, np.sqrt(np.abs(residuals)), alpha=0.5, s=20)
plt.xlabel('Fitted'); plt.ylabel('√|Residuals|')
plt.title('Scale-Location (Homoscedasticity Check)')
plt.savefig('outputs/figures/scale_location.png', dpi=150, bbox_inches='tight')
plt.close()

bp_stat, bp_p, _, _ = het_breuschpagan(residuals, model_ols.model.exog)
print(f"Breusch-Pagan: χ² = {bp_stat:.4f}, p = {bp_p:.4f}")
# p > .05 → homoscedasticity holds

# ── I: Independence ── Durbin-Watson
dw = durbin_watson(residuals)
print(f"Durbin-Watson: {dw:.4f}")
# ≈ 2.0 → no autocorrelation

# ── Multicollinearity ── VIF
vif_df = pd.DataFrame({
    'Feature': X_train.columns,
    'VIF': [variance_inflation_factor(X_train.values, i)
            for i in range(X_train.shape[1])]
}).sort_values('VIF', ascending=False)
print(vif_df)
# All < 5 → acceptable

# ── Influential Points ── Cook's Distance
influence = model_ols.get_influence()
cooks_d   = influence.cooks_distance[0]
threshold = 4 / len(y_train)
flagged   = np.where(cooks_d > threshold)[0]
print(f"Cook's D > {threshold:.4f}: observations {flagged}")

plt.figure(figsize=(8,4))
plt.stem(range(len(cooks_d)), cooks_d, markerfmt=',', basefmt=' ')
plt.axhline(threshold, color='red', linestyle='--', label=f'4/n = {threshold:.3f}')
plt.xlabel('Observation index'); plt.ylabel("Cook's distance")
plt.title("Cook's Distance — Influential Points")
plt.legend()
plt.savefig('outputs/figures/cooks_distance.png', dpi=150, bbox_inches='tight')
plt.close()
```

---

## Step 7 — Print Paper-Ready Results Table

```python
# Coefficient table with CIs
coef_table = pd.DataFrame({
    'β̂':      model_ols.params.round(4),
    'SE':     model_ols.bse.round(4),
    't':      model_ols.tvalues.round(3),
    'p-value': model_ols.pvalues.round(4),
    'CI_low':  model_ols.conf_int()[0].round(4),
    'CI_high': model_ols.conf_int()[1].round(4),
})
print(coef_table.to_markdown())

# Model fit summary
print(f"\nR²       = {model_ols.rsquared:.4f}")
print(f"adj-R²   = {model_ols.rsquared_adj:.4f}")
print(f"F-stat   = {model_ols.fvalue:.2f}")
print(f"F p-val  = {model_ols.f_pvalue:.4e}")
print(f"Test RMSE = {rmse:.4f}")
print(f"Test R²   = {r2_test:.4f}")
```

---

## Step 8 — Reporting Language Templates

```
# Significant predictor:
"TV advertising significantly predicted sales
(β̂ = 0.046, SE = 0.001, t(156) = 32.81, p < .001, 95% CI [0.043, 0.049])."

# Non-significant predictor:
"Newspaper advertising was not a significant predictor of sales
(β̂ = −0.001, SE = 0.006, t(156) = −0.18, p = .860)."

# Overall model:
"The overall multiple regression model was significant
(F(3, 156) = 454.3, p < .001) and explained 89.7% of variance in sales
(R² = .897, adj-R² = .896). Test-set RMSE = 1.69 thousand units."

# Assumption: holds
"Residuals were approximately normally distributed (Shapiro-Wilk W = 0.991, p = .142)."

# Assumption: violated
"Breusch-Pagan test indicated mild heteroscedasticity (χ²(3) = 8.12, p = .044);
robust standard errors were therefore applied."
```

---

## Diagnostic Summary Table (for paper)

```python
diag_summary = {
    'Test': ['Shapiro-Wilk (N)', 'Breusch-Pagan (E)', 'Durbin-Watson (I)',
             'Residual plot (L)', 'Max VIF (Multicollinearity)'],
    'Statistic': [f'W={sw_stat:.3f}', f'χ²={bp_stat:.3f}',
                  f'd={dw:.3f}', 'Visual', f'{vif_df.VIF.max():.2f}'],
    'p-value': [f'{sw_p:.3f}', f'{bp_p:.3f}', '—', '—', '—'],
    'Verdict': ['Normality holds' if sw_p>.05 else 'Violation',
                'Homoscedasticity holds' if bp_p>.05 else 'Violation',
                'No autocorrelation' if 1.5<dw<2.5 else 'Violation',
                'Inspect plot', 'OK' if vif_df.VIF.max()<5 else 'High VIF']
}
print(pd.DataFrame(diag_summary).to_markdown(index=False))
```
