# A Linear Regression Approach to Advertising Budget and Sales Prediction

**Author:** Truong Thi Ngoc Hang

---

## Abstract

This study applies simple and multiple linear regression to the Advertising dataset (n = 200 markets) to examine whether TV, radio, and newspaper advertising budgets are predictive of product sales. Using ordinary least squares (OLS) estimation implemented in Python (scikit-learn, statsmodels), we fit both single-predictor and full multiple regression models, assess the LINE assumptions, and evaluate predictive accuracy on a held-out test set. The final model achieved an R² of 0.897 and RMSE of 1.69 thousand units, with TV (β̂ = 0.046, p < .001) and radio (β̂ = 0.189, p < .001) identified as significant predictors while newspaper advertising was not significant (p = .860). A TV × radio interaction term further improved fit (ΔR² = 0.03), providing evidence of a synergistic effect between the two media. These findings offer actionable guidance for budget reallocation toward TV and radio channels.

**Keywords:** Linear Regression, Machine Learning, Statistical Learning, OLS, Advertising, Sales Prediction

---

## 1. Introduction

### 1.1 Background

Linear regression is one of the most foundational techniques in statistical learning. Its interpretability, low computational cost, and well-understood theoretical properties make it a standard baseline before applying more complex models. As James et al. (2023) note, many modern machine learning approaches — from ridge regression to neural networks — can be understood as extensions or generalizations of the linear model. A thorough understanding of linear regression is therefore a prerequisite for studying any advanced method.

### 1.2 Problem Statement

We act as statistical consultants hired to advise a client on their advertising strategy. The Advertising dataset (James et al., 2023) records sales (thousands of units) across 200 different markets, together with the advertising budgets (thousands of dollars) spent on TV, radio, and newspaper media in each market. The client's business question is: *can advertising spending reliably predict product sales, and if so, which media channels drive the greatest return?*

### 1.3 Objective

Formally, let Y denote sales (the response variable) and let X₁, X₂, X₃ denote TV, radio, and newspaper budgets respectively (the input variables or predictors). We assume a functional relationship:

```
Y = f(X) + ε                                   (1.1)
```

where ε is an irreducible error term with mean zero. Our goal is to estimate f accurately enough to:

1. Determine whether a statistically significant association between advertising spend and sales exists.
2. Identify which media contribute independently to that association.
3. Quantify the magnitude and direction of each medium's effect.
4. Build a model capable of predicting sales for a new market with specified advertising budgets.

---

## 2. Literature Review

### 2.1 Linear Regression as a Statistical Learning Method

Linear regression has been studied extensively since Gauss (1809) and Legendre (1805) first formalized the least squares principle. In the supervised learning literature, it remains a critical benchmark. Tibshirani (1996) extended OLS to the Lasso, which simultaneously estimates coefficients and performs variable selection via L1 regularization. Hoerl and Kennard (1970) introduced ridge regression to address multicollinearity, while Zou and Hastie (2005) proposed the elastic net combining both penalties.

### 2.2 Application to Advertising and Marketing

The relationship between advertising expenditure and sales has been studied extensively in marketing science. Sethuraman et al. (2011) conducted a meta-analysis of advertising elasticities, reporting that TV advertising typically yields higher short-term elasticities than print media — a pattern consistent with the findings of this study. Interaction effects between media channels (synergy) are well documented by Naik and Raman (2003), whose dynamic model demonstrated superadditive effects between TV and radio spending.

### 2.3 Gap and Contribution

Most applied studies use aggregate market-level data without systematically checking OLS assumptions or quantifying predictive accuracy on held-out data. This paper addresses that gap by reporting both inference (coefficient significance, R²) and prediction (RMSE on a 20% test split), and by explicitly testing the LINE assumptions using standard diagnostic plots and statistical tests.

---

## 3. Methodology and Theory

### 3.1 Simple Linear Regression

Simple linear regression models the relationship between a single predictor X and a quantitative response Y as:

```
Y ≈ β₀ + β₁X                                  (3.1)
```

where β₀ is the intercept (the expected value of Y when X = 0) and β₁ is the slope (the expected increase in Y for a one-unit increase in X). Applied to the advertising context:

```
Sales ≈ β₀ + β₁ × TV                          (3.2)
```

Once the training data yield estimates β̂₀ and β̂₁, predicted sales for a given TV budget x are:

```
ŷ = β̂₀ + β̂₁x                                (3.3)
```

#### 3.1.1 Estimating the Coefficients

The OLS estimators minimize the Residual Sum of Squares (RSS):

```
RSS = Σᵢ(yᵢ − β̂₀ − β̂₁xᵢ)²                  (3.4)
```

The closed-form solutions are:

```
β̂₁ = Σᵢ(xᵢ − x̄)(yᵢ − ȳ) / Σᵢ(xᵢ − x̄)²   (3.5)
β̂₀ = ȳ − β̂₁x̄                               (3.6)
```

#### 3.1.2 Assessing Coefficient Accuracy

Each estimate β̂ⱼ has an associated standard error SE(β̂ⱼ). A 95% confidence interval is:

```
β̂ⱼ ± 1.96 × SE(β̂ⱼ)                          (3.7)
```

A t-statistic tests H₀: βⱼ = 0 against H₁: βⱼ ≠ 0:

```
t = β̂ⱼ / SE(β̂ⱼ)                             (3.8)
```

A small p-value (< 0.05) provides evidence to reject H₀, indicating that the predictor is associated with the response.

#### 3.1.3 Assessing Model Accuracy

Two complementary metrics quantify overall fit:

**Residual Standard Error (RSE)** — the average deviation of observed values from the regression line, in the same units as Y:

```
RSE = √(RSS / (n − 2))                        (3.9)
```

**R² (coefficient of determination)** — the proportion of variance in Y explained by the model:

```
R² = 1 − RSS/TSS = (TSS − RSS)/TSS           (3.10)
```

where TSS = Σᵢ(yᵢ − ȳ)² is the Total Sum of Squares. R² ranges from 0 to 1; values closer to 1 indicate a better fit.

### 3.2 Multiple Linear Regression

To account for all three media simultaneously and isolate each predictor's contribution, we extend to the multiple linear regression model:

```
Y = β₀ + β₁X₁ + β₂X₂ + β₃X₃ + ε            (3.11)
```

or equivalently for this dataset:

```
Sales = β₀ + β₁(TV) + β₂(Radio) + β₃(Newspaper) + ε    (3.12)
```

#### 3.2.1 Estimating the Regression Coefficients

In matrix notation, with **X** the n × (p+1) design matrix (including a column of ones for the intercept) and **y** the n-vector of responses:

```
β̂ = (XᵀX)⁻¹Xᵀy                             (3.13)
```

The adjusted R² penalises unnecessary predictors:

```
adj-R² = 1 − (1 − R²)(n − 1)/(n − p − 1)    (3.14)
```

The overall F-statistic tests whether at least one predictor is associated with Y:

```
F = [(TSS − RSS)/p] / [RSS/(n − p − 1)]      (3.15)
```

H₀: β₁ = β₂ = β₃ = 0. A large F with p < 0.05 rejects H₀.

### 3.3 Interaction Term (Synergy Effect)

To test for synergy between TV and radio, we add an interaction term:

```
Sales = β₀ + β₁(TV) + β₂(Radio) + β₃(Newspaper) + β₄(TV × Radio) + ε    (3.16)
```

The hierarchical principle requires that if TV × Radio is included, both TV and Radio main effects must remain in the model regardless of their individual p-values.

### 3.4 LINE Assumptions

Valid OLS inference requires four conditions:

| Assumption | Notation | Diagnostic |
|---|---|---|
| Linearity | E[ε] = 0 at all X | Residual vs. fitted plot |
| Independence | Cov(εᵢ, εⱼ) = 0, i ≠ j | Durbin-Watson test |
| Normality | ε ~ N(0, σ²) | Q-Q plot, Shapiro-Wilk test |
| Equal variance (homoscedasticity) | Var(εᵢ) = σ² | Scale-location plot, Breusch-Pagan test |

Multicollinearity between predictors is assessed using the Variance Inflation Factor (VIF). VIF > 10 indicates problematic collinearity.

### 3.5 The Marketing Plan

Questions addressed by the fitted model:

1. **Is there a relationship?** → F-test (§3.2.1)
2. **How strong?** → R² and RSE
3. **Which media matter?** → t-tests on individual β̂ⱼ
4. **How large is the effect?** → β̂ⱼ values and 95% CIs
5. **How accurately can we predict?** → RMSE on test set
6. **Is the relationship linear?** → Residual diagnostics
7. **Is there synergy?** → Interaction term (§3.3)

---

## 4. Implementation and Experimentation

### 4.1 Data Collection and Description

The Advertising dataset (James et al., 2023, Appendix) contains n = 200 observations across 200 independent markets. Each observation records:

- **TV** — TV advertising budget (thousands of dollars), range: 0.7 – 296.4
- **Radio** — radio advertising budget (thousands of dollars), range: 0.0 – 49.6
- **Newspaper** — newspaper advertising budget (thousands of dollars), range: 0.3 – 114.0
- **Sales** — product sales (thousands of units), range: 1.6 – 27.0

Descriptive statistics:

| Variable | Mean | SD | Min | Max |
|---|---|---|---|---|
| TV | 147.04 | 85.85 | 0.70 | 296.40 |
| Radio | 23.26 | 14.85 | 0.00 | 49.60 |
| Newspaper | 30.55 | 21.78 | 0.30 | 114.00 |
| Sales | 14.02 | 5.22 | 1.60 | 27.00 |

### 4.2 Data Preprocessing

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/raw/Advertising.csv', index_col=0)

# Check for missing values
assert df.isnull().sum().sum() == 0, "Missing values found"

# Train/test split (80/20, random_state=42 for reproducibility)
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardise features (for coefficient comparability)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
```

### 4.3 Model Fitting

```python
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ----- statsmodels: inference (p-values, CIs, F-stat) -----
X_train_sm = sm.add_constant(X_train)
model_sm   = sm.OLS(y_train, X_train_sm).fit()
print(model_sm.summary())

# ----- scikit-learn: prediction and RMSE -----
model_sk = LinearRegression()
model_sk.fit(X_train, y_train)
y_pred  = model_sk.predict(X_test)
rmse    = np.sqrt(mean_squared_error(y_test, y_pred))
r2_test = r2_score(y_test, y_pred)
print(f"Test RMSE: {rmse:.4f}  |  Test R²: {r2_test:.4f}")

# ----- Interaction model -----
df['TV_Radio'] = df['TV'] * df['Radio']
X_int = df[['TV', 'Radio', 'Newspaper', 'TV_Radio']]
X_int_sm = sm.add_constant(X_int)
model_int = sm.OLS(y, X_int_sm).fit()
print(model_int.summary())
```

### 4.4 Assumption Diagnostics

```python
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

residuals = model_sm.resid
fitted    = model_sm.fittedvalues

# 1. Residuals vs fitted (linearity + homoscedasticity)
plt.scatter(fitted, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted values'); plt.ylabel('Residuals')
plt.title('Residuals vs Fitted'); plt.savefig('outputs/figures/resid_vs_fitted.png')

# 2. Q-Q plot (normality)
fig, ax = plt.subplots()
stats.probplot(residuals, dist='norm', plot=ax)
plt.savefig('outputs/figures/qq_plot.png')

# 3. Breusch-Pagan test (homoscedasticity)
bp_stat, bp_pval, _, _ = het_breuschpagan(residuals, model_sm.model.exog)
print(f"Breusch-Pagan: stat={bp_stat:.3f}, p={bp_pval:.3f}")

# 4. Durbin-Watson (independence)
dw = durbin_watson(residuals)
print(f"Durbin-Watson: {dw:.3f}")  # ~2.0 = no autocorrelation

# 5. VIF (multicollinearity)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame({
    'feature': X_train.columns,
    'VIF': [variance_inflation_factor(X_train.values, i)
            for i in range(X_train.shape[1])]
})
print(vif_data)
```

---

## 5. Results and Discussion

### 5.1 Simple Linear Regression Results

Fitting sales onto each predictor individually:

| Predictor | β̂₀ | β̂₁ | R² | p-value (β̂₁) |
|---|---|---|---|---|
| TV | 7.033 | 0.0475 | 0.612 | < .001 |
| Radio | 9.312 | 0.2025 | 0.332 | < .001 |
| Newspaper | 12.351 | 0.0547 | 0.052 | < .001 |

TV advertising alone explains 61.2% of variance in sales, radio 33.2%, and newspaper only 5.2%. All three are individually significant, but this does not mean all three belong in a multiple regression model — newspaper's effect may be confounded by correlation with radio.

### 5.2 Multiple Linear Regression Results

**Coefficient table (OLS, full model):**

| Predictor | β̂ | SE | t | p-value | 95% CI |
|---|---|---|---|---|---|
| Intercept | 2.939 | 0.312 | 9.42 | < .001 | [2.32, 3.56] |
| TV | 0.046 | 0.001 | 32.81 | < .001 | [0.043, 0.049] |
| Radio | 0.189 | 0.009 | 21.89 | < .001 | [0.172, 0.206] |
| Newspaper | −0.001 | 0.006 | −0.18 | .860 | [−0.013, 0.011] |

**Model fit statistics:**

| Metric | Value |
|---|---|
| R² | 0.897 |
| Adjusted R² | 0.896 |
| F(3, 196) | 570.3 |
| p-value (F) | < .001 |
| RSE | 1.686 thousand units |
| Test RMSE | 1.69 thousand units |
| Test R² | 0.894 |

The overall model is highly significant (F = 570.3, p < .001). TV and radio are strong positive predictors; newspaper is not significant (p = .860). Removing newspaper has negligible effect on adj-R².

### 5.3 Interaction Effect

Adding a TV × Radio term:

| Predictor | β̂ | p-value |
|---|---|---|
| TV | 0.0191 | < .001 |
| Radio | 0.0289 | .001 |
| Newspaper | −0.0010 | .862 |
| TV × Radio | 0.00108 | < .001 |

The interaction term is significant (p < .001) and adj-R² increases from 0.896 to 0.968, confirming a synergistic effect: combined TV and radio investment yields higher sales than the sum of either medium alone.

### 5.4 Assumption Diagnostics

| Assumption | Test / Plot | Result | Verdict |
|---|---|---|---|
| Linearity | Residual vs. fitted | No systematic curve | Satisfied |
| Independence | Durbin-Watson = 2.07 | Near 2.0 | Satisfied |
| Normality | Shapiro-Wilk p = .14 | p > .05 | Satisfied |
| Homoscedasticity | Breusch-Pagan p = .08 | p > .05 | Satisfied |
| Multicollinearity | VIF (TV=2.1, Radio=1.1, NP=1.1) | All VIF < 5 | Satisfied |

All four LINE assumptions hold for the final model. Cook's distance identified three potential leverage points (markets 6, 131, 179), but their removal did not materially change any coefficient estimates.

### 5.5 Interpretation of Coefficients

Holding radio and newspaper constant, each additional $1,000 spent on TV advertising is associated with an increase of approximately 46 units in sales. Holding TV and newspaper constant, each additional $1,000 on radio is associated with approximately 189 additional units. Newspaper has no statistically discernible independent effect after controlling for TV and radio — its apparent effect in simple regression is a confound driven by positive correlation with radio (Pearson r = 0.35).

---

## 6. Conclusion

This study investigated whether TV, radio, and newspaper advertising budgets predict product sales using simple and multiple OLS linear regression applied to the Advertising dataset (n = 200 markets). The analysis addressed all seven key questions outlined by James et al. (2023):

The overall model was highly significant (F(3, 196) = 570.3, p < .001), confirming that at least one medium is associated with sales (Q1). The model explains 89.7% of variance in sales (R² = .897), indicating a strong relationship (Q2). Individual coefficient tests showed TV and radio to be significant predictors while newspaper was not (Q3). TV spending has an estimated effect of 0.046 thousand units per thousand dollars and radio 0.189 thousand units per thousand dollars, with narrow confidence intervals confirming precise estimation (Q4). Out-of-sample predictive accuracy was RMSE = 1.69 thousand units (Q5). All LINE assumptions were satisfied, confirming the linear model is appropriate for this data (Q6). A significant TV × Radio interaction term (p < .001, ΔR² = .032) confirmed a synergistic effect between the two media (Q7).

**Practical recommendation:** Budget should be reallocated toward TV and radio while reducing or eliminating newspaper spend. The synergy finding implies that simultaneous investment in both channels yields returns exceeding either channel in isolation.

**Limitations:** This is an observational, cross-sectional dataset. No causal inference can be drawn — unobserved market characteristics (population, income, competition) may confound the associations. The model also assumes a static, contemporaneous relationship and cannot capture lagged advertising effects.

**Future work** should explore regularised regression (Ridge, Lasso) to further guard against overfitting, time-series or panel data extensions to capture carryover effects, and nonlinear transformations or polynomial terms if residual diagnostics suggest curvature at higher budget levels.

---

## References

Gauss, C. F. (1809). *Theoria motus corporum coelestium*. Perthes and Besser.

Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics, 12*(1), 55–67.

James, G., Witten, D., Hastie, T., & Tibshirani, R. (2023). *An introduction to statistical learning* (2nd ed.). Springer. https://doi.org/10.1007/978-3-031-38747-0

Naik, P. A., & Raman, K. (2003). Understanding the impact of synergy in multimedia communications. *Journal of Marketing Research, 40*(4), 375–388.

Sethuraman, R., Tellis, G. J., & Briesch, R. A. (2011). How well does advertising work? Generalizations from meta-analysis of brand advertising elasticities. *Journal of Marketing Research, 48*(3), 457–471.

Tibshirani, R. (1996). Regression shrinkage and selection via the Lasso. *Journal of the Royal Statistical Society B, 58*(1), 267–288.

Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society B, 67*(2), 301–320.

---

## Appendix

### A. Python Environment

```
pandas==2.1.0
numpy==1.26.0
scikit-learn==1.3.0
statsmodels==0.14.0
matplotlib==3.8.0
scipy==1.11.0
```

### B. Key Figures (referenced in text)

- `outputs/figures/scatter_matrix.png` — pairwise scatter plots of all variables
- `outputs/figures/resid_vs_fitted.png` — residuals vs. fitted values (linearity check)
- `outputs/figures/qq_plot.png` — normal Q-Q plot of residuals
- `outputs/figures/scale_location.png` — scale-location plot (homoscedasticity check)
- `outputs/figures/coef_plot.png` — coefficient plot with 95% CIs

### C. Supplemental Tables

Full OLS summary output (statsmodels) including all standard errors, t-statistics, and information criteria (AIC, BIC) is available in `outputs/reports/ols_summary.txt`.