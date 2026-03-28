# Statistics Reference
## ISL Ch.3 · Linear Regression · Advertising Dataset

---

## LINE Assumptions — Full Table

| # | Assumption | Formal Statement | Diagnostic | Remediation |
|---|---|---|---|---|
| L | Linearity | E[εᵢ\|Xᵢ] = 0 | Residuals vs. fitted plot (no curve) | Polynomial / log transform |
| I | Independence | Cov(εᵢ,εⱼ) = 0, i≠j | Durbin-Watson ≈ 2.0 | GLS, HAC standard errors |
| N | Normality | ε ~ N(0,σ²) | Q-Q plot; Shapiro-Wilk p > .05 | Box-Cox, bootstrap CI |
| E | Equal variance | Var(εᵢ) = σ² (homoscedastic) | Scale-location; Breusch-Pagan p > .05 | WLS, robust SEs |

**Additional checks:**
| Check | Threshold | If violated |
|---|---|---|
| Multicollinearity (VIF) | VIF < 5 acceptable; > 10 critical | Ridge / Lasso regression |
| Influential points (Cook's D) | D > 4/n → investigate | Remove / robust regression |
| Outliers (studentised residuals) | \|r\| > 3 → investigate | Winsorise or remove |

---

## All Hypothesis Tests

| Test | H₀ | Statistic | Reject H₀ when |
|---|---|---|---|
| Overall model fit | β₁=β₂=…=βₚ=0 | F = [(TSS−RSS)/p] / [RSS/(n−p−1)] | p < α (.05) |
| Individual coefficient | βⱼ = 0 | t = β̂ⱼ / SE(β̂ⱼ) | p < α |
| Normality of residuals | ε ~ Normal | Shapiro-Wilk W | p < .05 → NOT normal |
| Homoscedasticity | Var(ε) constant | Breusch-Pagan χ² | p < .05 → heteroscedastic |
| No autocorrelation | ρ = 0 | Durbin-Watson d | d < 1.5 or d > 2.5 → autocorrelation |

---

## Regression Metrics — All Formulas

```
RSS     = Σᵢ (yᵢ − ŷᵢ)²
TSS     = Σᵢ (yᵢ − ȳ)²
R²      = 1 − RSS/TSS                           ← 0–1, higher = better
adj-R²  = 1 − (1−R²)(n−1)/(n−p−1)              ← penalises extra predictors
RSE     = √(RSS / (n−p−1))                       ← in units of Y
RMSE    = √[(1/n) Σ(yᵢ−ŷᵢ)²]                   ← test set preferred
MAE     = (1/n) Σ|yᵢ−ŷᵢ|                        ← robust to outliers
MAPE    = (100/n) Σ|yᵢ−ŷᵢ|/|yᵢ|               ← percentage error
```

---

## OLS Coefficient Formulas

```
Simple LR:
  β̂₁ = Σ(xᵢ−x̄)(yᵢ−ȳ) / Σ(xᵢ−x̄)²
  β̂₀ = ȳ − β̂₁x̄

Multiple LR (matrix form):
  β̂ = (XᵀX)⁻¹Xᵀy

95% Confidence Interval:
  β̂ⱼ ± 1.96 × SE(β̂ⱼ)

t-statistic:
  t = β̂ⱼ / SE(β̂ⱼ),   df = n − p − 1
```

---

## VIF Formula

```
VIFⱼ = 1 / (1 − R²ⱼ)
```
R²ⱼ = R² from regressing Xⱼ on all other predictors.

---

## Confidence Interval vs. Prediction Interval

| Type | Formula | Use |
|---|---|---|
| CI for mean response | ŷ ± t × SE(ŷ) | "Where is the average Y for all markets spending $X?" |
| PI for new observation | ŷ ± t × √(SE(ŷ)² + σ̂²) | "What sales will this specific new market achieve?" |

PI is always wider. State which you report.

---

## 4-Element Reporting Rule (use in every paper result)

Every statistical result must include:
```
[statistic]([df]) = [value], p = [value], [effect size or CI]
```

Examples:
```
F(3, 196) = 570.3, p < .001, R² = .897
t(196) = 32.81, p < .001, 95% CI [0.043, 0.049]
W(200) = 0.991, p = .142  (Shapiro-Wilk — normality holds)
χ²(3) = 6.14, p = .105    (Breusch-Pagan — homoscedasticity holds)
DW = 2.07                  (Durbin-Watson — independence holds)
```

---

## Common Reporting Mistakes

| Wrong | Correct |
|---|---|
| "p = 0.000" | "p < .001" |
| Only report p-value | Report β̂, SE, t/F, p, CI |
| "proves causation" | "indicates association" |
| Checking assumptions after modelling | Always check BEFORE (pre-fit) and AFTER (residuals) |
| Using R² alone | Also report adj-R² and RMSE on test set |
| "not significant, so no effect" | "no evidence of a significant effect at α = .05" |
