# Statistics Reference
## Formulas · Reporting Rules · LINE Assumptions

---

## All Formulas (numbered for paper)

```
(1)  Simple LR:       Sales ≈ β₀ + β₁X
(2)  RSS:             RSS = Σᵢ(yᵢ − β̂₀ − β̂₁xᵢ)²
(3)  Normal eqs:      β̂ = (XᵀX)⁻¹Xᵀy
(4)  Multiple LR:     Sales = β₀ + β₁TV + β₂Radio + β₃Newspaper + ε
(5)  Interaction:     Sales = β₀ + β₁TV + β₂Radio + β₃NP + β₄(TV×Radio) + ε
(6)  R²:              R² = 1 − RSS/TSS = (TSS − RSS)/TSS
(7)  adj-R²:          adj-R² = 1 − (1−R²)(n−1)/(n−p−1)
(8)  RSE:             RSE = √(RSS / (n−p−1))
(9)  RMSE:            RMSE = √[(1/n)Σ(yᵢ−ŷᵢ)²]
(10) MAE:             MAE = (1/n)Σ|yᵢ−ŷᵢ|
(11) SE(β̂₁):          SE(β̂₁)² = σ² / Σ(xᵢ−x̄)²
(12) 95% CI:          β̂ⱼ ± 1.96 × SE(β̂ⱼ)
(13) t-statistic:     t = β̂ⱼ / SE(β̂ⱼ)
(14) F-statistic:     F = [(TSS−RSS)/p] / [RSS/(n−p−1)]
(15) VIF:             VIFⱼ = 1 / (1 − R²ⱼ)
```

---

## LINE Assumptions — Full Table

| # | Assumption | Formal Statement | Diagnostic | Test | Remediation |
|---|---|---|---|---|---|
| L | Linearity | E[εᵢ\|Xᵢ] = 0 | Residuals vs Fitted (no curve) | Visual | Polynomial / log transform |
| I | Independence | Cov(εᵢ,εⱼ) = 0 | Durbin-Watson d ≈ 2.0 | DW test | GLS, HAC SEs |
| N | Normality | ε ~ N(0,σ²) | Q-Q plot; Shapiro-Wilk p > .05 | SW test | Box-Cox, bootstrap CI |
| E | Equal variance | Var(εᵢ) = σ² | Scale-location; Breusch-Pagan p > .05 | BP test | WLS, robust SEs |

Additional:
- VIF < 5: acceptable · VIF > 10: critical → ridge/lasso
- Cook's D > 4/n: investigate influential point

---

## 4-Element Reporting Rule

Every result must contain ALL four elements:
```
[statistic]([df]) = [value], p = [value], [CI or effect size]
```

Examples:
```
F(3, 196) = 570.3, p < .001, R² = .897
t(196) = 32.81, p < .001, 95% CI [0.043, 0.049]
W(200) = 0.991, p = .142    ← Shapiro-Wilk
χ²(3) = 6.14, p = .105     ← Breusch-Pagan
d = 2.07                    ← Durbin-Watson (no p needed)
```

---

## Key Results (Advertising dataset — verified values)

| Metric | Value |
|---|---|
| n (total) | 200 |
| n_train | 160 |
| n_test | 40 |
| R² (multiple LR, train) | 0.900 |
| adj-R² | 0.896 |
| R² (test) | 0.894 |
| RMSE (test) | 1.69 K units |
| F-statistic | 570.3 |
| TV β̂ | 0.046 |
| Radio β̂ | 0.189 |
| Newspaper β̂ | −0.001 |
| TV p-value | < .001 |
| Radio p-value | < .001 |
| Newspaper p-value | .860 |
| TV 95% CI | [0.043, 0.049] |
| Radio 95% CI | [0.172, 0.206] |
| Newspaper 95% CI | [−0.013, 0.011] |
| TV VIF | 1.005 |
| Radio VIF | 1.145 |
| Newspaper VIF | 1.145 |
| Shapiro-Wilk W | 0.991 |
| Shapiro-Wilk p | .142 |
| Breusch-Pagan χ² | 6.14 |
| Breusch-Pagan p | .105 |
| Durbin-Watson d | 2.07 |
| Interaction β̂₄ | 0.00108 |
| Interaction p | < .001 |
| R² with interaction | 0.968 |
| ΔR² | 0.071 |
| RMSE with interaction | 0.93 K units |
| Baseline RMSE | 5.22 K units |

---

## Prohibited Language (formal academic style)

| Never write | Write instead |
|---|---|
| "I found that..." | "The analysis revealed that..." |
| "We ran the model..." | "The model was fitted using..." |
| "Our results show..." | "The results indicate..." |
| "My hypothesis..." | "The study hypothesis..." |
| "We used Python..." | "Python was employed..." |
| "As I mentioned..." | "As noted in Section III..." |
| "It seems like..." | "The evidence suggests..." |
| "More research needed" | [3 specific concrete directions] |
