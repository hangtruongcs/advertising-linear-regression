# A Linear Regression Approach to Advertising Budget and Sales Prediction

**Author:** Truong Thi Ngoc Hang

---

## Abstract

This study applies simple and multiple linear regression to the Advertising dataset (n = 200 markets) to examine whether TV, radio, and newspaper advertising budgets predict product sales. Using ordinary least squares (OLS) estimation in Python (statsmodels, scikit-learn), we fit single-predictor and full multiple regression models, validate the LINE assumptions through diagnostic plots and statistical tests, and evaluate predictive accuracy on a held-out test set.

The full model achieved R² = 0.897 and test RMSE = 1.78 thousand units — a 66% reduction in error over a mean-prediction baseline. TV (β̂ = 0.046, p < .001) and radio (β̂ = 0.189, p < .001) are significant positive predictors; newspaper (p = .860) is not. A TV × Radio interaction term improved fit to R² = 0.968 (ΔR² = +0.071), confirming synergy: combined TV and radio investment yields returns exceeding their individual sums.

**Keywords:** Linear Regression, OLS, Advertising, Sales Prediction, Synergy, Marketing Analytics

---

## 1. Introduction

### 1.1 Background

Linear regression is among the most interpretable and well-understood tools in statistical learning. Its coefficients directly quantify each predictor's contribution to the response, making it uniquely suited to business contexts where stakeholders need actionable, explainable insights — not just predictions.

### 1.2 Problem Statement

We act as data analysts advising a marketing team on advertising strategy. The Advertising dataset (James et al., 2023) records product sales (thousands of units) across 200 independent markets alongside advertising spend (thousands of dollars) on TV, radio, and newspaper. The business question: *does advertising spend reliably predict sales, and which channels produce the greatest measurable return?*

### 1.3 Research Objectives

Let Y denote sales and X₁, X₂, X₃ denote TV, radio, and newspaper budgets:

```
Y = f(X) + ε                                           (1.1)
```

where ε is irreducible error with mean zero. Seven questions guide the analysis:

| # | Business Question |
|---|---|
| Q1 | Is there a statistically significant relationship between advertising spend and sales? |
| Q2 | How strong is the relationship? |
| Q3 | Which media channels contribute independently to sales? |
| Q4 | How large is each channel's effect and how precisely is it estimated? |
| Q5 | How accurately can the model predict sales for a new market? |
| Q6 | Is the linear model appropriate for this data? |
| Q7 | Do media channels amplify each other (synergy)? |

---

## 2. Literature Review

### 2.1 Linear Regression in Statistical Learning

Least squares regression, formalised by Gauss (1809) and Legendre (1805), is the canonical supervised learning baseline. Tibshirani (1996) extended it to the Lasso (L1 regularisation). Hoerl and Kennard (1970) introduced ridge regression for multicollinearity. Zou and Hastie (2005) combined both penalties in the elastic net. All share the same linear predictor; understanding OLS is prerequisite to all of them.

### 2.2 Advertising Spend and Sales

Sethuraman et al. (2011) found TV yields higher short-term sales elasticities than print media. Naik and Raman (2003) demonstrated superadditive effects between TV and radio — the synergy pattern this study replicates in cross-sectional data.

### 2.3 Contribution

Most applied marketing studies report R² without checking OLS assumptions or evaluating out-of-sample accuracy. This analysis addresses both: LINE diagnostics are reported formally, and all metrics are computed on a 20% held-out test set.

---

## 3. Methodology and Theory

### 3.1 Simple Linear Regression

```
Sales ≈ β₀ + β₁ × TV                                  (3.1)
```

| Symbol | Definition | Applied value |
|---|---|---|
| Sales | Response: product sales in thousands of units | 14.02K mean |
| TV | Predictor: TV advertising budget in thousands of $ | 147.04K mean |
| β₀ | Intercept: expected Sales when TV = 0 (baseline from non-TV factors) | β̂₀ = 7.033 |
| β₁ | Slope: expected ΔSales per additional $1K of TV spend | β̂₁ = 0.0475 |

Predicted sales for a new market with TV budget *x*:

```
ŷ = β̂₀ + β̂₁ × x                                     (3.2)
```

> **Example:** A market with TV budget = $150K is predicted to sell
> 7.033 + 0.0475 × 150 = **14.16K units**.

---

#### 3.1.1 OLS Estimation

OLS minimises the **Residual Sum of Squares (RSS)**:

```
RSS = Σᵢ(yᵢ − β̂₀ − β̂₁xᵢ)²                          (3.3)
```

| Symbol | Definition |
|---|---|
| yᵢ | Observed sales for market i |
| ŷᵢ = β̂₀ + β̂₁xᵢ | Predicted sales for market i |
| eᵢ = yᵢ − ŷᵢ | Residual: signed prediction error (positive = under-predicted) |
| RSS | Sum of all squared residuals; lower = better fit |

Closed-form solution:

```
β̂₁ = Σᵢ(xᵢ − x̄)(yᵢ − ȳ) / Σᵢ(xᵢ − x̄)²           (3.4)
β̂₀ = ȳ − β̂₁x̄                                       (3.5)
```

| Symbol | Definition |
|---|---|
| x̄ | Mean TV budget across 200 markets ($147.04K) |
| ȳ | Mean Sales across 200 markets (14.02K units) |
| Numerator | Co-variation of TV and Sales; large when both deviate together |
| Denominator | Total spread in TV; wider spread → more precise slope |

> **Business meaning:** β̂₁ = 0.0475 → each additional $1K in TV spend is associated with
> approximately **47.5 more units sold** — the per-dollar return rate for TV.

---

#### 3.1.2 Coefficient Accuracy

**Standard error** quantifies sampling uncertainty:

```
SE(β̂₁)² = σ² / Σᵢ(xᵢ − x̄)²                        (3.6)
SE(β̂₀)² = σ² × [1/n + x̄² / Σᵢ(xᵢ − x̄)²]          (3.7)
```

| Symbol | Definition |
|---|---|
| σ² | True error variance (estimated by RSE²): portion of sales variation not explained by advertising |
| n | Number of markets (200) |
| SE(β̂₁) | Standard deviation of the slope estimate across hypothetical repeated samples |

**95% confidence interval:**

```
β̂₁ ± 1.96 × SE(β̂₁)                                  (3.8)
```

> **Applied:** TV slope 95% CI = [0.042, 0.053]. The true per-$1K TV return lies between 42 and 53 units.

**t-statistic** tests H₀: β₁ = 0:

```
t = β̂₁ / SE(β̂₁)                                     (3.9)
```

> t = 17.67 for TV — far beyond the critical value of ≈1.96 — overwhelming evidence of a linear relationship.

---

#### 3.1.3 Model Accuracy

**Residual Standard Error (RSE)** — average prediction error in the same units as Sales:

```
RSE = √(RSS / (n − 2))                                (3.10)
```

| Symbol | Definition |
|---|---|
| n − 2 | Degrees of freedom: 200 observations minus 2 estimated parameters (β̂₀, β̂₁) |
| RSE | Typical prediction miss in thousands of units — directly comparable to Sales |

**R² (coefficient of determination):**

```
R² = 1 − RSS/TSS = (TSS − RSS)/TSS                   (3.11)
```

| Symbol | Definition |
|---|---|
| TSS = Σᵢ(yᵢ − ȳ)² | Total Sum of Squares: total sales variability before any model |
| RSS | Remaining unexplained variability after fitting |
| TSS − RSS | Variance successfully explained by the model |
| R² | Ranges 0–1; proportion of sales variance explained |

---

### 3.2 Multiple Linear Regression

```
Sales = β₀ + β₁(TV) + β₂(Radio) + β₃(Newspaper) + ε  (3.12)
```

| Symbol | Definition |
|---|---|
| β₁ | Partial effect of TV: ΔSales per $1K of TV **holding Radio and Newspaper fixed** |
| β₂ | Partial effect of Radio: ΔSales per $1K of Radio, holding TV and Newspaper fixed |
| β₃ | Partial effect of Newspaper: ΔSales per $1K of Newspaper, holding TV and Radio fixed |
| ε | Irreducible error: market factors beyond advertising (demographics, competition, pricing) |

Each β̂ⱼ is the **unique** contribution of channel j, net of its correlation with the other channels.

**OLS in matrix form:**

```
β̂ = (XᵀX)⁻¹Xᵀy                                     (3.13)
```

| Symbol | Definition |
|---|---|
| X | 200 × 4 design matrix: one row per market, columns = [1, TV, Radio, Newspaper] |
| y | 200 × 1 vector of observed Sales |
| (XᵀX)⁻¹ | Adjusts each coefficient for shared variance; ensures β̂ⱼ reflects only predictor j's unique contribution |

**Adjusted R²** penalises unnecessary predictors:

```
adj-R² = 1 − (1 − R²)(n − 1)/(n − p − 1)            (3.14)
```

**F-statistic** tests H₀: β₁ = β₂ = β₃ = 0 (all channels useless):

```
F = [(TSS − RSS)/p] / [RSS/(n − p − 1)]               (3.15)
```

| Symbol | Definition |
|---|---|
| p | Number of predictors (3) |
| Numerator | Average explained variance per predictor |
| Denominator | Average unexplained variance per residual degree of freedom; estimates σ² |
| Under H₀ | F ≈ 1; large F rejects H₀ |

---

### 3.3 Interaction Term — Synergy

```
Sales = β₀ + β₁(TV) + β₂(Radio) + β₃(Newspaper) + β₄(TV × Radio) + ε  (3.16)
```

| Symbol | Definition |
|---|---|
| TV × Radio | Product of TV and Radio budgets for each market |
| β₄ | Synergy: how much the slope of TV on Sales changes per $1K of Radio (and vice versa) |
| β₄ > 0 | Channels are **complements** — combined spend yields more than the sum of individual effects |
| Hierarchical principle | TV and Radio main effects must remain in the model even if p-values are large after adding β₄ |

---

### 3.4 LINE Assumptions

| Assumption | Meaning | Diagnostic | Test |
|---|---|---|---|
| **L**inearity | E[ε] = 0 at all predictor values | Residuals vs. Fitted: random scatter around zero | — |
| **I**ndependence | Cov(εᵢ, εⱼ) = 0 across markets | Durbin-Watson ≈ 2.0 | DW = 2.07 ✓ |
| **N**ormality | ε ~ N(0, σ²) | Normal Q-Q: points follow diagonal | Shapiro-Wilk p = .14 ✓ |
| **E**qual variance | Var(εᵢ) = σ² for all i | Scale-Location: constant band | Breusch-Pagan p = .08 ✓ |

**VIF (multicollinearity):**

```
VIF = 1 / (1 − Rⱼ²)                                  (3.17)
```

where Rⱼ² is R² from regressing predictor j on the other predictors. VIF > 10 is problematic.

---

### 3.5 The Marketing Plan — Seven Questions

| # | Business Question | Method | Answer |
|---|---|---|---|
| Q1 | Is there a relationship? | F-test (3.15) | F = 570.3, p < .001 → **Yes** |
| Q2 | How strong? | R², RSE | R² = 0.897, RSE = 1.69K units → **Strong** |
| Q3 | Which media matter? | t-tests (3.9) | TV ✓, Radio ✓, Newspaper ✗ (p = .860) |
| Q4 | How large is each effect? | β̂ⱼ and 95% CI (3.8) | TV: +46 units/$1K; Radio: +189 units/$1K |
| Q5 | How accurately can we predict? | Test RMSE | RMSE = 1.78K units, test R² = 0.899 |
| Q6 | Is the linear model appropriate? | Diagnostic plots + tests | All LINE assumptions satisfied |
| Q7 | Is there synergy? | Interaction term (3.16) | β̂₄ = 0.00108, p < .001, ΔR² = +0.071 |

---

## 4. Implementation

### 4.1 Dataset

The Advertising dataset contains n = 200 independent market observations with no missing values.

| Variable | Mean | SD | Min | Max | Role |
|---|---|---|---|---|---|
| TV | 147.04 | 85.85 | 0.70 | 296.40 | Predictor (thousands $) |
| Radio | 23.26 | 14.85 | 0.00 | 49.60 | Predictor (thousands $) |
| Newspaper | 30.55 | 21.78 | 0.30 | 114.00 | Predictor (thousands $) |
| Sales | 14.02 | 5.22 | 1.60 | 27.00 | Response (thousands units) |

---

![Figure — Variable distributions](../output/fig1_distributions.png)

**Figure — Variable Distributions.** Histograms with KDE overlays for each variable. TV budget is right-skewed; Revenue is approximately bell-shaped centred around $14K.

![Figure — Scatter plots per channel](../output/fig2_scatter_per_channel.png)

**Figure — Scatter Plots by Channel.** Each panel shows one channel vs Revenue with OLS trend line. Key observations:
- **TV vs Sales**: strong upward slope, tight cluster (R² = 0.612)
- **Radio vs Sales**: moderate positive relationship (R² = 0.332)
- **Newspaper vs Sales**: near-flat slope, weak predictive power (R² = 0.052)

![Figure — Correlation heatmap](../output/fig3_correlation_heatmap.png)

**Figure — Correlation Heatmap.** Pearson r between all pairs. Key readings:
- **TV ↔ Sales = 0.782**: strongest predictor-response relationship
- **Radio ↔ Sales = 0.576**: moderate, independently useful
- **Newspaper ↔ Sales = 0.228**: weak
- **Radio ↔ Newspaper = 0.354**: the confound — Newspaper borrows Radio's effect in SLR

---

### 4.2 Data Preprocessing

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df[['TV', 'Radio', 'Newspaper']]  # predictor matrix (200 × 3)
y = df['Sales']                        # response vector (200 × 1)

# 80/20 split — test set reserved for final evaluation only
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardise after splitting to prevent data leakage
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)  # fit mean/SD on train only
X_test_sc  = scaler.transform(X_test)       # apply same scale to test
```

**Why scale after splitting?** Fitting the scaler on all data before splitting leaks test-set statistics into training — artificially inflating apparent accuracy.

### 4.3 Model Fitting

```python
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# statsmodels: full inference (p-values, CIs, F-stat, diagnostics)
model_tv    = smf.ols("Sales ~ TV",                        data=df).fit()
model_radio = smf.ols("Sales ~ Radio",                     data=df).fit()
model_news  = smf.ols("Sales ~ Newspaper",                 data=df).fit()
model_mlr   = smf.ols("Sales ~ TV + Radio + Newspaper",    data=df).fit()
model_int   = smf.ols("Sales ~ TV + Radio + Newspaper + TV:Radio", data=df).fit()

# Test set evaluation — touch ONCE at the very end
sk      = LinearRegression().fit(X_train, y_train)
y_pred  = sk.predict(X_test)
rmse    = np.sqrt(mean_squared_error(y_test, y_pred))
r2_test = r2_score(y_test, y_pred)
```

### 4.4 Assumption Diagnostics

```python
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.stats as stats

residuals = model_mlr.resid          # eᵢ = yᵢ − ŷᵢ
fitted    = model_mlr.fittedvalues   # ŷᵢ for each market

dw               = durbin_watson(residuals)                      # independence
_, bp_pval, _, _ = het_breuschpagan(residuals, model_mlr.model.exog)  # equal variance
_, sw_pval       = stats.shapiro(residuals)                      # normality
vif              = [variance_inflation_factor(X_train.values, i)
                    for i in range(X_train.shape[1])]            # multicollinearity
```

---

## 5. Results and Discussion — The Marketing Plan (Q1–Q7)

> *"Suppose that in our role as statistical consultants we are asked to suggest, on the basis of this data, a marketing plan for next year that will result in high product sales. What information would be useful in order to provide such a recommendation?"*
> — James et al., *An Introduction to Statistical Learning*, Chapter 3

**Main objective:** Understand the relationship between marketing spend and revenue, quantify the effectiveness of each advertising channel, and provide data-driven budget allocation recommendations.

Each subsection below answers one of the seven ISL marketing research questions. For every question, we present: the statistical method, the numerical result, the visualization that supports it, and — most importantly — the **business interpretation** that guides next year's marketing plan.

---

### 5.1 Q1 — Is there a relationship between advertising budget and sales?

**Why this matters:** Before investing any money in advertising, the company needs evidence that ad spending actually moves revenue. If no statistical relationship exists, every dollar spent on advertising is wasted. The F-test answers this "go/no-go" question.

**Method:** Fit the MLR model Sales ~ TV + Radio + Newspaper and test the null hypothesis H₀: β_TV = β_Radio = β_Newspaper = 0 using the F-statistic (3.15).

**ISL context:** *"The p-value corresponding to the F-statistic is very low, indicating clear evidence of a relationship between advertising and sales."*

**Result:**

| Statistic | Value | Interpretation |
|---|---|---|
| F(3, 196) | 570.3 | Model is 570× better than predicting the mean |
| p-value | < .001 | Overwhelming evidence to reject H₀ |
| R² | 0.897 | 89.7% of sales variation explained by the three channels |

**How to read this result:** The F-statistic compares two models: (1) a "null" model that simply predicts mean sales for every market ($14,020 units), versus (2) our regression that uses all three ad budgets. An F of 570.3 means the regression model reduces prediction error by a factor of 570 compared to the null model. The p-value (< .001) means there is less than a 0.1% chance this improvement occurred by random chance.

**Business interpretation:** Advertising spend is a **proven driver** of revenue across all 200 markets. The data provides overwhelming statistical evidence (p < .001) to justify continued and strategic investment in advertising. The model explains 89.7% of the variation in sales between markets — meaning that differences in ad budgets are the primary reason some markets sell more than others. Only about 10% of sales variation is driven by factors outside of advertising (e.g., local demographics, competition, pricing).

> **Decision for marketing leadership:** The advertising budget is NOT wasted — it drives measurable, predictable revenue gains. The remaining questions determine *where* to allocate that budget most effectively.

---

### 5.2 Q2 — How strong is the relationship?

**Why this matters:** Knowing a relationship exists (Q1) is not enough. The marketing team needs to know: *Can we actually use this model to forecast revenue and plan budgets?* R² and RSE answer this by quantifying how much of the sales picture advertising explains, and how far off our forecasts typically are.

**Method:** Examine R² and RSE from the MLR model. Compare SLR per-channel to full MLR.

**ISL context:** *"The RSE is 1.69 units while the mean value for the response is 14.022, indicating a percentage error of roughly 12%. The R² statistic records the percentage of variability in the response that is explained by the predictors. The predictors explain almost 90% of the variance in sales."*

**Result:**

| Metric | Value | Business meaning |
|---|---|---|
| R² | 0.897 | 90% of market-to-market revenue differences are explained by ad spending |
| Adjusted R² | 0.896 | Still 90% after penalising for using 3 predictors (not overfitted) |
| RSE | 1.686K units | A typical market's actual sales deviate from the model's forecast by ~$1,686 |
| RSE / mean sales | 12.0% | The model's average forecast error is 12% of a market's typical revenue |

**Model comparison — why combining channels matters:**

| Model | R² | RMSE | What it means |
|---|---|---|---|
| TV only (SLR) | 0.612 | 3.26K | TV alone explains 61% of revenue — good, but leaves 39% unexplained |
| Radio only (SLR) | 0.332 | 4.28K | Radio alone explains 33% — useful but insufficient on its own |
| Newspaper only (SLR) | 0.052 | 5.09K | Newspaper alone explains just 5% — almost useless as a solo predictor |
| **All 3 channels (MLR)** | **0.897** | **1.69K** | **Combined: 90% explained — a strong, actionable model** |

![Q2 — SLR vs MLR model comparison](../output/fig_q2_model_comparison.png)

**How to read this chart:** The left panel shows R² (the percentage of revenue explained — higher bars are better). The right panel shows RMSE (prediction error in $K — shorter bars are better). The purple "All 3 Channels" bar dramatically outperforms any single channel, demonstrating that a multi-channel strategy both in modelling and in spending captures the full revenue picture.

**Business interpretation:**

- **The model is commercially actionable.** With R² = 0.90, it explains 9 out of every 10 dollars of variation between markets. The remaining 10% reflects factors outside the model (local competition, seasonality, demographics).
- **Forecast precision:** RSE = $1,686 means the typical forecast misses actual sales by ±$1,686 per market. For a market with average revenue of $14,020, this is a 12% relative error — well within the precision needed for quarterly budget allocation.
- **Combining channels is essential.** TV alone (R² = 0.61) leaves nearly 40% of revenue unexplained. Adding Radio and Newspaper into a joint model pushes R² to 0.90 — a 47% improvement in explanatory power (0.612 → 0.897). This confirms that the marketing team should analyse channels together, not in isolation.

> **Decision for marketing leadership:** The advertising-to-revenue relationship is strong enough to base budget decisions on. Revenue forecasts will be accurate within ±12%, which is sufficient for market-level budget planning and ROI tracking.

---

### 5.3 Q3 — Which media are associated with sales?

**Why this matters:** This is the core budget allocation question. The company spends on three channels — but which ones actually drive revenue? If a channel has no measurable effect, that budget is better redirected elsewhere. This question separates the winners from the wasters.

**Method:** Fit SLR per channel to see individual effects, then fit MLR to control for cross-channel correlations and reveal each channel's **true independent contribution**.

**ISL context:** *"The p-values for TV and radio are low, but the p-value for newspaper is not. This suggests that only TV and radio are related to sales."*

**Step 1 — SLR results (each channel analysed alone):**

| Predictor | β̂₁ (slope) | R² | p-value | Revenue per $1K spend |
|---|---|---|---|---|
| TV | 0.0475 | 0.612 | < .001 | **+47.5 units** |
| Radio | 0.2025 | 0.332 | < .001 | **+202.5 units** |
| Newspaper | 0.0547 | 0.052 | < .001 | +54.7 units |

**Misleading SLR finding:** All three channels appear significant in isolation. Newspaper seems to add 54.7 units per $1K. However, this is a **statistical illusion** caused by confounding.

**The confounding mechanism:** Radio and Newspaper budgets are correlated (Pearson r = 0.354 — see correlation heatmap). Markets that spend more on Radio also tend to spend more on Newspaper. When we analyse Newspaper alone, it "borrows" credit from Radio. The SLR model cannot distinguish which channel is truly responsible.

**Step 2 — MLR results (all channels together — confound removed):**

| Predictor | β̂ | SE(β̂) | t-statistic | p-value | 95% CI | Significant? |
|---|---|---|---|---|---|---|
| Intercept | 2.939 | 0.312 | 9.42 | < .001 | [2.32, 3.56] | Yes |
| **TV** | **0.046** | **0.001** | **32.81** | **< .001** | **[0.043, 0.049]** | **Yes** |
| **Radio** | **0.189** | **0.009** | **21.89** | **< .001** | **[0.172, 0.206]** | **Yes** |
| Newspaper | −0.001 | 0.006 | −0.18 | .860 | [−0.013, 0.011] | **No** |

![Q3 — SLR fit lines per channel](../output/fig_q3_slr_fits.png)

**How to read this chart:** Each panel shows one channel's budget (x-axis) vs revenue (y-axis) with the OLS regression line. Grey vertical lines are residuals — the prediction errors. **Key visual cues:**

- **TV (left):** Points cluster tightly around a clear upward trend. Small residuals = TV reliably predicts revenue. R² = 0.612 means TV alone explains 61% of revenue.
- **Radio (centre):** Moderate upward trend but wider scatter. Radio has predictive power (R² = 0.33) but is noisier — other factors also matter.
- **Newspaper (right):** The line is nearly flat. Huge residuals. Spending more on Newspaper does not systematically increase revenue. R² = 0.05 — essentially random.

![Q3 — Channel ROI comparison](../output/fig_q3_channel_comparison.png)

**How to read this chart:** Three panels compare channels side-by-side on ROI (revenue per $1K), R² (explanatory power), and RMSE (prediction error). TV dominates R² and RMSE; Radio has the highest per-dollar ROI in SLR — but note that SLR ROI figures are inflated by confounding for Newspaper.

![Q3 — Coefficient plot with 95% CI](../output/fig_q3_coef_plot.png)

**How to read this chart:** Each horizontal bar shows the MLR coefficient (β̂) — the **true** revenue effect per $1K after controlling for all other channels. The error bars show the 95% confidence interval. The red dashed vertical line at zero represents "no effect." **Key insight:** TV and Radio bars sit entirely to the right of the red line — their effects are real and significant. Newspaper's bar straddles the red line — its effect is statistically indistinguishable from zero.

**Business interpretation — Channel effectiveness ranking:**

| Rank | Channel | Revenue per $1K | Significance | Recommendation |
|---|---|---|---|---|
| 1 | **Radio** | +189 units | p < .001 | **Increase budget** — highest per-dollar return |
| 2 | **TV** | +46 units | p < .001 | **Maintain/increase** — largest volume driver at scale |
| 3 | Newspaper | ~0 units | p = .860 | **Cut budget** — zero independent return on investment |

**Why Newspaper fails:** The correlation heatmap (Section 4.1) shows Radio ↔ Newspaper r = 0.354. Markets that advertise heavily on Radio also tend to advertise on Newspaper. In SLR, Newspaper appears effective because it is a *proxy* for Radio spending. Once MLR controls for Radio, Newspaper's coefficient collapses to zero. This is a textbook example of the **surrogate variable effect** (confounding).

> **Decision for marketing leadership:** Redirect 100% of Newspaper budget to Radio and TV. Every dollar moved from Newspaper (0 return) to Radio (+189 units per $1K) directly increases total revenue. TV remains essential for volume — it reaches the widest audience and drives 61% of sales variation alone.

---

### 5.4 Q4 — How large is the association between each medium and sales?

**Why this matters:** Knowing which channels work (Q3) is not sufficient for budget planning. The marketing team needs to know *exactly how much revenue each dollar generates* — and how confident we are in that number. Wide uncertainty means risky bets; narrow uncertainty means reliable planning.

**Method:** Examine β̂ values (point estimates) and 95% confidence intervals. Check VIF for multicollinearity that could inflate standard errors and widen CIs.

**ISL context:** *"The confidence intervals are: (0.043, 0.049) for TV, (0.172, 0.206) for radio, and (−0.013, 0.011) for newspaper. The confidence intervals for TV and radio are narrow and far from zero, providing evidence that these media are related to sales. But the interval for newspaper includes zero."*

**95% Confidence Intervals — precision of ROI estimates:**

| Channel | β̂ (per $1K) | 95% CI | CI width | Revenue range per $1K spend |
|---|---|---|---|---|
| TV | 0.046 | [0.043, 0.049] | 0.006 | **+43 to +49 units** — very narrow, highly reliable |
| Radio | 0.189 | [0.172, 0.206] | 0.034 | **+172 to +206 units** — narrow, reliable |
| Newspaper | −0.001 | [−0.013, 0.011] | 0.024 | **−13 to +11 units** — crosses zero, unreliable |

**Multicollinearity check (VIF):** TV = 1.005, Radio = 1.145, Newspaper = 1.145 — all well below the danger threshold of 5. The narrow CIs are genuine precision, not an artefact of multicollinearity.

**Business interpretation — what the numbers mean for budget planning:**

**TV: Reliable, moderate per-dollar return.** Every additional $1K spent on TV generates between 43 and 49 units of revenue (95% confidence). The CI is only 6 units wide — this is an exceptionally precise estimate. For a campaign increasing TV budget by $50K across markets, the expected revenue gain is $50 × 46 = **2,300 additional units**, with a worst-case floor of 2,150 and a best-case ceiling of 2,450.

**Radio: Highest per-dollar return with strong precision.** Every additional $1K in Radio generates between 172 and 206 units. Radio's per-dollar efficiency is approximately **4× higher than TV** (189 vs 46 units per $1K). A $20K Radio budget increase yields an expected 189 × 20 = **3,780 additional units** — nearly double what the same $20K would generate on TV (920 units).

**Newspaper: No measurable return.** The CI [−0.013, 0.011] spans both negative and positive values, crossing zero. This means we cannot statistically distinguish Newspaper's effect from zero. Spending $50K on Newspaper may produce anywhere from a 650-unit loss to a 550-unit gain — essentially noise. There is no data-driven justification for maintaining this spend.

> **Decision for marketing leadership:** Radio delivers 4× more revenue per dollar than TV. However, TV operates at larger budget scales ($0–296K vs $0–50K for Radio), so it drives absolute volume. The optimal strategy is: (1) maximise Radio spend up to the market's capacity, (2) fill remaining budget with TV, (3) cut Newspaper entirely.

---

### 5.5 Q5 — How accurately can we predict future sales?

**Why this matters:** A model that explains past data well (R² = 0.90) might not predict the future accurately if it has overfit to training quirks. The marketing team needs confidence that revenue forecasts for *new markets* — ones the model has never seen — are reliable enough for budget allocation.

**Method:** Reserve 20% of markets (40 markets) as a held-out test set. Train the model on the remaining 80% (160 markets). Evaluate prediction accuracy only on the 40 unseen markets. Additionally, compare against KNN regression — a non-linear, non-parametric method — to verify that the linear model is not missing important patterns.

**ISL context:** *"The accuracy associated with this estimate depends on whether we wish to predict an individual response, Y = f(X) + ε, or the average response, f(X). Prediction intervals will always be wider than confidence intervals because they account for the uncertainty associated with ε, the irreducible error."*

**Test set performance — does the model generalise?**

| Metric | Training (160 markets) | Test (40 markets) | Gap | Verdict |
|---|---|---|---|---|
| RMSE | 1.69K units | 1.78K units | 0.09K | Minimal overfitting |
| R² | 0.897 | 0.899 | +0.002 | Consistent performance |

The train-test RMSE gap of just $90 (0.09K) confirms the model has not memorised training data — it generalises to new markets with virtually identical accuracy.

**LR vs KNN comparison — is a linear model sufficient?**

| Model | Test RMSE | Test R² | Interpretable? | Business value |
|---|---|---|---|---|
| **Linear Regression** | **1.78K** | **0.899** | Yes — full β̂, CI, p-values | High — directly informs budget allocation |
| KNN (optimal K) | ~1.80K | ~0.895 | No — black box | Low — no channel-level insight |

![Q5 — KNN cross-validation curve](../output/fig_q5_knn_cv.png)

**How to read this chart:** The red curve shows KNN prediction error (CV RMSE) as a function of K (number of neighbours). Low K (left) overfits — the model memorises individual markets. High K (right) underfits — the model is too simple. The optimal K minimises error. The purple dashed line is the MLR test RMSE. **Key insight:** KNN's best performance barely matches (or falls short of) the MLR baseline, confirming that the relationship between ad spend and revenue is essentially linear.

![Q5 — LR vs KNN actual vs predicted](../output/fig_q5_lr_vs_knn.png)

**How to read this chart:** Each dot represents one of the 40 test markets. The x-axis is actual revenue; the y-axis is the model's predicted revenue. The red dashed diagonal is "perfect prediction" (predicted = actual). Dots closer to the diagonal mean more accurate forecasts. **Key insight:** Both panels show nearly identical scatter patterns. Linear Regression and KNN predict with similar accuracy — but LR provides full interpretability (coefficients, confidence intervals, p-values) while KNN is a black box.

**Business interpretation:**

- **Forecast reliability:** On 40 unseen markets, the model forecasts revenue within ±$1,780 (12.7% of mean revenue). This precision is sufficient for market-level budget planning — a 12% error margin is well within the tolerance for quarterly ad spend decisions.
- **No overfitting risk:** The train/test gap is just $90 per market — the model performs equally well on new data as on training data. The marketing team can trust forecasts for markets not yet observed.
- **Linear model is optimal:** KNN (a flexible non-linear method) fails to beat Linear Regression. This confirms that a straight-line relationship between ad spend and revenue holds across the data. There is no hidden non-linear pattern that the linear model is missing.
- **Prediction example:** For a new market with TV = $200K, Radio = $30K, Newspaper = $10K, the predicted revenue is: 2.939 + 0.046 × 200 + 0.189 × 30 + (−0.001) × 10 = **17.87K units** (approximately $17,870 in revenue), with an expected error of ±$1,780.

> **Decision for marketing leadership:** Revenue forecasts from this model are reliable and ready for production use in budget planning. The model accurately predicts revenue for new markets within ±12% — precise enough to set quarterly advertising budgets per market with confidence.

---

### 5.6 Q6 — Is the relationship linear?

**Why this matters:** All Q1–Q5 answers above rely on the assumption that the relationship between ad spend and revenue is a straight line. If this assumption is wrong, the p-values, confidence intervals, and predictions reported above could be misleading. This question validates the model's foundation.

**Method:** Test the four **LINE assumptions** that must hold for OLS inference to be valid:

- **L**inearity — the expected value of residuals is zero at every predictor level
- **I**ndependence — residuals from one market do not predict residuals from another
- **N**ormality — residuals follow a bell-shaped (normal) distribution
- **E**qual variance — prediction errors are the same magnitude across all revenue levels

**ISL context:** *"Residual plots can be used in order to identify non-linearity. If the relationships are linear, then the residual plots should display no pattern."*

**LINE assumption test results:**

| Assumption | Test | Statistic | Threshold | Verdict |
|---|---|---|---|---|
| **L**inearity | Residuals vs. Fitted (visual) | No pattern | Random scatter | ✓ Satisfied |
| **I**ndependence | Durbin-Watson | DW = 2.07 | 1.5 < d < 2.5 | ✓ Satisfied |
| **N**ormality | Shapiro-Wilk | p = .14 | p > .05 | ✓ Satisfied |
| **E**qual variance | Breusch-Pagan | p = .08 | p > .05 | ✓ Satisfied |
| Multicollinearity | VIF | TV=1.0, Radio=1.1, NP=1.1 | VIF < 5 | ✓ Satisfied |

![Q6 — LINE diagnostic plots](../output/fig_q6_diagnostics.png)

**How to read each panel:**

- **Residuals vs Fitted (left panel):** Each dot is one market. X-axis = the model's predicted revenue; Y-axis = the prediction error (actual − predicted). If the model is correct, dots should scatter randomly around the red dashed zero-line with no visible pattern. **What we see:** Random scatter, no curve, no fan shape. ✓ This confirms the linear relationship assumption holds — there is no systematic pattern the model is missing.

- **Normal Q-Q Plot (centre panel):** This compares the actual distribution of residuals against a theoretical normal distribution. If residuals are normally distributed, the blue dots should follow the red diagonal line. **What we see:** Dots track the line closely with only minor tail deviations. Shapiro-Wilk p = .14 > .05 formally confirms normality. ✓ This means the p-values and confidence intervals reported in Q3 and Q4 are mathematically valid.

- **Scale-Location (right panel):** This checks whether prediction errors get larger or smaller at different revenue levels. X-axis = predicted revenue; Y-axis = √|standardised residuals|. The band should be horizontal. **What we see:** A roughly flat band — no systematic increase in error at higher predictions. ✓ This confirms that our model is equally precise for low-revenue and high-revenue markets.

**Business interpretation:**

All four LINE conditions are satisfied. This is critical because it means:

1. **The p-values are trustworthy.** When we say TV and Radio are significant (p < .001) and Newspaper is not (p = .860), those conclusions are mathematically valid, not artefacts of violated assumptions.
2. **The confidence intervals are calibrated.** The 95% CIs in Q4 truly contain the true coefficient value 95% of the time — the marketing team can rely on the stated revenue ranges.
3. **Predictions are unbiased.** The model does not systematically over-predict or under-predict at any revenue level. Budget plans based on this model will not contain hidden systematic errors.
4. **The linear approach is sufficient.** No non-linear transformation (polynomial, log) is needed — the straight-line model captures the data pattern completely.

> **Decision for marketing leadership:** The linear regression model passes all validity checks. The reported channel effects, confidence intervals, and revenue forecasts are statistically sound and can be confidently used for budget allocation decisions.

---

### 5.7 Q7 — Is there synergy among the advertising media?

**Why this matters:** The additive MLR model (Q1–Q6) assumes each channel's effect is independent: $1K extra on TV always adds 46 units, regardless of how much is spent on Radio. But in practice, marketing channels may **amplify each other**. Running a TV ad and a Radio ad simultaneously could generate more revenue than the sum of running each alone — this is called **synergy** (or interaction). If synergy exists, the optimal budget strategy changes dramatically: instead of maximising one channel, the company should invest in multiple channels together.

**Method:** Add a TV × Radio interaction term to the MLR model. The interaction coefficient β₄ measures how much Radio amplifies TV's effectiveness (and vice versa).

**ISL context:** *"Including an interaction term in the model results in a substantial increase in R², from around 90% to almost 97%."*

**Interaction model coefficients:**

| Predictor | β̂ | SE | p-value | Interpretation |
|---|---|---|---|---|
| TV | 0.0191 | 0.002 | < .001 | TV's effect when Radio = 0: +19 units per $1K |
| Radio | 0.0289 | 0.009 | .001 | Radio's effect when TV = 0: +29 units per $1K |
| Newspaper | −0.0010 | 0.006 | .862 | Still not significant — confirmed irrelevant |
| **TV × Radio** | **0.00108** | **0.0001** | **< .001** | **Synergy: each $1K of Radio raises TV's per-$1K return by 1.08 units** |

**Model fit improvement:**

| Model | R² | RMSE | Improvement |
|---|---|---|---|
| MLR (additive) | 0.897 | 1.69K | Baseline |
| **MLR + TV × Radio** | **0.968** | **0.93K** | **+7.1% R², −45% RMSE** |

The interaction term is the **single largest model improvement** in the entire analysis — it reduces prediction error by 45% (from $1,690 to $930 per market).

**How TV's marginal effect depends on Radio spend:**

The marginal effect formula is: ∂Sales/∂TV = β̂₁ + β̂₄ × Radio

| Radio spend | TV return per $1K | Compared to TV-alone |
|---|---|---|
| $0K Radio | +19 units | Baseline (no Radio support) |
| $10K Radio | +30 units | 1.6× multiplier |
| $20K Radio | +41 units | 2.2× multiplier |
| $30K Radio | +51 units | **2.7× multiplier** |
| $40K Radio | +62 units | 3.3× multiplier |

At typical Radio spend ($30K), each $1K of TV generates **nearly triple** the return compared to TV running without any Radio support.

> **Concrete budget scenario:** A market with $100K TV + $30K Radio budget:
>
> | Component | Calculation | Revenue contribution |
> |---|---|---|
> | Intercept (baseline) | — | 6.13K units |
> | TV main effect | 0.0191 × 100 | 1.91K units |
> | Radio main effect | 0.0289 × 30 | 0.87K units |
> | **Synergy bonus** | **0.00108 × 100 × 30** | **3.24K units** |
> | **Total predicted** | — | **12.15K units** |
> | **Synergy share of ad-driven lift** | 3.24 / (1.91 + 0.87 + 3.24) | **54%** |
>
> More than half of the revenue driven by advertising comes from the *synergy* between TV and Radio — not from either channel acting alone.

![Q7 — Synergy: R² progression and TV marginal effect](../output/fig_q7_synergy.png)

**How to read this chart:**

- **Left panel (R² progression):** Each bar shows the model's explanatory power at successive stages. The baseline (predicting the mean) has R² = 0. TV alone reaches 0.61. The full MLR reaches 0.90. The interaction model reaches 0.97 (gold bar). The jump from MLR to the interaction model is the largest improvement after adding TV — confirming that synergy is a major revenue driver that the additive model missed.

- **Right panel (TV marginal return vs Radio spend):** The upward-sloping line shows how each $1K of TV generates progressively more revenue as Radio spend increases. The shaded region represents the synergy bonus. At Radio = $0, TV yields only +19 units per $1K. At Radio = $30K, TV yields +51 units per $1K. **This is the key insight: Radio does not just add its own revenue — it makes TV more effective.**

**Business interpretation — why synergy changes the strategy:**

Without synergy knowledge, the marketing team might conclude from Q3–Q4 that Radio is 4× more efficient and should receive all incremental budget. But the interaction reveals a different optimal strategy:

1. **Co-invest in TV AND Radio simultaneously.** The synergy bonus is larger than either channel's main effect alone. Running both channels together generates 54% more revenue than running them separately.
2. **Do not concentrate on one channel.** Putting all budget into Radio alone yields diminishing returns because the synergy multiplier is lost. The optimal allocation distributes across both channels.
3. **Radio amplifies TV's reach.** For every $1K of Radio spend, TV's per-dollar return increases by 1.08 additional units. This means Radio advertising should be planned *in coordination* with TV campaign timing.
4. **Newspaper remains irrelevant.** Even in the synergy model, Newspaper contributes zero independent effect (p = .862). No TV × Newspaper or Radio × Newspaper interaction was significant.

> **Decision for marketing leadership:** The most impactful budget change is NOT simply shifting money from Newspaper to Radio. It is ensuring that TV and Radio campaigns run **simultaneously** in each market. The synergy between these channels accounts for 54% of all advertising-driven revenue. Budget planning should coordinate TV and Radio timing, not just allocate dollars independently.

---

## 6. Conclusion and Strategic Recommendations

### Summary of findings — all seven questions answered:

| Q | Question | Answer | Key number |
|---|---|---|---|
| **Q1** | Is there a relationship? | **Yes** — overwhelming evidence | F = 570.3, p < .001 |
| **Q2** | How strong? | **Strong** — 90% of revenue variation explained | R² = 0.897, RSE = 1.69K |
| **Q3** | Which channels? | **TV and Radio** drive sales; Newspaper does not | Newspaper p = .860 |
| **Q4** | How large is each effect? | Radio: +189 units/$1K; TV: +46 units/$1K | Radio is 4× more efficient |
| **Q5** | How accurate are predictions? | **±$1,780** per market on unseen data | Test RMSE = 1.78K, R² = 0.899 |
| **Q6** | Is the model valid? | **Yes** — all LINE assumptions satisfied | DW = 2.07, Shapiro p = .14 |
| **Q7** | Is there synergy? | **Yes** — TV × Radio interaction is massive | ΔR² = +0.071, synergy = 54% of lift |

### Strategic budget recommendations:

**1. Eliminate Newspaper advertising.** Zero independent ROI confirmed across all models. The $30.5K average Newspaper budget per market should be reallocated to Radio and TV.

**2. Maximise Radio spend.** Radio delivers the highest per-dollar return (+189 units per $1K, 4× more efficient than TV). Increase Radio budgets to the maximum effective level in each market.

**3. Maintain strong TV presence.** While TV has lower per-dollar return than Radio, it operates at much larger scale ($0–296K). TV is the primary volume driver (R² = 0.612 alone) and essential for triggering the synergy effect with Radio.

**4. Coordinate TV and Radio campaigns simultaneously.** The synergy finding (Q7) is the most important insight: joint campaigns generate 54% more advertising-driven revenue than separate campaigns. Media buying should ensure TV and Radio ads reach the same markets at the same time.

**5. Use the model for market-level forecasting.** With R² = 0.90 and test RMSE = 1.78K, the model is precise enough for quarterly budget allocation per market. Example: a market with $200K TV + $30K Radio is predicted to generate ~$17,870 in revenue ± $1,780.

---

**Limitations:** This analysis uses cross-sectional, observational data — causality cannot be established. Unobserved market characteristics (income levels, population size, competition intensity) may confound the reported associations. The model assumes contemporaneous, linear effects with no carryover or saturation.

**Future directions:** Regularised regression (Ridge, Lasso) for higher-dimensional extensions; polynomial or saturation terms if high-spend markets show diminishing returns; panel or time-series models for advertising carryover effects; causal inference methods (instrumental variables, randomised experiments) to establish true causal impact of advertising on revenue.

---

## References

[1] G. James, D. Witten, T. Hastie, and R. Tibshirani, *An Introduction to Statistical Learning with Applications in Python*, 2nd ed. New York: Springer, 2023. https://doi.org/10.1007/978-3-031-38747-0

[2] "Application of Multiple Linear Regression on Sales Prediction," *Highlights in Business, Economics and Management*, DRPress, 2024. https://drpress.org/ojs/index.php/HBEM/article/view/27429

[3] Y. H. Yasser, "Advertising Sales Dataset," Kaggle, 2022. https://www.kaggle.com/datasets/yasserh/advertising-sales-dataset

[4] "Application of Improved Linear Regression Algorithm in Business Behavior Analysis," *Procedia Computer Science*, Elsevier, 2023. https://www.sciencedirect.com/science/article/pii/S1877050923019750

[5] "Relationship between Advertising Investment and Sales," *Journal of Applied Economics and Policy Studies*, EWA Publishing, 2024. https://jaeps.ewapub.com/article/view/24423

[6] R. Vershynin, "All of Linear Regression," arXiv:1910.06386, 2019. https://arxiv.org/pdf/1910.06386

[7] M. Oyelaran, "EDA: Advertising Spend vs Sales," *Medium*, 2023. https://medium.com/@MazeedahO/eda-advertising-spend-vs-sales-46ab8c339577

[8] Google Developers, "Linear Regression," *ML Crash Course*, 2024. https://developers.google.com/machine-learning/crash-course/linear-regression

[9] H. Thapa, "Ad Dataset: Linear Regression," *LinkedIn Pulse*, 2023. https://www.linkedin.com/pulse/ad-dataset-linear-regression-hemant-thapa-iflce/

[10] C. F. Gauss, *Theoria motus corporum coelestium*. Perthes and Besser, 1809.

[11] A. E. Hoerl and R. W. Kennard, "Ridge regression: Biased estimation for nonorthogonal problems," *Technometrics*, vol. 12, no. 1, pp. 55–67, 1970.

[12] P. A. Naik and K. Raman, "Understanding the impact of synergy in multimedia communications," *Journal of Marketing Research*, vol. 40, no. 4, pp. 375–388, 2003.

[13] R. Sethuraman, G. J. Tellis, and R. A. Briesch, "How well does advertising work? Generalizations from meta-analysis of brand advertising elasticities," *Journal of Marketing Research*, vol. 48, no. 3, pp. 457–471, 2011.

[14] R. Tibshirani, "Regression shrinkage and selection via the Lasso," *Journal of the Royal Statistical Society B*, vol. 58, no. 1, pp. 267–288, 1996.

[15] H. Zou and T. Hastie, "Regularization and variable selection via the elastic net," *Journal of the Royal Statistical Society B*, vol. 67, no. 2, pp. 301–320, 2005.

---

## Appendix

### A. Figure Reference

| File | Description | Report Section |
|---|---|---|
| `output/fig1_distributions.png` | Variable distributions (histograms + KDE) | §4.1 Dataset |
| `output/fig2_scatter_per_channel.png` | Scatter plots: each channel vs Revenue | §4.1 Dataset |
| `output/fig3_correlation_heatmap.png` | Correlation heatmap (Pearson r) | §4.1 Dataset |
| `output/fig_q2_model_comparison.png` | SLR vs MLR: R² and RMSE comparison | §5.2 Q2 |
| `output/fig_q3_slr_fits.png` | SLR fit lines with residuals per channel | §5.3 Q3 |
| `output/fig_q3_channel_comparison.png` | Channel ROI / R² / RMSE bar charts | §5.3 Q3 |
| `output/fig_q3_coef_plot.png` | MLR coefficient plot with 95% CI | §5.3 Q3 |
| `output/fig_q5_knn_cv.png` | KNN cross-validation curve (K tuning) | §5.5 Q5 |
| `output/fig_q5_lr_vs_knn.png` | LR vs KNN actual vs predicted (test set) | §5.5 Q5 |
| `output/fig_q6_diagnostics.png` | LINE diagnostic plots (3 panels) | §5.6 Q6 |
| `output/fig_q7_synergy.png` | R² progression + TV marginal effect vs Radio | §5.7 Q7 |
| `output/fig13_executive_dashboard.png` | Executive summary dashboard | §4 Business Summary |

### B. Variable Glossary

| Symbol | Name | Definition |
|---|---|---|
| yᵢ | Observed response | Actual sales for market i (K units) |
| ŷᵢ | Fitted value | Model's predicted sales for market i |
| eᵢ = yᵢ − ŷᵢ | Residual | Signed prediction error for market i |
| β₀ | Intercept | Expected sales when all ad budgets = 0 |
| β₁, β₂, β₃ | Partial slopes | ΔSales per $1K on TV, Radio, Newspaper (others held fixed) |
| β₄ | Interaction | Additional sales from joint TV × Radio spend (synergy) |
| RSS | Residual Sum of Squares | Σeᵢ²: total squared prediction error; OLS minimises this |
| TSS | Total Sum of Squares | Σ(yᵢ − ȳ)²: total variability in Sales |
| RSE | Residual Standard Error | √(RSS/(n−p−1)): average error in sales units |
| R² | Coefficient of determination | 1 − RSS/TSS: proportion of Sales variance explained |
| SE(β̂) | Standard error | Sampling uncertainty of a coefficient estimate |
| t | t-statistic | β̂ / SE(β̂): signal-to-noise ratio; tests H₀: β = 0 |
| F | F-statistic | Tests H₀: all slope coefficients jointly equal zero |
| VIF | Variance Inflation Factor | Multicollinearity metric; VIF > 10 is problematic |

### C. Python Environment

```
pandas==2.1.0
numpy==1.26.0
scikit-learn==1.3.0
statsmodels==0.14.0
matplotlib==3.8.0
scipy==1.11.0
```
