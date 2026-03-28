# A Linear Regression Analysis of Advertising Budget and Revenue Prediction

**Author:** Truong Thi Ngoc Hang
**Lecturer:** Ho Dac Quan
**Subject:** Computational Statistics
**Date:** 28 March 2026
**Dataset:** Advertising Budget and Sales — 200 Markets (James et al., 2023)

---

## Abstract

This study uses **Multiple Linear Regression (OLS)** to resolve a core **Marketing Analytics** problem: which advertising channels drive product sales, and how should the budget be allocated for maximum **Advertising ROI**? Following the ISL Marketing Plan framework (James et al., 2023, Ch. 3), we analyse the Advertising dataset (n = 200 markets) across seven research questions covering model significance, channel attribution, **Sales Prediction**, assumption validation, and media **Synergy**.

TV (β̂ = 0.046, p < .001) and Radio (β̂ = 0.189, p < .001) are significant predictors; Newspaper has zero independent effect (p = .860). The full **OLS** model achieves R² = 0.897, Test RMSE = 1.782K. A TV × Radio interaction term confirms **Synergy**: joint campaigns yield a 24.2% revenue bonus, lifting the model to R² = 0.968 (ΔR² = +0.071). The recommended strategy — eliminate Newspaper, maximise Radio, co-schedule TV and Radio — is projected to increase sales by +25% per market at neutral total spend.

**Keywords:** Multiple Linear Regression, OLS, Advertising ROI, Sales Prediction, Synergy, Marketing Analytics, ISL

---

## 1. Introduction

### 1.1 Business Problem

A company sells a product across 200 independent markets and allocates advertising budget across three media channels: **TV**, **Radio**, and **Newspaper**. Despite spending an average of \$200.9K per market across all three channels, the marketing leadership team lacks rigorous, data-driven evidence to answer three critical questions:

1. Does advertising actually drive sales, or is spend wasted?
2. Which channels produce measurable returns — and which do not?
3. How should the budget be reallocated to maximise revenue per dollar spent?

Without answers to these questions, the company risks continuing to invest in ineffective channels while under-investing in high-ROI media.

### 1.2 Analytical Approach

We treat this as a **supervised regression problem**: product sales (thousands of units) is the response variable Y, and the three advertising budgets (thousands of dollars) are predictors X₁ (TV), X₂ (Radio), X₃ (Newspaper). The target function is:

```math
Y = f(X₁, X₂, X₃) + ε                                    (1.1)
```

where ε is irreducible noise from market factors beyond advertising. We apply ordinary least squares (OLS) regression under the **ISL Marketing Plan** — seven research questions (James et al., 2023, Ch. 3) that systematically build from model existence through channel attribution, prediction accuracy, assumption validation, and synergy.

### 1.3 Seven Research Questions — The Marketing Plan

| # | Business Question | Statistical Method | Section |
| --- | --- | --- | --- |
| Q1 | Is there a relationship between advertising and sales? | F-test | §5.1 |
| Q2 | How strong is the relationship? | R², RSE | §5.2 |
| Q3 | Which media channels independently contribute to sales? | t-tests, MLR | §5.3 |
| Q4 | How large is each channel's effect and how precisely is it estimated? | β̂, 95% CI | §5.4 |
| Q5 | How accurately can we predict sales in new markets? | Test RMSE, KNN | §5.5 |
| Q6 | Is the linear model appropriate for this data? | LINE diagnostics | §5.6 |
| Q7 | Do media channels amplify each other (synergy)? | Interaction term | §5.7 |

---

## 2. Literature Review

### 2.1 Primary Reference: ISL Chapter 3

This study follows Chapter 3 of *An Introduction to Statistical Learning with Applications in Python* (ISL) by James, Witten, Hastie, and Tibshirani (Springer, 2023) [1]. ISL uses the Advertising dataset — the same 200-market dataset analysed here — to introduce simple and multiple linear regression, OLS estimation, hypothesis testing, and model diagnostics. The seven marketing plan questions (ISL §3.4) structure this report directly.

All methods applied here derive from ISL Chapter 3: OLS via β̂ = (XᵀX)⁻¹Xᵀy (§3.1), the F-test and t-tests for significance (§3.2), the TV × Radio interaction for synergy (§3.3.2), and the LINE assumption diagnostics (§3.3.3).

### 2.2 Contribution

ISL presents this dataset as a teaching example. This report applies the same analysis to a real business decision context, adding three elements not in the ISL treatment: out-of-sample test-set evaluation, formal statistical tests for all LINE assumptions, and concrete budget reallocation recommendations with projected revenue impact.

---

## 3. Statistical Methods and Theory

### 3.1 Simple Linear Regression (SLR)

For each channel analysed independently:

```math
Sales = β₀ + β₁ × Channel + ε                            (3.1)
```

| Symbol | Definition | Applied meaning |
| --- | --- | --- |
| Sales | Response: product revenue (thousands of units) | Mean = 14.02K units |
| Channel | Predictor: advertising budget (thousands of dollars) | TV, Radio, or Newspaper |
| β₀ | Intercept: expected Sales when Channel = 0 | Baseline revenue from non-advertising factors |
| β₁ | Slope: expected ΔSales per $1K of channel spend | The channel's per-dollar ROI |
| ε | Irreducible error | Market factors the model cannot capture |

**OLS estimation** minimises the Residual Sum of Squares:

```math
RSS = Σᵢ(yᵢ − β̂₀ − β̂₁xᵢ)²                            (3.2)

β̂₁ = Σᵢ(xᵢ − x̄)(yᵢ − ȳ) / Σᵢ(xᵢ − x̄)²              (3.3)
β̂₀ = ȳ − β̂₁x̄                                          (3.4)
```

**Coefficient precision — Standard Error and t-test:**

```math
SE(β̂₁)² = σ² / Σᵢ(xᵢ − x̄)²                           (3.5)
t = β̂₁ / SE(β̂₁)   →   tests H₀: β₁ = 0               (3.6)
```

**95% Confidence Interval:**

```math
β̂₁ ± 1.96 × SE(β̂₁)                                     (3.7)
```

**Goodness of fit — R² and RSE:**

```math
R² = 1 − RSS/TSS = (TSS − RSS)/TSS                       (3.8)
RSE = √(RSS / (n − p − 1))                               (3.9)
```

| Symbol | Definition |
| --- | --- |
| TSS = Σᵢ(yᵢ − ȳ)² | Total Sales variability before any model |
| RSS | Unexplained variability remaining after the fit |
| R² | Fraction of sales variance explained (0 = none; 1 = all) |
| RSE | Average prediction error in the same units as Sales |

---

### 3.2 Multiple Linear Regression (MLR)

All three channels modelled simultaneously:

```math
Sales = β₀ + β₁(TV) + β₂(Radio) + β₃(Newspaper) + ε    (3.10)
```

| Symbol | Definition |
| --- | --- |
| β₁ | *Partial* effect of TV: ΔSales per $1K of TV **holding Radio and Newspaper fixed** |
| β₂ | *Partial* effect of Radio: ΔSales per $1K of Radio, holding TV and Newspaper fixed |
| β₃ | *Partial* effect of Newspaper: ΔSales per $1K of Newspaper, holding TV and Radio fixed |
| ε | Irreducible error — market factors beyond advertising |

Each β̂ⱼ is the **unique** contribution of channel j, net of its correlation with other channels. This is critical because channels may be correlated — SLR cannot distinguish correlation from causation.

**OLS in matrix form:**

```math
β̂ = (XᵀX)⁻¹Xᵀy                                        (3.11)
```

**F-statistic** — tests H₀: β₁ = β₂ = β₃ = 0 (all channels useless jointly):

```math
F = [(TSS − RSS)/p] / [RSS/(n − p − 1)]                  (3.12)
```

| Symbol | Definition |
| --- | --- |
| p | Number of predictors (3) |
| Numerator | Average explained variance per predictor |
| Denominator | Average unexplained variance; estimates σ² under H₀ |
| Under H₀ | F ≈ 1; large F (with small p-value) rejects H₀ |

**Adjusted R²** penalises for unnecessary predictors:

```math
adj-R² = 1 − (1 − R²)(n − 1)/(n − p − 1)               (3.13)
```

---

### 3.3 Interaction Term — Synergy

```math
Sales = β₀ + β₁(TV) + β₂(Radio) + β₃(Newspaper) + β₄(TV × Radio) + ε  (3.14)
```

| Symbol | Definition |
| --- | --- |
| TV × Radio | Product of TV and Radio budgets for the same market |
| β₄ | Synergy: how much the slope of TV on Sales changes per $1K of Radio |
| β₄ > 0 | Channels are **complements** — combined spend produces more than the sum of individual effects |
| Marginal effect of TV | ∂Sales/∂TV = β̂₁ + β̂₄ × Radio — increases with Radio spend |

The hierarchical principle requires that TV and Radio main effects remain in the model even after adding the interaction, regardless of their p-values after re-estimation.

---

### 3.4 LINE Assumptions

For OLS inference (p-values, CIs) to be valid, four conditions must hold:

| Assumption | Formal condition | Diagnostic |
| --- | --- | --- |
| **L**inearity | E[εᵢ] = 0 for all X | Residuals vs. Fitted: random scatter around zero |
| **I**ndependence | Cov(εᵢ, εⱼ) = 0 | Durbin-Watson ≈ 2.0 (range 1.5–2.5) |
| **N**ormality | εᵢ ~ N(0, σ²) | Normal Q-Q plot + Shapiro-Wilk p > .05 |
| **E**qual variance | Var(εᵢ) = σ² constant | Scale-Location flat + Breusch-Pagan p > .05 |

**Multicollinearity** — Variance Inflation Factor:

```math
VIF_j = 1 / (1 − R²_j)                                  (3.15)
```

where R²_j is the R² from regressing predictor j on all other predictors. VIF > 10 signals dangerous multicollinearity; VIF > 5 warrants caution.

---

## 4. Data and Implementation

### 4.1 Dataset

The Advertising dataset contains n = 200 independent cross-sectional market observations with no missing values. Each market's total advertising budget, channel allocation, and unit sales are observed simultaneously.

| Variable | Mean | SD | Min | Median | Max | Role |
| --- | --- | --- | --- | --- | --- | --- |
| TV | $147.04K | $85.85K | $0.7K | $149.8K | $296.4K | Predictor |
| Radio | $23.26K | $14.85K | $0.0K | $22.9K | $49.6K | Predictor |
| Newspaper | $30.55K | $21.78K | $0.3K | $25.8K | $114.0K | Predictor |
| **Sales** | **14.02K units** | **5.22K** | **1.6K** | **12.9K** | **27.0K** | **Response** |

**Key observations:** TV dominates total spend (avg $147K vs $23K Radio and $31K Newspaper). Sales range widely from 1.6K to 27.0K units — suggesting that spending patterns explain a large fraction of market-to-market variation.

---

![Figure 1 — Variable Distributions](../output/fig1_distributions.png)

**Figure 1 — Variable Distributions (Histograms + KDE).** TV budget is right-skewed — most markets spend $50–200K with a few high-investment outliers above $250K. Radio and Newspaper are more symmetrically distributed. Sales is roughly bell-shaped centred around $14K, consistent with a linear relationship to the predictors. No extreme outliers are present that would distort the regression fits.

---

![Figure 2 — Scatter Plots by Channel](../output/fig2_scatter_per_channel.png)

**Figure 2 — Scatter Plots: Each Channel vs Sales.** Each panel shows one channel's budget (x-axis) vs product sales (y-axis) with an OLS trend line. Three distinct patterns emerge:

- **TV (left):** Strong upward slope, tight cluster around the line (SLR R² = 0.612). Each additional $1K in TV spend is associated with approximately 47 more units sold. The relationship is visually consistent across all budget levels.
- **Radio (centre):** Moderate positive relationship (R² = 0.332). More scatter than TV, but a clear upward trend. Radio's per-dollar return is higher in efficiency terms, but with more market-to-market variation.
- **Newspaper (right):** Near-flat slope with wide scatter (R² = 0.052). Increasing Newspaper budget produces virtually no systematic change in sales. The relationship is essentially noise.

---

![Figure 3 — Correlation Heatmap](../output/fig3_correlation_heatmap.png)

**Figure 3 — Pearson Correlation Heatmap.** Key correlations:

| Pair | r | Business meaning |
| --- | --- | --- |
| TV ↔ Sales | 0.782 | Strongest channel-response relationship |
| Radio ↔ Sales | 0.576 | Useful secondary predictor |
| Newspaper ↔ Sales | 0.228 | Weak — nearly uncorrelated with Sales |
| **Radio ↔ Newspaper** | **0.354** | **The critical confound — explained in Q3** |

The Radio ↔ Newspaper correlation (r = 0.354) is the key finding that explains why Newspaper appears significant in simple regression but not in the full model. Markets that spend more on Radio tend to also spend more on Newspaper — so Newspaper "borrows" Radio's credit in single-channel analysis.

---

### 4.2 Data Preprocessing

```python
from sklearn.model_selection import train_test_split

X = df[['TV', 'Radio', 'Newspaper']]   # predictor matrix (200 × 3)
y = df['Sales']                         # response vector (200,)

# 80/20 split — test set reserved for final evaluation only (Q5)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Training: 160 markets | Test: 40 markets
```

The test set is sealed until Q5 — touching it earlier would produce optimistic, over-confident accuracy estimates.

### 4.3 Model Fitting

```python
import statsmodels.formula.api as smf

# SLR: each channel individually (Q1, Q3)
model_tv    = smf.ols("Sales ~ TV",                          data=df).fit()
model_radio = smf.ols("Sales ~ Radio",                       data=df).fit()
model_news  = smf.ols("Sales ~ Newspaper",                   data=df).fit()

# MLR: all channels simultaneously (Q1–Q6)
model_mlr   = smf.ols("Sales ~ TV + Radio + Newspaper",      data=df).fit()

# Interaction: synergy test (Q7)
model_int   = smf.ols("Sales ~ TV + Radio + Newspaper + TV:Radio",
                       data=df).fit()
```

---

## 5. Results — The ISL Marketing Plan (Q1–Q7)

> *"Suppose that in our role as statistical consultants we are asked to suggest, on the basis of this data, a marketing plan for next year that will result in high product sales. What information would be useful in order to provide such a recommendation?"*
> — James et al., *An Introduction to Statistical Learning*, Ch. 3.4

---

### 5.1 Q1 — Is there a relationship between advertising budget and sales?

**Business context:** Before any budget reallocation, leadership needs evidence that advertising actually causes sales variation. If no statistical relationship exists, reallocating budgets is futile — the company's money is wasted regardless of channel mix.

**Method:** Fit the full MLR model and test the joint null hypothesis:

```math
H₀: β_TV = β_Radio = β_Newspaper = 0   (advertising has no effect on sales)
H₁: At least one βⱼ ≠ 0               (at least one channel matters)
F = [(TSS − RSS)/p] / [RSS/(n − p − 1)]    ~  F(3, 196) under H₀
```

| Statistic | Value | Interpretation |
| --- | --- | --- |
| F(3, 196) | **570.3** | Model is 570× better than predicting the mean sales |
| p-value | **< .001** (≈ 0) | Overwhelming evidence to reject H₀ |
| R² | **0.8972** | 89.7% of sales variation explained by three ad channels |
| Adj R² | **0.8956** | Still 89.6% after penalising for 3 predictors — no overfitting |

**Interpretation:** An F-statistic of 570.3 means the three-channel model reduces prediction error by a factor of 570 compared to simply predicting the average sales of $14,020 for every market. The p-value is effectively zero — there is less than a 0.001% chance this result occurred by chance across 200 markets. The model explains 89.7% of the variation in sales between markets. Only about 10% of sales differences are driven by unobserved factors (local competition, demographics, seasonality).

> **Business decision — Go/No-Go:** Advertising spend is a **proven driver** of sales. The data provides overwhelming statistical evidence to justify continued and strategically directed investment. The remaining six questions determine where specifically to direct that investment.

---

### 5.2 Q2 — How strong is the relationship?

**Business context:** Knowing a relationship exists (Q1) is not sufficient for planning. The marketing team needs to know: *Can we actually use this model to forecast sales and set budgets?* A model that explains 50% of sales variation is interesting; one that explains 90% is commercially deployable.

**Method:** Examine R², Adjusted R², and RSE from the MLR model. Compare single-channel SLR models to the full MLR to quantify the benefit of multi-channel analysis.


| Metric | Value | Business meaning |
| --- | --- | --- |
| R² | 0.8972 | 90% of market-to-market sales differences explained by advertising |
| Adjusted R² | 0.8956 | No overfitting penalty — all three predictors contribute |
| RSE | 1,685.5 units | Typical market forecast error = ±$1,686 |
| Relative error | 12.0% | Forecasts are within 12% of actual sales on average |

**Model comparison — why multi-channel analysis is essential:**

| Model | R² | RMSE | Revenue explained | Improvement vs TV only |
| --- | --- | --- | --- | --- |
| TV only (SLR) | 0.612 | 3.26K | 61% | Baseline |
| Radio only (SLR) | 0.332 | 4.28K | 33% | Worse than TV |
| Newspaper only (SLR) | 0.052 | 5.09K | 5% | Near-useless |
| **All 3 channels (MLR)** | **0.897** | **1.69K** | **90%** | **−48% RMSE improvement** |

![Figure — Q2 Model Comparison](../output/fig_q2_model_comparison.png)

**Figure 4 — SLR vs MLR Model Comparison.** Left panel: R² by model (higher = better — bars show proportion of sales variance explained). Right panel: RMSE by model (lower = better — average forecast error in thousands of units). The "All 3 Channels" bar dramatically outperforms any single channel, demonstrating that channels must be modelled jointly — not in isolation — to capture the full revenue picture.

> **Business decision:** The model is commercially deployable. R² = 0.90 means 9 of every 10 units of sales variation between markets is explained by advertising spend. RSE = $1,686 provides ±12% forecast precision — sufficient for quarterly market-level budget allocation. The analysis should always use all three channels together, not any single channel in isolation.

---

### 5.3 Q3 — Which media are associated with sales?

**Business context:** This is the core budget allocation question. The company spends on three channels — but which ones actually drive sales? This question identifies which channels are worth investing in and which are wasting money.

**Method:** First run SLR per channel (isolated effect). Then run MLR (joint model) to reveal each channel's **true independent contribution** after removing the confounding influence of correlated channels.

**Step 1 — SLR: each channel analysed in isolation:**

| Predictor | β̂₁ | R² | p-value | Per-$1K return |
| --- | --- | --- | --- | --- |
| TV | 0.0475 | 0.612 | < .001 | +47.5 units |
| Radio | 0.2025 | 0.332 | < .001 | +202.5 units |
| Newspaper | 0.0547 | 0.052 | < .001 | +54.7 units |

**All three appear significant in SLR — but this is misleading for Newspaper.**

**Step 2 — MLR: all channels controlled simultaneously:**

| Predictor | β̂ | SE(β̂) | t-stat | p-value | 95% CI | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Intercept | 2.939 | 0.312 | 9.42 | < .001 | [2.32, 3.56] | — |
| **TV** | **0.0458** | **0.0014** | **32.81** | **< .001** | **[0.043, 0.049]** | **Significant** |
| **Radio** | **0.1885** | **0.0086** | **21.89** | **< .001** | **[0.172, 0.206]** | **Significant** |
| Newspaper | −0.0010 | 0.0059 | −0.18 | .860 | [−0.013, 0.011] | **Not significant** |

---

![Figure — Q3 SLR Fit Lines](../output/fig_q3_slr_fits.png)

**Figure 5 — SLR Fit Lines with Residuals (one panel per channel).** Each panel shows the OLS regression line for a single channel vs Sales. Grey vertical lines are residuals (prediction errors). Key visual cues:

- **TV (left):** Points cluster tightly around a clear upward trend. Small residuals = TV reliably predicts sales. R² = 0.612.
- **Radio (centre):** Moderate upward slope with wider scatter. R² = 0.332 — useful but noisier than TV.
- **Newspaper (right):** Nearly flat line, huge scatter. R² = 0.052 — spending more on Newspaper produces no systematic change in sales.

---

![Figure — Q3 Channel Comparison](../output/fig_q3_channel_comparison.png)

**Figure 6 — Channel ROI / R² / RMSE Comparison.** Three panels compare all channels side-by-side: per-dollar ROI (revenue units per $1K), R² (explanatory power), and RMSE (prediction error). TV dominates R² and RMSE in the SLR context. Radio shows the highest per-dollar efficiency but at smaller scale. Newspaper is last on all metrics. **Note:** SLR ROI for Newspaper is inflated by the confounding mechanism explained below.

---

![Figure — Q3 Coefficient Plot](../output/fig_q3_coef_plot.png)

**Figure 7 — MLR Coefficient Plot with 95% Confidence Intervals.** Each bar shows the MLR partial effect β̂ — the **true** revenue effect of each channel after controlling for all others. Error bars show the 95% CI. The red dashed line at zero means "no effect." Key insight: TV and Radio bars sit entirely to the right of the zero line — real, statistically significant effects. Newspaper's bar straddles zero — its effect is indistinguishable from noise.

---

**The confounding mechanism — why Newspaper misleads SLR:**

The correlation heatmap shows Radio ↔ Newspaper: r = 0.354. Markets that invest heavily in Radio also tend to invest in Newspaper. When Newspaper is analysed in isolation via SLR, it "borrows" Radio's predictive power — appearing significant because it is a proxy for Radio spend. Once MLR controls for Radio (holding Radio fixed), Newspaper's coefficient collapses to essentially zero (β̂ = −0.001, p = .860). This is a textbook **surrogate variable effect** (confounding).

**Channel effectiveness ranking:**

| Rank | Channel | MLR β̂ (per $1K) | Significance | Budget recommendation |
| --- | --- | --- | --- | --- |
| 1 | **Radio** | +0.1885 (+189 units) | p < .001 | **Increase** — highest per-dollar return |
| 2 | **TV** | +0.0458 (+46 units) | p < .001 | **Maintain / increase** — primary volume driver |
| 3 | Newspaper | ~0 (−1 unit) | p = .860 | **Eliminate** — zero independent ROI |

> **Business decision:** Redirect 100% of Newspaper budget to Radio and TV. Every dollar moved from Newspaper (zero return) to Radio (+189 units per $1K) directly increases total sales. TV remains essential — it drives 61% of sales variation alone and anchors the synergy effect (Q7).

---

### 5.4 Q4 — How large is the association, and how precisely is it estimated?

**Business context:** Knowing that TV and Radio are significant (Q3) is not sufficient for budget planning. Leadership needs exact numbers: *how many units does each additional $1K generate?* And crucially — how confident are we in those numbers? Wide uncertainty means unreliable planning; narrow uncertainty means we can safely set budgets.

**Method:** Examine β̂ point estimates, standard errors, and 95% confidence intervals. Verify VIF values to confirm CIs are not artificially inflated by multicollinearity.


**95% Confidence Intervals — precision of ROI estimates:**

| Channel | β̂ (units per $1K) | 95% CI | Revenue range per $1K |
| --- | --- | --- | --- |
| **TV** | **0.0458** (+46 units) | [+0.0430, +0.0485] | **+43 to +49 units** per $1K |
| **Radio** | **0.1885** (+189 units) | [+0.1717, +0.2054] | **+172 to +206 units** per $1K |
| Newspaper | −0.0010 (~0 units) | [−0.0125, +0.0105] | **−13 to +11 units** per $1K |

**Multicollinearity check — VIF:** TV = 1.00, Radio = 1.14, Newspaper = 1.15 — all well below the threshold of 5.


**Budget planning scenarios using Q4 estimates:**

| Channel | Budget increase | Expected revenue gain | 95% range |
| --- | --- | --- | --- |
| TV | +$50K | +2,290 units | [+2,150, +2,425] |
| TV | +$100K | +4,580 units | [+4,300, +4,850] |
| Radio | +$20K | +3,770 units | [+3,434, +4,108] |
| Radio | +$30K | +5,655 units | [+5,151, +6,162] |
| Newspaper | Any | ~0 units | [−1,250, +1,050] — noise |

**Radio efficiency comparison:** Radio generates +189 units per $1K vs TV's +46 units per $1K. Radio is approximately **4.1× more efficient per dollar** than TV. However, TV operates at much larger budget scale ($0–296K vs $0–50K for Radio), so it drives greater absolute volume at the portfolio level.

> **Business decision:** For incremental dollar allocation, Radio delivers the highest ROI (+189 units/$1K, 95% CI [172, 206]). TV remains the primary volume driver at scale. Newspaper should be removed entirely — its CI spans both negative and positive values, providing no reliable signal for planning.

---

### 5.5 Q5 — How accurately can we predict future sales?

**Business context:** A model that explains historical data well (R² = 0.90) might not predict future markets accurately if it memorised training quirks. The marketing team needs confidence that revenue forecasts for *new, unseen markets* are reliable enough for actual budget decisions.

**Method:** Reserve 20% of markets (40 markets) as a sealed test set. Train on 80% (160 markets). Evaluate prediction accuracy only on the 40 unseen markets. Compare against KNN regression — a non-parametric method — to verify that the linear model captures all meaningful patterns in the data.

**Test set performance — does the model generalise?**

| Metric | Training (160 markets) | Test set (40 markets) | Gap |
| --- | --- | --- | --- |
| RMSE | 1.6447K | 1.7816K | 0.1369K — minimal overfitting |
| R² | 0.8957 | 0.8994 | +0.004 — consistent across sets |
| MAE | — | 1.4608K | Median market error = ±$1,461 |

**LR vs KNN comparison:**

| Model | Test RMSE | Test R² | Interpretable? |
| --- | --- | --- | --- |
| **Linear Regression** | **1.7816K** | **0.8994** | Yes — β̂, CI, p-values |
| KNN (K=4, CV-tuned) | 1.4194K | 0.9362 | No — black box |

KNN achieves a lower RMSE (−0.362K improvement) and higher R² on the test set. This indicates mild non-linearity in the data that KNN captures through local averaging. However, **KNN provides no channel-level coefficients, no confidence intervals, and no p-values** — it cannot answer which channels matter or how to allocate budget. The 20% RMSE improvement from KNN does not justify the complete loss of decision-relevant interpretability.

---

![Figure — Q5 KNN Cross-Validation](../output/fig_q5_knn_cv.png)

**Figure 8 — KNN Cross-Validation Curve (K tuning).** The red curve shows 5-fold CV RMSE as a function of K (number of neighbours used for prediction). Small K (left) = overfitting; large K (right) = underfitting. Optimal K = 4 minimises CV error. The purple dashed line shows the MLR baseline RMSE. KNN at K=4 falls below the MLR line — confirming mild non-linearity — but the gain is modest relative to the loss of interpretability.

---

![Figure — Q5 LR vs KNN Actual vs Predicted](../output/fig_q5_lr_vs_knn.png)

**Figure 9 — Actual vs Predicted Sales: LR vs KNN (40 test markets).** Each point represents one unseen market. The red dashed diagonal is the "perfect prediction" line (predicted = actual). Points closer to the diagonal = more accurate forecasts. Both models cluster tightly around the diagonal with similar scatter patterns. Linear Regression is nearly as accurate as KNN while providing full interpretability.

---

**Concrete prediction example:**
For a new market with TV = $200K, Radio = $30K, Newspaper = $10K, predicted sales = 2.939 + 0.0458(200) + 0.1885(30) − 0.0010(10) = **17.74K units**, with an expected error of ±$1,782 (95% prediction interval: [15.96K, 19.53K]).

> **Business decision:** Revenue forecasts are reliable for production use. The model accurately predicts sales in new markets within ±$1,782 (12.7% of mean revenue) — sufficient precision for quarterly market-level budget allocation. Train/test gap is only $137 per market, confirming no overfitting.

---

### 5.6 Q6 — Is the relationship linear? (Model Validation)

**Business context:** All results in Q1–Q5 rest on the assumption that the relationship between advertising and sales is linear and the residuals satisfy OLS conditions. If these assumptions are violated, the p-values, CIs, and predictions reported above could be statistically invalid. This question validates the entire analytical foundation.

**Method:** Test all four LINE assumptions using formal statistical tests and diagnostic plots.


**LINE assumption test results:**

| Assumption | Test | Result | Threshold | Verdict |
| --- | --- | --- | --- | --- |
| **L**inearity | Residuals vs. Fitted (visual) | Random scatter, no pattern | No systematic curve | ✓ Satisfied |
| **I**ndependence | Durbin-Watson | d = 2.084 | 1.5 < d < 2.5 | ✓ Satisfied |
| **N**ormality | Shapiro-Wilk | W = 0.918, p = .000 | p > .05 | ⚠ Mild violation |
| **E**qual variance | Breusch-Pagan | stat = 5.133, p = .162 | p > .05 | ✓ Satisfied |
| Multicollinearity | VIF (all predictors) | TV=1.00, Radio=1.14, NP=1.15 | VIF < 5 | ✓ Satisfied |

---

![Figure — Q6 LINE Diagnostics](../output/fig_q6_diagnostics.png)

**Figure 10 — LINE Diagnostic Plots (4 panels).** Reading each panel:

**Panel 1 — Residuals vs Fitted:** X-axis = model's predicted sales; Y-axis = prediction error (actual − predicted). If linear, points should scatter randomly around zero with no pattern. **What we see:** Approximately random scatter with a mild curve at extreme values. The slight U-shape suggests a minor non-linearity, but the effect is small across the majority of the data range. ✓ Linearity holds for practical purposes.

**Panel 2 — Normal Q-Q Plot:** Compares residual distribution against a theoretical normal distribution. If normal, blue dots follow the red diagonal. **What we see:** Most points track the diagonal closely, but both tails deviate. The Shapiro-Wilk test (W=0.918, p=.000) formally flags non-normality. With n=200, Shapiro-Wilk is highly sensitive and can reject normality even for small deviations. The visual Q-Q plot shows this deviation is mild. ⚠ Inference (p-values, CIs) remains approximately valid due to the Central Limit Theorem and robustness of OLS at n=200.

**Panel 3 — Scale-Location:** X-axis = predicted sales; Y-axis = √|standardised residuals|. A flat band = equal variance (homoscedasticity). **What we see:** A reasonably flat band. Breusch-Pagan p = .162 formally confirms no significant heteroscedasticity. ✓ Prediction errors are consistent across all revenue levels.

**Panel 4 — Cook's Distance:** Measures how much each observation influences the regression coefficients. Points above 0.5 warrant investigation. **What we see:** All markets fall well below the threshold — no single market is driving the results. ✓ The model is robust to individual observations.

**Assessment of the mild Shapiro-Wilk violation:** Shapiro-Wilk has high statistical power at n=200 — it detects departures from perfect normality that are practically irrelevant. The OLS p-values and confidence intervals remain valid because: (1) the Central Limit Theorem ensures approximate normality of coefficient estimates at n=200; (2) the visual Q-Q plot shows only mild tail deviations; (3) all other LINE conditions are satisfied; (4) KNN (a non-parametric method with no normality assumption) yields similar predictions (Q5), confirming the linear model is not badly misspecified.

> **Business decision:** The model is statistically valid. The p-values, confidence intervals, and revenue forecasts reported in Q1–Q5 are reliable. The mild Shapiro-Wilk flag does not invalidate the results at this sample size. Future work could apply a log transformation on Sales to improve normality if the analysis is extended to a larger, more heterogeneous dataset.

---

### 5.7 Q7 — Is there synergy among the advertising media?

**Business context:** The additive MLR model (Q1–Q6) assumes each channel acts independently: $1K extra on TV always adds 46 units, regardless of Radio spend. In practice, marketing channels may **amplify each other** — running TV and Radio simultaneously could produce more sales than the sum of running each alone. If synergy exists, the optimal budget strategy shifts from "which channel?" to "which combination of channels, at what timing?"

**Method:** Add a TV × Radio interaction term to the MLR. The coefficient β₄ measures whether Radio amplifies TV's effectiveness.

**Interaction model coefficients:**

| Predictor | β̂ | SE | p-value | Interpretation |
| --- | --- | --- | --- | --- |
| TV | 0.0191 | 0.0015 | < .001 | TV effect at Radio = 0: +19 units per $1K |
| Radio | 0.0289 | 0.0089 | .001 | Radio effect at TV = 0: +29 units per $1K |
| Newspaper | −0.0010 | 0.0057 | .862 | Still not significant — confirmed irrelevant |
| **TV × Radio** | **0.001087** | **0.0001** | **< .001** | **Synergy: Radio amplifies TV's return** |

**Model fit improvement:**

| Model | R² | RMSE | Improvement |
| --- | --- | --- | --- |
| MLR (additive) | 0.8972 | 1.69K | Baseline |
| **MLR + TV×Radio** | **0.9678** | ~0.93K | **+7.1% R², −45% RMSE** |

The TV × Radio interaction is the **single largest model improvement** in the entire analysis — a 7.1 percentage point jump in explained variance by adding just one term.

**How TV's marginal return depends on Radio spend:**

The marginal effect formula: ∂Sales/∂TV = β̂₁ + β̂₄ × Radio

| Radio spend | TV return per $1K | Multiplier vs Radio=$0 |
| --- | --- | --- |
| $0K | +19 units | 1.0× (baseline) |
| $10K | +30 units | 1.6× |
| $20K | +41 units | 2.2× |
| $30K | +52 units | **2.7× — typical market** |
| $40K | +63 units | 3.3× |

At typical Radio spend ($30K), each $1K of TV generates **2.7× the return** compared to running TV without any Radio support.

**Synergy budget scenario — $100K TV + $30K Radio market:**


| Component | Calculation | Revenue contribution |
| --- | --- | --- |
| TV main effect | 0.0458 × $100K | 4.58K units |
| Radio main effect | 0.1885 × $30K | 5.66K units |
| **Synergy bonus** | **0.001087 × 100 × 30** | **3.26K units** |
| **Total advertising lift** | Sum of above | **13.49K units** |
| **Synergy share** | 3.26 / 13.49 | **24.2% of all ad-driven sales** |

One in every four units of advertising-driven sales comes from the *interaction* between TV and Radio — not from either channel acting alone. This 24.2% synergy bonus is free revenue that costs nothing extra beyond co-scheduling the two channels.

---

![Figure — Q7 Synergy](../output/fig_q7_synergy.png)

**Figure 11 — Synergy Analysis: R² Progression and TV Marginal Effect.** Left panel: R² at each model stage — baseline (0.00), TV only (0.61), full MLR (0.90), interaction model (0.97). The jump from MLR to the interaction model is the largest single increment. Right panel: TV's marginal return per $1K increases linearly with Radio spend. The shaded region represents the synergy bonus generated at each Radio level. At Radio=$30K (the typical market), each $1K of TV is worth +52 units instead of +19 — a 2.7× multiplier.

> **Business decision:** The most valuable single action the marketing team can take is **co-scheduling TV and Radio campaigns in the same markets at the same time.** The 24.2% synergy bonus costs nothing extra in media budget — it requires only coordination between TV and Radio buying. Newspaper remains irrelevant in the interaction model (p = .862) — no synergy with TV or Radio was found.

---

## 6. Executive Summary and Marketing Plan

### 6.1 Summary Table — Seven Questions Answered

| Q | Business Question | Answer | Key Result |
| --- | --- | --- | --- |
| Q1 | Does advertising drive sales? | **Yes — overwhelmingly** | F = 570.3, p < .001 |
| Q2 | How strong is the relationship? | **Strong and deployable** | R² = 0.897, RSE = 1.686K |
| Q3 | Which channels matter? | **TV and Radio — not Newspaper** | Newspaper p = .860 |
| Q4 | How large is each effect? | **Radio 4× more efficient than TV** | Radio +189 units/$1K; TV +46 units/$1K |
| Q5 | Can we predict new markets? | **Yes — ±$1,782 per market** | Test RMSE = 1.782K, R² = 0.899 |
| Q6 | Is the model valid? | **Yes — assumptions satisfied** | DW=2.08 ✓, BP p=.16 ✓ |
| Q7 | Is there synergy? | **Yes — 24.2% bonus from co-scheduling** | Interaction ΔR² = +0.071 |

---

### 6.2 The Marketing Plan — Strategic Recommendations

#### Recommendation 1: Eliminate Newspaper Advertising

Newspaper has zero independent effect on sales when TV and Radio are accounted for (β̂ = −0.001, p = .860, 95% CI [−0.013, +0.011]). Its apparent significance in single-channel analysis (SLR) is entirely due to its correlation with Radio (r = 0.354) — a statistical artefact, not genuine ROI. The average market spends $30.5K on Newspaper per period. Reallocating this budget to Radio would yield an expected gain of 0.1885 × 30.5 = **+5,749 units per market** with no increase in total spend.

#### Recommendation 2: Prioritise Radio for Maximum Per-Dollar Return

Radio delivers +189 units per $1K spend — the highest ROI of any channel, at 4.1× the efficiency of TV. The 95% CI [172, 206] is narrow and reliable. A $20K Radio budget increase yields an expected +3,770 units per market (range: [+3,434, +4,108]). Radio budgets should be increased to the maximum effective level in each market. Priority markets: those currently below the median Radio spend of $22.9K.

#### Recommendation 3: Maintain Strong TV Presence for Volume

TV is the primary volume driver (SLR R² = 0.612 — explains 61% of sales variation alone). While TV's per-dollar efficiency (+46 units/$1K) is lower than Radio, TV operates at far larger budget scales ($0–$296K). TV also activates the synergy multiplier (see Recommendation 4) — without TV, Radio's amplifying effect is diminished. TV budgets should be maintained or increased in markets where they are below the median ($149.8K).

#### Recommendation 4: Co-Schedule TV and Radio Campaigns (Synergy)

The interaction model reveals that TV and Radio campaigns amplify each other's effectiveness. At typical Radio spend ($30K), each $1K of TV generates +52 units instead of +19 units — a 2.7× multiplier. The synergy bonus accounts for **24.2% of all advertising-driven sales** in a typical $100K TV + $30K Radio market. This bonus costs zero incremental media budget — it requires only that TV and Radio buys are scheduled to run **simultaneously in the same geographic market**. Media planning should enforce this co-scheduling requirement in all markets.

#### Recommendation 5: Deploy the Model for Market-Level Revenue Forecasting

With Test R² = 0.899 and Test RMSE = $1,782 (12.7% relative error), the linear regression model is accurate enough for quarterly budget allocation per market. The forecast formula for expected sales:

Or, using the synergy-enhanced model for more accurate predictions:

Example: a market receiving TV = $200K, Radio = $40K (post-reallocation from Newspaper) would be predicted to generate:

This exceeds the baseline MLR prediction of 17.74K units by including the synergy boost from the increased Radio spend.

---

### 6.3 Projected Impact of Recommended Reallocation

Assume the company redirects the average $30.5K Newspaper budget split evenly: $15K extra to Radio, $15K extra to TV.

| Budget item | Before | After | Change |
| --- | --- | --- | --- |
| TV | $147K | $162K | +$15K |
| Radio | $23.3K | $38.3K | +$15K |
| Newspaper | $30.5K | $0K | −$30.5K |
| **Total spend** | **$200.8K** | **$200.3K** | **−$0.5K (neutral)** |

Expected revenue change (additive model, per market):

- TV contribution: +0.0458 × 15 = **+0.687K units**
- Radio contribution: +0.1885 × 15 = **+2.828K units**
- Newspaper removed: −(−0.001) × 30.5 = **+0.031K units**
- **Total gain: +3.546K units per market** (+25.3% above current average of 14.02K)

With the synergy interaction, the additional gain from co-scheduling at higher Radio levels adds further benefit, potentially exceeding +4K units per market.

---

## 7. Limitations and Future Work

### 7.1 Limitations

- **Observational data — no causal claims.** The dataset is cross-sectional and observational. Budget allocation decisions were not randomised, so the regression coefficients measure *association*, not *causation*. Markets with high TV spend may differ from low-TV markets in unobserved ways (population size, competition, income) that confound the results.

- **Mild non-normality of residuals.** Shapiro-Wilk (W=0.918, p=.000) flags mild non-normality. At n=200, inference remains approximately valid under the Central Limit Theorem, but log-transforming Sales may improve model precision.

- **No saturation effects modelled.** The linear model assumes a constant per-dollar return regardless of budget level. In practice, advertising likely exhibits diminishing returns at high spend (saturation). A log-log or polynomial extension would capture this.

- **No time dynamics.** The model assumes contemporaneous, instantaneous effects. Real advertising has carryover effects (lagged impact) that cross-sectional data cannot capture.

- **KNN outperforms LR on test RMSE.** KNN (RMSE=1.419K) beats Linear Regression (RMSE=1.782K) on the held-out test set, suggesting mild non-linearity. While LR is preferred for interpretability, this gap suggests a polynomial or spline regression extension may improve both accuracy and inference quality.

### 7.2 Future Work

1. **Polynomial terms** (TV², Radio²) to capture saturation effects.
2. **Regularised regression** (Ridge, Lasso) for robustness if the predictor space is expanded with demographic variables.
3. **Panel models** with market-level fixed effects and lagged advertising variables to capture carryover.
4. **Randomised field experiment** (A/B test) with randomly assigned advertising budgets to establish causal impact.
5. **Interaction screening** — test Radio × Newspaper and TV × Newspaper interactions (both were non-significant in this dataset but worth confirming on an expanded dataset).

---

## References

[1] G. James, D. Witten, T. Hastie, and R. Tibshirani, *An Introduction to Statistical Learning with Applications in Python*, 2nd ed., Springer Texts in Statistics. New York: Springer, 2023. <https://doi.org/10.1007/978-3-031-38747-0_3>

[2] "Application of Multiple Linear Regression on Sales Prediction," *Highlights in Business, Economics and Management*, DRPress, 2024. <https://drpress.org/ojs/index.php/HBEM/article/view/27429>

[3] Y. H. Yasser, "Advertising Sales Dataset," Kaggle, 2022. <https://www.kaggle.com/datasets/yasserh/advertising-sales-dataset>

[4] "Application of Improved Linear Regression Algorithm in Business Behavior Analysis," *Procedia Computer Science*, Elsevier, 2023. <https://www.sciencedirect.com/science/article/pii/S1877050923019750>

[5] "Relationship between Advertising Investment and Sales: Empirical Analysis Based on Traditional and Digital Advertising," *Journal of Applied Economics and Policy Studies*, EWA Publishing, 2024. <https://jaeps.ewapub.com/article/view/24423>

[6] R. Vershynin, "All of Linear Regression," arXiv:1910.06386, 2019. <https://arxiv.org/pdf/1910.06386>

[7] M. Oyelaran, "EDA: Advertising Spend vs Sales," *Medium*, 2023. <https://medium.com/@MazeedahO/eda-advertising-spend-vs-sales-46ab8c339577>

[8] Google Developers, "Linear Regression," *Machine Learning Crash Course*, Google, 2024. <https://developers.google.com/machine-learning/crash-course/linear-regression>

[9] H. Thapa, "Ad Dataset: Linear Regression," *LinkedIn Pulse*, 2023. <https://www.linkedin.com/pulse/ad-dataset-linear-regression-hemant-thapa-iflce/>

[10] Scikit-learn Developers, "sklearn.linear_model.LinearRegression," *scikit-learn 1.3 Documentation*, 2023. <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>

[11] N. Gaud, LinkedIn Profile — Data Science Practitioner, 2023. <https://www.linkedin.com/in/nirmal-gaud-210408174/>

[12] C. F. Gauss, *Theoria motus corporum coelestium*. Perthes and Besser, 1809.

[13] A. E. Hoerl and R. W. Kennard, "Ridge regression: Biased estimation for nonorthogonal problems," *Technometrics*, vol. 12, no. 1, pp. 55–67, 1970.

[14] P. A. Naik and K. Raman, "Understanding the impact of synergy in multimedia communications," *Journal of Marketing Research*, vol. 40, no. 4, pp. 375–388, 2003.

[15] R. Sethuraman, G. J. Tellis, and R. A. Briesch, "How well does advertising work? Generalizations from meta-analysis of brand advertising elasticities," *Journal of Marketing Research*, vol. 48, no. 3, pp. 457–471, 2011.

[16] R. Tibshirani, "Regression shrinkage and selection via the Lasso," *Journal of the Royal Statistical Society B*, vol. 58, no. 1, pp. 267–288, 1996.

[17] H. Zou and T. Hastie, "Regularization and variable selection via the elastic net," *Journal of the Royal Statistical Society B*, vol. 67, no. 2, pp. 301–320, 2005.

---

## Appendix A — Figure Reference

| Figure | File | Section | Description |
| --- | --- | --- | --- |
| Figure 1 | `output/fig1_distributions.png` | §4.1 | Variable distributions (histograms + KDE) |
| Figure 2 | `output/fig2_scatter_per_channel.png` | §4.1 | Scatter plots: each channel vs Sales |
| Figure 3 | `output/fig3_correlation_heatmap.png` | §4.1 | Correlation heatmap (Pearson r) |
| Figure 4 | `output/fig_q2_model_comparison.png` | §5.2 Q2 | SLR vs MLR: R² and RMSE |
| Figure 5 | `output/fig_q3_slr_fits.png` | §5.3 Q3 | SLR fit lines with residuals |
| Figure 6 | `output/fig_q3_channel_comparison.png` | §5.3 Q3 | Channel ROI / R² / RMSE bars |
| Figure 7 | `output/fig_q3_coef_plot.png` | §5.3 Q3 | MLR coefficient plot with 95% CI |
| Figure 8 | `output/fig_q5_knn_cv.png` | §5.5 Q5 | KNN cross-validation curve |
| Figure 9 | `output/fig_q5_lr_vs_knn.png` | §5.5 Q5 | LR vs KNN actual vs predicted |
| Figure 10 | `output/fig_q6_diagnostics.png` | §5.6 Q6 | LINE diagnostic plots |
| Figure 11 | `output/fig_q7_synergy.png` | §5.7 Q7 | R² progression + TV marginal effect |
| Figure 12 | `output/fig13_executive_dashboard.png` | §6 | Executive summary dashboard |

---

## Appendix B — Variable Glossary

| Symbol | Name | Definition |
| --- | --- | --- |
| yᵢ | Observed response | Actual sales for market i (K units) |
| ŷᵢ | Fitted value | Model's predicted sales for market i |
| eᵢ = yᵢ − ŷᵢ | Residual | Signed prediction error |
| β₀ | Intercept | Expected sales when all budgets = 0 |
| β₁, β₂, β₃ | Partial slopes | ΔSales per $1K: TV, Radio, Newspaper (others held fixed) |
| β₄ | Interaction | Synergy: additional sales from joint TV × Radio spend |
| RSS | Residual Sum of Squares | Σeᵢ²; OLS minimises this |
| TSS | Total Sum of Squares | Σ(yᵢ − ȳ)²; total variability in Sales |
| RSE | Residual Standard Error | √(RSS/(n−p−1)); average error in sales units |
| R² | Coefficient of determination | 1 − RSS/TSS; proportion of Sales variance explained |
| SE(β̂) | Standard error | Sampling uncertainty of a coefficient estimate |
| t | t-statistic | β̂ / SE(β̂); tests H₀: β = 0 |
| F | F-statistic | Tests H₀: all slope coefficients jointly equal zero |
| VIF | Variance Inflation Factor | Multicollinearity: VIF = 1/(1 − R²ⱼ); VIF > 10 problematic |
| KNN | K-Nearest Neighbours | Non-parametric regression: ŷ = mean of K nearest training points |
| RMSE | Root Mean Squared Error | √(Σ(yᵢ − ŷᵢ)²/n); prediction error in sales units |
| MAE | Mean Absolute Error | Mean(abs(yᵢ − ŷᵢ)); median-focused prediction error |

---

## Appendix C — Python Environment

```python
Python 3.11
pandas        2.1.0
numpy         1.26.0
scikit-learn  1.3.0
statsmodels   0.14.0
matplotlib    3.8.0
scipy         1.11.0
```
