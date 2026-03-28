---
name: research-paper
description: >
  Use this skill whenever the user wants to write, structure, draft, complete, or review
  an academic research paper, journal article, conference paper, or technical report —
  especially for machine learning, statistical learning, or data science topics.
  Triggers include: "write a paper", "write my research", "draft abstract", "related works",
  "write introduction", "paper structure", "academic writing", "IEEE / APA / Springer format",
  "conclusion for my paper", "literature review", or any section name (Abstract, Introduction,
  Methodology, Results, Conclusion). Also trigger when the user uploads a partial draft and
  asks to complete or improve it. ALWAYS apply for ML/statistical papers even phrased as
  "write a report", "write a summary", or "fill in my skeleton".
  For the Advertising Sales / Linear Regression topic: use the canonical ISL reference
  (James et al., 2023) as the primary source and the provided reference list in
  references/isl-advertising-refs.md for related works.
---

# Research Paper Writing Skill
## Focused on: ISL Ch.3 · Linear Regression · Advertising Sales Dataset

Read `references/isl-advertising-refs.md` when writing Related Works for this topic.
Read `references/statistics.md` for all statistical formulas and diagnostics.

---

## Paper Structure (Springer / IEEE style)

```
Title
Author(s) & Affiliation
Abstract          ← 150–250 words, no citations
Keywords          ← 4–8 terms, alphabetical, semicolons
1. Introduction
2. Related Works
3. Methodology
   3.1 Dataset
   3.2 Data Preprocessing
   3.3 Model Formulation (Simple LR → Multiple LR → Extensions)
   3.4 Assumptions (LINE)
   3.5 Evaluation Metrics
4. Implementation & Experiments
   4.1 Tools & Environment
   4.2 EDA & Visualisation
   4.3 Model Training
   4.4 Results
5. Discussion
6. Conclusion
References
Appendix (optional)
```

---

## Section Rules

### Abstract  *(150–250 words, ONE paragraph, NO citations)*

**5-sentence mandatory structure:**

| Sentence | Content |
|---|---|
| 1 | Research question — what problem does this study address? |
| 2 | Motivation — why does it matter / what gap does it fill? |
| 3 | Method — how was the study conducted? (dataset + algorithm) |
| 4 | Key quantitative findings (R², RMSE, significant predictors) |
| 5 | Significance / practical implication / recommendation |

**Template for Advertising/LR papers:**
```
This study investigates the relationship between advertising expenditure across TV,
radio, and newspaper media and product sales using [simple / multiple] linear
regression applied to the Advertising Sales Dataset (n = 200 markets, Kaggle /
James et al., 2023). Accurate sales prediction enables data-driven marketing budget
allocation, yet prior work rarely validates OLS assumptions or reports out-of-sample
accuracy simultaneously. Ordinary least squares (OLS) regression was implemented in
Python (scikit-learn, statsmodels) with an 80/20 train–test split; all four LINE
assumptions were verified via residual diagnostics. The final model achieved
R² = [X], RMSE = [X] thousand units, with TV (β̂ = [X], p < .001) and radio
(β̂ = [X], p < .001) identified as significant predictors; newspaper was not
significant (p = [X]). These findings support reallocation of advertising budgets
toward TV and radio channels and confirm the utility of linear regression as a
transparent, interpretable baseline for sales forecasting.
```

---

### Keywords

- **Alphabetical order**, separated by semicolons.
- 4–8 terms. No title words. No vague terms like "study" or "analysis".
- Abbreviations only if universally known.

**Standard set for this topic:**
```
Advertising Budget; Linear Regression; OLS; Python; Sales Prediction;
Statistical Learning; Supervised Learning
```

---

### 1. Introduction

**Mandatory 6-part structure:**

1. **Hook / motivation** (1–2 sentences): why does sales prediction matter?
2. **Background** (2–3 sentences): linear regression as the foundational supervised
   learning method (cite James et al., 2023 here).
3. **Problem statement** (1–2 sentences): the 7 research questions from ISL Ch.3
   (relationship? strength? which media? how large? accuracy? linearity? synergy?).
4. **Proposed approach** (1 sentence): dataset + method.
5. **Contributions** (2–4 bullets): what is novel or thorough about this work.
6. **Paper organisation** (1 sentence): "The remainder of this paper is organised
   as follows: Section 2 reviews…"

**ISL hook to use or adapt:**
> Linear regression is a very simple approach for supervised learning that is
> useful for predicting a quantitative response. It serves as a good jumping-off
> point for newer approaches: many fancy statistical learning methods can be seen
> as generalisations or extensions of linear regression (James et al., 2023).

**The 7 ISL research questions** (address all in the Introduction):
1. Is there a relationship between advertising budget and sales?
2. How strong is the relationship?
3. Which media are associated with sales?
4. How large is the association between each medium and sales?
5. How accurately can we predict future sales?
6. Is the relationship linear?
7. Is there synergy among the advertising media (interaction effect)?

---

### 2. Related Works

**Format per paper (strictly follow):**

```
The study [N] by FirstAuthor[, SecondAuthor] et al. proposes [method] for [task],
achieving [result/finding]. However, [specific limitation of that work].
```

**Author citation rules:**
- 1 author:  "Smith [N]"
- 2 authors: "Smith and Jones [N]"
- 3+ authors: "Smith, Jones et al. [N]"  ← always use "et al." for 3 or more

**Final paragraph (REQUIRED):**
Synthesise ALL limitations from the papers above into one paragraph, then
explicitly state how the present work is novel and superior.

```
The above studies collectively [common finding/theme]. However, they share
limitations including [L1], [L2], and [L3]. The present work addresses these
gaps by [novelty 1] and [novelty 2], thereby [advantage].
```

**Reference entries for this paper** — see `references/isl-advertising-refs.md`
for pre-written summary + limitation for each of the 9 provided references.

---

### 3. Methodology

#### 3.1 Dataset

**Always describe:**
- Source: Kaggle (yasserh/advertising-sales-dataset) — 200 rows, 4 columns.
- Original source: James et al. (2023), ISL Appendix.
- Variables: TV (continuous, $0–$296K), Radio ($0–$50K), Newspaper ($0–$114K),
  Sales (thousands of units, response variable).
- No missing values; all variables continuous.

#### 3.2 Data Preprocessing

State steps explicitly:
1. Load CSV, verify shape and null counts.
2. Descriptive statistics (mean, SD, min/max per column).
3. Outlier check (IQR / boxplot).
4. Feature scaling (StandardScaler) — fit on train only.
5. Train/test split: 80% train, 20% test, `random_state=42`.

#### 3.3 Model Formulation

**Number every equation.** Use the exact ISL notation.

Simple linear regression (one predictor):
```
Sales ≈ β₀ + β₁ × TV                          (3.1)
ŷ = β̂₀ + β̂₁x                               (3.2)
```

OLS estimation (minimise RSS):
```
RSS = Σᵢ (yᵢ − β̂₀ − β̂₁xᵢ)²               (3.3)
β̂₁ = Σ(xᵢ−x̄)(yᵢ−ȳ) / Σ(xᵢ−x̄)²          (3.4)
β̂₀ = ȳ − β̂₁x̄                              (3.5)
```

Multiple linear regression (full model):
```
Sales = β₀ + β₁(TV) + β₂(Radio) + β₃(Newspaper) + ε   (3.6)
β̂ = (XᵀX)⁻¹Xᵀy                             (3.7)
```

Interaction model (synergy, Q7):
```
Sales = β₀ + β₁(TV) + β₂(Radio) + β₃(Newspaper)
        + β₄(TV × Radio) + ε                  (3.8)
```

#### 3.4 LINE Assumptions

Present as a table — see `references/statistics.md` for full table.

| | Assumption | Diagnostic |
|---|---|---|
| L | Linearity | Residuals vs. fitted plot |
| I | Independence | Durbin-Watson test |
| N | Normality | Q-Q plot, Shapiro-Wilk |
| E | Equal variance | Breusch-Pagan, scale-location plot |

Also check: VIF < 5 (multicollinearity), Cook's D < 4/n (influential points).

#### 3.5 Evaluation Metrics

Define every metric with its formula:
```
R²      = 1 − RSS/TSS                           (3.9)
adj-R²  = 1 − (1−R²)(n−1)/(n−p−1)             (3.10)
RSE     = √(RSS / (n−p−1))                      (3.11)
RMSE    = √[(1/n) Σ(yᵢ − ŷᵢ)²]                (3.12)
MAE     = (1/n) Σ|yᵢ − ŷᵢ|                     (3.13)
```

---

### 4. Implementation & Experiments

#### 4.1 Tools & Environment

State explicitly:
- Python 3.11+
- pandas, numpy, matplotlib, seaborn (EDA + visualisation)
- statsmodels (OLS inference: coefficients, p-values, F-statistic, CIs)
- scikit-learn (preprocessing, train/test split, cross-validation, RMSE)
- Dataset: Kaggle `yasserh/advertising-sales-dataset`

#### 4.2 EDA & Visualisation

**Required figures (all must appear in Results):**
1. Pairplot or scatter matrix (sales vs each predictor).
2. Correlation heatmap.
3. Distribution histograms for all 4 variables.
4. Boxplots (outlier detection).

#### 4.3 Model Training

Present training in this order:
1. Simple LR on TV alone (baseline / first model).
2. Simple LR on Radio, then Newspaper (comparison).
3. Multiple LR (all 3 predictors).
4. Multiple LR + TV×Radio interaction term.

#### 4.4 Results

**Required tables:**

*Table 1 — Simple regression per predictor:*
| Predictor | β̂₀ | β̂₁ | R² | p(β̂₁) |
|---|---|---|---|---|

*Table 2 — Multiple regression coefficient table:*
| Predictor | β̂ | SE | t | p-value | 95% CI |
|---|---|---|---|---|---|

*Table 3 — Model comparison:*
| Model | R² | adj-R² | RMSE | F-stat | p |
|---|---|---|---|---|---|
Bold best value per column.

**Required diagnostic figures:**
1. Residuals vs. fitted (linearity + homoscedasticity).
2. Normal Q-Q plot.
3. Scale-location plot.
4. Cook's distance plot.

---

### 5. Discussion

Structure:
1. Interpret each β̂ in plain, domain-relevant language (per $1K spent on TV → X units).
2. Connect results back to each of the 7 ISL research questions.
3. Explain why newspaper is not significant (correlation with radio confound).
4. Interpret the interaction term (synergy between TV and radio).
5. Limitations: observational design (no causality), single period, omitted variables.

---

### 6. Conclusion

**Strict 6-step structure — never skip any:**

| Step | Content |
|---|---|
| 1 | Restate the research problem (1 sentence) |
| 2 | Restate the method used (1 sentence) |
| 3 | Key quantitative findings (2–3 sentences with metrics) |
| 4 | Hypothesis verdict — explicitly accept/reject H₀ for each predictor |
| 5 | Limitations (1–2 sentences, honest) |
| 6 | Future work (2–3 concrete directions, not vague) |

**Rules:**
- No new results or citations in the conclusion.
- Must answer all 7 ISL questions implicitly.
- Future work must be concrete: "Ridge/Lasso regularisation", "time-series
  extension", "inclusion of digital advertising spend", NOT "more research is needed".

**Template:**
```
This study investigated whether TV, radio, and newspaper advertising budgets
predict product sales using [simple / multiple] OLS linear regression applied
to the Advertising Sales Dataset (n = 200 markets; James et al., 2023; Kaggle).
The overall model was highly significant (F([df1],[df2]) = [X], p < .001),
explaining [X]% of variance in sales (R² = [X], adj-R² = [X]). TV
(β̂ = [X], p < .001) and radio (β̂ = [X], p < .001) were strong positive
predictors, confirming H₁ for both; newspaper was not significant (p = [X]),
supporting H₀ for that predictor. A TV × Radio interaction term confirmed
synergistic effects (ΔR² = [X], p < .001), answering ISL Q7. Test-set
accuracy was RMSE = [X] thousand units. Limitations include the observational,
cross-sectional design — no causal inference can be drawn — and the absence of
digital advertising channels. Future work should apply Lasso regularisation to
extend variable selection, incorporate longitudinal data to capture advertising
carryover effects, and include digital spend (social media, search) as additional
predictors.
```

---

### References (IEEE numbered format)

Always include all 9 provided references. See `references/isl-advertising-refs.md`
for pre-formatted IEEE citation strings and the related-works paragraph for each.

---

## Paper Quality Checklist

- [ ] Abstract 150–250 words, no citations, 5-sentence structure
- [ ] Keywords alphabetical, 4–8, no title words
- [ ] Introduction addresses all 7 ISL questions
- [ ] Every related-work entry has: summary + limitation
- [ ] Related-works final paragraph states novelty
- [ ] All equations numbered (3.1), (3.2) …
- [ ] Results tables: baseline row + bold best value
- [ ] All 4 diagnostic figures present
- [ ] Conclusion follows 6-step structure, no new citations
- [ ] All 9 references cited in-text and in reference list
