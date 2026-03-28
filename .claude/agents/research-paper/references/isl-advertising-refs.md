# ISL Advertising References
## Pre-written Related Works entries + IEEE citation strings

Use these entries verbatim or lightly adapted in Section 2 (Related Works).
Cite as [1]…[9] in IEEE order of appearance.

---

## [1] Primary ISL Reference

**IEEE citation:**
> G. James, D. Witten, T. Hastie, and R. Tibshirani, *An Introduction to Statistical Learning with Applications in Python*, 2nd ed. New York, NY: Springer, 2023. https://doi.org/10.1007/978-3-031-38747-0_3

**Related works paragraph:**
> The foundational text [1] by James, Witten et al. introduces linear regression as the
> canonical supervised learning method for quantitative response prediction, using the
> Advertising dataset (TV, radio, newspaper → sales, n = 200) to illustrate all aspects
> of OLS estimation, inference, and model diagnostics. The book formally poses seven
> research questions covering relationship existence, effect magnitude, predictor selection,
> prediction accuracy, linearity, and interaction effects. However, the treatment is
> primarily pedagogical and does not include out-of-sample evaluation on a held-out
> test set, nor does it apply regularised regression (Ridge, Lasso) to this dataset.

---

## [2] Multiple Linear Regression on Sales Prediction

**IEEE citation:**
> "Application of Multiple Linear Regression on Sales Prediction," *Highlights in Business, Economics and Management*, DRPress, 2024. [Online]. Available: https://drpress.org/ojs/index.php/HBEM/article/view/27429

**Related works paragraph:**
> The study [2] applies multiple linear regression to a sales prediction problem,
> demonstrating that model accuracy improves substantially when multiple advertising
> channels are considered simultaneously rather than in isolation. The work confirms
> that OLS achieves acceptable RMSE on tabular marketing data. However, the study
> does not report confidence intervals on individual coefficients or verify the LINE
> assumptions through formal diagnostic tests, limiting the statistical rigour of
> its conclusions.

---

## [3] Advertising Sales Dataset (Kaggle)

**IEEE citation:**
> Y. H. Yasser, "Advertising Sales Dataset," Kaggle, 2022. [Online]. Available: https://www.kaggle.com/datasets/yasserh/advertising-sales-dataset

**Related works paragraph:**
> The Advertising Sales Dataset [3], publicly available on Kaggle, is a reproduction
> of the dataset used throughout ISL Chapter 3, containing 200 market-level observations
> across three advertising media (TV, radio, newspaper) and product sales. It has been
> widely used in community notebooks and tutorials for regression benchmarking. However,
> the dataset covers a single, unspecified time period and product category, precluding
> generalisation to other industries or longitudinal analysis of carryover advertising
> effects.

---

## [4] Improved Linear Regression in Business Behaviour Analysis

**IEEE citation:**
> "Application of Improved Linear Regression Algorithm in Business Behavior Analysis," *Procedia Computer Science*, vol. [X], Elsevier, 2023. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S1877050923019750

**Related works paragraph:**
> The study [4] proposes an improved linear regression formulation for business
> behaviour analysis, incorporating feature transformation and outlier handling to
> increase predictive stability. Results show measurable improvement over vanilla
> OLS on real-world business transaction data. Nonetheless, the proposed approach
> increases model complexity without providing interpretable coefficient-level
> inference, making it harder to translate findings into actionable marketing
> recommendations compared to standard OLS.

---

## [5] Advertising Investment and Sales — Traditional vs Digital

**IEEE citation:**
> "Relationship between Advertising Investment and Sales: Empirical Analysis Based on Traditional and Digital Advertising," *Journal of Applied Economics and Policy Studies*, EWA Publishing, 2024. [Online]. Available: https://jaeps.ewapub.com/article/view/24423

**Related works paragraph:**
> The research [5] empirically analyses the differential impact of traditional media
> (TV, print) versus digital advertising (social media, search) on sales across
> multiple product categories. The study finds that digital channels increasingly
> dominate sales response in consumer goods markets. However, its regression models
> do not report standard errors or test for multicollinearity between digital and
> traditional spend variables, which are typically highly correlated in modern
> marketing mixes.

---

## [6] All of Linear Regression (arXiv)

**IEEE citation:**
> R. Vershynin, "All of Linear Regression," arXiv preprint arXiv:1910.06386, 2019. [Online]. Available: https://arxiv.org/pdf/1910.06386

**Related works paragraph:**
> The theoretical monograph [6] provides a unified statistical and probabilistic
> treatment of linear regression, covering OLS, ridge, and high-dimensional
> regression under a single mathematical framework. It rigorously characterises the
> conditions under which OLS estimates are consistent and normally distributed.
> While the work is mathematically comprehensive, it does not include applied
> case studies or data-driven examples, limiting its direct usability as a
> practical guide for practitioners implementing regression on real datasets.

---

## [7] EDA — Advertising Spend vs Sales (Medium)

**IEEE citation:**
> M. Oyelaran, "EDA: Advertising Spend vs Sales," *Medium*, 2023. [Online]. Available: https://medium.com/@MazeedahO/eda-advertising-spend-vs-sales-46ab8c339577

**Related works paragraph:**
> The notebook [7] presents an exploratory data analysis of advertising spend
> against sales using the same Advertising dataset, producing pairplots, correlation
> heatmaps, and regression scatter plots in Python. The visualisation demonstrates
> a strong positive relationship between TV spend and sales. However, the analysis
> stops at EDA and does not proceed to formal OLS model fitting, assumption checking,
> or predictive evaluation on a held-out test set.

---

## [8] Google Machine Learning Crash Course — Linear Regression

**IEEE citation:**
> Google Developers, "Linear Regression," *Machine Learning Crash Course*, Google, 2024. [Online]. Available: https://developers.google.com/machine-learning/crash-course/linear-regression

**Related works paragraph:**
> The Google Machine Learning Crash Course [8] introduces linear regression as a
> practical tool for prediction, emphasising gradient descent as an alternative
> optimisation strategy to the normal equations. The course material provides clear
> intuition for loss functions and model training. However, it does not address
> statistical inference (p-values, confidence intervals, F-tests) or the classical
> assumption framework required for valid hypothesis testing in academic research.

---

## [9] Ad Dataset Linear Regression — LinkedIn Article

**IEEE citation:**
> H. Thapa, "Ad Dataset: Linear Regression," *LinkedIn Pulse*, 2023. [Online]. Available: https://www.linkedin.com/pulse/ad-dataset-linear-regression-hemant-thapa-iflce/

**Related works paragraph:**
> The applied notebook [9] by Thapa fits a linear regression model to the
> Advertising dataset and reports R² and coefficient values, confirming that TV
> is the dominant predictor. The work is accessible and practically oriented.
> However, it does not test the independence, normality, or equal-variance
> assumptions of OLS, and does not compare against regularised alternatives
> such as Ridge or Lasso, which are important when predictors show correlation.

---

## Related Works — Final Synthesis Paragraph (Template)

```
The above studies [1–9] collectively confirm that linear regression is a widely
applicable and interpretable method for sales prediction from advertising budgets,
with TV consistently emerging as the strongest predictor. However, they share
several limitations: (i) few papers verify all four LINE assumptions through formal
diagnostic tests [2, 7, 9]; (ii) most do not evaluate out-of-sample predictive
accuracy on a held-out test set [1, 7, 8]; (iii) interaction effects between media
channels are rarely modelled [2, 4, 9]; and (iv) none simultaneously provides
both statistical inference (coefficients, p-values, CIs) and machine learning
evaluation (RMSE, cross-validation). The present work addresses all four gaps
by: (a) formally testing all LINE assumptions with Shapiro-Wilk, Breusch-Pagan,
and Durbin-Watson tests; (b) evaluating the final model on a held-out 20% test
set; (c) fitting and comparing an interaction model; and (d) reporting both
inferential statistics from statsmodels and predictive metrics from scikit-learn.
```

---

## Quick Reference: All 9 IEEE Citations

```
[1] G. James, D. Witten, T. Hastie, and R. Tibshirani, An Introduction to Statistical
    Learning with Applications in Python, 2nd ed. Springer, 2023.
    https://doi.org/10.1007/978-3-031-38747-0_3

[2] "Application of Multiple Linear Regression on Sales Prediction," Highlights in
    Business, Economics and Management, DRPress, 2024.
    https://drpress.org/ojs/index.php/HBEM/article/view/27429

[3] Y. H. Yasser, "Advertising Sales Dataset," Kaggle, 2022.
    https://www.kaggle.com/datasets/yasserh/advertising-sales-dataset

[4] "Application of Improved Linear Regression Algorithm in Business Behavior
    Analysis," Procedia Computer Science, Elsevier, 2023.
    https://www.sciencedirect.com/science/article/pii/S1877050923019750

[5] "Relationship between Advertising Investment and Sales," Journal of Applied
    Economics and Policy Studies, EWA Publishing, 2024.
    https://jaeps.ewapub.com/article/view/24423

[6] R. Vershynin, "All of Linear Regression," arXiv:1910.06386, 2019.
    https://arxiv.org/pdf/1910.06386

[7] M. Oyelaran, "EDA: Advertising Spend vs Sales," Medium, 2023.
    https://medium.com/@MazeedahO/eda-advertising-spend-vs-sales-46ab8c339577

[8] Google Developers, "Linear Regression," ML Crash Course, 2024.
    https://developers.google.com/machine-learning/crash-course/linear-regression

[9] H. Thapa, "Ad Dataset: Linear Regression," LinkedIn Pulse, 2023.
    https://www.linkedin.com/pulse/ad-dataset-linear-regression-hemant-thapa-iflce/
```
