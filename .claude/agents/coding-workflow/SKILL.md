---
name: coding-workflow
description: >
  Use this skill for any Python coding task in a data science, ML, or statistical analysis
  project, especially for the Advertising Sales / ISL linear regression pipeline.
  Triggers: "write code for", "implement", "build a pipeline", "set up the project",
  "Python code for regression", "data preprocessing code", "train test split",
  "model evaluation", "visualisation code", "write functions", "refactor", "add docstrings",
  "unit tests", "download Kaggle dataset", "kaggle API", "full code from scratch",
  or any request for Python code in a data/ML/statistics context.
  For the Advertising dataset: always use Kaggle API to download, follow the
  src/ module structure, and produce all 4 diagnostic plots plus the ISL Figure 2.1
  scatter recreation.
---

# Coding Workflow Skill
## Project: Linear Regression · Advertising Sales · ISL Ch.3

---

## Repository Structure

```
advertising-lr/
├── data/
│   ├── raw/                     ← never modify
│   │   └── advertising.csv      ← downloaded from Kaggle
│   └── processed/               ← cleaned outputs
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── features.py
│   ├── model.py
│   └── evaluate.py
├── tests/
│   ├── test_features.py
│   └── test_model.py
├── outputs/
│   ├── figures/                 ← all plots saved here
│   └── reports/                 ← model summaries, metrics JSON
├── config.yaml
├── requirements.txt
└── README.md
```

---

## Step 0 — Download Dataset from Kaggle

```bash
# Install Kaggle CLI
pip install kaggle

# Place kaggle.json in ~/.kaggle/  (get from kaggle.com → Account → API)
# Then:
kaggle datasets download -d yasserh/advertising-sales-dataset -p data/raw/ --unzip

# Verify
ls data/raw/
# → advertising.csv  (200 rows × 4 columns)
```

---

## config.yaml

```yaml
data:
  raw_path:        data/raw/advertising.csv
  processed_path:  data/processed/advertising_clean.csv
  target:          Sales
  features:        [TV, Radio, Newspaper]
  test_size:       0.2
  random_state:    42

model:
  type:   ols          # ols | ridge | lasso
  alpha:  1.0

output:
  figures_dir:  outputs/figures
  reports_dir:  outputs/reports
  model_path:   outputs/model.pkl
```

---

## src/data_loader.py

```python
"""Data loading, validation, and train/test splitting for the Advertising dataset."""

import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml

logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    """Load YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_advertising(config: dict) -> pd.DataFrame:
    """Load and validate the Advertising CSV (Kaggle: yasserh/advertising-sales-dataset).

    Args:
        config: Loaded config dict.

    Returns:
        Validated DataFrame with columns [TV, Radio, Newspaper, Sales].

    Raises:
        ValueError: If required columns are missing or data has unexpected shape.
    """
    df = pd.read_csv(config["data"]["raw_path"])

    # Drop unnamed index column if present (common in Kaggle CSVs)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    expected_cols = {"TV", "Radio", "Newspaper", "Sales"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df.isnull().sum().sum() > 0:
        logger.warning("Missing values detected:\n%s", df.isnull().sum())

    logger.info("Loaded advertising data: %d rows × %d cols", *df.shape)
    return df


def split_data(
    df: pd.DataFrame,
    config: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target into train and test sets.

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    X = df[config["data"]["features"]]
    y = df[config["data"]["target"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"]
    )
    logger.info("Train=%d  Test=%d", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test
```

---

## src/features.py

```python
"""Feature engineering: interaction terms and scaling."""

import pandas as pd
from sklearn.preprocessing import StandardScaler


def add_interaction(
    df: pd.DataFrame,
    col_a: str = "TV",
    col_b: str = "Radio"
) -> pd.DataFrame:
    """Add TV × Radio interaction term (ISL Ch.3 Q7 — synergy effect).

    Args:
        df: DataFrame containing col_a and col_b.
        col_a: First predictor (default 'TV').
        col_b: Second predictor (default 'Radio').

    Returns:
        Copy of df with new column '{col_a}_x_{col_b}'.
    """
    df = df.copy()
    df[f"{col_a}_x_{col_b}"] = df[col_a] * df[col_b]
    return df


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit StandardScaler on train, transform both splits.

    IMPORTANT: scaler is fit on X_train only to prevent data leakage.

    Returns:
        (X_train_scaled, X_test_scaled, fitted_scaler)
    """
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns, index=X_train.index
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns, index=X_test.index
    )
    return X_train_sc, X_test_sc, scaler
```

---

## src/model.py

```python
"""OLS and regularised regression models for the Advertising dataset."""

import statsmodels.api as sm
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV


def fit_ols(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit OLS with statsmodels for full inference output.

    Returns:
        Fitted statsmodels RegressionResults (contains .summary(), .pvalues,
        .rsquared, .fvalue, .conf_int(), .get_influence()).
    """
    return sm.OLS(y_train, sm.add_constant(X_train)).fit()


def fit_sklearn_lr(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> LinearRegression:
    """Fit sklearn LinearRegression (for prediction and cross-validation)."""
    return LinearRegression().fit(X_train, y_train)


def fit_interaction_model(
    df: pd.DataFrame,
    target: str = "Sales"
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit OLS with TV × Radio interaction term (ISL Q7 — synergy).

    Applies the hierarchical principle: main effects kept regardless of p-value.
    """
    from src.features import add_interaction
    df_int = add_interaction(df)
    X_int  = df_int[['TV','Radio','Newspaper','TV_x_Radio']]
    y      = df_int[target]
    return sm.OLS(y, sm.add_constant(X_int)).fit()


def fit_ridge(
    X_train_sc: pd.DataFrame,
    y_train: pd.Series
) -> RidgeCV:
    """Fit Ridge with cross-validated alpha (regularisation alternative)."""
    model = RidgeCV(alphas=[0.01,0.1,1,10,100], cv=5)
    model.fit(X_train_sc, y_train)
    return model


def fit_lasso(
    X_train_sc: pd.DataFrame,
    y_train: pd.Series
) -> LassoCV:
    """Fit Lasso with cross-validated alpha (performs variable selection)."""
    model = LassoCV(cv=5, random_state=42, max_iter=10_000)
    model.fit(X_train_sc, y_train)
    return model
```

---

## src/evaluate.py

```python
"""Metrics, diagnostics, and visualisations for the Advertising regression."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Compute RMSE, MAE, R² on test set."""
    return {
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "MAE":  round(float(mean_absolute_error(y_true, y_pred)), 4),
        "R2":   round(float(r2_score(y_true, y_pred)), 4),
    }


def run_diagnostics(
    model,
    X_train: pd.DataFrame
) -> dict:
    """Run all LINE assumption tests. Returns dict of test statistics and verdicts."""
    residuals = model.resid
    fitted    = model.fittedvalues

    sw_stat, sw_p = stats.shapiro(residuals)
    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, model.model.exog)
    dw_val = durbin_watson(residuals)
    vif_max = max(
        variance_inflation_factor(X_train.values, i)
        for i in range(X_train.shape[1])
    )

    return {
        "shapiro_wilk":  {"stat": round(sw_stat,4), "p": round(sw_p,4),
                          "holds": sw_p > 0.05},
        "breusch_pagan": {"stat": round(bp_stat,4), "p": round(bp_p,4),
                          "holds": bp_p > 0.05},
        "durbin_watson": {"stat": round(dw_val,4),
                          "holds": 1.5 < dw_val < 2.5},
        "max_vif":       {"stat": round(vif_max,2),
                          "holds": vif_max < 5},
    }


def plot_all_diagnostics(
    model,
    output_dir: str = "outputs/figures"
) -> None:
    """Save all 4 required diagnostic plots + ISL-style scatter to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    residuals = model.resid
    fitted    = model.fittedvalues

    # 1. Residuals vs Fitted
    plt.figure(figsize=(6,4))
    plt.scatter(fitted, residuals, alpha=0.5, s=20)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Fitted values"); plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted")
    plt.savefig(f"{output_dir}/resid_vs_fitted.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Q-Q Plot
    fig, ax = plt.subplots(figsize=(5,5))
    stats.probplot(residuals, dist='norm', plot=ax)
    ax.set_title("Normal Q-Q")
    plt.savefig(f"{output_dir}/qq_plot.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Scale-Location
    plt.figure(figsize=(6,4))
    plt.scatter(fitted, np.sqrt(np.abs(residuals)), alpha=0.5, s=20)
    plt.xlabel("Fitted"); plt.ylabel("√|Residuals|")
    plt.title("Scale-Location")
    plt.savefig(f"{output_dir}/scale_location.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Cook's Distance
    cooks_d   = model.get_influence().cooks_distance[0]
    threshold = 4 / model.nobs
    plt.figure(figsize=(8,4))
    plt.stem(range(len(cooks_d)), cooks_d, markerfmt=',', basefmt=' ')
    plt.axhline(threshold, color='red', linestyle='--',
                label=f"4/n = {threshold:.3f}")
    plt.xlabel("Observation"); plt.ylabel("Cook's distance")
    plt.title("Cook's Distance")
    plt.legend()
    plt.savefig(f"{output_dir}/cooks_distance.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_isl_figure_2_1(
    df: pd.DataFrame,
    output_dir: str = "outputs/figures"
) -> None:
    """Recreate ISL Figure 2.1: sales vs TV/Radio/Newspaper with fitted lines."""
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, col in zip(axes, ['TV', 'Radio', 'Newspaper']):
        ax.scatter(df[col], df['Sales'], alpha=0.35, s=15, color='lightblue',
                   edgecolors='steelblue', linewidths=0.5)
        m, b = np.polyfit(df[col], df['Sales'], 1)
        x_line = np.linspace(df[col].min(), df[col].max(), 200)
        ax.plot(x_line, m * x_line + b, color='blue', linewidth=2)
        ax.set_xlabel(f'{col} budget ($K)', fontsize=11)
        ax.set_ylabel('Sales (K units)', fontsize=11)
        ax.set_title(f'Sales vs {col}', fontsize=12)

    plt.suptitle('ISL Figure 2.1 — Advertising Data', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/isl_figure_2_1.png", dpi=150, bbox_inches='tight')
    plt.close()
```

---

## tests/test_model.py

```python
"""Unit tests for src/model.py."""
import pytest
import pandas as pd
import numpy as np
from src.model import fit_ols, fit_sklearn_lr


@pytest.fixture
def ad_sample():
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({
        "TV":        np.random.uniform(0, 300, n),
        "Radio":     np.random.uniform(0, 50, n),
        "Newspaper": np.random.uniform(0, 100, n),
    })
    y = 2 + 0.046*X["TV"] + 0.189*X["Radio"] - 0.001*X["Newspaper"] \
        + np.random.normal(0, 1.7, n)
    return X, pd.Series(y, name="Sales")


def test_ols_tv_positive(ad_sample):
    X, y = ad_sample
    m = fit_ols(X, y)
    assert m.params["TV"] > 0, "TV coefficient should be positive"


def test_ols_newspaper_small(ad_sample):
    X, y = ad_sample
    m = fit_ols(X, y)
    assert abs(m.params["Newspaper"]) < abs(m.params["TV"]), \
        "Newspaper coef should be smaller than TV coef"


def test_ols_r2_above_threshold(ad_sample):
    X, y = ad_sample
    m = fit_ols(X, y)
    assert m.rsquared > 0.8, f"R² = {m.rsquared:.3f}, expected > 0.80"


def test_sklearn_no_nan(ad_sample):
    X, y = ad_sample
    m    = fit_sklearn_lr(X, y)
    preds = m.predict(X)
    assert not np.isnan(preds).any(), "Predictions contain NaN"


def test_sklearn_correct_length(ad_sample):
    X, y = ad_sample
    m     = fit_sklearn_lr(X, y)
    preds = m.predict(X)
    assert len(preds) == len(y)
```

---

## requirements.txt

```
pandas>=2.1.0
numpy>=1.26.0
scikit-learn>=1.3.0
statsmodels>=0.14.0
matplotlib>=3.8.0
seaborn>=0.13.0
scipy>=1.11.0
pyyaml>=6.0
pytest>=7.4.0
kaggle>=1.6.0
```

---

## Code Quality Checklist

- [ ] Kaggle dataset downloaded via API (not manually copied).
- [ ] `random_state=42` on all stochastic operations.
- [ ] Train/test split done ONCE before any preprocessing.
- [ ] `StandardScaler` fit on train only.
- [ ] All plots saved to `outputs/figures/`, not shown inline.
- [ ] All functions have docstrings and type hints.
- [ ] Unit tests cover: correct sign of TV coef, R² > threshold, no NaN predictions.
- [ ] `config.yaml` used for all paths and hyperparameters (no hardcoded strings in src/).
- [ ] ISL Figure 2.1 recreation saved as `isl_figure_2_1.png`.
