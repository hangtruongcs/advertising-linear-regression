# Setup Guide — Advertising Linear Regression

Step-by-step instructions to clone, configure, and run this project locally in VS Code.

---

## Prerequisites

| Tool | Version | Check |
|---|---|---|
| Python | >= 3.11 | `python3 --version` |
| pip | latest | `pip --version` |
| Git | any | `git --version` |
| VS Code | latest | [Download](https://code.visualstudio.com/) |

**Required VS Code extensions** (install from Extensions sidebar):

- **Python** (`ms-python.python`) — language support, linting, debugging
- **Jupyter** (`ms-toolsai.jupyter`) — run `.ipynb` notebooks inside VS Code

---

## 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/advertising-linear-regression.git
cd advertising-linear-regression
```

---

## 2. Create a Virtual Environment

```bash
python3 -m venv .venv
```

Activate it:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Windows (CMD)
.\.venv\Scripts\activate.bat
```

You should see `(.venv)` at the start of your terminal prompt.

---

## 3. Install Dependencies

```bash
pip install --upgrade pip
pip install jupyter ipykernel pandas numpy matplotlib statsmodels scipy scikit-learn
```

All required packages and their purposes:

| Package | Purpose |
|---|---|
| `jupyter` | Notebook server |
| `ipykernel` | Register Python kernel for VS Code |
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical computation |
| `matplotlib` | Charts and visualisation |
| `statsmodels` | OLS regression, p-values, F-test, confidence intervals |
| `scipy` | Shapiro-Wilk normality test |
| `scikit-learn` | Train/test split, RMSE, StandardScaler |

---

## 4. Register the Jupyter Kernel

```bash
python -m ipykernel install --user --name advertising-lr --display-name "Python (advertising-lr)"
```

This makes the kernel visible in VS Code's kernel picker.

---

## 5. Open in VS Code

```bash
code .
```

Or open VS Code manually and use **File > Open Folder** to select the project directory.

---

## 6. Select the Kernel and Run the Notebook

1. Open `notebooks/ch3_linear_regression.ipynb`
2. Click the **kernel selector** in the top-right corner of the notebook
3. Select **"Python (advertising-lr)"** from the list
4. Click **Run All** (the double-play button) or press `Shift + Alt + Enter`

If the kernel doesn't appear, reload VS Code (`Cmd+Shift+P` > `Developer: Reload Window`).

---

## Project Structure

```
advertising-linear-regression/
├── data/
│   ├── advertising_budget_and_sales.csv   # Dataset (200 markets, 4 columns)
│   └── advertising_sales_dataset.md       # Dataset documentation
├── notebooks/
│   └── ch3_linear_regression.ipynb        # Main analysis notebook (36 cells)
├── output/                                # Generated figures (auto-created)
│   ├── fig_slr_tv.png                     # OLS fit: Sales ~ TV
│   ├── fig_rss_surface.png                # RSS loss surface
│   ├── fig_population_vs_ols.png          # Sampling variability demo
│   ├── fig_mlr_plane.png                  # 3-D regression plane
│   ├── fig_q5_actual_vs_pred.png          # Actual vs Predicted (test set)
│   ├── fig_q6_diagnostics.png             # LINE assumption plots
│   └── fig_q7_synergy.png                 # TV×Radio synergy curve
├── report/
│   ├── ad-analytis.md                     # Full academic report (Markdown)
│   └── advertising_lr_paper.docx          # Report (Word format)
├── src/                                   # Source modules (if any)
├── tests/                                 # Unit tests (if any)
├── pyproject.toml                         # Project metadata and dependencies
├── SETUP.md                               # This file
└── README.md                              # Project overview
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'numpy'`

The notebook is using a different Python kernel. Fix:

1. Click the kernel selector (top-right of the notebook)
2. Switch to **"Python (advertising-lr)"**
3. If not listed, re-run: `python -m ipykernel install --user --name advertising-lr --display-name "Python (advertising-lr)"`

### Kernel not showing in VS Code

```bash
# Check registered kernels
jupyter kernelspec list

# If missing, re-register
source .venv/bin/activate
python -m ipykernel install --user --name advertising-lr --display-name "Python (advertising-lr)"
```

Then reload VS Code: `Cmd+Shift+P` > `Developer: Reload Window`

### `FileNotFoundError: ../data/advertising_budget_and_sales.csv`

The notebook expects to be run from the `notebooks/` directory. VS Code handles this automatically when you open the `.ipynb` file. If running from terminal:

```bash
cd notebooks
jupyter notebook ch3_linear_regression.ipynb
```

### Output folder not created

The notebook creates `output/` automatically on the first run (`os.makedirs("../output", exist_ok=True)`). No manual action needed.

---

## Quick Start (copy-paste)

```bash
git clone https://github.com/<your-username>/advertising-linear-regression.git
cd advertising-linear-regression
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install jupyter ipykernel pandas numpy matplotlib statsmodels scipy scikit-learn
python -m ipykernel install --user --name advertising-lr --display-name "Python (advertising-lr)"
code .
```

Then open `notebooks/ch3_linear_regression.ipynb`, select the kernel, and Run All.
