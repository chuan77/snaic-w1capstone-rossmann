# ICT6001 Week01 Group Capstone — Rossmann Store Sales

## Team

| Name | Role |
|------|------|
| Goh Hui Min | Data Architect & Engineer |
| Wong Chuan Sern | AI Modeling Engineer |
| Xu Hongming | Optimization & Tuning Engineer |
| Ling Kheng Aik Jony | Risk & Business Strategist |

## 1. Project Objective

To carry out data validation, baseline feasibility using default classical models with feature engineering / finetuning, transitioning to pipeline engineering, ablation studies and decision making.

The objective is not to brute-force a high accuracy score (e.g. by running thousands of hyperparameter tuning iterations) or chasing leaderboard rankings. The objective is to demonstrate data preparation via pipelines, logical model selection, controlled experimentation, and a mechanical understanding of model failures before translating probabilistic or continuous outputs into actionable business policies.

## 2. Scope and Hard Constraints

**Permitted:**

- **Advanced Data Preparation:** Feature scaling, advanced encoding (e.g., target encoding), complex imputation, and data imbalance mitigation (e.g., SMOTE).
- **Model Ensembling:** Classical ensemble methods (e.g., Random Forest, Gradient Boosting, Model Stacking).
- **Hyperparameter Tuning:** Justify what you search and why. Roughly 3 parameters per model, with grid/random searches of 50 total iterations.

**Strictly Prohibited:**

- **Deep Learning / LLMs:** Language models or deep learning models remain prohibited as the primary predictive component. They may be used for feature engineering but not for prediction.
- **Massive Automated Search:** Unjustified computational brute-forcing (e.g., 100+ grid points).
- **Test Set Overuse:** The hold-out test set may only be evaluated once to generate the final deployment metrics. All model selection and tuning must use cross-validation on the training set.

## 3. Dataset

**Kaggle Rossmann Store Sales** — Rossmann operates over 3,000 drug stores in 7 European countries. Store managers are tasked with predicting daily sales up to six weeks in advance. Sales are influenced by promotions, competition, school and state holidays, seasonality, and locality.

## 3. Task Requirements

The assignment is split into four parts.

### Part A: Advanced Data Preparation & Pipeline Engineering

- **Requirement:** Finalize CRISP-DM Phase 3. Implement robust handling for scaling, encoding, and missing values.
- **Constraint:** All data transformations and the estimator must be strictly encapsulated within a formal pipeline object (e.g., `sklearn.pipeline.Pipeline`). This proves the prevention of data leakage during validation.

### Part B: Champion Model Selection

- **Requirement:** Compare 2 to 3 distinct algorithmic families (e.g., a Linear/Distance-based model versus a Tree-based ensemble). Model stacking approaches are also permitted.
- **Deliverable:** Evaluate base models using k-fold cross-validation (e.g., 5-fold) or a time-series split. Declare one model as the "Champion" based on the mean and standard deviation of your primary evaluation metric.

### Part C: Controlled Ablations & Tuning (The Champion)

- **Requirement:** Perform a maximum of 4 controlled experiments exclusively on the chosen Champion model. This may involve adding a specific engineered feature, applying a class balancing technique, or tuning a specific set of hyperparameters.
- **Deliverable:** An Ablation Log (Table) explicitly detailing: Hypothesis, Controlled Change, CV Metric Impact (Mean ± Std Dev), and Conclusion.

### Part D: Mechanical Failure Analysis *(optional)*

- **Requirement:** Aggregate metrics obscure underlying model flaws. Inspect the raw validation data where the model was confidently incorrect.
  - **For Classification:** Extract 5–10 instances of False Positives or False Negatives with high prediction confidence. Mechanically explain why the model failed on these specific instances based on their feature values, and propose a targeted technical fix.
  - **For Regression:** Extract the 5–10 instances from your validation set that exhibit the highest absolute error (extreme under/over-predictions). Analyze the feature values of these specific extreme outliers to explain the failure and propose a technical fix.

### Part E: Decision Making

- **Requirement:** A model's raw output must be translated into a business decision. Default thresholds (0.5) are rarely optimal in the real world. You must logically evaluate the risks of your model's errors based on the context defined in Stage 1.

  **For Classification:**
  - Identify which error is more damaging to the business: a False Positive or a False Negative.
  - Logically argue whether your operating threshold should be shifted higher (more conservative) or lower (more aggressive) than the default 0.5. Justify the direction of the shift — an exact optimal threshold is not required.

  **For Regression:**
  - Identify which error carries a heavier operational penalty: over-predicting or under-predicting the target variable.
  - Logically argue whether the business should apply a "safety margin" (e.g., systematically adding or subtracting a buffer to raw predictions) before acting on the model's output.

## 4. Setup & Usage

### Prerequisites

- **Python 3.11.5+**
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager (replaces pip/venv)
  ```bash
  # Install uv (macOS / Linux)
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Jupyter** (for the notebook) — installed automatically via `uv sync` below

### Clone the repository

```bash
git clone https://github.com/chuan77/snaic-w1capstone-rossmann.git
cd snaic-w1capstone-rossmann
```

If you already have the repo, pull the latest changes:

```bash
git pull origin main
```

### Install dependencies

```bash
uv sync
```

This creates a `.venv` virtual environment and installs all locked dependencies (`lightgbm`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`).

### Download the dataset

The `data/` folder is not tracked by git. Download the Kaggle Rossmann Store Sales dataset and place the files here:

```
data/
├── train.csv
├── test.csv
└── store.csv
```

### Run the training script

```bash
uv run python rossmann_final.py
```

Outputs (plots, ablation log, submission CSV, serialised pipeline) are written to `outputs/` and `submissions/`.

### Run the failure analysis script

```bash
uv run python part_d_failure_analysis.py
```

Requires `rossmann_formal_pipeline.pkl` to exist (generated by `rossmann_final.py`).

### Open the EDA notebook

```bash
uv run jupyter notebook eda_JL_CPU.ipynb
```

Or open it directly in VS Code — the `.venv` kernel will be detected automatically.

## 5. Project Structure

```
snaic-w1capstone-rossmann/
├── data/
│   ├── train.csv                        # Historical daily sales used for training
│   ├── test.csv                         # Hold-out set for final evaluation / Kaggle submission
│   ├── store.csv                        # Store-level metadata (type, assortment, competition)
│   └── mechanical_failure_analysis.csv  # Derived dataset capturing model failure cases
│
├── outputs/
│   ├── ablation_log_report.csv          # Structured results from each ablation experiment
│   ├── actual_vs_predicted.png          # Diagnostic plot comparing predictions to ground truth
│   ├── feature_importance.png           # Bar chart of top model features
│   ├── plot_d1_error_distribution.png   # Residual / error distribution histogram
│   ├── plot_d2_scatter_outliers.png     # Scatter plot highlighting outlier predictions
│   ├── plot_d3_failure_modes.png        # Categorised failure mode breakdown
│   └── submission.csv                   # Final Kaggle-formatted submission file
│
├── submissions/
│   ├── rossmann_lgbm_kaggle.csv         # LightGBM predictions formatted for Kaggle
│   ├── rossmann_lgbm_business.csv       # LightGBM predictions with business-policy labels
│   ├── rossmann_lgbm_predictions_full.csv  # Full LightGBM prediction set (all rows)
│   ├── rossmann_rf_kaggle.csv           # Random Forest predictions formatted for Kaggle
│   ├── rossmann_rf_business.csv         # Random Forest predictions with business-policy labels
│   └── rossmann_rf_predictions_full.csv # Full Random Forest prediction set (all rows)
│
├── rossmann_final.py                    # Chuan Sern own exploratory EDA python code, model selection, ablation analysis, mechanical failure analysis, submission files
├── part_d_failure_analysis.py           # Failure mode analysis: error bucketing and visualisations
├── rossmann_formal_pipeline.pkl         # Serialised best-model pipeline for inference
├── eda_JL_CPU.ipynb                     # Main training script: pipeline, ablations, model export data analysis notebook for the project
├── main.py                              # Project entry point (placeholder)
├── pyproject.toml                       # Project metadata and dependencies (managed via uv)
├── uv.lock                              # Locked dependency versions for reproducibility
├── .python-version                      # Pinned Python version for the project
└── .vscode/settings.json                # VSCode workspace settings
```
