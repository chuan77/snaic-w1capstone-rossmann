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

## 4. Project Structure

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
├── rossmann_final.py                    # Chuan Sern own exploratory EDA notebook, model selection, ablation analysis, mechanical failure analysis, submission files
├── part_d_failure_analysis.py           # Failure mode analysis: error bucketing and visualisations
├── rossmann_formal_pipeline.pkl         # Serialised best-model pipeline for inference
├── eda_JL_CPU.ipynb                     # Main training script: pipeline, ablations, model export data analysis notebook for the project
├── main.py                              # Project entry point (placeholder)
├── pyproject.toml                       # Project metadata and dependencies (managed via uv)
├── uv.lock                              # Locked dependency versions for reproducibility
├── .python-version                      # Pinned Python version for the project
└── .vscode/settings.json                # VSCode workspace settings
```
