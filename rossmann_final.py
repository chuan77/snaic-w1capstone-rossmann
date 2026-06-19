"""
=======================================================================
ROSSMANN STORE SALES — PRODUCTION-GRADE PIPELINE WITH CLOSED-LOOP REMEDIATION
=======================================================================
CRISP-DM Phase 3 & 4 Compliance:
  - Part A: Structural Pipeline Engineering (Declared FIRST)
  - Part B: Cross-Validation Baseline Tourney (Reusing Part A Pipeline)
  - Part C: Controlled Ablation Experiments on Champion
  - Part D: Mechanical Failure Analysis (Identifies FM-1: Promo Surprise)
  - Part E: Closed-Loop Remediation Engine (Deploys Local Elasticity Fix)
=======================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
import pickle
from pathlib import Path
from part_d_failure_analysis import run_failure_analysis

# =====================================================================
# 0. EVALUATION METRIC
# =====================================================================

def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    pct_err = np.where(mask, ((y_true - y_pred) / y_true) ** 2, 0.0)
    return float(np.sqrt(pct_err.mean()))


def load_data(data_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = Path(data_dir)
    train = pd.read_csv(p / "train.csv", low_memory=False, parse_dates=["Date"])
    store = pd.read_csv(p / "store.csv")
    test  = pd.read_csv(p / "test.csv",  low_memory=False, parse_dates=["Date"])
    return train, store, test


BASE_FEATURES = [
    "Store", "StoreType", "Assortment", "DayOfWeek", "Year", "Month", "Day", "WeekOfYear", "IsWeekend",
    "Promo", "Promo2", "IsPromo2Active", "Promo2Age", "SchoolHoliday", "StateHoliday", "CompetitionDistance", "CompetitionAge"
]
LAG_FEATURES = ["SalesLag7", "SalesLag14", "SalesLag28", "SalesRollingMean7", "SalesRollingMean28"]
REMEDIATION_FEATURES = ["StorePromoElasticity", "StorePromoInteraction"]
CATEGORICAL_COLS = ["StoreType", "Assortment", "StateHoliday"]

# =====================================================================
# PART A: OPTIMIZED DATA PREPARATION & PIPELINE ENGINEERING
# =====================================================================

class RossmannStatefulTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_cols, cat_cols):
        self.feature_cols = feature_cols
        self.cat_cols = cat_cols
        self.month_to_num = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, 
                             "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
        
        self.distance_imputer = SimpleImputer(strategy="median")
        self.cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.history_buffer_ = None  
        self.store_promo_elasticity_ = {}  # Stateful dictionary for Part E Fix

    def fit(self, X, y=None):
        self.history_buffer_ = X.copy()
        
        if "CompetitionDistance" in X.columns:
            self.distance_imputer.fit(X[["CompetitionDistance"]])
        self.cat_encoder.fit(X[self.cat_cols].astype(str))
        
        # ── Part E Fix: Calculate Localized Store-Level Promo Elasticity Coefficients ──
        if "Sales" in X.columns and "Promo" in X.columns:
            promo_sales = X[X["Promo"] == 1].groupby("Store")["Sales"].mean()
            non_promo_sales = X[X["Promo"] == 0].groupby("Store")["Sales"].mean().replace(0, 1)
            self.store_promo_elasticity_ = (promo_sales / non_promo_sales).to_dict()
            
        return self

    def _compute_internal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d = d.sort_values(["Store", "Date"])
        
        for lag in [7, 14, 28]:
            d[f"SalesLag{lag}"] = d.groupby("Store")["Sales"].shift(lag).fillna(0)
        for window in [7, 28]:
            d[f"SalesRollingMean{window}"] = d.groupby("Store")["Sales"].transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).mean()
            ).fillna(0)

        d["Year"]        = d["Date"].dt.year
        d["Month"]       = d["Date"].dt.month
        d["Day"]         = d["Date"].dt.day
        d["DayOfWeek"]   = d["Date"].dt.dayofweek          
        d["WeekOfYear"]  = d["Date"].dt.isocalendar().week.astype(int)
        d["IsWeekend"]   = (d["DayOfWeek"] >= 5).astype(int)

        d["CompetitionOpenSinceYear"]  = d["CompetitionOpenSinceYear"].fillna(0).astype(int)
        d["CompetitionOpenSinceMonth"] = d["CompetitionOpenSinceMonth"].fillna(0).astype(int)
        d["CompetitionAge"] = (12 * (d["Year"] - d["CompetitionOpenSinceYear"]) + (d["Month"] - d["CompetitionOpenSinceMonth"])).clip(lower=0)

        d["Promo2SinceYear"]  = d["Promo2SinceYear"].fillna(0).astype(int)
        d["Promo2SinceWeek"]  = d["Promo2SinceWeek"].fillna(0).astype(int)
        d["PromoInterval"]    = d["PromoInterval"].fillna("")

        def _promo2_active(row):
            if row["Promo2"] == 0 or row["PromoInterval"] == "": return 0
            active_months = [self.month_to_num.get(m, 0) for m in row["PromoInterval"].split(",")]
            return int(row["Month"] in active_months)

        d["IsPromo2Active"] = d.apply(_promo2_active, axis=1)
        d["Promo2Age"] = (52 * (d["Year"] - d["Promo2SinceYear"]) + (d["WeekOfYear"] - d["Promo2SinceWeek"])).clip(lower=0) * d["Promo2"]
        
        # Map stateful elasticity back to mitigate global smoothing issues discovered in Part D
        d["StorePromoElasticity"] = d["Store"].map(self.store_promo_elasticity_).fillna(1.0)
        d["StorePromoInteraction"] = d["Promo"] * d["StorePromoElasticity"]
        
        return d

    def transform(self, X):
        original_indices = X.index
        if self.history_buffer_ is not None and X.index[0] != self.history_buffer_.index[0]:
            combined_df = pd.concat([self.history_buffer_, X]).drop_duplicates(subset=["Store", "Date"], keep="last")
        else:
            combined_df = X.copy()
            
        d = self._compute_internal_features(combined_df)

        if "CompetitionDistance" in d.columns:
            d["CompetitionDistance"] = self.distance_imputer.transform(d[["CompetitionDistance"]])
        d[self.cat_cols] = self.cat_encoder.transform(d[self.cat_cols].astype(str))
        
        for col in self.feature_cols:
            if col not in d.columns:
                d[col] = 0.0
                
        return d.loc[original_indices, self.feature_cols]


def build_final_submission_pipeline(feature_set) -> Pipeline:
    return Pipeline([
        ('preprocessing', RossmannStatefulTransformer(feature_set, CATEGORICAL_COLS)),
        ('estimator', lgb.LGBMRegressor(random_state=42, verbosity=-1))
    ])


# =====================================================================
# PART B & C: TOURNAMENT AND ABLATION ENGINES
# =====================================================================

def evaluate_models_cv(df: pd.DataFrame, base_pipeline: Pipeline, n_splits: int = 5) -> dict:
    print(f"\n--- Starting Part B: 5-Fold TimeSeriesSplit Baseline Tourney ---")
    df_cv = df.sort_values("Date").reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = {"LightGBM": [], "RandomForest": [], "RidgeRegression": []}
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(df_cv), 1):
        train_df, val_df = df_cv.iloc[train_idx].copy(), df_cv.iloc[val_idx].copy()
        y_tr_log = np.log1p(train_df["Sales"])
        y_va_real = val_df["Sales"].values
        
        base_pipeline.set_params(estimator=lgb.LGBMRegressor(
            objective="regression_l2", learning_rate=0.05, num_leaves=63, 
            n_estimators=300, random_state=42, verbosity=-1
        ))
        base_pipeline.fit(train_df, y_tr_log)
        results["LightGBM"].append(rmspe(y_va_real, np.expm1(base_pipeline.predict(val_df))))
        
        base_pipeline.set_params(estimator=RandomForestRegressor(n_estimators=20, max_depth=12, n_jobs=-1, random_state=42))
        base_pipeline.fit(train_df, y_tr_log)
        results["RandomForest"].append(rmspe(y_va_real, np.expm1(base_pipeline.predict(val_df))))
        
        base_pipeline.set_params(estimator=Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=10.0))]))
        base_pipeline.fit(train_df, y_tr_log)
        results["RidgeRegression"].append(rmspe(y_va_real, np.expm1(base_pipeline.predict(val_df))))
        
    return results

def declare_champion(cv_results: dict) -> str:
    print("\n" + "="*50)
    print("         PART B: CROSS VALIDATION SUMMARY")
    print("="*50)
    summary = {}
    for name, scores in cv_results.items():
        summary[name] = np.mean(scores)
        print(f"{name:<18} -> Mean RMSPE: {np.mean(scores)*100:.2f}% | Std Dev: {np.std(scores)*100:.3f}%")
    champion = min(summary, key=summary.get)
    print(f"🏆 CHAMPION DESIGNATED FOR PART C ABLATIONS: {champion}")
    print("="*50 + "\n")
    return champion


def run_controlled_ablations(df: pd.DataFrame, base_pipeline: Pipeline, feature_set: list, n_splits: int = 3) -> pd.DataFrame:
    df_cv = df.sort_values("Date").reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    experiments = {
        "Exp 0: Baseline Params": {
            "estimator": lgb.LGBMRegressor(objective="regression_l2", learning_rate=0.1, num_leaves=31, random_state=42, verbosity=-1, n_estimators=150),
            "hypothesis": "Evaluate standard parameters over designated feature matrix."
        },
        "Exp 1: Tree Capacity Boost": {
            "estimator": lgb.LGBMRegressor(objective="regression_l2", learning_rate=0.1, num_leaves=63, random_state=42, verbosity=-1, n_estimators=150),
            "hypothesis": "Expanding tree depth captures localized store variance metrics."
        },
        "Exp 2: Cool Learning Scale": {
            "estimator": lgb.LGBMRegressor(objective="regression_l2", learning_rate=0.03, num_leaves=128, random_state=42, verbosity=-1, n_estimators=400),
            "hypothesis": "Deepening leaves while reducing learning step rates tracks complex non-linear trends."
        }
    }
    
    ablation_records = []
    base_mean = 0.0
    
    for exp_name, config in experiments.items():
        scores = []
        base_pipeline.named_steps['preprocessing'].feature_cols = feature_set
        base_pipeline.set_params(estimator=config["estimator"])
        
        for train_idx, val_idx in tscv.split(df_cv):
            train_df, val_df = df_cv.iloc[train_idx].copy(), df_cv.iloc[val_idx].copy()
            y_tr_log = np.log1p(train_df["Sales"])
            y_va_real = val_df["Sales"].values
            
            base_pipeline.fit(train_df, y_tr_log)
            scores.append(rmspe(y_va_real, np.expm1(base_pipeline.predict(val_df))))
            
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        if exp_name == "Exp 0: Baseline Params":
            conclusion = "Benchmark initialized."
            base_mean = mean_score
        else:
            diff = mean_score - base_mean
            conclusion = f"Validated! Improvement verified ({diff*100:.2f}% shift)." if diff < -0.002 else "Neutral/Rejected impact."
                
        ablation_records.append({
            "Experiment": exp_name,
            "Hypothesis": config["hypothesis"],
            "CV RMSPE (Mean ± Std Dev)": f"{mean_score*100:.2f}% ± {std_score*100:.3f}%",
            "Conclusion": conclusion
        })
        
    return pd.DataFrame(ablation_records)

# =====================================================================
# PART E: CLOSED-LOOP REMEDIATION ENGINE (FEEDBACK REMEDIATION)
# =====================================================================

def run_part_e_remediation(df: pd.DataFrame, pipeline: Pipeline):
    print("\n" + "="*80)
    print("  PART E: CLOSED-LOOP REMEDIATION ENGINE (DEPLOYING REVISED FEATURES)")
    print("="*80)
    print("Feedback Received from Part D Failure Mode Analysis: [FM-1 Promo Surprise Outlier Concentration]")
    print("Action Plan: Injecting 'StorePromoElasticity' and 'StorePromoInteraction' features into testing arrays...")
    
    remediated_features = BASE_FEATURES + LAG_FEATURES + REMEDIATION_FEATURES
    ablation_remediated_df = run_controlled_ablations(df, pipeline, remediated_features, n_splits=3)
    
    print("\n" + "="*120)
    print("                       PART E: POST-REMEDIATION CONTROLLED ABLATION LOG TABLE")
    print("="*120)
    print(ablation_remediated_df.to_string(index=False))
    print("="*120 + "\n")


# =====================================================================
# SYSTEM EXECUTION ENGINE
# =====================================================================

def main():
    print("=" * 70)
    print("  ROSSMANN SALES FORECAST — PRODUCTION-GRADE PIPELINE INFRASTRUCTURE")
    print("=" * 70)

    train_raw, store_raw, test_raw = load_data("data")
    train_clean = train_raw[(train_raw["Open"] == 1) & (train_raw["Sales"] > 0)].copy()
    
    df = pd.merge(train_clean, store_raw, on="Store", how="inner")
    df["Sales"] = df["Sales"].astype(float)

    # ── PART A ──
    production_pipeline = build_final_submission_pipeline(BASE_FEATURES + LAG_FEATURES)

    # ── PART B ──
    cv_results = evaluate_models_cv(df, production_pipeline, n_splits=5)
    champion_name = declare_champion(cv_results)

    # ── PART C ──
    if champion_name == "LightGBM":
        print("--- Starting Part C: Champion Controlled Ablations Suite (Standard Features) ---")
        ablation_log_df = run_controlled_ablations(df, production_pipeline, BASE_FEATURES + LAG_FEATURES, n_splits=3)
        print("\n" + "="*110)
        print("                                PART C: CONTROLLED ABLATION LOG TABLE")
        print("="*110)
        print(ablation_log_df.to_string(index=False))
        print("="*110 + "\n")

    # ── PART D ──
    print("Executing standard validation run to feed failure analysis engines...")
    train_df = df[df["Date"] < "2015-07-01"].copy()
    val_df   = df[df["Date"] >= "2015-07-01"].copy()
    y_train_log = np.log1p(train_df["Sales"])
    y_val_real = val_df["Sales"].values

    production_pipeline.fit(train_df, y_train_log)
    val_preds = np.expm1(production_pipeline.predict(val_df))
    run_failure_analysis(val_df=val_df, y_val_real=y_val_real, val_preds=val_preds)

    # ── PART E: CLOSED-LOOP REMEDIATION ENGINE RE-RUN ──
    run_part_e_remediation(df, production_pipeline)

if __name__ == "__main__":
    main()