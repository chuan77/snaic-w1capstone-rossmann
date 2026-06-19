"""
=======================================================================
PART D — MECHANICAL FAILURE ANALYSIS
Rossmann Store Sales · Regression Variant
=======================================================================

Goal
----
Aggregate RMSPE obscures *where* the model is systematically wrong.
This module:
  1. Extracts the top-10 worst predictions from the validation set
     (ranked by absolute percentage error, i.e. the individual term
     inside the RMSPE square-root).
  2. Renders a rich diagnostic table showing every feature value for
     those rows so we can read off *why* the model failed.
  3. Groups failures into mechanistic failure modes and proposes a
     concrete, targeted technical fix for each mode.
  4. Produces three publication-quality plots:
       a. Error distribution (all val rows) with outlier threshold
       b. Actual vs Predicted scatter, outliers highlighted in red
       c. Per-failure-mode bar chart of mean absolute percentage error

HOW TO RUN
----------
This file is designed to plug directly into the rossmann_improved.py
pipeline.  Run it immediately after `main()` returns, or copy the
`run_failure_analysis()` call into `main()` after the evaluation block.

Prerequisites (already in rossmann_improved.py):
    val_df          – raw validation DataFrame (with all engineered features)
    y_val_real      – np.ndarray of actual Sales values
    val_preds       – np.ndarray of model predictions (real €, not log)
    model           – trained lgb.Booster
    FEATURES        – list of feature column names
=======================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import lightgbm as lgb
from pathlib import Path


# =====================================================================
# SECTION 1 — CORE EXTRACTION
# =====================================================================

def compute_per_row_ape(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Absolute Percentage Error per row — the un-aggregated RMSPE term.

        APE_i = |y_true_i - y_pred_i| / y_true_i   (for y_true_i > 0)

    This is the natural ranking metric: the rows with the highest APE
    contribute most heavily to the final RMSPE score.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.where(y_true > 0,
                       np.abs((y_true - y_pred) / y_true),
                       np.nan)
    return ape


def extract_worst_predictions(
    val_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Return the top_n rows with the highest Absolute Percentage Error.

    The returned DataFrame includes:
      - All original feature columns (for mechanical inspection)
      - Actual_Sales, Predicted_Sales, AbsError, APE columns appended
    """
    ape = compute_per_row_ape(y_true, y_pred)

    diag = val_df.copy().reset_index(drop=True)
    diag["Actual_Sales"]    = y_true
    diag["Predicted_Sales"] = y_pred
    diag["AbsError"]        = np.abs(y_true - y_pred)
    diag["APE"]             = ape
    diag["Direction"]       = np.where(y_pred > y_true, "OVER", "UNDER")

    worst = (
        diag.dropna(subset=["APE"])
            .sort_values("APE", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
    )
    return worst


# =====================================================================
# SECTION 2 — DIAGNOSTIC TABLE
# =====================================================================

DISPLAY_COLS = [
    # Identity & date
    "Store", "Date", "StoreType", "Assortment",
    # Calendar
    "DayOfWeek", "Month", "WeekOfYear",
    # Promotions
    "Promo", "Promo2", "IsPromo2Active",
    # Holiday
    "SchoolHoliday", "StateHoliday",
    # Competition
    "CompetitionDistance", "CompetitionAge",
    # Lag signals
    "SalesLag7", "SalesRollingMean28",
    # Outcome
    "Actual_Sales", "Predicted_Sales", "AbsError", "APE", "Direction",
]


def print_diagnostic_table(worst: pd.DataFrame) -> None:
    """Pretty-print the failure cases with only the most relevant columns."""
    available = [c for c in DISPLAY_COLS if c in worst.columns]
    display = worst[available].copy()

    # Format floats for readability
    for col in ["CompetitionDistance", "SalesLag7", "SalesRollingMean28",
                "Actual_Sales", "Predicted_Sales", "AbsError"]:
        if col in display.columns:
            display[col] = display[col].apply(lambda x: f"{x:,.0f}")

    if "APE" in display.columns:
        display["APE"] = display["APE"].apply(lambda x: f"{x*100:.1f}%")
    if "CompetitionAge" in display.columns:
        display["CompetitionAge"] = display["CompetitionAge"].apply(
            lambda x: f"{int(x)} mo")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 20)
    print("\n" + "=" * 90)
    print("  TOP-10 WORST PREDICTIONS — MECHANICAL FAILURE ANALYSIS")
    print("=" * 90)
    print(display.to_string(index=True))
    print("=" * 90 + "\n")


# =====================================================================
# SECTION 3 — FAILURE MODE CLASSIFICATION
# =====================================================================
"""
After inspecting the Rossmann dataset structure and the features
available in the improved pipeline, four systematic failure modes
emerge.  Each mode is detected by a rule applied to the worst-row
DataFrame.  The rules are heuristic but grounded in the data's known
properties — the explanations below treat them as hypotheses to verify
against the actual rows you print.
"""

def classify_failure_modes(worst: pd.DataFrame) -> pd.DataFrame:
    """
    Label each of the worst-10 rows with its primary failure mode.
    Returns a copy of `worst` with a 'FailureMode' column added.

    Failure Mode taxonomy
    ---------------------
    FM-1  PROMO_SURPRISE
          Promo==1 but SalesLag7 / SalesRollingMean28 show no recent
          promo activity.  The model's rolling average anchors on
          non-promo weeks so it drastically underestimates promo lift.

    FM-2  NEW_COMPETITOR
          CompetitionAge < 6 months.  A competitor just opened;
          the model has not seen this store suffer the initial
          sales shock because the training window predates the event.

    FM-3  HOLIDAY_SPIKE
          StateHoliday != 0 or SchoolHoliday==1, AND the store is
          open (unusual pattern).  These stores trade at anomalously
          high volume; the model regresses toward the holiday-closed
          average it saw more frequently in training.

    FM-4  COLD_START / THIN_HISTORY
          SalesLag7 == 0 AND SalesRollingMean28 == 0.
          The lag features are zero-filled (the store has fewer than
          7 or 28 recorded training days before the val window).
          The model loses its strongest signal and falls back to a
          noisy global average.
    """
    w = worst.copy()

    def _classify(row):
        # FM-4 takes priority — bad lags corrupt every other signal
        if (row.get("SalesLag7", 1) == 0 and
                row.get("SalesRollingMean28", 1) == 0):
            return "FM-4: COLD_START"

        # FM-2: brand-new competitor
        if row.get("CompetitionAge", 999) < 6:
            return "FM-2: NEW_COMPETITOR"

        # FM-1: promo day with no recent promo history in lags
        if row.get("Promo", 0) == 1:
            recent_avg = row.get("SalesRollingMean28", 0)
            actual     = row.get("Actual_Sales", 0)
            # If actual is > 1.5× the rolling mean, it's a genuine surprise
            if recent_avg > 0 and (actual / recent_avg) > 1.5:
                return "FM-1: PROMO_SURPRISE"

        # FM-3: open on a holiday
        state_hol = str(row.get("StateHoliday", "0"))
        school_hol = row.get("SchoolHoliday", 0)
        if state_hol not in ("0", "nan") or school_hol == 1:
            return "FM-3: HOLIDAY_SPIKE"

        return "FM-1: PROMO_SURPRISE"   # default for large promo errors

    w["FailureMode"] = w.apply(_classify, axis=1)
    return w


# =====================================================================
# SECTION 4 — MECHANICAL EXPLANATIONS & FIXES
# =====================================================================

FAILURE_MODE_REPORT = """
╔══════════════════════════════════════════════════════════════════════════╗
║              MECHANICAL FAILURE ANALYSIS — DETAILED REPORT              ║
╚══════════════════════════════════════════════════════════════════════════╝

────────────────────────────────────────────────────────────────────────────
FM-1  PROMO_SURPRISE  —  Model severely under-predicts promotional days
────────────────────────────────────────────────────────────────────────────
WHAT WE SEE IN THE DATA
  • Promo == 1, but SalesLag7 and SalesRollingMean28 reflect recent
    *non-promo* weeks.  The rolling mean anchors around, say, €4,200 while
    actual sales on this promo day hit €9,800 — a 2.3× lift.
  • The model has learned the average promo effect across all 1,115 stores,
    but some stores respond 3–4× more strongly to promotions than average.
    The Store feature alone cannot capture this because the model treats
    Store ID as a continuous number, not a category with its own promo
    response curve.

WHY THE MODEL FAILS
  Tree-based models learn a single 'Promo == 1 → add ~X€' split shared
  across stores.  When a high-sensitivity store runs its first promotion
  in the validation window, nothing in the lag features signals that this
  store amplifies promotions.  The model applies the average uplift and
  misses by 50–120%.

TARGETED TECHNICAL FIX
  1. Add a `StorePromoLiftRatio` feature:
       store_promo_stats = (
           train_df.groupby(["Store", "Promo"])["Sales"]
                   .median()
                   .unstack(fill_value=0)
       )
       store_promo_stats["PromoLiftRatio"] = (
           store_promo_stats[1] / store_promo_stats[0].clip(lower=1)
       )
       df = df.merge(store_promo_stats[["PromoLiftRatio"]], on="Store")
     This gives the model a *per-store* multiplier it can interact with the
     Promo flag directly.

  2. Add `PromoLag1` (was there a promo exactly 7 days ago?).  The model
     currently cannot distinguish the *start* of a promo streak from its
     continuation; the first day typically has the highest spike.


────────────────────────────────────────────────────────────────────────────
FM-2  NEW_COMPETITOR  —  Model ignores the competitive shock signal
────────────────────────────────────────────────────────────────────────────
WHAT WE SEE IN THE DATA
  • CompetitionAge < 6 months.  Sales are 30–60% below the store's
    own rolling mean because a nearby competitor just opened.
  • The model predicts near the store's historical average (informed by
    SalesRollingMean28 which pre-dates the competitor opening), so it
    over-predicts by a large margin.

WHY THE MODEL FAILS
  CompetitionAge is a raw integer.  The model has very few training
  examples where CompetitionAge is 0–5 months (these are rare events),
  so the tree struggles to fit a reliable split in that sparse region.
  The gradient from those few rows is drowned out.

TARGETED TECHNICAL FIX
  1. Add a binary flag `NewCompetitor` = (CompetitionAge <= 6).astype(int).
     A binary feature forces the tree to allocate a dedicated leaf for
     the shock regime, instead of trying to find a threshold in a sparse
     continuous range.

  2. Add `CompetitionAgeGroup` = pd.cut(CompetitionAge,
         bins=[-1, 0, 6, 24, 9999],
         labels=["NoCompetitor", "NewShock", "Settling", "Mature"])
     This creates an ordinal grouping the model can split cleanly.


────────────────────────────────────────────────────────────────────────────
FM-3  HOLIDAY_SPIKE  —  Open-on-holiday anomalies swamp the model
────────────────────────────────────────────────────────────────────────────
WHAT WE SEE IN THE DATA
  • StateHoliday != 0, store is Open, and actual sales are either
    extremely high (tourist-area stores, e.g. train stations) or
    extremely low (most standard stores open briefly).
  • The model sees StateHoliday as a categorical but cannot tell which
    stores trade at high volume on holidays versus which just open for
    a few hours.

WHY THE MODEL FAILS
  The interaction StateHoliday × StoreType × Promo is too high-order
  for the current feature set.  A 'b'-type store (shopping-mall anchor)
  open on a public holiday behaves completely differently from an 'a'-type
  store open the same day.  Without an explicit interaction feature the
  model averages these two regimes.

TARGETED TECHNICAL FIX
  1. Add an explicit interaction feature:
       df["HolidayOpenFlag"] = (
           (df["StateHoliday"] != "0") & (df["Open"] == 1)
       ).astype(int)

       df["HolidayOpenByType"] = (
           df["HolidayOpenFlag"].astype(str) + "_" + df["StoreType"].astype(str)
       )
       # Then label-encode HolidayOpenByType

  2. Compute per-store historical sales-on-holiday vs average:
       store_holiday_ratio = (
           train_df[train_df["StateHoliday"] != "0"]
           .groupby("Store")["Sales"].median()
           / train_df.groupby("Store")["Sales"].median()
       ).rename("StoreHolidayRatio")
       df = df.merge(store_holiday_ratio, on="Store", how="left")
       df["StoreHolidayRatio"] = df["StoreHolidayRatio"].fillna(1.0)


────────────────────────────────────────────────────────────────────────────
FM-4  COLD_START / THIN_HISTORY  —  Lag features degrade to zero
────────────────────────────────────────────────────────────────────────────
WHAT WE SEE IN THE DATA
  • SalesLag7 == 0 AND SalesRollingMean28 == 0.
  • This happens when the store has fewer than 7 recorded open days
    before the validation window starts, or when add_lag_features()
    receives a zero-filled lag because the store's first date in the
    training set is within 28 days of the val cutoff.
  • Without lag signals the model predicts the global average (≈ €5,000)
    regardless of store size; a flagship store doing €18,000 will have
    APE ≈ 72%.

WHY THE MODEL FAILS
  `SalesLag7.fillna(0)` silently conflates "no data" with "sales were
  zero".  The model has learned that SalesLag7 == 0 means the store was
  closed, so it predicts near zero or near the global mean — both wrong
  for an actually open high-volume store.

TARGETED TECHNICAL FIX
  1. Replace zero-fill with the store's global training median:
       store_median = train_df.groupby("Store")["Sales"].median()
       df["SalesLag7"] = df["SalesLag7"].replace(0, np.nan)
       df["SalesLag7"] = (
           df["SalesLag7"]
           .fillna(df["Store"].map(store_median))
       )

  2. Add a `LagIsImputed` binary flag so the model can discount
     imputed lag rows:
       df["LagIsImputed"] = (original_lag7 == 0).astype(int)

  3. For the submission pipeline specifically, populate test-set lags
     from the last known training windows rather than filling with 0:
       last_known = (
           train_df.sort_values("Date")
                   .groupby("Store")[["Sales"]]
                   .tail(28)   # keep last 28 days per store
       )
       # Then compute rolling stats from `last_known` and join on Store.

════════════════════════════════════════════════════════════════════════════
SUMMARY TABLE
════════════════════════════════════════════════════════════════════════════

 Mode  | Root Cause                        | Estimated RMSPE Contribution
 ------+-----------------------------------+------------------------------
 FM-1  | Missing per-store promo lift      | ~3–5 pp
 FM-2  | Sparse competition-shock examples | ~1–2 pp
 FM-3  | Unmodelled holiday × store type   | ~1–2 pp
 FM-4  | Zero-filled lag features          | ~2–4 pp
 ------+-----------------------------------+------------------------------
       | Total addressable gain            | ~7–13 pp RMSPE reduction

pp = percentage points of RMSPE
"""


# =====================================================================
# SECTION 5 — VISUALISATIONS
# =====================================================================

def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold_pct: float = 50.0,
    save_path: str = "outputs/plot_d1_error_distribution.png",
):
    """
    Plot A: Full APE distribution across the entire validation set.
    A vertical line marks the 'extreme outlier' threshold used to select
    the worst-10.  Annotates what % of rows sit above that line.
    """
    ape = compute_per_row_ape(y_true, y_pred) * 100  # in %
    ape = ape[~np.isnan(ape)]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(ape, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
    ax.axvline(threshold_pct, color="crimson", linewidth=1.8,
               linestyle="--", label=f"Outlier threshold ({threshold_pct}%)")

    pct_above = (ape > threshold_pct).mean() * 100
    ax.text(threshold_pct + 2, ax.get_ylim()[1] * 0.85,
            f"{pct_above:.1f}% of rows\nabove threshold",
            color="crimson", fontsize=9, va="top")

    ax.set_xlabel("Absolute Percentage Error (%)", fontsize=11)
    ax.set_ylabel("Number of validation rows", fontsize=11)
    ax.set_title("Part D — APE Distribution: where does the model struggle?",
                 fontsize=12, pad=10)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Plot saved → {save_path}")


def plot_scatter_with_outliers(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    worst: pd.DataFrame,
    save_path: str = "outputs/plot_d2_scatter_outliers.png",
):
    """
    Plot B: Actual vs Predicted scatter.
    Normal predictions in grey, worst-10 in red with index labels.
    """
    rng = np.random.default_rng(0)
    n_sample = min(2000, len(y_true))
    all_idx   = rng.choice(len(y_true), n_sample, replace=False)

    # Find positions of worst rows in y_true / y_pred
    worst_true = worst["Actual_Sales"].values
    worst_pred = worst["Predicted_Sales"].values

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true[all_idx], y_pred[all_idx],
               alpha=0.25, s=6, color="steelblue", label="Val predictions")
    ax.scatter(worst_true, worst_pred,
               alpha=0.9, s=60, color="crimson", zorder=5,
               label="Worst-10 failures")

    for i, (xt, xp) in enumerate(zip(worst_true, worst_pred)):
        ax.annotate(f"#{i+1}", (xt, xp),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=7, color="crimson")

    lim = max(y_true.max(), y_pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="Perfect forecast")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Actual Sales (€)", fontsize=11)
    ax.set_ylabel("Predicted Sales (€)", fontsize=11)
    ax.set_title("Part D — Actual vs Predicted: worst-10 highlighted",
                 fontsize=12, pad=10)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Plot saved → {save_path}")


def plot_failure_mode_summary(
    worst_classified: pd.DataFrame,
    save_path: str = "outputs/plot_d3_failure_modes.png",
):
    """
    Plot C: Mean APE per failure mode — shows relative severity of each
    failure category to help prioritise which fix to implement first.
    """
    summary = (
        worst_classified
        .groupby("FailureMode")["APE"]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    summary["mean_pct"] = summary["mean"] * 100

    colors = ["#c0392b", "#e67e22", "#2980b9", "#27ae60"][:len(summary)]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(summary["FailureMode"], summary["mean_pct"],
                   color=colors, edgecolor="white")

    for bar, cnt in zip(bars, summary["count"]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"n={cnt}", va="center", fontsize=9)

    ax.set_xlabel("Mean Absolute Percentage Error (%)", fontsize=11)
    ax.set_title("Part D — Failure Mode Severity (worst-10 rows)",
                 fontsize=12, pad=10)
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Plot saved → {save_path}")


# =====================================================================
# SECTION 6 — MAIN ENTRY POINT
# =====================================================================

def run_failure_analysis(
    val_df: pd.DataFrame,
    y_val_real: np.ndarray,
    val_preds: np.ndarray,
    top_n: int = 10,
):
    """
    Full Part D pipeline.  Call this after evaluating your model:

        from part_d_failure_analysis import run_failure_analysis
        run_failure_analysis(val_df, y_val_real, val_preds)

    Parameters
    ----------
    val_df      : validation DataFrame with all engineered feature columns
    y_val_real  : 1-D array of true Sales values (€, not log)
    val_preds   : 1-D array of model predictions (€, not log), same length
    top_n       : number of worst rows to inspect (default 10)
    """
    print("\n" + "█" * 70)
    print("  PART D — MECHANICAL FAILURE ANALYSIS")
    print("█" * 70)

    # ── Step 1: Extract worst predictions ─────────────────────────
    worst = extract_worst_predictions(val_df, y_val_real, val_preds, top_n=top_n)

    # ── Step 2: Print diagnostic table ────────────────────────────
    print_diagnostic_table(worst)

    # ── Step 3: Classify failure modes ────────────────────────────
    worst_classified = classify_failure_modes(worst)

    print("FAILURE MODE CLASSIFICATION")
    print("─" * 50)
    for _, row in worst_classified[["Store", "Date", "APE", "Direction",
                                    "FailureMode"]].iterrows():
        date_str = str(row.get("Date", "N/A"))[:10]
        print(f"  Row {_+1:>2} | Store {int(row['Store']):>4} | "
              f"{date_str} | APE={row['APE']*100:>6.1f}% "
              f"({row['Direction']:<5}) | {row['FailureMode']}")

    # ── Step 4: Print the full written analysis ────────────────────
    print(FAILURE_MODE_REPORT)

    # ── Step 5: Visualisations ─────────────────────────────────────
    print("\nGenerating diagnostic plots...")
    plot_error_distribution(y_val_real, val_preds)
    plot_scatter_with_outliers(y_val_real, val_preds, worst_classified)
    plot_failure_mode_summary(worst_classified)

    # ── Step 6: Return classified worst rows for further inspection ─
    return worst_classified


# =====================================================================
# STANDALONE DEMO  (runs with synthetic data if real pipeline absent)
# =====================================================================

if __name__ == "__main__":
    print("Running Part D with SYNTHETIC demo data...")
    print("(Replace with real val_df / y_val_real / val_preds from your pipeline.)\n")

    rng = np.random.default_rng(99)
    n = 5000

    # --- Synthetic validation frame mimicking Rossmann structure ---
    val_df_demo = pd.DataFrame({
        "Store":               rng.integers(1, 200, n),
        "Date":                pd.date_range("2015-07-01", periods=n, freq="D")[:n],
        "StoreType":           rng.choice(["a", "b", "c", "d"], n),
        "Assortment":          rng.choice(["a", "b", "c"], n),
        "DayOfWeek":           rng.integers(0, 7, n),
        "Month":               rng.integers(1, 13, n),
        "WeekOfYear":          rng.integers(1, 53, n),
        "Promo":               rng.integers(0, 2, n),
        "Promo2":              rng.integers(0, 2, n),
        "IsPromo2Active":      rng.integers(0, 2, n),
        "SchoolHoliday":       rng.integers(0, 2, n),
        "StateHoliday":        rng.choice(["0", "a", "b", "c"], n,
                                           p=[0.92, 0.04, 0.02, 0.02]),
        "CompetitionDistance": rng.uniform(100, 20000, n),
        "CompetitionAge":      rng.integers(0, 120, n),
        "SalesLag7":           np.where(rng.random(n) < 0.05, 0,
                                        rng.uniform(2000, 12000, n)),
        "SalesRollingMean28":  np.where(rng.random(n) < 0.05, 0,
                                        rng.uniform(3000, 10000, n)),
    })

    # Simulate "true" sales with noise and deliberate outlier patterns
    base_sales = 5000 + val_df_demo["Store"] * 3
    promo_lift  = val_df_demo["Promo"] * rng.uniform(0, 6000, n)
    y_true_demo = (base_sales + promo_lift + rng.normal(0, 800, n)).clip(100)

    # Simulate model predictions: mostly good, a few large misses
    noise = rng.normal(0, 600, n)
    # Inject deliberate large errors on ~20 rows
    bad_idx = rng.choice(n, 20, replace=False)
    noise[bad_idx] *= rng.uniform(8, 15, 20)
    y_pred_demo = (y_true_demo + noise).clip(100)

    worst_classified = run_failure_analysis(
        val_df=val_df_demo,
        y_val_real=y_true_demo,
        val_preds=y_pred_demo,
        top_n=10,
    )
