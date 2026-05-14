"""End-to-end pipeline.

Runs every model through the rolling-origin simulator, prints the headline and
per-district tables, and writes one CSV per model to submissions/.

Usage:
    python run.py                 # all models
    python run.py --models seasonal_naive xgboost_with_sn
    python run.py --no-write      # don't write CSVs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from forecasting import (  # noqa: E402
    load_data, build_panel,
    LastWeekNaive, SeasonalNaive, PoissonForecaster,
    SeasonalNaivePlus, SeasonalNaivePlusV2,
    XGBoostForecaster, XGBoostForecasterV2,
    run_simulation, compute_metrics, compute_metrics_grouped,
)

ALL_MODELS = {
    "last_week_naive":       LastWeekNaive,
    "seasonal_naive":        SeasonalNaive,
    "poisson_pooled":        PoissonForecaster,
    "seasonal_naive_plus":   SeasonalNaivePlus,
    "seasonal_naive_plus_v2": SeasonalNaivePlusV2,
    "xgboost_pooled":        XGBoostForecaster,
    "xgboost_with_sn":       XGBoostForecasterV2,
}


def verify_harness(df: pd.DataFrame, baseline: pd.DataFrame) -> None:
    """Check that our SeasonalNaive matches the shipped baseline on well-formed rows."""
    sn = SeasonalNaive()
    sn.fit(df)
    check = baseline.copy()
    check["issue_dt"]  = pd.to_datetime(check["issue_iso_week"]  + "-1", format="%G-W%V-%u")
    check["target_dt"] = pd.to_datetime(check["target_iso_week"] + "-1", format="%G-W%V-%u")
    check["real_horizon"] = (check["target_dt"] - check["issue_dt"]).dt.days // 7
    check["well_formed"]  = check["real_horizon"] == check["horizon_weeks"]
    check["our_mean"] = check.apply(
        lambda r: sn.predict(r["issue_dt"], int(r["horizon_weeks"]), r["district_id"])["mean"],
        axis=1,
    )
    check["matches"] = (check["our_mean"] - check["mean"]).abs() < 0.5

    n_well = int(check["well_formed"].sum())
    n_ok   = int((check["well_formed"] & check["matches"]).sum())
    n_bad  = int((check["well_formed"] & ~check["matches"]).sum())
    print(f"  well-formed rows in shipped baseline: {n_well} (concerns: {n_bad}, matched: {n_ok})")
    assert n_bad == 0, "SeasonalNaive does not match the shipped baseline — harness is suspect"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(ALL_MODELS),
                    choices=list(ALL_MODELS),
                    help="Subset of models to run (default: all).")
    ap.add_argument("--first-issue", default="2024-W01",
                    help="First issue ISO week (YYYY-Www). Default 2024-W01.")
    ap.add_argument("--last-issue", default="2024-W48",
                    help="Last issue ISO week. Default 2024-W48 (4w horizon stays inside data range).")
    ap.add_argument("--no-write", action="store_true",
                    help="Don't write forecast CSVs to submissions/.")
    args = ap.parse_args()

    print("=" * 70)
    print("Dengue forecasting pipeline")
    print("=" * 70)

    print("\n[1/4] Loading data ...")
    cases, weather, districts, baseline = load_data()
    df = build_panel(cases, weather, districts)
    print(f"  panel: {df.shape}, {df['week_start'].min().date()} -> {df['week_start'].max().date()}")

    print("\n[2/4] Verifying harness against shipped baseline ...")
    verify_harness(df, baseline)

    print("\n[3/4] Running rolling-origin simulator ...")
    first_issue = pd.to_datetime(args.first_issue + "-1", format="%G-W%V-%u")
    last_issue  = pd.to_datetime(args.last_issue  + "-1", format="%G-W%V-%u")
    tier_map = districts.set_index("district_id")["tier"].to_dict()

    forecasts = {}
    for name in args.models:
        m = ALL_MODELS[name]()
        print(f"  - {name}")
        fc = run_simulation(m, df, first_issue, last_issue, refit_each_step=False)
        fc["tier"] = fc["district_id"].map(tier_map)
        fc["target_week_of_year"] = fc["target_week_dt"].dt.isocalendar().week
        forecasts[name] = fc

    if not args.no_write:
        out_dir = ROOT / "submissions"
        out_dir.mkdir(exist_ok=True)
        for name, fc in forecasts.items():
            cols = ["issue_iso_week", "target_iso_week", "district_id",
                    "horizon_weeks", "mean", "q10", "q90"]
            (fc[cols]).to_csv(out_dir / f"{name}.csv", index=False)
        print(f"  wrote {len(forecasts)} forecast CSVs to submissions/")

    print("\n[4/4] Metrics")
    print("\n--- Headline (all districts, all horizons, full year) ---")
    headline = pd.concat(
        [compute_metrics(fc, name) for name, fc in forecasts.items()],
        ignore_index=True,
    )
    print(headline.to_string(index=False))

    print("\n--- Surge season per-district (target weeks W22-W36) ---")
    for name, fc in forecasts.items():
        surge = fc[(fc["target_week_of_year"] >= 22) & (fc["target_week_of_year"] <= 36)]
        if surge.empty:
            continue
        grouped = compute_metrics_grouped(surge, name, ["district_id"])
        print(f"\n{name}:")
        print(grouped[["district_id", "n", "mean_actual", "MAE",
                       "MAE_pct_of_mean", "bias", "coverage_80"]].to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
