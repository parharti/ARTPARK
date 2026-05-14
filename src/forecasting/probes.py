"""Failure-mode probes.

These exist to find the conditions under which the forecast should NOT be
trusted. They are a required deliverable — see docs/EVAL_DESIGN.md s.9.

Probe 1 — Reporting collapse (health-specific):
    Real reporting drops during festivals, staff shortages, PHC closures.
    We simulate this by reducing one district's reported cases for a few
    peak-surge weeks, then ask: how long do the models stay wrong after?

Probe 2 — Outbreak-week performance:
    A forecast can look fine on average and still be useless on the weeks
    that matter. We score the models only on target weeks where actuals
    jumped >50% from the prior week.
"""
from __future__ import annotations

from typing import Type

import numpy as np
import pandas as pd

from .forecasters import Forecaster
from .simulator import run_simulation


def reporting_collapse_probe(
    df: pd.DataFrame,
    model_classes: list[Type[Forecaster]],
    first_issue: pd.Timestamp,
    last_issue: pd.Timestamp,
    district: str = "D04",
    drop_pct: float = 0.40,
    n_weeks: int = 3,
    surge_window: tuple[int, int] = (22, 36),
    test_year: int = 2024,
    recovery_weeks: int = 6,
) -> dict:
    """
    Reduce `district`'s reported cases by `drop_pct` for the top `n_weeks`
    by case count during the surge window of `test_year`. Re-run each model
    in `model_classes` on both the clean and corrupted panels and compare
    per-district MAE on the impacted window (corrupted weeks + recovery_weeks
    after the last corrupted week).

    Returns a dict with: corrupted_weeks, per-model {clean_mae, corrupted_mae,
    delta_mae, n_impacted_weeks}, and an estimate of recovery time
    (weeks after corruption until MAE returns to within 10% of the clean run).
    """
    sub = df[df["district_id"] == district].copy()
    sub["year"] = sub["week_start"].dt.isocalendar().year
    sub["woy"] = sub["week_start"].dt.isocalendar().week
    in_surge = (
        (sub["year"] == test_year)
        & (sub["woy"] >= surge_window[0])
        & (sub["woy"] <= surge_window[1])
    )
    surge = sub[in_surge].nlargest(n_weeks, "cases")
    corrupted_weeks = sorted(surge["week_start"].tolist())
    last_corrupt = max(corrupted_weeks)
    impact_end = last_corrupt + pd.Timedelta(weeks=recovery_weeks)

    corrupted_panel = df.copy()
    mask = (
        (corrupted_panel["district_id"] == district)
        & (corrupted_panel["week_start"].isin(corrupted_weeks))
    )
    corrupted_panel.loc[mask, "cases"] = (
        corrupted_panel.loc[mask, "cases"] * (1 - drop_pct)
    ).round()

    results = {
        "district": district,
        "drop_pct": drop_pct,
        "corrupted_weeks": [w.date().isoformat() for w in corrupted_weeks],
        "recovery_window_weeks": recovery_weeks,
        "models": {},
    }

    # Truth source: ALWAYS the clean panel. We're testing how the model's
    # predictions change when its inputs are corrupted, not how well its
    # predictions match a corrupted ground truth.
    true_truth = df[["district_id", "week_start", "cases"]].rename(
        columns={"week_start": "target_week_dt", "cases": "true_actual"}
    )

    for cls in model_classes:
        m_clean = cls()
        m_corr = cls()
        fc_clean = run_simulation(m_clean, df, first_issue, last_issue, refit_each_step=False)
        fc_corr  = run_simulation(m_corr, corrupted_panel, first_issue, last_issue, refit_each_step=False)

        def slice_window(fc):
            mask = (
                (fc["district_id"] == district)
                & (fc["target_week_dt"] >= corrupted_weeks[0])
                & (fc["target_week_dt"] <= impact_end)
            )
            sliced = fc[mask].merge(true_truth, on=["district_id", "target_week_dt"], how="left")
            return sliced.dropna(subset=["true_actual", "mean"])

        clean_window = slice_window(fc_clean)
        corr_window  = slice_window(fc_corr)
        clean_mae = float((clean_window["mean"] - clean_window["true_actual"]).abs().mean())
        corr_mae  = float((corr_window["mean"]  - corr_window["true_actual"]).abs().mean())

        # Per-target-week MAE timeline, to estimate recovery
        timeline = []
        recovery = None
        target_weeks = sorted(corr_window["target_week_dt"].unique())
        for tw in target_weeks:
            c = corr_window[corr_window["target_week_dt"] == tw]
            cl = clean_window[clean_window["target_week_dt"] == tw]
            if c.empty:
                continue
            c_err  = float((c["mean"]  - c["true_actual"]).abs().mean())
            cl_err = float((cl["mean"] - cl["true_actual"]).abs().mean()) if not cl.empty else np.nan
            timeline.append({
                "target_week": pd.Timestamp(tw).date().isoformat(),
                "weeks_after_last_corrupt": int((pd.Timestamp(tw) - last_corrupt).days // 7),
                "clean_abs_err": cl_err,
                "corrupted_abs_err": c_err,
            })
            # First week after the last corrupted week where the model is back within 10%
            if recovery is None and pd.Timestamp(tw) > last_corrupt:
                if cl_err > 0 and abs(c_err - cl_err) / max(cl_err, 1.0) < 0.10:
                    recovery = int((pd.Timestamp(tw) - last_corrupt).days // 7)

        results["models"][m_clean.name] = {
            "clean_mae_window":     clean_mae,
            "corrupted_mae_window": corr_mae,
            "delta_mae":            corr_mae - clean_mae,
            "n_window_rows":        int(len(corr_window)),
            "recovery_weeks_to_within_10pct": recovery,
            "weekly_timeline":      timeline,
        }

    return results


def outbreak_week_probe(
    forecasts_by_model: dict[str, pd.DataFrame],
    panel: pd.DataFrame,
    jump_threshold: float = 0.50,
) -> pd.DataFrame:
    """
    Tag each forecast target week as 'outbreak' (target-week actuals jumped
    > `jump_threshold` from the prior week, same district) or 'quiet'. Return
    a per-model summary with MAE on each subset and the n in each.
    """
    p = panel.sort_values(["district_id", "week_start"]).copy()
    p["prior_cases"] = p.groupby("district_id")["cases"].shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p["jump_pct"] = (p["cases"] - p["prior_cases"]) / p["prior_cases"]
    p["is_outbreak"] = p["jump_pct"] > jump_threshold

    rows = []
    for name, fc in forecasts_by_model.items():
        merged = fc.merge(
            p[["district_id", "week_start", "is_outbreak"]],
            left_on=["district_id", "target_week_dt"],
            right_on=["district_id", "week_start"],
            how="left",
        ).dropna(subset=["actual", "mean", "is_outbreak"])
        merged["abs_err"] = (merged["mean"] - merged["actual"]).abs()

        outbreak = merged[merged["is_outbreak"]]
        quiet    = merged[~merged["is_outbreak"]]

        rows.append({
            "model": name,
            "outbreak_n":   int(len(outbreak)),
            "outbreak_MAE": float(outbreak["abs_err"].mean()) if len(outbreak) else np.nan,
            "quiet_n":      int(len(quiet)),
            "quiet_MAE":    float(quiet["abs_err"].mean())    if len(quiet)    else np.nan,
            "outbreak_vs_quiet_ratio": (
                float(outbreak["abs_err"].mean() / quiet["abs_err"].mean())
                if len(outbreak) and len(quiet) and quiet["abs_err"].mean() > 0
                else np.nan
            ),
        })
    return pd.DataFrame(rows)
