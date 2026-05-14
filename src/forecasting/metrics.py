"""Accuracy and calibration metrics for forecast tables."""
import numpy as np
import pandas as pd


def compute_metrics(forecasts: pd.DataFrame, label: str) -> pd.DataFrame:
    """One-row summary: MAE, RMSE, bias, 80% interval coverage."""
    f = forecasts.dropna(subset=["actual", "mean"]).copy()
    f["error"] = f["mean"] - f["actual"]
    f["abs_error"] = f["error"].abs()
    f["squared_error"] = f["error"] ** 2
    f["inside_80"] = (f["actual"] >= f["q10"]) & (f["actual"] <= f["q90"])

    return pd.DataFrame([{
        "model": label,
        "n": len(f),
        "MAE": f["abs_error"].mean(),
        "RMSE": np.sqrt(f["squared_error"].mean()),
        "bias": f["error"].mean(),
        "coverage_80": f["inside_80"].mean(),
    }])


def compute_metrics_grouped(forecasts: pd.DataFrame, label: str, group_cols: list[str]) -> pd.DataFrame:
    """Same metrics, sliced by `group_cols` (e.g. ['district_id'] or ['tier', 'horizon_weeks'])."""
    f = forecasts.dropna(subset=["actual", "mean"]).copy()
    f["error"] = f["mean"] - f["actual"]
    f["abs_error"] = f["error"].abs()
    f["squared_error"] = f["error"] ** 2
    f["inside_80"] = (f["actual"] >= f["q10"]) & (f["actual"] <= f["q90"])

    g = f.groupby(group_cols).agg(
        n=("error", "size"),
        MAE=("abs_error", "mean"),
        RMSE=("squared_error", lambda x: np.sqrt(x.mean())),
        bias=("error", "mean"),
        coverage_80=("inside_80", "mean"),
        mean_actual=("actual", "mean"),
    ).reset_index()
    g["model"] = label
    g["MAE_pct_of_mean"] = (g["MAE"] / g["mean_actual"] * 100).round(1)
    return g
