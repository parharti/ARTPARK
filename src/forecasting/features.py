"""Feature builders shared by the Poisson and XGBoost forecasters."""
import numpy as np
import pandas as pd

POP_PER_100K = 100_000


def build_poisson_features(panel: pd.DataFrame, target_week: pd.Timestamp,
                           district_id: str, history: pd.DataFrame) -> dict | None:
    """
    Build features for a single (target_week, district) prediction for the
    pooled Poisson model. Uses `history` only — never future data.

    Lags are counted back from the most recent observation; if the most
    recent observation is `weeks_ago` from the target, lag k means
    arr[len-1 - max(0, k - weeks_ago)].
    """
    h = history[(history["district_id"] == district_id) &
                (history["week_start"] < target_week)].sort_values("week_start")
    if len(h) < 6:
        return None

    pop = h.iloc[0]["population"]
    log_rate = np.log((h["cases"] / pop * POP_PER_100K) + 1).values
    rainfall = h["rainfall_mm"].fillna(h["rainfall_mm"].median()).values

    last_obs_date = h.iloc[-1]["week_start"]
    weeks_ago = (target_week - last_obs_date).days // 7

    def lag(arr, k):
        idx = len(arr) - 1 - max(0, k - weeks_ago)
        if idx < 0:
            return arr[0]
        return arr[idx]

    feats = {
        "log_rate_lag_1": lag(log_rate, 1),
        "log_rate_lag_2": lag(log_rate, 2),
        "log_rate_lag_4": lag(log_rate, 4),
        "rainfall_lag_4": lag(rainfall, 4),
        "rainfall_lag_6": lag(rainfall, 6),
    }

    target_woy = target_week.isocalendar().week
    feats["sin_woy"] = np.sin(2 * np.pi * target_woy / 52)
    feats["cos_woy"] = np.cos(2 * np.pi * target_woy / 52)

    return feats


def build_poisson_training_table(history: pd.DataFrame, districts_to_include) -> pd.DataFrame:
    """For each (district, week) in history with enough lags, build a training row."""
    rows = []
    for did in districts_to_include:
        h_d = history[history["district_id"] == did].sort_values("week_start")
        for _, row in h_d.iterrows():
            target_week = row["week_start"]
            feats = build_poisson_features(history, target_week, did, history)
            if feats is None:
                continue
            feats["district_id"] = did
            feats["cases"] = row["cases"]
            feats["population"] = row["population"]
            rows.append(feats)
    return pd.DataFrame(rows)


def build_xgb_features(panel: pd.DataFrame, district_id: str,
                       issue_week: pd.Timestamp, horizon: int) -> dict | None:
    """
    Build features for one (district, issue_week, horizon) prediction.
    All lags are relative to issue_week — same definition at train and predict
    time, which is what fixes the leakage in attempt 1.
    """
    h = panel[(panel["district_id"] == district_id) &
              (panel["week_start"] <= issue_week)].sort_values("week_start")
    if len(h) < 6:
        return None

    pop = h.iloc[0]["population"]
    rates = (h["cases"] / pop * POP_PER_100K).values

    rate_lag_0 = rates[-1]
    rate_lag_1 = rates[-2] if len(rates) >= 2 else rates[-1]
    rate_lag_2 = rates[-3] if len(rates) >= 3 else rates[-1]
    rate_lag_4 = rates[-5] if len(rates) >= 5 else rates[-1]

    target_week = issue_week + pd.Timedelta(weeks=horizon)

    def rain_at(target, weeks_back):
        ref = target - pd.Timedelta(weeks=weeks_back)
        r = panel[(panel["district_id"] == district_id) &
                  (panel["week_start"] == ref)]
        if r.empty:
            return 0.0
        v = r.iloc[0]["rainfall_mm"]
        return 0.0 if pd.isna(v) else float(v)

    rain_lag_4 = rain_at(target_week, 4)
    rain_lag_6 = rain_at(target_week, 6)

    target_woy = target_week.isocalendar().week
    sin_woy = np.sin(2 * np.pi * target_woy / 52)
    cos_woy = np.cos(2 * np.pi * target_woy / 52)

    return {
        "rate_lag_0": rate_lag_0,
        "rate_lag_1": rate_lag_1,
        "rate_lag_2": rate_lag_2,
        "rate_lag_4": rate_lag_4,
        "rain_lag_4": rain_lag_4,
        "rain_lag_6": rain_lag_6,
        "sin_woy": sin_woy,
        "cos_woy": cos_woy,
        "horizon": horizon,
        "district_id": district_id,
        "population": pop,
    }


def build_xgb_features_v2(panel: pd.DataFrame, district_id: str,
                          issue_week: pd.Timestamp, horizon: int) -> dict | None:
    """Same as v1 but adds the seasonal-naive prediction as a feature."""
    feats = build_xgb_features(panel, district_id, issue_week, horizon)
    if feats is None:
        return None

    target_week = issue_week + pd.Timedelta(weeks=horizon)
    sn_lookup = target_week - pd.Timedelta(weeks=52)
    sn_rows = panel[(panel["district_id"] == district_id) &
                    (panel["week_start"] == sn_lookup)]
    if sn_rows.empty:
        feats["sn_rate"] = feats["rate_lag_0"]
    else:
        pop = feats["population"]
        feats["sn_rate"] = float(sn_rows.iloc[0]["cases"]) / pop * POP_PER_100K
    return feats
