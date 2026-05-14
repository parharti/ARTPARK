"""Forecaster implementations.

The `Forecaster` ABC is the only contract the simulator depends on:
implement `.fit(history)` and `.predict(issue_week, horizon_weeks, district_id)`
and the model is swappable.

Models, in the order they were tried:
  - LastWeekNaive         persistence baseline
  - SeasonalNaive         cases at (target - 52w); matches the shipped baseline
  - PoissonForecaster     attempt 1 — pooled GLM, beaten by seasonal-naive
  - SeasonalNaivePlus     attempt 2 — SN + learned correction (failed: see EVAL_DESIGN s.5)
  - SeasonalNaivePlusV2   variant: anchor is the (district, week-of-year) mean
  - XGBoostForecaster     pooled, rate-scaled, one-hot district
  - XGBoostForecasterV2   adds the seasonal-naive prediction as a feature
"""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor, Ridge
import xgboost as xgb

from .features import (
    POP_PER_100K,
    build_poisson_features,
    build_poisson_training_table,
    build_xgb_features,
    build_xgb_features_v2,
)


class Forecaster(ABC):
    """The simulator only ever calls .fit() and .predict()."""
    name: str = "unnamed"

    @abstractmethod
    def fit(self, history: pd.DataFrame): ...

    @abstractmethod
    def predict(self, issue_week: pd.Timestamp, horizon_weeks: int, district_id: str) -> dict: ...


class LastWeekNaive(Forecaster):
    """forecast(W + h) = cases at week W. Same prediction for every horizon."""
    name = "last_week_naive"

    def __init__(self):
        self.history = None

    def fit(self, history):
        self.history = history

    def predict(self, issue_week, horizon_weeks, district_id):
        district_history = self.history[
            (self.history["district_id"] == district_id) &
            (self.history["week_start"] <= issue_week)
        ]
        if district_history.empty:
            return {"mean": np.nan, "q10": np.nan, "q90": np.nan}
        last_value = district_history.iloc[-1]["cases"]
        return {
            "mean": float(last_value),
            "q10": float(last_value * 0.5),
            "q90": float(last_value * 1.5),
        }


class SeasonalNaive(Forecaster):
    """forecast(target) = cases at (target - 52 weeks) for the same district."""
    name = "seasonal_naive"

    def __init__(self, seasonal_period_weeks: int = 52):
        self.history = None
        self.seasonal_period = seasonal_period_weeks

    def fit(self, history):
        self.history = history

    def predict(self, issue_week, horizon_weeks, district_id):
        target_week = issue_week + pd.Timedelta(weeks=horizon_weeks)
        lookup_week = target_week - pd.Timedelta(weeks=self.seasonal_period)
        rows = self.history[
            (self.history["district_id"] == district_id) &
            (self.history["week_start"] == lookup_week)
        ]
        if rows.empty:
            return {"mean": np.nan, "q10": np.nan, "q90": np.nan}
        value = rows.iloc[0]["cases"]
        return {
            "mean": float(value),
            "q10": float(value * 0.5),
            "q90": float(value * 1.5),
        }


class PoissonForecaster(Forecaster):
    """Pooled Poisson regression with district fixed effects (attempt 1)."""
    name = "poisson_pooled"

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.model = None
        self.feature_cols = None
        self.history = None
        self._residuals_by_district = {}

    def fit(self, history):
        self.history = history
        district_ids = sorted(history["district_id"].unique())
        training = build_poisson_training_table(history, district_ids)

        if len(training) < 30:
            self.model = None
            return

        X = pd.get_dummies(training.drop(columns=["cases", "population"]),
                           columns=["district_id"], drop_first=False)
        for did in district_ids:
            col = f"district_id_{did}"
            if col not in X.columns:
                X[col] = 0
        X = X.reindex(sorted(X.columns), axis=1)
        self.feature_cols = X.columns.tolist()

        y = training["cases"]
        self.model = PoissonRegressor(alpha=self.alpha, max_iter=2000)
        self.model.fit(X, y)

        preds = self.model.predict(X)
        training["pred"] = preds
        self._residuals_by_district = training.groupby("district_id").apply(
            lambda g: (g["cases"] - g["pred"]).values
        ).to_dict()

    def predict(self, issue_week, horizon_weeks, district_id):
        if self.model is None:
            return {"mean": np.nan, "q10": np.nan, "q90": np.nan}

        target_week = issue_week + pd.Timedelta(weeks=horizon_weeks)
        feats = build_poisson_features(self.history, target_week, district_id, self.history)
        if feats is None:
            return {"mean": np.nan, "q10": np.nan, "q90": np.nan}

        feats["district_id"] = district_id
        X_row = pd.DataFrame([feats])
        X_row = pd.get_dummies(X_row, columns=["district_id"], drop_first=False)
        for col in self.feature_cols:
            if col not in X_row.columns:
                X_row[col] = 0
        X_row = X_row[self.feature_cols]

        mean_pred = float(self.model.predict(X_row)[0])

        resids = self._residuals_by_district.get(district_id, None)
        if resids is not None and len(resids) > 10:
            q10 = max(0.0, mean_pred + float(np.quantile(resids, 0.10)))
            q90 = mean_pred + float(np.quantile(resids, 0.90))
        else:
            q10 = mean_pred * 0.5
            q90 = mean_pred * 1.5

        return {"mean": mean_pred, "q10": float(q10), "q90": float(q90)}


class SeasonalNaivePlus(Forecaster):
    """SN value + a learned residual correction (attempt 2)."""
    name = "seasonal_naive_plus"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = None
        self.feature_cols = None
        self.history = None
        self._residuals_by_district = {}

    def _seasonal_naive_lookup(self, target_week, district_id, hist):
        lookup = target_week - pd.Timedelta(weeks=52)
        rows = hist[(hist["district_id"] == district_id) &
                    (hist["week_start"] == lookup)]
        if rows.empty:
            return None
        return float(rows.iloc[0]["cases"])

    def _build_correction_features(self, target_week, district_id, hist):
        sn_value = self._seasonal_naive_lookup(target_week, district_id, hist)
        if sn_value is None:
            return None, None

        h_d = hist[(hist["district_id"] == district_id) &
                   (hist["week_start"] < target_week)].sort_values("week_start")
        recent_residuals = []
        for _, row in h_d.tail(8).iterrows():
            sn_past = self._seasonal_naive_lookup(row["week_start"], district_id, hist)
            if sn_past is not None:
                recent_residuals.append(row["cases"] - sn_past)
        recent_resid_mean = float(np.mean(recent_residuals[-4:])) if len(recent_residuals) >= 4 else 0.0

        def get_rainfall_lag(weeks_back):
            ref = target_week - pd.Timedelta(weeks=weeks_back)
            r = hist[(hist["district_id"] == district_id) &
                     (hist["week_start"] == ref)]
            if r.empty:
                return 0.0
            v = r.iloc[0]["rainfall_mm"]
            return 0.0 if pd.isna(v) else float(v)

        woy = target_week.isocalendar().week
        feats = {
            "recent_resid_mean": recent_resid_mean,
            "rainfall_lag_4": get_rainfall_lag(4),
            "rainfall_lag_6": get_rainfall_lag(6),
            "sin_woy": np.sin(2 * np.pi * woy / 52),
            "cos_woy": np.cos(2 * np.pi * woy / 52),
            "district_id": district_id,
        }
        return feats, sn_value

    def fit(self, history):
        self.history = history
        district_ids = sorted(history["district_id"].unique())

        rows = []
        for did in district_ids:
            h_d = history[history["district_id"] == did].sort_values("week_start")
            for _, row in h_d.iterrows():
                target_week = row["week_start"]
                feats, sn_value = self._build_correction_features(target_week, did, history)
                if feats is None:
                    continue
                feats["residual_target"] = row["cases"] - sn_value
                rows.append(feats)

        if len(rows) < 30:
            self.model = None
            return

        train = pd.DataFrame(rows)
        X = pd.get_dummies(train.drop(columns=["residual_target"]),
                           columns=["district_id"], drop_first=False)
        for did in district_ids:
            col = f"district_id_{did}"
            if col not in X.columns:
                X[col] = 0
        X = X.reindex(sorted(X.columns), axis=1)
        self.feature_cols = X.columns.tolist()

        y = train["residual_target"]
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X, y)

        preds = self.model.predict(X)
        train["final_resid"] = train["residual_target"] - preds
        self._residuals_by_district = train.groupby("district_id").apply(
            lambda g: g["final_resid"].values
        ).to_dict()

    def predict(self, issue_week, horizon_weeks, district_id):
        target_week = issue_week + pd.Timedelta(weeks=horizon_weeks)
        sn_value = self._seasonal_naive_lookup(target_week, district_id, self.history)
        if sn_value is None:
            return {"mean": np.nan, "q10": np.nan, "q90": np.nan}

        if self.model is None:
            return {"mean": sn_value, "q10": sn_value * 0.5, "q90": sn_value * 1.5}

        feats, _ = self._build_correction_features(target_week, district_id, self.history)
        if feats is None:
            return {"mean": sn_value, "q10": sn_value * 0.5, "q90": sn_value * 1.5}

        X_row = pd.DataFrame([feats])
        X_row = pd.get_dummies(X_row, columns=["district_id"], drop_first=False)
        for col in self.feature_cols:
            if col not in X_row.columns:
                X_row[col] = 0
        X_row = X_row[self.feature_cols]

        correction = float(self.model.predict(X_row)[0])
        mean_pred = max(0.0, sn_value + correction)

        resids = self._residuals_by_district.get(district_id, None)
        if resids is not None and len(resids) > 10:
            q10 = max(0.0, mean_pred + float(np.quantile(resids, 0.10)))
            q90 = mean_pred + float(np.quantile(resids, 0.90))
        else:
            q10 = mean_pred * 0.5
            q90 = mean_pred * 1.5

        return {"mean": mean_pred, "q10": float(q10), "q90": float(q90)}


class SeasonalNaivePlusV2(Forecaster):
    """Variant of SeasonalNaivePlus where the anchor is the (district, week-of-year) mean."""
    name = "seasonal_naive_plus_v2"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = None
        self.feature_cols = None
        self.history = None
        self.seasonal_means = None
        self._residuals_by_district = {}

    def _build_seasonal_means(self, history):
        h = history.copy()
        h["woy"] = h["week_start"].dt.isocalendar().week
        return h.groupby(["district_id", "woy"])["cases"].mean().to_dict()

    def _seasonal_anchor(self, target_week, district_id):
        woy = target_week.isocalendar().week
        return self.seasonal_means.get((district_id, woy), None)

    def _build_features(self, target_week, district_id, hist):
        anchor = self._seasonal_anchor(target_week, district_id)
        if anchor is None:
            return None, None

        h_d = hist[(hist["district_id"] == district_id) &
                   (hist["week_start"] < target_week)].sort_values("week_start").tail(4)
        recent_drifts = []
        for _, row in h_d.iterrows():
            past_anchor = self._seasonal_anchor(row["week_start"], district_id)
            if past_anchor is not None:
                recent_drifts.append(row["cases"] - past_anchor)
        recent_drift = float(np.mean(recent_drifts)) if recent_drifts else 0.0

        def get_rain(weeks_back):
            ref = target_week - pd.Timedelta(weeks=weeks_back)
            r = hist[(hist["district_id"] == district_id) &
                     (hist["week_start"] == ref)]
            if r.empty:
                return 0.0
            v = r.iloc[0]["rainfall_mm"]
            return 0.0 if pd.isna(v) else float(v)

        feats = {
            "recent_drift": recent_drift,
            "rainfall_lag_4": get_rain(4),
            "rainfall_lag_6": get_rain(6),
            "district_id": district_id,
        }
        return feats, anchor

    def fit(self, history):
        self.history = history
        self.seasonal_means = self._build_seasonal_means(history)
        district_ids = sorted(history["district_id"].unique())

        rows = []
        for did in district_ids:
            h_d = history[history["district_id"] == did].sort_values("week_start")
            for _, row in h_d.iterrows():
                target_week = row["week_start"]
                feats, anchor = self._build_features(target_week, did, history)
                if feats is None:
                    continue
                feats["residual"] = row["cases"] - anchor
                rows.append(feats)

        if len(rows) < 30:
            self.model = None
            return

        train = pd.DataFrame(rows)
        X = pd.get_dummies(train.drop(columns=["residual"]),
                           columns=["district_id"], drop_first=False)
        for did in district_ids:
            col = f"district_id_{did}"
            if col not in X.columns:
                X[col] = 0
        X = X.reindex(sorted(X.columns), axis=1)
        self.feature_cols = X.columns.tolist()

        y = train["residual"]
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X, y)

        preds = self.model.predict(X)
        train["final_resid"] = train["residual"] - preds
        self._residuals_by_district = train.groupby("district_id").apply(
            lambda g: g["final_resid"].values
        ).to_dict()

    def predict(self, issue_week, horizon_weeks, district_id):
        target_week = issue_week + pd.Timedelta(weeks=horizon_weeks)
        anchor = self._seasonal_anchor(target_week, district_id)
        if anchor is None:
            return {"mean": np.nan, "q10": np.nan, "q90": np.nan}

        if self.model is None:
            return {"mean": anchor, "q10": anchor * 0.5, "q90": anchor * 1.5}

        feats, _ = self._build_features(target_week, district_id, self.history)
        if feats is None:
            return {"mean": anchor, "q10": anchor * 0.5, "q90": anchor * 1.5}

        X_row = pd.DataFrame([feats])
        X_row = pd.get_dummies(X_row, columns=["district_id"], drop_first=False)
        for col in self.feature_cols:
            if col not in X_row.columns:
                X_row[col] = 0
        X_row = X_row[self.feature_cols]

        correction = float(self.model.predict(X_row)[0])
        mean_pred = max(0.0, anchor + correction)

        resids = self._residuals_by_district.get(district_id, None)
        if resids is not None and len(resids) > 10:
            q10 = max(0.0, mean_pred + float(np.quantile(resids, 0.10)))
            q90 = mean_pred + float(np.quantile(resids, 0.90))
        else:
            q10 = mean_pred * 0.5
            q90 = mean_pred * 1.5

        return {"mean": mean_pred, "q10": float(q10), "q90": float(q90)}


def _fit_xgb_pooled(self, history, feature_builder):
    """Shared training body for the two XGBoost variants."""
    self.history = history
    district_ids = sorted(history["district_id"].unique())

    rows = []
    for did in district_ids:
        h_d = history[history["district_id"] == did].sort_values("week_start")
        for _, row in h_d.iterrows():
            target_week = row["week_start"]
            pop = row["population"]
            for h in (2, 4):
                issue_week = target_week - pd.Timedelta(weeks=h)
                available = history[(history["district_id"] == did) &
                                    (history["week_start"] <= issue_week)]
                if len(available) < 6:
                    continue
                feats = feature_builder(history, did, issue_week, h)
                if feats is None:
                    continue
                feats["target_rate"] = row["cases"] / pop * POP_PER_100K
                rows.append(feats)

    if len(rows) < 30:
        self.model = None
        return

    train = pd.DataFrame(rows)
    X = pd.get_dummies(train.drop(columns=["target_rate", "population"]),
                       columns=["district_id"], drop_first=False)
    for did in district_ids:
        col = f"district_id_{did}"
        if col not in X.columns:
            X[col] = 0
    X = X.reindex(sorted(X.columns), axis=1)
    self.feature_cols = X.columns.tolist()
    y = train["target_rate"]

    self.model = xgb.XGBRegressor(
        n_estimators=self.n_estimators,
        max_depth=self.max_depth,
        learning_rate=self.learning_rate,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    self.model.fit(X, y)

    preds = self.model.predict(X)
    train["final_resid_rate"] = train["target_rate"] - preds
    train["final_resid_count"] = train["final_resid_rate"] * train["population"] / POP_PER_100K
    self._residuals_by_district = train.groupby("district_id").apply(
        lambda g: g["final_resid_count"].values
    ).to_dict()


def _predict_xgb_pooled(self, issue_week, horizon_weeks, district_id, feature_builder):
    if self.model is None:
        return {"mean": np.nan, "q10": np.nan, "q90": np.nan}

    feats = feature_builder(self.history, district_id, issue_week, horizon_weeks)
    if feats is None:
        return {"mean": np.nan, "q10": np.nan, "q90": np.nan}

    pop = feats["population"]
    X_row = pd.DataFrame([feats]).drop(columns=["population"])
    X_row = pd.get_dummies(X_row, columns=["district_id"], drop_first=False)
    for col in self.feature_cols:
        if col not in X_row.columns:
            X_row[col] = 0
    X_row = X_row[self.feature_cols]

    rate_pred = float(self.model.predict(X_row)[0])
    mean_pred = max(0.0, rate_pred * pop / POP_PER_100K)

    resids = self._residuals_by_district.get(district_id, None)
    if resids is not None and len(resids) > 10:
        q10 = max(0.0, mean_pred + float(np.quantile(resids, 0.10)))
        q90 = mean_pred + float(np.quantile(resids, 0.90))
    else:
        q10 = mean_pred * 0.5
        q90 = mean_pred * 1.5

    return {"mean": mean_pred, "q10": float(q10), "q90": float(q90)}


class XGBoostForecaster(Forecaster):
    """Pooled XGBoost on rate-per-100k with one-hot district."""
    name = "xgboost_pooled"

    def __init__(self, n_estimators: int = 200, max_depth: int = 3, learning_rate: float = 0.05):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None
        self.feature_cols = None
        self.history = None
        self._residuals_by_district = {}

    def fit(self, history):
        _fit_xgb_pooled(self, history, build_xgb_features)

    def predict(self, issue_week, horizon_weeks, district_id):
        return _predict_xgb_pooled(self, issue_week, horizon_weeks, district_id, build_xgb_features)


class XGBoostForecasterV2(Forecaster):
    """XGBoost variant that adds the seasonal-naive prediction as a feature."""
    name = "xgboost_with_sn"

    def __init__(self, n_estimators: int = 200, max_depth: int = 3, learning_rate: float = 0.05):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None
        self.feature_cols = None
        self.history = None
        self._residuals_by_district = {}

    def fit(self, history):
        _fit_xgb_pooled(self, history, build_xgb_features_v2)

    def predict(self, issue_week, horizon_weeks, district_id):
        return _predict_xgb_pooled(self, issue_week, horizon_weeks, district_id, build_xgb_features_v2)
