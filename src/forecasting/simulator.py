"""Rolling-origin simulator: walk forward one Monday at a time."""
import numpy as np
import pandas as pd


def run_simulation(model, full_panel: pd.DataFrame,
                   first_issue_week: pd.Timestamp, last_issue_week: pd.Timestamp,
                   horizons=(2, 4), refit_each_step: bool = False) -> pd.DataFrame:
    """
    Walk forward week by week. At every Monday, ask the model to forecast
    every horizon for every district. Compare to truth.

    The model only ever sees data with week_start <= issue_week. If
    `refit_each_step` is False, the model is fit once on data strictly before
    `first_issue_week` and then has its `.history` swapped in at each step
    (cheap for the lookup-based naive models). If True, `.fit()` is called
    each week with the expanding history.
    """
    district_ids = sorted(full_panel["district_id"].unique())
    issue_weeks = pd.date_range(start=first_issue_week, end=last_issue_week, freq="W-MON")

    if not refit_each_step:
        initial_history = full_panel[full_panel["week_start"] < first_issue_week].copy()
        model.fit(initial_history)

    rows = []
    for issue_week in issue_weeks:
        available_history = full_panel[full_panel["week_start"] <= issue_week].copy()

        if refit_each_step:
            model.fit(available_history)
        else:
            model.history = available_history

        for district_id in district_ids:
            for h in horizons:
                target_week = issue_week + pd.Timedelta(weeks=h)
                pred = model.predict(issue_week, h, district_id)

                actual_rows = full_panel[
                    (full_panel["district_id"] == district_id) &
                    (full_panel["week_start"] == target_week)
                ]
                actual = float(actual_rows["cases"].iloc[0]) if not actual_rows.empty else np.nan

                rows.append({
                    "issue_iso_week": issue_week.strftime("%G-W%V"),
                    "target_iso_week": target_week.strftime("%G-W%V"),
                    "issue_week_dt": issue_week,
                    "target_week_dt": target_week,
                    "district_id": district_id,
                    "horizon_weeks": h,
                    "mean": pred["mean"],
                    "q10": pred["q10"],
                    "q90": pred["q90"],
                    "actual": actual,
                })

    return pd.DataFrame(rows)
