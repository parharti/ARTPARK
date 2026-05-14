"""Data loading and panel construction for the dengue forecasting project."""
from pathlib import Path
import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def iso_week_to_date(iso_week_str: str) -> pd.Timestamp:
    return pd.to_datetime(iso_week_str + "-1", format="%G-W%V-%u")


def load_data(data_dir: Path | str = DEFAULT_DATA_DIR):
    """Load the four raw CSVs and return (cases, weather, districts, baseline)."""
    data_dir = Path(data_dir)
    cases = pd.read_csv(data_dir / "cases_weekly.csv")
    weather = pd.read_csv(data_dir / "weather_weekly.csv")
    districts = pd.read_csv(data_dir / "district_registry.csv")
    baseline_path = data_dir.parent / "submissions" / "baseline_seasonal_naive.csv"
    baseline = pd.read_csv(baseline_path)
    return cases, weather, districts, baseline


def build_panel(cases: pd.DataFrame, weather: pd.DataFrame, districts: pd.DataFrame) -> pd.DataFrame:
    """Merge the three sources into one long panel keyed by (district_id, week_start)."""
    cases = cases.copy()
    weather = weather.copy()
    cases["week_start"] = cases["iso_week"].apply(iso_week_to_date)
    weather["week_start"] = weather["iso_week"].apply(iso_week_to_date)

    df = cases.merge(weather, on=["iso_week", "district_id", "week_start"], how="outer")
    df = df.merge(
        districts[["district_id", "canonical_name", "tier", "population"]],
        on="district_id",
        how="left",
    )
    df = df.sort_values(["district_id", "week_start"]).reset_index(drop=True)
    return df
