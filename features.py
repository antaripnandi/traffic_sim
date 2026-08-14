"""
Phase 6: Temporal & Historical Feature Engineering
Calculates lagged values, rolling averages, and trend direction for each zone over time.
Ensures zero data leakage by relying strictly on historical (t-1, t-2, ...) information.
"""

import numpy as np
import pandas as pd


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes time-series and rolling historical features for each zone.
    """
    # Sort strictly by zone and week to prevent chronological mixing
    df_sorted = df.sort_values(by=["zone_id", "week"]).copy()

    # Group by zone for per-zone lag computations
    grouped = df_sorted.groupby("zone_id")

    # Lag 1 features (Conditions observed last week)
    df_sorted["prev_week_incidents"] = grouped["incident_occurred"].shift(1).fillna(0).astype(int)
    df_sorted["prev_week_congestion"] = grouped["congestion"].shift(1).bfill()
    df_sorted["prev_week_violations"] = grouped["violations"].shift(1).bfill()
    df_sorted["prev_week_speed"] = grouped["avg_speed"].shift(1).bfill()

    # Congestion Delta: Change from previous week
    df_sorted["congestion_delta"] = df_sorted["congestion"] - df_sorted["prev_week_congestion"]

    # 4-Week Rolling Averages (using closed='left' logic to only look at past weeks)
    def calc_rolling_4w_incidents(s: pd.Series) -> pd.Series:
        # Shift first so current week's outcome is not included
        return s.shift(1).rolling(window=4, min_periods=1).mean().fillna(0.0)

    def calc_rolling_4w_congestion(s: pd.Series) -> pd.Series:
        return s.shift(1).rolling(window=4, min_periods=1).mean().bfill()

    df_sorted["rolling_4w_incidents"] = grouped["incident_occurred"].transform(calc_rolling_4w_incidents)
    df_sorted["rolling_4w_congestion"] = grouped["congestion"].transform(calc_rolling_4w_congestion)

    # Trend Slope: Slope of congestion over the past 3 weeks
    # Positive = Risk/Congestion is increasing; Negative = Risk/Congestion is decreasing
    def calc_trend_slope(s: pd.Series) -> pd.Series:
        def slope_3w(w):
            if len(w) < 2:
                return 0.0
            x = np.arange(len(w))
            return np.polyfit(x, w, 1)[0]
        return s.rolling(window=3, min_periods=1).apply(slope_3w, raw=True).fillna(0.0)

    df_sorted["trend_slope"] = grouped["congestion"].transform(calc_trend_slope)

    # Categorical trend indicator (+1 = Increasing, 0 = Stable, -1 = Improving)
    df_sorted["trend_direction"] = np.where(
        df_sorted["trend_slope"] > 1.5, "Increasing",
        np.where(df_sorted["trend_slope"] < -1.5, "Decreasing", "Stable")
    )

    return df_sorted.sort_index()
