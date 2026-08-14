"""
Phases 8 & 9: Exposure and Multi-Factor Priority Scoring Engine
Implements the core project philosophy: Risk != Exposure != Priority.
- Risk: Predicted probability of an incident occurring (%)
- Exposure: Citizen & vehicle population affected (0-100 score)
- Trend: Momentum of risk/congestion changes (Increasing / Decreasing / Stable)
- Priority: Actionable decision score directing limited municipal intervention resources
"""

import pandas as pd
import numpy as np
from config import (
    WEIGHT_EXPOSURE_POP,
    WEIGHT_EXPOSURE_VEH,
    WEIGHT_PRIORITY_RISK,
    WEIGHT_PRIORITY_EXPOSURE,
    WEIGHT_PRIORITY_TREND,
    PRIORITY_TIERS,
)


def calculate_exposure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 8: Computes normalized Exposure score (0 to 100) per observation.
    Exposure = w_pop * Norm(Population) + w_veh * Norm(Vehicles)
    """
    df_out = df.copy()

    # Global min-max scaling for consistent baseline reference across weeks
    p_min, p_max = df_out["population_density"].min(), df_out["population_density"].max()
    v_min, v_max = df_out["vehicle_density"].min(), df_out["vehicle_density"].max()

    p_norm = 100.0 * (df_out["population_density"] - p_min) / max(1.0, (p_max - p_min))
    v_norm = 100.0 * (df_out["vehicle_density"] - v_min) / max(1.0, (v_max - v_min))

    df_out["exposure_score"] = (WEIGHT_EXPOSURE_POP * p_norm + WEIGHT_EXPOSURE_VEH * v_norm).round(1)
    return df_out


def calculate_priority(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 9: Computes final Priority score (0 to 100) and assigns actionable urgency tiers.
    Priority = w_risk * Risk + w_exp * Exposure + w_trend * Trend
    """
    df_out = calculate_exposure(df)

    # Convert trend slope into a 0 - 100 momentum score (50 = baseline stable)
    trend_score = np.clip(50.0 + df_out["trend_slope"] * 6.0, 10.0, 95.0)
    df_out["trend_score"] = trend_score.round(1)

    # Composite Priority Index
    raw_priority = (
        WEIGHT_PRIORITY_RISK * df_out["predicted_risk_pct"]
        + WEIGHT_PRIORITY_EXPOSURE * df_out["exposure_score"]
        + WEIGHT_PRIORITY_TREND * df_out["trend_score"]
    )
    df_out["priority_score"] = np.clip(raw_priority, 0.0, 100.0).round(1)

    # Assign Priority Tiers & Badges
    def get_tier_badge(score: float):
        for threshold, tier, badge in PRIORITY_TIERS:
            if score >= threshold:
                return tier, badge
        return "MODERATE", "[MODERATE]"

    tier_results = df_out["priority_score"].apply(get_tier_badge)
    df_out["priority_tier"] = [t[0] for t in tier_results]
    df_out["priority_badge"] = [t[1] for t in tier_results]

    return df_out
