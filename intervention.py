"""
Phase 10a: Counterfactual Intervention Simulator
Simulates targeted municipal interventions on high-priority zones and evaluates before vs after metrics.
Supports:
1. Signal Timing Optimization (Adaptive traffic signal control)
2. Targeted Police Patrol & Speed Enforcement
3. Dynamic Lane Management & Diversions
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.pipeline import Pipeline
from scoring import calculate_priority
from model import FEATURE_COLS_NUMERIC, FEATURE_COLS_CATEGORICAL


INTERVENTIONS = {
    "SIGNAL_OPTIMIZATION": {
        "name": "Adaptive Signal Timing Optimization",
        "description": "Adjusts green light cycles to relieve bottlenecks; reduces congestion and smooths flow.",
        "congestion_mult": 0.78,    # 22% reduction in congestion
        "speed_add": 6.5,           # +6.5 km/h improvement in avg speed
        "violations_mult": 0.85,    # 15% reduction in red-light violations
    },
    "PATROL_ENFORCEMENT": {
        "name": "Targeted Traffic Police & Speed Enforcement",
        "description": "Deploys visible patrol units to curb aggressive driving and violations.",
        "congestion_mult": 0.92,
        "speed_add": 2.0,
        "violations_mult": 0.45,    # 55% reduction in violations
    },
    "DYNAMIC_LANE_MANAGEMENT": {
        "name": "Dynamic Lane Reversible Flow & Diversion",
        "description": "Opens peak-direction reversible lanes and reroutes non-local traffic.",
        "congestion_mult": 0.72,    # 28% reduction in congestion
        "speed_add": 8.0,
        "violations_mult": 0.80,
    },
}


def recommend_intervention(zone_row: pd.Series) -> str:
    """
    Selects the optimal intervention strategy based on the primary risk driver in the zone.
    """
    if zone_row["violations"] >= 8:
        return "PATROL_ENFORCEMENT"
    elif zone_row["congestion"] >= 70:
        return "DYNAMIC_LANE_MANAGEMENT"
    else:
        return "SIGNAL_OPTIMIZATION"


def simulate_intervention(
    zone_row: pd.Series,
    model_pipeline: Pipeline,
    intervention_key: str = None
) -> Tuple[Dict[str, Any], pd.Series]:
    """
    Applies counterfactual intervention to a zone row, recalculates ML predicted risk,
    exposure, and priority score, and returns a detailed before/after comparison dictionary.
    """
    if intervention_key is None:
        intervention_key = recommend_intervention(zone_row)

    cfg = INTERVENTIONS[intervention_key]

    # Create modified counterfactual row
    row_after = zone_row.copy()
    row_after["congestion"] = max(5.0, round(float(zone_row["congestion"]) * cfg["congestion_mult"], 1))
    row_after["avg_speed"] = min(60.0, round(float(zone_row["avg_speed"]) + cfg["speed_add"], 1))
    row_after["violations"] = max(0, int(round(float(zone_row["violations"]) * cfg["violations_mult"])))
    row_after["congestion_delta"] = round(row_after["congestion"] - float(zone_row["prev_week_congestion"]), 1)
    row_after["trend_slope"] = round(float(zone_row["trend_slope"]) - 3.5, 2)

    # Re-evaluate with ML model
    df_after_temp = pd.DataFrame([row_after])
    X_after = df_after_temp[FEATURE_COLS_NUMERIC + FEATURE_COLS_CATEGORICAL]
    new_risk_prob = float(model_pipeline.predict_proba(X_after)[0, 1])
    row_after["predicted_risk_prob"] = new_risk_prob
    row_after["predicted_risk_pct"] = round(new_risk_prob * 100.0, 1)

    # Recalculate priority
    df_after_scored = calculate_priority(pd.DataFrame([row_after]))
    row_after = df_after_scored.iloc[0]

    comparison = {
        "zone_id": zone_row["zone_id"],
        "week": int(zone_row["week"]),
        "intervention_key": intervention_key,
        "intervention_name": cfg["name"],
        "intervention_desc": cfg["description"],
        "before": {
            "congestion": float(zone_row["congestion"]),
            "avg_speed": float(zone_row["avg_speed"]),
            "violations": int(zone_row["violations"]),
            "risk_pct": float(zone_row["predicted_risk_pct"]),
            "exposure": float(zone_row["exposure_score"]),
            "priority": float(zone_row["priority_score"]),
            "tier": str(zone_row["priority_tier"]),
        },
        "after": {
            "congestion": float(row_after["congestion"]),
            "avg_speed": float(row_after["avg_speed"]),
            "violations": int(row_after["violations"]),
            "risk_pct": float(row_after["predicted_risk_pct"]),
            "exposure": float(row_after["exposure_score"]),
            "priority": float(row_after["priority_score"]),
            "tier": str(row_after["priority_tier"]),
        },
        "delta": {
            "risk_pct_reduction": round(float(zone_row["predicted_risk_pct"]) - float(row_after["predicted_risk_pct"]), 1),
            "congestion_reduction": round(float(zone_row["congestion"]) - float(row_after["congestion"]), 1),
            "speed_gain": round(float(row_after["avg_speed"]) - float(zone_row["avg_speed"]), 1),
            "priority_reduction": round(float(zone_row["priority_score"]) - float(row_after["priority_score"]), 1),
        }
    }

    return comparison, row_after
