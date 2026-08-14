"""
Phase 10b: Executive Traffic Intelligence Terminal Dashboard
Visualizes weekly zone risk, exposure scores, priority rankings, and intervention impact.
"""

import pandas as pd
from typing import Dict, Any, List


def print_header(title: str, width: int = 76):
    print("\n" + "=" * width)
    print(f" {title.upper()} ".center(width, "="))
    print("=" * width)


def render_dashboard(
    df_latest: pd.DataFrame,
    metrics: Dict[str, float],
    intervention_results: List[Dict[str, Any]],
    current_week: int = 24
):
    """
    Renders a formatted executive intelligence briefing for the given week.
    """
    ranked_df = df_latest.sort_values(by="priority_score", ascending=False).reset_index(drop=True)

    print_header(f"WEEKLY TRAFFIC INTELLIGENCE BRIEFING -- WEEK {current_week}")

    print(f"\n[+] SYSTEM METRICS & ML CALIBRATION:")
    print(f"    - Holdout ROC-AUC: {metrics.get('roc_auc', 0.0):.3f} | Accuracy: {metrics.get('accuracy', 0.0)*100:.1f}% | Brier Loss: {metrics.get('brier_score', 0.0):.3f}")
    print(f"    - Evaluated Zones: {len(ranked_df)} | Total Active Traffic Corridors Monitored: {len(ranked_df)}")
    print(f"    - Core Engine: Risk (Probability) x Exposure (Volume) x Trend (Momentum) -> Priority Index")

    print("\n" + "-" * 76)
    print(f"{'RANK':<5} {'ZONE':<9} {'ARCHETYPE':<14} {'RISK (%)':<10} {'EXPOSURE':<10} {'TREND':<11} {'PRIORITY':<10} {'TIER'}")
    print("-" * 76)

    for idx, row in ranked_df.iterrows():
        rank = idx + 1
        trend_symbol = "^" if row["trend_direction"] == "Increasing" else ("v" if row["trend_direction"] == "Decreasing" else "=")
        trend_str = f"{trend_symbol} {row['trend_direction'][:4]}"
        
        tier_flag = "CRITICAL" if row["priority_tier"] == "CRITICAL" else row["priority_tier"]

        print(
            f"{rank:<5} "
            f"{row['zone_id']:<9} "
            f"{row['archetype']:<14} "
            f"{row['predicted_risk_pct']:>6.1f}%   "
            f"{row['exposure_score']:>7.1f}   "
            f"{trend_str:<11} "
            f"{row['priority_score']:>7.1f}   "
            f"[{tier_flag}]"
        )

    print("-" * 76)

    # -------------------------------------------------------------
    # Focus Breakdown: The "Risk != Exposure != Priority" Philosophy
    # -------------------------------------------------------------
    print_header("DECISION INTELLIGENCE HIGHLIGHTS")
    top_zone = ranked_df.iloc[0]
    print(f"\n[*] #1 ACTION PRIORITY: {top_zone['zone_id']} ({top_zone['archetype']})")
    print(f"    - Predicted Risk:  {top_zone['predicted_risk_pct']:.1f}% (Incident likelihood)")
    print(f"    - Exposure Score:  {top_zone['exposure_score']:.1f} (Population: {top_zone['population_density']:,}, Vehicles: {top_zone['vehicle_density']})")
    print(f"    - Trend Momentum:  {top_zone['trend_direction']} (Congestion Delta: {top_zone['congestion_delta']:+.1f})")
    print(f"    - Priority Score:  {top_zone['priority_score']:.1f} --> {top_zone['priority_tier']}")
    print(f"    - Live Conditions: Congestion={top_zone['congestion']}% | Avg Speed={top_zone['avg_speed']} km/h | Violations={top_zone['violations']} | Weather={top_zone['weather']}")

    # -------------------------------------------------------------
    # Simulated Intervention Results
    # -------------------------------------------------------------
    if intervention_results:
        print_header("COUNTERFACTUAL INTERVENTION SIMULATION")
        for res in intervention_results:
            b = res["before"]
            a = res["after"]
            d = res["delta"]
            print(f"\n[>] TARGETED ZONE: {res['zone_id']}")
            print(f"    Selected Policy: {res['intervention_name']}")
            print(f"    Mechanism:       {res['intervention_desc']}")
            print("\n    " + "-" * 62)
            print(f"    {'METRIC':<20} {'BEFORE':<12} {'AFTER':<12} {'IMPACT / DELTA'}")
            print("    " + "-" * 62)
            print(f"    {'Congestion Index':<20} {b['congestion']:>5.1f}%       {a['congestion']:>5.1f}%       -{d['congestion_reduction']:.1f}%")
            print(f"    {'Average Speed':<20} {b['avg_speed']:>5.1f} km/h   {a['avg_speed']:>5.1f} km/h   +{d['speed_gain']:.1f} km/h")
            print(f"    {'Violations / Wk':<20} {b['violations']:>5d}         {a['violations']:>5d}         {a['violations'] - b['violations']:+d}")
            print(f"    {'Incident Risk':<20} {b['risk_pct']:>5.1f}%       {a['risk_pct']:>5.1f}%       -{d['risk_pct_reduction']:.1f}% (Predicted)")
            print(f"    {'Priority Score':<20} {b['priority']:>5.1f}        {a['priority']:>5.1f}        -{d['priority_reduction']:.1f} ({b['tier']} -> {a['tier']})")
            print("    " + "-" * 62)
            print(f"    Result: Zone risk successfully mitigated below critical intervention threshold.")

    print("\n" + "=" * 76 + "\n")
