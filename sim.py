"""
Traffic Intelligence & Intervention Priority System
End-to-End Simulation, ML Risk Modeling, Exposure Weighting, and Counterfactual Intervention Pipeline.
"""

import os
import pandas as pd
from config import NUM_ZONES, NUM_WEEKS, RANDOM_SEED
from simulation import simulate_city_traffic
from features import compute_temporal_features
from model import train_risk_model
from scoring import calculate_priority
from intervention import simulate_intervention, recommend_intervention
from dashboard import render_dashboard


def run_pipeline():
    print("\n=======================================================")
    print(" TRAFFIC INTELLIGENCE & INTERVENTION SIMULATION SYSTEM ")
    print("=======================================================\n")

    # ---------------------------------------------------------
    # Phases 1 - 5: Synthetic City Simulation
    # ---------------------------------------------------------
    print(f"[1/5] Simulating synthetic city ({NUM_ZONES} Zones x {NUM_WEEKS} Weeks = {NUM_ZONES * NUM_WEEKS} observations)...")
    raw_df = simulate_city_traffic(num_zones=NUM_ZONES, num_weeks=NUM_WEEKS, seed=RANDOM_SEED)

    # ---------------------------------------------------------
    # Phase 6: Temporal & Historical Feature Engineering
    # ---------------------------------------------------------
    print("[2/5] Engineering temporal features, lag indicators, and 4-week rolling trends...")
    df_features = compute_temporal_features(raw_df)

    # ---------------------------------------------------------
    # Phase 7: Machine Learning Incident Risk Prediction Model
    # ---------------------------------------------------------
    train_split = int(NUM_WEEKS * 0.75)
    print(f"[3/5] Training ML Risk Classifier on Weeks 1..{train_split} & evaluating on Weeks {train_split + 1}..{NUM_WEEKS}...")
    pipeline, metrics, df_risk = train_risk_model(df_features, train_split_week=train_split)
    print(f"      -> Model Holdout ROC-AUC: {metrics['roc_auc']:.3f} | Accuracy: {metrics['accuracy'] * 100:.1f}%")

    # ---------------------------------------------------------
    # Phases 8 & 9: Exposure & Priority Scoring
    # ---------------------------------------------------------
    print("[4/5] Computing Exposure index and multi-factor Priority rankings (Risk x Exposure x Trend)...")
    df_scored = calculate_priority(df_risk)

    # ---------------------------------------------------------
    # Phase 10: Counterfactual Intervention Simulation & Executive Dashboard
    # ---------------------------------------------------------
    latest_week = NUM_WEEKS
    df_latest = df_scored[df_scored["week"] == latest_week].copy()

    # Identify top priority zone for municipal intervention
    top_zone_row = df_latest.sort_values(by="priority_score", ascending=False).iloc[0]
    recommended_policy = recommend_intervention(top_zone_row)
    print(f"[5/5] Simulating counterfactual intervention '{recommended_policy}' on #1 priority ({top_zone_row['zone_id']})...")

    intervention_res, row_after = simulate_intervention(
        zone_row=top_zone_row,
        model_pipeline=pipeline,
        intervention_key=recommended_policy
    )

    # Render Executive Briefing Dashboard
    render_dashboard(
        df_latest=df_latest,
        metrics=metrics,
        intervention_results=[intervention_res],
        current_week=latest_week
    )

    # Save output datasets
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    output_csv = os.path.join(data_dir, "traffic_simulation_full.csv")
    df_scored.to_csv(output_csv, index=False)

    # Backwards compatibility with initial Phase 1 export
    phase1_csv = os.path.join(data_dir, "phase1_data.csv")
    df_scored[["zone_id", "week", "population_density", "road_capacity"]].to_csv(phase1_csv, index=False)

    print(f"[+] Full simulation dataset ({len(df_scored)} rows) saved to: {output_csv}")


if __name__ == "__main__":
    run_pipeline()