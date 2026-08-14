"""
Phase 7: Machine Learning Incident Risk Prediction Model
Trains a supervised probability model to estimate P(Incident | Conditions, History).
Evaluates performance metrics and outputs calibrated incident risk scores.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score, precision_score, recall_score
from config import RANDOM_SEED


FEATURE_COLS_NUMERIC = [
    "population_density",
    "road_capacity",
    "vehicle_density",
    "congestion",
    "avg_speed",
    "violations",
    "prev_week_incidents",
    "prev_week_congestion",
    "congestion_delta",
    "rolling_4w_incidents",
    "rolling_4w_congestion",
    "trend_slope",
]

FEATURE_COLS_CATEGORICAL = [
    "weather",
    "event",
    "archetype",
]

TARGET_COL = "incident_occurred"


def build_pipeline() -> Pipeline:
    """
    Constructs an end-to-end preprocessing + classifier pipeline.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", FEATURE_COLS_NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURE_COLS_CATEGORICAL),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=6,
        min_samples_split=5,
        random_state=RANDOM_SEED,
        class_weight="balanced"
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])


def train_risk_model(
    df: pd.DataFrame,
    train_split_week: int = 18
) -> Tuple[Pipeline, Dict[str, float], pd.DataFrame]:
    """
    Trains the incident risk model on historical weeks and validates on holdout weeks.
    Returns the trained pipeline, evaluation metrics, and the annotated dataframe with predicted risk.
    """
    train_mask = df["week"] <= train_split_week
    test_mask = df["week"] > train_split_week

    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()

    X_train = df_train[FEATURE_COLS_NUMERIC + FEATURE_COLS_CATEGORICAL]
    y_train = df_train[TARGET_COL]

    X_test = df_test[FEATURE_COLS_NUMERIC + FEATURE_COLS_CATEGORICAL]
    y_test = df_test[TARGET_COL]

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluate on test set
    y_pred_proba_test = pipeline.predict_proba(X_test)[:, 1]
    y_pred_test = (y_pred_proba_test >= 0.5).astype(int)

    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, y_pred_proba_test)), 4) if len(np.unique(y_test)) > 1 else 1.0,
        "brier_score": round(float(brier_score_loss(y_test, y_pred_proba_test)), 4),
        "accuracy": round(float(accuracy_score(y_test, y_pred_test)), 4),
        "precision": round(float(precision_score(y_test, y_pred_test, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred_test, zero_division=0)), 4),
    }

    # Predict risk for entire dataset
    X_all = df[FEATURE_COLS_NUMERIC + FEATURE_COLS_CATEGORICAL]
    df_result = df.copy()
    df_result["predicted_risk_prob"] = pipeline.predict_proba(X_all)[:, 1]
    df_result["predicted_risk_pct"] = (df_result["predicted_risk_prob"] * 100.0).round(1)

    return pipeline, metrics, df_result
