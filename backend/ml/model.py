"""Lightweight tree-based ranking model for Phase 3B recommendation training."""

from __future__ import annotations

from statistics import fmean
from typing import Iterable

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from backend.evaluation.baselines import BASELINES
from backend.evaluation.metrics import ranking_metrics

IDENTIFIER_COLUMNS = [
    "customer_id",
    "target_order_id",
    "target_order_number",
    "candidate_product_id",
    "product_id",
]
MODEL_EXCLUDED_COLUMNS = IDENTIFIER_COLUMNS + ["product_name", "y"]
DEFAULT_K_VALUES = (5, 10, 20)
DEFAULT_RANDOM_STATE = 42


def get_model_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return the numeric model feature columns after excluding IDs and leakage."""
    if frame.empty:
        raise ValueError("Model input is empty; no feature columns can be inferred.")
    features: list[str] = []
    for column in frame.columns:
        if column in MODEL_EXCLUDED_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            features.append(column)
    if not features:
        raise ValueError("No numeric feature columns remain after excluding identifiers and leakage.")
    return features


def validate_training_inputs(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> list[str]:
    """Validate the Phase 3A train/validation frame shapes."""
    if train_df.empty:
        raise ValueError("Training data is empty.")
    if validation_df.empty:
        raise ValueError("Validation data is empty.")
    required_columns = [
        "customer_id",
        "target_order_id",
        "target_order_number",
        "candidate_product_id",
        "y",
    ]
    missing_train = [column for column in required_columns if column not in train_df.columns]
    missing_validation = [column for column in required_columns if column not in validation_df.columns]
    if missing_train:
        raise ValueError(f"Training data is missing required columns: {missing_train}")
    if missing_validation:
        raise ValueError(f"Validation data is missing required columns: {missing_validation}")
    feature_columns = get_model_feature_columns(train_df)
    missing_features = [column for column in feature_columns if column not in validation_df.columns]
    if missing_features:
        raise ValueError(f"Validation data is missing expected features: {missing_features}")
    if train_df["y"].nunique() < 2:
        raise ValueError("Training data must contain both positive and negative labels.")
    return feature_columns


def compute_candidate_recall(frame: pd.DataFrame) -> float:
    """Mean candidate recall across all validation targets before ranking."""
    if frame.empty:
        return 0.0
    scores: list[float] = []
    for _, group in frame.groupby(["customer_id", "target_order_id"], sort=True):
        relevant = set(group.loc[group["y"] == 1, "candidate_product_id"].astype(int))
        candidates = set(group["candidate_product_id"].astype(int))
        if relevant:
            scores.append(len(candidates & relevant) / len(relevant))
        else:
            scores.append(0.0)
    return float(fmean(scores)) if scores else 0.0


def rank_validation_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Sort every validation target's candidates by model score then product ID."""
    if predictions.empty:
        return predictions.copy()
    required = {"customer_id", "target_order_id", "candidate_product_id", "model_score"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Predictions are missing required ranking columns: {missing}")
    ranked = predictions.copy()
    ranked["model_score"] = pd.to_numeric(ranked["model_score"], errors="coerce")
    ranked["candidate_product_id"] = ranked["candidate_product_id"].astype(int)
    return ranked.sort_values(
        ["customer_id", "target_order_id", "model_score", "candidate_product_id"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def compute_grouped_ranking_metrics(
    predictions: pd.DataFrame,
    *,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
) -> dict[str, float]:
    """Aggregate ranking metrics across all customer-target groups."""
    if predictions.empty:
        return {f"precision_at_{k}": 0.0 for k in k_values} | {
            f"recall_at_{k}": 0.0 for k in k_values
        } | {
            f"f1_at_{k}": 0.0 for k in k_values
        } | {
            f"map_at_{k}": 0.0 for k in k_values
        } | {
            f"ndcg_at_{k}": 0.0 for k in k_values
        }
    k_values = tuple(int(k) for k in k_values)
    rows: list[dict[str, float]] = []
    for _, group in predictions.groupby(["customer_id", "target_order_id"], sort=True):
        ordered = group.sort_values(
            ["model_score", "candidate_product_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        recommended = ordered["candidate_product_id"].astype(int).tolist()
        relevant = set(ordered.loc[ordered["y"] == 1, "candidate_product_id"].astype(int))
        rows.append(ranking_metrics(recommended, relevant, k_values))

    summary: dict[str, float] = {}
    for metric_name in ("precision", "recall", "f1", "map", "ndcg"):
        for k in k_values:
            values = [row[f"{metric_name}_at_{k}"] for row in rows]
            summary[f"{metric_name}_at_{k}"] = float(fmean(values)) if values else 0.0
    return summary


def fit_ranking_model(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
) -> dict[str, object]:
    """Train and evaluate a lightweight tree-based ranking model."""
    feature_columns = validate_training_inputs(train_df, validation_df)
    X_train = train_df[feature_columns].copy()
    X_valid = validation_df[feature_columns].copy()
    y_train = train_df["y"].astype(int).copy()
    if X_train.isnull().all().all():
        raise ValueError("Training feature matrix is entirely null.")

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=None,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.0,
        random_state=random_state,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    validation_predictions = validation_df.copy()
    validation_predictions["model_score"] = model.predict_proba(X_valid)[:, 1].astype(float)
    ranked_predictions = rank_validation_predictions(validation_predictions)
    metrics = compute_grouped_ranking_metrics(ranked_predictions, k_values=k_values)
    candidate_recall = compute_candidate_recall(validation_df)
    metadata = {
        "model_type": "HistGradientBoostingClassifier",
        "training_dataset_path": "data/processed/ml_dev/train.csv",
        "validation_dataset_path": "data/processed/ml_dev/validation.csv",
        "feature_columns": feature_columns,
        "excluded_identifier_columns": IDENTIFIER_COLUMNS,
        "excluded_leakage_columns": ["product_name"],
        "training_row_count": int(len(train_df)),
        "validation_row_count": int(len(validation_df)),
        "positive_training_rows": int(train_df["y"].sum()),
        "negative_training_rows": int((train_df["y"] == 0).sum()),
        "positive_validation_rows": int(validation_df["y"].sum()),
        "negative_validation_rows": int((validation_df["y"] == 0).sum()),
        "model_configuration": {
            "learning_rate": 0.05,
            "max_depth": None,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 0.0,
            "random_state": random_state,
            "class_weight": "balanced",
        },
        "random_state": random_state,
        "k_values": [int(k) for k in k_values],
        "candidate_recall": candidate_recall,
    }
    return {
        "model": model,
        "feature_columns": feature_columns,
        "training_rows": train_df.copy(),
        "validation_predictions": validation_predictions,
        "ranked_predictions": ranked_predictions,
        "metrics": metrics,
        "metadata": metadata,
    }


def evaluate_baselines_on_validation(
    history,
    validation_df: pd.DataFrame,
    *,
    baseline_names: Iterable[str] = ("previously_purchased", "customer_frequency"),
    k_values: Iterable[int] = DEFAULT_K_VALUES,
) -> dict[str, dict[str, float]]:
    """Compute Phase 2C baseline metrics on the same validation targets."""
    if validation_df.empty:
        raise ValueError("Validation data is empty; cannot evaluate baselines.")
    results: dict[str, dict[str, float]] = {}
    selected = list(baseline_names)
    for baseline_name in selected:
        if baseline_name not in BASELINES:
            raise ValueError(f"Unsupported baseline: {baseline_name}")
        values: list[dict[str, float]] = []
        for _, group in validation_df.groupby(["customer_id", "target_order_id"], sort=True):
            customer_id = int(group["customer_id"].iloc[0])
            target_order_id = int(group["target_order_id"].iloc[0])
            target_order_number = int(group["target_order_number"].iloc[0])
            candidate_ids = sorted(group["candidate_product_id"].astype(int).unique())
            relevant = set(group.loc[group["y"] == 1, "candidate_product_id"].astype(int))
            history_context = history.history_for_target(customer_id, target_order_id, target_order_number)
            ranking = BASELINES[baseline_name](
                history,
                customer_id,
                target_order_id,
                target_order_number,
                candidate_ids,
                history_context=history_context,
            )
            values.append(ranking_metrics(ranking, relevant, tuple(int(k) for k in k_values)))
        summary: dict[str, float] = {}
        for metric_name in ("precision", "recall", "f1", "map", "ndcg"):
            for k in (int(k) for k in k_values):
                metric_values = [row[f"{metric_name}_at_{k}"] for row in values]
                summary[f"{metric_name}_at_{k}"] = float(fmean(metric_values)) if metric_values else 0.0
        results[baseline_name] = summary
    return results
