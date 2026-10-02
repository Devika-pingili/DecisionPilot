"""Train the first lightweight ranking model from the bounded Phase 3A dataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.features.history import HistoryStore
from backend.ml.model import DEFAULT_K_VALUES, evaluate_baselines_on_validation, fit_ranking_model


def main() -> None:
    repo_root = REPO_ROOT
    data_dir = repo_root / "data"
    processed_dir = data_dir / "processed" / "ml_dev"
    raw_dir = data_dir / "raw" / "instacart"

    train_df = pd.read_csv(processed_dir / "train.csv")
    validation_df = pd.read_csv(processed_dir / "validation.csv")
    history = HistoryStore.from_csv(raw_dir, index_path=data_dir / "processed" / "history.sqlite")

    artifacts = fit_ranking_model(train_df, validation_df, random_state=42, k_values=DEFAULT_K_VALUES)
    model_path = processed_dir / "model.joblib"
    prediction_path = processed_dir / "validation_predictions.csv"
    metrics_path = processed_dir / "model_metrics.json"
    metadata_path = processed_dir / "model_metadata.json"

    joblib.dump(artifacts["model"], model_path)
    artifacts["validation_predictions"].to_csv(prediction_path, index=False)
    metrics_payload = {
        **artifacts["metrics"],
        "candidate_recall": artifacts["metadata"]["candidate_recall"],
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8")
    metadata_payload = {
        **artifacts["metadata"],
        "training_dataset_path": str(processed_dir / "train.csv"),
        "validation_dataset_path": str(processed_dir / "validation.csv"),
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2, sort_keys=True), encoding="utf-8")

    baseline_results = evaluate_baselines_on_validation(
        history,
        validation_df,
        baseline_names=("customer_frequency", "previously_purchased"),
        k_values=DEFAULT_K_VALUES,
    )

    print("## ML model training")
    print(f"Model: {artifacts['metadata']['model_type']}")
    print(f"Training rows: {artifacts['metadata']['training_row_count']}")
    print(f"Validation rows: {artifacts['metadata']['validation_row_count']}")
    print(f"Features: {', '.join(artifacts['feature_columns'])}")
    print(f"Positive training rows: {artifacts['metadata']['positive_training_rows']}")
    print(f"Negative training rows: {artifacts['metadata']['negative_training_rows']}")
    print()
    print("## ML ranking results")
    print("K Precision Recall F1 MAP NDCG")
    for k in DEFAULT_K_VALUES:
        metric_prefix = f"precision_at_{k}"
        row = {
            "K": k,
            "Precision": artifacts["metrics"][f"precision_at_{k}"],
            "Recall": artifacts["metrics"][f"recall_at_{k}"],
            "F1": artifacts["metrics"][f"f1_at_{k}"],
            "MAP": artifacts["metrics"][f"map_at_{k}"],
            "NDCG": artifacts["metrics"][f"ndcg_at_{k}"],
        }
        print(f"{row['K']} {row['Precision']:.6f} {row['Recall']:.6f} {row['F1']:.6f} {row['MAP']:.6f} {row['NDCG']:.6f}")
    print(f"Candidate recall: {artifacts['metadata']['candidate_recall']:.6f}")
    print()
    print("## Phase 2C development baseline comparison")
    for baseline_name in ("customer_frequency", "previously_purchased"):
        print(f"Phase 2C development baseline ({baseline_name}): {baseline_results[baseline_name]}")
    print(f"Phase 3B ML validation result: {artifacts['metrics']}")


if __name__ == "__main__":
    main()
