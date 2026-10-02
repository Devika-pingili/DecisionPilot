"""Load a saved Phase 3B model and print the validation ranking summary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.ml.model import DEFAULT_K_VALUES, compute_grouped_ranking_metrics


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    model_dir = repo_root / "data" / "processed" / "ml_dev"
    model = joblib.load(model_dir / "model.joblib")
    validation = pd.read_csv(model_dir / "validation_predictions.csv")
    metrics = compute_grouped_ranking_metrics(validation, k_values=DEFAULT_K_VALUES)
    print("## ML ranking results")
    print("K Precision Recall F1 MAP NDCG")
    for k in DEFAULT_K_VALUES:
        print(
            f"{k} {metrics[f'precision_at_{k}']:.6f} {metrics[f'recall_at_{k}']:.6f} "
            f"{metrics[f'f1_at_{k}']:.6f} {metrics[f'map_at_{k}']:.6f} {metrics[f'ndcg_at_{k}']:.6f}"
        )
    print("")
    print(f"Model type: {type(model).__name__}")

    metadata = json.loads((model_dir / "model_metadata.json").read_text(encoding="utf-8"))
    print(f"Candidate recall: {metadata['candidate_recall']:.6f}")


if __name__ == "__main__":
    main()
