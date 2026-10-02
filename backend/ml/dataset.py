"""Build bounded, chronologically separated supervised ranking datasets."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from backend.features.candidates import PREDICTION_KEYS, generate_candidates, measure_candidate_recall
from backend.features.history import HistoryStore
from backend.features.labels import build_labels
from backend.features.point_in_time import FEATURE_COLUMNS, build_point_in_time_features


DATASET_COLUMNS = FEATURE_COLUMNS + ["y"]
DEFAULT_VALIDATION_FRACTION = 0.2


def select_rolling_targets(
    history: HistoryStore,
    *,
    customer_ids: Optional[Iterable[int]] = None,
    customer_limit: Optional[int] = None,
    min_history: int = 1,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
) -> tuple[list[dict[str, int | str]], list[dict[str, int | str]], dict[str, object]]:
    """Select deterministic prior-order targets and split them chronologically."""
    if min_history < 1:
        raise ValueError("min_history must be at least 1")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if customer_limit is not None and customer_limit < 1:
        raise ValueError("customer_limit must be positive")

    allowed = None if customer_ids is None else {int(value) for value in customer_ids}
    prior_orders = history.orders.loc[history.orders["eval_set"] == "prior"].sort_values(
        ["user_id", "order_number", "order_id"]
    )
    eligible_customers = sorted(
        int(customer_id)
        for customer_id in prior_orders["user_id"].unique()
        if allowed is None or int(customer_id) in allowed
    )
    if customer_limit is not None:
        eligible_customers = eligible_customers[:customer_limit]

    train_targets: list[dict[str, int | str]] = []
    validation_targets: list[dict[str, int | str]] = []
    excluded_count = 0
    for customer_id in eligible_customers:
        customer_orders = prior_orders.loc[prior_orders["user_id"] == customer_id].reset_index(drop=True)
        eligible = []
        for position, target in customer_orders.iterrows():
            if position < min_history:
                excluded_count += 1
                continue
            eligible.append({
                "customer_id": int(customer_id),
                "target_order_id": int(target["order_id"]),
                "target_order_number": int(target["order_number"]),
                "history_length": int(position),
            })
        if not eligible:
            continue
        validation_count = max(1, math.ceil(len(eligible) * validation_fraction))
        split_at = max(0, len(eligible) - validation_count)
        train_targets.extend(eligible[:split_at])
        validation_targets.extend(eligible[split_at:])

    metadata = {
        "customer_selection": "sorted eligible customer IDs",
        "customer_count": len(eligible_customers),
        "customer_ids_first": eligible_customers[0] if eligible_customers else None,
        "customer_ids_last": eligible_customers[-1] if eligible_customers else None,
        "minimum_history": min_history,
        "validation_fraction": validation_fraction,
        "split_rule": "per customer, final ceil(validation_fraction * eligible targets) are validation",
        "excluded_insufficient_history_targets": excluded_count,
        "training_target_count": len(train_targets),
        "validation_target_count": len(validation_targets),
    }
    return train_targets, validation_targets, metadata


@dataclass
class DevelopmentDataset:
    train: pd.DataFrame
    validation: pd.DataFrame
    metadata: dict[str, object]
    candidate_diagnostics: pd.DataFrame

    def write(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.train[DATASET_COLUMNS].to_csv(output_dir / "train.csv", index=False)
        self.validation[DATASET_COLUMNS].to_csv(output_dir / "validation.csv", index=False)
        self.candidate_diagnostics.to_csv(
            output_dir / "candidate_diagnostics.csv", index=False
        )
        (output_dir / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True), encoding="utf-8"
        )


def _empty_dataset() -> pd.DataFrame:
    return pd.DataFrame(columns=DATASET_COLUMNS)


def _build_partition(
    history: HistoryStore,
    targets: list[dict[str, int | str]],
    *,
    candidate_limit: int,
    source_limit: int,
    recent_order_count: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    partitions: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    for target in targets:
        customer_id = int(target["customer_id"])
        target_order_id = int(target["target_order_id"])
        target_order_number = int(target["target_order_number"])
        candidates = generate_candidates(
            history,
            customer_id,
            target_order_id,
            target_order_number,
            candidate_limit=candidate_limit,
            source_limit=source_limit,
            recent_order_count=recent_order_count,
        )
        target_products = history.target_products(target_order_id)
        labels = build_labels(candidates, target_products)
        features = build_point_in_time_features(
            history, candidates, recent_order_count=recent_order_count
        )
        rows = features.merge(
            labels,
            on=PREDICTION_KEYS,
            how="inner",
            validate="one_to_one",
        )
        if len(rows) != len(candidates):
            raise ValueError("feature and label row counts differ for a target")
        if rows.duplicated(PREDICTION_KEYS).any():
            raise ValueError("development dataset contains duplicate prediction rows")
        partitions.append(rows[DATASET_COLUMNS])
        target_products_set = set(target_products["product_id"].astype(int))
        candidate_set = set(candidates["candidate_product_id"].astype(int))
        diagnostics.append({
            **target,
            "target_product_count": len(target_products_set),
            "candidate_count": len(candidate_set),
            "matched_target_products": len(target_products_set & candidate_set),
            "missed_target_products": len(target_products_set - candidate_set),
            "candidate_recall": (
                len(target_products_set & candidate_set) / len(target_products_set)
                if target_products_set else 0.0
            ),
        })

    if not partitions:
        return _empty_dataset(), diagnostics
    return pd.concat(partitions, ignore_index=True).sort_values(PREDICTION_KEYS).reset_index(drop=True), diagnostics


def build_development_dataset(
    history: HistoryStore,
    *,
    customer_ids: Optional[Iterable[int]] = None,
    customer_limit: Optional[int] = 100,
    min_history: int = 1,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    candidate_limit: int = 200,
    source_limit: int = 100,
    recent_order_count: int = 3,
    output_dir: Optional[Path] = None,
) -> DevelopmentDataset:
    """Build a deterministic bounded train/validation ranking dataset."""
    train_targets, validation_targets, split_metadata = select_rolling_targets(
        history,
        customer_ids=customer_ids,
        customer_limit=customer_limit,
        min_history=min_history,
        validation_fraction=validation_fraction,
    )
    train, train_diagnostics = _build_partition(
        history,
        train_targets,
        candidate_limit=candidate_limit,
        source_limit=source_limit,
        recent_order_count=recent_order_count,
    )
    validation, validation_diagnostics = _build_partition(
        history,
        validation_targets,
        candidate_limit=candidate_limit,
        source_limit=source_limit,
        recent_order_count=recent_order_count,
    )
    diagnostics = pd.DataFrame(train_diagnostics + validation_diagnostics)
    total_rows = len(train) + len(validation)
    total_positive = int(train["y"].sum()) + int(validation["y"].sum())
    metadata = {
        "dataset_policy": "prior-only rolling targets with point-in-time features",
        "official_train_target_used": False,
        "target_source": "order_products__prior.csv",
        "candidate_source": "existing Phase 2B generator",
        "feature_source": "existing Phase 2B point-in-time feature builder",
        "random_seed": None,
        "candidate_limit": candidate_limit,
        "source_limit": source_limit,
        "recent_order_count": recent_order_count,
        "total_rows": total_rows,
        "positive_rows": total_positive,
        "negative_rows": total_rows - total_positive,
        "positive_negative_ratio": (
            total_positive / (total_rows - total_positive)
            if total_rows - total_positive else None
        ),
        "target_group_count": int(len(diagnostics)),
        "average_candidates_per_target": float(diagnostics["candidate_count"].mean())
        if not diagnostics.empty else 0.0,
        "candidate_recall": float(diagnostics["candidate_recall"].mean())
        if not diagnostics.empty else 0.0,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_positive_rows": int(train["y"].sum()) if not train.empty else 0,
        "train_negative_rows": int((train["y"] == 0).sum()) if not train.empty else 0,
        "validation_positive_rows": int(validation["y"].sum()) if not validation.empty else 0,
        "validation_negative_rows": int((validation["y"] == 0).sum()) if not validation.empty else 0,
        "train_target_count": len(train_targets),
        "validation_target_count": len(validation_targets),
        "feature_columns": FEATURE_COLUMNS,
        "dataset_columns": DATASET_COLUMNS,
        **split_metadata,
    }
    result = DevelopmentDataset(train, validation, metadata, diagnostics)
    if output_dir is not None:
        result.write(output_dir)
    return result
