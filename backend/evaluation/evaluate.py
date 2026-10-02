"""Rolling, point-in-time evaluation for deterministic recommendation baselines."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from backend.features.candidates import generate_candidates
from backend.features.history import HistoryStore

from .baselines import BASELINES
from .metrics import ranking_metrics


DEFAULT_K_VALUES = (5, 10, 20)


def _history_bucket(history_length: int) -> str:
    if history_length == 1:
        return "1 order"
    if history_length <= 3:
        return "2-3 orders"
    if history_length <= 7:
        return "4-7 orders"
    return "8+ orders"


def _basket_bucket(basket_size: int) -> str:
    if basket_size == 1:
        return "1"
    if basket_size <= 3:
        return "2-3"
    if basket_size <= 6:
        return "4-6"
    return "7+"


@dataclass
class EvaluationResult:
    """Machine-readable rolling evaluation outputs."""

    summary: pd.DataFrame
    segments: pd.DataFrame
    per_target: pd.DataFrame
    recommendations: pd.DataFrame
    candidate_diagnostics: pd.DataFrame
    excluded_targets: pd.DataFrame
    metadata: dict[str, object]

    def write(self, output_dir: Path, prefix: str = "baseline_evaluation") -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.summary.to_csv(output_dir / f"{prefix}_summary.csv", index=False)
        self.segments.to_csv(output_dir / f"{prefix}_segments.csv", index=False)
        self.per_target.to_csv(output_dir / f"{prefix}_per_target.csv", index=False)
        self.recommendations.to_csv(output_dir / f"{prefix}_recommendations.csv", index=False)
        self.candidate_diagnostics.to_csv(
            output_dir / f"{prefix}_candidate_diagnostics.csv", index=False
        )
        self.excluded_targets.to_csv(output_dir / f"{prefix}_excluded_targets.csv", index=False)
        (output_dir / f"{prefix}_metadata.json").write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True), encoding="utf-8"
        )


def _target_rows(
    history: HistoryStore,
    min_history: int,
    customer_ids: Optional[set[int]],
) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    prior_orders = history.orders.loc[history.orders["eval_set"] == "prior"].sort_values(
        ["user_id", "order_number", "order_id"]
    )
    eligible: list[dict[str, int]] = []
    excluded: list[dict[str, int]] = []
    for customer_id, customer_orders in prior_orders.groupby("user_id", sort=True):
        customer_id = int(customer_id)
        if customer_ids is not None and customer_id not in customer_ids:
            continue
        customer_orders = customer_orders.reset_index(drop=True)
        for position, target in customer_orders.iterrows():
            row = {
                "customer_id": customer_id,
                "target_order_id": int(target["order_id"]),
                "target_order_number": int(target["order_number"]),
                "history_length": int(position),
            }
            if position < min_history:
                excluded.append(row)
            else:
                eligible.append(row)
    return eligible, excluded


def _aggregate(rows: pd.DataFrame, k_values: tuple[int, ...], excluded_count: int) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    result: list[dict[str, object]] = []
    metric_names = ["precision", "recall", "f1", "map", "ndcg"]
    for (baseline, k), group in rows.groupby(["baseline", "k"], sort=True):
        record: dict[str, object] = {
            "baseline": baseline,
            "k": int(k),
            "evaluated_targets": int(group["target_order_id"].nunique()),
            "customers": int(group["customer_id"].nunique()),
            "average_target_basket_size": float(group["target_basket_size"].mean()),
            "average_candidate_count": float(group["candidate_count"].mean()),
            "candidate_recall": float(group["candidate_recall"].mean()),
            "excluded_targets": int(excluded_count),
        }
        record.update({metric: float(group[metric].mean()) for metric in metric_names})
        result.append(record)
    return pd.DataFrame(result)


def _segment_summary(rows: pd.DataFrame, excluded_count: int) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    metric_names = ["precision", "recall", "f1", "map", "ndcg"]
    outputs: list[pd.DataFrame] = []
    for segment_type, column in [("history_length", "history_bucket"), ("target_basket_size", "basket_bucket")]:
        grouped = rows.groupby(["baseline", "k", column], sort=True).agg(
            evaluated_targets=("target_order_id", "nunique"),
            customers=("customer_id", "nunique"),
            average_target_basket_size=("target_basket_size", "mean"),
            average_candidate_count=("candidate_count", "mean"),
            candidate_recall=("candidate_recall", "mean"),
            **{metric: (metric, "mean") for metric in metric_names},
        ).reset_index()
        grouped = grouped.rename(columns={column: "segment"})
        grouped["segment_type"] = segment_type
        grouped["excluded_targets"] = excluded_count
        outputs.append(grouped)
    return pd.concat(outputs, ignore_index=True)


def evaluate_rolling(
    history: HistoryStore,
    *,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
    min_history: int = 1,
    customer_ids: Optional[Iterable[int]] = None,
    max_targets: Optional[int] = None,
    candidate_limit: int = 200,
    source_limit: int = 100,
    recent_order_count: int = 3,
) -> EvaluationResult:
    """Evaluate deterministic baselines on prior-only rolling target orders."""
    k_values = tuple(dict.fromkeys(int(k) for k in k_values))
    if not k_values or any(k < 1 for k in k_values):
        raise ValueError("k_values must contain positive integers")
    if min_history < 1:
        raise ValueError("min_history must be at least 1")
    customer_set = None if customer_ids is None else {int(value) for value in customer_ids}
    targets, excluded = _target_rows(history, min_history, customer_set)
    if max_targets is not None:
        if max_targets < 1:
            raise ValueError("max_targets must be positive")
        targets = targets[:max_targets]

    per_target_rows: list[dict[str, object]] = []
    recommendation_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    for target in targets:
        customer_id = target["customer_id"]
        target_order_id = target["target_order_id"]
        target_order_number = target["target_order_number"]
        historical_orders, historical_transactions = history.history_for_target(
            customer_id, target_order_id, target_order_number
        )
        candidates = generate_candidates(
            history,
            customer_id,
            target_order_id,
            target_order_number,
            recent_order_count=recent_order_count,
            candidate_limit=candidate_limit,
            source_limit=source_limit,
        )
        target_products = history.target_products(target_order_id)
        relevant = set(target_products["product_id"].astype(int))
        candidate_ids = candidates["candidate_product_id"].astype(int).tolist()
        matched = relevant.intersection(candidate_ids)
        candidate_record = {
            **target,
            "target_basket_size": len(relevant),
            "candidate_count": len(candidate_ids),
            "matched_target_products": len(matched),
            "missed_target_products": len(relevant - set(candidate_ids)),
            "candidate_recall": len(matched) / len(relevant) if relevant else 0.0,
        }
        candidate_rows.append(candidate_record)
        history_context = (historical_orders, historical_transactions)
        for baseline_name, baseline in BASELINES.items():
            ranking = baseline(
                history,
                customer_id,
                target_order_id,
                target_order_number,
                candidate_ids,
                history_context=history_context,
            )
            for rank, product_id in enumerate(ranking, start=1):
                recommendation_rows.append({
                    **target,
                    "baseline": baseline_name,
                    "rank": rank,
                    "candidate_product_id": product_id,
                })
            for k in k_values:
                metric_values = ranking_metrics(ranking, relevant, (k,))
                per_target_rows.append({
                    **target,
                    "baseline": baseline_name,
                    "k": k,
                    "target_basket_size": len(relevant),
                    "candidate_count": len(candidate_ids),
                    "candidate_recall": candidate_record["candidate_recall"],
                    "history_bucket": _history_bucket(target["history_length"]),
                    "basket_bucket": _basket_bucket(len(relevant)),
                    "precision": metric_values[f"precision_at_{k}"],
                    "recall": metric_values[f"recall_at_{k}"],
                    "f1": metric_values[f"f1_at_{k}"],
                    "map": metric_values[f"map_at_{k}"],
                    "ndcg": metric_values[f"ndcg_at_{k}"],
                })

    per_target = pd.DataFrame(per_target_rows)
    candidate_diagnostics = pd.DataFrame(candidate_rows)
    recommendations = pd.DataFrame(recommendation_rows)
    excluded_targets = pd.DataFrame(excluded)
    summary = _aggregate(per_target, k_values, len(excluded_targets))
    segments = _segment_summary(per_target, len(excluded_targets))
    metadata = {
        "evaluation_policy": "prior-only rolling targets",
        "official_train_target_used": False,
        "min_history": min_history,
        "k_values": list(k_values),
        "candidate_limit": candidate_limit,
        "source_limit": source_limit,
        "recent_order_count": recent_order_count,
        "eligible_target_count": len(targets),
        "excluded_target_count": len(excluded_targets),
    }
    return EvaluationResult(
        summary, segments, per_target, recommendations,
        candidate_diagnostics, excluded_targets, metadata,
    )
