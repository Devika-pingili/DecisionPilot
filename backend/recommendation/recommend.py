"""Generate point-in-time recommendations with auditable explanations."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from backend.features.candidates import generate_candidates
from backend.features.history import HistoryStore
from backend.features.point_in_time import build_point_in_time_features

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = REPOSITORY_ROOT / "data" / "processed" / "ml_dev" / "model.joblib"
DEFAULT_MODEL_METADATA_PATH = (
    REPOSITORY_ROOT / "data" / "processed" / "ml_dev" / "model_metadata.json"
)
RECOMMENDATION_COLUMNS = [
    "customer_id",
    "target_order_id",
    "product_id",
    "product_name",
    "department_id",
    "aisle_id",
    "model_score",
    "rank",
    "explanation_short",
    "explanation_reason_codes",
]
CSV_COLUMNS = RECOMMENDATION_COLUMNS.copy()


def load_model(model_path: Optional[Path] = None):
    """Load the already-trained Phase 3B model without fitting or modifying it."""
    path = DEFAULT_MODEL_PATH if model_path is None else Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Trained recommendation model was not found: {path}")
    model = joblib.load(path)
    if not callable(getattr(model, "predict_proba", None)):
        raise ValueError(f"Model artifact does not provide predict_proba: {path}")
    return model


def _model_feature_columns(model) -> list[str]:
    columns = getattr(model, "feature_names_in_", None)
    if columns is not None:
        return [str(column) for column in columns]
    if DEFAULT_MODEL_METADATA_PATH.is_file():
        metadata = json.loads(DEFAULT_MODEL_METADATA_PATH.read_text(encoding="utf-8"))
        columns = metadata.get("feature_columns")
        if columns:
            return [str(column) for column in columns]
    raise ValueError("Model artifact does not declare its expected feature columns.")


def _number(row: pd.Series, column: str) -> float:
    value = row.get(column, 0)
    if pd.isna(value):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Feature {column!r} must be numeric") from error


def explain_historical_features(row: pd.Series | dict[str, object]) -> dict[str, str]:
    """Explain only signals that are present in the point-in-time feature row."""
    values = row if isinstance(row, pd.Series) else pd.Series(row)
    reasons: list[tuple[str, str]] = []

    purchases = _number(values, "customer_product_purchase_count")
    if purchases >= 2:
        reasons.append(("CUSTOMER_FREQUENT", "Purchased in multiple previous orders."))

    if _number(values, "customer_recent_product_flag") >= 1:
        reasons.append(("CUSTOMER_RECENT", "Purchased in the customer's recent orders."))

    if _number(values, "customer_product_reorder_count") > 0:
        reasons.append(("CUSTOMER_REORDER", "Previously reordered by this customer."))

    if _number(values, "customer_department_affinity") > 0:
        reasons.append(("DEPARTMENT_AFFINITY", "Matches a department the customer has purchased from."))

    if _number(values, "customer_aisle_affinity") > 0:
        reasons.append(("AISLE_AFFINITY", "Matches an aisle the customer has purchased from."))

    if _number(values, "product_global_purchase_count") > 0:
        reasons.append(("GLOBAL_POPULARITY", "Has prior purchase history across customers."))

    if not reasons:
        return {
            "explanation_short": "No strong historical signal identified before this order.",
            "explanation_reason_codes": "",
        }
    return {
        "explanation_short": " ".join(message for _, message in reasons),
        "explanation_reason_codes": "|".join(code for code, _ in reasons),
    }


def _empty_recommendations() -> pd.DataFrame:
    return pd.DataFrame(columns=RECOMMENDATION_COLUMNS)


def recommend_products(
    history: HistoryStore,
    customer_id: int,
    target_order_id: int,
    target_order_number: Optional[int] = None,
    *,
    model=None,
    model_path: Optional[Path] = None,
    candidate_limit: int = 200,
    source_limit: int = 100,
    recent_order_count: int = 3,
    top_k: Optional[int] = None,
) -> pd.DataFrame:
    """Return ranked recommendations using history strictly before a prior order.

    ``target_order_id`` identifies the prediction cutoff, not a source of labels.
    The target must be a prior order; its products are never read by this function.
    """
    try:
        customer_id = int(customer_id)
        target_order_id = int(target_order_id)
        if target_order_number is not None:
            target_order_number = int(target_order_number)
    except (TypeError, ValueError) as error:
        raise ValueError("customer_id, target_order_id, and target_order_number must be integers") from error
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive")

    target = history.target_order(target_order_id)
    if int(target["user_id"]) != customer_id:
        raise ValueError("target_order_id does not belong to customer_id")
    if str(target["eval_set"]) != "prior":
        raise ValueError("target_order_id must identify a prior order prediction context")
    actual_target_order_number = int(target["order_number"])
    if target_order_number is not None and target_order_number != actual_target_order_number:
        raise ValueError("target_order_number does not match target_order_id")

    target_order_number = actual_target_order_number
    candidates = generate_candidates(
        history,
        customer_id,
        target_order_id,
        target_order_number,
        candidate_limit=candidate_limit,
        source_limit=source_limit,
        recent_order_count=recent_order_count,
    )
    if candidates.empty:
        return _empty_recommendations()

    features = build_point_in_time_features(
        history, candidates, recent_order_count=recent_order_count
    )
    model = load_model(model_path) if model is None else model
    feature_columns = _model_feature_columns(model)
    missing = [column for column in feature_columns if column not in features.columns]
    if missing:
        raise ValueError(f"Point-in-time feature builder is missing model features: {missing}")

    model_input = features[feature_columns].copy()
    for column in feature_columns:
        model_input[column] = pd.to_numeric(model_input[column], errors="coerce")
    probabilities = model.predict_proba(model_input)
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 not in classes:
        raise ValueError("Trained model does not expose a positive-class probability")
    positive_class_index = classes.index(1)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(features):
        raise ValueError("Model returned an invalid score matrix for the candidate rows")
    if positive_class_index >= probabilities.shape[1]:
        raise ValueError("Model returned no probability column for the positive class")

    result = features[
        ["customer_id", "target_order_id", "product_id", "product_name", "department_id", "aisle_id"]
    ].copy()
    result["model_score"] = pd.to_numeric(
        probabilities[:, positive_class_index], errors="coerce"
    )
    if result["model_score"].isna().any() or not result["model_score"].map(
        lambda score: math.isfinite(float(score))
    ).all():
        raise ValueError("Model produced a non-finite recommendation score")

    explanations = features.apply(explain_historical_features, axis=1, result_type="expand")
    result["explanation_short"] = explanations["explanation_short"].to_numpy()
    result["explanation_reason_codes"] = explanations["explanation_reason_codes"].to_numpy()
    result = result.sort_values(
        ["model_score", "product_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    if result["product_id"].duplicated().any():
        raise ValueError("Candidate generation produced duplicate product recommendations")
    result["rank"] = range(1, len(result) + 1)
    if top_k is not None:
        result = result.head(top_k).copy()
    return result[RECOMMENDATION_COLUMNS]


def write_recommendations(recommendations: pd.DataFrame, output_path: Path) -> Path:
    """Write one customer's ranked recommendations as a stable CSV artifact."""
    missing = [column for column in CSV_COLUMNS if column not in recommendations.columns]
    if missing:
        raise ValueError(f"Recommendations are missing output columns: {missing}")
    if recommendations.duplicated(["customer_id", "target_order_id", "product_id"]).any():
        raise ValueError("Recommendations contain duplicate customer/context/product rows")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recommendations[CSV_COLUMNS].to_csv(output_path, index=False, lineterminator="\n")
    return output_path