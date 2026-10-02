"""Thin HTTP layer around the existing point-in-time recommendation engine."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Optional
from urllib.parse import urlsplit

import pandas as pd
from fastapi import FastAPI, HTTPException, Path as PathParameter, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from backend.ai.explanation import generate_ai_explanation, get_ai_provider_diagnostics
from backend.config import DEFAULT_DATA_ROOT, PROJECT_ROOT, resolve_data_paths
from backend.api.schemas import (
    AIProviderDiagnostics,
    CustomerSummary,
    HealthResponse,
    RecommendationExplanationRequest,
    RecommendationExplanationResponse,
    RecommendationItem,
    RecommendationsResponse,
    ServiceInfo,
)
from backend.features.history import HistoryStore
from backend.recommendation.recommend import load_model, recommend_products

logger = logging.getLogger(__name__)
DEFAULT_LOCAL_DATA_PATHS = resolve_data_paths(DEFAULT_DATA_ROOT)
DEFAULT_RAW_DIR = DEFAULT_LOCAL_DATA_PATHS.raw_dir
DEFAULT_INDEX_PATH = DEFAULT_LOCAL_DATA_PATHS.index_path
ALLOWED_ORIGINS = ["http://localhost:3000", "http://localhost:5173", "http://localhost:5174"]


def _cors_origins() -> list[str]:
    origins = list(ALLOWED_ORIGINS)
    configured_origins = os.getenv("DECISIONPILOT_CORS_ORIGINS", "")
    for value in configured_origins.split(","):
        origin = value.strip().rstrip("/")
        if not origin:
            continue
        if origin == "*":
            raise ValueError("DECISIONPILOT_CORS_ORIGINS must not contain a wildcard origin")
        parsed = urlsplit(origin)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("DECISIONPILOT_CORS_ORIGINS must contain comma-separated origins")
        if origin not in origins:
            origins.append(origin)
    return origins


def _resolve_prediction_context(
    history: HistoryStore,
    customer_id: int,
    order_number: Optional[int],
) -> tuple[int, int]:
    prior_orders = history.orders.loc[
        (history.orders["user_id"] == customer_id)
        & (history.orders["eval_set"] == "prior")
    ].sort_values(["order_number", "order_id"])
    if prior_orders.empty:
        raise HTTPException(status_code=400, detail="Customer has insufficient prior-order history.")

    if order_number is None:
        eligible_orders = prior_orders.iloc[1:]
        if eligible_orders.empty:
            raise HTTPException(status_code=400, detail="Customer has insufficient prior-order history.")
        target = eligible_orders.iloc[-1]
    else:
        matching_orders = prior_orders.loc[prior_orders["order_number"] == order_number]
        if matching_orders.empty:
            raise HTTPException(
                status_code=400,
                detail="order_number is not a prior-order prediction context for this customer.",
            )
        target = matching_orders.iloc[-1]
        earlier_orders = prior_orders.loc[prior_orders["order_number"] < order_number]
        if earlier_orders.empty:
            raise HTTPException(
                status_code=400,
                detail="Customer has insufficient history before the requested order_number.",
            )

    return int(target["order_id"]), int(target["order_number"])


def _customer_exists(history: HistoryStore, customer_id: int) -> bool:
    return bool((history.orders["user_id"] == customer_id).any())


def _resources(request: Request) -> tuple[HistoryStore, object]:
    if request.app.state.resource_error is not None:
        raise HTTPException(status_code=503, detail=request.app.state.resource_error)
    return request.app.state.history, request.app.state.model


def create_app(
    *,
    history: Optional[HistoryStore] = None,
    model=None,
    raw_dir: Optional[Path] = None,
    index_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
) -> FastAPI:
    """Build an application with model and indexed history cached for its lifetime."""
    injected_history = history
    injected_model = model
    configured_paths = resolve_data_paths()
    active_raw_dir = Path(raw_dir) if raw_dir is not None else configured_paths.raw_dir
    active_index_path = Path(index_path) if index_path is not None else configured_paths.index_path
    active_model_path = Path(model_path) if model_path is not None else configured_paths.model_path

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.history = injected_history
        application.state.model = injected_model
        application.state.resource_error = None
        try:
            if application.state.history is None:
                if not active_index_path.is_file() and not configured_paths.build_index_if_missing:
                    application.state.resource_error = "Indexed customer history is unavailable."
                else:
                    application.state.history = HistoryStore.from_csv(
                        active_raw_dir, index_path=active_index_path, progress=False
                    )
            if application.state.model is None and application.state.resource_error is None:
                if not active_model_path.is_file():
                    application.state.resource_error = "The recommendation model is unavailable."
                else:
                    application.state.model = load_model(active_model_path)
        except Exception:
            logger.exception("DecisionPilot API resource initialization failed")
            application.state.resource_error = "Recommendation resources are unavailable."
        try:
            yield
        finally:
            active_history = application.state.history
            if active_history is not None:
                try:
                    active_history.close()
                except Exception:
                    logger.exception("Failed to close indexed customer history")

    application = FastAPI(
        title="DecisionPilot API",
        version="1.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @application.exception_handler(Exception)
    async def handle_unexpected_error(_request: Request, error: Exception) -> JSONResponse:
        logger.error("Unexpected DecisionPilot API error", exc_info=(type(error), error, error.__traceback__))
        return JSONResponse(
            status_code=500,
            content={"detail": "An unexpected internal error occurred."},
        )

    @application.get("/", response_model=ServiceInfo)
    async def root() -> ServiceInfo:
        return ServiceInfo(name="DecisionPilot", status="ok", version="1.0")

    @application.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        _resources(request)
        return HealthResponse(status="healthy")

    @application.get("/diagnostics/ai", response_model=AIProviderDiagnostics)
    async def ai_diagnostics(request: Request) -> AIProviderDiagnostics:
        client_host = request.client.host if request.client is not None else ""
        if client_host not in {"127.0.0.1", "::1", "testclient"}:
            raise HTTPException(status_code=404, detail="Not found.")
        return AIProviderDiagnostics(**get_ai_provider_diagnostics())

    @application.get(
        "/customers/{customer_id}/recommendations",
        response_model=RecommendationsResponse,
    )
    async def customer_recommendations(
        request: Request,
        customer_id: Annotated[int, PathParameter(ge=1)],
        top_k: Annotated[int, Query(ge=1, le=20)] = 5,
        order_number: Annotated[Optional[int], Query(ge=1)] = None,
    ) -> RecommendationsResponse:
        active_history, active_model = _resources(request)
        if not _customer_exists(active_history, customer_id):
            raise HTTPException(status_code=404, detail="Customer was not found.")
        target_order_id, target_order_number = _resolve_prediction_context(
            active_history, customer_id, order_number
        )
        try:
            ranked = recommend_products(
                active_history,
                customer_id,
                target_order_id,
                target_order_number,
                model=active_model,
                top_k=top_k,
            )
        except ValueError as error:
            logger.info("Recommendation request rejected: %s", error)
            raise HTTPException(
                status_code=400,
                detail="Unable to generate recommendations for this prediction context.",
            ) from error

        recommendations = [
            RecommendationItem(
                rank=int(row.rank),
                product_id=int(row.product_id),
                product_name=str(row.product_name),
                department_id=int(row.department_id),
                aisle_id=int(row.aisle_id),
                model_score=float(row.model_score),
                explanation_short=str(row.explanation_short),
                explanation_reason_codes=[
                    code for code in str(row.explanation_reason_codes).split("|") if code
                ],
            )
            for row in ranked.itertuples(index=False)
        ]
        return RecommendationsResponse(customer_id=customer_id, recommendations=recommendations)

    @application.post(
        "/customers/{customer_id}/recommendations/explain",
        response_model=RecommendationExplanationResponse,
    )
    async def explain_recommendation(
        request: Request,
        customer_id: Annotated[int, PathParameter(ge=1)],
        explanation_request: RecommendationExplanationRequest,
    ) -> RecommendationExplanationResponse:
        active_history, active_model = _resources(request)
        if not _customer_exists(active_history, customer_id):
            raise HTTPException(status_code=404, detail="Customer was not found.")

        target_order_id, target_order_number = _resolve_prediction_context(
            active_history, customer_id, explanation_request.order_number
        )
        try:
            ranked = recommend_products(
                active_history,
                customer_id,
                target_order_id,
                target_order_number,
                model=active_model,
                top_k=explanation_request.top_k,
            )
        except ValueError as error:
            logger.info("Recommendation explanation context rejected: %s", error)
            raise HTTPException(
                status_code=400,
                detail="Unable to generate recommendations for this prediction context.",
            ) from error

        selected = ranked.loc[ranked["product_id"] == explanation_request.product_id]
        if selected.empty:
            raise HTTPException(
                status_code=404,
                detail="The selected product is not in the current recommendation list.",
            )

        row = selected.iloc[0]
        reason_codes = [
            code
            for code in str(row["explanation_reason_codes"]).split("|")
            if code in {
                "CUSTOMER_FREQUENT",
                "CUSTOMER_RECENT",
                "CUSTOMER_REORDER",
                "DEPARTMENT_AFFINITY",
                "AISLE_AFFINITY",
                "GLOBAL_POPULARITY",
            }
        ]
        deterministic_explanation = str(row["explanation_short"])
        ai_explanation = await run_in_threadpool(
            generate_ai_explanation,
            str(row["product_name"]),
            reason_codes,
        )

        return RecommendationExplanationResponse(
            product_name=str(row["product_name"]),
            model_score=float(row["model_score"]),
            reason_codes=reason_codes,
            deterministic_explanation=deterministic_explanation,
            ai_explanation=ai_explanation,
            available=ai_explanation is not None,
            message=None if ai_explanation is not None else "AI explanation is currently unavailable.",
        )

    @application.get("/customers/{customer_id}/summary", response_model=CustomerSummary)
    async def customer_summary(
        request: Request,
        customer_id: Annotated[int, PathParameter(ge=1)],
    ) -> CustomerSummary:
        active_history, _ = _resources(request)
        if not _customer_exists(active_history, customer_id):
            raise HTTPException(status_code=404, detail="Customer was not found.")

        prior_orders = active_history.prior_orders_for_customer(customer_id)
        order_ids = prior_orders["order_id"].astype(int).tolist()
        transaction_frames = [active_history.target_products(order_id) for order_id in order_ids]
        transactions = (
            pd.concat(transaction_frames, ignore_index=True)
            if transaction_frames
            else pd.DataFrame(columns=["order_id", "product_id", "reordered"])
        )
        most_active_department_id = None
        if not transactions.empty:
            product_departments = active_history.catalog()[["product_id", "department_id"]]
            department_counts = transactions.merge(
                product_departments,
                on="product_id",
                how="left",
                validate="many_to_one",
            ).groupby("department_id").size().rename("purchase_count").reset_index()
            department_counts = department_counts.sort_values(
                ["purchase_count", "department_id"],
                ascending=[False, True],
                kind="mergesort",
            )
            most_active_department_id = int(department_counts.iloc[0]["department_id"])

        return CustomerSummary(
            customer_id=customer_id,
            total_orders=int(len(prior_orders)),
            products_purchased=int(transactions["product_id"].nunique()),
            most_active_department_id=most_active_department_id,
            recent_order_number=(
                int(prior_orders["order_number"].max()) if not prior_orders.empty else None
            ),
        )

    return application


app = create_app()