"""Public JSON response schemas for the DecisionPilot API."""

from pydantic import BaseModel, Field


class ServiceInfo(BaseModel):
    name: str
    status: str
    version: str


class HealthResponse(BaseModel):
    status: str


class RecommendationItem(BaseModel):
    rank: int
    product_id: int
    product_name: str
    department_id: int
    aisle_id: int
    model_score: float
    explanation_short: str
    explanation_reason_codes: list[str]


class RecommendationsResponse(BaseModel):
    customer_id: int
    recommendations: list[RecommendationItem]


class RecommendationExplanationRequest(BaseModel):
    product_id: int = Field(ge=1)
    top_k: int = Field(default=5, ge=1, le=20)
    order_number: int | None = Field(default=None, ge=1)


class RecommendationExplanationResponse(BaseModel):
    product_name: str | None = None
    model_score: float | None = None
    reason_codes: list[str] = Field(default_factory=list)
    deterministic_explanation: str
    ai_explanation: str | None = None
    available: bool
    message: str | None = None


class CustomerSummary(BaseModel):
    customer_id: int
    total_orders: int
    products_purchased: int
    most_active_department_id: int | None
    recent_order_number: int | None