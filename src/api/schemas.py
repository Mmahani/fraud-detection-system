"""Pydantic response models for the fraud-detection API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

PredictionLabel = Literal["Fraud", "Legitimate"]
RiskLevel = Literal["Low", "Medium", "High"]


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    model_loaded: bool
    classification_threshold: float | None = None


class ModelInfoResponse(BaseModel):
    model_type: str
    raw_feature_count: int
    transformed_feature_count: int
    classification_threshold: float
    risk_levels: dict[str, str]
    explainability_available: bool


class PredictionResponse(BaseModel):
    transaction_id: Any | None = None
    fraud_probability: float = Field(ge=0.0, le=1.0)
    classification_threshold: float
    prediction: PredictionLabel
    risk_level: RiskLevel


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]


class Factor(BaseModel):
    feature: str
    contribution: float


class ExplanationResponse(PredictionResponse):
    top_fraud_factors: list[Factor]
    top_legitimate_factors: list[Factor]


class ErrorResponse(BaseModel):
    detail: str
