"""FastAPI application exposing the frozen fraud-detection model.

Run from the project root:

    uvicorn src.api.main:app --host 127.0.0.1 --port 8000

The model, preprocessor, and threshold configuration are loaded once at startup.
Every endpoint expects the same raw feature contract used during training.
"""

from __future__ import annotations

import logging
import math
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Body, FastAPI, HTTPException, Query

from src.prediction import (
    explain_transaction,
    get_required_features,
    load_configuration,
    load_model,
    load_preprocessor,
    predict_batch,
    predict_transaction,
)

from .schemas import (
    BatchPredictionResponse,
    ExplanationResponse,
    Factor,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

LOGGER = logging.getLogger("fraud_api")
MAX_BATCH_SIZE = 500


def _scalar(value: Any) -> Any:
    """Convert numpy/pandas scalars and NaN into JSON-safe values."""
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


class ModelService:
    """Lazy holder for the frozen inference artifacts."""

    def __init__(self) -> None:
        self.model: Any | None = None
        self.preprocessor: Any | None = None
        self.threshold_config: dict[str, Any] | None = None
        self.required_features: list[str] | None = None
        self.load_error: str | None = None

    @property
    def loaded(self) -> bool:
        return all(
            item is not None
            for item in (self.model, self.preprocessor, self.threshold_config, self.required_features)
        )

    def load(self) -> None:
        self.threshold_config = load_configuration()
        self.preprocessor = load_preprocessor()
        self.model = load_model()
        self.required_features = get_required_features(self.preprocessor)
        self.load_error = None
        LOGGER.info(
            "Loaded model %s with %d raw features and threshold %.2f",
            type(self.model).__name__,
            len(self.required_features),
            float(self.threshold_config["classification_threshold"]),
        )


service = ModelService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        service.load()
    except Exception as exc:  # pragma: no cover - depends on local artifacts
        service.load_error = str(exc)
        LOGGER.exception("Could not load the model artifacts")
    yield


app = FastAPI(
    title="IEEE-CIS Fraud Detection API",
    description=(
        "Inference service for the frozen Phase 8 tuned Random Forest. "
        "The threshold is 0.35 and the risk bands are presentation-only."
    ),
    version="1.0.1",
    lifespan=lifespan,
)


def _require_model() -> None:
    if not service.loaded:
        detail = "Model artifacts are not loaded."
        if service.load_error:
            detail = f"{detail} {service.load_error}"
        raise HTTPException(status_code=503, detail=detail)


def _prediction_response(result: dict[str, Any]) -> PredictionResponse:
    return PredictionResponse(
        transaction_id=_scalar(result.get("transaction_id")),
        fraud_probability=float(result["fraud_probability"]),
        classification_threshold=float(result["classification_threshold"]),
        prediction=result["prediction"],
        risk_level=result["risk_level"],
    )


@app.get("/", tags=["meta"])
def root() -> dict[str, str]:
    return {
        "name": "IEEE-CIS Fraud Detection API",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    threshold = None
    if service.threshold_config:
        threshold = float(service.threshold_config["classification_threshold"])
    return HealthResponse(
        status="ok" if service.loaded else "degraded",
        model_loaded=service.loaded,
        classification_threshold=threshold,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["model"])
def model_info() -> ModelInfoResponse:
    _require_model()
    assert service.model is not None
    assert service.preprocessor is not None
    assert service.threshold_config is not None
    assert service.required_features is not None
    return ModelInfoResponse(
        model_type=type(service.model).__name__,
        raw_feature_count=len(service.required_features),
        transformed_feature_count=int(service.model.n_features_in_),
        classification_threshold=float(service.threshold_config["classification_threshold"]),
        risk_levels=dict(service.threshold_config["risk_levels"]),
        explainability_available=True,
    )


@app.get("/model/features", tags=["model"])
def model_features() -> dict[str, Any]:
    _require_model()
    assert service.required_features is not None
    return {
        "count": len(service.required_features),
        "features": service.required_features,
        "derived_features": ["DeviceInfo_is_missing", "id_33_is_missing"],
        "note": (
            "The two derived features are created from the raw DeviceInfo and id_33 "
            "fields, so clients do not need to send them explicitly."
        ),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(
    transaction: Annotated[dict[str, Any], Body(description="One raw transaction record")],
) -> PredictionResponse:
    _require_model()
    try:
        result = predict_transaction(
            transaction,
            model=service.model,
            preprocessor=service.preprocessor,
            threshold_config=service.threshold_config,
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except MemoryError as exc:
        raise HTTPException(status_code=503, detail="Not enough memory to score this request.") from exc
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("Single prediction failed")
        raise HTTPException(status_code=500, detail="Internal prediction error.") from exc
    return _prediction_response(result)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["inference"])
def predict_batch_endpoint(
    transactions: Annotated[
        list[dict[str, Any]], Body(description="Up to 500 raw transaction records")
    ],
) -> BatchPredictionResponse:
    _require_model()
    if not transactions:
        raise HTTPException(status_code=422, detail="transactions must contain at least one record.")
    if len(transactions) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Batch size {len(transactions)} exceeds the limit of {MAX_BATCH_SIZE}.",
        )
    try:
        import pandas as pd

        frame = predict_batch(
            pd.DataFrame(transactions),
            model=service.model,
            preprocessor=service.preprocessor,
            threshold_config=service.threshold_config,
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except MemoryError as exc:
        raise HTTPException(status_code=503, detail="Not enough memory to score this request.") from exc
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail="Internal batch prediction error.") from exc

    predictions = [
        PredictionResponse(
            transaction_id=_scalar(row["TransactionID"]),
            fraud_probability=float(row["Fraud_Probability"]),
            classification_threshold=float(row["Classification_Threshold"]),
            prediction=row["Prediction"],
            risk_level=row["Risk_Level"],
        )
        for _, row in frame.iterrows()
    ]
    return BatchPredictionResponse(count=len(predictions), predictions=predictions)


@app.post("/explain", response_model=ExplanationResponse, tags=["inference"])
def explain(
    transaction: Annotated[dict[str, Any], Body(description="One raw transaction record")],
    top_n: Annotated[
        int,
        Query(ge=1, le=20, description="Number of positive and negative factors to return"),
    ] = 5,
) -> ExplanationResponse:
    _require_model()
    try:
        result = explain_transaction(
            transaction,
            model=service.model,
            preprocessor=service.preprocessor,
            threshold_config=service.threshold_config,
            top_n=top_n,
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except MemoryError as exc:
        raise HTTPException(status_code=503, detail="Not enough memory to explain this request.") from exc
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("Explanation failed")
        raise HTTPException(status_code=500, detail="Internal explanation error.") from exc

    base = _prediction_response(result)
    return ExplanationResponse(
        **base.model_dump(),
        top_fraud_factors=[Factor(**item) for item in result["top_fraud_factors"]],
        top_legitimate_factors=[Factor(**item) for item in result["top_legitimate_factors"]],
    )
