"""
FastAPI application for model serving.

Provides REST endpoints for predictions and model information.
"""

import logging
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import settings
from src.serving import predict
from src.utils import CustomerInput

# Configure logging
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────
# FastAPI app initialization
# ─────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Predictor",
    description="ML pipeline for predicting customer churn using XGBoost",
    version="1.0.0",
)

# ─────────────────────────────────────────────────────
# CORS middleware
# ─────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────
# Response models
# ─────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    """Response model for predictions."""

    prediction: int
    churn_probability: float
    label: str


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Response model for model information."""

    algorithm: str
    model_path: str
    feature_count: int
    categorical_features: int
    numeric_features: int
    version: str


# ─────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns:
        Health status and timestamp.

    Example:
        >>> response = client.get("/health")
        >>> assert response.status_code == 200
    """
    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def make_prediction(customer: CustomerInput) -> PredictionResponse:
    """
    Predict customer churn.

    Args:
        customer: Customer input data (all 10 features required).

    Returns:
        Prediction result with probability and label.

    Raises:
        HTTPException: If prediction fails.

    Example:
        >>> data = {
        ...     'Age': 30, 'Tenure': 12, 'Usage Frequency': 14,
        ...     'Support Calls': 5, 'Payment Delay': 18,
        ...     'Subscription Type': 'Standard', 'Contract Length': 'Annual',
        ...     'Total Spend': 932, 'Last Interaction': 17, 'Gender': 'Female'
        ... }
        >>> response = client.post("/predict", json=data)
        >>> assert response.status_code == 200
    """
    try:
        # Convert Pydantic model to prediction input format
        input_data = customer.to_prediction_input()

        # Get prediction
        result = predict(input_data)

        logger.info(f"Prediction: {result['label']}")

        return PredictionResponse(
            prediction=result["prediction"],
            churn_probability=result["churn_probability"],
            label=result["label"],
        )

    except ValueError as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e

    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction service error",
        ) from e


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    """
    Get model metadata.

    Returns:
        Information about the deployed model.

    Example:
        >>> response = client.get("/model-info")
        >>> assert response.json()['algorithm'] == 'XGBoost'
    """
    return ModelInfoResponse(
        algorithm="XGBoost",
        model_path=settings.MODEL_PATH,
        feature_count=len(settings.FEATURE_NAMES),
        numeric_features=len(settings.NUMERIC_FEATURES),
        categorical_features=len(settings.CATEGORICAL_FEATURES),
        version="1.0.0",
    )


@app.get("/")
async def root() -> dict:
    """Root endpoint."""
    return {
        "message": "Customer Churn Predictor API",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model-info",
        "predict": "/predict (POST)",
    }


# ─────────────────────────────────────────────────────
# Exception handlers
# ─────────────────────────────────────────────────────


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions."""
    logger.error(f"ValueError: {str(exc)}")
    return {
        "error": "Invalid input",
        "detail": str(exc),
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting FastAPI server on 0.0.0.0:8000")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level=settings.LOG_LEVEL.lower(),
    )
