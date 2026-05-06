"""
Inference module.

Loads trained model and provides prediction interface.
"""

import logging
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_model():
    """
    Load trained XGBoost model from disk or MLflow (cached).

    Tries in this order:
    1. Local .joblib file (fastest, development)
    2. MLflow logged model (production, from tracking server)

    Returns:
        Trained sklearn Pipeline.

    Raises:
        FileNotFoundError: If model not found in either location.
    """
    model_path = Path(settings.MODEL_PATH)

    # Try local file first (fastest for development/serving)
    if model_path.exists():
        logger.info(f"Loading model from local file: {model_path}")
        model = joblib.load(model_path)
        logger.info("✓ Model loaded successfully")
        return model

    # Fallback to MLflow artifact store (for production serving)
    logger.info(
        f"Local model not found, loading from MLflow: {settings.MLFLOW_TRACKING_URI}"
    )
    import mlflow.sklearn

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    try:
        # Load latest version from model registry (registered as 'xgboost-churn-classifier')
        model = mlflow.sklearn.load_model("models:/xgboost-churn-classifier/latest")
        logger.info("✓ Model loaded from MLflow model registry")
        return model
    except Exception as e:
        raise FileNotFoundError(
            f"Model not found in local path ({model_path}) or MLflow ({settings.MLFLOW_TRACKING_URI}). "
            f"Train model first: python scripts/run_pipeline.py\nError: {str(e)}"
        )


def predict(input_data: dict) -> dict:
    """
    Generate prediction for a single customer.

    Args:
        input_data: Dictionary with 10 features matching settings.FEATURE_NAMES.

    Returns:
        Dictionary with keys:
            - 'prediction': int (0 or 1)
            - 'churn_probability': float (0.0 to 1.0)
            - 'label': str ('No Churn' or 'Churn')

    Raises:
        ValueError: If features are missing or input shape is invalid.

    Example:
        >>> data = {
        ...     'Age': 30, 'Tenure': 12, 'Usage Frequency': 14,
        ...     'Support Calls': 5, 'Payment Delay': 18,
        ...     'Subscription Type': 'Standard', 'Contract Length': 'Annual',
        ...     'Total Spend': 932, 'Last Interaction': 17, 'Gender': 'Female'
        ... }
        >>> result = predict(data)
        >>> print(result['label'])
        'No Churn'
    """
    logger.debug(f"Received prediction request with keys: {list(input_data.keys())}")

    # ─────────────────────────────────────────────────────
    # Validate input shape
    # ─────────────────────────────────────────────────────
    missing_features = set(settings.FEATURE_NAMES) - set(input_data.keys())
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    # ─────────────────────────────────────────────────────
    # Create single-row DataFrame with correct feature order
    # ─────────────────────────────────────────────────────
    df = pd.DataFrame(
        [input_data],
        columns=settings.FEATURE_NAMES,
    )

    # ─────────────────────────────────────────────────────
    # Load model (cached)
    # ─────────────────────────────────────────────────────
    model = _load_model()

    # ─────────────────────────────────────────────────────
    # Predict
    # ─────────────────────────────────────────────────────
    try:
        prediction = model.predict(df)[0]
        churn_probability = model.predict_proba(df)[0][1]
    except Exception as e:
        raise ValueError(f"Model prediction failed: {str(e)}")

    # ─────────────────────────────────────────────────────
    # Format response
    # ─────────────────────────────────────────────────────
    label = "Churn" if prediction == 1 else "No Churn"

    result = {
        "prediction": int(prediction),
        "churn_probability": float(churn_probability),
        "label": label,
    }

    logger.debug(
        f"Prediction: {label} (probability: {churn_probability:.4f})"
    )

    return result


if __name__ == "__main__":
    # Smoke test
    test_input = {
        "Age": 30,
        "Tenure": 12,
        "Usage Frequency": 14,
        "Support Calls": 5,
        "Payment Delay": 18,
        "Subscription Type": "Standard",
        "Contract Length": "Annual",
        "Total Spend": 932,
        "Last Interaction": 17,
        "Gender": "Female",
    }

    try:
        result = predict(test_input)
        print("✓ Inference test successful")
        print(f"  Result: {result}")
    except FileNotFoundError as e:
        print(f"⚠ Model not yet trained: {e}")
    except Exception as e:
        print(f"✗ Error: {e}")
