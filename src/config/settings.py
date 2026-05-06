"""
Centralized configuration for the ML pipeline.

Contains all constants, feature definitions, and environment settings.
"""

import logging
from pathlib import Path
from typing import List

try:
    # pydantic v2 uses a separate pydantic-settings package
    from pydantic_settings import BaseSettings
except Exception:  # pragma: no cover - fallback for environments without pydantic-settings
    try:
        # Fallback to pydantic's BaseSettings (works with pydantic v1)
        from pydantic import BaseSettings
    except Exception as exc:  # pragma: no cover - user must install pydantic
        raise ImportError(
            "pydantic or pydantic-settings is required. Install with: pip install pydantic pydantic-settings"
        ) from exc


class Settings(BaseSettings):
    """
    Application settings using Pydantic BaseSettings.

    Loads from environment variables with defaults.
    """

    # ─────────────────────────────────────────────────────
    # Paths and Model Configuration
    # ─────────────────────────────────────────────────────
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    MODEL_DIR: Path = PROJECT_ROOT / "src" / "serving" / "model"
    MODEL_PATH: str = str(MODEL_DIR / "xgboost_model.joblib")
    MLFLOW_TRACKING_URI: str = "./mlruns"

    # ─────────────────────────────────────────────────────
    # API Configuration
    # ─────────────────────────────────────────────────────
    API_URL: str = "http://localhost:8000"
    PREDICT_ENDPOINT: str = "/predict"
    HEALTH_ENDPOINT: str = "/health"

    # ─────────────────────────────────────────────────────
    # Logging Configuration
    # ─────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"

    # ─────────────────────────────────────────────────────
    # Feature Definitions (11 features + 1 target)
    # ─────────────────────────────────────────────────────
    # All numeric features (will be MinMaxScaled)
    NUMERIC_FEATURES: List[str] = [
        "Age",
        "Tenure",
        "Usage Frequency",
        "Support Calls",
        "Payment Delay",
        "Total Spend",
        "Last Interaction",
    ]

    # All categorical features (will be OneHotEncoded)
    CATEGORICAL_FEATURES: List[str] = [
        "Subscription Type",
        "Contract Length",
        "Gender",
    ]

    # Complete ordered list of input features (for DataFrame column ordering)
    FEATURE_NAMES: List[str] = [
        "Age",
        "Tenure",
        "Usage Frequency",
        "Support Calls",
        "Payment Delay",
        "Subscription Type",
        "Contract Length",
        "Total Spend",
        "Last Interaction",
        "Gender",
    ]

    # Target column
    TARGET_COLUMN: str = "Churn"

    # ─────────────────────────────────────────────────────
    # Model Training Configuration
    # ─────────────────────────────────────────────────────
    RANDOM_STATE: int = 30
    TEST_SIZE: float = 0.2
    CV_SPLITS: int = 5
    OPTUNA_N_TRIALS: int = 50
    OPTUNA_SCORING: str = "recall"

    # ─────────────────────────────────────────────────────
    # MLflow Configuration
    # ─────────────────────────────────────────────────────
    MLFLOW_EXPERIMENT_NAME: str = "churn-classification"

    # ─────────────────────────────────────────────────────
    # Streamlit Configuration
    # ─────────────────────────────────────────────────────
    STREAMLIT_PAGE_CONFIG: dict = {
        "page_title": "Customer Churn Predictor",
        "page_icon": "📊",
        "layout": "wide",
        "initial_sidebar_state": "expanded",
    }

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        case_sensitive = False


# Singleton instance
settings = Settings()

# Configure logging based on settings
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


if __name__ == "__main__":
    # Smoke test
    print("✓ Settings loaded successfully")
    print(f"  Project Root: {settings.PROJECT_ROOT}")
    print(f"  Model Path: {settings.MODEL_PATH}")
    print(f"  Features: {len(settings.FEATURE_NAMES)} input features")
    print(f"  Numeric: {settings.NUMERIC_FEATURES}")
    print(f"  Categorical: {settings.CATEGORICAL_FEATURES}")
