"""
Feature engineering module.

Builds sklearn preprocessing pipelines for numeric and categorical features.
"""

import logging

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from src.config import settings

logger = logging.getLogger(__name__)


def build_feature_pipeline() -> Pipeline:
    """
    Build sklearn feature preprocessing pipeline.

    Applies:
    - MinMaxScaler to numeric features
    - OneHotEncoder to categorical features

    Returns:
        Fitted sklearn Pipeline with ColumnTransformer.

    Example:
        >>> pipeline = build_feature_pipeline()
        >>> X_scaled = pipeline.fit_transform(X_train)
        >>> X_scaled.shape
        (800, 13)
    """
    logger.info("Building feature engineering pipeline")

    # ─────────────────────────────────────────────────────
    # Numeric pipeline: MinMaxScaler
    # ─────────────────────────────────────────────────────
    numeric_pipeline = Pipeline(
        steps=[
            ("scaler", MinMaxScaler()),
        ]
    )

    # ─────────────────────────────────────────────────────
    # Categorical pipeline: OneHotEncoder
    # ─────────────────────────────────────────────────────
    categorical_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    drop="first",
                    sparse_output=False,
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    # ─────────────────────────────────────────────────────
    # ColumnTransformer: combine pipelines
    # ─────────────────────────────────────────────────────
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, settings.NUMERIC_FEATURES),
            (
                "categorical",
                categorical_pipeline,
                settings.CATEGORICAL_FEATURES,
            ),
        ],
        remainder="passthrough",
    )

    # ─────────────────────────────────────────────────────
    # Final pipeline
    # ─────────────────────────────────────────────────────
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
        ]
    )

    logger.info(
        f"Pipeline built with {len(settings.NUMERIC_FEATURES)} numeric "
        f"and {len(settings.CATEGORICAL_FEATURES)} categorical features"
    )

    return pipeline


if __name__ == "__main__":
    # Smoke test
    import pandas as pd
    from src.data import load_data, preprocess_data

    train_path = settings.DATA_DIR / "train.csv"
    test_path = settings.DATA_DIR / "test.csv"

    train_df = load_data(str(train_path))
    test_df = load_data(str(test_path))

    df = preprocess_data([train_df, test_df])
    X = df.drop(columns=[settings.TARGET_COLUMN])

    pipeline = build_feature_pipeline()
    X_transformed = pipeline.fit_transform(X)

    print(f"✓ Pipeline built successfully")
    print(f"  Input shape: {X.shape}")
    print(f"  Output shape: {X_transformed.shape}")
