"""
Data preprocessing module.

Handles data cleaning, type conversions, and feature engineering preparation.
"""

import logging
from typing import List, Union

import pandas as pd

from src.config import settings

logger = logging.getLogger(__name__)


def preprocess_data(
    df_input: Union[pd.DataFrame, List[pd.DataFrame]]
) -> pd.DataFrame:
    """
    Preprocess raw data for modeling.

    Handles:
    - Dropping CustomerID
    - Concatenating train/test if passed as list
    - Type conversions (int/float)
    - Dropping NaN rows
    - Encoding target variable

    Args:
        df_input: Single DataFrame or list of [train_df, test_df].

    Returns:
        Cleaned DataFrame ready for feature engineering.

    Example:
        >>> from src.data import load_data
        >>> train = load_data('data/train.csv')
        >>> test = load_data('data/test.csv')
        >>> df = preprocess_data([train, test])
        >>> df.shape
        (2000, 11)
    """
    # Handle list input (train/test concatenation)
    if isinstance(df_input, list):
        logger.info(f"Concatenating {len(df_input)} DataFrames")
        df = pd.concat(df_input, axis=0, ignore_index=True)
    else:
        df = df_input.copy()

    logger.info(f"Starting with shape: {df.shape}")

    # ─────────────────────────────────────────────────────
    # Drop CustomerID if present
    # ─────────────────────────────────────────────────────
    if "CustomerID" in df.columns:
        logger.info("Dropping CustomerID column")
        df = df.drop(columns=["CustomerID"])

    # ─────────────────────────────────────────────────────
    # Type conversions: int columns
    # ─────────────────────────────────────────────────────
    int_columns = [
        "Age",
        "Usage Frequency",
        "Support Calls",
        "Payment Delay",
        "Last Interaction",
    ]

    for col in int_columns:
        if col in df.columns:
            logger.info(f"Converting {col} to int")
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # ─────────────────────────────────────────────────────
    # Type conversions: float columns
    # ─────────────────────────────────────────────────────
    float_columns = ["Tenure", "Total Spend"]

    for col in float_columns:
        if col in df.columns:
            logger.info(f"Converting {col} to float")
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ─────────────────────────────────────────────────────
    # Encode target: Churn (Yes/No or 1/0 -> 1/0)
    # ─────────────────────────────────────────────────────
    if settings.TARGET_COLUMN in df.columns:
        if df[settings.TARGET_COLUMN].dtype == "object":
            logger.info("Encoding Churn from Yes/No to 1/0")
            df[settings.TARGET_COLUMN] = (
                df[settings.TARGET_COLUMN].map({"Yes": 1, "No": 0})
            )

    # ─────────────────────────────────────────────────────
    # Drop NaN rows
    # ─────────────────────────────────────────────────────
    rows_before = len(df)
    df = df.dropna()
    rows_dropped = rows_before - len(df)
    if rows_dropped > 0:
        logger.info(f"Dropped {rows_dropped} rows with NaN values")

    logger.info(f"Final shape: {df.shape}")
    logger.info(f"Dtypes after preprocessing:\n{df.dtypes}")

    return df


if __name__ == "__main__":
    # Smoke test
    from src.data import load_data

    train_path = settings.DATA_DIR / "train.csv"
    test_path = settings.DATA_DIR / "test.csv"

    train_df = load_data(str(train_path))
    test_df = load_data(str(test_path))

    df = preprocess_data([train_df, test_df])
    print(f"✓ Preprocessed shape: {df.shape}")
    print(f"  Dtypes:\n{df.dtypes}")
