"""
Data loading module.

Provides utilities for loading and validating CSV data.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load CSV data from file.

    Args:
        filepath: Path to CSV file.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file is not a CSV.

    Example:
        >>> df = load_data('data/train.csv')
        >>> df.shape
        (1000, 12)
    """
    filepath = Path(filepath)

    # Validate file exists
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Validate CSV extension
    if filepath.suffix.lower() != ".csv":
        raise ValueError(f"Expected CSV file, got: {filepath.suffix}")

    # Load data
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)

    # Log info
    logger.info(f"Loaded shape: {df.shape}")
    logger.info(f"Dtypes:\n{df.dtypes}")
    logger.info(f"Missing values:\n{df.isna().sum()}")

    return df


if __name__ == "__main__":
    # Smoke test
    from src.config import settings

    train_path = settings.DATA_DIR / "train.csv"
    df = load_data(str(train_path))
    print(f"✓ Loaded {df.shape[0]} rows, {df.shape[1]} columns")
