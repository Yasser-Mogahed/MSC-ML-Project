"""
ML Pipeline Orchestration Script.

Orchestrates the full pipeline: load data -> preprocess -> train model -> save artifacts.

MLflow Integration: Logs model, hyperparameters, metrics, and classification report.
Model Serialization: Saves to both MLflow (production serving) and local file (development).

Usage:
    python scripts/run_pipeline.py
"""

import json
import logging
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Ensure project root is on sys.path so `src` can be imported when running this script directly
import os
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import settings
from src.data import load_data, preprocess_data
from src.models import train_model

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """
    Run the complete ML pipeline.

    Pipeline steps:
    1. Load train and test CSV files
    2. Preprocess (type conversion, encoding)
    3. Split into train/test (80/20 stratified)
    4. Train XGBoost model with Optuna hyperparameter tuning
    5. Evaluate on test set and log metrics
    6. Save model to MLflow + local file for serving
    7. Print detailed classification report

    Returns:
        Tuple of (trained_ml_pipeline, mlflow_run_id)
    """
    print("\n" + "=" * 80)
    print("[RUNNING] CUSTOMER CHURN PREDICTION - ML PIPELINE")
    print("=" * 80)

    # ─────────────────────────────────────────────────────
    # MLflow Setup
    # ─────────────────────────────────────────────────────
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    # ─────────────────────────────────────────────────────
    # [1/6] Load Data
    # ─────────────────────────────────────────────────────
    print("\n[1/6] [DATA] Loading data...")
    train_path = settings.DATA_DIR / "train.csv"
    test_path = settings.DATA_DIR / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")

    train_df = load_data(str(train_path))
    test_df = load_data(str(test_path))
    print(
        f"   [OK] Train: {train_df.shape[0]} rows × {train_df.shape[1]} cols"
    )
    print(f"   [OK] Test:  {test_df.shape[0]} rows × {test_df.shape[1]} cols")

    # ─────────────────────────────────────────────────────
    # [2/6] Preprocess
    # ─────────────────────────────────────────────────────
    print("\n[2/6] [PREPROCESS] Preprocessing data...")
    df = preprocess_data([train_df, test_df])
    print(f"   [OK] Combined: {df.shape[0]} rows × {df.shape[1]} cols")

    # ─────────────────────────────────────────────────────
    # [3/6] Split Data
    # ─────────────────────────────────────────────────────
    print("\n[3/6] [SPLIT] Splitting data (80/20 stratified)...")
    X = df.drop(columns=[settings.TARGET_COLUMN])
    y = df[settings.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=settings.TEST_SIZE,
        stratify=y,
        random_state=settings.RANDOM_STATE,
    )

    print(f"   [OK] Train: {X_train.shape[0]} samples")
    print(f"   [OK] Test:  {X_test.shape[0]} samples")
    print(
        f"   [OK] Class dist: {dict(y_train.value_counts().to_dict())}"
    )

    # ─────────────────────────────────────────────────────
    # [4/6] Train Model
    # ─────────────────────────────────────────────────────
    print(
        f"\n[4/6] [TRAIN] Training XGBoost with Optuna..."
    )
    print(
        f"   • CV: {settings.CV_SPLITS}-fold"
    )
    print(
        f"   • Trials: {settings.OPTUNA_N_TRIALS}"
    )
    print(
        f"   • Metric: {settings.OPTUNA_SCORING}"
    )

    train_start = time.time()
    ml_pipeline, run_id = train_model(
        X_train, y_train, X_test, y_test
    )
    train_time = time.time() - train_start

    print(f"   [OK] Training completed in {train_time:.2f}s")
    print(f"   [OK] MLflow Run ID: {run_id}")

    # ─────────────────────────────────────────────────────
    # [5/6] Evaluate Model
    # ─────────────────────────────────────────────────────
    print("\n[5/6] [EVAL] Evaluating on test set...")

    pred_start = time.time()
    y_pred = ml_pipeline.predict(X_test)
    y_pred_proba = ml_pipeline.predict_proba(X_test)[:, 1]
    pred_time = time.time() - pred_start

    # Calculate metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"   [OK] Precision:  {precision:.4f}")
    print(f"   [OK] Recall:     {recall:.4f}")
    print(f"   [OK] F1 Score:   {f1:.4f}")
    print(f"   [OK] ROC AUC:    {roc_auc:.4f}")

    # ─────────────────────────────────────────────────────
    # [6/6] Save Model & Log Artifacts
    # ─────────────────────────────────────────────────────
    print("\n[6/6] [SAVE] Saving model artifacts...")

    # Save locally
    settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(ml_pipeline, settings.MODEL_PATH)
    file_size_mb = Path(settings.MODEL_PATH).stat().st_size / (1024 ** 2)
    print(f"   [OK] MLflow: models:/xgboost-churn-classifier/latest")
    print(f"   [OK] Local:  {settings.MODEL_PATH} ({file_size_mb:.2f} MB)")

    # Log classification report as text artifact
    class_report = classification_report(
        y_test, y_pred, digits=3, zero_division=0
    )

    # ─────────────────────────────────────────────────────
    # Print Summary
    # ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("[SUCCESS] PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)

    print("\n[METRICS] PERFORMANCE SUMMARY")
    print("-" * 80)
    print(f"Training Time:        {train_time:.2f} seconds")
    print(f"Prediction Time:      {pred_time:.4f} seconds")
    print(f"Throughput:           {len(X_test) / pred_time:.0f} samples/sec")
    print("-" * 80)
    print(f"Test Precision:       {precision:.4f}")
    print(f"Test Recall:          {recall:.4f}")
    print(f"Test F1 Score:        {f1:.4f}")
    print(f"Test ROC AUC:         {roc_auc:.4f}")

    print("\n[REPORT] DETAILED CLASSIFICATION REPORT")
    print("-" * 80)
    print(class_report)
    print("-" * 80)

    print("\n[ARTIFACTS] MODEL ARTIFACTS")
    print("-" * 80)
    print(f"MLflow Tracking URI:  {settings.MLFLOW_TRACKING_URI}")
    print(f"Experiment:           {settings.MLFLOW_EXPERIMENT_NAME}")
    print(f"Run ID:               {run_id}")
    print(f"Model Path (Local):   {settings.MODEL_PATH}")
    print(f"Model Path (MLflow):  models:/xgboost-churn-classifier/latest")
    print("-" * 80)

    print("\n[NEXT] NEXT STEPS")
    print("-" * 80)
    print("1. Start API server:     uvicorn src.app.main:app --reload")
    print("2. Start dashboard:      streamlit run src/app/ui.py")
    print("3. View MLflow:          mlflow ui --backend-store-uri ./mlruns")
    print("-" * 80)

    print("\n[DONE] Pipeline finished! Model ready for serving.\n")

    return ml_pipeline, run_id


if __name__ == "__main__":
    try:
        model, run_id = main()
        exit(0)
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {str(e)}", exc_info=True)
        print(f"\n[ERROR] {str(e)}\n")
        exit(1)
