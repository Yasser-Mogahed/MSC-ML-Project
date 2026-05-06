"""
Model training module.

Handles hyperparameter tuning with Optuna and MLflow experiment tracking.
"""

import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import accuracy_score, classification_report, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline, clone
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import settings
from src.features import build_feature_pipeline

logger = logging.getLogger(__name__)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray = None,
    y_test: np.ndarray = None,
) -> tuple[Pipeline, str]:
    """
    Train XGBoost model with Optuna hyperparameter tuning and MLflow tracking.

    Performs:
    - Optuna hyperparameter optimization (50 trials)
    - StratifiedKFold cross-validation (5 splits)
    - MLflow experiment tracking and model logging
    - Local model saving to MODEL_PATH

    Args:
        X_train: Training features.
        y_train: Training target.
        X_test: Optional test features for evaluation.
        y_test: Optional test target for evaluation.

    Returns:
        Tuple of (trained_pipeline, mlflow_run_id).

    Example:
        >>> from src.data import load_data, preprocess_data
        >>> from src.features import build_feature_pipeline
        >>> df = preprocess_data([train_df, test_df])
        >>> X = df.drop(columns=['Churn'])
        >>> y = df['Churn']
        >>> model, run_id = train_model(X, y)
        >>> print(f"Model saved: {settings.MODEL_PATH}")
    """
    # ─────────────────────────────────────────────────────
    # Setup MLflow
    # ─────────────────────────────────────────────────────
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    # ─────────────────────────────────────────────────────
    # Build feature pipeline
    # ─────────────────────────────────────────────────────
    feature_pipeline = build_feature_pipeline()

    # ─────────────────────────────────────────────────────
    # Setup cross-validation
    # ─────────────────────────────────────────────────────
    skf = StratifiedKFold(
        n_splits=settings.CV_SPLITS,
        shuffle=True,
        random_state=settings.RANDOM_STATE,
    )

    # ─────────────────────────────────────────────────────
    # Optuna objective function for XGBoost
    # ─────────────────────────────────────────────────────
    def objective_xgb(trial: optuna.Trial) -> float:
        """Optuna objective for XGBoost hyperparameter tuning."""
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0001, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0001, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
        }

        model = XGBClassifier(
            **params,
            random_state=settings.RANDOM_STATE,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=-1,
        )

        ml_pipeline = Pipeline(
            [
                ("preprocessor", clone(feature_pipeline)),
                ("classifier", model),
            ]
        )

        # Cross-validation scoring (using recall as per notebook)
        scores = cross_val_score(
            ml_pipeline,
            X_train,
            y_train,
            cv=skf,
            scoring=settings.OPTUNA_SCORING,
            n_jobs=-1,
        )

        # Return mean score with penalty for variance
        mean_score = scores.mean()
        std_penalty = scores.std() * 0.5
        return mean_score - std_penalty

    # ─────────────────────────────────────────────────────
    # Optuna study
    # ─────────────────────────────────────────────────────
    logger.info("Starting Optuna hyperparameter tuning")
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=settings.RANDOM_STATE),
    )
    study.optimize(
        objective_xgb,
        n_trials=settings.OPTUNA_N_TRIALS,
        show_progress_bar=True,
    )

    logger.info(f"✓ Best CV {settings.OPTUNA_SCORING}: {study.best_value:.4f}")
    logger.info(f"  Best params: {study.best_params}")

    # ─────────────────────────────────────────────────────
    # Train final model with best hyperparameters
    # ─────────────────────────────────────────────────────
    final_model = XGBClassifier(
        **study.best_params,
        random_state=settings.RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0,
        n_jobs=-1,
    )

    final_pipeline = Pipeline(
        [
            ("preprocessor", clone(feature_pipeline)),
            ("classifier", final_model),
        ]
    )

    final_pipeline.fit(X_train, y_train)
    logger.info("✓ Model training completed")

    # ─────────────────────────────────────────────────────
    # MLflow tracking context
    # ─────────────────────────────────────────────────────
    with mlflow.start_run() as run:
        # Log hyperparameters
        mlflow.log_params(study.best_params)
        mlflow.log_param("cv_splits", settings.CV_SPLITS)
        mlflow.log_param("random_state", settings.RANDOM_STATE)

        # Log metrics
        mlflow.log_metric("best_cv_score", study.best_value)

        # Evaluate on test set if provided
        if X_test is not None and y_test is not None:
            y_pred = final_pipeline.predict(X_test)
            y_pred_proba = final_pipeline.predict_proba(X_test)[:, 1]

            test_accuracy = accuracy_score(y_test, y_pred)
            test_recall = recall_score(y_test, y_pred)

            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("test_recall", test_recall)

            logger.info(f"Test Accuracy: {test_accuracy:.4f}")
            logger.info(f"Test Recall: {test_recall:.4f}")
            logger.info("Classification Report:")
            logger.info(
                f"\n{classification_report(y_test, y_pred, digits=3)}"
            )

        # Log model to MLflow
        mlflow.sklearn.log_model(
            final_pipeline,
            artifact_path="model",
            registered_model_name="xgboost-churn-classifier",
        )

        # Save model locally to MODEL_PATH
        settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)

        import joblib

        joblib.dump(final_pipeline, settings.MODEL_PATH)
        logger.info(f"✓ Model saved locally to {settings.MODEL_PATH}")

        run_id = run.info.run_id
        logger.info(f"✓ MLflow run ID: {run_id}")

    return final_pipeline, run_id


if __name__ == "__main__":
    # Smoke test
    from sklearn.model_selection import train_test_split

    from src.data import load_data, preprocess_data

    train_path = settings.DATA_DIR / "train.csv"
    test_path = settings.DATA_DIR / "test.csv"

    train_df = load_data(str(train_path))
    test_df = load_data(str(test_path))

    df = preprocess_data([train_df, test_df])
    X = df.drop(columns=[settings.TARGET_COLUMN])
    y = df[settings.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=settings.TEST_SIZE,
        stratify=y,
        random_state=settings.RANDOM_STATE,
    )

    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    model, run_id = train_model(X_train, y_train, X_test, y_test)
    print(f"✓ Model trained and saved")
    print(f"  Run ID: {run_id}")
