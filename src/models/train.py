"""Model training pipeline for ACL injury risk prediction."""

import logging
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import RANDOM_SEED, TEST_SIZE, CV_FOLDS, MODELS_DIR, PROCESSED_DATA_DIR
from src.features.feature_pipeline import FEATURE_NAMES

logger = logging.getLogger(__name__)


def prepare_data(X: np.ndarray, y: np.ndarray):
    """Split and scale data for training.

    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    logger.info(
        f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples"
    )
    return X_train, X_test, y_train, y_test, scaler


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Train Random Forest with hyperparameter tuning via GridSearchCV."""
    param_grid = {
        "n_estimators": [100, 200, 500],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    logger.info("Training Random Forest with GridSearchCV...")
    grid_search.fit(X_train, y_train)

    logger.info(f"Best params: {grid_search.best_params_}")
    logger.info(f"Best CV ROC AUC: {grid_search.best_score_:.4f}")

    return {
        "model": grid_search.best_estimator_,
        "best_params": grid_search.best_params_,
        "cv_score": grid_search.best_score_,
        "cv_results": grid_search.cv_results_,
    }


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Train Logistic Regression baseline with regularization tuning."""
    param_grid = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "penalty": ["l1", "l2"],
    }

    lr = LogisticRegression(
        solver="saga",
        class_weight="balanced",
        random_state=RANDOM_SEED,
        max_iter=5000,
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    grid_search = GridSearchCV(
        lr,
        param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    logger.info("Training Logistic Regression with GridSearchCV...")
    grid_search.fit(X_train, y_train)

    logger.info(f"Best params: {grid_search.best_params_}")
    logger.info(f"Best CV ROC AUC: {grid_search.best_score_:.4f}")

    return {
        "model": grid_search.best_estimator_,
        "best_params": grid_search.best_params_,
        "cv_score": grid_search.best_score_,
        "cv_results": grid_search.cv_results_,
    }


def save_model(
    model, scaler: StandardScaler, feature_names: list,
    model_name: str, metrics: dict = None
) -> Path:
    """Save trained model with metadata."""
    save_path = MODELS_DIR / f"{model_name}.joblib"

    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "model_name": model_name,
        "metrics": metrics or {},
    }

    joblib.dump(artifact, save_path)
    logger.info(f"Model saved to {save_path}")
    return save_path


def train_pipeline(X: np.ndarray, y: np.ndarray) -> dict:
    """Full training pipeline: split, train both models, save.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)

    Returns:
        Dict with training results for both models.
    """
    X_train, X_test, y_train, y_test, scaler = prepare_data(X, y)

    # Train Random Forest
    rf_results = train_random_forest(X_train, y_train)
    rf_model = rf_results["model"]
    save_model(
        rf_model, scaler, FEATURE_NAMES,
        "random_forest",
        {"cv_roc_auc": rf_results["cv_score"]},
    )

    # Train Logistic Regression
    lr_results = train_logistic_regression(X_train, y_train)
    lr_model = lr_results["model"]
    save_model(
        lr_model, scaler, FEATURE_NAMES,
        "logistic_regression",
        {"cv_roc_auc": lr_results["cv_score"]},
    )

    return {
        "random_forest": rf_results,
        "logistic_regression": lr_results,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
    }
