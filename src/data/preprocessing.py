"""Data preprocessing utilities."""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import GAIT_CYCLE_POINTS

logger = logging.getLogger(__name__)


def normalize_gait_cycle(timeseries: np.ndarray, n_points: int = GAIT_CYCLE_POINTS) -> np.ndarray:
    """Resample a time series to a fixed number of points (0-100% gait cycle)."""
    ts = np.asarray(timeseries, dtype=float)
    ts = ts[~np.isnan(ts)]
    if len(ts) < 2:
        return np.full(n_points, np.nan)

    x_old = np.linspace(0, 100, len(ts))
    x_new = np.linspace(0, 100, n_points)
    return np.interp(x_new, x_old, ts)


def remove_outliers(df: pd.DataFrame, columns: list, threshold: float = 1.5) -> pd.DataFrame:
    """Remove outliers using the IQR method."""
    mask = pd.Series(True, index=df.index)

    for col in columns:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        mask &= df[col].between(lower, upper)

    n_removed = (~mask).sum()
    if n_removed > 0:
        logger.info(f"Removed {n_removed} outlier rows ({n_removed / len(df) * 100:.1f}%)")

    return df[mask].reset_index(drop=True)


def standardize_features(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit scaler on training data and transform both sets."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def handle_missing(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """Impute missing values in numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if df[col].isna().any():
            if strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == "zero":
                df[col] = df[col].fillna(0)

    return df


def validate_feature_matrix(X: np.ndarray, y: np.ndarray) -> bool:
    """Validate feature matrix before training."""
    if X.shape[0] != y.shape[0]:
        logger.error(f"Shape mismatch: X has {X.shape[0]} samples, y has {y.shape[0]}")
        return False

    if np.any(np.isnan(X)):
        nan_count = np.isnan(X).sum()
        logger.warning(f"Feature matrix contains {nan_count} NaN values")
        return False

    if np.any(np.isinf(X)):
        logger.warning("Feature matrix contains infinite values")
        return False

    logger.info(f"Feature matrix valid: {X.shape[0]} samples, {X.shape[1]} features")
    return True
