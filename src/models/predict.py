"""Inference pipeline for ACL injury risk prediction from video."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import joblib

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import MODELS_DIR, RISK_THRESHOLDS

logger = logging.getLogger(__name__)


@dataclass
class RiskAssessment:
    """ACL injury risk assessment result."""
    risk_score: float
    risk_level: str
    contributing_factors: list = field(default_factory=list)
    raw_features: dict = field(default_factory=dict)
    confidence: float = 0.0
    model_name: str = ""

    def to_dict(self) -> dict:
        return {
            "risk_score": round(self.risk_score, 4),
            "risk_level": self.risk_level,
            "contributing_factors": self.contributing_factors,
            "confidence": round(self.confidence, 4),
            "model_name": self.model_name,
        }


def classify_risk_level(score: float) -> str:
    """Map risk score to risk level."""
    if score < RISK_THRESHOLDS["low"]:
        return "Low"
    elif score < RISK_THRESHOLDS["moderate"]:
        return "Moderate"
    else:
        return "High"


def get_contributing_factors(
    features: np.ndarray, feature_names: list, model, top_n: int = 5
) -> list:
    """Identify top contributing features to the risk prediction."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return []

    # Weight by feature value magnitude and importance
    weighted = importances * np.abs(features.flatten())
    top_indices = np.argsort(weighted)[-top_n:][::-1]

    factors = []
    for idx in top_indices:
        factors.append({
            "feature": feature_names[idx],
            "importance": float(importances[idx]),
            "value": float(features.flatten()[idx]),
        })

    return factors


def load_model(model_path: Path = None, model_name: str = "random_forest") -> dict:
    """Load saved model artifact."""
    if model_path is None:
        model_path = MODELS_DIR / f"{model_name}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    return joblib.load(model_path)


def predict_from_features(
    features: np.ndarray, model_path: Path = None, model_name: str = "random_forest"
) -> RiskAssessment:
    """Predict risk from pre-extracted feature vector.

    Args:
        features: Feature vector of shape (n_features,) or (1, n_features).
        model_path: Path to model .joblib file.
        model_name: Model name if model_path not provided.

    Returns:
        RiskAssessment with score, level, and contributing factors.
    """
    artifact = load_model(model_path, model_name)
    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_names = artifact["feature_names"]

    X = features.reshape(1, -1) if features.ndim == 1 else features
    X_scaled = scaler.transform(X)

    risk_score = float(model.predict_proba(X_scaled)[0, 1])
    risk_level = classify_risk_level(risk_score)
    factors = get_contributing_factors(X_scaled, feature_names, model)

    return RiskAssessment(
        risk_score=risk_score,
        risk_level=risk_level,
        contributing_factors=factors,
        raw_features=dict(zip(feature_names, features.flatten())),
        confidence=abs(risk_score - 0.5) * 2,  # Distance from decision boundary
        model_name=model_name,
    )


def predict_from_video(
    video_path: str, model_path: Path = None, model_name: str = "random_forest"
) -> Optional[RiskAssessment]:
    """End-to-end prediction from video file.

    Pipeline:
    1. Extract pose landmarks from video
    2. Compute joint angles per frame
    3. Extract gait features
    4. Predict risk score

    Args:
        video_path: Path to input video file.
        model_path: Path to model .joblib file.
        model_name: Model name if model_path not provided.

    Returns:
        RiskAssessment or None if pose extraction fails.
    """
    from src.pose.video_processor import process_video_to_features

    result = process_video_to_features(video_path)
    if result is None:
        logger.error("Failed to extract features from video.")
        return None

    features, landmarks_seq, angles_seq = result
    assessment = predict_from_features(features, model_path, model_name)

    # Compute confidence from landmark visibility
    visibility_scores = []
    for frame_landmarks in landmarks_seq:
        if frame_landmarks and "visibility" in frame_landmarks:
            visibility_scores.append(frame_landmarks["visibility"])

    if visibility_scores:
        assessment.confidence *= np.mean(visibility_scores)

    return assessment
