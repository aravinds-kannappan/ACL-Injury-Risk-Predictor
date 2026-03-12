"""Tests for ML model training and prediction."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.predict import classify_risk_level, RiskAssessment


class TestRiskClassification:
    """Tests for risk level classification."""

    def test_low_risk(self):
        assert classify_risk_level(0.1) == "Low"
        assert classify_risk_level(0.0) == "Low"
        assert classify_risk_level(0.29) == "Low"

    def test_moderate_risk(self):
        assert classify_risk_level(0.3) == "Moderate"
        assert classify_risk_level(0.5) == "Moderate"
        assert classify_risk_level(0.69) == "Moderate"

    def test_high_risk(self):
        assert classify_risk_level(0.7) == "High"
        assert classify_risk_level(0.9) == "High"
        assert classify_risk_level(1.0) == "High"


class TestRiskAssessment:
    """Tests for RiskAssessment dataclass."""

    def test_to_dict(self):
        ra = RiskAssessment(
            risk_score=0.65,
            risk_level="Moderate",
            contributing_factors=[{"feature": "knee_valgus", "importance": 0.3}],
            confidence=0.8,
            model_name="random_forest",
        )
        d = ra.to_dict()
        assert d["risk_score"] == 0.65
        assert d["risk_level"] == "Moderate"
        assert len(d["contributing_factors"]) == 1

    def test_default_values(self):
        ra = RiskAssessment(risk_score=0.5, risk_level="Moderate")
        assert ra.contributing_factors == []
        assert ra.raw_features == {}
        assert ra.confidence == 0.0


class TestModelTraining:
    """Tests for model training on small synthetic data."""

    def test_train_small_dataset(self):
        """Models should train without error on small data."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = np.array([0] * 30 + [1] * 20)

        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X, y)
        assert hasattr(rf, "predict_proba")

        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X, y)
        assert hasattr(lr, "predict_proba")

    def test_predictions_in_range(self):
        """Probabilities should be in [0, 1]."""
        from sklearn.ensemble import RandomForestClassifier

        np.random.seed(42)
        X_train = np.random.randn(50, 10)
        y_train = np.array([0] * 30 + [1] * 20)
        X_test = np.random.randn(10, 10)

        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X_train, y_train)

        proba = rf.predict_proba(X_test)[:, 1]
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)
