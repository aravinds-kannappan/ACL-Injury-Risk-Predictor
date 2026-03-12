"""Tests for feature engineering modules."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.joint_angles import (
    compute_angle_3points,
    compute_knee_flexion,
    compute_knee_valgus,
    compute_all_angles,
)
from src.features.gait_features import (
    extract_cycle_features,
    compute_asymmetry,
    aggregate_cycles,
)
from src.features.feature_pipeline import (
    extract_features_from_timeseries,
    get_feature_names,
    FEATURE_NAMES,
    N_FEATURES,
)


class TestJointAngles:
    """Tests for joint angle computation."""

    def test_right_angle(self):
        """90-degree angle from perpendicular vectors."""
        a = np.array([1, 0, 0])
        b = np.array([0, 0, 0])
        c = np.array([0, 1, 0])
        angle = compute_angle_3points(a, b, c)
        assert abs(angle - 90.0) < 0.01

    def test_straight_line(self):
        """180-degree angle from collinear points."""
        a = np.array([0, 0, 0])
        b = np.array([1, 0, 0])
        c = np.array([2, 0, 0])
        angle = compute_angle_3points(a, b, c)
        assert abs(angle - 180.0) < 0.01

    def test_60_degree_triangle(self):
        """60-degree angle from equilateral triangle vertices."""
        a = np.array([0, 0, 0])
        b = np.array([1, 0, 0])
        c = np.array([0.5, np.sqrt(3) / 2, 0])
        angle = compute_angle_3points(a, b, c)
        assert abs(angle - 60.0) < 0.01

    def test_zero_length_vector(self):
        """Should return 0 for coincident points."""
        a = np.array([1, 0, 0])
        b = np.array([1, 0, 0])
        c = np.array([2, 0, 0])
        angle = compute_angle_3points(a, b, c)
        assert angle == 0.0

    def test_knee_flexion_straight_leg(self):
        """Straight leg should give ~180 degrees."""
        hip = np.array([0, 0, 0])
        knee = np.array([0, 0, -0.4])
        ankle = np.array([0, 0, -0.8])
        angle = compute_knee_flexion(hip, knee, ankle)
        assert abs(angle - 180.0) < 1.0

    def test_knee_valgus_aligned(self):
        """Aligned hip-knee-ankle should give ~180 degrees (no valgus)."""
        hip = np.array([0, 0, 0])
        knee = np.array([0, -0.4, 0])
        ankle = np.array([0, -0.8, 0])
        angle = compute_knee_valgus(hip, knee, ankle)
        assert abs(angle - 180.0) < 1.0

    def test_compute_all_angles(self):
        """All angles should be computed from complete landmarks."""
        landmarks = {
            "left_shoulder": np.array([-0.2, -0.5, 0]),
            "right_shoulder": np.array([0.2, -0.5, 0]),
            "left_hip": np.array([-0.15, 0, 0]),
            "right_hip": np.array([0.15, 0, 0]),
            "left_knee": np.array([-0.15, 0.4, 0]),
            "right_knee": np.array([0.15, 0.4, 0]),
            "left_ankle": np.array([-0.15, 0.8, 0]),
            "right_ankle": np.array([0.15, 0.8, 0]),
            "left_foot_index": np.array([-0.1, 0.85, 0.1]),
            "right_foot_index": np.array([0.1, 0.85, 0.1]),
        }
        angles = compute_all_angles(landmarks)

        assert "left_knee_flexion" in angles
        assert "right_knee_flexion" in angles
        assert "left_hip_flexion" in angles
        assert "trunk_lean" in angles
        assert all(isinstance(v, float) for v in angles.values())


class TestGaitFeatures:
    """Tests for gait cycle feature extraction."""

    def test_extract_cycle_features(self):
        """Should extract all expected features from a sine wave."""
        ts = np.sin(np.linspace(0, 2 * np.pi, 101)) * 30 + 150
        features = extract_cycle_features(ts)

        assert "mean" in features
        assert "std" in features
        assert "max" in features
        assert "min" in features
        assert "range" in features
        assert "angular_velocity_max" in features
        assert abs(features["range"] - 60.0) < 1.0

    def test_extract_empty_series(self):
        """Should return empty dict for too-short series."""
        features = extract_cycle_features(np.array([1, 2]))
        assert features == {}

    def test_asymmetry_symmetric(self):
        """Identical left/right should give near-zero asymmetry."""
        left = {"mean": 150.0, "max": 170.0}
        right = {"mean": 150.0, "max": 170.0}
        asym = compute_asymmetry(left, right)
        assert all(v < 0.001 for v in asym.values())

    def test_asymmetry_asymmetric(self):
        """Known difference should give expected ratio."""
        left = {"mean": 100.0}
        right = {"mean": 50.0}
        asym = compute_asymmetry(left, right)
        assert abs(asym["mean_asymmetry"] - 0.5) < 0.01

    def test_aggregate_cycles(self):
        """Average should be correct across multiple cycles."""
        cycles = [{"mean": 10.0}, {"mean": 20.0}, {"mean": 30.0}]
        agg = aggregate_cycles(cycles)
        assert abs(agg["mean"] - 20.0) < 0.01


class TestFeaturePipeline:
    """Tests for unified feature pipeline."""

    def test_feature_names_count(self):
        """Feature names list should have correct count."""
        names = get_feature_names()
        assert len(names) == N_FEATURES
        assert len(names) > 0

    def test_feature_names_unique(self):
        """All feature names should be unique."""
        names = get_feature_names()
        assert len(names) == len(set(names))

    def test_extract_features_returns_correct_shape(self):
        """Feature vector should match expected length."""
        angle_data = {
            "left_knee_flexion": np.sin(np.linspace(0, 2 * np.pi, 101)) * 30 + 150,
            "right_knee_flexion": np.sin(np.linspace(0, 2 * np.pi, 101)) * 28 + 148,
            "left_hip_flexion": np.sin(np.linspace(0, 2 * np.pi, 101)) * 20 + 160,
            "right_hip_flexion": np.sin(np.linspace(0, 2 * np.pi, 101)) * 22 + 162,
        }
        features = extract_features_from_timeseries(angle_data)
        assert features is not None
        assert features.shape == (N_FEATURES,)

    def test_extract_features_no_nans(self):
        """Output should not contain NaN values."""
        angle_data = {
            "left_knee_flexion": np.random.randn(101) * 10 + 150,
            "right_knee_flexion": np.random.randn(101) * 10 + 150,
        }
        features = extract_features_from_timeseries(angle_data)
        if features is not None:
            assert not np.any(np.isnan(features))
