"""Unified feature engineering pipeline.

Produces identical feature vectors from either:
1. COMPWALK-ACL / UCI dataset angle time series
2. MediaPipe-extracted landmark sequences

This bridge ensures models trained on clinical data can generalize
to video-based pose estimation input.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .gait_features import (
    extract_cycle_features,
    compute_asymmetry,
    aggregate_cycles,
    detect_gait_cycles,
)
from .joint_angles import compute_all_angles

logger = logging.getLogger(__name__)

# Canonical feature order for model input
JOINTS_FOR_FEATURES = [
    "knee_flexion",
    "hip_flexion",
    "ankle_dorsiflexion",
    "knee_valgus",
]

STAT_NAMES = ["mean", "std", "max", "min", "range"]
PHASE_NAMES = ["initial_contact", "midstance", "toe_off", "peak_swing"]
VELOCITY_NAMES = ["angular_velocity_max", "angular_velocity_mean"]
ALL_CYCLE_FEATURES = STAT_NAMES + PHASE_NAMES + VELOCITY_NAMES

SIDES = ["left", "right"]


def get_feature_names() -> list:
    """Return ordered list of all feature names produced by the pipeline."""
    names = []

    # Per-side, per-joint features
    for side in SIDES:
        for joint in JOINTS_FOR_FEATURES:
            for feat in ALL_CYCLE_FEATURES:
                names.append(f"{side}_{joint}_{feat}")

    # Asymmetry features (one per joint per cycle feature)
    for joint in JOINTS_FOR_FEATURES:
        for feat in ALL_CYCLE_FEATURES:
            names.append(f"{joint}_{feat}_asymmetry")

    # Trunk lean features
    for feat in STAT_NAMES:
        names.append(f"trunk_lean_{feat}")

    return names


FEATURE_NAMES = get_feature_names()
N_FEATURES = len(FEATURE_NAMES)


def extract_features_from_timeseries(
    angle_data: dict, participant_id: str = ""
) -> Optional[np.ndarray]:
    """Extract feature vector from dataset angle time series.

    Args:
        angle_data: Dict mapping (joint, side) tuples or
            "{side}_{joint}" strings to numpy arrays of angle values
            over a gait cycle.
        participant_id: For logging.

    Returns:
        Feature vector of shape (N_FEATURES,) or None if insufficient data.
    """
    features = {}

    for side in SIDES:
        for joint in JOINTS_FOR_FEATURES:
            key_options = [
                (joint, side),
                f"{side}_{joint}",
                joint,  # If side is already handled in the data
            ]

            ts = None
            for key in key_options:
                if key in angle_data:
                    ts = angle_data[key]
                    break

            if ts is not None:
                cycle_feats = extract_cycle_features(ts)
                for feat_name, value in cycle_feats.items():
                    features[f"{side}_{joint}_{feat_name}"] = value

    # Compute asymmetry
    for joint in JOINTS_FOR_FEATURES:
        left_feats = {
            k.replace(f"left_{joint}_", ""): v
            for k, v in features.items()
            if k.startswith(f"left_{joint}_")
        }
        right_feats = {
            k.replace(f"right_{joint}_", ""): v
            for k, v in features.items()
            if k.startswith(f"right_{joint}_")
        }
        asymm = compute_asymmetry(left_feats, right_feats)
        for feat_name, value in asymm.items():
            features[f"{joint}_{feat_name}"] = value

    # Build feature vector in canonical order
    vector = []
    for name in FEATURE_NAMES:
        vector.append(features.get(name, np.nan))

    vector = np.array(vector, dtype=float)

    # Check completeness
    nan_ratio = np.isnan(vector).sum() / len(vector)
    if nan_ratio > 0.8:
        logger.warning(
            f"Participant {participant_id}: {nan_ratio:.0%} features missing"
        )
        return None

    # Fill remaining NaNs with 0 (for features not computable from this data)
    vector = np.nan_to_num(vector, nan=0.0)
    return vector


def extract_features_from_landmarks(
    landmarks_sequence: list, fps: float = 30.0
) -> Optional[np.ndarray]:
    """Extract feature vector from a sequence of MediaPipe landmark frames.

    Args:
        landmarks_sequence: List of dicts, each mapping landmark names
            to (x, y, z) arrays. One dict per video frame.
        fps: Video frame rate for gait cycle detection.

    Returns:
        Feature vector of shape (N_FEATURES,) or None if insufficient data.
    """
    if not landmarks_sequence or len(landmarks_sequence) < 10:
        return None

    # Compute angles for every frame
    frame_angles = []
    for frame_landmarks in landmarks_sequence:
        if frame_landmarks is not None:
            angles = compute_all_angles(frame_landmarks)
            frame_angles.append(angles)
        else:
            frame_angles.append({})

    if not frame_angles:
        return None

    # Collect angle time series per joint-side
    angle_keys = set()
    for fa in frame_angles:
        angle_keys.update(fa.keys())

    angle_timeseries = {}
    for key in angle_keys:
        ts = [fa.get(key, np.nan) for fa in frame_angles]
        angle_timeseries[key] = np.array(ts)

    # Try to detect gait cycles from ankle position
    left_ankle_y = []
    for fl in landmarks_sequence:
        if fl and "left_ankle" in fl:
            left_ankle_y.append(fl["left_ankle"][1])
        else:
            left_ankle_y.append(np.nan)
    left_ankle_y = np.array(left_ankle_y)

    cycles = detect_gait_cycles(left_ankle_y[~np.isnan(left_ankle_y)], fps)

    if cycles:
        # Extract features per cycle, then average
        cycle_angle_data = []
        for start, end in cycles:
            cycle_data = {}
            for key, ts in angle_timeseries.items():
                valid_ts = ts[~np.isnan(ts)]
                if start < len(valid_ts) and end <= len(valid_ts):
                    cycle_segment = valid_ts[start:end]
                    # Normalize to 101 points
                    x_old = np.linspace(0, 100, len(cycle_segment))
                    x_new = np.linspace(0, 100, 101)
                    cycle_data[key] = np.interp(x_new, x_old, cycle_segment)
            cycle_angle_data.append(cycle_data)

        # Average across cycles
        all_keys = set()
        for cd in cycle_angle_data:
            all_keys.update(cd.keys())

        averaged_data = {}
        for key in all_keys:
            arrays = [cd[key] for cd in cycle_angle_data if key in cd]
            if arrays:
                averaged_data[key] = np.mean(arrays, axis=0)

        return extract_features_from_timeseries(averaged_data)
    else:
        # No gait cycles detected: use the full sequence as one "cycle"
        normalized_data = {}
        for key, ts in angle_timeseries.items():
            valid_ts = ts[~np.isnan(ts)]
            if len(valid_ts) >= 10:
                x_old = np.linspace(0, 100, len(valid_ts))
                x_new = np.linspace(0, 100, 101)
                normalized_data[key] = np.interp(x_new, x_old, valid_ts)

        return extract_features_from_timeseries(normalized_data)


def build_feature_matrix(dataset_df: pd.DataFrame) -> tuple:
    """Build feature matrix from dataset DataFrame.

    Expects DataFrame with columns: participant_id, side, joint,
    angle_timeseries, label.

    Groups by participant and speed, pivots joints into angle_data dict,
    then extracts features.

    Returns:
        (X, y, participant_ids) where X is (n_samples, n_features),
        y is (n_samples,), participant_ids is list of strings.
    """
    X_list = []
    y_list = []
    pid_list = []

    group_cols = ["participant_id"]
    if "speed" in dataset_df.columns:
        group_cols.append("speed")

    for group_key, group in dataset_df.groupby(group_cols):
        if isinstance(group_key, tuple):
            pid = group_key[0]
        else:
            pid = group_key

        label = group["label"].iloc[0]

        # Build angle_data dict
        angle_data = {}
        for _, row in group.iterrows():
            joint = row["joint"]
            side = row.get("side", "right")
            key = f"{side}_{joint}"
            angle_data[key] = row["angle_timeseries"]

            # Also store without side prefix for joints that may not be bilateral
            if joint not in angle_data:
                angle_data[joint] = row["angle_timeseries"]

        features = extract_features_from_timeseries(angle_data, participant_id=pid)

        if features is not None:
            X_list.append(features)
            y_list.append(label)
            pid_list.append(pid)

    if not X_list:
        logger.error("No valid feature vectors extracted.")
        return np.array([]), np.array([]), []

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=int)

    logger.info(
        f"Feature matrix: {X.shape[0]} samples, {X.shape[1]} features. "
        f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}"
    )

    return X, y, pid_list
