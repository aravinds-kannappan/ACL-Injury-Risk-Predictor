"""Gait cycle detection and per-cycle feature extraction.

Extracts statistical and biomechanical features from joint angle
time series over normalized gait cycles.
"""

import numpy as np
from scipy.signal import find_peaks


def detect_gait_cycles(ankle_y_positions: np.ndarray, fps: float = 30.0) -> list:
    """Detect gait cycles from ankle vertical position (heel strikes).

    Heel strikes correspond to local minima in ankle y-position
    (in MediaPipe, y increases downward, so heel strikes are local maxima).

    Args:
        ankle_y_positions: Array of ankle y-coordinates over time.
        fps: Video frame rate.

    Returns:
        List of (start_frame, end_frame) tuples for each detected gait cycle.
    """
    if len(ankle_y_positions) < 10:
        return []

    # Smooth the signal
    kernel_size = max(3, int(fps / 10))
    if kernel_size % 2 == 0:
        kernel_size += 1
    smoothed = np.convolve(
        ankle_y_positions, np.ones(kernel_size) / kernel_size, mode="same"
    )

    # Find peaks (heel strikes = local maxima in y since y points down)
    min_distance = int(fps * 0.4)  # Minimum 0.4s between heel strikes
    peaks, _ = find_peaks(smoothed, distance=min_distance, prominence=0.01)

    if len(peaks) < 2:
        return []

    cycles = [(peaks[i], peaks[i + 1]) for i in range(len(peaks) - 1)]
    return cycles


def extract_cycle_features(angle_series: np.ndarray) -> dict:
    """Extract statistical features from a single gait cycle angle series.

    Args:
        angle_series: Joint angle values over one gait cycle
            (ideally 101 points, 0-100% gait cycle).

    Returns:
        Dict of feature names to values.
    """
    ts = np.asarray(angle_series, dtype=float)
    ts = ts[~np.isnan(ts)]

    if len(ts) < 5:
        return {}

    n = len(ts)
    features = {
        "mean": float(np.mean(ts)),
        "std": float(np.std(ts)),
        "max": float(np.max(ts)),
        "min": float(np.min(ts)),
        "range": float(np.ptp(ts)),
        # Gait phase values
        "initial_contact": float(ts[0]),
        "midstance": float(ts[int(n * 0.3)]),
        "toe_off": float(ts[int(n * 0.6)]),
        "peak_swing": float(np.max(ts[int(n * 0.6):])),
        # Angular velocity features
        "angular_velocity_max": float(np.max(np.abs(np.gradient(ts)))),
        "angular_velocity_mean": float(np.mean(np.abs(np.gradient(ts)))),
    }

    return features


def compute_asymmetry(left_features: dict, right_features: dict) -> dict:
    """Compute left-right asymmetry ratios for matching features.

    Asymmetry = |left - right| / max(|left|, |right|, epsilon)
    Values near 0 = symmetric, near 1 = highly asymmetric.
    """
    asymmetry = {}
    epsilon = 1e-6

    for key in left_features:
        if key in right_features:
            l_val = left_features[key]
            r_val = right_features[key]
            denom = max(abs(l_val), abs(r_val), epsilon)
            asymmetry[f"{key}_asymmetry"] = abs(l_val - r_val) / denom

    return asymmetry


def aggregate_cycles(cycle_features_list: list) -> dict:
    """Average features across multiple gait cycles for robustness.

    Args:
        cycle_features_list: List of feature dicts from extract_cycle_features.

    Returns:
        Dict with averaged feature values.
    """
    if not cycle_features_list:
        return {}

    all_keys = cycle_features_list[0].keys()
    aggregated = {}

    for key in all_keys:
        values = [cf[key] for cf in cycle_features_list if key in cf]
        if values:
            aggregated[key] = float(np.mean(values))

    return aggregated
