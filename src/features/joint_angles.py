"""Joint angle computation from 3D landmark coordinates.

Computes biomechanically relevant joint angles from either MediaPipe
landmarks or dataset coordinate systems. All angles are returned in degrees.
"""

import numpy as np


def compute_angle_3points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Compute the angle at point b formed by segments ba and bc.

    Uses the dot product formula:
        angle = arccos((ba . bc) / (|ba| * |bc|))

    Args:
        a: 3D point (x, y, z)
        b: Vertex point (x, y, z)
        c: 3D point (x, y, z)

    Returns:
        Angle in degrees at point b.
    """
    ba = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    bc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return 0.0

    cos_angle = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def compute_knee_flexion(hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> float:
    """Compute knee flexion angle in the sagittal plane.

    Projects points onto the sagittal plane (x-z in MediaPipe coords,
    where y is vertical) and computes the angle at the knee.

    Full extension = ~180 degrees, full flexion = ~0 degrees.
    """
    # Project to sagittal plane (use x and z, ignoring lateral component y)
    # In MediaPipe: x = horizontal, y = vertical (down), z = depth
    hip_sag = np.array([hip[0], hip[2], 0.0])
    knee_sag = np.array([knee[0], knee[2], 0.0])
    ankle_sag = np.array([ankle[0], ankle[2], 0.0])
    return compute_angle_3points(hip_sag, knee_sag, ankle_sag)


def compute_hip_flexion(
    shoulder: np.ndarray, hip: np.ndarray, knee: np.ndarray
) -> float:
    """Compute hip flexion angle in the sagittal plane.

    Angle formed by shoulder-hip-knee projected onto sagittal plane.
    Standing upright = ~180 degrees.
    """
    shoulder_sag = np.array([shoulder[0], shoulder[2], 0.0])
    hip_sag = np.array([hip[0], hip[2], 0.0])
    knee_sag = np.array([knee[0], knee[2], 0.0])
    return compute_angle_3points(shoulder_sag, hip_sag, knee_sag)


def compute_ankle_dorsiflexion(
    knee: np.ndarray, ankle: np.ndarray, foot: np.ndarray
) -> float:
    """Compute ankle dorsiflexion angle in the sagittal plane.

    Angle at ankle formed by knee-ankle-foot.
    Neutral position = ~90 degrees.
    """
    knee_sag = np.array([knee[0], knee[2], 0.0])
    ankle_sag = np.array([ankle[0], ankle[2], 0.0])
    foot_sag = np.array([foot[0], foot[2], 0.0])
    return compute_angle_3points(knee_sag, ankle_sag, foot_sag)


def compute_knee_valgus(
    hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray
) -> float:
    """Compute knee valgus angle in the frontal plane.

    Projects onto the frontal plane (x-y in MediaPipe coords) and
    computes the angle at the knee. Deviations from 180 degrees
    indicate valgus (inward) or varus (outward) alignment.

    Returns:
        Angle in degrees. Values < 180 suggest valgus.
    """
    # Frontal plane projection (x = lateral, y = vertical)
    hip_front = np.array([hip[0], hip[1], 0.0])
    knee_front = np.array([knee[0], knee[1], 0.0])
    ankle_front = np.array([ankle[0], ankle[1], 0.0])
    return compute_angle_3points(hip_front, knee_front, ankle_front)


def compute_trunk_lean(
    shoulder_mid: np.ndarray, hip_mid: np.ndarray
) -> float:
    """Compute trunk lean angle from vertical.

    0 degrees = perfectly upright, positive = forward lean.
    """
    vertical = np.array([0, -1, 0])  # MediaPipe y-axis points down
    trunk = np.asarray(shoulder_mid, dtype=float) - np.asarray(hip_mid, dtype=float)

    norm_trunk = np.linalg.norm(trunk)
    if norm_trunk < 1e-8:
        return 0.0

    cos_angle = np.clip(np.dot(trunk, vertical) / norm_trunk, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def compute_all_angles(landmarks: dict) -> dict:
    """Compute all relevant joint angles from a set of landmarks.

    Args:
        landmarks: Dict mapping landmark names to (x, y, z) arrays.
            Expected keys: left_hip, right_hip, left_knee, right_knee,
            left_ankle, right_ankle, left_shoulder, right_shoulder,
            left_foot_index, right_foot_index

    Returns:
        Dict mapping angle names to values in degrees.
    """
    angles = {}

    for side in ["left", "right"]:
        hip = landmarks.get(f"{side}_hip")
        knee = landmarks.get(f"{side}_knee")
        ankle = landmarks.get(f"{side}_ankle")
        shoulder = landmarks.get(f"{side}_shoulder")
        foot = landmarks.get(f"{side}_foot_index")

        if all(v is not None for v in [hip, knee, ankle]):
            angles[f"{side}_knee_flexion"] = compute_knee_flexion(hip, knee, ankle)
            angles[f"{side}_knee_valgus"] = compute_knee_valgus(hip, knee, ankle)

        if all(v is not None for v in [shoulder, hip, knee]):
            angles[f"{side}_hip_flexion"] = compute_hip_flexion(shoulder, hip, knee)

        if all(v is not None for v in [knee, ankle, foot]):
            angles[f"{side}_ankle_dorsiflexion"] = compute_ankle_dorsiflexion(
                knee, ankle, foot
            )

    # Trunk lean
    ls = landmarks.get("left_shoulder")
    rs = landmarks.get("right_shoulder")
    lh = landmarks.get("left_hip")
    rh = landmarks.get("right_hip")
    if all(v is not None for v in [ls, rs, lh, rh]):
        shoulder_mid = (np.asarray(ls) + np.asarray(rs)) / 2
        hip_mid = (np.asarray(lh) + np.asarray(rh)) / 2
        angles["trunk_lean"] = compute_trunk_lean(shoulder_mid, hip_mid)

    return angles
