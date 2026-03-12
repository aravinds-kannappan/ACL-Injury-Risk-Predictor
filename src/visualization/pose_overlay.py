"""Pose skeleton overlay and angle annotation on video frames."""

import cv2
import numpy as np
from pathlib import Path

# Skeleton connections for lower body + trunk
CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"),
    ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"),
    ("right_ankle", "right_heel"),
    ("left_ankle", "left_foot_index"),
    ("right_ankle", "right_foot_index"),
]


def draw_pose_on_frame(
    frame: np.ndarray,
    landmarks: dict,
    risk_score: float = None,
    angles: dict = None,
) -> np.ndarray:
    """Draw skeleton overlay on a frame with optional risk coloring.

    Args:
        frame: BGR image.
        landmarks: Dict with '_image_landmarks' containing pixel coords.
        risk_score: Optional 0-1 score for color coding.
        angles: Optional dict of angle values to annotate.

    Returns:
        Annotated frame (copy).
    """
    annotated = frame.copy()
    img_landmarks = landmarks.get("_image_landmarks", {})

    if not img_landmarks:
        return annotated

    # Determine color based on risk
    if risk_score is not None:
        if risk_score < 0.3:
            color = (0, 200, 0)  # Green
        elif risk_score < 0.7:
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 0, 255)  # Red
    else:
        color = (0, 255, 0)  # Default green

    # Draw connections
    for start_name, end_name in CONNECTIONS:
        if start_name in img_landmarks and end_name in img_landmarks:
            pt1 = tuple(img_landmarks[start_name].astype(int))
            pt2 = tuple(img_landmarks[end_name].astype(int))
            cv2.line(annotated, pt1, pt2, color, 2, cv2.LINE_AA)

    # Draw landmarks
    for name, coords in img_landmarks.items():
        pt = tuple(coords.astype(int))
        cv2.circle(annotated, pt, 5, color, -1, cv2.LINE_AA)
        cv2.circle(annotated, pt, 6, (255, 255, 255), 1, cv2.LINE_AA)

    # Annotate angles
    if angles:
        angle_positions = {
            "left_knee_flexion": "left_knee",
            "right_knee_flexion": "right_knee",
            "left_hip_flexion": "left_hip",
            "right_hip_flexion": "right_hip",
            "left_knee_valgus": "left_knee",
        }

        for angle_name, landmark_name in angle_positions.items():
            if angle_name in angles and landmark_name in img_landmarks:
                pt = img_landmarks[landmark_name].astype(int)
                text = f"{angles[angle_name]:.0f}"
                cv2.putText(
                    annotated, text,
                    (pt[0] + 10, pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2, cv2.LINE_AA,
                )
                cv2.putText(
                    annotated, text,
                    (pt[0] + 10, pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA,
                )

    # Risk score overlay
    if risk_score is not None:
        text = f"Risk: {risk_score:.1%}"
        cv2.putText(
            annotated, text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            color, 2, cv2.LINE_AA,
        )

    return annotated


def create_annotated_video(
    video_path: str,
    landmarks_sequence: list,
    angles_sequence: list = None,
    risk_score: float = None,
    output_path: str = None,
) -> str:
    """Create output video with pose overlay.

    Args:
        video_path: Input video path.
        landmarks_sequence: List of per-frame landmark dicts.
        angles_sequence: Optional list of per-frame angle dicts.
        risk_score: Overall risk score for color coding.
        output_path: Output video path. Defaults to input_annotated.mp4.

    Returns:
        Path to output video.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if output_path is None:
        p = Path(video_path)
        output_path = str(p.parent / f"{p.stem}_annotated.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(landmarks_sequence) and landmarks_sequence[frame_idx] is not None:
            angles = angles_sequence[frame_idx] if angles_sequence and frame_idx < len(angles_sequence) else None
            frame = draw_pose_on_frame(frame, landmarks_sequence[frame_idx], risk_score, angles)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return output_path
