"""MediaPipe pose estimation for extracting body keypoints from video frames."""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# MediaPipe landmark indices for lower body + trunk
LANDMARK_NAMES = {
    11: "left_shoulder",
    12: "right_shoulder",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}


class PoseEstimator:
    """MediaPipe Pose wrapper for biomechanical analysis."""

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        import mediapipe as mp

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def process_frame(self, frame: np.ndarray) -> dict:
        """Run pose detection on a single BGR frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            Dict mapping landmark names to (x, y, z) numpy arrays,
            or None if no pose detected. Also includes 'visibility'
            key with mean visibility score.
        """
        import cv2

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if not results.pose_world_landmarks:
            return None

        landmarks = {}
        visibility_scores = []

        for idx, name in LANDMARK_NAMES.items():
            lm = results.pose_world_landmarks.landmark[idx]
            landmarks[name] = np.array([lm.x, lm.y, lm.z])
            visibility_scores.append(lm.visibility)

        landmarks["visibility"] = float(np.mean(visibility_scores))

        # Also store the image-space landmarks for visualization
        if results.pose_landmarks:
            h, w = frame.shape[:2]
            landmarks["_image_landmarks"] = {}
            for idx, name in LANDMARK_NAMES.items():
                lm = results.pose_landmarks.landmark[idx]
                landmarks["_image_landmarks"][name] = np.array([
                    int(lm.x * w), int(lm.y * h)
                ])

        return landmarks

    def get_drawing_data(self, frame: np.ndarray):
        """Get MediaPipe results for drawing utilities."""
        import cv2
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.pose.process(rgb_frame)

    def close(self):
        """Release MediaPipe resources."""
        self.pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
