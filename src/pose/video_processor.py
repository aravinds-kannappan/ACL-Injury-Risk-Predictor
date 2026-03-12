"""Video I/O and frame-by-frame pose processing."""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from .mediapipe_estimator import PoseEstimator
from src.features.feature_pipeline import extract_features_from_landmarks

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Handle video I/O and orchestrate frame-by-frame pose processing."""

    def __init__(self, video_path: str):
        self.video_path = str(video_path)
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0

    def get_metadata(self) -> dict:
        return {
            "fps": self.fps,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "duration": self.duration,
        }

    def process_video(
        self, estimator: PoseEstimator, progress: bool = True, max_frames: int = None
    ) -> list:
        """Process all frames through pose estimation.

        Args:
            estimator: PoseEstimator instance.
            progress: Show progress bar.
            max_frames: Maximum frames to process (None = all).

        Returns:
            List of per-frame landmark dicts (or None for frames with no detection).
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        landmarks_sequence = []
        total = max_frames or self.frame_count

        iterator = range(total)
        if progress:
            iterator = tqdm(iterator, desc="Processing video", unit="frame")

        for _ in iterator:
            ret, frame = self.cap.read()
            if not ret:
                break

            landmarks = estimator.process_frame(frame)
            landmarks_sequence.append(landmarks)

        detection_rate = sum(1 for l in landmarks_sequence if l is not None) / max(len(landmarks_sequence), 1)
        logger.info(
            f"Processed {len(landmarks_sequence)} frames, "
            f"detection rate: {detection_rate:.1%}"
        )

        return landmarks_sequence

    def extract_frames(self, interval: int = 1):
        """Generator yielding frames at specified interval."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                yield frame_idx, frame
            frame_idx += 1

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()


def process_video_to_features(video_path: str) -> Optional[tuple]:
    """High-level function: video → features.

    Args:
        video_path: Path to input video.

    Returns:
        Tuple of (feature_vector, landmarks_sequence, angles_sequence)
        or None if processing fails.
    """
    from src.features.joint_angles import compute_all_angles

    try:
        with VideoProcessor(video_path) as vp:
            metadata = vp.get_metadata()
            logger.info(
                f"Video: {metadata['width']}x{metadata['height']}, "
                f"{metadata['fps']:.1f} fps, {metadata['duration']:.1f}s"
            )

            with PoseEstimator() as estimator:
                landmarks_seq = vp.process_video(estimator)

        # Compute angles per frame
        angles_seq = []
        for landmarks in landmarks_seq:
            if landmarks is not None:
                # Remove non-coordinate entries before computing angles
                coord_landmarks = {
                    k: v for k, v in landmarks.items()
                    if isinstance(v, np.ndarray) and not k.startswith("_")
                }
                angles = compute_all_angles(coord_landmarks)
                angles_seq.append(angles)
            else:
                angles_seq.append(None)

        # Extract features from landmark sequence
        clean_landmarks = [
            {k: v for k, v in l.items() if isinstance(v, np.ndarray) and not k.startswith("_")}
            if l is not None else None
            for l in landmarks_seq
        ]

        features = extract_features_from_landmarks(clean_landmarks, metadata["fps"])

        if features is None:
            logger.error("Feature extraction returned None.")
            return None

        return features, landmarks_seq, angles_seq

    except Exception as e:
        logger.error(f"Video processing failed: {e}")
        return None
