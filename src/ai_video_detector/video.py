from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import settings
from .models import VideoMetadata


@dataclass
class SampledVideo:
    metadata: VideoMetadata
    frames_bgr: list[np.ndarray]


def sample_video_frames(video_path: str, sample_count: int | None = None) -> SampledVideo:
    sample_count = sample_count or settings.frame_sample_count
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_seconds = frame_count / fps if fps > 0 else 0.0

    if duration_seconds > settings.max_video_duration_seconds:
        capture.release()
        raise ValueError(
            f"Video duration {duration_seconds:.2f}s exceeds limit of "
            f"{settings.max_video_duration_seconds}s"
        )

    if frame_count <= 0:
        capture.release()
        raise ValueError("Video has no readable frames")

    sampled_indices = sorted(
        {min(int(i), frame_count - 1) for i in np.linspace(0, frame_count - 1, num=sample_count)}
    )

    frames_bgr: list[np.ndarray] = []
    for index in sampled_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if ok and frame is not None:
            frames_bgr.append(frame)

    capture.release()

    if not frames_bgr:
        raise ValueError("No frames could be sampled from the video")

    metadata = VideoMetadata(
        path=str(path.resolve()),
        fps=fps,
        frame_count=frame_count,
        duration_seconds=duration_seconds,
        width=width,
        height=height,
        sampled_indices=sampled_indices[: len(frames_bgr)],
    )
    return SampledVideo(metadata=metadata, frames_bgr=frames_bgr)
