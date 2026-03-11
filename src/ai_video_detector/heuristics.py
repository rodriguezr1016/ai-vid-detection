from __future__ import annotations

import cv2
import numpy as np

from .models import HeuristicSignals


def _normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def compute_heuristics(frames_bgr: list[np.ndarray]) -> HeuristicSignals:
    if not frames_bgr:
        raise ValueError("At least one frame is required")

    grayscale_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames_bgr]
    duplicate_distances: list[float] = []
    edge_densities: list[float] = []
    color_variances: list[float] = []
    motion_instabilities: list[float] = []

    for idx, frame in enumerate(frames_bgr):
        gray = grayscale_frames[idx]
        edges = cv2.Canny(gray, 100, 200)
        edge_densities.append(float(np.count_nonzero(edges)) / float(edges.size))
        color_variances.append(float(np.var(frame)))

        if idx > 0:
            prev_gray = grayscale_frames[idx - 1]
            mean_delta = float(np.mean(cv2.absdiff(prev_gray, gray)))
            duplicate_distances.append(mean_delta)
            motion_instabilities.append(abs(mean_delta - float(np.mean(duplicate_distances))))

    if duplicate_distances:
        duplicate_frame_ratio = sum(distance < 2.0 for distance in duplicate_distances) / len(duplicate_distances)
    else:
        duplicate_frame_ratio = 0.0

    edge_density_mean = float(np.mean(edge_densities))
    edge_density_std = float(np.std(edge_densities))
    color_variance_mean = float(np.mean(color_variances))
    motion_instability_mean = float(np.mean(motion_instabilities)) if motion_instabilities else 0.0

    artifact_score = np.mean(
        [
            _normalize(duplicate_frame_ratio, 0.05, 0.35),
            _normalize(edge_density_std, 0.02, 0.10),
            1.0 - _normalize(color_variance_mean, 1500.0, 8000.0),
            _normalize(motion_instability_mean, 1.0, 12.0),
        ]
    )

    return HeuristicSignals(
        duplicate_frame_ratio=float(duplicate_frame_ratio),
        edge_density_mean=edge_density_mean,
        edge_density_std=edge_density_std,
        color_variance_mean=color_variance_mean,
        motion_instability_mean=motion_instability_mean,
        synthetic_artifact_score=float(max(0.0, min(1.0, artifact_score))),
    )
