import numpy as np

from ai_video_detector.heuristics import compute_heuristics


def test_compute_heuristics_returns_bounded_score() -> None:
    frames = [
        np.zeros((32, 32, 3), dtype=np.uint8),
        np.full((32, 32, 3), 10, dtype=np.uint8),
        np.full((32, 32, 3), 20, dtype=np.uint8),
        np.full((32, 32, 3), 30, dtype=np.uint8),
    ]

    result = compute_heuristics(frames)

    assert 0.0 <= result.synthetic_artifact_score <= 1.0
    assert 0.0 <= result.duplicate_frame_ratio <= 1.0
