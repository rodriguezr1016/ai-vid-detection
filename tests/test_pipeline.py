from pathlib import Path

from ai_video_detector.models import DeepfakeModelReview, HeuristicSignals, LLMReview
from ai_video_detector.pipeline import VideoDetectionPipeline


def test_fuse_decision_prefers_deepfake_model_and_llm_consensus() -> None:
    pipeline = VideoDetectionPipeline()
    heuristics = HeuristicSignals(
        duplicate_frame_ratio=0.1,
        edge_density_mean=0.2,
        edge_density_std=0.03,
        color_variance_mean=2500.0,
        motion_instability_mean=1.0,
        synthetic_artifact_score=0.25,
    )
    deepfake_review = DeepfakeModelReview(
        verdict="likely_ai_generated",
        confidence=0.84,
        average_fake_probability=0.84,
        frame_probabilities=[0.8, 0.85, 0.87],
        reasoning="Dedicated model flagged frame artifacts.",
        model_path="/tmp/model.onnx",
    )
    llm_review = LLMReview(
        verdict="likely_ai_generated",
        confidence=0.71,
        reasoning="The LLM saw mouth-region texture instability.",
        suspicious_cues=["texture shimmer"],
    )

    label, confidence, reasoning = pipeline._fuse_decision(heuristics, deepfake_review, llm_review)

    assert label == "likely_ai_generated"
    assert confidence >= 0.68
    assert "deepfake_model" in reasoning


def test_fuse_decision_without_model_or_llm_falls_back_to_heuristics() -> None:
    pipeline = VideoDetectionPipeline()
    heuristics = HeuristicSignals(
        duplicate_frame_ratio=0.3,
        edge_density_mean=0.12,
        edge_density_std=0.07,
        color_variance_mean=1800.0,
        motion_instability_mean=8.0,
        synthetic_artifact_score=0.82,
    )

    label, confidence, reasoning = pipeline._fuse_decision(heuristics, None, None)

    assert label == "likely_ai_generated"
    assert confidence == 0.82
    assert "Only heuristic evidence was available" in reasoning


def test_mintime_availability_requires_repo_and_weights(tmp_path: Path) -> None:
    repo_path = tmp_path / "mintime"
    repo_path.mkdir()
    (repo_path / "predict.py").write_text("", encoding="utf-8")
    config_path = repo_path / "config.yaml"
    config_path.write_text("", encoding="utf-8")
    weights_path = tmp_path / "weights.pth"
    weights_path.write_text("", encoding="utf-8")

    client = VideoDetectionPipeline().deepfake_model_client.__class__(
        repo_path=str(repo_path),
        config_path=str(config_path),
        model_weights=str(weights_path),
    )

    assert client.available() is True
