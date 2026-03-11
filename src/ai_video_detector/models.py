from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class VideoMetadata(BaseModel):
    path: str
    fps: float
    frame_count: int
    duration_seconds: float
    width: int
    height: int
    sampled_indices: list[int]


class HeuristicSignals(BaseModel):
    duplicate_frame_ratio: float = Field(ge=0.0, le=1.0)
    edge_density_mean: float = Field(ge=0.0)
    edge_density_std: float = Field(ge=0.0)
    color_variance_mean: float = Field(ge=0.0)
    motion_instability_mean: float = Field(ge=0.0)
    synthetic_artifact_score: float = Field(ge=0.0, le=1.0)


class LLMReview(BaseModel):
    verdict: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    suspicious_cues: list[str] = Field(default_factory=list)
    raw_response: dict[str, Any] | None = None


class DeepfakeModelReview(BaseModel):
    verdict: str
    confidence: float = Field(ge=0.0, le=1.0)
    average_fake_probability: float = Field(ge=0.0, le=1.0)
    frame_probabilities: list[float] = Field(default_factory=list)
    reasoning: str
    model_path: str | None = None


class DetectionResult(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    metadata: VideoMetadata
    heuristics: HeuristicSignals
    preview_frames: list[str] = Field(default_factory=list)
    deepfake_model_review: DeepfakeModelReview | None = None
    llm_review: LLMReview | None = None
