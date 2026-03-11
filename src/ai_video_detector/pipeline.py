from __future__ import annotations

from .deepfake_model import DeepfakeModelClient
from .heuristics import compute_heuristics
from .llm import LLMClient
from .models import DeepfakeModelReview, DetectionResult, HeuristicSignals, LLMReview
from .presentation import frame_to_data_url
from .video import sample_video_frames


class VideoDetectionPipeline:
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        deepfake_model_client: DeepfakeModelClient | None = None,
    ) -> None:
        self.llm_client = llm_client or LLMClient()
        self.deepfake_model_client = deepfake_model_client or DeepfakeModelClient()

    def analyze(self, video_path: str) -> DetectionResult:
        sampled = sample_video_frames(video_path)
        heuristics = compute_heuristics(sampled.frames_bgr)
        deepfake_model_review = self.deepfake_model_client.review_video(sampled.metadata.path)
        llm_review = self.llm_client.review_video(
            metadata=sampled.metadata,
            heuristics=heuristics,
            frames_bgr=sampled.frames_bgr,
        )
        label, confidence, reasoning = self._fuse_decision(
            heuristics,
            deepfake_model_review,
            llm_review,
        )
        return DetectionResult(
            label=label,
            confidence=confidence,
            reasoning=reasoning,
            metadata=sampled.metadata,
            heuristics=heuristics,
            preview_frames=[frame_to_data_url(frame) for frame in sampled.frames_bgr[:4]],
            deepfake_model_review=deepfake_model_review,
            llm_review=llm_review,
        )

    def _fuse_decision(
        self,
        heuristics: HeuristicSignals,
        deepfake_model_review: DeepfakeModelReview | None,
        llm_review: LLMReview | None,
    ) -> tuple[str, float, str]:
        heuristic_score = heuristics.synthetic_artifact_score
        weighted_scores: list[tuple[str, float, float]] = [("heuristics", heuristic_score, 0.2)]

        if deepfake_model_review is not None:
            weighted_scores.append(
                (
                    "deepfake_model",
                    self._review_to_ai_score(deepfake_model_review.verdict, deepfake_model_review.confidence),
                    0.45,
                )
            )
        if llm_review is not None:
            weighted_scores.append(
                (
                    "llm",
                    self._review_to_ai_score(llm_review.verdict, llm_review.confidence),
                    0.35 if deepfake_model_review is not None else 0.8,
                )
            )

        if len(weighted_scores) == 1:
            combined_score = heuristic_score
        else:
            total_weight = sum(weight for _, _, weight in weighted_scores)
            combined_score = sum(score * weight for _, score, weight in weighted_scores) / total_weight

        if combined_score >= 0.6:
            label = "likely_ai_generated"
        elif combined_score <= 0.4:
            label = "likely_authentic"
        else:
            label = "uncertain"

        evidence = [f"heuristics={heuristic_score:.2f}"]
        if deepfake_model_review is not None:
            evidence.append(
                "deepfake_model="
                f"{deepfake_model_review.verdict}:{deepfake_model_review.average_fake_probability:.2f}"
            )
        else:
            evidence.append("deepfake_model=unavailable")
        if llm_review is not None:
            evidence.append(f"llm={llm_review.verdict}:{llm_review.confidence:.2f}")
        else:
            evidence.append("llm=unavailable")

        reasoning_parts = [f"Fused evidence ({', '.join(evidence)})."]
        if deepfake_model_review is not None:
            reasoning_parts.append(deepfake_model_review.reasoning)
        if llm_review is not None:
            reasoning_parts.append(llm_review.reasoning)
        if deepfake_model_review is None and llm_review is None:
            reasoning_parts.append("Only heuristic evidence was available for this run.")

        reasoning = " ".join(reasoning_parts)
        confidence = combined_score if label != "likely_authentic" else 1.0 - combined_score
        return label, round(confidence, 2), reasoning

    def _review_to_ai_score(self, verdict: str, confidence: float) -> float:
        if verdict == "likely_ai_generated":
            return confidence
        if verdict == "likely_authentic":
            return 1.0 - confidence
        return 0.5
