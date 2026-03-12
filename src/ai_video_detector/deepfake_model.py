from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .config import settings
from .models import DeepfakeModelReview


class DeepfakeModelClient:
    def __init__(
        self,
        python_bin: str | None = None,
        repo_path: str | None = None,
        config_path: str | None = None,
        model_weights: str | None = None,
        extractor_model: int | None = None,
        extractor_weights: str | None = None,
        detector_type: str | None = None,
        device: str | None = None,
        gpu_id: int | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        self.python_bin = python_bin or settings.mintime_python_bin
        self.repo_path = Path(repo_path or settings.mintime_repo_path)
        self.config_path = Path(config_path or settings.mintime_config_path)
        self.model_weights = model_weights or settings.mintime_model_weights
        self.extractor_model = (
            settings.mintime_extractor_model if extractor_model is None else extractor_model
        )
        self.extractor_weights = extractor_weights or settings.mintime_extractor_weights
        self.detector_type = detector_type or settings.mintime_detector_type
        self.device = device or settings.mintime_device
        self.gpu_id = settings.mintime_gpu_id if gpu_id is None else gpu_id
        self.timeout_seconds = (
            settings.mintime_timeout_seconds if timeout_seconds is None else timeout_seconds
        )

    def available(self) -> bool:
        return (
            self.repo_path.exists()
            and (self.repo_path / "predict.py").exists()
            and self.config_path.exists()
            and bool(self.model_weights)
            and Path(self.model_weights).exists()
        )

    def review_video(self, video_path: str) -> DeepfakeModelReview | None:
        if not self.available():
            return None

        command = [
            self.python_bin,
            "predict.py",
            "--video_path",
            video_path,
            "--model_weights",
            str(self.model_weights),
            "--extractor_model",
            str(self.extractor_model),
            "--extractor_weights",
            self.extractor_weights,
            "--config",
            str(self.config_path),
            "--detector_type",
            self.detector_type,
            "--device",
            self.device,
            "--gpu_id",
            str(self.gpu_id),
            "--output_type",
            "0",
            "--workers",
            "0",
        ]

        completed = subprocess.run(
            command,
            cwd=str(self.repo_path),
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )

        if completed.returncode != 0:
            raise RuntimeError(self._build_error_message(completed.stderr, completed.stdout))

        prediction = self._parse_prediction(completed.stdout)
        if prediction is None:
            raise RuntimeError("MINTIME inference completed but no prediction value was found in stdout.")

        if prediction >= 0.6:
            verdict = "likely_ai_generated"
            confidence = prediction
        elif prediction <= 0.4:
            verdict = "likely_authentic"
            confidence = 1.0 - prediction
        else:
            verdict = "uncertain"
            confidence = 0.5

        return DeepfakeModelReview(
            verdict=verdict,
            confidence=round(confidence, 2),
            average_fake_probability=round(prediction, 4),
            frame_probabilities=[],
            reasoning=(
                "MINTIME video-level inference returned a fake probability of "
                f"{prediction:.2f} using the official `predict.py` pipeline."
            ),
            model_path=str(self.model_weights),
        )

    def _parse_prediction(self, stdout: str) -> float | None:
        matches = re.findall(r"Prediction\s+([0-9]*\.?[0-9]+)", stdout)
        if not matches:
            return None
        return float(matches[-1])

    def _build_error_message(self, stderr: str, stdout: str) -> str:
        detail = stderr.strip() or stdout.strip() or "Unknown MINTIME error."
        return f"MINTIME inference failed: {detail}"
