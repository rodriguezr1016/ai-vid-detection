from __future__ import annotations

import json

import requests

from .config import settings
from .models import HeuristicSignals, LLMReview, VideoMetadata
from .presentation import frame_to_data_url


class LLMClient:
    def __init__(self) -> None:
        self.api_key = settings.llm_api_key
        self.base_url = settings.llm_base_url.rstrip("/")
        self.model = settings.llm_model
        self.timeout = settings.llm_timeout_seconds

    def available(self) -> bool:
        return bool(self.api_key)

    def review_video(
        self,
        metadata: VideoMetadata,
        heuristics: HeuristicSignals,
        frames_bgr: list,
    ) -> LLMReview | None:
        if not self.available():
            return None

        images = [frame_to_data_url(frame, max_size=(768, 768)) for frame in frames_bgr[:4]]
        prompt = self._build_prompt(metadata, heuristics)
        payload = {
            "model": self.model,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a forensic media analyst. "
                        "Estimate whether a video is AI-generated based on sampled frames "
                        "and numeric forensic signals. Return strict JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                    + [
                        {"type": "image_url", "image_url": {"url": data_url}}
                        for data_url in images
                    ],
                },
            ],
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        content = body["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        return LLMReview(
            verdict=parsed.get("verdict", "uncertain"),
            confidence=float(parsed.get("confidence", 0.5)),
            reasoning=parsed.get("reasoning", "No reasoning returned."),
            suspicious_cues=list(parsed.get("suspicious_cues", [])),
            raw_response=body,
        )

    def _build_prompt(self, metadata: VideoMetadata, heuristics: HeuristicSignals) -> str:
        return (
            "Review the sampled video frames and forensic signals. "
            "Decide whether the video is likely AI-generated or likely authentic.\n\n"
            f"Video metadata:\n{metadata.model_dump_json(indent=2)}\n\n"
            f"Heuristics:\n{heuristics.model_dump_json(indent=2)}\n\n"
            "Return JSON with keys: verdict, confidence, reasoning, suspicious_cues. "
            "Use verdict values: likely_ai_generated, likely_authentic, or uncertain."
        )
