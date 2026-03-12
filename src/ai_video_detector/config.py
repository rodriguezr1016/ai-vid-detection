from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()
project_root = Path(__file__).resolve().parents[2]


class Settings(BaseModel):
    llm_api_key: str | None = Field(default=os.getenv("LLM_API_KEY"))
    llm_base_url: str = Field(default=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"))
    llm_model: str = Field(default=os.getenv("LLM_MODEL", "gpt-4o-mini"))
    llm_timeout_seconds: int = Field(default=int(os.getenv("LLM_TIMEOUT_SECONDS", "45")))
    frame_sample_count: int = Field(default=int(os.getenv("FRAME_SAMPLE_COUNT", "8")))
    max_video_duration_seconds: int = Field(default=int(os.getenv("MAX_VIDEO_DURATION_SECONDS", "120")))
    mintime_python_bin: str = Field(default=os.getenv("MINTIME_PYTHON_BIN", "python"))
    mintime_repo_path: str = Field(
        default=os.getenv(
            "MINTIME_REPO_PATH",
            str(project_root / "vendor" / "mintime"),
        )
    )
    mintime_config_path: str = Field(
        default=os.getenv(
            "MINTIME_CONFIG_PATH",
            str(project_root / "vendor" / "mintime" / "config" / "size_invariant_timesformer.yaml"),
        )
    )
    mintime_model_weights: str | None = Field(default=os.getenv("MINTIME_MODEL_WEIGHTS"))
    mintime_extractor_model: int = Field(default=int(os.getenv("MINTIME_EXTRACTOR_MODEL", "0")))
    mintime_extractor_weights: str = Field(default=os.getenv("MINTIME_EXTRACTOR_WEIGHTS", "ImageNet"))
    mintime_detector_type: str = Field(default=os.getenv("MINTIME_DETECTOR_TYPE", "FacenetDetector"))
    mintime_device: str = Field(default=os.getenv("MINTIME_DEVICE", "cpu"))
    mintime_gpu_id: int = Field(default=int(os.getenv("MINTIME_GPU_ID", "0")))
    mintime_timeout_seconds: int = Field(default=int(os.getenv("MINTIME_TIMEOUT_SECONDS", "300")))


settings = Settings()
