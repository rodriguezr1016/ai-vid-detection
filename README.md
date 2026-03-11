# AI Video Detector

This project is a practical MVP for detecting whether a video is likely AI-generated.

It does not rely on a single model claim. Instead, it combines:

- frame sampling from the input video
- simple forensic heuristics across frames
- MINTIME video-level deepfake inference
- an LLM or multimodal model that reviews selected frames plus the heuristic summary
- a final fused confidence score and explanation

## Why this approach

A pure LLM verdict on a video is weak. A stronger MVP is:

1. extract representative frames from the video
2. compute lightweight signals that often correlate with synthetic video artifacts
3. run MINTIME against the full video for temporal deepfake detection
4. ask a multimodal LLM to inspect those frames and explain suspicious patterns
5. combine all sources into one score

This gives you a system that is easier to improve over time with better detectors.

## Features

- CLI for local video analysis
- FastAPI service for programmatic use
- Browser UI for drag-and-upload style analysis
- OpenAI-compatible multimodal API support
- Official MINTIME integration through its published `predict.py` pipeline
- Heuristic metrics for temporal duplication, edge density, color variance, and motion instability
- JSON output with an explanation and evidence

## Project structure

```text
src/ai_video_detector/
  api.py
  cli.py
  config.py
  deepfake_model.py
  heuristics.py
  llm.py
  models.py
  presentation.py
  pipeline.py
  video.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

For Linux + AMD ROCm deployment, see [linux-rocm.md](/Users/renerodriguez/Desktop/projects/ai-video-detector/docs/linux-rocm.md).

## Environment variables

```bash
LLM_API_KEY=your_key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
LLM_TIMEOUT_SECONDS=45
FRAME_SAMPLE_COUNT=8
MAX_VIDEO_DURATION_SECONDS=120
MINTIME_PYTHON_BIN=/absolute/path/to/mintime-env/bin/python
MINTIME_REPO_PATH=/absolute/path/to/ai-video-detector/vendor/mintime
MINTIME_CONFIG_PATH=/absolute/path/to/ai-video-detector/vendor/mintime/config/size_invariant_timesformer.yaml
MINTIME_MODEL_WEIGHTS=/absolute/path/to/mintime_model_weights
MINTIME_EXTRACTOR_MODEL=0
MINTIME_EXTRACTOR_WEIGHTS=ImageNet
MINTIME_DETECTOR_TYPE=FacenetDetector
MINTIME_GPU_ID=0
MINTIME_TIMEOUT_SECONDS=300
```

The LLM endpoint must be OpenAI-compatible and support image inputs if you want the vision step.

## MINTIME setup

This project now shells out to the official MINTIME repository at [vendor/mintime](/Users/renerodriguez/Desktop/projects/ai-video-detector/vendor/mintime), using the command pattern documented by the authors:

```bash
python3 predict.py \
  --video_path path/to/video.mp4 \
  --model_weights path/to/model_weights \
  --extractor_weights path/to/extractor_weights \
  --config config/size_invariant_timesformer.yaml
```

You still need to provide the MINTIME model weights. The upstream README points to their model zoo hosted on Google Drive:

- [MINTIME repository](https://github.com/davide-coccomini/MINTIME-Multi-Identity-size-iNvariant-TIMEsformer-for-Video-Deepfake-Detection)
- [MINTIME paper](https://ieeexplore.ieee.org/document/10547206)
- [Model zoo link from the README](https://drive.google.com/drive/folders/19bNOs8_rZ7LmPP3boDS3XvZcR1iryHR1?usp=sharing)

MINTIME has a heavier runtime than the rest of this app. In practice, create a separate Python 3.8-3.10 environment for it, install the dependencies from [vendor/mintime/environment.yml](/Users/renerodriguez/Desktop/projects/ai-video-detector/vendor/mintime/environment.yml), then point `MINTIME_PYTHON_BIN` at that environment's `python`.

For ROCm/Linux specifically, this repo includes:

- [setup_app_env.sh](/Users/renerodriguez/Desktop/projects/ai-video-detector/scripts/setup_app_env.sh)
- [setup_mintime_rocm.sh](/Users/renerodriguez/Desktop/projects/ai-video-detector/scripts/setup_mintime_rocm.sh)

## CLI usage

```bash
ai-video-detector /absolute/path/to/video.mp4
```

Example output:

```json
{
  "label": "likely_ai_generated",
  "confidence": 0.74,
  "reasoning": "The model observed repeated facial texture instability..."
}
```

## API usage

Start the API:

```bash
uvicorn ai_video_detector.api:app --reload
```

Open the browser UI at:

```text
http://127.0.0.1:8000/
```

Analyze a file:

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -F "video=@/absolute/path/to/video.mp4"
```

## Notes

- This is an MVP, not a courtroom-grade detector.
- MINTIME is the strongest detector in this stack and should be treated as the primary model signal once its weights are configured.
- Stronger results usually come from combining this pipeline with dedicated deepfake detectors, audio analysis, and metadata checks.
- The included heuristics are designed to be transparent and easy to replace.
- Do not commit `.env` or MINTIME weights; the root `.gitignore` excludes both.
