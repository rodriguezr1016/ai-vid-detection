# macOS MINTIME

You can test this project on a MacBook, including the MINTIME path, but expect slower inference than on Linux GPU hardware.

## Recommended local setup

- Use the normal app venv for the FastAPI app
- Use a separate Python 3.10 venv for MINTIME
- Start with `MINTIME_DEVICE=cpu`
- Optionally try `MINTIME_DEVICE=mps` later if your Mac supports PyTorch MPS well enough

## Setup

```bash
cd /Users/renerodriguez/Desktop/projects/ai-video-detector
./scripts/setup_app_env.sh
PYTHON_BIN=python3.10 ./scripts/setup_mintime_mac.sh
cp .env.example .env
```

## `.env`

```bash
LLM_API_KEY=your_openai_api_key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini

MINTIME_PYTHON_BIN=/Users/renerodriguez/Desktop/projects/ai-video-detector/vendor/mintime/.venv-mintime/bin/python
MINTIME_REPO_PATH=/Users/renerodriguez/Desktop/projects/ai-video-detector/vendor/mintime
MINTIME_CONFIG_PATH=/Users/renerodriguez/Desktop/projects/ai-video-detector/vendor/mintime/config/size_invariant_timesformer.yaml
MINTIME_MODEL_WEIGHTS=/absolute/path/to/mintime_model_weights
MINTIME_EXTRACTOR_MODEL=0
MINTIME_EXTRACTOR_WEIGHTS=ImageNet
MINTIME_DETECTOR_TYPE=FacenetDetector
MINTIME_DEVICE=cpu
MINTIME_GPU_ID=0
MINTIME_TIMEOUT_SECONDS=300
```

## Run

```bash
source .venv/bin/activate
uvicorn ai_video_detector.api:app --reload
```

Open `http://127.0.0.1:8000/`.

## Notes

- The MINTIME weights are still required.
- The macOS setup script intentionally skips `av` and `pytorchvideo`; the local prediction path does not need them.
- If `cpu` works, you can experiment with `MINTIME_DEVICE=mps`.
- If MINTIME is still unstable locally, keep using the app on macOS and run MINTIME on Linux for the final signal.
