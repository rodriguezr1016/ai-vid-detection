# Linux ROCm Deployment

This project is intended to run on Linux with AMD ROCm when you want MINTIME acceleration.

## Recommended host

- Ubuntu 24.04.3 or Ubuntu 22.04.5
- ROCm 7.2
- Python 3.10 for the MINTIME environment
- A separate venv for the app and for MINTIME

AMD's current ROCm documentation lists Radeon RX 9070 XT support on Linux, and the PyTorch on ROCm docs recommend their tested Docker images or ROCm wheel installs for PyTorch 2.9.1.

## 1. Clone the repo

```bash
git clone <your-repo-url>
cd ai-video-detector
```

## 2. Create the app environment

```bash
./scripts/setup_app_env.sh
```

## 3. Create the MINTIME ROCm environment

```bash
PYTHON_BIN=python3.10 ./scripts/setup_mintime_rocm.sh
```

## 4. Download MINTIME weights

Download the upstream MINTIME checkpoint into:

```text
vendor/mintime/weights/
```

The upstream project links its model zoo from Google Drive.

## 5. Configure `.env`

Set these values:

```bash
LLM_API_KEY=your_openai_api_key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini

MINTIME_PYTHON_BIN=/absolute/path/to/ai-video-detector/vendor/mintime/.venv-mintime/bin/python
MINTIME_REPO_PATH=/absolute/path/to/ai-video-detector/vendor/mintime
MINTIME_CONFIG_PATH=/absolute/path/to/ai-video-detector/vendor/mintime/config/size_invariant_timesformer.yaml
MINTIME_MODEL_WEIGHTS=/absolute/path/to/ai-video-detector/vendor/mintime/weights/model.pth
MINTIME_EXTRACTOR_MODEL=0
MINTIME_EXTRACTOR_WEIGHTS=ImageNet
MINTIME_DETECTOR_TYPE=FacenetDetector
MINTIME_GPU_ID=0
MINTIME_TIMEOUT_SECONDS=300
```

## 6. Start the app

```bash
source .venv/bin/activate
uvicorn ai_video_detector.api:app --host 0.0.0.0 --port 8000
```

Open `http://<linux-host>:8000/`.

## Notes

- MINTIME itself is upstream code and may still need dependency tuning on some ROCm installations.
- If bare-metal ROCm gives you trouble, AMD recommends using their ROCm PyTorch Docker images.
