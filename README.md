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

## Recommended deployment

The best target for this project is a Windows PC with an NVIDIA GPU such as an RTX 4070.

That setup is preferred because:

- the FastAPI app and UI are lightweight
- MINTIME was originally built around a CUDA/PyTorch workflow
- a dedicated NVIDIA desktop is a better target than a MacBook for video-level inference

macOS can still be used for UI-only testing, but the main detector should be treated as a Windows GPU deployment target.

## App setup

Prerequisites:

- Python 3.10
- Git
- NVIDIA driver installed
- A CUDA-compatible PyTorch build for your GPU

PowerShell:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
Copy-Item .env.example .env
```

For Linux + AMD ROCm deployment, see [linux-rocm.md](/Users/renerodriguez/Desktop/projects/ai-video-detector/docs/linux-rocm.md).
For local macOS testing, see [macos-mintime.md](/Users/renerodriguez/Desktop/projects/ai-video-detector/docs/macos-mintime.md).

## Environment variables

```bash
LLM_API_KEY=your_key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
LLM_TIMEOUT_SECONDS=45
FRAME_SAMPLE_COUNT=8
MAX_VIDEO_DURATION_SECONDS=120
MINTIME_PYTHON_BIN=C:\path\to\ai-video-detector\vendor\mintime\.venv-mintime\Scripts\python.exe
MINTIME_REPO_PATH=C:\path\to\ai-video-detector\vendor\mintime
MINTIME_CONFIG_PATH=C:\path\to\ai-video-detector\vendor\mintime\config\size_invariant_timesformer.yaml
MINTIME_MODEL_WEIGHTS=C:\path\to\model_checkpoint
MINTIME_EXTRACTOR_MODEL=1
MINTIME_EXTRACTOR_WEIGHTS=C:\path\to\extractor_checkpoint
MINTIME_DETECTOR_TYPE=FacenetDetector
MINTIME_DEVICE=cuda
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

You still need to provide the MINTIME weights. The upstream README points to their model zoo hosted on Google Drive:

- [MINTIME repository](https://github.com/davide-coccomini/MINTIME-Multi-Identity-size-iNvariant-TIMEsformer-for-Video-Deepfake-Detection)
- [MINTIME paper](https://ieeexplore.ieee.org/document/10547206)
- [Model zoo link from the README](https://drive.google.com/drive/folders/19bNOs8_rZ7LmPP3boDS3XvZcR1iryHR1?usp=sharing)

For the current app integration, use the `XC` pair of checkpoints from the model zoo:

- `MINTIME_XC_Model_checkpoint30`
- `MINTIME_XC_Extractor_checkpoint30`

These match the Xception-backed path, so the correct settings are:

```bash
MINTIME_EXTRACTOR_MODEL=1
MINTIME_MODEL_WEIGHTS=/absolute/path/to/MINTIME_XC_Model_checkpoint30
MINTIME_EXTRACTOR_WEIGHTS=/absolute/path/to/MINTIME_XC_Extractor_checkpoint30
```

MINTIME has a heavier runtime than the rest of this app. In practice, create a separate Python 3.10 environment for it and point `MINTIME_PYTHON_BIN` at that environment's `python.exe`.

For ROCm/Linux specifically, this repo includes:

- [setup_app_env.sh](/Users/renerodriguez/Desktop/projects/ai-video-detector/scripts/setup_app_env.sh)
- [setup_mintime_mac.sh](/Users/renerodriguez/Desktop/projects/ai-video-detector/scripts/setup_mintime_mac.sh)
- [setup_mintime_rocm.sh](/Users/renerodriguez/Desktop/projects/ai-video-detector/scripts/setup_mintime_rocm.sh)

For a Windows NVIDIA machine such as a PC with an RTX 4070, use a dedicated MINTIME venv with CUDA-enabled PyTorch. Do not assume anything is installed. Install Python 3.10 first, then create the venv, then install PyTorch, then install the remaining MINTIME dependencies.

Example outline:

```powershell
cd vendor/mintime
py -3.10 -m venv .venv-mintime
.venv-mintime\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio
python -m pip install \
  facenet-pytorch==2.5.3 \
  efficientnet-pytorch==0.7.1 \
  einops==0.6.0 \
  albumentations==0.5.2 \
  imgaug==0.4.0 \
  imageio==2.37.0 \
  opencv-python==4.11.0.86 \
  pillow==10.2.0 \
  scikit-image==0.22.0 \
  scikit-learn==1.5.2 \
  scipy==1.13.1 \
  pyyaml==6.0.2 \
  pandas==2.2.3 \
  matplotlib==3.8.4 \
  networkx==3.3 \
  shapely==2.0.7 \
  tabulate==0.9.0 \
  termcolor==2.5.0 \
  torchsummary==1.5.1 \
  tqdm==4.67.1 \
  yacs==0.1.8 \
  timm==0.9.16
```

If the plain `torch torchvision torchaudio` install does not give you a CUDA build, install the correct PyTorch command for your CUDA version from the official PyTorch selector first, then run the remaining package install.

For local macOS use, start with `MINTIME_DEVICE=cpu`. The vendored MINTIME inference path has been patched to support `cpu`, `mps`, and `cuda`, but `cpu` is the safest default on a MacBook.
The macOS setup script intentionally skips `av` and `pytorchvideo`, because the local prediction path does not require them.

## Windows RTX 4070 quick start

On the target Windows PC:

```powershell
git clone <your-repo-url>
cd ai-video-detector
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
Copy-Item .env.example .env
```

Create the MINTIME environment, download the two `XC` checkpoints into `vendor/mintime/weights/`, and set `.env` to something like:

```env
LLM_API_KEY=your_openai_api_key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini

MINTIME_PYTHON_BIN=C:\path\to\ai-video-detector\vendor\mintime\.venv-mintime\Scripts\python.exe
MINTIME_REPO_PATH=C:\path\to\ai-video-detector\vendor\mintime
MINTIME_CONFIG_PATH=C:\path\to\ai-video-detector\vendor\mintime\config\size_invariant_timesformer.yaml
MINTIME_MODEL_WEIGHTS=C:\path\to\ai-video-detector\vendor\mintime\weights\MINTIME_XC_Model_checkpoint30
MINTIME_EXTRACTOR_MODEL=1
MINTIME_EXTRACTOR_WEIGHTS=C:\path\to\ai-video-detector\vendor\mintime\weights\MINTIME_XC_Extractor_checkpoint30
MINTIME_DETECTOR_TYPE=FacenetDetector
MINTIME_DEVICE=cuda
MINTIME_GPU_ID=0
MINTIME_TIMEOUT_SECONDS=300
```

Start the app:

```powershell
.venv\Scripts\Activate.ps1
uvicorn ai_video_detector.api:app --host 0.0.0.0 --port 8000
```

Then open the UI from another machine at `http://<windows-host>:8000/`.

## CLI usage

```powershell
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

```powershell
uvicorn ai_video_detector.api:app --reload
```

Open the browser UI at:

```text
http://127.0.0.1:8000/
```

Analyze a file:

```powershell
curl -X POST "http://127.0.0.1:8000/analyze" \
  -F "video=@/absolute/path/to/video.mp4"
```

## Notes

- This is an MVP, not a courtroom-grade detector.
- MINTIME is the strongest detector in this stack and should be treated as the primary model signal once its weights are configured.
- Stronger results usually come from combining this pipeline with dedicated deepfake detectors, audio analysis, and metadata checks.
- The included heuristics are designed to be transparent and easy to replace.
- Do not commit `.env` or MINTIME weights; the root `.gitignore` excludes both.
