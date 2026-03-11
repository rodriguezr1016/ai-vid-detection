from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .pipeline import VideoDetectionPipeline

app = FastAPI(title="AI Video Detector", version="0.1.0")
pipeline = VideoDetectionPipeline()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parents[2] / "templates"))
app.mount(
    "/static",
    StaticFiles(directory=str(Path(__file__).resolve().parents[2] / "static")),
    name="static",
)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "result": None,
            "error": None,
        },
    )


@app.post("/analyze")
async def analyze(video: UploadFile = File(...)) -> dict:
    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    temp_path = ""
    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(await video.read())
            temp_path = temp_file.name

        result = pipeline.analyze(temp_path)
        return result.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


@app.post("/", response_class=HTMLResponse)
async def analyze_form(request: Request, video: UploadFile = File(...)) -> HTMLResponse:
    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    temp_path = ""
    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(await video.read())
            temp_path = temp_file.name

        result = pipeline.analyze(temp_path)
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "result": result.model_dump(),
                "error": None,
            },
        )
    except Exception as exc:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "result": None,
                "error": str(exc),
            },
            status_code=400,
        )
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)
