"""
FastAPI server for Sarashina-TTS with OpenAI-compatible API.
"""

import os
import hashlib
import tempfile
import uuid
import shutil
import asyncio
from typing import List, Optional, Dict
from contextlib import asynccontextmanager
from enum import Enum

import soundfile as sf
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn
from starlette.background import BackgroundTask

from sarashina_tts.generate.generate import SarashinaTTSGenerator
from sarashina_tts.flow_matching.decoder import FlowDecoder
from sarashina_tts.text_frontend.text_splitter import split_text, STRATEGY_AUTO
from sarashina_tts.utils.audio_concat import concat_wavs


# Global generator instance
gen: Optional[SarashinaTTSGenerator] = None
cache: dict = {}

# Task storage for async processing
tasks: Dict[str, Dict] = {}


class TaskStatus(str, Enum):
    """Task status enumeration."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    init_generator()
    yield
    # Shutdown
    pass


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Sarashina TTS API",
    description="OpenAI-compatible Text-to-Speech API for Sarashina-TTS",
    version="1.0.0",
    lifespan=lifespan,
)


class SpeechRequest(BaseModel):
    """OpenAI-compatible speech request."""

    model: str = Field(default="sarashina-tts", description="Model name")
    input: str = Field(..., description="Text to synthesize", max_length=4096)
    voice: str = Field(default="default", description="Voice preset (currently unused)")
    response_format: str = Field(default="mp3", description="Audio format: mp3, wav")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Audio speed")


class ModelInfo(BaseModel):
    """Model information."""

    id: str
    name: str
    description: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    supported_formats: List[str]


class TaskResponse(BaseModel):
    """Task creation response."""

    task_id: str
    status: TaskStatus
    message: str


class TaskStatusResponse(BaseModel):
    """Task status response."""

    task_id: str
    status: TaskStatus
    progress: Optional[str] = None
    error: Optional[str] = None


def _get_audio_duration(filepath: str) -> float:
    """Return the duration of an audio file in seconds."""
    info = sf.info(filepath)
    return info.duration


def init_generator():
    """Initialize the SarashinaTTSGenerator."""
    global gen
    if gen is None:
        gen = SarashinaTTSGenerator()
    return gen


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok", model_loaded=gen is not None, supported_formats=["mp3", "wav"]
    )


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "sarashina-tts",
                "object": "model",
                "created": 1234567890,
                "owned_by": "sarashina-tts",
            }
        ],
    }


@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    """
    Generate speech from text (OpenAI-compatible endpoint).

    This endpoint requires an audio prompt file to be uploaded separately
    or provided via a different mechanism since Sarashina-TTS is a
    zero-shot TTS that needs a reference audio.
    """
    if gen is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    # For now, we'll return an error since we need the audio prompt
    # This will be implemented in the next step
    raise HTTPException(
        status_code=400,
        detail="Audio prompt file required. Use /tts endpoint with file upload.",
    )


async def process_tts_task(
    task_id: str,
    text: str,
    prompt_file_path: str,
    prompt_text: str,
    max_length: int,
    temperature: float,
    top_p: float,
):
    """Process TTS generation in background."""
    global tasks

    try:
        tasks[task_id]["status"] = TaskStatus.PROCESSING
        tasks[task_id]["progress"] = "Loading prompt audio..."

        # Load prompt audio
        prompt_audio, sr = sf.read(prompt_file_path)
        if sr != FlowDecoder.sample_rate:
            from scipy import signal

            num_samples = int(len(prompt_audio) * FlowDecoder.sample_rate / sr)
            prompt_audio = signal.resample(prompt_audio, num_samples)

        tasks[task_id]["progress"] = "Extracting audio prompt features..."

        # Extract prompt features
        if gen is None:
            raise HTTPException(status_code=503, detail="Model not initialized")

        key = hashlib.md5(
            prompt_audio.tobytes() + prompt_text.encode("utf-8")
        ).hexdigest()

        if key not in cache:
            tokens, feat = gen.extract_audio_prompt(
                prompt_audio, prompt_text=prompt_text
            )
            cache[key] = (tokens, feat)
        else:
            tokens, feat = cache[key]

        tasks[task_id]["progress"] = "Splitting text into segments..."

        # Split text
        segments = split_text(
            text,
            strategy=STRATEGY_AUTO,
            prompt_duration_sec=len(prompt_audio) / FlowDecoder.sample_rate,
        )

        tasks[task_id]["progress"] = f"Generating {len(segments)} segments..."

        all_wavs = []
        gen_kwargs = {
            "max_tokens": max_length,
            "temperature": temperature,
            "top_p": top_p,
        }

        for i, seg in enumerate(segments):
            tasks[task_id]["progress"] = (
                f"Generating segment {i + 1}/{len(segments)}..."
            )
            wavs = gen.generate(
                text=seg,
                audio_prompt_text=prompt_text,
                audio_prompt_tokens=tokens,
                audio_prompt_feat=feat,
                watermark=True,
                gen_kwargs=gen_kwargs,
            )
            all_wavs.extend(wavs)

        tasks[task_id]["progress"] = "Concatenating segments..."

        # Concatenate all segments
        final_wav = concat_wavs(all_wavs, sample_rate=FlowDecoder.sample_rate)

        tasks[task_id]["progress"] = "Saving audio file..."

        # Save to temporary directory using gen.save_audios
        tmp_dir = tempfile.mkdtemp()
        paths = gen.save_audios([final_wav], output_dir=tmp_dir)
        output_path = paths[0]

        tasks[task_id]["status"] = TaskStatus.COMPLETED
        tasks[task_id]["progress"] = "Completed"
        tasks[task_id]["output_path"] = output_path
        tasks[task_id]["tmp_dir"] = tmp_dir

    except Exception as e:
        tasks[task_id]["status"] = TaskStatus.FAILED
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["progress"] = "Failed"


@app.post("/tts")
async def text_to_speech(
    text: str = Form(...),
    prompt_file: UploadFile = File(...),
    prompt_text: str = Form(default=""),
    response_format: str = Form(default="mp3"),
    max_length: int = Form(default=2000),
    temperature: float = Form(default=0.9),
    top_p: float = Form(default=0.95),
):
    """
    Generate speech from text with audio prompt (async task).

    This endpoint creates a background task and returns a task ID.
    Use /tts/{task_id} to check status and /tts/{task_id}/download to get the result.
    """
    if gen is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    # Create task ID
    task_id = str(uuid.uuid4())

    # Save uploaded prompt file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_prompt:
        tmp_prompt.write(await prompt_file.read())
        tmp_prompt_path = tmp_prompt.name

    # Initialize task
    tasks[task_id] = {
        "status": TaskStatus.PENDING,
        "progress": "Queued",
        "error": None,
        "output_path": None,
        "tmp_dir": None,
    }

    # Start background task
    asyncio.create_task(
        process_tts_task(
            task_id,
            text,
            tmp_prompt_path,
            prompt_text,
            max_length,
            temperature,
            top_p,
        )
    )

    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message="Task created. Use /tts/{task_id} to check status.",
    )


@app.get("/tts/{task_id}")
async def get_task_status(task_id: str):
    """Get task status."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]
    return TaskStatusResponse(
        task_id=task_id,
        status=task["status"],
        progress=task["progress"],
        error=task["error"],
    )


@app.get("/tts/{task_id}/download")
async def download_task_result(task_id: str):
    """Download task result audio file."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]

    if task["status"] != TaskStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Task not completed. Current status: {task['status']}",
        )

    if task["output_path"] is None or not os.path.exists(task["output_path"]):
        raise HTTPException(status_code=404, detail="Output file not found")

    return FileResponse(
        task["output_path"],
        media_type="audio/wav",
        filename="output.wav",
        background=BackgroundTask(shutil.rmtree, task["tmp_dir"]),
    )


@app.post("/tts/sync")
async def text_to_speech_sync(
    text: str = Form(...),
    prompt_file: UploadFile = File(...),
    prompt_text: str = Form(default=""),
    response_format: str = Form(default="mp3"),
    max_length: int = Form(default=2000),
    temperature: float = Form(default=0.9),
    top_p: float = Form(default=0.95),
):
    """
    Generate speech from text with audio prompt (synchronous).

    This is the original synchronous endpoint for backward compatibility.
    Note: This may timeout for long texts.
    """
    if gen is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    # Save uploaded prompt file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_prompt:
        tmp_prompt.write(await prompt_file.read())
        tmp_prompt_path = tmp_prompt.name

    try:
        # Load prompt audio
        prompt_audio, sr = sf.read(tmp_prompt_path)
        if sr != FlowDecoder.sample_rate:
            from scipy import signal

            num_samples = int(len(prompt_audio) * FlowDecoder.sample_rate / sr)
            prompt_audio = signal.resample(prompt_audio, num_samples)

        # Extract prompt features
        key = hashlib.md5(
            prompt_audio.tobytes() + prompt_text.encode("utf-8")
        ).hexdigest()

        if key not in cache:
            tokens, feat = gen.extract_audio_prompt(
                prompt_audio, prompt_text=prompt_text
            )
            cache[key] = (tokens, feat)
        else:
            tokens, feat = cache[key]

        # Split text
        segments = split_text(
            text,
            strategy=STRATEGY_AUTO,
            prompt_duration_sec=len(prompt_audio) / FlowDecoder.sample_rate,
        )

        all_wavs = []
        gen_kwargs = {
            "max_tokens": max_length,
            "temperature": temperature,
            "top_p": top_p,
        }

        for seg in segments:
            wavs = gen.generate(
                text=seg,
                audio_prompt_text=prompt_text,
                audio_prompt_tokens=tokens,
                audio_prompt_feat=feat,
                watermark=True,
                gen_kwargs=gen_kwargs,
            )
            all_wavs.extend(wavs)

        # Concatenate all segments
        final_wav = concat_wavs(all_wavs, sample_rate=FlowDecoder.sample_rate)

        # Save to temporary directory using gen.save_audios
        tmp_dir = tempfile.mkdtemp()
        paths = gen.save_audios([final_wav], output_dir=tmp_dir)
        output_path = paths[0]

        # Return WAV file with background cleanup
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="output.wav",
            background=BackgroundTask(shutil.rmtree, tmp_dir),
        )

    finally:
        # Clean up temporary prompt file
        if os.path.exists(tmp_prompt_path):
            os.unlink(tmp_prompt_path)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        timeout_keep_alive=600,  # 10 minutes keep-alive timeout
        timeout_graceful_shutdown=30,  # 30 seconds graceful shutdown
        workers=1,  # Single worker for TTS generation (model is memory-intensive)
        limit_concurrency=5,  # Allow more concurrent requests for retry scenarios
    )
