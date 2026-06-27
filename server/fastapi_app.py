"""
FastAPI server for Sarashina-TTS with OpenAI-compatible API.
"""

import os
import hashlib
import tempfile
from typing import List, Optional
from contextlib import asynccontextmanager

import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn

from sarashina_tts.generate.generate import SarashinaTTSGenerator
from sarashina_tts.flow_matching.decoder import FlowDecoder
from sarashina_tts.text_frontend.text_splitter import split_text, STRATEGY_AUTO
from sarashina_tts.utils.audio_concat import concat_wavs


# Global generator instance
gen: Optional[SarashinaTTSGenerator] = None
cache: dict = {}


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
    Generate speech from text with audio prompt.

    This is the main endpoint for Sarashina-TTS which requires:
    - text: Text to synthesize
    - prompt_file: Reference audio file for voice cloning
    - prompt_text: Transcription of the reference audio
    """
    if gen is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    # Save uploaded prompt file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_prompt:
        tmp_prompt.write(await prompt_file.read())
        tmp_prompt_path = tmp_prompt.name

    try:
        # Compute cache key
        with open(tmp_prompt_path, "rb") as f:
            data = f.read()
        key = hashlib.md5(data + prompt_text.encode("utf-8")).hexdigest()

        if key not in cache:
            flow_emb = gen._extract_zero_shot_embedding(tmp_prompt_path)
            tokens = gen._extract_audio_prompt_tokens(tmp_prompt_path)
            feat = gen._extract_audio_prompt_feat(tmp_prompt_path)
            cache[key] = (flow_emb, tokens, feat)
        else:
            flow_emb, tokens, feat = cache[key]

        gen_kwargs = {
            "max_length": max_length,
            "repetition_penalty": 1.0,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
        }

        # Split text into segments
        prompt_duration = _get_audio_duration(tmp_prompt_path)
        segments = split_text(
            text,
            strategy=STRATEGY_AUTO,
            prompt_duration_s=prompt_duration,
        )

        # Generate each segment
        all_wavs: List[torch.Tensor] = []
        for segment in segments:
            wavs = gen.generate(
                texts=[segment],
                flow_embedding=flow_emb,
                audio_prompt_path=tmp_prompt_path,
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
        # This handles the audio format correctly
        import shutil

        tmp_dir = tempfile.mkdtemp()
        paths = gen.save_audios([final_wav], output_dir=tmp_dir)
        output_path = paths[0]

        # Return WAV file with background cleanup
        from starlette.background import BackgroundTask

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
