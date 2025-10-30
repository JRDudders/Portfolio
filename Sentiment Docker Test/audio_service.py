"""
Audio Deepfake Detection Microservice

This service runs on Python 3.10 with fairseq and provides
audio deepfake detection via HTTP API.

Independent service for the Audio tab in CiceroWatch.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import uvicorn
import tempfile
import os
from pathlib import Path

# Import URL fetch utility
from url_fetch import fetch_url, guess_file_extension

# Import audio processing (requires fairseq)
import audio_antispoofing

app = FastAPI(
    title="CiceroWatch Audio Service",
    description="Audio deepfake detection microservice",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = audio_antispoofing.check_models_available()
    return {
        "status": "healthy",
        "fairseq_available": status.get("fairseq", False),
        "mode": "local" if status.get("fairseq", False) else "api"
    }


@app.get("/model-status")
async def model_status():
    """Check if anti-spoofing models are ready"""
    status = audio_antispoofing.check_models_available()
    return {
        "model_ready": status.get("fairseq", False),
        "mode": "local" if status.get("fairseq", False) else "api",
        "fairseq_available": status.get("fairseq", False)
    }


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Analyze audio file for deepfake detection

    Returns:
        - prediction: "bonafide" or "spoofed"
        - confidence: confidence score (0-1)
        - spoof_score: raw score (higher = more likely spoofed)
    """
    # Validate file type
    if not file.filename.lower().endswith(('.wav', '.flac', '.mp3')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Supported: WAV, FLAC, MP3"
        )

    # Save uploaded file temporarily
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"audio_{os.getpid()}_{file.filename}")

    try:
        # Write uploaded file
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Analyze audio
        prediction, confidence, spoof_score = audio_antispoofing.predict_audio(temp_path)

        return {
            "success": True,
            "prediction": prediction,
            "confidence": float(confidence),
            "spoof_score": float(spoof_score),
            "filename": file.filename
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Audio analysis failed: {str(e)}"
        )

    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


class AudioURLRequest(BaseModel):
    """Request model for analyzing audio from URL"""
    url: HttpUrl


@app.post("/analyze-from-url")
async def analyze_audio_from_url(request: AudioURLRequest):
    """
    Analyze audio file from URL for deepfake detection

    Returns:
        - prediction: "bonafide" or "spoofed"
        - confidence: confidence score (0-1)
        - spoof_score: raw score (higher = more likely spoofed)
    """
    temp_path = None
    try:
        # Fetch audio file from URL
        audio_bytes, content_type = await fetch_url(str(request.url))

        # Guess extension
        ext = guess_file_extension(str(request.url), content_type)
        if ext not in ['.wav', '.flac', '.mp3']:
            # Try to use extension from URL
            url_ext = str(request.url).lower().split('.')[-1].split('?')[0]
            if url_ext in ['wav', 'flac', 'mp3']:
                ext = f'.{url_ext}'
            else:
                ext = '.wav'  # Default to wav

        # Save to temporary file
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"audio_url_{os.getpid()}{ext}")

        with open(temp_path, "wb") as f:
            f.write(audio_bytes)

        # Analyze audio
        prediction, confidence, spoof_score = audio_antispoofing.predict_audio(temp_path)

        return {
            "success": True,
            "prediction": prediction,
            "confidence": float(confidence),
            "spoof_score": float(spoof_score),
            "url": str(request.url)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Audio analysis from URL failed: {str(e)}"
        )

    finally:
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/download-models")
async def download_models():
    """
    Pre-download models (useful for initialization)
    """
    try:
        audio_antispoofing.download_models()
        return {"success": True, "message": "Models ready"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model download failed: {str(e)}"
        )


if __name__ == "__main__":
    port = int(os.getenv("SERVICE_PORT", 8003))

    print("=" * 70)
    print("CiceroWatch Audio Service")
    print("=" * 70)
    print("Python 3.10 + fairseq")
    print(f"Listening on: http://0.0.0.0:{port}")
    print("Features: Audio Deepfake Detection")
    print("=" * 70)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
