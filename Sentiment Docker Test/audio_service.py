"""
Audio Deepfake Detection Microservice

This service runs on Python 3.10 with fairseq and provides
audio deepfake detection via HTTP API.

The main app (Python 3.12) calls this service for audio analysis.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import tempfile
import os
from pathlib import Path

# Import audio processing (requires fairseq)
import audio_antispoofing

app = FastAPI(title="Audio Deepfake Detection Service", version="1.0.0")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = audio_antispoofing.check_models_available()
    return {
        "status": "healthy",
        "fairseq_available": status.get("fairseq", False),
        "mode": "local" if status.get("fairseq", False) else "api"
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
    print("=" * 70)
    print("Audio Deepfake Detection Service")
    print("=" * 70)
    print("Python 3.10 + fairseq")
    print("Listening on: http://0.0.0.0:8081")
    print("=" * 70)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8081,
        log_level="info"
    )
