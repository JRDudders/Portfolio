"""
Audio Deepfake Detection Microservice

This service runs on Python 3.10 with fairseq and provides
audio deepfake detection via HTTP API.

Independent service for the Audio tab in CiceroWatch.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os
from pathlib import Path

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
