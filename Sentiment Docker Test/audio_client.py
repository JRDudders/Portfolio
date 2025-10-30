"""
Audio Service Client

This module handles communication with the audio deepfake detection service.
Falls back to local processing if service is unavailable.
"""

import os
import requests
from typing import Tuple, Optional

# Audio service URL from environment or default
AUDIO_SERVICE_URL = os.getenv("AUDIO_SERVICE_URL", "http://localhost:8081")


def check_audio_service() -> bool:
    """
    Check if audio service is available

    Returns:
        True if service is healthy, False otherwise
    """
    try:
        response = requests.get(f"{AUDIO_SERVICE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def analyze_audio_remote(file_path: str) -> Tuple[str, float, float]:
    """
    Analyze audio using remote service

    Args:
        file_path: Path to audio file

    Returns:
        (prediction, confidence, spoof_score)
    """
    with open(file_path, 'rb') as f:
        files = {'file': (os.path.basename(file_path), f)}
        response = requests.post(
            f"{AUDIO_SERVICE_URL}/analyze",
            files=files,
            timeout=300
        )

    if response.status_code != 200:
        raise Exception(f"Audio service error: {response.text}")

    result = response.json()
    return (
        result['prediction'],
        result['confidence'],
        result['spoof_score']
    )


def analyze_audio(file_path: str) -> Tuple[str, float, float]:
    """
    Analyze audio - uses remote service if available, falls back to local

    Args:
        file_path: Path to audio file

    Returns:
        (prediction, confidence, spoof_score)
    """
    # Try remote service first
    if check_audio_service():
        print("Using remote audio service (Python 3.10 + fairseq)")
        try:
            return analyze_audio_remote(file_path)
        except Exception as e:
            print(f"Remote service failed: {e}")
            print("Falling back to local processing...")

    # Fall back to local processing
    print("Using local audio processing")
    import audio_antispoofing
    return audio_antispoofing.predict_audio(file_path)


if __name__ == "__main__":
    # Test the service
    if check_audio_service():
        print(f"✅ Audio service is available at {AUDIO_SERVICE_URL}")
        response = requests.get(f"{AUDIO_SERVICE_URL}/health")
        print(f"Status: {response.json()}")
    else:
        print(f"❌ Audio service not available at {AUDIO_SERVICE_URL}")
        print("Will use local processing as fallback")
