"""
run_local.py - Local development server for Sentiment Analysis API

Fixes the uvicorn/asyncio compatibility issue in Conda environments.
Run with: python run_local.py
"""
import sys
import os

# Fix 1: Downgrade to uvicorn without loop_factory
def run_with_basic_uvicorn():
    """Run using basic uvicorn without asyncio patches"""
    try:
        import uvicorn
        # Use the basic run method without loop_factory
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8080,
            reload=True,
            log_level="info",
            loop="asyncio"  # Explicitly use asyncio loop
        )
    except TypeError as e:
        if "loop_factory" in str(e):
            print("⚠️  Detected uvicorn/asyncio compatibility issue in Conda environment")
            return False
        raise
    return True

# Fix 2: Use hypercorn instead (more compatible with Conda)
def run_with_hypercorn():
    """Alternative: Run using hypercorn (more Conda-friendly)"""
    try:
        from hypercorn.config import Config
        from hypercorn.asyncio import serve
        import asyncio

        config = Config()
        config.bind = ["0.0.0.0:8080"]

        from app import app

        print("🚀 Starting server with Hypercorn (Conda-compatible)")
        print("📍 http://localhost:8080")
        print("📚 API docs: http://localhost:8080/docs")

        asyncio.run(serve(app, config))
        return True
    except ImportError:
        print("⚠️  Hypercorn not installed. Install with: pip install hypercorn")
        return False

# Fix 3: Patch asyncio.run before importing uvicorn
def run_with_patched_asyncio():
    """Patch asyncio.run to ignore loop_factory argument"""
    import asyncio

    # Save original run
    _original_run = asyncio.run

    # Create wrapper that ignores loop_factory
    def patched_run(main, *, debug=False, loop_factory=None):
        # Ignore loop_factory argument
        return _original_run(main, debug=debug)

    # Apply patch
    asyncio.run = patched_run

    # Now import and run uvicorn
    import uvicorn
    print("🚀 Starting server with patched asyncio")
    print("📍 http://localhost:8080")
    print("📚 API docs: http://localhost:8080/docs")

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
    return True


if __name__ == "__main__":
    import platform

    print("=" * 60)
    print("Sentiment Analysis & Graph Analytics API - Local Server")
    print("=" * 60)

    # Check GPU availability
    print("\n🔍 GPU Status Check:")

    is_windows = platform.system() == 'Windows'
    py_version = sys.version_info
    py_supported = (3, 10) <= py_version[:2] <= (3, 12)

    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA version: {torch.version.cuda}")
        else:
            print("⚠ No CUDA GPU detected (using CPU)")
            print("  For GPU support, ensure:")
            print("  1. NVIDIA GPU with CUDA support")
            print("  2. NVIDIA drivers installed (check: nvidia-smi)")
            print("  3. PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121")
            if is_windows:
                print("  Or use Docker Desktop: docker build -f Dockerfile.gpu -t sentiment-gpu .")
            else:
                print("  Or use: ./run_docker_gpu.sh")
    except ImportError:
        print("⚠ PyTorch not installed (GPU detection unavailable)")
        print("  For GPU support: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        if is_windows:
            print("  Or use Docker Desktop: docker build -f Dockerfile.gpu -t sentiment-gpu .")
        else:
            print("  Or use: ./run_docker_gpu.sh")

    try:
        import cudf
        import cugraph
        print(f"✓ cuGraph {cugraph.__version__} and cuDF {cudf.__version__} available")
    except ImportError:
        print("⚠ cuGraph/cuDF not installed (graph GPU acceleration unavailable)")

        if is_windows:
            print("  ⚠ RAPIDS (cuGraph/cuDF) does NOT support Windows")
            print("  Solutions:")
            print("  1. Docker Desktop with WSL2 (recommended):")
            print("     docker build -f Dockerfile.gpu -t sentiment-gpu .")
            print("     docker run --gpus all -p 8080:8080 sentiment-gpu")
            print("  2. Use WSL2 (Windows Subsystem for Linux)")
            print("  3. CPU fallback (current mode - functional but slower)")
        elif not py_supported:
            print(f"  ⚠ Python {py_version.major}.{py_version.minor} not supported by RAPIDS")
            print(f"  RAPIDS requires Python 3.10, 3.11, or 3.12")
            print("  Solutions:")
            print("  1. Use Docker (recommended): ./run_docker_gpu.sh")
            print("  2. Create environment: conda create -n rapids python=3.12")
            print("  3. CPU fallback (current mode)")
        else:
            print("  Install: pip install cudf-cu12 cugraph-cu12 --extra-index-url=https://pypi.nvidia.com")
            print("  Or use: ./run_docker_gpu.sh")
    print()

    # Try methods in order
    methods = [
        ("Standard uvicorn", run_with_basic_uvicorn),
        ("Patched asyncio", run_with_patched_asyncio),
        ("Hypercorn (alternative)", run_with_hypercorn),
    ]

    for name, method in methods:
        print(f"\n🔧 Attempting: {name}...")
        try:
            if method():
                break
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            continue
    else:
        print("\n" + "=" * 60)
        print("❌ All methods failed. Manual workarounds:")
        print("=" * 60)
        print("\n1. Downgrade uvicorn:")
        print("   pip install 'uvicorn<0.30.0'")
        print("\n2. Use hypercorn instead:")
        print("   pip install hypercorn")
        print("   hypercorn app:app --bind 0.0.0.0:8080")
        print("\n3. Use Python's http.server:")
        print("   python -m http.server 8080")
        print("\n4. Exit Conda and use system Python:")
        print("   conda deactivate")
        print("   pip install uvicorn fastapi")
        print("   uvicorn app:app --host 0.0.0.0 --port 8080")
        sys.exit(1)
