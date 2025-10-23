#!/usr/bin/env python
"""
GPU Setup Verification Script
Checks if your system is ready for GPU-accelerated graph analytics.
"""

import subprocess
import sys

def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def check_nvidia_driver():
    """Check if NVIDIA drivers are installed."""
    print_header("1. NVIDIA Driver Check")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ NVIDIA drivers installed")
            # Extract GPU info
            for line in result.stdout.split('\n'):
                if 'NVIDIA' in line or 'Driver Version' in line or 'CUDA Version' in line:
                    print(f"   {line.strip()}")
            return True
        else:
            print("❌ nvidia-smi failed")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found")
        print("   → NVIDIA drivers are NOT installed")
        print("   → GPU acceleration will NOT work")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_pytorch():
    """Check if PyTorch with CUDA is installed."""
    print_header("2. PyTorch CUDA Check")
    try:
        import torch
        print(f"✅ PyTorch installed: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.version.cuda}")
            print(f"   GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("⚠️  PyTorch installed but CUDA not available")
            print("   → Need to install PyTorch with CUDA support")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        print("   → Run: pip install torch")
        return False

def check_rapids():
    """Check if RAPIDS (cuDF/cuGraph) is installed."""
    print_header("3. RAPIDS (cuGraph/cuDF) Check")
    has_cudf = False
    has_cugraph = False

    try:
        import cudf
        print(f"✅ cuDF installed: {cudf.__version__}")
        has_cudf = True
    except ImportError:
        print("❌ cuDF not installed")

    try:
        import cugraph
        print(f"✅ cuGraph installed: {cugraph.__version__}")
        has_cugraph = True
    except ImportError:
        print("❌ cuGraph not installed")

    if not (has_cudf and has_cugraph):
        print("\n   → Install RAPIDS via conda:")
        print("   conda install -c rapidsai -c conda-forge -c nvidia \\")
        print("     cudf=24.04 cugraph=24.04 python=3.11 cuda-version=12.0")

    return has_cudf and has_cugraph

def check_graph_tasks():
    """Check if graph_tasks.py detects GPU."""
    print_header("4. graph_tasks.py GPU Detection")
    try:
        import graph_tasks
        if graph_tasks.HAS_GPU:
            print("✅ GPU acceleration ENABLED")
            print(f"   HAS_GPU: {graph_tasks.HAS_GPU}")
            print(f"   HAS_CUGRAPH: {graph_tasks.HAS_CUGRAPH}")
            return True
        else:
            print("⚠️  GPU acceleration DISABLED")
            print(f"   HAS_GPU: {graph_tasks.HAS_GPU}")
            print(f"   HAS_CUGRAPH: {graph_tasks.HAS_CUGRAPH}")
            print("   → Graph analytics will use CPU")
            return False
    except Exception as e:
        print(f"❌ Error importing graph_tasks: {e}")
        return False

def main():
    print("\n" + "🔍 GPU ACCELERATION VERIFICATION SCRIPT" + "\n")

    has_driver = check_nvidia_driver()
    has_pytorch = check_pytorch()
    has_rapids = check_rapids()
    has_gpu_enabled = check_graph_tasks()

    # Final summary
    print_header("Summary")

    if has_driver and has_pytorch and has_rapids and has_gpu_enabled:
        print("🎉 SUCCESS! GPU acceleration is fully configured and working!")
        print("\nYour graph analytics will run on GPU automatically.")
        print("Expected speedup: 10-100x for large graphs")
    else:
        print("⚠️  GPU acceleration is NOT configured")
        print("\nMissing components:")
        if not has_driver:
            print("  - NVIDIA drivers (hardware or drivers missing)")
        if not has_pytorch:
            print("  - PyTorch with CUDA support")
        if not has_rapids:
            print("  - RAPIDS (cuDF/cuGraph)")

        print("\n📌 RECOMMENDED ACTIONS:")
        print("\n1. Check if you have an NVIDIA GPU:")
        print("   lspci | grep -i nvidia")
        print("\n2. If you have GPU, install via Docker (easiest):")
        print("   docker build -f Dockerfile.gpu -t sentiment-api:gpu .")
        print("   docker run --gpus all -p 8080:8080 sentiment-api:gpu")
        print("\n3. OR install locally via conda:")
        print("   conda create -n sentiment-gpu python=3.11")
        print("   conda activate sentiment-gpu")
        print("   conda install -c rapidsai -c conda-forge -c nvidia \\")
        print("     cudf=24.04 cugraph=24.04 cuda-version=12.0")
        print("   pip install -r requirements-local.txt")

        print("\n4. If no NVIDIA GPU:")
        print("   → CPU-only mode will work fine for smaller graphs")
        print("   → App automatically falls back to CPU")

    print("\n" + "=" * 70 + "\n")

if __name__ == "__main__":
    main()
