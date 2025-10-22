"""
check_compatibility.py - Diagnose version compatibility issues

Run this before starting the server to check for common problems.
Usage: python check_compatibility.py
"""
import sys

def check_versions():
    """Check installed package versions and compatibility"""
    issues = []
    warnings = []

    print("=" * 60)
    print("Dependency Compatibility Check")
    print("=" * 60)

    # Check Python version
    print(f"\n✓ Python: {sys.version.split()[0]}")
    if sys.version_info < (3, 9):
        issues.append("Python 3.9+ is required")

    # Check PyTorch
    try:
        import torch
        torch_version = torch.__version__
        print(f"✓ PyTorch: {torch_version}")

        # Parse version
        torch_major = int(torch_version.split('.')[0])
        torch_minor = int(torch_version.split('.')[1])

        if torch_major < 2:
            warnings.append(f"PyTorch {torch_version} is old. Recommend PyTorch 2.2+")
        elif torch_major == 2 and torch_minor < 2:
            warnings.append(f"PyTorch {torch_version} may have compatibility issues. Recommend 2.2+")

    except ImportError:
        issues.append("PyTorch not installed: pip install torch>=2.2.0")
        torch_version = None
    except Exception as e:
        warnings.append(f"Could not check PyTorch version: {e}")
        torch_version = None

    # Check Transformers
    try:
        import transformers
        trans_version = transformers.__version__
        print(f"✓ Transformers: {trans_version}")

        # Check for known incompatible combinations
        if torch_version and trans_version:
            trans_major = int(trans_version.split('.')[0])
            trans_minor = int(trans_version.split('.')[1])

            # Transformers 4.44+ needs PyTorch 2.3+
            if trans_major == 4 and trans_minor >= 44:
                if torch_major == 2 and torch_minor < 3:
                    issues.append(
                        f"Transformers {trans_version} requires PyTorch 2.3+, "
                        f"but you have {torch_version}"
                    )

            # Very new transformers might be bleeding edge
            if trans_major == 4 and trans_minor >= 45:
                warnings.append(
                    f"Transformers {trans_version} is very new and may have issues. "
                    f"Consider 4.40-4.44"
                )

    except ImportError:
        issues.append("Transformers not installed: pip install transformers>=4.40.0,<4.45.0")
    except Exception as e:
        # This is the actual error we're seeing
        if "skip_code" in str(e) or "dynamo" in str(e):
            issues.append(
                "PyTorch/Transformers compatibility issue detected!\n"
                "    This is the 'skip_code' error you're seeing.\n"
                "    Fix: pip install torch>=2.3.0 transformers>=4.40.0,<4.45.0"
            )
        else:
            warnings.append(f"Could not check Transformers: {e}")

    # Check FastAPI
    try:
        import fastapi
        print(f"✓ FastAPI: {fastapi.__version__}")
    except ImportError:
        issues.append("FastAPI not installed: pip install fastapi>=0.111.0")

    # Check uvicorn
    try:
        import uvicorn
        uv_version = uvicorn.__version__
        print(f"✓ Uvicorn: {uv_version}")

        # Parse version
        uv_parts = uv_version.split('.')
        if len(uv_parts) >= 2:
            uv_minor = int(uv_parts[1])
            if uv_minor >= 30:
                warnings.append(
                    f"Uvicorn {uv_version} may have Conda compatibility issues. "
                    f"Consider downgrading: pip install 'uvicorn<0.30.0'"
                )
    except ImportError:
        issues.append("Uvicorn not installed: pip install uvicorn<0.30.0")

    # Check optional but useful packages
    optional = {
        "pandas": "Data processing",
        "numpy": "Numerical operations",
        "spacy": "NLP tasks",
        "hypercorn": "Alternative ASGI server (Conda-friendly)"
    }

    print("\nOptional packages:")
    for pkg, desc in optional.items():
        try:
            mod = __import__(pkg)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {pkg}: {version} - {desc}")
        except ImportError:
            print(f"  ✗ {pkg}: Not installed - {desc}")

    # Print results
    print("\n" + "=" * 60)

    if issues:
        print("❌ ISSUES FOUND (must fix):")
        print("=" * 60)
        for i, issue in enumerate(issues, 1):
            print(f"\n{i}. {issue}")

    if warnings:
        print("\n⚠️  WARNINGS (recommended to fix):")
        print("=" * 60)
        for i, warning in enumerate(warnings, 1):
            print(f"\n{i}. {warning}")

    if not issues and not warnings:
        print("✅ All checks passed! You're good to go.")
        print("\nStart the server with:")
        print("  python run_local.py")
        return 0

    # Provide fix instructions
    if issues:
        print("\n" + "=" * 60)
        print("QUICK FIX:")
        print("=" * 60)
        print("\n1. Uninstall conflicting packages:")
        print("   pip uninstall torch transformers -y")
        print("\n2. Install compatible versions:")
        print("   pip install torch>=2.2.0,<2.5.0 transformers>=4.40.0,<4.45.0")
        print("\n3. Or use the local requirements file:")
        print("   pip install -r requirements-local.txt")
        print("\n4. Re-run this check:")
        print("   python check_compatibility.py")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(check_versions())
