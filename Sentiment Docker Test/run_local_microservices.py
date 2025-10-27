"""
run_local_microservices.py - Run all microservices locally

Starts all services in separate processes:
- NLP Service: http://localhost:8001
- Graph Service: http://localhost:8002
- Audio Service: http://localhost:8003

Note: You still need to serve the frontend separately
or use: python -m http.server 80 (in static/ directory)
"""

import subprocess
import sys
import time
import signal
import os
from pathlib import Path

# Service configurations
SERVICES = [
    {
        "name": "NLP Service",
        "script": "service_nlp.py",
        "port": 8001,
        "color": "\033[94m",  # Blue
    },
    {
        "name": "Graph Service",
        "script": "service_graph.py",
        "port": 8002,
        "color": "\033[92m",  # Green
    },
    {
        "name": "Audio Service",
        "script": "audio_service.py",
        "port": 8003,
        "color": "\033[93m",  # Yellow
        "python": "3.10",  # Requires Python 3.10
    },
]

RESET = "\033[0m"

processes = []


def check_python_version(required_version=None):
    """Check if correct Python version is available"""
    version = sys.version_info
    if required_version:
        major, minor = required_version.split('.')
        if version.major != int(major) or version.minor != int(minor):
            return False
    return True


def start_service(service):
    """Start a microservice"""
    name = service["name"]
    script = service["script"]
    port = service["port"]
    color = service.get("color", "")
    required_python = service.get("python")

    print(f"{color}Starting {name} on port {port}...{RESET}")

    # Check Python version if required
    if required_python and not check_python_version(required_python):
        print(f"⚠️  {name} requires Python {required_python}")
        print(f"   Current version: {sys.version_info.major}.{sys.version_info.minor}")
        print(f"   Skipping {name}...")
        return None

    # Check if script exists
    if not Path(script).exists():
        print(f"❌ Script not found: {script}")
        return None

    # Set environment
    env = os.environ.copy()
    env["SERVICE_PORT"] = str(port)

    # Start process
    try:
        process = subprocess.Popen(
            [sys.executable, script],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        return process
    except Exception as e:
        print(f"❌ Failed to start {name}: {e}")
        return None


def cleanup(signum=None, frame=None):
    """Stop all services"""
    print("\n\n🛑 Stopping all services...")
    for process in processes:
        if process and process.poll() is None:
            process.terminate()

    # Wait for graceful shutdown
    time.sleep(2)

    # Force kill if needed
    for process in processes:
        if process and process.poll() is None:
            process.kill()

    print("✅ All services stopped")
    sys.exit(0)


def main():
    # Register signal handlers
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print("=" * 70)
    print("CiceroWatch Microservices - Local Development")
    print("=" * 70)
    print()

    # Start all services
    for service in SERVICES:
        process = start_service(service)
        if process:
            processes.append(process)
            time.sleep(1)  # Stagger startup
        print()

    if not processes:
        print("❌ No services started")
        sys.exit(1)

    print("=" * 70)
    print("✅ Services Running:")
    print("=" * 70)
    for i, service in enumerate(SERVICES):
        if i < len(processes) and processes[i]:
            print(f"  {service['name']}: http://localhost:{service['port']}")
            print(f"    API Docs: http://localhost:{service['port']}/docs")
    print()
    print("📝 Note: Frontend not started. To serve frontend:")
    print("   cd static && python -m http.server 80")
    print()
    print("🛑 Press Ctrl+C to stop all services")
    print("=" * 70)
    print()

    # Monitor services
    try:
        while True:
            for i, process in enumerate(processes):
                if process and process.poll() is not None:
                    service = SERVICES[i]
                    print(f"\n⚠️  {service['name']} stopped unexpectedly")

                    # Try to restart
                    print(f"🔄 Restarting {service['name']}...")
                    new_process = start_service(service)
                    if new_process:
                        processes[i] = new_process
                        print(f"✅ {service['name']} restarted")
                    else:
                        print(f"❌ Failed to restart {service['name']}")

            time.sleep(1)

    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
