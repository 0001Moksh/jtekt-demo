
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"
BACKEND_DIR = ROOT_DIR / "backend"

HOST = "0.0.0.0"
FRONTEND_PORT = 5175
BACKEND_PORT = 8003

VENV_PYTHON = BACKEND_DIR / ".venv" / "Scripts" / "python.exe"

processes = []


def start_services():
    if not (FRONTEND_DIR / "package.json").exists():
        raise FileNotFoundError(f"Missing frontend: {FRONTEND_DIR}")

    if not (BACKEND_DIR / "main.py").exists():
        raise FileNotFoundError(f"Missing backend: {BACKEND_DIR}")

    if not VENV_PYTHON.exists():
        raise FileNotFoundError(f"Missing Python environment: {VENV_PYTHON}")

    try:
        print("\nStarting SharePoint Chatbot...\n")

        # Start FastAPI backend on the LAN
        backend = subprocess.Popen(
            [
                str(VENV_PYTHON),
                "-m", "uvicorn",
                "main:app",
                "--reload",
                "--host", HOST,
                "--port", str(BACKEND_PORT),
            ],
            cwd=str(BACKEND_DIR),
        )
        processes.append(backend)

        # Start Vite frontend on the LAN
        frontend = subprocess.Popen(
            [
                "npm.cmd",
                "run", "dev",
                "--",
                "--host", HOST,
                "--port", str(FRONTEND_PORT),
                "--strictPort",
            ],
            cwd=str(FRONTEND_DIR),
        )
        processes.append(frontend)

        print("==========================================")
        print(" SharePoint Chatbot is starting")
        print("==========================================")
        print(f"Frontend LAN: http://192.168.10.8:{FRONTEND_PORT}/")
        print(f"Backend LAN:  http://192.168.10.8:{BACKEND_PORT}/")
        print(f"API Docs:     http://192.168.10.8:{BACKEND_PORT}/docs")
        print("\nPress Ctrl+C to stop both services.\n")

        while True:
            for process in processes:
                if process.poll() is not None:
                    print(f"Service exited with code {process.returncode}")
                    return
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping services...")

    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()

        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

        print("All services stopped.")


if __name__ == "__main__":
    try:
        start_services()
    except (FileNotFoundError, OSError) as error:
        print(f"Startup error: {error}")
        sys.exit(1)
