"""
Sarashina TTS Launcher
=====================
Pystray-based launcher for Gradio web UI and FastAPI server.
"""

import os
import runpy
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import pystray
from PIL import Image, ImageDraw

APP_NAME = "Sarashina TTS Launcher"
HOST = "127.0.0.1"
GRADIO_PORT = 7860
FASTAPI_PORT = 8000
GRADIO_URL = f"http://localhost:{GRADIO_PORT}"
FASTAPI_URL = f"http://localhost:{FASTAPI_PORT}"
STARTUP_TIMEOUT_SECONDS = 60
GRADIO_CHILD_ARG = "--gradio-child"
FASTAPI_CHILD_ARG = "--fastapi-child"
MAX_LOG_SIZE_BYTES = 1_000_000

gradio_process = None
fastapi_process = None
tray_icon = None
model_dir = None


def get_base_path() -> Path:
    """Return base path for both source and PyInstaller bundle execution."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


def get_gradio_app_path() -> Path:
    """Return the Gradio application path."""
    return get_base_path() / "server" / "gradio_app.py"


def get_fastapi_app_path() -> Path:
    """Return the FastAPI application path."""
    return get_base_path() / "server" / "fastapi_app.py"


def get_tray_icon_path() -> Path:
    """Return the tray icon image path."""
    return get_base_path() / "static" / "tts_icon.png"


def get_log_path() -> Path:
    """Return launcher log file path."""
    log_dir = Path.home() / ".sarashina_tts" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "tts_launcher.log"


def get_model_dir_path() -> Path:
    """Return model directory path.

    Priority:
    1. Global model_dir variable (set via command line arg)
    2. SARASHINA_MODEL_DIR environment variable
    3. Default: ~/.sarashina_tts/pretrained_models
    """
    global model_dir
    if model_dir:
        return Path(model_dir)

    env_model_dir = os.environ.get("SARASHINA_MODEL_DIR")
    if env_model_dir:
        return Path(env_model_dir)

    default_model_dir = Path.home() / ".sarashina_tts" / "pretrained_models"
    return default_model_dir


def should_use_vllm() -> bool:
    """Return whether vLLM backend should be enabled."""
    env_value = os.environ.get("SARASHINA_USE_VLLM")
    if env_value is not None:
        return env_value.lower() in {"1", "true", "yes", "on"}
    return bool(getattr(sys, "frozen", False))


def get_python_path() -> Path:
    """Return Python executable path for child server processes."""
    python_path = os.environ.get("SARASHINA_PYTHON_PATH")
    if python_path:
        return Path(python_path)

    vllm_metal_python = Path.home() / ".venv-vllm-metal" / "bin" / "python"
    if getattr(sys, "frozen", False) and vllm_metal_python.exists():
        return vllm_metal_python

    return Path(sys.executable)


def rotate_log_if_needed(log_path: Path) -> None:
    """Rotate the launcher log when it exceeds the maximum size."""
    if not log_path.exists() or log_path.stat().st_size <= MAX_LOG_SIZE_BYTES:
        return

    rotated_log_path = log_path.with_name(f"{log_path.name}.1")
    if rotated_log_path.exists():
        rotated_log_path.unlink()
    log_path.rename(rotated_log_path)


def write_log(message: str) -> None:
    """Write a message to the launcher log file."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_path = get_log_path()
    rotate_log_if_needed(log_path)
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"[{timestamp}] {message}\n")


def is_port_open(host: str, port: int) -> bool:
    """Check whether the Gradio server port is accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def wait_for_gradio() -> bool:
    """Wait until Gradio is ready or timeout is reached."""
    deadline = time.time() + STARTUP_TIMEOUT_SECONDS
    while time.time() < deadline:
        if is_port_open(HOST, GRADIO_PORT):
            return True
        time.sleep(0.5)
    return False


def wait_for_fastapi() -> bool:
    """Wait until FastAPI is ready or timeout is reached."""
    deadline = time.time() + STARTUP_TIMEOUT_SECONDS
    while time.time() < deadline:
        if is_port_open(HOST, FASTAPI_PORT):
            return True
        time.sleep(0.5)
    return False


def create_icon_image() -> Image.Image:
    """Create a simple tray icon image."""
    tray_icon_path = get_tray_icon_path()
    if tray_icon_path.exists():
        return Image.open(tray_icon_path).convert("RGBA")

    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((8, 8, 56, 56), radius=12, fill=(168, 85, 247, 255))
    draw.text((18, 20), "TTS", fill=(255, 255, 255, 255))
    return image


def start_gradio() -> None:
    """Start Gradio server if it is not already running."""
    global gradio_process

    if is_port_open(HOST, GRADIO_PORT):
        write_log(f"Gradio is already running at {GRADIO_URL}")
        return

    gradio_app_path = get_gradio_app_path()
    model_dir_path = get_model_dir_path()
    use_vllm = should_use_vllm()
    write_log(f"Starting Gradio with app path: {gradio_app_path}")
    write_log(f"Using model directory: {model_dir_path}")
    write_log(f"Using vLLM backend: {use_vllm}")

    python_path = get_python_path()

    if getattr(sys, "frozen", False) and python_path == Path(sys.executable):
        command = [
            str(python_path),
            GRADIO_CHILD_ARG,
            str(gradio_app_path),
            "--model-dir",
            str(model_dir_path),
        ]
    else:
        command = [
            str(python_path),
            str(gradio_app_path),
            "--model-dir",
            str(model_dir_path),
        ]

    if use_vllm:
        command.append("--use-vllm")

    write_log(f"Gradio command: {' '.join(command)}")
    log_file = open(get_log_path(), "a", encoding="utf-8")
    gradio_process = subprocess.Popen(
        command,
        cwd=str(get_base_path()),
        env={
            **os.environ,
            "SARASHINA_TTS_CHILD": "1",
        },
        stdout=log_file,
        stderr=log_file,
    )
    write_log(f"Gradio process started with PID: {gradio_process.pid}")


def stop_gradio() -> None:
    """Stop Gradio server if it is running."""
    global gradio_process

    if gradio_process is not None and gradio_process.poll() is None:
        write_log(f"Stopping Gradio process (PID: {gradio_process.pid})")
        gradio_process.terminate()
        try:
            gradio_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            write_log("Gradio process did not terminate gracefully, killing...")
            gradio_process.kill()
            gradio_process.wait()
        gradio_process = None
        write_log("Gradio process stopped")
    else:
        # Try to kill process on port 7860
        try:
            result = subprocess.run(
                ["lsof", "-ti", str(GRADIO_PORT)],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.stdout.strip():
                pids = result.stdout.strip().split("\n")
                for pid in pids:
                    write_log(f"Killing process {pid} on port {GRADIO_PORT}")
                    subprocess.run(["kill", "-9", pid], check=False)
        except Exception as e:
            write_log(f"Error killing process on port {GRADIO_PORT}: {e}")


def start_fastapi() -> None:
    """Start FastAPI server if it is not already running."""
    global fastapi_process

    if is_port_open(HOST, FASTAPI_PORT):
        write_log(f"FastAPI is already running at {FASTAPI_URL}")
        return

    fastapi_app_path = get_fastapi_app_path()
    model_dir_path = get_model_dir_path()
    use_vllm = should_use_vllm()
    write_log(f"Starting FastAPI with app path: {fastapi_app_path}")
    write_log(f"Using model directory: {model_dir_path}")
    write_log(f"Using vLLM backend: {use_vllm}")

    python_path = get_python_path()

    log_path = get_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "a", encoding="utf-8") as log_file:
        if getattr(sys, "frozen", False) and python_path == Path(sys.executable):
            cmd = [
                str(python_path),
                FASTAPI_CHILD_ARG,
                str(fastapi_app_path),
                "--model-dir",
                str(model_dir_path),
            ]
        else:
            cmd = [
                str(python_path),
                str(fastapi_app_path),
                "--model-dir",
                str(model_dir_path),
            ]
        if use_vllm:
            cmd.append("--use-vllm")
        write_log(f"FastAPI command: {' '.join(cmd)}")

        fastapi_process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
        )
        write_log(f"FastAPI process started with PID: {fastapi_process.pid}")


def stop_fastapi() -> None:
    """Stop FastAPI server if it is running."""
    global fastapi_process

    if fastapi_process is not None and fastapi_process.poll() is None:
        write_log(f"Stopping FastAPI process (PID: {fastapi_process.pid})")
        fastapi_process.terminate()
        try:
            fastapi_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            write_log("FastAPI process did not terminate gracefully, killing...")
            fastapi_process.kill()
            fastapi_process.wait()
        fastapi_process = None
        write_log("FastAPI process stopped")
    else:
        # Try to kill process on port 8000
        try:
            result = subprocess.run(
                ["lsof", "-ti", str(FASTAPI_PORT)],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.stdout.strip():
                pids = result.stdout.strip().split("\n")
                for pid in pids:
                    write_log(f"Killing process {pid} on port {FASTAPI_PORT}")
                    subprocess.run(["kill", "-9", pid], check=False)
        except Exception as e:
            write_log(f"Error killing process on port {FASTAPI_PORT}: {e}")


def open_tts_ui() -> None:
    """Open Sarashina TTS Web UI in the default browser."""
    webbrowser.open(GRADIO_URL)


def show_logs() -> None:
    """Open the launcher log file in the default text editor."""
    log_path = get_log_path()
    if log_path.exists():
        subprocess.run(["open", str(log_path)], check=False)
    else:
        write_log(f"Log file not found: {log_path}")


def quit_app(icon: pystray.Icon) -> None:
    """Stop Gradio, FastAPI and quit the tray application."""
    global gradio_process, fastapi_process

    stop_gradio()
    stop_fastapi()
    icon.stop()


def setup_tray() -> pystray.Icon:
    """Create and return tray icon."""
    menu = pystray.Menu(
        pystray.MenuItem("Open Sarashina TTS", lambda: open_tts_ui()),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Start Gradio", lambda: start_gradio()),
        pystray.MenuItem("Stop Gradio", lambda: stop_gradio()),
        pystray.MenuItem("Start FastAPI", lambda: start_fastapi()),
        pystray.MenuItem("Stop FastAPI", lambda: stop_fastapi()),
        pystray.MenuItem("Show Logs", lambda: show_logs()),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", quit_app),
    )
    return pystray.Icon(APP_NAME, create_icon_image(), APP_NAME, menu)


def main() -> None:
    """Start tray icon without auto-starting servers."""
    global tray_icon

    write_log("Launcher started")
    tray_icon = setup_tray()
    tray_icon.run()


def run_server_child(child_arg: str) -> None:
    """Run a bundled server script inside a child process."""
    try:
        child_arg_index = sys.argv.index(child_arg)
        app_path = sys.argv[child_arg_index + 1]
    except (ValueError, IndexError):
        raise SystemExit("Missing server app path.")

    child_args = sys.argv[child_arg_index + 2 :]
    sys.argv = [app_path, *child_args]
    runpy.run_path(app_path, run_name="__main__")


if __name__ == "__main__":
    if GRADIO_CHILD_ARG in sys.argv:
        run_server_child(GRADIO_CHILD_ARG)
    elif FASTAPI_CHILD_ARG in sys.argv:
        run_server_child(FASTAPI_CHILD_ARG)
    else:
        main()
