"""Launch the GARIC browser dashboard and training pipeline with one command."""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _python_has_module(python_exe: Path | str, module_name: str) -> bool:
    try:
        result = subprocess.run(
            [str(python_exe), "-c", f"import {module_name}"],
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def _pick_python_executable() -> Path:
    candidates = [
        Path(sys.executable),
        ROOT / "venv" / "Scripts" / "python.exe",
        ROOT / ".venv" / "Scripts" / "python.exe",
    ]
    seen = set()
    for candidate in candidates:
        if not candidate.exists():
            continue
        candidate_key = str(candidate.resolve())
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        if _python_has_module(candidate, "streamlit"):
            return candidate
    return Path(sys.executable)


def _wait_for_port(host: str, port: int, timeout_sec: float) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.8)
        try:
            if sock.connect_ex((host, port)) == 0:
                return True
        finally:
            sock.close()
        time.sleep(0.25)
    return False


def _stop_process(proc: subprocess.Popen | None):
    if proc is None or proc.poll() is not None:
        return

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch browser dashboard + training with one command")
    parser.add_argument("--config", default="configs/train_rtx2060.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--dashboard-timeout", type=float, default=20.0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-hold", action="store_true", help="Exit immediately when training finishes")
    args = parser.parse_args()

    dashboard_script = ROOT / "monitoring" / "training" / "dashboard.py"
    pipeline_script = ROOT / "pipeline.py"
    dashboard_log_path = ROOT / "dashboard.log"
    dashboard_url = f"http://{args.host}:{args.port}"
    python_exe = _pick_python_executable()

    env = os.environ.copy()
    env["GARIC_DASHBOARD_MODE"] = "web"

    if not _python_has_module(python_exe, "streamlit"):
        print(
            "[GARIC] Streamlit is not installed in the selected Python environment.",
            file=sys.stderr,
        )
        print(f"[GARIC] Python: {python_exe}", file=sys.stderr)
        print("[GARIC] Install it with: pip install streamlit plotly", file=sys.stderr)
        return 1

    dashboard_cmd = [
        str(python_exe),
        "-m",
        "streamlit",
        "run",
        str(dashboard_script),
        "--server.headless",
        "true",
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
        "--browser.gatherUsageStats",
        "false",
    ]

    dashboard_proc = None
    dashboard_log = dashboard_log_path.open("w", encoding="utf-8")
    try:
        print(f"[GARIC] Starting browser dashboard at {dashboard_url}")
        print(f"[GARIC] Using Python: {python_exe}")
        dashboard_proc = subprocess.Popen(
            dashboard_cmd,
            cwd=str(ROOT),
            env=env,
            stdout=dashboard_log,
            stderr=dashboard_log,
        )

        if not _wait_for_port(args.host, args.port, args.dashboard_timeout):
            print(f"[GARIC] Dashboard failed to start. Check {dashboard_log_path.name}", file=sys.stderr)
            return 1

        print(f"[GARIC] Dashboard ready: {dashboard_url}")
        print(f"[GARIC] Dashboard logs: {dashboard_log_path.name}")
        if not args.no_browser:
            webbrowser.open(dashboard_url)

        pipeline_cmd = [
            str(python_exe),
            str(pipeline_script),
            "--mode",
            "train",
            "--config",
            args.config,
        ]
        if args.no_cache:
            pipeline_cmd.append("--no-cache")

        print("[GARIC] Starting training pipeline...")
        exit_code = subprocess.call(
            pipeline_cmd,
            cwd=str(ROOT),
            env=env,
        )

        print(f"[GARIC] Training finished with exit code {exit_code}")
        print(f"[GARIC] Browser dashboard is still available at {dashboard_url}")

        if not args.no_hold and sys.stdin.isatty():
            input("Press Enter to close the dashboard...")
        return exit_code
    except KeyboardInterrupt:
        print("\n[GARIC] Interrupted by user")
        return 130
    finally:
        _stop_process(dashboard_proc)
        dashboard_log.close()


if __name__ == "__main__":
    raise SystemExit(main())
