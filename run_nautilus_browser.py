"""Launch GARIC Nautilus mode with browser dashboard."""

from __future__ import annotations

import argparse
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
        key = str(candidate.resolve())
        if key in seen:
            continue
        seen.add(key)
        if _python_has_module(candidate, "streamlit") and _python_has_module(candidate, "nautilus_trader"):
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
    parser = argparse.ArgumentParser(description="Launch GARIC Nautilus browser dashboard")
    parser.add_argument("--mode", choices=["backtest", "paper", "live"], default="backtest")
    parser.add_argument("--config", default="configs/nautilus.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8502)
    parser.add_argument("--dashboard-timeout", type=float, default=20.0)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-hold", action="store_true")
    parser.add_argument("--limit-bars", type=int, default=None)
    args = parser.parse_args()

    python_exe = _pick_python_executable()
    dashboard_script = ROOT / "monitoring" / "nautilus" / "dashboard.py"
    dashboard_log_path = ROOT / "nautilus_dashboard.log"
    url = f"http://{args.host}:{args.port}"

    if not _python_has_module(python_exe, "streamlit"):
        print("[GARIC] Streamlit is not installed in the selected Python environment.", file=sys.stderr)
        print(f"[GARIC] Python: {python_exe}", file=sys.stderr)
        return 1
    if not _python_has_module(python_exe, "nautilus_trader"):
        print("[GARIC] NautilusTrader is not installed in the selected Python environment.", file=sys.stderr)
        print(f"[GARIC] Python: {python_exe}", file=sys.stderr)
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

    if args.mode == "backtest":
        runner_cmd = [
            str(python_exe),
            "-m",
            "execution.nautilus.backtest_runner",
            "--config",
            args.config,
        ]
        if args.limit_bars:
            runner_cmd.extend(["--limit-bars", str(args.limit_bars)])
    else:
        runner_cmd = [
            str(python_exe),
            "-m",
            "execution.nautilus.live_runner",
            "--mode",
            args.mode,
            "--config",
            args.config,
        ]

    dashboard_log = dashboard_log_path.open("w", encoding="utf-8")
    dashboard_proc = None
    try:
        print(f"[GARIC] Starting Nautilus dashboard at {url}")
        print(f"[GARIC] Using Python: {python_exe}")
        dashboard_proc = subprocess.Popen(
            dashboard_cmd,
            cwd=str(ROOT),
            stdout=dashboard_log,
            stderr=dashboard_log,
        )
        if not _wait_for_port(args.host, args.port, args.dashboard_timeout):
            print(f"[GARIC] Dashboard failed to start. Check {dashboard_log_path.name}", file=sys.stderr)
            return 1
        print(f"[GARIC] Dashboard ready: {url}")
        if not args.no_browser:
            webbrowser.open(url)

        exit_code = subprocess.call(runner_cmd, cwd=str(ROOT))
        print(f"[GARIC] Nautilus {args.mode} finished with exit code {exit_code}")
        if not args.no_hold and sys.stdin.isatty():
            input("Press Enter to close the Nautilus dashboard...")
        return exit_code
    finally:
        _stop_process(dashboard_proc)
        dashboard_log.close()


if __name__ == "__main__":
    raise SystemExit(main())
