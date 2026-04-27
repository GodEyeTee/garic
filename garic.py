"""Unified GARIC launcher.

Use this entry point instead of memorizing multiple script names.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROFILE_CONFIGS = {
    "best": "configs/train_rtx2060.yaml",
    "cloud": "configs/default.yaml",
    "smoke": "configs/test_rtx2060.yaml",
}
NAUTILUS_DEFAULT_CONFIG = "configs/nautilus.yaml"


def _python_has_modules(python_exe: Path | str, modules: list[str]) -> bool:
    if not modules:
        return True
    check_script = (
        "import importlib.util, sys\n"
        "missing = [name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]\n"
        "raise SystemExit(1 if missing else 0)\n"
    )
    try:
        result = subprocess.run(
            [str(python_exe), "-c", check_script, *modules],
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def pick_python_executable(required_modules: list[str] | None = None) -> Path | None:
    """Prefer the current interpreter, then project venvs, if they satisfy requirements."""
    modules = list(required_modules or [])
    candidates = [
        Path(sys.executable),
        ROOT / "venv" / "Scripts" / "python.exe",
        ROOT / ".venv" / "Scripts" / "python.exe",
    ]
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate.exists():
            continue
        key = str(candidate.resolve())
        if key in seen:
            continue
        seen.add(key)
        if _python_has_modules(candidate, modules):
            return candidate
    return None


def resolve_config_path(
    config: str | None,
    profile: str | None,
    *,
    default_profile: str | None = None,
    fallback: str | None = None,
) -> str:
    """Resolve a custom config first, otherwise map a friendly profile to a YAML file."""
    if config:
        return config
    if profile:
        return PROFILE_CONFIGS[profile]
    if default_profile:
        return PROFILE_CONFIGS[default_profile]
    if fallback:
        return fallback
    raise ValueError("No config or profile could be resolved")


def build_train_command(args: argparse.Namespace, python_exe: Path | str) -> tuple[list[str], str]:
    config_path = resolve_config_path(args.config, args.profile, default_profile="best")
    if args.cli:
        command = [
            str(python_exe),
            str(ROOT / "pipeline.py"),
            "--mode",
            "train",
            "--config",
            config_path,
        ]
        if args.no_cache:
            command.append("--no-cache")
        return command, config_path

    command = [
        str(python_exe),
        str(ROOT / "run_training_browser.py"),
        "--config",
        config_path,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.no_cache:
        command.append("--no-cache")
    if args.no_browser:
        command.append("--no-browser")
    if args.no_hold:
        command.append("--no-hold")
    return command, config_path


def build_test_command(args: argparse.Namespace, python_exe: Path | str) -> tuple[list[str], str]:
    config_path = resolve_config_path(args.config, args.profile, default_profile="smoke")
    command = [
        str(python_exe),
        str(ROOT / "pipeline.py"),
        "--mode",
        "test",
        "--config",
        config_path,
    ]
    if args.no_cache:
        command.append("--no-cache")
    return command, config_path


def build_nautilus_command(
    mode: str,
    args: argparse.Namespace,
    python_exe: Path | str,
) -> tuple[list[str], str]:
    config_path = resolve_config_path(args.config, None, fallback=NAUTILUS_DEFAULT_CONFIG)
    if args.cli:
        if mode == "backtest":
            command = [
                str(python_exe),
                "-m",
                "execution.nautilus.backtest_runner",
                "--config",
                config_path,
            ]
            if args.limit_bars is not None:
                command.extend(["--limit-bars", str(args.limit_bars)])
            return command, config_path

        command = [
            str(python_exe),
            "-m",
            "execution.nautilus.live_runner",
            "--mode",
            mode,
            "--config",
            config_path,
        ]
        return command, config_path

    command = [
        str(python_exe),
        str(ROOT / "run_nautilus_browser.py"),
        "--mode",
        mode,
        "--config",
        config_path,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.no_browser:
        command.append("--no-browser")
    if args.no_hold:
        command.append("--no-hold")
    if mode == "backtest" and args.limit_bars is not None:
        command.extend(["--limit-bars", str(args.limit_bars)])
    return command, config_path


def build_download_command(args: argparse.Namespace, python_exe: Path | str) -> tuple[list[str], str]:
    command = [
        str(python_exe),
        "-m",
        "data.downloaders.binance_historical",
        "--pairs",
        *args.pairs,
        "--interval",
        args.interval,
        "--start",
        args.start,
    ]
    if args.end:
        command.extend(["--end", args.end])
    if getattr(args, "clear", False):
        command.append("--clear")
    return command, "none"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified GARIC launcher. Prefer this over calling individual helper scripts directly.",
    )
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser(
        "train",
        help="Recommended full local training run",
        description="Train the model with the current recommended local workflow.",
    )
    train_parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_CONFIGS),
        default="best",
        help="Config profile: best=current local recipe, cloud=base config, smoke=lightweight.",
    )
    train_parser.add_argument("--config", default=None, help="Custom YAML override. Takes precedence over --profile.")
    train_parser.add_argument("--cli", action="store_true", help="Run pipeline directly without the Streamlit dashboard.")
    train_parser.add_argument("--no-cache", action="store_true", help="Rebuild features instead of using cache.")
    train_parser.add_argument("--host", default="127.0.0.1", help="Dashboard host in web mode.")
    train_parser.add_argument("--port", type=int, default=8501, help="Dashboard port in web mode.")
    train_parser.add_argument("--no-browser", action="store_true", help="Do not auto-open the dashboard URL.")
    train_parser.add_argument("--no-hold", action="store_true", help="Exit immediately when the web launcher finishes.")

    test_parser = subparsers.add_parser(
        "test",
        help="Lightweight validation run",
        description="Run the train/test pipeline in test mode.",
    )
    test_parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_CONFIGS),
        default="smoke",
        help="Config profile for test mode. smoke is the default quick validation profile.",
    )
    test_parser.add_argument("--config", default=None, help="Custom YAML override. Takes precedence over --profile.")
    test_parser.add_argument("--cli", action="store_true", help="Run pipeline directly without the Streamlit dashboard.")
    test_parser.add_argument("--no-cache", action="store_true", help="Rebuild features instead of using cache.")
    test_parser.add_argument("--host", default="127.0.0.1", help="Dashboard host in web mode.")
    test_parser.add_argument("--port", type=int, default=8501, help="Dashboard port in web mode.")
    test_parser.add_argument("--no-browser", action="store_true", help="Do not auto-open the dashboard URL.")
    test_parser.add_argument("--no-hold", action="store_true", help="Exit immediately when the web launcher finishes.")

    download_parser = subparsers.add_parser(
        "download",
        help="Download historical data from Binance",
        description="Download OHLCV data from Binance Data Vision without API keys.",
    )
    download_parser.add_argument("--pairs", nargs="+", default=["BTCUSDT", "ETHUSDT"], help="Trading pairs to download.")
    download_parser.add_argument("--interval", default="1m", help="Candle interval (default 1m).")
    download_parser.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD).")
    download_parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD).")
    download_parser.add_argument("--clear", action="store_true", help="Clear old data for the pairs before downloading.")

    for mode, default_port, help_text in (
        ("backtest", 8502, "Run the Nautilus backtest flow"),
        ("paper", 8502, "Run the Nautilus paper-trading flow"),
        ("live", 8502, "Run the Nautilus live-trading flow"),
    ):
        mode_parser = subparsers.add_parser(mode, help=help_text, description=help_text)
        mode_parser.add_argument("--config", default=None, help="Custom Nautilus YAML config.")
        mode_parser.add_argument("--cli", action="store_true", help="Run the Nautilus module directly without dashboard.")
        mode_parser.add_argument("--host", default="127.0.0.1", help="Dashboard host in web mode.")
        mode_parser.add_argument("--port", type=int, default=default_port, help="Dashboard port in web mode.")
        mode_parser.add_argument("--no-browser", action="store_true", help="Do not auto-open the dashboard URL.")
        mode_parser.add_argument("--no-hold", action="store_true", help="Exit immediately when the web launcher finishes.")
        if mode == "backtest":
            mode_parser.add_argument("--limit-bars", type=int, default=None, help="Limit loaded bars for a faster backtest.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 2

    if args.command == "train":
        required = ["yaml", "numpy", "pandas", "stable_baselines3"]
        if not args.cli:
            required.insert(0, "streamlit")
        python_exe = pick_python_executable(required)
    elif args.command == "test":
        required = ["yaml", "numpy", "pandas", "stable_baselines3"]
        python_exe = pick_python_executable(required)
    elif args.command == "download":
        required = ["requests", "pandas"]
        python_exe = pick_python_executable(required)
    else:
        required = ["yaml", "nautilus_trader"]
        if not args.cli:
            required.insert(0, "streamlit")
        python_exe = pick_python_executable(required)

    if python_exe is None:
        modules_str = ", ".join(required)
        print("[GARIC] No compatible Python environment was found.", file=sys.stderr)
        print("[GARIC] Checked: current interpreter, venv\\Scripts\\python.exe, .venv\\Scripts\\python.exe", file=sys.stderr)
        print(f"[GARIC] Required modules: {modules_str}", file=sys.stderr)
        print("[GARIC] Recreate the project venv and reinstall requirements if your old venv points to a removed Python installation.", file=sys.stderr)
        return 1

    if args.command == "train":
        command, config_path = build_train_command(args, python_exe)
    elif args.command == "test":
        command, config_path = build_test_command(args, python_exe)
    elif args.command == "download":
        command, config_path = build_download_command(args, python_exe)
    else:
        command, config_path = build_nautilus_command(args.command, args, python_exe)

    print(f"[GARIC] Python: {python_exe}")
    print(f"[GARIC] Config: {config_path}")
    return subprocess.call(command, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
