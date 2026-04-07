"""Tests for the unified GARIC launcher."""

from argparse import Namespace
from pathlib import Path

import garic


def _ns(**overrides):
    base = {
        "profile": None,
        "config": None,
        "cli": False,
        "no_cache": False,
        "host": "127.0.0.1",
        "port": 8501,
        "no_browser": False,
        "no_hold": False,
        "limit_bars": None,
    }
    base.update(overrides)
    return Namespace(**base)


def test_resolve_config_uses_default_profile():
    assert garic.resolve_config_path(None, None, default_profile="best") == "configs/train_rtx2060.yaml"


def test_explicit_config_wins_over_profile():
    assert garic.resolve_config_path("configs/custom.yaml", "best", default_profile="smoke") == "configs/custom.yaml"


def test_build_train_command_defaults_to_web_launcher():
    args = _ns(profile="best")
    command, config_path = garic.build_train_command(args, Path("python.exe"))

    assert config_path == "configs/train_rtx2060.yaml"
    assert command[:2] == ["python.exe", str(garic.ROOT / "run_training_browser.py")]
    assert "--config" in command
    assert "--host" in command
    assert "--port" in command


def test_build_train_command_cli_uses_pipeline_and_no_cache():
    args = _ns(profile="cloud", cli=True, no_cache=True)
    command, config_path = garic.build_train_command(args, Path("python.exe"))

    assert config_path == "configs/default.yaml"
    assert command == [
        "python.exe",
        str(garic.ROOT / "pipeline.py"),
        "--mode",
        "train",
        "--config",
        "configs/default.yaml",
        "--no-cache",
    ]


def test_build_test_command_defaults_to_smoke_profile():
    args = _ns()
    command, config_path = garic.build_test_command(args, Path("python.exe"))

    assert config_path == "configs/test_rtx2060.yaml"
    assert command == [
        "python.exe",
        str(garic.ROOT / "pipeline.py"),
        "--mode",
        "test",
        "--config",
        "configs/test_rtx2060.yaml",
    ]


def test_build_backtest_command_defaults_to_web_launcher():
    args = _ns(limit_bars=400, port=8502)
    command, config_path = garic.build_nautilus_command("backtest", args, Path("python.exe"))

    assert config_path == "configs/nautilus.yaml"
    assert command[:2] == ["python.exe", str(garic.ROOT / "run_nautilus_browser.py")]
    assert "--mode" in command and "backtest" in command
    assert "--limit-bars" in command and "400" in command


def test_build_live_command_cli_uses_nautilus_module():
    args = _ns(cli=True)
    command, config_path = garic.build_nautilus_command("live", args, Path("python.exe"))

    assert config_path == "configs/nautilus.yaml"
    assert command == [
        "python.exe",
        "-m",
        "execution.nautilus.live_runner",
        "--mode",
        "live",
        "--config",
        "configs/nautilus.yaml",
    ]
