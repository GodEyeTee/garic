"""Shared config helpers for GARIC Nautilus integration."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml


DEFAULT_CONFIG: dict = {
    "nautilus": {
        "symbol": "BTCUSDT",
        "instrument_id": "BTCUSDT-PERP.BINANCE",
        "venue": "BINANCE",
        "bar_minutes": 15,
        "history_bars": 160,
        "request_history_days": 3,
        "trade_size": "0.002",
        "initial_balance_usdt": 10_000.0,
        "leverage": 1,
        "model_path": "",
        "data_path": "",
        "state_path": "checkpoints/nautilus_dashboard_state.json",
        "log_path": "nautilus.log",
        "dashboard_log_path": "nautilus_dashboard.log",
        "environment": "LIVE",
        "api_key_env": "BINANCE_API_KEY",
        "api_secret_env": "BINANCE_API_SECRET",
        "close_positions_on_stop": True,
        "reduce_only_on_stop": True,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_nautilus_config(path: str | None = None) -> dict:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        config = _deep_merge(config, payload)

    cfg = config.setdefault("nautilus", {})
    cfg["model_path"] = str(resolve_model_path(cfg.get("model_path", "")))
    cfg["data_path"] = str(resolve_data_path(cfg.get("symbol", "BTCUSDT"), cfg.get("data_path", "")))
    cfg["state_path"] = str(Path(cfg.get("state_path", "checkpoints/nautilus_dashboard_state.json")))
    cfg["log_path"] = str(Path(cfg.get("log_path", "nautilus.log")))
    cfg["dashboard_log_path"] = str(Path(cfg.get("dashboard_log_path", "nautilus_dashboard.log")))
    cfg["trade_size"] = str(cfg.get("trade_size", "0.002"))
    cfg["bar_minutes"] = int(cfg.get("bar_minutes", 15))
    cfg["history_bars"] = int(cfg.get("history_bars", 160))
    cfg["request_history_days"] = int(cfg.get("request_history_days", 3))
    cfg["initial_balance_usdt"] = float(cfg.get("initial_balance_usdt", 10_000.0))
    cfg["leverage"] = int(cfg.get("leverage", 1))
    cfg["environment"] = str(cfg.get("environment", "LIVE")).upper()
    return config


def resolve_model_path(configured: str | os.PathLike[str] | None = None) -> Path:
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            Path("checkpoints/rl_agent_supervised.joblib"),
            Path("checkpoints/rl_agent_final.zip"),
            Path("checkpoints/rl_agent_best.zip"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_data_path(symbol: str, configured: str | os.PathLike[str] | None = None) -> Path:
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured))
    candidates.append(Path("data/raw") / f"{symbol.upper()}_1m.parquet")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_live_credentials(config: dict) -> tuple[str, str]:
    cfg = config["nautilus"]
    api_key = os.getenv(cfg.get("api_key_env", "BINANCE_API_KEY"), "").strip()
    api_secret = os.getenv(cfg.get("api_secret_env", "BINANCE_API_SECRET"), "").strip()
    return api_key, api_secret

