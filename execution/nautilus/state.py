"""Realtime JSON state for the Nautilus browser dashboard."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def _to_builtin(value: Any):
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


class NautilusStateWriter:
    def __init__(self, path: str | Path, max_history: int = 512):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_history = max(int(max_history), 1)
        self.state: dict[str, Any] = self._default_state()
        if self.path.exists():
            try:
                existing = json.loads(self.path.read_text(encoding="utf-8"))
                self.state = self._merge_states(self.state, existing)
            except Exception:
                pass

    def _default_state(self) -> dict[str, Any]:
        return {
            "updated_at": 0.0,
            "status": "IDLE",
            "mode": "backtest",
            "symbol": "",
            "instrument_id": "",
            "model_path": "",
            "model_family": "",
            "history": {
                "ts": [],
                "price": [],
                "equity": [],
                "position": [],
                "upnl": [],
            },
            "action_counts": {"short": 0, "flat": 0, "long": 0},
            "recent_events": [],
        }

    def _merge_states(self, base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base)
        for key, value in incoming.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._merge_states(merged[key], value)
            elif isinstance(value, list):
                existing = merged.get(key, [])
                if len(value) >= len(existing):
                    merged[key] = [_to_builtin(v) for v in value]
            elif value not in (None, "", 0, 0.0):
                merged[key] = _to_builtin(value)
        return merged

    def reset(self, **fields) -> None:
        if self.path.exists():
            try:
                self.path.unlink()
            except Exception:
                pass
        self.state = self._default_state()
        self.update(**fields)

    def update(self, *, event: str | None = None, history: dict[str, float] | None = None, **fields) -> None:
        if self.path.exists():
            try:
                current = json.loads(self.path.read_text(encoding="utf-8"))
                self.state = self._merge_states(current, self.state)
            except Exception:
                pass

        self.state.update({k: _to_builtin(v) for k, v in fields.items()})
        self.state["updated_at"] = time.time()

        if history:
            for key, value in history.items():
                bucket = self.state.setdefault("history", {}).setdefault(key, [])
                bucket.append(_to_builtin(value))
                if len(bucket) > self.max_history:
                    del bucket[:-self.max_history]

        if event:
            events = self.state.setdefault("recent_events", [])
            events.append({"ts": time.time(), "message": str(event)})
            if len(events) > 40:
                del events[:-40]

        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = json.dumps(_to_builtin(self.state), ensure_ascii=False, indent=2)
        tmp_path.write_text(payload, encoding="utf-8")
        for _ in range(5):
            try:
                tmp_path.replace(self.path)
                return
            except PermissionError:
                time.sleep(0.05)
        try:
            self.path.write_text(payload, encoding="utf-8")
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
