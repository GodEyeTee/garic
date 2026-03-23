import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


pytest.importorskip("nautilus_trader")

from execution.nautilus.features import NautilusFeatureBuilder
from execution.nautilus.model import ACTION_TO_DIRECTION
from execution.nautilus.state import NautilusStateWriter
from execution.nautilus.strategy import GaricNautilusStrategy


def test_nautilus_feature_builder_outputs_30_features():
    n = 200
    close = 100 + np.cumsum(np.random.randn(n) * 0.2)
    frame = pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
            "open": close - 0.1,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "volume": np.random.uniform(10, 100, size=n),
            "trades": np.random.randint(50, 250, size=n),
            "taker_buy_volume": np.random.uniform(5, 50, size=n),
        }
    )
    builder = NautilusFeatureBuilder(history_bars=160)
    snapshot = builder.build_latest(frame)
    assert snapshot.feature_array.shape == (30,)
    assert np.isfinite(snapshot.feature_array).all()
    assert set(ACTION_TO_DIRECTION) == {0, 1, 2}


def test_nautilus_state_writer_persists_history(tmp_path: Path):
    path = tmp_path / "state.json"
    writer = NautilusStateWriter(path, max_history=3)
    writer.update(status="RUNNING", history={"ts": 1, "price": 100.0}, event="first")
    writer.update(status="RUNNING", history={"ts": 2, "price": 101.0}, event="second")
    writer.update(status="RUNNING", history={"ts": 3, "price": 102.0}, event="third")
    writer.update(status="RUNNING", history={"ts": 4, "price": 103.0}, event="fourth")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "RUNNING"
    assert payload["history"]["ts"] == [2, 3, 4]
    assert len(payload["recent_events"]) == 4


def test_nautilus_state_writer_merges_existing_state(tmp_path: Path):
    path = tmp_path / "state.json"
    first = NautilusStateWriter(path, max_history=4)
    first.update(
        status="RUNNING",
        action_counts={"short": 3, "flat": 1, "long": 2},
        history={"ts": 1, "price": 100.0},
        event="strategy update",
    )

    second = NautilusStateWriter(path, max_history=4)
    second.update(status="COMPLETE", backtest={"total_orders": 2}, event="runner summary")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "COMPLETE"
    assert payload["action_counts"] == {"short": 3, "flat": 1, "long": 2}
    assert payload["history"]["ts"] == [1]
    assert len(payload["recent_events"]) == 2


def test_nautilus_state_writer_reset_clears_previous_history(tmp_path: Path):
    path = tmp_path / "state.json"
    writer = NautilusStateWriter(path, max_history=4)
    writer.update(status="RUNNING", history={"ts": 1, "price": 100.0}, event="old")
    writer.reset(status="STARTING", mode="backtest", symbol="BTCUSDT", event="new")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "STARTING"
    assert payload["symbol"] == "BTCUSDT"
    assert payload["history"]["ts"] == []
    assert payload["recent_events"][-1]["message"] == "new"


def test_nautilus_strategy_position_lifecycle_updates_trade_stats():
    strategy = GaricNautilusStrategy.__new__(GaricNautilusStrategy)
    strategy._n_trades = 0
    strategy._n_wins = 0
    strategy._n_losses = 0
    strategy._latest_price = 0.0
    strategy._last_realized_pnl = 0.0
    strategy._seen_event_ids = set()
    events = []
    strategy._publish_runtime_state = lambda **kwargs: events.append(kwargs.get("event"))
    strategy._safe_float = GaricNautilusStrategy._safe_float.__get__(strategy, GaricNautilusStrategy)

    opened = type(
        "PositionOpened",
        (),
        {
            "last_px": 101.5,
            "side": SimpleNamespace(value="LONG"),
            "signed_qty": 0.002,
            "realized_pnl": 0.0,
            "realized_return": 0.0,
        },
    )()
    win_closed = type(
        "PositionClosed",
        (),
        {
            "last_px": 103.0,
            "realized_pnl": 12.5,
            "realized_return": 0.021,
            "side": SimpleNamespace(value="FLAT"),
            "signed_qty": 0.0,
        },
    )()
    loss_closed = type(
        "PositionClosed",
        (),
        {
            "last_px": 99.0,
            "realized_pnl": -3.25,
            "realized_return": -0.007,
            "side": SimpleNamespace(value="FLAT"),
            "signed_qty": 0.0,
        },
    )()

    GaricNautilusStrategy.on_event(strategy, opened)
    GaricNautilusStrategy._sync_trade_counters(
        strategy,
        previous_position=0.0,
        current_position=-1.0,
        realized_pnl_after=-0.5,
    )
    GaricNautilusStrategy.on_event(strategy, win_closed)
    GaricNautilusStrategy._sync_trade_counters(
        strategy,
        previous_position=-1.0,
        current_position=0.0,
        realized_pnl_after=12.5,
    )
    GaricNautilusStrategy._sync_trade_counters(
        strategy,
        previous_position=0.0,
        current_position=1.0,
        realized_pnl_after=12.0,
    )
    GaricNautilusStrategy.on_event(strategy, loss_closed)
    GaricNautilusStrategy._sync_trade_counters(
        strategy,
        previous_position=1.0,
        current_position=0.0,
        realized_pnl_after=9.25,
    )

    assert strategy._n_trades == 2
    assert strategy._n_wins == 1
    assert strategy._n_losses == 1
    assert any("Position opened" in str(msg) for msg in events)
    assert any("Position closed" in str(msg) for msg in events)
