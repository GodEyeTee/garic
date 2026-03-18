"""Test CandleAggregator — ต้อง action เฉพาะ candle close."""

import pytest
from data.adapters.live import CandleAggregator


def test_no_action_during_candle():
    """ระหว่างแท่งเทียน ต้องไม่ signal action."""
    agg = CandleAggregator(timeframe_seconds=60)  # 1 min candle

    # Tick ที่ t=0s, 10s, 30s → ยังอยู่ในแท่งเดียวกัน
    agg.on_trade(100.0, 1.0, 0)
    assert not agg.candle_closed

    agg.on_trade(101.0, 0.5, 10_000)
    assert not agg.candle_closed

    agg.on_trade(99.0, 2.0, 30_000)
    assert not agg.candle_closed


def test_action_on_candle_close():
    """เมื่อแท่งเทียนปิด ต้อง signal action."""
    agg = CandleAggregator(timeframe_seconds=60)

    # Tick ที่ t=0s → candle 1
    agg.on_trade(100.0, 1.0, 0)
    assert not agg.candle_closed

    # Tick ที่ t=30s → ยังอยู่ใน candle 1
    agg.on_trade(105.0, 2.0, 30_000)
    assert not agg.candle_closed

    # Tick ที่ t=60s → candle 1 ปิด, candle 2 เริ่ม
    agg.on_trade(103.0, 1.5, 60_000)
    assert agg.candle_closed


def test_consume_resets_flag():
    """หลัง consume แล้ว flag ต้อง reset."""
    agg = CandleAggregator(timeframe_seconds=60)

    agg.on_trade(100.0, 1.0, 0)
    agg.on_trade(103.0, 1.5, 60_000)
    assert agg.candle_closed

    agg.consume_closed_candle()
    assert not agg.candle_closed


def test_orderbook_accumulates():
    """Orderbook data ต้อง accumulate ไม่ใช่ trigger action."""
    agg = CandleAggregator(timeframe_seconds=60)

    agg.on_trade(100.0, 1.0, 0)
    agg.on_orderbook(99.9, 100.1)
    agg.on_orderbook(99.8, 100.2)
    assert not agg.candle_closed

    features = agg.get_accumulated_features()
    # 100.1-99.9=0.2, 100.2-99.8=0.4 → mean=0.3
    assert features["avg_spread"] == pytest.approx(0.3)
