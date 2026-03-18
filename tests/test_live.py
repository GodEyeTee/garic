"""Test live trading components."""

import numpy as np
import pytest


class TestDriftDetector:
    def test_no_drift_on_same_data(self):
        from monitoring.live.drift_detector import DriftDetector
        ref = np.random.randn(500, 10)
        dd = DriftDetector(reference_features=ref, window_size=200, check_interval=1)
        for i in range(200):
            dd.update(ref[i % 500], action=0.0, pnl_return=0.001)
        dd._step_count = 1  # force check
        result = dd.check_all()
        assert not result["should_pause"]

    def test_drift_on_shifted_data(self):
        from monitoring.live.drift_detector import DriftDetector
        ref = np.random.randn(500, 10)
        dd = DriftDetector(reference_features=ref, window_size=200, check_interval=1)
        for i in range(200):
            dd.update(ref[i % 500] + 10, action=0.0, pnl_return=-0.01)
        dd._step_count = 1
        result = dd.check_all()
        assert result["feature_drift"]


class TestOrderManager:
    def test_paper_order(self):
        from execution.live.order_manager import OrderManager
        om = OrderManager(paper_mode=True)
        result = om.place_order("BTCUSDT", "buy", 0.01, 50000.0)
        assert result.status == "filled"
        assert result.is_paper
        assert result.amount == 0.01

    def test_paper_balance_updates(self):
        from execution.live.order_manager import OrderManager
        om = OrderManager(paper_mode=True)
        initial = om.get_balance()
        om.place_order("BTCUSDT", "buy", 0.01, 50000.0)
        assert om.get_balance() < initial  # spent money


class TestDashboardData:
    def test_add_and_metrics(self):
        from monitoring.live.dashboard import DashboardData
        dd = DashboardData()
        for i in range(100):
            dd.add_record(
                timestamp=i * 60000,
                balance=10000 + i * 10,
                position=0.5,
                pnl=10.0,
                action=0.5,
                confidence=0.8,
            )
        metrics = dd.get_metrics()
        assert metrics["total_return_pct"] > 0
        assert metrics["n_records"] == 100


class TestLiveAdapter:
    def test_candle_aggregator_returns_closed_candle(self):
        from data.adapters.live import CandleAggregator

        agg = CandleAggregator(timeframe_seconds=60)
        agg.on_trade(100.0, 1.0, 0)
        agg.on_trade(101.0, 2.0, 30_000)
        agg.on_trade(102.0, 3.0, 60_000)

        assert agg.candle_closed
        candle = agg.consume_closed_candle()
        assert candle is not None
        assert candle.timestamp == 0
        assert candle.open == 100.0
        assert candle.high == 101.0
        assert candle.close == 101.0


class TestTradingEngine:
    def test_engine_init_paper(self):
        from execution.live.trading_engine import TradingEngine
        engine = TradingEngine(symbol="BTCUSDT", paper_mode=True)
        assert engine.paper_mode
        assert engine._last_action == 0.0

    def test_estimate_equity_uses_balance_and_position(self):
        from execution.live.trading_engine import TradingEngine

        engine = TradingEngine(symbol="BTCUSDT", paper_mode=True)
        engine.order_manager._paper_balance = 500.0
        engine.order_manager._paper_position = 2.0
        assert engine._estimate_equity(100.0) == 700.0

    def test_warmup_no_crash(self):
        from execution.live.trading_engine import TradingEngine
        engine = TradingEngine(symbol="BTCUSDT", paper_mode=True, timeframe_seconds=60)
        # Feed some trades — should not crash during warmup
        for i in range(10):
            engine.on_trade(50000.0 + i, 0.1, i * 1000)
