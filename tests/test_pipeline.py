"""Tests for pipeline data splitting and aggregation helpers."""

import numpy as np
import pandas as pd

from features.builder import FeatureBuilder
from pipeline import (
    _aggregate_ohlcv_15m,
    _build_nautilus_frame,
    _compute_data_ranges,
    _run_nautilus_backtest_segment,
    _score_nautilus_summary,
    add_naive_forecast,
    backtest_with_model,
)


class TestPipelineHelpers:
    def test_feature_builder_batch_array_is_compact(self):
        df = pd.DataFrame(
            {
                "open": np.linspace(100.0, 130.0, 120),
                "high": np.linspace(101.0, 131.0, 120),
                "low": np.linspace(99.0, 129.0, 120),
                "close": np.linspace(100.5, 130.5, 120),
                "volume": np.ones(120),
            }
        )
        feature_array, ta_slice, micro_slice = FeatureBuilder(lookback=60).build_batch_array(df)
        assert feature_array.shape == (60, 25)
        assert ta_slice.shape[1] == 15
        assert micro_slice.shape[1] == 5

    def test_add_naive_forecast_appends_compact_block_without_overwriting(self):
        base = np.ones((80, 25), dtype=np.float32)
        prices = np.linspace(100.0, 120.0, 80, dtype=np.float32)
        enriched = add_naive_forecast(base, prices)
        assert enriched.shape == (80, 30)
        np.testing.assert_array_equal(enriched[:, :25], base)

    def test_compute_data_ranges_reserves_holdout_tail(self):
        ranges = _compute_data_ranges(100, test_ratio=0.2, validation_ratio_within_train=0.1)
        assert ranges["train"] == (0, 72)
        assert ranges["validation"] == (72, 80)
        assert ranges["test"] == (80, 100)

    def test_aggregate_ohlcv_15m_vectorized_values(self):
        df = pd.DataFrame(
            {
                "open_time": pd.date_range("2026-01-01", periods=30, freq="min"),
                "open": list(range(1, 31)),
                "high": list(range(2, 32)),
                "low": list(range(0, 30)),
                "close": list(range(1, 31)),
                "volume": [1.0] * 30,
            }
        )

        agg = _aggregate_ohlcv_15m(df, period=15)
        assert len(agg) == 2
        assert agg.iloc[0]["open"] == 1
        assert agg.iloc[0]["high"] == 16
        assert agg.iloc[0]["low"] == 0
        assert agg.iloc[0]["close"] == 15
        assert agg.iloc[0]["volume"] == 15.0
        assert agg.iloc[1]["open"] == 16
        assert agg.iloc[1]["close"] == 30

    def test_backtest_uses_caller_ranges_when_provided(self):
        prices = np.linspace(100.0, 200.0, 100, dtype=np.float32)
        features = np.zeros((100, 25), dtype=np.float32)
        config = {"trading": {"min_trade_pct": 0.05}, "training": {"validation": {}}}
        ranges = {"train": (0, 60), "validation": (60, 80), "test": (10, 20)}

        result = backtest_with_model(None, features, prices, config, data_ranges=ranges)
        assert result["test_range"] == [10, 20]

    def test_build_nautilus_frame_from_ohlcv_arrays(self):
        prices = np.array([100.0, 101.0, 102.0], dtype=np.float32)
        ohlcv = np.array(
            [
                [99.5, 100.5, 99.0, 100.0],
                [100.0, 101.5, 99.8, 101.0],
                [101.0, 102.5, 100.5, 102.0],
            ],
            dtype=np.float32,
        )
        frame = _build_nautilus_frame(prices, ohlcv)
        assert list(frame.columns) == ["open_time", "open", "high", "low", "close", "volume"]
        assert len(frame) == 3
        assert frame["close"].tolist() == [100.0, 101.0, 102.0]
        assert frame["volume"].tolist() == [1.0, 1.0, 1.0]

    def test_score_nautilus_summary_rejects_no_trade_collapse(self):
        score = _score_nautilus_summary(
            {
                "n_trades": 0,
                "total_return": -0.0069,
                "outperformance_vs_bh": -0.02,
                "win_rate": 0.0,
                "flat_ratio": 1.0,
                "eval_dominant_action_ratio": 1.0,
                "eval_action_entropy": 0.0,
                "max_drawdown": -0.0069,
            }
        )
        assert score == float("-inf")

    def test_run_nautilus_backtest_segment_passes_aligned_slice(self, monkeypatch):
        captured = {}

        def fake_run_backtest_frame(frame_15m, **kwargs):
            captured["frame_len"] = len(frame_15m)
            captured["first_close"] = float(frame_15m["close"].iloc[0])
            captured["kwargs"] = kwargs
            return {
                "n_trades": 3,
                "total_return": 0.02,
                "gross_total_return": 0.025,
                "outperformance_vs_bh": 0.01,
                "win_rate": 0.66,
                "flat_ratio": 0.2,
                "eval_dominant_action_ratio": 0.5,
                "eval_action_entropy": 0.8,
                "max_drawdown": -0.04,
            }

        monkeypatch.setattr("execution.nautilus.backtest_runner.run_backtest_frame", fake_run_backtest_frame)

        frame = pd.DataFrame(
            {
                "open_time": pd.date_range("2026-01-01", periods=200, freq="15min", tz="UTC"),
                "open": np.linspace(100.0, 150.0, 200),
                "high": np.linspace(101.0, 151.0, 200),
                "low": np.linspace(99.0, 149.0, 200),
                "close": np.linspace(100.5, 150.5, 200),
                "volume": np.ones(200),
            }
        )
        config = {
            "_active_symbol": "BTCUSDT",
            "trading": {"monthly_server_cost_usd": 100.0, "periods_per_day": 96, "leverage": 1.0},
            "training": {
                "nautilus_validation": {
                    "history_bars": 160,
                    "request_history_days": 3,
                    "trade_size_pct_of_equity": 1.0,
                    "initial_balance_usdt": 10000.0,
                    "leverage": 1.0,
                }
            },
            "data": {"pairs": ["BTCUSDT"]},
        }

        summary = _run_nautilus_backtest_segment(
            model_path="checkpoints/rl_agent_supervised.joblib",
            frame_15m=frame,
            segment_range=(20, 180),
            config=config,
            label="validation_supervised_logreg",
        )

        assert captured["frame_len"] == 160
        assert captured["first_close"] == float(frame["close"].iloc[20])
        assert captured["kwargs"]["symbol"] == "BTCUSDT"
        assert float(captured["kwargs"]["trade_size"]) > 0.0
        assert summary["segment_range"] == [20, 180]
