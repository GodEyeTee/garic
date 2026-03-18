"""Tests for pipeline data splitting and aggregation helpers."""

import numpy as np
import pandas as pd

from pipeline import _aggregate_ohlcv_15m, _compute_data_ranges, backtest_with_model


class TestPipelineHelpers:
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
        features = np.zeros((100, 30), dtype=np.float32)
        config = {"trading": {"min_trade_pct": 0.05}, "training": {"validation": {}}}
        ranges = {"train": (0, 60), "validation": (60, 80), "test": (10, 20)}

        result = backtest_with_model(None, features, prices, config, data_ranges=ranges)
        assert result["test_range"] == [10, 20]
