"""Test feature engineering modules."""

import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(n: int = 300) -> pd.DataFrame:
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    open_ = close + np.random.randn(n) * 0.1
    high = np.maximum(close, open_) + abs(np.random.randn(n)) * 0.5
    low = np.minimum(close, open_) - abs(np.random.randn(n)) * 0.5
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": np.random.rand(n) * 1000 + 100,
        "trades": np.random.randint(50, 500, n),
        "taker_buy_volume": np.random.rand(n) * 500 + 50,
    })


class TestTechnicalIndicators:
    def test_compute_all_shape(self):
        from features.technical.indicators import compute_all
        df = _make_ohlcv()
        result = compute_all(df)
        assert result.shape == (300, 15)
        assert result.dtype == np.float32

    def test_rsi_range(self):
        from features.technical.indicators import rsi
        df = _make_ohlcv()
        r = rsi(df["close"]).dropna()
        assert r.min() >= 0
        assert r.max() <= 100

    def test_atr_positive(self):
        from features.technical.indicators import atr
        df = _make_ohlcv()
        a = atr(df["high"], df["low"], df["close"]).dropna()
        assert (a >= 0).all()


class TestMicrostructure:
    def test_compute_all_shape(self):
        from features.technical.microstructure import compute_all
        df = _make_ohlcv()
        result = compute_all(df)
        assert result.shape == (300, 5)
        assert result.dtype == np.float32

    def test_ofi_range(self):
        from features.technical.microstructure import order_flow_imbalance
        buy = pd.Series([100, 200, 150])
        sell = pd.Series([50, 200, 300])
        ofi = order_flow_imbalance(buy, sell)
        assert (ofi >= -1).all()
        assert (ofi <= 1).all()


class TestValidation:
    def test_purged_kfold(self):
        from features.validation import PurgedKFold
        cv = PurgedKFold(n_splits=5, embargo_pct=0.01)
        splits = list(cv.split(1000))
        assert len(splits) == 5
        for train_idx, test_idx in splits:
            # No overlap
            assert len(np.intersect1d(train_idx, test_idx)) == 0

    def test_feature_consistency_same_data(self):
        from features.validation import check_feature_consistency
        data = np.random.randn(500, 10)
        result = check_feature_consistency(data, data)
        assert result["passed"]

    def test_feature_consistency_drift(self):
        from features.validation import check_feature_consistency
        train = np.random.randn(500, 10)
        live = np.random.randn(500, 10) + 5  # shifted
        result = check_feature_consistency(train, live)
        assert not result["passed"]
        assert len(result["drifted_features"]) > 0
