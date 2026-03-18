"""Test data quality pipeline."""

import numpy as np
import pandas as pd
import pytest

from data.quality import remove_outliers, validate_ohlcv, detect_gaps


def _make_ohlcv(n: int = 100) -> pd.DataFrame:
    """สร้าง OHLCV data สำหรับ test."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    open_ = close + np.random.randn(n) * 0.1
    high = np.maximum(close, open_) + abs(np.random.randn(n)) * 0.5
    low = np.minimum(close, open_) - abs(np.random.randn(n)) * 0.5
    return pd.DataFrame({
        "open_time": pd.date_range("2024-01-01", periods=n, freq="1min"),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.rand(n) * 1000,
    })


def test_validate_clean_data():
    df = _make_ohlcv()
    issues = validate_ohlcv(df)
    assert len(issues) == 0


def test_validate_detects_bad_hl():
    df = _make_ohlcv()
    df.loc[5, "high"] = df.loc[5, "low"] - 1  # high < low
    issues = validate_ohlcv(df)
    assert any("high < low" in i for i in issues)


def test_remove_outliers():
    df = _make_ohlcv(200)
    df.loc[50, "close"] = 9999  # spike
    cleaned = remove_outliers(df, zscore_threshold=3.0, window=50)
    assert cleaned.loc[50, "close"] < 9999


def test_detect_gaps():
    df = _make_ohlcv()
    # Remove rows 30-35 to create gap
    df = df.drop(range(30, 36)).reset_index(drop=True)
    gaps = detect_gaps(df)
    assert len(gaps) == 1
