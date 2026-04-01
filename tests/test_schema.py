"""Test StandardFeatureVector consistency."""

import numpy as np
import pytest

from data.schema import StandardFeatureVector


def _make_dummy_vector(**overrides) -> StandardFeatureVector:
    defaults = dict(
        ohlcv=np.random.rand(60, 5),
        returns=np.random.randn(5),
        ta_indicators=np.zeros(15),
        microstructure=np.zeros(5),
        price_forecast=np.zeros(12),
        forecast_uncertainty=0.0,
        funding_rate=0.0001,
        open_interest_change=0.0,
        sentiment_score=0.5,
        onchain_metrics=np.zeros(5),
        timestamp=1700000000000,
        symbol="BTCUSDT",
    )
    defaults.update(overrides)
    return StandardFeatureVector(**defaults)


def test_to_array_shape():
    vec = _make_dummy_vector()
    arr = vec.to_array()
    expected_dim = StandardFeatureVector.feature_dim()
    assert arr.shape == (expected_dim,)
    assert arr.dtype == np.float32
    assert expected_dim == 25


def test_to_array_deterministic():
    """Same input → same output."""
    vec1 = _make_dummy_vector(
        ohlcv=np.ones((60, 5)),
        returns=np.zeros(5),
    )
    vec2 = _make_dummy_vector(
        ohlcv=np.ones((60, 5)),
        returns=np.zeros(5),
    )
    np.testing.assert_array_equal(vec1.to_array(), vec2.to_array())


def test_feature_dim_matches():
    dim = StandardFeatureVector.feature_dim(n_return_periods=5, n_ta=15, n_micro=5)
    assert dim == 25


def test_feature_dim_with_forecast_matches():
    dim = StandardFeatureVector.feature_dim(
        n_return_periods=5,
        n_ta=15,
        n_micro=5,
        include_forecast=True,
        forecast_dims=5,
    )
    assert dim == 30


def test_zero_forecast_block_stays_zero_when_enabled():
    vec = _make_dummy_vector(
        ohlcv=np.full((60, 5), 100.0),
        price_forecast=np.zeros(12, dtype=np.float32),
        forecast_uncertainty=0.0,
    )
    arr = vec.to_array(include_forecast=True)
    assert arr.shape == (30,)
    np.testing.assert_array_equal(arr[-5:], np.zeros(5, dtype=np.float32))
