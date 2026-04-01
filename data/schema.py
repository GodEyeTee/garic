"""Shared data schemas for GARIC."""

from dataclasses import dataclass

import numpy as np


@dataclass
class OHLCV:
    """Single candle data."""

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str = ""
    timeframe: str = ""


@dataclass
class StandardFeatureVector:
    """Compatibility container for train/live features.

    The model-facing layout is intentionally compact:
    - multi-period returns
    - TA indicators
    - microstructure
    - optional compact forecast block
    """

    ohlcv: np.ndarray
    returns: np.ndarray
    ta_indicators: np.ndarray
    microstructure: np.ndarray
    price_forecast: np.ndarray
    forecast_uncertainty: float
    funding_rate: float
    open_interest_change: float
    sentiment_score: float
    onchain_metrics: np.ndarray
    timestamp: int = 0
    symbol: str = ""

    def compact_forecast_block(self) -> np.ndarray:
        """Return the 5-dim forecast block expected by the optional 15m Mamba path."""
        block = np.zeros(5, dtype=np.float32)
        forecast = np.asarray(self.price_forecast, dtype=np.float32).reshape(-1)
        if forecast.size == 0 or not np.isfinite(forecast).any() or np.allclose(forecast, 0.0):
            block[4] = float(self.forecast_uncertainty)
            return block

        ohlcv = np.asarray(self.ohlcv, dtype=np.float32)
        last_close = float(ohlcv[-1, 3]) if ohlcv.ndim == 2 and ohlcv.shape[0] > 0 else 0.0
        usable = min(4, forecast.size)
        if last_close > 0:
            block[:usable] = (forecast[:usable] / max(last_close, 1e-9)) - 1.0
        else:
            block[:usable] = forecast[:usable]
        block[4] = float(self.forecast_uncertainty)
        return block

    def to_array(self, include_forecast: bool = False) -> np.ndarray:
        """Flatten to the compact layout used by GARIC models."""
        parts = [
            np.asarray(self.returns, dtype=np.float32).reshape(-1),
            np.asarray(self.ta_indicators, dtype=np.float32).reshape(-1),
            np.asarray(self.microstructure, dtype=np.float32).reshape(-1),
        ]
        if include_forecast:
            parts.append(self.compact_forecast_block())
        return np.concatenate(parts).astype(np.float32)

    @staticmethod
    def feature_dim(
        n_return_periods: int = 5,
        n_ta: int = 15,
        n_micro: int = 5,
        include_forecast: bool = False,
        forecast_dims: int = 5,
    ) -> int:
        """Return the compact feature dimension used by model-facing arrays."""
        base = int(n_return_periods) + int(n_ta) + int(n_micro)
        return base + (int(forecast_dims) if include_forecast else 0)
