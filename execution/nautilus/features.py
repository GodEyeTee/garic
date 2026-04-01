"""Feature extraction for Nautilus bar-driven inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from features.technical.indicators import compute_all as compute_ta
from features.technical.microstructure import compute_all as compute_micro
from models.forecast.naive import NaiveForecaster


@dataclass
class FeatureSnapshot:
    feature_array: np.ndarray
    latest_price: float
    ta: np.ndarray
    micro: np.ndarray
    returns: np.ndarray
    forecast: np.ndarray
    vol_20: float


class NautilusFeatureBuilder:
    """Build the compact GARIC feature vector from 15m bars."""

    def __init__(self, history_bars: int = 160, include_forecast: bool = False):
        self.history_bars = max(int(history_bars), 128)
        self.return_periods = (1, 4, 16, 48, 96)
        self.forecaster = NaiveForecaster()
        self.include_forecast = bool(include_forecast)

    @property
    def warmup_bars(self) -> int:
        return max(self.history_bars, 128)

    def build_latest(self, df: pd.DataFrame) -> FeatureSnapshot:
        frame = df.tail(self.history_bars).copy()
        if len(frame) < self.warmup_bars:
            raise ValueError(f"Need at least {self.warmup_bars} bars, got {len(frame)}")

        frame["open"] = frame["open"].astype(float)
        frame["high"] = frame["high"].astype(float)
        frame["low"] = frame["low"].astype(float)
        frame["close"] = frame["close"].astype(float)
        frame["volume"] = frame["volume"].astype(float)

        ta = compute_ta(frame).astype(np.float32)
        micro = compute_micro(frame).astype(np.float32)
        closes = frame["close"].to_numpy(dtype=np.float64)
        latest_price = float(closes[-1])

        safe_closes = np.maximum(closes, 1e-9)
        log_returns = np.diff(np.log(safe_closes), prepend=np.log(safe_closes[0]))
        returns = np.array(
            [float(np.sum(log_returns[-period:])) for period in self.return_periods],
            dtype=np.float32,
        )

        forecast = np.zeros(5, dtype=np.float32)
        if self.include_forecast:
            preds, uncertainty = self.forecaster.predict(safe_closes[-60:], horizon=4)
            forecast[:4] = ((preds[:4] / max(latest_price, 1e-9)) - 1.0).astype(np.float32)
            forecast[4] = float(uncertainty)

        feature_parts = [
            returns,
            ta[-1],
            micro[-1],
        ]
        if self.include_forecast:
            feature_parts.append(forecast)
        feature_array = np.concatenate(feature_parts).astype(np.float32)
        vol_20 = float(pd.Series(log_returns).rolling(20).std().fillna(0.0).iloc[-1])

        return FeatureSnapshot(
            feature_array=feature_array,
            latest_price=latest_price,
            ta=ta[-1],
            micro=micro[-1],
            returns=returns,
            forecast=forecast,
            vol_20=vol_20,
        )
