"""Naive forecaster — baseline สำหรับเปรียบเทียบ.

ใช้ last value + drift เป็น forecast.
ทุก model ต้องชนะ naive ก่อนจึงจะใช้ได้.
"""

import numpy as np

from models.forecast.base import BaseForecaster


class NaiveForecaster(BaseForecaster):
    """Last value forecast with optional drift."""

    def predict(self, price_series: np.ndarray, horizon: int = 12) -> tuple[np.ndarray, float]:
        last_price = price_series[-1]

        # Simple drift = average recent return
        if len(price_series) > 1:
            returns = np.diff(np.log(price_series[-min(60, len(price_series)):]))
            avg_return = returns.mean()
            std_return = returns.std() if len(returns) > 1 else 0.01
        else:
            avg_return = 0.0
            std_return = 0.01

        # Forecast with drift
        steps = np.arange(1, horizon + 1)
        forecast = last_price * np.exp(avg_return * steps)

        # Uncertainty: rough estimate from return volatility
        uncertainty = std_return * np.sqrt(horizon)

        return forecast.astype(np.float32), float(uncertainty)

    def name(self) -> str:
        return "naive"
