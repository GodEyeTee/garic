"""Base forecast model interface.

ทุก forecast model ต้อง implement interface นี้
เพื่อให้ swap ได้ (TimesFM, CryptoMamba, Chronos, etc.)
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseForecaster(ABC):
    """Abstract interface สำหรับ price forecaster."""

    @abstractmethod
    def predict(
        self,
        price_series: np.ndarray,
        horizon: int = 12,
    ) -> tuple[np.ndarray, float]:
        """Forecast future prices.

        Args:
            price_series: historical close prices, shape (context_len,)
            horizon: number of steps to forecast

        Returns:
            (point_forecast, uncertainty)
            point_forecast: shape (horizon,)
            uncertainty: scalar, quantile spread (Q90-Q10)/Q50
        """
        ...

    @abstractmethod
    def name(self) -> str:
        ...
