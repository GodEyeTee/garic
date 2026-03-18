"""Shared data schemas — StandardFeatureVector and OHLCV types.

ทุก component ใช้ schema เดียวกัน ทั้ง training และ live.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class OHLCV:
    """Single candle data."""
    timestamp: int          # Unix ms
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str = ""
    timeframe: str = ""


@dataclass
class StandardFeatureVector:
    """Format เดียวสำหรับทั้ง training และ live inference.

    *** train กับ live ต้องได้ vector นี้เหมือนกัน 100% ***
    ถ้าไม่เหมือนกัน โมเดลจะทำงานผิดพลาด
    """
    # === Price features ===
    ohlcv: np.ndarray               # shape: (lookback, 5) — O,H,L,C,V
    returns: np.ndarray             # shape: (n_return_periods,) — log returns

    # === Technical features ===
    ta_indicators: np.ndarray       # shape: (n_ta,) — RSI, MACD, BB, ATR, etc.
    microstructure: np.ndarray      # shape: (n_micro,) — OFI, spread, imbalance

    # === Forecast features ===
    price_forecast: np.ndarray      # shape: (horizon,) — predicted prices
    forecast_uncertainty: float     # quantile spread (Q90-Q10)/Q50

    # === External features ===
    funding_rate: float
    open_interest_change: float
    sentiment_score: float          # -1 to +1
    onchain_metrics: np.ndarray     # shape: (n_onchain,) — TVL change, whale flow

    # === Meta (ไม่เข้าโมเดล) ===
    timestamp: int = 0
    symbol: str = ""

    def to_array(self) -> np.ndarray:
        """Flatten ทุก feature เป็น 1D array สำหรับ model input."""
        parts = [
            self.ohlcv.flatten(),
            self.returns,
            self.ta_indicators,
            self.microstructure,
            self.price_forecast,
            np.array([self.forecast_uncertainty]),
            np.array([self.funding_rate]),
            np.array([self.open_interest_change]),
            np.array([self.sentiment_score]),
            self.onchain_metrics,
        ]
        return np.concatenate(parts).astype(np.float32)

    @staticmethod
    def feature_dim(
        lookback: int = 60,
        n_return_periods: int = 5,
        n_ta: int = 15,
        n_micro: int = 5,
        horizon: int = 12,
        n_onchain: int = 5,
    ) -> int:
        """คำนวณ total feature dimension."""
        return (
            lookback * 5        # ohlcv
            + n_return_periods  # returns
            + n_ta              # ta_indicators
            + n_micro           # microstructure
            + horizon           # price_forecast
            + 1                 # forecast_uncertainty
            + 1                 # funding_rate
            + 1                 # open_interest_change
            + 1                 # sentiment_score
            + n_onchain         # onchain_metrics
        )
