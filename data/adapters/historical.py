"""Historical data adapter — แปลง Parquet/CSV เป็น StandardFeatureVector.

ใช้สำหรับ training และ backtesting.
ทุก row = 1 candle close = 1 decision point.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data.adapters.base import BaseDataAdapter
from data.schema import StandardFeatureVector

logger = logging.getLogger(__name__)


class HistoricalAdapter(BaseDataAdapter):
    """แปลง historical OHLCV data เป็น StandardFeatureVector."""

    def __init__(
        self,
        ohlcv_path: str | Path,
        lookback: int = 60,
        return_periods: list[int] | None = None,
        funding_path: str | Path | None = None,
        sentiment_path: str | Path | None = None,
    ):
        self.lookback = lookback
        self.return_periods = return_periods or [1, 5, 15, 60, 240]

        # Load data
        self.ohlcv = pd.read_parquet(ohlcv_path)
        self._ensure_columns()

        # Optional: merge funding rate
        self.funding = None
        if funding_path and Path(funding_path).exists():
            self.funding = pd.read_parquet(funding_path)

        # Optional: merge sentiment
        self.sentiment = None
        if sentiment_path and Path(sentiment_path).exists():
            self.sentiment = pd.read_parquet(sentiment_path)

        # Pre-compute log returns
        self.ohlcv["log_return"] = np.log(
            self.ohlcv["close"] / self.ohlcv["close"].shift(1)
        )

        logger.info(f"Loaded {len(self.ohlcv)} candles from {ohlcv_path}")

    def _ensure_columns(self):
        required = ["open_time", "open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in self.ohlcv.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    def __len__(self) -> int:
        return len(self.ohlcv) - self.lookback

    def get_feature_vector(self, symbol: str, idx: int) -> StandardFeatureVector:
        """สร้าง feature vector ณ index ที่กำหนด.

        idx คือ offset จาก lookback (0 = row แรกที่มี lookback ครบ)
        """
        actual_idx = idx + self.lookback
        if actual_idx >= len(self.ohlcv):
            raise IndexError(f"Index {actual_idx} out of range")

        row = self.ohlcv.iloc[actual_idx]
        window = self.ohlcv.iloc[actual_idx - self.lookback:actual_idx]

        # OHLCV window
        ohlcv = window[["open", "high", "low", "close", "volume"]].values

        # Returns at multiple periods
        returns = np.array([
            self.ohlcv["log_return"].iloc[max(0, actual_idx - p):actual_idx + 1].sum()
            for p in self.return_periods
        ])

        # Placeholder features (จะเติมใน Phase 2)
        ta_indicators = np.zeros(15, dtype=np.float32)
        microstructure = np.zeros(5, dtype=np.float32)
        price_forecast = np.zeros(12, dtype=np.float32)
        onchain_metrics = np.zeros(5, dtype=np.float32)

        # Funding rate lookup
        funding_rate = 0.0
        if self.funding is not None:
            ts = row["open_time"]
            mask = self.funding["fundingTime"] <= ts
            if mask.any():
                funding_rate = float(self.funding.loc[mask, "fundingRate"].iloc[-1])

        # Sentiment lookup
        sentiment_score = 0.0
        if self.sentiment is not None:
            ts = row["open_time"]
            mask = self.sentiment["timestamp"] <= ts
            if mask.any():
                raw_value = float(self.sentiment.loc[mask, "value"].iloc[-1])
                sentiment_score = (raw_value - 50) / 50  # normalize to -1..+1

        return StandardFeatureVector(
            ohlcv=ohlcv,
            returns=returns,
            ta_indicators=ta_indicators,
            microstructure=microstructure,
            price_forecast=price_forecast,
            forecast_uncertainty=0.0,
            funding_rate=funding_rate,
            open_interest_change=0.0,
            sentiment_score=sentiment_score,
            onchain_metrics=onchain_metrics,
            timestamp=int(row["open_time"].timestamp() * 1000)
                      if hasattr(row["open_time"], "timestamp")
                      else int(row["open_time"]),
            symbol=symbol,
        )

    def is_candle_closed(self) -> bool:
        """Historical data: ทุก row คือ candle ที่ปิดแล้ว."""
        return True
