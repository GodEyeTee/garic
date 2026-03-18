"""Feature builder — สร้าง StandardFeatureVector จาก raw OHLCV data.

รวม technical indicators + microstructure เข้าด้วยกัน.
ใช้ได้ทั้ง batch (training) และ incremental (live).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data.schema import StandardFeatureVector
from features.technical.indicators import compute_all as compute_ta
from features.technical.microstructure import compute_all as compute_micro

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """สร้าง feature vectors จาก OHLCV data."""

    def __init__(
        self,
        lookback: int = 60,
        return_periods: list[int] | None = None,
    ):
        self.lookback = lookback
        self.return_periods = return_periods or [1, 5, 15, 60, 240]

    def build_batch_array(
        self,
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized batch builder — returns (feature_array, ta_all, micro_all).

        ใช้สำหรับ training pipeline กับ large datasets.
        ไม่สร้าง StandardFeatureVector objects — ทำงานเร็วกว่า build_batch() มาก.

        Returns:
            feature_array: shape (n_out, feature_dim) — ready for model input
            ta_all: shape (n_out, n_ta) — TA indicators aligned with feature_array
            micro_all: shape (n_out, n_micro) — microstructure features aligned
        """
        logger.info("Computing indicators...")
        ta_all = compute_ta(df)        # (n, 15)
        micro_all = compute_micro(df)  # (n, 5)

        # Log returns
        close = df["close"].values.astype(np.float64)
        log_returns = np.log(close / np.roll(close, 1))
        log_returns[0] = 0.0

        ohlcv_data = df[["open", "high", "low", "close", "volume"]].values.astype(np.float32)

        n = len(df)
        lb = self.lookback
        n_out = n - lb

        logger.info(f"Building {n_out} feature vectors (vectorized)...")

        # OHLCV windows via stride tricks: (n_out, lookback, 5)
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(ohlcv_data, lb, axis=0)  # (n-lb+1, 5, lb)
        # windows[j] corresponds to ohlcv_data[j:j+lb] for each column
        # We need windows for positions 0..n_out-1 (i.e. i-lb for i in range(lb, n))
        # Transpose to get (n-lb+1, lb, 5) then flatten to (n_out, lb*5)
        ohlcv_windows = np.moveaxis(windows[:n_out], 1, 2)  # (n_out, lb, 5)
        ohlcv_flat = ohlcv_windows.reshape(n_out, lb * 5).astype(np.float32)

        # Period returns via cumsum
        cum_lr = np.concatenate([[0.0], np.cumsum(log_returns)])
        idx = np.arange(lb, n)
        returns_all = np.zeros((n_out, len(self.return_periods)), dtype=np.float32)
        for j, p in enumerate(self.return_periods):
            prev_idx = np.maximum(0, idx - p)
            returns_all[:, j] = (cum_lr[idx + 1] - cum_lr[prev_idx]).astype(np.float32)

        # TA and micro: slice from lookback onwards
        ta_slice = ta_all[lb:].astype(np.float32)
        micro_slice = micro_all[lb:].astype(np.float32)

        # Zeros for forecast, funding, sentiment, onchain (will be filled later if available)
        zeros = np.zeros((n_out, 1), dtype=np.float32)

        # Concatenate all features (same order as StandardFeatureVector.to_array)
        feature_array = np.concatenate([
            ohlcv_flat,                                         # (n_out, 300)
            returns_all,                                        # (n_out, 5)
            ta_slice,                                           # (n_out, 15)
            micro_slice,                                        # (n_out, 5)
            np.zeros((n_out, 12), dtype=np.float32),            # price_forecast
            zeros,                                              # forecast_uncertainty
            zeros,                                              # funding_rate
            zeros,                                              # open_interest_change
            zeros,                                              # sentiment_score
            np.zeros((n_out, 5), dtype=np.float32),             # onchain_metrics
        ], axis=1)

        logger.info(f"Built feature array: {feature_array.shape}")
        return feature_array, ta_slice, micro_slice

    def build_batch(
        self,
        df: pd.DataFrame,
        funding_df: pd.DataFrame | None = None,
        sentiment_df: pd.DataFrame | None = None,
    ) -> list[StandardFeatureVector]:
        """สร้าง feature vectors สำหรับ DataFrame (live/small batch mode).

        สำหรับ large training datasets ใช้ build_batch_array() แทน.
        Returns list ของ StandardFeatureVector, 1 ต่อ 1 candle close.
        """
        # Pre-compute all indicators
        ta_all = compute_ta(df)        # (n, 15)
        micro_all = compute_micro(df)  # (n, 5)

        # Pre-extract to numpy (avoid pandas in loop)
        df = df.copy()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1)).fillna(0)

        ohlcv_data = df[["open", "high", "low", "close", "volume"]].values
        log_returns = df["log_return"].values
        cum_lr = np.concatenate([[0.0], np.cumsum(log_returns)])

        # Timestamps
        if "open_time" in df.columns:
            ts_values = df["open_time"].values
        else:
            ts_values = None

        symbol = df.attrs.get("symbol", "")
        n = len(df)

        vectors = []
        for i in range(self.lookback, n):
            ohlcv = ohlcv_data[i - self.lookback:i]

            returns = np.array([
                cum_lr[i + 1] - cum_lr[max(0, i - p)]
                for p in self.return_periods
            ], dtype=np.float32)

            # Timestamp
            if ts_values is not None:
                ts = ts_values[i]
                ts_ms = int(pd.Timestamp(ts).timestamp() * 1000) if not isinstance(ts, (int, float)) else int(ts)
            else:
                ts = None
                ts_ms = 0

            # Funding rate lookup
            funding_rate = 0.0
            if funding_df is not None and "fundingTime" in funding_df.columns:
                mask = funding_df["fundingTime"] <= ts
                if mask.any():
                    funding_rate = float(funding_df.loc[mask, "fundingRate"].iloc[-1])

            # Sentiment lookup
            sentiment_score = 0.0
            if sentiment_df is not None and "timestamp" in sentiment_df.columns:
                mask = sentiment_df["timestamp"] <= ts
                if mask.any():
                    raw = float(sentiment_df.loc[mask, "value"].iloc[-1])
                    sentiment_score = (raw - 50) / 50

            vec = StandardFeatureVector(
                ohlcv=ohlcv,
                returns=returns,
                ta_indicators=ta_all[i],
                microstructure=micro_all[i],
                price_forecast=np.zeros(12, dtype=np.float32),
                forecast_uncertainty=0.0,
                funding_rate=funding_rate,
                open_interest_change=0.0,
                sentiment_score=sentiment_score,
                onchain_metrics=np.zeros(5, dtype=np.float32),
                timestamp=ts_ms,
                symbol=symbol,
            )
            vectors.append(vec)

        logger.info(f"Built {len(vectors)} feature vectors (lookback={self.lookback})")
        return vectors

    def vectors_to_array(self, vectors: list[StandardFeatureVector]) -> np.ndarray:
        """แปลง list of vectors เป็น 2D array สำหรับ training."""
        return np.stack([v.to_array() for v in vectors])
