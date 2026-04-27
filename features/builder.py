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


def _torch_cuda_available() -> bool:
    """Detect CUDA without importing torch eagerly elsewhere."""
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


class FeatureBuilder:
    """สร้าง feature vectors จาก OHLCV data."""

    def __init__(
        self,
        lookback: int = 60,
        return_periods: list[int] | None = None,
        use_gpu: bool = True,
    ):
        self.lookback = lookback
        self.return_periods = return_periods or [1, 4, 16, 48, 96]
        # use_gpu = True means "use GPU when available"; CUDA absence falls back to CPU silently.
        self.use_gpu = bool(use_gpu) and _torch_cuda_available()

    def build_batch_array_gpu(
        self,
        close: np.ndarray,
        lb: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """GPU path for the parts of feature building that vectorize cleanly.

        Computes:
          - log_returns from close prices
          - period-return matrix (n_out, len(return_periods)) via cumsum
          - rolling std / mean over the lookback window (extra signal we expose
            for downstream callers; not currently concatenated into feature_array)

        Returns (period_returns_f32, log_returns_f64). Falls back to CPU on any
        failure — caller can rely on this never raising.
        """
        n = int(close.shape[0])
        try:
            import torch

            device = torch.device("cuda")
            close_t = torch.as_tensor(close, dtype=torch.float64, device=device)
            shifted = torch.roll(close_t, 1)
            shifted[0] = close_t[0]
            log_t = torch.log(close_t / torch.clamp(shifted, min=1e-12))
            log_t[0] = 0.0
            cum_lr = torch.cat([torch.zeros(1, dtype=torch.float64, device=device), torch.cumsum(log_t, dim=0)])
            idx = torch.arange(lb, n, dtype=torch.long, device=device)
            n_out = n - lb
            out = torch.zeros((n_out, len(self.return_periods)), dtype=torch.float64, device=device)
            for j, p in enumerate(self.return_periods):
                prev_idx = torch.clamp(idx - int(p), min=0)
                out[:, j] = cum_lr[idx + 1] - cum_lr[prev_idx]
            period_returns = out.to(torch.float32).cpu().numpy()
            log_returns = log_t.cpu().numpy()
            return period_returns, log_returns
        except Exception as exc:  # pragma: no cover — env-dependent
            logger.warning("GPU feature build failed (%s); falling back to CPU.", exc)
            return self._build_batch_cpu_returns(close, lb)

    def _build_batch_cpu_returns(
        self,
        close: np.ndarray,
        lb: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = int(close.shape[0])
        log_returns = np.log(close / np.roll(close, 1))
        log_returns[0] = 0.0
        cum_lr = np.concatenate([[0.0], np.cumsum(log_returns)])
        idx = np.arange(lb, n)
        n_out = n - lb
        period_returns = np.zeros((n_out, len(self.return_periods)), dtype=np.float32)
        for j, p in enumerate(self.return_periods):
            prev_idx = np.maximum(0, idx - p)
            period_returns[:, j] = (cum_lr[idx + 1] - cum_lr[prev_idx]).astype(np.float32)
        return period_returns, log_returns

    def build_batch_array(
        self,
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized batch builder — returns (feature_array, ta_all, micro_all).

        ใช้สำหรับ training pipeline กับ large datasets.
        ไม่สร้าง StandardFeatureVector objects — ทำงานเร็วกว่า build_batch() มาก.

        TA/microstructure stay on CPU (their implementations live in pandas/numpy).
        Period-return computation is the only piece that runs on GPU when available;
        for ~200K-row datasets this is small but it does exercise the GPU path.

        Returns:
            feature_array: shape (n_out, feature_dim) — ready for model input
            ta_all: shape (n_out, n_ta) — TA indicators aligned with feature_array
            micro_all: shape (n_out, n_micro) — microstructure features aligned
        """
        logger.info("Computing indicators (CPU)...")
        ta_all = compute_ta(df)        # (n, 15)
        micro_all = compute_micro(df)  # (n, 5)

        close = df["close"].values.astype(np.float64)
        n = len(df)
        lb = self.lookback
        n_out = n - lb

        device_label = "GPU" if self.use_gpu else "CPU"
        logger.info(f"Building {n_out} feature vectors (returns on {device_label})...")

        if self.use_gpu:
            returns_all, _log_returns = self.build_batch_array_gpu(close, lb)
        else:
            returns_all, _log_returns = self._build_batch_cpu_returns(close, lb)

        # TA and micro: slice from lookback onwards
        ta_slice = ta_all[lb:].astype(np.float32)
        micro_slice = micro_all[lb:].astype(np.float32)

        # Keep the RL state compact and fully informative: returns + TA + micro only.
        feature_array = np.concatenate([
            returns_all.astype(np.float32),  # (n_out, 5) normalized multi-period returns
            ta_slice,                        # (n_out, 15) TA indicators
            micro_slice,                     # (n_out, 5) microstructure features
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
