"""TimesFM 2.5 zero-shot forecaster — Google foundation model.

200M params, decoder-only transformer, patch-based tokenization.
Ranked #1 zero-shot time series forecasting (ICML 2024).

RTX 2060 6GB: พอสำหรับ inference (200M params ≈ 0.8GB fp32)
ถ้า timesfm ไม่ได้ติดตั้ง → fallback เป็น NaiveForecaster

Usage:
    pip install timesfm[torch]
    forecaster = TimesFMForecaster(device="cuda")
    forecast, uncertainty = forecaster.predict(prices, horizon=12)
"""

import logging

import numpy as np

from models.forecast.base import BaseForecaster

logger = logging.getLogger(__name__)


class TimesFMForecaster(BaseForecaster):
    """TimesFM 2.5 200M zero-shot forecaster.

    ไม่ต้อง fine-tune — ใช้ pretrained weights เลย.
    Context window สูงสุด 1024 candles, horizon สูงสุด 256.
    """

    def __init__(
        self,
        device: str = "cuda",
        max_context: int = 512,
        max_horizon: int = 128,
    ):
        self._model = None
        self._config = None
        self._device = device
        self._max_context = max_context
        self._max_horizon = max_horizon
        self._available = False

        self._try_load()

    def _try_load(self):
        try:
            import timesfm

            self._model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                "google/timesfm-2.5-200m-pytorch",
            )
            self._config = timesfm.ForecastConfig(
                max_context=self._max_context,
                max_horizon=self._max_horizon,
            )
            self._model.compile(self._config)
            self._available = True
            logger.info(f"TimesFM 2.5 loaded (device={self._device})")

        except ImportError:
            logger.warning(
                "timesfm not installed — run: pip install timesfm[torch]  "
                "Falling back to NaiveForecaster"
            )
        except Exception as e:
            logger.warning(f"TimesFM load failed: {e} — falling back to NaiveForecaster")

    @property
    def available(self) -> bool:
        return self._available

    def predict(
        self,
        price_series: np.ndarray,
        horizon: int = 12,
    ) -> tuple[np.ndarray, float]:
        """Zero-shot forecast.

        Returns (point_forecast, uncertainty).
        point_forecast: shape (horizon,)
        uncertainty: quantile spread (Q90-Q10)/Q50
        """
        if not self._available:
            return self._fallback_predict(price_series, horizon)

        # Clamp context to max
        context = price_series[-self._max_context:].astype(np.float32)
        h = min(horizon, self._max_horizon)

        try:
            point_forecast, quantile_forecast = self._model.forecast(
                horizon=h,
                inputs=[context],
            )
            # point_forecast: list of arrays, take first
            pf = np.array(point_forecast[0][:horizon], dtype=np.float32)

            # Compute uncertainty from quantiles if available
            if quantile_forecast is not None and len(quantile_forecast) > 0:
                qf = np.array(quantile_forecast[0])  # shape: (horizon, n_quantiles)
                if qf.ndim == 2 and qf.shape[1] >= 2:
                    q_high = qf[:, -1]  # ~Q90
                    q_low = qf[:, 0]    # ~Q10
                    q_mid = np.abs(pf)
                    q_mid = np.where(q_mid < 1e-8, 1.0, q_mid)
                    spreads = (q_high - q_low) / q_mid
                    uncertainty = float(np.mean(spreads))
                else:
                    uncertainty = 0.1
            else:
                uncertainty = 0.1

            # Pad if horizon > model output
            if len(pf) < horizon:
                pad = np.full(horizon - len(pf), pf[-1], dtype=np.float32)
                pf = np.concatenate([pf, pad])

            return pf, uncertainty

        except Exception as e:
            logger.warning(f"TimesFM predict failed: {e}")
            return self._fallback_predict(price_series, horizon)

    def _fallback_predict(
        self,
        price_series: np.ndarray,
        horizon: int,
    ) -> tuple[np.ndarray, float]:
        """Fallback: naive drift forecast."""
        from models.forecast.naive import NaiveForecaster
        return NaiveForecaster().predict(price_series, horizon)

    def name(self) -> str:
        if self._available:
            return "TimesFM-2.5-200M"
        return "TimesFM-2.5-200M (fallback:naive)"
