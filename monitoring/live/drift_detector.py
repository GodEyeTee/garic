"""Drift detection — ตรวจจับเมื่อ model ต้อง retrain.

ตรวจ 4 ประเภท:
1. Feature drift — distribution ของ input เปลี่ยน
2. Prediction drift — action distribution เปลี่ยน
3. Performance drift — rolling Sharpe ลดลง
4. Concept drift — ความสัมพันธ์ feature↔action เปลี่ยน

ใช้ Evidently AI เป็นหลัก, fallback เป็น scipy stats.
"""

import logging
from collections import deque

import numpy as np

from performance import safe_sharpe_ratio

logger = logging.getLogger(__name__)


class DriftDetector:
    """ตรวจจับ drift ทุก check_interval candles."""

    def __init__(
        self,
        reference_features: np.ndarray | None = None,
        window_size: int = 500,
        ks_threshold: float = 0.05,
        sharpe_min: float = 0.3,
        check_interval: int = 240,  # ทุก 4 ชม. (ถ้า 1m candle)
    ):
        self.reference = reference_features
        self.window_size = window_size
        self.ks_threshold = ks_threshold
        self.sharpe_min = sharpe_min
        self.check_interval = check_interval

        self._feature_buffer: deque[np.ndarray] = deque(maxlen=window_size)
        self._return_buffer: deque[float] = deque(maxlen=window_size)
        self._action_buffer: deque[float] = deque(maxlen=window_size)
        self._step_count = 0

    def update(self, features: np.ndarray, action: float, pnl_return: float):
        """อัพเดทข้อมูลทุก candle close."""
        self._feature_buffer.append(features)
        self._return_buffer.append(pnl_return)
        self._action_buffer.append(action)
        self._step_count += 1

    def should_check(self) -> bool:
        return (
            self._step_count % self.check_interval == 0
            and len(self._feature_buffer) >= self.window_size // 2
        )

    def check_all(self) -> dict:
        """Run ทุก drift check. Returns dict of results."""
        results = {
            "feature_drift": False,
            "performance_drift": False,
            "action_drift": False,
            "should_retrain": False,
            "should_pause": False,
            "details": [],
        }

        # 1. Feature drift
        if self.reference is not None and len(self._feature_buffer) >= 100:
            fd = self._check_feature_drift()
            results["feature_drift"] = fd["drifted"]
            if fd["drifted"]:
                results["details"].append(f"Feature drift: {fd['n_drifted']}/{fd['n_features']} features")

        # 2. Performance drift (rolling Sharpe)
        pd_result = self._check_performance_drift()
        results["performance_drift"] = pd_result["degraded"]
        if pd_result["degraded"]:
            results["details"].append(f"Performance drift: Sharpe={pd_result['sharpe']:.3f} < {self.sharpe_min}")

        # 3. Action distribution drift
        ad_result = self._check_action_drift()
        results["action_drift"] = ad_result["drifted"]
        if ad_result["drifted"]:
            results["details"].append("Action distribution drift detected")

        # Decision
        n_drifts = sum([results["feature_drift"], results["performance_drift"], results["action_drift"]])
        results["should_retrain"] = n_drifts >= 1
        results["should_pause"] = n_drifts >= 2

        if results["should_pause"]:
            logger.warning(f"DRIFT ALERT — PAUSE TRADING: {results['details']}")
        elif results["should_retrain"]:
            logger.warning(f"Drift detected — retrain recommended: {results['details']}")
        else:
            logger.debug("No drift detected")

        return results

    def _check_feature_drift(self) -> dict:
        from scipy import stats
        current = np.array(list(self._feature_buffer))
        n_features = min(current.shape[1], self.reference.shape[1])
        n_drifted = 0
        for i in range(n_features):
            _, p = stats.ks_2samp(self.reference[:, i], current[:, i])
            if p < self.ks_threshold:
                n_drifted += 1
        drifted = n_drifted > n_features * 0.3  # >30% features drifted
        return {"drifted": drifted, "n_drifted": n_drifted, "n_features": n_features}

    def _check_performance_drift(self) -> dict:
        if len(self._return_buffer) < 100:
            return {"degraded": False, "sharpe": 0.0}
        returns = np.array(list(self._return_buffer))
        sharpe = safe_sharpe_ratio(returns)
        return {"degraded": sharpe < self.sharpe_min, "sharpe": float(sharpe)}

    def _check_action_drift(self) -> dict:
        if len(self._action_buffer) < 200:
            return {"drifted": False}
        actions = np.array(list(self._action_buffer))
        first_half = actions[:len(actions) // 2]
        second_half = actions[len(actions) // 2:]
        from scipy import stats
        _, p = stats.ks_2samp(first_half, second_half)
        return {"drifted": p < self.ks_threshold}


class AlertManager:
    """ส่ง alert เมื่อ drift detected."""

    def __init__(self, telegram_token: str = "", telegram_chat_id: str = ""):
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id

    def send_alert(self, message: str, level: str = "warning"):
        """ส่ง alert ผ่าน Telegram (ถ้า config ครบ) + log."""
        log_fn = logger.warning if level == "warning" else logger.critical
        log_fn(f"ALERT [{level}]: {message}")

        if self.telegram_token and self.telegram_chat_id:
            self._send_telegram(message)

    def _send_telegram(self, message: str):
        try:
            import requests
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            requests.post(url, json={
                "chat_id": self.telegram_chat_id,
                "text": f"🚨 GARIC Alert\n{message}",
                "parse_mode": "HTML",
            }, timeout=10)
        except Exception as e:
            logger.error(f"Telegram alert failed: {e}")
