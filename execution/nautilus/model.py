"""Model loading and prediction helpers for Nautilus strategies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from models.rl.environment import AGENT_STATE_DIM, build_agent_state


ACTION_TO_DIRECTION = {
    0: -1.0,
    1: 0.0,
    2: 1.0,
}


@dataclass
class ModelPrediction:
    action_index: int
    direction: float
    confidence: float
    probabilities: dict[str, float]


class GaricModelAdapter:
    def __init__(self, model_path: str | Path):
        from stable_baselines3 import PPO

        self.model_path = Path(model_path)
        self.model_family = "ppo"
        self._model = None
        self.feature_dim = 25

        suffix = self.model_path.suffix.lower()
        if suffix == ".joblib":
            from models.rl.supervised import SupervisedActionModel

            self._model = SupervisedActionModel.load(self.model_path)
            self.model_family = f"supervised_{self._model.metadata.get('model_type', 'unknown')}"
            self.feature_dim = int(self._model.feature_dim)
        else:
            self._model = PPO.load(str(self.model_path))
            obs_shape = getattr(self._model.observation_space, "shape", None) or (33,)
            self.feature_dim = max(int(obs_shape[0]) - AGENT_STATE_DIM, 1)

    def _build_obs(
        self,
        feature_array: np.ndarray,
        position_state: float,
        flat_steps: int,
        pos_steps: int,
        upnl: float = 0.0,
        equity_ratio: float = 0.0,
        drawdown: float = 0.0,
        rolling_volatility: float = 0.0,
        turnover_last_step: float = 0.0,
    ) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(feature_array[: self.feature_dim], dtype=np.float32),
                build_agent_state(
                    position=position_state,
                    upnl=upnl,
                    equity_ratio=equity_ratio,
                    drawdown=drawdown,
                    rolling_volatility=rolling_volatility,
                    turnover_last_step=turnover_last_step,
                    flat_steps=flat_steps,
                    pos_steps=pos_steps,
                ),
            ],
        ).astype(np.float32)

    def predict(
        self,
        feature_array: np.ndarray,
        *,
        position_state: float,
        flat_steps: int,
        pos_steps: int,
        upnl: float = 0.0,
        equity_ratio: float = 0.0,
        drawdown: float = 0.0,
        rolling_volatility: float = 0.0,
        turnover_last_step: float = 0.0,
    ) -> ModelPrediction:
        obs = self._build_obs(
            feature_array,
            position_state=position_state,
            flat_steps=flat_steps,
            pos_steps=pos_steps,
            upnl=upnl,
            equity_ratio=equity_ratio,
            drawdown=drawdown,
            rolling_volatility=rolling_volatility,
            turnover_last_step=turnover_last_step,
        )

        if self.model_path.suffix.lower() == ".joblib":
            action, _ = self._model.predict(obs, deterministic=True)
            x = self._model._prepare_obs(obs)  # noqa: SLF001
            if self._model.scaler is not None:
                x = self._model.scaler.transform(x)
            proba_raw = self._model.classifier.predict_proba(x)[0]
            cls_order = [int(v) for v in self._model.classifier.classes_]
            probs = {cls: 0.0 for cls in (0, 1, 2)}
            for cls, prob in zip(cls_order, proba_raw, strict=False):
                probs[int(cls)] = float(prob)
        else:
            action, _ = self._model.predict(obs, deterministic=True)
            probs = {0: 0.0, 1: 0.0, 2: 0.0}
            try:
                obs_tensor, _ = self._model.policy.obs_to_tensor(obs.reshape(1, -1))
                dist = self._model.policy.get_distribution(obs_tensor)
                values = dist.distribution.probs.detach().cpu().numpy()[0]
                for idx, prob in enumerate(values.tolist()):
                    probs[idx] = float(prob)
            except Exception:
                pass

        action_index = int(action.item()) if isinstance(action, np.ndarray) else int(action)
        confidence = float(max(probs.values())) if probs else 0.0
        prob_map = {
            "short": float(probs.get(0, 0.0)),
            "flat": float(probs.get(1, 0.0)),
            "long": float(probs.get(2, 0.0)),
        }
        return ModelPrediction(
            action_index=action_index,
            direction=float(ACTION_TO_DIRECTION.get(action_index, 0.0)),
            confidence=confidence,
            probabilities=prob_map,
        )
