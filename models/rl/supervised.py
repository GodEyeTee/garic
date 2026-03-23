"""Supervised fallback model for action classification.

This module provides a practical fallback when PPO collapses into a single action.
It learns discrete Short / Flat / Long labels from future returns on the training split
and selects a validation-tuned confidence threshold before evaluating on the test split.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

ACTION_SHORT = 0
ACTION_FLAT = 1
ACTION_LONG = 2


def _normalize_range(total_len: int, segment_range: tuple[int, int]) -> tuple[int, int]:
    start, end = int(segment_range[0]), int(segment_range[1])
    start = max(0, min(start, total_len - 1))
    end = max(start + 1, min(end, total_len))
    return start, end


def _build_action_labels(
    prices: np.ndarray,
    segment_range: tuple[int, int],
    horizon: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_len = len(prices)
    start, end = _normalize_range(total_len, segment_range)
    last_start = end - max(int(horizon), 1)
    if last_start <= start:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    indices = np.arange(start, last_start, dtype=np.int32)
    future_indices = indices + int(horizon)
    future_returns = (prices[future_indices] / prices[indices]) - 1.0
    labels = np.full(indices.shape, ACTION_FLAT, dtype=np.int32)
    labels[future_returns > threshold] = ACTION_LONG
    labels[future_returns < -threshold] = ACTION_SHORT
    return indices, labels, future_returns.astype(np.float32)


@dataclass
class SupervisedActionModel:
    scaler: StandardScaler | None
    classifier: Any
    feature_dim: int
    confidence_threshold: float
    min_hold_steps: int
    reversal_margin: float
    metadata: dict

    def _prepare_obs(self, obs: np.ndarray) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr[:, : self.feature_dim]

    def _extract_policy_state(self, obs: np.ndarray) -> tuple[int, float]:
        arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        if arr.shape[0] < self.feature_dim + 4:
            return ACTION_FLAT, 0.0
        position = float(arr[self.feature_dim])
        pos_steps = max(float(arr[self.feature_dim + 3]) * 100.0, 0.0)
        if position > 0.5:
            return ACTION_LONG, pos_steps
        if position < -0.5:
            return ACTION_SHORT, pos_steps
        return ACTION_FLAT, 0.0

    def predict(self, obs, deterministic: bool = True):
        x = self._prepare_obs(obs)
        if self.scaler is not None:
            x = self.scaler.transform(x)
        proba = self.classifier.predict_proba(x)
        best_idx = int(np.argmax(proba[0]))
        best_prob = float(proba[0, best_idx])
        best_action = int(self.classifier.classes_[best_idx])
        current_action, pos_steps = self._extract_policy_state(obs)
        prob_by_action = {
            int(cls): float(prob)
            for cls, prob in zip(self.classifier.classes_, proba[0], strict=False)
        }
        current_prob = float(prob_by_action.get(current_action, 0.0))

        if current_action != ACTION_FLAT:
            if pos_steps < float(self.min_hold_steps) and current_prob >= 0.15:
                return current_action, None
            switch_threshold = max(float(self.confidence_threshold), current_prob + float(self.reversal_margin))
            if best_action == current_action:
                action = current_action
            elif best_prob >= switch_threshold:
                action = best_action
            else:
                action = current_action
        else:
            if best_action != ACTION_FLAT and best_prob < float(self.confidence_threshold):
                action = ACTION_FLAT
            else:
                action = best_action
        return action, None

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        if target.suffix.lower() != ".joblib":
            target = target.with_suffix(".joblib")
        payload = {
            "scaler": self.scaler,
            "classifier": self.classifier,
            "feature_dim": self.feature_dim,
            "confidence_threshold": self.confidence_threshold,
            "min_hold_steps": self.min_hold_steps,
            "reversal_margin": self.reversal_margin,
            "metadata": self.metadata,
        }
        joblib.dump(payload, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "SupervisedActionModel":
        payload = joblib.load(path)
        return cls(
            scaler=payload["scaler"],
            classifier=payload["classifier"],
            feature_dim=int(payload["feature_dim"]),
            confidence_threshold=float(payload["confidence_threshold"]),
            min_hold_steps=int(payload.get("min_hold_steps", 0)),
            reversal_margin=float(payload.get("reversal_margin", 0.0)),
            metadata=dict(payload.get("metadata", {})),
        )


def train_supervised_action_model(
    feature_array: np.ndarray,
    prices: np.ndarray,
    train_range: tuple[int, int],
    validation_range: tuple[int, int],
    *,
    model_type: str = "logreg",
    horizon: int = 16,
    min_return_threshold: float = 0.003,
    threshold_quantile: float = 0.65,
    max_train_samples: int = 120_000,
    logistic_c: float = 1.0,
    extra_trees_n_estimators: int = 160,
    extra_trees_max_depth: int = 12,
    extra_trees_min_samples_leaf: int = 32,
    min_hold_steps: int = 16,
    reversal_margin: float = 0.08,
    random_state: int = 42,
) -> tuple[SupervisedActionModel, dict]:
    feature_dim = int(feature_array.shape[1])
    train_idx, _, train_future_returns = _build_action_labels(
        prices, train_range, horizon=horizon, threshold=min_return_threshold,
    )
    if train_idx.size == 0:
        raise ValueError("Not enough samples for supervised fallback training.")

    adaptive_threshold = max(
        float(min_return_threshold),
        float(np.quantile(np.abs(train_future_returns), threshold_quantile)),
    )
    train_idx, train_labels, train_future_returns = _build_action_labels(
        prices, train_range, horizon=horizon, threshold=adaptive_threshold,
    )
    valid_idx, valid_labels, valid_future_returns = _build_action_labels(
        prices, validation_range, horizon=horizon, threshold=adaptive_threshold,
    )
    if train_idx.size == 0 or valid_idx.size == 0:
        raise ValueError("Insufficient train/validation samples for supervised fallback.")

    if train_idx.size > max_train_samples:
        stride = max(int(train_idx.size // max_train_samples), 1)
        train_idx = train_idx[::stride][:max_train_samples]
        train_labels = train_labels[::stride][:max_train_samples]
        train_future_returns = train_future_returns[::stride][:max_train_samples]

    x_train = np.asarray(feature_array[train_idx], dtype=np.float32)
    model_type = str(model_type).strip().lower()
    scaler: StandardScaler | None = None
    if model_type == "logreg":
        scaler = StandardScaler()
        x_train_fit = scaler.fit_transform(x_train)
        classifier = LogisticRegression(
            max_iter=400,
            solver="lbfgs",
            class_weight="balanced",
            C=float(logistic_c),
            random_state=int(random_state),
        )
    elif model_type == "extratrees":
        x_train_fit = x_train
        classifier = ExtraTreesClassifier(
            n_estimators=max(int(extra_trees_n_estimators), 64),
            max_depth=max(int(extra_trees_max_depth), 2),
            min_samples_leaf=max(int(extra_trees_min_samples_leaf), 1),
            class_weight="balanced_subsample",
            max_features="sqrt",
            n_jobs=-1,
            random_state=int(random_state),
        )
    else:
        raise ValueError(f"Unsupported supervised model_type: {model_type}")

    classifier.fit(x_train_fit, train_labels)

    metadata = {
        "horizon": int(horizon),
        "label_threshold": float(adaptive_threshold),
        "train_samples": int(len(train_idx)),
        "validation_samples": int(len(valid_idx)),
        "train_class_counts": {
            "short": int(np.sum(train_labels == ACTION_SHORT)),
            "flat": int(np.sum(train_labels == ACTION_FLAT)),
            "long": int(np.sum(train_labels == ACTION_LONG)),
        },
        "validation_class_counts": {
            "short": int(np.sum(valid_labels == ACTION_SHORT)),
            "flat": int(np.sum(valid_labels == ACTION_FLAT)),
            "long": int(np.sum(valid_labels == ACTION_LONG)),
        },
        "validation_return_mean": float(np.mean(valid_future_returns)) if len(valid_future_returns) else 0.0,
        "model_type": model_type,
        "min_hold_steps": int(min_hold_steps),
        "reversal_margin": float(reversal_margin),
    }

    logger.info(
        "Supervised fallback dataset: model=%s horizon=%d threshold=%.4f train=%d valid=%d "
        "train_labels={short:%d, flat:%d, long:%d}",
        model_type,
        int(horizon),
        float(adaptive_threshold),
        int(metadata["train_samples"]),
        int(metadata["validation_samples"]),
        int(metadata["train_class_counts"]["short"]),
        int(metadata["train_class_counts"]["flat"]),
        int(metadata["train_class_counts"]["long"]),
    )

    model = SupervisedActionModel(
        scaler=scaler,
        classifier=classifier,
        feature_dim=feature_dim,
        confidence_threshold=0.34,
        min_hold_steps=max(int(min_hold_steps), 0),
        reversal_margin=max(float(reversal_margin), 0.0),
        metadata=metadata,
    )
    return model, metadata
