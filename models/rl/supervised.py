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

from models.rl.environment import (
    AGENT_STATE_DIM,
    STATE_IDX_DRAWDOWN,
    STATE_IDX_FLAT_STEPS,
    STATE_IDX_POSITION,
    STATE_IDX_POS_STEPS,
    STATE_IDX_ROLLING_VOL,
    STATE_IDX_UPNL,
)

logger = logging.getLogger(__name__)

ACTION_SHORT = 0
ACTION_FLAT = 1
ACTION_LONG = 2


def compute_trend_score(features_row: np.ndarray) -> float:
    row = np.asarray(features_row, dtype=np.float32).reshape(-1)
    if row.size < 5:
        return 0.0
    returns = np.zeros(5, dtype=np.float32)
    usable = min(5, row.size)
    returns[:usable] = row[:usable]
    ema9_ratio = float(row[18]) if row.size > 18 else 1.0
    ema21_ratio = float(row[19]) if row.size > 19 else ema9_ratio
    return float(
        float(returns[2]) * 0.55
        + float(returns[3]) * 0.85
        + float(returns[4]) * 1.10
        + (1.0 - ema9_ratio) * 0.35
        + (1.0 - ema21_ratio) * 0.50
    )


class TrendRuleClassifier:
    """Heuristic probability model based on multi-horizon trend context."""

    classes_ = np.array([ACTION_SHORT, ACTION_FLAT, ACTION_LONG], dtype=np.int32)

    def __init__(
        self,
        *,
        feature_dim: int,
        entry_threshold: float = 0.004,
        neutral_band: float = 0.0015,
    ):
        self.feature_dim = int(feature_dim)
        self.entry_threshold = max(float(entry_threshold), 1e-6)
        self.neutral_band = max(float(neutral_band), 1e-6)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        probs = np.zeros((arr.shape[0], 3), dtype=np.float32)
        scale = max(self.entry_threshold * 3.0, self.neutral_band * 2.0, 1e-6)
        for i, row in enumerate(arr):
            score = compute_trend_score(row[: self.feature_dim])
            mag = abs(score)
            if score >= self.entry_threshold:
                strength = min((score - self.entry_threshold) / scale, 1.0)
                long_prob = 0.56 + 0.32 * strength
                flat_prob = 0.36 - 0.24 * strength
                short_prob = max(1.0 - long_prob - flat_prob, 0.02)
            elif score <= -self.entry_threshold:
                strength = min((-score - self.entry_threshold) / scale, 1.0)
                short_prob = 0.56 + 0.32 * strength
                flat_prob = 0.36 - 0.24 * strength
                long_prob = max(1.0 - short_prob - flat_prob, 0.02)
            else:
                centered = min(mag / self.neutral_band, 1.0)
                flat_prob = 0.72 - 0.16 * centered
                directional = (1.0 - flat_prob) / 2.0
                if score >= 0:
                    long_prob = directional + 0.05 * centered
                    short_prob = max(1.0 - flat_prob - long_prob, 0.02)
                else:
                    short_prob = directional + 0.05 * centered
                    long_prob = max(1.0 - flat_prob - short_prob, 0.02)
            total = max(short_prob + flat_prob + long_prob, 1e-6)
            probs[i] = np.array(
                [short_prob / total, flat_prob / total, long_prob / total],
                dtype=np.float32,
            )
        return probs


class ConstantClassifier:
    """Degenerate classifier that always emits the same action."""

    classes_ = np.array([ACTION_SHORT, ACTION_FLAT, ACTION_LONG], dtype=np.int32)

    def __init__(self, action: int = ACTION_FLAT):
        self.action = int(action)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        probs = np.zeros((arr.shape[0], 3), dtype=np.float32)
        probs[:, self.action] = 1.0
        return probs


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
    entry_margin: float
    exit_to_flat_margin: float
    max_hold_steps: int
    stop_loss_threshold: float
    drawdown_exit_threshold: float
    trend_alignment_threshold: float
    countertrend_margin: float
    metadata: dict
    long_confidence_threshold: float | None = None
    short_confidence_threshold: float | None = None
    regime_confidence_relief: float = 0.0
    flat_reentry_cooldown_steps: int = 0

    def _prepare_obs(self, obs: np.ndarray) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr[:, : self.feature_dim]

    def _threshold_for_action(self, action: int) -> float:
        base = float(max(self.confidence_threshold, 0.0))
        if action == ACTION_LONG and self.long_confidence_threshold is not None:
            return float(max(self.long_confidence_threshold, 0.0))
        if action == ACTION_SHORT and self.short_confidence_threshold is not None:
            return float(max(self.short_confidence_threshold, 0.0))
        return base

    def _extract_policy_state(self, obs: np.ndarray) -> tuple[int, float, float, float, float, float]:
        arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        if arr.shape[0] < self.feature_dim + AGENT_STATE_DIM:
            return ACTION_FLAT, 0.0, 0.0, 0.0, 0.0, 0.0
        position = float(arr[self.feature_dim + STATE_IDX_POSITION])
        flat_steps = max(float(arr[self.feature_dim + STATE_IDX_FLAT_STEPS]) * 100.0, 0.0)
        pos_steps = max(float(arr[self.feature_dim + STATE_IDX_POS_STEPS]) * 100.0, 0.0)
        upnl = float(arr[self.feature_dim + STATE_IDX_UPNL])
        drawdown = max(float(arr[self.feature_dim + STATE_IDX_DRAWDOWN]), 0.0)
        rolling_vol = max(float(arr[self.feature_dim + STATE_IDX_ROLLING_VOL]), 0.0)
        if position > 0.5:
            return ACTION_LONG, flat_steps, pos_steps, upnl, drawdown, rolling_vol
        if position < -0.5:
            return ACTION_SHORT, flat_steps, pos_steps, upnl, drawdown, rolling_vol
        return ACTION_FLAT, flat_steps, 0.0, upnl, drawdown, rolling_vol

    def _trend_score(self, features_row: np.ndarray) -> float:
        return compute_trend_score(features_row)

    def predict(self, obs, deterministic: bool = True):
        x_raw = self._prepare_obs(obs)
        x = np.array(x_raw, copy=True)
        if self.scaler is not None:
            x = self.scaler.transform(x)
        proba = self.classifier.predict_proba(x)
        current_action, flat_steps, pos_steps, upnl, drawdown, _rolling_vol = self._extract_policy_state(obs)
        prob_by_action = {
            int(cls): float(prob)
            for cls, prob in zip(self.classifier.classes_, proba[0], strict=False)
        }
        short_prob = float(prob_by_action.get(ACTION_SHORT, 0.0))
        flat_prob = float(prob_by_action.get(ACTION_FLAT, 0.0))
        long_prob = float(prob_by_action.get(ACTION_LONG, 0.0))
        current_prob = float(prob_by_action.get(current_action, 0.0))
        best_action = ACTION_LONG if long_prob >= max(short_prob, flat_prob) else (
            ACTION_SHORT if short_prob >= max(long_prob, flat_prob) else ACTION_FLAT
        )
        best_prob = float(max(short_prob, flat_prob, long_prob))
        entry_margin = float(max(self.entry_margin, 0.0))
        exit_to_flat_margin = float(max(self.exit_to_flat_margin, 0.0))
        reversal_margin = float(max(self.reversal_margin, 0.0))
        confidence_threshold = float(max(self.confidence_threshold, 0.0))
        stop_loss_threshold = float(min(self.stop_loss_threshold, 0.0))
        drawdown_exit_threshold = float(max(self.drawdown_exit_threshold, 0.0))
        max_hold_steps = max(int(self.max_hold_steps), 0)
        trend_alignment_threshold = float(max(self.trend_alignment_threshold, 0.0))
        countertrend_margin = float(max(self.countertrend_margin, 0.0))
        trend_score = self._trend_score(x_raw[0])
        bullish_regime = trend_score >= trend_alignment_threshold
        bearish_regime = trend_score <= -trend_alignment_threshold
        regime_confidence_relief = float(max(self.regime_confidence_relief, 0.0))
        long_confidence_threshold = max(
            self._threshold_for_action(ACTION_LONG) - (regime_confidence_relief if bullish_regime else 0.0),
            0.0,
        )
        short_confidence_threshold = max(
            self._threshold_for_action(ACTION_SHORT) - (regime_confidence_relief if bearish_regime else 0.0),
            0.0,
        )
        reentry_cooldown_steps = max(int(self.flat_reentry_cooldown_steps), 0)
        cooldown_active = current_action == ACTION_FLAT and flat_steps < reentry_cooldown_steps
        if cooldown_active:
            long_confidence_threshold += 0.05
            short_confidence_threshold += 0.05
            entry_margin += 0.04

        if current_action != ACTION_FLAT:
            current_threshold = self._threshold_for_action(current_action)
            if stop_loss_threshold < 0.0 and upnl <= stop_loss_threshold:
                return ACTION_FLAT, None
            if drawdown_exit_threshold > 0.0 and drawdown >= drawdown_exit_threshold:
                return ACTION_FLAT, None
            if max_hold_steps > 0 and pos_steps >= max_hold_steps:
                stale_signal = flat_prob >= current_prob - min(exit_to_flat_margin, 0.03)
                weak_hold = current_prob < max(current_threshold - 0.08, 0.18)
                if stale_signal or weak_hold:
                    return ACTION_FLAT, None
            if pos_steps < float(self.min_hold_steps) and current_prob >= 0.15:
                return current_action, None
            opposite_action = ACTION_SHORT if current_action == ACTION_LONG else ACTION_LONG
            opposite_prob = float(prob_by_action.get(opposite_action, 0.0))
            opposite_regime_ok = bearish_regime if opposite_action == ACTION_SHORT else bullish_regime
            opposite_threshold = short_confidence_threshold if opposite_action == ACTION_SHORT else long_confidence_threshold
            reverse_signal = (
                opposite_prob >= max(
                    opposite_threshold,
                    current_prob + reversal_margin,
                    flat_prob + entry_margin,
                )
                and (
                    opposite_regime_ok
                    or opposite_prob >= current_prob + countertrend_margin
                )
            )
            flatten_signal = (
                flat_prob >= current_prob + exit_to_flat_margin
                or (
                    current_prob < max(current_threshold - 0.05, 0.20)
                    and flat_prob >= current_prob - 0.02
                )
            )
            if reverse_signal:
                action = opposite_action
            elif flatten_signal:
                action = ACTION_FLAT
            elif best_action == current_action:
                action = current_action
            else:
                action = current_action
        else:
            long_regime_ok = bullish_regime or long_prob >= short_prob + countertrend_margin
            short_regime_ok = bearish_regime or short_prob >= long_prob + countertrend_margin
            long_signal = (
                long_prob >= long_confidence_threshold
                and long_prob >= short_prob
                and long_prob >= flat_prob + entry_margin
                and long_regime_ok
            )
            short_signal = (
                short_prob >= short_confidence_threshold
                and short_prob > long_prob
                and short_prob >= flat_prob + entry_margin
                and short_regime_ok
            )
            if long_signal:
                action = ACTION_LONG
            elif short_signal:
                action = ACTION_SHORT
            else:
                action = ACTION_FLAT
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
            "entry_margin": self.entry_margin,
            "exit_to_flat_margin": self.exit_to_flat_margin,
            "max_hold_steps": self.max_hold_steps,
            "stop_loss_threshold": self.stop_loss_threshold,
            "drawdown_exit_threshold": self.drawdown_exit_threshold,
            "trend_alignment_threshold": self.trend_alignment_threshold,
            "countertrend_margin": self.countertrend_margin,
            "metadata": self.metadata,
            "long_confidence_threshold": self.long_confidence_threshold,
            "short_confidence_threshold": self.short_confidence_threshold,
            "regime_confidence_relief": self.regime_confidence_relief,
            "flat_reentry_cooldown_steps": self.flat_reentry_cooldown_steps,
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
            entry_margin=float(payload.get("entry_margin", 0.08)),
            exit_to_flat_margin=float(payload.get("exit_to_flat_margin", 0.05)),
            max_hold_steps=int(payload.get("max_hold_steps", 0)),
            stop_loss_threshold=float(payload.get("stop_loss_threshold", -0.01)),
            drawdown_exit_threshold=float(payload.get("drawdown_exit_threshold", 0.04)),
            trend_alignment_threshold=float(payload.get("trend_alignment_threshold", 0.0)),
            countertrend_margin=float(payload.get("countertrend_margin", 0.08)),
            metadata=dict(payload.get("metadata", {})),
            long_confidence_threshold=(
                None if payload.get("long_confidence_threshold") is None
                else float(payload.get("long_confidence_threshold"))
            ),
            short_confidence_threshold=(
                None if payload.get("short_confidence_threshold") is None
                else float(payload.get("short_confidence_threshold"))
            ),
            regime_confidence_relief=float(payload.get("regime_confidence_relief", 0.0)),
            flat_reentry_cooldown_steps=int(payload.get("flat_reentry_cooldown_steps", 0)),
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
    entry_margin: float = 0.08,
    exit_to_flat_margin: float = 0.05,
    max_hold_steps: int = 64,
    stop_loss_threshold: float = -0.012,
    drawdown_exit_threshold: float = 0.04,
    trend_alignment_threshold: float = 0.0015,
    countertrend_margin: float = 0.08,
    taker_fee: float = 0.0005,
    slippage_bps: float = 1.0,
    leverage: float = 1.0,
    regime_confidence_relief: float = 0.0,
    flat_reentry_cooldown_steps: int = 0,
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
    round_trip_cost = (2.0 * (float(taker_fee) + float(slippage_bps) / 10000.0)) / max(float(leverage), 1e-9)
    effective_threshold = adaptive_threshold + round_trip_cost
    train_idx, train_labels, train_future_returns = _build_action_labels(
        prices, train_range, horizon=horizon, threshold=effective_threshold,
    )
    valid_idx, valid_labels, valid_future_returns = _build_action_labels(
        prices, validation_range, horizon=horizon, threshold=effective_threshold,
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

    sample_weights = 1.0 + np.clip(
        np.abs(train_future_returns) / max(effective_threshold, 1e-6),
        0.0,
        4.0,
    )
    classifier.fit(x_train_fit, train_labels, sample_weight=sample_weights)

    metadata = {
        "horizon": int(horizon),
        "label_threshold": float(effective_threshold),
        "signal_threshold": float(adaptive_threshold),
        "round_trip_cost": float(round_trip_cost),
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
        "entry_margin": float(entry_margin),
        "exit_to_flat_margin": float(exit_to_flat_margin),
        "max_hold_steps": int(max_hold_steps),
        "stop_loss_threshold": float(stop_loss_threshold),
        "drawdown_exit_threshold": float(drawdown_exit_threshold),
        "trend_alignment_threshold": float(trend_alignment_threshold),
        "countertrend_margin": float(countertrend_margin),
        "regime_confidence_relief": float(regime_confidence_relief),
        "flat_reentry_cooldown_steps": int(max(flat_reentry_cooldown_steps, 0)),
    }

    x_valid = np.asarray(feature_array[valid_idx], dtype=np.float32)
    x_valid_eval = scaler.transform(x_valid) if scaler is not None else x_valid
    valid_proba = np.asarray(classifier.predict_proba(x_valid_eval), dtype=np.float64)
    aligned_valid_proba = np.zeros((valid_proba.shape[0], 3), dtype=np.float64)
    for cls_idx, cls in enumerate(getattr(classifier, "classes_", np.array([0, 1, 2], dtype=np.int32))):
        cls_int = int(cls)
        if 0 <= cls_int < 3:
            aligned_valid_proba[:, cls_int] = valid_proba[:, cls_idx]
    valid_one_hot = np.eye(3, dtype=np.float64)[valid_labels.astype(np.int32)]
    metadata["validation_brier"] = float(np.mean(np.sum((aligned_valid_proba - valid_one_hot) ** 2, axis=1)))
    metadata["validation_label_abs_mean"] = float(np.mean(np.abs(valid_future_returns))) if len(valid_future_returns) else 0.0

    logger.info(
        "Supervised fallback dataset: model=%s horizon=%d signal_threshold=%.4f effective_threshold=%.4f "
        "round_trip_cost=%.4f train=%d valid=%d brier=%.4f train_labels={short:%d, flat:%d, long:%d}",
        model_type,
        int(horizon),
        float(adaptive_threshold),
        float(effective_threshold),
        float(round_trip_cost),
        int(metadata["train_samples"]),
        int(metadata["validation_samples"]),
        float(metadata["validation_brier"]),
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
        entry_margin=max(float(entry_margin), 0.0),
        exit_to_flat_margin=max(float(exit_to_flat_margin), 0.0),
        max_hold_steps=max(int(max_hold_steps), 0),
        stop_loss_threshold=min(float(stop_loss_threshold), 0.0),
        drawdown_exit_threshold=max(float(drawdown_exit_threshold), 0.0),
        trend_alignment_threshold=max(float(trend_alignment_threshold), 0.0),
        countertrend_margin=max(float(countertrend_margin), 0.0),
        metadata=metadata,
        long_confidence_threshold=None,
        short_confidence_threshold=None,
        regime_confidence_relief=max(float(regime_confidence_relief), 0.0),
        flat_reentry_cooldown_steps=max(int(flat_reentry_cooldown_steps), 0),
    )
    return model, metadata


def build_constant_action_model(
    *,
    feature_dim: int,
    action: int = ACTION_FLAT,
    metadata: dict | None = None,
) -> SupervisedActionModel:
    return SupervisedActionModel(
        scaler=None,
        classifier=ConstantClassifier(action=action),
        feature_dim=int(feature_dim),
        confidence_threshold=1.0,
        min_hold_steps=0,
        reversal_margin=0.0,
        entry_margin=0.0,
        exit_to_flat_margin=0.0,
        max_hold_steps=0,
        stop_loss_threshold=0.0,
        drawdown_exit_threshold=0.0,
        trend_alignment_threshold=0.0,
        countertrend_margin=0.0,
        metadata=dict(metadata or {}),
        long_confidence_threshold=1.0,
        short_confidence_threshold=1.0,
        regime_confidence_relief=0.0,
        flat_reentry_cooldown_steps=0,
    )
