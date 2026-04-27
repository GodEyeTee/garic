"""Supervised fallback model for action classification.

This module provides a practical fallback when PPO collapses into a single action.
It learns discrete Short / Flat / Long labels from future returns on the training split
and selects a validation-tuned confidence threshold before evaluating on the test split.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

from models.rl.environment import (
    AGENT_STATE_DIM,
    STATE_IDX_DRAWDOWN,
    STATE_IDX_EQUITY_RATIO,
    STATE_IDX_FLAT_STEPS,
    STATE_IDX_POSITION,
    STATE_IDX_POS_STEPS,
    STATE_IDX_ROLLING_VOL,
    STATE_IDX_TURNOVER,
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


class TrendRuleBinaryClassifier:
    """Binary edge classifier backed by the same trend score heuristic."""

    classes_ = np.array([0, 1], dtype=np.int32)

    def __init__(
        self,
        *,
        feature_dim: int,
        positive_direction: int,
        entry_threshold: float = 0.004,
        neutral_band: float = 0.0015,
    ):
        self.feature_dim = int(feature_dim)
        self.positive_direction = int(positive_direction)
        self.entry_threshold = max(float(entry_threshold), 1e-6)
        self.neutral_band = max(float(neutral_band), 1e-6)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        probs = np.zeros((arr.shape[0], 2), dtype=np.float32)
        scale = max(self.entry_threshold * 3.0, self.neutral_band * 2.0, 1e-6)
        for i, row in enumerate(arr):
            score = compute_trend_score(row[: self.feature_dim])
            aligned_score = score if self.positive_direction == ACTION_LONG else -score
            if aligned_score >= self.entry_threshold:
                strength = min((aligned_score - self.entry_threshold) / scale, 1.0)
                positive_prob = 0.56 + 0.32 * strength
            elif aligned_score <= -self.neutral_band:
                strength = min((-aligned_score - self.neutral_band) / scale, 1.0)
                positive_prob = 0.12 - 0.08 * strength
            else:
                centered = min(abs(aligned_score) / max(self.neutral_band, 1e-6), 1.0)
                positive_prob = 0.32 + (0.12 * centered)
            positive_prob = float(np.clip(positive_prob, 0.02, 0.98))
            probs[i] = np.array([1.0 - positive_prob, positive_prob], dtype=np.float32)
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


class ConstantBinaryClassifier:
    """Binary classifier that always emits a fixed positive probability."""

    classes_ = np.array([0, 1], dtype=np.int32)

    def __init__(self, positive_prob: float = 0.0):
        self.positive_prob = float(np.clip(positive_prob, 0.0, 1.0))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        pos = np.full((arr.shape[0], 1), self.positive_prob, dtype=np.float32)
        neg = 1.0 - pos
        return np.concatenate([neg, pos], axis=1)


def _binary_positive_probability(classifier: Any, x: np.ndarray) -> np.ndarray:
    probs = np.asarray(classifier.predict_proba(x), dtype=np.float32)
    if probs.ndim == 1:
        probs = probs[None, :]
    classes = np.asarray(getattr(classifier, "classes_", np.array([0, 1], dtype=np.int32)), dtype=np.int32)
    if probs.shape[1] == 1:
        only_class = int(classes[0]) if classes.size else 0
        return np.ones(probs.shape[0], dtype=np.float32) if only_class == 1 else np.zeros(probs.shape[0], dtype=np.float32)
    pos_idx = next((idx for idx, cls in enumerate(classes.tolist()) if int(cls) == 1), probs.shape[1] - 1)
    return np.asarray(probs[:, pos_idx], dtype=np.float32)


class BinaryEdgeActionClassifier:
    """Compose long/short binary edge models into action probabilities."""

    classes_ = np.array([ACTION_SHORT, ACTION_FLAT, ACTION_LONG], dtype=np.int32)

    def __init__(self, *, long_classifier: Any, short_classifier: Any):
        self.long_classifier = long_classifier
        self.short_classifier = short_classifier

    def predict_action_components(self, x: np.ndarray) -> dict[str, np.ndarray]:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        long_prob = np.clip(_binary_positive_probability(self.long_classifier, arr), 0.0, 1.0)
        short_prob = np.clip(_binary_positive_probability(self.short_classifier, arr), 0.0, 1.0)
        flat_prob = np.clip(1.0 - np.maximum(long_prob, short_prob), 0.0, 1.0)
        raw = np.stack([short_prob, flat_prob, long_prob], axis=1).astype(np.float32)
        total = np.maximum(np.sum(raw, axis=1, keepdims=True), 1e-6)
        normalized = (raw / total).astype(np.float32)
        return {
            "raw": raw,
            "normalized": normalized,
        }

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.predict_action_components(x)["normalized"]


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


def _build_binary_edge_targets(
    future_returns: np.ndarray,
    *,
    round_trip_cost: float,
    edge_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    returns = np.asarray(future_returns, dtype=np.float32).reshape(-1)
    long_edges = returns - float(round_trip_cost)
    short_edges = (-returns) - float(round_trip_cost)
    long_targets = (long_edges > float(edge_threshold)).astype(np.int32)
    short_targets = (short_edges > float(edge_threshold)).astype(np.int32)
    return long_targets, short_targets, long_edges.astype(np.float32), short_edges.astype(np.float32)


def _normalize_probability_thresholds(
    thresholds: list[float] | tuple[float, ...] | np.ndarray | None,
) -> list[float]:
    if thresholds is None:
        values = np.array([0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90], dtype=np.float32)
    else:
        values = np.asarray(thresholds, dtype=np.float32).reshape(-1)
    if values.size == 0:
        values = np.array([0.40, 0.60, 0.80], dtype=np.float32)
    clipped = np.clip(values, 0.01, 0.99)
    unique_sorted = np.unique(np.round(clipped, 4))
    return [float(v) for v in unique_sorted.tolist()]


def _edge_summary(edges: np.ndarray) -> dict:
    arr = np.asarray(edges, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {
            "count": 0,
            "mean_edge": 0.0,
            "median_edge": 0.0,
            "positive_rate": 0.0,
            "edge_std": 0.0,
        }
    return {
        "count": int(arr.size),
        "mean_edge": float(np.mean(arr)),
        "median_edge": float(np.median(arr)),
        "positive_rate": float(np.mean(arr > 0.0)),
        "edge_std": float(np.std(arr)),
    }


def _build_post_cost_calibration(
    aligned_valid_proba: np.ndarray,
    valid_future_returns: np.ndarray,
    round_trip_cost: float,
    *,
    probability_thresholds: list[float] | tuple[float, ...] | np.ndarray | None,
    min_samples: int,
    window_count: int = 4,
    window_min_samples: int | None = None,
) -> dict:
    thresholds = _normalize_probability_thresholds(probability_thresholds)
    min_samples = max(int(min_samples), 8)
    window_count = max(int(window_count), 1)
    if window_min_samples is None:
        window_min_samples = max(min_samples // max(window_count, 1), 8)
    window_min_samples = max(int(window_min_samples), 4)
    future_returns = np.asarray(valid_future_returns, dtype=np.float64).reshape(-1)
    window_indices = [
        np.asarray(chunk, dtype=np.int32)
        for chunk in np.array_split(np.arange(future_returns.size, dtype=np.int32), window_count)
        if len(chunk) > 0
    ]
    calibration: dict[str, Any] = {
        "probability_thresholds": thresholds,
        "min_samples": int(min_samples),
        "window_count": int(len(window_indices)),
        "window_min_samples": int(window_min_samples),
    }

    side_specs = (
        ("long", ACTION_LONG, future_returns - float(round_trip_cost)),
        ("short", ACTION_SHORT, -future_returns - float(round_trip_cost)),
    )
    for side_name, action, side_edges in side_specs:
        probs = np.asarray(aligned_valid_proba[:, action], dtype=np.float64).reshape(-1)
        global_summary = _edge_summary(side_edges)
        global_window_summaries: list[dict[str, float]] = []
        for idx in window_indices:
            if idx.size < window_min_samples:
                continue
            global_window_summaries.append(_edge_summary(side_edges[idx]))
        if global_window_summaries:
            global_mean_edges = np.asarray([row["mean_edge"] for row in global_window_summaries], dtype=np.float64)
            global_positive_rates = np.asarray([row["positive_rate"] for row in global_window_summaries], dtype=np.float64)
            global_summary["active_window_ratio"] = float(len(global_window_summaries) / max(len(window_indices), 1))
            global_summary["robust_mean_edge"] = float(
                (np.median(global_mean_edges) * 0.65) + (np.min(global_mean_edges) * 0.35)
            )
            global_summary["robust_positive_rate"] = float(
                (np.median(global_positive_rates) * 0.65) + (np.min(global_positive_rates) * 0.35)
            )
        else:
            global_summary["active_window_ratio"] = 0.0
            global_summary["robust_mean_edge"] = float(global_summary["mean_edge"])
            global_summary["robust_positive_rate"] = float(global_summary["positive_rate"])
        tail_stats: list[dict[str, float | int]] = []
        for threshold in thresholds:
            mask = probs >= float(threshold)
            count = int(np.sum(mask))
            if count < min_samples:
                continue
            slice_summary = _edge_summary(side_edges[mask])
            shrink_reference = max(min_samples // 2, 12)
            shrink = count / float(count + shrink_reference)
            shrunk_mean_edge = (
                float(slice_summary["mean_edge"]) * shrink
                + float(global_summary["mean_edge"]) * (1.0 - shrink)
            )
            shrunk_positive_rate = (
                float(slice_summary["positive_rate"]) * shrink
                + float(global_summary["positive_rate"]) * (1.0 - shrink)
            )
            threshold_window_summaries: list[dict[str, float]] = []
            for idx in window_indices:
                if idx.size == 0:
                    continue
                window_mask = mask[idx]
                window_hits = int(np.sum(window_mask))
                if window_hits < window_min_samples:
                    continue
                threshold_window_summaries.append(_edge_summary(side_edges[idx][window_mask]))
            if threshold_window_summaries:
                window_mean_edges = np.asarray(
                    [row["mean_edge"] for row in threshold_window_summaries],
                    dtype=np.float64,
                )
                window_positive_rates = np.asarray(
                    [row["positive_rate"] for row in threshold_window_summaries],
                    dtype=np.float64,
                )
                active_window_ratio = float(len(threshold_window_summaries) / max(len(window_indices), 1))
                robust_mean_edge = float(
                    (np.median(window_mean_edges) * 0.65) + (np.min(window_mean_edges) * 0.35)
                )
                robust_positive_rate = float(
                    (np.median(window_positive_rates) * 0.65) + (np.min(window_positive_rates) * 0.35)
                )
            else:
                active_window_ratio = 0.0
                robust_mean_edge = float(global_summary["robust_mean_edge"])
                robust_positive_rate = float(global_summary["robust_positive_rate"])
            tail_stats.append(
                {
                    "threshold": float(threshold),
                    "count": count,
                    "mean_edge": float(shrunk_mean_edge),
                    "raw_mean_edge": float(slice_summary["mean_edge"]),
                    "median_edge": float(slice_summary["median_edge"]),
                    "positive_rate": float(shrunk_positive_rate),
                    "raw_positive_rate": float(slice_summary["positive_rate"]),
                    "edge_std": float(slice_summary["edge_std"]),
                    "active_window_ratio": float(active_window_ratio),
                    "robust_mean_edge": float(robust_mean_edge),
                    "robust_positive_rate": float(robust_positive_rate),
                }
            )

        if not tail_stats and probs.size > 0:
            top_n = min(probs.size, max(min_samples, 24))
            top_idx = np.argsort(probs)[-top_n:]
            top_summary = _edge_summary(side_edges[top_idx])
            tail_stats.append(
                {
                    "threshold": float(np.min(probs[top_idx])),
                    "count": int(top_summary["count"]),
                    "mean_edge": float(top_summary["mean_edge"]),
                    "raw_mean_edge": float(top_summary["mean_edge"]),
                    "median_edge": float(top_summary["median_edge"]),
                    "positive_rate": float(top_summary["positive_rate"]),
                    "raw_positive_rate": float(top_summary["positive_rate"]),
                    "edge_std": float(top_summary["edge_std"]),
                    "active_window_ratio": float(global_summary.get("active_window_ratio", 0.0)),
                    "robust_mean_edge": float(global_summary.get("robust_mean_edge", top_summary["mean_edge"])),
                    "robust_positive_rate": float(global_summary.get("robust_positive_rate", top_summary["positive_rate"])),
                }
            )

        calibration[side_name] = {
            "global": global_summary,
            "thresholds": tail_stats,
        }

    return calibration


def _align_action_probability_views(classifier: Any, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(classifier, "predict_action_components"):
        components = classifier.predict_action_components(x)
        raw = np.asarray(components.get("raw"), dtype=np.float64)
        normalized = np.asarray(components.get("normalized", raw), dtype=np.float64)
        return raw, normalized

    probs = np.asarray(classifier.predict_proba(x), dtype=np.float64)
    if probs.ndim == 1:
        probs = probs[None, :]
    aligned = np.zeros((probs.shape[0], 3), dtype=np.float64)
    for cls_idx, cls in enumerate(getattr(classifier, "classes_", np.array([0, 1, 2], dtype=np.int32))):
        cls_int = int(cls)
        if 0 <= cls_int < 3:
            aligned[:, cls_int] = probs[:, cls_idx]
    return aligned, aligned


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
    meta_label_min_edge: float = 0.0
    meta_label_edge_margin: float = 0.0
    meta_label_exit_edge: float = 0.0
    meta_label_min_positive_rate: float = 0.47
    calibration_min_active_window_ratio: float = 0.50

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

    def _extract_policy_state(self, obs: np.ndarray) -> tuple[int, float, float, float, float, float, float, float]:
        arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        if arr.shape[0] < self.feature_dim + AGENT_STATE_DIM:
            return ACTION_FLAT, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        position = float(arr[self.feature_dim + STATE_IDX_POSITION])
        flat_steps = max(float(arr[self.feature_dim + STATE_IDX_FLAT_STEPS]) * 100.0, 0.0)
        pos_steps = max(float(arr[self.feature_dim + STATE_IDX_POS_STEPS]) * 100.0, 0.0)
        upnl = float(arr[self.feature_dim + STATE_IDX_UPNL])
        equity_ratio = float(arr[self.feature_dim + STATE_IDX_EQUITY_RATIO])
        drawdown = max(float(arr[self.feature_dim + STATE_IDX_DRAWDOWN]), 0.0)
        rolling_vol = max(float(arr[self.feature_dim + STATE_IDX_ROLLING_VOL]), 0.0)
        turnover = max(float(arr[self.feature_dim + STATE_IDX_TURNOVER]), 0.0)
        if position > 0.5:
            return ACTION_LONG, flat_steps, pos_steps, upnl, equity_ratio, drawdown, rolling_vol, turnover
        if position < -0.5:
            return ACTION_SHORT, flat_steps, pos_steps, upnl, equity_ratio, drawdown, rolling_vol, turnover
        return ACTION_FLAT, flat_steps, 0.0, upnl, equity_ratio, drawdown, rolling_vol, turnover

    def _trend_score(self, features_row: np.ndarray) -> float:
        return compute_trend_score(features_row)

    def _estimate_post_cost_edge(self, action: int, probability: float) -> tuple[float, float]:
        calibration = self.metadata.get("post_cost_calibration")
        if not isinstance(calibration, dict):
            return 0.0, 0.5
        if action == ACTION_LONG:
            side_calibration = calibration.get("long")
        elif action == ACTION_SHORT:
            side_calibration = calibration.get("short")
        else:
            return 0.0, 1.0
        if not isinstance(side_calibration, dict):
            return 0.0, 0.5

        prob = float(np.clip(probability, 0.0, 1.0))
        selected = None
        for row in side_calibration.get("thresholds", []):
            if prob >= float(row.get("threshold", 1.0)):
                selected = row
        if selected is None:
            selected = side_calibration.get("global", {})
        active_window_ratio = float(selected.get("active_window_ratio", 1.0))
        min_active_window_ratio = float(max(self.calibration_min_active_window_ratio, 0.0))
        robust_mean_edge = float(selected.get("robust_mean_edge", selected.get("mean_edge", 0.0)))
        median_edge = float(selected.get("median_edge", robust_mean_edge))
        raw_mean_edge = float(selected.get("raw_mean_edge", robust_mean_edge))
        positive_rate = float(
            selected.get("robust_positive_rate", selected.get("positive_rate", 0.5))
        )
        raw_positive_rate = float(selected.get("raw_positive_rate", selected.get("positive_rate", positive_rate)))
        mean_edge = robust_mean_edge
        # Avoid collapsing to permanent flat when a sparse threshold has
        # slightly negative robust mean edge but still positive median edge
        # and acceptable hit rate after costs.
        if robust_mean_edge < 0.0 and median_edge > 0.0:
            mean_edge = float((robust_mean_edge * 0.45) + (median_edge * 0.55))
        elif robust_mean_edge <= 0.0 and raw_mean_edge > 0.0:
            mean_edge = float((robust_mean_edge * 0.60) + (raw_mean_edge * 0.40))
        elif median_edge > robust_mean_edge:
            mean_edge = float((robust_mean_edge * 0.75) + (median_edge * 0.25))
        if raw_positive_rate > positive_rate:
            positive_rate = float((positive_rate * 0.65) + (raw_positive_rate * 0.35))
        if min_active_window_ratio > 0.0 and active_window_ratio < min_active_window_ratio:
            damp = float(np.clip(active_window_ratio, 0.0, 1.0))
            mean_edge *= damp
            positive_rate = 0.5 + ((positive_rate - 0.5) * damp)
        return (mean_edge, positive_rate)

    def predict(self, obs, deterministic: bool = True):
        x_raw = self._prepare_obs(obs)
        x = np.array(x_raw, copy=True)
        if self.scaler is not None:
            x = self.scaler.transform(x)
        raw_proba, _ = _align_action_probability_views(self.classifier, x)
        current_action, flat_steps, pos_steps, upnl, equity_ratio, drawdown, rolling_vol, turnover = self._extract_policy_state(obs)
        prob_by_action = {
            int(cls): float(prob)
            for cls, prob in zip((ACTION_SHORT, ACTION_FLAT, ACTION_LONG), raw_proba[0], strict=False)
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
        countertrend_threshold_penalty = float(
            max(self.metadata.get("countertrend_threshold_penalty", 0.0), 0.0)
        )
        countertrend_entry_penalty = float(
            max(self.metadata.get("countertrend_entry_penalty", 0.0), 0.0)
        )
        flat_patience_steps = max(int(self.metadata.get("flat_patience_steps", 0)), 0)
        flat_patience_threshold_relief = float(
            max(self.metadata.get("flat_patience_threshold_relief", 0.0), 0.0)
        )
        flat_patience_entry_margin_relief = float(
            max(self.metadata.get("flat_patience_entry_margin_relief", 0.0), 0.0)
        )
        long_confidence_threshold = max(
            self._threshold_for_action(ACTION_LONG) - (regime_confidence_relief if bullish_regime else 0.0),
            0.0,
        )
        short_confidence_threshold = max(
            self._threshold_for_action(ACTION_SHORT) - (regime_confidence_relief if bearish_regime else 0.0),
            0.0,
        )
        long_entry_margin = float(entry_margin)
        short_entry_margin = float(entry_margin)
        if bullish_regime:
            short_confidence_threshold += countertrend_threshold_penalty
            short_entry_margin += countertrend_entry_penalty
        elif bearish_regime:
            long_confidence_threshold += countertrend_threshold_penalty * 0.5
            long_entry_margin += countertrend_entry_penalty * 0.5
        reentry_cooldown_steps = max(int(self.flat_reentry_cooldown_steps), 0)
        cooldown_active = current_action == ACTION_FLAT and flat_steps < reentry_cooldown_steps
        if cooldown_active:
            long_confidence_threshold += 0.05
            short_confidence_threshold += 0.05
            long_entry_margin += 0.04
            short_entry_margin += 0.04
        if current_action == ACTION_FLAT and flat_patience_steps > 0 and flat_steps >= flat_patience_steps:
            if bullish_regime:
                long_confidence_threshold = max(
                    long_confidence_threshold - flat_patience_threshold_relief,
                    0.0,
                )
                long_entry_margin = max(
                    long_entry_margin - flat_patience_entry_margin_relief,
                    0.0,
                )
            if bearish_regime:
                short_confidence_threshold = max(
                    short_confidence_threshold - flat_patience_threshold_relief,
                    0.0,
                )
                short_entry_margin = max(
                    short_entry_margin - flat_patience_entry_margin_relief,
                    0.0,
                )

        meta_label_min_edge = float(max(self.meta_label_min_edge, 0.0))
        meta_label_edge_margin = float(max(self.meta_label_edge_margin, 0.0))
        meta_label_exit_edge = float(self.meta_label_exit_edge)
        meta_label_min_positive_rate = float(np.clip(self.meta_label_min_positive_rate, 0.0, 1.0))
        relaxed_positive_rate_floor = max(meta_label_min_positive_rate - 0.08, 0.50)
        edge_override_threshold = max(meta_label_min_edge * 1.25, meta_label_min_edge + 0.0002)
        long_edge, long_positive_rate = self._estimate_post_cost_edge(ACTION_LONG, long_prob)
        short_edge, short_positive_rate = self._estimate_post_cost_edge(ACTION_SHORT, short_prob)
        current_edge = 0.0
        current_positive_rate = 1.0
        if current_action == ACTION_LONG:
            current_edge, current_positive_rate = long_edge, long_positive_rate
        elif current_action == ACTION_SHORT:
            current_edge, current_positive_rate = short_edge, short_positive_rate
        long_edge_ok = (
            long_edge >= meta_label_min_edge
            and long_edge >= short_edge + meta_label_edge_margin
            and (
                long_positive_rate >= meta_label_min_positive_rate
                or (
                    long_edge >= edge_override_threshold
                    and long_positive_rate >= relaxed_positive_rate_floor
                )
            )
        )
        short_edge_ok = (
            short_edge >= meta_label_min_edge
            and short_edge >= long_edge + meta_label_edge_margin
            and (
                short_positive_rate >= meta_label_min_positive_rate
                or (
                    short_edge >= edge_override_threshold
                    and short_positive_rate >= relaxed_positive_rate_floor
                )
            )
        )

        if current_action != ACTION_FLAT:
            current_threshold = self._threshold_for_action(current_action)
            if stop_loss_threshold < 0.0 and upnl <= stop_loss_threshold:
                return ACTION_FLAT, None
            if drawdown_exit_threshold > 0.0 and drawdown >= drawdown_exit_threshold:
                return ACTION_FLAT, None
            opposite_action = ACTION_SHORT if current_action == ACTION_LONG else ACTION_LONG
            opposite_prob = float(prob_by_action.get(opposite_action, 0.0))
            opposite_edge = short_edge if opposite_action == ACTION_SHORT else long_edge
            opposite_regime_ok = bearish_regime if opposite_action == ACTION_SHORT else bullish_regime
            opposite_threshold = short_confidence_threshold if opposite_action == ACTION_SHORT else long_confidence_threshold
            opposite_entry_margin = short_entry_margin if opposite_action == ACTION_SHORT else long_entry_margin
            reverse_signal = (
                opposite_prob >= max(
                    opposite_threshold,
                    current_prob + reversal_margin,
                    flat_prob + opposite_entry_margin,
                )
                and opposite_edge >= max(meta_label_min_edge, current_edge + meta_label_edge_margin)
                and (
                    opposite_regime_ok
                    or opposite_prob >= current_prob + countertrend_margin
                )
            )
            if pos_steps < float(self.min_hold_steps):
                strong_reverse = reverse_signal and opposite_prob >= current_prob + reversal_margin + 0.15
                protective_flat = (
                    upnl <= min(stop_loss_threshold * 0.50, -0.0050)
                )
                if strong_reverse:
                    return opposite_action, None
                elif protective_flat:
                    return ACTION_FLAT, None
                return current_action, None
            if max_hold_steps > 0 and pos_steps >= max_hold_steps:
                stale_signal = flat_prob >= current_prob - min(exit_to_flat_margin, 0.03)
                weak_hold = (
                    current_prob < max(current_threshold - 0.08, 0.18)
                    or current_edge <= meta_label_exit_edge
                )
                if stale_signal or weak_hold:
                    return ACTION_FLAT, None
            flatten_signal = (
                current_edge <= meta_label_exit_edge
                or current_positive_rate < 0.48
                or flat_prob >= current_prob + exit_to_flat_margin
                or (
                    current_prob < max(current_threshold - 0.05, 0.20)
                    and flat_prob >= current_prob - 0.02
                )
                or (
                    upnl < -0.0025
                    and flat_prob >= current_prob - 0.04
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
                and long_prob >= flat_prob + long_entry_margin
                and long_regime_ok
                and long_edge_ok
            )
            short_signal = (
                short_prob >= short_confidence_threshold
                and short_prob > long_prob
                and short_prob >= flat_prob + short_entry_margin
                and short_regime_ok
                and short_edge_ok
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
            "meta_label_min_edge": self.meta_label_min_edge,
            "meta_label_edge_margin": self.meta_label_edge_margin,
            "meta_label_exit_edge": self.meta_label_exit_edge,
            "meta_label_min_positive_rate": self.meta_label_min_positive_rate,
            "calibration_min_active_window_ratio": self.calibration_min_active_window_ratio,
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
            meta_label_min_edge=float(payload.get("meta_label_min_edge", 0.0)),
            meta_label_edge_margin=float(payload.get("meta_label_edge_margin", 0.0)),
            meta_label_exit_edge=float(payload.get("meta_label_exit_edge", 0.0)),
            meta_label_min_positive_rate=float(payload.get("meta_label_min_positive_rate", 0.47)),
            calibration_min_active_window_ratio=float(payload.get("calibration_min_active_window_ratio", 0.50)),
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
    logistic_multi_class: str = "ovr",
    logistic_target_mode: str = "binary_meta",
    catboost_iterations: int = 256,
    catboost_depth: int = 6,
    catboost_learning_rate: float = 0.05,
    catboost_l2_leaf_reg: float = 3.0,
    catboost_border_count: int = 128,
    catboost_task_type: str = "CPU",
    catboost_devices: str = "0",
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
    flat_patience_steps: int = 0,
    flat_patience_threshold_relief: float = 0.0,
    flat_patience_entry_margin_relief: float = 0.0,
    countertrend_threshold_penalty: float = 0.0,
    countertrend_entry_penalty: float = 0.0,
    meta_label_min_edge: float = 0.0,
    meta_label_edge_margin: float = 0.0,
    meta_label_exit_edge: float = 0.0,
    meta_label_min_positive_rate: float = 0.47,
    meta_label_target_edge_threshold: float | None = None,
    calibration_min_samples: int = 64,
    calibration_probability_thresholds: list[float] | tuple[float, ...] | np.ndarray | None = None,
    calibration_window_count: int = 4,
    calibration_window_min_samples: int | None = None,
    calibration_min_active_window_ratio: float = 0.50,
    random_state: int = 42,
    dashboard=None,
) -> tuple[SupervisedActionModel, dict]:
    if dashboard:
        dashboard.update(status_msg=f"Training Supervised Model ({model_type})...")
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
    target_mode = str(logistic_target_mode).strip().lower()
    if model_type == "trend_rule":
        target_mode = "trend_rule"
    elif target_mode not in {"binary_meta", "multiclass"}:
        target_mode = "binary_meta"

    x_valid = np.asarray(feature_array[valid_idx], dtype=np.float32)
    if model_type == "logreg":
        scaler = StandardScaler()
        x_train_fit = scaler.fit_transform(x_train)
        x_valid_eval = scaler.transform(x_valid)
    elif model_type == "catboost":
        x_train_fit = x_train
        x_valid_eval = x_valid
    elif model_type == "extratrees":
        x_train_fit = x_train
        x_valid_eval = x_valid
    elif model_type == "trend_rule":
        x_train_fit = x_train
        x_valid_eval = x_valid
    else:
        raise ValueError(f"Unsupported supervised model_type: {model_type}")

    def _make_estimator(seed_offset: int = 0):
        if model_type == "logreg":
            base = LogisticRegression(
                max_iter=400,
                solver="lbfgs",
                class_weight="balanced",
                C=float(logistic_c),
                random_state=int(random_state + seed_offset),
            )
            if target_mode == "multiclass":
                multi_class_mode = str(logistic_multi_class).strip().lower()
                if multi_class_mode not in {"ovr", "multinomial", "auto"}:
                    multi_class_mode = "ovr"
                return OneVsRestClassifier(base) if multi_class_mode == "ovr" else base
            return base
        if model_type == "catboost":
            catboost_mode = str(catboost_task_type).strip().upper() or "CPU"
            params = {
                "iterations": max(int(catboost_iterations), 32),
                "depth": max(int(catboost_depth), 2),
                "learning_rate": float(catboost_learning_rate),
                "l2_leaf_reg": float(catboost_l2_leaf_reg),
                "border_count": max(int(catboost_border_count), 32),
                "random_seed": int(random_state + seed_offset),
                "verbose": False,
                "allow_writing_files": False,
                "thread_count": -1,
                "auto_class_weights": "Balanced",
                "task_type": catboost_mode,
            }
            if catboost_mode == "GPU":
                params["devices"] = str(catboost_devices)
            if target_mode == "multiclass":
                params["loss_function"] = "MultiClass"
                params["eval_metric"] = "MultiClass"
            else:
                params["loss_function"] = "Logloss"
                params["eval_metric"] = "Logloss"
            return CatBoostClassifier(**params)
        return ExtraTreesClassifier(
            n_estimators=max(int(extra_trees_n_estimators), 64),
            max_depth=max(int(extra_trees_max_depth), 2),
            min_samples_leaf=max(int(extra_trees_min_samples_leaf), 1),
            class_weight="balanced_subsample",
            max_features="sqrt",
            n_jobs=-1,
            random_state=int(random_state + seed_offset),
        )

    sample_weights = 1.0 + np.clip(
        np.abs(train_future_returns) / max(effective_threshold, 1e-6),
        0.0,
        4.0,
    )

    edge_target_threshold = (
        float(meta_label_target_edge_threshold)
        if meta_label_target_edge_threshold is not None
        else max(min(float(min_return_threshold), float(round_trip_cost)) * 0.25, 0.0001)
    )
    train_long_targets, train_short_targets, train_long_edges, train_short_edges = _build_binary_edge_targets(
        train_future_returns,
        round_trip_cost=round_trip_cost,
        edge_threshold=edge_target_threshold,
    )
    valid_long_targets, valid_short_targets, valid_long_edges, valid_short_edges = _build_binary_edge_targets(
        valid_future_returns,
        round_trip_cost=round_trip_cost,
        edge_threshold=edge_target_threshold,
    )

    if model_type == "trend_rule":
        trend_entry_threshold = max(
            float(adaptive_threshold) * 0.75,
            float(round_trip_cost) * 1.25,
            0.0008,
        )
        trend_neutral_band = max(float(adaptive_threshold) * 0.35, 0.0004)
        classifier = BinaryEdgeActionClassifier(
            long_classifier=TrendRuleBinaryClassifier(
                feature_dim=feature_dim,
                positive_direction=ACTION_LONG,
                entry_threshold=trend_entry_threshold,
                neutral_band=trend_neutral_band,
            ),
            short_classifier=TrendRuleBinaryClassifier(
                feature_dim=feature_dim,
                positive_direction=ACTION_SHORT,
                entry_threshold=trend_entry_threshold,
                neutral_band=trend_neutral_band,
            ),
        )
    elif target_mode == "binary_meta":
        def _fit_binary_head(targets: np.ndarray, edges: np.ndarray, *, seed_offset: int):
            if np.unique(targets).size < 2:
                positive_prob = float(np.mean(targets)) if targets.size else 0.0
                return ConstantBinaryClassifier(positive_prob=positive_prob)
            estimator = _make_estimator(seed_offset=seed_offset)
            binary_weights = 1.0 + np.clip(
                np.abs(edges) / max(edge_target_threshold + round_trip_cost, 1e-6),
                0.0,
                4.0,
            )
            fit_task_type = (
                str(catboost_task_type).strip().upper()
                if model_type == "catboost"
                else "CPU"
            )
            fit_devices = str(catboost_devices) if model_type == "catboost" else "-"
            fit_start = perf_counter()
            if isinstance(estimator, OneVsRestClassifier):
                estimator.fit(x_train_fit, targets)
            else:
                estimator.fit(x_train_fit, targets, sample_weight=binary_weights)
            fit_seconds = perf_counter() - fit_start
            logger.info(
                "Supervised fit head complete: model=%s mode=%s task_type=%s devices=%s "
                "samples=%d features=%d duration=%.2fs",
                model_type,
                str(target_mode),
                fit_task_type,
                fit_devices,
                int(len(targets)),
                int(x_train_fit.shape[1]) if np.ndim(x_train_fit) == 2 else 0,
                float(fit_seconds),
            )
            return estimator

        long_classifier = _fit_binary_head(train_long_targets, train_long_edges, seed_offset=11)
        short_classifier = _fit_binary_head(train_short_targets, train_short_edges, seed_offset=29)
        classifier = BinaryEdgeActionClassifier(
            long_classifier=long_classifier,
            short_classifier=short_classifier,
        )
    else:
        classifier = _make_estimator(seed_offset=0)
        fit_task_type = (
            str(catboost_task_type).strip().upper()
            if model_type == "catboost"
            else "CPU"
        )
        fit_devices = str(catboost_devices) if model_type == "catboost" else "-"
        fit_start = perf_counter()
        if isinstance(classifier, OneVsRestClassifier):
            classifier.fit(x_train_fit, train_labels)
        else:
            classifier.fit(x_train_fit, train_labels, sample_weight=sample_weights)
        fit_seconds = perf_counter() - fit_start
        logger.info(
            "Supervised fit complete: model=%s mode=%s task_type=%s devices=%s "
            "samples=%d features=%d duration=%.2fs",
            model_type,
            str(target_mode),
            fit_task_type,
            fit_devices,
            int(len(train_labels)),
            int(x_train_fit.shape[1]) if np.ndim(x_train_fit) == 2 else 0,
            float(fit_seconds),
        )

    metadata = {
        "horizon": int(horizon),
        "label_threshold": float(effective_threshold),
        "signal_threshold": float(adaptive_threshold),
        "round_trip_cost": float(round_trip_cost),
        "classifier_mode": str(target_mode),
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
        "logistic_multi_class": str(logistic_multi_class).strip().lower(),
        "logistic_target_mode": str(target_mode),
        "catboost_iterations": int(max(catboost_iterations, 0)),
        "catboost_depth": int(max(catboost_depth, 0)),
        "catboost_learning_rate": float(catboost_learning_rate),
        "catboost_l2_leaf_reg": float(catboost_l2_leaf_reg),
        "catboost_border_count": int(max(catboost_border_count, 0)),
        "catboost_task_type": str(catboost_task_type).strip().upper(),
        "catboost_devices": str(catboost_devices),
        "binary_target_edge_threshold": float(edge_target_threshold),
        "binary_target_counts": {
            "train_long_positive": int(np.sum(train_long_targets)),
            "train_short_positive": int(np.sum(train_short_targets)),
            "validation_long_positive": int(np.sum(valid_long_targets)),
            "validation_short_positive": int(np.sum(valid_short_targets)),
        },
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
        "flat_patience_steps": int(max(flat_patience_steps, 0)),
        "flat_patience_threshold_relief": float(max(flat_patience_threshold_relief, 0.0)),
        "flat_patience_entry_margin_relief": float(max(flat_patience_entry_margin_relief, 0.0)),
        "countertrend_threshold_penalty": float(max(countertrend_threshold_penalty, 0.0)),
        "countertrend_entry_penalty": float(max(countertrend_entry_penalty, 0.0)),
    }
    if model_type == "trend_rule":
        metadata["trend_rule_entry_threshold"] = float(trend_entry_threshold)
        metadata["trend_rule_neutral_band"] = float(trend_neutral_band)

    aligned_valid_proba_raw, aligned_valid_proba = _align_action_probability_views(classifier, x_valid_eval)
    valid_one_hot = np.eye(3, dtype=np.float64)[valid_labels.astype(np.int32)]
    metadata["validation_brier"] = float(np.mean(np.sum((aligned_valid_proba - valid_one_hot) ** 2, axis=1)))
    metadata["validation_brier_long"] = float(np.mean((aligned_valid_proba_raw[:, ACTION_LONG] - valid_long_targets) ** 2))
    metadata["validation_brier_short"] = float(np.mean((aligned_valid_proba_raw[:, ACTION_SHORT] - valid_short_targets) ** 2))
    metadata["validation_label_abs_mean"] = float(np.mean(np.abs(valid_future_returns))) if len(valid_future_returns) else 0.0
    metadata["post_cost_calibration"] = _build_post_cost_calibration(
        aligned_valid_proba_raw,
        valid_future_returns,
        round_trip_cost,
        probability_thresholds=calibration_probability_thresholds,
        min_samples=int(calibration_min_samples),
        window_count=int(calibration_window_count),
        window_min_samples=calibration_window_min_samples,
    )
    metadata["meta_label_min_edge"] = float(max(meta_label_min_edge, 0.0))
    metadata["meta_label_edge_margin"] = float(max(meta_label_edge_margin, 0.0))
    metadata["meta_label_exit_edge"] = float(meta_label_exit_edge)
    metadata["meta_label_min_positive_rate"] = float(np.clip(meta_label_min_positive_rate, 0.0, 1.0))
    metadata["calibration_min_active_window_ratio"] = float(max(calibration_min_active_window_ratio, 0.0))

    logger.info(
        "Supervised fallback dataset: model=%s mode=%s horizon=%d signal_threshold=%.4f effective_threshold=%.4f "
        "round_trip_cost=%.4f edge_target=%.4f train=%d valid=%d brier=%.4f meta_label[min_edge=%.4f margin=%.4f exit=%.4f min_pos=%.2f] "
        "train_labels={short:%d, flat:%d, long:%d} binary_targets={long:%d, short:%d}",
        model_type,
        str(target_mode),
        int(horizon),
        float(adaptive_threshold),
        float(effective_threshold),
        float(round_trip_cost),
        float(edge_target_threshold),
        int(metadata["train_samples"]),
        int(metadata["validation_samples"]),
        float(metadata["validation_brier"]),
        float(metadata["meta_label_min_edge"]),
        float(metadata["meta_label_edge_margin"]),
        float(metadata["meta_label_exit_edge"]),
        float(metadata["meta_label_min_positive_rate"]),
        int(metadata["train_class_counts"]["short"]),
        int(metadata["train_class_counts"]["flat"]),
        int(metadata["train_class_counts"]["long"]),
        int(metadata["binary_target_counts"]["train_long_positive"]),
        int(metadata["binary_target_counts"]["train_short_positive"]),
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
        meta_label_min_edge=max(float(meta_label_min_edge), 0.0),
        meta_label_edge_margin=max(float(meta_label_edge_margin), 0.0),
        meta_label_exit_edge=float(meta_label_exit_edge),
        meta_label_min_positive_rate=float(np.clip(meta_label_min_positive_rate, 0.0, 1.0)),
        calibration_min_active_window_ratio=float(max(calibration_min_active_window_ratio, 0.0)),
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
        meta_label_min_edge=0.0,
        meta_label_edge_margin=0.0,
        meta_label_exit_edge=0.0,
        meta_label_min_positive_rate=0.0,
    )
