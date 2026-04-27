"""Main pipeline โ€” เธฃเธงเธกเธ—เธธเธ component เน€เธเนเธฒเธ”เนเธงเธขเธเธฑเธ.

เนเธเนเนเธ”เนเธ—เธฑเนเธ:
1. train: historical data โ’ features โ’ train RL โ’ backtest โ’ validate
2. test:  subsample data โ’ features โ’ train RL (เน€เธฃเนเธง) โ’ backtest โ’ validate เธ—เธธเธ component
3. paper: live data โ’ features โ’ model โ’ paper orders
4. live:  live data โ’ features โ’ model โ’ risk โ’ real orders

*** เธ—เธธเธ mode เนเธเน code path เน€เธ”เธตเธขเธงเธเธฑเธ ***
*** Action เน€เธเธเธฒเธฐ candle close เน€เธ—เนเธฒเธเธฑเนเธ ***

Usage:
  python pipeline.py --mode test                           # เธ—เธ”เธชเธญเธเธ—เธฑเนเธเธฃเธฐเธเธ (RTX 2060)
  python pipeline.py --mode test --config configs/test_rtx2060.yaml
  python pipeline.py --mode train                          # full training
  python pipeline.py --mode train --config configs/default.yaml
"""

import logging
import time
import sys
import copy
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from configs import load_config
from data.quality import clean_pipeline
from features.builder import FeatureBuilder
from models.forecast.naive import NaiveForecaster
from models.rl.environment import build_agent_state
from execution.backtest.runner import BacktestRunner, BacktestConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Helper: GPU info
# =============================================================================

def _log_gpu_info():
    """Log GPU info if available."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU: {name} ({vram:.1f} GB VRAM)")
            logger.info(f"CUDA: {torch.version.cuda}")
            return True
        else:
            logger.warning("No CUDA GPU detected โ€” will use CPU (slower)")
            return False
    except ImportError:
        logger.warning("PyTorch not installed โ€” will use CPU")
        return False


def _log_gpu_memory():
    """Log current GPU memory usage."""
    try:
        import torch
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated(0) / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU Memory: {used:.2f}/{total:.1f} GB")
    except Exception:
        pass


# =============================================================================
# Phase 1: Load & Clean Data
# =============================================================================

def load_and_clean_data(config: dict, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """Load OHLCV + optional funding/sentiment data, then clean."""
    data_config = config["data"]
    raw_dir = Path(data_config["paths"]["raw"])

    # OHLCV
    ohlcv_path = raw_dir / f"{symbol}_1m.parquet"
    if not ohlcv_path.exists():
        logger.info(f"Data not found. Downloading {symbol}...")
        from data.downloaders.binance_historical import download_range
        from datetime import date
        download_range(symbol, "1m", date(2020, 1, 1), output_dir=str(raw_dir))
        
        if not ohlcv_path.exists():
            raise FileNotFoundError(f"Data not found: {ohlcv_path} โ€” run downloaders first")

    df = pd.read_parquet(ohlcv_path)
    logger.info(f"Loaded {symbol}: {len(df):,} rows")

    # Subsample if configured (for test mode)
    subsample = data_config.get("subsample_rows")
    if subsample and subsample < len(df):
        df = df.tail(subsample).reset_index(drop=True)
        logger.info(f"Subsampled to last {subsample:,} rows")

    # Clean
    df = clean_pipeline(df, zscore_threshold=data_config["quality"]["zscore_threshold"])
    logger.info(f"After cleaning: {len(df):,} rows")

    # Funding rate (optional)
    funding_df = None
    funding_path = raw_dir / f"{symbol}_funding_rate.parquet"
    if funding_path.exists():
        funding_df = pd.read_parquet(funding_path)
        logger.info(f"Loaded funding rate: {len(funding_df):,} rows")

    # Sentiment (optional)
    sentiment_df = None
    sentiment_path = raw_dir / "fear_greed_index.parquet"
    if sentiment_path.exists():
        sentiment_df = pd.read_parquet(sentiment_path)
        logger.info(f"Loaded sentiment: {len(sentiment_df):,} rows")

    return df, funding_df, sentiment_df


# =============================================================================
# Phase 2: Build Features
# =============================================================================

def build_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build feature arrays from cleaned OHLCV data.

    Returns: (feature_array, ta_slice, micro_slice, prices)
    """
    builder = FeatureBuilder(lookback=60)
    feature_array, ta_slice, micro_slice = builder.build_batch_array(df)

    if len(feature_array) == 0:
        raise ValueError("No feature vectors generated")

    prices = df["close"].values[60:]  # align with features
    logger.info(f"Features: {feature_array.shape}, Prices: {prices.shape}")

    return feature_array, ta_slice, micro_slice, prices


# =============================================================================
# Phase 3: Naive Forecast (baseline)
# =============================================================================

def add_naive_forecast(feature_array: np.ndarray, prices: np.ndarray) -> np.ndarray:
    """Append a compact causal naive-forecast block to the feature array."""
    forecaster = NaiveForecaster()
    forecast_feats = np.zeros((len(prices), 5), dtype=np.float32)

    for i in range(60, len(prices)):
        price_window = prices[max(0, i - 60):i]
        if len(price_window) > 1:
            forecast, uncertainty = forecaster.predict(price_window, horizon=4)
            last_price = max(float(price_window[-1]), 1e-9)
            usable = min(4, len(forecast))
            forecast_feats[i, :usable] = (np.asarray(forecast[:usable], dtype=np.float32) / last_price) - 1.0
            forecast_feats[i, 4] = float(uncertainty)

    return np.concatenate([feature_array, forecast_feats], axis=1).astype(np.float32)


def _compute_data_ranges(
    total_len: int,
    test_ratio: float = 0.20,
    validation_ratio_within_train: float = 0.10,
) -> dict[str, tuple[int, int]]:
    """Return exclusive [start, end) ranges for train/validation/test."""
    total_len = max(int(total_len), 0)
    if total_len <= 2:
        return {
            "train": (0, total_len),
            "validation": (0, total_len),
            "test": (0, total_len),
        }

    test_ratio = float(np.clip(test_ratio, 0.05, 0.45))
    validation_ratio_within_train = float(np.clip(validation_ratio_within_train, 0.05, 0.40))

    test_len = max(int(round(total_len * test_ratio)), 1)
    test_len = min(test_len, total_len - 2)
    train_val_end = total_len - test_len

    validation_len = max(int(round(train_val_end * validation_ratio_within_train)), 1)
    validation_len = min(validation_len, train_val_end - 1)

    train_end = max(train_val_end - validation_len, 1)
    return {
        "train": (0, train_end),
        "validation": (train_end, train_val_end),
        "test": (train_val_end, total_len),
    }


def _robust_validation_score(scores: list[float] | tuple[float, ...]) -> float:
    """Aggregate validation scores conservatively across multiple seed runs."""
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return float("-inf")
    median = float(np.median(arr))
    worst = float(np.min(arr))
    return float(median + 0.25 * worst)


def _aggregate_validation_metrics(metric_runs: list[dict], scores: list[float] | tuple[float, ...]) -> dict:
    """Summarize repeated validation runs into a conservative representative metrics dict."""
    if not metric_runs:
        return {}

    aggregated = dict(metric_runs[0])
    numeric_keys = set()
    for run in metric_runs:
        numeric_keys.update(
            key for key, value in run.items()
            if isinstance(value, (int, float, np.integer, np.floating))
        )

    for key in numeric_keys:
        values = [
            float(run[key]) for run in metric_runs
            if isinstance(run.get(key), (int, float, np.integer, np.floating))
        ]
        if len(values) != len(metric_runs):
            continue
        arr = np.asarray(values, dtype=np.float64)
        if key in {"eval_dominant_action_ratio", "flat_ratio", "max_drawdown"}:
            aggregated[key] = float(np.max(arr))
        else:
            aggregated[key] = float(np.median(arr))

    score_arr = np.asarray(scores, dtype=np.float64)
    finite_mask = np.isfinite(score_arr)
    aggregated["validation_seed_runs"] = int(len(metric_runs))
    aggregated["validation_seed_all_finite"] = 1.0 if bool(np.all(finite_mask)) else 0.0
    if np.any(finite_mask):
        finite_scores = score_arr[finite_mask]
        aggregated["validation_score_median"] = float(np.median(finite_scores))
        aggregated["validation_score_worst"] = float(np.min(finite_scores))
    else:
        aggregated["validation_score_median"] = float("-inf")
        aggregated["validation_score_worst"] = float("-inf")
    aggregated["validation_score_robust"] = _robust_validation_score(list(scores))
    return aggregated


def _combine_supervised_validation_scores(
    episodic_score: float,
    walkforward_metrics: dict,
    trainer,
    *,
    max_dominant_action_ratio: float,
    min_avg_trades_per_episode: float,
    min_action_entropy: float,
    walkforward_score_weight: float = 0.20,
    walkforward_min_active_ratio: float = 0.10,
    walkforward_inactivity_penalty_scale: float = 0.10,
    walkforward_worst_net_penalty_scale: float = 0.10,
    walkforward_worst_gross_penalty_scale: float = 0.10,
    walkforward_worst_alpha_penalty_scale: float = 0.10,
    walkforward_dominance_penalty_scale: float = 0.10,
) -> tuple[float, float, dict]:
    """Blend episodic validation with walk-forward stress testing without hard-failing on one flat window."""
    if not np.isfinite(episodic_score):
        return float("-inf"), float("-inf"), {}

    walkforward_soft_score = float(
        trainer.score_candidate(
            walkforward_metrics,
            max_dominant_action_ratio=1.0,
            min_avg_trades_per_episode=0.0,
            min_action_entropy=0.0,
        )
    )
    if not np.isfinite(walkforward_soft_score):
        walkforward_soft_score = -2.0

    active_ratio = float(walkforward_metrics.get("walkforward_active_window_ratio", 0.0))
    worst_net = float(walkforward_metrics.get("walkforward_min_total_return", walkforward_metrics.get("total_return", 0.0)))
    worst_gross = float(
        walkforward_metrics.get("walkforward_min_gross_total_return", walkforward_metrics.get("gross_total_return", 0.0))
    )
    worst_alpha = float(walkforward_metrics.get("walkforward_min_alpha", walkforward_metrics.get("outperformance_vs_bh", 0.0)))
    worst_dom_ratio = float(
        walkforward_metrics.get(
            "walkforward_worst_dominant_action_ratio",
            walkforward_metrics.get("eval_dominant_action_ratio", 1.0),
        )
    )
    median_trades = float(
        walkforward_metrics.get(
            "walkforward_median_trades",
            walkforward_metrics.get("avg_trades_per_episode", 0.0),
        )
    )

    if active_ratio <= 0.0 and median_trades < max(min_avg_trades_per_episode * 0.5, 0.5):
        return float("-inf"), walkforward_soft_score, {
            "walkforward_active_window_ratio": active_ratio,
            "walkforward_soft_score": walkforward_soft_score,
            "walkforward_penalty": float("inf"),
            "walkforward_bonus": 0.0,
        }

    penalty = 0.0
    penalty += max(walkforward_min_active_ratio - active_ratio, 0.0) * walkforward_inactivity_penalty_scale
    penalty += max(-worst_net, 0.0) * walkforward_worst_net_penalty_scale
    penalty += max(-worst_gross, 0.0) * walkforward_worst_gross_penalty_scale
    penalty += max(-worst_alpha, 0.0) * walkforward_worst_alpha_penalty_scale
    penalty += max(worst_dom_ratio - max_dominant_action_ratio, 0.0) * walkforward_dominance_penalty_scale

    bonus = active_ratio * 0.50
    bonus += max(float(walkforward_metrics.get("walkforward_positive_net_ratio", 0.0)) - 0.50, 0.0) * 0.25
    bonus += max(float(walkforward_metrics.get("walkforward_positive_alpha_ratio", 0.0)) - 0.50, 0.0) * 0.15

    combined_score = (
        float(episodic_score)
        + (float(np.clip(walkforward_soft_score, -5.0, 5.0)) * walkforward_score_weight)
        + bonus
        - penalty
    )
    details = {
        "walkforward_active_window_ratio": active_ratio,
        "walkforward_soft_score": float(walkforward_soft_score),
        "walkforward_penalty": float(penalty),
        "walkforward_bonus": float(bonus),
    }
    return float(combined_score), float(walkforward_soft_score), details


def _score_sparse_supervised_watchlist_candidate(metrics: dict) -> float:
    """Soft score for sparse candidates that merit realistic Nautilus validation."""
    net_return = float(metrics.get("total_return", 0.0))
    gross_return = float(metrics.get("gross_total_return", net_return))
    alpha = float(metrics.get("outperformance_vs_bh", 0.0))
    max_drawdown = abs(float(metrics.get("max_drawdown", 0.0)))
    avg_trades = float(metrics.get("avg_trades_per_episode", metrics.get("n_trades", 0.0)))
    action_entropy = float(metrics.get("eval_action_entropy", 0.0))
    dominant_action_ratio = float(metrics.get("eval_dominant_action_ratio", 1.0))
    flat_ratio = float(metrics.get("flat_ratio", 1.0))
    worst_net_return = float(metrics.get("walkforward_min_total_return", net_return))
    worst_gross_return = float(metrics.get("walkforward_min_gross_total_return", gross_return))
    worst_alpha = float(metrics.get("walkforward_min_alpha", alpha))
    active_ratio = float(metrics.get("walkforward_active_window_ratio", 0.0))
    positive_net_ratio = float(metrics.get("walkforward_positive_net_ratio", 0.0))
    positive_alpha_ratio = float(metrics.get("walkforward_positive_alpha_ratio", 0.0))
    brier = float(metrics.get("supervised_validation_brier", float("inf")))
    trade_density = float(metrics.get("trade_density", 0.0))

    if avg_trades < 0.5 and active_ratio <= 0.0:
        return float("-inf")
    if (
        gross_return <= 0.0
        and worst_gross_return <= 0.0
        and positive_net_ratio <= 0.0
        and positive_alpha_ratio < 0.5
    ):
        return float("-inf")
    if net_return <= -0.01 and worst_net_return <= -0.01 and positive_net_ratio <= 0.0:
        return float("-inf")

    turnover_penalty = max(avg_trades - 4.0, 0.0) * 0.05
    turnover_penalty += max(avg_trades - 8.0, 0.0) * 0.08
    turnover_penalty += max(trade_density - 0.01, 0.0) * 8.0
    turnover_penalty += max(trade_density - 0.02, 0.0) * 14.0
    dominance_penalty = max(dominant_action_ratio - 0.95, 0.0) * 6.0
    flat_penalty = max(flat_ratio - 0.985, 0.0) * 1.5
    flat_penalty += max(flat_ratio - 0.995, 0.0) * 4.0
    score = (
        net_return * 8.0
        + gross_return * 10.0
        + alpha * 2.0
        + worst_net_return * 4.0
        + worst_gross_return * 5.0
        + worst_alpha * 1.0
        + active_ratio * 0.75
        + positive_net_ratio * 0.60
        + positive_alpha_ratio * 0.40
        - max_drawdown * 5.0
        - turnover_penalty
        - dominance_penalty
        - flat_penalty
        + action_entropy * 0.10
    )
    if np.isfinite(brier):
        score -= max(brier - 0.75, 0.0) * 0.25
    return float(score)


def _score_anchor_supervised_probe_candidate(metrics: dict) -> float:
    """Allow very conservative high-threshold candidates to reach Nautilus once."""
    net_return = float(metrics.get("total_return", 0.0))
    gross_return = float(metrics.get("gross_total_return", net_return))
    alpha = float(metrics.get("outperformance_vs_bh", 0.0))
    avg_trades = float(metrics.get("avg_trades_per_episode", metrics.get("n_trades", 0.0)))
    dominant_action_ratio = float(metrics.get("eval_dominant_action_ratio", 1.0))
    flat_ratio = float(metrics.get("flat_ratio", 1.0))
    max_drawdown = abs(float(metrics.get("max_drawdown", 0.0)))
    worst_net_return = float(metrics.get("walkforward_min_total_return", net_return))
    worst_gross_return = float(metrics.get("walkforward_min_gross_total_return", gross_return))

    if gross_return < -0.0025 or worst_gross_return < -0.0040:
        return float("-inf")
    if net_return < -0.0040 or worst_net_return < -0.0080:
        return float("-inf")
    # Was: avg_trades > 2.0 → -inf. That cap was too tight: it excluded
    # legitimately active anchor probes whose Nautilus validation might
    # still be informative. 8 trades / episode keeps the "conservative
    # anchor" intent without hard-blocking selective active models.
    if avg_trades > 8.0:
        return float("-inf")

    return float(
        gross_return * 8.0
        + net_return * 6.0
        + alpha * 1.5
        - max_drawdown * 3.0
        - max(flat_ratio - 0.999, 0.0) * 0.5
        - max(dominant_action_ratio - 0.995, 0.0) * 2.0
        - max(avg_trades - 1.0, 0.0) * 0.10
    )


def _is_collapsed_supervised_probe_metrics(metrics: dict) -> bool:
    """Return True when a supervised probe is effectively flat/collapsed."""
    avg_trades = float(metrics.get("avg_trades_per_episode", metrics.get("n_trades", 0.0)))
    flat_ratio = float(metrics.get("flat_ratio", 1.0))
    dominant_ratio = float(metrics.get("eval_dominant_action_ratio", 1.0))
    action_entropy = float(metrics.get("eval_action_entropy", 0.0))
    position_ratio = float(metrics.get("position_ratio", 0.0))
    gross_return = float(metrics.get("gross_total_return", metrics.get("total_return", 0.0)))
    return bool(
        avg_trades <= 0.10
        and flat_ratio >= 0.999
        and dominant_ratio >= 0.999
        and action_entropy <= 0.01
        and position_ratio <= 0.001
        and abs(gross_return) <= 0.0025
    )


def _derive_calibrated_confidence_grid(
    side_calibration: dict | None,
    fallback_grid: list[float] | tuple[float, ...],
    *,
    min_edge: float,
    min_positive_rate: float,
    min_active_window_ratio: float = 0.50,
    respect_fallback_floor: bool = True,
    anchor_thresholds: list[float] | tuple[float, ...] | None = None,
    preserve_top_fallback_count: int = 2,
    top_k: int = 3,
) -> list[float]:
    fallback = sorted({round(float(v), 2) for v in fallback_grid if np.isfinite(v)})
    fallback_floor = fallback[0] if fallback else 0.0
    anchors = sorted(
        {
            round(float(v), 2)
            for v in (anchor_thresholds or [])
            if np.isfinite(v)
        }
    )
    if respect_fallback_floor and fallback:
        anchors = [value for value in anchors if value + 1e-6 >= fallback_floor]
    if not isinstance(side_calibration, dict):
        return sorted(set(fallback).union(anchors))

    threshold_rows = side_calibration.get("thresholds", [])
    if not isinstance(threshold_rows, list):
        return fallback

    scored: list[tuple[float, float]] = []
    relaxed_positive_rate = max(float(min_positive_rate) - 0.05, 0.40)
    relaxed_min_edge = min(float(min_edge), 0.0)
    for row in threshold_rows:
        threshold = round(float(row.get("threshold", 0.0)), 2)
        if respect_fallback_floor and fallback and threshold + 1e-6 < fallback_floor:
            continue
        mean_edge = float(row.get("robust_mean_edge", row.get("mean_edge", 0.0)))
        positive_rate = float(row.get("robust_positive_rate", row.get("positive_rate", 0.0)))
        active_window_ratio = float(row.get("active_window_ratio", 1.0))
        count = int(row.get("count", 0))
        if (
            mean_edge < relaxed_min_edge
            or positive_rate < relaxed_positive_rate
            or active_window_ratio < min_active_window_ratio
        ):
            continue
        quality = (
            mean_edge * 1200.0
            + positive_rate * 1.25
            + active_window_ratio * 0.45
            + threshold * 0.35
            - float(row.get("edge_std", 0.0)) * 4.0
            + np.log1p(max(count, 1)) * 0.03
        )
        scored.append((quality, threshold))

    if not scored:
        for row in threshold_rows:
            threshold = round(float(row.get("threshold", 0.0)), 2)
            if respect_fallback_floor and fallback and threshold + 1e-6 < fallback_floor:
                continue
            mean_edge = float(row.get("robust_mean_edge", row.get("mean_edge", 0.0)))
            positive_rate = float(row.get("robust_positive_rate", row.get("positive_rate", 0.0)))
            active_window_ratio = float(row.get("active_window_ratio", 1.0))
            count = int(row.get("count", 0))
            quality = (
                mean_edge * 800.0
                + positive_rate * 1.0
                + active_window_ratio * 0.35
                + threshold * 0.30
                - float(row.get("edge_std", 0.0)) * 3.0
                + np.log1p(max(count, 1)) * 0.02
            )
            scored.append((quality, threshold))

    selected = [threshold for _, threshold in sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]]
    if not selected:
        return sorted(set(fallback).union(anchors))

    merged = set(selected)
    fallback_lookup = {value: idx for idx, value in enumerate(fallback)}
    for threshold in selected:
        idx = fallback_lookup.get(threshold)
        if idx is None:
            continue
        if idx - 1 >= 0:
            merged.add(fallback[idx - 1])
        if idx + 1 < len(fallback):
            merged.add(fallback[idx + 1])

    if fallback:
        preserve_top_fallback_count = max(int(preserve_top_fallback_count), 0)
        if preserve_top_fallback_count > 0:
            merged.update(fallback[-preserve_top_fallback_count:])
        else:
            merged.add(fallback[-1])
    merged.update(anchors)
    return sorted(merged)


def _build_supervised_search_grid(
    calibrated_grid: list[float] | tuple[float, ...],
    *,
    exploration_grid: list[float] | tuple[float, ...] | None = None,
    anchor_grid: list[float] | tuple[float, ...] | None = None,
) -> list[float]:
    values = {
        round(float(v), 2)
        for v in list(calibrated_grid or [])
        if np.isfinite(v)
    }
    values.update(
        round(float(v), 2)
        for v in list(exploration_grid or [])
        if np.isfinite(v)
    )
    values.update(
        round(float(v), 2)
        for v in list(anchor_grid or [])
        if np.isfinite(v)
    )
    return sorted(values)


def _build_nautilus_frame(
    prices: np.ndarray,
    ohlcv_data: np.ndarray | None = None,
    timestamps: np.ndarray | None = None,
) -> pd.DataFrame:
    n = len(prices)
    if timestamps is None or len(timestamps) != n:
        timestamps = pd.date_range("2020-01-01", periods=n, freq="15min", tz="UTC")
    else:
        timestamps = pd.to_datetime(timestamps, utc=True)

    if ohlcv_data is not None and len(ohlcv_data) == n:
        ohlcv = np.asarray(ohlcv_data, dtype=np.float64)
        open_col = ohlcv[:, 0]
        high_col = ohlcv[:, 1]
        low_col = ohlcv[:, 2]
        close_col = ohlcv[:, 3]
        volume_col = ohlcv[:, 4] if ohlcv.shape[1] >= 5 else np.ones(n, dtype=np.float64)
    else:
        close_col = np.asarray(prices, dtype=np.float64)
        open_col = close_col.copy()
        high_col = close_col.copy()
        low_col = close_col.copy()
        volume_col = np.ones(n, dtype=np.float64)

    frame = pd.DataFrame(
        {
            "open_time": timestamps,
            "open": open_col,
            "high": high_col,
            "low": low_col,
            "close": close_col,
            "volume": volume_col,
        }
    )
    return frame.reset_index(drop=True)


def _prepare_nautilus_segment_frame(
    frame_15m: pd.DataFrame,
    segment_range: tuple[int, int],
    *,
    history_bars: int,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int, int]:
    start, end = int(segment_range[0]), int(segment_range[1])
    if end <= start:
        raise ValueError(f"Invalid Nautilus segment range: [{start}:{end})")
    warmup_bars = max(int(history_bars) - 1, 0)
    padded_start = max(0, start - warmup_bars)
    prepared = frame_15m.iloc[padded_start:end].reset_index(drop=True)
    target = frame_15m.iloc[start:end].reset_index(drop=True)
    score_start_index = start - padded_start
    if len(target) < 128:
        raise ValueError(f"Nautilus target segment too short: {len(target)} bars")
    if len(prepared) <= score_start_index:
        raise ValueError(
            f"Nautilus prepared segment has no scored region: prepared={len(prepared)} score_start={score_start_index}"
        )
    return prepared, target, padded_start, start, score_start_index


def _apply_nautilus_target_slice_metrics(
    summary: dict,
    *,
    target_frame: pd.DataFrame,
    target_start: int,
    target_end: int,
    prepared_start: int,
    score_start_index: int,
    initial_balance: float,
    monthly_server_cost_usd: float,
    periods_per_day: int,
) -> dict:
    adjusted = dict(summary)
    target_bars = int(len(target_frame))
    server_cost_paid = 0.0
    if monthly_server_cost_usd > 0.0 and periods_per_day > 0 and target_bars > 0:
        server_cost_paid = float(monthly_server_cost_usd) * (
            float(target_bars) / float(periods_per_day * 30)
        )

    gross_total_return = float(adjusted.get("gross_total_return", adjusted.get("total_return", 0.0)))
    net_total_return = gross_total_return - (server_cost_paid / max(float(initial_balance), 1e-9))
    bh_eval_return = 0.0
    if target_bars >= 2:
        first_price = float(target_frame["close"].iloc[0])
        last_price = float(target_frame["close"].iloc[-1])
        if first_price > 0.0:
            bh_eval_return = last_price / first_price - 1.0

    adjusted["server_cost_paid"] = float(server_cost_paid)
    adjusted["gross_total_return"] = float(gross_total_return)
    adjusted["total_return"] = float(net_total_return)
    adjusted["bh_eval_return"] = float(bh_eval_return)
    adjusted["outperformance_vs_bh"] = float(net_total_return - bh_eval_return)
    adjusted["segment_range"] = [int(target_start), int(target_end)]
    adjusted["segment_input_range"] = [int(prepared_start), int(target_end)]
    adjusted["segment_score_start_index"] = int(score_start_index)
    adjusted["segment_target_bars"] = int(target_bars)
    adjusted["segment_warmup_bars"] = int(max(score_start_index, 0))
    adjusted["trade_density"] = float(adjusted.get("n_trades", 0.0)) / max(float(target_bars), 1.0)
    return _normalize_nautilus_summary_metrics(adjusted)


def _normalize_nautilus_summary_metrics(summary: dict) -> dict:
    """Recompute execution metrics from action counts / position events when available."""
    normalized = dict(summary)
    backtest = normalized.get("backtest", {})
    if not isinstance(backtest, dict):
        backtest = {}

    total_orders = int(
        normalized.get(
            "total_orders",
            backtest.get("total_orders", 0),
        )
        or 0
    )
    total_positions = int(
        normalized.get(
            "total_positions",
            backtest.get("total_positions", 0),
        )
        or 0
    )
    n_trades = max(
        int(normalized.get("n_trades", 0) or 0),
        int(backtest.get("n_trades", 0) or 0),
        total_positions,
    )

    n_wins = max(
        int(normalized.get("n_wins", 0) or 0),
        int(backtest.get("n_wins", 0) or 0),
    )
    n_losses = max(
        int(normalized.get("n_losses", 0) or 0),
        int(backtest.get("n_losses", 0) or 0),
    )
    total_closed = n_wins + n_losses
    win_rate = (
        float(n_wins / total_closed)
        if total_closed > 0
        else float(normalized.get("win_rate", backtest.get("win_rate", 0.0)) or 0.0)
    )

    action_counts = normalized.get("action_counts")
    if not isinstance(action_counts, dict):
        action_counts = backtest.get("action_counts", {})
    if not isinstance(action_counts, dict):
        action_counts = {}
    combined_action_counts = {
        "short": int(action_counts.get("short", normalized.get("eval_short_actions", 0.0)) or 0),
        "flat": int(action_counts.get("flat", normalized.get("eval_flat_actions", 0.0)) or 0),
        "long": int(action_counts.get("long", normalized.get("eval_long_actions", 0.0)) or 0),
    }
    total_actions = int(sum(combined_action_counts.values()))

    if total_actions > 0:
        probs = np.array(
            [
                combined_action_counts["short"],
                combined_action_counts["flat"],
                combined_action_counts["long"],
            ],
            dtype=np.float64,
        ) / float(total_actions)
        nonzero = probs[probs > 0]
        entropy = float(-np.sum(nonzero * np.log(nonzero)) / np.log(3.0))
        if abs(entropy) < 1e-12:
            entropy = 0.0
        dominant_action = int(np.argmax(probs))
        dominant_ratio = float(np.max(probs))
        flat_ratio = float(combined_action_counts["flat"] / total_actions)
        position_ratio = float(
            (combined_action_counts["short"] + combined_action_counts["long"]) / total_actions
        )
    else:
        entropy = float(normalized.get("eval_action_entropy", backtest.get("eval_action_entropy", 0.0)) or 0.0)
        if abs(entropy) < 1e-12:
            entropy = 0.0
        dominant_action = int(
            normalized.get("eval_dominant_action", backtest.get("eval_dominant_action", 1.0)) or 1
        )
        dominant_ratio = float(
            normalized.get(
                "eval_dominant_action_ratio",
                backtest.get("eval_dominant_action_ratio", 1.0),
            )
            or 1.0
        )
        flat_ratio = float(normalized.get("flat_ratio", backtest.get("flat_ratio", 1.0)) or 1.0)
        position_ratio = float(
            normalized.get("position_ratio", backtest.get("position_ratio", 0.0)) or 0.0
        )

    target_bars = int(normalized.get("segment_target_bars", 0) or 0)
    trade_density = (
        float(n_trades) / max(float(target_bars), 1.0)
        if target_bars > 0
        else float(normalized.get("trade_density", 0.0) or 0.0)
    )

    normalized.update(
        {
            "total_orders": int(total_orders),
            "total_positions": int(total_positions),
            "n_trades": int(n_trades),
            "n_wins": int(n_wins),
            "n_losses": int(n_losses),
            "win_rate": float(win_rate),
            "action_counts": dict(combined_action_counts),
            "flat_ratio": float(flat_ratio),
            "position_ratio": float(position_ratio),
            "eval_action_entropy": float(entropy),
            "eval_dominant_action": float(dominant_action),
            "eval_dominant_action_ratio": float(dominant_ratio),
            "eval_short_actions": float(combined_action_counts["short"]),
            "eval_flat_actions": float(combined_action_counts["flat"]),
            "eval_long_actions": float(combined_action_counts["long"]),
            "avg_trades_per_episode": float(
                max(float(normalized.get("avg_trades_per_episode", 0.0) or 0.0), float(n_trades))
            ),
            "trade_density": float(trade_density),
        }
    )
    return normalized


def _estimate_directional_trade_counts(summary: dict) -> tuple[int, int]:
    """Approximate long/short trade counts from action mix when explicit counts are unavailable."""
    n_longs = int(summary.get("n_longs", 0) or 0)
    n_shorts = int(summary.get("n_shorts", 0) or 0)
    if n_longs > 0 or n_shorts > 0:
        return n_longs, n_shorts

    n_trades = int(summary.get("n_trades", 0) or 0)
    if n_trades <= 0:
        return 0, 0

    action_counts = summary.get("action_counts", {})
    if not isinstance(action_counts, dict):
        action_counts = {}
    long_actions = float(action_counts.get("long", summary.get("eval_long_actions", 0.0)) or 0.0)
    short_actions = float(action_counts.get("short", summary.get("eval_short_actions", 0.0)) or 0.0)
    directional_actions = max(long_actions + short_actions, 0.0)
    if directional_actions <= 0.0:
        return 0, 0

    est_longs = int(round(float(n_trades) * (long_actions / directional_actions)))
    est_longs = max(0, min(est_longs, n_trades))
    est_shorts = max(0, n_trades - est_longs)
    return est_longs, est_shorts


def _promote_nautilus_test_metrics(metrics: dict, final_summary: dict) -> dict:
    """Promote execution-aligned Nautilus held-out test metrics to top-level display fields."""
    promoted = dict(metrics or {})
    if "env_eval_metrics" not in promoted:
        promoted["env_eval_metrics"] = copy.deepcopy(dict(metrics or {}))

    normalized = _normalize_nautilus_summary_metrics(dict(final_summary or {}))
    promoted["nautilus_test"] = dict(normalized)
    promoted["metrics_source"] = "nautilus_test"
    promoted["metrics_source_label"] = "Nautilus held-out test"

    top_level_keys = [
        "total_return",
        "gross_total_return",
        "bh_eval_return",
        "outperformance_vs_bh",
        "n_trades",
        "n_wins",
        "n_losses",
        "win_rate",
        "server_cost_paid",
        "flat_ratio",
        "position_ratio",
        "eval_action_entropy",
        "eval_dominant_action",
        "eval_dominant_action_ratio",
        "eval_short_actions",
        "eval_flat_actions",
        "eval_long_actions",
        "avg_trades_per_episode",
        "max_drawdown",
        "trade_density",
        "total_orders",
        "total_positions",
        "segment_range",
        "segment_input_range",
        "segment_score_start_index",
        "segment_target_bars",
        "segment_warmup_bars",
        "status",
    ]
    for key in top_level_keys:
        if key in normalized:
            promoted[key] = copy.deepcopy(normalized[key])

    stats_returns = normalized.get("stats_returns", {})
    if isinstance(stats_returns, dict):
        if "Sharpe Ratio (252 days)" in stats_returns:
            promoted["sharpe"] = float(stats_returns.get("Sharpe Ratio (252 days)", 0.0))
        if "Sortino Ratio (252 days)" in stats_returns:
            promoted["sortino"] = float(stats_returns.get("Sortino Ratio (252 days)", 0.0))

    n_longs, n_shorts = _estimate_directional_trade_counts(normalized)
    promoted["n_longs"] = float(n_longs)
    promoted["n_shorts"] = float(n_shorts)
    promoted["eval_episodes"] = int(normalized.get("eval_episodes", 1) or 1)
    return promoted


def _aggregate_nautilus_window_summaries(
    summaries: list[dict],
    scores: list[float],
    windows: list[tuple[int, int]],
) -> tuple[dict, float]:
    if not summaries:
        return {}, float("-inf")

    normalized_summaries = [_normalize_nautilus_summary_metrics(item) for item in summaries]

    if len(normalized_summaries) == 1:
        summary = dict(normalized_summaries[0])
        score = float(scores[0]) if scores else float("-inf")
        summary["nautilus_window_count"] = 1
        summary["nautilus_score_median"] = score
        summary["nautilus_score_worst"] = score
        summary["nautilus_score_robust"] = score
        summary["nautilus_min_total_return"] = float(summary.get("total_return", 0.0))
        summary["nautilus_min_gross_total_return"] = float(summary.get("gross_total_return", 0.0))
        summary["nautilus_min_alpha"] = float(summary.get("outperformance_vs_bh", 0.0))
        summary["nautilus_active_window_ratio"] = 1.0 if score != float("-inf") else 0.0
        summary["nautilus_median_trade_density"] = float(summary.get("trade_density", 0.0))
        summary["nautilus_worst_dominant_action_ratio"] = float(summary.get("eval_dominant_action_ratio", 1.0))
        summary["nautilus_windows"] = [[int(windows[0][0]), int(windows[0][1])]] if windows else []
        return summary, score

    def _arr(key: str, default: float = 0.0) -> np.ndarray:
        return np.asarray([float(item.get(key, default)) for item in normalized_summaries], dtype=np.float64)

    score_arr = np.asarray(scores, dtype=np.float64)
    finite_scores = score_arr[np.isfinite(score_arr)]
    robust_score = float("-inf")
    active_ratio = 0.0
    if finite_scores.size > 0:
        robust_score = float(np.median(finite_scores) + 0.5 * np.min(finite_scores))
        active_ratio = float(finite_scores.size / max(len(scores), 1))
        robust_score -= max(0.75 - active_ratio, 0.0) * 6.0

    net = _arr("total_return")
    gross = _arr("gross_total_return")
    alpha = _arr("outperformance_vs_bh")
    dd = np.abs(_arr("max_drawdown"))
    trades = _arr("n_trades")
    trade_density = _arr("trade_density")
    win_rate = _arr("win_rate")
    flat_ratio = _arr("flat_ratio", 1.0)
    pos_ratio = _arr("position_ratio")
    entropy = _arr("eval_action_entropy")
    dom_ratio = _arr("eval_dominant_action_ratio", 1.0)
    server_cost = _arr("server_cost_paid")
    dominant_idx = int(np.argmax(np.where(np.isfinite(score_arr), score_arr, -np.inf))) if np.isfinite(score_arr).any() else 0
    total_target_bars = float(
        np.sum([float(item.get("segment_target_bars", 0.0)) for item in normalized_summaries])
    )
    total_trades = float(
        np.sum(
            [
                max(
                    float(item.get("n_trades", 0.0)),
                    float(item.get("total_positions", 0.0)),
                )
                for item in normalized_summaries
            ]
        )
    )
    total_wins = int(np.sum([int(item.get("n_wins", 0)) for item in normalized_summaries]))
    total_losses = int(np.sum([int(item.get("n_losses", 0)) for item in normalized_summaries]))
    total_orders = int(np.sum([int(item.get("total_orders", 0)) for item in normalized_summaries]))
    total_positions = int(np.sum([int(item.get("total_positions", 0)) for item in normalized_summaries]))
    combined_action_counts = {"short": 0, "flat": 0, "long": 0}
    for item in normalized_summaries:
        counts = item.get("action_counts", {})
        if not isinstance(counts, dict):
            continue
        for key in ("short", "flat", "long"):
            combined_action_counts[key] += int(counts.get(key, 0))
    total_actions = sum(combined_action_counts.values())
    if total_actions > 0:
        probs = np.array(
            [
                combined_action_counts["short"],
                combined_action_counts["flat"],
                combined_action_counts["long"],
            ],
            dtype=np.float64,
        ) / float(total_actions)
        nonzero = probs[probs > 0]
        combined_action_entropy = float(-np.sum(nonzero * np.log(nonzero)) / np.log(3.0))
        if abs(combined_action_entropy) < 1e-12:
            combined_action_entropy = 0.0
        combined_dominant_idx = int(np.argmax(probs))
        combined_dominant_ratio = float(np.max(probs))
        combined_flat_ratio = float(combined_action_counts["flat"] / total_actions)
        combined_position_ratio = float(
            (combined_action_counts["short"] + combined_action_counts["long"]) / total_actions
        )
    else:
        combined_action_entropy = float(np.median(entropy))
        combined_dominant_idx = int(np.argmax(np.array([0.0, 1.0, 0.0])))
        combined_dominant_ratio = 1.0
        combined_flat_ratio = float(np.median(flat_ratio))
        combined_position_ratio = float(np.median(pos_ratio))
    weighted_trade_density = (
        total_trades / max(total_target_bars, 1.0)
        if total_target_bars > 0.0
        else float(np.median(trade_density))
    )
    aggregate_win_rate = (
        float(total_wins / max(total_wins + total_losses, 1))
        if (total_wins + total_losses) > 0
        else float(np.median(win_rate))
    )

    combined = dict(normalized_summaries[dominant_idx])
    combined.update(
        {
            "total_return": float(np.median(net)),
            "gross_total_return": float(np.median(gross)),
            "outperformance_vs_bh": float(np.median(alpha)),
            "max_drawdown": float(np.max(dd)),
            "total_orders": int(total_orders),
            "total_positions": int(total_positions),
            "n_trades": float(total_trades),
            "n_wins": int(total_wins),
            "n_losses": int(total_losses),
            "win_rate": float(aggregate_win_rate),
            "flat_ratio": float(combined_flat_ratio),
            "position_ratio": float(combined_position_ratio),
            "eval_action_entropy": float(combined_action_entropy),
            "eval_dominant_action": float(combined_dominant_idx),
            "eval_dominant_action_ratio": float(combined_dominant_ratio),
            "eval_short_actions": float(combined_action_counts["short"]),
            "eval_flat_actions": float(combined_action_counts["flat"]),
            "eval_long_actions": float(combined_action_counts["long"]),
            "action_counts": dict(combined_action_counts),
            "server_cost_paid": float(np.median(server_cost)),
            "trade_density": float(weighted_trade_density),
            "nautilus_window_count": int(len(normalized_summaries)),
            "nautilus_score_median": float(np.median(finite_scores)) if finite_scores.size > 0 else float("-inf"),
            "nautilus_score_worst": float(np.min(finite_scores)) if finite_scores.size > 0 else float("-inf"),
            "nautilus_score_robust": float(robust_score),
            "nautilus_min_total_return": float(np.min(net)),
            "nautilus_min_gross_total_return": float(np.min(gross)),
            "nautilus_min_alpha": float(np.min(alpha)),
            "nautilus_active_window_ratio": float(active_ratio),
            "nautilus_median_trade_density": float(np.median(trade_density)),
            "nautilus_worst_dominant_action_ratio": float(np.max(dom_ratio)),
            "nautilus_windows": [[int(start), int(end)] for start, end in windows],
        }
    )
    return combined, robust_score


def _select_nautilus_candidate_subset(
    candidate_records: list[dict],
    *,
    limit: int,
) -> list[dict]:
    if limit <= 0 or len(candidate_records) <= limit:
        return list(candidate_records)

    def _metric(candidate: dict, key: str, default: float = 0.0) -> float:
        metrics = candidate.get("env_validation_metrics")
        if not isinstance(metrics, dict):
            return float(default)
        return float(metrics.get(key, default))

    def _sort_key(candidate: dict) -> tuple[float, float, float, float]:
        return (
            float(candidate.get("env_validation_score", float("-inf"))),
            _metric(candidate, "walkforward_selection_score", float("-inf")),
            _metric(candidate, "total_return", float("-inf")),
            -_metric(candidate, "supervised_validation_brier", float("inf")),
        )

    def _candidate_id(candidate: dict) -> str:
        return f"{candidate.get('candidate_name', '')}|{candidate.get('model_path', '')}"

    def _confidence(candidate: dict, key: str) -> float:
        return float(candidate.get(key, candidate.get("supervised_confidence_threshold", 0.0)))

    def _min_confidence(candidate: dict) -> float:
        return min(
            _confidence(candidate, "supervised_long_confidence_threshold"),
            _confidence(candidate, "supervised_short_confidence_threshold"),
        )

    def _avg_trades(candidate: dict) -> float:
        return _metric(candidate, "avg_trades_per_episode", float("inf"))

    def _walkforward_active_ratio(candidate: dict) -> float:
        return _metric(candidate, "walkforward_active_window_ratio", 0.0)

    def _gross_return(candidate: dict) -> float:
        return _metric(candidate, "gross_total_return", float("-inf"))

    def _worst_gross_return(candidate: dict) -> float:
        return _metric(
            candidate,
            "walkforward_min_gross_total_return",
            _gross_return(candidate),
        )

    ordered = sorted(candidate_records, key=_sort_key, reverse=True)
    selected: list[dict] = []
    seen_ids: set[str] = set()

    def _append_candidate(candidate: dict | None) -> None:
        if candidate is None:
            return
        candidate_id = _candidate_id(candidate)
        if candidate_id in seen_ids:
            return
        seen_ids.add(candidate_id)
        selected.append(candidate)

    def _best_remaining(metric_keys: tuple[str, ...]) -> dict | None:
        pool = [item for item in ordered if _candidate_id(item) not in seen_ids]
        if not pool:
            return None
        return max(
            pool,
            key=lambda item: tuple(_metric(item, key, float("-inf")) for key in metric_keys),
        )

    def _best_remaining_by_key(key_fn) -> dict | None:
        pool = [item for item in ordered if _candidate_id(item) not in seen_ids]
        if not pool:
            return None
        return max(pool, key=key_fn)

    def _append_best_by_family() -> None:
        family_values = []
        for item in ordered:
            family = str(item.get("family", ""))
            if family and family not in family_values:
                family_values.append(family)
        for family in family_values:
            pool = [
                item
                for item in ordered
                if _candidate_id(item) not in seen_ids
                and str(item.get("family", "")) == family
            ]
            if not pool:
                continue
            candidate = max(
                pool,
                key=lambda item: (
                    _walkforward_active_ratio(item),
                    1.0 if _gross_return(item) > 0.0 else 0.0,
                    float(item.get("env_validation_score", float("-inf"))),
                    -_avg_trades(item),
                ),
            )
            _append_candidate(candidate)
            if len(selected) >= limit:
                return

    def _append_best_active_sparse_candidate() -> None:
        pool = [
            item
            for item in ordered
            if _candidate_id(item) not in seen_ids
            and _walkforward_active_ratio(item) > 0.0
        ]
        if not pool:
            return
        candidate = max(
            pool,
            key=lambda item: (
                _walkforward_active_ratio(item),
                1.0 if _worst_gross_return(item) > -0.003 else 0.0,
                1.0 if _gross_return(item) > 0.0 else 0.0,
                -_confidence(item, "supervised_short_confidence_threshold"),
                -abs(_avg_trades(item) - 4.0),
                float(item.get("env_validation_score", float("-inf"))),
            ),
        )
        _append_candidate(candidate)

    _append_candidate(ordered[0] if ordered else None)

    def _append_next_stricter_short_candidate() -> None:
        if not selected:
            return
        current_short_conf = _confidence(
            selected[0],
            "supervised_short_confidence_threshold",
        )
        pool = [
            item
            for item in ordered
            if _candidate_id(item) not in seen_ids
            and _confidence(item, "supervised_short_confidence_threshold") > current_short_conf + 1e-6
        ]
        if not pool:
            return
        target_short_conf = min(
            _confidence(item, "supervised_short_confidence_threshold") for item in pool
        )
        candidate = max(
            pool,
            key=lambda item, target=target_short_conf: (
                1.0
                if np.isclose(
                    _confidence(item, "supervised_short_confidence_threshold"),
                    target,
                )
                else 0.0,
                _walkforward_active_ratio(item),
                1.0
                if _metric(
                    item,
                    "walkforward_min_gross_total_return",
                    _metric(item, "gross_total_return", float("-inf")),
                )
                > -0.002
                else 0.0,
                _metric(item, "gross_total_return", float("-inf")),
                -abs(_avg_trades(item) - 4.0),
                float(item.get("env_validation_score", float("-inf"))),
            ),
        )
        _append_candidate(candidate)

    def _append_best_by_confidence(key: str) -> None:
        values = sorted({_confidence(item, key) for item in ordered}, reverse=True)
        for value in values:
            candidate = _best_remaining_by_key(
                lambda item, conf=value: (
                    1.0 if np.isclose(_confidence(item, key), conf) else 0.0,
                    float(item.get("env_validation_score", float("-inf"))),
                    _gross_return(item),
                    -_avg_trades(item),
                )
            )
            if candidate is None or not np.isclose(_confidence(candidate, key), value):
                continue
            _append_candidate(candidate)
            if len(selected) >= limit:
                return

    def _append_extreme_by_confidence(key: str, *, highest: bool) -> None:
        values = sorted({_confidence(item, key) for item in ordered}, reverse=highest)
        for value in values:
            pool = [
                item
                for item in ordered
                if _candidate_id(item) not in seen_ids
                and np.isclose(_confidence(item, key), value)
            ]
            if not pool:
                continue
            candidate = max(
                pool,
                key=lambda item: (
                    _walkforward_active_ratio(item),
                    1.0 if _gross_return(item) > 0.0 else 0.0,
                    1.0 if _worst_gross_return(item) > -0.003 else 0.0,
                    -abs(_avg_trades(item) - 4.0),
                    float(item.get("env_validation_score", float("-inf"))),
                ),
            )
            _append_candidate(candidate)
            return

    _append_best_active_sparse_candidate()
    _append_best_by_family()
    _append_next_stricter_short_candidate()
    _append_extreme_by_confidence("supervised_short_confidence_threshold", highest=True)
    _append_extreme_by_confidence("supervised_long_confidence_threshold", highest=True)
    _append_extreme_by_confidence("supervised_short_confidence_threshold", highest=False)
    _append_extreme_by_confidence("supervised_long_confidence_threshold", highest=False)
    _append_candidate(
        _best_remaining_by_key(
            lambda item: (
                _min_confidence(item),
                1.0 if _gross_return(item) > 0.0 else 0.0,
                _walkforward_active_ratio(item),
                -_avg_trades(item),
                float(item.get("env_validation_score", float("-inf"))),
            )
        )
    )
    _append_candidate(_best_remaining(("walkforward_min_total_return", "walkforward_min_gross_total_return", "total_return")))
    _append_candidate(
        _best_remaining_by_key(
            lambda item: (
                _confidence(item, "supervised_long_confidence_threshold")
                + _confidence(item, "supervised_short_confidence_threshold"),
                _gross_return(item),
                float(item.get("env_validation_score", float("-inf"))),
                -_avg_trades(item),
            )
        )
    )
    _append_candidate(
        _best_remaining_by_key(
            lambda item: (
                _walkforward_active_ratio(item),
                1.0 if _gross_return(item) > 0.0 else 0.0,
                -_avg_trades(item),
                _gross_return(item),
                float(item.get("env_validation_score", float("-inf"))),
            )
        )
    )
    _append_candidate(_best_remaining(("gross_total_return", "total_return", "outperformance_vs_bh")))
    _append_best_by_confidence("supervised_long_confidence_threshold")
    _append_best_by_confidence("supervised_short_confidence_threshold")

    for candidate in ordered:
        if len(selected) >= limit:
            break
        _append_candidate(candidate)

    return selected[:limit]


def _prune_supervised_candidate_pool(candidate_pool: list[dict]) -> list[dict]:
    if not candidate_pool:
        return []

    def _metric(candidate: dict, key: str, default: float = 0.0) -> float:
        metrics = candidate.get("validation_metrics")
        if not isinstance(metrics, dict):
            return float(default)
        return float(metrics.get(key, default))

    def _behavior_key(candidate: dict) -> tuple:
        model_type = str(candidate.get("meta", {}).get("model_type", "fallback"))
        long_conf = round(float(candidate.get("long_confidence", candidate.get("confidence", 0.0))), 2)
        short_conf = round(float(candidate.get("short_confidence", candidate.get("confidence", 0.0))), 2)
        long_actions = round(_metric(candidate, "eval_long_actions", 0.0), 1)
        short_actions = round(_metric(candidate, "eval_short_actions", 0.0), 1)
        active_ratio = round(_metric(candidate, "walkforward_active_window_ratio", 0.0), 2)
        avg_trades = round(_metric(candidate, "avg_trades_per_episode", _metric(candidate, "n_trades", 0.0)), 1)
        dominant_ratio = round(_metric(candidate, "eval_dominant_action_ratio", 1.0), 2)
        gross_return = round(_metric(candidate, "gross_total_return", 0.0), 4)
        net_return = round(_metric(candidate, "total_return", 0.0), 4)

        if long_actions <= 0.0 and short_actions > 0.0:
            action_mode = "short_only"
        elif short_actions <= 0.0 and long_actions > 0.0:
            action_mode = "long_only"
        elif short_actions <= 0.0 and long_actions <= 0.0:
            action_mode = "flat_only"
        else:
            action_mode = "mixed"

        return (
            model_type,
            action_mode,
            long_conf,
            short_conf,
            active_ratio,
            avg_trades,
            dominant_ratio,
            gross_return,
            net_return,
        )

    deduped: dict[tuple, dict] = {}
    for candidate in candidate_pool:
        key = _behavior_key(candidate)
        incumbent = deduped.get(key)
        if incumbent is None:
            deduped[key] = candidate
            continue
        incumbent_key = (
            float(incumbent.get("score", float("-inf"))),
            int(incumbent.get("priority", -999)),
            -_metric(incumbent, "supervised_validation_brier", float("inf")),
        )
        candidate_key = (
            float(candidate.get("score", float("-inf"))),
            int(candidate.get("priority", -999)),
            -_metric(candidate, "supervised_validation_brier", float("inf")),
        )
        if candidate_key > incumbent_key:
            deduped[key] = candidate

    return sorted(
        deduped.values(),
        key=lambda item: (
            float(item.get("score", float("-inf"))),
            int(item.get("priority", -999)),
            -_metric(item, "supervised_validation_brier", float("inf")),
        ),
        reverse=True,
    )


def _trim_supervised_candidate_pool(
    candidate_pool: list[dict],
    *,
    limit: int,
) -> list[dict]:
    """Trim the supervised candidate pool while preserving family diversity.

    The candidate search evaluates model families sequentially. If we truncate the
    pool immediately during search, later families can be starved by earlier
    families even when they should still get a realistic Nautilus check.
    """
    if limit <= 0 or len(candidate_pool) <= limit:
        return list(candidate_pool)

    def _metric(candidate: dict, key: str, default: float = 0.0) -> float:
        metrics = candidate.get("validation_metrics")
        if not isinstance(metrics, dict):
            return float(default)
        return float(metrics.get(key, default))

    ordered = sorted(
        candidate_pool,
        key=lambda item: (
            float(item.get("score", float("-inf"))),
            int(item.get("priority", -999)),
            _metric(item, "walkforward_active_window_ratio", 0.0),
            _metric(item, "gross_total_return", float("-inf")),
            -_metric(item, "supervised_validation_brier", float("inf")),
        ),
        reverse=True,
    )

    selected: list[dict] = []
    seen_ids: set[str] = set()

    def _candidate_id(candidate: dict) -> str:
        return "|".join(
            [
                str(candidate.get("name", "")),
                str(candidate.get("meta", {}).get("model_type", "")),
                f"{float(candidate.get('long_confidence', candidate.get('confidence', 0.0))):.4f}",
                f"{float(candidate.get('short_confidence', candidate.get('confidence', 0.0))):.4f}",
            ]
        )

    def _append(candidate: dict | None) -> None:
        if candidate is None:
            return
        candidate_id = _candidate_id(candidate)
        if candidate_id in seen_ids:
            return
        seen_ids.add(candidate_id)
        selected.append(candidate)

    family_best: dict[str, dict] = {}
    for item in ordered:
        family = str(item.get("meta", {}).get("model_type", "fallback"))
        family_best.setdefault(family, item)
    for family in sorted(family_best.keys()):
        _append(family_best[family])
        if len(selected) >= limit:
            return selected[:limit]

    for item in ordered:
        if len(selected) >= limit:
            break
        _append(item)

    return selected[:limit]


def _score_nautilus_summary(
    summary: dict,
    *,
    min_trades: float = 1.0,
    max_dominant_action_ratio: float = 0.995,
    min_active_window_ratio: float = 0.50,
) -> float:
    if summary.get("error"):
        return float("-inf")

    trades = float(summary.get("n_trades", 0.0))
    net_return = float(summary.get("total_return", 0.0))
    gross_return = float(summary.get("gross_total_return", net_return))
    alpha = float(summary.get("outperformance_vs_bh", 0.0))
    win_rate = float(summary.get("win_rate", 0.0))
    flat_ratio = float(summary.get("flat_ratio", 1.0))
    dominant_ratio = float(summary.get("eval_dominant_action_ratio", 1.0))
    action_entropy = float(summary.get("eval_action_entropy", 0.0))
    max_drawdown = abs(float(summary.get("max_drawdown", 0.0)))
    worst_net_return = float(summary.get("nautilus_min_total_return", net_return))
    worst_gross_return = float(summary.get("nautilus_min_gross_total_return", gross_return))
    worst_alpha = float(summary.get("nautilus_min_alpha", alpha))
    active_window_ratio = float(summary.get("nautilus_active_window_ratio", 1.0))
    trade_density = float(summary.get("trade_density", summary.get("nautilus_median_trade_density", 0.0)))
    returns_stats = summary.get("stats_returns", {}) if isinstance(summary.get("stats_returns"), dict) else {}
    sharpe = float(returns_stats.get("Sharpe Ratio (252 days)", 0.0) or 0.0)
    wrong_side_penalty = max(float(summary.get("position_ratio", 0.0)) - 0.98, 0.0) * 0.5
    sparse_active_override = _allows_sparse_active_nautilus_override(
        summary,
        min_trades=min_trades,
        max_dominant_action_ratio=max_dominant_action_ratio,
        min_active_window_ratio=min_active_window_ratio,
    )

    if trades < min_trades:
        return float("-inf")
    if active_window_ratio < float(min_active_window_ratio) and not sparse_active_override:
        return float("-inf")
    if dominant_ratio > max_dominant_action_ratio and trades <= (min_trades + 1.0) and not sparse_active_override:
        return float("-inf")
    if gross_return < -0.002 and worst_gross_return < -0.005:
        return float("-inf")
    if net_return < -0.005 and worst_net_return < -0.01:
        return float("-inf")
    # Reject models that lose more than 10%
    if net_return < -0.10:
        return float("-inf")

    # Sharpe-priority: previously sharpe weight was 0.20 vs net_return*10 → net_return dominated.
    # Boost sharpe + alpha so we reward consistent risk-adjusted profitability over raw P&L.
    profit_bonus = max(net_return, 0.0) * 8.0
    gross_bonus = max(gross_return, 0.0) * 5.0
    turnover_penalty = max(trade_density - 0.02, 0.0) * 8.0
    turnover_penalty += max(trade_density - 0.05, 0.0) * 18.0
    return (
        sharpe * 3.0                    # 0.20 → 3.0 (PRIMARY)
        + alpha * 3.0                   # 1.5 → 3.0  (must beat passive)
        + worst_alpha * 1.5
        + net_return * 5.0              # 10 → 5
        + gross_return * 3.0            # 6 → 3
        + worst_net_return * 4.0        # 6 → 4
        + worst_gross_return * 2.5      # 4 → 2.5
        + profit_bonus
        + gross_bonus
        + win_rate * 0.50
        + min(trades, 16.0) * 0.01
        + active_window_ratio * 0.80
        + action_entropy * 0.20
        - max(-gross_return, 0.0) * 14.0
        - max(-net_return, 0.0) * 10.0
        - max(-worst_gross_return, 0.0) * 10.0
        - max(-worst_net_return, 0.0) * 8.0
        - max(flat_ratio - 0.95, 0.0) * 2.0
        - max(dominant_ratio - 0.85, 0.0) * 1.5
        - max(max_drawdown - 0.15, 0.0) * 5.0   # 2.5 → 5.0 (survival first)
        - wrong_side_penalty
        - turnover_penalty
    )


def _allows_sparse_active_nautilus_override(
    summary: dict,
    *,
    min_trades: float = 1.0,
    max_dominant_action_ratio: float = 0.995,
    min_active_window_ratio: float = 1.0 / 3.0,
) -> bool:
    trades = float(summary.get("n_trades", 0.0))
    net_return = float(summary.get("total_return", 0.0))
    gross_return = float(summary.get("gross_total_return", net_return))
    worst_net_return = float(summary.get("nautilus_min_total_return", net_return))
    worst_gross_return = float(summary.get("nautilus_min_gross_total_return", gross_return))
    active_window_ratio = float(summary.get("nautilus_active_window_ratio", 0.0))
    dominant_ratio = float(summary.get("eval_dominant_action_ratio", 1.0))
    trade_density = float(summary.get("trade_density", summary.get("nautilus_median_trade_density", 0.0)))
    max_drawdown = abs(float(summary.get("max_drawdown", 0.0)))

    return bool(
        trades >= max(min_trades + 1.0, 2.0)
        and active_window_ratio >= min(float(min_active_window_ratio), 1.0 / 3.0)
        and gross_return >= -0.0005
        and worst_gross_return >= -0.0015
        and net_return >= -0.0025
        and worst_net_return >= -0.0035
        and dominant_ratio <= min(max_dominant_action_ratio, 0.985)
        and trade_density <= 0.01
        and max_drawdown <= 0.02
    )


def _score_nautilus_rejected_fallback(summary: dict) -> float:
    if not isinstance(summary, dict) or summary.get("error"):
        return float("-inf")

    preliminary_score = float(summary.get("nautilus_score_preliminary", float("-inf")))
    if not np.isfinite(preliminary_score):
        return float("-inf")

    trades = float(summary.get("n_trades", 0.0))
    net_return = float(summary.get("total_return", 0.0))
    gross_return = float(summary.get("gross_total_return", net_return))
    alpha = float(summary.get("outperformance_vs_bh", 0.0))
    worst_net_return = float(summary.get("nautilus_min_total_return", net_return))
    worst_gross_return = float(summary.get("nautilus_min_gross_total_return", gross_return))
    active_window_ratio = float(summary.get("nautilus_active_window_ratio", 0.0))
    trade_density = float(summary.get("nautilus_median_trade_density", summary.get("trade_density", 0.0)))
    dominant_ratio = float(summary.get("eval_dominant_action_ratio", 1.0))
    max_drawdown = abs(float(summary.get("max_drawdown", 0.0)))

    if trades < 1.0 or active_window_ratio <= 0.0:
        return float("-inf")
    if active_window_ratio < 0.25:
        return float("-inf")
    if gross_return < -0.005 or net_return < -0.010:
        return float("-inf")
    if worst_gross_return < -0.010 or worst_net_return < -0.015:
        return float("-inf")

    turnover_penalty = max(trade_density - 0.01, 0.0) * 18.0
    turnover_penalty += max(trade_density - 0.02, 0.0) * 28.0

    return float(
        preliminary_score * 0.35
        + gross_return * 8.0
        + net_return * 6.0
        + worst_gross_return * 5.0
        + worst_net_return * 4.0
        + alpha * 1.2
        + active_window_ratio * 0.75
        - trade_density * 14.0
        - turnover_penalty
        - max(dominant_ratio - 0.93, 0.0) * 5.0
        - max_drawdown * 4.0
    )


def _run_nautilus_backtest_segment(
    *,
    model_path: str,
    frame_15m: pd.DataFrame,
    segment_range: tuple[int, int],
    config: dict,
    label: str,
) -> dict:
    from execution.nautilus.backtest_runner import run_backtest_frame

    training_config = config.get("training", {})
    nav_config = training_config.get("nautilus_validation", {})
    trading_config = config.get("trading", {})
    backtest_log_level = str(nav_config.get("backtest_log_level", "ERROR"))
    backtest_bypass_logging = bool(nav_config.get("backtest_bypass_logging", False))
    backtest_state_update_interval_bars = max(int(nav_config.get("backtest_state_update_interval_bars", 64)), 1)
    backtest_publish_event_details = bool(nav_config.get("backtest_publish_event_details", False))
    backtest_publish_warmup_updates = bool(nav_config.get("backtest_publish_warmup_updates", False))
    history_bars = int(nav_config.get("history_bars", 384))
    segment, target_segment, prepared_start, target_start, score_start_index = _prepare_nautilus_segment_frame(
        frame_15m,
        segment_range,
        history_bars=history_bars,
    )
    start, end = int(segment_range[0]), int(segment_range[1])
    symbol = str(config.get("_active_symbol") or config.get("data", {}).get("pairs", ["BTCUSDT"])[0])
    initial_balance = float(nav_config.get("initial_balance_usdt", 10_000.0))
    leverage = float(nav_config.get("leverage", trading_config.get("leverage", 1.0)))
    trade_size_pct = float(nav_config.get("trade_size_pct_of_equity", 1.0))
    first_price = float(target_segment["close"].iloc[0])
    trade_qty = max((initial_balance * trade_size_pct) / max(first_price, 1e-9), 1e-6)
    state_name = f"nautilus_{label}_{Path(model_path).stem}.json"
    state_path = str(Path("checkpoints") / state_name)

    summary = run_backtest_frame(
        segment,
        symbol=symbol,
        model_path=model_path,
        venue=str(nav_config.get("venue", "BINANCE")),
        bar_minutes=int(nav_config.get("bar_minutes", 15)),
        history_bars=history_bars,
        request_history_days=int(nav_config.get("request_history_days", 8)),
        trade_size=f"{trade_qty:.12f}",
        initial_balance_usdt=initial_balance,
        leverage=leverage,
        state_path=state_path,
        mode=f"train_{label}",
        trading_start_bar_index=score_start_index,
        close_positions_on_stop=True,
        reduce_only_on_stop=True,
        monthly_server_cost_usd=float(trading_config.get("monthly_server_cost_usd", 100.0)),
        periods_per_day=int(trading_config.get("periods_per_day", 96)),
        log_level=backtest_log_level,
        bypass_logging=backtest_bypass_logging,
        state_update_interval_bars=backtest_state_update_interval_bars,
        publish_event_details=backtest_publish_event_details,
        publish_warmup_updates=backtest_publish_warmup_updates,
    )
    summary = _apply_nautilus_target_slice_metrics(
        summary,
        target_frame=target_segment,
        target_start=target_start,
        target_end=end,
        prepared_start=prepared_start,
        score_start_index=score_start_index,
        initial_balance=initial_balance,
        monthly_server_cost_usd=float(trading_config.get("monthly_server_cost_usd", 100.0)),
        periods_per_day=int(trading_config.get("periods_per_day", 96)),
    )
    summary["segment_label"] = label
    summary["trade_size_qty"] = float(trade_qty)
    summary["trade_size_notional_usdt"] = float(trade_qty * first_price)
    return summary


def _run_nautilus_backtest_segment_subprocess(
    *,
    model_path: str,
    frame_15m: pd.DataFrame,
    segment_range: tuple[int, int],
    config: dict,
    label: str,
) -> dict:
    training_config = config.get("training", {})
    nav_config = training_config.get("nautilus_validation", {})
    trading_config = config.get("trading", {})
    backtest_log_level = str(nav_config.get("backtest_log_level", "ERROR"))
    backtest_bypass_logging = bool(nav_config.get("backtest_bypass_logging", False))
    backtest_state_update_interval_bars = max(int(nav_config.get("backtest_state_update_interval_bars", 64)), 1)
    backtest_publish_event_details = bool(nav_config.get("backtest_publish_event_details", False))
    backtest_publish_warmup_updates = bool(nav_config.get("backtest_publish_warmup_updates", False))
    history_bars = int(nav_config.get("history_bars", 384))
    segment, target_segment, prepared_start, target_start, score_start_index = _prepare_nautilus_segment_frame(
        frame_15m,
        segment_range,
        history_bars=history_bars,
    )
    start, end = int(segment_range[0]), int(segment_range[1])
    symbol = str(config.get("_active_symbol") or config.get("data", {}).get("pairs", ["BTCUSDT"])[0])
    initial_balance = float(nav_config.get("initial_balance_usdt", 10_000.0))
    leverage = float(nav_config.get("leverage", trading_config.get("leverage", 1.0)))
    trade_size_pct = float(nav_config.get("trade_size_pct_of_equity", 1.0))
    first_price = float(target_segment["close"].iloc[0])
    trade_qty = max((initial_balance * trade_size_pct) / max(first_price, 1e-9), 1e-6)

    safe_label = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in label)
    segment_path = Path("checkpoints") / f"nautilus_segment_{safe_label}.parquet"
    summary_path = Path("checkpoints") / f"nautilus_segment_{safe_label}_summary.json"
    state_path = Path("checkpoints") / f"nautilus_{safe_label}.json"
    log_path = Path("checkpoints") / f"nautilus_{safe_label}.log"
    segment.to_parquet(segment_path, index=False)
    if summary_path.exists():
        summary_path.unlink()
    if log_path.exists():
        log_path.unlink()

    def _load_json_dict(path: Path) -> dict | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _log_tail(limit_chars: int = 4000) -> str:
        if not log_path.exists():
            return ""
        try:
            data = log_path.read_bytes()
        except Exception:
            return ""
        return data.decode("utf-8", errors="replace")[-limit_chars:]

    cmd = [
        sys.executable,
        "-X",
        "utf8",
        "-m",
        "execution.nautilus.backtest_runner",
        "--frame-parquet",
        str(segment_path),
        "--model-path",
        str(model_path),
        "--symbol",
        symbol,
        "--venue",
        str(nav_config.get("venue", "BINANCE")),
        "--bar-minutes",
        str(int(nav_config.get("bar_minutes", 15))),
        "--history-bars",
        str(history_bars),
        "--request-history-days",
        str(int(nav_config.get("request_history_days", 8))),
        "--trade-size",
        f"{trade_qty:.12f}",
        "--initial-balance-usdt",
        str(initial_balance),
        "--leverage",
        str(leverage),
        "--state-path",
        str(state_path),
        "--mode",
        f"train_{label}",
        "--trading-start-bar-index",
        str(int(score_start_index)),
        "--monthly-server-cost-usd",
        str(float(trading_config.get("monthly_server_cost_usd", 100.0))),
        "--periods-per-day",
        str(int(trading_config.get("periods_per_day", 96))),
        "--summary-json",
        str(summary_path),
        "--log-path",
        str(log_path),
        "--log-level",
        backtest_log_level,
        "--state-update-interval-bars",
        str(backtest_state_update_interval_bars),
    ]
    if backtest_bypass_logging:
        cmd.append("--bypass-logging")
    if backtest_publish_event_details:
        cmd.append("--publish-event-details")
    if backtest_publish_warmup_updates:
        cmd.append("--publish-warmup-updates")
    configured_collapse_probe_bars = max(int(nav_config.get("collapse_probe_bars", 0)), 0)
    collapse_probe_min_fraction = float(np.clip(nav_config.get("collapse_probe_min_fraction", 0.75), 0.0, 1.0))
    target_probe_floor = int(round(len(target_segment) * collapse_probe_min_fraction))
    collapse_probe_bars = min(
        len(target_segment),
        max(configured_collapse_probe_bars, target_probe_floor),
    ) if configured_collapse_probe_bars > 0 else 0
    collapse_probe_min_flat_ratio = float(nav_config.get("collapse_probe_min_flat_ratio", 0.995))
    collapse_probe_max_trades = max(int(nav_config.get("collapse_probe_max_trades", 0)), 0)
    base_timeout_seconds = max(int(nav_config.get("subprocess_timeout_seconds", 240)), 30)
    test_timeout_seconds = max(
        int(nav_config.get("subprocess_test_timeout_seconds", base_timeout_seconds)),
        base_timeout_seconds,
    )
    state_poll_interval_seconds = max(
        float(nav_config.get("subprocess_state_poll_interval_seconds", 4.0)),
        0.5,
    )
    cleanup_validation_artifacts = bool(nav_config.get("cleanup_validation_artifacts", True))
    subprocess_timeout_seconds = (
        test_timeout_seconds if str(label).startswith("test_") else base_timeout_seconds
    )
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    start_time = time.time()
    killed_for_collapse = False
    killed_for_timeout = False
    collapse_state: dict | None = None
    with open(log_path, "ab") as log_handle:
        proc = subprocess.Popen(
            cmd,
            cwd=str(Path.cwd()),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=False,
            env=env,
        )

        while True:
            returncode = proc.poll()
            if returncode is not None:
                proc.communicate()
                break

            runtime_state = _load_json_dict(state_path)
            if collapse_probe_bars > 0 and runtime_state is not None:
                try:
                    action_counts = runtime_state.get("action_counts", {})
                    total_actions = int(sum(int(action_counts.get(k, 0)) for k in ("short", "flat", "long")))
                    flat_actions = int(action_counts.get("flat", 0))
                    n_trades = int(runtime_state.get("n_trades", 0))
                    flat_ratio = float(flat_actions) / max(total_actions, 1)
                    if (
                        total_actions >= collapse_probe_bars
                        and n_trades <= collapse_probe_max_trades
                        and flat_ratio >= collapse_probe_min_flat_ratio
                    ):
                        killed_for_collapse = True
                        collapse_state = runtime_state
                        proc.terminate()
                        try:
                            proc.communicate(timeout=15)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.communicate()
                        break
                except Exception:
                    pass

            if (time.time() - start_time) >= float(subprocess_timeout_seconds):
                killed_for_timeout = True
                collapse_state = runtime_state if isinstance(runtime_state, dict) else None
                proc.terminate()
                try:
                    proc.communicate(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    try:
                        proc.communicate(timeout=15)
                    except subprocess.TimeoutExpired:
                        pass
                break

            time.sleep(state_poll_interval_seconds)

    log_tail = _log_tail()
    runtime_state = _load_json_dict(state_path)

    if killed_for_collapse:
        action_counts = collapse_state.get("action_counts", {}) if isinstance(collapse_state, dict) else {}
        short_actions = int(action_counts.get("short", 0))
        flat_actions = int(action_counts.get("flat", 0))
        long_actions = int(action_counts.get("long", 0))
        total_actions = max(short_actions + flat_actions + long_actions, 1)
        flat_ratio = float(flat_actions) / float(total_actions)
        position_ratio = 1.0 - flat_ratio
        probabilities = np.array([short_actions, flat_actions, long_actions], dtype=np.float64) / float(total_actions)
        entropy = float(-np.sum(np.where(probabilities > 0, probabilities * np.log(probabilities), 0.0)))
        periods_per_month = max(int(trading_config.get("periods_per_day", 96)) * 30, 1)
        server_cost_paid = float(total_actions / periods_per_month) * float(
            trading_config.get("monthly_server_cost_usd", 100.0)
        )
        gross_total_return = 0.0
        total_return = -server_cost_paid / max(initial_balance, 1e-9)
        last_idx = min(total_actions, len(target_segment)) - 1
        last_close = float(target_segment["close"].iloc[last_idx])
        bh_eval_return = (last_close / max(first_price, 1e-9)) - 1.0
        dominant_counts = [short_actions, flat_actions, long_actions]
        dominant_action = int(np.argmax(dominant_counts))
        dominant_ratio = float(max(dominant_counts)) / float(total_actions)
        summary = {
            "status": "EARLY_ABORT_COLLAPSED",
            "early_abort_reason": "flat_inactive_probe",
            "n_trades": int(collapse_state.get("n_trades", 0)) if isinstance(collapse_state, dict) else 0,
            "n_wins": int(collapse_state.get("n_wins", 0)) if isinstance(collapse_state, dict) else 0,
            "n_losses": int(collapse_state.get("n_losses", 0)) if isinstance(collapse_state, dict) else 0,
            "win_rate": float(collapse_state.get("win_rate", 0.0)) if isinstance(collapse_state, dict) else 0.0,
            "action_counts": {"short": short_actions, "flat": flat_actions, "long": long_actions},
            "last_signal": float(collapse_state.get("action_value", 0.0)) if isinstance(collapse_state, dict) else 0.0,
            "model_family": str(collapse_state.get("model_family", "")) if isinstance(collapse_state, dict) else "",
            "model_path": str(collapse_state.get("model_path", model_path)) if isinstance(collapse_state, dict) else str(model_path),
            "max_drawdown": float(collapse_state.get("max_drawdown", 0.0)) if isinstance(collapse_state, dict) else 0.0,
            "flat_ratio": flat_ratio,
            "position_ratio": position_ratio,
            "eval_action_entropy": max(entropy, 0.0),
            "eval_dominant_action": float(dominant_action),
            "eval_dominant_action_ratio": dominant_ratio,
            "eval_short_actions": float(short_actions),
            "eval_flat_actions": float(flat_actions),
            "eval_long_actions": float(long_actions),
            "server_cost_paid": server_cost_paid,
            "gross_total_return": gross_total_return,
            "total_return": total_return,
            "bh_eval_return": bh_eval_return,
            "outperformance_vs_bh": total_return - bh_eval_return,
            "avg_trades_per_episode": float(int(collapse_state.get("n_trades", 0))) if isinstance(collapse_state, dict) else 0.0,
            "eval_episodes": 1,
            "log_tail": log_tail[-1000:],
            "nautilus_log_path": str(log_path).replace("\\", "/"),
        }
    else:
        if killed_for_timeout:
            state_status = str(runtime_state.get("status", "UNKNOWN")) if isinstance(runtime_state, dict) else "UNKNOWN"
            state_event = ""
            if (
                isinstance(runtime_state, dict)
                and isinstance(runtime_state.get("recent_events"), list)
                and runtime_state.get("recent_events")
                and isinstance(runtime_state.get("recent_events")[-1], dict)
            ):
                state_event = str(runtime_state.get("recent_events")[-1].get("message", ""))
            raise RuntimeError(
                f"Nautilus subprocess validation timed out for {label} after {subprocess_timeout_seconds}s "
                f"(state={state_status} event={state_event!r} log={str(log_path)}) "
                f"log_tail={log_tail[-2000:]}"
            )
        if proc.returncode != 0:
            state_status = str(runtime_state.get("status", "UNKNOWN")) if isinstance(runtime_state, dict) else "UNKNOWN"
            raise RuntimeError(
                f"Nautilus subprocess validation failed for {label}: exit={proc.returncode} state={state_status} "
                f"log={str(log_path)} log_tail={log_tail[-2000:]}"
            )
        summary = _load_json_dict(summary_path)
        if summary is None:
            if isinstance(runtime_state, dict) and str(runtime_state.get("status", "")).upper() == "COMPLETE":
                action_counts = runtime_state.get("action_counts", {}) if isinstance(runtime_state.get("action_counts"), dict) else {}
                summary = {
                    "status": str(runtime_state.get("status", "COMPLETE")),
                    "model_family": str(runtime_state.get("model_family", "")),
                    "model_path": str(runtime_state.get("model_path", model_path)),
                    "n_trades": int(runtime_state.get("n_trades", 0)),
                    "n_wins": int(runtime_state.get("n_wins", 0)),
                    "n_losses": int(runtime_state.get("n_losses", 0)),
                    "win_rate": float(runtime_state.get("win_rate", 0.0)),
                    "action_counts": {
                        "short": int(action_counts.get("short", 0)),
                        "flat": int(action_counts.get("flat", 0)),
                        "long": int(action_counts.get("long", 0)),
                    },
                    "max_drawdown": float(runtime_state.get("max_drawdown", 0.0)),
                    "flat_ratio": float(runtime_state.get("flat_ratio", 1.0)),
                    "position_ratio": float(runtime_state.get("position_ratio", 0.0)),
                    "eval_action_entropy": float(runtime_state.get("eval_action_entropy", 0.0)),
                    "eval_dominant_action_ratio": float(runtime_state.get("eval_dominant_action_ratio", 1.0)),
                    "eval_short_actions": float(action_counts.get("short", 0.0)),
                    "eval_flat_actions": float(action_counts.get("flat", 0.0)),
                    "eval_long_actions": float(action_counts.get("long", 0.0)),
                    "gross_total_return": float(runtime_state.get("gross_total_return", runtime_state.get("total_return", 0.0))),
                    "total_return": float(runtime_state.get("total_return", 0.0)),
                    "server_cost_paid": float(runtime_state.get("server_cost_paid", 0.0)),
                    "bh_eval_return": float(runtime_state.get("bh_eval_return", 0.0)),
                    "outperformance_vs_bh": float(runtime_state.get("outperformance_vs_bh", 0.0)),
                    "avg_trades_per_episode": float(runtime_state.get("n_trades", 0.0)),
                    "eval_episodes": 1,
                }
            else:
                state_status = str(runtime_state.get("status", "UNKNOWN")) if isinstance(runtime_state, dict) else "UNKNOWN"
                state_event = ""
                if (
                    isinstance(runtime_state, dict)
                    and isinstance(runtime_state.get("recent_events"), list)
                    and runtime_state.get("recent_events")
                    and isinstance(runtime_state.get("recent_events")[-1], dict)
                ):
                    state_event = str(runtime_state.get("recent_events")[-1].get("message", ""))
                raise RuntimeError(
                    f"Nautilus subprocess validation did not produce a usable summary for {label}: "
                    f"state={state_status} event={state_event!r} summary={str(summary_path)} "
                    f"log={str(log_path)} log_tail={log_tail[-2000:]}"
                )
        if isinstance(runtime_state, dict) and isinstance(runtime_state.get("history"), dict):
            summary["history"] = copy.deepcopy(runtime_state["history"])

    summary = _apply_nautilus_target_slice_metrics(
        summary,
        target_frame=target_segment,
        target_start=target_start,
        target_end=end,
        prepared_start=prepared_start,
        score_start_index=score_start_index,
        initial_balance=initial_balance,
        monthly_server_cost_usd=float(trading_config.get("monthly_server_cost_usd", 100.0)),
        periods_per_day=int(trading_config.get("periods_per_day", 96)),
    )
    summary["segment_label"] = label
    summary["trade_size_qty"] = float(trade_qty)
    summary["trade_size_notional_usdt"] = float(trade_qty * first_price)
    summary["subprocess_validation"] = True
    summary["nautilus_log_path"] = str(log_path).replace("\\", "/")

    if cleanup_validation_artifacts:
        for artifact_path in (segment_path, state_path, log_path):
            try:
                artifact_path.unlink(missing_ok=True)
            except Exception:
                pass
    return summary


# =============================================================================
# Phase 4: MoE Routing
# =============================================================================

def test_moe_routing(prices: np.ndarray) -> dict:
    """Test MoE router on price data."""
    from models.moe.router import MoERouter

    router = MoERouter(n_experts=6, top_k=2)

    # Compute returns for routing
    returns = np.diff(np.log(prices))
    if len(returns) < 20:
        return {"status": "SKIP", "reason": "not enough data"}

    # Test routing at different points
    regime_counts = {}
    n_samples = min(1000, len(returns) - 20)
    sample_indices = np.linspace(20, len(returns) - 1, n_samples, dtype=int)

    for idx in sample_indices:
        vol = returns[max(0, idx - 20):idx].std()
        routing = router.route(returns[:idx], vol)
        top_expert = routing[0][0]
        regime_counts[top_expert] = regime_counts.get(top_expert, 0) + 1

    total = sum(regime_counts.values())
    regime_pcts = {k: v / total * 100 for k, v in sorted(regime_counts.items())}

    logger.info(f"MoE regime distribution: {regime_pcts}")
    return {"status": "OK", "regime_distribution": regime_pcts, "n_samples": n_samples}


# =============================================================================
# Phase 5: RL Training
# =============================================================================

def train_rl_agent(
    feature_array: np.ndarray,
    prices: np.ndarray,
    config: dict,
    dashboard=None,
) -> tuple[object | None, dict]:
    """Train the selected policy family and return (model, metrics)."""
    training_config = config.get("training", {})
    rl_config = training_config.get("rl", {})
    primary_model = str(training_config.get("primary_model", "hybrid")).strip().lower()
    ppo_enabled = bool(rl_config.get("enabled", True))
    if primary_model in {
        "supervised",
        "supervised_logreg",
        "supervised_catboost",
        "supervised_gpu",
        "catboost",
        "logreg",
        "supervised_only",
    }:
        ppo_enabled = False

    total_timesteps = rl_config.get("total_timesteps", 10000)
    learning_rate = rl_config.get("learning_rate", 3e-4)
    n_steps = rl_config.get("n_steps", 512)
    max_episode_steps = int(rl_config.get("max_episode_steps", n_steps))
    batch_size = rl_config.get("batch_size", 32)
    n_epochs = rl_config.get("n_epochs", 5)
    algo = rl_config.get("algo", "PPO")
    training_device = str(training_config.get("device", "auto"))
    supervised_config = training_config.get("supervised_fallback", {})
    eval_episodes = rl_config.get("eval_episodes", 8)

    try:
        from models.rl.trainer import RLTrainer
    except ImportError:
        logger.error("Cannot import RLTrainer")
        return None, {"error": "import failed"}

    # Trading params โ€” เนเธเนเธเนเธฒเน€เธ”เธตเธขเธงเธเธฑเธเธ—เธฑเนเธ Env เนเธฅเธฐ BacktestRunner
    trading_config = config.get("trading", {})
    validation_config = training_config.get("validation", {})
    selection_windows = int(validation_config.get("selection_windows", 4))
    selection_min_window_bars = int(validation_config.get("selection_min_window_bars", 4096))
    ranges = _compute_data_ranges(
        len(prices),
        test_ratio=validation_config.get("holdout_test_ratio", 0.20),
        validation_ratio_within_train=validation_config.get("validation_ratio_within_train", 0.10),
    )
    config["_data_ranges"] = ranges

    # OHLCV data เธชเธณเธซเธฃเธฑเธ realistic intra-candle simulation
    ohlcv_data = config.get("_ohlcv_data")  # passed from pipeline

    trainer = RLTrainer(
        feature_arrays=feature_array,
        price_series=prices,
        ohlcv_data=ohlcv_data,
        algo=algo,
        checkpoint_dir="checkpoints",
        checkpoint_interval=training_config.get("checkpoint_interval", 300),
        leverage=trading_config.get("leverage", 1.0),
        min_trade_pct=trading_config.get("min_trade_pct", 0.02),
        maintenance_margin=trading_config.get("maintenance_margin", 0.005),
        funding_interval=rl_config.get("funding_interval", 32),
        max_episode_steps=max_episode_steps,
        monthly_server_cost_usd=trading_config.get("monthly_server_cost_usd", 100.0),
        periods_per_day=trading_config.get("periods_per_day", 96),
        pnl_reward_scale=rl_config.get("pnl_reward_scale", 100.0),
        drawdown_penalty_scale=rl_config.get("drawdown_penalty_scale", 2.0),
        turnover_penalty_scale=rl_config.get("turnover_penalty_scale", 0.0),
        inactive_episode_penalty=rl_config.get("inactive_episode_penalty", 0.5),
        static_position_episode_penalty=rl_config.get("static_position_episode_penalty", 0.0),
        # Reward v3.1 hyperparams — wire from config so env actually sees the configured values.
        opp_cost_scale=rl_config.get("opp_cost_scale", 0.4),
        hold_bonus_scale=rl_config.get("hold_bonus_scale", 0.5),
        sharpe_bonus_scale=rl_config.get("sharpe_bonus_scale", 0.3),
        balanced_sampling=rl_config.get("balanced_sampling", True),
        regime_label_threshold=rl_config.get("regime_label_threshold", 0.02),
        selection_max_dominant_action_ratio=rl_config.get("selection_max_dominant_action_ratio", 0.92),
        selection_min_avg_trades_per_episode=rl_config.get("selection_min_avg_trades_per_episode", 1.5),
        selection_min_action_entropy=rl_config.get("selection_min_action_entropy", 0.05),
        train_range=ranges["train"],
        eval_range=ranges["validation"],
        test_range=ranges["test"],
    )
    if dashboard is not None:
        trainer._dashboard = dashboard

    logger.info(
        "%s %s: %d steps, lr=%s, batch=%d, n_steps=%d",
        "Training" if ppo_enabled else "Configured RL candidate",
        algo,
        int(total_timesteps),
        learning_rate,
        int(batch_size),
        int(n_steps),
    )
    logger.info("Requested training device: %s", training_device)
    logger.info(
        "Data split: train=[%d:%d) validation=[%d:%d) test=[%d:%d)",
        ranges["train"][0],
        ranges["train"][1],
        ranges["validation"][0],
        ranges["validation"][1],
        ranges["test"][0],
        ranges["test"][1],
    )

    _log_gpu_memory()

    if ppo_enabled:
        metrics = trainer.train(
            total_timesteps=total_timesteps,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            ent_coef=rl_config.get("ent_coef", 0.08),
            eval_every_steps=rl_config.get("eval_every_steps", 50000),
            eval_episodes=eval_episodes,
            device=training_device,
            collapse_eval_patience=rl_config.get("collapse_eval_patience", 4),
            collapse_min_steps=rl_config.get("collapse_min_steps", 200000),
        )
    else:
        metrics = {
            "model_family": "ppo_disabled",
            "selection_best_score": float("-inf"),
            "selection_gate_passed": 0.0,
            "ppo_disabled": 1.0,
        }
        logger.info(
            "Primary model strategy '%s' -> skipping PPO training and selecting from supervised candidates only.",
            primary_model or "hybrid",
        )

    _log_gpu_memory()

    # Load trained model for backtesting
    ppo_model = None
    selected_model = None
    ppo_model_path = Path("checkpoints/rl_agent_final.zip")
    if ppo_enabled and ppo_model_path.exists():
        try:
            from stable_baselines3 import PPO, SAC
            ModelClass = PPO if algo == "PPO" else SAC
            ppo_model = ModelClass.load(str(ppo_model_path), device=training_device)
            logger.info(f"Loaded trained model for evaluation on device: {training_device}")
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
    selected_model = ppo_model
    selected_metrics = dict(metrics)
    if ppo_model is not None:
        selected_metrics["model_family"] = "ppo"
        selected_metrics["model_path"] = str(ppo_model_path).replace("\\", "/")
    candidate_records: list[dict] = []

    ppo_validation_score = float("-inf")
    ppo_validation_metrics = None
    if ppo_model is not None:
        ppo_validation_metrics = trainer._eval_walkforward_segments(
            ppo_model,
            segment_range=ranges["validation"],
            n_windows=selection_windows,
            min_window_size=selection_min_window_bars,
        )
        ppo_validation_score = trainer._selection_score(ppo_validation_metrics)
        logger.info(
            "PPO walk-forward selection: score=%s median_alpha=%.4f median_net=%.4f worst_net=%.4f trades=%.1f dominant_ratio=%.2f%% windows=%d",
            f"{ppo_validation_score:.4f}" if np.isfinite(ppo_validation_score) else "-inf",
            float(ppo_validation_metrics.get("outperformance_vs_bh", 0.0)),
            float(ppo_validation_metrics.get("total_return", 0.0)),
            float(ppo_validation_metrics.get("walkforward_min_total_return", ppo_validation_metrics.get("total_return", 0.0))),
            float(ppo_validation_metrics.get("avg_trades_per_episode", 0.0)),
            float(ppo_validation_metrics.get("eval_dominant_action_ratio", 0.0) * 100.0),
            int(ppo_validation_metrics.get("walkforward_window_count", 1)),
        )
        candidate_records.append(
            {
                "family": "ppo",
                "model": ppo_model,
                "model_path": str(ppo_model_path).replace("\\", "/"),
                "display_metrics": dict(selected_metrics),
                "env_validation_score": float(ppo_validation_score),
                "env_validation_metrics": dict(ppo_validation_metrics),
            }
        )

    if supervised_config.get("enabled", True):
        try:
            from models.rl.supervised import build_constant_action_model, train_supervised_action_model

            confidence_grid = supervised_config.get("confidence_grid", [0.34, 0.38, 0.42, 0.46, 0.50])
            long_confidence_grid = supervised_config.get("long_confidence_grid", confidence_grid)
            short_confidence_grid = supervised_config.get("short_confidence_grid", confidence_grid)
            anchor_long_confidence_grid = supervised_config.get("anchor_long_confidence_grid", [])
            anchor_short_confidence_grid = supervised_config.get("anchor_short_confidence_grid", [])
            exploration_long_confidence_grid = supervised_config.get("exploration_long_confidence_grid", [])
            exploration_short_confidence_grid = supervised_config.get("exploration_short_confidence_grid", [])
            preserve_top_fallback_count = int(
                supervised_config.get("preserve_top_confidence_grid_count", 2)
            )
            model_types = supervised_config.get("model_types", ["logreg", "extratrees"])
            anchor_long_confidence_values = {
                round(float(value), 2)
                for value in anchor_long_confidence_grid
                if np.isfinite(value)
            }
            anchor_short_confidence_values = {
                round(float(value), 2)
                for value in anchor_short_confidence_grid
                if np.isfinite(value)
            }
            if isinstance(model_types, str):
                model_types = [model_types]
            recent_train_bars = int(supervised_config.get("recent_train_bars", 0))
            supervised_validation_episodes = int(supervised_config.get("validation_episodes", max(eval_episodes, 10)))
            supervised_validation_seed_runs = max(int(supervised_config.get("validation_seed_runs", 1)), 1)
            supervised_validation_seed_stride = max(int(supervised_config.get("validation_seed_stride", 500)), 1)
            supervised_validation_probe_seed_runs = min(
                supervised_validation_seed_runs,
                max(int(supervised_config.get("validation_probe_seed_runs", 1)), 1),
            )
            supervised_validation_skip_walkforward_on_probe_collapse = bool(
                supervised_config.get("validation_probe_skip_walkforward_on_collapse", True)
            )
            supervised_min_selection_score = float(supervised_config.get("min_selection_score", 0.0))
            supervised_fallback_to_safe_flat = bool(supervised_config.get("fallback_to_safe_flat", True))
            validation_seed_base = int(training_config.get("seed", 42)) + 1000
            test_seed_base = int(training_config.get("seed", 42)) + 2000

            relaxed_min_avg_trades = float(
                supervised_config.get("relaxed_min_avg_trades_per_episode", 0.5)
            )
            relaxed_max_dominant_ratio = float(
                supervised_config.get("relaxed_max_dominant_action_ratio", 0.985)
            )
            relaxed_min_action_entropy = float(
                supervised_config.get("relaxed_min_action_entropy", 0.0)
            )
            regime_confidence_relief = float(
                supervised_config.get("regime_confidence_relief", 0.0)
            )
            walkforward_score_weight = float(supervised_config.get("walkforward_score_weight", 0.20))
            walkforward_min_active_ratio = float(supervised_config.get("walkforward_min_active_ratio", 0.10))
            # Defaults aligned with default.yaml — previous 4.0/3.0/3.0 fallbacks
            # silently overrode the config when keys were missing or typo'd.
            walkforward_inactivity_penalty_scale = float(
                supervised_config.get("walkforward_inactivity_penalty_scale", 0.10)
            )
            walkforward_worst_net_penalty_scale = float(
                supervised_config.get("walkforward_worst_net_penalty_scale", 0.10)
            )
            walkforward_worst_gross_penalty_scale = float(
                supervised_config.get("walkforward_worst_gross_penalty_scale", 0.10)
            )
            walkforward_worst_alpha_penalty_scale = float(
                supervised_config.get("walkforward_worst_alpha_penalty_scale", 0.10)
            )
            walkforward_dominance_penalty_scale = float(
                supervised_config.get("walkforward_dominance_penalty_scale", 0.10)
            )

            sup_best_model = None
            sup_best_meta = None
            sup_best_conf = None
            sup_best_long_conf = None
            sup_best_short_conf = None
            sup_best_score = float("-inf")
            sup_best_priority = -1
            sup_best_mode = "strict"
            sup_best_validation_metrics = None
            supervised_candidate_pool: list[dict] = []
            supervised_candidate_pool_size = max(
                int(supervised_config.get("candidate_pool_size", 32)),
                1,
            )
            supervised_train_range = ranges["train"]
            if recent_train_bars > 0:
                train_end = int(ranges["train"][1])
                train_start = max(int(ranges["train"][0]), train_end - recent_train_bars)
                supervised_train_range = (train_start, train_end)
            logger.info(
                "Supervised fallback selection: train=[%d:%d) validation=[%d:%d) "
                "recent_train_bars=%d validation_episodes=%d validation_seed_runs=%d probe_seed_runs=%d min_selection_score=%.4f "
                "walkforward_windows=%d",
                int(supervised_train_range[0]),
                int(supervised_train_range[1]),
                int(ranges["validation"][0]),
                int(ranges["validation"][1]),
                recent_train_bars,
                supervised_validation_episodes,
                supervised_validation_seed_runs,
                supervised_validation_probe_seed_runs,
                supervised_min_selection_score,
                selection_windows,
            )

            for model_type in model_types:
                sup_model, sup_meta = train_supervised_action_model(
                    feature_array=feature_array,
                    prices=prices,
                    train_range=supervised_train_range,
                    validation_range=ranges["validation"],
                    model_type=model_type,
                    horizon=supervised_config.get("horizon", 16),
                    min_return_threshold=supervised_config.get("min_return_threshold", 0.003),
                    threshold_quantile=supervised_config.get("threshold_quantile", 0.65),
                    max_train_samples=supervised_config.get("max_train_samples", 120_000),
                    logistic_c=supervised_config.get("logistic_c", 1.0),
                    logistic_multi_class=supervised_config.get("logistic_multi_class", "ovr"),
                    logistic_target_mode=supervised_config.get("logistic_target_mode", "binary_meta"),
                    catboost_iterations=supervised_config.get("catboost_iterations", 256),
                    catboost_depth=supervised_config.get("catboost_depth", 6),
                    catboost_learning_rate=supervised_config.get("catboost_learning_rate", 0.05),
                    catboost_l2_leaf_reg=supervised_config.get("catboost_l2_leaf_reg", 3.0),
                    catboost_border_count=supervised_config.get("catboost_border_count", 128),
                    catboost_task_type=supervised_config.get(
                        "catboost_task_type",
                        "GPU" if str(training_device).strip().lower() == "cuda" else "CPU",
                    ),
                    catboost_devices=supervised_config.get("catboost_devices", "0"),
                    extra_trees_n_estimators=supervised_config.get("extra_trees_n_estimators", 160),
                    extra_trees_max_depth=supervised_config.get("extra_trees_max_depth", 12),
                    extra_trees_min_samples_leaf=supervised_config.get("extra_trees_min_samples_leaf", 32),
                    min_hold_steps=supervised_config.get("min_hold_steps", 16),
                    reversal_margin=supervised_config.get("reversal_margin", 0.08),
                    entry_margin=supervised_config.get("entry_margin", 0.08),
                    exit_to_flat_margin=supervised_config.get("exit_to_flat_margin", 0.05),
                    max_hold_steps=supervised_config.get("max_hold_steps", 64),
                    stop_loss_threshold=supervised_config.get("stop_loss_threshold", -0.012),
                    drawdown_exit_threshold=supervised_config.get("drawdown_exit_threshold", 0.04),
                    trend_alignment_threshold=supervised_config.get("trend_alignment_threshold", 0.0015),
                    countertrend_margin=supervised_config.get("countertrend_margin", 0.08),
                    regime_confidence_relief=regime_confidence_relief,
                    flat_reentry_cooldown_steps=supervised_config.get("flat_reentry_cooldown_steps", 0),
                    flat_patience_steps=supervised_config.get("flat_patience_steps", 0),
                    flat_patience_threshold_relief=supervised_config.get("flat_patience_threshold_relief", 0.0),
                    flat_patience_entry_margin_relief=supervised_config.get("flat_patience_entry_margin_relief", 0.0),
                    countertrend_threshold_penalty=supervised_config.get("countertrend_threshold_penalty", 0.0),
                    countertrend_entry_penalty=supervised_config.get("countertrend_entry_penalty", 0.0),
                    meta_label_min_edge=supervised_config.get("meta_label_min_edge", 0.0),
                    meta_label_edge_margin=supervised_config.get("meta_label_edge_margin", 0.0),
                    meta_label_exit_edge=supervised_config.get("meta_label_exit_edge", 0.0),
                    meta_label_min_positive_rate=supervised_config.get("meta_label_min_positive_rate", 0.47),
                    meta_label_target_edge_threshold=supervised_config.get("meta_label_target_edge_threshold"),
                    calibration_min_samples=supervised_config.get("calibration_min_samples", 64),
                    calibration_probability_thresholds=supervised_config.get("calibration_probability_thresholds"),
                    calibration_window_count=supervised_config.get("calibration_window_count", 4),
                    calibration_window_min_samples=supervised_config.get("calibration_window_min_samples"),
                    calibration_min_active_window_ratio=supervised_config.get("calibration_min_active_window_ratio", 0.50),
                    taker_fee=trading_config.get("taker_fee", 0.0005),
                    slippage_bps=trading_config.get("slippage_bps", 1.0),
                    leverage=trading_config.get("leverage", 1.0),
                    random_state=training_config.get("seed", 42),
                    dashboard=dashboard,
                )

                post_cost_calibration = sup_meta.get("post_cost_calibration", {})
                calibrated_long_confidence_grid = _derive_calibrated_confidence_grid(
                    post_cost_calibration.get("long") if isinstance(post_cost_calibration, dict) else None,
                    long_confidence_grid,
                    min_edge=float(supervised_config.get("meta_label_min_edge", 0.0)),
                    min_positive_rate=float(supervised_config.get("meta_label_min_positive_rate", 0.47)),
                    min_active_window_ratio=float(supervised_config.get("calibration_min_active_window_ratio", 0.50)),
                    anchor_thresholds=anchor_long_confidence_grid,
                    preserve_top_fallback_count=preserve_top_fallback_count,
                )
                calibrated_short_confidence_grid = _derive_calibrated_confidence_grid(
                    post_cost_calibration.get("short") if isinstance(post_cost_calibration, dict) else None,
                    short_confidence_grid,
                    min_edge=float(supervised_config.get("meta_label_min_edge", 0.0)),
                    min_positive_rate=float(supervised_config.get("meta_label_min_positive_rate", 0.47)),
                    min_active_window_ratio=float(supervised_config.get("calibration_min_active_window_ratio", 0.50)),
                    anchor_thresholds=anchor_short_confidence_grid,
                    preserve_top_fallback_count=preserve_top_fallback_count,
                )
                search_long_confidence_grid = _build_supervised_search_grid(
                    calibrated_long_confidence_grid,
                    exploration_grid=exploration_long_confidence_grid,
                    anchor_grid=anchor_long_confidence_grid,
                )
                search_short_confidence_grid = _build_supervised_search_grid(
                    calibrated_short_confidence_grid,
                    exploration_grid=exploration_short_confidence_grid,
                    anchor_grid=anchor_short_confidence_grid,
                )
                calibrated_long_confidence_values = {
                    round(float(value), 2)
                    for value in calibrated_long_confidence_grid
                    if np.isfinite(value)
                }
                calibrated_short_confidence_values = {
                    round(float(value), 2)
                    for value in calibrated_short_confidence_grid
                    if np.isfinite(value)
                }
                logger.info(
                    "Calibrated confidence grids: model=%s long=%s short=%s search_long=%s search_short=%s",
                    model_type,
                    calibrated_long_confidence_grid,
                    calibrated_short_confidence_grid,
                    search_long_confidence_grid,
                    search_short_confidence_grid,
                )
                search_pair_count = int(len(search_long_confidence_grid) * len(search_short_confidence_grid))
                projected_seed_eval_passes = int(search_pair_count * supervised_validation_seed_runs)
                projected_walkforward_passes = int(search_pair_count * selection_windows)
                logger.info(
                    "Supervised search workload: model=%s threshold_pairs=%d projected_seed_eval_passes=%d "
                    "projected_walkforward_passes=%d probe_seed_runs=%d",
                    model_type,
                    search_pair_count,
                    projected_seed_eval_passes,
                    projected_walkforward_passes,
                    supervised_validation_probe_seed_runs,
                )

                for long_confidence in search_long_confidence_grid:
                    for short_confidence in search_short_confidence_grid:
                        exploration_only = (
                            round(float(long_confidence), 2) not in calibrated_long_confidence_values
                            or round(float(short_confidence), 2) not in calibrated_short_confidence_values
                        )
                        sup_model.confidence_threshold = float(min(float(long_confidence), float(short_confidence)))
                        sup_model.long_confidence_threshold = float(long_confidence)
                        sup_model.short_confidence_threshold = float(short_confidence)
                        sup_val_runs: list[dict] = []
                        strict_scores: list[float] = []
                        relaxed_scores: list[float] = []
                        collapsed_probe = False
                        for seed_idx in range(supervised_validation_seed_runs):
                            seed_base = validation_seed_base + seed_idx * supervised_validation_seed_stride
                            sup_val_metrics = trainer._eval_multi_episode(
                                sup_model,
                                n_episodes=supervised_validation_episodes,
                                segment_range=ranges["validation"],
                                log_episodes=False,
                                seed_base=seed_base,
                            )
                            sup_val_runs.append(dict(sup_val_metrics))
                            strict_scores.append(float(trainer.score_candidate(sup_val_metrics)))
                            relaxed_scores.append(float(trainer.score_candidate(
                                sup_val_metrics,
                                max_dominant_action_ratio=relaxed_max_dominant_ratio,
                                min_avg_trades_per_episode=relaxed_min_avg_trades,
                                min_action_entropy=relaxed_min_action_entropy,
                            )))
                            if (
                                supervised_validation_skip_walkforward_on_probe_collapse
                                and seed_idx + 1 == supervised_validation_probe_seed_runs
                                and len(sup_val_runs) >= supervised_validation_probe_seed_runs
                                and all(
                                    _is_collapsed_supervised_probe_metrics(run_metrics)
                                    for run_metrics in sup_val_runs[-supervised_validation_probe_seed_runs:]
                                )
                            ):
                                collapsed_probe = True
                                logger.info(
                                    "Skipping remaining validation seeds and walk-forward for collapsed probe: "
                                    "model=%s long_conf=%.2f short_conf=%.2f probe_runs=%d",
                                    model_type,
                                    float(long_confidence),
                                    float(short_confidence),
                                    supervised_validation_probe_seed_runs,
                                )
                                break
                        strict_score = _robust_validation_score(strict_scores)
                        relaxed_score = _robust_validation_score(relaxed_scores)
                        if collapsed_probe:
                            walkforward_metrics = dict(sup_val_runs[-1]) if sup_val_runs else {}
                            walkforward_metrics["walkforward_window_count"] = 0
                            walkforward_metrics["walkforward_min_total_return"] = float(
                                walkforward_metrics.get("total_return", 0.0)
                            )
                            walkforward_metrics["walkforward_min_gross_total_return"] = float(
                                walkforward_metrics.get("gross_total_return", walkforward_metrics.get("total_return", 0.0))
                            )
                            walkforward_metrics["walkforward_min_alpha"] = float(
                                walkforward_metrics.get("outperformance_vs_bh", 0.0)
                            )
                            walkforward_metrics["walkforward_net_std"] = 0.0
                            walkforward_metrics["walkforward_gross_std"] = 0.0
                            walkforward_metrics["walkforward_active_window_ratio"] = 0.0
                            walkforward_metrics["walkforward_active_window_count"] = 0.0
                            walkforward_metrics["walkforward_positive_net_ratio"] = 0.0
                            walkforward_metrics["walkforward_positive_alpha_ratio"] = 0.0
                            walkforward_metrics["walkforward_worst_dominant_action_ratio"] = float(
                                walkforward_metrics.get("eval_dominant_action_ratio", 1.0)
                            )
                            walkforward_strict_score = float("-inf")
                            walkforward_relaxed_score = float("-inf")
                            strict_combined_score = float("-inf")
                            relaxed_combined_score = float("-inf")
                            walkforward_strict_soft_score = float("-inf")
                            walkforward_relaxed_soft_score = float("-inf")
                            strict_wf_details = {
                                "walkforward_active_window_ratio": 0.0,
                                "walkforward_soft_score": float("-inf"),
                                "walkforward_penalty": 0.0,
                                "walkforward_bonus": 0.0,
                            }
                            relaxed_wf_details = dict(strict_wf_details)
                        else:
                            walkforward_metrics = trainer._eval_walkforward_segments(
                                sup_model,
                                segment_range=ranges["validation"],
                                n_windows=selection_windows,
                                min_window_size=selection_min_window_bars,
                            )
                            walkforward_strict_score = float(trainer.score_candidate(walkforward_metrics))
                            walkforward_relaxed_score = float(trainer.score_candidate(
                                walkforward_metrics,
                                max_dominant_action_ratio=relaxed_max_dominant_ratio,
                                min_avg_trades_per_episode=relaxed_min_avg_trades,
                                min_action_entropy=relaxed_min_action_entropy,
                            ))
                            strict_combined_score, walkforward_strict_soft_score, strict_wf_details = _combine_supervised_validation_scores(
                                strict_score,
                                walkforward_metrics,
                                trainer,
                                max_dominant_action_ratio=trainer.selection_max_dominant_action_ratio,
                                min_avg_trades_per_episode=trainer.selection_min_avg_trades_per_episode,
                                min_action_entropy=trainer.selection_min_action_entropy,
                                walkforward_score_weight=walkforward_score_weight,
                                walkforward_min_active_ratio=walkforward_min_active_ratio,
                                walkforward_inactivity_penalty_scale=walkforward_inactivity_penalty_scale,
                                walkforward_worst_net_penalty_scale=walkforward_worst_net_penalty_scale,
                                walkforward_worst_gross_penalty_scale=walkforward_worst_gross_penalty_scale,
                                walkforward_worst_alpha_penalty_scale=walkforward_worst_alpha_penalty_scale,
                                walkforward_dominance_penalty_scale=walkforward_dominance_penalty_scale,
                            )
                            relaxed_combined_score, walkforward_relaxed_soft_score, relaxed_wf_details = _combine_supervised_validation_scores(
                                relaxed_score,
                                walkforward_metrics,
                                trainer,
                                max_dominant_action_ratio=relaxed_max_dominant_ratio,
                                min_avg_trades_per_episode=relaxed_min_avg_trades,
                                min_action_entropy=relaxed_min_action_entropy,
                                walkforward_score_weight=walkforward_score_weight,
                                walkforward_min_active_ratio=walkforward_min_active_ratio,
                                walkforward_inactivity_penalty_scale=walkforward_inactivity_penalty_scale,
                                walkforward_worst_net_penalty_scale=walkforward_worst_net_penalty_scale,
                                walkforward_worst_gross_penalty_scale=walkforward_worst_gross_penalty_scale,
                                walkforward_worst_alpha_penalty_scale=walkforward_worst_alpha_penalty_scale,
                                walkforward_dominance_penalty_scale=walkforward_dominance_penalty_scale,
                            )
                        strict_metrics = _aggregate_validation_metrics(sup_val_runs, strict_scores)
                        relaxed_metrics = _aggregate_validation_metrics(sup_val_runs, relaxed_scores)
                        watchlist_metrics = dict(walkforward_metrics)
                        watchlist_metrics["supervised_validation_brier"] = float(
                            sup_meta.get("validation_brier", float("inf"))
                        )
                        watchlist_metrics["validation_seed_runs"] = int(supervised_validation_seed_runs)
                        watchlist_metrics["validation_score_median"] = float(
                            relaxed_metrics.get("validation_score_median", float("-inf"))
                        )
                        watchlist_metrics["validation_score_worst"] = float(
                            relaxed_metrics.get("validation_score_worst", float("-inf"))
                        )
                        watchlist_metrics["walkforward_selection_score"] = float(walkforward_relaxed_soft_score)
                        watchlist_metrics["trade_density"] = float(
                            relaxed_metrics.get(
                                "avg_trades_per_episode",
                                relaxed_metrics.get("n_trades", 0.0),
                            )
                        ) / max(float(trainer.max_episode_steps), 1.0)
                        watchlist_score = _score_sparse_supervised_watchlist_candidate(watchlist_metrics)
                        for metric_bundle, wf_score, wf_details in (
                            (strict_metrics, walkforward_strict_soft_score, strict_wf_details),
                            (relaxed_metrics, walkforward_relaxed_soft_score, relaxed_wf_details),
                        ):
                            metric_bundle["walkforward_window_count"] = int(
                                walkforward_metrics.get("walkforward_window_count", 1)
                            )
                            metric_bundle["walkforward_min_total_return"] = float(
                                walkforward_metrics.get(
                                    "walkforward_min_total_return",
                                    walkforward_metrics.get("total_return", 0.0),
                                )
                            )
                            metric_bundle["walkforward_min_gross_total_return"] = float(
                                walkforward_metrics.get(
                                    "walkforward_min_gross_total_return",
                                    walkforward_metrics.get("gross_total_return", 0.0),
                                )
                            )
                            metric_bundle["walkforward_min_alpha"] = float(
                                walkforward_metrics.get(
                                    "walkforward_min_alpha",
                                    walkforward_metrics.get("outperformance_vs_bh", 0.0),
                                )
                            )
                            metric_bundle["walkforward_net_std"] = float(
                                walkforward_metrics.get("walkforward_net_std", 0.0)
                            )
                            metric_bundle["walkforward_gross_std"] = float(
                                walkforward_metrics.get("walkforward_gross_std", 0.0)
                            )
                            metric_bundle["walkforward_selection_score"] = float(wf_score)
                            metric_bundle["walkforward_active_window_ratio"] = float(
                                walkforward_metrics.get("walkforward_active_window_ratio", 0.0)
                            )
                            metric_bundle["walkforward_active_window_count"] = float(
                                walkforward_metrics.get("walkforward_active_window_count", 0.0)
                            )
                            metric_bundle["walkforward_positive_net_ratio"] = float(
                                walkforward_metrics.get("walkforward_positive_net_ratio", 0.0)
                            )
                            metric_bundle["walkforward_positive_alpha_ratio"] = float(
                                walkforward_metrics.get("walkforward_positive_alpha_ratio", 0.0)
                            )
                            metric_bundle["walkforward_worst_dominant_action_ratio"] = float(
                                walkforward_metrics.get(
                                    "walkforward_worst_dominant_action_ratio",
                                    walkforward_metrics.get("eval_dominant_action_ratio", 1.0),
                                )
                            )
                            metric_bundle["walkforward_penalty"] = float(wf_details.get("walkforward_penalty", 0.0))
                            metric_bundle["walkforward_bonus"] = float(wf_details.get("walkforward_bonus", 0.0))
                            metric_bundle["supervised_validation_brier"] = float(
                                sup_meta.get("validation_brier", float("nan"))
                            )
                        logger.info(
                            "Supervised validation: model=%s long_conf=%.2f short_conf=%.2f strict=%s relaxed=%s watchlist=%s "
                            "wf_strict=%s wf_relaxed=%s strict_combined=%s relaxed_combined=%s seed_runs=%d "
                            "alpha=%.4f net=%.4f gross=%.4f trades=%.1f dominant_ratio=%.2f%% "
                            "wf_active=%.2f%% wf_penalty=%.4f brier=%.4f episodes=%d",
                            model_type,
                            float(long_confidence),
                            float(short_confidence),
                            f"{strict_score:.4f}" if np.isfinite(strict_score) else "-inf",
                            f"{relaxed_score:.4f}" if np.isfinite(relaxed_score) else "-inf",
                            f"{watchlist_score:.4f}" if np.isfinite(watchlist_score) else "-inf",
                            f"{walkforward_strict_soft_score:.4f}" if np.isfinite(walkforward_strict_soft_score) else "-inf",
                            f"{walkforward_relaxed_soft_score:.4f}" if np.isfinite(walkforward_relaxed_soft_score) else "-inf",
                            f"{strict_combined_score:.4f}" if np.isfinite(strict_combined_score) else "-inf",
                            f"{relaxed_combined_score:.4f}" if np.isfinite(relaxed_combined_score) else "-inf",
                            supervised_validation_seed_runs,
                            float(relaxed_metrics.get("outperformance_vs_bh", 0.0)),
                            float(relaxed_metrics.get("total_return", 0.0)),
                            float(relaxed_metrics.get("gross_total_return", 0.0)),
                            float(relaxed_metrics.get("avg_trades_per_episode", 0.0)),
                            float(relaxed_metrics.get("eval_dominant_action_ratio", 0.0) * 100.0),
                            float(relaxed_metrics.get("walkforward_active_window_ratio", 0.0) * 100.0),
                            float(relaxed_metrics.get("walkforward_penalty", 0.0)),
                            float(relaxed_metrics.get("supervised_validation_brier", float("nan"))),
                            int(relaxed_metrics.get("eval_episodes", supervised_validation_episodes)),
                        )
                        candidate_mode = None
                        candidate_score = float("-inf")
                        candidate_priority = -1
                        candidate_validation_metrics = None
                        candidate_options = []
                        if np.isfinite(strict_combined_score):
                            candidate_options.append(("strict", float(strict_combined_score), 1, strict_metrics))
                        if np.isfinite(relaxed_combined_score):
                            candidate_options.append(("relaxed", float(relaxed_combined_score), 0, relaxed_metrics))
                        if candidate_options and not exploration_only:
                            candidate_mode, candidate_score, candidate_priority, candidate_validation_metrics = max(
                                candidate_options,
                                key=lambda item: (
                                    item[1],
                                    item[2],
                                    -float(item[3].get("supervised_validation_brier", float("inf"))),
                                ),
                            )

                        if (not exploration_only) and candidate_mode is not None and (
                            candidate_score > sup_best_score
                            or (np.isclose(candidate_score, sup_best_score) and candidate_priority > sup_best_priority)
                        ):
                            sup_best_model = sup_model
                            sup_best_meta = dict(sup_meta)
                            sup_best_conf = float(min(float(long_confidence), float(short_confidence)))
                            sup_best_long_conf = float(long_confidence)
                            sup_best_short_conf = float(short_confidence)
                            sup_best_score = float(candidate_score)
                            sup_best_priority = int(candidate_priority)
                            sup_best_mode = str(candidate_mode)
                            sup_best_validation_metrics = dict(candidate_validation_metrics or relaxed_metrics)
                        pool_mode = candidate_mode
                        pool_score = candidate_score
                        pool_priority = candidate_priority
                        pool_validation_metrics = candidate_validation_metrics
                        if pool_mode is None and np.isfinite(watchlist_score):
                            pool_mode = "watchlist"
                            pool_score = float(watchlist_score)
                            pool_priority = -1
                            pool_validation_metrics = dict(watchlist_metrics)
                        is_anchor_candidate = (
                            round(float(long_confidence), 2) in anchor_long_confidence_values
                            and round(float(short_confidence), 2) in anchor_short_confidence_values
                        )
                        if pool_mode is None and is_anchor_candidate:
                            anchor_metrics = dict(relaxed_metrics)
                            anchor_probe_score = _score_anchor_supervised_probe_candidate(anchor_metrics)
                            if np.isfinite(anchor_probe_score):
                                pool_mode = "anchor_probe"
                                pool_score = float(anchor_probe_score)
                                pool_priority = -2
                                pool_validation_metrics = anchor_metrics
                        if pool_mode is not None and np.isfinite(pool_score):
                            candidate_name = (
                                f"{model_type}_{pool_mode}_"
                                f"lc{float(long_confidence):.2f}_sc{float(short_confidence):.2f}"
                            )
                            candidate_snapshot = copy.deepcopy(sup_model)
                            candidate_snapshot.confidence_threshold = float(
                                min(float(long_confidence), float(short_confidence))
                            )
                            candidate_snapshot.long_confidence_threshold = float(long_confidence)
                            candidate_snapshot.short_confidence_threshold = float(short_confidence)
                            supervised_candidate_pool.append(
                                {
                                    "model": candidate_snapshot,
                                    "meta": dict(sup_meta),
                                    "score": float(pool_score),
                                    "priority": int(pool_priority),
                                    "mode": str(pool_mode),
                                    "name": candidate_name,
                                    "confidence": float(min(float(long_confidence), float(short_confidence))),
                                    "long_confidence": float(long_confidence),
                                    "short_confidence": float(short_confidence),
                                    "validation_metrics": dict(pool_validation_metrics or relaxed_metrics),
                                }
                            )
                            supervised_candidate_pool.sort(
                                key=lambda item: (
                                    float(item["score"]),
                                    int(item["priority"]),
                                    -float(item["validation_metrics"].get("supervised_validation_brier", float("inf"))),
                                ),
                                reverse=True,
                            )

            supervised_candidate_eligible = (
                sup_best_model is not None
                and sup_best_conf is not None
                and np.isfinite(sup_best_score)
                and sup_best_score >= supervised_min_selection_score
            )
            original_candidate_pool_size = len(supervised_candidate_pool)
            supervised_candidate_pool = _prune_supervised_candidate_pool(supervised_candidate_pool)
            if len(supervised_candidate_pool) != original_candidate_pool_size:
                logger.info(
                    "Pruned supervised candidate pool by behavior signature: %d -> %d",
                    original_candidate_pool_size,
                    len(supervised_candidate_pool),
                )
            if len(supervised_candidate_pool) > supervised_candidate_pool_size:
                trimmed_size = len(supervised_candidate_pool)
                supervised_candidate_pool = _trim_supervised_candidate_pool(
                    supervised_candidate_pool,
                    limit=supervised_candidate_pool_size,
                )
                logger.info(
                    "Trimmed supervised candidate pool with family preservation: %d -> %d",
                    trimmed_size,
                    len(supervised_candidate_pool),
                )

            supervised_path = None
            supervised_metrics = None
            if sup_best_model is not None and sup_best_conf is not None:
                sup_best_model.confidence_threshold = float(sup_best_conf)
                sup_best_model.long_confidence_threshold = float(
                    sup_best_long_conf if sup_best_long_conf is not None else sup_best_conf
                )
                sup_best_model.short_confidence_threshold = float(
                    sup_best_short_conf if sup_best_short_conf is not None else sup_best_conf
                )
                supervised_path = sup_best_model.save("checkpoints/rl_agent_supervised.joblib")
                logger.info(
                    "Saved supervised fallback -> %s (mode=%s long_conf=%.2f short_conf=%.2f validation_score=%s)",
                    supervised_path,
                    sup_best_mode,
                    float(sup_best_long_conf if sup_best_long_conf is not None else sup_best_conf),
                    float(sup_best_short_conf if sup_best_short_conf is not None else sup_best_conf),
                    f"{sup_best_score:.4f}" if np.isfinite(sup_best_score) else "-inf",
                )

                use_supervised = False
                if supervised_candidate_eligible and np.isfinite(sup_best_score):
                    use_supervised = (not np.isfinite(ppo_validation_score)) or (sup_best_score > ppo_validation_score)
                if supervised_candidate_eligible and not np.isfinite(ppo_validation_score) and np.isfinite(sup_best_score):
                    use_supervised = True

                if use_supervised:
                    supervised_metrics = trainer._eval_multi_episode(
                        sup_best_model,
                        n_episodes=10,
                        segment_range=ranges["test"],
                        log_episodes=True,
                        seed_base=test_seed_base,
                    )
                    trainer._save_chart(sup_best_model, supervised_metrics)
                    supervised_metrics["selection_gate_passed"] = 1.0
                    supervised_metrics["selection_best_score"] = float(sup_best_score)
                    supervised_metrics["model_family"] = f"supervised_{sup_best_meta.get('model_type', 'fallback')}"
                    supervised_metrics["model_path"] = str(supervised_path).replace("\\", "/")
                    supervised_metrics["supervised_confidence_threshold"] = float(sup_best_conf)
                    supervised_metrics["supervised_long_confidence_threshold"] = float(
                        sup_best_long_conf if sup_best_long_conf is not None else sup_best_conf
                    )
                    supervised_metrics["supervised_short_confidence_threshold"] = float(
                        sup_best_short_conf if sup_best_short_conf is not None else sup_best_conf
                    )
                    supervised_metrics["supervised_label_threshold"] = float(sup_best_meta.get("label_threshold", 0.0))
                    supervised_metrics["supervised_model_type"] = str(sup_best_meta.get("model_type", "fallback"))
                    supervised_metrics["supervised_validation_brier"] = float(
                        sup_best_meta.get("validation_brier", float("nan"))
                    )
                    supervised_metrics["selection_mode"] = sup_best_mode
                    supervised_metrics["selection_engine"] = "episodic_validation_multiseed_walkforward"
                    if sup_best_validation_metrics is not None:
                        supervised_metrics["validation_score_median"] = float(
                            sup_best_validation_metrics.get("validation_score_median", float("nan"))
                        )
                        supervised_metrics["validation_score_worst"] = float(
                            sup_best_validation_metrics.get("validation_score_worst", float("nan"))
                        )
                        supervised_metrics["validation_score_robust"] = float(
                            sup_best_validation_metrics.get("validation_score_robust", float("nan"))
                        )
                        supervised_metrics["validation_seed_runs"] = float(
                            sup_best_validation_metrics.get("validation_seed_runs", supervised_validation_seed_runs)
                        )
                    logger.info(
                        "Using supervised fallback model: mode=%s long_conf=%.2f short_conf=%.2f "
                        "validation_score=%s ppo_validation_score=%s",
                        sup_best_mode,
                        float(sup_best_long_conf if sup_best_long_conf is not None else sup_best_conf),
                        float(sup_best_short_conf if sup_best_short_conf is not None else sup_best_conf),
                        f"{sup_best_score:.4f}" if np.isfinite(sup_best_score) else "-inf",
                        f"{ppo_validation_score:.4f}" if np.isfinite(ppo_validation_score) else "-inf",
                    )
                    selected_model = sup_best_model
                    selected_metrics = supervised_metrics
                else:
                    if supervised_candidate_eligible:
                        logger.info(
                            "Keeping PPO model: best supervised mode=%s score=%s ppo validation score=%s",
                            sup_best_mode,
                            f"{sup_best_score:.4f}" if np.isfinite(sup_best_score) else "-inf",
                            f"{ppo_validation_score:.4f}" if np.isfinite(ppo_validation_score) else "-inf",
                        )
                    else:
                        logger.warning(
                            "Best supervised candidate score %.4f is below the fast-env minimum %.4f. "
                            "Keeping it available for Nautilus validation instead of forcing safe_flat immediately.",
                            float(sup_best_score),
                            float(supervised_min_selection_score),
                        )
            if supervised_candidate_pool and sup_best_model is None:
                logger.info(
                    "No strict/relaxed supervised winner passed fast-env gates, but %d watchlist/anchor candidates remain for Nautilus validation.",
                    len(supervised_candidate_pool),
                )

            for idx, candidate_info in enumerate(supervised_candidate_pool):
                candidate_model = candidate_info["model"]
                candidate_meta = candidate_info["meta"]
                candidate_mode_value = str(candidate_info["mode"])
                candidate_score_value = float(candidate_info["score"])
                candidate_conf_value = float(candidate_info["confidence"])
                candidate_long_conf_value = float(candidate_info["long_confidence"])
                candidate_short_conf_value = float(candidate_info["short_confidence"])
                candidate_validation_metrics = dict(candidate_info["validation_metrics"])
                is_selected_env_candidate = (
                    sup_best_model is not None
                    and np.isclose(candidate_score_value, sup_best_score)
                    and np.isclose(candidate_long_conf_value, float(sup_best_long_conf or sup_best_conf or 0.0))
                    and np.isclose(candidate_short_conf_value, float(sup_best_short_conf or sup_best_conf or 0.0))
                )
                if is_selected_env_candidate and supervised_path is not None:
                    candidate_path = supervised_path
                else:
                    candidate_path = candidate_model.save(
                        f"checkpoints/rl_agent_supervised_{candidate_info['name']}.joblib"
                    )
                candidate_records.append(
                    {
                        "family": f"supervised_{candidate_meta.get('model_type', 'fallback')}",
                        "candidate_name": str(candidate_info.get("name", f"candidate_{idx + 1}")),
                        "model": candidate_model,
                        "model_path": str(candidate_path).replace("\\", "/"),
                        "display_metrics": (
                            dict(supervised_metrics) if (is_selected_env_candidate and supervised_metrics is not None) else None
                        ),
                        "env_validation_score": float(candidate_score_value),
                        "env_validation_metrics": candidate_validation_metrics,
                        "selection_mode": candidate_mode_value,
                        "supervised_confidence_threshold": float(candidate_conf_value),
                        "supervised_long_confidence_threshold": float(candidate_long_conf_value),
                        "supervised_short_confidence_threshold": float(candidate_short_conf_value),
                        "supervised_label_threshold": float(candidate_meta.get("label_threshold", 0.0)),
                        "supervised_model_type": str(candidate_meta.get("model_type", "fallback")),
                    }
                )

            if (
                supervised_fallback_to_safe_flat
                and selected_model is None
                and not candidate_records
            ):
                safe_flat_model = build_constant_action_model(
                    feature_dim=int(feature_array.shape[1]),
                    action=1,
                    metadata={"model_type": "safe_flat", "reason": "no_profitable_candidate"},
                )
                safe_flat_path = safe_flat_model.save("checkpoints/rl_agent_safe_flat.joblib")
                safe_flat_metrics = trainer._eval_multi_episode(
                    safe_flat_model,
                    n_episodes=10,
                    segment_range=ranges["test"],
                    log_episodes=True,
                    seed_base=test_seed_base,
                )
                trainer._save_chart(safe_flat_model, safe_flat_metrics)
                safe_flat_metrics["selection_gate_passed"] = 1.0
                safe_flat_metrics["selection_best_score"] = 0.0
                safe_flat_metrics["model_family"] = "safe_flat"
                safe_flat_metrics["model_path"] = str(safe_flat_path).replace("\\", "/")
                safe_flat_metrics["selection_mode"] = "safety"
                safe_flat_metrics["selection_reason"] = "no_supervised_candidate"
                selected_model = safe_flat_model
                selected_metrics = safe_flat_metrics
                candidate_records.append(
                    {
                        "family": "safe_flat",
                        "model": safe_flat_model,
                        "model_path": str(safe_flat_path).replace("\\", "/"),
                        "display_metrics": dict(safe_flat_metrics),
                        "env_validation_score": 0.0,
                        "env_validation_metrics": None,
                        "selection_mode": "safety",
                    }
                )
                logger.warning(
                    "No supervised candidate could be built. Falling back to safe flat policy."
                )
        except Exception as e:
            logger.exception("Supervised fallback training failed: %s", e)

    force_safe_flat_after_nautilus_rejection = False
    nautilus_config = training_config.get("nautilus_validation", {})
    nautilus_enabled = False  # Completely abandoned Nautilus in favor of custom vectorized execution
    has_non_safe_flat_candidate = any(
        str(candidate.get("family", "")) != "safe_flat"
        for candidate in candidate_records
    )
    if nautilus_enabled and candidate_records and has_non_safe_flat_candidate:
        frame_15m = config.get("_nautilus_frame")
        if not isinstance(frame_15m, pd.DataFrame) or len(frame_15m) != len(prices):
            logger.warning(
                "Nautilus validation enabled but no aligned 15m frame is available; skipping realistic selection."
            )
        else:
            min_trades = float(nautilus_config.get("selection_min_trades", 1.0))
            max_dom = float(nautilus_config.get("selection_max_dominant_action_ratio", 0.995))
            min_active_window_ratio = float(nautilus_config.get("selection_min_active_window_ratio", 1.0 / 3.0))
            min_nautilus_selection_score = float(nautilus_config.get("min_selection_score", 0.0))
            conservative_fallback_min_score = float(
                nautilus_config.get("conservative_fallback_min_score", 0.75)
            )
            nautilus_selection_windows = max(int(nautilus_config.get("selection_windows", 2)), 1)
            nautilus_selection_min_window_bars = max(int(nautilus_config.get("selection_min_window_bars", 4096)), 128)
            use_for_selection = bool(nautilus_config.get("use_for_model_selection", True))
            evaluate_final_test = bool(nautilus_config.get("evaluate_final_test", True))
            # Default flipped to False — subprocess spawn per candidate adds 10-50x
            # wall time and the in-process Nautilus runner already isolates state.
            # Configs can opt back in by setting subprocess_on_windows: true.
            use_subprocess_validation = bool(nautilus_config.get("subprocess_on_windows", False)) and sys.platform.startswith("win")
            if use_subprocess_validation:
                logger.info(
                    "Using subprocess Nautilus validation on Windows to avoid logger re-initialization crashes."
                )
            best_candidate = None
            best_nautilus_score = float("-inf")
            best_rejected_candidate = None
            best_rejected_score = float("-inf")
            nautilus_candidate_limit = max(int(nautilus_config.get("top_supervised_candidates", 2)), 1)
            candidate_pool = _select_nautilus_candidate_subset(
                candidate_records,
                limit=nautilus_candidate_limit,
            )
            logger.info(
                "Nautilus candidate subset (%d/%d): %s",
                len(candidate_pool),
                len(candidate_records),
                ", ".join(
                    str(item.get("candidate_name", item.get("family", "candidate")))
                    for item in candidate_pool
                ) or "none",
            )

            for candidate in candidate_pool:
                candidate_name = str(candidate.get("candidate_name", candidate.get("family", "candidate")))
                if str(candidate.get("family", "")) == "safe_flat":
                    candidate["nautilus_validation"] = {"skipped": "safe_flat_candidate"}
                    candidate["nautilus_validation_score"] = float("-inf")
                    logger.info("Skipping Nautilus validation for safe_flat candidate.")
                    continue
                env_score = float(candidate.get("env_validation_score", float("-inf")))
                if not np.isfinite(env_score):
                    logger.info(
                        "Skipping Nautilus validation for %s because fast-env score is non-finite (%s)",
                        candidate.get("family", "unknown"),
                        "-inf" if not np.isfinite(env_score) else f"{env_score:.4f}",
                    )
                    candidate["nautilus_validation"] = {"skipped": "non_finite_fast_env_score"}
                    candidate["nautilus_validation_score"] = float("-inf")
                    continue
                try:
                    validation_runner = (
                        _run_nautilus_backtest_segment_subprocess
                        if use_subprocess_validation
                        else _run_nautilus_backtest_segment
                    )
                    nautilus_windows = trainer._build_walkforward_windows(
                        segment_range=ranges["validation"],
                        n_windows=nautilus_selection_windows,
                        min_window_size=nautilus_selection_min_window_bars,
                    )
                    validation_window_summaries: list[dict] = []
                    validation_window_scores: list[float] = []
                    for window_idx, window_range in enumerate(nautilus_windows, start=1):
                        window_label = (
                            f"validation_{candidate_name}"
                            if len(nautilus_windows) == 1
                            else f"validation_{candidate_name}_w{window_idx}"
                        )
                        window_summary = validation_runner(
                            model_path=candidate["model_path"],
                            frame_15m=frame_15m,
                            segment_range=window_range,
                            config=config,
                            label=window_label,
                        )
                        validation_window_summaries.append(window_summary)
                        validation_window_scores.append(
                            _score_nautilus_summary(
                                window_summary,
                                min_trades=min_trades,
                                max_dominant_action_ratio=max_dom,
                                min_active_window_ratio=min_active_window_ratio,
                            )
                        )
                    validation_summary, validation_score = _aggregate_nautilus_window_summaries(
                        validation_window_summaries,
                        validation_window_scores,
                        nautilus_windows,
                    )
                    validation_preliminary_score = float(validation_score)
                    validation_summary["nautilus_score_preliminary"] = validation_preliminary_score
                    validation_score = _score_nautilus_summary(
                        validation_summary,
                        min_trades=min_trades,
                        max_dominant_action_ratio=max_dom,
                        min_active_window_ratio=min_active_window_ratio,
                    )
                    validation_summary["nautilus_score_final"] = (
                        float(validation_score) if np.isfinite(validation_score) else float("-inf")
                    )
                    fallback_score = _score_nautilus_rejected_fallback(validation_summary)
                    validation_summary["nautilus_fallback_score"] = (
                        float(fallback_score) if np.isfinite(fallback_score) else float("-inf")
                    )
                    candidate["nautilus_validation"] = validation_summary
                    candidate["nautilus_validation_score"] = float(validation_score)
                    candidate["nautilus_fallback_score"] = (
                        float(fallback_score) if np.isfinite(fallback_score) else float("-inf")
                    )
                    logger.info(
                        "Nautilus validation: family=%s candidate=%s long_conf=%.2f short_conf=%.2f "
                        "score=%s pre_score=%s fallback_score=%s windows=%d active_ratio=%.2f%% net=%.4f gross=%.4f alpha=%.4f "
                        "trades=%d trade_density=%.4f dominant_ratio=%.2f%% status=%s",
                        candidate["family"],
                        candidate_name,
                        float(candidate.get("supervised_long_confidence_threshold", candidate.get("supervised_confidence_threshold", 0.0))),
                        float(candidate.get("supervised_short_confidence_threshold", candidate.get("supervised_confidence_threshold", 0.0))),
                        f"{validation_score:.4f}" if np.isfinite(validation_score) else "-inf",
                        f"{validation_preliminary_score:.4f}" if np.isfinite(validation_preliminary_score) else "-inf",
                        f"{fallback_score:.4f}" if np.isfinite(fallback_score) else "-inf",
                        int(validation_summary.get("nautilus_window_count", 1)),
                        float(validation_summary.get("nautilus_active_window_ratio", 1.0) * 100.0),
                        float(validation_summary.get("total_return", 0.0)),
                        float(validation_summary.get("gross_total_return", 0.0)),
                        float(validation_summary.get("outperformance_vs_bh", 0.0)),
                        int(validation_summary.get("n_trades", 0)),
                        float(validation_summary.get("trade_density", 0.0)),
                        float(validation_summary.get("eval_dominant_action_ratio", 0.0) * 100.0),
                        str(validation_summary.get("status", "UNKNOWN")),
                    )
                    if np.isfinite(validation_score) and validation_score > best_nautilus_score:
                        best_nautilus_score = float(validation_score)
                        best_candidate = candidate
                    if (
                        not np.isfinite(validation_score)
                        and np.isfinite(fallback_score)
                        and fallback_score > best_rejected_score
                    ):
                        best_rejected_score = float(fallback_score)
                        best_rejected_candidate = candidate
                except Exception as exc:
                    candidate["nautilus_validation"] = {"error": str(exc)}
                    candidate["nautilus_validation_score"] = float("-inf")
                    candidate["nautilus_fallback_score"] = float("-inf")
                    logger.warning(
                        "Nautilus validation failed for %s: %s",
                        candidate["family"],
                        exc,
                    )

            nautilus_selection_engine = "nautilus_validation"
            if best_candidate is None and best_rejected_candidate is not None:
                rejected_summary = (
                    best_rejected_candidate.get("nautilus_validation", {})
                    if isinstance(best_rejected_candidate, dict)
                    else {}
                )
                allow_sparse_rejected = (
                    isinstance(rejected_summary, dict)
                    and _allows_sparse_active_nautilus_override(
                        rejected_summary,
                        min_trades=min_trades,
                        max_dominant_action_ratio=max_dom,
                        min_active_window_ratio=min_active_window_ratio,
                    )
                )
                if float(best_rejected_score) >= conservative_fallback_min_score or allow_sparse_rejected:
                    logger.warning(
                        "No model passed Nautilus validation gates; selecting best conservative Nautilus fallback candidate: "
                        "family=%s fallback_score=%.4f gate_score=-inf min_fallback_score=%.4f sparse_override=%s",
                        best_rejected_candidate["family"],
                        float(best_rejected_score),
                        float(conservative_fallback_min_score),
                        allow_sparse_rejected,
                    )
                    best_candidate = best_rejected_candidate
                    best_nautilus_score = float(best_rejected_score)
                    nautilus_selection_engine = "nautilus_conservative_fallback"
                else:
                    logger.warning(
                        "No model passed Nautilus validation gates and best fallback score %.4f is below "
                        "conservative minimum %.4f; deferring to downstream safe-flat fallback if enabled.",
                        float(best_rejected_score),
                        float(conservative_fallback_min_score),
                    )

            if best_candidate is not None:
                logger.info(
                    "Best Nautilus candidate: family=%s score=%.4f use_for_selection=%s min_selection_score=%.4f",
                    best_candidate["family"],
                    float(best_nautilus_score),
                    use_for_selection,
                    min_nautilus_selection_score,
                )
                if (
                    use_for_selection
                    and best_candidate["model"] is not None
                    and np.isfinite(best_nautilus_score)
                    and best_nautilus_score >= min_nautilus_selection_score
                ):
                    selected_model = best_candidate["model"]
                    if best_candidate.get("display_metrics") is None:
                        selected_metrics = trainer._eval_multi_episode(
                            best_candidate["model"],
                            n_episodes=10,
                            segment_range=ranges["test"],
                            log_episodes=True,
                        )
                        if str(best_candidate["family"]).startswith("supervised_"):
                            trainer._save_chart(best_candidate["model"], selected_metrics)
                    else:
                        selected_metrics = dict(best_candidate["display_metrics"])
                    selected_metrics["model_family"] = str(best_candidate["family"])
                    selected_metrics["model_path"] = str(best_candidate["model_path"])
                    selected_metrics["selection_engine"] = str(nautilus_selection_engine)
                    selected_metrics["selection_mode"] = str(best_candidate.get("selection_mode", "unknown"))
                    if str(best_candidate["family"]).startswith("supervised_"):
                        selected_metrics["supervised_confidence_threshold"] = float(
                            best_candidate.get("supervised_confidence_threshold", 0.0)
                        )
                        selected_metrics["supervised_long_confidence_threshold"] = float(
                            best_candidate.get("supervised_long_confidence_threshold", 0.0)
                        )
                        selected_metrics["supervised_short_confidence_threshold"] = float(
                            best_candidate.get("supervised_short_confidence_threshold", 0.0)
                        )
                        selected_metrics["supervised_label_threshold"] = float(
                            best_candidate.get("supervised_label_threshold", 0.0)
                        )
                        selected_metrics["supervised_model_type"] = str(
                            best_candidate.get("supervised_model_type", "fallback")
                        )
                    logger.info(
                        "Selecting final model using %s -> %s",
                        str(nautilus_selection_engine),
                        best_candidate["family"],
                    )
                elif use_for_selection:
                    logger.warning(
                        "Best Nautilus validation score %.4f is below the minimum %.4f. Keeping fast-env selection.",
                        float(best_nautilus_score),
                        float(min_nautilus_selection_score),
                    )
                    if supervised_fallback_to_safe_flat:
                        force_safe_flat_after_nautilus_rejection = True

                selected_metrics["nautilus_validation"] = dict(best_candidate["nautilus_validation"])
                selected_metrics["nautilus_validation_score"] = float(best_nautilus_score)

                if evaluate_final_test:
                    try:
                        final_runner = (
                            _run_nautilus_backtest_segment_subprocess
                            if use_subprocess_validation
                            else _run_nautilus_backtest_segment
                        )
                        final_summary = final_runner(
                            model_path=str(best_candidate["model_path"]),
                            frame_15m=frame_15m,
                            segment_range=ranges["test"],
                            config=config,
                            label=f"test_{str(best_candidate.get('candidate_name', best_candidate['family']))}",
                        )
                        selected_metrics = _promote_nautilus_test_metrics(selected_metrics, final_summary)
                        logger.info(
                            "Nautilus held-out test: family=%s net=%.4f gross=%.4f alpha=%.4f "
                            "trades=%d win_rate=%.2f%%",
                            best_candidate["family"],
                            float(final_summary.get("total_return", 0.0)),
                            float(final_summary.get("gross_total_return", 0.0)),
                            float(final_summary.get("outperformance_vs_bh", 0.0)),
                            int(final_summary.get("n_trades", 0)),
                            float(final_summary.get("win_rate", 0.0) * 100.0),
                        )
                    except Exception as exc:
                        selected_metrics["nautilus_test"] = {"error": str(exc)}
                        logger.warning("Nautilus held-out test failed for %s: %s", best_candidate["family"], exc)
            else:
                if use_for_selection and supervised_fallback_to_safe_flat:
                    logger.warning(
                        "No model passed Nautilus validation gates; deferring to downstream safe-flat fallback."
                    )
                    force_safe_flat_after_nautilus_rejection = True
                else:
                    logger.warning(
                        "No model passed Nautilus validation gates; keeping fast-env selection."
                    )

    if force_safe_flat_after_nautilus_rejection:
        selected_model = None
        selected_metrics = {}

    if supervised_fallback_to_safe_flat and selected_model is None:
        logger.warning(
            "No candidate survived fast-env plus Nautilus selection. Falling back to safe flat policy."
        )
        safe_flat_model = build_constant_action_model(
            feature_dim=int(feature_array.shape[1]),
            action=1,
            metadata={"model_type": "safe_flat", "reason": "no_candidate_survived_selection"},
        )
        safe_flat_path = safe_flat_model.save("checkpoints/rl_agent_safe_flat.joblib")
        safe_flat_metrics = trainer._eval_multi_episode(
            safe_flat_model,
            n_episodes=10,
            segment_range=ranges["test"],
            log_episodes=True,
            seed_base=test_seed_base,
        )
        trainer._save_chart(safe_flat_model, safe_flat_metrics)
        safe_flat_metrics["selection_gate_passed"] = 1.0
        safe_flat_metrics["selection_best_score"] = 0.0
        safe_flat_metrics["model_family"] = "safe_flat"
        safe_flat_metrics["model_path"] = str(safe_flat_path).replace("\\", "/")
        safe_flat_metrics["selection_mode"] = "safety"
        safe_flat_metrics["selection_reason"] = (
            "nautilus_rejected_all_candidates"
            if has_non_safe_flat_candidate
            else "no_supervised_candidate"
        )
        if candidate_records:
            best_candidate = max(
                candidate_records,
                key=lambda item: float(item.get("nautilus_validation_score", item.get("env_validation_score", float("-inf")))),
            )
            if isinstance(best_candidate.get("nautilus_validation"), dict):
                safe_flat_metrics["nautilus_validation"] = dict(best_candidate["nautilus_validation"])
                safe_flat_metrics["nautilus_validation_score"] = float(
                    best_candidate.get("nautilus_validation_score", float("-inf"))
                )
        selected_model = safe_flat_model
        selected_metrics = safe_flat_metrics

    if selected_model is not None and selected_metrics:
        trainer._save_chart(selected_model, selected_metrics)

    return selected_model, selected_metrics


# =============================================================================
# Phase 6: Backtest with Trained Model
# =============================================================================

def backtest_with_model(
    model,
    feature_array: np.ndarray,
    prices: np.ndarray,
    config: dict | None = None,
    data_ranges: dict[str, tuple[int, int]] | None = None,
) -> dict:
    """Run backtest using trained RL model via standardized RLTrainer evaluation."""
    from models.rl.trainer import RLTrainer
    
    trading_config = (config or {}).get("trading", {})
    validation_config = (config or {}).get("training", {}).get("validation", {})
    ranges = data_ranges or (config or {}).get("_data_ranges")
    if ranges is None:
        ranges = _compute_data_ranges(
            len(prices),
            test_ratio=validation_config.get("holdout_test_ratio", 0.20),
            validation_ratio_within_train=validation_config.get("validation_ratio_within_train", 0.10),
        )
    test_range = ranges["test"]

    if model is None:
        # For tests or scenarios with no model, return a baseline
        from models.rl.supervised import build_constant_action_model
        model = build_constant_action_model(action=1, feature_dim=int(feature_array.shape[1])) # FLAT (Stay in cash/safe)
        logger.warning("backtest_with_model called with model=None, defaulting to Constant FLAT model.")

    trainer = RLTrainer(
        feature_arrays=feature_array,
        price_series=prices,
        train_range=ranges["train"],
        test_range=test_range,
        taker_fee=float(trading_config.get("taker_fee", 0.0005)),
        leverage=float(trading_config.get("leverage", 1.0)),
        min_trade_pct=float(trading_config.get("min_trade_pct", 0.02)),
        monthly_server_cost_usd=float(trading_config.get("monthly_server_cost_usd", 100.0)),
        periods_per_day=int(trading_config.get("periods_per_day", 96)),
    )
    # Global 'dash' variable from run_training_pipeline
    try:
        from monitoring.display import Dashboard
        # Attempt to find the active dashboard instance if not passed
        trainer._dashboard = config.get("_dashboard")
    except:
        pass
    
    metrics = trainer._eval_full_segment(model, segment_range=test_range)
    
    return {
        "backtest_metrics": metrics,
        "n_bars": test_range[1] - test_range[0],
        "test_range": list(test_range),
    }


def backtest_baseline(prices: np.ndarray, ta_slice: np.ndarray) -> dict:
    """Run naive RSI baseline backtest for comparison."""
    rsi_values = ta_slice[:, 0]
    signals = np.where(rsi_values < 30, 0.5, np.where(rsi_values > 70, -0.5, 0.0))

    bt_config = BacktestConfig(maker_fee=0.0002, taker_fee=0.0005, slippage_bps=1.0)
    runner = BacktestRunner(bt_config)
    result = runner.run(prices, signals)
    return result.metrics

# =============================================================================
# Phase 7: Validation (CPCV, DSR)
# =============================================================================

def backtest_baseline(prices: np.ndarray, ta_slice: np.ndarray) -> dict:
    """Run naive RSI baseline backtest for comparison."""
    rsi_values = ta_slice[:, 0]
    signals = np.where(rsi_values < 30, 0.5, np.where(rsi_values > 70, -0.5, 0.0))

    bt_config = BacktestConfig(maker_fee=0.0002, taker_fee=0.0005, slippage_bps=1.0)
    runner = BacktestRunner(bt_config)
    result = runner.run(prices, signals)
    return result.metrics


# =============================================================================
# Phase 7: Validation (CPCV, DSR)
# =============================================================================

def run_validation(
    feature_array: np.ndarray,
    prices: np.ndarray,
    backtest_metrics: dict,
    config: dict,
) -> dict:
    """Run CPCV + DSR + PBO validation."""
    from features.validation import PurgedKFold, deflated_sharpe_ratio

    val_config = config.get("training", {}).get("validation", {})
    n_splits = val_config.get("cpcv_splits", 4)
    embargo_pct = val_config.get("embargo_pct", 0.01)
    n_trials = val_config.get("n_trials_dsr", 1)

    results = {}

    # CPCV splits
    pkf = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct)
    fold_sizes = []
    for train_idx, test_idx in pkf.split(len(feature_array)):
        fold_sizes.append((len(train_idx), len(test_idx)))

    results["cpcv"] = {
        "n_splits": n_splits,
        "embargo_pct": embargo_pct,
        "fold_sizes": fold_sizes,
        "status": "OK",
    }
    logger.info(f"CPCV: {n_splits} splits, embargo={embargo_pct}")

    # Deflated Sharpe Ratio
    sharpe = backtest_metrics.get("sharpe", 0)
    n_obs = len(prices)
    try:
        dsr_p = deflated_sharpe_ratio(
            sharpe_observed=sharpe,
            n_trials=max(n_trials, 1),
            n_observations=n_obs,
        )
        results["dsr"] = {
            "sharpe_observed": sharpe,
            "p_value": dsr_p,
            "significant": dsr_p > 0.95,
            "status": "OK",
        }
    except Exception as e:
        results["dsr"] = {"status": "ERROR", "error": str(e)}

    # Feature consistency check (subsample to realistic window size)
    # KS test เธเธฑเธ 100K samples เธญเนเธญเธเนเธซเธงเน€เธเธดเธเนเธ (detect noise เนเธกเนเนเธเน real drift)
    # เนเธเน 500 samples เน€เธซเธกเธทเธญเธ production drift detector window
    from features.validation import check_feature_consistency
    half = len(feature_array) // 2
    subsample_n = min(500, half)
    rng = np.random.RandomState(42)
    try:
        idx_a = rng.choice(half, size=subsample_n, replace=False)
        idx_b = rng.choice(np.arange(half, len(feature_array)), size=subsample_n, replace=False)
        consistency = check_feature_consistency(
            feature_array[idx_a],
            feature_array[idx_b],
        )
        results["feature_consistency"] = {
            "passed": consistency["passed"],
            "n_drifted": len(consistency["drifted_features"]),
            "n_features": feature_array.shape[1],
            "subsample_size": subsample_n,
            "status": "OK",
        }
    except Exception as e:
        results["feature_consistency"] = {"status": "ERROR", "error": str(e)}

    return results


# =============================================================================
# Phase 8: Risk Manager Test
# =============================================================================

def test_risk_manager() -> dict:
    """Test risk manager with sample scenarios."""
    from risk.manager import RiskManager

    rm = RiskManager(
        max_drawdown=0.15,
        daily_loss_limit=0.03,
        max_open_positions=5,
    )

    # Normal trade
    decision1 = rm.evaluate(
        symbol="BTCUSDT", direction=1.0, equity=10000,
        model_confidence=0.7, win_rate=0.55, avg_win=0.02,
        avg_loss=0.01, atr_value=0.015, current_drawdown=0.05,
    )

    # Drawdown exceeded
    decision2 = rm.evaluate(
        symbol="BTCUSDT", direction=1.0, equity=10000,
        model_confidence=0.7, win_rate=0.55, avg_win=0.02,
        avg_loss=0.01, atr_value=0.015, current_drawdown=0.20,
    )

    # Low confidence
    decision3 = rm.evaluate(
        symbol="BTCUSDT", direction=1.0, equity=10000,
        model_confidence=0.2, win_rate=0.55, avg_win=0.02,
        avg_loss=0.01, atr_value=0.015, current_drawdown=0.05,
    )

    results = {
        "normal_trade": {"approved": decision1.approved, "size": decision1.size},
        "drawdown_exceeded": {"approved": decision2.approved, "reason": decision2.reject_reason},
        "low_confidence": {"approved": decision3.approved, "size": decision3.size},
        "status": "OK",
    }

    passed = (
        decision1.approved
        and not decision2.approved
        and decision3.approved
        and decision3.size < decision1.size  # low confidence โ’ smaller size
    )
    results["all_checks_passed"] = passed
    logger.info(f"Risk Manager: {'PASS' if passed else 'FAIL'} โ€” {results}")
    return results


# =============================================================================
# Phase 9: Drift Detection Test
# =============================================================================

def test_drift_detection(feature_array: np.ndarray) -> dict:
    """Test drift detector with feature data."""
    from monitoring.live.drift_detector import DriftDetector

    # Subsample reference เน€เธเธทเนเธญเนเธซเน KS test เนเธกเน oversensitive
    # Production: reference = training data (~500 samples window)
    half = len(feature_array) // 2
    rng = np.random.RandomState(42)
    ref_idx = rng.choice(half, size=min(500, half), replace=False)
    ref = feature_array[ref_idx]

    detector = DriftDetector(
        reference_features=ref,
        window_size=min(500, half),
        check_interval=min(240, half // 2),
    )

    # Feed second half as if it's live data
    for i in range(half, min(half + 500, len(feature_array))):
        detector.update(feature_array[i], 0.0, 0.0)

    if detector.should_check():
        drift_result = detector.check_all()
    else:
        drift_result = detector.check_all()

    results = {
        "feature_drift": drift_result.get("feature_drift", False),
        "performance_drift": drift_result.get("performance_drift", False),
        "action_drift": drift_result.get("action_drift", False),
        "should_retrain": drift_result.get("should_retrain", False),
        "should_pause": drift_result.get("should_pause", False),
        "status": "OK",
    }
    logger.info(f"Drift Detection: {results}")
    return results


# =============================================================================
# Phase 10: Live Components Test (paper mode simulation)
# =============================================================================

def test_live_components() -> dict:
    """Test live trading components without real connection."""
    results = {}

    # 1. CandleAggregator
    from data.adapters.live import CandleAggregator
    agg = CandleAggregator(timeframe_seconds=900)

    base_ts = 1710000000000  # some timestamp
    # Simulate trades within one candle
    agg.on_trade(50000.0, 1.0, base_ts)
    agg.on_trade(50100.0, 0.5, base_ts + 60000)
    agg.on_trade(49900.0, 0.8, base_ts + 120000)
    assert not agg.candle_closed, "Candle should NOT close mid-candle"

    # Next candle period โ’ triggers close
    agg.on_trade(50200.0, 1.2, base_ts + 900000)
    assert agg.candle_closed, "Candle SHOULD close when new period starts"

    results["candle_aggregator"] = {"status": "OK", "candle_close_logic": "PASS"}
    logger.info("CandleAggregator: PASS")

    # 2. OrderManager (paper mode)
    from execution.live.order_manager import OrderManager
    om = OrderManager(paper_mode=True)
    assert om.get_balance() == 10000.0

    order = om.place_order("BTCUSDT", "buy", 0.1, 50000.0)
    assert order.status == "filled"
    assert order.is_paper

    results["order_manager"] = {"status": "OK", "paper_order": "PASS"}
    logger.info("OrderManager (paper): PASS")

    # 3. DashboardData
    from monitoring.live.dashboard import DashboardData
    dash = DashboardData()
    dash.add_record(
        timestamp=base_ts, balance=10000, position=0.1,
        pnl=0, action=0.5, confidence=0.7, symbol="BTCUSDT",
    )
    dash.add_record(
        timestamp=base_ts + 900000, balance=10050, position=0.1,
        pnl=50, action=0.5, confidence=0.7, symbol="BTCUSDT",
    )
    metrics = dash.get_metrics()
    assert "sharpe" in metrics

    results["dashboard"] = {"status": "OK", "metrics_computed": True}
    logger.info("DashboardData: PASS")

    # 4. AlertManager
    from monitoring.live.drift_detector import AlertManager
    am = AlertManager()  # no token = log only
    am.send_alert("Test alert", level="warning")
    results["alert_manager"] = {"status": "OK"}
    logger.info("AlertManager: PASS")

    # 5. W&B Tracker (disabled mode)
    from monitoring.training.wandb_tracker import WandbTracker
    tracker = WandbTracker(enabled=False)
    tracker.log({"test": 1.0})
    results["wandb_tracker"] = {"status": "OK", "mode": "disabled"}
    logger.info("WandbTracker (disabled): PASS")

    return results


# =============================================================================
# Phase 11: Forecasters (TimesFM + CryptoMamba)
# =============================================================================

def test_forecasters(prices: np.ndarray) -> dict:
    """Test TimesFM + CryptoMamba forecasters."""
    results = {}
    test_prices = prices[-200:]  # last 200 candles for quick test

    # TimesFM
    try:
        from models.forecast.timesfm_forecaster import TimesFMForecaster
        tfm = TimesFMForecaster(device="cpu")
        forecast, uncertainty = tfm.predict(test_prices, horizon=12)
        results["timesfm"] = {
            "available": tfm.available,
            "name": tfm.name(),
            "forecast_shape": list(forecast.shape),
            "uncertainty": float(uncertainty),
            "status": "OK",
        }
        logger.info(f"TimesFM: {tfm.name()}, uncertainty={uncertainty:.4f}")
    except Exception as e:
        results["timesfm"] = {"status": "ERROR", "error": str(e)}

    # CryptoMamba
    try:
        from models.forecast.crypto_mamba import CryptoMambaForecaster
        mamba = CryptoMambaForecaster(
            context_len=128, horizon=12,
            d_model=32, n_layers=2,  # tiny for test
            device="cpu",
        )

        # Predict (untrained โ€” expect random output)
        forecast, uncertainty = mamba.predict(test_prices, horizon=12)
        results["crypto_mamba_predict"] = {
            "available": mamba.available,
            "name": mamba.name(),
            "forecast_shape": list(forecast.shape),
            "uncertainty": float(uncertainty),
            "status": "OK",
        }
        logger.info(f"CryptoMamba predict: shape={forecast.shape}, unc={uncertainty:.4f}")

        # Quick fine-tune (tiny: 5 epochs on small data)
        if mamba.available:
            try:
                ft_result = mamba.fine_tune(
                    test_prices, epochs=5, lr=1e-3, batch_size=16,
                    save_path="checkpoints/crypto_mamba_test.pt",
                )
                results["crypto_mamba_finetune"] = {
                    "final_loss": ft_result.get("final_loss"),
                    "n_samples": ft_result.get("n_samples"),
                    "status": "OK" if "error" not in ft_result else "FAIL",
                }
                logger.info(f"CryptoMamba fine-tune: loss={ft_result.get('final_loss', 0):.6f}, "
                            f"samples={ft_result.get('n_samples', 0)}")

                # Predict after fine-tune
                forecast_ft, unc_ft = mamba.predict(test_prices, horizon=12)
                results["crypto_mamba_after_ft"] = {
                    "forecast_shape": list(forecast_ft.shape),
                    "uncertainty": float(unc_ft),
                }
                logger.info(f"CryptoMamba after fine-tune: unc={unc_ft:.4f}")
            except Exception as e:
                logger.error(f"CryptoMamba fine-tune failed: {e}")
                results["crypto_mamba_finetune"] = {"status": "ERROR", "error": str(e)}
    except Exception as e:
        logger.error(f"CryptoMamba failed: {e}")
        results["crypto_mamba"] = {"status": "ERROR", "error": str(e)}

    return results


# =============================================================================
# Phase 12: Benchmark Comparison
# =============================================================================

def test_benchmarks(prices: np.ndarray) -> dict:
    """Run all benchmark strategies."""
    from execution.backtest.benchmarks import run_all_benchmarks

    logger.info("Running benchmark strategies...")
    results_list = run_all_benchmarks(prices)

    return {
        "benchmarks": results_list,
        "n_strategies": len(results_list),
    }


# =============================================================================
# Phase 13: Multi-Regime Test
# =============================================================================

def test_regime_backtest(df: pd.DataFrame, ta_slice: np.ndarray) -> dict:
    """Run backtest across different market regimes.

    *** เนเธเน FULL data + aggregate เน€เธเนเธ 15m เธเนเธญเธเธฃเธฑเธ strategy ***
    - Full data เน€เธเธทเนเธญเธเธฃเธญเธเธเธฅเธธเธกเธ—เธธเธ regime (COVID เธ–เธถเธ Present)
    - 15m aggregation เน€เธเธทเนเธญเนเธกเน overtrade เน€เธซเธกเธทเธญเธ benchmarks
    """
    from execution.backtest.regime_test import run_regime_backtest, compute_regime_stats
    from execution.backtest.benchmarks import aggregate_to_candles

    # Load full data for regime coverage
    raw_path = Path("data/raw/BTCUSDT_1m.parquet")
    if raw_path.exists():
        df_full = pd.read_parquet(raw_path)
        logger.info(f"Loaded full data for regime test: {len(df_full):,} rows")
    else:
        df_full = df
        logger.warning("Full data not found, using subsampled data")

    # Aggregate to 15m for realistic strategy signals
    # *** RSI เธเธ 1m เธเธฐ overtrade 16K+ times โ’ -99% เธ—เธธเธ regime ***
    # *** RSI เธเธ 15m เธเธฐ trade ~1-2K times โ’ เธเธฅเธฅเธฑเธเธเนเธชเธกเธเธฃเธดเธ ***
    prices_1m = df_full["close"].values
    prices_15m = aggregate_to_candles(prices_1m, period=15)

    # Compute RSI on 15m candles
    from execution.backtest.benchmarks import _compute_rsi
    rsi_15m = _compute_rsi(prices_15m, period=14)
    signals_15m = np.where(rsi_15m < 30, 0.5, np.where(rsi_15m > 70, -0.5, 0.0))

    # Regime stats (on original 1m data for accuracy)
    logger.info("Computing regime statistics...")
    stats = compute_regime_stats(df_full)

    # Build a 15m DataFrame for regime test
    n_15m = len(prices_15m)
    # Create timestamps: take every 15th timestamp from original
    if "open_time" in df_full.columns:
        ts_1m = pd.to_datetime(df_full["open_time"].values)
        indices_15m = np.arange(14, min(n_15m * 15, len(ts_1m)), 15)[:n_15m]
        ts_15m = ts_1m[indices_15m]
    else:
        ts_15m = pd.date_range("2020-01-01", periods=n_15m, freq="15min")

    df_15m = pd.DataFrame({"open_time": ts_15m, "close": prices_15m[:len(ts_15m)]})

    # Regime backtest on 15m candles
    logger.info(f"Running regime backtest on {len(df_15m):,} 15m candles...")
    regime_results = run_regime_backtest(df_15m, signals_15m[:len(df_15m)])

    return {
        "regime_stats": stats,
        "regime_backtest": regime_results,
        "n_regimes": len([r for r in regime_results if r.get("regime") != "OVERALL"]),
        "timeframe": "15m",
    }


# =============================================================================
# Phase 14: Feature Importance
# =============================================================================

def test_feature_importance(feature_array: np.ndarray, prices: np.ndarray) -> dict:
    """Compute feature importance using MDI + MDA + SFI."""
    from features.importance import compute_feature_importance

    # Create binary labels: 1 if next return > 0, else 0
    returns = np.diff(prices) / prices[:-1]
    labels = (returns > 0).astype(int)

    # Align: features[:-1] โ’ labels
    X = feature_array[:-1]
    y = labels

    # Subsample for speed (full run would take too long)
    n_sub = min(10000, len(X))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X), size=n_sub, replace=False)
    idx.sort()
    X_sub = X[idx]
    y_sub = y[idx]

    # Replace NaN/inf
    X_sub = np.nan_to_num(X_sub, nan=0.0, posinf=0.0, neginf=0.0)

    # Train/test split (80/20, time-ordered)
    split = int(n_sub * 0.8)
    X_train, X_test = X_sub[:split], X_sub[split:]
    y_train, y_test = y_sub[:split], y_sub[split:]

    result = compute_feature_importance(
        X_train, y_train, X_test, y_test, top_k=20,
    )

    # Summary
    top5 = result["top_features"][:5]
    top5_summary = [(f['index'], round(f['combined'], 3)) for f in top5]
    logger.info(f"Top 5 features: {top5_summary}")

    return {
        "n_features": result["n_features"],
        "top_features": result["top_features"][:10],
        "n_subsample": n_sub,
    }


# =============================================================================
# Main Pipeline: test mode
# =============================================================================

def run_test_pipeline(config_path: str | None = None):
    """Full system test โ€” เธ—เธ”เธชเธญเธเธ—เธธเธ component เธ”เนเธงเธข RTX 2060 6GB.

    เธ—เธณเธ—เธธเธเธญเธขเนเธฒเธเน€เธซเธกเธทเธญเธ full training เนเธ•เนเนเธเน data เธเนเธญเธขเธฅเธ + timesteps เธเนเธญเธขเธฅเธ.
    เน€เธเนเธฒเธซเธกเธฒเธข: validate logic เธ—เธฑเนเธเธฃเธฐเธเธ เธเนเธญเธ deploy cloud GPU.
    """
    config_path = config_path or "configs/test_rtx2060.yaml"
    config = load_config(config_path)

    start_time = time.time()
    report = {"phases": {}, "errors": []}

    logger.info("=" * 70)
    logger.info("GARIC โ€” Full System Test (RTX 2060 Mode)")
    logger.info("=" * 70)

    # GPU check
    has_gpu = _log_gpu_info()
    report["gpu_available"] = has_gpu

    # ---- Phase 1: Data ----
    logger.info("\n>>> Phase 1: Load & Clean Data")
    t0 = time.time()
    try:
        symbol = config["data"]["pairs"][0]
        df, funding_df, sentiment_df = load_and_clean_data(config, symbol)
        report["phases"]["1_data"] = {
            "status": "OK",
            "symbol": symbol,
            "rows": len(df),
            "has_funding": funding_df is not None,
            "has_sentiment": sentiment_df is not None,
            "time_sec": round(time.time() - t0, 1),
        }
    except Exception as e:
        logger.error(f"Phase 1 FAILED: {e}")
        report["phases"]["1_data"] = {"status": "FAIL", "error": str(e)}
        report["errors"].append(f"Phase 1: {e}")
        _print_report(report, start_time)
        return report

    # ---- Phase 2: Features ----
    logger.info("\n>>> Phase 2: Build Features")
    t0 = time.time()
    try:
        feature_array, ta_slice, micro_slice, prices = build_features(df)
        report["phases"]["2_features"] = {
            "status": "OK",
            "shape": list(feature_array.shape),
            "time_sec": round(time.time() - t0, 1),
        }
    except Exception as e:
        logger.error(f"Phase 2 FAILED: {e}")
        report["phases"]["2_features"] = {"status": "FAIL", "error": str(e)}
        report["errors"].append(f"Phase 2: {e}")
        _print_report(report, start_time)
        return report

    # ---- Phase 3: Naive Forecast ----
    logger.info("\n>>> Phase 3: Naive Forecast Baseline")
    t0 = time.time()
    try:
        feature_array = add_naive_forecast(feature_array, prices)
        baseline_metrics = backtest_baseline(prices, ta_slice)
        report["phases"]["3_baseline"] = {
            "status": "OK",
            "metrics": baseline_metrics,
            "time_sec": round(time.time() - t0, 1),
        }
        logger.info(f"Baseline RSI backtest: {baseline_metrics}")
    except Exception as e:
        logger.error(f"Phase 3 FAILED: {e}")
        report["phases"]["3_baseline"] = {"status": "FAIL", "error": str(e)}
        report["errors"].append(f"Phase 3: {e}")

    # ---- Phase 4: MoE Routing ----
    logger.info("\n>>> Phase 4: MoE Routing")
    t0 = time.time()
    try:
        moe_result = test_moe_routing(prices)
        report["phases"]["4_moe"] = {**moe_result, "time_sec": round(time.time() - t0, 1)}
    except Exception as e:
        logger.error(f"Phase 4 FAILED: {e}")
        report["phases"]["4_moe"] = {"status": "FAIL", "error": str(e)}
        report["errors"].append(f"Phase 4: {e}")

    # ---- Phase 5: RL Training ----
    logger.info("\n>>> Phase 5: RL Agent Training")
    t0 = time.time()
    model = None
    try:
        model, rl_metrics = train_rl_agent(feature_array, prices, config)
        report["phases"]["5_rl_training"] = {
            "status": "OK" if "error" not in rl_metrics else "FAIL",
            "metrics": rl_metrics,
            "time_sec": round(time.time() - t0, 1),
        }
    except Exception as e:
        logger.error(f"Phase 5 FAILED: {e}")
        report["phases"]["5_rl_training"] = {"status": "FAIL", "error": str(e)}
        report["errors"].append(f"Phase 5: {e}")

    # ---- Phase 6: Backtest with Trained Model ----
    logger.info("\n>>> Phase 6: Backtest with RL Model")
    t0 = time.time()
    try:
        bt_result = backtest_with_model(
            model,
            feature_array,
            prices,
            config,
            data_ranges=config.get("_data_ranges"),
        )
        report["phases"]["6_rl_backtest"] = {
            "status": "OK",
            **bt_result,
            "time_sec": round(time.time() - t0, 1),
        }
        logger.info(f"RL backtest: {bt_result['backtest_metrics']}")
    except Exception as e:
        logger.error(f"Phase 6 FAILED: {e}")
        report["phases"]["6_rl_backtest"] = {"status": "FAIL", "error": str(e)}
        report["errors"].append(f"Phase 6: {e}")

    # ---- Phase 7: Validation ----
    logger.info("\n>>> Phase 7: Validation (CPCV + DSR)")
    t0 = time.time()
    try:
        bt_metrics = report.get("phases", {}).get("6_rl_backtest", {}).get("backtest_metrics", {})
        val_result = run_validation(feature_array, prices, bt_metrics, config)
        report["phases"]["7_validation"] = {
            "status": "OK",
            **val_result,
            "time_sec": round(time.time() - t0, 1),
        }
    except Exception as e:
        logger.error(f"Phase 7 FAILED: {e}")
        report["phases"]["7_validation"] = {"status": "FAIL", "error": str(e)}
        report["errors"].append(f"Phase 7: {e}")

    # ---- Phase 8: Risk Manager ----
    logger.info("\n>>> Phase 8: Risk Manager")
    t0 = time.time()
    try:
        risk_result = test_risk_manager()
        report["phases"]["8_risk"] = {**risk_result, "time_sec": round(time.time() - t0, 1)}
    except Exception as e:
        logger.error(f"Phase 8 FAILED: {e}")
        report["phases"]["8_risk"] = {"status": "FAIL", "error": str(e)}
        report["errors"].append(f"Phase 8: {e}")

    # ---- Phase 9: Drift Detection ----
    logger.info("\n>>> Phase 9: Drift Detection")
    t0 = time.time()
    try:
        drift_result = test_drift_detection(feature_array)
        report["phases"]["9_drift"] = {**drift_result, "time_sec": round(time.time() - t0, 1)}
    except Exception as e:
        logger.error(f"Phase 9 FAILED: {e}")
        report["phases"]["9_drift"] = {"status": "FAIL", "error": str(e)}
        report["errors"].append(f"Phase 9: {e}")

    # ---- Phase 10: Live Components ----
    logger.info("\n>>> Phase 10: Live Components (paper mode)")
    t0 = time.time()
    try:
        live_result = test_live_components()
        report["phases"]["10_live_components"] = {
            "status": "OK",
            "components": live_result,
            "time_sec": round(time.time() - t0, 1),
        }
    except Exception as e:
        logger.error(f"Phase 10 FAILED: {e}")
        report["phases"]["10_live_components"] = {"status": "FAIL", "error": str(e)}
        report["errors"].append(f"Phase 10: {e}")

    # ---- Phase 11: Forecasters (TimesFM + CryptoMamba) ----
    logger.info("\n>>> Phase 11: Forecasters (TimesFM + CryptoMamba)")
    t0 = time.time()
    try:
        forecast_result = test_forecasters(prices)
        report["phases"]["11_forecasters"] = {
            "status": "OK",
            **forecast_result,
            "time_sec": round(time.time() - t0, 1),
        }
    except Exception as e:
        logger.error(f"Phase 11 FAILED: {e}")
        report["phases"]["11_forecasters"] = {"status": "FAIL", "error": str(e)}
        report["errors"].append(f"Phase 11: {e}")

    # ---- Phase 12: Benchmark Comparison ----
    logger.info("\n>>> Phase 12: Benchmark Comparison")
    t0 = time.time()
    try:
        benchmark_result = test_benchmarks(prices)
        report["phases"]["12_benchmarks"] = {
            "status": "OK",
            **benchmark_result,
            "time_sec": round(time.time() - t0, 1),
        }
    except Exception as e:
        logger.error(f"Phase 12 FAILED: {e}")
        report["phases"]["12_benchmarks"] = {"status": "FAIL", "error": str(e)}
        report["errors"].append(f"Phase 12: {e}")

    # ---- Phase 13: Multi-Regime Test ----
    logger.info("\n>>> Phase 13: Multi-Regime Test")
    t0 = time.time()
    try:
        regime_result = test_regime_backtest(df, ta_slice)
        report["phases"]["13_regime_test"] = {
            "status": "OK",
            **regime_result,
            "time_sec": round(time.time() - t0, 1),
        }
    except Exception as e:
        logger.error(f"Phase 13 FAILED: {e}")
        report["phases"]["13_regime_test"] = {"status": "FAIL", "error": str(e)}
        report["errors"].append(f"Phase 13: {e}")

    # ---- Phase 14: Feature Importance ----
    logger.info("\n>>> Phase 14: Feature Importance (MDI + MDA + SFI)")
    t0 = time.time()
    try:
        fi_result = test_feature_importance(feature_array, prices)
        report["phases"]["14_feature_importance"] = {
            "status": "OK",
            **fi_result,
            "time_sec": round(time.time() - t0, 1),
        }
    except Exception as e:
        logger.error(f"Phase 14 FAILED: {e}")
        report["phases"]["14_feature_importance"] = {"status": "FAIL", "error": str(e)}
        report["errors"].append(f"Phase 14: {e}")

    # ---- Print Report ----
    _print_report(report, start_time)
    return report


# =============================================================================
# Main Pipeline: full training
# =============================================================================

def _aggregate_ohlcv_15m(df: pd.DataFrame, period: int = 15) -> pd.DataFrame:
    """Aggregate 1m OHLCV โ’ 15m OHLCV (proper aggregation, not just close)."""
    n = len(df)
    n_candles = n // period
    if n_candles <= 0:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])

    trim_n = n_candles * period
    opens = df["open"].to_numpy(dtype=np.float64, copy=False)[:trim_n].reshape(n_candles, period)
    highs = df["high"].to_numpy(dtype=np.float64, copy=False)[:trim_n].reshape(n_candles, period)
    lows = df["low"].to_numpy(dtype=np.float64, copy=False)[:trim_n].reshape(n_candles, period)
    closes = df["close"].to_numpy(dtype=np.float64, copy=False)[:trim_n].reshape(n_candles, period)
    volumes = df["volume"].to_numpy(dtype=np.float64, copy=False)[:trim_n].reshape(n_candles, period)

    agg_open = opens[:, 0]
    agg_high = highs.max(axis=1)
    agg_low = lows.min(axis=1)
    agg_close = closes[:, -1]
    agg_volume = volumes.sum(axis=1)

    if "open_time" in df.columns:
        ts = pd.to_datetime(df["open_time"].values[:trim_n])
        agg_ts = ts[::period]
    else:
        agg_ts = range(n_candles)

    return pd.DataFrame({
        "open_time": agg_ts,
        "open": agg_open, "high": agg_high,
        "low": agg_low, "close": agg_close,
        "volume": agg_volume,
    })


def run_training_pipeline(config_path: str | None = None, no_cache: bool = False):
    """Full training pipeline โ€” aggregate 15m โ’ features โ’ train โ’ backtest.

    *** KEY FIXES ***
    1. Aggregate 1m โ’ 15m เธเนเธญเธเธ—เธธเธเธญเธขเนเธฒเธ (216K candles เนเธ—เธ 3.2M)
    2. Use compact returns + TA + micro features (25 dims instead of the old 346-dim raw window)
    3. Short episodes (2000 candles) + random start
    4. PPO ent_coef=0.02 เธเนเธญเธเธเธฑเธ entropy collapse
    """
    from monitoring.display import Dashboard

    config = load_config(config_path)
    data_config = config["data"]
    trading_config = config.get("trading", {})
    validation_config = config.get("training", {}).get("validation", {})
    pipeline_start = time.time()

    # GPU info
    gpu_name, gpu_vram, cuda_ver = "", 0, ""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            cuda_ver = torch.version.cuda or ""
    except Exception:
        pass

    dash = Dashboard()
    dash.update(gpu=gpu_name, gpu_vram=gpu_vram, cuda=cuda_ver)
    config["_dashboard"] = dash

    pairs = data_config["pairs"]
    raw_dir = Path(data_config["paths"]["raw"])

    use_cache = not no_cache
    from data.cache import FEATURE_SCHEMA_VERSION, load_features, save_features

    for symbol in pairs:
        config["_active_symbol"] = symbol
        ohlcv_path = raw_dir / f"{symbol}_1m.parquet"
        if not ohlcv_path.exists():
            dash.update(status_msg=f"Downloading {symbol} data...")
            logger.info(f"Data not found. Downloading {symbol}...")
            from data.downloaders.binance_historical import download_range
            from datetime import date
            download_range(symbol, "1m", date(2020, 1, 1), output_dir=str(raw_dir))
            
            if not ohlcv_path.exists():
                dash.add_phase(f"Data {symbol}", "fail")
                continue

        dash.update(symbol=symbol, status_msg=f"Loading {symbol}...")

        # --- Try loading from cache first ---
        source_paths = [ohlcv_path]
        # Also invalidate if funding/sentiment data changed
        for extra in [raw_dir / f"{symbol}_funding_rate.parquet",
                      raw_dir / "fear_greed_index.parquet"]:
            if extra.exists():
                source_paths.append(extra)

        cached = load_features(symbol, source_paths) if use_cache else None

        # Validate cache: reject if too small (stale from test run with subsample)
        if cached is not None:
            try:
                cached_schema_version = int(np.asarray(cached.get("schema_version", [0])).reshape(-1)[0])
                if cached_schema_version != FEATURE_SCHEMA_VERSION:
                    logger.warning(
                        "Cache invalidated: schema version %d != expected %d. Rebuilding.",
                        cached_schema_version,
                        FEATURE_SCHEMA_VERSION,
                    )
                    cached = None
            except Exception:
                cached = None

        if cached is not None:
            try:
                import pyarrow.parquet as pq
                expected_1m_rows = pq.read_metadata(ohlcv_path).num_rows
                expected_15m_rows = expected_1m_rows // 15
                cached_rows = len(cached["prices"])
                if cached_rows < expected_15m_rows * 0.5:
                    logger.warning(
                        "Cache invalidated: %d rows vs expected ~%d (stale from test run?). Rebuilding.",
                        cached_rows, expected_15m_rows,
                    )
                    cached = None
            except Exception:
                pass

        if cached is not None:
            feature_array = cached["features"]
            prices = cached["prices"]
            ohlcv_data = cached["ohlcv"]
            close_15m = cached["close_15m"]
            config["_nautilus_frame"] = _build_nautilus_frame(prices, ohlcv_data)

            n_feats = feature_array.shape[1]
            dash.update(data_1m=0, data_15m=len(prices))
            dash.add_phase("Loaded from cache", "ok", 0)
            dash.update(features=n_feats)
            logger.info(f"Using cached features: {n_feats} dims, {len(prices):,} rows")

            if config.get("training", {}).get("nautilus_validation", {}).get("enabled", False):
                t0 = time.time()
                dash.update(status_msg="Preparing cached Nautilus validation frame...")
                df_nav, _, _ = load_and_clean_data(config, symbol)
                df_nav_15m = _aggregate_ohlcv_15m(df_nav, period=15)
                if len(df_nav_15m) == len(prices):
                    config["_nautilus_frame"] = df_nav_15m[
                        ["open_time", "open", "high", "low", "close", "volume"]
                    ].reset_index(drop=True)
                else:
                    logger.warning(
                        "Cached Nautilus frame length mismatch for %s: raw_15m=%d cached=%d. Falling back to reconstructed frame.",
                        symbol,
                        len(df_nav_15m),
                        len(prices),
                    )
                dash.add_phase("Nautilus Validation Frame", "ok", time.time() - t0)
        else:
            # Step 1: Load & clean 1m data
            t0 = time.time()
            df, funding_df, sentiment_df = load_and_clean_data(config, symbol)
            dash.add_phase("Data Loading & Cleaning", "ok", time.time() - t0)
            dash.update(data_1m=len(df))

            # Step 2: Aggregate to 15m
            t0 = time.time()
            dash.update(status_msg="Aggregating 1m -> 15m...")
            df_15m = _aggregate_ohlcv_15m(df, period=15)
            dash.add_phase("15m OHLCV Aggregation", "ok", time.time() - t0)
            dash.update(data_15m=len(df_15m))

            # Step 3: Compute compact 15m features (returns + TA + micro = 25 dims)
            from features.technical.indicators import compute_all as compute_ta
            from features.technical.microstructure import compute_all as compute_micro

            ta_15m = compute_ta(df_15m).astype(np.float32)      # (n, 15)
            micro_15m = compute_micro(df_15m).astype(np.float32)  # (n, 5)

            # Returns on 15m
            close_15m = df_15m["close"].values.astype(np.float64)
            log_ret = np.log(close_15m[1:] / close_15m[:-1])
            returns_1 = np.concatenate([[0], log_ret])
            returns_4 = pd.Series(returns_1).rolling(4).sum().fillna(0).values
            returns_16 = pd.Series(returns_1).rolling(16).sum().fillna(0).values
            returns_48 = pd.Series(returns_1).rolling(48).sum().fillna(0).values
            returns_96 = pd.Series(returns_1).rolling(96).sum().fillna(0).values
            ret_features = np.column_stack(
                [returns_1, returns_4, returns_16, returns_48, returns_96]
            ).astype(np.float32)

            # CryptoMamba forecast features are optional and disabled by default.
            # This implementation uses a sequential scan, so it can be expensive
            # on Windows/consumer GPUs unless kept on a small budget.
            n_15m = len(close_15m)
            forecast_feats = np.zeros((n_15m, 5), dtype=np.float32)
            mamba_quality_ok = False
            mamba_phase_label = "CryptoMamba Forecast (DISABLED)"
            mamba_phase_status = "warn"
            mamba_cfg = (
                config.get("training", {})
                .get("forecast_features", {})
                .get("crypto_mamba", {})
            )
            mamba_enabled = bool(mamba_cfg.get("enabled", False))
            mamba = None

            if mamba_enabled:
                dash.update(status_msg="Evaluating CryptoMamba...")
                from models.forecast.crypto_mamba import CryptoMambaForecaster

                ctx_len = int(mamba_cfg.get("context_len", 96))
                pred_horizon = max(1, min(int(mamba_cfg.get("horizon", 4)), 4))
                sample_step = max(1, int(mamba_cfg.get("sample_step", 15)))
                mamba = CryptoMambaForecaster(
                    context_len=ctx_len,
                    horizon=pred_horizon,
                    d_model=int(mamba_cfg.get("d_model", 32)),
                    n_layers=int(mamba_cfg.get("n_layers", 2)),
                    device=str(mamba_cfg.get("device", "cuda" if gpu_name else "cpu")),
                )
                mamba_phase_label = "CryptoMamba Forecast (UNAVAILABLE)"

                if mamba.available:
                    logger.info(
                        "Fine-tuning CryptoMamba on 15m data with conservative budget: "
                        "ctx=%d horizon=%d stride=%d epochs=%d batch=%d max_samples=%d patience=%d",
                        ctx_len,
                        pred_horizon,
                        int(mamba_cfg.get("window_stride", 4)),
                        int(mamba_cfg.get("epochs", 6)),
                        int(mamba_cfg.get("batch_size", 128)),
                        int(mamba_cfg.get("max_samples", 12_000)),
                        int(mamba_cfg.get("patience", 2)),
                    )
                    ft_result = mamba.fine_tune(
                        close_15m,
                        epochs=int(mamba_cfg.get("epochs", 6)),
                        lr=float(mamba_cfg.get("lr", 1e-3)),
                        batch_size=int(mamba_cfg.get("batch_size", 128)),
                        max_samples=int(mamba_cfg.get("max_samples", 12_000)),
                        patience=int(mamba_cfg.get("patience", 2)),
                        min_improvement=float(mamba_cfg.get("min_improvement", 0.002)),
                        window_stride=int(mamba_cfg.get("window_stride", 4)),
                        eval_batch_size=int(mamba_cfg.get("eval_batch_size", 1024)),
                        use_amp=bool(mamba_cfg.get("use_amp", True)),
                        save_path=str(mamba_cfg.get("save_path", "checkpoints/crypto_mamba_15m.pt")),
                    )

                    mamba_quality_ok = bool(ft_result.get("better_than_naive", False))
                    dir_acc = float(ft_result.get("directional_accuracy", 0.0))
                    rmse_ratio = float(ft_result.get("rmse_ratio_vs_naive", 999.0))

                    if mamba_quality_ok:
                        logger.info(
                            "CryptoMamba PASSED quality gate: dir_acc=%.1f%% rmse_ratio=%.3f - using forecast features",
                            dir_acc * 100,
                            rmse_ratio,
                        )
                        indices = list(range(ctx_len, n_15m, sample_step))
                        n_preds = len(indices)
                        logger.info("Generating CryptoMamba forecasts (%d predictions, batched)...", n_preds)
                        dash.update(status_msg=f"CryptoMamba batch predict ({n_preds})...")

                        windows = np.array([close_15m[i - ctx_len:i] for i in indices], dtype=np.float32)
                        try:
                            preds, uncs = mamba.predict_batch(
                                windows,
                                horizon=pred_horizon,
                                batch_size=int(mamba_cfg.get("predict_batch_size", 1024)),
                            )
                            for k, i in enumerate(indices):
                                last_p = close_15m[i - 1]
                                if last_p > 0:
                                    fc = (preds[k, :pred_horizon] / last_p - 1.0).astype(np.float32)
                                    end = min(i + sample_step, n_15m)
                                    forecast_feats[i:end, :pred_horizon] = fc
                                    forecast_feats[i:end, 4] = float(uncs[k])
                            mamba_phase_label = "CryptoMamba Forecast (GOOD)"
                            mamba_phase_status = "ok"
                        except Exception as e:
                            logger.warning("CryptoMamba batch predict failed: %s", e)
                            forecast_feats[:] = 0.0
                            mamba_quality_ok = False
                            mamba_phase_label = "CryptoMamba Forecast (PREDICT FAILED)"
                    else:
                        logger.warning(
                            "CryptoMamba FAILED quality gate: dir_acc=%.1f%% rmse_ratio=%.3f - forecast features zeroed out",
                            dir_acc * 100,
                            rmse_ratio,
                        )
                        mamba_phase_label = "CryptoMamba Forecast (ZEROED)"
                else:
                    logger.warning("CryptoMamba unavailable on this environment - forecast features zeroed out.")
            else:
                logger.info("CryptoMamba disabled by config - forecast features zeroed out.")

            if mamba is not None:
                mamba.release_gpu()
            dash.add_phase(mamba_phase_label, mamba_phase_status)

            # Combine a compact feature state; include forecast block only when it passed the quality gate.
            feature_blocks = [ret_features, ta_15m, micro_15m]
            if mamba_quality_ok:
                feature_blocks.append(forecast_feats)
            feature_array = np.concatenate(feature_blocks, axis=1)
            feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)
            prices = close_15m.astype(np.float32)
            ohlcv_data = df_15m[["open", "high", "low", "close"]].values.astype(np.float32)
            config["_nautilus_frame"] = df_15m[["open_time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)

            n_feats = feature_array.shape[1]
            dash.add_phase(f"Features + CryptoMamba ({n_feats} dims)", "ok", time.time() - t0)
            dash.update(features=n_feats)

            # Save to cache for next run unless this execution explicitly bypassed cache usage.
            if use_cache:
                save_features(symbol, feature_array, prices, ohlcv_data, close_15m)
            else:
                logger.info("Skipping feature cache write because --no-cache was requested.")

        # Step 4: Baseline
        bh_return = float(prices[-1] / prices[0] - 1)
        ranges = _compute_data_ranges(
            len(prices),
            test_ratio=validation_config.get("holdout_test_ratio", 0.20),
            validation_ratio_within_train=validation_config.get("validation_ratio_within_train", 0.10),
        )
        config["_data_ranges"] = ranges
        logger.info(
            "Dataset split for %s: train=[%d:%d) validation=[%d:%d) test=[%d:%d)",
            symbol,
            ranges["train"][0],
            ranges["train"][1],
            ranges["validation"][0],
            ranges["validation"][1],
            ranges["test"][0],
            ranges["test"][1],
        )
        dash.update(bh_return=bh_return, bh_full_return=bh_return)

        # Step 5: Model training / selection
        training_config = config.get("training", {})
        rl_config = training_config.get("rl", {})
        total_steps = rl_config.get("total_timesteps", 200000)

        dash.update(train_total=total_steps, status_msg="Initializing training...")

        t0 = time.time()
        dash.update(status_msg="Training and model selection in progress...")
        config["_ohlcv_data"] = ohlcv_data
        _, metrics = train_rl_agent(
            feature_array=feature_array,
            prices=prices,
            config=config,
            dashboard=dash,
        )
        dash.add_phase(f"Model Training ({total_steps//1000}K steps)", "ok", time.time() - t0)
        dash.add_phase("Multi-Episode Eval (10 eps)", "ok")

        # Update dashboard with results
        dash.update(
            bh_return=metrics.get("bh_eval_return", bh_return),
            bh_full_return=bh_return,
            rl_return=metrics.get("total_return", 0),
            gross_return=metrics.get("gross_total_return", metrics.get("total_return", 0)),
            server_cost_paid=metrics.get("server_cost_paid", 0),
            total_server_cost_paid=metrics.get("total_server_cost_paid", metrics.get("server_cost_paid", 0)),
            avg_trades_per_episode=metrics.get("avg_trades_per_episode", 0),
            eval_episodes=metrics.get("eval_episodes", 0),
            flat_ratio=metrics.get("flat_ratio", 0),
            position_ratio=metrics.get("position_ratio", 0),
            alpha_vs_bh=metrics.get("outperformance_vs_bh", 0),
            avg_reward_sum=metrics.get("avg_reward_sum", 0),
            wrong_side_moves=metrics.get("wrong_side_moves", 0),
            eval_short_actions=metrics.get("eval_short_actions", 0),
            eval_flat_actions=metrics.get("eval_flat_actions", 0),
            eval_long_actions=metrics.get("eval_long_actions", 0),
            eval_action_entropy=metrics.get("eval_action_entropy", 0),
            eval_dominant_action_ratio=metrics.get("eval_dominant_action_ratio", 1),
            selection_gate_passed=metrics.get("selection_gate_passed", 1),
            selection_best_score=metrics.get("selection_best_score", 0),
            sharpe=metrics.get("sharpe", 0),
            sortino=metrics.get("sortino", 0),
            max_dd=metrics.get("max_drawdown", 0),
            n_trades=metrics.get("n_trades", 0),
            n_longs=metrics.get("n_longs", 0),
            n_shorts=metrics.get("n_shorts", 0),
            n_wins=metrics.get("n_wins", 0),
            n_losses=metrics.get("n_losses", 0),
            win_rate=metrics.get("win_rate", 0),
            model_path=metrics.get("model_path", "checkpoints/rl_agent_final.zip"),
            chart_path="checkpoints/training_results.png",
            status_msg=f"COMPLETE ({metrics.get('model_family', 'ppo')})",
        )

        # Log to file
        logger.info(f"RL metrics: {metrics}")
        logger.info(f"Buy & Hold full-history return: {bh_return:.4f}")
        logger.info(f"Buy & Hold eval-window return: {metrics.get('bh_eval_return', bh_return):.4f}")

    time.sleep(2)  # show final results
    dash.stop()
    logger.info("=== Training pipeline complete ===")


# =============================================================================
# Report
# =============================================================================

def _print_report(report: dict, start_time: float):
    """Print final test report."""
    total_time = time.time() - start_time

    logger.info("\n" + "=" * 70)
    logger.info("GARIC โ€” System Test Report")
    logger.info("=" * 70)

    n_ok = 0
    n_fail = 0

    for phase_name, phase_data in report.get("phases", {}).items():
        status = phase_data.get("status", "UNKNOWN")
        time_sec = phase_data.get("time_sec", 0)
        status_icon = "PASS" if status == "OK" else "FAIL"

        if status == "OK":
            n_ok += 1
        else:
            n_fail += 1

        logger.info(f"  [{status_icon}] {phase_name} ({time_sec}s)")

        # Print key metrics
        if "metrics" in phase_data and isinstance(phase_data["metrics"], dict):
            for k, v in phase_data["metrics"].items():
                if isinstance(v, float):
                    logger.info(f"         {k}: {v:.4f}")

    logger.info("-" * 70)
    logger.info(f"  Total: {n_ok} passed, {n_fail} failed")
    logger.info(f"  Time:  {total_time:.1f} seconds ({total_time / 60:.1f} min)")
    logger.info(f"  GPU:   {'Yes' if report.get('gpu_available') else 'No (CPU mode)'}")

    if report["errors"]:
        logger.info("\n  Errors:")
        for err in report["errors"]:
            logger.info(f"    - {err}")

    logger.info("=" * 70)

    if n_fail == 0:
        logger.info("ALL PHASES PASSED - System is ready")
        logger.info("Next: python pipeline.py --mode train (full data, cloud GPU)")
    else:
        logger.info(f"{n_fail} PHASES FAILED โ€” เธ•เนเธญเธเนเธเนเนเธเธเนเธญเธ deploy")


# =============================================================================
# Entry point
# =============================================================================

def main():
    import argparse
    import os
    import warnings

    parser = argparse.ArgumentParser(description="GARIC train/test pipeline")
    parser.add_argument("--config", default=None)
    parser.add_argument("--mode", choices=["train", "test"], default="test")
    parser.add_argument("--no-cache", action="store_true", help="Force recompute features (ignore cache)")
    args = parser.parse_args()

    run_logs_dir = Path("logs")
    run_logs_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = run_logs_dir / f"pipeline_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("pipeline.log", mode="w", encoding="utf-8"),
            logging.FileHandler(run_log_path, mode="w", encoding="utf-8"),
        ],
        force=True,
    )
    os.environ["GARIC_RUN_TAG"] = run_log_path.stem.replace("pipeline_", "", 1)
    logger.info("Run log archive: %s", run_log_path.as_posix())
    if args.mode == "train":
        warnings.filterwarnings("ignore")  # suppress all warnings in train mode

    if args.mode == "test":
        run_test_pipeline(args.config)
    elif args.mode == "train":
        run_training_pipeline(args.config, no_cache=args.no_cache)


if __name__ == "__main__":
    main()
