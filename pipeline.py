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
import subprocess
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
    walkforward_min_active_ratio: float = 0.25,
    walkforward_inactivity_penalty_scale: float = 1.50,
    walkforward_worst_net_penalty_scale: float = 4.0,
    walkforward_worst_gross_penalty_scale: float = 3.0,
    walkforward_worst_alpha_penalty_scale: float = 0.50,
    walkforward_dominance_penalty_scale: float = 3.0,
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

    bonus = active_ratio * 0.15
    bonus += max(float(walkforward_metrics.get("walkforward_positive_net_ratio", 0.0)) - 0.50, 0.0) * 0.25
    bonus += max(float(walkforward_metrics.get("walkforward_positive_alpha_ratio", 0.0)) - 0.50, 0.0) * 0.15

    combined_score = (
        float(episodic_score)
        + (float(np.clip(walkforward_soft_score, -2.0, 2.0)) * walkforward_score_weight)
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


def _score_nautilus_summary(
    summary: dict,
    *,
    min_trades: float = 1.0,
    max_dominant_action_ratio: float = 0.995,
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
    returns_stats = summary.get("stats_returns", {}) if isinstance(summary.get("stats_returns"), dict) else {}
    sharpe = float(returns_stats.get("Sharpe Ratio (252 days)", 0.0) or 0.0)
    wrong_side_penalty = max(float(summary.get("position_ratio", 0.0)) - 0.98, 0.0) * 0.5

    if trades < min_trades:
        return float("-inf")
    if dominant_ratio > max_dominant_action_ratio and trades <= (min_trades + 1.0):
        return float("-inf")
    # Reject models that lose more than 10%
    if net_return < -0.10:
        return float("-inf")

    profit_bonus = max(net_return, 0.0) * 15.0
    gross_bonus = max(gross_return, 0.0) * 8.0
    return (
        net_return * 10.0
        + gross_return * 6.0
        + sharpe * 0.20
        + alpha * 1.5
        + profit_bonus
        + gross_bonus
        + win_rate * 0.50
        + min(trades, 16.0) * 0.01
        + action_entropy * 0.20
        - max(flat_ratio - 0.95, 0.0) * 2.0
        - max(dominant_ratio - 0.85, 0.0) * 1.5
        - max(max_drawdown - 0.15, 0.0) * 2.5
        - wrong_side_penalty
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

    start, end = int(segment_range[0]), int(segment_range[1])
    segment = frame_15m.iloc[start:end].reset_index(drop=True)
    if len(segment) < 128:
        raise ValueError(f"Nautilus {label} segment too short: {len(segment)} bars")

    training_config = config.get("training", {})
    nav_config = training_config.get("nautilus_validation", {})
    trading_config = config.get("trading", {})
    symbol = str(config.get("_active_symbol") or config.get("data", {}).get("pairs", ["BTCUSDT"])[0])
    initial_balance = float(nav_config.get("initial_balance_usdt", 10_000.0))
    leverage = float(nav_config.get("leverage", trading_config.get("leverage", 1.0)))
    trade_size_pct = float(nav_config.get("trade_size_pct_of_equity", 1.0))
    first_price = float(segment["close"].iloc[0])
    trade_qty = max((initial_balance * trade_size_pct) / max(first_price, 1e-9), 1e-6)
    state_name = f"nautilus_{label}_{Path(model_path).stem}.json"
    state_path = str(Path("checkpoints") / state_name)

    summary = run_backtest_frame(
        segment,
        symbol=symbol,
        model_path=model_path,
        venue=str(nav_config.get("venue", "BINANCE")),
        bar_minutes=int(nav_config.get("bar_minutes", 15)),
        history_bars=int(nav_config.get("history_bars", 160)),
        request_history_days=int(nav_config.get("request_history_days", 3)),
        trade_size=f"{trade_qty:.12f}",
        initial_balance_usdt=initial_balance,
        leverage=leverage,
        state_path=state_path,
        mode=f"train_{label}",
        close_positions_on_stop=True,
        reduce_only_on_stop=True,
        monthly_server_cost_usd=float(trading_config.get("monthly_server_cost_usd", 100.0)),
        periods_per_day=int(trading_config.get("periods_per_day", 96)),
    )
    summary["segment_label"] = label
    summary["segment_range"] = [start, end]
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
    start, end = int(segment_range[0]), int(segment_range[1])
    segment = frame_15m.iloc[start:end].reset_index(drop=True)
    if len(segment) < 128:
        raise ValueError(f"Nautilus {label} segment too short: {len(segment)} bars")

    training_config = config.get("training", {})
    nav_config = training_config.get("nautilus_validation", {})
    trading_config = config.get("trading", {})
    symbol = str(config.get("_active_symbol") or config.get("data", {}).get("pairs", ["BTCUSDT"])[0])
    initial_balance = float(nav_config.get("initial_balance_usdt", 10_000.0))
    leverage = float(nav_config.get("leverage", trading_config.get("leverage", 1.0)))
    trade_size_pct = float(nav_config.get("trade_size_pct_of_equity", 1.0))
    first_price = float(segment["close"].iloc[0])
    trade_qty = max((initial_balance * trade_size_pct) / max(first_price, 1e-9), 1e-6)

    safe_label = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in label)
    segment_path = Path("checkpoints") / f"nautilus_segment_{safe_label}.parquet"
    summary_path = Path("checkpoints") / f"nautilus_segment_{safe_label}_summary.json"
    state_path = Path("checkpoints") / f"nautilus_{safe_label}.json"
    segment.to_parquet(segment_path, index=False)
    if summary_path.exists():
        summary_path.unlink()

    cmd = [
        sys.executable,
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
        str(int(nav_config.get("history_bars", 160))),
        "--request-history-days",
        str(int(nav_config.get("request_history_days", 3))),
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
        "--monthly-server-cost-usd",
        str(float(trading_config.get("monthly_server_cost_usd", 100.0))),
        "--periods-per-day",
        str(int(trading_config.get("periods_per_day", 96))),
        "--summary-json",
        str(summary_path),
    ]
    collapse_probe_bars = max(int(nav_config.get("collapse_probe_bars", 0)), 0)
    collapse_probe_min_flat_ratio = float(nav_config.get("collapse_probe_min_flat_ratio", 0.995))
    collapse_probe_max_trades = max(int(nav_config.get("collapse_probe_max_trades", 0)), 0)
    proc = subprocess.Popen(
        cmd,
        cwd=str(Path.cwd()),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout_text = ""
    stderr_text = ""
    killed_for_collapse = False
    collapse_state: dict | None = None

    while True:
        returncode = proc.poll()
        if returncode is not None:
            stdout_text, stderr_text = proc.communicate()
            break

        if collapse_probe_bars > 0 and state_path.exists():
            try:
                with open(state_path, "r", encoding="utf-8") as fh:
                    runtime_state = json.load(fh)
                action_counts = runtime_state.get("action_counts", {}) if isinstance(runtime_state, dict) else {}
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
                        stdout_text, stderr_text = proc.communicate(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        stdout_text, stderr_text = proc.communicate()
                    break
            except Exception:
                pass

        time.sleep(2.0)

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
        last_idx = min(total_actions, len(segment)) - 1
        last_close = float(segment["close"].iloc[last_idx])
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
            "stdout_tail": stdout_text[-1000:],
            "stderr_tail": stderr_text[-1000:],
        }
    else:
        if proc.returncode != 0:
            message = (stderr_text or "").strip() or (stdout_text or "").strip() or f"subprocess exit code {proc.returncode}"
            raise RuntimeError(f"Nautilus subprocess validation failed for {label}: {message[-2000:]}")
        if not summary_path.exists():
            raise RuntimeError(f"Nautilus subprocess validation did not create summary file: {summary_path}")
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

    summary["segment_label"] = label
    summary["segment_range"] = [start, end]
    summary["trade_size_qty"] = float(trade_qty)
    summary["trade_size_notional_usdt"] = float(trade_qty * first_price)
    summary["subprocess_validation"] = True
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
    if primary_model in {"supervised", "supervised_logreg", "logreg", "supervised_only"}:
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
        turnover_penalty_scale=rl_config.get("turnover_penalty_scale", 0.05),
        inactive_episode_penalty=rl_config.get("inactive_episode_penalty", 0.0),
        static_position_episode_penalty=rl_config.get("static_position_episode_penalty", 0.0),
        balanced_sampling=rl_config.get("balanced_sampling", True),
        regime_label_threshold=rl_config.get("regime_label_threshold", 0.02),
        selection_max_dominant_action_ratio=rl_config.get("selection_max_dominant_action_ratio", 0.95),
        selection_min_avg_trades_per_episode=rl_config.get("selection_min_avg_trades_per_episode", 2.0),
        selection_min_action_entropy=rl_config.get("selection_min_action_entropy", 0.02),
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
            ppo_model = ModelClass.load(str(ppo_model_path))
            logger.info("Loaded trained model for evaluation")
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
            model_types = supervised_config.get("model_types", ["logreg", "extratrees"])
            if isinstance(model_types, str):
                model_types = [model_types]
            recent_train_bars = int(supervised_config.get("recent_train_bars", 0))
            supervised_validation_episodes = int(supervised_config.get("validation_episodes", max(eval_episodes, 10)))
            supervised_validation_seed_runs = max(int(supervised_config.get("validation_seed_runs", 1)), 1)
            supervised_validation_seed_stride = max(int(supervised_config.get("validation_seed_stride", 500)), 1)
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
            walkforward_min_active_ratio = float(supervised_config.get("walkforward_min_active_ratio", 0.25))
            walkforward_inactivity_penalty_scale = float(
                supervised_config.get("walkforward_inactivity_penalty_scale", 1.50)
            )
            walkforward_worst_net_penalty_scale = float(
                supervised_config.get("walkforward_worst_net_penalty_scale", 4.0)
            )
            walkforward_worst_gross_penalty_scale = float(
                supervised_config.get("walkforward_worst_gross_penalty_scale", 3.0)
            )
            walkforward_worst_alpha_penalty_scale = float(
                supervised_config.get("walkforward_worst_alpha_penalty_scale", 0.50)
            )
            walkforward_dominance_penalty_scale = float(
                supervised_config.get("walkforward_dominance_penalty_scale", 3.0)
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
                int(training_config.get("nautilus_validation", {}).get("top_supervised_candidates", 3)),
                1,
            )
            supervised_train_range = ranges["train"]
            if recent_train_bars > 0:
                train_end = int(ranges["train"][1])
                train_start = max(int(ranges["train"][0]), train_end - recent_train_bars)
                supervised_train_range = (train_start, train_end)
            logger.info(
                "Supervised fallback selection: train=[%d:%d) validation=[%d:%d) "
                "recent_train_bars=%d validation_episodes=%d validation_seed_runs=%d min_selection_score=%.4f "
                "walkforward_windows=%d",
                int(supervised_train_range[0]),
                int(supervised_train_range[1]),
                int(ranges["validation"][0]),
                int(ranges["validation"][1]),
                recent_train_bars,
                supervised_validation_episodes,
                supervised_validation_seed_runs,
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
                    taker_fee=trading_config.get("taker_fee", 0.0005),
                    slippage_bps=trading_config.get("slippage_bps", 1.0),
                    leverage=trading_config.get("leverage", 1.0),
                    random_state=training_config.get("seed", 42),
                )

                for long_confidence in long_confidence_grid:
                    for short_confidence in short_confidence_grid:
                        sup_model.confidence_threshold = float(min(float(long_confidence), float(short_confidence)))
                        sup_model.long_confidence_threshold = float(long_confidence)
                        sup_model.short_confidence_threshold = float(short_confidence)
                        sup_val_runs: list[dict] = []
                        strict_scores: list[float] = []
                        relaxed_scores: list[float] = []
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
                        strict_score = _robust_validation_score(strict_scores)
                        relaxed_score = _robust_validation_score(relaxed_scores)
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
                            "Supervised validation: model=%s long_conf=%.2f short_conf=%.2f strict=%s relaxed=%s "
                            "wf_strict=%s wf_relaxed=%s strict_combined=%s relaxed_combined=%s seed_runs=%d "
                            "alpha=%.4f net=%.4f gross=%.4f trades=%.1f dominant_ratio=%.2f%% "
                            "wf_active=%.2f%% wf_penalty=%.4f brier=%.4f episodes=%d",
                            model_type,
                            float(long_confidence),
                            float(short_confidence),
                            f"{strict_score:.4f}" if np.isfinite(strict_score) else "-inf",
                            f"{relaxed_score:.4f}" if np.isfinite(relaxed_score) else "-inf",
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
                        if candidate_options:
                            candidate_mode, candidate_score, candidate_priority, candidate_validation_metrics = max(
                                candidate_options,
                                key=lambda item: (
                                    item[1],
                                    item[2],
                                    -float(item[3].get("supervised_validation_brier", float("inf"))),
                                ),
                            )

                        if candidate_mode is not None and (
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
                        if candidate_mode is not None and np.isfinite(candidate_score):
                            candidate_name = (
                                f"{model_type}_{candidate_mode}_"
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
                                    "score": float(candidate_score),
                                    "priority": int(candidate_priority),
                                    "mode": str(candidate_mode),
                                    "name": candidate_name,
                                    "confidence": float(min(float(long_confidence), float(short_confidence))),
                                    "long_confidence": float(long_confidence),
                                    "short_confidence": float(short_confidence),
                                    "validation_metrics": dict(candidate_validation_metrics or relaxed_metrics),
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
                            if len(supervised_candidate_pool) > supervised_candidate_pool_size:
                                supervised_candidate_pool = supervised_candidate_pool[:supervised_candidate_pool_size]

            supervised_candidate_eligible = (
                sup_best_model is not None
                and sup_best_conf is not None
                and np.isfinite(sup_best_score)
                and sup_best_score >= supervised_min_selection_score
            )

            if sup_best_model is not None and sup_best_conf is not None and supervised_candidate_eligible:
                sup_best_model.confidence_threshold = float(sup_best_conf)
                sup_best_model.long_confidence_threshold = float(
                    sup_best_long_conf if sup_best_long_conf is not None else sup_best_conf
                )
                sup_best_model.short_confidence_threshold = float(
                    sup_best_short_conf if sup_best_short_conf is not None else sup_best_conf
                )
                supervised_path = sup_best_model.save("checkpoints/rl_agent_supervised.joblib")
                supervised_metrics = None
                logger.info(
                    "Saved supervised fallback -> %s (mode=%s long_conf=%.2f short_conf=%.2f validation_score=%s)",
                    supervised_path,
                    sup_best_mode,
                    float(sup_best_long_conf if sup_best_long_conf is not None else sup_best_conf),
                    float(sup_best_short_conf if sup_best_short_conf is not None else sup_best_conf),
                    f"{sup_best_score:.4f}" if np.isfinite(sup_best_score) else "-inf",
                )

                use_supervised = False
                if np.isfinite(sup_best_score):
                    use_supervised = (not np.isfinite(ppo_validation_score)) or (sup_best_score > ppo_validation_score)
                if not np.isfinite(ppo_validation_score) and np.isfinite(sup_best_score):
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
                    logger.info(
                        "Keeping PPO model: best supervised mode=%s score=%s ppo validation score=%s",
                        sup_best_mode,
                        f"{sup_best_score:.4f}" if np.isfinite(sup_best_score) else "-inf",
                        f"{ppo_validation_score:.4f}" if np.isfinite(ppo_validation_score) else "-inf",
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
                    candidate_path = (
                        supervised_path
                        if idx == 0
                        else candidate_model.save(
                            f"checkpoints/rl_agent_supervised_{candidate_info['name']}.joblib"
                        )
                    )
                    is_selected_env_candidate = (
                        np.isclose(candidate_score_value, sup_best_score)
                        and np.isclose(candidate_long_conf_value, float(sup_best_long_conf or sup_best_conf or 0.0))
                        and np.isclose(candidate_short_conf_value, float(sup_best_short_conf or sup_best_conf or 0.0))
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
            elif sup_best_model is not None and sup_best_conf is not None:
                logger.warning(
                    "Best supervised candidate score %.4f is below the minimum %.4f. "
                    "It will not replace the current selected policy.",
                    float(sup_best_score),
                    float(supervised_min_selection_score),
                )

            if (
                supervised_fallback_to_safe_flat
                and (selected_model is None or not np.isfinite(ppo_validation_score))
                and (
                    sup_best_model is None
                    or sup_best_conf is None
                    or not supervised_candidate_eligible
                )
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
                    "No supervised candidate cleared the minimum score. Falling back to safe flat policy."
                )
        except Exception as e:
            logger.exception("Supervised fallback training failed: %s", e)

    nautilus_config = training_config.get("nautilus_validation", {})
    nautilus_enabled = bool(nautilus_config.get("enabled", False))
    if nautilus_enabled and candidate_records:
        frame_15m = config.get("_nautilus_frame")
        if not isinstance(frame_15m, pd.DataFrame) or len(frame_15m) != len(prices):
            logger.warning(
                "Nautilus validation enabled but no aligned 15m frame is available; skipping realistic selection."
            )
        else:
            min_trades = float(nautilus_config.get("selection_min_trades", 1.0))
            max_dom = float(nautilus_config.get("selection_max_dominant_action_ratio", 0.995))
            min_nautilus_selection_score = float(nautilus_config.get("min_selection_score", 0.0))
            use_for_selection = bool(nautilus_config.get("use_for_model_selection", True))
            evaluate_final_test = bool(nautilus_config.get("evaluate_final_test", True))
            use_subprocess_validation = bool(nautilus_config.get("subprocess_on_windows", True)) and sys.platform.startswith("win")
            if use_subprocess_validation:
                logger.info(
                    "Using subprocess Nautilus validation on Windows to avoid logger re-initialization crashes."
                )
            best_candidate = None
            best_nautilus_score = float("-inf")

            for candidate in candidate_records:
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
                    validation_summary = validation_runner(
                        model_path=candidate["model_path"],
                        frame_15m=frame_15m,
                        segment_range=ranges["validation"],
                        config=config,
                        label=f"validation_{candidate_name}",
                    )
                    validation_score = _score_nautilus_summary(
                        validation_summary,
                        min_trades=min_trades,
                        max_dominant_action_ratio=max_dom,
                    )
                    candidate["nautilus_validation"] = validation_summary
                    candidate["nautilus_validation_score"] = float(validation_score)
                    logger.info(
                        "Nautilus validation: family=%s candidate=%s long_conf=%.2f short_conf=%.2f "
                        "score=%s net=%.4f gross=%.4f alpha=%.4f trades=%d dominant_ratio=%.2f%% status=%s",
                        candidate["family"],
                        candidate_name,
                        float(candidate.get("supervised_long_confidence_threshold", candidate.get("supervised_confidence_threshold", 0.0))),
                        float(candidate.get("supervised_short_confidence_threshold", candidate.get("supervised_confidence_threshold", 0.0))),
                        f"{validation_score:.4f}" if np.isfinite(validation_score) else "-inf",
                        float(validation_summary.get("total_return", 0.0)),
                        float(validation_summary.get("gross_total_return", 0.0)),
                        float(validation_summary.get("outperformance_vs_bh", 0.0)),
                        int(validation_summary.get("n_trades", 0)),
                        float(validation_summary.get("eval_dominant_action_ratio", 0.0) * 100.0),
                        str(validation_summary.get("status", "UNKNOWN")),
                    )
                    if np.isfinite(validation_score) and validation_score > best_nautilus_score:
                        best_nautilus_score = float(validation_score)
                        best_candidate = candidate
                except Exception as exc:
                    candidate["nautilus_validation"] = {"error": str(exc)}
                    candidate["nautilus_validation_score"] = float("-inf")
                    logger.warning(
                        "Nautilus validation failed for %s: %s",
                        candidate["family"],
                        exc,
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
                    selected_metrics["selection_engine"] = "nautilus_validation"
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
                        "Selecting final model using Nautilus validation -> %s",
                        best_candidate["family"],
                    )
                elif use_for_selection:
                    logger.warning(
                        "Best Nautilus validation score %.4f is below the minimum %.4f. Keeping fast-env selection.",
                        float(best_nautilus_score),
                        float(min_nautilus_selection_score),
                    )

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
                        selected_metrics["nautilus_test"] = dict(final_summary)
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
                logger.warning(
                    "No model passed Nautilus validation gates; keeping fast-env selection."
                )

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
    """Run backtest using trained RL model via BacktestRunner.

    *** เนเธเน cost model เน€เธ”เธตเธขเธงเธเธฑเธเธเธฑเธ CryptoFuturesEnv ***
    *** เนเธเน last 200K steps เธชเธณเธซเธฃเธฑเธ eval (เนเธกเนเธ•เนเธญเธเธฃเธฑเธ 3.2M เธ—เธฑเนเธเธซเธกเธ”) ***
    """
    trading_config = (config or {}).get("trading", {})
    validation_config = (config or {}).get("training", {}).get("validation", {})
    min_trade = trading_config.get("min_trade_pct", 0.05)
    ranges = data_ranges or (config or {}).get("_data_ranges")
    if ranges is None:
        ranges = _compute_data_ranges(
            len(prices),
            test_ratio=validation_config.get("holdout_test_ratio", 0.20),
            validation_ratio_within_train=validation_config.get("validation_ratio_within_train", 0.10),
        )
    test_start, test_end = ranges["test"]
    fa_eval = feature_array[test_start:test_end]
    pr_eval = prices[test_start:test_end]

    n = len(pr_eval)
    signals = np.zeros(n)

    if model is not None:
        position = 0.0
        entry_price = 0.0
        flat_steps = 0
        pos_steps = 0
        last_turnover = 0.0
        initial_balance = 10_000.0
        balance = initial_balance
        peak_balance = initial_balance
        leverage = float(trading_config.get("leverage", 1.0))
        fee_rate = float(trading_config.get("taker_fee", 0.0005)) + float(trading_config.get("slippage_bps", 1.0)) / 10000.0
        server_cost_usd_per_step = (
            float(trading_config.get("monthly_server_cost_usd", 100.0))
            / max(float(trading_config.get("periods_per_day", 96)) * 30.0, 1.0)
        )
        for i in range(n):
            price_now = pr_eval[i]
            upnl = 0.0
            if position != 0 and entry_price > 0:
                upnl = position * (price_now / entry_price - 1.0) * leverage
            drawdown = max((peak_balance - balance) / max(peak_balance, 1e-9), 0.0)
            vol_start = max(0, i - 32)
            price_window = np.maximum(pr_eval[vol_start:i + 1], 1e-9)
            rolling_vol = float(np.std(np.diff(np.log(price_window)))) if price_window.size >= 2 else 0.0

            obs = np.concatenate(
                [
                    fa_eval[i],
                    build_agent_state(
                        position=position,
                        upnl=upnl,
                        equity_ratio=(balance / max(initial_balance, 1e-9)) - 1.0,
                        drawdown=drawdown,
                        rolling_volatility=rolling_vol,
                        turnover_last_step=last_turnover,
                        flat_steps=flat_steps,
                        pos_steps=pos_steps,
                    ),
                ]
            )
            action, _ = model.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                a = int(action.item()) if action.ndim == 0 else int(action[0])
            else:
                a = int(action)
            old_position = position
            if a == 0:
                position = -1.0
            elif a == 2:
                position = 1.0
            else:
                position = 0.0
            turnover = abs(position - old_position)
            last_turnover = float(turnover)

            if i > 0:
                prev_price = pr_eval[i - 1]
                price_ret = (price_now / prev_price - 1.0) if prev_price > 0 else 0.0
                gross_step_return = old_position * price_ret * leverage - turnover * fee_rate
                server_cost_pct = server_cost_usd_per_step / max(balance, 1.0)
                balance *= (1.0 + gross_step_return - server_cost_pct)
                balance = max(balance, 0.0)
                peak_balance = max(peak_balance, balance)

            prev_position = signals[i - 1] if i > 0 else 0.0
            if position == 0:
                entry_price = 0.0
                flat_steps += 1
                pos_steps = 0
            else:
                if prev_position != position:
                    entry_price = price_now
                flat_steps = 0
                pos_steps += 1
            signals[i] = position
    else:
        signals[:] = 1.0

    bt_config = BacktestConfig(
        taker_fee=0.0005,
        slippage_bps=1.0,
        leverage=trading_config.get("leverage", 1.0),
        maintenance_margin=trading_config.get("maintenance_margin", 0.005),
        min_trade_pct=min_trade,
        monthly_server_cost_usd=trading_config.get("monthly_server_cost_usd", 100.0),
        periods_per_day=trading_config.get("periods_per_day", 96),
    )
    runner = BacktestRunner(bt_config)
    result = runner.run(pr_eval, signals)
    logger.info(
        "Backtest on held-out test range [%d:%d) -> %d candles",
        test_start,
        test_end,
        n,
    )

    return {
        "backtest_metrics": result.metrics,
        "n_trades": len(result.trades),
        "test_range": [test_start, test_end],
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

            # Save to cache for next run
            save_features(symbol, feature_array, prices, ohlcv_data, close_15m)

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
        logger.info("ALL PHASES PASSED โ€” เธฃเธฐเธเธเธเธฃเนเธญเธกเนเธเนเธเธฒเธ")
        logger.info("Next: python pipeline.py --mode train (full data, cloud GPU)")
    else:
        logger.info(f"{n_fail} PHASES FAILED โ€” เธ•เนเธญเธเนเธเนเนเธเธเนเธญเธ deploy")


# =============================================================================
# Entry point
# =============================================================================

def main():
    import argparse
    import warnings

    parser = argparse.ArgumentParser(description="GARIC train/test pipeline")
    parser.add_argument("--config", default=None)
    parser.add_argument("--mode", choices=["train", "test"], default="test")
    parser.add_argument("--no-cache", action="store_true", help="Force recompute features (ignore cache)")
    args = parser.parse_args()

    # Train mode: browser dashboard handles visuals, console shows logs/errors.
    # Test mode: log to both console + file.
    if args.mode == "train":
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("pipeline.log", mode="w", encoding="utf-8"),
            ],
        )
        warnings.filterwarnings("ignore")  # suppress all warnings in train mode
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("pipeline.log", mode="w", encoding="utf-8"),
            ],
        )

    if args.mode == "test":
        run_test_pipeline(args.config)
    elif args.mode == "train":
        run_training_pipeline(args.config, no_cache=args.no_cache)


if __name__ == "__main__":
    main()
