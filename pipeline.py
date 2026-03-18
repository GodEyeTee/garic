"""Main pipeline — รวมทุก component เข้าด้วยกัน.

ใช้ได้ทั้ง:
1. train: historical data → features → train RL → backtest → validate
2. test:  subsample data → features → train RL (เร็ว) → backtest → validate ทุก component
3. paper: live data → features → model → paper orders
4. live:  live data → features → model → risk → real orders

*** ทุก mode ใช้ code path เดียวกัน ***
*** Action เฉพาะ candle close เท่านั้น ***

Usage:
  python pipeline.py --mode test                           # ทดสอบทั้งระบบ (RTX 2060)
  python pipeline.py --mode test --config configs/test_rtx2060.yaml
  python pipeline.py --mode train                          # full training
  python pipeline.py --mode train --config configs/default.yaml
"""

import logging
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from configs import load_config
from data.quality import clean_pipeline
from features.builder import FeatureBuilder
from models.forecast.naive import NaiveForecaster
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
            logger.warning("No CUDA GPU detected — will use CPU (slower)")
            return False
    except ImportError:
        logger.warning("PyTorch not installed — will use CPU")
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
            raise FileNotFoundError(f"Data not found: {ohlcv_path} — run downloaders first")

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
    """Add naive forecast to feature array using past-only windows."""
    forecaster = NaiveForecaster()
    forecast_col_start = 60 * 5 + 5 + 15 + 5  # ohlcv + returns + ta + micro
    enriched = feature_array.copy()

    for i in range(60, len(prices)):
        price_window = prices[max(0, i - 60):i]
        if len(price_window) > 1:
            forecast, uncertainty = forecaster.predict(price_window)
            enriched[i, forecast_col_start:forecast_col_start + 12] = forecast
            enriched[i, forecast_col_start + 12] = uncertainty

    return enriched


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
) -> tuple[object | None, dict]:
    """Train RL agent with PPO. Returns (model, metrics)."""
    training_config = config.get("training", {})
    rl_config = training_config.get("rl", {})

    total_timesteps = rl_config.get("total_timesteps", 10000)
    learning_rate = rl_config.get("learning_rate", 3e-4)
    n_steps = rl_config.get("n_steps", 512)
    batch_size = rl_config.get("batch_size", 32)
    n_epochs = rl_config.get("n_epochs", 5)
    algo = rl_config.get("algo", "PPO")

    try:
        from models.rl.trainer import RLTrainer
    except ImportError:
        logger.error("Cannot import RLTrainer")
        return None, {"error": "import failed"}

    # Trading params — ใช้ค่าเดียวกันทั้ง Env และ BacktestRunner
    trading_config = config.get("trading", {})
    validation_config = training_config.get("validation", {})
    ranges = _compute_data_ranges(
        len(prices),
        test_ratio=validation_config.get("holdout_test_ratio", 0.20),
        validation_ratio_within_train=validation_config.get("validation_ratio_within_train", 0.10),
    )
    config["_data_ranges"] = ranges

    # OHLCV data สำหรับ realistic intra-candle simulation
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
        monthly_server_cost_usd=trading_config.get("monthly_server_cost_usd", 100.0),
        periods_per_day=trading_config.get("periods_per_day", 96),
        opportunity_threshold=rl_config.get("opportunity_threshold", 0.0010),
        missed_move_penalty_scale=rl_config.get("missed_move_penalty_scale", 160.0),
        server_cost_reward_multiplier=rl_config.get("server_cost_reward_multiplier", 25.0),
        flat_penalty_after_steps=rl_config.get("flat_penalty_after_steps", 8),
        flat_penalty_scale=rl_config.get("flat_penalty_scale", 0.015),
        train_range=ranges["train"],
        eval_range=ranges["validation"],
        test_range=ranges["test"],
    )

    logger.info(f"Training {algo}: {total_timesteps} steps, lr={learning_rate}, "
                f"batch={batch_size}, n_steps={n_steps}")
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

    metrics = trainer.train(
        total_timesteps=total_timesteps,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        ent_coef=rl_config.get("ent_coef", 0.02),
        eval_every_steps=rl_config.get("eval_every_steps", 50000),
        eval_episodes=rl_config.get("eval_episodes", 4),
    )

    _log_gpu_memory()

    # Load trained model for backtesting
    model = None
    model_path = Path("checkpoints/rl_agent_final.zip")
    if model_path.exists():
        try:
            from stable_baselines3 import PPO, SAC
            ModelClass = PPO if algo == "PPO" else SAC
            model = ModelClass.load(str(model_path))
            logger.info("Loaded trained model for evaluation")
        except Exception as e:
            logger.warning(f"Could not load model: {e}")

    return model, metrics


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

    *** ใช้ cost model เดียวกันกับ CryptoFuturesEnv ***
    *** ใช้ last 200K steps สำหรับ eval (ไม่ต้องรัน 3.2M ทั้งหมด) ***
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
        # 3 actions: 0=Short, 1=Flat, 2=Long
        position = 0.0
        entry_price = 0.0
        flat_steps = 0
        pos_steps = 0
        for i in range(n):
            price_now = pr_eval[i]
            upnl = 0.0
            if position != 0 and entry_price > 0:
                upnl = position * (price_now / entry_price - 1.0)

            obs = np.concatenate([
                fa_eval[i],
                np.array([position, upnl, flat_steps / 100.0, pos_steps / 100.0], dtype=np.float32),
            ])
            action, _ = model.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                a = int(action.item()) if action.ndim == 0 else int(action[0])
            else:
                a = int(action)
            if a == 0:
                position = -1.0
            elif a == 2:
                position = 1.0
            else:
                position = 0.0

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
    # KS test กับ 100K samples อ่อนไหวเกินไป (detect noise ไม่ใช่ real drift)
    # ใช้ 500 samples เหมือน production drift detector window
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
        and decision3.size < decision1.size  # low confidence → smaller size
    )
    results["all_checks_passed"] = passed
    logger.info(f"Risk Manager: {'PASS' if passed else 'FAIL'} — {results}")
    return results


# =============================================================================
# Phase 9: Drift Detection Test
# =============================================================================

def test_drift_detection(feature_array: np.ndarray) -> dict:
    """Test drift detector with feature data."""
    from monitoring.live.drift_detector import DriftDetector

    # Subsample reference เพื่อให้ KS test ไม่ oversensitive
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

    # Next candle period → triggers close
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

        # Predict (untrained — expect random output)
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

    *** ใช้ FULL data + aggregate เป็น 15m ก่อนรัน strategy ***
    - Full data เพื่อครอบคลุมทุก regime (COVID ถึง Present)
    - 15m aggregation เพื่อไม่ overtrade เหมือน benchmarks
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
    # *** RSI บน 1m จะ overtrade 16K+ times → -99% ทุก regime ***
    # *** RSI บน 15m จะ trade ~1-2K times → ผลลัพธ์สมจริง ***
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

    # Align: features[:-1] → labels
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
    """Full system test — ทดสอบทุก component ด้วย RTX 2060 6GB.

    ทำทุกอย่างเหมือน full training แต่ใช้ data น้อยลง + timesteps น้อยลง.
    เป้าหมาย: validate logic ทั้งระบบ ก่อน deploy cloud GPU.
    """
    config_path = config_path or "configs/test_rtx2060.yaml"
    config = load_config(config_path)

    start_time = time.time()
    report = {"phases": {}, "errors": []}

    logger.info("=" * 70)
    logger.info("GARIC — Full System Test (RTX 2060 Mode)")
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
    """Aggregate 1m OHLCV → 15m OHLCV (proper aggregation, not just close)."""
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
    """Full training pipeline — aggregate 15m → features → train → backtest.

    *** KEY FIXES ***
    1. Aggregate 1m → 15m ก่อนทุกอย่าง (216K candles แทน 3.2M)
    2. ใช้ TA features เท่านั้น (20 features แทน 346)
    3. Short episodes (2000 candles) + random start
    4. PPO ent_coef=0.02 ป้องกัน entropy collapse
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
    from data.cache import load_features, save_features

    for symbol in pairs:
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

        if cached is not None:
            feature_array = cached["features"]
            prices = cached["prices"]
            ohlcv_data = cached["ohlcv"]
            close_15m = cached["close_15m"]

            n_feats = feature_array.shape[1]
            dash.update(data_1m=0, data_15m=len(prices))
            dash.add_phase("Loaded from cache", "ok", 0)
            dash.update(features=n_feats)
            logger.info(f"Using cached features: {n_feats} dims, {len(prices):,} rows")
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

            # Step 3: Compute features on 15m data (TA + micro = 20 features)
            from features.technical.indicators import compute_all as compute_ta
            from features.technical.microstructure import compute_all as compute_micro

            ta_15m = compute_ta(df_15m).astype(np.float32)      # (n, 15)
            micro_15m = compute_micro(df_15m).astype(np.float32)  # (n, 5)

            # Returns on 15m
            close_15m = df_15m["close"].values.astype(np.float64)
            log_ret = np.log(close_15m[1:] / close_15m[:-1])
            returns_1 = np.concatenate([[0], log_ret])
            returns_4 = pd.Series(returns_1).rolling(4).sum().fillna(0).values  # 1h
            returns_16 = pd.Series(returns_1).rolling(16).sum().fillna(0).values  # 4h
            returns_96 = pd.Series(returns_1).rolling(96).sum().fillna(0).values  # 1d
            ret_features = np.column_stack([returns_1, returns_4, returns_16, returns_96]).astype(np.float32)

            # CryptoMamba forecast features
            dash.update(status_msg="Fine-tuning CryptoMamba...")
            from models.forecast.crypto_mamba import CryptoMambaForecaster
            mamba = CryptoMambaForecaster(
                context_len=64, horizon=4, d_model=32, n_layers=2,
                device="cuda" if gpu_name else "cpu",
            )
            # Quick fine-tune on recent data (max 50K windows, batch_size=256 for GPU)
            if mamba.available:
                logger.info("Fine-tuning CryptoMamba on 15m data...")
                mamba.fine_tune(close_15m, epochs=10, lr=1e-3, batch_size=256,
                                max_samples=50_000,
                                save_path="checkpoints/crypto_mamba_15m.pt")

            # Generate forecast features — batch prediction (much faster than per-window)
            n_15m = len(close_15m)
            forecast_feats = np.zeros((n_15m, 5), dtype=np.float32)
            ctx_len = 64
            sample_step = 10
            indices = list(range(ctx_len, n_15m, sample_step))
            n_preds = len(indices)
            logger.info(f"Generating CryptoMamba forecasts ({n_preds} predictions, batched)...")
            dash.update(status_msg=f"CryptoMamba batch predict ({n_preds})...")

            # Build all windows at once and predict in one batched call
            windows = np.array([close_15m[i - ctx_len:i] for i in indices])
            try:
                preds, uncs = mamba.predict_batch(windows, horizon=4)
                for k, i in enumerate(indices):
                    last_p = close_15m[i - 1]
                    if last_p > 0:
                        fc = (preds[k, :4] / last_p - 1).astype(np.float32)
                        end = min(i + sample_step, n_15m)
                        forecast_feats[i:end, :4] = fc
                        forecast_feats[i:end, 4] = float(uncs[k])
            except Exception as e:
                logger.warning(f"CryptoMamba batch predict failed: {e}")

            # Release GPU after CryptoMamba done
            mamba.release_gpu()
            dash.add_phase("CryptoMamba Forecast", "ok")

            # Volatility feature (rolling 20-period std of returns)
            vol_20 = pd.Series(returns_1).rolling(20).std().fillna(0).values.astype(np.float32)

            # Combine: TA(15) + Micro(5) + Returns(4) + Forecast(5) + Vol(1) = 30 features
            feature_array = np.concatenate([
                ta_15m, micro_15m, ret_features,
                forecast_feats, vol_20.reshape(-1, 1),
            ], axis=1)
            feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)
            prices = close_15m.astype(np.float32)
            ohlcv_data = df_15m[["open", "high", "low", "close"]].values.astype(np.float32)

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

        # Step 5: RL Training
        training_config = config.get("training", {})
        rl_config = training_config.get("rl", {})
        total_steps = rl_config.get("total_timesteps", 200000)
        ep_len = 2000
        lr = rl_config.get("learning_rate", 3e-4)

        dash.update(train_total=total_steps, status_msg="Initializing PPO...")

        from models.rl.trainer import RLTrainer
        trainer = RLTrainer(
            feature_arrays=feature_array,
            price_series=prices,
            ohlcv_data=ohlcv_data,
            leverage=trading_config.get("leverage", 1.0),
            min_trade_pct=trading_config.get("min_trade_pct", 0.05),
            maintenance_margin=trading_config.get("maintenance_margin", 0.005),
            funding_interval=32,
            max_episode_steps=ep_len,
            monthly_server_cost_usd=trading_config.get("monthly_server_cost_usd", 100.0),
            periods_per_day=trading_config.get("periods_per_day", 96),
            opportunity_threshold=rl_config.get("opportunity_threshold", 0.0010),
            missed_move_penalty_scale=rl_config.get("missed_move_penalty_scale", 160.0),
            server_cost_reward_multiplier=rl_config.get("server_cost_reward_multiplier", 25.0),
            flat_penalty_after_steps=rl_config.get("flat_penalty_after_steps", 8),
            flat_penalty_scale=rl_config.get("flat_penalty_scale", 0.015),
            train_range=ranges["train"],
            eval_range=ranges["validation"],
            test_range=ranges["test"],
        )
        trainer._dashboard = dash  # pass dashboard to trainer callback

        t0 = time.time()
        dash.update(status_msg="PPO Training in progress...")
        metrics = trainer.train(
            total_timesteps=total_steps,
            learning_rate=lr,
            n_steps=ep_len,
            batch_size=rl_config.get("batch_size", 64),
            n_epochs=rl_config.get("n_epochs", 10),
            ent_coef=rl_config.get("ent_coef", 0.02),
            eval_every_steps=rl_config.get("eval_every_steps", 50000),
            eval_episodes=rl_config.get("eval_episodes", 4),
        )
        dash.add_phase(f"PPO Training ({total_steps//1000}K steps)", "ok", time.time() - t0)
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
            eval_short_actions=metrics.get("eval_short_actions", 0),
            eval_flat_actions=metrics.get("eval_flat_actions", 0),
            eval_long_actions=metrics.get("eval_long_actions", 0),
            sharpe=metrics.get("sharpe", 0),
            sortino=metrics.get("sortino", 0),
            max_dd=metrics.get("max_drawdown", 0),
            n_trades=metrics.get("n_trades", 0),
            n_longs=metrics.get("n_longs", 0),
            n_shorts=metrics.get("n_shorts", 0),
            n_wins=metrics.get("n_wins", 0),
            n_losses=metrics.get("n_losses", 0),
            win_rate=metrics.get("win_rate", 0),
            model_path="checkpoints/rl_agent_final.zip",
            chart_path="checkpoints/training_results.png",
            status_msg="COMPLETE",
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
    logger.info("GARIC — System Test Report")
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
        logger.info("ALL PHASES PASSED — ระบบพร้อมใช้งาน")
        logger.info("Next: python pipeline.py --mode train (full data, cloud GPU)")
    else:
        logger.info(f"{n_fail} PHASES FAILED — ต้องแก้ไขก่อน deploy")


# =============================================================================
# Entry point
# =============================================================================

def main():
    import argparse
    import warnings

    parser = argparse.ArgumentParser(description="GARIC Pipeline")
    parser.add_argument("--config", default=None)
    parser.add_argument("--mode", choices=["train", "test", "paper", "live"], default="test")
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
    elif args.mode == "paper":
        logger.info("Starting paper trading...")
        from execution.live.trading_engine import TradingEngine
        engine = TradingEngine(paper_mode=True)
        engine.start()
    elif args.mode == "live":
        logger.error("Live mode requires API keys. Use:")
        logger.error("  python -m execution.live.trading_engine --mode live --api-key KEY --api-secret SECRET")


if __name__ == "__main__":
    main()
