"""Benchmark strategies สำหรับเปรียบเทียบกับ RL agent.

RL agent ต้องชนะ benchmarks เหล่านี้ ถ้าไม่ชนะ → strategy ไม่มีค่า.

Benchmarks:
1. Buy and Hold (long-only)
2. MACD Crossover
3. Bollinger Band Mean Reversion
4. Dual Moving Average Crossover
5. RSI Overbought/Oversold (baseline ที่ใช้อยู่แล้ว)
"""

import logging

import numpy as np

from execution.backtest.runner import BacktestRunner, BacktestConfig

logger = logging.getLogger(__name__)


def _run_backtest(prices: np.ndarray, signals: np.ndarray, name: str) -> dict:
    """Run backtest and return metrics with strategy name."""
    config = BacktestConfig(maker_fee=0.0002, taker_fee=0.0005, slippage_bps=1.0)
    runner = BacktestRunner(config)
    result = runner.run(prices, signals)
    metrics = result.metrics
    metrics["strategy"] = name
    metrics["n_trades"] = len(result.trades)
    return metrics


def buy_and_hold(prices: np.ndarray) -> dict:
    """Long 100% ตลอด — ง่ายที่สุด, benchmark พื้นฐาน."""
    signals = np.ones(len(prices)) * 1.0
    return _run_backtest(prices, signals, "Buy & Hold")


def macd_crossover(
    prices: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
    position_size: float = 0.5,
) -> dict:
    """MACD crossover: long เมื่อ MACD > signal, short เมื่อ MACD < signal."""
    ema_fast = _ema(prices, fast)
    ema_slow = _ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal_period)

    signals = np.where(macd_line > signal_line, position_size, -position_size)
    # Warmup: flat during first slow period
    signals[:slow] = 0.0

    return _run_backtest(prices, signals, f"MACD({fast},{slow},{signal_period})")


def bollinger_mean_reversion(
    prices: np.ndarray,
    window: int = 20,
    n_std: float = 2.0,
    position_size: float = 0.5,
) -> dict:
    """Bollinger Band mean reversion: long เมื่อ price < lower, short เมื่อ > upper."""
    sma = _rolling_mean(prices, window)
    std = _rolling_std(prices, window)

    upper = sma + n_std * std
    lower = sma - n_std * std

    signals = np.zeros(len(prices))
    for i in range(window, len(prices)):
        if prices[i] < lower[i]:
            signals[i] = position_size   # long: below lower band
        elif prices[i] > upper[i]:
            signals[i] = -position_size  # short: above upper band
        else:
            signals[i] = signals[i - 1] * 0.95  # decay position toward 0

    return _run_backtest(prices, signals, f"BB-MR({window},{n_std})")


def dual_ma_crossover(
    prices: np.ndarray,
    fast: int = 10,
    slow: int = 50,
    position_size: float = 0.5,
) -> dict:
    """Dual Moving Average: long เมื่อ fast MA > slow MA."""
    ma_fast = _rolling_mean(prices, fast)
    ma_slow = _rolling_mean(prices, slow)

    signals = np.where(ma_fast > ma_slow, position_size, -position_size)
    signals[:slow] = 0.0

    return _run_backtest(prices, signals, f"DualMA({fast},{slow})")


def rsi_strategy(
    prices: np.ndarray,
    period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
    position_size: float = 0.5,
) -> dict:
    """RSI overbought/oversold — baseline ที่ใช้อยู่."""
    rsi = _compute_rsi(prices, period)
    signals = np.where(rsi < oversold, position_size, np.where(rsi > overbought, -position_size, 0.0))
    return _run_backtest(prices, signals, f"RSI({period})")


def aggregate_to_candles(prices_1m: np.ndarray, period: int = 15) -> np.ndarray:
    """Aggregate 1m close prices to higher timeframe (e.g. 15m).

    ใช้ close ของ candle สุดท้ายในแต่ละ period.
    สำคัญ: strategies ต้องทำงานบน candle ที่เหมาะสม ไม่ใช่ 1m ตรงๆ
    """
    n = len(prices_1m)
    n_candles = n // period
    # Take close price at end of each period
    indices = np.arange(period - 1, n_candles * period, period)
    return prices_1m[indices]


def run_all_benchmarks(
    prices: np.ndarray,
    candle_period: int = 15,
) -> list[dict]:
    """Run ทุก benchmark strategy บน aggregated candles.

    *** สำคัญ: strategies ออกแบบสำหรับ 15m+ ไม่ใช่ 1m ***
    ถ้ารันบน 1m จะ overtrade → ค่า fee กินทุนหมด.
    """
    # Aggregate to higher timeframe for signal-based strategies
    prices_agg = aggregate_to_candles(prices, candle_period)
    logger.info(f"Aggregated {len(prices):,} 1m candles → {len(prices_agg):,} {candle_period}m candles")

    benchmarks = [
        ("Buy & Hold", buy_and_hold),
        ("MACD", macd_crossover),
        ("BB Mean Rev", bollinger_mean_reversion),
        ("Dual MA", dual_ma_crossover),
        ("RSI", rsi_strategy),
    ]

    results = []
    for name, func in benchmarks:
        try:
            metrics = func(prices_agg)
            metrics["timeframe"] = f"{candle_period}m"
            results.append(metrics)
            logger.info(
                f"  {metrics['strategy']:20s} | "
                f"Sharpe={metrics.get('sharpe', 0):8.3f} | "
                f"Return={metrics.get('total_return', 0):8.3%} | "
                f"MaxDD={metrics.get('max_drawdown', 0):7.3%} | "
                f"Trades={metrics.get('n_trades', 0):6d}"
            )
        except Exception as e:
            logger.error(f"Benchmark {name} failed: {e}")

    return results


# =============================================================================
# Helpers (pure numpy)
# =============================================================================

def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average."""
    alpha = 2 / (period + 1)
    result = np.zeros_like(data, dtype=np.float64)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return result


def _rolling_mean(data: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean."""
    cumsum = np.cumsum(np.insert(data.astype(np.float64), 0, 0))
    result = np.zeros_like(data, dtype=np.float64)
    result[window - 1:] = (cumsum[window:] - cumsum[:-window]) / window
    result[:window - 1] = data[:window - 1]
    return result


def _rolling_std(data: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation."""
    result = np.zeros_like(data, dtype=np.float64)
    for i in range(window - 1, len(data)):
        result[i] = data[i - window + 1:i + 1].std()
    result[:window - 1] = result[window - 1]
    return np.maximum(result, 1e-10)


def _compute_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI calculation."""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = _ema(gains, period)
    avg_loss = _ema(losses, period)
    avg_loss = np.where(avg_loss == 0, 1e-10, avg_loss)

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Prepend first value
    return np.concatenate([[50.0], rsi])
