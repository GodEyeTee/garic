"""Multi-regime backtesting — ทดสอบ strategy ใน market regime ต่างๆ.

ทดสอบว่า strategy ทำงานได้ทุก regime หรือแค่ regime เดียว.
ถ้าทำได้แค่ regime เดียว → ต้องใช้ MoE routing.

Regime periods (BTC):
  - COVID Crash: 2020-02 to 2020-04
  - Bull Run 2021: 2021-01 to 2021-11
  - Luna Crash: 2022-05 to 2022-07
  - FTX Collapse: 2022-10 to 2022-12
  - Recovery 2023: 2023-01 to 2023-12
  - Bull 2024-25: 2024-01 to 2025-01
  - Recent: 2025-01 to present
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from execution.backtest.runner import BacktestRunner, BacktestConfig

logger = logging.getLogger(__name__)


@dataclass
class RegimePeriod:
    name: str
    start: str      # YYYY-MM-DD
    end: str         # YYYY-MM-DD
    description: str


# Known crypto market regimes
KNOWN_REGIMES = [
    RegimePeriod("COVID Crash", "2020-02-15", "2020-04-15",
                 "BTC -50% in 2 days, V-shaped recovery"),
    RegimePeriod("Bull Run 2021", "2021-01-01", "2021-11-10",
                 "BTC 29K→69K, high momentum"),
    RegimePeriod("Luna Crash", "2022-05-01", "2022-07-01",
                 "UST depeg, BTC 40K→19K, contagion"),
    RegimePeriod("FTX Collapse", "2022-10-01", "2022-12-31",
                 "BTC 20K→15K, trust crisis"),
    RegimePeriod("Recovery 2023", "2023-01-01", "2023-12-31",
                 "BTC 16K→42K, slow recovery + ranging"),
    RegimePeriod("Bull 2024", "2024-01-01", "2024-12-31",
                 "ETF approval, halving, BTC→100K"),
    RegimePeriod("Recent 2025-26", "2025-01-01", "2026-03-15",
                 "Current market conditions"),
]


def run_regime_backtest(
    df: pd.DataFrame,
    signals: np.ndarray,
    time_col: str = "open_time",
    regimes: list[RegimePeriod] | None = None,
) -> list[dict]:
    """Run backtest on each regime period separately.

    Args:
        df: full OHLCV DataFrame (ต้องมี time_col + close)
        signals: position signals aligned with df (same length)
        time_col: timestamp column name
        regimes: list of regime periods (default: KNOWN_REGIMES)

    Returns:
        list of dicts: metrics per regime + overall
    """
    if regimes is None:
        regimes = KNOWN_REGIMES

    results = []
    config = BacktestConfig(maker_fee=0.0002, taker_fee=0.0005, slippage_bps=1.0)
    runner = BacktestRunner(config)

    # Ensure timestamps are datetime
    timestamps = pd.to_datetime(df[time_col])

    logger.info(f"{'Regime':25s} | {'Sharpe':>8s} | {'Return':>8s} | {'MaxDD':>7s} | {'WinRate':>7s} | {'Rows':>8s}")
    logger.info("-" * 80)

    for regime in regimes:
        start = pd.Timestamp(regime.start)
        end = pd.Timestamp(regime.end)

        mask = (timestamps >= start) & (timestamps < end)
        n_rows = mask.sum()

        if n_rows < 100:
            logger.info(f"  {regime.name:25s} | SKIP (only {n_rows} rows)")
            continue

        prices_r = df.loc[mask, "close"].values
        signals_r = signals[mask.values] if isinstance(signals, np.ndarray) else signals[mask]

        result = runner.run(prices_r, signals_r)
        metrics = result.metrics
        metrics["regime"] = regime.name
        metrics["start"] = regime.start
        metrics["end"] = regime.end
        metrics["n_rows"] = n_rows
        metrics["description"] = regime.description

        logger.info(
            f"  {regime.name:25s} | "
            f"{metrics.get('sharpe', 0):8.3f} | "
            f"{metrics.get('total_return', 0):8.3%} | "
            f"{metrics.get('max_drawdown', 0):7.3%} | "
            f"{metrics.get('win_rate', 0):7.3%} | "
            f"{n_rows:8,d}"
        )
        results.append(metrics)

    # Overall
    if len(results) > 0:
        avg_sharpe = np.mean([r.get("sharpe", 0) for r in results])
        avg_return = np.mean([r.get("total_return", 0) for r in results])
        worst_dd = max([r.get("max_drawdown", 0) for r in results])
        n_positive = sum(1 for r in results if r.get("total_return", 0) > 0)

        summary = {
            "regime": "OVERALL",
            "avg_sharpe": avg_sharpe,
            "avg_return": avg_return,
            "worst_max_drawdown": worst_dd,
            "n_regimes_tested": len(results),
            "n_positive_regimes": n_positive,
            "consistency": n_positive / len(results),  # % regimes profitable
        }
        results.append(summary)

        logger.info("-" * 80)
        logger.info(
            f"  {'OVERALL':25s} | "
            f"{avg_sharpe:8.3f} | "
            f"{avg_return:8.3%} | "
            f"{worst_dd:7.3%} | "
            f"{'':7s} | "
            f"profitable: {n_positive}/{len(results) - 1}"
        )

    return results


def compute_regime_stats(df: pd.DataFrame, time_col: str = "open_time") -> list[dict]:
    """Compute statistics for each regime period (ไม่ต้อง signals).

    ใช้ดู characteristics ของแต่ละ regime.
    """
    timestamps = pd.to_datetime(df[time_col])
    stats = []

    for regime in KNOWN_REGIMES:
        start = pd.Timestamp(regime.start)
        end = pd.Timestamp(regime.end)
        mask = (timestamps >= start) & (timestamps < end)

        if mask.sum() < 100:
            continue

        prices = df.loc[mask, "close"].values
        log_returns = np.diff(np.log(prices))

        stats.append({
            "regime": regime.name,
            "start": regime.start,
            "end": regime.end,
            "n_rows": int(mask.sum()),
            "price_start": float(prices[0]),
            "price_end": float(prices[-1]),
            "total_return": float(prices[-1] / prices[0] - 1),
            "annualized_vol": float(log_returns.std() * np.sqrt(525600)),  # 1m → yearly
            "avg_return_1m": float(log_returns.mean()),
            "skewness": float(_skewness(log_returns)),
            "kurtosis": float(_kurtosis(log_returns)),
            "max_drawdown": float(_max_drawdown(prices)),
        })

    return stats


def _skewness(x: np.ndarray) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    m = x.mean()
    s = x.std()
    if s == 0:
        return 0.0
    return (n / ((n - 1) * (n - 2))) * np.sum(((x - m) / s) ** 3)


def _kurtosis(x: np.ndarray) -> float:
    n = len(x)
    if n < 4:
        return 3.0
    m = x.mean()
    s = x.std()
    if s == 0:
        return 3.0
    return np.mean(((x - m) / s) ** 4)


def _max_drawdown(prices: np.ndarray) -> float:
    peak = np.maximum.accumulate(prices)
    dd = (peak - prices) / np.where(peak > 0, peak, 1)
    return float(dd.max())
