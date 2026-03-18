"""Regression tests for performance calculations and inactivity costs."""

import numpy as np
import pytest

from execution.backtest.runner import BacktestConfig, BacktestRunner
from models.rl.environment import CryptoFuturesEnv
from performance import format_drawdown_pct, safe_sharpe_ratio, safe_sortino_ratio


def test_zero_variance_ratios_do_not_explode():
    returns = np.full(128, -0.001, dtype=np.float64)
    assert safe_sharpe_ratio(returns) == 0.0
    assert safe_sortino_ratio(returns) == 0.0


def test_drawdown_display_uses_negative_sign():
    assert format_drawdown_pct(0.0069444444) == "-0.69%"


def test_env_inactivity_pays_server_cost_but_keeps_gross_flat():
    n = 200
    features = np.zeros((n, 8), dtype=np.float32)
    prices = np.full(n, 100.0, dtype=np.float64)

    env = CryptoFuturesEnv(
        features,
        prices,
        max_episode_steps=100,
        monthly_server_cost_usd=100.0,
        periods_per_day=96,
    )
    obs, _ = env.reset(seed=42)

    for _ in range(100):
        obs, reward, term, trunc, info = env.step(1)  # stay flat
        if term or trunc:
            break

    metrics = env.get_metrics()
    assert metrics["n_trades"] == 0
    assert metrics["gross_total_return"] == pytest.approx(0.0)
    assert metrics["total_return"] < 0.0
    assert metrics["sharpe"] == pytest.approx(0.0)
    assert metrics["sortino"] == pytest.approx(0.0)
    assert metrics["server_cost_paid"] > 0.0


def test_backtest_inactivity_tracks_server_cost_separately():
    prices = np.full(97, 100.0, dtype=np.float64)
    signals = np.zeros(97, dtype=np.float64)

    runner = BacktestRunner(
        BacktestConfig(
            funding_interval=0,
            monthly_server_cost_usd=100.0,
            periods_per_day=96,
        )
    )
    result = runner.run(prices, signals)
    metrics = result.metrics

    assert metrics["n_trades"] == 0
    assert metrics["gross_total_return"] == pytest.approx(0.0)
    assert metrics["total_return"] < 0.0
    assert metrics["sharpe"] == pytest.approx(0.0)
    assert metrics["sortino"] == pytest.approx(0.0)
    assert metrics["server_cost_paid"] == pytest.approx(100.0 / 30.0, rel=1e-3)
