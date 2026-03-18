"""Shared performance metric helpers for training, backtests, and dashboards."""

from __future__ import annotations

import math

import numpy as np


ANNUALIZATION_15M = 252 * 24 * 4
_EPS = 1e-12
_MIN_STD = 1e-8
_MIN_SPREAD = 1e-7


def sanitize_returns(returns: np.ndarray | list[float]) -> np.ndarray:
    """Return a finite float64 array of returns."""
    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr[np.isfinite(arr)]


def equity_to_returns(equity_curve: np.ndarray | list[float]) -> np.ndarray:
    """Convert an equity curve to simple returns."""
    equity = np.asarray(equity_curve, dtype=np.float64)
    if equity.size < 2:
        return np.array([], dtype=np.float64)

    prev = np.where(np.abs(equity[:-1]) > _EPS, equity[:-1], 1.0)
    return sanitize_returns(np.diff(equity) / prev)


def drawdown_series(equity_curve: np.ndarray | list[float]) -> np.ndarray:
    """Return drawdown magnitudes in [0, 1]."""
    equity = np.asarray(equity_curve, dtype=np.float64)
    if equity.size == 0:
        return np.array([], dtype=np.float64)

    peak = np.maximum.accumulate(equity)
    denom = np.where(peak > _EPS, peak, 1.0)
    return (peak - equity) / denom


def max_drawdown(equity_curve: np.ndarray | list[float]) -> float:
    """Return max drawdown as a positive magnitude."""
    dd = drawdown_series(equity_curve)
    return float(dd.max()) if dd.size else 0.0


def format_drawdown_pct(drawdown: float) -> str:
    """Format drawdown as a negative percentage for display."""
    return f"{-abs(drawdown):.2%}"


def safe_sharpe_ratio(
    returns: np.ndarray | list[float],
    annualization_factor: float = ANNUALIZATION_15M,
) -> float:
    """Sharpe ratio with guards for flat or numerically unstable series."""
    arr = sanitize_returns(returns)
    if arr.size < 2:
        return 0.0

    std = float(arr.std())
    spread = float(arr.max() - arr.min())
    if std < _MIN_STD or spread < _MIN_SPREAD:
        return 0.0

    scale = math.sqrt(max(annualization_factor, 1.0))
    return float(arr.mean() / std * scale)


def safe_sortino_ratio(
    returns: np.ndarray | list[float],
    annualization_factor: float = ANNUALIZATION_15M,
) -> float:
    """Sortino ratio with the same numerical guards as Sharpe."""
    arr = sanitize_returns(returns)
    if arr.size < 2:
        return 0.0

    downside = arr[arr < 0]
    if downside.size < 2:
        return safe_sharpe_ratio(arr, annualization_factor=annualization_factor)

    downside_std = float(downside.std())
    downside_spread = float(downside.max() - downside.min())
    if downside_std < _MIN_STD or downside_spread < _MIN_SPREAD:
        return 0.0

    scale = math.sqrt(max(annualization_factor, 1.0))
    return float(arr.mean() / downside_std * scale)


def summarize_equity_curve(
    equity_curve: np.ndarray | list[float],
    annualization_factor: float = ANNUALIZATION_15M,
) -> dict[str, float]:
    """Return core metrics from an equity curve."""
    equity = np.asarray(equity_curve, dtype=np.float64)
    if equity.size == 0:
        return {
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "total_return": 0.0,
        }

    returns = equity_to_returns(equity)
    total_return = float(equity[-1] / equity[0] - 1) if equity[0] != 0 else 0.0

    return {
        "sharpe": safe_sharpe_ratio(returns, annualization_factor=annualization_factor),
        "sortino": safe_sortino_ratio(returns, annualization_factor=annualization_factor),
        "max_drawdown": max_drawdown(equity),
        "total_return": total_return,
    }
