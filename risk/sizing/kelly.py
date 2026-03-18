"""Position sizing — Fractional Kelly Criterion + ATR method.

ใช้ Quarter Kelly สำหรับ crypto (Full Kelly เสี่ยงเกินไป).
Max 5% equity ต่อ trade (BTC), 2% (altcoin).
"""

import numpy as np


def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.25,  # Quarter Kelly
) -> float:
    """Fractional Kelly Criterion.

    K% = W - (1-W)/R  where W=win_rate, R=avg_win/avg_loss
    แล้วคูณ fraction (0.25 = Quarter Kelly)
    """
    if avg_loss == 0 or avg_win == 0:
        return 0.0

    r = avg_win / avg_loss
    kelly = win_rate - (1 - win_rate) / r
    return max(0.0, kelly * fraction)


def atr_position_size(
    equity: float,
    atr_value: float,
    risk_pct: float = 0.02,
    atr_multiplier: float = 2.0,
) -> float:
    """ATR-based position sizing.

    position_size = (equity × risk_pct) / (ATR × multiplier)
    """
    if atr_value <= 0:
        return 0.0
    return (equity * risk_pct) / (atr_value * atr_multiplier)


def compute_position_size(
    equity: float,
    model_confidence: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    atr_value: float,
    is_major: bool = True,
    regime_uncertain: bool = False,
) -> float:
    """คำนวณ position size สุดท้าย.

    Combine Kelly + ATR + constraints.
    """
    # Kelly fraction
    kf = kelly_fraction(win_rate, avg_win, avg_loss, fraction=0.25)

    # ATR size
    atr_size = atr_position_size(equity, atr_value)

    # Take minimum of Kelly and ATR
    size = min(kf * equity, atr_size)

    # Scale by model confidence
    size *= np.clip(model_confidence, 0.1, 1.0)

    # Max position constraint
    max_pct = 0.05 if is_major else 0.02
    size = min(size, equity * max_pct)

    # Reduce 50% if regime uncertain
    if regime_uncertain:
        size *= 0.5

    return max(0.0, size)
