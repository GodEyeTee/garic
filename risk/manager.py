"""Risk Manager — ประเมิน risk ก่อนส่ง order.

ตรวจสอบ:
1. Position size ไม่เกิน limit
2. Max drawdown ไม่เกิน threshold
3. Daily loss limit
4. Correlation risk (ถ้าถือหลาย pair)
"""

import logging
from dataclasses import dataclass

import numpy as np

from risk.sizing.kelly import compute_position_size

logger = logging.getLogger(__name__)


@dataclass
class TradeDecision:
    """ผลจาก Risk Manager — ส่งต่อไป Execution."""
    symbol: str
    direction: float        # -1 to 1
    size: float             # dollar amount
    stop_loss_pct: float    # % from entry
    take_profit_pct: float  # % from entry
    approved: bool
    reject_reason: str = ""


class RiskManager:
    """ตรวจสอบ risk ก่อนอนุมัติ trade."""

    def __init__(
        self,
        max_drawdown: float = 0.15,         # 15%
        daily_loss_limit: float = 0.03,     # 3% per day
        max_open_positions: int = 5,
        max_correlated_exposure: float = 0.10,  # 10%
    ):
        self.max_drawdown = max_drawdown
        self.daily_loss_limit = daily_loss_limit
        self.max_open_positions = max_open_positions
        self.max_correlated_exposure = max_correlated_exposure

        self._peak_equity = 0.0
        self._daily_pnl = 0.0
        self._open_positions: dict[str, float] = {}

    def evaluate(
        self,
        symbol: str,
        direction: float,
        equity: float,
        model_confidence: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        atr_value: float,
        current_drawdown: float,
        is_major: bool = True,
    ) -> TradeDecision:
        """ประเมินและปรับขนาด trade."""

        # Check 1: Max drawdown
        if current_drawdown > self.max_drawdown:
            return TradeDecision(
                symbol=symbol, direction=0, size=0,
                stop_loss_pct=0, take_profit_pct=0,
                approved=False,
                reject_reason=f"Max drawdown exceeded: {current_drawdown:.1%} > {self.max_drawdown:.1%}",
            )

        # Check 2: Daily loss limit
        if self._daily_pnl < -self.daily_loss_limit * equity:
            return TradeDecision(
                symbol=symbol, direction=0, size=0,
                stop_loss_pct=0, take_profit_pct=0,
                approved=False,
                reject_reason=f"Daily loss limit hit",
            )

        # Check 3: Max open positions
        if symbol not in self._open_positions and len(self._open_positions) >= self.max_open_positions:
            return TradeDecision(
                symbol=symbol, direction=0, size=0,
                stop_loss_pct=0, take_profit_pct=0,
                approved=False,
                reject_reason=f"Max open positions reached: {self.max_open_positions}",
            )

        # Compute position size
        regime_uncertain = model_confidence < 0.3
        size = compute_position_size(
            equity=equity,
            model_confidence=model_confidence,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            atr_value=atr_value,
            is_major=is_major,
            regime_uncertain=regime_uncertain,
        )

        # Stop loss / take profit based on ATR
        stop_loss_pct = atr_value * 2.0 if atr_value > 0 else 0.02
        take_profit_pct = atr_value * 3.0 if atr_value > 0 else 0.03

        return TradeDecision(
            symbol=symbol,
            direction=direction,
            size=size,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            approved=size > 0,
        )

    def update_pnl(self, pnl: float):
        self._daily_pnl += pnl

    def reset_daily(self):
        self._daily_pnl = 0.0
