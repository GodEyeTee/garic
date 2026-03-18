"""Backtest runner — realistic Binance Futures simulation.

รวม:
- Slippage model (spread + market impact + fees)
- Funding rate ทุก 8 ชม.
- Leverage + liquidation เหมือน Binance จริง
- Minimum trade size (ป้องกัน micro-adjustments)
- Position sizing จาก Risk Manager
- Performance metrics + CPCV validation

*** ใช้ cost model เดียวกันกับ CryptoFuturesEnv ***
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from performance import summarize_equity_curve

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    initial_balance: float = 10000.0
    maker_fee: float = 0.0002      # 0.02%
    taker_fee: float = 0.0005      # 0.05%
    slippage_bps: float = 1.0      # 1 basis point
    funding_interval: int = 480    # candles between funding (8h at 1m)
    leverage: float = 1.0          # default 1x (conservative)
    maintenance_margin: float = 0.005  # 0.5% Binance BTC default
    min_trade_pct: float = 0.05    # minimum 5% position change
    monthly_server_cost_usd: float = 100.0
    periods_per_day: int = 96


@dataclass
class BacktestResult:
    equity_curve: list[float] = field(default_factory=list)
    gross_equity_curve: list[float] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


class BacktestRunner:
    """Run backtest with realistic cost model — SAME as CryptoFuturesEnv."""

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()

    def run(
        self,
        prices: np.ndarray,
        signals: np.ndarray,
        funding_rates: np.ndarray | None = None,
    ) -> BacktestResult:
        """Run backtest.

        Args:
            prices: close prices, shape (n,)
            signals: position signals [-1, 1], shape (n,)
            funding_rates: optional, shape (n,)

        *** signals ต้องมาจาก candle close decision เท่านั้น ***
        *** cost model เดียวกับ CryptoFuturesEnv ***
        """
        n = len(prices)
        assert len(signals) == n

        if funding_rates is None:
            funding_rates = np.zeros(n)

        cfg = self.config
        slippage_rate = cfg.slippage_bps / 10000

        balance = cfg.initial_balance
        gross_balance = cfg.initial_balance
        position = 0.0
        entry_price = 0.0
        entry_fee_return = 0.0
        equity_curve = [balance]
        gross_equity_curve = [gross_balance]
        trades = []
        n_longs = 0
        n_shorts = 0
        n_wins = 0
        n_losses = 0
        server_cost_paid = 0.0
        flat_steps_total = 0
        position_steps_total = 0

        for i in range(1, n):
            target = float(signals[i])
            price = prices[i]
            prev_price = prices[i - 1]

            # Check minimum trade size (same as env)
            pos_change = abs(target - position)
            if pos_change < cfg.min_trade_pct:
                target = position  # keep current
                pos_change = 0.0

            # PnL from existing position WITH LEVERAGE
            price_return = (price / prev_price) - 1
            pnl_return = position * price_return * cfg.leverage

            # Transaction cost: taker fee + slippage (SAME formula as env)
            fee_rate = cfg.taker_fee + slippage_rate
            fee_return = pos_change * fee_rate

            pnl = pnl_return - fee_return

            if pos_change >= cfg.min_trade_pct:
                trade = {
                    "step": i,
                    "price": price,
                    "old_pos": position,
                    "new_pos": target,
                    "cost": fee_return * balance,
                }

                close_fee_return = 0.0
                open_fee_return = 0.0

                if position != 0:
                    if target == 0 or np.sign(target) != np.sign(position):
                        close_fee_return = abs(position) * fee_rate
                    elif np.sign(target) == np.sign(position) and abs(target) < abs(position):
                        close_fee_return = abs(position - target) * fee_rate

                if target != 0:
                    if position == 0 or np.sign(target) != np.sign(position):
                        open_fee_return = abs(target) * fee_rate
                    elif np.sign(target) == np.sign(position) and abs(target) > abs(position):
                        open_fee_return = abs(target - position) * fee_rate

                if position != 0 and (target == 0 or np.sign(target) != np.sign(position)):
                    if entry_price > 0:
                        trade_return = position * (price / entry_price - 1) * cfg.leverage
                        trade_return -= entry_fee_return + close_fee_return
                        trade["realized_return"] = trade_return
                        if trade_return > 0:
                            n_wins += 1
                        else:
                            n_losses += 1

                if target != 0 and (position == 0 or np.sign(target) != np.sign(position)):
                    entry_price = price
                    entry_fee_return = open_fee_return
                    if target > 0:
                        n_longs += 1
                    else:
                        n_shorts += 1
                elif target == 0:
                    entry_price = 0.0
                    entry_fee_return = 0.0
                elif np.sign(target) == np.sign(position) and abs(target) > abs(position):
                    entry_fee_return += open_fee_return

                trades.append(trade)

            # Funding rate cost (ONLY at intervals — same as env)
            if cfg.funding_interval > 0 and i % cfg.funding_interval == 0:
                funding_cost = abs(position) * funding_rates[i]
                pnl -= funding_cost

            gross_pnl = pnl

            server_cost_usd_per_step = 0.0
            if cfg.monthly_server_cost_usd > 0 and cfg.periods_per_day > 0:
                server_cost_usd_per_step = cfg.monthly_server_cost_usd / (cfg.periods_per_day * 30.0)
                server_cost_pct = server_cost_usd_per_step / max(balance, 1.0)
                pnl -= server_cost_pct
                server_cost_paid += server_cost_usd_per_step

            gross_balance *= (1 + gross_pnl)
            gross_balance = max(gross_balance, 0)
            gross_equity_curve.append(gross_balance)

            balance *= (1 + pnl)
            position = target
            equity_curve.append(balance)
            if position == 0:
                flat_steps_total += 1
            else:
                position_steps_total += 1

            # Liquidation check (same logic as env)
            if cfg.leverage > 1:
                margin_ratio = balance / cfg.initial_balance
                min_margin = cfg.maintenance_margin * abs(position) * cfg.leverage
                if margin_ratio < min_margin or balance <= 0:
                    logger.warning(f"Liquidated at step {i} (leverage={cfg.leverage}x)")
                    break
            elif balance <= 0:
                logger.warning(f"Liquidated at step {i}")
                break

        result = BacktestResult(
            equity_curve=equity_curve,
            gross_equity_curve=gross_equity_curve,
            trades=trades,
        )
        result.metrics = self._compute_metrics(
            equity_curve,
            gross_equity_curve,
            trades,
            n_longs=n_longs,
            n_shorts=n_shorts,
            n_wins=n_wins,
            n_losses=n_losses,
            server_cost_paid=server_cost_paid,
            flat_steps_total=flat_steps_total,
            position_steps_total=position_steps_total,
        )
        return result

    def _compute_metrics(
        self,
        equity_curve: list[float],
        gross_equity_curve: list[float],
        trades: list[dict],
        n_longs: int = 0,
        n_shorts: int = 0,
        n_wins: int = 0,
        n_losses: int = 0,
        server_cost_paid: float = 0.0,
        flat_steps_total: int = 0,
        position_steps_total: int = 0,
    ) -> dict:
        equity = np.array(equity_curve)
        if len(equity) < 2:
            return {}

        summary = summarize_equity_curve(equity)
        gross_summary = summarize_equity_curve(np.array(gross_equity_curve))
        closed = n_wins + n_losses
        win_rate = float(n_wins / closed) if closed > 0 else 0.0
        total_steps = max(len(equity) - 1, 1)
        flat_ratio = float(flat_steps_total / total_steps)

        if len(trades) == 0 and abs(gross_summary["total_return"]) < 1e-10:
            summary["sharpe"] = 0.0
            summary["sortino"] = 0.0

        return {
            "sharpe": float(summary["sharpe"]),
            "sortino": float(summary["sortino"]),
            "max_drawdown": float(summary["max_drawdown"]),
            "total_return": float(summary["total_return"]),
            "gross_total_return": float(gross_summary["total_return"]),
            "server_cost_paid": float(server_cost_paid),
            "flat_ratio": float(flat_ratio),
            "position_ratio": float(position_steps_total / total_steps),
            "win_rate": float(win_rate),
            "n_trades": len(trades),
            "n_longs": int(n_longs),
            "n_shorts": int(n_shorts),
            "n_wins": int(n_wins),
            "n_losses": int(n_losses),
        }
