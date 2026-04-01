"""Trading environment for Binance USDT-M futures.

3 actions: Short, Flat, Long

Reward (v2 โ€” simplified for profitability):
- Net PnL from position (fees + funding included) ร— scale
- Continuous opportunity cost when flat (proportional to market move)
- No double-counting, no alpha benchmark, no server cost in reward
- Server cost tracked in balance only (sunk cost the agent can't control)

OHLCV path: Open -> High/Low (liquidation) -> Close
"""

import logging
import numpy as np

from performance import summarize_equity_curve

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

logger = logging.getLogger(__name__)

ACTION_SHORT = 0
ACTION_FLAT = 1
ACTION_LONG = 2
AGENT_STATE_DIM = 8
STATE_IDX_POSITION = 0
STATE_IDX_UPNL = 1
STATE_IDX_EQUITY_RATIO = 2
STATE_IDX_DRAWDOWN = 3
STATE_IDX_ROLLING_VOL = 4
STATE_IDX_TURNOVER = 5
STATE_IDX_FLAT_STEPS = 6
STATE_IDX_POS_STEPS = 7


def build_agent_state(
    *,
    position: float,
    upnl: float,
    equity_ratio: float,
    drawdown: float,
    rolling_volatility: float,
    turnover_last_step: float,
    flat_steps: int,
    pos_steps: int,
) -> np.ndarray:
    return np.array(
        [
            float(np.clip(position, -1.0, 1.0)),
            float(upnl),
            float(equity_ratio),
            float(max(drawdown, 0.0)),
            float(max(rolling_volatility, 0.0)),
            float(max(turnover_last_step, 0.0)),
            float(max(flat_steps, 0)) / 100.0,
            float(max(pos_steps, 0)) / 100.0,
        ],
        dtype=np.float32,
    )


class CryptoFuturesEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        feature_arrays: np.ndarray,
        price_series: np.ndarray,
        ohlcv_data: np.ndarray | None = None,
        initial_balance: float = 10000.0,
        taker_fee: float = 0.0005,
        slippage_bps: float = 1.0,
        funding_rates: np.ndarray | None = None,
        funding_interval: int = 32,
        leverage: float = 1.0,
        maintenance_margin: float = 0.005,
        max_episode_steps: int = 2000,
        monthly_server_cost_usd: float = 100.0,
        periods_per_day: int = 96,
        pnl_reward_scale: float = 100.0,
        drawdown_penalty_scale: float = 2.0,
        turnover_penalty_scale: float = 0.05,
        inactive_episode_penalty: float = 0.0,
        static_position_episode_penalty: float = 0.0,
        balanced_sampling: bool = False,
        regime_label_threshold: float = 0.02,
        segment_start: int = 0,
        segment_end: int | None = None,
        **kwargs,
    ):
        super().__init__()
        n = len(feature_arrays)
        self.features = feature_arrays
        self.prices = price_series.astype(np.float64)
        self.initial_balance = initial_balance
        self.fee_rate = taker_fee + slippage_bps / 10000
        self.funding_rates = funding_rates if funding_rates is not None else np.zeros(n)
        self.funding_interval = funding_interval
        self.leverage = leverage
        self.maintenance_margin = maintenance_margin
        self.max_episode_steps = max_episode_steps
        self.monthly_server_cost_usd = monthly_server_cost_usd
        self.periods_per_day = periods_per_day
        self.pnl_reward_scale = max(float(pnl_reward_scale), 0.0)
        self.drawdown_penalty_scale = max(float(drawdown_penalty_scale), 0.0)
        self.turnover_penalty_scale = max(float(turnover_penalty_scale), 0.0)
        self.inactive_episode_penalty = max(float(inactive_episode_penalty), 0.0)
        self.static_position_episode_penalty = max(float(static_position_episode_penalty), 0.0)
        self.balanced_sampling = bool(balanced_sampling)
        self.regime_label_threshold = max(float(regime_label_threshold), 0.0)
        self.total_len = n
        self.segment_start = max(0, int(segment_start))
        resolved_segment_end = n if segment_end is None else int(segment_end)
        self.segment_end = min(max(self.segment_start + 1, resolved_segment_end), n)

        if ohlcv_data is not None and ohlcv_data.shape[1] >= 4:
            self.highs = ohlcv_data[:, 1].astype(np.float64)
            self.lows = ohlcv_data[:, 2].astype(np.float64)
        else:
            self.highs = self.lows = self.prices

        feature_dim = feature_arrays.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(feature_dim + AGENT_STATE_DIM,), dtype=np.float32,
        )
        # 3 actions: Short, Flat, Long
        self.action_space = spaces.Discrete(3)
        self._sampling_cursor = 0
        self._sampled_regime = "uniform"
        self._sampling_buckets = self._build_sampling_buckets()
        self.reset()

    def _build_sampling_buckets(self) -> dict[str, np.ndarray]:
        latest_start = max(self.segment_start, self.segment_end - self.max_episode_steps - 1)
        if latest_start <= self.segment_start:
            candidate_starts = np.array([self.segment_start], dtype=np.int32)
        else:
            candidate_starts = np.arange(self.segment_start, latest_start + 1, dtype=np.int32)

        if not self.balanced_sampling or candidate_starts.size == 0:
            self.sampling_stats = {
                "mode": "uniform",
                "threshold": float(self.regime_label_threshold),
                "total_candidates": int(candidate_starts.size),
                "up_count": 0,
                "down_count": 0,
                "flat_count": 0,
            }
            return {}

        horizon_end = np.minimum(candidate_starts + self.max_episode_steps, self.segment_end - 1)
        episode_returns = (self.prices[horizon_end] / self.prices[candidate_starts]) - 1.0
        threshold = self.regime_label_threshold
        up_mask = episode_returns > threshold
        down_mask = episode_returns < -threshold
        flat_mask = ~(up_mask | down_mask)
        buckets = {
            "down": candidate_starts[down_mask],
            "flat": candidate_starts[flat_mask],
            "up": candidate_starts[up_mask],
        }
        usable_buckets = {name: starts for name, starts in buckets.items() if starts.size > 0}
        self.sampling_stats = {
            "mode": "balanced" if usable_buckets else "uniform",
            "threshold": float(threshold),
            "total_candidates": int(candidate_starts.size),
            "up_count": int(buckets["up"].size),
            "down_count": int(buckets["down"].size),
            "flat_count": int(buckets["flat"].size),
        }
        if len(usable_buckets) >= 2:
            return usable_buckets
        self.sampling_stats["mode"] = "uniform"
        return {}

    def _sample_start_index(self) -> int:
        latest_start = max(self.segment_start, self.segment_end - self.max_episode_steps - 1)
        if latest_start <= self.segment_start:
            self._sampled_regime = "uniform"
            return self.segment_start
        if self._sampling_buckets:
            ordered = [name for name in ("down", "flat", "up") if name in self._sampling_buckets]
            regime_name = ordered[self._sampling_cursor % len(ordered)]
            self._sampling_cursor += 1
            starts = self._sampling_buckets[regime_name]
            sampled = int(starts[self.np_random.integers(0, len(starts))])
            self._sampled_regime = regime_name
            return sampled
        self._sampled_regime = "uniform"
        return int(self.np_random.integers(self.segment_start, latest_start + 1))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._start = self._sample_start_index()
        self._step = 0
        self.position = 0.0  # start flat
        self.entry_price = 0.0
        self.balance = self.initial_balance
        self.gross_balance = self.initial_balance
        self.equity_curve = [self.initial_balance]
        self.gross_equity_curve = [self.initial_balance]
        self._trades = 0
        self._longs = 0
        self._shorts = 0
        self._wins = 0
        self._losses = 0
        self._flat_steps = 0
        self._pos_steps = 0
        self._flat_steps_total = 0
        self._pos_steps_total = 0
        self._missed_moves = 0.0  # cumulative missed opportunity
        self._wrong_side_moves = 0.0
        self._server_cost_paid = 0.0
        self._direction_steps = 0
        self._last_turnover = 0.0
        self._peak_balance = self.initial_balance
        self._prev_drawdown = 0.0
        return self._obs(), {}

    @property
    def _idx(self):
        return self._start + self._step

    def _obs(self):
        f = self.features[self._idx].astype(np.float32)
        upnl = self._unrealized_pnl()
        return np.concatenate(
            [
                f,
                build_agent_state(
                    position=self.position,
                    upnl=upnl,
                    equity_ratio=(self.balance / max(self.initial_balance, 1e-9)) - 1.0,
                    drawdown=self._current_drawdown(),
                    rolling_volatility=self._rolling_volatility(),
                    turnover_last_step=self._last_turnover,
                    flat_steps=self._flat_steps,
                    pos_steps=self._pos_steps,
                ),
            ]
        )

    def _unrealized_pnl(self):
        if self.position == 0 or self.entry_price == 0:
            return 0.0
        return self.position * (self.prices[self._idx] / self.entry_price - 1) * self.leverage

    def _current_drawdown(self) -> float:
        peak = max(self._peak_balance, self.balance, 1e-9)
        return max((peak - self.balance) / peak, 0.0)

    def _rolling_volatility(self, window: int = 32) -> float:
        idx = self._idx
        start = max(self._start, idx - max(int(window), 1))
        if idx <= start:
            return 0.0
        prices = np.maximum(self.prices[start: idx + 1], 1e-9)
        if prices.size < 2:
            return 0.0
        returns = np.diff(np.log(prices))
        if returns.size == 0:
            return 0.0
        return float(np.std(returns))

    def step(self, action):
        if isinstance(action, np.ndarray):
            a = int(action.item()) if action.ndim == 0 else int(action[0])
        else:
            a = int(action)

        idx = self._idx
        old_balance = float(self.balance)
        old_pos = float(self.position)
        prev_close = self.prices[max(idx - 1, self._start)]
        close_p = self.prices[idx]
        high_p = self.highs[idx]
        low_p = self.lows[idx]

        price_ret = (close_p / prev_close - 1.0) if self._step > 0 else 0.0
        abs_move = abs(price_ret)

        if a == ACTION_LONG:
            requested_pos = 1.0
        elif a == ACTION_SHORT:
            requested_pos = -1.0
        else:
            requested_pos = 0.0

        gross_pnl = 0.0
        funding_cost = 0.0
        liquidation_cost = 0.0
        liquidated = False

        if self._step > 0 and old_pos != 0.0:
            gross_pnl = old_pos * price_ret * self.leverage

            if self.funding_interval > 0 and self._step % self.funding_interval == 0:
                funding_cost = abs(old_pos) * abs(self.funding_rates[idx])

            if self.leverage > 1 and self.entry_price > 0:
                ld = (1.0 / self.leverage) - self.maintenance_margin
                if old_pos > 0 and low_p <= self.entry_price * (1.0 - ld):
                    gross_pnl = -ld * self.leverage
                    liquidated = True
                elif old_pos < 0 and high_p >= self.entry_price * (1.0 + ld):
                    gross_pnl = -ld * self.leverage
                    liquidated = True

        actual_new_pos = 0.0 if liquidated else requested_pos
        turnover = abs(actual_new_pos - old_pos)
        trading_cost = turnover * self.fee_rate
        self._last_turnover = float(turnover)

        if self._step > 0 and old_pos == 0.0:
            self._missed_moves += abs_move
        elif self._step > 0 and old_pos != 0.0:
            if (old_pos > 0 and price_ret < 0) or (old_pos < 0 and price_ret > 0):
                self._wrong_side_moves += abs_move

        if old_pos == 0.0 and actual_new_pos != 0.0:
            self._trades += 1
            if actual_new_pos > 0:
                self._longs += 1
            else:
                self._shorts += 1
        elif old_pos != 0.0 and actual_new_pos != old_pos:
            trade_ret = old_pos * (close_p / self.entry_price - 1.0) * self.leverage - self.fee_rate - funding_cost
            if liquidated:
                trade_ret = gross_pnl - self.fee_rate - funding_cost
            if trade_ret > 0:
                self._wins += 1
            else:
                self._losses += 1
            self._trades += 1
            if actual_new_pos != 0.0:
                if actual_new_pos > 0:
                    self._longs += 1
                else:
                    self._shorts += 1

        if actual_new_pos != 0.0 and actual_new_pos != old_pos:
            self.entry_price = close_p
        elif actual_new_pos == 0.0:
            self.entry_price = 0.0

        gross_step_return = gross_pnl - trading_cost - funding_cost - liquidation_cost

        server_cost_usd_per_step = 0.0
        server_cost_pct = 0.0
        if self.monthly_server_cost_usd > 0 and self.periods_per_day > 0 and old_balance > 0:
            server_cost_usd_per_step = self.monthly_server_cost_usd / (self.periods_per_day * 30.0)
            server_cost_pct = server_cost_usd_per_step / max(old_balance, 1.0)
            self._server_cost_paid += server_cost_usd_per_step

        net_step_return = gross_step_return - server_cost_pct
        self.gross_balance *= (1.0 + gross_step_return)
        self.gross_balance = max(self.gross_balance, 0.0)
        self.gross_equity_curve.append(self.gross_balance)

        self.balance *= (1.0 + net_step_return)
        self.balance = max(self.balance, 0.0)
        self.equity_curve.append(self.balance)
        self._peak_balance = max(self._peak_balance, self.balance)

        drawdown = self._current_drawdown()
        drawdown_increase = max(0.0, drawdown - self._prev_drawdown)
        equity_ret = (self.balance - old_balance) / max(old_balance, 1e-9)
        reward = (
            self.pnl_reward_scale * equity_ret
            - self.drawdown_penalty_scale * drawdown_increase
            - self.turnover_penalty_scale * turnover
        )
        reward = float(np.clip(reward, -5.0, 5.0))
        self._prev_drawdown = drawdown

        self.position = actual_new_pos
        if self.position != 0.0:
            self._pos_steps += 1
            self._pos_steps_total += 1
            if old_pos != 0.0 and self.position == old_pos:
                self._direction_steps += 1
            else:
                self._direction_steps = 1
            self._flat_steps = 0
        else:
            self._pos_steps = 0
            self._flat_steps += 1
            self._flat_steps_total += 1
            self._direction_steps = 0

        self._step += 1
        done = self._step >= self.max_episode_steps or self._idx >= self.segment_end - 1
        trunc = self.balance <= 0.0
        if trunc:
            reward = -5.0

        if (done or trunc) and self.position != 0.0:
            trade_ret = self.position * (close_p / self.entry_price - 1.0) * self.leverage - self.fee_rate
            if trade_ret > 0:
                self._wins += 1
            else:
                self._losses += 1

        if done or trunc:
            total_steps = max(self._flat_steps_total + self._pos_steps_total, 1)
            position_ratio = self._pos_steps_total / total_steps
            if self._trades == 0 and self.inactive_episode_penalty > 0:
                reward -= self.inactive_episode_penalty
            elif self._trades <= 1 and position_ratio >= 0.98 and self.static_position_episode_penalty > 0:
                reward -= self.static_position_episode_penalty
            reward = float(np.clip(reward, -5.0, 5.0))

        obs = self._obs() if not (done or trunc) else np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {
            "balance": self.balance,
            "position": self.position,
            "pnl": net_step_return,
            "gross_pnl": gross_step_return,
            "upnl": self._unrealized_pnl(),
            "n_trades": self._trades,
            "drawdown": drawdown,
            "turnover": turnover,
        }
        return obs, float(reward), done, trunc, info

    def get_metrics(self):
        eq = np.array(self.equity_curve)
        gross_eq = np.array(self.gross_equity_curve)
        ncl = self._wins + self._losses
        summary = summarize_equity_curve(eq)
        gross_summary = summarize_equity_curve(gross_eq)
        total_steps = max(len(eq) - 1, 1)

        if self._trades == 0 and abs(gross_summary["total_return"]) < 1e-10:
            summary["sharpe"] = 0.0
            summary["sortino"] = 0.0

        return {
            "sharpe": float(summary["sharpe"]),
            "sortino": float(summary["sortino"]),
            "max_drawdown": float(summary["max_drawdown"]),
            "total_return": float(summary["total_return"]),
            "gross_total_return": float(gross_summary["total_return"]),
            "flat_ratio": float(self._flat_steps_total / total_steps),
            "position_ratio": float(self._pos_steps_total / total_steps),
            "n_trades": self._trades, "n_longs": self._longs, "n_shorts": self._shorts,
            "n_wins": self._wins, "n_losses": self._losses,
            "win_rate": float(self._wins / ncl) if ncl > 0 else 0,
            "missed_moves": float(self._missed_moves),
            "wrong_side_moves": float(self._wrong_side_moves),
            "server_cost_paid": float(self._server_cost_paid),
        }
