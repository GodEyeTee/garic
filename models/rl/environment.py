"""Trading environment for Binance USDT-M futures.

3 actions: Short, Flat, Long

Reward (v2 — simplified for profitability):
- Net PnL from position (fees + funding included) × scale
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
        pnl_reward_scale: float = 200.0,
        opportunity_cost_scale: float = 30.0,
        inactive_episode_penalty: float = 3.0,
        static_position_episode_penalty: float = 1.0,
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
        self.pnl_reward_scale = pnl_reward_scale
        self.opportunity_cost_scale = opportunity_cost_scale
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
            low=-np.inf, high=np.inf, shape=(feature_dim + 4,), dtype=np.float32,
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
        return self._obs(), {}

    @property
    def _idx(self):
        return self._start + self._step

    def _obs(self):
        f = self.features[self._idx].astype(np.float32)
        upnl = self._unrealized_pnl()
        # Extra state helps the policy reason about current exposure and time in/out of market.
        return np.concatenate([f, np.array([
            self.position, upnl, self._flat_steps / 100.0, self._pos_steps / 100.0
        ], dtype=np.float32)])

    def _unrealized_pnl(self):
        if self.position == 0 or self.entry_price == 0:
            return 0.0
        return self.position * (self.prices[self._idx] / self.entry_price - 1) * self.leverage

    def step(self, action):
        if isinstance(action, np.ndarray):
            a = int(action.item()) if action.ndim == 0 else int(action[0])
        else:
            a = int(action)

        idx = self._idx
        old_pos_for_streak = self.position
        prev_close = self.prices[max(idx - 1, self._start)]
        close_p = self.prices[idx]
        high_p = self.highs[idx]
        low_p = self.lows[idx]

        price_ret = (close_p / prev_close - 1) if self._step > 0 else 0.0
        abs_move = abs(price_ret)

        # === Determine new position ===
        if a == ACTION_LONG:
            new_pos = 1.0
        elif a == ACTION_SHORT:
            new_pos = -1.0
        else:  # Flat / close position
            new_pos = 0.0

        # === PnL from current position ===
        pnl = 0.0
        liquidated = False

        if self._step > 0 and self.position != 0:
            pnl = self.position * price_ret * self.leverage

            # Intra-candle liquidation
            if self.leverage > 1 and self.entry_price > 0:
                ld = (1.0 / self.leverage) - self.maintenance_margin
                if self.position > 0 and low_p <= self.entry_price * (1 - ld):
                    pnl = -ld * self.leverage
                    liquidated = True
                elif self.position < 0 and high_p >= self.entry_price * (1 + ld):
                    pnl = -ld * self.leverage
                    liquidated = True

            if self.funding_interval > 0 and self._step % self.funding_interval == 0:
                pnl -= abs(self.position) * abs(self.funding_rates[idx])

        # === Trade execution ===
        entered = False
        exited = False

        if liquidated:
            self._losses += 1
            self._trades += 1
            self.position = 0.0
            self.entry_price = 0.0
            exited = True

        if not liquidated:
            old_pos = self.position

            # Entering from flat
            if old_pos == 0 and new_pos != 0:
                pnl -= self.fee_rate  # entry fee
                self._trades += 1
                if new_pos > 0:
                    self._longs += 1
                else:
                    self._shorts += 1
                self.entry_price = close_p
                entered = True
                self._flat_steps = 0

            # Reversing direction
            elif old_pos != 0 and new_pos != 0 and old_pos != new_pos:
                # Close old + open new
                trade_ret = old_pos * (close_p / self.entry_price - 1) * self.leverage - (2.0 * self.fee_rate)
                if trade_ret > 0:
                    self._wins += 1
                else:
                    self._losses += 1
                pnl -= 2.0 * self.fee_rate  # close + open
                self._trades += 1
                if new_pos > 0:
                    self._longs += 1
                else:
                    self._shorts += 1
                self.entry_price = close_p
                entered = True
                closed_trade_ret = trade_ret

            # Exiting to flat 
            elif old_pos != 0 and new_pos == 0:
                # Close old
                trade_ret = old_pos * (close_p / self.entry_price - 1) * self.leverage - (2.0 * self.fee_rate)
                if trade_ret > 0:
                    self._wins += 1
                else:
                    self._losses += 1
                pnl -= self.fee_rate  # close fee
                self._trades += 1
                self.entry_price = 0.0
                exited = True
                closed_trade_ret = trade_ret

            self.position = new_pos

        if self.position != 0:
            self._pos_steps += 1
            self._pos_steps_total += 1
            if old_pos_for_streak != 0 and self.position == old_pos_for_streak:
                self._direction_steps += 1
            else:
                self._direction_steps = 1
        else:
            self._pos_steps = 0
            self._flat_steps_total += 1
            self._direction_steps = 0

        # === REWARD (v2 — clean PnL + opportunity cost) ===
        gross_pnl = pnl  # trading PnL: position PnL - fees - funding

        # Server cost: track for balance but NOT in reward (sunk cost)
        server_cost_usd_per_step = 0.0
        server_cost_pct = 0.0
        if self.monthly_server_cost_usd > 0 and self.periods_per_day > 0:
            server_cost_usd_per_step = self.monthly_server_cost_usd / (self.periods_per_day * 30.0)
            server_cost_pct = server_cost_usd_per_step / max(self.balance, 1.0)
            self._server_cost_paid += server_cost_usd_per_step

        # Core reward: net PnL from trading decisions only
        reward = pnl * self.pnl_reward_scale

        # Opportunity cost: penalize flat proportional to actual market movement
        # Creates correct incentive ordering: right position > flat > wrong position
        if self.position == 0:
            if self._step > 0:
                self._flat_steps += 1
                self._missed_moves += abs_move
                reward -= abs_move * self.opportunity_cost_scale
        else:
            self._flat_steps = 0
            # Track wrong-side moves for metrics (PnL already penalizes this)
            if self._step > 0:
                if (self.position > 0 and price_ret < 0) or (self.position < 0 and price_ret > 0):
                    self._wrong_side_moves += abs_move

        # Apply server cost to pnl for balance tracking (after reward calculation)
        pnl -= server_cost_pct

        # Update balances
        self.gross_balance *= (1 + gross_pnl)
        self.gross_balance = max(self.gross_balance, 0)
        self.gross_equity_curve.append(self.gross_balance)

        self.balance *= (1 + pnl)
        self.balance = max(self.balance, 0)
        self.equity_curve.append(self.balance)

        self._step += 1
        done = self._step >= self.max_episode_steps or self._idx >= self.segment_end - 1
        trunc = self.balance <= 0
        if trunc:
            reward = -10.0

        # Evaluate final open position for accurate win/loss stats at episode end
        if (done or trunc) and self.position != 0:
            trade_ret = self.position * (close_p / self.entry_price - 1) * self.leverage - self.fee_rate
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

        obs = self._obs() if not (done or trunc) else np.zeros(self.observation_space.shape, dtype=np.float32)
        
        info = {
            "balance": self.balance, "position": self.position,
            "pnl": pnl, "upnl": self._unrealized_pnl(), "n_trades": self._trades,
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
