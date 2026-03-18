"""RL Trainer — PPO Discrete + PnL reward + multi-episode eval.

Key fixes:
- n_steps = max_episode_steps (PPO เห็นจบ episode ทุก rollout)
- ent_coef = 0.05 (ป้องกัน entropy collapse อย่างเด็ดขาด)
- Multi-episode eval (ไม่ใช่แค่ 1 episode)
- RL vs Buy&Hold comparison chart
"""

import logging
import time
from pathlib import Path

import numpy as np

from performance import format_drawdown_pct

logger = logging.getLogger(__name__)


class RLTrainer:
    def __init__(
        self,
        feature_arrays: np.ndarray,
        price_series: np.ndarray,
        ohlcv_data: np.ndarray | None = None,
        funding_rates: np.ndarray | None = None,
        algo: str = "PPO",
        checkpoint_dir: str = "checkpoints",
        leverage: float = 1.0,
        taker_fee: float = 0.0005,
        slippage_bps: float = 1.0,
        funding_interval: int = 32,
        maintenance_margin: float = 0.005,
        min_trade_pct: float = 0.05,
        max_episode_steps: int = 2000,
        monthly_server_cost_usd: float = 100.0,
        periods_per_day: int = 96,
        train_range: tuple[int, int] | None = None,
        eval_range: tuple[int, int] | None = None,
        test_range: tuple[int, int] | None = None,
        **kwargs,
    ):
        self.feature_arrays = feature_arrays
        self.price_series = price_series
        self.ohlcv_data = ohlcv_data
        self.funding_rates = funding_rates
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.leverage = leverage
        self.taker_fee = taker_fee
        self.slippage_bps = slippage_bps
        self.funding_interval = funding_interval
        self.maintenance_margin = maintenance_margin
        self.min_trade_pct = min_trade_pct
        self.max_episode_steps = max_episode_steps
        self.monthly_server_cost_usd = monthly_server_cost_usd
        self.periods_per_day = periods_per_day
        self.env_kwargs = dict(kwargs)
        self.total_len = len(self.price_series)
        self.train_range = self._normalize_range(train_range)
        self.eval_range = self._normalize_range(eval_range or self.train_range)
        self.test_range = self._normalize_range(test_range or self.eval_range)

    def _normalize_range(self, segment_range: tuple[int, int] | None) -> tuple[int, int]:
        if self.total_len <= 0:
            return (0, 0)
        if segment_range is None:
            return (0, self.total_len)
        start, end = int(segment_range[0]), int(segment_range[1])
        start = max(0, min(start, self.total_len - 1))
        end = max(start + 1, min(end, self.total_len))
        return (start, end)

    def create_env(self, feature_arrays=None, price_series=None, ohlcv_data=None, segment_range=None):
        from models.rl.environment import CryptoFuturesEnv
        segment_start, segment_end = self._normalize_range(segment_range)
        return CryptoFuturesEnv(
            feature_arrays=feature_arrays if feature_arrays is not None else self.feature_arrays,
            price_series=price_series if price_series is not None else self.price_series,
            ohlcv_data=ohlcv_data if ohlcv_data is not None else self.ohlcv_data,
            funding_rates=self.funding_rates,
            leverage=self.leverage, taker_fee=self.taker_fee,
            slippage_bps=self.slippage_bps,
            funding_interval=self.funding_interval,
            maintenance_margin=self.maintenance_margin,
            min_trade_pct=self.min_trade_pct,
            max_episode_steps=self.max_episode_steps,
            monthly_server_cost_usd=self.monthly_server_cost_usd,
            periods_per_day=self.periods_per_day,
            segment_start=segment_start,
            segment_end=segment_end,
            **self.env_kwargs,
        )

    def train(
        self,
        total_timesteps: int = 200_000,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        ent_coef: float = 0.02,
        eval_every_steps: int = 50_000,
        eval_episodes: int = 4,
        **kwargs,
    ) -> dict:
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.callbacks import CheckpointCallback
        except ImportError:
            return {"error": "stable-baselines3 not installed"}

        env = self.create_env(segment_range=self.train_range)
        n_feat = env.observation_space.shape[0]

        # *** PPO config ที่ทำงานจริง ***
        # n_steps = full episode → PPO เห็น outcome ของทุก action
        actual_n_steps = min(n_steps, self.max_episode_steps)

        # batch_size ต้องหาร n_steps ลงตัว ไม่งั้น SB3 warning
        actual_batch = batch_size
        while actual_n_steps % actual_batch != 0 and actual_batch > 1:
            actual_batch -= 1

        import warnings
        warnings.filterwarnings("ignore", message=".*mini-batch.*")
        warnings.filterwarnings("ignore", message=".*GPU.*MlpPolicy.*")

        model = PPO(
            "MlpPolicy", env,
            learning_rate=learning_rate,
            n_steps=actual_n_steps,
            batch_size=actual_batch,
            n_epochs=n_epochs,
            ent_coef=ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            device="auto",  # Changed to use GPU if available
            policy_kwargs={"net_arch": dict(pi=[256, 256], vf=[256, 256])},
            verbose=0,  # *** ปิด SB3 print — ใช้ dashboard แทน ***
        )

        logger.info(f"PPO: features={n_feat}, ent_coef={ent_coef}, n_steps={actual_n_steps}, "
                     f"ep_len={self.max_episode_steps}, total={total_timesteps}")
        logger.info(
            "Training config: lr=%.6f batch=%d epochs=%d leverage=%.2f min_trade_pct=%.4f "
            "server_cost=$%.2f/mo periods_per_day=%d",
            learning_rate,
            actual_batch,
            n_epochs,
            self.leverage,
            self.min_trade_pct,
            self.monthly_server_cost_usd,
            self.periods_per_day,
        )
        logger.info(
            "Reward shaping: opportunity_threshold=%.6f missed_move_penalty_scale=%.2f "
            "server_cost_reward_multiplier=%.2f flat_penalty_after_steps=%d flat_penalty_scale=%.4f",
            float(self.env_kwargs.get("opportunity_threshold", 0.0)),
            float(self.env_kwargs.get("missed_move_penalty_scale", 0.0)),
            float(self.env_kwargs.get("server_cost_reward_multiplier", 0.0)),
            int(self.env_kwargs.get("flat_penalty_after_steps", 0)),
            float(self.env_kwargs.get("flat_penalty_scale", 0.0)),
        )
        logger.info(
            "Dataset ranges: train=[%d:%d) validation=[%d:%d) test=[%d:%d)",
            self.train_range[0],
            self.train_range[1],
            self.eval_range[0],
            self.eval_range[1],
            self.test_range[0],
            self.test_range[1],
        )

        # Custom callback: update dashboard + checkpoint
        from stable_baselines3.common.callbacks import BaseCallback
        trainer_self = self

        class DashboardCallback(BaseCallback):
            """Update dashboard during training (no stdout print)."""
            def __init__(self, dash=None, save_path="checkpoints", save_freq=10000, eval_freq=50_000, eval_eps=4):
                super().__init__(verbose=0)
                self.dash = dash
                self.save_path = Path(save_path)
                self.save_freq = save_freq
                self.eval_freq = max(int(eval_freq), 1)
                self.eval_eps = max(int(eval_eps), 1)
                self._start_time = time.time()
                self._last_log_step = 0
                self._last_ui_refresh = 0.0
                self._last_eval_step = 0
                self.best_score = -np.inf
                self.best_path = None

            def _on_step(self) -> bool:
                # Update more frequently to catch intra-episode stats before reset!
                now = time.time()
                should_refresh_ui = (
                    self.dash
                    and (
                        self.num_timesteps == 1
                        or now - self._last_ui_refresh >= 0.4
                        or self.num_timesteps >= self.locals.get("total_timesteps", 0)
                    )
                )
                if should_refresh_ui:
                    elapsed = time.time() - self._start_time
                    fps = self.num_timesteps / max(elapsed, 1)
                    # Get training info from logger
                    ent = 0
                    loss_val = 0
                    if hasattr(self.model, 'logger') and self.model.logger is not None:
                        try:
                            kvs = self.model.logger.name_to_value
                            ent = abs(kvs.get("train/entropy_loss", 0))
                            loss_val = kvs.get("train/loss", 0)
                        except Exception:
                            pass
                    pos = 0.0
                    upnl = 0.0
                    action_str = "None"
                    
                    try:
                        actions = self.locals.get("actions")
                        if actions is not None and len(actions) > 0:
                            a = int(actions[0])
                            action_str = "Short" if a == 0 else "Long" if a == 2 else "Flat"
                    except:
                        pass
                        
                    n_tr, n_l, n_s, n_w, n_lo, n_wr = 0,0,0,0,0,0.0
                    try:
                        infos = self.locals.get("infos", [{}])
                        if infos and len(infos) > 0:
                            info = infos[0]
                            pos = info.get("position", 0.0)
                            upnl = info.get("upnl", 0.0)
                            n_tr = info.get("n_trades", 0)
                            
                        if hasattr(self.model, 'env') and self.model.env is not None:
                            eval_env = self.model.env.envs[0].unwrapped
                            m = eval_env.get_metrics()
                            n_tr, n_l, n_s, n_w, n_lo, n_wr = m.get("n_trades", 0), m.get("n_longs", 0), m.get("n_shorts", 0), m.get("n_wins", 0), m.get("n_losses", 0), m.get("win_rate", 0.0)
                    except Exception as e:
                        open("dash_err.log", "a").write(f"Dash callback error: {e}\n")

                    self.dash.update(
                        train_step=self.num_timesteps,
                        train_fps=fps,
                        train_elapsed=elapsed,
                        entropy=ent,
                        loss=loss_val,
                        status_msg=f"Training... {self.num_timesteps:,}/{self.locals.get('total_timesteps', 0):,}",
                        current_pos=pos,
                        current_pnl=upnl,
                        current_action=action_str,
                        n_trades=n_tr, n_longs=n_l, n_shorts=n_s, 
                        n_wins=n_w, n_losses=n_lo, win_rate=n_wr,
                    )
                    self._last_ui_refresh = now
                    if self.num_timesteps - self._last_log_step >= 5000:
                        logger.info(
                            "Train step=%d/%d fps=%.0f loss=%.6f entropy=%.6f action=%s pos=%.2f "
                            "upnl=%.5f trades=%d win_rate=%.2f%%",
                            self.num_timesteps,
                            self.locals.get("total_timesteps", 0),
                            fps,
                            float(loss_val),
                            float(ent),
                            action_str,
                            float(pos),
                            float(upnl),
                            int(n_tr),
                            float(n_wr * 100.0),
                        )
                        self._last_log_step = self.num_timesteps
                # Checkpoint
                if self.num_timesteps % self.save_freq == 0:
                    p = self.save_path / f"rl_agent_{self.num_timesteps}.zip"
                    self.model.save(str(p))

                should_eval = (
                    self.num_timesteps >= self.eval_freq
                    and (
                        self.num_timesteps - self._last_eval_step >= self.eval_freq
                        or self.num_timesteps >= total_timesteps
                    )
                )
                if should_eval:
                    metrics = trainer_self._eval_multi_episode(
                        self.model,
                        n_episodes=self.eval_eps,
                        segment_range=trainer_self.eval_range,
                        log_episodes=False,
                    )
                    score = trainer_self._selection_score(metrics)
                    logger.info(
                        "Checkpoint eval step=%d score=%.4f alpha=%.4f net=%.4f gross=%.4f "
                        "trades=%.1f flat_ratio=%.2f%%",
                        self.num_timesteps,
                        float(score),
                        float(metrics.get("outperformance_vs_bh", 0.0)),
                        float(metrics.get("total_return", 0.0)),
                        float(metrics.get("gross_total_return", 0.0)),
                        float(metrics.get("n_trades", 0.0)),
                        float(metrics.get("flat_ratio", 0.0) * 100.0),
                    )
                    if score > self.best_score:
                        self.best_score = score
                        self.best_path = self.save_path / "rl_agent_best"
                        self.model.save(str(self.best_path))
                        logger.info(
                            "New best checkpoint at step=%d -> %s (score=%.4f)",
                            self.num_timesteps,
                            self.best_path,
                            float(score),
                        )
                    self._last_eval_step = self.num_timesteps
                return True

        # Get dashboard from pipeline (passed via _dashboard key in self)
        dash = getattr(self, '_dashboard', None)
        cb = DashboardCallback(dash=dash, save_path=str(self.checkpoint_dir),
                               save_freq=max(total_timesteps // 5, 2000),
                               eval_freq=eval_every_steps,
                               eval_eps=eval_episodes)
        model.learn(total_timesteps=total_timesteps, callback=cb)

        if cb.best_path is not None and cb.best_path.exists():
            logger.info("Loading best checkpoint -> %s", cb.best_path)
            model = PPO.load(str(cb.best_path), env=env)

        final_path = self.checkpoint_dir / "rl_agent_final"
        model.save(str(final_path))
        logger.info(f"Saved model -> {final_path}")

        # === Multi-episode eval ===
        metrics = self._eval_multi_episode(model, n_episodes=10, segment_range=self.test_range)
        logger.info(f"Eval ({10} episodes avg): {metrics}")

        # === Save chart ===
        self._save_chart(model, metrics)

        return metrics

    def _selection_score(self, metrics: dict) -> float:
        """Prefer alpha, but penalize policies that collapse to flat."""
        alpha = float(metrics.get("outperformance_vs_bh", 0.0))
        flat_ratio = float(metrics.get("flat_ratio", 1.0))
        avg_trades = float(metrics.get("avg_trades_per_episode", 0.0))
        gross_return = float(metrics.get("gross_total_return", 0.0))
        return alpha + (gross_return * 0.5) + min(avg_trades, 8.0) * 0.01 - flat_ratio * 0.25

    def _eval_multi_episode(
        self,
        model,
        n_episodes: int = 10,
        segment_range: tuple[int, int] | None = None,
        log_episodes: bool = True,
    ) -> dict:
        """Run multiple episodes, aggregate metrics."""
        all_metrics = []
        total_equity = []
        bh_eval_returns = []
        server_costs = []
        missed_moves = []
        reward_sums = []
        total_action_counts = {0: 0, 1: 0, 2: 0}
        active_range = self._normalize_range(segment_range or self.test_range)

        for ep in range(n_episodes):
            env = self.create_env(segment_range=active_range)
            obs, _ = env.reset()
            action_counts = {0: 0, 1: 0, 2: 0}
            rewards = []
            while True:
                action, _ = model.predict(obs, deterministic=True)
                action_int = int(action.item()) if isinstance(action, np.ndarray) else int(action)
                action_counts[action_int] = action_counts.get(action_int, 0) + 1
                obs, r, t, tr, info = env.step(action)
                rewards.append(float(r))
                if t or tr:
                    break
            m = env.get_metrics()
            all_metrics.append(m)
            total_equity.extend(env.equity_curve)
            prices_slice = self.price_series[env._start: env._start + len(env.equity_curve) - 1]
            if len(prices_slice) >= 2:
                bh_eval_returns.append(float(prices_slice[-1] / prices_slice[0] - 1.0))
            else:
                bh_eval_returns.append(0.0)
            server_costs.append(float(m.get("server_cost_paid", 0.0)))
            missed_moves.append(float(m.get("missed_moves", 0.0)))
            reward_sums.append(float(np.sum(rewards)))
            for key, value in action_counts.items():
                total_action_counts[key] = total_action_counts.get(key, 0) + int(value)
            if log_episodes:
                logger.info(
                    "Eval episode %d/%d: start=%d steps=%d net=%.4f gross=%.4f dd=%.4f trades=%d "
                    "win_rate=%.2f%% flat_ratio=%.2f%% pos_ratio=%.2f%% server_cost=$%.2f "
                    "missed_moves=%.4f actions={short:%d, flat:%d, long:%d} reward_sum=%.4f",
                    ep + 1,
                    n_episodes,
                    env._start,
                    len(env.equity_curve) - 1,
                    m.get("total_return", 0.0),
                    m.get("gross_total_return", 0.0),
                    m.get("max_drawdown", 0.0),
                    int(m.get("n_trades", 0)),
                    float(m.get("win_rate", 0.0) * 100.0),
                    float(m.get("flat_ratio", 0.0) * 100.0),
                    float(m.get("position_ratio", 0.0) * 100.0),
                    float(m.get("server_cost_paid", 0.0)),
                    float(m.get("missed_moves", 0.0)),
                    int(action_counts.get(0, 0)),
                    int(action_counts.get(1, 0)),
                    int(action_counts.get(2, 0)),
                    float(np.sum(rewards)),
                )

        # Average metrics and sum trade counts
        avg = {}
        for key in all_metrics[0]:
            vals = [m[key] for m in all_metrics]
            if isinstance(vals[0], (int, float)):
                if key.startswith("n_"):
                    avg[key] = float(np.sum(vals))
                else:
                    avg[key] = float(np.mean(vals))

        # Recalculate Win Rate from the new totals
        ncl = avg.get("n_wins", 0) + avg.get("n_losses", 0)
        avg["win_rate"] = float(avg["n_wins"] / ncl) if ncl > 0 else 0.0
        avg["avg_trades_per_episode"] = float(avg.get("n_trades", 0.0) / max(n_episodes, 1))
        avg["bh_eval_return"] = float(np.mean(bh_eval_returns)) if bh_eval_returns else 0.0
        avg["server_cost_paid"] = float(np.mean(server_costs)) if server_costs else 0.0
        avg["total_server_cost_paid"] = float(np.sum(server_costs)) if server_costs else 0.0
        avg["missed_moves"] = float(np.mean(missed_moves)) if missed_moves else 0.0
        avg["total_missed_moves"] = float(np.sum(missed_moves)) if missed_moves else 0.0
        avg["outperformance_vs_bh"] = float(avg.get("total_return", 0.0) - avg["bh_eval_return"])
        avg["avg_reward_sum"] = float(np.mean(reward_sums)) if reward_sums else 0.0
        avg["eval_short_actions"] = float(total_action_counts.get(0, 0))
        avg["eval_flat_actions"] = float(total_action_counts.get(1, 0))
        avg["eval_long_actions"] = float(total_action_counts.get(2, 0))
        avg["eval_episodes"] = n_episodes
        avg["eval_range_start"] = int(active_range[0])
        avg["eval_range_end"] = int(active_range[1])
        avg["eval_range_len"] = int(active_range[1] - active_range[0])
        logger.info(
            "Eval aggregate [%d:%d): net=%.4f gross=%.4f dd=%.4f trades=%d win_rate=%.2f%% "
            "flat_ratio=%.2f%% pos_ratio=%.2f%% avg_server_cost=$%.2f total_server_cost=$%.2f "
            "avg_missed_moves=%.4f bh_eval=%.4f alpha=%.4f",
            int(active_range[0]),
            int(active_range[1]),
            float(avg.get("total_return", 0.0)),
            float(avg.get("gross_total_return", 0.0)),
            float(avg.get("max_drawdown", 0.0)),
            int(avg.get("n_trades", 0)),
            float(avg.get("win_rate", 0.0) * 100.0),
            float(avg.get("flat_ratio", 0.0) * 100.0),
            float(avg.get("position_ratio", 0.0) * 100.0),
            float(avg.get("server_cost_paid", 0.0)),
            float(avg.get("total_server_cost_paid", 0.0)),
            float(avg.get("missed_moves", 0.0)),
            float(avg.get("bh_eval_return", 0.0)),
            float(avg.get("outperformance_vs_bh", 0.0)),
        )
        return avg

    def _save_chart(self, model, metrics: dict):
        """Save RL vs Buy&Hold comparison chart."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Run one deterministic episode for chart
            env = self.create_env(segment_range=self.test_range)
            obs, _ = env.reset()
            positions = []
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, t, tr, info = env.step(action)
                positions.append(info["position"])
                if t or tr:
                    break

            equity = np.array(env.equity_curve)
            start_idx = env._start
            end_idx = start_idx + len(equity) - 1
            bh_prices = self.price_series[start_idx:end_idx].astype(np.float64)
            if len(bh_prices) >= 2:
                bh_equity = np.concatenate((
                    [env.initial_balance],
                    env.initial_balance * bh_prices / bh_prices[0],
                ))
                bh_ret = bh_prices[-1] / bh_prices[0] - 1
            else:
                bh_equity = np.full(max(len(equity), 1), env.initial_balance, dtype=np.float64)
                bh_ret = 0.0

            fig, axes = plt.subplots(3, 2, figsize=(16, 12))
            fig.suptitle(f"GARIC Training Results — {len(equity)} steps", fontsize=14, fontweight="bold")

            # 1. RL vs Buy&Hold Equity
            ax = axes[0, 0]
            ax.plot(equity, label=f"RL Agent ({metrics.get('total_return', 0):.1%})", linewidth=1)
            ax.plot(bh_equity, label=f"Buy & Hold ({bh_ret:.1%})", linewidth=1, alpha=0.7)
            ax.set_title("Equity: RL Agent vs Buy & Hold")
            ax.legend()
            ax.set_ylabel("Balance ($)")

            # 2. Drawdown
            ax = axes[0, 1]
            peak = np.maximum.accumulate(equity)
            dd = (peak - equity) / np.where(peak > 0, peak, 1) * 100
            ax.fill_between(range(len(dd)), dd, alpha=0.7, color="red")
            ax.set_title(f"Drawdown (Max: {format_drawdown_pct(metrics.get('max_drawdown', 0))})")
            ax.invert_yaxis()
            ax.set_ylabel("%")

            # 3. Position over time
            ax = axes[1, 0]
            ax.plot(positions, linewidth=0.5, alpha=0.8)
            ax.set_title(f"Position History (Trades: {metrics.get('n_trades', 0):.0f})")
            ax.set_ylabel("Position")
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)

            # 4. Metrics table
            ax = axes[1, 1]
            ax.axis("off")
            table_data = [
                ["Sharpe", f"{metrics.get('sharpe', 0):.3f}"],
                ["Sortino", f"{metrics.get('sortino', 0):.3f}"],
                ["Net Return", f"{metrics.get('total_return', 0):.2%}"],
                ["Gross Return", f"{metrics.get('gross_total_return', metrics.get('total_return', 0)):.2%}"],
                ["Server Cost", f"-${metrics.get('server_cost_paid', 0):.2f}"],
                ["Max DD", format_drawdown_pct(metrics.get("max_drawdown", 0))],
                ["Trades", f"{metrics.get('n_trades', 0):.0f}"],
                ["Longs", f"{metrics.get('n_longs', 0):.0f}"],
                ["Shorts", f"{metrics.get('n_shorts', 0):.0f}"],
                ["Wins", f"{metrics.get('n_wins', 0):.0f}"],
                ["Losses", f"{metrics.get('n_losses', 0):.0f}"],
                ["Win Rate", f"{metrics.get('win_rate', 0):.1%}"],
            ]
            tbl = ax.table(cellText=table_data, colLabels=["Metric", "Value"],
                           loc="center", cellLoc="left")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            tbl.scale(1, 1.4)
            ax.set_title("Performance Metrics (avg 10 episodes)")

            # 5. Returns distribution
            ax = axes[2, 0]
            rets = np.diff(equity) / np.where(equity[:-1] != 0, equity[:-1], 1)
            rets = rets[np.isfinite(rets)]
            if len(rets) > 0:
                ax.hist(rets, bins=80, alpha=0.7, color="steelblue", edgecolor="none")
                ax.axvline(x=0, color="red", linestyle="--")
                ax.set_title("Returns Distribution")

            # 6. RL vs B&H comparison bar
            ax = axes[2, 1]
            rl_ret = metrics.get("total_return", 0) * 100
            bh_ret = bh_ret * 100
            colors = ["green" if rl_ret > 0 else "red", "gray"]
            ax.bar(["RL Agent", "Buy & Hold"], [rl_ret, bh_ret], color=colors, alpha=0.8)
            ax.set_title("Return Comparison (%)")
            ax.set_ylabel("%")
            for i, v in enumerate([rl_ret, bh_ret]):
                ax.text(i, v + (1 if v >= 0 else -3), f"{v:.1f}%", ha="center", fontweight="bold")

            plt.tight_layout()
            path = "checkpoints/training_results.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Chart saved → {path}")
        except Exception as e:
            logger.warning(f"Chart failed: {e}")
