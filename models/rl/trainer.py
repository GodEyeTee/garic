"""RL Trainer — PPO Discrete + PnL reward + multi-episode eval.

Key fixes:
- n_steps = max_episode_steps (PPO เห็นจบ episode ทุก rollout)
- ent_coef = 0.05 (ป้องกัน entropy collapse อย่างเด็ดขาด)
- Multi-episode eval (ไม่ใช่แค่ 1 episode)
- RL vs Buy&Hold comparison chart
"""

import logging
import os
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
        selection_max_dominant_action_ratio: float = 0.95,
        selection_min_avg_trades_per_episode: float = 2.0,
        selection_min_action_entropy: float = 0.02,
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
        self.selection_max_dominant_action_ratio = float(selection_max_dominant_action_ratio)
        self.selection_min_avg_trades_per_episode = float(selection_min_avg_trades_per_episode)
        self.selection_min_action_entropy = float(selection_min_action_entropy)
        self._dashboard = None

    def _normalize_range(self, segment_range: tuple[int, int] | None) -> tuple[int, int]:
        if self.total_len <= 0:
            return (0, 0)
        if segment_range is None:
            return (0, self.total_len)
        start, end = int(segment_range[0]), int(segment_range[1])
        start = max(0, min(start, self.total_len - 1))
        end = max(start + 1, min(end, self.total_len))
        return (start, end)

    @staticmethod
    def _saved_model_path(base_path: Path | str) -> Path:
        path = Path(base_path)
        return path if path.suffix == ".zip" else path.with_suffix(".zip")

    def create_env(
        self,
        feature_arrays=None,
        price_series=None,
        ohlcv_data=None,
        segment_range=None,
        balanced_sampling=None,
        max_episode_steps_override=None,
    ):
        from models.rl.environment import CryptoFuturesEnv
        segment_start, segment_end = self._normalize_range(segment_range)
        env_kwargs = dict(self.env_kwargs)
        if balanced_sampling is None:
            balanced_sampling = env_kwargs.pop("balanced_sampling", False)
        else:
            env_kwargs.pop("balanced_sampling", None)
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
            max_episode_steps=(
                self.max_episode_steps
                if max_episode_steps_override is None
                else max(int(max_episode_steps_override), 1)
            ),
            monthly_server_cost_usd=self.monthly_server_cost_usd,
            periods_per_day=self.periods_per_day,
            segment_start=segment_start,
            segment_end=segment_end,
            balanced_sampling=balanced_sampling,
            **env_kwargs,
        )

    def _run_eval_episode(self, model, env, *, seed: int | None = None) -> tuple[dict, dict[int, int], float]:
        obs, _ = env.reset(seed=seed)
        action_counts = {0: 0, 1: 0, 2: 0}
        rewards = []
        while True:
            action, _ = model.predict(obs, deterministic=True)
            action_int = int(action.item()) if isinstance(action, np.ndarray) else int(action)
            action_counts[action_int] = action_counts.get(action_int, 0) + 1
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(float(reward))
            if terminated or truncated:
                break
        return env.get_metrics(), action_counts, float(np.sum(rewards))

    def train(
        self,
        total_timesteps: int = 200_000,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        ent_coef: float = 0.02,
        eval_every_steps: int = 50_000,
        eval_episodes: int = 8,
        device: str = "auto",
        collapse_eval_patience: int = 4,
        collapse_min_steps: int = 200_000,
        **kwargs,
    ) -> dict:
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.callbacks import CheckpointCallback
        except ImportError:
            return {"error": "stable-baselines3 not installed"}

        env_kwargs = dict(self.env_kwargs)
        balanced_sampling_enabled = bool(env_kwargs.pop("balanced_sampling", False))
        env = self.create_env(segment_range=self.train_range, balanced_sampling=balanced_sampling_enabled)
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

        requested_device = str(device or "auto").strip().lower()
        resolved_device = requested_device
        cuda_available = False
        torch_version = ""
        torch_cuda_version = ""
        try:
            import torch

            torch_version = str(getattr(torch, "__version__", ""))
            torch_cuda_version = str(getattr(torch.version, "cuda", "") or "")
            cuda_available = bool(torch.cuda.is_available())
            if requested_device == "auto":
                resolved_device = "cuda" if cuda_available else "cpu"
            elif requested_device.startswith("cuda") and not cuda_available:
                logger.warning(
                    "Requested PPO device '%s' but CUDA is unavailable. Falling back to CPU.",
                    requested_device,
                )
                resolved_device = "cpu"
        except Exception:
            if requested_device == "auto":
                resolved_device = "cpu"
            elif requested_device.startswith("cuda"):
                logger.warning(
                    "Requested PPO device '%s' but torch.cuda check failed. Falling back to CPU.",
                    requested_device,
                )
                resolved_device = "cpu"

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
            device=resolved_device,
            policy_kwargs={"net_arch": dict(pi=[256, 256], vf=[256, 256])},
            verbose=0,  # *** ปิด SB3 print — ใช้ dashboard แทน ***
        )

        logger.info(f"PPO: features={n_feat}, ent_coef={ent_coef}, n_steps={actual_n_steps}, "
                     f"ep_len={self.max_episode_steps}, total={total_timesteps}")
        logger.info(
            "PPO device: requested=%s resolved=%s cuda_available=%s torch=%s torch_cuda=%s",
            requested_device,
            resolved_device,
            cuda_available,
            torch_version,
            torch_cuda_version or "none",
        )
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
            "Reward v3: pnl_reward_scale=%.1f drawdown_penalty_scale=%.2f "
            "turnover_penalty_scale=%.3f inactive_episode_penalty=%.2f "
            "static_position_episode_penalty=%.2f",
            float(self.env_kwargs.get("pnl_reward_scale", 100.0)),
            float(self.env_kwargs.get("drawdown_penalty_scale", 2.0)),
            float(self.env_kwargs.get("turnover_penalty_scale", 0.05)),
            float(self.env_kwargs.get("inactive_episode_penalty", 0.0)),
            float(self.env_kwargs.get("static_position_episode_penalty", 0.0)),
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
        sampling_stats = getattr(env, "sampling_stats", {})
        logger.info(
            "Train sampler: mode=%s threshold=%.4f candidates=%d buckets={down:%d, flat:%d, up:%d}",
            sampling_stats.get("mode", "unknown"),
            float(sampling_stats.get("threshold", 0.0)),
            int(sampling_stats.get("total_candidates", 0)),
            int(sampling_stats.get("down_count", 0)),
            int(sampling_stats.get("flat_count", 0)),
            int(sampling_stats.get("up_count", 0)),
        )
        logger.info(
            "Checkpoint gates: max_dominant_action_ratio=%.2f min_avg_trades_per_episode=%.2f "
            "min_action_entropy=%.3f",
            self.selection_max_dominant_action_ratio,
            self.selection_min_avg_trades_per_episode,
            self.selection_min_action_entropy,
        )
        logger.info(
            "Collapse early-stop: patience=%d min_steps=%d",
            int(max(collapse_eval_patience, 0)),
            int(max(collapse_min_steps, 0)),
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
                self.best_base_path = None
                self.collapse_eval_count = 0
                self.stopped_early = False
                self.stop_reason = ""

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
                        "trades=%.1f flat_ratio=%.2f%% action_entropy=%.3f dominant_ratio=%.2f%%",
                        self.num_timesteps,
                        float(score),
                        float(metrics.get("outperformance_vs_bh", 0.0)),
                        float(metrics.get("total_return", 0.0)),
                        float(metrics.get("gross_total_return", 0.0)),
                        float(metrics.get("n_trades", 0.0)),
                        float(metrics.get("flat_ratio", 0.0) * 100.0),
                        float(metrics.get("eval_action_entropy", 0.0)),
                        float(metrics.get("eval_dominant_action_ratio", 0.0) * 100.0),
                    )
                    if np.isfinite(score) and score > self.best_score:
                        self.best_score = score
                        self.best_base_path = self.save_path / "rl_agent_best"
                        self.model.save(str(self.best_base_path))
                        self.best_path = trainer_self._saved_model_path(self.best_base_path)
                        logger.info(
                            "New best checkpoint at step=%d -> %s (score=%.4f)",
                            self.num_timesteps,
                            self.best_path,
                            float(score),
                        )
                    self._last_eval_step = self.num_timesteps
                    collapsed_eval = (
                        self.num_timesteps >= int(max(collapse_min_steps, 0))
                        and float(metrics.get("flat_ratio", 0.0)) >= 0.995
                        and float(metrics.get("avg_trades_per_episode", metrics.get("n_trades", 0.0))) < 0.25
                        and float(metrics.get("eval_dominant_action_ratio", 0.0)) >= 0.995
                        and float(metrics.get("eval_action_entropy", 0.0)) <= 0.01
                    )
                    if collapsed_eval:
                        self.collapse_eval_count += 1
                        logger.warning(
                            "Collapsed eval detected (%d/%d) at step=%d: flat_ratio=%.2f%% trades=%.2f entropy=%.3f dominant_ratio=%.2f%%",
                            self.collapse_eval_count,
                            int(max(collapse_eval_patience, 0)),
                            self.num_timesteps,
                            float(metrics.get("flat_ratio", 0.0) * 100.0),
                            float(metrics.get("avg_trades_per_episode", metrics.get("n_trades", 0.0))),
                            float(metrics.get("eval_action_entropy", 0.0)),
                            float(metrics.get("eval_dominant_action_ratio", 0.0) * 100.0),
                        )
                        if int(max(collapse_eval_patience, 0)) > 0 and self.collapse_eval_count >= int(max(collapse_eval_patience, 0)):
                            self.stopped_early = True
                            self.stop_reason = "collapsed_policy"
                            logger.warning(
                                "Early stopping PPO after %d collapsed validation checkpoints. Moving on to model selection.",
                                self.collapse_eval_count,
                            )
                            return False
                    else:
                        self.collapse_eval_count = 0
                return True

        # Get dashboard from pipeline (passed via _dashboard key in self)
        dash = getattr(self, '_dashboard', None)
        cb = DashboardCallback(dash=dash, save_path=str(self.checkpoint_dir),
                               save_freq=max(total_timesteps // 5, 2000),
                               eval_freq=eval_every_steps,
                               eval_eps=eval_episodes)
        model.learn(total_timesteps=total_timesteps, callback=cb)

        selection_gate_passed = cb.best_path is not None and cb.best_path.exists()
        if selection_gate_passed:
            logger.info("Loading best checkpoint -> %s", cb.best_path)
            model = PPO.load(str(cb.best_path), env=env)
        else:
            logger.warning(
                "No checkpoint passed selection gates; keeping final in-memory model. "
                "Training likely remained collapsed."
            )

        final_path = self.checkpoint_dir / "rl_agent_final"
        model.save(str(final_path))
        logger.info(f"Saved model -> {final_path}")

        # === Multi-episode eval ===
        metrics = self._eval_multi_episode(model, n_episodes=10, segment_range=self.test_range)
        metrics["selection_gate_passed"] = 1.0 if selection_gate_passed else 0.0
        metrics["selection_best_score"] = float(cb.best_score) if np.isfinite(cb.best_score) else float("-inf")
        metrics["ppo_early_stopped"] = 1.0 if cb.stopped_early else 0.0
        metrics["ppo_stop_reason"] = cb.stop_reason or ""
        logger.info(f"Eval ({10} episodes avg): {metrics}")

        # === Save chart ===
        self._save_chart(model, metrics)

        return metrics

    def score_candidate(
        self,
        metrics: dict,
        *,
        max_dominant_action_ratio: float | None = None,
        min_avg_trades_per_episode: float | None = None,
        min_action_entropy: float | None = None,
    ) -> float:
        """Prefer positive alpha and penalize action-collapse policies."""
        alpha = float(metrics.get("outperformance_vs_bh", 0.0))
        net_return = float(metrics.get("total_return", 0.0))
        gross_return = float(metrics.get("gross_total_return", net_return))
        sharpe = float(metrics.get("sharpe", 0.0))
        max_drawdown = abs(float(metrics.get("max_drawdown", 1.0)))
        flat_ratio = float(metrics.get("flat_ratio", 1.0))
        position_ratio = float(metrics.get("position_ratio", 0.0))
        avg_trades = float(metrics.get("avg_trades_per_episode", 0.0))
        action_entropy = float(metrics.get("eval_action_entropy", 0.0))
        dominant_action_ratio = float(metrics.get("eval_dominant_action_ratio", 1.0))
        wrong_side_moves = float(metrics.get("wrong_side_moves", 0.0))
        worst_net_return = float(metrics.get("walkforward_min_total_return", net_return))
        worst_gross_return = float(metrics.get("walkforward_min_gross_total_return", gross_return))
        worst_alpha = float(metrics.get("walkforward_min_alpha", alpha))
        walkforward_net_std = float(metrics.get("walkforward_net_std", 0.0))
        walkforward_gross_std = float(metrics.get("walkforward_gross_std", 0.0))

        max_dominant_action_ratio = (
            self.selection_max_dominant_action_ratio
            if max_dominant_action_ratio is None
            else float(max_dominant_action_ratio)
        )
        min_avg_trades_per_episode = (
            self.selection_min_avg_trades_per_episode
            if min_avg_trades_per_episode is None
            else float(min_avg_trades_per_episode)
        )
        min_action_entropy = (
            self.selection_min_action_entropy
            if min_action_entropy is None
            else float(min_action_entropy)
        )

        if dominant_action_ratio > max_dominant_action_ratio:
            return float("-inf")
        if avg_trades < min_avg_trades_per_episode:
            return float("-inf")
        if action_entropy < min_action_entropy:
            return float("-inf")
        # Hard gate: reject models that lose more than 10% net
        if net_return < -0.10:
            return float("-inf")

        # Sharpe-priority scoring (v2):
        # Primary objective is risk-adjusted return + alpha vs buy&hold,
        # NOT raw net return. A model that wins 40% with 3:1 profit:loss
        # is preferred over one that wins 70% with 1:1.
        collapse_penalty = max(dominant_action_ratio - 0.75, 0.0) * 2.0
        collapse_penalty += max(flat_ratio - 0.90, 0.0) * 3.0   # raised, but only above 90% flat

        trade_bonus = min(avg_trades, 8.0) * 0.015
        turnover_penalty = max(avg_trades - 10.0, 0.0) * 0.020
        turnover_penalty += max(avg_trades - 20.0, 0.0) * 0.030
        loss_penalty = max(-net_return, 0.0) * 6.0              # 8 → 6, sharpe carries more weight
        profit_bonus = max(net_return, 0.0) * 6.0               # 12 → 6, sharpe is primary
        gross_bonus = max(gross_return, 0.0) * 4.0              # 8 → 4
        wrong_side_penalty = wrong_side_moves * 0.05
        return (
            sharpe * 4.0                       # 3.0 → 4.0  (PRIMARY: risk-adjusted return)
            + alpha * 3.0                      # 1.5 → 3.0  (PRIMARY: must beat passive)
            + net_return * 2.0                 # 6.0 → 2.0  (secondary)
            + gross_return * 1.5               # 4.0 → 1.5
            + worst_net_return * 2.0           # 4.0 → 2.0
            + worst_gross_return * 1.5         # 3.0 → 1.5
            + worst_alpha * 1.5                # 0.75 → 1.5
            + profit_bonus
            + gross_bonus
            - max_drawdown * 5.0               # 4.0 → 5.0  (survival first)
            + action_entropy * 0.15
            + trade_bonus
            - collapse_penalty
            - turnover_penalty
            - loss_penalty
            - wrong_side_penalty
            - walkforward_net_std * 1.5        # 2.0 → 1.5
            - walkforward_gross_std * 1.0      # 1.5 → 1.0
        )

    def _selection_score(self, metrics: dict) -> float:
        return self.score_candidate(metrics)

    def _eval_multi_episode(
        self,
        model,
        n_episodes: int = 10,
        segment_range: tuple[int, int] | None = None,
        log_episodes: bool = True,
        seed_base: int | None = None,
    ) -> dict:
        """Run multiple episodes, aggregate metrics."""
        all_metrics = []
        total_equity = []
        bh_eval_returns = []
        server_costs = []
        missed_moves = []
        wrong_side_moves = []
        reward_sums = []
        total_action_counts = {0: 0, 1: 0, 2: 0}
        active_range = self._normalize_range(segment_range or self.test_range)

        for ep in range(n_episodes):
            if self._dashboard:
                self._dashboard.update(status_msg=f"Validating Candidate (Episode {ep+1}/{n_episodes})...")
            env = self.create_env(segment_range=active_range, balanced_sampling=False)
            seed = None if seed_base is None else int(seed_base) + ep
            m, action_counts, reward_sum = self._run_eval_episode(model, env, seed=seed)
            all_metrics.append(m)
            total_equity.extend(env.equity_curve)
            prices_slice = self.price_series[env._start: env._start + len(env.equity_curve) - 1]
            if len(prices_slice) >= 2:
                bh_eval_returns.append(float(prices_slice[-1] / prices_slice[0] - 1.0))
            else:
                bh_eval_returns.append(0.0)
            server_costs.append(float(m.get("server_cost_paid", 0.0)))
            missed_moves.append(float(m.get("missed_moves", 0.0)))
            wrong_side_moves.append(float(m.get("wrong_side_moves", 0.0)))
            reward_sums.append(float(reward_sum))
            for key, value in action_counts.items():
                total_action_counts[key] = total_action_counts.get(key, 0) + int(value)
            if log_episodes:
                logger.info(
                    "Eval episode %d/%d: start=%d steps=%d net=%.4f gross=%.4f dd=%.4f trades=%d "
                    "win_rate=%.2f%% flat_ratio=%.2f%% pos_ratio=%.2f%% server_cost=$%.2f "
                    "missed_moves=%.4f wrong_side_moves=%.4f actions={short:%d, flat:%d, long:%d} reward_sum=%.4f",
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
                    float(m.get("wrong_side_moves", 0.0)),
                    int(action_counts.get(0, 0)),
                    int(action_counts.get(1, 0)),
                    int(action_counts.get(2, 0)),
                    float(reward_sum),
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
        avg["wrong_side_moves"] = float(np.mean(wrong_side_moves)) if wrong_side_moves else 0.0
        avg["total_wrong_side_moves"] = float(np.sum(wrong_side_moves)) if wrong_side_moves else 0.0
        avg["outperformance_vs_bh"] = float(avg.get("total_return", 0.0) - avg["bh_eval_return"])
        avg["avg_reward_sum"] = float(np.mean(reward_sums)) if reward_sums else 0.0
        avg["eval_short_actions"] = float(total_action_counts.get(0, 0))
        avg["eval_flat_actions"] = float(total_action_counts.get(1, 0))
        avg["eval_long_actions"] = float(total_action_counts.get(2, 0))
        avg["eval_episodes"] = n_episodes
        avg["eval_range_start"] = int(active_range[0])
        avg["eval_range_end"] = int(active_range[1])
        avg["eval_range_len"] = int(active_range[1] - active_range[0])
        total_actions = float(sum(total_action_counts.values()))
        if total_actions > 0:
            action_probs = np.array([
                total_action_counts.get(0, 0),
                total_action_counts.get(1, 0),
                total_action_counts.get(2, 0),
            ], dtype=np.float64) / total_actions
            nonzero_probs = action_probs[action_probs > 0]
            entropy = float(-np.sum(nonzero_probs * np.log(nonzero_probs)) / np.log(3.0))
            if abs(entropy) < 1e-12:
                entropy = 0.0
            dominant_idx = int(np.argmax(action_probs))
            dominant_ratio = float(np.max(action_probs))
        else:
            entropy = 0.0
            dominant_idx = 1
            dominant_ratio = 1.0
        avg["eval_action_entropy"] = entropy
        avg["eval_dominant_action"] = float(dominant_idx)
        avg["eval_dominant_action_ratio"] = dominant_ratio
        logger.info(
            "Eval aggregate [%d:%d): net=%.4f gross=%.4f dd=%.4f trades=%d win_rate=%.2f%% "
            "flat_ratio=%.2f%% pos_ratio=%.2f%% avg_server_cost=$%.2f total_server_cost=$%.2f "
            "avg_missed_moves=%.4f avg_wrong_side_moves=%.4f bh_eval=%.4f alpha=%.4f "
            "action_entropy=%.3f dominant_action=%d dominant_ratio=%.2f%%",
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
            float(avg.get("wrong_side_moves", 0.0)),
            float(avg.get("bh_eval_return", 0.0)),
            float(avg.get("outperformance_vs_bh", 0.0)),
            float(avg.get("eval_action_entropy", 0.0)),
            int(avg.get("eval_dominant_action", 1)),
            float(avg.get("eval_dominant_action_ratio", 0.0) * 100.0),
        )
        return avg

    def _eval_full_segment(
        self,
        model,
        segment_range: tuple[int, int] | None = None,
        *,
        log_episode: bool = False,
    ) -> dict:
        active_range = self._normalize_range(segment_range or self.test_range)
        segment_len = max(active_range[1] - active_range[0], 1)
        env = self.create_env(
            segment_range=active_range,
            balanced_sampling=False,
            max_episode_steps_override=max(segment_len - 1, 1),
        )
        metrics, action_counts, reward_sum = self._run_eval_episode(model, env)
        total_actions = max(sum(action_counts.values()), 1)
        action_probs = np.array(
            [
                action_counts.get(0, 0),
                action_counts.get(1, 0),
                action_counts.get(2, 0),
            ],
            dtype=np.float64,
        ) / float(total_actions)
        nonzero_probs = action_probs[action_probs > 0.0]
        entropy = float(-np.sum(nonzero_probs * np.log(nonzero_probs)) / np.log(3.0)) if nonzero_probs.size else 0.0
        if abs(entropy) < 1e-12:
            entropy = 0.0
        dominant_idx = int(np.argmax(action_probs))
        dominant_ratio = float(np.max(action_probs))

        prices_slice = self.price_series[env._start: env._start + len(env.equity_curve) - 1]
        bh_eval_return = float(prices_slice[-1] / prices_slice[0] - 1.0) if len(prices_slice) >= 2 else 0.0
        metrics = dict(metrics)
        metrics["avg_trades_per_episode"] = float(metrics.get("n_trades", 0.0))
        metrics["bh_eval_return"] = float(bh_eval_return)
        metrics["total_server_cost_paid"] = float(metrics.get("server_cost_paid", 0.0))
        metrics["total_missed_moves"] = float(metrics.get("missed_moves", 0.0))
        metrics["total_wrong_side_moves"] = float(metrics.get("wrong_side_moves", 0.0))
        metrics["outperformance_vs_bh"] = float(metrics.get("total_return", 0.0) - bh_eval_return)
        metrics["avg_reward_sum"] = float(reward_sum)
        metrics["eval_short_actions"] = float(action_counts.get(0, 0))
        metrics["eval_flat_actions"] = float(action_counts.get(1, 0))
        metrics["eval_long_actions"] = float(action_counts.get(2, 0))
        metrics["eval_episodes"] = 1
        metrics["eval_range_start"] = int(active_range[0])
        metrics["eval_range_end"] = int(active_range[1])
        metrics["eval_range_len"] = int(active_range[1] - active_range[0])
        metrics["eval_action_entropy"] = float(entropy)
        metrics["eval_dominant_action"] = float(dominant_idx)
        metrics["eval_dominant_action_ratio"] = float(dominant_ratio)

        logger.info(
            "Eval full segment [%d:%d): net=%.4f gross=%.4f dd=%.4f trades=%d win_rate=%.2f%% "
            "flat_ratio=%.2f%% pos_ratio=%.2f%% server_cost=$%.2f missed_moves=%.4f "
            "wrong_side_moves=%.4f bh_eval=%.4f alpha=%.4f action_entropy=%.3f "
            "dominant_action=%d dominant_ratio=%.2f%%",
            int(active_range[0]),
            int(active_range[1]),
            float(metrics.get("total_return", 0.0)),
            float(metrics.get("gross_total_return", 0.0)),
            float(metrics.get("max_drawdown", 0.0)),
            int(metrics.get("n_trades", 0)),
            float(metrics.get("win_rate", 0.0) * 100.0),
            float(metrics.get("flat_ratio", 0.0) * 100.0),
            float(metrics.get("position_ratio", 0.0) * 100.0),
            float(metrics.get("server_cost_paid", 0.0)),
            float(metrics.get("missed_moves", 0.0)),
            float(metrics.get("wrong_side_moves", 0.0)),
            float(metrics.get("bh_eval_return", 0.0)),
            float(metrics.get("outperformance_vs_bh", 0.0)),
            float(metrics.get("eval_action_entropy", 0.0)),
            int(metrics.get("eval_dominant_action", 1)),
            float(metrics.get("eval_dominant_action_ratio", 0.0) * 100.0),
        )
        if log_episode:
            logger.info(
                "Eval full segment actions: {short:%d, flat:%d, long:%d} reward_sum=%.4f",
                int(action_counts.get(0, 0)),
                int(action_counts.get(1, 0)),
                int(action_counts.get(2, 0)),
                float(reward_sum),
            )
        return metrics

    def _build_walkforward_windows(
        self,
        segment_range: tuple[int, int] | None = None,
        *,
        n_windows: int = 4,
        min_window_size: int = 4096,
    ) -> list[tuple[int, int]]:
        active_range = self._normalize_range(segment_range or self.eval_range)
        range_len = max(active_range[1] - active_range[0], 1)
        n_windows = max(int(n_windows), 1)
        min_window_size = max(int(min_window_size), 64)
        if n_windows <= 1 or range_len < (min_window_size * 2):
            return [active_range]

        max_windows = max(range_len // min_window_size, 1)
        n_actual = min(n_windows, max_windows)
        if n_actual <= 1:
            return [active_range]

        window_step = range_len // n_actual
        windows: list[tuple[int, int]] = []
        cursor = active_range[0]
        for idx in range(n_actual):
            end = active_range[1] if idx == n_actual - 1 else min(cursor + window_step, active_range[1])
            if end - cursor >= max(min_window_size, 32):
                windows.append((cursor, end))
            cursor = end
        return windows or [active_range]

    def _eval_walkforward_segments(
        self,
        model,
        segment_range: tuple[int, int] | None = None,
        *,
        n_windows: int = 4,
        min_window_size: int = 4096,
    ) -> dict:
        windows = self._build_walkforward_windows(
            segment_range=segment_range,
            n_windows=n_windows,
            min_window_size=min_window_size,
        )
        metrics_list = []
        for i, window in enumerate(windows):
            if self._dashboard:
                self._dashboard.update(status_msg=f"Walk-forward Validation ({i+1}/{len(windows)})...")
            metrics_list.append(self._eval_full_segment(model, segment_range=window, log_episode=False))
        if len(metrics_list) == 1:
            metrics = dict(metrics_list[0])
            active = bool(
                float(metrics.get("avg_trades_per_episode", metrics.get("n_trades", 0.0))) >= 1.0
                and float(metrics.get("flat_ratio", 1.0)) < 0.999
                and float(metrics.get("eval_dominant_action_ratio", 1.0)) < 0.9995
            )
            metrics["walkforward_window_count"] = 1
            metrics["walkforward_min_total_return"] = float(metrics.get("total_return", 0.0))
            metrics["walkforward_min_gross_total_return"] = float(metrics.get("gross_total_return", 0.0))
            metrics["walkforward_min_alpha"] = float(metrics.get("outperformance_vs_bh", 0.0))
            metrics["walkforward_net_std"] = 0.0
            metrics["walkforward_gross_std"] = 0.0
            metrics["walkforward_active_window_ratio"] = 1.0 if active else 0.0
            metrics["walkforward_active_window_count"] = 1 if active else 0
            metrics["walkforward_positive_net_ratio"] = 1.0 if float(metrics.get("total_return", 0.0)) > 0.0 else 0.0
            metrics["walkforward_positive_alpha_ratio"] = (
                1.0 if float(metrics.get("outperformance_vs_bh", 0.0)) > 0.0 else 0.0
            )
            metrics["walkforward_worst_dominant_action_ratio"] = float(
                metrics.get("eval_dominant_action_ratio", 1.0)
            )
            metrics["walkforward_median_trades"] = float(
                metrics.get("avg_trades_per_episode", metrics.get("n_trades", 0.0))
            )
            metrics["walkforward_windows"] = [[int(windows[0][0]), int(windows[0][1])]]
            return metrics

        def _arr(key: str, default: float = 0.0) -> np.ndarray:
            return np.asarray([float(m.get(key, default)) for m in metrics_list], dtype=np.float64)

        net = _arr("total_return")
        gross = _arr("gross_total_return")
        alpha = _arr("outperformance_vs_bh")
        dd = np.abs(_arr("max_drawdown"))
        sharpe = _arr("sharpe")
        sortino = _arr("sortino")
        trades = _arr("avg_trades_per_episode")
        win_rate = _arr("win_rate")
        flat_ratio = _arr("flat_ratio")
        pos_ratio = _arr("position_ratio")
        entropy = _arr("eval_action_entropy")
        dom_ratio = _arr("eval_dominant_action_ratio")
        missed = _arr("missed_moves")
        wrong_side = _arr("wrong_side_moves")
        reward_sum = _arr("avg_reward_sum")
        server_cost = _arr("server_cost_paid")
        active_mask = (trades >= 1.0) & (flat_ratio < 0.999) & (dom_ratio < 0.9995)
        positive_net_mask = net > 0.0
        positive_alpha_mask = alpha > 0.0

        dominant_idx = int(np.argmax(dom_ratio))
        combined = dict(metrics_list[dominant_idx])
        combined.update(
            {
                "sharpe": float(np.median(sharpe)),
                "sortino": float(np.median(sortino)),
                "max_drawdown": float(np.max(dd)),
                "total_return": float(np.median(net)),
                "gross_total_return": float(np.median(gross)),
                "flat_ratio": float(np.median(flat_ratio)),
                "position_ratio": float(np.median(pos_ratio)),
                "n_trades": float(np.median(trades)),
                "win_rate": float(np.median(win_rate)),
                "missed_moves": float(np.median(missed)),
                "wrong_side_moves": float(np.median(wrong_side)),
                "server_cost_paid": float(np.median(server_cost)),
                "avg_trades_per_episode": float(np.median(trades)),
                "outperformance_vs_bh": float(np.median(alpha)),
                "avg_reward_sum": float(np.median(reward_sum)),
                "eval_action_entropy": float(np.median(entropy)),
                "eval_dominant_action_ratio": float(np.max(dom_ratio)),
                "walkforward_window_count": int(len(metrics_list)),
                "walkforward_min_total_return": float(np.min(net)),
                "walkforward_min_gross_total_return": float(np.min(gross)),
                "walkforward_min_alpha": float(np.min(alpha)),
                "walkforward_net_std": float(np.std(net)),
                "walkforward_gross_std": float(np.std(gross)),
                "walkforward_active_window_ratio": float(np.mean(active_mask.astype(np.float64))),
                "walkforward_active_window_count": int(np.sum(active_mask)),
                "walkforward_positive_net_ratio": float(np.mean(positive_net_mask.astype(np.float64))),
                "walkforward_positive_alpha_ratio": float(np.mean(positive_alpha_mask.astype(np.float64))),
                "walkforward_worst_dominant_action_ratio": float(np.max(dom_ratio)),
                "walkforward_median_trades": float(np.median(trades)),
                "walkforward_windows": [[int(start), int(end)] for start, end in windows],
            }
        )
        logger.info(
            "Walk-forward summary [%d:%d): windows=%d median_net=%.4f worst_net=%.4f "
            "median_gross=%.4f worst_gross=%.4f median_alpha=%.4f worst_alpha=%.4f "
            "median_trades=%.2f active_windows=%d/%d worst_dom_ratio=%.2f%%",
            int(windows[0][0]),
            int(windows[-1][1]),
            int(len(windows)),
            float(np.median(net)),
            float(np.min(net)),
            float(np.median(gross)),
            float(np.min(gross)),
            float(np.median(alpha)),
            float(np.min(alpha)),
            float(np.median(trades)),
            int(np.sum(active_mask)),
            int(len(windows)),
            float(np.max(dom_ratio) * 100.0),
        )
        return combined

    def _save_chart(self, model, metrics: dict):
        """Save RL vs Buy&Hold comparison chart."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            metrics_source = str(metrics.get("metrics_source", "env_eval"))
            metrics_source_label = str(
                metrics.get(
                    "metrics_source_label",
                    "Nautilus held-out test" if metrics_source == "nautilus_test" else "Env episodic eval",
                )
            )
            nautilus_test = metrics.get("nautilus_test", {})
            nautilus_history = nautilus_test.get("history", {}) if isinstance(nautilus_test, dict) else {}
            use_nautilus_history = (
                metrics_source == "nautilus_test"
                and isinstance(nautilus_history, dict)
                and len(nautilus_history.get("equity", [])) >= 2
                and len(nautilus_history.get("price", [])) >= 2
            )

            chart_note = ""
            if use_nautilus_history:
                equity = np.asarray(nautilus_history.get("equity", []), dtype=np.float64)
                positions = list(np.asarray(nautilus_history.get("position", []), dtype=np.float64))
                bh_prices = np.asarray(nautilus_history.get("price", []), dtype=np.float64)
                initial_balance = float(equity[0]) if equity.size else 10_000.0
                if bh_prices.size >= 2 and bh_prices[0] > 0.0:
                    bh_equity = initial_balance * bh_prices / bh_prices[0]
                    bh_ret_proxy = float(bh_prices[-1] / bh_prices[0] - 1.0)
                else:
                    bh_equity = np.full(max(int(equity.size), 1), initial_balance, dtype=np.float64)
                    bh_ret_proxy = 0.0
                chart_note = f"Equity/position panels use {metrics_source_label.lower()} history."
            else:
                # Run one deterministic episode for chart proxy
                env = self.create_env(segment_range=self.test_range, balanced_sampling=False)
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
                    bh_ret_proxy = float(bh_prices[-1] / bh_prices[0] - 1.0)
                else:
                    bh_equity = np.full(max(len(equity), 1), env.initial_balance, dtype=np.float64)
                    bh_ret_proxy = 0.0
                if metrics_source == "nautilus_test":
                    chart_note = (
                        "Performance table and return bars use Nautilus held-out test. "
                        "Equity/position panels remain an env proxy rollout for visualization."
                    )
                else:
                    chart_note = "Equity/position panels and performance table use env episodic evaluation."

            bh_ret_display = float(metrics.get("bh_eval_return", bh_ret_proxy))

            fig, axes = plt.subplots(3, 2, figsize=(16, 12))
            model_family = str(metrics.get("model_family", "unknown"))
            selection_mode = str(metrics.get("selection_mode", "unknown"))
            selection_reason = str(metrics.get("selection_reason", ""))
            fig.suptitle(
                f"GARIC Training Results - {len(equity)} steps\n"
                f"Selected model: {model_family} ({selection_mode})",
                fontsize=14,
                fontweight="bold",
            )
            fig.text(
                0.5,
                0.955,
                f"Metrics source: {metrics_source_label}",
                ha="center",
                va="top",
                fontsize=9,
                color="#334155",
            )
            nautilus_validation = metrics.get("nautilus_validation", {})
            if model_family == "safe_flat":
                if selection_reason == "no_supervised_candidate":
                    rejected_text = (
                        "Safe-flat fallback: no supervised candidate survived fast-env validation.\n"
                        "The chart shows a defensive flat policy, not an active trading model."
                    )
                elif isinstance(nautilus_validation, dict) and isinstance(nautilus_validation.get("skipped"), str):
                    rejected_text = (
                        "Safe-flat fallback: no candidate reached execution-grade validation.\n"
                        f"Last selector state: {nautilus_validation.get('skipped')}"
                    )
                elif isinstance(nautilus_validation, dict) and nautilus_validation:
                    rejected_text = (
                        "Safe-flat fallback: Nautilus rejected the best supervised candidate.\n"
                        f"Candidate {nautilus_validation.get('segment_label', 'unknown')} | "
                        f"net {float(nautilus_validation.get('total_return', 0.0)):.2%} | "
                        f"gross {float(nautilus_validation.get('gross_total_return', 0.0)):.2%} | "
                        f"alpha {float(nautilus_validation.get('outperformance_vs_bh', 0.0)):.2%} | "
                        f"trades {float(nautilus_validation.get('n_trades', 0.0)):.0f} | "
                        f"dom {float(nautilus_validation.get('eval_dominant_action_ratio', 0.0)):.2%}"
                    )
                else:
                    rejected_text = (
                        "Safe-flat fallback: the selector did not approve any active trading candidate."
                    )
                fig.text(
                    0.5,
                    0.935,
                    rejected_text,
                    ha="center",
                    va="top",
                    fontsize=9,
                    color="firebrick",
                    bbox={
                        "facecolor": "#fff5f5",
                        "edgecolor": "firebrick",
                        "boxstyle": "round,pad=0.35",
                    },
                )
            elif chart_note:
                fig.text(
                    0.5,
                    0.935,
                    chart_note,
                    ha="center",
                    va="top",
                    fontsize=9,
                    color="#475569",
                    bbox={
                        "facecolor": "#f8fafc",
                        "edgecolor": "#cbd5e1",
                        "boxstyle": "round,pad=0.30",
                    },
                )

            # 1. RL vs Buy&Hold Equity
            ax = axes[0, 0]
            ax.plot(equity, label=f"RL Agent ({metrics.get('total_return', 0):.1%})", linewidth=1)
            ax.plot(bh_equity, label=f"Buy & Hold ({bh_ret_display:.1%})", linewidth=1, alpha=0.7)
            ax.set_title(
                "Equity: RL Agent vs Buy & Hold"
                + (f" ({metrics_source_label})" if use_nautilus_history else " (env proxy)")
            )
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
            ax.set_title(
                f"Position History (Trades: {metrics.get('n_trades', 0):.0f})"
                + ("" if use_nautilus_history else " [env proxy]")
            )
            ax.set_ylabel("Position")
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)

            # 4. Metrics table
            ax = axes[1, 1]
            ax.axis("off")
            table_data = [
                ["Source", metrics_source_label],
                ["Model", model_family],
                ["Selection", selection_mode],
                ["Reason", selection_reason or "-"],
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
            ax.set_title(
                "Performance Metrics (Nautilus held-out test)"
                if metrics_source == "nautilus_test"
                else "Performance Metrics (avg 10 episodes)"
            )

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
            bh_ret = bh_ret_display * 100
            colors = ["green" if rl_ret > 0 else "red", "gray"]
            ax.bar(["RL Agent", "Buy & Hold"], [rl_ret, bh_ret], color=colors, alpha=0.8)
            ax.set_title("Return Comparison (%)")
            ax.set_ylabel("%")
            for i, v in enumerate([rl_ret, bh_ret]):
                ax.text(i, v + (1 if v >= 0 else -3), f"{v:.1f}%", ha="center", fontweight="bold")

            plt.tight_layout(rect=[0, 0, 1, 0.90])
            path = Path("checkpoints/training_results.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            run_tag = str(os.environ.get("GARIC_RUN_TAG", "")).strip()
            if run_tag:
                archive_dir = Path("logs")
                archive_dir.mkdir(parents=True, exist_ok=True)
                archive_path = archive_dir / f"training_results_{run_tag}.png"
                plt.savefig(archive_path, dpi=150, bbox_inches="tight")
                logger.info("Chart archive saved -> %s", archive_path.as_posix())
            plt.close()
            logger.info("Chart saved -> %s", path.as_posix())
        except Exception as e:
            logger.warning(f"Chart failed: {e}")
