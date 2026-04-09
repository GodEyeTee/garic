"""Test model components."""

import numpy as np
import pytest


class TestNaiveForecaster:
    def test_output_shape(self):
        from models.forecast.naive import NaiveForecaster
        f = NaiveForecaster()
        prices = np.array([100, 101, 102, 103, 104, 105], dtype=np.float64)
        forecast, uncertainty = f.predict(prices, horizon=12)
        assert forecast.shape == (12,)
        assert isinstance(uncertainty, float)

    def test_uptrend_forecast(self):
        from models.forecast.naive import NaiveForecaster
        f = NaiveForecaster()
        prices = np.linspace(100, 110, 60)
        forecast, _ = f.predict(prices, horizon=5)
        assert forecast[0] > prices[-1]  # should predict higher


class TestMoERouter:
    def test_route_returns_top_k(self):
        from models.moe.router import MoERouter
        router = MoERouter(n_experts=6, top_k=2)
        returns = np.random.randn(20) * 0.01
        routing = router.route(returns, volatility=0.5)
        assert len(routing) == 2
        # Weights should sum to ~1
        total_weight = sum(w for _, w in routing)
        assert abs(total_weight - 1.0) < 0.01

    def test_combine_outputs(self):
        from models.moe.router import MoERouter
        router = MoERouter(n_experts=6, top_k=2)
        routing = [(0, 0.6), (1, 0.4)]
        outputs = {0: np.array([1.0, 2.0]), 1: np.array([3.0, 4.0])}
        combined = router.combine_expert_outputs(outputs, routing)
        expected = np.array([1.0 * 0.6 + 3.0 * 0.4, 2.0 * 0.6 + 4.0 * 0.4])
        np.testing.assert_array_almost_equal(combined, expected)


class TestRLEnvironment:
    def test_env_creation(self):
        from models.rl.environment import CryptoFuturesEnv
        n = 500
        features = np.random.randn(n, 25).astype(np.float32)
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        env = CryptoFuturesEnv(features, prices)
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape == (33,)  # 25 + 8 agent-state features

    def test_step_returns(self):
        from models.rl.environment import CryptoFuturesEnv
        n = 100
        features = np.random.randn(n, 25).astype(np.float32)
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        env = CryptoFuturesEnv(features, prices)
        obs, _ = env.reset()
        action = 2  # Long
        obs, reward, term, trunc, info = env.step(action)
        assert isinstance(reward, float)
        assert "balance" in info

    def test_segment_range_limits_reset(self):
        from models.rl.environment import CryptoFuturesEnv
        n = 200
        features = np.random.randn(n, 25).astype(np.float32)
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        env = CryptoFuturesEnv(
            features,
            prices,
            max_episode_steps=20,
            segment_start=120,
            segment_end=160,
        )
        env.reset(seed=7)
        assert 120 <= env._start < 160
        assert env.segment_end == 160

    def test_balanced_sampling_spreads_regimes(self):
        from models.rl.environment import CryptoFuturesEnv

        down = np.linspace(100, 80, 60, dtype=np.float32)
        flat = np.linspace(80, 80.4, 60, dtype=np.float32)
        up = np.linspace(80.4, 104, 60, dtype=np.float32)
        prices = np.concatenate([down, flat, up])
        features = np.random.randn(len(prices), 16).astype(np.float32)

        env = CryptoFuturesEnv(
            features,
            prices,
            max_episode_steps=20,
            balanced_sampling=True,
            regime_label_threshold=0.03,
        )
        stats = env.sampling_stats
        assert stats["mode"] == "balanced"
        assert stats["down_count"] > 0
        assert stats["flat_count"] > 0
        assert stats["up_count"] > 0

        seen_regimes = set()
        for _ in range(6):
            env.reset()
            seen_regimes.add(env._sampled_regime)

        assert seen_regimes == {"down", "flat", "up"}

    def test_inactive_episode_penalty_applies_when_no_trade(self):
        from models.rl.environment import CryptoFuturesEnv

        prices = np.full(8, 100.0, dtype=np.float32)
        features = np.zeros((8, 4), dtype=np.float32)
        env = CryptoFuturesEnv(
            features,
            prices,
            max_episode_steps=3,
            monthly_server_cost_usd=0.0,
            inactive_episode_penalty=1.0,
        )

        env.reset(seed=1)
        rewards = []
        for _ in range(3):
            _, reward, done, trunc, _ = env.step(1)
            rewards.append(reward)
            if done or trunc:
                break

        assert rewards[-1] <= -1.0

    def test_wrong_side_gets_negative_reward_right_side_positive(self):
        """Long in downtrend → negative PnL reward; Short in downtrend → positive."""
        from models.rl.environment import CryptoFuturesEnv

        prices = np.array([100.0, 99.0, 98.0, 97.0, 96.0], dtype=np.float32)
        features = np.zeros((len(prices), 4), dtype=np.float32)

        env_long = CryptoFuturesEnv(
            features, prices, max_episode_steps=3,
            monthly_server_cost_usd=0.0, pnl_reward_scale=200.0,
        )
        env_short = CryptoFuturesEnv(
            features, prices, max_episode_steps=3,
            monthly_server_cost_usd=0.0, pnl_reward_scale=200.0,
        )

        env_long.reset(seed=1)
        env_short.reset(seed=1)
        env_long.step(2)   # Long
        env_short.step(0)  # Short
        _, reward_long, _, _, _ = env_long.step(2)   # hold long in downtrend
        _, reward_short, _, _, _ = env_short.step(0)  # hold short in downtrend

        # Short should profit, long should lose in a downtrend
        assert reward_short > reward_long
        assert reward_long < 0  # wrong side = negative reward


class TestRiskManager:
    def test_reject_on_drawdown(self):
        from risk.manager import RiskManager
        rm = RiskManager(max_drawdown=0.10)
        decision = rm.evaluate(
            symbol="BTCUSDT", direction=1.0, equity=10000,
            model_confidence=0.8, win_rate=0.55, avg_win=0.03, avg_loss=0.02,
            atr_value=0.02, current_drawdown=0.15, is_major=True,
        )
        assert not decision.approved
        assert "drawdown" in decision.reject_reason.lower()

    def test_approve_normal(self):
        from risk.manager import RiskManager
        rm = RiskManager()
        decision = rm.evaluate(
            symbol="BTCUSDT", direction=1.0, equity=10000,
            model_confidence=0.8, win_rate=0.55, avg_win=0.03, avg_loss=0.02,
            atr_value=0.02, current_drawdown=0.05, is_major=True,
        )
        assert decision.approved
        assert decision.size > 0

    def test_daily_reset_and_position_tracking(self):
        from risk.manager import RiskManager
        rm = RiskManager(max_open_positions=1)
        rm.update_pnl(-500.0)
        rm.set_position("BTCUSDT", 2500.0)
        rm.reset_daily()
        assert rm._daily_pnl == 0.0
        assert "BTCUSDT" in rm._open_positions


class TestRLTrainerSelection:
    def test_saved_model_path_uses_zip_suffix(self):
        from models.rl.trainer import RLTrainer

        assert str(RLTrainer._saved_model_path("checkpoints/rl_agent_best")).endswith("rl_agent_best.zip")
        assert str(RLTrainer._saved_model_path("checkpoints/rl_agent_final.zip")).endswith("rl_agent_final.zip")

    def test_selection_score_penalizes_collapsed_policy(self):
        from models.rl.trainer import RLTrainer

        features = np.zeros((100, 10), dtype=np.float32)
        prices = np.linspace(100, 120, 100, dtype=np.float32)
        trainer = RLTrainer(features, prices)

        collapsed = {
            "outperformance_vs_bh": -0.008,
            "total_return": 0.12,
            "flat_ratio": 0.0,
            "position_ratio": 1.0,
            "avg_trades_per_episode": 1.0,
            "eval_action_entropy": 0.0,
            "eval_dominant_action_ratio": 1.0,
        }
        diversified = {
            "outperformance_vs_bh": 0.010,
            "total_return": 0.05,
            "flat_ratio": 0.35,
            "position_ratio": 0.65,
            "avg_trades_per_episode": 6.0,
            "eval_action_entropy": 0.75,
            "eval_dominant_action_ratio": 0.55,
        }

        assert trainer._selection_score(diversified) > trainer._selection_score(collapsed)

    def test_selection_score_rejects_single_action_policy(self):
        from models.rl.trainer import RLTrainer

        trainer = RLTrainer(np.zeros((100, 4), dtype=np.float32), np.linspace(100, 110, 100, dtype=np.float32))
        collapsed = {
            "outperformance_vs_bh": 0.02,
            "total_return": 0.03,
            "flat_ratio": 0.0,
            "position_ratio": 1.0,
            "avg_trades_per_episode": 1.0,
            "eval_action_entropy": 0.0,
            "eval_dominant_action_ratio": 1.0,
            "wrong_side_moves": 0.0,
        }

        assert trainer._selection_score(collapsed) == float("-inf")

    def test_relaxed_selection_score_accepts_sparse_noncollapsed_policy(self):
        from models.rl.trainer import RLTrainer

        trainer = RLTrainer(np.zeros((100, 4), dtype=np.float32), np.linspace(100, 110, 100, dtype=np.float32))
        sparse = {
            "outperformance_vs_bh": 0.01,
            "total_return": 0.01,
            "flat_ratio": 0.60,
            "position_ratio": 0.40,
            "avg_trades_per_episode": 1.0,
            "eval_action_entropy": 0.01,
            "eval_dominant_action_ratio": 0.70,
            "wrong_side_moves": 0.2,
        }

        assert trainer._selection_score(sparse) == float("-inf")
        assert np.isfinite(
            trainer.score_candidate(
                sparse,
                min_avg_trades_per_episode=0.5,
                min_action_entropy=0.005,
            )
        )

    def test_selection_score_penalizes_extreme_turnover(self):
        from models.rl.trainer import RLTrainer

        trainer = RLTrainer(np.zeros((100, 4), dtype=np.float32), np.linspace(100, 110, 100, dtype=np.float32))
        moderate = {
            "outperformance_vs_bh": 0.01,
            "total_return": 0.01,
            "gross_total_return": 0.015,
            "flat_ratio": 0.80,
            "position_ratio": 0.20,
            "avg_trades_per_episode": 6.0,
            "eval_action_entropy": 0.10,
            "eval_dominant_action_ratio": 0.80,
            "wrong_side_moves": 0.1,
            "max_drawdown": 0.03,
            "sharpe": 0.5,
        }
        extreme = dict(moderate)
        extreme["avg_trades_per_episode"] = 72.0

        assert trainer.score_candidate(moderate) > trainer.score_candidate(extreme)

    def test_eval_full_segment_uses_requested_slice(self):
        from models.rl.trainer import RLTrainer

        class FlatModel:
            def predict(self, obs, deterministic=True):
                return 1, None

        prices = np.linspace(100.0, 120.0, 100, dtype=np.float32)
        features = np.zeros((100, 6), dtype=np.float32)
        trainer = RLTrainer(features, prices, max_episode_steps=16)
        metrics = trainer._eval_full_segment(FlatModel(), segment_range=(20, 40), log_episode=False)

        assert metrics["eval_episodes"] == 1
        assert metrics["eval_range_start"] == 20
        assert metrics["eval_range_end"] == 40
        assert metrics["eval_flat_actions"] > 0

    def test_eval_walkforward_segments_aggregates_windows(self):
        from models.rl.trainer import RLTrainer

        class FlatModel:
            def predict(self, obs, deterministic=True):
                return 1, None

        prices = np.linspace(100.0, 120.0, 300, dtype=np.float32)
        features = np.zeros((300, 6), dtype=np.float32)
        trainer = RLTrainer(features, prices, max_episode_steps=32)
        metrics = trainer._eval_walkforward_segments(
            FlatModel(),
            segment_range=(20, 260),
            n_windows=3,
            min_window_size=40,
        )

        assert metrics["walkforward_window_count"] == 3
        assert len(metrics["walkforward_windows"]) == 3
        assert metrics["walkforward_min_total_return"] <= metrics["total_return"]

    def test_eval_walkforward_single_window_reports_activity_stats(self):
        from models.rl.trainer import RLTrainer

        class FlatModel:
            def predict(self, obs, deterministic=True):
                return 1, None

        prices = np.linspace(100.0, 120.0, 120, dtype=np.float32)
        features = np.zeros((120, 6), dtype=np.float32)
        trainer = RLTrainer(features, prices, max_episode_steps=16)
        metrics = trainer._eval_walkforward_segments(
            FlatModel(),
            segment_range=(20, 70),
            n_windows=4,
            min_window_size=64,
        )

        assert metrics["walkforward_window_count"] == 1
        assert "walkforward_active_window_ratio" in metrics
        assert "walkforward_median_trades" in metrics


class TestSupervisedFallback:
    def test_policy_uses_side_specific_entry_thresholds(self):
        from models.rl.supervised import ACTION_LONG, SupervisedActionModel

        class DummyClassifier:
            classes_ = np.array([0, 1, 2], dtype=np.int32)

            def predict_proba(self, x):
                return np.array([[0.60, 0.05, 0.79]], dtype=np.float32)

        model = SupervisedActionModel(
            scaler=None,
            classifier=DummyClassifier(),
            feature_dim=20,
            confidence_threshold=0.82,
            min_hold_steps=4,
            reversal_margin=0.08,
            entry_margin=0.10,
            exit_to_flat_margin=0.05,
            max_hold_steps=32,
            stop_loss_threshold=-0.012,
            drawdown_exit_threshold=0.04,
            trend_alignment_threshold=0.0015,
            countertrend_margin=0.08,
            metadata={},
            long_confidence_threshold=0.78,
            short_confidence_threshold=0.84,
            regime_confidence_relief=0.0,
        )
        features = np.zeros(20, dtype=np.float32)
        features[2] = 0.004
        features[3] = 0.006
        features[4] = 0.008

        action, _ = model.predict(features)
        assert action == ACTION_LONG

    def test_policy_relaxes_aligned_entry_after_extended_flat_patience(self):
        from models.rl.environment import build_agent_state
        from models.rl.supervised import ACTION_LONG, SupervisedActionModel

        class DummyClassifier:
            classes_ = np.array([0, 1, 2], dtype=np.int32)

            def predict_proba(self, x):
                return np.array([[0.05, 0.76, 0.80]], dtype=np.float32)

        features = np.zeros(20, dtype=np.float32)
        features[2] = 0.004
        features[3] = 0.007
        features[4] = 0.010
        features[18] = 0.998
        features[19] = 0.997
        obs = np.concatenate(
            [
                features,
                build_agent_state(
                    position=0.0,
                    upnl=0.0,
                    equity_ratio=0.0,
                    drawdown=0.0,
                    rolling_volatility=0.01,
                    turnover_last_step=0.0,
                    flat_steps=72,
                    pos_steps=0,
                ),
            ]
        ).astype(np.float32)

        model = SupervisedActionModel(
            scaler=None,
            classifier=DummyClassifier(),
            feature_dim=20,
            confidence_threshold=0.82,
            min_hold_steps=4,
            reversal_margin=0.08,
            entry_margin=0.04,
            exit_to_flat_margin=0.05,
            max_hold_steps=32,
            stop_loss_threshold=-0.012,
            drawdown_exit_threshold=0.04,
            trend_alignment_threshold=0.0015,
            countertrend_margin=0.08,
            metadata={
                "post_cost_calibration": {
                    "long": {
                        "global": {"mean_edge": 0.0020, "positive_rate": 0.70},
                        "thresholds": [],
                    },
                    "short": {
                        "global": {"mean_edge": -0.0010, "positive_rate": 0.30},
                        "thresholds": [],
                    },
                },
                "flat_patience_steps": 48,
                "flat_patience_threshold_relief": 0.03,
                "flat_patience_entry_margin_relief": 0.02,
            },
            long_confidence_threshold=0.82,
            short_confidence_threshold=0.84,
            regime_confidence_relief=0.0,
            flat_reentry_cooldown_steps=0,
            meta_label_min_edge=0.0005,
            meta_label_edge_margin=0.0002,
            meta_label_exit_edge=0.0,
            meta_label_min_positive_rate=0.47,
        )

        action, _ = model.predict(obs)
        assert action == ACTION_LONG

    def test_policy_blocks_countertrend_short_with_regime_penalty(self):
        from models.rl.environment import build_agent_state
        from models.rl.supervised import ACTION_FLAT, SupervisedActionModel

        class DummyClassifier:
            classes_ = np.array([0, 1, 2], dtype=np.int32)

            def predict_proba(self, x):
                return np.array([[0.84, 0.79, 0.10]], dtype=np.float32)

        features = np.zeros(20, dtype=np.float32)
        features[2] = 0.004
        features[3] = 0.007
        features[4] = 0.010
        features[18] = 0.998
        features[19] = 0.997
        obs = np.concatenate(
            [
                features,
                build_agent_state(
                    position=0.0,
                    upnl=0.0,
                    equity_ratio=0.0,
                    drawdown=0.0,
                    rolling_volatility=0.01,
                    turnover_last_step=0.0,
                    flat_steps=80,
                    pos_steps=0,
                ),
            ]
        ).astype(np.float32)

        model = SupervisedActionModel(
            scaler=None,
            classifier=DummyClassifier(),
            feature_dim=20,
            confidence_threshold=0.82,
            min_hold_steps=4,
            reversal_margin=0.08,
            entry_margin=0.02,
            exit_to_flat_margin=0.05,
            max_hold_steps=32,
            stop_loss_threshold=-0.012,
            drawdown_exit_threshold=0.04,
            trend_alignment_threshold=0.0015,
            countertrend_margin=0.08,
            metadata={
                "post_cost_calibration": {
                    "long": {
                        "global": {"mean_edge": 0.0010, "positive_rate": 0.55},
                        "thresholds": [],
                    },
                    "short": {
                        "global": {"mean_edge": 0.0020, "positive_rate": 0.70},
                        "thresholds": [],
                    },
                },
                "countertrend_threshold_penalty": 0.04,
                "countertrend_entry_penalty": 0.02,
            },
            long_confidence_threshold=0.82,
            short_confidence_threshold=0.82,
            regime_confidence_relief=0.0,
            flat_reentry_cooldown_steps=0,
            meta_label_min_edge=0.0005,
            meta_label_edge_margin=0.0002,
            meta_label_exit_edge=0.0,
            meta_label_min_positive_rate=0.47,
        )

        action, _ = model.predict(obs)
        assert action == ACTION_FLAT

    def test_train_supervised_model_reports_validation_brier(self):
        from models.rl.supervised import train_supervised_action_model

        n = 260
        prices = np.concatenate(
            [
                np.linspace(100.0, 110.0, 90, dtype=np.float32),
                np.linspace(110.0, 108.0, 80, dtype=np.float32),
                np.linspace(108.0, 116.0, 90, dtype=np.float32),
            ]
        )
        horizon = 4
        future_ret = np.zeros(n, dtype=np.float32)
        future_ret[:-horizon] = (prices[horizon:] / prices[:-horizon]) - 1.0
        features = np.column_stack(
            [
                future_ret,
                np.roll(future_ret, 1),
                np.roll(future_ret, 2),
                np.sign(future_ret),
                future_ret ** 2,
                np.linspace(-1.0, 1.0, n, dtype=np.float32),
            ]
        ).astype(np.float32)

        _, meta = train_supervised_action_model(
            feature_array=features,
            prices=prices,
            train_range=(0, 180),
            validation_range=(180, 240),
            model_type="logreg",
            horizon=horizon,
            min_return_threshold=0.002,
            threshold_quantile=0.40,
            max_train_samples=2000,
        )

        assert "validation_brier" in meta
        assert np.isfinite(meta["validation_brier"])
        assert meta["validation_brier"] >= 0.0
        assert "post_cost_calibration" in meta
        assert "long" in meta["post_cost_calibration"]
        assert "short" in meta["post_cost_calibration"]
        assert "robust_mean_edge" in meta["post_cost_calibration"]["long"]["global"]
        assert "active_window_ratio" in meta["post_cost_calibration"]["short"]["global"]

    def test_policy_respects_flat_reentry_cooldown(self):
        from models.rl.environment import build_agent_state
        from models.rl.supervised import ACTION_FLAT, SupervisedActionModel

        class DummyClassifier:
            classes_ = np.array([0, 1, 2], dtype=np.int32)

            def predict_proba(self, x):
                return np.array([[0.12, 0.18, 0.82]], dtype=np.float32)

        model = SupervisedActionModel(
            scaler=None,
            classifier=DummyClassifier(),
            feature_dim=8,
            confidence_threshold=0.80,
            min_hold_steps=4,
            reversal_margin=0.08,
            entry_margin=0.10,
            exit_to_flat_margin=0.05,
            max_hold_steps=32,
            stop_loss_threshold=-0.012,
            drawdown_exit_threshold=0.04,
            trend_alignment_threshold=0.0015,
            countertrend_margin=0.08,
            metadata={},
            long_confidence_threshold=0.80,
            short_confidence_threshold=0.84,
            flat_reentry_cooldown_steps=4,
        )
        obs = np.concatenate(
            [
                np.zeros(8, dtype=np.float32),
                build_agent_state(
                    position=0.0,
                    upnl=0.0,
                    equity_ratio=0.0,
                    drawdown=0.0,
                    rolling_volatility=0.0,
                    turnover_last_step=1.0,
                    flat_steps=1,
                    pos_steps=0,
                ),
            ]
        ).astype(np.float32)

        action, _ = model.predict(obs)
        assert action == ACTION_FLAT

    def test_policy_stays_flat_when_post_cost_edge_is_negative(self):
        from models.rl.environment import build_agent_state
        from models.rl.supervised import ACTION_FLAT, SupervisedActionModel

        class DummyClassifier:
            classes_ = np.array([0, 1, 2], dtype=np.int32)

            def predict_proba(self, x):
                return np.array([[0.06, 0.10, 0.84]], dtype=np.float32)

        model = SupervisedActionModel(
            scaler=None,
            classifier=DummyClassifier(),
            feature_dim=6,
            confidence_threshold=0.80,
            min_hold_steps=8,
            reversal_margin=0.08,
            entry_margin=0.08,
            exit_to_flat_margin=0.05,
            max_hold_steps=32,
            stop_loss_threshold=-0.012,
            drawdown_exit_threshold=0.04,
            trend_alignment_threshold=0.0,
            countertrend_margin=0.08,
            metadata={
                "post_cost_calibration": {
                    "long": {
                        "global": {"mean_edge": -0.0005, "positive_rate": 0.45},
                        "thresholds": [
                            {"threshold": 0.80, "mean_edge": -0.0012, "positive_rate": 0.42},
                        ],
                    },
                    "short": {
                        "global": {"mean_edge": -0.0006, "positive_rate": 0.44},
                        "thresholds": [
                            {"threshold": 0.80, "mean_edge": -0.0008, "positive_rate": 0.43},
                        ],
                    },
                },
            },
            long_confidence_threshold=0.80,
            short_confidence_threshold=0.84,
            meta_label_min_edge=0.0005,
            meta_label_edge_margin=0.0002,
            meta_label_exit_edge=0.0,
        )
        obs = np.concatenate(
            [
                np.zeros(6, dtype=np.float32),
                build_agent_state(
                    position=0.0,
                    upnl=0.0,
                    equity_ratio=0.0,
                    drawdown=0.0,
                    rolling_volatility=0.0,
                    turnover_last_step=0.0,
                    flat_steps=10,
                    pos_steps=0,
                ),
            ]
        ).astype(np.float32)

        action, _ = model.predict(obs)
        assert action == ACTION_FLAT

    def test_policy_enters_when_post_cost_edge_is_positive_enough(self):
        from models.rl.environment import build_agent_state
        from models.rl.supervised import ACTION_LONG, SupervisedActionModel

        class DummyClassifier:
            classes_ = np.array([0, 1, 2], dtype=np.int32)

            def predict_proba(self, x):
                return np.array([[0.05, 0.08, 0.87]], dtype=np.float32)

        features = np.zeros(6, dtype=np.float32)
        features[2] = 0.004
        features[3] = 0.006
        features[4] = 0.008
        model = SupervisedActionModel(
            scaler=None,
            classifier=DummyClassifier(),
            feature_dim=6,
            confidence_threshold=0.80,
            min_hold_steps=8,
            reversal_margin=0.08,
            entry_margin=0.08,
            exit_to_flat_margin=0.05,
            max_hold_steps=32,
            stop_loss_threshold=-0.012,
            drawdown_exit_threshold=0.04,
            trend_alignment_threshold=0.001,
            countertrend_margin=0.08,
            metadata={
                "post_cost_calibration": {
                    "long": {
                        "global": {"mean_edge": 0.0008, "positive_rate": 0.56},
                        "thresholds": [
                            {"threshold": 0.80, "mean_edge": 0.0021, "positive_rate": 0.67},
                        ],
                    },
                    "short": {
                        "global": {"mean_edge": -0.0004, "positive_rate": 0.46},
                        "thresholds": [
                            {"threshold": 0.80, "mean_edge": -0.0010, "positive_rate": 0.42},
                        ],
                    },
                },
            },
            long_confidence_threshold=0.80,
            short_confidence_threshold=0.84,
            meta_label_min_edge=0.0005,
            meta_label_edge_margin=0.0002,
            meta_label_exit_edge=0.0,
        )
        obs = np.concatenate(
            [
                features,
                build_agent_state(
                    position=0.0,
                    upnl=0.0,
                    equity_ratio=0.0,
                    drawdown=0.0,
                    rolling_volatility=0.0,
                    turnover_last_step=0.0,
                    flat_steps=10,
                    pos_steps=0,
                ),
            ]
        ).astype(np.float32)

        action, _ = model.predict(obs)
        assert action == ACTION_LONG

    def test_policy_damps_sparse_calibration_support(self):
        from models.rl.environment import build_agent_state
        from models.rl.supervised import ACTION_FLAT, SupervisedActionModel

        class DummyClassifier:
            classes_ = np.array([0, 1, 2], dtype=np.int32)

            def predict_proba(self, x):
                return np.array([[0.04, 0.10, 0.86]], dtype=np.float32)

        features = np.zeros(6, dtype=np.float32)
        features[2] = 0.006
        features[3] = 0.007
        features[4] = 0.009
        model = SupervisedActionModel(
            scaler=None,
            classifier=DummyClassifier(),
            feature_dim=6,
            confidence_threshold=0.80,
            min_hold_steps=8,
            reversal_margin=0.08,
            entry_margin=0.08,
            exit_to_flat_margin=0.05,
            max_hold_steps=32,
            stop_loss_threshold=-0.012,
            drawdown_exit_threshold=0.04,
            trend_alignment_threshold=0.001,
            countertrend_margin=0.08,
            metadata={
                "post_cost_calibration": {
                    "long": {
                        "global": {
                            "mean_edge": 0.0012,
                            "positive_rate": 0.62,
                            "robust_mean_edge": 0.0012,
                            "robust_positive_rate": 0.62,
                            "active_window_ratio": 1.0,
                        },
                        "thresholds": [
                            {
                                "threshold": 0.80,
                                "mean_edge": 0.0025,
                                "positive_rate": 0.69,
                                "robust_mean_edge": 0.0025,
                                "robust_positive_rate": 0.69,
                                "active_window_ratio": 0.25,
                            },
                        ],
                    },
                    "short": {
                        "global": {
                            "mean_edge": -0.0005,
                            "positive_rate": 0.46,
                            "robust_mean_edge": -0.0005,
                            "robust_positive_rate": 0.46,
                            "active_window_ratio": 1.0,
                        },
                        "thresholds": [],
                    },
                },
            },
            long_confidence_threshold=0.80,
            short_confidence_threshold=0.84,
            meta_label_min_edge=0.0008,
            meta_label_edge_margin=0.0002,
            meta_label_exit_edge=0.0,
            meta_label_min_positive_rate=0.55,
            calibration_min_active_window_ratio=0.50,
        )
        obs = np.concatenate(
            [
                features,
                build_agent_state(
                    position=0.0,
                    upnl=0.0,
                    equity_ratio=0.0,
                    drawdown=0.0,
                    rolling_volatility=0.0,
                    turnover_last_step=0.0,
                    flat_steps=10,
                    pos_steps=0,
                ),
            ]
        ).astype(np.float32)

        action, _ = model.predict(obs)
        assert action == ACTION_FLAT

    def test_meta_label_calibration_round_trips(self, tmp_path):
        from models.rl.supervised import SupervisedActionModel, TrendRuleClassifier

        model = SupervisedActionModel(
            scaler=None,
            classifier=TrendRuleClassifier(feature_dim=6, entry_threshold=0.004, neutral_band=0.0015),
            feature_dim=6,
            confidence_threshold=0.80,
            min_hold_steps=8,
            reversal_margin=0.08,
            entry_margin=0.08,
            exit_to_flat_margin=0.05,
            max_hold_steps=32,
            stop_loss_threshold=-0.012,
            drawdown_exit_threshold=0.04,
            trend_alignment_threshold=0.001,
            countertrend_margin=0.08,
            metadata={
                "post_cost_calibration": {
                    "long": {
                        "global": {"mean_edge": 0.0007, "positive_rate": 0.55},
                        "thresholds": [{"threshold": 0.70, "mean_edge": 0.0015, "positive_rate": 0.61}],
                    },
                    "short": {
                        "global": {"mean_edge": -0.0004, "positive_rate": 0.46},
                        "thresholds": [{"threshold": 0.70, "mean_edge": -0.0009, "positive_rate": 0.43}],
                    },
                },
            },
            long_confidence_threshold=0.80,
            short_confidence_threshold=0.84,
            meta_label_min_edge=0.0008,
            meta_label_edge_margin=0.0004,
            meta_label_exit_edge=0.0001,
            meta_label_min_positive_rate=0.47,
            calibration_min_active_window_ratio=0.50,
        )

        saved_path = model.save(tmp_path / "meta_label_model.joblib")
        loaded = SupervisedActionModel.load(saved_path)

        assert loaded.meta_label_min_edge == pytest.approx(0.0008)
        assert loaded.meta_label_edge_margin == pytest.approx(0.0004)
        assert loaded.meta_label_exit_edge == pytest.approx(0.0001)
        assert loaded.meta_label_min_positive_rate == pytest.approx(0.47)
        assert loaded.calibration_min_active_window_ratio == pytest.approx(0.50)
        assert loaded.metadata["post_cost_calibration"]["long"]["thresholds"][0]["mean_edge"] == pytest.approx(0.0015)

    def test_stateful_policy_holds_position_before_min_hold(self):
        from models.rl.environment import build_agent_state
        from models.rl.supervised import ACTION_SHORT, SupervisedActionModel

        class DummyClassifier:
            classes_ = np.array([0, 1, 2], dtype=np.int32)

            def predict_proba(self, x):
                return np.array([[0.20, 0.70, 0.10]], dtype=np.float32)

        model = SupervisedActionModel(
            scaler=None,
            classifier=DummyClassifier(),
            feature_dim=6,
            confidence_threshold=0.34,
            min_hold_steps=16,
            reversal_margin=0.08,
            entry_margin=0.08,
            exit_to_flat_margin=0.05,
            max_hold_steps=64,
            stop_loss_threshold=-0.012,
            drawdown_exit_threshold=0.04,
            trend_alignment_threshold=0.0015,
            countertrend_margin=0.08,
            metadata={},
        )
        obs = np.concatenate(
            [
                np.zeros(6, dtype=np.float32),
                build_agent_state(
                    position=-1.0,
                    upnl=0.0,
                    equity_ratio=0.0,
                    drawdown=0.0,
                    rolling_volatility=0.0,
                    turnover_last_step=0.0,
                    flat_steps=0,
                    pos_steps=5,
                ),
            ]
        ).astype(np.float32)
        action, _ = model.predict(obs)

        assert action == ACTION_SHORT

    def test_extratrees_model_round_trip(self, tmp_path):
        from models.rl.supervised import SupervisedActionModel, train_supervised_action_model

        n = 320
        prices = np.concatenate(
            [
                np.linspace(100.0, 120.0, 110, dtype=np.float32),
                np.linspace(120.0, 120.3, 90, dtype=np.float32),
                np.linspace(120.3, 95.0, 120, dtype=np.float32),
            ]
        )
        horizon = 4
        future_ret = np.zeros(n, dtype=np.float32)
        future_ret[:-horizon] = (prices[horizon:] / prices[:-horizon]) - 1.0
        features = np.column_stack(
            [
                future_ret,
                np.sign(future_ret),
                future_ret ** 2,
                np.roll(future_ret, 1),
                np.roll(future_ret, 2),
                np.linspace(-1.0, 1.0, n, dtype=np.float32),
            ]
        ).astype(np.float32)

        model, meta = train_supervised_action_model(
            feature_array=features,
            prices=prices,
            train_range=(0, 220),
            validation_range=(220, 300),
            model_type="extratrees",
            horizon=horizon,
            min_return_threshold=0.002,
            threshold_quantile=0.40,
            max_train_samples=2000,
            extra_trees_n_estimators=64,
            extra_trees_max_depth=8,
            extra_trees_min_samples_leaf=4,
            min_hold_steps=12,
            reversal_margin=0.05,
            entry_margin=0.07,
            exit_to_flat_margin=0.04,
        )
        model.confidence_threshold = 0.30
        action_before, _ = model.predict(features[240])
        assert action_before in (0, 1, 2)
        assert meta["model_type"] == "extratrees"

        saved_path = model.save(tmp_path / "supervised_model.joblib")
        loaded = SupervisedActionModel.load(saved_path)
        action_after, _ = loaded.predict(features[240])

        assert action_before == action_after
        assert loaded.min_hold_steps == 12
        assert loaded.reversal_margin == pytest.approx(0.05)
        assert loaded.entry_margin == pytest.approx(0.07)
        assert loaded.exit_to_flat_margin == pytest.approx(0.04)
        assert loaded.max_hold_steps == 64
        assert loaded.stop_loss_threshold == pytest.approx(-0.012)
        assert "post_cost_calibration" in loaded.metadata
        assert loaded.meta_label_min_edge == pytest.approx(0.0)
        assert loaded.meta_label_edge_margin == pytest.approx(0.0)

    def test_build_constant_action_model_predicts_flat(self):
        from models.rl.supervised import ACTION_FLAT, build_constant_action_model

        model = build_constant_action_model(feature_dim=6, action=ACTION_FLAT, metadata={"kind": "safe"})
        action, _ = model.predict(np.zeros(6, dtype=np.float32))

        assert action == ACTION_FLAT

    def test_eval_multi_episode_seed_base_is_deterministic(self):
        from models.rl.trainer import RLTrainer

        class AlternatingModel:
            def predict(self, obs, deterministic=True):
                feature_sum = float(np.asarray(obs).sum())
                return (0 if feature_sum < 0 else 1), None

        prices = np.linspace(100.0, 120.0, 400, dtype=np.float32)
        features = np.random.default_rng(42).normal(size=(400, 6)).astype(np.float32)
        trainer = RLTrainer(features, prices, max_episode_steps=32)

        metrics_a = trainer._eval_multi_episode(
            AlternatingModel(),
            n_episodes=4,
            segment_range=(50, 250),
            log_episodes=False,
            seed_base=123,
        )
        metrics_b = trainer._eval_multi_episode(
            AlternatingModel(),
            n_episodes=4,
            segment_range=(50, 250),
            log_episodes=False,
            seed_base=123,
        )

        assert metrics_a["total_return"] == metrics_b["total_return"]
        assert metrics_a["n_trades"] == metrics_b["n_trades"]

    def test_policy_stays_flat_when_edge_over_flat_is_too_small(self):
        from models.rl.supervised import ACTION_FLAT, SupervisedActionModel

        class DummyClassifier:
            classes_ = np.array([0, 1, 2], dtype=np.int32)

            def predict_proba(self, x):
                return np.array([[0.23, 0.32, 0.45]], dtype=np.float32)

        model = SupervisedActionModel(
            scaler=None,
            classifier=DummyClassifier(),
            feature_dim=4,
            confidence_threshold=0.40,
            min_hold_steps=8,
            reversal_margin=0.08,
            entry_margin=0.15,
            exit_to_flat_margin=0.05,
            max_hold_steps=32,
            stop_loss_threshold=-0.012,
            drawdown_exit_threshold=0.04,
            trend_alignment_threshold=0.0015,
            countertrend_margin=0.08,
            metadata={},
        )

        action, _ = model.predict(np.zeros(4, dtype=np.float32))
        assert action == ACTION_FLAT

    def test_policy_exits_to_flat_when_flat_probability_overtakes_position(self):
        from models.rl.environment import build_agent_state
        from models.rl.supervised import ACTION_FLAT, SupervisedActionModel

        class DummyClassifier:
            classes_ = np.array([0, 1, 2], dtype=np.int32)

            def predict_proba(self, x):
                return np.array([[0.10, 0.63, 0.27]], dtype=np.float32)

        model = SupervisedActionModel(
            scaler=None,
            classifier=DummyClassifier(),
            feature_dim=5,
            confidence_threshold=0.38,
            min_hold_steps=4,
            reversal_margin=0.08,
            entry_margin=0.10,
            exit_to_flat_margin=0.12,
            max_hold_steps=32,
            stop_loss_threshold=-0.012,
            drawdown_exit_threshold=0.04,
            trend_alignment_threshold=0.0015,
            countertrend_margin=0.08,
            metadata={},
        )
        obs = np.concatenate(
            [
                np.zeros(5, dtype=np.float32),
                build_agent_state(
                    position=1.0,
                    upnl=0.0,
                    equity_ratio=0.0,
                    drawdown=0.0,
                    rolling_volatility=0.0,
                    turnover_last_step=0.0,
                    flat_steps=0,
                    pos_steps=12,
                ),
            ]
        ).astype(np.float32)

        action, _ = model.predict(obs)
        assert action == ACTION_FLAT

    def test_policy_flattens_on_stop_loss(self):
        from models.rl.environment import build_agent_state
        from models.rl.supervised import ACTION_FLAT, SupervisedActionModel

        class DummyClassifier:
            classes_ = np.array([0, 1, 2], dtype=np.int32)

            def predict_proba(self, x):
                return np.array([[0.08, 0.22, 0.70]], dtype=np.float32)

        model = SupervisedActionModel(
            scaler=None,
            classifier=DummyClassifier(),
            feature_dim=5,
            confidence_threshold=0.38,
            min_hold_steps=4,
            reversal_margin=0.08,
            entry_margin=0.10,
            exit_to_flat_margin=0.12,
            max_hold_steps=32,
            stop_loss_threshold=-0.010,
            drawdown_exit_threshold=0.04,
            trend_alignment_threshold=0.0015,
            countertrend_margin=0.08,
            metadata={},
        )
        obs = np.concatenate(
            [
                np.zeros(5, dtype=np.float32),
                build_agent_state(
                    position=1.0,
                    upnl=-0.02,
                    equity_ratio=-0.01,
                    drawdown=0.01,
                    rolling_volatility=0.0,
                    turnover_last_step=0.0,
                    flat_steps=0,
                    pos_steps=20,
                ),
            ]
        ).astype(np.float32)

        action, _ = model.predict(obs)
        assert action == ACTION_FLAT

    def test_policy_flattens_on_stale_hold(self):
        from models.rl.environment import build_agent_state
        from models.rl.supervised import ACTION_FLAT, SupervisedActionModel

        class DummyClassifier:
            classes_ = np.array([0, 1, 2], dtype=np.int32)

            def predict_proba(self, x):
                return np.array([[0.10, 0.47, 0.43]], dtype=np.float32)

        model = SupervisedActionModel(
            scaler=None,
            classifier=DummyClassifier(),
            feature_dim=20,
            confidence_threshold=0.40,
            min_hold_steps=4,
            reversal_margin=0.08,
            entry_margin=0.10,
            exit_to_flat_margin=0.06,
            max_hold_steps=12,
            stop_loss_threshold=-0.012,
            drawdown_exit_threshold=0.04,
            trend_alignment_threshold=0.0015,
            countertrend_margin=0.08,
            metadata={},
        )
        obs = np.concatenate(
            [
                np.zeros(20, dtype=np.float32),
                build_agent_state(
                    position=1.0,
                    upnl=0.001,
                    equity_ratio=0.0,
                    drawdown=0.0,
                    rolling_volatility=0.0,
                    turnover_last_step=0.0,
                    flat_steps=0,
                    pos_steps=20,
                ),
            ]
        ).astype(np.float32)

        action, _ = model.predict(obs)
        assert action == ACTION_FLAT


class TestBacktest:
    def test_basic_run(self):
        from execution.backtest.runner import BacktestRunner
        prices = 100 + np.cumsum(np.random.randn(500) * 0.5)
        signals = np.zeros(500)
        signals[100:300] = 0.5  # long for a period

        runner = BacktestRunner()
        result = runner.run(prices, signals)
        assert len(result.equity_curve) > 0
        assert "sharpe" in result.metrics
        assert "max_drawdown" in result.metrics
