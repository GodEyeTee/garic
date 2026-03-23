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
        features = np.random.randn(n, 346).astype(np.float32)
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        env = CryptoFuturesEnv(features, prices)
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape == (350,)  # 346 + 4 agent-state features

    def test_step_returns(self):
        from models.rl.environment import CryptoFuturesEnv
        n = 100
        features = np.random.randn(n, 346).astype(np.float32)
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
        features = np.random.randn(n, 346).astype(np.float32)
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
            opportunity_cost_scale=0.0,
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
            opportunity_cost_scale=0.0,
        )
        env_short = CryptoFuturesEnv(
            features, prices, max_episode_steps=3,
            monthly_server_cost_usd=0.0, pnl_reward_scale=200.0,
            opportunity_cost_scale=0.0,
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


class TestSupervisedFallback:
    def test_stateful_policy_holds_position_before_min_hold(self):
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
            metadata={},
        )
        obs = np.array([0, 0, 0, 0, 0, 0, -1.0, 0.0, 0.0, 0.05], dtype=np.float32)
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
