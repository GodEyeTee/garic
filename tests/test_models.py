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
            opportunity_threshold=10.0,
            flat_penalty_after_steps=99,
            flat_penalty_scale=0.0,
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
