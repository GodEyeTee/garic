"""Tests for pipeline data splitting and aggregation helpers."""

import numpy as np
import pandas as pd

from features.builder import FeatureBuilder
from pipeline import (
    _aggregate_ohlcv_15m,
    _aggregate_nautilus_window_summaries,
    _aggregate_validation_metrics,
    _apply_nautilus_target_slice_metrics,
    _build_supervised_search_grid,
    _build_nautilus_frame,
    _derive_calibrated_confidence_grid,
    _is_collapsed_supervised_probe_metrics,
    _prepare_nautilus_segment_frame,
    _combine_supervised_validation_scores,
    _compute_data_ranges,
    _prune_supervised_candidate_pool,
    _promote_nautilus_test_metrics,
    _robust_validation_score,
    _run_nautilus_backtest_segment,
    _score_nautilus_rejected_fallback,
    _score_anchor_supervised_probe_candidate,
    _score_sparse_supervised_watchlist_candidate,
    _select_nautilus_candidate_subset,
    _score_nautilus_summary,
    _trim_supervised_candidate_pool,
    add_naive_forecast,
    backtest_with_model,
)


class TestPipelineHelpers:
    def test_feature_builder_batch_array_is_compact(self):
        df = pd.DataFrame(
            {
                "open": np.linspace(100.0, 130.0, 120),
                "high": np.linspace(101.0, 131.0, 120),
                "low": np.linspace(99.0, 129.0, 120),
                "close": np.linspace(100.5, 130.5, 120),
                "volume": np.ones(120),
            }
        )
        feature_array, ta_slice, micro_slice = FeatureBuilder(lookback=60).build_batch_array(df)
        assert feature_array.shape == (60, 25)
        assert ta_slice.shape[1] == 15
        assert micro_slice.shape[1] == 5

    def test_add_naive_forecast_appends_compact_block_without_overwriting(self):
        base = np.ones((80, 25), dtype=np.float32)
        prices = np.linspace(100.0, 120.0, 80, dtype=np.float32)
        enriched = add_naive_forecast(base, prices)
        assert enriched.shape == (80, 30)
        np.testing.assert_array_equal(enriched[:, :25], base)

    def test_compute_data_ranges_reserves_holdout_tail(self):
        ranges = _compute_data_ranges(100, test_ratio=0.2, validation_ratio_within_train=0.1)
        assert ranges["train"] == (0, 72)
        assert ranges["validation"] == (72, 80)
        assert ranges["test"] == (80, 100)

    def test_robust_validation_score_requires_all_finite(self):
        assert _robust_validation_score([1.0, 2.0, 3.0]) == 2.0 + 0.25 * 1.0
        assert _robust_validation_score([1.0, float("-inf"), 3.0]) == float("-inf")

    def test_derive_calibrated_confidence_grid_prefers_positive_edge_thresholds(self):
        grid = _derive_calibrated_confidence_grid(
            {
                "thresholds": [
                    {"threshold": 0.35, "mean_edge": -0.0004, "positive_rate": 0.44, "count": 80},
                    {
                        "threshold": 0.45,
                        "mean_edge": 0.0008,
                        "positive_rate": 0.58,
                        "robust_mean_edge": 0.0007,
                        "robust_positive_rate": 0.57,
                        "active_window_ratio": 0.75,
                        "count": 64,
                    },
                    {
                        "threshold": 0.55,
                        "mean_edge": 0.0016,
                        "positive_rate": 0.62,
                        "robust_mean_edge": 0.0014,
                        "robust_positive_rate": 0.61,
                        "active_window_ratio": 0.75,
                        "count": 28,
                    },
                ]
            },
            [0.80, 0.84],
            min_edge=0.0002,
            min_positive_rate=0.47,
            respect_fallback_floor=False,
        )

        assert 0.45 in grid
        assert 0.55 in grid

    def test_derive_calibrated_confidence_grid_filters_sparse_window_support(self):
        grid = _derive_calibrated_confidence_grid(
            {
                "thresholds": [
                    {
                        "threshold": 0.50,
                        "mean_edge": 0.0018,
                        "positive_rate": 0.66,
                        "robust_mean_edge": 0.0017,
                        "robust_positive_rate": 0.65,
                        "active_window_ratio": 0.25,
                        "count": 40,
                    },
                    {
                        "threshold": 0.60,
                        "mean_edge": 0.0011,
                        "positive_rate": 0.58,
                        "robust_mean_edge": 0.0010,
                        "robust_positive_rate": 0.57,
                        "active_window_ratio": 0.75,
                        "count": 26,
                    },
                ]
            },
            [0.70, 0.75],
            min_edge=0.0002,
            min_positive_rate=0.47,
            min_active_window_ratio=0.50,
            respect_fallback_floor=False,
        )

        assert 0.60 in grid
        assert 0.50 not in grid

    def test_derive_calibrated_confidence_grid_respects_fallback_floor(self):
        grid = _derive_calibrated_confidence_grid(
            {
                "thresholds": [
                    {
                        "threshold": 0.30,
                        "mean_edge": 0.0030,
                        "positive_rate": 0.70,
                        "robust_mean_edge": 0.0028,
                        "robust_positive_rate": 0.68,
                        "active_window_ratio": 0.90,
                        "count": 120,
                    },
                    {
                        "threshold": 0.55,
                        "mean_edge": 0.0012,
                        "positive_rate": 0.58,
                        "robust_mean_edge": 0.0011,
                        "robust_positive_rate": 0.57,
                        "active_window_ratio": 0.75,
                        "count": 36,
                    },
                ]
            },
            [0.55, 0.60, 0.65],
            min_edge=0.0002,
            min_positive_rate=0.47,
            min_active_window_ratio=0.50,
        )

        assert 0.30 not in grid
        assert 0.55 in grid
        assert 0.60 in grid

    def test_derive_calibrated_confidence_grid_preserves_anchor_and_top_fallback_thresholds(self):
        grid = _derive_calibrated_confidence_grid(
            {
                "thresholds": [
                    {
                        "threshold": 0.55,
                        "mean_edge": 0.0011,
                        "positive_rate": 0.58,
                        "robust_mean_edge": 0.0010,
                        "robust_positive_rate": 0.57,
                        "active_window_ratio": 0.75,
                        "count": 32,
                    },
                    {
                        "threshold": 0.60,
                        "mean_edge": 0.0014,
                        "positive_rate": 0.61,
                        "robust_mean_edge": 0.0013,
                        "robust_positive_rate": 0.60,
                        "active_window_ratio": 0.75,
                        "count": 24,
                    },
                ]
            },
            [0.55, 0.60, 0.70, 0.82, 0.84],
            min_edge=0.0002,
            min_positive_rate=0.47,
            min_active_window_ratio=0.50,
            anchor_thresholds=[0.82],
            preserve_top_fallback_count=2,
        )

        assert 0.82 in grid
        assert 0.84 in grid
        assert 0.60 in grid

    def test_build_supervised_search_grid_includes_exploration_and_anchor_thresholds(self):
        grid = _build_supervised_search_grid(
            [0.80, 0.82, 0.84],
            exploration_grid=[0.72, 0.76, 0.80],
            anchor_grid=[0.82, 0.86],
        )

        assert grid == [0.72, 0.76, 0.80, 0.82, 0.84, 0.86]

    def test_aggregate_validation_metrics_is_conservative(self):
        metrics = [
            {
                "total_return": 0.01,
                "gross_total_return": 0.015,
                "flat_ratio": 0.98,
                "eval_dominant_action_ratio": 0.97,
                "max_drawdown": 0.03,
                "avg_trades_per_episode": 2.0,
            },
            {
                "total_return": 0.02,
                "gross_total_return": 0.025,
                "flat_ratio": 0.96,
                "eval_dominant_action_ratio": 0.95,
                "max_drawdown": 0.02,
                "avg_trades_per_episode": 4.0,
            },
        ]
        out = _aggregate_validation_metrics(metrics, [0.5, 1.5])
        assert out["total_return"] == 0.015
        assert out["gross_total_return"] == 0.02
        assert out["flat_ratio"] == 0.98
        assert out["eval_dominant_action_ratio"] == 0.97
        assert out["max_drawdown"] == 0.03
        assert out["avg_trades_per_episode"] == 3.0
        assert out["validation_score_median"] == 1.0
        assert out["validation_score_worst"] == 0.5
        assert out["validation_score_robust"] == 1.0 + 0.25 * 0.5

    def test_promote_nautilus_test_metrics_overrides_top_level_display_fields(self):
        env_metrics = {
            "total_return": -0.0073,
            "gross_total_return": -0.0055,
            "bh_eval_return": -0.0322,
            "outperformance_vs_bh": 0.0249,
            "n_trades": 60.0,
            "model_family": "supervised_logreg",
            "model_path": "checkpoints/env_proxy.joblib",
            "selection_mode": "watchlist",
            "selection_reason": "",
        }
        nautilus_summary = {
            "total_return": 0.0156,
            "gross_total_return": 0.0249,
            "bh_eval_return": -0.1510,
            "outperformance_vs_bh": 0.1666,
            "n_trades": 12,
            "n_wins": 2,
            "n_losses": 10,
            "server_cost_paid": 92.60,
            "flat_ratio": 0.9651,
            "position_ratio": 0.0349,
            "eval_action_entropy": 0.1377,
            "eval_dominant_action": 1.0,
            "eval_dominant_action_ratio": 0.9651,
            "eval_short_actions": 0.0,
            "eval_flat_actions": 2574.0,
            "eval_long_actions": 93.0,
            "max_drawdown": -0.0373,
            "action_counts": {"short": 0, "flat": 2574, "long": 93},
            "stats_returns": {
                "Sharpe Ratio (252 days)": 1.4630,
                "Sortino Ratio (252 days)": 4.1297,
            },
            "model_path": "checkpoints/rl_agent_supervised.joblib",
            "model_family": "supervised_logreg",
        }

        promoted = _promote_nautilus_test_metrics(env_metrics, nautilus_summary)

        assert promoted["metrics_source"] == "nautilus_test"
        assert promoted["total_return"] == nautilus_summary["total_return"]
        assert promoted["gross_total_return"] == nautilus_summary["gross_total_return"]
        assert promoted["bh_eval_return"] == nautilus_summary["bh_eval_return"]
        assert promoted["outperformance_vs_bh"] == nautilus_summary["outperformance_vs_bh"]
        assert promoted["n_trades"] == nautilus_summary["n_trades"]
        assert promoted["sharpe"] == 1.4630
        assert promoted["sortino"] == 4.1297
        assert promoted["n_longs"] == 12.0
        assert promoted["n_shorts"] == 0.0
        assert promoted["env_eval_metrics"]["total_return"] == env_metrics["total_return"]

    def test_combine_supervised_validation_scores_softens_one_flat_window(self):
        class DummyTrainer:
            def score_candidate(self, metrics, **kwargs):
                return float(metrics.get("soft_score", -0.5))

        combined, walkforward_soft, details = _combine_supervised_validation_scores(
            1.40,
            {
                "soft_score": -0.90,
                "walkforward_active_window_ratio": 0.50,
                "walkforward_min_total_return": -0.015,
                "walkforward_min_gross_total_return": 0.0,
                "walkforward_min_alpha": -0.20,
                "walkforward_worst_dominant_action_ratio": 1.0,
                "walkforward_median_trades": 2.0,
                "walkforward_positive_net_ratio": 0.50,
                "walkforward_positive_alpha_ratio": 0.50,
            },
            DummyTrainer(),
            max_dominant_action_ratio=0.99,
            min_avg_trades_per_episode=0.5,
            min_action_entropy=0.0,
        )

        assert np.isfinite(combined)
        assert walkforward_soft == -0.90
        assert details["walkforward_active_window_ratio"] == 0.50

    def test_combine_supervised_validation_scores_rejects_fully_inactive_walkforward(self):
        class DummyTrainer:
            def score_candidate(self, metrics, **kwargs):
                return -0.5

        combined, _, _ = _combine_supervised_validation_scores(
            1.20,
            {
                "walkforward_active_window_ratio": 0.0,
                "walkforward_median_trades": 0.0,
                "walkforward_min_total_return": -0.015,
            },
            DummyTrainer(),
            max_dominant_action_ratio=0.99,
            min_avg_trades_per_episode=0.5,
            min_action_entropy=0.0,
        )

        assert combined == float("-inf")

    def test_aggregate_ohlcv_15m_vectorized_values(self):
        df = pd.DataFrame(
            {
                "open_time": pd.date_range("2026-01-01", periods=30, freq="min"),
                "open": list(range(1, 31)),
                "high": list(range(2, 32)),
                "low": list(range(0, 30)),
                "close": list(range(1, 31)),
                "volume": [1.0] * 30,
            }
        )

        agg = _aggregate_ohlcv_15m(df, period=15)
        assert len(agg) == 2
        assert agg.iloc[0]["open"] == 1
        assert agg.iloc[0]["high"] == 16
        assert agg.iloc[0]["low"] == 0
        assert agg.iloc[0]["close"] == 15
        assert agg.iloc[0]["volume"] == 15.0
        assert agg.iloc[1]["open"] == 16
        assert agg.iloc[1]["close"] == 30

    def test_backtest_uses_caller_ranges_when_provided(self):
        prices = np.linspace(100.0, 200.0, 100, dtype=np.float32)
        features = np.zeros((100, 25), dtype=np.float32)
        config = {"trading": {"min_trade_pct": 0.05}, "training": {"validation": {}}}
        ranges = {"train": (0, 60), "validation": (60, 80), "test": (10, 20)}

        result = backtest_with_model(None, features, prices, config, data_ranges=ranges)
        assert result["test_range"] == [10, 20]

    def test_build_nautilus_frame_from_ohlcv_arrays(self):
        prices = np.array([100.0, 101.0, 102.0], dtype=np.float32)
        ohlcv = np.array(
            [
                [99.5, 100.5, 99.0, 100.0],
                [100.0, 101.5, 99.8, 101.0],
                [101.0, 102.5, 100.5, 102.0],
            ],
            dtype=np.float32,
        )
        frame = _build_nautilus_frame(prices, ohlcv)
        assert list(frame.columns) == ["open_time", "open", "high", "low", "close", "volume"]
        assert len(frame) == 3
        assert frame["close"].tolist() == [100.0, 101.0, 102.0]
        assert frame["volume"].tolist() == [1.0, 1.0, 1.0]

    def test_score_nautilus_summary_rejects_no_trade_collapse(self):
        score = _score_nautilus_summary(
            {
                "n_trades": 0,
                "total_return": -0.0069,
                "outperformance_vs_bh": -0.02,
                "win_rate": 0.0,
                "flat_ratio": 1.0,
                "eval_dominant_action_ratio": 1.0,
                "eval_action_entropy": 0.0,
                "max_drawdown": -0.0069,
            }
        )
        assert score == float("-inf")

    def test_score_nautilus_summary_rejects_execution_loser(self):
        score = _score_nautilus_summary(
            {
                "n_trades": 3,
                "total_return": -0.008,
                "gross_total_return": -0.004,
                "outperformance_vs_bh": 0.01,
                "win_rate": 0.66,
                "flat_ratio": 0.97,
                "eval_dominant_action_ratio": 0.99,
                "eval_action_entropy": 0.10,
                "max_drawdown": -0.03,
                "nautilus_active_window_ratio": 1.0,
                "nautilus_min_total_return": -0.012,
                "nautilus_min_gross_total_return": -0.006,
            }
        )
        assert score == float("-inf")

    def test_score_nautilus_summary_rejects_low_active_window_ratio(self):
        score = _score_nautilus_summary(
            {
                "n_trades": 5,
                "total_return": -0.002,
                "gross_total_return": -0.001,
                "outperformance_vs_bh": 0.01,
                "win_rate": 0.6,
                "flat_ratio": 0.92,
                "eval_dominant_action_ratio": 0.90,
                "eval_action_entropy": 0.25,
                "max_drawdown": -0.03,
                "nautilus_active_window_ratio": 1.0 / 3.0,
                "nautilus_min_total_return": -0.006,
                "nautilus_min_gross_total_return": -0.004,
            }
        )
        assert score == float("-inf")

    def test_score_nautilus_summary_allows_sparse_active_override(self):
        score = _score_nautilus_summary(
            {
                "n_trades": 2,
                "total_return": -0.0012,
                "gross_total_return": 0.0,
                "outperformance_vs_bh": -0.0047,
                "win_rate": 0.5,
                "flat_ratio": 1041.0 / 1067.0,
                "position_ratio": 26.0 / 1067.0,
                "eval_action_entropy": 0.104,
                "eval_dominant_action_ratio": 1041.0 / 1067.0,
                "max_drawdown": -0.0084,
                "nautilus_active_window_ratio": 1.0 / 3.0,
                "nautilus_min_total_return": -0.0012,
                "nautilus_min_gross_total_return": 0.0,
                "nautilus_min_alpha": -0.0077,
                "trade_density": 2.0 / 1067.0,
                "stats_returns": {"Sharpe Ratio (252 days)": 2.29},
            }
        )
        assert np.isfinite(score)

    def test_run_nautilus_backtest_segment_passes_aligned_slice(self, monkeypatch):
        captured = {}

        def fake_run_backtest_frame(frame_15m, **kwargs):
            captured["frame_len"] = len(frame_15m)
            captured["first_close"] = float(frame_15m["close"].iloc[0])
            captured["kwargs"] = kwargs
            return {
                "n_trades": 3,
                "total_return": 0.02,
                "gross_total_return": 0.025,
                "outperformance_vs_bh": 0.01,
                "win_rate": 0.66,
                "flat_ratio": 0.2,
                "eval_dominant_action_ratio": 0.5,
                "eval_action_entropy": 0.8,
                "max_drawdown": -0.04,
            }

        monkeypatch.setattr("execution.nautilus.backtest_runner.run_backtest_frame", fake_run_backtest_frame)

        frame = pd.DataFrame(
            {
                "open_time": pd.date_range("2026-01-01", periods=400, freq="15min", tz="UTC"),
                "open": np.linspace(100.0, 150.0, 400),
                "high": np.linspace(101.0, 151.0, 400),
                "low": np.linspace(99.0, 149.0, 400),
                "close": np.linspace(100.5, 150.5, 400),
                "volume": np.ones(400),
            }
        )
        config = {
            "_active_symbol": "BTCUSDT",
            "trading": {"monthly_server_cost_usd": 100.0, "periods_per_day": 96, "leverage": 1.0},
            "training": {
                "nautilus_validation": {
                    "history_bars": 160,
                    "request_history_days": 3,
                    "trade_size_pct_of_equity": 1.0,
                    "initial_balance_usdt": 10000.0,
                    "leverage": 1.0,
                }
            },
            "data": {"pairs": ["BTCUSDT"]},
        }

        summary = _run_nautilus_backtest_segment(
            model_path="checkpoints/rl_agent_supervised.joblib",
            frame_15m=frame,
            segment_range=(200, 360),
            config=config,
            label="validation_supervised_logreg",
        )

        assert captured["frame_len"] == 319
        assert captured["first_close"] == float(frame["close"].iloc[41])
        assert captured["kwargs"]["symbol"] == "BTCUSDT"
        assert float(captured["kwargs"]["trade_size"]) > 0.0
        assert summary["segment_range"] == [200, 360]
        assert summary["segment_input_range"] == [41, 360]
        assert summary["segment_score_start_index"] == 159
        assert summary["segment_target_bars"] == 160

    def test_prepare_nautilus_segment_frame_adds_history_warmup(self):
        frame = pd.DataFrame(
            {
                "open_time": pd.date_range("2026-01-01", periods=500, freq="15min", tz="UTC"),
                "open": np.linspace(100.0, 200.0, 500),
                "high": np.linspace(101.0, 201.0, 500),
                "low": np.linspace(99.0, 199.0, 500),
                "close": np.linspace(100.5, 200.5, 500),
                "volume": np.ones(500),
            }
        )
        prepared, target, padded_start, target_start, score_start_index = _prepare_nautilus_segment_frame(
            frame,
            (220, 380),
            history_bars=160,
        )
        assert len(target) == 160
        assert len(prepared) == 319
        assert padded_start == 61
        assert target_start == 220
        assert score_start_index == 159

    def test_apply_nautilus_target_slice_metrics_uses_target_bars_for_costs(self):
        target = pd.DataFrame(
            {
                "close": [100.0, 102.0, 103.0, 104.0],
            }
        )
        adjusted = _apply_nautilus_target_slice_metrics(
            {
                "gross_total_return": 0.05,
                "total_return": 0.01,
            },
            target_frame=target,
            target_start=10,
            target_end=14,
            prepared_start=0,
            score_start_index=10,
            initial_balance=10_000.0,
            monthly_server_cost_usd=100.0,
            periods_per_day=96,
        )
        expected_cost = 100.0 * (4.0 / (96.0 * 30.0))
        assert np.isclose(adjusted["server_cost_paid"], expected_cost)
        assert adjusted["gross_total_return"] == 0.05
        assert np.isclose(adjusted["total_return"], 0.05 - (expected_cost / 10_000.0))
        assert np.isclose(adjusted["bh_eval_return"], 0.04)
        assert adjusted["segment_range"] == [10, 14]
        assert adjusted["segment_input_range"] == [0, 14]
        assert np.isclose(adjusted["trade_density"], 0.0)

    def test_apply_nautilus_target_slice_metrics_normalizes_trade_counts_from_actions(self):
        target = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0, 104.0]})
        adjusted = _apply_nautilus_target_slice_metrics(
            {
                "gross_total_return": 0.01,
                "total_return": 0.005,
                "n_trades": 0,
                "n_wins": 1,
                "n_losses": 1,
                "total_orders": 4,
                "total_positions": 2,
                "action_counts": {"short": 0, "flat": 3, "long": 2},
            },
            target_frame=target,
            target_start=20,
            target_end=25,
            prepared_start=0,
            score_start_index=20,
            initial_balance=10_000.0,
            monthly_server_cost_usd=100.0,
            periods_per_day=96,
        )
        assert adjusted["n_trades"] == 2
        assert adjusted["total_positions"] == 2
        assert adjusted["total_orders"] == 4
        assert np.isclose(adjusted["flat_ratio"], 3.0 / 5.0)
        assert np.isclose(adjusted["position_ratio"], 2.0 / 5.0)
        assert np.isclose(adjusted["trade_density"], 2.0 / 5.0)
        assert adjusted["eval_long_actions"] == 2.0
        assert adjusted["win_rate"] == 0.5

    def test_aggregate_nautilus_window_summaries_is_conservative(self):
        summaries = [
            {
                "status": "COMPLETE",
                "total_return": 0.01,
                "gross_total_return": 0.02,
                "outperformance_vs_bh": 0.03,
                "max_drawdown": -0.04,
                "n_trades": 4,
                "win_rate": 0.60,
                "flat_ratio": 0.95,
                "position_ratio": 0.05,
                "eval_action_entropy": 0.10,
                "eval_dominant_action_ratio": 0.95,
                "trade_density": 0.01,
            },
            {
                "status": "COMPLETE",
                "total_return": -0.02,
                "gross_total_return": -0.01,
                "outperformance_vs_bh": -0.04,
                "max_drawdown": -0.06,
                "n_trades": 12,
                "win_rate": 0.40,
                "flat_ratio": 0.80,
                "position_ratio": 0.20,
                "eval_action_entropy": 0.30,
                "eval_dominant_action_ratio": 0.80,
                "trade_density": 0.08,
            },
        ]
        combined, robust_score = _aggregate_nautilus_window_summaries(
            summaries,
            [1.0, -0.5],
            [(100, 200), (200, 300)],
        )
        assert np.isclose(robust_score, 0.0)
        assert combined["nautilus_window_count"] == 2
        assert np.isclose(combined["nautilus_score_robust"], 0.0)
        assert np.isclose(combined["nautilus_min_total_return"], -0.02)
        assert np.isclose(combined["nautilus_min_gross_total_return"], -0.01)
        assert np.isclose(combined["nautilus_min_alpha"], -0.04)
        assert np.isclose(combined["nautilus_active_window_ratio"], 1.0)
        assert np.isclose(combined["trade_density"], 0.045)
        assert combined["nautilus_windows"] == [[100, 200], [200, 300]]

    def test_aggregate_nautilus_window_summaries_preserves_sparse_active_window_metrics(self):
        summaries = [
            {
                "status": "COMPLETE",
                "segment_target_bars": 355,
                "n_trades": 0,
                "total_orders": 0,
                "total_positions": 0,
                "n_wins": 0,
                "n_losses": 0,
                "action_counts": {"short": 0, "flat": 355, "long": 0},
                "flat_ratio": 1.0,
                "position_ratio": 0.0,
                "eval_action_entropy": 0.0,
                "eval_dominant_action_ratio": 1.0,
                "gross_total_return": 0.0,
                "total_return": -0.0012,
                "outperformance_vs_bh": 0.01,
            },
            {
                "status": "COMPLETE",
                "segment_target_bars": 355,
                "n_trades": 0,
                "total_orders": 0,
                "total_positions": 0,
                "n_wins": 0,
                "n_losses": 0,
                "action_counts": {"short": 0, "flat": 355, "long": 0},
                "flat_ratio": 1.0,
                "position_ratio": 0.0,
                "eval_action_entropy": 0.0,
                "eval_dominant_action_ratio": 1.0,
                "gross_total_return": 0.0,
                "total_return": -0.0012,
                "outperformance_vs_bh": -0.01,
            },
            {
                "status": "COMPLETE",
                "segment_target_bars": 357,
                "n_trades": 0,
                "total_orders": 4,
                "total_positions": 2,
                "n_wins": 1,
                "n_losses": 1,
                "action_counts": {"short": 0, "flat": 331, "long": 26},
                "flat_ratio": 1.0,
                "position_ratio": 0.0,
                "eval_action_entropy": 0.0,
                "eval_dominant_action_ratio": 1.0,
                "gross_total_return": 0.0069,
                "total_return": 0.0043,
                "outperformance_vs_bh": 0.1250,
            },
        ]
        combined, robust_score = _aggregate_nautilus_window_summaries(
            summaries,
            [float("-inf"), float("-inf"), 1.35],
            [(9599, 9954), (9954, 10309), (10309, 10666)],
        )
        expected_score = (1.35 + (0.5 * 1.35)) - ((0.75 - (1.0 / 3.0)) * 6.0)
        assert np.isclose(robust_score, expected_score)
        assert combined["n_trades"] == 2
        assert combined["total_orders"] == 4
        assert combined["total_positions"] == 2
        assert combined["n_wins"] == 1
        assert combined["n_losses"] == 1
        assert np.isclose(combined["win_rate"], 0.5)
        assert combined["action_counts"] == {"short": 0, "flat": 1041, "long": 26}
        assert np.isclose(combined["flat_ratio"], 1041.0 / 1067.0)
        assert np.isclose(combined["position_ratio"], 26.0 / 1067.0)
        assert np.isclose(combined["trade_density"], 2.0 / 1067.0)
        assert combined["eval_long_actions"] == 26.0
        assert combined["nautilus_window_count"] == 3
        assert np.isclose(combined["nautilus_active_window_ratio"], 1.0 / 3.0)

    def test_select_nautilus_candidate_subset_keeps_next_stricter_short_candidate(self):
        candidates = [
            {
                "candidate_name": "best_env",
                "model_path": "a.joblib",
                "supervised_long_confidence_threshold": 0.80,
                "supervised_short_confidence_threshold": 0.55,
                "env_validation_score": 1.5,
                "env_validation_metrics": {
                    "walkforward_selection_score": 1.0,
                    "walkforward_active_window_ratio": 0.75,
                    "total_return": 0.02,
                    "gross_total_return": 0.03,
                    "avg_trades_per_episode": 1.0,
                    "supervised_validation_brier": 0.7,
                },
            },
            {
                "candidate_name": "best_gross",
                "model_path": "b.joblib",
                "supervised_long_confidence_threshold": 0.80,
                "supervised_short_confidence_threshold": 0.65,
                "env_validation_score": 1.1,
                "env_validation_metrics": {
                    "walkforward_selection_score": 0.8,
                    "walkforward_active_window_ratio": 0.25,
                    "walkforward_min_gross_total_return": 0.001,
                    "total_return": 0.01,
                    "gross_total_return": 0.05,
                    "avg_trades_per_episode": 1.0,
                    "supervised_validation_brier": 0.6,
                },
            },
            {
                "candidate_name": "best_worst_case",
                "model_path": "c.joblib",
                "supervised_long_confidence_threshold": 0.82,
                "supervised_short_confidence_threshold": 0.70,
                "env_validation_score": 1.0,
                "env_validation_metrics": {
                    "walkforward_selection_score": 0.7,
                    "walkforward_active_window_ratio": 0.0,
                    "walkforward_min_total_return": 0.02,
                    "walkforward_min_gross_total_return": 0.02,
                    "total_return": 0.015,
                    "gross_total_return": 0.02,
                    "avg_trades_per_episode": 2.0,
                    "supervised_validation_brier": 0.65,
                },
            },
            {
                "candidate_name": "conservative_high_conf",
                "model_path": "d.joblib",
                "supervised_long_confidence_threshold": 0.86,
                "supervised_short_confidence_threshold": 0.88,
                "env_validation_score": 0.7,
                "env_validation_metrics": {
                    "walkforward_selection_score": 0.3,
                    "total_return": 0.012,
                    "gross_total_return": 0.018,
                    "avg_trades_per_episode": 1.2,
                    "supervised_validation_brier": 0.63,
                },
            },
            {
                "candidate_name": "next_stricter_short",
                "model_path": "e.joblib",
                "supervised_long_confidence_threshold": 0.78,
                "supervised_short_confidence_threshold": 0.60,
                "env_validation_score": 1.05,
                "env_validation_metrics": {
                    "walkforward_selection_score": 0.85,
                    "walkforward_active_window_ratio": 0.75,
                    "walkforward_min_gross_total_return": 0.004,
                    "total_return": 0.011,
                    "gross_total_return": 0.015,
                    "avg_trades_per_episode": 4.0,
                    "supervised_validation_brier": 0.62,
                },
            },
        ]
        selected = _select_nautilus_candidate_subset(candidates, limit=3)
        names = [item["candidate_name"] for item in selected]
        assert "best_env" in names
        assert "next_stricter_short" in names
        assert len(names) == 3

    def test_select_nautilus_candidate_subset_keeps_conservative_high_conf_candidate(self):
        candidates = [
            {
                "candidate_name": "best_env",
                "model_path": "a.joblib",
                "supervised_long_confidence_threshold": 0.80,
                "supervised_short_confidence_threshold": 0.55,
                "env_validation_score": 1.5,
                "env_validation_metrics": {
                    "walkforward_selection_score": 1.0,
                    "walkforward_active_window_ratio": 0.75,
                    "total_return": 0.02,
                    "gross_total_return": 0.03,
                    "avg_trades_per_episode": 3.0,
                    "supervised_validation_brier": 0.7,
                },
            },
            {
                "candidate_name": "conservative_high_conf",
                "model_path": "b.joblib",
                "supervised_long_confidence_threshold": 0.84,
                "supervised_short_confidence_threshold": 0.84,
                "env_validation_score": 0.9,
                "env_validation_metrics": {
                    "walkforward_selection_score": 0.6,
                    "walkforward_active_window_ratio": 0.50,
                    "total_return": 0.01,
                    "gross_total_return": 0.02,
                    "avg_trades_per_episode": 1.0,
                    "supervised_validation_brier": 0.62,
                },
            },
            {
                "candidate_name": "mid_conf",
                "model_path": "c.joblib",
                "supervised_long_confidence_threshold": 0.75,
                "supervised_short_confidence_threshold": 0.60,
                "env_validation_score": 1.1,
                "env_validation_metrics": {
                    "walkforward_selection_score": 0.7,
                    "walkforward_active_window_ratio": 0.75,
                    "total_return": 0.012,
                    "gross_total_return": 0.018,
                    "avg_trades_per_episode": 5.0,
                    "supervised_validation_brier": 0.63,
                },
            },
        ]

        selected = _select_nautilus_candidate_subset(candidates, limit=3)
        names = [item["candidate_name"] for item in selected]
        assert "best_env" in names
        assert "conservative_high_conf" in names

    def test_select_nautilus_candidate_subset_keeps_best_active_sparse_candidate(self):
        candidates = [
            {
                "candidate_name": "top_score",
                "model_path": "a.joblib",
                "supervised_long_confidence_threshold": 0.70,
                "supervised_short_confidence_threshold": 0.60,
                "env_validation_score": 1.95,
                "env_validation_metrics": {
                    "walkforward_selection_score": 1.20,
                    "walkforward_active_window_ratio": 0.25,
                    "walkforward_min_gross_total_return": -0.001,
                    "total_return": 0.012,
                    "gross_total_return": 0.018,
                    "avg_trades_per_episode": 1.0,
                    "supervised_validation_brier": 0.68,
                },
            },
            {
                "candidate_name": "best_active_sparse",
                "model_path": "b.joblib",
                "supervised_long_confidence_threshold": 0.70,
                "supervised_short_confidence_threshold": 0.55,
                "env_validation_score": 0.65,
                "env_validation_metrics": {
                    "walkforward_selection_score": 0.60,
                    "walkforward_active_window_ratio": 0.75,
                    "walkforward_min_gross_total_return": 0.002,
                    "total_return": 0.006,
                    "gross_total_return": 0.012,
                    "avg_trades_per_episode": 4.0,
                    "supervised_validation_brier": 0.64,
                },
            },
            {
                "candidate_name": "next_stricter_short",
                "model_path": "c.joblib",
                "supervised_long_confidence_threshold": 0.72,
                "supervised_short_confidence_threshold": 0.70,
                "env_validation_score": 1.10,
                "env_validation_metrics": {
                    "walkforward_selection_score": 0.75,
                    "walkforward_active_window_ratio": 0.50,
                    "walkforward_min_gross_total_return": 0.001,
                    "total_return": 0.007,
                    "gross_total_return": 0.010,
                    "avg_trades_per_episode": 3.0,
                    "supervised_validation_brier": 0.63,
                },
            },
            {
                "candidate_name": "conservative_high_conf",
                "model_path": "d.joblib",
                "supervised_long_confidence_threshold": 0.84,
                "supervised_short_confidence_threshold": 0.84,
                "env_validation_score": 0.90,
                "env_validation_metrics": {
                    "walkforward_selection_score": 0.40,
                    "walkforward_active_window_ratio": 0.0,
                    "walkforward_min_gross_total_return": 0.0,
                    "total_return": 0.004,
                    "gross_total_return": 0.006,
                    "avg_trades_per_episode": 1.0,
                    "supervised_validation_brier": 0.62,
                },
            },
            {
                "candidate_name": "extra_candidate",
                "model_path": "e.joblib",
                "supervised_long_confidence_threshold": 0.78,
                "supervised_short_confidence_threshold": 0.78,
                "env_validation_score": 0.85,
                "env_validation_metrics": {
                    "walkforward_selection_score": 0.45,
                    "walkforward_active_window_ratio": 0.20,
                    "walkforward_min_gross_total_return": -0.002,
                    "total_return": 0.002,
                    "gross_total_return": 0.004,
                    "avg_trades_per_episode": 2.0,
                    "supervised_validation_brier": 0.66,
                },
            },
        ]

        selected = _select_nautilus_candidate_subset(candidates, limit=4)
        names = [item["candidate_name"] for item in selected]
        assert "top_score" in names
        assert "best_active_sparse" in names
        assert "next_stricter_short" in names
        assert "conservative_high_conf" in names

    def test_select_nautilus_candidate_subset_keeps_low_confidence_explorer(self):
        candidates = [
            {
                "candidate_name": "baseline",
                "model_path": "a.joblib",
                "supervised_long_confidence_threshold": 0.84,
                "supervised_short_confidence_threshold": 0.82,
                "env_validation_score": 0.80,
                "env_validation_metrics": {
                    "walkforward_selection_score": 0.60,
                    "walkforward_active_window_ratio": 0.25,
                    "walkforward_min_gross_total_return": 0.0,
                    "total_return": 0.002,
                    "gross_total_return": 0.004,
                    "avg_trades_per_episode": 1.0,
                    "supervised_validation_brier": 0.70,
                },
            },
            {
                "candidate_name": "low_short_explorer",
                "model_path": "b.joblib",
                "supervised_long_confidence_threshold": 0.80,
                "supervised_short_confidence_threshold": 0.74,
                "env_validation_score": 0.45,
                "env_validation_metrics": {
                    "walkforward_selection_score": 0.25,
                    "walkforward_active_window_ratio": 0.75,
                    "walkforward_min_gross_total_return": 0.001,
                    "total_return": 0.003,
                    "gross_total_return": 0.006,
                    "avg_trades_per_episode": 4.0,
                    "supervised_validation_brier": 0.68,
                },
            },
            {
                "candidate_name": "high_short_conservative",
                "model_path": "c.joblib",
                "supervised_long_confidence_threshold": 0.84,
                "supervised_short_confidence_threshold": 0.86,
                "env_validation_score": 0.50,
                "env_validation_metrics": {
                    "walkforward_selection_score": 0.35,
                    "walkforward_active_window_ratio": 0.25,
                    "walkforward_min_gross_total_return": 0.0,
                    "total_return": 0.001,
                    "gross_total_return": 0.002,
                    "avg_trades_per_episode": 1.0,
                    "supervised_validation_brier": 0.69,
                },
            },
        ]

        selected = _select_nautilus_candidate_subset(candidates, limit=3)
        names = [item["candidate_name"] for item in selected]
        assert "low_short_explorer" in names

    def test_select_nautilus_candidate_subset_keeps_best_of_distinct_family(self):
        candidates = [
            {
                "family": "supervised_logreg",
                "candidate_name": "logreg_top",
                "model_path": "a.joblib",
                "supervised_long_confidence_threshold": 0.82,
                "supervised_short_confidence_threshold": 0.76,
                "env_validation_score": 1.20,
                "env_validation_metrics": {
                    "walkforward_selection_score": 0.90,
                    "walkforward_active_window_ratio": 0.25,
                    "gross_total_return": 0.004,
                    "avg_trades_per_episode": 1.0,
                    "supervised_validation_brier": 0.62,
                },
            },
            {
                "family": "supervised_trend_rule",
                "candidate_name": "trend_family_candidate",
                "model_path": "b.joblib",
                "supervised_long_confidence_threshold": 0.78,
                "supervised_short_confidence_threshold": 0.72,
                "env_validation_score": 0.95,
                "env_validation_metrics": {
                    "walkforward_selection_score": 0.70,
                    "walkforward_active_window_ratio": 0.50,
                    "gross_total_return": 0.006,
                    "avg_trades_per_episode": 2.0,
                    "supervised_validation_brier": 0.66,
                },
            },
            {
                "family": "supervised_logreg",
                "candidate_name": "logreg_second",
                "model_path": "c.joblib",
                "supervised_long_confidence_threshold": 0.84,
                "supervised_short_confidence_threshold": 0.80,
                "env_validation_score": 1.10,
                "env_validation_metrics": {
                    "walkforward_selection_score": 0.85,
                    "walkforward_active_window_ratio": 0.10,
                    "gross_total_return": 0.003,
                    "avg_trades_per_episode": 1.0,
                    "supervised_validation_brier": 0.60,
                },
            },
        ]

        selected = _select_nautilus_candidate_subset(candidates, limit=2)
        names = [item["candidate_name"] for item in selected]
        assert "logreg_top" in names
        assert "trend_family_candidate" in names

    def test_prune_supervised_candidate_pool_preserves_threshold_variants(self):
        candidates = [
            {
                "meta": {"model_type": "logreg"},
                "long_confidence": 0.84,
                "short_confidence": 0.76,
                "score": 0.10,
                "priority": 0,
                "validation_metrics": {
                    "eval_long_actions": 0.0,
                    "eval_short_actions": 0.0,
                    "walkforward_active_window_ratio": 0.25,
                    "avg_trades_per_episode": 0.0,
                    "eval_dominant_action_ratio": 1.0,
                    "gross_total_return": 0.0,
                    "total_return": 0.0,
                    "supervised_validation_brier": 0.70,
                },
            },
            {
                "meta": {"model_type": "logreg"},
                "long_confidence": 0.84,
                "short_confidence": 0.82,
                "score": 0.09,
                "priority": 0,
                "validation_metrics": {
                    "eval_long_actions": 0.0,
                    "eval_short_actions": 0.0,
                    "walkforward_active_window_ratio": 0.25,
                    "avg_trades_per_episode": 0.0,
                    "eval_dominant_action_ratio": 1.0,
                    "gross_total_return": 0.0,
                    "total_return": 0.0,
                    "supervised_validation_brier": 0.71,
                },
            },
        ]

        pruned = _prune_supervised_candidate_pool(candidates)
        assert len(pruned) == 2

    def test_trim_supervised_candidate_pool_keeps_distinct_family_when_late_family_scores_lower(self):
        candidates = []
        for idx in range(6):
            candidates.append(
                {
                    "name": f"logreg_{idx}",
                    "meta": {"model_type": "logreg"},
                    "score": 10.0 - idx,
                    "priority": 0,
                    "long_confidence": 0.80,
                    "short_confidence": 0.72 + idx * 0.01,
                    "validation_metrics": {
                        "walkforward_active_window_ratio": 0.5,
                        "gross_total_return": 0.002,
                        "supervised_validation_brier": 0.7,
                    },
                }
            )
        candidates.append(
            {
                "name": "trend_best",
                "meta": {"model_type": "trend_rule"},
                "score": 1.0,
                "priority": -1,
                "long_confidence": 0.82,
                "short_confidence": 0.74,
                "validation_metrics": {
                    "walkforward_active_window_ratio": 1.0,
                    "gross_total_return": 0.001,
                    "supervised_validation_brier": 0.8,
                },
            }
        )

        trimmed = _trim_supervised_candidate_pool(candidates, limit=4)
        families = {item["meta"]["model_type"] for item in trimmed}
        assert "logreg" in families
        assert "trend_rule" in families

    def test_score_nautilus_summary_allows_one_of_three_active_windows_when_configured(self):
        summary = {
            "n_trades": 3.0,
            "total_return": 0.002,
            "gross_total_return": 0.004,
            "outperformance_vs_bh": 0.001,
            "win_rate": 0.66,
            "flat_ratio": 0.97,
            "eval_dominant_action_ratio": 0.99,
            "eval_action_entropy": 0.05,
            "max_drawdown": -0.01,
            "nautilus_min_total_return": -0.001,
            "nautilus_min_gross_total_return": 0.0,
            "nautilus_min_alpha": -0.001,
            "nautilus_active_window_ratio": 1.0 / 3.0,
            "trade_density": 0.002,
            "stats_returns": {"Sharpe Ratio (252 days)": 0.5},
        }

        score = _score_nautilus_summary(
            summary,
            min_trades=1.0,
            max_dominant_action_ratio=0.995,
            min_active_window_ratio=1.0 / 3.0,
        )
        assert np.isfinite(score)

    def test_score_sparse_supervised_watchlist_candidate_accepts_sparse_profitable_model(self):
        score = _score_sparse_supervised_watchlist_candidate(
            {
                "total_return": 0.004,
                "gross_total_return": 0.008,
                "outperformance_vs_bh": 0.10,
                "max_drawdown": 0.02,
                "avg_trades_per_episode": 3.0,
                "eval_action_entropy": 0.12,
                "eval_dominant_action_ratio": 0.97,
                "flat_ratio": 0.96,
                "walkforward_min_total_return": 0.002,
                "walkforward_min_gross_total_return": 0.004,
                "walkforward_min_alpha": 0.05,
            }
        )
        assert np.isfinite(score)

    def test_score_sparse_supervised_watchlist_candidate_rejects_inactive_or_losing_model(self):
        score = _score_sparse_supervised_watchlist_candidate(
            {
                "total_return": -0.02,
                "gross_total_return": -0.01,
                "avg_trades_per_episode": 0.0,
                "eval_dominant_action_ratio": 1.0,
            }
        )
        assert score == float("-inf")

    def test_is_collapsed_supervised_probe_metrics_detects_flat_candidate(self):
        assert _is_collapsed_supervised_probe_metrics(
            {
                "avg_trades_per_episode": 0.0,
                "flat_ratio": 1.0,
                "eval_dominant_action_ratio": 1.0,
                "eval_action_entropy": 0.0,
                "position_ratio": 0.0,
                "gross_total_return": 0.0,
            }
        )
        assert not _is_collapsed_supervised_probe_metrics(
            {
                "avg_trades_per_episode": 1.2,
                "flat_ratio": 0.94,
                "eval_dominant_action_ratio": 0.90,
                "eval_action_entropy": 0.12,
                "position_ratio": 0.06,
                "gross_total_return": 0.004,
            }
        )

    def test_score_anchor_supervised_probe_candidate_accepts_sparse_near_flat_candidate(self):
        score = _score_anchor_supervised_probe_candidate(
            {
                "total_return": -0.0015,
                "gross_total_return": 0.0,
                "outperformance_vs_bh": 0.02,
                "max_drawdown": 0.003,
                "avg_trades_per_episode": 0.0,
                "eval_dominant_action_ratio": 1.0,
                "flat_ratio": 1.0,
                "walkforward_min_total_return": -0.0020,
                "walkforward_min_gross_total_return": 0.0,
            }
        )
        assert np.isfinite(score)

    def test_score_nautilus_rejected_fallback_rejects_negative_sparse_candidate(self):
        score = _score_nautilus_rejected_fallback(
            {
                "nautilus_score_preliminary": 0.40,
                "n_trades": 3.0,
                "total_return": -0.0129,
                "gross_total_return": -0.0117,
                "nautilus_min_total_return": -0.0189,
                "nautilus_min_gross_total_return": -0.0177,
                "outperformance_vs_bh": -0.003,
                "nautilus_active_window_ratio": 1.0 / 3.0,
                "trade_density": 0.0085,
                "eval_dominant_action_ratio": 0.949,
                "max_drawdown": -0.0395,
            }
        )
        assert score == float("-inf")

    def test_score_nautilus_rejected_fallback_accepts_near_flat_active_candidate(self):
        score = _score_nautilus_rejected_fallback(
            {
                "nautilus_score_preliminary": 0.85,
                "n_trades": 3.0,
                "total_return": -0.0015,
                "gross_total_return": 0.0004,
                "nautilus_min_total_return": -0.0030,
                "nautilus_min_gross_total_return": -0.0010,
                "outperformance_vs_bh": 0.012,
                "nautilus_active_window_ratio": 2.0 / 3.0,
                "trade_density": 0.006,
                "eval_dominant_action_ratio": 0.90,
                "max_drawdown": -0.010,
            }
        )
        assert np.isfinite(score)
