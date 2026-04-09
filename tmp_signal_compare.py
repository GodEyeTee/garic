from pathlib import Path

import numpy as np
import pandas as pd

from execution.nautilus.features import NautilusFeatureBuilder
from models.rl.environment import build_agent_state
from models.rl.supervised import SupervisedActionModel
from features.builder import FeatureBuilder
from pipeline import _aggregate_ohlcv_15m


def build_obs(feature_row: np.ndarray) -> np.ndarray:
    state = build_agent_state(
        position=0.0,
        upnl=0.0,
        equity_ratio=0.0,
        drawdown=0.0,
        rolling_volatility=0.0,
        turnover_last_step=0.0,
        flat_steps=64,
        pos_steps=0,
    )
    return np.concatenate([feature_row.astype(np.float32), state]).astype(np.float32)


def main() -> None:
    data_path = Path("data/raw/BTCUSDT_1m.parquet")
    df = pd.read_parquet(data_path).tail(200_000).reset_index(drop=True)
    agg = _aggregate_ohlcv_15m(df, period=15).reset_index(drop=True)

    builder = FeatureBuilder(lookback=60)
    feature_array, _, _ = builder.build_batch_array(agg)
    model = SupervisedActionModel.load("checkpoints/rl_agent_supervised.joblib")

    ranges = {"validation": (9599, 10666), "test": (10666, 13333)}

    for history_bars in (160, 256, 384):
        naut = NautilusFeatureBuilder(history_bars=history_bars, include_forecast=False)
        print("=== history_bars", history_bars, "===")
        for name, (start, end) in ranges.items():
            train_actions = []
            naut_actions = []
            for idx in range(start, end):
                bar_end = idx + 60 + 1
                obs_train = build_obs(feature_array[idx])
                action_train, _ = model.predict(obs_train, deterministic=True)
                train_actions.append(int(action_train))

                snap = naut.build_latest(agg.iloc[:bar_end].copy())
                obs_naut = build_obs(snap.feature_array)
                action_naut, _ = model.predict(obs_naut, deterministic=True)
                naut_actions.append(int(action_naut))

            train_counts = {a: train_actions.count(a) for a in (0, 1, 2)}
            naut_counts = {a: naut_actions.count(a) for a in (0, 1, 2)}
            same = sum(int(a == b) for a, b in zip(train_actions, naut_actions, strict=False))
            print(
                name,
                "train_counts",
                train_counts,
                "naut_counts",
                naut_counts,
                "match_ratio",
                same / max(len(train_actions), 1),
            )


if __name__ == "__main__":
    main()
