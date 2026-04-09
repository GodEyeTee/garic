from pathlib import Path

import numpy as np
import pandas as pd

from execution.nautilus.features import NautilusFeatureBuilder
from features.builder import FeatureBuilder
from pipeline import _aggregate_ohlcv_15m


def main() -> None:
    data_path = Path("data/raw/BTCUSDT_1m.parquet")
    df = pd.read_parquet(data_path).tail(15 * 400).reset_index(drop=True)
    agg = _aggregate_ohlcv_15m(df, period=15).reset_index(drop=True)

    fb = FeatureBuilder(lookback=60)
    feature_array, _, _ = fb.build_batch_array(agg)
    for history_bars in (160, 256, 384, 512, 640):
        naut = NautilusFeatureBuilder(history_bars=history_bars, include_forecast=False)
        rows = []
        for end in range(max(history_bars, 160), len(agg) + 1):
            snapshot = naut.build_latest(agg.iloc[:end].copy())
            train_idx = end - 60 - 1
            if train_idx < 0 or train_idx >= len(feature_array):
                continue
            diff = np.abs(feature_array[train_idx] - snapshot.feature_array)
            rows.append((end, float(diff.max()), float(diff.mean())))

        print("history_bars", history_bars, "num_rows", len(rows))
        print("last5", rows[-5:])
        print("max_of_max", max(r[1] for r in rows))
        print("mean_of_mean", sum(r[2] for r in rows) / len(rows))

        end = len(agg)
        snapshot = naut.build_latest(agg.copy())
        train_idx = end - 60 - 1
        latest_diff = feature_array[train_idx] - snapshot.feature_array
        print("latest_train_idx", train_idx, "feature_dim", latest_diff.shape[0])
        print("largest_latest_abs", float(np.max(np.abs(latest_diff))))


if __name__ == "__main__":
    main()
