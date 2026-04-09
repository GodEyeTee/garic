from __future__ import annotations

import numpy as np
import yaml

from models.rl.supervised import train_supervised_action_model


def main() -> None:
    data = np.load("data/cache/BTCUSDT_train_features.npz", allow_pickle=True)
    config = yaml.safe_load(open("configs/test_rtx2060.yaml", "r", encoding="utf-8"))
    features = np.asarray(data["features"], dtype=np.float32)
    prices = np.asarray(data["prices"], dtype=np.float32)
    n = len(prices)
    train_end = int(n * 0.72)
    val_end = int(n * 0.80)
    supervised = config["training"]["supervised_fallback"]
    trading = config["trading"]

    model, meta = train_supervised_action_model(
        feature_array=features,
        prices=prices,
        train_range=(0, train_end),
        validation_range=(train_end, val_end),
        model_type="logreg",
        horizon=supervised["horizon"],
        min_return_threshold=supervised["min_return_threshold"],
        threshold_quantile=supervised["threshold_quantile"],
        max_train_samples=supervised["max_train_samples"],
        logistic_c=supervised["logistic_c"],
        min_hold_steps=supervised["min_hold_steps"],
        reversal_margin=supervised["reversal_margin"],
        entry_margin=supervised["entry_margin"],
        exit_to_flat_margin=supervised["exit_to_flat_margin"],
        max_hold_steps=supervised["max_hold_steps"],
        stop_loss_threshold=supervised["stop_loss_threshold"],
        drawdown_exit_threshold=supervised["drawdown_exit_threshold"],
        trend_alignment_threshold=supervised["trend_alignment_threshold"],
        countertrend_margin=supervised["countertrend_margin"],
        regime_confidence_relief=supervised["regime_confidence_relief"],
        flat_reentry_cooldown_steps=supervised["flat_reentry_cooldown_steps"],
        meta_label_min_edge=supervised["meta_label_min_edge"],
        meta_label_edge_margin=supervised["meta_label_edge_margin"],
        meta_label_exit_edge=supervised["meta_label_exit_edge"],
        meta_label_min_positive_rate=supervised["meta_label_min_positive_rate"],
        calibration_min_samples=supervised["calibration_min_samples"],
        calibration_probability_thresholds=supervised["calibration_probability_thresholds"],
        taker_fee=trading.get("taker_fee", 0.0005),
        slippage_bps=trading.get("slippage_bps", 1.0),
        leverage=trading.get("leverage", 1.0),
        random_state=config["training"]["seed"],
    )

    print("model_feature_dim", model.feature_dim)
    print("meta_label", meta["meta_label_min_edge"], meta["meta_label_edge_margin"], meta["meta_label_exit_edge"], meta["meta_label_min_positive_rate"])
    print("label_threshold", meta["label_threshold"], "round_trip_cost", meta["round_trip_cost"])
    calibration = meta["post_cost_calibration"]
    x_valid = np.asarray(features[train_end:val_end - supervised["horizon"]], dtype=np.float32)
    if model.scaler is not None:
        x_valid = model.scaler.transform(x_valid)
    raw_proba = np.asarray(model.classifier.predict_proba(x_valid), dtype=np.float64)
    aligned_proba = np.zeros((raw_proba.shape[0], 3), dtype=np.float64)
    for cls_idx, cls in enumerate(model.classifier.classes_):
        aligned_proba[:, int(cls)] = raw_proba[:, cls_idx]
    for side_name, action in (("long", 2), ("short", 0), ("flat", 1)):
        values = aligned_proba[:, action]
        print(
            "PROBA",
            side_name,
            "max",
            float(np.max(values)),
            "q90",
            float(np.quantile(values, 0.90)),
            "q95",
            float(np.quantile(values, 0.95)),
            "q99",
            float(np.quantile(values, 0.99)),
            "count>=0.70",
            int(np.sum(values >= 0.70)),
            "count>=0.80",
            int(np.sum(values >= 0.80)),
        )
    for side in ("long", "short"):
        side_cal = calibration[side]
        print("SIDE", side)
        print(" global", side_cal["global"])
        for row in side_cal["thresholds"]:
            print(" ", row)


if __name__ == "__main__":
    main()
