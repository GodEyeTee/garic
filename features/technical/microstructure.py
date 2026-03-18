"""Microstructure features — Order Flow Imbalance, spread, volume analysis.

สำคัญสำหรับ short-term prediction:
"Better input สำคัญกว่าการเพิ่ม hidden layer" — arXiv:2506.05764
"""

import numpy as np
import pandas as pd


def order_flow_imbalance(
    buy_volume: pd.Series,
    sell_volume: pd.Series,
) -> pd.Series:
    """Order Flow Imbalance (OFI) = (buy_vol - sell_vol) / total_vol.

    ใช้ taker_buy_volume จาก Binance data.
    """
    total = buy_volume + sell_volume
    return ((buy_volume - sell_volume) / total.replace(0, np.nan)).fillna(0)


def volume_imbalance_ratio(
    volume: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Volume relative to rolling average — detect abnormal activity."""
    avg = volume.rolling(period).mean()
    return (volume / avg.replace(0, np.nan)).fillna(1.0)


def price_range_ratio(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Current candle range vs average range — detect volatility expansion."""
    candle_range = high - low
    avg_range = candle_range.rolling(period).mean()
    return (candle_range / avg_range.replace(0, np.nan)).fillna(1.0)


def close_position_in_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Where close sits in H-L range. 1=top, 0=bottom."""
    hl_range = high - low
    return ((close - low) / hl_range.replace(0, np.nan)).fillna(0.5)


def trade_intensity(
    trades: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Number of trades relative to average — proxy for activity."""
    avg = trades.rolling(period).mean()
    return (trades / avg.replace(0, np.nan)).fillna(1.0)


def compute_all(df: pd.DataFrame) -> np.ndarray:
    """คำนวณ microstructure features ทั้งหมด return shape (n_rows, 5).

    Output columns:
    0: Order Flow Imbalance (OFI)
    1: Volume imbalance ratio
    2: Price range ratio
    3: Close position in range
    4: Trade intensity
    """
    # OFI: ใช้ taker_buy_volume ถ้ามี ไม่งั้นประมาณจาก close position
    if "taker_buy_volume" in df.columns:
        buy_vol = df["taker_buy_volume"].astype(float)
        sell_vol = df["volume"] - buy_vol
        ofi = order_flow_imbalance(buy_vol, sell_vol)
    else:
        # Approximate: ถ้า close > open → buy dominant
        ofi = close_position_in_range(df["high"], df["low"], df["close"]) * 2 - 1

    vol_imb = volume_imbalance_ratio(df["volume"])
    price_rr = price_range_ratio(df["high"], df["low"], df["close"])
    close_pos = close_position_in_range(df["high"], df["low"], df["close"])

    if "trades" in df.columns:
        trade_int = trade_intensity(df["trades"].astype(float))
    else:
        trade_int = pd.Series(1.0, index=df.index)

    result = pd.DataFrame({
        "ofi": ofi,
        "vol_imbalance": vol_imb,
        "price_range_ratio": price_rr,
        "close_position": close_pos,
        "trade_intensity": trade_int,
    }).fillna(0.0)

    return result.values.astype(np.float32)
