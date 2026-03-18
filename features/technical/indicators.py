"""Technical indicators — RSI, MACD, Bollinger Bands, ATR, etc.

ใช้ pure numpy/pandas เพื่อลด dependency และควบคุม output ให้ตรงกันทั้ง train/live.
"""

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, histogram."""
    ema_fast = close.ewm(span=fast, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Upper band, middle band (SMA), lower band."""
    middle = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, min_periods=period).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(period).mean()


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_period: int = 14, d_period: int = 3) -> tuple[pd.Series, pd.Series]:
    """Stochastic %K and %D."""
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    denom = highest_high - lowest_low
    k = 100 * (close - lowest_low) / denom.replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index."""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr_val = atr(high, low, close, period)
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr_val)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr_val)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, min_periods=period).mean()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def vwap_rolling(high: pd.Series, low: pd.Series, close: pd.Series,
                 volume: pd.Series, period: int = 20) -> pd.Series:
    """Rolling VWAP."""
    typical = (high + low + close) / 3
    return (typical * volume).rolling(period).sum() / volume.rolling(period).sum()


def compute_all(df: pd.DataFrame) -> np.ndarray:
    """คำนวณ TA indicators ทั้งหมด return shape (n_rows, 15).

    Output columns:
    0: RSI(14)
    1: MACD line
    2: MACD signal
    3: MACD histogram
    4: BB upper
    5: BB middle
    6: BB lower
    7: BB %B (position within bands)
    8: ATR(14)
    9: Stochastic %K
    10: Stochastic %D
    11: ADX(14)
    12: OBV (normalized)
    13: EMA(9) / close ratio
    14: EMA(21) / close ratio
    """
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    rsi_val = rsi(c)
    macd_line, macd_sig, macd_hist = macd(c)
    bb_upper, bb_mid, bb_lower = bollinger_bands(c)
    bb_pctb = (c - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
    atr_val = atr(h, l, c)
    stoch_k, stoch_d = stochastic(h, l, c)
    adx_val = adx(h, l, c)
    obv_val = obv(c, v)
    obv_norm = (obv_val - obv_val.rolling(100).mean()) / obv_val.rolling(100).std().replace(0, 1)
    ema9_ratio = ema(c, 9) / c
    ema21_ratio = ema(c, 21) / c

    result = pd.DataFrame({
        "rsi": rsi_val,
        "macd_line": macd_line,
        "macd_signal": macd_sig,
        "macd_hist": macd_hist,
        "bb_upper": bb_upper,
        "bb_mid": bb_mid,
        "bb_lower": bb_lower,
        "bb_pctb": bb_pctb,
        "atr": atr_val,
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
        "adx": adx_val,
        "obv_norm": obv_norm,
        "ema9_ratio": ema9_ratio,
        "ema21_ratio": ema21_ratio,
    }).fillna(0.0)

    return result.values.astype(np.float32)
