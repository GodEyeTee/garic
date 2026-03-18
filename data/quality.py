"""Data quality pipeline — gap detection, outlier removal, validation.

ใช้ทั้ง historical data cleaning และ live data validation.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def detect_gaps(df: pd.DataFrame, time_col: str = "open_time", freq: str = "1min") -> pd.DataFrame:
    """Detect time gaps in OHLCV data.

    Returns DataFrame with gap info: start, end, duration.
    """
    expected_delta = pd.Timedelta(freq)
    deltas = df[time_col].diff()
    gap_mask = deltas > expected_delta * 1.5

    gaps = []
    for idx in df.index[gap_mask]:
        prev_idx = idx - 1
        gaps.append({
            "gap_start": df.loc[prev_idx, time_col],
            "gap_end": df.loc[idx, time_col],
            "duration": deltas.loc[idx],
        })

    gap_df = pd.DataFrame(gaps)
    if not gap_df.empty:
        logger.info(f"Found {len(gap_df)} gaps, max duration: {gap_df['duration'].max()}")
    return gap_df


def fill_gaps(
    df: pd.DataFrame,
    time_col: str = "open_time",
    freq: str = "1min",
    max_forward_fill_sec: int = 300,
) -> pd.DataFrame:
    """Fill gaps: forward-fill < max_forward_fill_sec, interpolate longer gaps."""
    df = df.set_index(time_col).sort_index()
    n_original = len(df)
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    n_gap_rows = len(full_idx) - n_original

    df = df.reindex(full_idx)

    # Short gaps: forward fill
    max_periods = max_forward_fill_sec // pd.Timedelta(freq).total_seconds()
    df_filled = df.ffill(limit=int(max_periods))

    # Longer gaps: interpolate numeric columns
    numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
    df_filled[numeric_cols] = df_filled[numeric_cols].interpolate(method="linear")
    df_filled[numeric_cols] = df_filled[numeric_cols].bfill()  # handle leading NaNs

    df_filled.index.name = time_col
    df_filled.reset_index(inplace=True)

    if n_gap_rows > 0:
        pct = n_gap_rows / len(df_filled) * 100
        logger.info(f"Filled {n_gap_rows} gap rows ({pct:.1f}% of total)")
        if pct > 10:
            logger.warning(f"High gap fill ratio ({pct:.1f}%) — check data completeness")

    # Repair OHLCV consistency after interpolation
    df_filled = _repair_ohlcv(df_filled)

    return df_filled


def remove_outliers(
    df: pd.DataFrame,
    price_cols: list[str] | None = None,
    zscore_threshold: float = 3.0,
    window: int = 100,
) -> pd.DataFrame:
    """Remove outliers using rolling Z-score > threshold.

    Replaces outlier values with NaN then interpolates.
    Uses close as anchor — outliers in close are detected first,
    then rows with outlier close get all OHLC set to NaN together
    to preserve consistency.
    """
    if price_cols is None:
        price_cols = ["open", "high", "low", "close"]

    df = df.copy()

    # Detect outlier rows based on close price (most reliable)
    if "close" in df.columns:
        rolling_mean = df["close"].rolling(window, min_periods=10).mean()
        rolling_std = df["close"].rolling(window, min_periods=10).std()
        zscore = ((df["close"] - rolling_mean) / rolling_std).abs()
        outlier_rows = zscore > zscore_threshold
        n_outliers = outlier_rows.sum()
        if n_outliers > 0:
            for col in price_cols:
                if col in df.columns:
                    df.loc[outlier_rows, col] = np.nan
            logger.info(f"Cleaned {n_outliers} outlier rows (NaN → interpolate, rows preserved)")

    # Interpolate all price columns
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear")
            df[col] = df[col].bfill()  # handle leading NaNs

    # Enforce OHLCV consistency
    df = _repair_ohlcv(df)

    return df


def _repair_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce OHLCV constraints: high >= max(open,close), low <= min(open,close)."""
    df = df.copy()
    if all(c in df.columns for c in ["open", "high", "low", "close"]):
        row_max = df[["open", "close"]].max(axis=1)
        row_min = df[["open", "close"]].min(axis=1)

        # high must be >= open, close, and >= low
        df["high"] = df[["high"]].assign(row_max=row_max).max(axis=1)
        # low must be <= open, close, and <= high
        df["low"] = df[["low"]].assign(row_min=row_min).min(axis=1)
    return df


def validate_ohlcv(df: pd.DataFrame) -> list[str]:
    """Validate OHLCV data integrity. Returns list of issues."""
    issues = []

    # Check high >= low
    bad_hl = (df["high"] < df["low"]).sum()
    if bad_hl > 0:
        issues.append(f"high < low in {bad_hl} rows")

    # Check high >= open and high >= close
    bad_ho = (df["high"] < df["open"]).sum()
    bad_hc = (df["high"] < df["close"]).sum()
    if bad_ho > 0:
        issues.append(f"high < open in {bad_ho} rows")
    if bad_hc > 0:
        issues.append(f"high < close in {bad_hc} rows")

    # Check low <= open and low <= close
    bad_lo = (df["low"] > df["open"]).sum()
    bad_lc = (df["low"] > df["close"]).sum()
    if bad_lo > 0:
        issues.append(f"low > open in {bad_lo} rows")
    if bad_lc > 0:
        issues.append(f"low > close in {bad_lc} rows")

    # Check volume >= 0
    neg_vol = (df["volume"] < 0).sum()
    if neg_vol > 0:
        issues.append(f"negative volume in {neg_vol} rows")

    # Check for zero prices
    zero_price = ((df[["open", "high", "low", "close"]] == 0).any(axis=1)).sum()
    if zero_price > 0:
        issues.append(f"zero price in {zero_price} rows")

    # Check duplicates
    if "open_time" in df.columns:
        dupes = df.duplicated(subset=["open_time"]).sum()
        if dupes > 0:
            issues.append(f"{dupes} duplicate timestamps")

    if issues:
        for issue in issues:
            logger.warning(f"Data issue: {issue}")
    else:
        logger.info("OHLCV validation passed")

    return issues


def clean_pipeline(
    df: pd.DataFrame,
    time_col: str = "open_time",
    freq: str = "1min",
    zscore_threshold: float = 3.0,
    max_forward_fill_sec: int = 300,
) -> pd.DataFrame:
    """Full cleaning pipeline: validate → outliers → gaps → validate."""
    logger.info(f"Cleaning {len(df)} rows...")

    # Step 1: Remove duplicates
    df = df.drop_duplicates(subset=[time_col], keep="last")
    df = df.sort_values(time_col).reset_index(drop=True)

    # Step 2: Validate
    issues_before = validate_ohlcv(df)

    # Step 3: Remove outliers
    df = remove_outliers(df, zscore_threshold=zscore_threshold)

    # Step 4: Fill gaps
    df = fill_gaps(df, time_col=time_col, freq=freq,
                   max_forward_fill_sec=max_forward_fill_sec)

    # Step 5: Validate again
    issues_after = validate_ohlcv(df)

    logger.info(f"Done: {len(issues_before)} issues before → {len(issues_after)} after, {len(df)} rows")
    return df
