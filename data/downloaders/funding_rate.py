"""Download historical funding rate from Binance public data repository.

Source: data.binance.vision (static CDN, not geo-restricted like API)
- Monthly ZIP files containing CSV with funding rate data
- Data available since ~Oct 2019

Usage:
  python -m data.downloaders.funding_rate --pairs BTCUSDT ETHUSDT
"""

import io
import logging
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision/data/futures/um/monthly/fundingRate"
REQUEST_TIMEOUT = 30


def _month_range(start: date, end: date) -> list[str]:
    """Generate YYYY-MM strings from start to end (inclusive)."""
    months = []
    current = start.replace(day=1)
    while current <= end:
        months.append(current.strftime("%Y-%m"))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    return months


def _download_month(symbol: str, year_month: str) -> pd.DataFrame | None:
    """Download one monthly ZIP and return DataFrame, or None if not available."""
    url = f"{BASE_URL}/{symbol}/{symbol}-fundingRate-{year_month}.zip"
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)

    if resp.status_code == 404:
        return None
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(f)
    return df


def download_funding_rate(
    symbol: str,
    start_date: date = date(2019, 10, 1),
    end_date: date | None = None,
    output_dir: str = "data/raw",
) -> Path:
    """Download all funding rate history from Binance public data."""
    if end_date is None:
        end_date = date.today()

    output_path = Path(output_dir) / f"{symbol}_funding_rate.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    months = _month_range(start_date, end_date)
    frames: list[pd.DataFrame] = []

    for ym in months:
        df = _download_month(symbol, ym)
        if df is None:
            logger.debug(f"{symbol} {ym}: not available, skipping")
            continue
        frames.append(df)
        logger.info(f"{symbol} {ym}: {len(df)} records")

    if not frames:
        logger.warning(f"No funding rate data for {symbol}")
        return output_path

    df = pd.concat(frames, ignore_index=True)

    # Normalize column names (Binance CSV headers vary slightly)
    col_map = {}
    for col in df.columns:
        low = col.strip().lower()
        if "time" in low and "funding" in low:
            col_map[col] = "fundingTime"
        elif "funding" in low and "rate" in low:
            col_map[col] = "fundingRate"
        elif "mark" in low and "price" in low:
            col_map[col] = "markPrice"
    df.rename(columns=col_map, inplace=True)

    # Parse time — could be ms timestamp or ISO string
    if df["fundingTime"].dtype == "object":
        df["fundingTime"] = pd.to_datetime(df["fundingTime"])
    else:
        df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")

    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce").fillna(0.0)
    if "markPrice" in df.columns:
        df["markPrice"] = pd.to_numeric(
            df["markPrice"].replace("", None), errors="coerce"
        ).fillna(0.0)
    else:
        df["markPrice"] = 0.0

    df = df[["fundingTime", "fundingRate", "markPrice"]]
    df.drop_duplicates(subset=["fundingTime"], keep="last", inplace=True)
    df.sort_values("fundingTime", inplace=True)

    # Filter to requested date range
    df = df[
        (df["fundingTime"] >= pd.Timestamp(start_date))
        & (df["fundingTime"] <= pd.Timestamp(end_date))
    ]
    df.reset_index(drop=True, inplace=True)

    df.to_parquet(output_path, index=False)
    logger.info(f"Saved {symbol} funding rate: {len(df)} rows → {output_path}")
    return output_path


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Download Binance funding rate history")
    parser.add_argument("--pairs", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--start", default="2019-10-01")
    parser.add_argument("--output", default="data/raw")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    for pair in args.pairs:
        download_funding_rate(pair, start, output_dir=args.output)


if __name__ == "__main__":
    main()
