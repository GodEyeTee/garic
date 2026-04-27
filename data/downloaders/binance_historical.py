"""Download historical OHLCV data from Binance Data Vision (ฟรี ไม่มี rate limit).

URL pattern:
  https://data.binance.vision/data/futures/um/monthly/klines/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{YYYY}-{MM}.zip

Usage:
  python -m data.downloaders.binance_historical --pairs BTCUSDT ETHUSDT --start 2020-01-01
"""

import io
import zipfile
import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BINANCE_DATA_VISION = "https://data.binance.vision/data/futures/um"

OHLCV_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_volume",
    "taker_buy_quote_volume", "ignore",
]


def download_monthly_klines(
    symbol: str,
    interval: str,
    year: int,
    month: int,
    timeout: int = 60,
) -> pd.DataFrame | None:
    """Download 1 month of kline data. Returns None if not available."""
    filename = f"{symbol}-{interval}-{year}-{month:02d}"
    url = f"{BINANCE_DATA_VISION}/monthly/klines/{symbol}/{interval}/{filename}.zip"

    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 404:
            logger.debug(f"Not found: {filename}")
            return None
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"Failed to download {filename}: {e}")
        return None

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            # บางไฟล์มี header row บางไฟล์ไม่มี — ต้องตรวจจับ
            first_line = f.readline().decode("utf-8").strip()
            f.seek(0)
            if first_line.startswith("open_time") or "open" in first_line.split(",")[:3]:
                df = pd.read_csv(f)
                # Rename ให้ตรงกับ schema ถ้าจำเป็น
                col_map = {c: c for c in df.columns}
                df.columns = [c.strip().lower() for c in df.columns]
            else:
                df = pd.read_csv(f, header=None, names=OHLCV_COLUMNS)

    # ตัด column ที่เกินมา (บาง period มี 12 col บางอันมี 9)
    for col in list(df.columns):
        if col not in OHLCV_COLUMNS:
            df.drop(columns=[col], inplace=True, errors="ignore")

    # Drop rows ที่เป็น header ปนมา (string ใน numeric column)
    df = df[pd.to_numeric(df["open"], errors="coerce").notna()].copy()

    # Clean up types
    for col in ["open", "high", "low", "close", "volume", "quote_volume",
                "taker_buy_volume", "taker_buy_quote_volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(pd.to_numeric(df["open_time"], errors="coerce"), unit="ms")
    if "close_time" in df.columns:
        df["close_time"] = pd.to_datetime(pd.to_numeric(df["close_time"], errors="coerce"), unit="ms")
    if "trades" in df.columns:
        df["trades"] = pd.to_numeric(df["trades"], errors="coerce").fillna(0).astype(int)
    df.drop(columns=["ignore"], inplace=True, errors="ignore")

    logger.info(f"Downloaded {filename}: {len(df)} rows")
    return df


def download_range(
    symbol: str,
    interval: str = "1m",
    start_date: date = date(2020, 1, 1),
    end_date: date | None = None,
    output_dir: str = "data/raw",
    clear: bool = False,
) -> Path:
    """Download klines for a date range, save as Parquet."""
    if end_date is None:
        end_date = date.today()

    output_path = Path(output_dir) / f"{symbol}_{interval}.parquet"
    if clear and output_path.exists():
        logger.info(f"Clearing old data at {output_path}")
        output_path.unlink()
        
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    current = date(start_date.year, start_date.month, 1)

    while current <= end_date:
        df = download_monthly_klines(symbol, interval, current.year, current.month)
        if df is not None:
            all_dfs.append(df)

        # Next month
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)

    if not all_dfs:
        logger.error(f"No data downloaded for {symbol}")
        return output_path

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.drop_duplicates(subset=["open_time"], keep="last", inplace=True)
    combined.sort_values("open_time", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    combined.to_parquet(output_path, index=False)
    logger.info(f"Saved {symbol} {interval}: {len(combined)} rows → {output_path}")
    return output_path


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Download Binance Futures historical data")
    parser.add_argument("--pairs", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--output", default="data/raw")
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else None

    for pair in args.pairs:
        download_range(pair, args.interval, start, end, args.output, args.clear)


if __name__ == "__main__":
    main()
