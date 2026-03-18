"""Download sentiment data — Fear & Greed Index + Google Trends.

Fear & Greed: ฟรี ไม่ต้อง API key, data ตั้งแต่ Feb 2018
Google Trends: ผ่าน pytrends, 5 ปี weekly resolution

Usage:
  python -m data.downloaders.sentiment
"""

import logging
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=0&format=json"


def download_fear_greed(output_dir: str = "data/raw") -> Path:
    """Download full Fear & Greed Index history."""
    output_path = Path(output_dir) / "fear_greed_index.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    resp = requests.get(FEAR_GREED_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()["data"]

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
    df["value"] = df["value"].astype(int)
    df = df[["timestamp", "value", "value_classification"]].copy()
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_parquet(output_path, index=False)
    logger.info(f"Fear & Greed: {len(df)} days → {output_path}")
    return output_path


def download_google_trends(
    keywords: list[str] | None = None,
    output_dir: str = "data/raw",
) -> Path:
    """Download Google Trends data for crypto keywords."""
    output_path = Path(output_dir) / "google_trends.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if keywords is None:
        keywords = ["bitcoin", "crypto", "ethereum"]

    try:
        from pytrends.request import TrendReq
    except ImportError:
        logger.error("pytrends not installed: pip install pytrends")
        return output_path

    pytrends = TrendReq(hl="en-US", tz=0)
    pytrends.build_payload(keywords, timeframe="today 5-y")
    df = pytrends.interest_over_time()

    if df.empty:
        logger.warning("No Google Trends data returned")
        return output_path

    df = df.drop(columns=["isPartial"], errors="ignore")
    df.reset_index(inplace=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Google Trends: {len(df)} weeks → {output_path}")
    return output_path


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    download_fear_greed()
    download_google_trends()


if __name__ == "__main__":
    main()
