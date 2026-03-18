"""Download on-chain data from DeFiLlama (ฟรี ไม่ต้อง API key).

Endpoints:
  - TVL: https://api.llama.fi/v2/historicalChainTvl/{chain}
  - Protocol TVL: https://api.llama.fi/protocol/{protocol}
  - Stablecoin: https://stablecoins.llama.fi/stablecoincharts/all

Usage:
  python -m data.downloaders.onchain
"""

import logging
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DEFILLAMA_BASE = "https://api.llama.fi"


def download_chain_tvl(
    chain: str = "Ethereum",
    output_dir: str = "data/raw",
) -> Path:
    """Download historical TVL for a chain."""
    output_path = Path(output_dir) / f"tvl_{chain.lower()}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    url = f"{DEFILLAMA_BASE}/v2/historicalChainTvl/{chain}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"], unit="s")
    df["tvl"] = df["tvl"].astype(float)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_parquet(output_path, index=False)
    logger.info(f"TVL {chain}: {len(df)} days → {output_path}")
    return output_path


def download_protocol_tvl(
    protocol: str = "aave",
    output_dir: str = "data/raw",
) -> Path:
    """Download historical TVL for a specific protocol."""
    output_path = Path(output_dir) / f"tvl_protocol_{protocol}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    url = f"{DEFILLAMA_BASE}/protocol/{protocol}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    tvl_data = data.get("tvl", [])
    if not tvl_data:
        logger.warning(f"No TVL data for {protocol}")
        return output_path

    df = pd.DataFrame(tvl_data)
    df["date"] = pd.to_datetime(df["date"], unit="s")
    df["totalLiquidityUSD"] = df["totalLiquidityUSD"].astype(float)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_parquet(output_path, index=False)
    logger.info(f"TVL {protocol}: {len(df)} days → {output_path}")
    return output_path


def download_stablecoin_supply(output_dir: str = "data/raw") -> Path:
    """Download total stablecoin market cap over time."""
    output_path = Path(output_dir) / "stablecoin_supply.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    url = "https://stablecoins.llama.fi/stablecoincharts/all?stablecoin=1"  # USDT
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for item in data:
        row = {"date": item["date"]}
        if "totalCirculating" in item:
            row["circulating"] = item["totalCirculating"].get("peggedUSD", 0)
        if "totalMintedToday" in item:
            row["minted_today"] = item["totalMintedToday"].get("peggedUSD", 0)
        rows.append(row)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], unit="s")
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_parquet(output_path, index=False)
    logger.info(f"Stablecoin supply: {len(df)} days → {output_path}")
    return output_path


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    for chain in ["Ethereum", "Solana", "Arbitrum"]:
        download_chain_tvl(chain)

    download_stablecoin_supply()


if __name__ == "__main__":
    main()
