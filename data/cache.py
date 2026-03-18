"""Feature cache — บันทึก processed features เพื่อไม่ต้องคำนวณซ้ำทุกรอบ.

Cache invalidation:
- Raw data file เปลี่ยน (mtime ใหม่กว่า cache) → recompute
- ลบ data/cache/ ทั้ง folder → recompute ทุกอย่าง
- ใช้ --no-cache flag → ข้าม cache

Cache files:
  data/cache/{SYMBOL}_train_features.npz  — feature_array, prices, ohlcv, close_15m
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache")


def _is_cache_valid(cache_path: Path, source_paths: list[Path]) -> bool:
    """Check if cache is newer than all source files."""
    if not cache_path.exists():
        return False

    cache_mtime = cache_path.stat().st_mtime
    for src in source_paths:
        if src.exists() and src.stat().st_mtime > cache_mtime:
            logger.info(f"Cache invalidated: {src.name} is newer than cache")
            return False
    return True


def load_features(symbol: str, source_paths: list[Path]) -> dict | None:
    """Load cached features if valid, else return None."""
    cache_path = CACHE_DIR / f"{symbol}_train_features.npz"

    if not _is_cache_valid(cache_path, source_paths):
        return None

    data = np.load(cache_path)
    result = {k: data[k] for k in data.files}
    logger.info(f"Loaded cached features: {cache_path} "
                f"({result['features'].shape[1]} dims, {len(result['prices']):,} rows)")
    return result


def save_features(
    symbol: str,
    feature_array: np.ndarray,
    prices: np.ndarray,
    ohlcv_data: np.ndarray,
    close_15m: np.ndarray,
) -> Path:
    """Save features to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{symbol}_train_features.npz"

    np.savez(
        cache_path,
        features=feature_array,
        prices=prices,
        ohlcv=ohlcv_data,
        close_15m=close_15m,
    )
    logger.info(f"Saved feature cache: {cache_path} "
                f"({feature_array.shape[1]} dims, {len(prices):,} rows)")
    return cache_path
