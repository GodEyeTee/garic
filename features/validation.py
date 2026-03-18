"""Validation framework — anti-overfitting tools.

CPCV (Combinatorial Purged Cross-Validation)
DSR (Deflated Sharpe Ratio)
PBO (Probability of Backtest Overfitting)
Feature consistency check (train vs live)
"""

import logging
from datetime import timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# CPCV — Combinatorial Purged Cross-Validation
# =============================================================================

class PurgedKFold:
    """Purged K-Fold CV สำหรับ time series.

    - Purging: ลบ observation ที่ label overlap กับ test set
    - Embargo: เพิ่ม gap หลัง test set เพื่อป้องกัน leakage

    สำหรับ CPCV เต็มรูปแบบ ใช้ library `timeseriescv`:
      pip install timeseriescv
      from timeseriescv import CombPurgedKFoldCV
    """

    def __init__(
        self,
        n_splits: int = 6,
        embargo_pct: float = 0.01,
    ):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, n_samples: int):
        """Yield (train_indices, test_indices) tuples."""
        fold_size = n_samples // self.n_splits
        embargo_size = int(n_samples * self.embargo_pct)

        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n_samples)

            test_idx = np.arange(test_start, test_end)

            # Train = everything except test + embargo zone
            embargo_end = min(test_end + embargo_size, n_samples)
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_start:embargo_end] = False
            train_idx = np.where(train_mask)[0]

            yield train_idx, test_idx


# =============================================================================
# Deflated Sharpe Ratio
# =============================================================================

def deflated_sharpe_ratio(
    sharpe_observed: float,
    n_trials: int,
    n_observations: int,
    sharpe_std: float = 1.0,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio (Lopez de Prado).

    คำนวณความน่าจะเป็นที่ Sharpe ratio จะ significant
    หลังแก้ selection bias จากการทดลองหลายครั้ง.

    Args:
        sharpe_observed: Sharpe ratio ที่วัดได้
        n_trials: จำนวนครั้งที่ทดลอง (strategies tested), ต้อง >= 2
        n_observations: จำนวน data points
        sharpe_std: standard deviation of Sharpe estimates
        skewness: skewness of returns
        kurtosis: kurtosis of returns (3 = normal)
    """
    from scipy import stats

    if n_observations < 2:
        logger.warning("DSR: n_observations < 2, cannot compute")
        return 0.0

    # n_trials=1: DSR ไม่มีความหมาย (ไม่มี multiple testing)
    # fallback เป็น simple t-test ของ Sharpe > 0
    if n_trials < 2:
        se = np.sqrt(
            (1 - skewness * sharpe_observed + (kurtosis - 1) / 4 * sharpe_observed ** 2)
            / (n_observations - 1)
        )
        se = max(se, 1e-10)
        z = sharpe_observed / se
        p_value = stats.norm.cdf(z)
        logger.info(
            f"DSR (single trial, no deflation): observed={sharpe_observed:.3f}, "
            f"z={z:.3f}, p={p_value:.4f}"
        )
        return p_value

    # Expected max Sharpe under null (Euler-Mascheroni approximation)
    euler_mascheroni = 0.5772156649
    e_max_sharpe = sharpe_std * (
        (1 - euler_mascheroni) * stats.norm.ppf(1 - 1 / n_trials)
        + euler_mascheroni * stats.norm.ppf(1 - 1 / (n_trials * np.e))
    )

    # Sharpe ratio standard error with non-normal adjustment
    se = np.sqrt(
        (1 - skewness * sharpe_observed + (kurtosis - 1) / 4 * sharpe_observed ** 2)
        / (n_observations - 1)
    )
    se = max(se, 1e-10)

    # Test statistic
    z = (sharpe_observed - e_max_sharpe) / se
    p_value = stats.norm.cdf(z)

    logger.info(
        f"DSR: observed={sharpe_observed:.3f}, "
        f"E[max]={e_max_sharpe:.3f}, "
        f"p={p_value:.4f}, "
        f"n_trials={n_trials}"
    )
    return p_value


# =============================================================================
# PBO — Probability of Backtest Overfitting
# =============================================================================

def probability_of_backtest_overfitting(
    performance_matrix: np.ndarray,
) -> float:
    """Probability of Backtest Overfitting (PBO) via CSCV.

    Args:
        performance_matrix: shape (n_splits, n_strategies)
            แต่ละ cell = performance ของ strategy i ใน split j

    Returns:
        PBO: fraction ที่ IS-optimal strategy ทำแย่กว่า OOS median
        PBO > 0.5 = overfitting likely
    """
    n_splits, n_strategies = performance_matrix.shape

    if n_splits < 4:
        logger.warning("PBO needs >= 4 splits for meaningful results")
        return 0.5

    n_overfit = 0
    n_total = 0
    half = n_splits // 2

    # Simple CSCV: split data in half, check if IS-best = OOS-best
    from itertools import combinations
    for is_indices in combinations(range(n_splits), half):
        oos_indices = [i for i in range(n_splits) if i not in is_indices]

        is_perf = performance_matrix[list(is_indices), :].mean(axis=0)
        oos_perf = performance_matrix[oos_indices, :].mean(axis=0)

        best_is = np.argmax(is_perf)
        oos_median = np.median(oos_perf)

        if oos_perf[best_is] < oos_median:
            n_overfit += 1
        n_total += 1

    pbo = n_overfit / n_total if n_total > 0 else 0.5
    logger.info(f"PBO: {pbo:.3f} ({n_overfit}/{n_total} overfit)")
    return pbo


# =============================================================================
# Feature consistency check
# =============================================================================

def check_feature_consistency(
    train_features: np.ndarray,
    live_features: np.ndarray,
    threshold_ks: float = 0.05,
) -> dict:
    """ตรวจว่า feature distribution ของ train กับ live ไม่ต่างกันมากเกินไป.

    ใช้ KS test ต่อ feature column.
    """
    from scipy import stats

    n_features = train_features.shape[1]
    results = {"passed": True, "drifted_features": [], "ks_pvalues": []}

    for i in range(n_features):
        ks_stat, p_value = stats.ks_2samp(train_features[:, i], live_features[:, i])
        results["ks_pvalues"].append(p_value)

        if p_value < threshold_ks:
            results["drifted_features"].append(i)
            results["passed"] = False

    n_drifted = len(results["drifted_features"])
    if n_drifted > 0:
        logger.warning(f"Feature drift detected in {n_drifted}/{n_features} features")
    else:
        logger.info(f"Feature consistency check passed ({n_features} features)")

    return results
