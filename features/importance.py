"""Feature importance — MDI, MDA, SFI.

วัดว่า feature ไหนสำคัญ ก่อนส่งเข้า model.
ช่วยลด overfitting + เข้าใจว่า model ใช้อะไรตัดสินใจ.

Methods:
  MDI: Mean Decrease in Impurity (from Random Forest)
  MDA: Mean Decrease in Accuracy (permutation importance)
  SFI: Single Feature Importance (train on each feature alone)

Usage:
    from features.importance import compute_feature_importance
    result = compute_feature_importance(X_train, y_train, X_test, y_test)
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_mdi(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """Mean Decrease in Impurity — feature importance จาก Random Forest.

    เร็ว แต่มี bias ต่อ high-cardinality features.
    ใช้เป็น quick filter ก่อน MDA.

    Returns: importance scores, shape (n_features,)
    """
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X, y)
    return clf.feature_importances_


def compute_mda(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 5,
    n_repeats: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean Decrease in Accuracy — permutation importance.

    Unbiased, ใช้ test set → ไม่ overfit.
    แต่ช้ากว่า MDI (ต้อง permute ทีละ feature).

    Returns: (importance_means, importance_stds), shape (n_features,) each
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    result = permutation_importance(
        clf, X_test, y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    return result.importances_mean, result.importances_std


def compute_sfi(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_depth: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """Single Feature Importance — train model on each feature alone.

    ช้าที่สุด แต่บอกได้ว่า feature เดี่ยวๆ มีพลังพยากรณ์แค่ไหน.

    Returns: accuracy scores per feature, shape (n_features,)
    """
    from sklearn.ensemble import RandomForestClassifier

    n_features = X_train.shape[1]
    scores = np.zeros(n_features)

    for i in range(n_features):
        clf = RandomForestClassifier(
            n_estimators=50,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        clf.fit(X_train[:, i:i + 1], y_train)
        scores[i] = clf.score(X_test[:, i:i + 1], y_test)

    return scores


def compute_feature_importance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str] | None = None,
    top_k: int = 30,
) -> dict:
    """Run all 3 importance methods and return combined results.

    Args:
        X_train, y_train: training data
        X_test, y_test: test data
        feature_names: optional names for features
        top_k: return top K important features

    Returns:
        dict with MDI, MDA, SFI scores + combined ranking
    """
    n_features = X_train.shape[1]
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]

    logger.info(f"Computing feature importance: {n_features} features...")

    # MDI
    logger.info("  MDI (Mean Decrease in Impurity)...")
    mdi_scores = compute_mdi(X_train, y_train)

    # MDA
    logger.info("  MDA (Mean Decrease in Accuracy)...")
    mda_means, mda_stds = compute_mda(X_train, y_train, X_test, y_test)

    # SFI
    logger.info("  SFI (Single Feature Importance)...")
    sfi_scores = compute_sfi(X_train, y_train, X_test, y_test)

    # Combined ranking: normalize each to [0,1] then average
    def _normalize(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-10)

    combined = (_normalize(mdi_scores) + _normalize(mda_means) + _normalize(sfi_scores)) / 3
    ranking = np.argsort(-combined)  # descending

    # Top K features
    top_features = []
    for i in ranking[:top_k]:
        top_features.append({
            "index": int(i),
            "name": feature_names[i],
            "mdi": float(mdi_scores[i]),
            "mda_mean": float(mda_means[i]),
            "mda_std": float(mda_stds[i]),
            "sfi": float(sfi_scores[i]),
            "combined": float(combined[i]),
        })

    logger.info(f"Top 5 features: {[f['name'] for f in top_features[:5]]}")

    return {
        "mdi_scores": mdi_scores,
        "mda_means": mda_means,
        "mda_stds": mda_stds,
        "sfi_scores": sfi_scores,
        "combined": combined,
        "ranking": ranking,
        "top_features": top_features,
        "n_features": n_features,
    }
