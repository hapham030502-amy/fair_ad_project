from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score


def choose_threshold_on_val(
    scores_val: np.ndarray,
    y_val: np.ndarray,
    percentiles: Iterable[float] = np.linspace(70, 99.5, 240),
) -> Dict[str, float]:
    """Chọn threshold trên validation set; tuyệt đối không chọn threshold trên test."""
    scores_val = np.asarray(scores_val, dtype=float).ravel()
    y_val = np.asarray(y_val, dtype=int).ravel()

    if len(scores_val) != len(y_val):
        raise ValueError("scores_val và y_val phải có cùng số phần tử")

    best = {"theta": float("nan"), "val_f1": -1.0, "f1": -1.0, "percentile": float("nan")}

    for p in percentiles:
        theta = float(np.percentile(scores_val, p))
        y_pred = (scores_val >= theta).astype(int)
        val_f1 = float(f1_score(y_val, y_pred, zero_division=0))
        if val_f1 > best["val_f1"]:
            best = {"theta": theta, "val_f1": val_f1, "f1": val_f1, "percentile": float(p)}

    return best


def _safe_auc(metric_func, y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(metric_func(y_true, scores))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    scores = np.asarray(scores, dtype=float).ravel()

    return {
        "roc_auc": _safe_auc(roc_auc_score, y_true, scores),
        "pr_auc": _safe_auc(average_precision_score, y_true, scores),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _fpr_fnr(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return float(fpr), float(fnr)


def compute_fairness_metrics(y_true: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    sensitive = np.asarray(sensitive).ravel()

    groups = sorted(pd.Series(sensitive).dropna().unique().tolist())
    if len(groups) < 2:
        return {"delta_fpr": 0.0, "delta_fnr": 0.0, "eo_gap": 0.0}

    fprs, fnrs = [], []
    for g in groups:
        mask = sensitive == g
        fpr, fnr = _fpr_fnr(y_true[mask], y_pred[mask])
        fprs.append(fpr)
        fnrs.append(fnr)

    delta_fpr = float(max(fprs) - min(fprs))
    delta_fnr = float(max(fnrs) - min(fnrs))
    return {"delta_fpr": delta_fpr, "delta_fnr": delta_fnr, "eo_gap": float(delta_fpr + delta_fnr)}


def fairness_multigroup(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups,
    min_group_size: int = 20,
) -> Dict:
    """Hàm tương thích cho test cũ: tính fairness cho nhiều nhóm."""
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    groups = pd.Series(groups).reset_index(drop=True)

    per_group = {}
    fprs, fnrs = [], []
    for g, idx in groups.groupby(groups).groups.items():
        mask = np.array(list(idx), dtype=int)
        if len(mask) < min_group_size:
            continue
        fpr, fnr = _fpr_fnr(y_true[mask], y_pred[mask])
        per_group[str(g)] = {"n": int(len(mask)), "fpr": fpr, "fnr": fnr}
        fprs.append(fpr)
        fnrs.append(fnr)

    if len(fprs) < 2:
        return {"delta_fpr": 0.0, "delta_fnr": 0.0, "eo_gap": 0.0, "per_group": per_group}

    delta_fpr = float(max(fprs) - min(fprs))
    delta_fnr = float(max(fnrs) - min(fnrs))
    return {"delta_fpr": delta_fpr, "delta_fnr": delta_fnr, "eo_gap": delta_fpr + delta_fnr, "per_group": per_group}


def bin_age(age: pd.Series) -> pd.Series:
    """Chia nhóm tuổi để audit fairness theo age."""
    age_num = pd.to_numeric(age, errors="coerce")
    bins = [-np.inf, 25, 35, 45, 55, np.inf]
    labels = ["<=25", "26-35", "36-45", "46-55", ">=56"]
    return pd.cut(age_num, bins=bins, labels=labels, right=True, include_lowest=True)
