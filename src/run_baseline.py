from __future__ import annotations
from pathlib import Path
import json
import argparse

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix


# -------------------------
# Helpers: load / dense
# -------------------------
def load_sensitive(processed_dir: Path, split: str) -> pd.DataFrame:
    """
    Hỗ trợ cả parquet và csv (tùy bạn đang lưu kiểu nào).
    """
    p_parquet = processed_dir / f"sensitive_{split}_raw.parquet"
    p_csv = processed_dir / f"sensitive_{split}_raw.csv"
    if p_parquet.exists():
        return pd.read_parquet(p_parquet)
    if p_csv.exists():
        return pd.read_csv(p_csv)
    raise FileNotFoundError(f"Không tìm thấy sensitive file cho {split} trong {processed_dir}")

def ensure_dense(X):
    """
    Fix: X trong transformed.npz có thể là csr_matrix hoặc ndarray(dtype=object) chứa csr_matrix.
    IsolationForest cần ndarray float.
    """
    if isinstance(X, np.ndarray) and X.dtype == object:
        # thường là object array size=1 chứa csr_matrix
        if X.size == 1:
            X = X.item()
        else:
            X = X.ravel()[0]

    if sp.issparse(X):
        X = X.toarray()

    return np.asarray(X, dtype=np.float32)


# -------------------------
# Threshold selection (val)
# -------------------------
def choose_threshold_on_val(scores_val: np.ndarray, y_val: np.ndarray):
    """
    Chọn θ bằng cách quét percentile trên scores_val để tối đa F1 (tuning trên validation).
    Không dùng test => đúng chuẩn thực nghiệm.
    """
    best = {"theta": None, "f1": -1.0, "percentile": None}
    for p in np.linspace(80, 99.9, 200):
        theta = float(np.percentile(scores_val, p))
        yhat = (scores_val >= theta).astype(int)
        f1 = float(f1_score(y_val, yhat))
        if f1 > best["f1"]:
            best = {"theta": theta, "f1": f1, "percentile": float(p)}
    return best


# -------------------------
# Fairness: multi-group + age binning
# -------------------------
def _fpr_fnr(y_true: np.ndarray, y_pred: np.ndarray):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn + 1e-12)
    fnr = fn / (fn + tp + 1e-12)
    return float(fpr), float(fnr)

def bin_age(age_series: pd.Series) -> pd.Series:
    """
    Age binning chuẩn, dễ giải thích trong luận văn:
      <=25, 26–35, 36–45, 46–55, >=56
    """
    bins = [0, 25, 35, 45, 55, 200]
    labels = ["<=25", "26-35", "36-45", "46-55", ">=56"]
    age_num = pd.to_numeric(age_series, errors="coerce")
    return pd.cut(age_num, bins=bins, labels=labels, include_lowest=True)

def fairness_multigroup(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: pd.Series,
    min_group_size: int = 50,
):
    """
    Multi-group fairness:
      ΔFPR = max_g FPR_g - min_g FPR_g
      ΔFNR = max_g FNR_g - min_g FNR_g
      EO_gap = ΔFPR + ΔFNR
    Trả về thêm per_group để viết luận văn.
    """
    g = groups.copy()
    mask = ~g.isna()
    y_t = y_true[mask.to_numpy()]
    y_p = y_pred[mask.to_numpy()]
    g = g[mask]

    per_group = {}
    fprs, fnrs = [], []

    for val, idx in g.groupby(g).groups.items():
        idx = np.array(list(idx), dtype=int)
        if idx.size < min_group_size:
            continue
        fpr, fnr = _fpr_fnr(y_t[idx], y_p[idx])
        per_group[str(val)] = {"n": int(idx.size), "fpr": fpr, "fnr": fnr}
        fprs.append(fpr)
        fnrs.append(fnr)

    if len(fprs) <= 1:
        return {
            "delta_fpr": 0.0,
            "delta_fnr": 0.0,
            "eo_gap": 0.0,
            "per_group": per_group,
            "note": "Không đủ nhóm (sau khi lọc min_group_size) để tính gap."
        }

    delta_fpr = float(max(fprs) - min(fprs))
    delta_fnr = float(max(fnrs) - min(fnrs))
    return {
        "delta_fpr": delta_fpr,
        "delta_fnr": delta_fnr,
        "eo_gap": float(delta_fpr + delta_fnr),
        "per_group": per_group
    }


# -------------------------
# Main
# -------------------------
def main(dataset: str, seed: int):
    root = Path(".")
    processed_dir = root / "data" / "processed" / dataset
    out_dir = root / "results" / dataset / "isolation_forest"
    out_dir.mkdir(parents=True, exist_ok=True)

    npz = np.load(processed_dir / "transformed.npz", allow_pickle=True)

    X_train = ensure_dense(npz["X_train"])
    X_val   = ensure_dense(npz["X_val"])
    X_test  = ensure_dense(npz["X_test"])

    y_val  = npz["y_val"].astype(int)
    y_test = npz["y_test"].astype(int)

    # Baseline AD model
    model = IsolationForest(n_estimators=200, random_state=seed, contamination="auto")
    model.fit(X_train)

    # score_samples: lớn => bình thường => đổi dấu để lớn => bất thường
    scores_val  = -model.score_samples(X_val)
    scores_test = -model.score_samples(X_test)

    # Threshold on validation
    best = choose_threshold_on_val(scores_val, y_val)
    theta = best["theta"]

    yhat_test = (scores_test >= theta).astype(int)

    # Utility
    utility = {
        "roc_auc": float(roc_auc_score(y_test, scores_test)),
        "pr_auc": float(average_precision_score(y_test, scores_test)),   # PRIMARY
        "f1_at_theta": float(f1_score(y_test, yhat_test)),
        "theta": float(theta),
        "theta_selected_on": "validation",
        "theta_search": best
    }

    # Fairness (multi-group)
    s_test = load_sensitive(processed_dir, "test")  # DataFrame
    fairness = {}

    for col in s_test.columns:
        col_l = col.lower()
        if col_l == "age":
            g = bin_age(s_test[col])
            key = "age_bin"
        else:
            g = s_test[col].astype("object")
            key = col

        fairness[key] = fairness_multigroup(
            y_true=y_test,
            y_pred=yhat_test,
            groups=g,
            min_group_size=50
        )

    result = {
        "dataset": dataset,
        "seed": seed,
        "utility": utility,
        "fairness": fairness
    }

    out_path = out_dir / f"seed_{seed}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[DONE]", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["adult", "credit_default"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.dataset, args.seed)