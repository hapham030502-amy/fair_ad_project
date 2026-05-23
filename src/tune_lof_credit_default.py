from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data_loader import load_processed_data
from src.metrics import choose_threshold_on_val, compute_fairness_metrics, compute_metrics
from src.models.fast_lof import FastKDTreeLOF

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
OUT_FILE = RESULTS_DIR / "lof_tuning_credit_default.csv"
CANDIDATE_N_NEIGHBORS = [5, 10, 20, 35, 50, 75, 100, 150, 200, 300, 500, 750, 1000]


def main() -> None:
    d = load_processed_data("credit_default")
    X_train, X_val, X_test = d["X_train"], d["X_val"], d["X_test"]
    y_val, y_test = d["y_val"], d["y_test"]
    s_test = d["s_test"]

    rows = []
    for k in CANDIDATE_N_NEIGHBORS:
        model = FastKDTreeLOF(n_neighbors=k)
        model.fit(X_train)
        scores_val = -model.decision_function(X_val)
        scores_test = -model.decision_function(X_test)

        best = choose_threshold_on_val(scores_val, y_val)
        y_pred = (scores_test >= best["theta"]).astype(int)

        rows.append({
            "dataset": "credit_default",
            "model": "LOF",
            "n_neighbors": int(k),
            "validation_metric": "pr_auc",
            "val_pr_auc": float(average_precision_score(y_val, scores_val)),
            "val_roc_auc": float(roc_auc_score(y_val, scores_val)),
            "threshold": float(best["theta"]),
            "threshold_val_f1": float(best["val_f1"]),
            "threshold_percentile": float(best["percentile"]),
            **compute_metrics(y_test, y_pred, scores_test),
            **compute_fairness_metrics(y_test, y_pred, s_test),
        })

    tuning = pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False).reset_index(drop=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tuning.to_csv(OUT_FILE, index=False)
    print(f"Đã lưu {OUT_FILE}")
    print("Best LOF setting theo validation PR-AUC:")
    print(tuning.head(1).to_string(index=False))


if __name__ == "__main__":
    main()
