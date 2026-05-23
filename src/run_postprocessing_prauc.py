from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
OUT_DIR = RESULTS_DIR / "prauc"

ALL_RESULTS_FILE = RESULTS_DIR / "all_results.csv"
SCORE_OUTPUTS_FILE = RESULTS_DIR / "score_outputs.csv"


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)

    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def compute_pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Tự tính Average Precision / PR-AUC không phụ thuộc sklearn.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if len(np.unique(y_true)) < 2:
        return 0.0

    order = np.argsort(scores)[::-1]
    y_sorted = y_true[order]

    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)

    total_pos = int((y_true == 1).sum())
    if total_pos == 0:
        return 0.0

    precision = tp_cum / (tp_cum + fp_cum)
    recall = tp_cum / total_pos

    # average precision
    ap = 0.0
    prev_recall = 0.0
    for p, r in zip(precision, recall):
        ap += p * max(0.0, r - prev_recall)
        prev_recall = r

    return float(ap)


def compute_group_fpr_fnr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
) -> Tuple[float, float, float]:
    groups = sorted(pd.Series(sensitive).dropna().unique().tolist())
    if len(groups) < 2:
        return 0.0, 0.0, 0.0

    fprs = []
    fnrs = []

    for g in groups[:2]:
        mask = sensitive == g
        yt = y_true[mask]
        yp = y_pred[mask]

        fp = int(((yp == 1) & (yt == 0)).sum())
        tn = int(((yp == 0) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        tp = int(((yp == 1) & (yt == 1)).sum())

        fpr = safe_div(fp, fp + tn)
        fnr = safe_div(fn, fn + tp)

        fprs.append(fpr)
        fnrs.append(fnr)

    delta_fpr = abs(fprs[0] - fprs[1])
    delta_fnr = abs(fnrs[0] - fnrs[1])
    eo_gap = delta_fpr + delta_fnr
    return float(delta_fpr), float(delta_fnr), float(eo_gap)


def choose_best_base_model(all_results: pd.DataFrame) -> pd.DataFrame:
    """
    Chọn base model cho mỗi dataset theo mean PR-AUC cao nhất.
    """
    grp = (
        all_results.groupby(["dataset", "model"], as_index=False)["pr_auc"]
        .mean()
        .rename(columns={"pr_auc": "mean_pr_auc"})
    )

    idx = grp.groupby("dataset")["mean_pr_auc"].idxmax()
    chosen = grp.loc[idx].copy().reset_index(drop=True)
    chosen = chosen.rename(columns={"model": "base_model"})
    return chosen[["dataset", "base_model", "mean_pr_auc"]]


def global_threshold_rule(
    scores: np.ndarray,
    percent: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    adjusted_score = score - global_threshold
    Như vậy ranking có thể giữ nguyên, nhưng đây là baseline global-threshold rule.
    """
    theta = float(np.percentile(scores, percent))
    adjusted_score = scores - theta
    y_pred = (adjusted_score >= 0).astype(int)
    return adjusted_score, y_pred, {"threshold": theta}


def per_group_threshold_rule(
    scores: np.ndarray,
    sensitive: np.ndarray,
    target_positive_rate: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    adjusted_score_g = score - theta_g
    Với theta_g là quantile riêng từng group.
    Điều này thay đổi ranking toàn cục giữa các nhóm -> PR-AUC thay đổi có nghĩa.
    """
    adjusted_score = np.zeros(len(scores), dtype=float)
    y_pred = np.zeros(len(scores), dtype=int)
    info: Dict[str, float] = {}

    groups = sorted(pd.Series(sensitive).dropna().unique().tolist())
    for g in groups:
        mask = sensitive == g
        s = scores[mask]

        q = 1.0 - target_positive_rate
        q = min(max(q, 0.0), 1.0)

        theta_g = float(np.quantile(s, q))
        adjusted_score[mask] = s - theta_g
        y_pred[mask] = (adjusted_score[mask] >= 0).astype(int)

        info[f"threshold_group_{g}"] = theta_g

    return adjusted_score, y_pred, info


def topk_per_group_rule(
    scores: np.ndarray,
    sensitive: np.ndarray,
    ratio: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Tạo adjusted_score theo percentile-rank trong từng group:
      adjusted_score = rank_percentile - (1 - ratio)
    Khi ratio thay đổi thì adjusted_score thay đổi,
    đồng thời ranking toàn cục theo "độ nổi bật trong group" cũng thay đổi.
    """
    adjusted_score = np.zeros(len(scores), dtype=float)
    y_pred = np.zeros(len(scores), dtype=int)

    ratio = float(min(max(ratio, 0.0), 1.0))
    groups = sorted(pd.Series(sensitive).dropna().unique().tolist())

    for g in groups:
        idx = np.where(sensitive == g)[0]
        if len(idx) == 0:
            continue

        s = scores[idx]
        order = np.argsort(s)
        ranks = np.empty(len(s), dtype=float)
        ranks[order] = np.arange(len(s), dtype=float)

        if len(s) > 1:
            rank_percentile = ranks / (len(s) - 1)
        else:
            rank_percentile = np.zeros(len(s), dtype=float)

        adjusted_score[idx] = rank_percentile - (1.0 - ratio)
        y_pred[idx] = (adjusted_score[idx] >= 0).astype(int)

    return adjusted_score, y_pred, {}


def evaluate_one_setting(
    y_true: np.ndarray,
    adjusted_score: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
) -> Dict[str, float]:
    pr_auc = compute_pr_auc(y_true, adjusted_score)
    f1 = compute_f1(y_true, y_pred)
    delta_fpr, delta_fnr, eo_gap = compute_group_fpr_fnr(y_true, y_pred, sensitive)

    return {
        "pr_auc": pr_auc,
        "f1": f1,
        "delta_fpr": delta_fpr,
        "delta_fnr": delta_fnr,
        "eo_gap": eo_gap,
    }


def is_pareto_optimal(points: pd.DataFrame) -> pd.Series:
    """
    Tối ưu đồng thời:
      - PR-AUC càng cao càng tốt
      - EO_gap càng thấp càng tốt
    """
    pareto = np.ones(len(points), dtype=bool)

    for i in range(len(points)):
        pr_i = points.iloc[i]["pr_auc_mean"]
        eo_i = points.iloc[i]["eo_gap_mean"]

        for j in range(len(points)):
            if i == j:
                continue

            pr_j = points.iloc[j]["pr_auc_mean"]
            eo_j = points.iloc[j]["eo_gap_mean"]

            dominates = (
                (pr_j >= pr_i and eo_j <= eo_i)
                and (pr_j > pr_i or eo_j < eo_i)
            )
            if dominates:
                pareto[i] = False
                break

    return pd.Series(pareto, index=points.index)


def distance_to_ideal(
    pr_auc_mean: float,
    eo_gap_mean: float,
    max_pr: float,
    min_eo: float,
) -> float:
    """
    Điểm lý tưởng = (max PR-AUC, min EO_gap).
    """
    return float(np.sqrt((max_pr - pr_auc_mean) ** 2 + (eo_gap_mean - min_eo) ** 2))


def plot_dataset_pareto(df_plot: pd.DataFrame, dataset: str, out_path: Path) -> None:
    plt.figure(figsize=(8, 6))

    for rule in df_plot["rule"].unique():
        sub = df_plot[df_plot["rule"] == rule].sort_values("eo_gap_mean")
        plt.plot(
            sub["eo_gap_mean"],
            sub["pr_auc_mean"],
            marker="o",
            linewidth=1.5,
            label=rule,
        )

    pareto = df_plot[df_plot["is_pareto"]].copy()
    if not pareto.empty:
        plt.scatter(
            pareto["eo_gap_mean"],
            pareto["pr_auc_mean"],
            s=120,
            marker="*",
            label="Pareto-optimal",
        )

    plt.xlabel("EO_gap (lower is better)")
    plt.ylabel("PR-AUC (higher is better)")
    plt.title(f"Pareto Front - {dataset}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    if not ALL_RESULTS_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {ALL_RESULTS_FILE}")
    if not SCORE_OUTPUTS_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {SCORE_OUTPUTS_FILE}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = pd.read_csv(ALL_RESULTS_FILE)
    score_outputs = pd.read_csv(SCORE_OUTPUTS_FILE)

    required_all = {"dataset", "model", "seed", "pr_auc", "f1", "eo_gap"}
    required_scores = {"dataset", "model", "seed", "score", "sensitive", "y_true"}

    missing_all = required_all - set(all_results.columns)
    missing_scores = required_scores - set(score_outputs.columns)

    if missing_all:
        raise ValueError(f"Thiếu cột trong all_results.csv: {missing_all}")
    if missing_scores:
        raise ValueError(f"Thiếu cột trong score_outputs.csv: {missing_scores}")

    base_models = choose_best_base_model(all_results)
    candidate_rows: List[Dict[str, float]] = []

    global_grid = np.arange(80.0, 100.0, 0.5)
    per_group_grid = np.linspace(0.01, 0.30, 40)
    topk_grid = np.linspace(0.02, 0.30, 40)

    for _, base_row in base_models.iterrows():
        dataset = base_row["dataset"]
        base_model = base_row["base_model"]

        base_scores = score_outputs[
            (score_outputs["dataset"] == dataset)
            & (score_outputs["model"] == base_model)
        ].copy()

        if base_scores.empty:
            continue

        for seed, sub in base_scores.groupby("seed"):
            sub = sub.sort_index()

            scores = sub["score"].to_numpy(dtype=float)
            y_true = sub["y_true"].to_numpy(dtype=int)
            sensitive = sub["sensitive"].to_numpy()

            # Rule 1: Global threshold
            for p in global_grid:
                adjusted_score, y_pred, extra = global_threshold_rule(scores, p)
                metrics = evaluate_one_setting(y_true, adjusted_score, y_pred, sensitive)

                candidate_rows.append(
                    {
                        "dataset": dataset,
                        "base_model": base_model,
                        "seed": int(seed),
                        "rule": "Global threshold",
                        "param": float(p),
                        "threshold": extra["threshold"],
                        **metrics,
                        "threshold_group_0": np.nan,
                        "threshold_group_1": np.nan,
                    }
                )

            # Rule 2: Per-group threshold
            for q in per_group_grid:
                adjusted_score, y_pred, extra = per_group_threshold_rule(scores, sensitive, q)
                metrics = evaluate_one_setting(y_true, adjusted_score, y_pred, sensitive)

                candidate_rows.append(
                    {
                        "dataset": dataset,
                        "base_model": base_model,
                        "seed": int(seed),
                        "rule": "Per-group threshold",
                        "param": float(q),
                        "threshold": np.nan,
                        **metrics,
                        "threshold_group_0": extra.get("threshold_group_0", np.nan),
                        "threshold_group_1": extra.get("threshold_group_1", np.nan),
                    }
                )

            # Rule 3: Top-k per group
            for q in topk_grid:
                adjusted_score, y_pred, _ = topk_per_group_rule(scores, sensitive, q)
                metrics = evaluate_one_setting(y_true, adjusted_score, y_pred, sensitive)

                candidate_rows.append(
                    {
                        "dataset": dataset,
                        "base_model": base_model,
                        "seed": int(seed),
                        "rule": "Top-k per group",
                        "param": float(q),
                        "threshold": np.nan,
                        **metrics,
                        "threshold_group_0": np.nan,
                        "threshold_group_1": np.nan,
                    }
                )

    candidates = pd.DataFrame(candidate_rows)
    candidates.to_csv(OUT_DIR / "postprocessing_candidates_prauc.csv", index=False)

    mean_table = (
        candidates.groupby(["dataset", "base_model", "rule", "param"], as_index=False)
        .agg(
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            eo_gap_mean=("eo_gap", "mean"),
            eo_gap_std=("eo_gap", "std"),
            delta_fpr_mean=("delta_fpr", "mean"),
            delta_fnr_mean=("delta_fnr", "mean"),
        )
        .sort_values(["dataset", "rule", "param"])
        .reset_index(drop=True)
    )

    mean_table["is_pareto"] = False
    mean_table["distance_to_ideal"] = np.nan

    for dataset, idx in mean_table.groupby("dataset").groups.items():
        idx = list(idx)
        sub = mean_table.loc[idx].copy()

        pareto_mask = is_pareto_optimal(sub)
        mean_table.loc[sub.index, "is_pareto"] = pareto_mask.values

        max_pr = sub["pr_auc_mean"].max()
        min_eo = sub["eo_gap_mean"].min()

        distances = [
            distance_to_ideal(r["pr_auc_mean"], r["eo_gap_mean"], max_pr, min_eo)
            for _, r in sub.iterrows()
        ]
        mean_table.loc[sub.index, "distance_to_ideal"] = distances

    mean_table.to_csv(OUT_DIR / "pareto_front_candidates_mean_prauc.csv", index=False)

    baseline_summary = (
        all_results.merge(
            base_models[["dataset", "base_model"]],
            left_on=["dataset", "model"],
            right_on=["dataset", "base_model"],
            how="inner",
        )
        .groupby(["dataset", "base_model"], as_index=False)
        .agg(
            baseline_pr_auc_mean=("pr_auc", "mean"),
            baseline_pr_auc_std=("pr_auc", "std"),
            baseline_eo_gap_mean=("eo_gap", "mean"),
            baseline_f1_mean=("f1", "mean"),
        )
    )

    best_tradeoff = (
        mean_table.groupby(["dataset", "base_model", "rule"], as_index=False)
        .apply(lambda x: x.loc[x["distance_to_ideal"].idxmin()])
        .reset_index(drop=True)
    )

    best_tradeoff = best_tradeoff.merge(
        baseline_summary,
        on=["dataset", "base_model"],
        how="left",
    )

    best_tradeoff.to_csv(OUT_DIR / "best_tradeoff_table_prauc.csv", index=False)

    for dataset, sub in mean_table.groupby("dataset"):
        plot_dataset_pareto(sub, dataset, OUT_DIR / f"pareto_{dataset}_prauc.png")
        plot_dataset_pareto(sub, dataset, OUT_DIR / f"pareto_{dataset}_prauc.pdf")

    print("Đã tạo xong D9 (PR-AUC version) tại thư mục:")
    print(f"  {OUT_DIR}")
    print("\nCác file chính:")
    print(" - postprocessing_candidates_prauc.csv")
    print(" - pareto_front_candidates_mean_prauc.csv")
    print(" - best_tradeoff_table_prauc.csv")
    print(" - pareto_<dataset>_prauc.png/.pdf")

if __name__ == "__main__":
    main()