from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
DIR = RESULTS_DIR / "prauc"
OUT_DIR = RESULTS_DIR / "prauc"

ALL_RESULTS_FILE = RESULTS_DIR / "all_results.csv"
BEST_FILE = DIR / "best_tradeoff_table_prauc.csv"

# File in-processing thật của bạn có thể đặt ở 1 trong các vị trí dưới đây
INPROC_CANDIDATES = [
    RESULTS_DIR / "inprocessing_results_prauc.csv",
    OUT_DIR / "inprocessing_results_prauc.csv",
    RESULTS_DIR / "prauc" / "inprocessing_results_prauc.csv",
]


def fmt_mean_std(mean_val: float, std_val: float, digits: int = 4) -> str:
    if pd.isna(mean_val):
        return ""
    if pd.isna(std_val):
        std_val = 0.0
    return f"{mean_val:.{digits}f} ± {std_val:.{digits}f}"


def find_inprocessing_file() -> Optional[Path]:
    for p in INPROC_CANDIDATES:
        if p.exists():
            return p
    return None


def choose_best_base_model(all_results: pd.DataFrame) -> pd.DataFrame:
    """
    Chọn model baseline tốt nhất theo mean PR-AUC của mỗi dataset.
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


def build_baseline_table(
    all_results: pd.DataFrame,
    base_models: pd.DataFrame,
) -> pd.DataFrame:
    df = all_results.merge(
        base_models[["dataset", "base_model"]],
        left_on=["dataset", "model"],
        right_on=["dataset", "base_model"],
        how="inner",
    )

    baseline = (
        df.groupby(["dataset", "base_model"], as_index=False)
        .agg(
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            eo_gap_mean=("eo_gap", "mean"),
            eo_gap_std=("eo_gap", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            delta_fpr_mean=("delta_fpr", "mean"),
            delta_fnr_mean=("delta_fnr", "mean"),
        )
        .copy()
    )

    baseline["method"] = "Baseline"
    baseline["rule"] = "No post-processing"
    baseline["param"] = np.nan
    baseline["note"] = "Best base model selected by mean PR-AUC from all_results.csv"

    return baseline[
        [
            "dataset",
            "base_model",
            "method",
            "rule",
            "param",
            "pr_auc_mean",
            "pr_auc_std",
            "eo_gap_mean",
            "eo_gap_std",
            "f1_mean",
            "f1_std",
            "delta_fpr_mean",
            "delta_fnr_mean",
            "note",
        ]
    ].copy()


def build_best_post_table(best: pd.DataFrame) -> pd.DataFrame:
    """
    Từ bảng D9 best per rule, chọn 1 post-processing tốt nhất cho mỗi dataset
    theo distance_to_ideal nhỏ nhất.
    """
    required = {
        "dataset",
        "base_model",
        "rule",
        "param",
        "pr_auc_mean",
        "pr_auc_std",
        "eo_gap_mean",
        "eo_gap_std",
        "f1_mean",
        "f1_std",
        "distance_to_ideal",
    }
    missing = required - set(best.columns)
    if missing:
        raise ValueError(f"best_tradeoff_table_prauc.csv thiếu cột: {missing}")

    idx = best.groupby("dataset")["distance_to_ideal"].idxmin()
    best_post = best.loc[idx].copy().reset_index(drop=True)

    if "delta_fpr_mean" not in best_post.columns:
        best_post["delta_fpr_mean"] = np.nan
    if "delta_fnr_mean" not in best_post.columns:
        best_post["delta_fnr_mean"] = np.nan

    best_post["method"] = "Post-processing"
    best_post["note"] = "Best post-processing chosen by minimum distance_to_ideal from D9"

    return best_post[
        [
            "dataset",
            "base_model",
            "method",
            "rule",
            "param",
            "pr_auc_mean",
            "pr_auc_std",
            "eo_gap_mean",
            "eo_gap_std",
            "f1_mean",
            "f1_std",
            "delta_fpr_mean",
            "delta_fnr_mean",
            "note",
        ]
    ].copy()


def build_inprocessing_table(inproc_file: Path) -> pd.DataFrame:
    df = pd.read_csv(inproc_file)

    required = {
        "dataset",
        "model",
        "pr_auc_mean",
        "pr_auc_std",
        "eo_gap_mean",
        "eo_gap_std",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"File in-processing {inproc_file.name} thiếu cột bắt buộc: {missing}"
        )

    if "f1_mean" not in df.columns:
        df["f1_mean"] = np.nan
    if "f1_std" not in df.columns:
        df["f1_std"] = np.nan
    if "delta_fpr_mean" not in df.columns:
        df["delta_fpr_mean"] = np.nan
    if "delta_fnr_mean" not in df.columns:
        df["delta_fnr_mean"] = np.nan

    out = df.copy()
    out["base_model"] = out["model"]
    out["method"] = "In-processing"
    out["rule"] = "In-processing method"
    out["param"] = np.nan
    out["note"] = f"Loaded from {inproc_file.name}"

    return out[
        [
            "dataset",
            "base_model",
            "method",
            "rule",
            "param",
            "pr_auc_mean",
            "pr_auc_std",
            "eo_gap_mean",
            "eo_gap_std",
            "f1_mean",
            "f1_std",
            "delta_fpr_mean",
            "delta_fnr_mean",
            "note",
        ]
    ].copy()


def build_display_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["PR-AUC"] = [
        fmt_mean_std(m, s) for m, s in zip(out["pr_auc_mean"], out["pr_auc_std"])
    ]
    out["EO_gap"] = [
        fmt_mean_std(m, s) for m, s in zip(out["eo_gap_mean"], out["eo_gap_std"])
    ]
    out["F1"] = [
        fmt_mean_std(m, s) if pd.notna(m) else "" for m, s in zip(out["f1_mean"], out["f1_std"])
    ]
    out["Δ_FPR"] = out["delta_fpr_mean"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    out["Δ_FNR"] = out["delta_fnr_mean"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")

    return out[
        [
            "dataset",
            "base_model",
            "method",
            "rule",
            "param",
            "PR-AUC",
            "EO_gap",
            "F1",
            "Δ_FPR",
            "Δ_FNR",
            "note",
        ]
    ].copy()


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for dataset, sub in df.groupby("dataset"):
        # tốt nhất về utility
        best_utility = sub.sort_values(
            ["pr_auc_mean", "eo_gap_mean"],
            ascending=[False, True]
        ).iloc[0]

        # tốt nhất về fairness
        best_fairness = sub.sort_values(
            ["eo_gap_mean", "pr_auc_mean"],
            ascending=[True, False]
        ).iloc[0]

        # trade-off đơn giản: chuẩn hóa utility và fairness
        sub = sub.copy()
        pr_min, pr_max = sub["pr_auc_mean"].min(), sub["pr_auc_mean"].max()
        eo_min, eo_max = sub["eo_gap_mean"].min(), sub["eo_gap_mean"].max()

        def norm(v, lo, hi):
            if hi - lo == 0:
                return 1.0
            return (v - lo) / (hi - lo)

        sub["tradeoff_score"] = sub.apply(
            lambda r: 0.5 * norm(r["pr_auc_mean"], pr_min, pr_max)
                    + 0.5 * (1.0 - norm(r["eo_gap_mean"], eo_min, eo_max)),
            axis=1,
        )

        best_tradeoff = sub.sort_values(
            ["tradeoff_score", "pr_auc_mean"],
            ascending=[False, False]
        ).iloc[0]

        rows.append(
            {
                "dataset": dataset,
                "best_utility_method": best_utility["method"],
                "best_utility_model": best_utility["base_model"],
                "best_utility_pr_auc": best_utility["pr_auc_mean"],
                "best_fairness_method": best_fairness["method"],
                "best_fairness_model": best_fairness["base_model"],
                "best_fairness_eo_gap": best_fairness["eo_gap_mean"],
                "best_tradeoff_method": best_tradeoff["method"],
                "best_tradeoff_model": best_tradeoff["base_model"],
                "best_tradeoff_rule": best_tradeoff["rule"],
                "best_tradeoff_score": best_tradeoff["tradeoff_score"],
            }
        )

    return pd.DataFrame(rows)


def plot_dataset_comparison(df: pd.DataFrame, dataset: str, out_path: Path) -> None:
    sub = df[df["dataset"] == dataset].copy()

    plt.figure(figsize=(8, 6))

    color_map = {
        "Baseline": "#1f77b4",
        "Post-processing": "#ff7f0e",
        "In-processing": "#2ca02c",
    }

    for _, row in sub.iterrows():
        c = color_map.get(row["method"], "#7f7f7f")
        plt.scatter(row["eo_gap_mean"], row["pr_auc_mean"], s=100, color=c)

        label = f"{row['method']} | {row['base_model']}"
        if row["method"] == "Post-processing":
            label += f" | {row['rule']}"

        plt.annotate(
            label,
            (row["eo_gap_mean"], row["pr_auc_mean"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
        )

    plt.xlabel("EO_gap (lower is better)")
    plt.ylabel("PR-AUC (higher is better)")
    plt.title(f"Comparison - {dataset}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    if not ALL_RESULTS_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {ALL_RESULTS_FILE}")
    if not BEST_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {BEST_FILE}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = pd.read_csv(ALL_RESULTS_FILE)
    best = pd.read_csv(BEST_FILE)

    base_models = choose_best_base_model(all_results)
    baseline = build_baseline_table(all_results, base_models)
    post = build_best_post_table(best)

    tables = [baseline, post]

    inproc_file = find_inprocessing_file()
    if inproc_file is not None:
        inproc = build_inprocessing_table(inproc_file)
        tables.append(inproc)
        print(f"Đã tìm thấy file in-processing: {inproc_file}")
    else:
        print("Chưa có file in-processing_results_prauc.csv -> D10 hiện chỉ gồm Baseline và Post-processing.")

    comparison = pd.concat(tables, ignore_index=True, sort=False)

    method_order = {"Baseline": 0, "Post-processing": 1, "In-processing": 2}
    comparison["method_order"] = comparison["method"].map(method_order).fillna(99)
    comparison = comparison.sort_values(
        ["dataset", "method_order", "base_model"]
    ).drop(columns=["method_order"])

    display_table = build_display_table(comparison)
    summary_table = build_summary_table(comparison)

    comparison.to_csv(OUT_DIR / "comparison_numeric_prauc.csv", index=False)
    display_table.to_csv(OUT_DIR / "comparison_display_prauc.csv", index=False)
    summary_table.to_csv(OUT_DIR / "analysis_summary_prauc.csv", index=False)

    for dataset in comparison["dataset"].unique():
        plot_dataset_comparison(comparison, dataset, OUT_DIR / f"scatter_{dataset}_prauc.png")
        plot_dataset_comparison(comparison, dataset, OUT_DIR / f"scatter_{dataset}_prauc.pdf")

    print("\nĐã tạo xong D10 tại:")
    print(f"  {OUT_DIR}")
    print("\nCác file chính:")
    print(" - comparison_numeric_prauc.csv")
    print(" - comparison_display_prauc.csv")
    print(" - analysis_summary_prauc.csv")
    print(" - scatter_<dataset>_prauc.png/.pdf")


if __name__ == "__main__":
    main()