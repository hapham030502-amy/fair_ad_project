from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
INPUT_FILE = RESULTS_DIR / "all_results.csv"
NUMERIC_OUTPUT_FILE = RESULTS_DIR / "baseline_results_numeric.csv"
DISPLAY_OUTPUT_FILE = RESULTS_DIR / "baseline_results.csv"
PER_SEED_OUTPUT_FILE = RESULTS_DIR / "per_seed_results.csv"


def fmt_mean_std(mean_val, std_val, digits: int = 4) -> str:
    if pd.isna(mean_val):
        return ""
    if pd.isna(std_val):
        std_val = 0.0
    return f"{mean_val:.{digits}f} ± {std_val:.{digits}f}"


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError("Không thấy results/all_results.csv. Hãy chạy: python -m src.run_all_models")

    df = pd.read_csv(INPUT_FILE)
    required = {"dataset", "model", "seed", "roc_auc", "pr_auc", "f1", "delta_fpr", "delta_fnr", "eo_gap"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"results/all_results.csv thiếu cột: {missing}")

    df = df.sort_values(["dataset", "model", "seed"]).reset_index(drop=True)
    df.to_csv(PER_SEED_OUTPUT_FILE, index=False)

    metrics = ["roc_auc", "pr_auc", "f1", "delta_fpr", "delta_fnr", "eo_gap"]
    summary = df.groupby(["dataset", "model"])[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = [f"{a}_{b}" if b else a for a, b in summary.columns.to_flat_index()]
    summary = summary.sort_values(["dataset", "model"]).reset_index(drop=True)
    summary.to_csv(NUMERIC_OUTPUT_FILE, index=False)

    display_df = summary[["dataset", "model"]].copy()
    for m in metrics:
        display_df[m] = [fmt_mean_std(mean_val, std_val) for mean_val, std_val in zip(summary[f"{m}_mean"], summary[f"{m}_std"])]

    display_df["best_pr_auc"] = ""
    display_df["best_roc_auc"] = ""
    display_df["best_fairness_eo_gap"] = ""

    for _, idxs in summary.groupby("dataset").groups.items():
        idxs = list(idxs)
        sub = summary.loc[idxs]
        display_df.loc[sub["pr_auc_mean"].idxmax(), "best_pr_auc"] = "Best"
        display_df.loc[sub["roc_auc_mean"].idxmax(), "best_roc_auc"] = "Best"
        display_df.loc[sub["eo_gap_mean"].idxmin(), "best_fairness_eo_gap"] = "Best"

    display_df.to_csv(DISPLAY_OUTPUT_FILE, index=False)
    print(f"Đã lưu: {PER_SEED_OUTPUT_FILE}")
    print(f"Đã lưu: {NUMERIC_OUTPUT_FILE}")
    print(f"Đã lưu: {DISPLAY_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
