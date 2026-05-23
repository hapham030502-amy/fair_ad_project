from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "results" / "score_outputs.csv"
OUT_DIR = ROOT / "results" / "figures"
OUT_REP_SEEDS = ROOT / "results" / "D7_representative_seeds.csv"
OUT_INDEX = ROOT / "results" / "D7_figure_index.csv"


def _compute_per_seed_pr_auc(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, model, seed), sub in df.groupby(["dataset", "model", "seed"]):
        y_true = sub["y_true"].to_numpy()
        scores = sub["score"].to_numpy()
        pr_auc = float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) > 1 else 0.0
        rows.append({"dataset": dataset, "model": model, "seed": int(seed), "pr_auc": pr_auc})
    return pd.DataFrame(rows)


def choose_representative_seed(df: pd.DataFrame) -> pd.DataFrame:
    per_seed = _compute_per_seed_pr_auc(df)
    rows = []
    for (dataset, model), sub in per_seed.groupby(["dataset", "model"]):
        mean_pr = sub["pr_auc"].mean()
        sub = sub.copy()
        sub["distance_to_mean"] = (sub["pr_auc"] - mean_pr).abs()
        chosen = sub.sort_values(["distance_to_mean", "seed"]).iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "chosen_seed": int(chosen["seed"]),
                "chosen_seed_pr_auc": float(chosen["pr_auc"]),
                "mean_pr_auc_5seeds": float(mean_pr),
                "abs_distance": float(chosen["distance_to_mean"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "model"]).reset_index(drop=True)


def _draw_hist(ax, df_plot: pd.DataFrame, xlim=None, zoom: bool = False) -> None:
    groups = sorted(df_plot["sensitive"].dropna().unique().tolist())
    for g in groups:
        sub = df_plot[df_plot["sensitive"] == g]
        ax.hist(sub["score"], bins=30, alpha=0.5, density=True, label=f"Group {g} (n={len(sub)})")
        mean_score = sub["score"].mean()
        ax.axvline(mean_score, linestyle="--", linewidth=1.5, label=f"Mean group {g} = {mean_score:.3f}")

    threshold = float(df_plot["threshold"].iloc[0])
    ax.axvline(threshold, linestyle="-", linewidth=2, label=f"Threshold = {threshold:.3f}")
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Density")
    ax.set_title("Zoom 1%-99%" if zoom else "Full range")


def plot_one(df_plot: pd.DataFrame, dataset: str, model: str, seed: int, out_dir: Path):
    scores = df_plot["score"].to_numpy()
    q_low, q_high = np.quantile(scores, [0.01, 0.99])
    if q_low == q_high:
        q_low, q_high = np.quantile(scores, [0.005, 0.995])
    pad = 0.05 * (q_high - q_low) if q_high > q_low else 0.1
    zoom_xlim = (q_low - pad, q_high + pad)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _draw_hist(axes[0], df_plot, xlim=None, zoom=False)
    _draw_hist(axes[1], df_plot, xlim=zoom_xlim, zoom=True)
    fig.suptitle(f"Score distribution by sensitive group\nDataset={dataset}, Model={model}, Seed={seed}", fontsize=15)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8)
    plt.tight_layout(rect=[0, 0, 0.86, 0.92])

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset}_{model}_score_dist"
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    return png_path, pdf_path


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError("Không tìm thấy results/score_outputs.csv. Hãy chạy: python -m src.run_all_models")

    df = pd.read_csv(INPUT_FILE)
    required_cols = {"dataset", "model", "seed", "score", "threshold", "sensitive", "y_true"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột trong score_outputs.csv: {missing}")

    rep_df = choose_representative_seed(df)
    OUT_REP_SEEDS.parent.mkdir(parents=True, exist_ok=True)
    rep_df.to_csv(OUT_REP_SEEDS, index=False)

    figure_rows = []
    for _, row in rep_df.iterrows():
        dataset = row["dataset"]
        model = row["model"]
        chosen_seed = int(row["chosen_seed"])
        sub = df[(df["dataset"] == dataset) & (df["model"] == model) & (df["seed"] == chosen_seed)].copy()
        png_path, pdf_path = plot_one(sub, dataset, model, chosen_seed, OUT_DIR)
        figure_rows.append({
            "dataset": dataset,
            "model": model,
            "chosen_seed": chosen_seed,
            "png_file": str(png_path.relative_to(ROOT)).replace("\\", "/"),
            "pdf_file": str(pdf_path.relative_to(ROOT)).replace("\\", "/"),
        })

    pd.DataFrame(figure_rows).to_csv(OUT_INDEX, index=False)
    print(f"Đã lưu: {OUT_REP_SEEDS}")
    print(f"Đã lưu: {OUT_INDEX}")
    print(f"Đã tạo hình tại: {OUT_DIR}")


if __name__ == "__main__":
    main()
