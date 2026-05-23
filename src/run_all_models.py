from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from src.data_loader import SUPPORTED_DATASETS, load_processed_data, read_config
from src.metrics import choose_threshold_on_val, compute_fairness_metrics, compute_metrics
from src.models.autoencoder import AutoEncoder
from src.models.deep_svdd import score_deep_svdd, train_deep_svdd
from src.models.fast_lof import FastKDTreeLOF

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def get_seeds() -> List[int]:
    cfg = read_config()
    return list(cfg.get("reproducibility", {}).get("seeds", [42, 123, 456, 789, 1011]))


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)


def get_lof_n_neighbors(dataset_name: str) -> int:
    tuned = {"credit_default": 1000}
    return int(tuned.get(dataset_name, 20))


def get_models(seed: int, dataset_name: str) -> Dict[str, object]:
    return {
        "IsolationForest": IsolationForest(
            contamination="auto",
            random_state=seed,
            n_estimators=200,
            n_jobs=1,
        ),
        "LOF": FastKDTreeLOF(n_neighbors=get_lof_n_neighbors(dataset_name)),
        "OCSVM": OneClassSVM(kernel="rbf", nu=0.1, gamma="scale"),
    }


def maybe_sample_train(X_train: np.ndarray, seed: int, max_train_samples: int) -> np.ndarray:
    if max_train_samples <= 0 or len(X_train) <= max_train_samples:
        return X_train
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_train), size=max_train_samples, replace=False)
    return X_train[idx]


def classical_scores(model_name: str, model, X: np.ndarray) -> np.ndarray:
    # Quy ước: score càng cao => càng bất thường.
    if model_name == "IsolationForest":
        return -model.score_samples(X)
    return -model.decision_function(X)


def run_autoencoder(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    epochs: int = 20,
    lr: float = 1e-3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    model = AutoEncoder(X_train_t.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        recon = model(X_train_t)
        loss = loss_fn(recon, X_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        recon_val = model(X_val_t)
        recon_test = model(X_test_t)
        scores_val = ((X_val_t - recon_val) ** 2).mean(dim=1).cpu().numpy()
        scores_test = ((X_test_t - recon_test) ** 2).mean(dim=1).cpu().numpy()
    return scores_val, scores_test


def append_score_rows(
    detailed_rows: list,
    dataset_name: str,
    model_name: str,
    seed: int,
    scores: np.ndarray,
    threshold: float,
    sensitive: np.ndarray,
    y_test: np.ndarray,
    val_f1: float,
    threshold_percentile: float,
) -> None:
    for i in range(len(scores)):
        detailed_rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "seed": int(seed),
                "sample_index": int(i),
                "score": float(scores[i]),
                "threshold": float(threshold),
                "sensitive": int(sensitive[i]),
                "y_true": int(y_test[i]),
                "y_pred": int(scores[i] >= threshold),
                "threshold_selected_on": "validation",
                "threshold_val_f1": float(val_f1),
                "threshold_percentile": float(threshold_percentile),
            }
        )


def run_one_result(
    dataset_name: str,
    model_name: str,
    seed: int,
    scores_val: np.ndarray,
    scores_test: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    s_test: np.ndarray,
    detailed_rows: list,
) -> dict:
    best = choose_threshold_on_val(scores_val, y_val)
    theta = float(best["theta"])
    y_pred = (scores_test >= theta).astype(int)

    utility = compute_metrics(y_test, y_pred, scores_test)
    fairness = compute_fairness_metrics(y_test, y_pred, s_test)

    append_score_rows(
        detailed_rows=detailed_rows,
        dataset_name=dataset_name,
        model_name=model_name,
        seed=seed,
        scores=scores_test,
        threshold=theta,
        sensitive=s_test,
        y_test=y_test,
        val_f1=best["val_f1"],
        threshold_percentile=best["percentile"],
    )

    return {
        "dataset": dataset_name,
        "model": model_name,
        "seed": int(seed),
        "threshold": theta,
        "threshold_val_f1": float(best["val_f1"]),
        "threshold_percentile": float(best["percentile"]),
        **utility,
        **fairness,
    }


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["roc_auc", "pr_auc", "f1", "delta_fpr", "delta_fnr", "eo_gap"]
    summary = df.groupby(["dataset", "model"])[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = [f"{a}_{b}" if b else a for a, b in summary.columns.to_flat_index()]
    return summary.sort_values(["dataset", "model"]).reset_index(drop=True)


def run_experiment(max_train_samples: int = 0, ae_epochs: int = 20, svdd_epochs: int = 20) -> None:
    seeds = get_seeds()
    results = []
    detailed_rows = []

    for dataset_name in SUPPORTED_DATASETS:
        print(f"\n===== DATASET: {dataset_name} =====")
        d = load_processed_data(dataset_name)
        X_train, X_val, X_test = d["X_train"], d["X_val"], d["X_test"]
        y_train, y_val, y_test = d["y_train"], d["y_val"], d["y_test"]
        s_test = d["s_test"]

        print(
            {
                "X_train": X_train.shape,
                "X_val": X_val.shape,
                "X_test": X_test.shape,
                "y_train_sum": int(y_train.sum()),
                "sensitive_col": d["sensitive_col"],
                "sensitive_groups_test": dict(pd.Series(s_test).value_counts().sort_index()),
            }
        )

        for seed in seeds:
            set_global_seed(seed)
            X_train_seed = maybe_sample_train(X_train, seed, max_train_samples)

            for model_name, model in get_models(seed, dataset_name).items():
                print(f"Running {dataset_name} - {model_name} - seed={seed}")
                model.fit(X_train_seed)
                scores_val = classical_scores(model_name, model, X_val)
                scores_test = classical_scores(model_name, model, X_test)
                results.append(
                    run_one_result(dataset_name, model_name, seed, scores_val, scores_test, y_val, y_test, s_test, detailed_rows)
                )

            print(f"Running {dataset_name} - AutoEncoder - seed={seed}")
            scores_val_ae, scores_test_ae = run_autoencoder(
                X_train_seed, X_val, X_test, epochs=ae_epochs, seed=seed
            )
            results.append(
                run_one_result(dataset_name, "AutoEncoder", seed, scores_val_ae, scores_test_ae, y_val, y_test, s_test, detailed_rows)
            )

            print(f"Running {dataset_name} - DeepSVDD - seed={seed}")
            model_svdd, center = train_deep_svdd(X_train_seed, epochs=svdd_epochs, seed=seed)
            scores_val_svdd = score_deep_svdd(model_svdd, center, X_val)
            scores_test_svdd = score_deep_svdd(model_svdd, center, X_test)
            results.append(
                run_one_result(dataset_name, "DeepSVDD", seed, scores_val_svdd, scores_test_svdd, y_val, y_test, s_test, detailed_rows)
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results).sort_values(["dataset", "model", "seed"])
    detailed_df = pd.DataFrame(detailed_rows).sort_values(["dataset", "model", "seed", "sample_index"])
    summary_df = summarize_results(df)

    df.to_csv(RESULTS_DIR / "all_results.csv", index=False)
    df.to_csv(RESULTS_DIR / "per_seed_results.csv", index=False)
    detailed_df.to_csv(RESULTS_DIR / "score_outputs.csv", index=False)
    summary_df.to_csv(RESULTS_DIR / "summary.csv", index=False)

    print("\nĐã lưu kết quả D6 nền:")
    print(" - results/all_results.csv")
    print(" - results/per_seed_results.csv")
    print(" - results/summary.csv")
    print(" - results/score_outputs.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train-samples", type=int, default=0, help="0 = dùng toàn bộ train; >0 = lấy mẫu để chạy nhanh")
    parser.add_argument("--ae-epochs", type=int, default=20)
    parser.add_argument("--svdd-epochs", type=int, default=20)
    args = parser.parse_args()
    run_experiment(max_train_samples=args.max_train_samples, ae_epochs=args.ae_epochs, svdd_epochs=args.svdd_epochs)


if __name__ == "__main__":
    main()
