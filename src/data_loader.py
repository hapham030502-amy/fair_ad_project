from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"
PROCESSED_ROOT = ROOT / "data" / "processed"
SUPPORTED_DATASETS = ["adult", "credit_default"]

DEFAULT_SENSITIVE = {
    "adult": "sex",
    "credit_default": "sex",
}


def read_config(config_path: Path = CONFIG_PATH) -> Dict:
    if not config_path.exists() or yaml is None:
        return {
            "reproducibility": {"seeds": [42, 123, 456, 789, 1011]},
            "datasets": {
                "adult": {"default_sensitive": "sex"},
                "credit_default": {"default_sensitive": "sex"},
            },
        }
    with open(config_path, "r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f) or {}


def ensure_dense(X) -> np.ndarray:
    if isinstance(X, np.ndarray) and X.dtype == object:
        if X.size == 1:
            X = X.item()
        else:
            X = X.ravel()[0]
    if sparse.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def _read_table(parquet_path: Path, csv_path: Path) -> pd.DataFrame:
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception as e:
            raise RuntimeError(
                f"Không đọc được {parquet_path}. Cần cài pyarrow. "
                f"Lệnh cài: pip install pyarrow. Lỗi gốc: {e}"
            ) from e
    raise FileNotFoundError(f"Không tìm thấy {parquet_path.name} hoặc {csv_path.name}")


def _read_sensitive_file(dataset_dir: Path, split: str) -> pd.DataFrame:
    return _read_table(
        dataset_dir / f"sensitive_{split}_raw.parquet",
        dataset_dir / f"sensitive_{split}_raw.csv",
    )


def _binary_sensitive(dataset_name: str, s_df: pd.DataFrame, sensitive_col: str) -> np.ndarray:
    if sensitive_col not in s_df.columns:
        raise ValueError(f"Không có cột sensitive '{sensitive_col}' trong {list(s_df.columns)}")

    s = s_df[sensitive_col]

    if dataset_name == "adult" and sensitive_col == "sex":
        return (s.astype(str).str.strip().str.lower() == "male").astype(int).to_numpy()

    if dataset_name == "adult" and sensitive_col == "race":
        # 1 = White/majority, 0 = other groups. Dùng để binary audit ổn định.
        return (s.astype(str).str.strip().str.lower() == "white").astype(int).to_numpy()

    if dataset_name == "credit_default" and sensitive_col == "sex":
        # Theo UCI Credit Default: SEX=1 male, SEX=2 female.
        return (pd.to_numeric(s, errors="coerce") == 1).astype(int).to_numpy()

    if sensitive_col == "age":
        s_num = pd.to_numeric(s, errors="coerce")
        median_age = float(s_num.median())
        return (s_num >= median_age).astype(int).to_numpy()

    s_num = pd.to_numeric(s, errors="coerce")
    vals = sorted(pd.Series(s_num.dropna().unique()).tolist())
    if len(vals) == 2:
        return (s_num == vals[-1]).astype(int).to_numpy()

    mode_val = s.astype(str).str.strip().mode(dropna=True).iloc[0]
    return (s.astype(str).str.strip() == mode_val).astype(int).to_numpy()


def _load_sensitive_from_files(
    dataset_name: str,
    split: str,
    processed_dir: Path,
    sensitive_col: Optional[str] = None,
) -> Tuple[np.ndarray, str]:
    cfg = read_config()
    sensitive_name = (
        sensitive_col
        or cfg.get("datasets", {}).get(dataset_name, {}).get("default_sensitive")
        or DEFAULT_SENSITIVE.get(dataset_name, "sex")
    )
    s_df = _read_sensitive_file(processed_dir, split)
    return _binary_sensitive(dataset_name, s_df, sensitive_name), sensitive_name


def _get_sensitive_arrays(
    dataset_name: str,
    dataset_dir: Path,
    data: np.lib.npyio.NpzFile,
    sensitive_col: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    cfg = read_config()
    sensitive_name = (
        sensitive_col
        or cfg.get("datasets", {}).get(dataset_name, {}).get("default_sensitive")
        or DEFAULT_SENSITIVE.get(dataset_name, "sex")
    )

    if all(k in data.files for k in ["s_train", "s_val", "s_test"]):
        return (
            np.asarray(data["s_train"], dtype=int).ravel(),
            np.asarray(data["s_val"], dtype=int).ravel(),
            np.asarray(data["s_test"], dtype=int).ravel(),
            str(data["sensitive_name"]) if "sensitive_name" in data.files else sensitive_name,
        )

    s_train, sensitive_name = _load_sensitive_from_files(dataset_name, "train", dataset_dir, sensitive_name)
    s_val, _ = _load_sensitive_from_files(dataset_name, "val", dataset_dir, sensitive_name)
    s_test, _ = _load_sensitive_from_files(dataset_name, "test", dataset_dir, sensitive_name)
    return s_train, s_val, s_test, sensitive_name


def _validate_split_shapes(dataset_name: str, X: np.ndarray, y: np.ndarray, s: np.ndarray, split: str) -> None:
    if len(X) != len(y):
        raise ValueError(f"{dataset_name}/{split}: số dòng X={len(X)} không khớp y={len(y)}")
    if len(y) != len(s):
        raise ValueError(f"{dataset_name}/{split}: số dòng y={len(y)} không khớp sensitive={len(s)}")


def load_processed_data(dataset_name: str, sensitive_col: Optional[str] = None) -> Dict[str, np.ndarray]:
    dataset_name = dataset_name.lower().strip()
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Dataset không hỗ trợ: {dataset_name}. Chọn một trong {SUPPORTED_DATASETS}")

    dataset_dir = PROCESSED_ROOT / dataset_name
    npz_path = dataset_dir / "transformed.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    required = {"X_train", "X_val", "X_test", "y_train", "y_val", "y_test"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"{npz_path} thiếu các key: {sorted(missing)}")

    X_train = ensure_dense(data["X_train"])
    X_val = ensure_dense(data["X_val"])
    X_test = ensure_dense(data["X_test"])

    y_train = np.asarray(data["y_train"], dtype=int).ravel()
    y_val = np.asarray(data["y_val"], dtype=int).ravel()
    y_test = np.asarray(data["y_test"], dtype=int).ravel()

    s_train, s_val, s_test, sensitive_name = _get_sensitive_arrays(dataset_name, dataset_dir, data, sensitive_col)

    _validate_split_shapes(dataset_name, X_train, y_train, s_train, "train")
    _validate_split_shapes(dataset_name, X_val, y_val, s_val, "val")
    _validate_split_shapes(dataset_name, X_test, y_test, s_test, "test")

    if int(y_train.sum()) != 0:
        raise ValueError(
            f"Sai pipeline: y_train của {dataset_name} còn {int(y_train.sum())} anomaly. "
            "Train phải normal-only để học phân bố bình thường."
        )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "s_train": np.asarray(s_train, dtype=int).ravel(),
        "s_val": np.asarray(s_val, dtype=int).ravel(),
        "s_test": np.asarray(s_test, dtype=int).ravel(),
        "sensitive_col": sensitive_name,
        "dataset": dataset_name,
    }


def load_data(dataset_name: str):
    d = load_processed_data(dataset_name)
    return d["X_train"], d["X_test"], d["y_test"], d["s_test"]


def print_dataset_report(dataset_name: str, sensitive_col: Optional[str] = None) -> None:
    d = load_processed_data(dataset_name, sensitive_col=sensitive_col)
    report = {
        "dataset": dataset_name,
        "X_train": d["X_train"].shape,
        "X_val": d["X_val"].shape,
        "X_test": d["X_test"].shape,
        "y_train_sum": int(d["y_train"].sum()),
        "val_anomaly_rate": round(float(d["y_val"].mean()), 6),
        "test_anomaly_rate": round(float(d["y_test"].mean()), 6),
        "sensitive_col": d["sensitive_col"],
        "s_train_groups": dict(pd.Series(d["s_train"]).value_counts().sort_index()),
        "s_val_groups": dict(pd.Series(d["s_val"]).value_counts().sort_index()),
        "s_test_groups": dict(pd.Series(d["s_test"]).value_counts().sort_index()),
    }
    print(report)


def main() -> None:
    ap = argparse.ArgumentParser(description="Kiểm tra data loader cho D1-D8")
    ap.add_argument("--dataset", default="all", choices=["all"] + SUPPORTED_DATASETS)
    ap.add_argument("--sensitive", default=None, help="Ví dụ: sex, race, age")
    args = ap.parse_args()

    datasets = SUPPORTED_DATASETS if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        print_dataset_report(ds, sensitive_col=args.sensitive)


if __name__ == "__main__":
    main()
