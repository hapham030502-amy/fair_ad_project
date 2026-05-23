from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

# Các thuộc tính nhạy cảm được tách khỏi X để dùng cho fairness audit.
# Giữ nguyên logic hiện có của luận văn.
SENSITIVE_COLUMNS = {
    "adult": ["sex", "race"],
    "credit_default": ["sex", "age"],
}

# Chỉ khai báo các feature thật sự cần loại khỏi ma trận huấn luyện X.
# Hiện tại chỉ có Credit Default có cột định danh id/ID cần loại.
# Nếu sau này phát hiện duplicate hoặc near-zero variance thì bổ sung vào đúng danh sách,
# không tự động loại nếu chưa có bằng chứng.
FEATURE_REMOVAL_COLUMNS = {
    "adult": {
        "identifier": [],
        "duplicate": [],
        "near_zero_variance": [],
    },
    "credit_default": {
        "identifier": ["id", "ID"],
        "duplicate": [],
        "near_zero_variance": [],
    },
}


def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def read_split(dataset_dir: Path, split: str, prefix: str) -> pd.DataFrame:
    parquet = dataset_dir / f"{prefix}_{split}_raw.parquet"
    csv = dataset_dir / f"{prefix}_{split}_raw.csv"

    if parquet.exists():
        return read_table(parquet)

    if csv.exists():
        return read_table(csv)

    raise FileNotFoundError(f"Missing {prefix}_{split}_raw.parquet/csv in {dataset_dir}")


def find_columns_by_name(df: pd.DataFrame, candidate_cols: list[str]) -> list[str]:
    """
    Tìm cột theo tên, không phân biệt hoa/thường.
    Chỉ trả về các cột thật sự tồn tại trong DataFrame.
    """
    target = {str(c).strip().lower() for c in candidate_cols}
    return [c for c in df.columns if str(c).strip().lower() in target]


def find_sensitive_columns(df: pd.DataFrame, dataset: str) -> list[str]:
    return find_columns_by_name(df, SENSITIVE_COLUMNS.get(dataset, []))


def find_feature_removal_columns(df: pd.DataFrame, dataset: str) -> dict[str, list[str]]:
    """
    Trả về các feature ngoài sensitive cần loại khỏi X theo từng lý do:
    - identifier: cột định danh như id/ID.
    - duplicate: cột trùng lặp nếu đã xác định trước.
    - near_zero_variance: cột gần như hằng nếu đã xác định trước.

    Lưu ý: hàm này không tự ý phát hiện và loại thêm feature mới.
    Muốn loại feature nào thì phải khai báo rõ trong FEATURE_REMOVAL_COLUMNS.
    """
    cfg = FEATURE_REMOVAL_COLUMNS.get(dataset, {})
    out: dict[str, list[str]] = {}

    for reason in ["identifier", "duplicate", "near_zero_variance"]:
        out[reason] = find_columns_by_name(df, cfg.get(reason, []))

    return out


def unique_keep_order(cols: list[str]) -> list[str]:
    seen = set()
    out = []
    for col in cols:
        key = str(col).strip().lower()
        if key not in seen:
            out.append(col)
            seen.add(key)
    return out


def flatten_feature_removal_columns(feature_removal_by_reason: dict[str, list[str]]) -> list[str]:
    cols: list[str] = []
    for reason in ["identifier", "duplicate", "near_zero_variance"]:
        cols.extend(feature_removal_by_reason.get(reason, []))
    return unique_keep_order(cols)


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_onehot_encoder()),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ], remainder="drop")


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return [str(x) for x in preprocessor.get_feature_names_out()]
    except Exception:
        return []


def rebuild_dataset(dataset: str) -> None:
    dataset_dir = PROCESSED_DIR / dataset

    X_train_raw = read_split(dataset_dir, "train", "X")
    X_val_raw = read_split(dataset_dir, "val", "X")
    X_test_raw = read_split(dataset_dir, "test", "X")

    y_train = read_split(dataset_dir, "train", "y").iloc[:, 0].astype(int).to_numpy()
    y_val = read_split(dataset_dir, "val", "y").iloc[:, 0].astype(int).to_numpy()
    y_test = read_split(dataset_dir, "test", "y").iloc[:, 0].astype(int).to_numpy()

    if int(y_train.sum()) != 0:
        raise ValueError(
            f"{dataset}: y_train phải normal-only nhưng còn {int(y_train.sum())} anomaly"
        )

    # 1) Loại sensitive attributes khỏi X để đánh giá fairness đúng nguyên tắc.
    sensitive_drop_cols = find_sensitive_columns(X_train_raw, dataset)

    # 2) Loại các feature không dùng để học mô hình: ID, duplicate, near-zero variance nếu có.
    feature_removal_by_reason = find_feature_removal_columns(X_train_raw, dataset)
    feature_drop_cols = flatten_feature_removal_columns(feature_removal_by_reason)

    # 3) Tổng hợp danh sách drop khỏi X.
    # Giữ đúng thứ tự, tránh lặp cột.
    drop_cols = unique_keep_order(sensitive_drop_cols + feature_drop_cols)

    X_train_model = X_train_raw.drop(columns=drop_cols, errors="ignore")
    X_val_model = X_val_raw.drop(columns=drop_cols, errors="ignore")
    X_test_model = X_test_raw.drop(columns=drop_cols, errors="ignore")

    preprocessor = build_preprocessor(X_train_model)

    # Fit chỉ trên train, transform sang val/test để tránh data leakage.
    X_train = preprocessor.fit_transform(X_train_model).astype(np.float32)
    X_val = preprocessor.transform(X_val_model).astype(np.float32)
    X_test = preprocessor.transform(X_test_model).astype(np.float32)

    old_npz = dataset_dir / "transformed.npz"
    backup_npz = dataset_dir / "transformed_before_drop_sensitive.npz"

    if old_npz.exists() and not backup_npz.exists():
        old_npz.rename(backup_npz)

    np.savez_compressed(
        old_npz,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    joblib.dump(preprocessor, dataset_dir / "preprocessor_without_sensitive.joblib")

    feature_names = get_feature_names(preprocessor)

    stats = {
        "dataset": dataset,

        # Các cột sensitive bị loại khỏi X nhưng vẫn được giữ riêng để audit fairness.
        "drop_sensitive_columns_from_X": sensitive_drop_cols,

        # Các feature thật sự bị loại khỏi X vì không phù hợp làm biến huấn luyện.
        "drop_feature_columns_from_X": feature_drop_cols,
        "drop_feature_columns_by_reason": feature_removal_by_reason,

        # Tổng toàn bộ cột bị loại khỏi X.
        "drop_columns_from_X": drop_cols,

        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),

        # Giữ key cũ để các file khác không bị lỗi.
        "n_features_after_drop_sensitive": int(X_train.shape[1]),
        "feature_names_after_drop_sensitive": feature_names,

        # Key mới ghi rõ đã loại cả sensitive và feature không dùng như ID nếu có.
        "n_features_after_drop_sensitive_and_removed_features": int(X_train.shape[1]),
        "feature_names_after_drop_sensitive_and_removed_features": feature_names,

        "train_is_normal_only": bool(int(y_train.sum()) == 0),
        "anomaly_rate_val": float(y_val.mean()),
        "anomaly_rate_test": float(y_test.mean()),
    }

    with open(dataset_dir / "stats_rebuilt_without_sensitive.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print({
        "dataset": dataset,
        "drop_sensitive_columns_from_X": sensitive_drop_cols,
        "drop_feature_columns_from_X": feature_drop_cols,
        "drop_feature_columns_by_reason": feature_removal_by_reason,
        "drop_columns_from_X": drop_cols,
        "X_train": X_train.shape,
        "X_val": X_val.shape,
        "X_test": X_test.shape,
        "y_train_sum": int(y_train.sum()),
    })


def main():
    for dataset in ["adult", "credit_default"]:
        rebuild_dataset(dataset)


if __name__ == "__main__":
    main()