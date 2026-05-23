from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import SUPPORTED_DATASETS, load_processed_data

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "results" / "cards"


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def read_raw(dataset_dir: Path, prefix: str, split: str) -> pd.DataFrame:
    parquet = dataset_dir / f"{prefix}_{split}_raw.parquet"
    csv = dataset_dir / f"{prefix}_{split}_raw.csv"

    if parquet.exists():
        return read_table(parquet)

    if csv.exists():
        return read_table(csv)

    raise FileNotFoundError(
        f"Không tìm thấy {prefix}_{split}_raw.parquet hoặc .csv trong {dataset_dir}"
    )


def read_stats(dataset_dir: Path) -> dict:
    stats_path = dataset_dir / "stats_rebuilt_without_sensitive.json"

    if not stats_path.exists():
        return {}

    with open(stats_path, "r", encoding="utf-8") as f:
        return json.load(f)


def missing_rate(df: pd.DataFrame) -> float:
    return float(df.isna().mean().mean()) if not df.empty else 0.0


def group_distribution(s: np.ndarray) -> dict:
    vc = pd.Series(s).value_counts(normalize=False).sort_index()
    total = int(vc.sum())

    return {
        str(k): {
            "count": int(v),
            "percent": float(v / total) if total else 0.0,
        }
        for k, v in vc.items()
    }


def anomaly_rate_per_group(y: np.ndarray, s: np.ndarray) -> dict:
    out = {}

    for g in sorted(np.unique(s)):
        mask = s == g
        out[str(int(g))] = {
            "n": int(mask.sum()),
            "anomaly_rate": float(np.mean(y[mask])) if mask.sum() > 0 else 0.0,
        }

    return out


def normalize_col_name(col: str) -> str:
    return str(col).strip().lower()


def build_removed_reason_lookup(stats: dict) -> dict[str, str]:
    """
    Tạo bảng tra cứu:
        tên_cột -> lý do bị loại khỏi ma trận đặc trưng X.

    Chỉ phục vụ ghi chú trong Dataset Card, không thay đổi dữ liệu.
    """

    removed_by_reason = stats.get("drop_feature_columns_by_reason", {}) or {}

    reason_label = {
        "identifier": "ID/identifier; excluded from model X",
        "duplicate": "duplicate feature; excluded from model X",
        "near_zero_variance": "near-zero variance feature; excluded from model X",
    }

    lookup: dict[str, str] = {}

    for reason, cols in removed_by_reason.items():
        label = reason_label.get(reason, f"{reason}; excluded from model X")

        for col in cols:
            lookup[normalize_col_name(col)] = label

    return lookup


def make_schema_table(df: pd.DataFrame, stats: dict | None = None, max_cols: int = 120) -> str:
    """
    Sinh bảng schema raw features.

    Điểm sửa chính:
    - Nếu cột là sensitive attribute đã loại khỏi X, ghi rõ:
      sensitive attribute; excluded from model X, kept for fairness audit.
    - Nếu cột là ID/duplicate/near-zero variance đã loại khỏi X, ghi rõ lý do.
    - Các cột còn lại ghi là raw input feature used in model X.

    Hàm này chỉ thay đổi nội dung báo cáo Dataset Card, không thay đổi pipeline mô hình.
    """

    stats = stats or {}

    sensitive_cols = {
        normalize_col_name(c)
        for c in stats.get("drop_sensitive_columns_from_X", [])
    }

    removed_reason_lookup = build_removed_reason_lookup(stats)

    rows = []

    for col in df.columns[:max_cols]:
        key = normalize_col_name(col)

        if key in removed_reason_lookup:
            note = removed_reason_lookup[key]
        elif key in sensitive_cols:
            note = "sensitive attribute; excluded from model X, kept for fairness audit"
        else:
            note = "raw input feature used in model X"

        rows.append(f"| {col} | {str(df[col].dtype)} | {note} |")

    if len(df.columns) > max_cols:
        rows.append(f"| ... | ... | Còn {len(df.columns) - max_cols} cột khác |")

    return "\n".join(rows)


def append_removed_feature_notes(lines: list[str], stats: dict) -> None:
    """
    Ghi chú các feature đã loại khỏi X.

    Lưu ý:
    - Chỉ ghi các feature thật sự có trong stats.
    - Không tự bịa thêm duplicate hoặc near-zero variance nếu stats không ghi nhận.
    """

    sensitive_removed = stats.get("drop_sensitive_columns_from_X", [])
    if sensitive_removed:
        lines.append(f"- Các cột sensitive bị loại khỏi X: `{sensitive_removed}`.")

    removed_by_reason = stats.get("drop_feature_columns_by_reason", {}) or {}
    has_removed_feature = any(bool(cols) for cols in removed_by_reason.values())

    if has_removed_feature:
        lines.append("- Các feature khác bị loại khỏi X, nếu có:")

        reason_label_vi = {
            "identifier": "ID/định danh",
            "duplicate": "feature trùng lặp",
            "near_zero_variance": "feature gần như hằng/near-zero variance",
        }

        for reason in ["identifier", "duplicate", "near_zero_variance"]:
            cols = removed_by_reason.get(reason, [])
            if cols:
                label = reason_label_vi.get(reason, reason)
                lines.append(f"  - {label}: `{cols}`.")

    elif stats:
        lines.append("- Không ghi nhận feature khác bị loại khỏi X ngoài sensitive attributes.")


def make_card(dataset: str) -> str:
    dataset_dir = PROCESSED_DIR / dataset

    d = load_processed_data(dataset)

    X_train_raw = read_raw(dataset_dir, "X", "train")
    X_val_raw = read_raw(dataset_dir, "X", "val")
    X_test_raw = read_raw(dataset_dir, "X", "test")

    y_train, y_val, y_test = d["y_train"], d["y_val"], d["y_test"]
    s_test = d["s_test"]

    stats = read_stats(dataset_dir)

    lines = []

    lines.append(f"# D4 Dataset Card: {dataset}")
    lines.append("")

    lines.append("## 1. Mục đích sử dụng")
    lines.append("")
    lines.append(
        f"Bộ dữ liệu `{dataset}` được sử dụng cho bài toán phát hiện bất thường trên dữ liệu bảng "
        "và đánh giá công bằng mô hình theo nhóm nhạy cảm."
    )
    lines.append("")

    lines.append("## 2. Quy mô dữ liệu")
    lines.append("")
    lines.append("| Split | Số mẫu | Tỷ lệ anomaly |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Train | {len(y_train)} | {float(np.mean(y_train)):.4f} |")
    lines.append(f"| Validation | {len(y_val)} | {float(np.mean(y_val)):.4f} |")
    lines.append(f"| Test | {len(y_test)} | {float(np.mean(y_test)):.4f} |")
    lines.append("")
    lines.append(
        f"Ghi chú: `y_train_sum = {int(y_train.sum())}`; "
        "train set là normal-only để học normality."
    )
    lines.append("")

    lines.append("## 3. Số đặc trưng")
    lines.append("")
    lines.append("| Nội dung | Giá trị |")
    lines.append("|---|---:|")
    lines.append(f"| Số cột raw train | {X_train_raw.shape[1]} |")
    lines.append(f"| Số feature sau biến đổi | {d['X_train'].shape[1]} |")

    if stats:
        removed_cols = stats.get("drop_columns_from_X", [])
        feature_removed_cols = stats.get("drop_feature_columns_from_X", [])

        lines.append(f"| Số cột bị loại khỏi X | {len(removed_cols)} |")
        lines.append(f"| Số feature không dùng để huấn luyện bị loại khỏi X | {len(feature_removed_cols)} |")

    lines.append("")

    lines.append("## 4. Sensitive attribute")
    lines.append("")
    lines.append(f"Sensitive attribute mặc định dùng cho fairness audit: `{d['sensitive_col']}`.")
    lines.append("")

    lines.append("### Phân bố nhóm trên test")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(group_distribution(s_test), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    lines.append("### Anomaly rate theo nhóm trên test")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(anomaly_rate_per_group(y_test, s_test), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    lines.append("## 5. Missing rate")
    lines.append("")
    lines.append("| Split | Missing rate trung bình |")
    lines.append("|---|---:|")
    lines.append(f"| Train raw | {missing_rate(X_train_raw):.4f} |")
    lines.append(f"| Validation raw | {missing_rate(X_val_raw):.4f} |")
    lines.append(f"| Test raw | {missing_rate(X_test_raw):.4f} |")
    lines.append("")

    lines.append("## 6. Schema raw features")
    lines.append("")
    lines.append("| Tên cột | Kiểu dữ liệu | Ghi chú |")
    lines.append("|---|---|---|")
    lines.append(make_schema_table(X_train_raw, stats))
    lines.append("")

    lines.append("## 7. Tiền xử lý")
    lines.append("")
    lines.append("- Missing numerical values xử lý bằng median imputation.")
    lines.append("- Missing categorical values xử lý bằng most-frequent imputation.")
    lines.append("- Numerical features chuẩn hóa bằng StandardScaler.")
    lines.append("- Categorical features mã hóa bằng OneHotEncoder.")
    lines.append(
        "- Sensitive attributes dùng để audit fairness, không dùng trực tiếp trong X "
        "nếu đã rebuild without sensitive."
    )

    append_removed_feature_notes(lines, stats)

    lines.append("")

    lines.append("## 8. Rủi ro và giới hạn")
    lines.append("")
    lines.append("- Dữ liệu mất cân bằng nên Accuracy không dùng làm metric chính.")
    lines.append("- PR-AUC là metric utility chính.")
    lines.append("- EO-gap, ΔFPR, ΔFNR dùng để đánh giá fairness theo nhóm.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in SUPPORTED_DATASETS:
        md = make_card(dataset)
        out_path = OUT_DIR / f"D4_Dataset_Card_{dataset}.md"
        out_path.write_text(md, encoding="utf-8")
        print(f"Đã lưu {out_path}")


if __name__ == "__main__":
    main()