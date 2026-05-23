from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from src.data_loader import SUPPORTED_DATASETS, load_processed_data

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
D8_DIR = RESULTS_DIR / "d8"
SCORE_FILE = RESULTS_DIR / "score_outputs.csv"
BASELINE_NUMERIC_FILE = RESULTS_DIR / "summary.csv"


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": safe_div(fp, fp + tn),
        "fnr": safe_div(fn, fn + tp),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def make_subgroup_performance(score_df: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset", "model", "seed", "score", "threshold", "sensitive", "y_true"}
    missing = required - set(score_df.columns)
    if missing:
        raise ValueError(f"score_outputs.csv thiếu cột: {missing}")

    score_df = score_df.copy()
    if "y_pred" not in score_df.columns:
        score_df["y_pred"] = (score_df["score"] >= score_df["threshold"]).astype(int)

    rows: List[Dict] = []
    for keys, g in score_df.groupby(["dataset", "model", "seed", "sensitive"]):
        dataset, model, seed, sensitive = keys
        y_true = g["y_true"].astype(int).to_numpy()
        y_pred = g["y_pred"].astype(int).to_numpy()
        scores = g["score"].astype(float).to_numpy()
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "seed": int(seed),
                "sensitive_group": int(sensitive),
                "n_samples": int(len(g)),
                "anomaly_rate": float(np.mean(y_true)),
                "pred_positive_rate": float(np.mean(y_pred)),
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "median_score": float(np.median(scores)),
                **compute_binary_metrics(y_true, y_pred),
            }
        )
    return pd.DataFrame(rows)


def summarize_subgroup_performance(subgroup_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "n_samples",
        "anomaly_rate",
        "pred_positive_rate",
        "mean_score",
        "std_score",
        "median_score",
        "fpr",
        "fnr",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "tn",
        "fn",
    ]
    summary = subgroup_df.groupby(["dataset", "model", "sensitive_group"])[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = [f"{a}_{b}" if b else a for a, b in summary.columns.to_flat_index()]
    return summary


def make_bias_source_summary(subgroup_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in subgroup_df.groupby(["dataset", "model", "seed"]):
        dataset, model, seed = keys
        groups = sorted(g["sensitive_group"].unique().tolist())
        if len(groups) < 2:
            continue
        g0 = g[g["sensitive_group"] == groups[0]].iloc[0]
        g1 = g[g["sensitive_group"] == groups[-1]].iloc[0]

        anomaly_rate_gap = abs(g0["anomaly_rate"] - g1["anomaly_rate"])
        pred_positive_rate_gap = abs(g0["pred_positive_rate"] - g1["pred_positive_rate"])
        mean_score_gap = abs(g0["mean_score"] - g1["mean_score"])
        fpr_gap = abs(g0["fpr"] - g1["fpr"])
        fnr_gap = abs(g0["fnr"] - g1["fnr"])

        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "seed": int(seed),
                "group_low": int(groups[0]),
                "group_high": int(groups[-1]),
                "anomaly_rate_gap": float(anomaly_rate_gap),
                "pred_positive_rate_gap": float(pred_positive_rate_gap),
                "mean_score_gap": float(mean_score_gap),
                "fpr_gap": float(fpr_gap),
                "fnr_gap": float(fnr_gap),
                "eo_gap": float(fpr_gap + fnr_gap),
            }
        )

    gap_df = pd.DataFrame(rows)
    if gap_df.empty:
        return gap_df

    summary = gap_df.groupby(["dataset", "model"]).agg(
        anomaly_rate_gap_mean=("anomaly_rate_gap", "mean"),
        anomaly_rate_gap_std=("anomaly_rate_gap", "std"),
        pred_positive_rate_gap_mean=("pred_positive_rate_gap", "mean"),
        pred_positive_rate_gap_std=("pred_positive_rate_gap", "std"),
        mean_score_gap_mean=("mean_score_gap", "mean"),
        mean_score_gap_std=("mean_score_gap", "std"),
        fpr_gap_mean=("fpr_gap", "mean"),
        fpr_gap_std=("fpr_gap", "std"),
        fnr_gap_mean=("fnr_gap", "mean"),
        fnr_gap_std=("fnr_gap", "std"),
        eo_gap_mean=("eo_gap", "mean"),
        eo_gap_std=("eo_gap", "std"),
    ).reset_index()
    return summary


def _feature_names_after_drop_sensitive(dataset: str, n_features: int) -> List[str]:
    """
    Trả về tên feature sau tiền xử lý để tránh các mã chỉ số nội bộ khó diễn giải.
    Ưu tiên đọc stats_rebuilt_without_sensitive.json hoặc preprocessor.get_feature_names_out().
    Nếu không đọc được thì fallback về unmapped_col_000, unmapped_col_001, ...
    """
    stats_path = ROOT / "data" / "processed" / dataset / "stats_rebuilt_without_sensitive.json"
    if stats_path.exists():
        try:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            names = stats.get("feature_names_after_drop_sensitive", [])
            if len(names) == n_features:
                return [str(x) for x in names]
        except Exception:
            pass

    for fname in ["preprocessor_without_sensitive.joblib", "preprocessor.joblib"]:
        p = ROOT / "data" / "processed" / dataset / fname
        if p.exists():
            try:
                import joblib
                pre = joblib.load(p)
                names = list(pre.get_feature_names_out())
                if len(names) == n_features:
                    return [str(x) for x in names]
            except Exception:
                pass

    return [f"unmapped_col_{j:03d}" for j in range(n_features)]


def _humanize_feature_name(name: str) -> Dict[str, str]:
    """Tách tên feature one-hot thành cột gốc và giá trị để dễ diễn giải trong luận văn."""
    original = name
    category_value = ""
    clean = name
    if "__" in clean:
        clean = clean.split("__", 1)[1]

    # Adult: workclass_Private, marital_status_Married-civ-spouse, ...
    known_prefixes = [
        "marital_status", "native_country", "education", "occupation", "relationship", "workclass",
        "capital_gain", "capital_loss", "hours_per_week", "education_num", "fnlwgt", "age",
    ]
    original_column = clean
    for pref in known_prefixes:
        if clean == pref:
            original_column = pref
            category_value = ""
            break
        if clean.startswith(pref + "_"):
            original_column = pref
            category_value = clean[len(pref) + 1:]
            break

    # Credit Default: map x1, x3...x23 to common UCI meanings.
    credit_map = {
        "id": "ID",
        "x1": "LIMIT_BAL", "x2": "SEX", "x3": "EDUCATION", "x4": "MARRIAGE", "x5": "AGE",
        "x6": "PAY_0", "x7": "PAY_2", "x8": "PAY_3", "x9": "PAY_4", "x10": "PAY_5", "x11": "PAY_6",
        "x12": "BILL_AMT1", "x13": "BILL_AMT2", "x14": "BILL_AMT3", "x15": "BILL_AMT4", "x16": "BILL_AMT5", "x17": "BILL_AMT6",
        "x18": "PAY_AMT1", "x19": "PAY_AMT2", "x20": "PAY_AMT3", "x21": "PAY_AMT4", "x22": "PAY_AMT5", "x23": "PAY_AMT6",
    }
    if clean in credit_map:
        original_column = credit_map[clean]
        category_value = ""

    return {
        "feature_mapped": original,
        "original_column": original_column,
        "category_value": category_value,
    }


def make_proxy_feature_analysis(top_k: int = 30) -> pd.DataFrame:
    rows = []
    for dataset in SUPPORTED_DATASETS:
        d = load_processed_data(dataset)
        X_test = d["X_test"]
        s_test = d["s_test"].astype(int)
        feature_names = _feature_names_after_drop_sensitive(dataset, X_test.shape[1])

        correlations = []
        for j in range(X_test.shape[1]):
            x = X_test[:, j]
            if np.std(x) < 1e-12:
                corr = 0.0
            else:
                corr = float(np.corrcoef(x, s_test)[0, 1])
            if np.isnan(corr):
                corr = 0.0
            correlations.append(corr)

        try:
            mi = mutual_info_classif(X_test, s_test, discrete_features=False, random_state=42)
        except Exception:
            mi = np.zeros(X_test.shape[1])

        for j, (corr, mi_val) in enumerate(zip(correlations, mi)):
            mapped = _humanize_feature_name(feature_names[j])
            rows.append(
                {
                    "dataset": dataset,
                    "feature_mapped": mapped["feature_mapped"],
                    "original_column": mapped["original_column"],
                    "category_value": mapped["category_value"],
                    "correlation_with_sensitive": float(corr),
                    "abs_correlation_with_sensitive": float(abs(corr)),
                    "mutual_information_with_sensitive": float(mi_val),
                }
            )

    df = pd.DataFrame(rows)
    return df.sort_values(["dataset", "abs_correlation_with_sensitive", "mutual_information_with_sensitive"], ascending=[True, False, False]).groupby("dataset").head(top_k)


def _markdown_table(df: pd.DataFrame, max_rows: int = 12) -> str:
    if df is None or df.empty:
        return "_Không có dữ liệu._"
    return df.head(max_rows).to_markdown(index=False)


def identify_main_findings(bias_df: pd.DataFrame, baseline_df: pd.DataFrame) -> List[str]:
    findings = []
    if not baseline_df.empty:
        for dataset, sub in baseline_df.groupby("dataset"):
            best_pr = sub.sort_values("pr_auc_mean", ascending=False).iloc[0]
            best_fair = sub.sort_values("eo_gap_mean", ascending=True).iloc[0]
            worst_fair = sub.sort_values("eo_gap_mean", ascending=False).iloc[0]
            findings.append(
                f"- `{dataset}`: PR-AUC cao nhất thuộc `{best_pr['model']}` "
                f"({best_pr['pr_auc_mean']:.4f}); EO-gap thấp nhất thuộc `{best_fair['model']}` "
                f"({best_fair['eo_gap_mean']:.4f}); EO-gap cao nhất thuộc `{worst_fair['model']}` "
                f"({worst_fair['eo_gap_mean']:.4f})."
            )
    if not bias_df.empty:
        for dataset, sub in bias_df.groupby("dataset"):
            top = sub.sort_values("eo_gap_mean", ascending=False).iloc[0]
            source = "threshold/error-rate bias"
            if top["anomaly_rate_gap_mean"] > max(top["mean_score_gap_mean"], top["fpr_gap_mean"], top["fnr_gap_mean"]):
                source = "data bias"
            elif top["mean_score_gap_mean"] > max(top["anomaly_rate_gap_mean"], top["fpr_gap_mean"], top["fnr_gap_mean"]):
                source = "model score bias/proxy bias"
            findings.append(f"- `{dataset}`: mô hình cần chú ý nhất là `{top['model']}`; nguồn bias nổi bật theo audit là `{source}`.")
    return findings


def write_report(
    subgroup_summary: pd.DataFrame,
    bias_summary: pd.DataFrame,
    proxy_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> None:
    findings = identify_main_findings(bias_summary, baseline_df)

    lines = []
    lines.append("# D8 Error Audit Report: Bias đến từ đâu?")
    lines.append("")
    lines.append("## 1. Mục tiêu")
    lines.append("")
    lines.append("Báo cáo này phân tích lỗi theo nhóm nhạy cảm để xác định bias có thể đến từ dữ liệu, mô hình hoặc threshold.")
    lines.append("")
    lines.append("## 2. Tóm tắt phát hiện chính")
    lines.append("")
    lines.extend(findings or ["- Chưa đủ dữ liệu để rút ra phát hiện chính."])
    lines.append("")
    lines.append("## 3. Subgroup Performance Summary")
    lines.append("")
    lines.append(_markdown_table(subgroup_summary, max_rows=16))
    lines.append("")
    lines.append("## 4. Bias Source Summary")
    lines.append("")
    lines.append(_markdown_table(bias_summary, max_rows=16))
    lines.append("")
    lines.append("## 5. Proxy Feature Analysis")
    lines.append("")
    lines.append("Các feature có tương quan hoặc mutual information cao với sensitive attribute có thể đóng vai trò proxy.")
    lines.append("Bảng dưới đây chỉ trình bày tên thuộc tính gốc và giá trị sau mã hóa, không còn dùng các mã chỉ số nội bộ khó diễn giải. Tên feature được lấy từ `OneHotEncoder.get_feature_names_out()` hoặc `stats_rebuilt_without_sensitive.json`, giúp phần proxy feature analysis có ý nghĩa khi đưa vào luận văn.")
    lines.append("")
    proxy_cols = [
        "dataset", "original_column", "category_value", "feature_mapped",
        "abs_correlation_with_sensitive", "mutual_information_with_sensitive"
    ]
    proxy_display = proxy_df[[c for c in proxy_cols if c in proxy_df.columns]].copy()
    if "category_value" in proxy_display.columns:
        proxy_display["category_value"] = proxy_display["category_value"].fillna("")
    lines.append(_markdown_table(proxy_display, max_rows=20))
    lines.append("")
    lines.append("## 6. Diễn giải nguồn bias")
    lines.append("")
    lines.append("- **Data bias**: thể hiện qua `anomaly_rate_gap_mean` lớn giữa các nhóm.")
    lines.append("- **Model score bias/proxy bias**: thể hiện qua `mean_score_gap_mean` lớn hoặc proxy feature mạnh.")
    lines.append("- **Threshold bias/error-rate bias**: thể hiện qua `fpr_gap_mean`, `fnr_gap_mean`, `eo_gap_mean` lớn khi dùng cùng threshold toàn cục.")
    lines.append("")
    lines.append("## 7. Hàm ý cho D9-D11")
    lines.append("")
    lines.append("- Cần thử post-processing bằng threshold riêng theo nhóm nếu EO-gap cao do threshold/error-rate bias.")
    lines.append("- Cần thử reweighting hoặc in-processing nếu score gap hoặc proxy bias rõ.")
    lines.append("- Cần ablation theo imbalance, label noise và threshold để kiểm tra độ bền của kết luận.")
    lines.append("")

    out_path = D8_DIR / "D8_Error_Audit_Report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Đã lưu: {out_path}")


def main() -> None:
    if not SCORE_FILE.exists():
        raise FileNotFoundError("Không thấy results/score_outputs.csv. Hãy chạy: python -m src.run_all_models")
    if not BASELINE_NUMERIC_FILE.exists():
       raise FileNotFoundError("Không thấy results/summary.csv. Hãy chạy lại: python -m src.run_all_models")

    D8_DIR.mkdir(parents=True, exist_ok=True)
    score_df = pd.read_csv(SCORE_FILE)
    baseline_df = pd.read_csv(BASELINE_NUMERIC_FILE)

    subgroup_df = make_subgroup_performance(score_df)
    subgroup_summary = summarize_subgroup_performance(subgroup_df)
    bias_summary = make_bias_source_summary(subgroup_df)
    proxy_df = make_proxy_feature_analysis()

    subgroup_df.to_csv(D8_DIR / "subgroup_performance.csv", index=False)
    subgroup_summary.to_csv(D8_DIR / "subgroup_performance_summary.csv", index=False)
    bias_summary.to_csv(D8_DIR / "bias_source_summary.csv", index=False)
    proxy_df.to_csv(D8_DIR / "proxy_feature_analysis.csv", index=False)

    write_report(subgroup_summary, bias_summary, proxy_df, baseline_df)
    print("Đã hoàn thành D8 Error Audit.")


if __name__ == "__main__":
    main()
