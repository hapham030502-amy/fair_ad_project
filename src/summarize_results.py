from __future__ import annotations
from pathlib import Path
import json
import argparse
import pandas as pd

def read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))

def flatten_result(d: dict, dataset: str, model: str, file_path: str) -> dict:
    row = {"dataset": dataset, "model": model, "seed": d.get("seed"), "file": file_path}

    # utility: chỉ lấy số (bỏ chuỗi như theta_selected_on='validation')
    util = d.get("utility", {})
    for k, v in util.items():
        if isinstance(v, (int, float)):
            row[f"util__{k}"] = v

    # fairness: delta_fpr/delta_fnr/eo_gap là số
    fair = d.get("fairness", {})
    for attr, val in fair.items():
        if isinstance(val, dict):
            for k in ["delta_fpr", "delta_fnr", "eo_gap"]:
                if k in val and isinstance(val[k], (int, float)):
                    row[f"fair__{attr}__{k}"] = val[k]
    return row

def main(results_dir: str, out_summary: str, out_per_seed: str):
    results_dir = Path(results_dir)
    files = list(results_dir.rglob("seed_*.json"))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy seed_*.json trong {results_dir.resolve()}")

    rows = []
    for f in files:
        d = read_json(f)
        rel = f.relative_to(results_dir)
        dataset = rel.parts[0] if len(rel.parts) >= 1 else d.get("dataset", "unknown")
        model = rel.parts[1] if len(rel.parts) >= 2 else "unknown"
        rows.append(flatten_result(d, dataset, model, str(f)))

    df = pd.DataFrame(rows)

    Path(out_per_seed).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_per_seed, index=False, encoding="utf-8-sig")

    metric_cols = [c for c in df.columns if c.startswith("util__") or c.startswith("fair__")]
    for c in metric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    numeric_cols = [c for c in metric_cols if df[c].notna().any()]

    agg = df.groupby(["dataset", "model"])[numeric_cols].agg(["mean", "std"]).reset_index()

    # flatten MultiIndex columns
    flat_cols = []
    for col in agg.columns:
        if isinstance(col, tuple):
            if col[1] == "":
                flat_cols.append(col[0])
            else:
                flat_cols.append(f"{col[0]}__{col[1]}")
        else:
            flat_cols.append(col)
    agg.columns = flat_cols

    Path(out_summary).parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_summary, index=False, encoding="utf-8-sig")

    print("[DONE] per-seed ->", out_per_seed)
    print("[DONE] summary  ->", out_summary)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_summary", default="results/summary.csv")
    ap.add_argument("--out_per_seed", default="results/per_seed.csv")
    args = ap.parse_args()
    main(args.results_dir, args.out_summary, args.out_per_seed)
