#!/usr/bin/env bash
set -e

echo "=================================================="
echo " RUN ALL DELIVERABLES: D1 - D12"
echo "=================================================="

echo "[0/13] Check Python environment"
python --version

echo "[1/13] Generate D1, D2, D5"
python -m src.generate_d1_d2_d5

echo "[1b/13] Rebuild processed data without sensitive attributes and removed features"
python -m src.rebuild_processed_without_sensitive

echo "[2/13] Check D3 data loader and metrics"
bash scripts/check_d1_d8.sh

echo "[3/13] Build D4 dataset cards"
python -m src.make_dataset_cards

echo "[4/13] Tune LOF on Credit Default"
python -m src.tune_lof_credit_default

echo "[5/13] Run all baseline models for D6 with 10 seeds"
python -m src.run_all_models --max-train-samples 0 --ae-epochs 20 --svdd-epochs 20

echo "[6/13] Build D6 baseline tables"
python -m src.make_baseline_results

echo "[7/13] Build D7 score distribution figures"
python -m src.plot_score_distributions

echo "[8/13] Build D8 error audit"
python -m src.make_error_audit

echo "[9/13] Build D9 Pareto Front Analysis"
python -m src.make_d9_pareto

echo "[10/13] Build D10 Fairness-aware Results"
python -m src.make_d10_fairness_results

echo "[11/13] Build D11 Ablation Study"
python -m src.make_d11_ablation

echo "[12/13] Build D12 Reproducibility Package"
python -m src.make_d12_repro_package

echo "[13/13] Run tests if available"
if [ -d "tests" ]; then
    python -m pytest -q
else
    echo "[SKIP] tests/ folder not found"
fi

echo "=================================================="
echo " FINAL CHECK OUTPUT FILES"
echo "=================================================="

check_file() {
    if [ -f "$1" ]; then
        echo "[OK] $1"
    else
        echo "[ERROR] Missing file: $1"
        exit 1
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo "[OK] $1"
    else
        echo "[ERROR] Missing folder: $1"
        exit 1
    fi
}

check_file "results/summary.csv"
check_file "results/per_seed_results.csv"
check_file "results/lof_tuning_credit_default.csv"
check_dir  "results/figures"
check_dir  "results/d8"
check_dir  "results/d9"
check_dir  "results/d10"
check_dir  "results/d11"

check_file "results/d8/D8_Error_Audit_Report.md"
check_file "results/d8/proxy_feature_analysis.csv"

check_dir  "reproducibility"
check_file "reproducibility/README.md"
check_file "reproducibility/MANIFEST_sha256.csv"
check_file "reproducibility/D12_artifact_checklist.csv"
check_file "reproducibility/results/D12_final_statistical_summary.csv"
check_file "reproducibility/D12_Reproducibility_Report.md"

echo "=================================================="
echo "[DONE] D1-D12 outputs are ready."
echo "=================================================="

echo "Main outputs:"
echo "- results/summary.csv"
echo "- results/lof_tuning_credit_default.csv"
echo "- results/d8/"
echo "- results/d9/"
echo "- results/d10/"
echo "- results/d11/"
echo "- reproducibility/"
echo "- reproducibility/results/D12_final_statistical_summary.csv"