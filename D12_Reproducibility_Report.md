# D12 - Reproducibility Package Report

## 1. Mục tiêu
D12 đóng gói code, config, seeds, dữ liệu đã chia, kết quả thực nghiệm, báo cáo và manifest để hỗ trợ tái lập kết quả.

## 2. Kết quả kiểm tra artifact
Số mục tồn tại: **18/18**.

| item | path | exists |
| --- | --- | --- |
| D1 | D1_Problem_Statement_RQs_Hypotheses.docx | True |
| D2 | D2_Decision_Log.md | True |
| D3_config | config/config.yaml | True |
| D3_run_all | scripts/run_all.sh | True |
| D4_dataset_cards | results/cards | True |
| D5_metrics_table | D5_Metrics_Definition_Table.docx | True |
| D6_all_results | results/all_results.csv | True |
| D7_figures | results/figures | True |
| D8_error_audit | results/d8 | True |
| D9_pareto | results/d9 | True |
| D10_fairness_results | results/d10 | True |
| D11_ablation | results/d11 | True |
| D12_reproducibility | reproducibility | True |
| D12_readme | reproducibility/README.md | True |
| D12_manifest | reproducibility/MANIFEST_sha256.csv | True |
| D12_checklist | reproducibility/D12_artifact_checklist.csv | True |
| D12_stat_summary | reproducibility/results/D12_final_statistical_summary.csv | True |
| D12_report_md | reproducibility/D12_Reproducibility_Report.md | True |

## 3. Bảng thống kê cuối cùng
Bảng 4.10 dùng kiểm định Wilcoxon signed-rank theo từng cặp seed. Có 12 kiểm định non-baseline (2 bộ dữ liệu × 2 phương pháp × 3 metric); do đó p-value gốc được hiệu chỉnh bằng Holm-Bonferroni để kiểm soát rủi ro đa so sánh.

| dataset | method | detail | metric | mean | std | change_vs_baseline | p-value gốc | p-value hiệu chỉnh Holm | significant_holm_0_05 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | :---: |
| adult | Baseline | AutoEncoder | pr_auc | 0.5060 | 0.0136 | 0.0000 |  |  |  |
| adult | Baseline | AutoEncoder | f1 | 0.4106 | 0.0162 | 0.0000 |  |  |  |
| adult | Baseline | AutoEncoder | eo_gap | 0.0793 | 0.0255 | 0.0000 |  |  |  |
| adult | Post-processing | Global threshold - param=94.0000 | pr_auc | 0.5060 | 0.0136 | 0.0000 | 1.0000 | 1.0000 | No |
| adult | Post-processing | Global threshold - param=94.0000 | f1 | 0.3288 | 0.0015 | -0.0818 | 0.0020 | 0.0234 | Yes |
| adult | Post-processing | Global threshold - param=94.0000 | eo_gap | 0.0426 | 0.0031 | -0.0367 | 0.0020 | 0.0234 | Yes |
| adult | In-processing | Reweighted IsolationForest | pr_auc | 0.3644 | 0.0158 | -0.1416 | 0.0020 | 0.0234 | Yes |
| adult | In-processing | Reweighted IsolationForest | f1 | 0.4336 | 0.0094 | 0.0230 | 0.0039 | 0.0234 | Yes |
| adult | In-processing | Reweighted IsolationForest | eo_gap | 0.2028 | 0.0434 | 0.1236 | 0.0020 | 0.0234 | Yes |
| credit_default | Baseline | IsolationForest | pr_auc | 0.3172 | 0.0106 | 0.0000 |  |  |  |
| credit_default | Baseline | IsolationForest | f1 | 0.3685 | 0.0087 | 0.0000 |  |  |  |
| credit_default | Baseline | IsolationForest | eo_gap | 0.0605 | 0.0099 | 0.0000 |  |  |  |
| credit_default | Post-processing | Top-k per group - param=0.3000 | pr_auc | 0.3165 | 0.0107 | -0.0007 | 0.0020 | 0.0234 | Yes |
| credit_default | Post-processing | Top-k per group - param=0.3000 | f1 | 0.3714 | 0.0082 | 0.0030 | 0.0273 | 0.0820 | No |
| credit_default | Post-processing | Top-k per group - param=0.3000 | eo_gap | 0.0138 | 0.0065 | -0.0468 | 0.0020 | 0.0234 | Yes |
| credit_default | In-processing | Reweighted IsolationForest | pr_auc | 0.3136 | 0.0060 | -0.0036 | 0.3223 | 0.6445 | No |
| credit_default | In-processing | Reweighted IsolationForest | f1 | 0.3903 | 0.0055 | 0.0218 | 0.0020 | 0.0234 | Yes |
| credit_default | In-processing | Reweighted IsolationForest | eo_gap | 0.0186 | 0.0146 | -0.0420 | 0.0020 | 0.0234 | Yes |

## 4. Manifest
Tổng số file trong gói D12: **142**.
Manifest lưu tại `MANIFEST_sha256.csv`.

## 5. Kết luận
Nếu toàn bộ các dòng quan trọng trong `D12_artifact_checklist.csv` có `exists=True`, gói D12 đạt yêu cầu kỹ thuật cơ bản cho khả năng tái lập. Các kết luận thống kê chính nên ưu tiên đọc theo cột p_value_holm để tránh diễn giải quá mức khi có nhiều phép kiểm định.