# D9 - Pareto Front Analysis: Utility vs Fairness cho post-processing

## 1. Mục tiêu
D9 đánh giá trade-off giữa hiệu quả phát hiện bất thường và công bằng nhóm bằng các quy tắc hậu xử lý ngưỡng. Trục x là EO_gap, trục y là PR-AUC.

## 2. Dữ liệu đầu vào
- Kết quả baseline từ `results/all_results.csv`.
- Điểm bất thường từng mẫu từ `results/score_outputs.csv`.
- Mỗi dataset chọn base model có mean PR-AUC cao nhất từ D6.

## 3. Các quy tắc hậu xử lý
- **Global threshold**: dùng một ngưỡng chung cho toàn bộ nhóm.
- **Per-group FPR threshold**: chọn ngưỡng riêng theo từng nhóm để điều chỉnh FPR.
- **Top-k per group**: chọn top-k% điểm bất thường cao nhất trong từng nhóm.

## 4. Best trade-off theo từng quy tắc
| dataset | base_model | rule | param | pr_auc_mean | f1_mean | eo_gap_mean | eo_gap_reduction_vs_baseline | passes_f1_guardrail |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adult | AutoEncoder | Global threshold | 94.0000 | 0.5060 | 0.3288 | 0.0426 | 0.0367 | True |
| adult | AutoEncoder | Per-group FPR threshold | 0.3000 | 0.4993 | 0.4079 | 0.0633 | 0.0160 | True |
| adult | AutoEncoder | Top-k per group | 0.0918 | 0.4878 | 0.3770 | 0.1186 | -0.0393 | True |
| credit_default | IsolationForest | Global threshold | 80.5000 | 0.3172 | 0.3348 | 0.0467 | 0.0139 | True |
| credit_default | IsolationForest | Per-group FPR threshold | 0.2926 | 0.3160 | 0.3790 | 0.0063 | 0.0543 | True |
| credit_default | IsolationForest | Top-k per group | 0.3000 | 0.3165 | 0.3714 | 0.0138 | 0.0468 | True |

## 5. So sánh baseline và post-processing khuyến nghị
| dataset | base_model | baseline_pr_auc | baseline_f1 | baseline_eo_gap | recommended_rule | recommended_param | post_pr_auc | post_f1 | post_eo_gap | eo_gap_reduction | pr_auc_change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adult | AutoEncoder | 0.5060 | 0.4106 | 0.0793 | Global threshold | 94.0000 | 0.5060 | 0.3288 | 0.0426 | 0.0367 | 0.0000 |
| credit_default | IsolationForest | 0.3172 | 0.3685 | 0.0605 | Top-k per group | 0.3000 | 0.3165 | 0.3714 | 0.0138 | 0.0468 | -0.0007 |

## 6. Nhận xét
- **adult**: mô hình nền là **AutoEncoder**. Quy tắc khuyến nghị là **Global threshold** với tham số 94.0000; EO_gap giảm 0.0367, PR-AUC thay đổi 0.0000.
- **credit_default**: mô hình nền là **IsolationForest**. Quy tắc khuyến nghị là **Top-k per group** với tham số 0.3000; EO_gap giảm 0.0468, PR-AUC thay đổi -0.0007.

## 7. Kết luận D9
D9 đã tạo được bảng và hình Pareto Front để phân tích trade-off giữa utility và fairness. Kết quả này là đầu vào cho D10, nơi so sánh Baseline, Post-processing và In-processing.

## 8. File kết quả
- `results/d9/d9_postprocessing_candidates.csv`
- `results/d9/d9_pareto_mean.csv`
- `results/d9/D9_best_tradeoff_table.csv`
- `results/d9/D9_baseline_vs_recommended_postprocessing.csv`
- `results/d9/pareto_adult.png`
- `results/d9/pareto_credit_default.png`