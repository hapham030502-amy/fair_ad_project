# D10 - Fairness-aware Results

## 1. Mục tiêu
D10 so sánh ba nhóm kết quả: Baseline, Post-processing và In-processing. Các chỉ số chính gồm PR-AUC, F1 và EO_gap.

## 2. Thiết kế thực nghiệm
- **Baseline**: lấy mô hình nền tốt nhất theo PR-AUC từ D6/D9.
- **Post-processing**: lấy quy tắc hậu xử lý khuyến nghị từ D9.
- **In-processing**: dùng reweighted IsolationForest, trong đó mẫu ở nhóm sensitive ít hơn được gán trọng số cao hơn khi huấn luyện.
- Threshold của in-processing được chọn trên validation set theo F1; test set chỉ dùng để đánh giá cuối cùng.

## 3. Bảng so sánh tổng hợp
| dataset | method | method_detail | pr_auc_mean | f1_mean | eo_gap_mean | pr_auc_change_vs_baseline | f1_change_vs_baseline | eo_gap_reduction_vs_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adult | Baseline | AutoEncoder | 0.5060 | 0.4106 | 0.0793 | 0.0000 | 0.0000 | 0.0000 |
| adult | In-processing | Reweighted IsolationForest | 0.3644 | 0.4336 | 0.2028 | -0.1416 | 0.0230 | -0.1236 |
| adult | Post-processing | Global threshold - param=94.0000 | 0.5060 | 0.3288 | 0.0426 | 0.0000 | -0.0818 | 0.0367 |
| credit_default | Baseline | IsolationForest | 0.3172 | 0.3685 | 0.0605 | 0.0000 | 0.0000 | 0.0000 |
| credit_default | In-processing | Reweighted IsolationForest | 0.3136 | 0.3903 | 0.0186 | -0.0036 | 0.0218 | 0.0420 |
| credit_default | Post-processing | Top-k per group - param=0.3000 | 0.3165 | 0.3714 | 0.0138 | -0.0007 | 0.0030 | 0.0468 |

## 4. Method trade-off tốt nhất theo từng dataset
| dataset | method | method_detail | pr_auc_mean | f1_mean | eo_gap_mean | pr_auc_change_vs_baseline | eo_gap_reduction_vs_baseline | selection_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adult | Post-processing | Global threshold - param=94.0000 | 0.5060 | 0.3288 | 0.0426 | 0.0000 | 0.0367 | EO_gap thấp nhất trong nhóm không làm PR-AUC giảm quá 0.01 so với baseline |
| credit_default | Post-processing | Top-k per group - param=0.3000 | 0.3165 | 0.3714 | 0.0138 | -0.0007 | 0.0468 | EO_gap thấp nhất trong nhóm không làm PR-AUC giảm quá 0.01 so với baseline |

## 5. Nhận xét
- **adult**: phương pháp có trade-off tốt nhất là **Post-processing** (Global threshold - param=94.0000). PR-AUC = 0.5060, F1 = 0.3288, EO_gap = 0.0426. So với baseline, EO_gap thay đổi 0.0367.
- **credit_default**: phương pháp có trade-off tốt nhất là **Post-processing** (Top-k per group - param=0.3000). PR-AUC = 0.3165, F1 = 0.3714, EO_gap = 0.0138. So với baseline, EO_gap thay đổi 0.0468.

## 6. Kết luận D10
D10 đã tạo được bảng so sánh Baseline, Post-processing và In-processing. Kết quả cho phép đánh giá trực tiếp trade-off giữa hiệu quả phát hiện bất thường và công bằng nhóm. Phần này là đầu vào cho D11, nơi tiến hành ablation study.

## 7. File kết quả
- `results/d10/D10_inprocessing_raw_results.csv`
- `results/d10/D10_fairness_aware_results.csv`
- `results/d10/D10_best_method_by_dataset.csv`
- `results/d10/d10_eogap_comparison.png`
- `results/d10/D10_Fairness_Aware_Results_Report.md`
- `results/d10/D10_Fairness_Aware_Results_Report.docx` nếu đã cài `python-docx`