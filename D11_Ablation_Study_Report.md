# D11 - Ablation Study

## 1. Mục tiêu
D11 đánh giá độ nhạy của kết quả fairness-aware anomaly detection thông qua ba loại ablation: group imbalance, label noise và threshold rule.

## 2. Thiết kế ablation
- **A1 - Group Imbalance**: thay đổi tỉ lệ nhóm majority/minority theo các mức 50-50, 70-30, 80-20, 90-10, 95-5.
- **A2 - Label Noise**: đã sửa theo góp ý GVHD; nhiễu nhãn chỉ áp dụng trên train set bằng cách đưa anomaly từ validation vào train như mẫu normal; y_true của test được giữ nguyên khi tính PR-AUC/F1/EO_gap; số seed dùng cho A2 là 10.
- **A3 - Threshold Rule**: so sánh ba quy tắc hậu xử lý từ D9: Global threshold, Per-group FPR threshold và Top-k per group.

## 3. Bảng kết quả tổng hợp
| ablation | dataset | rule | setting | pr_auc_mean | f1_mean | eo_gap_mean | delta_fpr_mean | delta_fnr_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1_group_imbalance | adult | Global threshold | majority_ratio=0.50 | 0.4857 | 0.3559 | 0.0508 | 0.0047 | 0.0461 |
| A1_group_imbalance | adult | Global threshold | majority_ratio=0.70 | 0.5095 | 0.3251 | 0.0484 | 0.0050 | 0.0434 |
| A1_group_imbalance | adult | Global threshold | majority_ratio=0.80 | 0.5201 | 0.3112 | 0.0571 | 0.0042 | 0.0529 |
| A1_group_imbalance | adult | Global threshold | majority_ratio=0.90 | 0.5326 | 0.3031 | 0.0806 | 0.0045 | 0.0761 |
| A1_group_imbalance | adult | Global threshold | majority_ratio=0.95 | 0.5387 | 0.2981 | 0.0918 | 0.0091 | 0.0827 |
| A1_group_imbalance | credit_default | Top-k per group | majority_ratio=0.50 | 0.3175 | 0.3728 | 0.0186 | 0.0049 | 0.0137 |
| A1_group_imbalance | credit_default | Top-k per group | majority_ratio=0.70 | 0.3137 | 0.3707 | 0.0178 | 0.0057 | 0.0121 |
| A1_group_imbalance | credit_default | Top-k per group | majority_ratio=0.80 | 0.3092 | 0.3656 | 0.0328 | 0.0067 | 0.0261 |
| A1_group_imbalance | credit_default | Top-k per group | majority_ratio=0.90 | 0.3090 | 0.3664 | 0.0588 | 0.0140 | 0.0449 |
| A1_group_imbalance | credit_default | Top-k per group | majority_ratio=0.95 | 0.3048 | 0.3628 | 0.0646 | 0.0139 | 0.0507 |
| A2_label_noise | adult | Global threshold | train_label_noise=0.00 | 0.5060 | 0.3151 | 0.0533 | 0.0048 | 0.0485 |
| A2_label_noise | adult | Global threshold | train_label_noise=0.05 | 0.5046 | 0.3942 | 0.0573 | 0.0138 | 0.0434 |
| A2_label_noise | adult | Global threshold | train_label_noise=0.10 | 0.5029 | 0.3951 | 0.0543 | 0.0161 | 0.0382 |
| A2_label_noise | adult | Global threshold | train_label_noise=0.15 | 0.5013 | 0.3969 | 0.0539 | 0.0170 | 0.0369 |
| A2_label_noise | credit_default | Top-k per group | train_label_noise=0.00 | 0.3164 | 0.3714 | 0.0138 | 0.0044 | 0.0094 |
| A2_label_noise | credit_default | Top-k per group | train_label_noise=0.05 | 0.3058 | 0.3632 | 0.0140 | 0.0048 | 0.0092 |
| A2_label_noise | credit_default | Top-k per group | train_label_noise=0.10 | 0.2986 | 0.3568 | 0.0143 | 0.0056 | 0.0087 |
| A2_label_noise | credit_default | Top-k per group | train_label_noise=0.15 | 0.2886 | 0.3454 | 0.0134 | 0.0052 | 0.0082 |
| A3_threshold_rule | adult | Global threshold | Global threshold | 0.5060 | 0.3288 | 0.0426 | 0.0051 | 0.0375 |
| A3_threshold_rule | adult | Per-group FPR threshold | Per-group FPR threshold | 0.4993 | 0.4079 | 0.0633 | 0.0000 | 0.0633 |
| A3_threshold_rule | adult | Top-k per group | Top-k per group | 0.4878 | 0.3770 | 0.1186 | 0.0354 | 0.0832 |
| A3_threshold_rule | credit_default | Global threshold | Global threshold | 0.3172 | 0.3348 | 0.0467 | 0.0294 | 0.0173 |
| A3_threshold_rule | credit_default | Per-group FPR threshold | Per-group FPR threshold | 0.3160 | 0.3790 | 0.0063 | 0.0002 | 0.0061 |
| A3_threshold_rule | credit_default | Top-k per group | Top-k per group | 0.3164 | 0.3714 | 0.0138 | 0.0044 | 0.0094 |

## 4. Kết luận theo từng ablation
| dataset | ablation | finding | interpretation |
| --- | --- | --- | --- |
| adult | A1_group_imbalance | Khi majority_ratio tăng từ 0.50 lên 0.95, EO_gap thay đổi từ 0.0508 lên 0.0918. | Nếu EO_gap tăng khi mất cân bằng nhóm lớn hơn, điều này cho thấy fairness nhạy với phân bố nhóm trong dữ liệu đánh giá. |
| adult | A2_label_noise | Khi train_label_noise tăng từ 0.00 lên 0.15, PR-AUC giảm từ 0.5060 đến 0.5013; EO_gap thay đổi từ 0.0533 đến 0.0539. | A2 đã được sửa để label noise chỉ áp dụng trên train set; y_true của test được giữ nguyên khi tính metric. Nếu PR-AUC giảm khi train noise tăng, kết quả phù hợp kỳ vọng vì train bị contaminate làm giảm chất lượng học phân bố normal. |
| adult | A3_threshold_rule | Rule có EO_gap thấp nhất là Global threshold (EO_gap=0.0426); rule có PR-AUC cao nhất là Global threshold (PR-AUC=0.5060). | Kết quả này cho thấy lựa chọn threshold rule ảnh hưởng trực tiếp đến trade-off utility–fairness. |
| credit_default | A1_group_imbalance | Khi majority_ratio tăng từ 0.50 lên 0.95, EO_gap thay đổi từ 0.0186 lên 0.0646. | Nếu EO_gap tăng khi mất cân bằng nhóm lớn hơn, điều này cho thấy fairness nhạy với phân bố nhóm trong dữ liệu đánh giá. |
| credit_default | A2_label_noise | Khi train_label_noise tăng từ 0.00 lên 0.15, PR-AUC giảm từ 0.3164 đến 0.2886; EO_gap thay đổi từ 0.0138 đến 0.0134. | A2 đã được sửa để label noise chỉ áp dụng trên train set; y_true của test được giữ nguyên khi tính metric. Nếu PR-AUC giảm khi train noise tăng, kết quả phù hợp kỳ vọng vì train bị contaminate làm giảm chất lượng học phân bố normal. |
| credit_default | A3_threshold_rule | Rule có EO_gap thấp nhất là Per-group FPR threshold (EO_gap=0.0063); rule có PR-AUC cao nhất là Global threshold (PR-AUC=0.3172). | Kết quả này cho thấy lựa chọn threshold rule ảnh hưởng trực tiếp đến trade-off utility–fairness. |

## 5. Nhận xét
Ablation A1 cho biết mức độ nhạy của EO_gap khi phân bố nhóm thay đổi. Ablation A2 cho biết độ ổn định của PR-AUC và EO_gap khi train set bị nhiễu nhãn; y_true của test luôn giữ sạch. Ablation A3 cho thấy threshold rule là yếu tố quan trọng ảnh hưởng trực tiếp đến trade-off giữa utility và fairness.

## 6. Threats to validity and limitations
- **Giới hạn về dữ liệu**: Thực nghiệm mới được kiểm tra trên hai bộ dữ liệu dạng bảng là Adult và Credit Default, vì vậy khả năng khái quát sang dữ liệu ảnh, chuỗi thời gian hoặc đồ thị cần được kiểm chứng thêm.
- **Giới hạn về thuộc tính nhạy cảm**: Các phân tích fairness chủ yếu sử dụng một thuộc tính nhạy cảm mặc định cho mỗi bộ dữ liệu. Các thuộc tính khác như race hoặc age có thể tạo ra kết quả khác và cần được phân tích bổ sung trong các nghiên cứu tiếp theo.
- **Giới hạn về mô hình**: Ablation sử dụng điểm bất thường đã sinh từ các mô hình baseline và các quy tắc hậu xử lý. Do đó, kết quả phản ánh độ nhạy của pipeline hiện tại, chưa đại diện cho toàn bộ các thuật toán phát hiện bất thường khác.
- **Giới hạn về nhiễu nhãn**: Thí nghiệm label noise được mô phỏng bằng cách lật nhãn ngẫu nhiên. Trong dữ liệu thực tế, nhiễu nhãn có thể có cấu trúc phức tạp hơn và phụ thuộc vào nhóm người dùng.
- **Giới hạn về threshold rule**: Các quy tắc threshold được khảo sát gồm Global threshold, Per-group FPR threshold và Top-k per group. Những chiến lược hậu xử lý phức tạp hơn có thể đem lại trade-off khác giữa utility và fairness.

## 7. Kết luận D11
D11 đã tạo được bảng ablation cho ba yếu tố chính: group imbalance, label noise và threshold rule. Kết quả này dùng để viết phần robustness/ablation trong Chương 4 và làm cơ sở thảo luận ở Chương 5.

## 8. File kết quả
- `results/d11/D11_ablation_raw_results.csv`
- `results/d11/D11_ablation_summary.csv`
- `results/d11/D11_ablation_conclusions.csv`
- `results/d11/d11_a1_group_imbalance_eogap.png`
- `results/d11/d11_a2_label_noise_prauc.png`
- `results/d11/d11_a3_threshold_rule_eogap.png`
- `results/d11/D11_Ablation_Study_Report.md`
- `results/d11/D11_Ablation_Study_Report.docx` nếu có `python-docx`