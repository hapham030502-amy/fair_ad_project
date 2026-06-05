# D5. Metrics Definition Table

## 1. Utility metrics

| Metric | Công thức/Ý nghĩa | Ghi chú |
|---|---|---|
| ROC-AUC | Diện tích dưới đường ROC | Đánh giá khả năng xếp hạng tổng quát, threshold-independent |
| PR-AUC | Diện tích dưới đường Precision-Recall | Metric chính khi dữ liệu mất cân bằng |
| F1@θ | 2 × Precision × Recall / (Precision + Recall) tại threshold θ | Threshold-dependent |

## 2. Fairness metrics

| Metric | Công thức | Ý nghĩa |
|---|---|---|
| ΔFPR | \|FPR_group0 − FPR_group1\| | Sai khác false positive rate giữa nhóm |
| ΔFNR | \|FNR_group0 − FNR_group1\| | Sai khác false negative rate giữa nhóm |
| EO-gap | ΔFPR + ΔFNR | Mức vi phạm Equalized Odds, dùng làm fairness metric chính |

## 3. Quy tắc chọn threshold

| Rule | Mô tả | Dùng ở giai đoạn |
|---|---|---|
| Validation F1 | Quét percentile anomaly score trên validation, chọn θ có F1 cao nhất | D6-D8 baseline |
| Fixed contamination | Chọn θ theo tỷ lệ anomaly giả định | Đối chứng/ablation sau D8 |
| Top-k% | Gắn anomaly cho top k% score cao nhất | D9-D11 nếu làm post-processing |

## 4. Nguyên tắc báo cáo

- Không chọn threshold trên test set.
- Báo cáo mean ± std trên 10 seed.
- Với utility, PR-AUC là metric chính.
- Với fairness, EO-gap là metric chính.
- Cần báo cáo đồng thời utility và fairness để tránh kết luận một chiều.
