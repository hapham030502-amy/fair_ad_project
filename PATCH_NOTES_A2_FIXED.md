# Patch notes - Sửa A2 label noise theo góp ý GVHD

## Nội dung đã sửa

1. Sửa `src/make_d11_ablation.py`:
   - A2 không còn lật `y_true` của test.
   - Label noise chỉ áp dụng trên train set bằng cách đưa một tỷ lệ mẫu anomaly từ validation vào train như mẫu normal.
   - Mô hình được fit lại cho từng mức noise và từng seed.
   - Metric PR-AUC, F1, ΔFPR, ΔFNR, EO_gap luôn tính trên `y_test_clean`.
   - Raw result có các cột kiểm chứng: `noise_applied_to=train_only`, `test_y_unchanged=True`, `actual_train_noise`, `n_train_noisy`, `n_calibration`.

2. Sửa `config/config.yaml`:
   - Tăng seed từ 5 lên 10:
     `[42, 123, 456, 789, 1011, 2026, 31415, 27182, 16180, 14142]`.

3. Chạy lại D11:
   - Cập nhật `results/d11/D11_ablation_raw_results.csv`.
   - Cập nhật `results/d11/D11_ablation_summary.csv`.
   - Cập nhật `results/d11/D11_ablation_conclusions.csv`.
   - Cập nhật hình `results/d11/d11_a2_label_noise_prauc.png`.
   - Cập nhật báo cáo `D11_Ablation_Study_Report.md/.docx`.

4. Cập nhật D12:
   - Sửa `src/make_d12_repro_package.py` để danh sách seed là 10 seed.
   - Chạy lại reproducibility package, cập nhật manifest SHA-256.

## Kết quả A2 mới

Sau khi sửa, PR-AUC trên Credit Default không còn tăng theo noise:

| Dataset | Train label noise | PR-AUC mean | EO_gap mean |
|---|---:|---:|---:|
| adult | 0.00 | 0.505993 | 0.042313 |
| adult | 0.05 | 0.504569 | 0.054066 |
| adult | 0.10 | 0.502898 | 0.053186 |
| adult | 0.15 | 0.501332 | 0.053790 |
| credit_default | 0.00 | 0.308620 | 0.014098 |
| credit_default | 0.05 | 0.300505 | 0.016827 |
| credit_default | 0.10 | 0.292978 | 0.011729 |
| credit_default | 0.15 | 0.283270 | 0.011398 |

## Cách chạy lại

```bash
python -m src.make_d11_ablation
python -m src.make_d12_repro_package
```

