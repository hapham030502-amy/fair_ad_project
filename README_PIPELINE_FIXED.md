# Pipeline đã chuẩn hóa cho D6–D7

## Mục tiêu sửa

Code đã được chỉnh để dùng duy nhất dữ liệu đã xử lý tại:

```text
data/processed/<dataset>/transformed.npz
```

Quy trình mới:

```text
X_train, y_train  -> fit mô hình AD, bắt buộc y_train toàn 0
X_val, y_val      -> chọn threshold bằng F1 trên validation
X_test, y_test    -> đánh giá cuối cùng
s_test            -> tính fairness metrics
```

## Các file đã chỉnh/sửa chính

```text
src/data_loader.py
src/add_sensitive_to_npz.py
src/metrics.py
src/run_all_models.py
src/make_baseline_results.py
scripts/run_all.sh
```

## Cách chạy lại D6–D7

Từ thư mục gốc project:

```bash
bash scripts/run_all.sh
```

Nếu máy mạnh và muốn dùng nhiều mẫu train hơn:

```bash
python -u -m src.run_all_models --max-train-samples 5000
python -m src.make_baseline_results
python -m src.plot_score_distributions
```

Nếu muốn dùng toàn bộ train normal-only:

```bash
python -u -m src.run_all_models --max-train-samples 0
```

## Kiểm tra nhanh loader

```bash
python -m src.data_loader --dataset adult
python -m src.data_loader --dataset credit_default
```

Kết quả đúng cần thấy:

```text
y_train_sum = 0
X_train / X_val / X_test tồn tại
s_test_groups có ít nhất 2 nhóm
```

## Ghi chú về kết quả cũ

Các kết quả cũ trước khi sửa pipeline được để ở:

```text
results_legacy_before_pipeline_fix/
```

Một lần chạy thử ngắn để kiểm tra code được để ở:

```text
results_smoke_test_after_pipeline_fix/
```

Sau khi chạy lại `bash scripts/run_all.sh`, kết quả mới sẽ sinh trong:

```text
results/all_results.csv
results/per_seed_results.csv
results/summary.csv
results/score_outputs.csv
results/baseline_results_numeric.csv
results/baseline_results.csv
```
