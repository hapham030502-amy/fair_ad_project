# fair_ad_raw - D1 đến D8

Dự án phục vụ luận văn thạc sĩ: **Nghiên cứu bài toán phát hiện bất thường và sự công bằng trong mất cân bằng dữ liệu học máy**.

Bộ mã này hoàn thiện các sản phẩm D1-D8:

- **D1**: Problem Statement + Research Questions + Hypotheses.
- **D2**: Decision Log.
- **D3**: Code repository skeleton + `config/config.yaml`.
- **D4**: Dataset Cards cho Adult và Credit Default.
- **D5**: Metrics Definition Table.
- **D6**: Baseline Results v1.
- **D7**: Score Distribution Figures.
- **D8**: Error Audit Report.

## 1. Cấu trúc thư mục

```text
fair_ad_raw/
├── config/
│   └── config.yaml
├── data/
│   ├── adult.data
│   ├── adult.test
│   ├── default of credit card clients.xls
│   └── processed/
│       ├── adult/
│       └── credit_default/
├── src/
│   ├── data_loader.py
│   ├── metrics.py
│   ├── run_all_models.py
│   ├── make_baseline_results.py
│   ├── plot_score_distributions.py
│   ├── make_dataset_cards.py
│   ├── make_error_audit.py
│   ├── generate_d1_d2_d5.py
│   └── models/
├── tests/
│   └── test_metrics.py
├── scripts/
│   ├── check_d1_d8.sh
│   └── run_all.sh
├── results/
├── environment.yml
├── requirements.txt
└── README.md
```

## 2. Cài đặt môi trường

Khuyến nghị dùng Python 3.9.

```bash
conda env create -f environment.yml
conda activate fair_ad
```

Nếu dùng pip:

```bash
pip install -r requirements.txt
```

Dự án đọc file `.parquet`, vì vậy bắt buộc cần `pyarrow`.

## 3. Kiểm tra nhanh D1-D8

```bash
python -m src.data_loader --dataset adult
python -m src.data_loader --dataset credit_default
python -m unittest tests/test_metrics.py
```

Hoặc:

```bash
bash scripts/check_d1_d8.sh
```

## 4. Chạy toàn bộ D1-D8

```bash
bash scripts/run_all.sh
```

Nếu muốn chạy nhanh để kiểm tra trước:

```bash
python -m src.run_all_models --max-train-samples 3000 --ae-epochs 5 --svdd-epochs 5
```

Nếu muốn chạy đầy đủ:

```bash
python -m src.run_all_models --max-train-samples 0 --ae-epochs 20 --svdd-epochs 20
```

## 5. File đầu ra

Sau khi chạy, cần có:

```text
D1_Problem_Statement_RQs_Hypotheses.md
D2_Decision_Log.md
D5_Metrics_Definition_Table.md
results/cards/D4_Dataset_Card_adult.md
results/cards/D4_Dataset_Card_credit_default.md
results/all_results.csv
results/per_seed_results.csv
results/summary.csv
results/baseline_results_numeric.csv
results/baseline_results.csv
results/score_outputs.csv
results/figures/*.png
results/figures/*.pdf
results/d8/D8_Error_Audit_Report.md
results/d8/subgroup_performance.csv
results/d8/subgroup_performance_summary.csv
results/d8/bias_source_summary.csv
results/d8/proxy_feature_analysis.csv
```

## 6. Điều kiện đạt D1-D8

- D1, D2, D5 có file `.md` rõ nội dung.
- D3 có cấu trúc repo, config, loader, test, README và môi trường.
- D4 có dataset card cho 2 dataset.
- D6 có bảng baseline cho 2 dataset x 5 model x 10 seed.
- D7 có hình phân phối score theo nhóm nhạy cảm.
- D8 có subgroup audit, bias source summary, proxy feature analysis và report.
