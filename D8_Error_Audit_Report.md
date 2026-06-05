# D8 Error Audit Report: Bias đến từ đâu?

## 1. Mục tiêu

Báo cáo này phân tích lỗi theo nhóm nhạy cảm để xác định bias có thể đến từ dữ liệu, mô hình hoặc threshold. Các kết quả dưới đây đã được đồng bộ với `results/summary.csv`, sử dụng 10 seed và phiên bản LOF đã tinh chỉnh mở rộng trên Credit Default.

## 2. Tóm tắt phát hiện chính

- `adult`: PR-AUC cao nhất thuộc `AutoEncoder` (0.5060); EO-gap thấp nhất thuộc `OCSVM` (0.0765); EO-gap cao nhất thuộc `IsolationForest` (0.2577).
- `credit_default`: PR-AUC cao nhất thuộc `AutoEncoder` (0.3105); EO-gap thấp nhất thuộc `AutoEncoder` (0.0276); EO-gap cao nhất thuộc `LOF` (0.0985).
- LOF trên Credit Default đã được tinh chỉnh mở rộng theo PR-AUC validation. Cấu hình tốt nhất là `n_neighbors = 1000`, giúp ROC-AUC tăng từ 0.4858 lên 0.6281 và PR-AUC tăng từ 0.2104 lên 0.2829. Tuy nhiên EO-gap tăng lên 0.0985, cho thấy trade-off utility–fairness.

## 3. Tóm tắt Error Audit

| Dataset        | PR-AUC cao nhất      | EO-gap thấp nhất     | EO-gap cao nhất          | Nguồn bias nổi bật        |
|:---------------|:---------------------|:---------------------|:-------------------------|:--------------------------|
| adult          | AutoEncoder (0.5060) | OCSVM (0.0765)       | IsolationForest (0.2577) | threshold/error-rate bias |
| credit_default | AutoEncoder (0.3105) | AutoEncoder (0.0276) | LOF (0.0985)             | threshold/error-rate bias |

## 4. Kết quả tinh chỉnh LOF trên Credit Default

LOF được tinh chỉnh riêng trên Credit Default với `n_neighbors ∈ {1000, 750, 500, 300, 200, 150, 100, 75, 50, 35, 5, 20, 10}`. Tiêu chí chọn tham số là PR-AUC trên validation set; test set chỉ dùng để đánh giá sau khi đã chốt tham số.

|   n_neighbors |   val_pr_auc |   val_roc_auc |   roc_auc |   pr_auc |       f1 |    eo_gap |
|--------------:|-------------:|--------------:|----------:|---------:|---------:|----------:|
|          1000 |     0.290902 |      0.63829  |  0.628128 | 0.28291  | 0.382646 | 0.0985247 |
|           750 |     0.277456 |      0.621351 |  0.611411 | 0.269506 | 0.356987 | 0.0819097 |
|           500 |     0.260446 |      0.590445 |  0.581341 | 0.251664 | 0.318438 | 0.075055  |
|           300 |     0.244848 |      0.547645 |  0.539972 | 0.236498 | 0.291631 | 0.0712085 |
|           200 |     0.237703 |      0.521023 |  0.517215 | 0.23405  | 0.270624 | 0.0605625 |
|           150 |     0.233707 |      0.509029 |  0.508468 | 0.234939 | 0.263531 | 0.0435372 |
|           100 |     0.229172 |      0.497358 |  0.500864 | 0.233744 | 0.257018 | 0.0483813 |
|            75 |     0.225916 |      0.490669 |  0.499308 | 0.23128  | 0.247586 | 0.0186771 |
|            50 |     0.223174 |      0.491064 |  0.503621 | 0.226482 | 0.253386 | 0.0322805 |
|            35 |     0.217268 |      0.493513 |  0.496531 | 0.217383 | 0.25401  | 0.0532491 |

Diễn giải: mở rộng grid tới `n_neighbors = 1000` giúp LOF không còn gần mức ngẫu nhiên như cấu hình 20 hoặc 100 láng giềng. Tuy nhiên LOF vẫn không phải mô hình tốt nhất trên Credit Default vì AutoEncoder và IsolationForest vẫn có PR-AUC cao hơn. Đồng thời EO-gap của LOF tăng, nên luận văn cần ghi nhận đây là trade-off giữa utility và fairness.

## 5. Các dấu hiệu bias theo mô hình

| dataset        | model           |   anomaly_rate_gap_mean |   pred_positive_rate_gap_mean |   mean_score_gap_mean |   fpr_gap_mean |   fnr_gap_mean |   eo_gap_mean |
|:---------------|:----------------|------------------------:|------------------------------:|----------------------:|---------------:|---------------:|--------------:|
| adult          | AutoEncoder     |               0.185298  |                     0.0730186 |            0.313362   |      0.0448325 |     0.0344471  |     0.0792796 |
| adult          | DeepSVDD        |               0.185298  |                     0.0421425 |            0.00328094 |      0.0206852 |     0.0640289  |     0.0847141 |
| adult          | IsolationForest |               0.185298  |                     0.0426755 |            0.00367566 |      0.0446093 |     0.213099   |     0.257708  |
| adult          | LOF             |               0.185298  |                     0.0291099 |            0.0512592  |      0.0119898 |     0.1121     |     0.12409   |
| adult          | OCSVM           |               0.185298  |                     0.0363956 |            2.44878    |      0.0099627 |     0.0664893  |     0.076452  |
| credit_default | AutoEncoder     |               0.0304712 |                     0.0212962 |            0.0910931  |      0.0199983 |     0.00756946 |     0.0275678 |
| credit_default | DeepSVDD        |               0.0304712 |                     0.0368379 |            0.00144437 |      0.0280756 |     0.0467668  |     0.0748424 |
| credit_default | IsolationForest |               0.0304712 |                     0.0374586 |            0.00344898 |      0.031732  |     0.0339823  |     0.0657143 |
| credit_default | LOF             |               0.0304712 |                     0.0520428 |            0.04132    |      0.0435751 |     0.0549496  |     0.0985247 |
| credit_default | OCSVM           |               0.0304712 |                     0.0266379 |            3.12842    |      0.0166554 |     0.0364478  |     0.0531032 |

## 6. Proxy Feature Analysis

Bảng dưới đây chỉ trình bày tên thuộc tính gốc và giá trị sau mã hóa, không còn dùng các mã chỉ số nội bộ như `feature_050`. Tên feature được lấy từ `OneHotEncoder.get_feature_names_out()` hoặc `stats_rebuilt_without_sensitive.json`.

| dataset   | original_column   | category_value     | feature_mapped                         |   abs_correlation_with_sensitive |   mutual_information_with_sensitive |
|:----------|:------------------|:-------------------|:---------------------------------------|---------------------------------:|------------------------------------:|
| adult     | relationship      | Husband            | cat__relationship_Husband              |                        0.578084  |                          0.225886   |
| adult     | marital_status    | Married-civ-spouse | cat__marital_status_Married-civ-spouse |                        0.418548  |                          0.0940968  |
| adult     | relationship      | Wife               | cat__relationship_Wife                 |                        0.332597  |                          0.0653773  |
| adult     | relationship      | Unmarried          | cat__relationship_Unmarried            |                        0.329667  |                          0.0512219  |
| adult     | occupation        | Adm-clerical       | cat__occupation_Adm-clerical           |                        0.284799  |                          0.0439725  |
| adult     | hours_per_week    | nan                | num__hours_per_week                    |                        0.231961  |                          0.026386   |
| adult     | occupation        | Craft-repair       | cat__occupation_Craft-repair           |                        0.226164  |                          0.0327294  |
| adult     | marital_status    | Divorced           | cat__marital_status_Divorced           |                        0.221674  |                          0.0260934  |
| adult     | marital_status    | Widowed            | cat__marital_status_Widowed            |                        0.18674   |                          0.0325535  |
| adult     | relationship      | Not-in-family      | cat__relationship_Not-in-family        |                        0.172194  |                          0.0136264  |
| adult     | occupation        | Other-service      | cat__occupation_Other-service          |                        0.171672  |                          0.0195702  |
| adult     | marital_status    | Never-married      | cat__marital_status_Never-married      |                        0.162768  |                          0.0226678  |
| adult     | occupation        | Transport-moving   | cat__occupation_Transport-moving       |                        0.127042  |                          0.00335112 |
| adult     | marital_status    | Separated          | cat__marital_status_Separated          |                        0.112526  |                          0.00270634 |
| adult     | workclass         | Self-emp-not-inc   | cat__workclass_Self-emp-not-inc        |                        0.102706  |                          0.0145518  |
| adult     | occupation        | Handlers-cleaners  | cat__occupation_Handlers-cleaners      |                        0.0994992 |                          0          |
| adult     | occupation        | Farming-fishing    | cat__occupation_Farming-fishing        |                        0.0987881 |                          0.00295543 |
| adult     | workclass         | Self-emp-inc       | cat__workclass_Self-emp-inc            |                        0.0932721 |                          0.0152501  |
| adult     | age               | nan                | num__age                               |                        0.0923105 |                          0.00987626 |
| adult     | occupation        | Protective-serv    | cat__occupation_Protective-serv        |                        0.0810557 |                          0.00325603 |

## 7. Kết luận Error Audit

Bias trong bộ kết quả hiện tại không đến từ một nguồn duy nhất, mà là kết quả kết hợp giữa dữ liệu, mô hình và cơ chế chọn threshold. Trên Adult, IsolationForest có EO-gap cao nhất chủ yếu do chênh lệch FNR lớn. Trên Credit Default, sau khi LOF được tinh chỉnh theo PR-AUC, LOF cải thiện mạnh về ROC-AUC/PR-AUC nhưng trở thành mô hình có EO-gap cao nhất; điều này cho thấy cải thiện utility có thể kéo theo đánh đổi fairness. Vì vậy, D9–D10 cần tiếp tục dùng AutoEncoder làm base model theo PR-AUC và áp dụng post-processing để giảm EO-gap.
