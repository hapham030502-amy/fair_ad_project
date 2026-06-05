# Báo cáo tinh chỉnh LOF trên Credit Default

LOF được tinh chỉnh mở rộng trên validation set với `n_neighbors ∈ {5, 10, 20, 35, 50, 75, 100, 150, 200, 300, 500, 750, 1000}`. Tiêu chí chọn tham số là **PR-AUC trên validation set**; test set chỉ dùng để đánh giá sau khi đã chọn tham số.

## Kết quả top theo validation PR-AUC

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

## Kết luận

Kết quả tốt nhất theo validation PR-AUC là `n_neighbors = 1000`. Trên test set, cấu hình này đạt ROC-AUC = 0.6281, PR-AUC = 0.2829 và EO-gap = 0.0985. So với cấu hình cũ `n_neighbors = 20` (ROC-AUC = 0.4858, PR-AUC = 0.2104, EO-gap = 0.0798), LOF cải thiện rõ rệt về ROC-AUC và PR-AUC.

Tuy nhiên, LOF vẫn không phải mô hình tốt nhất trên Credit Default vì AutoEncoder và IsolationForest vẫn có PR-AUC cao hơn. Đồng thời EO-gap tăng so với cấu hình `n_neighbors = 100`, cho thấy cải thiện utility có thể đi kèm đánh đổi fairness. Vì vậy, LOF nên được giữ như baseline density-based để đối chứng, không phải mô hình khuyến nghị cuối cùng.
