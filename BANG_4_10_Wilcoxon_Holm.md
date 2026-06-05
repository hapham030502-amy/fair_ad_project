# Bảng 4.10. Thống kê tóm tắt cuối cùng với Wilcoxon và Holm-Bonferroni

Bảng này dùng để chèn vào Chương 4 của luận văn. Cột p-value hiệu chỉnh Holm là kết quả hiệu chỉnh Holm-Bonferroni cho 12 kiểm định non-baseline.

| Dataset | Phương pháp | Chi tiết | Metric | Mean | Std | Δ vs Baseline | p-value gốc | p-value hiệu chỉnh Holm | Có ý nghĩa sau Holm (α=0.05) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adult | Baseline | AutoEncoder | PR-AUC | 0.5060 | 0.0136 | 0.0000 |  |  |  |
| adult | Baseline | AutoEncoder | F1 | 0.4106 | 0.0162 | 0.0000 |  |  |  |
| adult | Baseline | AutoEncoder | EO-gap | 0.0793 | 0.0255 | 0.0000 |  |  |  |
| adult | Post-processing | Global threshold - param=94.0000 | PR-AUC | 0.5060 | 0.0136 | 0.0000 | 1.0000 | 1.0000 | No |
| adult | Post-processing | Global threshold - param=94.0000 | F1 | 0.3288 | 0.0015 | -0.0818 | 0.0020 | 0.0234 | Yes |
| adult | Post-processing | Global threshold - param=94.0000 | EO-gap | 0.0426 | 0.0031 | -0.0367 | 0.0020 | 0.0234 | Yes |
| adult | In-processing | Reweighted IsolationForest | PR-AUC | 0.3644 | 0.0158 | -0.1416 | 0.0020 | 0.0234 | Yes |
| adult | In-processing | Reweighted IsolationForest | F1 | 0.4336 | 0.0094 | 0.0230 | 0.0039 | 0.0234 | Yes |
| adult | In-processing | Reweighted IsolationForest | EO-gap | 0.2028 | 0.0434 | 0.1236 | 0.0020 | 0.0234 | Yes |
| credit_default | Baseline | AutoEncoder | PR-AUC | 0.3105 | 0.0042 | 0.0000 |  |  |  |
| credit_default | Baseline | AutoEncoder | F1 | 0.3795 | 0.0102 | 0.0000 |  |  |  |
| credit_default | Baseline | AutoEncoder | EO-gap | 0.0276 | 0.0082 | 0.0000 |  |  |  |
| credit_default | Post-processing | Per-group FPR threshold - param=0.1662 | PR-AUC | 0.3092 | 0.0040 | -0.0013 | 0.0020 | 0.0234 | Yes |
| credit_default | Post-processing | Per-group FPR threshold - param=0.1662 | F1 | 0.3405 | 0.0068 | -0.0390 | 0.0020 | 0.0234 | Yes |
| credit_default | Post-processing | Per-group FPR threshold - param=0.1662 | EO-gap | 0.0049 | 0.0031 | -0.0227 | 0.0020 | 0.0234 | Yes |
| credit_default | In-processing | Reweighted IsolationForest | PR-AUC | 0.3108 | 0.0073 | 0.0003 | 0.4922 | 0.9844 | No |
| credit_default | In-processing | Reweighted IsolationForest | F1 | 0.3936 | 0.0066 | 0.0141 | 0.0059 | 0.0234 | Yes |
| credit_default | In-processing | Reweighted IsolationForest | EO-gap | 0.0223 | 0.0125 | -0.0053 | 0.2754 | 0.8262 | No |
