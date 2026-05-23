from __future__ import annotations

from pathlib import Path
from datetime import date

ROOT = Path(__file__).resolve().parents[1]


def write(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"Đã lưu: {path}")


def make_d1() -> str:
    return f"""
# D1. Problem Statement + Research Questions + Hypotheses

## 1. Tên đề tài

**Nghiên cứu bài toán phát hiện bất thường và sự công bằng trong mất cân bằng dữ liệu học máy**

## 2. Bối cảnh và vấn đề nghiên cứu

Phát hiện bất thường trên dữ liệu bảng là một hướng quan trọng trong khai phá dữ liệu và học máy. Trong nhiều bối cảnh như tín dụng, thu nhập, rủi ro tài chính hoặc đánh giá hành vi bất thường, dữ liệu thường có đặc điểm mất cân bằng: số mẫu bất thường chiếm tỷ lệ thấp hơn nhiều so với mẫu bình thường. Nếu chỉ tối ưu hiệu năng dự đoán chung, mô hình có thể tạo ra sai khác lớn về tỷ lệ lỗi giữa các nhóm nhạy cảm như giới tính, độ tuổi hoặc chủng tộc.

Trong đề tài này, bất thường được xem là lớp quan tâm có nhãn `1`. Mô hình học từ tập huấn luyện chủ yếu hoặc hoàn toàn là dữ liệu bình thường để học biểu diễn normality, sau đó gán anomaly score cho validation/test. Điểm càng cao biểu thị khả năng bất thường càng lớn.

## 3. Đầu vào và đầu ra của hệ thống

- **Đầu vào**: dữ liệu bảng sau tiền xử lý, gồm các đặc trưng số và đặc trưng phân loại đã mã hóa.
- **Nhãn đánh giá**: `y = 1` là anomaly, `y = 0` là normal.
- **Thuộc tính nhạy cảm**: mặc định dùng `sex`; có thể mở rộng sang `race` với Adult hoặc `age` với Credit Default.
- **Đầu ra**: anomaly score, nhãn dự đoán sau threshold, các chỉ số utility và fairness.

## 4. Câu hỏi nghiên cứu

**RQ1.** Các mô hình phát hiện bất thường truyền thống và học sâu đạt hiệu năng như thế nào trên dữ liệu bảng mất cân bằng?

**RQ2.** Khi dùng cùng một threshold toàn cục, các mô hình có tạo ra sai khác đáng kể về FPR/FNR giữa các nhóm nhạy cảm hay không?

**RQ3.** Nguồn gốc sai lệch chủ yếu đến từ phân bố dữ liệu, điểm số mô hình hay cách chọn threshold?

## 5. Giả thuyết nghiên cứu

**H1.** Mô hình có PR-AUC cao chưa chắc có EO-gap thấp; do đó cần đánh giá đồng thời utility và fairness.

**H2.** Trong bối cảnh dữ liệu mất cân bằng, threshold toàn cục có thể làm tăng sai khác FPR/FNR giữa các nhóm, đặc biệt khi phân phối score hoặc anomaly rate khác nhau theo nhóm nhạy cảm.

## 6. Phạm vi D1-D8

Giai đoạn D1-D8 tập trung vào nền tảng nghiên cứu và baseline audit:

- Chốt bài toán, dữ liệu, metric.
- Xây dựng pipeline đọc dữ liệu và kiểm tra split.
- Chạy baseline với Isolation Forest, LOF, One-Class SVM, AutoEncoder, DeepSVDD.
- Phân tích phân phối score và error audit theo nhóm nhạy cảm.

Ngày sinh file: {date.today().isoformat()}
"""


def make_d2() -> str:
    return """
# D2. Decision Log

## 1. Vì sao chọn tabular anomaly detection?

Đề tài chọn dữ liệu bảng vì đây là dạng dữ liệu phổ biến trong các bài toán rủi ro tín dụng, thu nhập, hồ sơ khách hàng và ra quyết định tự động. Dữ liệu bảng cũng phù hợp với các mô hình baseline kinh điển như Isolation Forest, LOF, One-Class SVM và các mô hình học sâu đơn giản như AutoEncoder, DeepSVDD.

## 2. Vì sao không chọn time-series hoặc graph anomaly detection?

Time-series và graph anomaly detection có cấu trúc dữ liệu, giả định mô hình và cách đánh giá khác biệt. Nếu mở rộng sang các dạng này, phạm vi luận văn sẽ quá rộng và khó kiểm soát biến thực nghiệm. Vì vậy giai đoạn này giới hạn ở tabular AD để đảm bảo có pipeline tái lập và đo fairness rõ ràng.

## 3. Vì sao chọn Adult và Credit Default làm core datasets?

- **Adult Census**: có số mẫu đủ lớn, có thuộc tính nhạy cảm như `sex`, `race`, và có thể định nghĩa anomaly theo nhóm thu nhập cao.
- **Credit Default**: có ý nghĩa thực tiễn trong rủi ro tài chính, có `sex`, `age`, và nhãn default phù hợp với bài toán phát hiện rủi ro/bất thường.

Hai bộ dữ liệu này giúp so sánh giữa bối cảnh xã hội/thu nhập và bối cảnh tài chính/tín dụng.

## 4. Vì sao chọn group fairness?

Group fairness phù hợp với dữ liệu hiện có vì các thuộc tính nhạy cảm như `sex`, `race`, `age` có thể chia nhóm để đo sai khác FPR/FNR. Individual fairness yêu cầu định nghĩa khoảng cách công bằng giữa từng cá thể, khó xác định chắc chắn trong phạm vi dữ liệu bảng hiện có.

## 5. Vì sao chọn các mô hình baseline này?

- **Isolation Forest**: baseline mạnh cho anomaly detection trên dữ liệu bảng.
- **LOF**: đại diện cho hướng density-based anomaly detection.
- **One-Class SVM**: đại diện cho hướng boundary-based/one-class learning.
- **AutoEncoder**: đại diện cho deep reconstruction-based anomaly detection.
- **DeepSVDD**: đại diện cho deep one-class anomaly detection.

## 6. Rủi ro và biện pháp giảm thiểu

| Rủi ro | Biện pháp giảm thiểu |
|---|---|
| Data leakage | Fit tiền xử lý trên train; validation chọn threshold; test chỉ đánh giá cuối |
| Kết quả phụ thuộc seed | Chạy 10 seed: 42, 123, 456, 789, 1011, 2026, 31415, 27182, 16180, 14142 |
| Metric utility không phản ánh imbalance | Ưu tiên PR-AUC, vẫn báo cáo ROC-AUC và F1 |
| Fairness chỉ theo một sensitive attribute | Mặc định audit theo sex, có thể mở rộng race/age |
| Threshold làm méo fairness | Ghi rõ threshold chọn trên validation và phân tích threshold bias ở D8 |
"""


def make_d5() -> str:
    return """
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
"""


def main() -> None:
    write(ROOT / "D1_Problem_Statement_RQs_Hypotheses.md", make_d1())
    write(ROOT / "D2_Decision_Log.md", make_d2())
    write(ROOT / "D5_Metrics_Definition_Table.md", make_d5())


if __name__ == "__main__":
    main()
