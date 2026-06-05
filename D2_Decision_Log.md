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
