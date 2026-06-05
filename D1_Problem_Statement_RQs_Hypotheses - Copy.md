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

Ngày sinh file: 2026-05-05
