# Thiết kế giao diện xử lý ảnh tối giản

## Mục tiêu

Rút gọn ứng dụng về luồng chính: tải ảnh, chọn mục tiêu xử lý, chạy khử mờ và
xem kết quả OCR. Các tùy chọn kỹ thuật còn cần thiết được đặt trong mục
`Nâng cao` đóng mặc định.

## Phạm vi giao diện

Giao diện chính chỉ gồm:

1. Tải một ảnh JPG hoặc PNG hợp lệ.
2. Chọn chế độ `Tự nhiên` hoặc `Ưu tiên OCR`.
3. Mở mục `Nâng cao` nếu cần thay đổi cấu hình.
4. Bấm `Cải thiện ảnh`.
5. Xem ảnh trước/sau, văn bản OCR và backend thực tế.
6. Tùy chọn nhập ground truth để xem CER/WER.

Loại bỏ khỏi giao diện:

- Trang tạo ảnh mờ thử nghiệm.
- Phát hiện/crop vùng căn cước.
- Ảnh tham chiếu, PSNR và SSIM.
- Cải thiện và so sánh khuôn mặt.
- Sharpness và phần trăm tăng sharpness.
- Thống kê số lượng ký tự OCR.
- Lựa chọn OCR engine và DocRes task.

Các utility tạo blur vẫn được giữ trong mã nguồn để phục vụ kiểm thử.

## Cấu hình nâng cao

Mục `Nâng cao` đóng mặc định và chỉ chứa:

- Khôi phục tài liệu:
  - `Tự động (DocRes, fallback OpenCV)` là mặc định.
  - `Chỉ OpenCV` để bỏ qua DocRes theo lựa chọn người dùng.
- Tiền xử lý OCR:
  - Tự động theo engine.
  - Giữ ảnh màu/gốc.
  - OpenCV threshold.
- Wiener deconvolution, mặc định tắt.

## Luồng xử lý

Ở chế độ tự động, ứng dụng luôn yêu cầu DocRes với task `deblurring`. Nếu
DocRes thiếu cấu hình, timeout hoặc trả lỗi, pipeline dùng OpenCV và hiển thị
thông báo fallback ngắn gọn. Không yêu cầu người dùng cấu hình task.

OCR luôn ưu tiên PaddleOCR. Nếu PaddleOCR không khả dụng hoặc lỗi, pipeline
fallback sang EasyOCR. Engine thực tế được hiển thị trong kết quả nhưng không
được đưa thành lựa chọn ở giao diện.

Kết quả được lưu theo chữ ký gồm nội dung ảnh và các cấu hình xử lý. Nhập
ground truth hoặc rerun giao diện không chạy lại restoration/OCR. Thay ảnh hoặc
đổi cấu hình sẽ làm kết quả cũ không còn được hiển thị.

## Xử lý lỗi

- Tệp vượt giới hạn dung lượng/pixel hoặc không phải JPG/PNG bị từ chối trước
  khi chuyển thành mảng ảnh.
- Lỗi DocRes kích hoạt fallback OpenCV.
- Lỗi PaddleOCR kích hoạt fallback EasyOCR.
- Nếu cả backend chính và fallback đều lỗi, giao diện hiển thị lỗi đúng giai
  đoạn và không gắn nhãn sai thành lỗi đọc tệp.

## Kiểm thử

- Unit test xác nhận cấu hình mặc định chọn DocRes và fallback OpenCV.
- Unit test xác nhận lựa chọn `Chỉ OpenCV` không gọi DocRes.
- Regression test cho chữ ký kết quả khi thay đổi cấu hình nâng cao.
- Full pytest, compileall và pip check.
- Playthrough Streamlit: upload, xử lý mặc định, xác nhận backend/fallback,
  nhập ground truth và kiểm tra kết quả vẫn tồn tại sau rerun.
