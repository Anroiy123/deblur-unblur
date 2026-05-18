# ID Card Deblur + OCR Application

Ứng dụng Streamlit để khử mờ ảnh căn cước/giấy tờ, cải thiện vùng ảnh quan trọng và trích xuất văn bản OCR. Repo hiện giữ pipeline OpenCV/EasyOCR làm baseline chạy được ngay, đồng thời đã refactor để đi theo hướng khuyến nghị: thêm lớp khôi phục tài liệu kiểu DocRes và ưu tiên PaddleOCR khi môi trường đã cài backend AI.

## Kiến trúc Pipeline

```text
uploaded image
  -> card detection / perspective correction
  -> restoration backend
       - OpenCV baseline
       - DocRes command adapter (optional)
  -> optional face-region enhancement
  -> OCR preprocessing policy
       - auto by engine
       - preserve color/original
       - OpenCV threshold
  -> OCR backend
       - PaddleOCR (recommended, optional)
       - EasyOCR (baseline/fallback)
  -> OCR comparison + accuracy/CER/WER metrics
```

## Tính Năng

- **Card detection**: phát hiện vùng căn cước và hiệu chỉnh phối cảnh.
- **Document restoration**: OpenCV baseline chạy ngay; DocRes adapter có thể cấu hình ngoài qua `DOCRES_COMMAND`.
- **Enhancement modes**: chế độ tự nhiên để dễ xem và chế độ OCR để tăng nét chữ.
- **OCR backend switch**: PaddleOCR là hướng chính; EasyOCR được giữ làm baseline/fallback.
- **Backend-aware OCR preprocessing**: PaddleOCR mặc định giữ ảnh màu/gốc để tránh threshold phá nét chữ; EasyOCR baseline vẫn có thể dùng threshold OpenCV.
- **OCR metrics**: so sánh ký tự, độ chính xác đơn giản, CER và WER khi có ground truth.
- **Synthetic blur generator**: tạo ảnh Gaussian/motion/defocus blur để kiểm thử.

## Cài Đặt

Yêu cầu Python 3.8+.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Nếu muốn dùng PaddleOCR:

```powershell
pip install -r requirements-ai.txt
```

`requirements-ai.txt` được tách riêng vì PaddleOCR/PaddlePaddle nặng hơn và có thể cần chọn bản phù hợp CPU/GPU của máy.

## Chạy Ứng Dụng

```powershell
streamlit run app.py
```

Mở trình duyệt tại `http://localhost:8501`.

## Cấu Hình DocRes Adapter

DocRes không được nhúng trực tiếp vào repo này vì cần repo/weights riêng. App hỗ trợ gọi DocRes qua biến môi trường:

```powershell
$env:DOCRES_COMMAND='python C:\path\to\DocRes\inference.py --task {task} --input "{input}" --output "{output}"'
streamlit run app.py
```

Các placeholder:

- `{input}`: ảnh tạm đầu vào do app ghi ra.
- `{output}`: ảnh tạm mà lệnh DocRes phải tạo.
- `{task}`: `end2end` hoặc `deblurring`.

Nếu chọn DocRes nhưng chưa cấu hình hoặc lệnh lỗi, app tự fallback về OpenCV baseline để workflow không bị gãy.

## Kiểm Thử

```powershell
python -m pytest
```

Các test hiện khóa những contract chính:

- OCR EasyOCR cache và text extraction.
- PaddleOCR result parser và fallback sang EasyOCR.
- Policy tiền xử lý OCR theo engine.
- DocRes adapter fallback.
- Enhancement, detection, face, metrics và blur generator.

## Cấu Trúc

```text
deblur-unblur/
├── app.py
├── requirements.txt
├── requirements-ai.txt
├── utils/
│   ├── enhancement.py
│   ├── restoration.py
│   ├── text_processing.py
│   ├── ocr.py
│   ├── detection.py
│   ├── face.py
│   ├── metrics.py
│   └── blur_generator.py
└── tests/
```

## Ghi Chú Kỹ Thuật

- Không nên tối ưu theo sharpness/Laplacian đơn thuần. Chỉ số này vẫn hiển thị để tham khảo, nhưng mục tiêu chính là OCR đúng hơn.
- Với OCR hiện đại như PaddleOCR, không threshold ảnh quá sớm nếu chưa có số liệu chứng minh tốt hơn.
- OpenCV baseline vẫn hữu ích cho demo nhanh, CPU-only và fallback.
- DocRes nên được đánh giá bằng tập ảnh CCCD/giấy tờ thật với ground truth để chọn giữa `end2end` và `deblurring`.

## License

Project phục vụ mục đích học tập/thực nghiệm Computer Vision.
