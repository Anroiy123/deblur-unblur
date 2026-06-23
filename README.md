# ID Card Deblur + OCR Application

Ứng dụng Streamlit một trang để khử mờ ảnh căn cước/giấy tờ và trích xuất
văn bản OCR. Luồng mặc định ưu tiên DocRes rồi tự fallback OpenCV; OCR ưu tiên
PaddleOCR rồi fallback EasyOCR.

## Kiến trúc Pipeline

```text
uploaded image
  -> restoration backend
       - DocRes deblurring (default)
       - OpenCV fallback or manual override
  -> OCR preprocessing policy
       - auto by engine
       - preserve color/original
       - OpenCV threshold
  -> OCR backend
       - PaddleOCR (preferred)
       - EasyOCR fallback
  -> optional CER/WER against text ground truth
```

## Tính Năng

- **Document restoration**: mặc định thử DocRes `deblurring`, tự fallback OpenCV nếu lỗi.
- **Enhancement modes**: chế độ tự nhiên để dễ xem và chế độ OCR để tăng nét chữ.
- **OCR tự động**: ưu tiên PaddleOCR và fallback EasyOCR, không yêu cầu người dùng chọn engine.
- **Backend-aware OCR preprocessing**: PaddleOCR mặc định giữ ảnh màu/gốc để tránh threshold phá nét chữ; EasyOCR baseline vẫn có thể dùng threshold OpenCV.
- **Cấu hình nâng cao**: cho phép ép OpenCV, đổi preprocessing và bật Wiener deconvolution.
- **OCR metrics**: CER và WER khi người dùng có ground truth văn bản.

## Cài Đặt

Yêu cầu Python 3.9+. Cấu hình đã được kiểm thử chính trên Python 3.12.

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

DocRes được khai báo dưới dạng Git submodule. Khi clone mới, lấy mã nguồn
submodule bằng một trong hai cách:

```powershell
git clone --recurse-submodules <repository-url>
# Hoặc với repository đã clone:
git submodule update --init --recursive
```

Weights không nằm trong lịch sử Git. Đặt các tệp model tại:

- `external/DocRes/checkpoints/docres.pkl`
- `external/DocRes/data/MBD/checkpoint/mbd.pkl`

Khi mã nguồn và weights đã sẵn sàng, app tự dùng wrapper
`scripts/run_docres.py`. Ngoài ra có thể cấu hình command adapter riêng:

```powershell
$env:DOCRES_COMMAND='python C:\path\to\adapter.py --task {task} --input "{input}" --output "{output}"'
streamlit run app.py
```

Các placeholder của command adapter:

- `{input}`: ảnh tạm đầu vào do app ghi ra.
- `{output}`: ảnh tạm mà lệnh DocRes phải tạo.
- `{task}`: `end2end` hoặc `deblurring`.

DocRes là lựa chọn mặc định. Nếu chưa cấu hình, timeout hoặc lệnh lỗi, app tự
fallback về OpenCV. Trong `Cấu hình nâng cao`, người dùng có thể chọn `Chỉ
OpenCV` để bỏ qua DocRes.

## Phân Tích Tệp DOCX Tham Chiếu

Script `create_report.py` chỉ phân tích cấu trúc/định dạng của một tệp DOCX và
ghi kết quả ra tệp văn bản; script không tự cài dependency và không tự tạo báo
cáo khoa học hoàn chỉnh.

```powershell
python -m pip install python-docx
python create_report.py .\reference.docx --output .\BaoCao_DeBlur_analysis.txt
```

## Kiểm Thử

```powershell
python -m pytest
```

Các test hiện khóa những contract chính:

- OCR EasyOCR cache và text extraction.
- PaddleOCR result parser và fallback sang EasyOCR.
- Policy tiền xử lý OCR theo engine.
- DocRes adapter fallback.
- Chính sách mặc định DocRes và override OpenCV.
- Enhancement, detection, face, metrics và blur generator ở tầng utility.

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

- UI không hiển thị sharpness/PSNR/SSIM vì chúng không phản ánh trực tiếp độ đúng OCR.
- Với OCR hiện đại như PaddleOCR, không threshold ảnh quá sớm nếu chưa có số liệu chứng minh tốt hơn.
- OpenCV baseline vẫn hữu ích cho demo nhanh, CPU-only và fallback.
- App cố định DocRes task `deblurring` để giữ luồng sử dụng đơn giản.

## License

Project phục vụ mục đích học tập/thực nghiệm Computer Vision.
