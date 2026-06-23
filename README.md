# ID Card Deblur + OCR Application

Ứng dụng Streamlit để khử mờ ảnh giấy tờ và OCR văn bản. Pipeline khôi phục
ảnh hiện tại hỗ trợ:

```text
DocRes -> Restormer -> NAFNet -> OpenCV
```

OCR mặc định dùng `EasyOCR`.

## Yêu cầu

- Windows + PowerShell
- Python 3.12
- Repo được clone đầy đủ tại `deblur-unblur`

## Chạy app

### Cách nhanh nhất trên máy đã có sẵn môi trường

Nếu repo này đã có `.venv312`:

```powershell
.\.venv312\Scripts\python.exe -m streamlit run app.py
```

Mở trình duyệt tại:

```text
http://127.0.0.1:8501
```

Nếu `8501` đang bị chiếm cổng:

```powershell
.\.venv312\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

Khi đó mở:

```text
http://127.0.0.1:8502
```

### Chạy từ đầu từng bước

1. Tạo môi trường ảo Python 3.12:

```powershell
py -3.12 -m venv .venv312
```

2. Kích hoạt môi trường:

```powershell
.\.venv312\Scripts\Activate.ps1
```

3. Cài dependency nền:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

4. Cài dependency AI:

```powershell
python -m pip install -r requirements-ai.txt
python -m pip install gdown natsort yacs addict lmdb lpips joblib h5py tb-nightly yapf
```

5. Chạy app:

```powershell
python -m streamlit run app.py
```

6. Mở app trong trình duyệt:

```text
http://127.0.0.1:8501
```

## Các bước sử dụng trong app

1. Tải ảnh `.jpg`, `.jpeg` hoặc `.png`.
2. Chọn mục tiêu xử lý:
   - `Tự nhiên`: ưu tiên ảnh nhìn tự nhiên hơn.
   - `Ưu tiên OCR`: ưu tiên tương phản và khả năng đọc chữ.
3. Mở `Cấu hình nâng cao` nếu cần:
   - `Khôi phục tài liệu`
   - `Tiền xử lý OCR`
   - `Wiener deconvolution`
4. Nhấn `Cải thiện ảnh`.
5. Xem:
   - ảnh trước/sau xử lý
   - OCR từ ảnh gốc
   - OCR từ ảnh sau xử lý
   - backend thực tế đã dùng

## Backend khôi phục ảnh

Trong UI `Cấu hình nâng cao -> Khôi phục tài liệu` có các lựa chọn:

- `Tự động tốt nhất (DocRes -> Restormer -> NAFNet -> OpenCV)`
- `DocRes`
- `Restormer`
- `NAFNet`
- `Chỉ OpenCV`

Ý nghĩa:

- `Tự động tốt nhất`: thử `DocRes`, nếu lỗi sẽ fallback sang `Restormer`, tiếp đó
  `NAFNet`, cuối cùng là `OpenCV`.
- `DocRes`, `Restormer`, `NAFNet`: ép chạy backend tương ứng; nếu backend đó lỗi,
  app vẫn có thể fallback về backend khác hoặc `OpenCV`.
- `Chỉ OpenCV`: không gọi model ngoài.

## Repo/model ngoài đang dùng

### DocRes

Repo:

```text
external/DocRes
```

Weights:

- `external/DocRes/checkpoints/docres.pkl`
- `external/DocRes/data/MBD/checkpoint/mbd.pkl`

Adapter:

```text
scripts/run_docres.py
```

### Restormer

Repo:

```text
external/Restormer
```

Weight:

```text
external/Restormer/Motion_Deblurring/pretrained_models/motion_deblurring.pth
```

Adapter:

```text
scripts/run_restormer.py
```

### NAFNet

Repo:

```text
external/NAFNet
```

Weight:

```text
external/NAFNet/experiments/pretrained_models/NAFNet-GoPro-width64.pth
```

Config test:

```text
external/NAFNet/options/test/GoPro/NAFNet-width64.yml
```

Adapter:

```text
scripts/run_nafnet.py
```

## Biến môi trường tùy chọn

Nếu muốn override Python dùng cho backend AI:

```powershell
$env:RESTORATION_PYTHON='C:\path\to\python.exe'
```

Hoặc theo từng backend:

```powershell
$env:DOCRES_PYTHON='C:\path\to\python.exe'
$env:RESTORMER_PYTHON='C:\path\to\python.exe'
$env:NAFNET_PYTHON='C:\path\to\python.exe'
```

Nếu muốn override command adapter:

```powershell
$env:DOCRES_COMMAND='python C:\path\to\adapter.py --task {task} --input "{input}" --output "{output}"'
$env:RESTORMER_COMMAND='python C:\path\to\adapter.py --task {task} --input "{input}" --output "{output}"'
$env:NAFNET_COMMAND='python C:\path\to\adapter.py --task {task} --input "{input}" --output "{output}"'
```

## Smoke test backend riêng lẻ

### DocRes

```powershell
.\.venv312\Scripts\python.exe scripts\run_docres.py --task deblurring --input external\DocRes\input\for_debluring.png --output tmp\docres-smoke.png
```

### Restormer

```powershell
.\.venv312\Scripts\python.exe scripts\run_restormer.py --task deblurring --input external\DocRes\input\for_debluring.png --output tmp\restormer-smoke.png
```

### NAFNet

```powershell
.\.venv312\Scripts\python.exe scripts\run_nafnet.py --task deblurring --input external\DocRes\input\for_debluring.png --output tmp\nafnet-smoke.png
```

## Kiểm thử

Chạy toàn bộ test:

```powershell
python -m pytest
```

Chạy test parser/backend UI:

```powershell
python -m pytest tests/unit/test_app_support.py
python -m pytest tests/unit/test_restoration.py
```

## Lỗi thường gặp

### 1. Chọn `NAFNet` hoặc `Restormer` rồi báo lỗi chung trong UI

Nguyên nhân hay gặp nhất là app Streamlit đang chạy bằng instance cũ hoặc code cũ
chưa restart.

Cách xử lý:

1. Dừng app đang chạy bằng `Ctrl + C`.
2. Chạy lại:

```powershell
.\.venv312\Scripts\python.exe -m streamlit run app.py
```

3. Nếu vẫn nghi ngờ cổng cũ đang bị giữ:

```powershell
.\.venv312\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

4. Mở đúng URL mới, ví dụ `http://127.0.0.1:8502`.

Ghi chú: trạng thái repo hiện tại đã được kiểm tra lại với UI và cả `Restormer`
lẫn `NAFNet` đều chạy được trên instance mới.

### 2. Thiếu model/weight

Nếu backend AI không chạy, kiểm tra các đường dẫn:

- `external/DocRes/checkpoints/docres.pkl`
- `external/DocRes/data/MBD/checkpoint/mbd.pkl`
- `external/Restormer/Motion_Deblurring/pretrained_models/motion_deblurring.pth`
- `external/NAFNet/experiments/pretrained_models/NAFNet-GoPro-width64.pth`

### 3. Muốn chỉ chạy bản an toàn, ít phụ thuộc

Chọn `Chỉ OpenCV` trong `Khôi phục tài liệu`.

## Cấu trúc chính

```text
deblur-unblur/
├── app.py
├── requirements.txt
├── requirements-ai.txt
├── scripts/
│   ├── run_docres.py
│   ├── run_restormer.py
│   └── run_nafnet.py
├── external/
│   ├── DocRes/
│   ├── Restormer/
│   └── NAFNet/
├── utils/
└── tests/
```

## Trạng thái đã xác nhận

- App mở được bằng Streamlit local
- Upload ảnh mẫu thành công
- `DocRes`, `Restormer`, `NAFNet`, `OpenCV` đều đã được nối vào UI
- `Restormer` và `NAFNet` đã được smoke test riêng
- `Restormer` và `NAFNet` đã được bấm chạy thành công từ giao diện web trên
  instance mới

## License

Project phục vụ mục đích học tập và thực nghiệm Computer Vision.
