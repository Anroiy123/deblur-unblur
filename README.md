# Document Image Restoration

Ứng dụng Streamlit dùng để khử mờ và khôi phục ảnh giấy tờ. Pipeline hỗ trợ:

```text
DocRes -> Restormer -> NAFNet -> OpenCV
```

Ứng dụng chỉ xử lý và hiển thị ảnh trước/sau. Không có bước nhận dạng văn bản.

## Yêu cầu

- Windows + PowerShell
- Python 3.12
- Git có hỗ trợ submodule

## Lấy mã nguồn

Clone kèm các repository model ngoài:

```powershell
git clone --recurse-submodules https://github.com/Anroiy123/deblur-unblur.git
cd deblur-unblur
```

Nếu repo đã được clone trước đó:

```powershell
git submodule update --init --recursive
```

## Cài đặt

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-ai.txt
```

## Model weights

Weights không nằm trong Git. Đặt các file tại:

```text
external/DocRes/checkpoints/docres.pkl
external/DocRes/data/MBD/checkpoint/mbd.pkl
external/Restormer/Motion_Deblurring/pretrained_models/motion_deblurring.pth
external/NAFNet/experiments/pretrained_models/NAFNet-GoPro-width64.pth
```

NAFNet dùng config:

```text
external/NAFNet/options/test/GoPro/NAFNet-width64.yml
```

## Chạy ứng dụng

```powershell
.\.venv312\Scripts\python.exe -m streamlit run app.py
```

Mở `http://127.0.0.1:8501`.

Nếu cổng đang bị một instance cũ giữ, dừng process cũ trước khi chạy lại:

```powershell
$listener = Get-NetTCPConnection -State Listen -LocalPort 8501 -ErrorAction SilentlyContinue
if ($listener) { Stop-Process -Id $listener.OwningProcess -Force }
```

## Sử dụng

1. Tải ảnh JPG hoặc PNG.
2. Mở `Cấu hình nâng cao`.
3. Chọn backend khôi phục:
   - `Tự động tốt nhất`: DocRes → Restormer → NAFNet → OpenCV.
   - `DocRes`: ưu tiên khôi phục tài liệu.
   - `Restormer`: ưu tiên motion deblurring.
   - `NAFNet`: ưu tiên deblurring bằng NAFNet.
   - `Chỉ OpenCV`: không gọi model ngoài.
4. Có thể bật `Wiener deconvolution` cho blur nhẹ.
5. Nhấn `Cải thiện ảnh` để xem ảnh trước và sau xử lý.

Khi backend được chọn không chạy được, ứng dụng thử backend tiếp theo trong chuỗi
fallback và hiển thị backend thực tế đã sử dụng.

## Adapter

Các adapter được ứng dụng gọi qua subprocess:

```text
scripts/run_docres.py
scripts/run_restormer.py
scripts/run_nafnet.py
```

Có thể chỉ định Python riêng cho từng backend:

```powershell
$env:DOCRES_PYTHON='C:\path\to\python.exe'
$env:RESTORMER_PYTHON='C:\path\to\python.exe'
$env:NAFNET_PYTHON='C:\path\to\python.exe'
```

Hoặc override command adapter:

```powershell
$env:DOCRES_COMMAND='python C:\path\to\adapter.py --task {task} --input "{input}" --output "{output}"'
$env:RESTORMER_COMMAND='python C:\path\to\adapter.py --task {task} --input "{input}" --output "{output}"'
$env:NAFNET_COMMAND='python C:\path\to\adapter.py --task {task} --input "{input}" --output "{output}"'
```

## Smoke test backend

```powershell
.\.venv312\Scripts\python.exe scripts\run_docres.py --task deblurring --input path\to\input.jpg --output tmp\docres.png
.\.venv312\Scripts\python.exe scripts\run_restormer.py --task deblurring --input path\to\input.jpg --output tmp\restormer.png
.\.venv312\Scripts\python.exe scripts\run_nafnet.py --task deblurring --input path\to\input.jpg --output tmp\nafnet.png
```

## Kiểm thử

```powershell
.\.venv312\Scripts\python.exe -m pytest -q
```

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
│   ├── app_support.py
│   ├── enhancement.py
│   └── restoration.py
└── tests/
```

## License

Project phục vụ mục đích học tập và thực nghiệm Computer Vision.
