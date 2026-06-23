from pathlib import Path


APP_SOURCE = Path("app.py").read_text(encoding="utf-8")


def test_main_ui_contains_only_the_restoration_flow():
    required_labels = (
        "Khử mờ tài liệu",
        "Chọn ảnh cần cải thiện",
        "Cấu hình nâng cao",
        "Tự động tốt nhất (DocRes -> Restormer -> NAFNet -> OpenCV)",
        '"DocRes": DOCRES_RESTORATION',
        '"Restormer": RESTORMER_RESTORATION',
        '"NAFNet": NAFNET_RESTORATION',
        '"Chỉ OpenCV": OPENCV_RESTORATION',
        "Cải thiện ảnh",
        "Khôi phục thực tế",
    )
    removed_labels = (
        "Văn bản OCR",
        "OCR thực tế",
        "OCR từ ảnh đầu vào",
        "OCR từ ảnh sau xử lý",
        "ground truth",
        "CER",
        "WER",
        "Tiền xử lý OCR",
        "Ưu tiên OCR",
        "PADDLEOCR_BACKEND",
        "EASYOCR_BACKEND",
        "Chọn trang",
        "Tạo ảnh mờ thử nghiệm",
        "Phát hiện vùng căn cước",
        "ảnh tham chiếu",
        "cải thiện vùng khuôn mặt",
        "Độ nét ảnh đầu vào",
        "Số ký tự cải thiện",
    )

    for label in required_labels:
        assert label in APP_SOURCE
    for label in removed_labels:
        assert label not in APP_SOURCE


def test_app_no_longer_imports_removed_feature_modules():
    removed_imports = (
        "utils.detection",
        "utils.face",
        "utils.blur_generator",
        "calculate_psnr",
        "calculate_ssim",
        "calculate_sharpness",
        "utils.ocr",
        "utils.text_processing",
    )

    for removed_import in removed_imports:
        assert removed_import not in APP_SOURCE
