from pathlib import Path


APP_SOURCE = Path("app.py").read_text(encoding="utf-8")


def test_main_ui_contains_only_the_simplified_processing_flow():
    required_labels = (
        "Chọn ảnh cần cải thiện",
        "Mục tiêu xử lý",
        "Cấu hình nâng cao",
        "Tự động (DocRes, fallback OpenCV)",
        "Chỉ OpenCV",
        "Cải thiện ảnh",
        "Độ chính xác OCR với ground truth",
    )
    removed_labels = (
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
    )

    for removed_import in removed_imports:
        assert removed_import not in APP_SOURCE
