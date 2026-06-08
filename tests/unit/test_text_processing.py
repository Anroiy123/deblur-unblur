import numpy as np

from utils import text_processing


def test_convert_to_grayscale_from_bgr(text_like_image):
    gray = text_processing.convert_to_grayscale(text_like_image)
    assert gray.ndim == 2


def test_convert_to_grayscale_passthrough_for_gray(text_like_image):
    gray = text_like_image[:, :, 0]
    result = text_processing.convert_to_grayscale(gray)
    assert np.array_equal(result, gray)


def test_apply_text_clahe_preserves_shape(text_like_image):
    gray = text_processing.convert_to_grayscale(text_like_image)
    result = text_processing.apply_text_clahe(gray)
    assert result.shape == gray.shape


def test_sharpen_text_preserves_shape(text_like_image):
    gray = text_processing.convert_to_grayscale(text_like_image)
    result = text_processing.sharpen_text(gray)
    assert result.shape == gray.shape


def test_adaptive_threshold_supports_gaussian_and_mean(text_like_image):
    gray = text_processing.convert_to_grayscale(text_like_image)
    gaussian = text_processing.adaptive_threshold(gray, method="gaussian")
    mean = text_processing.adaptive_threshold(gray, method="mean")
    assert set(np.unique(gaussian)).issubset({0, 255})
    assert set(np.unique(mean)).issubset({0, 255})


def test_morphological_operations_support_all_modes():
    image = np.zeros((20, 20), dtype=np.uint8)
    image[5:15, 5:15] = 255
    for mode in ("opening", "closing", "both"):
        result = text_processing.morphological_operations(image, operation=mode)
        assert result.shape == image.shape


def test_preprocess_for_ocr_returns_binary_image(text_like_image):
    result = text_processing.preprocess_for_ocr(text_like_image)
    assert result.ndim == 2
    assert set(np.unique(result)).issubset({0, 255})


def test_prepare_image_for_ocr_auto_keeps_paddle_input_unthresholded(text_like_image):
    result = text_processing.prepare_image_for_ocr(text_like_image, backend="paddleocr", profile="auto")
    assert result.shape == text_like_image.shape
    assert np.array_equal(result, text_like_image)


def test_prepare_image_for_ocr_auto_thresholds_easyocr_input(text_like_image):
    result = text_processing.prepare_image_for_ocr(text_like_image, backend="easyocr", profile="auto")
    assert result.ndim == 2
    assert set(np.unique(result)).issubset({0, 255})


def test_prepare_image_for_ocr_grayscale_profile(text_like_image):
    result = text_processing.prepare_image_for_ocr(text_like_image, profile="grayscale")
    assert result.ndim == 2
