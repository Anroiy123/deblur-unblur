import cv2

from tests.conftest import FakeEasyOCRReader
from utils import ocr


def test_get_reader_is_cached(monkeypatch):
    created = []

    def factory(*_args, **_kwargs):
        reader = FakeEasyOCRReader()
        created.append(reader)
        return reader

    monkeypatch.setattr(ocr, "_reader", None)
    monkeypatch.setattr(ocr.easyocr, "Reader", factory)
    first = ocr.get_reader()
    second = ocr.get_reader()
    assert first is second
    assert len(created) == 1


def test_extract_text_concatenates_lines(text_like_image, monkeypatch):
    fake_reader = FakeEasyOCRReader(results=[(None, "line one", 0.9), (None, "line two", 0.8)])
    monkeypatch.setattr(ocr, "get_reader", lambda: fake_reader)
    result = ocr.extract_text(text_like_image)
    assert result == "line one\nline two"


def test_extract_text_accepts_grayscale(text_like_image, monkeypatch):
    gray = cv2.cvtColor(text_like_image, cv2.COLOR_BGR2GRAY)
    fake_reader = FakeEasyOCRReader(results=[(None, "gray text", 0.9)])
    monkeypatch.setattr(ocr, "get_reader", lambda: fake_reader)
    result = ocr.extract_text(gray)
    assert result == "gray text"
    assert fake_reader.calls[0].ndim == 2


def test_extract_text_returns_error_string_on_failure(text_like_image, monkeypatch):
    fake_reader = FakeEasyOCRReader(should_raise=True)
    monkeypatch.setattr(ocr, "get_reader", lambda: fake_reader)
    result = ocr.extract_text(text_like_image)
    assert result.startswith("Error during OCR:")


def test_extract_text_rejects_removed_paddle_backend(text_like_image):
    result = ocr.extract_text(text_like_image, backend="paddleocr")
    assert result == "Error during OCR: Unsupported OCR backend: paddleocr"


def test_extract_text_with_details_rejects_removed_paddle_backend(text_like_image):
    result = ocr.extract_text_with_details(text_like_image, backend="paddleocr")
    assert result.ok is False
    assert result.error == "Unsupported OCR backend: paddleocr"


def test_calculate_accuracy_handles_empty_text():
    assert ocr.calculate_accuracy("", "abc") == 0.0
    assert ocr.calculate_accuracy("abc", "") == 0.0


def test_calculate_accuracy_normalizes_whitespace_and_case():
    assert ocr.calculate_accuracy("ABC 123", "abc123") == 100.0


def test_calculate_accuracy_partial_match():
    assert ocr.calculate_accuracy("abcd", "abzz") == 50.0


def test_calculate_accuracy_uses_edit_distance_for_insertions():
    assert round(ocr.calculate_accuracy("abcdef", "xabcdef"), 2) == 83.33


def test_calculate_accuracy_is_clamped_at_zero():
    assert ocr.calculate_accuracy("a", "abcdefgh") == 0.0


def test_error_rate_metrics_use_edit_distance():
    assert ocr.calculate_character_error_rate("abc", "abc") == 0.0
    assert round(ocr.calculate_character_error_rate("abc", "axc"), 1) == 33.3
    assert ocr.calculate_word_error_rate("hello world", "hello there") == 50.0


def test_compare_ocr_results_returns_expected_stats():
    result = ocr.compare_ocr_results("a b\nc", "abcd\nef")
    assert result == {
        "original_chars": 3,
        "enhanced_chars": 6,
        "original_lines": 2,
        "enhanced_lines": 2,
        "char_improvement": 3,
        "line_improvement": 0,
    }
