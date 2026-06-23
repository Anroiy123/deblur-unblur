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


def test_extract_text_supports_paddleocr_legacy_result_shape(text_like_image, monkeypatch):
    class FakePaddleReader:
        def ocr(self, _image, cls=True):
            return [[
                [None, ("CONG HOA XA HOI", 0.98)],
                [None, ("123456789012", 0.97)],
            ]]

    monkeypatch.setattr(ocr, "get_paddleocr_reader", lambda: FakePaddleReader())
    result = ocr.extract_text(text_like_image, backend="paddleocr")
    assert result == "CONG HOA XA HOI\n123456789012"


def test_get_paddleocr_reader_retries_after_value_error(monkeypatch):
    attempts = []
    env_during_import = []

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            attempts.append(kwargs)
            if "show_log" in kwargs:
                raise ValueError("Unknown argument: show_log")

    monkeypatch.setattr(ocr, "_paddle_reader", None)
    monkeypatch.delenv("FLAGS_enable_pir_api", raising=False)

    class FakeModule:
        PaddleOCR = FakePaddleOCR

    def fake_import(name):
        env_during_import.append((name, ocr.os.environ.get("FLAGS_enable_pir_api")))
        return FakeModule()

    monkeypatch.setattr(ocr.importlib, "import_module", fake_import)

    reader = ocr.get_paddleocr_reader()

    assert isinstance(reader, FakePaddleOCR)
    assert env_during_import == [("paddleocr", "0")]
    assert attempts == [
        {
            "use_angle_cls": True,
            "lang": "vi",
            "show_log": False,
            "enable_mkldnn": False,
        },
        {
            "use_angle_cls": True,
            "lang": "vi",
            "enable_mkldnn": False,
        },
    ]


def test_collect_paddle_text_preserves_all_rec_texts():
    raw_result = {"rec_texts": ["LINE 1", "LINE 2", "LINE 3"]}

    assert ocr._collect_paddle_text(raw_result) == ["LINE 1", "LINE 2", "LINE 3"]


def test_extract_text_with_details_falls_back_to_easyocr(text_like_image, monkeypatch):
    def raise_missing_paddle():
        raise RuntimeError("paddle missing")

    fake_reader = FakeEasyOCRReader(results=[(None, "fallback text", 0.9)])
    monkeypatch.setattr(ocr, "get_paddleocr_reader", raise_missing_paddle)
    monkeypatch.setattr(ocr, "get_reader", lambda: fake_reader)

    result = ocr.extract_text_with_details(
        text_like_image,
        backend="paddleocr",
        fallback_backend="easyocr",
    )

    assert result.ok
    assert result.used_fallback is True
    assert result.backend == "easyocr"
    assert result.text == "fallback text"


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
