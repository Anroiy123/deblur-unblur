from tests.conftest import FakeEasyOCRReader
from utils.enhancement import enhance_image
from utils.metrics import calculate_sharpness
from utils.ocr import compare_ocr_results, extract_text
from utils.text_processing import preprocess_for_ocr


def test_enhancement_pipeline_preserves_image_contract(blurred_image):
    enhanced = enhance_image(blurred_image, use_deconvolution=False)
    assert enhanced.shape == blurred_image.shape
    assert enhanced.dtype == blurred_image.dtype


def test_enhancement_pipeline_can_improve_sharpness_on_synthetic_input(blurred_image):
    enhanced = enhance_image(blurred_image, use_deconvolution=False)
    assert calculate_sharpness(enhanced) >= calculate_sharpness(blurred_image)


def test_preprocess_plus_mocked_ocr_flow_returns_deterministic_text(text_like_image, monkeypatch):
    processed = preprocess_for_ocr(text_like_image)
    fake_reader = FakeEasyOCRReader(results=[(None, "ID CARD", 0.99), (None, "123456789", 0.99)])

    import utils.ocr as ocr

    monkeypatch.setattr(ocr, "get_reader", lambda: fake_reader)
    extracted = extract_text(processed)
    assert extracted == "ID CARD\n123456789"


def test_compare_ocr_results_shows_non_negative_improvement_for_enhanced_text():
    comparison = compare_ocr_results("ID\n12", "ID CARD\n123456789")
    assert comparison["char_improvement"] >= 0
    assert comparison["enhanced_chars"] >= comparison["original_chars"]
