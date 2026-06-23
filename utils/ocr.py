from dataclasses import dataclass
import importlib
import importlib.util
import os
import sys

import numpy as np

try:
    import easyocr
except ImportError:
    easyocr = None

EASYOCR_BACKEND = "easyocr"
PADDLEOCR_BACKEND = "paddleocr"
SUPPORTED_BACKENDS = (EASYOCR_BACKEND, PADDLEOCR_BACKEND)

# Keep _reader for backwards-compatible tests/imports.
_reader = None
_paddle_reader = None


@dataclass(frozen=True)
class OCRResult:
    """
    Structured OCR response used by the Streamlit app.
    """

    text: str
    backend: str
    error: str = ""
    used_fallback: bool = False
    requested_backend: str = ""

    @property
    def ok(self):
        return not self.error


def normalize_backend(backend):
    """
    Normalize a user-facing backend value to an internal backend key.
    """
    if not backend:
        return EASYOCR_BACKEND

    normalized = str(backend).strip().lower()
    if normalized in ("easy", "easyocr", "baseline"):
        return EASYOCR_BACKEND
    if normalized in ("paddle", "paddleocr", "paddle_ocr"):
        return PADDLEOCR_BACKEND

    raise ValueError(f"Unsupported OCR backend: {backend}")


def _module_is_available(module_name):
    if module_name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def is_backend_available(backend):
    """
    Return whether the OCR backend can be imported in this environment.
    """
    backend = normalize_backend(backend)
    if backend == EASYOCR_BACKEND:
        return easyocr is not None or _module_is_available("easyocr")
    if backend == PADDLEOCR_BACKEND:
        return _module_is_available("paddleocr")
    return False


def get_backend_label(backend):
    backend = normalize_backend(backend)
    if backend == PADDLEOCR_BACKEND:
        return "PaddleOCR"
    return "EasyOCR"


def get_available_backends():
    """
    Return installed OCR backends in preference order.
    """
    return [backend for backend in SUPPORTED_BACKENDS if is_backend_available(backend)]


def get_easyocr_reader():
    """
    Get or initialize EasyOCR reader.

    Returns:
        EasyOCR Reader instance
    """
    global _reader, easyocr
    if _reader is None:
        if easyocr is None:
            easyocr = importlib.import_module("easyocr")
        _reader = easyocr.Reader(['en', 'vi'], gpu=False)
    return _reader


def get_reader():
    """
    Backwards-compatible EasyOCR reader accessor.

    Returns:
        EasyOCR Reader instance
    """
    return get_easyocr_reader()


def get_paddleocr_reader():
    """
    Get or initialize PaddleOCR reader if the optional dependency is installed.
    """
    global _paddle_reader
    if _paddle_reader is not None:
        return _paddle_reader

    # Work around PaddleOCR 3.x CPU failures on Windows/oneDNN by
    # disabling the PIR path before the module initializes.
    os.environ.setdefault("FLAGS_enable_pir_api", "0")
    paddleocr_module = importlib.import_module("paddleocr")
    PaddleOCR = paddleocr_module.PaddleOCR

    # PaddleOCR has changed constructor flags across versions. Try the
    # Vietnamese-friendly legacy config first, then fall back to minimal config.
    init_attempts = (
        {"use_angle_cls": True, "lang": "vi", "show_log": False, "enable_mkldnn": False},
        {"use_angle_cls": True, "lang": "vi", "enable_mkldnn": False},
        {"lang": "vi", "enable_mkldnn": False},
        {"enable_mkldnn": False},
    )
    last_error = None
    for kwargs in init_attempts:
        try:
            _paddle_reader = PaddleOCR(**kwargs)
            return _paddle_reader
        except (TypeError, ValueError) as exc:
            last_error = exc

    raise RuntimeError(f"Unable to initialize PaddleOCR: {last_error}")


def _image_for_easyocr(image):
    # EasyOCR expects RGB or grayscale.
    if len(image.shape) == 3:
        import cv2
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _extract_easyocr_text(image):
    reader = get_reader()
    results = reader.readtext(_image_for_easyocr(image))
    text_lines = [result[1] for result in results if len(result) > 1]
    return '\n'.join(text_lines)


def _collect_paddle_text(raw_result):
    """
    Collect recognized strings from both legacy and newer PaddleOCR result shapes.
    """
    if raw_result is None:
        return []

    if isinstance(raw_result, str):
        return [raw_result] if raw_result.strip() else []

    if isinstance(raw_result, dict):
        texts = []
        priority_keys = ("rec_texts", "texts", "text", "transcription", "label")
        for key in priority_keys:
            if key in raw_result:
                texts.extend(_collect_paddle_text(raw_result[key]))
        if texts:
            return texts

        ignored_suffixes = ("path", "paths", "file", "files")
        for key, value in raw_result.items():
            if str(key).lower().endswith(ignored_suffixes):
                continue
            texts.extend(_collect_paddle_text(value))
        return texts

    if isinstance(raw_result, np.ndarray):
        return []

    if isinstance(raw_result, (list, tuple)):
        if raw_result and all(isinstance(item, str) for item in raw_result):
            return [item for item in raw_result if item.strip()]

        # Legacy PaddleOCR line format: [box, (text, confidence)].
        if len(raw_result) >= 2:
            second = raw_result[1]
            if isinstance(second, str):
                return [second]
            if isinstance(second, (list, tuple)) and second and isinstance(second[0], str):
                return [second[0]]

        texts = []
        for item in raw_result:
            texts.extend(_collect_paddle_text(item))
        return texts

    return []


def _extract_paddleocr_text(image):
    reader = get_paddleocr_reader()

    if hasattr(reader, "ocr"):
        try:
            raw_result = reader.ocr(image, cls=True)
        except TypeError:
            raw_result = reader.ocr(image)
    elif hasattr(reader, "predict"):
        raw_result = reader.predict(image)
    else:
        raise RuntimeError("PaddleOCR reader has no supported OCR method")

    text_lines = _collect_paddle_text(raw_result)
    return '\n'.join(text_lines)


def extract_text_with_details(image, backend=EASYOCR_BACKEND, fallback_backend=None):
    """
    Extract text and return backend/error metadata for UI decisions.
    """
    requested_backend = normalize_backend(backend)
    fallback = normalize_backend(fallback_backend) if fallback_backend else None

    try:
        if requested_backend == PADDLEOCR_BACKEND:
            text = _extract_paddleocr_text(image)
        else:
            text = _extract_easyocr_text(image)

        return OCRResult(text=text, backend=requested_backend, requested_backend=requested_backend)
    except Exception as exc:
        primary_error = str(exc)
        if fallback and fallback != requested_backend:
            try:
                if fallback == PADDLEOCR_BACKEND:
                    text = _extract_paddleocr_text(image)
                else:
                    text = _extract_easyocr_text(image)
                return OCRResult(
                    text=text,
                    backend=fallback,
                    used_fallback=True,
                    requested_backend=requested_backend,
                )
            except Exception as fallback_exc:
                primary_error = f"{primary_error}; fallback {fallback} failed: {fallback_exc}"

        return OCRResult(
            text="",
            backend=requested_backend,
            requested_backend=requested_backend,
            error=primary_error,
        )


def extract_text(image, backend=EASYOCR_BACKEND):
    """
    Extract text from image using the selected OCR backend.

    Args:
        image: Input image (BGR or grayscale)
        backend: 'easyocr' or 'paddleocr'

    Returns:
        str: Extracted text (concatenated from all detected regions)
    """
    result = extract_text_with_details(image, backend=backend)
    if result.ok:
        return result.text
    return f"Error during OCR: {result.error}"


def _levenshtein_distance(left, right):
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (left_item != right_item)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def calculate_character_error_rate(ground_truth, extracted_text):
    """
    Calculate character error rate (CER). Lower is better.
    """
    if not ground_truth:
        return 0.0 if not extracted_text else 100.0

    gt_normalized = ''.join(ground_truth.lower().split())
    ext_normalized = ''.join((extracted_text or "").lower().split())
    if not gt_normalized:
        return 0.0 if not ext_normalized else 100.0

    distance = _levenshtein_distance(gt_normalized, ext_normalized)
    return (distance / len(gt_normalized)) * 100


def calculate_word_error_rate(ground_truth, extracted_text):
    """
    Calculate word error rate (WER). Lower is better.
    """
    gt_words = (ground_truth or "").lower().split()
    ext_words = (extracted_text or "").lower().split()
    if not gt_words:
        return 0.0 if not ext_words else 100.0

    distance = _levenshtein_distance(gt_words, ext_words)
    return (distance / len(gt_words)) * 100


def calculate_accuracy(ground_truth, extracted_text):
    """
    Calculate character-level accuracy between ground truth and extracted text.

    Args:
        ground_truth: Ground truth text
        extracted_text: OCR extracted text

    Returns:
        float: Accuracy percentage (0-100)
    """
    if not ground_truth or not extracted_text:
        return 0.0

    # Normalize texts (remove extra whitespace, lowercase)
    gt_normalized = ''.join(ground_truth.lower().split())
    ext_normalized = ''.join(extracted_text.lower().split())

    if len(gt_normalized) == 0:
        return 0.0

    distance = _levenshtein_distance(gt_normalized, ext_normalized)
    accuracy = (1 - (distance / len(gt_normalized))) * 100
    return max(0.0, min(100.0, accuracy))


def compare_ocr_results(original_text, enhanced_text):
    """
    Compare OCR results from original and enhanced images.

    Args:
        original_text: Text extracted from original image
        enhanced_text: Text extracted from enhanced image

    Returns:
        dict: Comparison statistics
    """
    original_chars = len(''.join(original_text.split()))
    enhanced_chars = len(''.join(enhanced_text.split()))

    original_lines = len(original_text.split('\n'))
    enhanced_lines = len(enhanced_text.split('\n'))

    return {
        'original_chars': original_chars,
        'enhanced_chars': enhanced_chars,
        'original_lines': original_lines,
        'enhanced_lines': enhanced_lines,
        'char_improvement': enhanced_chars - original_chars,
        'line_improvement': enhanced_lines - original_lines
    }
