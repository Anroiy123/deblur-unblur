from dataclasses import dataclass
import importlib


try:
    import easyocr
except ImportError:
    easyocr = None


EASYOCR_BACKEND = "easyocr"
SUPPORTED_BACKENDS = (EASYOCR_BACKEND,)

# Keep _reader for backwards-compatible tests/imports.
_reader = None


@dataclass(frozen=True)
class OCRResult:
    text: str
    backend: str
    error: str = ""
    used_fallback: bool = False
    requested_backend: str = ""

    @property
    def ok(self):
        return not self.error


def normalize_backend(backend):
    if not backend:
        return EASYOCR_BACKEND

    normalized = str(backend).strip().lower()
    if normalized in ("easy", "easyocr", "baseline"):
        return EASYOCR_BACKEND

    raise ValueError(f"Unsupported OCR backend: {backend}")


def is_backend_available(backend):
    backend = normalize_backend(backend)
    return backend == EASYOCR_BACKEND and (
        easyocr is not None or importlib.util.find_spec("easyocr") is not None
    )


def get_backend_label(backend):
    normalize_backend(backend)
    return "EasyOCR"


def get_available_backends():
    return [backend for backend in SUPPORTED_BACKENDS if is_backend_available(backend)]


def get_easyocr_reader():
    global _reader, easyocr
    if _reader is None:
        if easyocr is None:
            easyocr = importlib.import_module("easyocr")
        _reader = easyocr.Reader(["en", "vi"], gpu=False)
    return _reader


def get_reader():
    return get_easyocr_reader()


def _image_for_easyocr(image):
    if len(image.shape) == 3:
        import cv2

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _extract_easyocr_text(image):
    reader = get_reader()
    results = reader.readtext(_image_for_easyocr(image))
    text_lines = [result[1] for result in results if len(result) > 1]
    return "\n".join(text_lines)


def extract_text_with_details(image, backend=EASYOCR_BACKEND, fallback_backend=None):
    try:
        requested_backend = normalize_backend(backend)
        if fallback_backend is not None:
            normalize_backend(fallback_backend)
    except ValueError as exc:
        requested_backend = str(backend).strip().lower() if backend else EASYOCR_BACKEND
        return OCRResult(
            text="",
            backend=requested_backend,
            requested_backend=requested_backend,
            error=str(exc),
        )

    try:
        text = _extract_easyocr_text(image)
        return OCRResult(
            text=text,
            backend=requested_backend,
            requested_backend=requested_backend,
        )
    except Exception as exc:
        return OCRResult(
            text="",
            backend=requested_backend,
            requested_backend=requested_backend,
            error=str(exc),
        )


def extract_text(image, backend=EASYOCR_BACKEND):
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
    if not ground_truth:
        return 0.0 if not extracted_text else 100.0

    gt_normalized = "".join(ground_truth.lower().split())
    ext_normalized = "".join((extracted_text or "").lower().split())
    if not gt_normalized:
        return 0.0 if not ext_normalized else 100.0

    distance = _levenshtein_distance(gt_normalized, ext_normalized)
    return (distance / len(gt_normalized)) * 100


def calculate_word_error_rate(ground_truth, extracted_text):
    gt_words = (ground_truth or "").lower().split()
    ext_words = (extracted_text or "").lower().split()
    if not gt_words:
        return 0.0 if not ext_words else 100.0

    distance = _levenshtein_distance(gt_words, ext_words)
    return (distance / len(gt_words)) * 100


def calculate_accuracy(ground_truth, extracted_text):
    if not ground_truth or not extracted_text:
        return 0.0

    gt_normalized = "".join(ground_truth.lower().split())
    ext_normalized = "".join(extracted_text.lower().split())

    if len(gt_normalized) == 0:
        return 0.0

    distance = _levenshtein_distance(gt_normalized, ext_normalized)
    accuracy = (1 - (distance / len(gt_normalized))) * 100
    return max(0.0, min(100.0, accuracy))


def compare_ocr_results(original_text, enhanced_text):
    original_chars = len("".join(original_text.split()))
    enhanced_chars = len("".join(enhanced_text.split()))

    original_lines = len(original_text.split("\n"))
    enhanced_lines = len(enhanced_text.split("\n"))

    return {
        "original_chars": original_chars,
        "enhanced_chars": enhanced_chars,
        "original_lines": original_lines,
        "enhanced_lines": enhanced_lines,
        "char_improvement": enhanced_chars - original_chars,
        "line_improvement": enhanced_lines - original_lines,
    }
