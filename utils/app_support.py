import hashlib
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG"}
PROCESSING_RESULT_KEY = "processing_result"
PROCESSING_PIPELINE_VERSION = "ocr-pipeline-v2"


class ImageValidationError(ValueError):
    pass


def _uploaded_bytes(uploaded_file):
    if hasattr(uploaded_file, "getvalue"):
        return uploaded_file.getvalue()

    current_position = uploaded_file.tell()
    uploaded_file.seek(0)
    content = uploaded_file.read()
    uploaded_file.seek(current_position)
    return content


def load_uploaded_image(
    uploaded_file,
    max_bytes=MAX_UPLOAD_BYTES,
    max_pixels=MAX_IMAGE_PIXELS,
):
    content = _uploaded_bytes(uploaded_file)
    if len(content) > max_bytes:
        raise ImageValidationError("Tệp ảnh vượt quá giới hạn 10 MB.")

    try:
        with Image.open(BytesIO(content)) as probe:
            if probe.format not in SUPPORTED_IMAGE_FORMATS:
                raise ImageValidationError("Định dạng ảnh không hợp lệ; chỉ hỗ trợ JPG và PNG.")
            width, height = probe.size
            if width <= 0 or height <= 0 or width * height > max_pixels:
                raise ImageValidationError(
                    "Ảnh có kích thước pixel quá lớn hoặc không hợp lệ."
                )
            probe.verify()

        with Image.open(BytesIO(content)) as source:
            source.load()
            return source.convert("RGB")
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError("Nội dung tệp ảnh không hợp lệ.") from exc


def image_to_bgr(image):
    rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def make_processing_signature(
    image_bytes,
    settings,
    version=PROCESSING_PIPELINE_VERSION,
):
    digest = hashlib.sha256()
    digest.update(str(version).encode("utf-8"))
    digest.update(image_bytes)
    digest.update(repr(tuple(settings)).encode("utf-8"))
    return digest.hexdigest()


def calculate_percentage_change(original_value, new_value):
    if original_value <= 0:
        return None
    return ((new_value - original_value) / original_value) * 100


def resolve_restoration_backend(choice):
    normalized = (choice or "").strip().lower()
    if normalized == "auto":
        return "docres"
    if normalized == "opencv":
        return "opencv"
    raise ValueError(f"Unsupported restoration choice: {choice}")


def store_processing_result(state, signature, result):
    state[PROCESSING_RESULT_KEY] = {
        "signature": signature,
        "result": result,
    }


def get_processing_result(state, signature):
    stored = state.get(PROCESSING_RESULT_KEY)
    if not stored or stored.get("signature") != signature:
        return None
    return stored.get("result")
