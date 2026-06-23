from io import BytesIO

from PIL import Image
import pytest

from utils.app_support import (
    ImageValidationError,
    PROCESSING_PIPELINE_VERSION,
    calculate_percentage_change,
    image_to_bgr,
    get_processing_result,
    load_uploaded_image,
    make_processing_signature,
    resolve_restoration_backend,
    store_processing_result,
)


def make_image_file(size=(32, 24), image_format="PNG"):
    buffer = BytesIO()
    Image.new("RGB", size, color=(10, 20, 30)).save(buffer, format=image_format)
    buffer.seek(0)
    buffer.name = f"sample.{image_format.lower()}"
    buffer.type = f"image/{image_format.lower()}"
    buffer.size = len(buffer.getvalue())
    return buffer


def test_load_uploaded_image_accepts_valid_png():
    image = load_uploaded_image(make_image_file())

    assert image.size == (32, 24)
    assert image.mode == "RGB"


def test_load_uploaded_image_rejects_files_over_limit():
    uploaded = make_image_file()

    with pytest.raises(ImageValidationError, match="10 MB"):
        load_uploaded_image(uploaded, max_bytes=uploaded.size - 1)


def test_load_uploaded_image_rejects_excessive_pixel_count():
    uploaded = make_image_file(size=(20, 20))

    with pytest.raises(ImageValidationError, match="kích thước pixel"):
        load_uploaded_image(uploaded, max_pixels=399)


def test_load_uploaded_image_rejects_non_image_content():
    uploaded = BytesIO(b"not an image")
    uploaded.name = "fake.png"
    uploaded.type = "image/png"
    uploaded.size = len(uploaded.getvalue())

    with pytest.raises(ImageValidationError, match="không hợp lệ"):
        load_uploaded_image(uploaded)


def test_image_to_bgr_normalizes_palette_images():
    image = Image.new("P", (10, 8))

    result = image_to_bgr(image)

    assert result.shape == (8, 10, 3)


def test_processing_signature_changes_with_content_or_settings():
    first = make_processing_signature(b"one", ("natural", "opencv"))
    second = make_processing_signature(b"two", ("natural", "opencv"))
    third = make_processing_signature(b"one", ("natural", "docres"))

    assert len({first, second, third}) == 3


def test_processing_signature_changes_with_pipeline_version():
    current = make_processing_signature(b"one", ("natural", "opencv"))
    next_version = make_processing_signature(
        b"one",
        ("natural", "opencv"),
        version=f"{PROCESSING_PIPELINE_VERSION}-next",
    )

    assert current != next_version


def test_percentage_change_returns_none_for_zero_baseline():
    assert calculate_percentage_change(0.0, 10.0) is None


def test_percentage_change_calculates_normal_case():
    assert calculate_percentage_change(10.0, 15.0) == 50.0


def test_processing_result_persists_only_for_matching_signature():
    state = {}
    result = {"enhanced": "image"}

    store_processing_result(state, "signature-a", result)

    assert get_processing_result(state, "signature-a") is result
    assert get_processing_result(state, "signature-b") is None


def test_resolve_restoration_backend_prefers_docres_for_auto():
    assert resolve_restoration_backend("auto") == "auto"


def test_resolve_restoration_backend_allows_docres_override():
    assert resolve_restoration_backend("docres") == "docres"


def test_resolve_restoration_backend_allows_restormer_override():
    assert resolve_restoration_backend("restormer") == "restormer"


def test_resolve_restoration_backend_allows_nafnet_override():
    assert resolve_restoration_backend("nafnet") == "nafnet"


def test_resolve_restoration_backend_allows_opencv_override():
    assert resolve_restoration_backend("opencv") == "opencv"


def test_resolve_restoration_backend_rejects_unknown_choice():
    with pytest.raises(ValueError, match="restoration choice"):
        resolve_restoration_backend("unknown")
