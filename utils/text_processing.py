import cv2
import numpy as np

OCR_PREPROCESS_AUTO = "auto"
OCR_PREPROCESS_PRESERVE = "preserve"
OCR_PREPROCESS_GRAYSCALE = "grayscale"
OCR_PREPROCESS_THRESHOLD = "threshold"


def convert_to_grayscale(image):
    """
    Convert image to grayscale for text processing.

    Args:
        image: Input image (BGR)

    Returns:
        Grayscale image
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def apply_text_clahe(image, clip_limit=3.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE tuned for text visibility.

    Args:
        image: Input grayscale image
        clip_limit: Threshold for contrast limiting (higher for text)
        tile_grid_size: Size of grid for histogram equalization

    Returns:
        Contrast-enhanced grayscale image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def sharpen_text(image):
    """
    Apply text-focused sharpening before thresholding.

    Args:
        image: Input grayscale image

    Returns:
        Sharpened grayscale image
    """
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)


def adaptive_threshold(image, method='gaussian', block_size=11, C=2):
    """
    Apply adaptive thresholding to binarize text.

    Args:
        image: Input grayscale image
        method: 'gaussian' or 'mean'
        block_size: Size of pixel neighborhood (must be odd)
        C: Constant subtracted from weighted mean

    Returns:
        Binary image
    """
    if method == 'gaussian':
        thresh_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    else:
        thresh_method = cv2.ADAPTIVE_THRESH_MEAN_C

    binary = cv2.adaptiveThreshold(
        image,
        255,
        thresh_method,
        cv2.THRESH_BINARY,
        block_size,
        C
    )

    return binary


def morphological_operations(image, operation='both', kernel_size=(2, 2)):
    """
    Apply morphological operations to clean text.

    Args:
        image: Input binary image
        operation: 'opening', 'closing', or 'both'
        kernel_size: Size of morphological kernel

    Returns:
        Cleaned binary image
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    if operation == 'opening':
        # Remove noise
        result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif operation == 'closing':
        # Fill gaps
        result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    else:  # both
        # Remove noise then fill gaps
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        result = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    return result


def preprocess_for_ocr(image):
    """
    Complete text preprocessing pipeline for OCR.

    Args:
        image: Input image (BGR or grayscale)

    Returns:
        Preprocessed image ready for OCR
    """
    # Convert to grayscale
    gray = convert_to_grayscale(image)

    # Enhance contrast
    enhanced = apply_text_clahe(gray)

    # Sharpen text before thresholding
    sharpened = sharpen_text(enhanced)

    # Apply adaptive thresholding
    binary = adaptive_threshold(sharpened)

    # Clean with morphological operations
    cleaned = morphological_operations(binary)

    return cleaned


def preserve_for_ocr(image):
    """
    Keep the image close to detector input for modern OCR engines.

    Modern OCR engines such as PaddleOCR usually perform their own detection
    and normalization, so aggressive thresholding can remove faint strokes.
    """
    return image.copy()


def prepare_image_for_ocr(image, backend="easyocr", profile=OCR_PREPROCESS_AUTO):
    """
    Prepare an image for OCR using a backend-aware policy.

    Args:
        image: Input image (BGR or grayscale)
        backend: OCR backend key, e.g. 'easyocr' or 'paddleocr'
        profile: 'auto', 'preserve', 'grayscale', or 'threshold'

    Returns:
        Image ready to send to the OCR backend.
    """
    profile = (profile or OCR_PREPROCESS_AUTO).strip().lower()
    backend = (backend or "easyocr").strip().lower()

    if profile == OCR_PREPROCESS_THRESHOLD:
        return preprocess_for_ocr(image)
    if profile == OCR_PREPROCESS_GRAYSCALE:
        return convert_to_grayscale(image)
    if profile == OCR_PREPROCESS_PRESERVE:
        return preserve_for_ocr(image)
    if profile != OCR_PREPROCESS_AUTO:
        raise ValueError(f"Unsupported OCR preprocessing profile: {profile}")

    if backend in ("paddleocr", "paddle", "paddle_ocr"):
        return preserve_for_ocr(image)
    return preprocess_for_ocr(image)
