import easyocr
import numpy as np


# Initialize EasyOCR reader (lazy loading)
_reader = None


def get_reader():
    """
    Get or initialize EasyOCR reader.

    Returns:
        EasyOCR Reader instance
    """
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(['en', 'vi'], gpu=False)
    return _reader


def extract_text(image):
    """
    Extract text from image using EasyOCR.

    Args:
        image: Input image (BGR or grayscale)

    Returns:
        str: Extracted text (concatenated from all detected regions)
    """
    try:
        reader = get_reader()

        # EasyOCR expects RGB or grayscale
        if len(image.shape) == 3:
            # Convert BGR to RGB
            import cv2
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        # Perform OCR
        results = reader.readtext(image_rgb)

        # Extract text from results
        text_lines = [result[1] for result in results]
        extracted_text = '\n'.join(text_lines)

        return extracted_text

    except Exception as e:
        return f"Error during OCR: {str(e)}"


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

    # Calculate character-level accuracy using simple matching
    if len(gt_normalized) == 0:
        return 0.0

    matches = sum(1 for a, b in zip(gt_normalized, ext_normalized) if a == b)
    max_len = max(len(gt_normalized), len(ext_normalized))

    accuracy = (matches / max_len) * 100

    return accuracy


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
