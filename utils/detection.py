import logging

import cv2
import numpy as np


logger = logging.getLogger(__name__)


def detect_card_edges(image):
    """
    Detect card edges using Canny edge detection.

    Args:
        image: Input image (BGR)

    Returns:
        Edge map (binary image)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    return edges


def find_card_contour(edges, image_shape):
    """
    Find the largest quadrilateral contour (card boundary).

    Args:
        edges: Edge map from Canny detection
        image_shape: Shape of original image

    Returns:
        Contour points or None if not found
    """
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Look for quadrilateral in top contours
    for contour in contours[:10]:
        # Approximate contour to polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # Check if it's a quadrilateral and large enough
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            image_area = image_shape[0] * image_shape[1]

            # Card should be at least 10% of image area
            if area > 0.1 * image_area:
                return approx

    return None


def order_points(pts):
    """
    Order points in consistent order: top-left, top-right, bottom-right, bottom-left.

    Args:
        pts: Array of 4 points

    Returns:
        Ordered points
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]

    sums = ordered.sum(axis=1)
    min_sum = sums.min()
    start_candidates = np.flatnonzero(np.isclose(sums, min_sum))
    start = start_candidates[np.argmin(ordered[start_candidates, 0])]
    return np.roll(ordered, -start, axis=0).astype(np.float32)


def apply_perspective_transform(image, contour):
    """
    Apply perspective transformation to get rectangular card view.

    Args:
        image: Input image (BGR)
        contour: Card boundary contour (4 points)

    Returns:
        Transformed image (rectangular card)
    """
    # Reshape contour points
    pts = contour.reshape(4, 2)

    # Order points
    rect = order_points(pts)

    # Calculate width and height of card
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Destination points for perspective transform
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype=np.float32)

    # Calculate perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)

    # Apply perspective transform
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def is_axis_aligned_rect(contour, angle_tolerance=10.0):
    """
    Check if detected contour is close to an axis-aligned rectangle.

    Args:
        contour: Card boundary contour (4 points)
        angle_tolerance: Maximum allowed angle (degrees) from axis alignment

    Returns:
        bool: True if contour is near axis-aligned, False otherwise
    """
    pts = contour.reshape(4, 2).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    angle = rect[2]

    # OpenCV angle conventions differ across versions; 0 and +/-90 are aligned.
    normalized_angle = abs(angle) % 90
    axis_delta = min(normalized_angle, 90 - normalized_angle)

    return axis_delta <= angle_tolerance


def crop_bounding_rect(image, contour):
    """
    Crop axis-aligned bounding rectangle for detected contour.

    Args:
        image: Input image (BGR)
        contour: Card boundary contour

    Returns:
        Cropped image
    """
    x, y, w, h = cv2.boundingRect(contour)
    return image[y:y + h, x:x + w]


def detect_and_extract_card(image):
    """
    Complete pipeline to detect and extract card region.

    Args:
        image: Input image (BGR)

    Returns:
        tuple: (extracted_card, success_flag, message)
    """
    fallback_message = "Card detection failed, using original image"

    try:
        # Detect edges
        edges = detect_card_edges(image)

        # Find card contour
        contour = find_card_contour(edges, image.shape)

        if contour is None:
            logger.warning(fallback_message)
            return image, False, fallback_message

        # Skip perspective correction when card is already aligned
        if is_axis_aligned_rect(contour):
            card = crop_bounding_rect(image, contour)
        else:
            card = apply_perspective_transform(image, contour)

        return card, True, "Card detected and extracted successfully"

    except Exception as e:
        logger.warning(f"{fallback_message}: {str(e)}")
        return image, False, fallback_message
