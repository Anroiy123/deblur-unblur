import cv2
import numpy as np


def detect_face(image):
    """
    Detect face region in image using Haar Cascade.

    Args:
        image: Input image (BGR)

    Returns:
        tuple: (x, y, w, h) of face bounding box, or None if not found
    """
    try:
        # Load Haar Cascade classifier for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            return None

        # Return the largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        return tuple(largest_face)

    except Exception as e:
        return None


def extract_face_region(image, face_bbox):
    """
    Extract face region from image.

    Args:
        image: Input image (BGR)
        face_bbox: Face bounding box (x, y, w, h)

    Returns:
        Face region image
    """
    x, y, w, h = face_bbox
    face_region = image[y:y+h, x:x+w].copy()
    return face_region


def enhance_face(face_region):
    """
    Apply face-specific enhancement.

    Args:
        face_region: Face region image (BGR)

    Returns:
        Enhanced face region
    """
    # Step 1: Denoise with bilateral filter
    denoised = cv2.bilateralFilter(face_region, 9, 75, 75)

    # Step 2: Apply CLAHE for contrast enhancement
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    lab_clahe = cv2.merge([l_clahe, a, b])
    contrast_enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # Step 3: Sharpen
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(contrast_enhanced, -1, kernel)

    return sharpened


def blend_face_back(image, enhanced_face, face_bbox):
    """
    Blend enhanced face back into card image seamlessly.

    Args:
        image: Original card image (BGR)
        enhanced_face: Enhanced face region
        face_bbox: Face bounding box (x, y, w, h)

    Returns:
        Image with enhanced face blended in
    """
    x, y, w, h = face_bbox

    # Resize enhanced face to match bounding box if needed
    if enhanced_face.shape[:2] != (h, w):
        enhanced_face = cv2.resize(enhanced_face, (w, h))

    # Create result image
    result = image.copy()

    # Create a mask for smooth blending
    mask = np.ones((h, w), dtype=np.float32)

    # Apply Gaussian blur to mask for smooth transition
    mask = cv2.GaussianBlur(mask, (21, 21), 11)
    mask = np.stack([mask] * 3, axis=2)

    # Blend face region
    result[y:y+h, x:x+w] = (
        enhanced_face * mask + result[y:y+h, x:x+w] * (1 - mask)
    ).astype(np.uint8)

    return result


def enhance_face_in_card(image):
    """
    Complete pipeline to detect and enhance face region in card.

    Args:
        image: Input card image (BGR)

    Returns:
        tuple: (enhanced_image, face_detected, original_face, enhanced_face, message)
    """
    # Detect face
    face_bbox = detect_face(image)

    if face_bbox is None:
        return image, False, None, None, "Face region not detected, skipping face enhancement"

    # Extract face region
    original_face = extract_face_region(image, face_bbox)

    # Enhance face
    enhanced_face = enhance_face(original_face)

    # Blend back into image
    result = blend_face_back(image, enhanced_face, face_bbox)

    return result, True, original_face, enhanced_face, "Face detected and enhanced successfully"
