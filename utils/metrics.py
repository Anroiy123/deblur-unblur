import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calculate_sharpness(image):
    """
    Calculate sharpness using Laplacian variance.

    Args:
        image: Input image (BGR or grayscale)

    Returns:
        float: Laplacian variance (higher = sharper)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()

    return variance


def calculate_psnr(original, enhanced):
    """
    Calculate Peak Signal-to-Noise Ratio between two images.

    Args:
        original: Reference image
        enhanced: Enhanced image

    Returns:
        float: PSNR value in dB
    """
    return peak_signal_noise_ratio(original, enhanced)


def calculate_ssim(original, enhanced):
    """
    Calculate Structural Similarity Index between two images.

    Args:
        original: Reference image
        enhanced: Enhanced image

    Returns:
        float: SSIM value (0-1 scale)
    """
    if len(original.shape) == 3:
        return structural_similarity(original, enhanced, channel_axis=2)
    else:
        return structural_similarity(original, enhanced)
