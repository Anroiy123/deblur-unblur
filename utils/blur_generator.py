import cv2
import numpy as np


def apply_gaussian_blur(image, kernel_size=15, sigma=0):
    """
    Apply Gaussian blur to simulate out-of-focus blur.

    Args:
        image: Input image (BGR)
        kernel_size: Size of Gaussian kernel (must be odd)
        sigma: Standard deviation (0 = auto-calculate)

    Returns:
        Blurred image
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return blurred


def apply_motion_blur(image, kernel_size=15, angle=0):
    """
    Apply motion blur to simulate camera shake.

    Args:
        image: Input image (BGR)
        kernel_size: Length of motion blur
        angle: Angle of motion in degrees

    Returns:
        Motion blurred image
    """
    # Create motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size

    # Rotate kernel to specified angle
    M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))

    # Apply motion blur
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred


def apply_defocus_blur(image, radius=10):
    """
    Apply defocus blur to simulate lens defocus.

    Args:
        image: Input image (BGR)
        radius: Radius of defocus disk

    Returns:
        Defocus blurred image
    """
    # Create circular kernel for defocus
    kernel_size = 2 * radius + 1
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)

    # Create circular mask
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    kernel[mask] = 1

    # Normalize kernel
    kernel = kernel / kernel.sum()

    # Apply defocus blur
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred


def generate_blurred_versions(image, blur_types=['gaussian', 'motion', 'defocus']):
    """
    Generate multiple blurred versions of an image.

    Args:
        image: Input sharp image (BGR)
        blur_types: List of blur types to generate

    Returns:
        dict: Dictionary of blur_type -> blurred_image
    """
    results = {}

    if 'gaussian' in blur_types:
        results['gaussian'] = apply_gaussian_blur(image, kernel_size=15)

    if 'motion' in blur_types:
        results['motion'] = apply_motion_blur(image, kernel_size=20, angle=45)

    if 'defocus' in blur_types:
        results['defocus'] = apply_defocus_blur(image, radius=8)

    return results
