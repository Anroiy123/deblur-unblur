import cv2
import numpy as np
from scipy.signal import convolve2d


def bilateral_denoise(image, d=9, sigma_color=75, sigma_space=75):
    """
    Apply bilateral filter for edge-preserving denoising.

    Args:
        image: Input image (BGR)
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in color space
        sigma_space: Filter sigma in coordinate space

    Returns:
        Denoised image
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Args:
        image: Input image (BGR)
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization

    Returns:
        Contrast-enhanced image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)

    # Merge channels and convert back to BGR
    lab_clahe = cv2.merge([l_clahe, a, b])
    enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    return enhanced


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.5, threshold=0):
    """
    Apply unsharp masking for sharpening.

    Args:
        image: Input image (BGR)
        kernel_size: Size of Gaussian kernel
        sigma: Standard deviation for Gaussian kernel
        amount: Strength of sharpening
        threshold: Minimum brightness change required

    Returns:
        Sharpened image
    """
    # Create blurred version
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)

    # Calculate sharpened image
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)

    # Apply threshold if specified
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        sharpened = np.where(low_contrast_mask, image, sharpened)

    return np.clip(sharpened, 0, 255).astype(np.uint8)


def wiener_deconvolution(image, kernel_size=5, noise_variance=0.01):
    """
    Apply Wiener deconvolution for blur reduction (simplified version).

    Args:
        image: Input image (BGR)
        kernel_size: Size of estimated blur kernel
        noise_variance: Estimated noise variance

    Returns:
        Deconvolved image
    """
    # Convert to grayscale for processing
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Create motion blur kernel (simplified)
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size

    # Apply deconvolution using Richardson-Lucy algorithm
    deconvolved = gray.astype(float)
    for _ in range(10):  # 10 iterations
        conv = convolve2d(deconvolved, kernel, mode='same', boundary='symm')
        relative_blur = gray / (conv + 1e-10)
        deconvolved = deconvolved * convolve2d(relative_blur, kernel, mode='same', boundary='symm')

    deconvolved = np.clip(deconvolved, 0, 255).astype(np.uint8)

    # Convert back to BGR if needed
    if len(image.shape) == 3:
        deconvolved = cv2.cvtColor(deconvolved, cv2.COLOR_GRAY2BGR)

    return deconvolved


def enhance_image(image, use_deconvolution=False):
    """
    Complete enhancement pipeline combining multiple techniques.

    Args:
        image: Input image (BGR)
        use_deconvolution: Whether to apply Wiener deconvolution

    Returns:
        Enhanced image
    """
    # Step 1: Denoise
    denoised = bilateral_denoise(image)

    # Step 2: Enhance contrast
    contrast_enhanced = apply_clahe(denoised)

    # Step 3: Sharpen
    sharpened = unsharp_mask(contrast_enhanced)

    # Step 4: Optional deconvolution
    if use_deconvolution:
        result = wiener_deconvolution(sharpened)
    else:
        result = sharpened

    return result
