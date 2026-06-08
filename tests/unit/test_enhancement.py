import numpy as np

from utils import enhancement
from utils.metrics import calculate_sharpness


def test_bilateral_denoise_preserves_shape_and_dtype(noisy_image):
    result = enhancement.bilateral_denoise(noisy_image)
    assert result.shape == noisy_image.shape
    assert result.dtype == noisy_image.dtype


def test_median_denoise_reduces_impulse_noise(clean_image):
    noisy = clean_image.copy()
    noisy[20, 20] = [0, 0, 0]
    result = enhancement.median_denoise(noisy, kernel_size=3)
    assert not np.array_equal(result[20, 20], noisy[20, 20])


def test_apply_clahe_changes_low_contrast_image(clean_image):
    low_contrast = np.full_like(clean_image, 120)
    low_contrast[40:120, 60:180] = 135
    result = enhancement.apply_clahe(low_contrast)
    assert result.shape == low_contrast.shape
    assert not np.array_equal(result, low_contrast)


def test_unsharp_mask_preserves_shape(clean_image):
    result = enhancement.unsharp_mask(clean_image)
    assert result.shape == clean_image.shape
    assert result.dtype == np.uint8


def test_unsharp_mask_threshold_can_preserve_low_contrast_regions():
    image = np.full((40, 40, 3), 128, dtype=np.uint8)
    result = enhancement.unsharp_mask(image, threshold=10)
    assert np.array_equal(result, image)


def test_wiener_deconvolution_returns_bgr_for_color_input(blurred_image):
    result = enhancement.wiener_deconvolution(blurred_image, kernel_size=3)
    assert result.shape == blurred_image.shape
    assert result.dtype == np.uint8


def test_wiener_deconvolution_preserves_color_channels_for_color_input(blurred_image):
    result = enhancement.wiener_deconvolution(blurred_image, kernel_size=3)
    assert not np.array_equal(result[:, :, 0], result[:, :, 1])


def test_wiener_deconvolution_returns_grayscale_for_grayscale_input(clean_image):
    gray = clean_image[:, :, 0]
    result = enhancement.wiener_deconvolution(gray, kernel_size=3)
    assert result.ndim == 2
    assert result.shape == gray.shape


def test_enhance_image_without_deconvolution_runs_pipeline(blurred_image):
    result = enhancement.enhance_image(blurred_image, use_deconvolution=False)
    assert result.shape == blurred_image.shape


def test_enhance_image_with_deconvolution_runs_pipeline(blurred_image):
    result = enhancement.enhance_image(blurred_image, use_deconvolution=True)
    assert result.shape == blurred_image.shape


def test_enhance_image_natural_mode_runs_pipeline(blurred_image):
    result = enhancement.enhance_image(blurred_image, mode='natural')
    assert result.shape == blurred_image.shape
    assert result.dtype == blurred_image.dtype


def test_enhance_image_improves_sharpness_on_blurred_fixture(blurred_image):
    result = enhancement.enhance_image(blurred_image, use_deconvolution=False)
    assert calculate_sharpness(result) >= calculate_sharpness(blurred_image)
