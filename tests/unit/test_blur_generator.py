import numpy as np

from utils import blur_generator
from utils.metrics import calculate_sharpness


def test_apply_gaussian_blur_preserves_shape(clean_image):
    result = blur_generator.apply_gaussian_blur(clean_image, kernel_size=8)
    assert result.shape == clean_image.shape


def test_apply_gaussian_blur_normalizes_even_kernel_to_odd(clean_image):
    even = blur_generator.apply_gaussian_blur(clean_image, kernel_size=8)
    odd = blur_generator.apply_gaussian_blur(clean_image, kernel_size=9)
    assert np.array_equal(even, odd)


def test_apply_motion_blur_preserves_shape(clean_image):
    result = blur_generator.apply_motion_blur(clean_image, kernel_size=9, angle=30)
    assert result.shape == clean_image.shape


def test_apply_defocus_blur_preserves_shape(clean_image):
    result = blur_generator.apply_defocus_blur(clean_image, radius=4)
    assert result.shape == clean_image.shape


def test_generated_blurs_reduce_sharpness(clean_image):
    blurred = blur_generator.apply_gaussian_blur(clean_image, kernel_size=15)
    assert calculate_sharpness(blurred) < calculate_sharpness(clean_image)


def test_generate_blurred_versions_returns_requested_keys(clean_image):
    result = blur_generator.generate_blurred_versions(clean_image, ["gaussian", "motion", "defocus"])
    assert set(result) == {"gaussian", "motion", "defocus"}


def test_generate_blurred_versions_can_select_subset(clean_image):
    result = blur_generator.generate_blurred_versions(clean_image, ["motion"])
    assert set(result) == {"motion"}
