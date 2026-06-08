import cv2

from utils import metrics


def test_calculate_sharpness_higher_for_clean_than_blurred(clean_image, blurred_image):
    assert metrics.calculate_sharpness(clean_image) > metrics.calculate_sharpness(blurred_image)


def test_calculate_sharpness_accepts_grayscale(clean_image):
    gray = cv2.cvtColor(clean_image, cv2.COLOR_BGR2GRAY)
    assert metrics.calculate_sharpness(gray) >= 0


def test_calculate_psnr_identical_images_is_infinite(clean_image):
    result = metrics.calculate_psnr(clean_image, clean_image.copy())
    assert result == float("inf")


def test_calculate_psnr_decreases_for_blurred_image(clean_image, blurred_image):
    assert metrics.calculate_psnr(clean_image, blurred_image) < float("inf")


def test_calculate_ssim_identical_images_is_one(clean_image):
    assert metrics.calculate_ssim(clean_image, clean_image.copy()) == 1.0


def test_calculate_ssim_grayscale_supported(clean_image, blurred_image):
    gray_clean = cv2.cvtColor(clean_image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
    value = metrics.calculate_ssim(gray_clean, gray_blurred)
    assert 0 <= value < 1
