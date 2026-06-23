import numpy as np

from utils import detection


def test_detect_card_edges_returns_binary_map(card_like_image):
    edges = detection.detect_card_edges(card_like_image)
    assert edges.shape == card_like_image.shape[:2]
    assert edges.dtype == np.uint8
    assert edges.sum() > 0


def test_find_card_contour_returns_quadrilateral(card_like_image):
    edges = detection.detect_card_edges(card_like_image)
    contour = detection.find_card_contour(edges, card_like_image.shape)
    assert contour is not None
    assert contour.shape[0] == 4


def test_find_card_contour_returns_none_when_no_contours():
    edges = np.zeros((100, 100), dtype=np.uint8)
    assert detection.find_card_contour(edges, (100, 100, 3)) is None


def test_order_points_returns_consistent_order():
    pts = np.array([[100, 100], [20, 20], [20, 100], [100, 20]], dtype=np.float32)
    ordered = detection.order_points(pts)
    assert np.array_equal(ordered[0], [20, 20])
    assert np.array_equal(ordered[1], [100, 20])
    assert np.array_equal(ordered[2], [100, 100])
    assert np.array_equal(ordered[3], [20, 100])


def test_order_points_keeps_four_unique_points_when_sums_tie():
    pts = np.array([[0, 1], [1, 0], [2, 1], [1, 2]], dtype=np.float32)

    ordered = detection.order_points(pts)

    assert len(np.unique(ordered, axis=0)) == 4
    assert np.array_equal(ordered[0], [0, 1])
    assert np.array_equal(ordered[1], [1, 0])
    assert np.array_equal(ordered[2], [2, 1])
    assert np.array_equal(ordered[3], [1, 2])


def test_apply_perspective_transform_returns_non_empty_crop(card_like_image):
    contour = np.array([[[60, 70]], [[360, 70]], [[360, 240]], [[60, 240]]], dtype=np.int32)
    warped = detection.apply_perspective_transform(card_like_image, contour)
    assert warped.size > 0
    assert warped.ndim == 3


def test_is_axis_aligned_rect_detects_aligned_rectangle():
    contour = np.array([[[10, 10]], [[110, 10]], [[110, 60]], [[10, 60]]], dtype=np.int32)
    assert detection.is_axis_aligned_rect(contour)


def test_crop_bounding_rect_returns_expected_size(card_like_image):
    contour = np.array([[[60, 70]], [[360, 70]], [[360, 240]], [[60, 240]]], dtype=np.int32)
    cropped = detection.crop_bounding_rect(card_like_image, contour)
    assert cropped.shape[:2] == (171, 301)


def test_detect_and_extract_card_success(card_like_image):
    card, success, message = detection.detect_and_extract_card(card_like_image)
    assert success is True
    assert card.size > 0
    assert "successfully" in message.lower()


def test_detect_and_extract_card_fallback_when_missing_contour():
    image = np.full((160, 240, 3), 255, dtype=np.uint8)
    card, success, message = detection.detect_and_extract_card(image)
    assert success is False
    assert np.array_equal(card, image)
    assert "failed" in message.lower()
