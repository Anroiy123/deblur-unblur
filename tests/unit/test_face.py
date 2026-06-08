import cv2
import numpy as np

from utils import face


def test_detect_face_returns_none_when_classifier_finds_nothing(clean_image, monkeypatch):
    class DummyCascade:
        def detectMultiScale(self, *args, **kwargs):
            return ()

    monkeypatch.setattr(cv2, "CascadeClassifier", lambda *_args, **_kwargs: DummyCascade())
    assert face.detect_face(clean_image) is None


def test_detect_face_returns_largest_face(clean_image, monkeypatch):
    class DummyCascade:
        def detectMultiScale(self, *args, **kwargs):
            return np.array([[1, 2, 10, 10], [5, 5, 20, 20]])

    monkeypatch.setattr(cv2, "CascadeClassifier", lambda *_args, **_kwargs: DummyCascade())
    assert face.detect_face(clean_image) == (5, 5, 20, 20)


def test_extract_face_region_returns_requested_crop(face_like_image):
    region = face.extract_face_region(face_like_image, (65, 65, 90, 90))
    assert region.shape[:2] == (90, 90)


def test_enhance_face_preserves_shape(face_like_image):
    region = face.extract_face_region(face_like_image, (65, 65, 90, 90))
    enhanced = face.enhance_face(region)
    assert enhanced.shape == region.shape


def test_blend_face_back_replaces_target_region(face_like_image):
    bbox = (65, 65, 90, 90)
    replacement = np.zeros((90, 90, 3), dtype=np.uint8)
    blended = face.blend_face_back(face_like_image, replacement, bbox)
    assert blended.shape == face_like_image.shape
    assert not np.array_equal(blended[70:150, 70:150], face_like_image[70:150, 70:150])


def test_enhance_face_in_card_returns_fallback_when_no_face(clean_image, monkeypatch):
    monkeypatch.setattr(face, "detect_face", lambda _image: None)
    result, found, original_face, enhanced_face, message = face.enhance_face_in_card(clean_image)
    assert found is False
    assert original_face is None
    assert enhanced_face is None
    assert np.array_equal(result, clean_image)
    assert "not detected" in message.lower()


def test_enhance_face_in_card_runs_full_pipeline(face_like_image, monkeypatch):
    monkeypatch.setattr(face, "detect_face", lambda _image: (65, 65, 90, 90))
    result, found, original_face, enhanced_face, message = face.enhance_face_in_card(face_like_image)
    assert found is True
    assert original_face.shape[:2] == (90, 90)
    assert enhanced_face.shape[:2] == (90, 90)
    assert result.shape == face_like_image.shape
    assert "successfully" in message.lower()
