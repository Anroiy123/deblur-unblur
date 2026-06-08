import sys
import types

import cv2
import numpy as np
import pytest


class FakeEasyOCRReader:
    def __init__(self, results=None, should_raise=False):
        self.results = results or []
        self.should_raise = should_raise
        self.calls = []

    def readtext(self, image):
        self.calls.append(image)
        if self.should_raise:
            raise RuntimeError("mock OCR failure")
        return self.results


@pytest.fixture(autouse=True)
def stub_easyocr_module(monkeypatch):
    fake_module = types.SimpleNamespace(Reader=lambda *args, **kwargs: FakeEasyOCRReader())
    monkeypatch.setitem(sys.modules, "easyocr", fake_module)

    import utils.ocr as ocr

    monkeypatch.setattr(ocr, "_reader", None)
    monkeypatch.setattr(ocr, "easyocr", fake_module)
    yield
    monkeypatch.setattr(ocr, "_reader", None)


@pytest.fixture
def clean_image():
    image = np.full((160, 240, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (30, 30), (210, 130), (40, 40, 40), 2)
    cv2.line(image, (40, 60), (200, 60), (70, 70, 70), 2)
    cv2.line(image, (40, 90), (180, 90), (70, 70, 70), 2)
    return image


@pytest.fixture
def blurred_image(clean_image):
    return cv2.GaussianBlur(clean_image, (11, 11), 0)


@pytest.fixture
def noisy_image(clean_image):
    noise = np.random.default_rng(123).normal(0, 20, clean_image.shape)
    noisy = np.clip(clean_image.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


@pytest.fixture
def text_like_image():
    image = np.full((180, 320, 3), 255, dtype=np.uint8)
    cv2.putText(image, "ID CARD", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, "123456789", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, "NGUYEN VAN A", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return image


@pytest.fixture
def document_like_image(text_like_image):
    image = text_like_image.copy()
    cv2.rectangle(image, (10, 10), (310, 170), (120, 120, 120), 2)
    return image


@pytest.fixture
def face_like_image():
    image = np.full((220, 320, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (300, 200), (180, 180, 180), 2)
    center = (110, 110)
    cv2.circle(image, center, 45, (210, 190, 170), -1)
    cv2.circle(image, (95, 98), 5, (0, 0, 0), -1)
    cv2.circle(image, (125, 98), 5, (0, 0, 0), -1)
    cv2.ellipse(image, (110, 125), (18, 10), 0, 0, 180, (60, 60, 120), 2)
    return image


@pytest.fixture
def card_like_image(document_like_image):
    canvas = np.full((300, 420, 3), 240, dtype=np.uint8)
    cv2.rectangle(canvas, (60, 70), (360, 240), (255, 255, 255), -1)
    cv2.rectangle(canvas, (60, 70), (360, 240), (30, 30, 30), 3)
    resized = cv2.resize(document_like_image, (280, 150))
    canvas[80:230, 70:350] = resized
    return canvas
