import numpy as np
import pytest

from src.utils import resize_image


def test_resize_maintains_aspect_ratio():
    image = np.zeros((200, 400, 3), dtype=np.uint8)
    resized = resize_image(image, width=200)
    assert resized.shape[1] == 200
    assert resized.shape[0] == 100


def test_resize_default_width():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    resized = resize_image(image)
    assert resized.shape[1] == 640
