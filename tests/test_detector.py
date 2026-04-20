import numpy as np

from src.detector import Detector


def test_detect_returns_list():
    detector = Detector(model_name="yolov8n.pt", confidence=0.5)
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(blank)
    assert isinstance(detections, list)


def test_detect_and_annotate_returns_image_and_list():
    detector = Detector(model_name="yolov8n.pt", confidence=0.5)
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated, detections = detector.detect_and_annotate(blank)
    assert annotated.shape == blank.shape
    assert isinstance(detections, list)
