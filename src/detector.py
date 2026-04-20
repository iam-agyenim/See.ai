from ultralytics import YOLO
import cv2
import numpy as np


class Detector:
    """Object detector powered by YOLOv8."""

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.5):
        self.model = YOLO(model_name)
        self.confidence = confidence

    def detect(self, image: np.ndarray) -> list[dict]:
        """Run detection on a single image.

        Args:
            image: BGR image as a NumPy array.

        Returns:
            A list of detections, each containing:
                - label: class name
                - confidence: detection confidence
                - box: [x1, y1, x2, y2] bounding box coordinates
        """
        results = self.model(image, conf=self.confidence, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                detections.append({
                    "label": label,
                    "confidence": round(conf, 3),
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                })
        return detections

    def detect_and_annotate(self, image: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """Run detection and draw bounding boxes on the image.

        Args:
            image: BGR image as a NumPy array.

        Returns:
            A tuple of (annotated_image, detections).
        """
        detections = self.detect(image)
        annotated = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            label = f"{det['label']} {det['confidence']:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(
                annotated, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1,
            )
        return annotated, detections
