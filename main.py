import argparse
import sys

import cv2

from src.camera import Camera
from src.detector import Detector
from src.utils import load_image, save_image


def run_image(path: str, model: str, confidence: float, output: str | None) -> None:
    """Run detection on a single image."""
    detector = Detector(model_name=model, confidence=confidence)
    image = load_image(path)
    annotated, detections = detector.detect_and_annotate(image)

    for det in detections:
        print(f"  {det['label']}: {det['confidence']:.2f}  {det['box']}")

    if output:
        save_image(output, annotated)
        print(f"\nSaved annotated image to {output}")
    else:
        cv2.imshow("See.ai - Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_camera(source: int | str, model: str, confidence: float) -> None:
    """Run real-time detection on a camera feed."""
    detector = Detector(model_name=model, confidence=confidence)

    with Camera(source) as cam:
        print("Press 'q' to quit.")
        while True:
            frame = cam.read()
            if frame is None:
                break

            annotated, detections = detector.detect_and_annotate(frame)
            cv2.imshow("See.ai - Live Detection", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="See.ai — AI that sees and detects things",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # image sub-command
    img_parser = subparsers.add_parser("image", help="Detect objects in an image")
    img_parser.add_argument("path", help="Path to the image file")
    img_parser.add_argument(
        "-o", "--output", default=None,
        help="Save annotated image to this path instead of displaying it",
    )

    # camera sub-command
    cam_parser = subparsers.add_parser("camera", help="Real-time detection from a camera")
    cam_parser.add_argument(
        "-s", "--source", default=0,
        help="Camera index or video file path (default: 0)",
    )

    # shared options
    for p in (img_parser, cam_parser):
        p.add_argument(
            "-m", "--model", default="yolov8n.pt",
            help="YOLO model to use (default: yolov8n.pt)",
        )
        p.add_argument(
            "-c", "--confidence", type=float, default=0.5,
            help="Minimum detection confidence (default: 0.5)",
        )

    args = parser.parse_args()

    if args.command == "image":
        run_image(args.path, args.model, args.confidence, args.output)
    elif args.command == "camera":
        source = int(args.source) if str(args.source).isdigit() else args.source
        run_camera(source, args.model, args.confidence)


if __name__ == "__main__":
    main()
