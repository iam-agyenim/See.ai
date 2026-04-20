# See.ai

AI that sees and detects things — a computer vision object detection tool powered by YOLOv8.

## Features

- **Image detection** — detect and annotate objects in any image file
- **Live camera detection** — real-time object detection from a webcam or video file
- **Configurable** — choose your YOLO model and confidence threshold

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Detect objects in an image

```bash
python main.py image path/to/image.jpg
```

Save the annotated result instead of displaying it:

```bash
python main.py image path/to/image.jpg -o output.jpg
```

### Real-time detection from a camera

```bash
python main.py camera
```

Use a specific camera index or video file:

```bash
python main.py camera -s 1
python main.py camera -s path/to/video.mp4
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-m`, `--model` | YOLO model to use | `yolov8n.pt` |
| `-c`, `--confidence` | Minimum detection confidence | `0.5` |
| `-o`, `--output` | Save annotated image to path (image mode) | — |
| `-s`, `--source` | Camera index or video path (camera mode) | `0` |

## Project Structure

```
See.ai/
├── main.py              # CLI entry point
├── src/
│   ├── detector.py      # YOLOv8 object detector
│   ├── camera.py        # Video capture wrapper
│   └── utils.py         # Image I/O and helpers
├── tests/
│   ├── test_detector.py
│   └── test_utils.py
├── requirements.txt
└── .gitignore
```

## Running Tests

```bash
pip install pytest
pytest
```
