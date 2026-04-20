import cv2


class Camera:
    """Wrapper around an OpenCV video capture source."""

    def __init__(self, source: int | str = 0):
        """Open a camera or video file.

        Args:
            source: Camera index (0 for default webcam) or path to a video file.
        """
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

    def read(self):
        """Read a single frame.

        Returns:
            A BGR image as a NumPy array, or None if the stream has ended.
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        """Release the video source."""
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
