import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """Load an image from disk.

    Args:
        path: Path to the image file.

    Returns:
        BGR image as a NumPy array.

    Raises:
        FileNotFoundError: If the image cannot be loaded.
    """
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return image


def save_image(path: str, image: np.ndarray) -> None:
    """Save an image to disk.

    Args:
        path: Destination file path.
        image: BGR image as a NumPy array.
    """
    cv2.imwrite(path, image)


def resize_image(image: np.ndarray, width: int = 640) -> np.ndarray:
    """Resize an image while maintaining aspect ratio.

    Args:
        image: BGR image as a NumPy array.
        width: Target width in pixels.

    Returns:
        Resized image.
    """
    h, w = image.shape[:2]
    ratio = width / w
    new_h = int(h * ratio)
    return cv2.resize(image, (width, new_h))
