"""Platform-specific OpenCV capture choice shared by camera entry points."""
import sys


def open_camera(cv2, camera_index: int, *, platform: str | None = None):
    platform = sys.platform if platform is None else platform
    if platform.startswith("linux"):
        return cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    return cv2.VideoCapture(camera_index)
