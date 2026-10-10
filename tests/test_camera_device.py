"""A single webcam backend policy must also work on Apple Silicon."""
import unittest

from handd_core.camera_device import open_camera


class CameraDeviceTests(unittest.TestCase):
    def test_linux_uses_v4l2_and_macos_windows_use_native_capture(self):
        class CV2Stub:
            CAP_V4L2 = 200

            def __init__(self):
                self.calls = []

            def VideoCapture(self, *args):
                self.calls.append(args)
                return args

        cv2 = CV2Stub()
        self.assertEqual(open_camera(cv2, 2, platform="linux"), (2, 200))
        self.assertEqual(open_camera(cv2, 1, platform="darwin"), (1,))
        self.assertEqual(open_camera(cv2, 0, platform="win32"), (0,))
        self.assertEqual(cv2.calls, [(2, 200), (1,), (0,)])


if __name__ == "__main__":
    unittest.main()
