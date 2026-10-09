"""Preview must not steal inference cycles or halve a 30 Hz webcam."""
import unittest

from handd_core.preview_pacing import PreviewPacer


class PreviewPacingTests(unittest.TestCase):
    def test_30_hz_source_keeps_every_frame_with_live_preview_consumers(self):
        pacer = PreviewPacer(target_fps=30)
        enabled = [pacer.due(frame / 29.5, subscribers=1) for frame in range(120)]
        self.assertGreaterEqual(sum(enabled), 118)

    def test_60_hz_source_is_limited_near_30_hz(self):
        pacer = PreviewPacer(target_fps=30)
        enabled = [pacer.due(frame / 60, subscribers=1) for frame in range(120)]
        self.assertGreaterEqual(sum(enabled), 55)
        self.assertLessEqual(sum(enabled), 62)

    def test_no_mjpeg_consumers_means_no_jpeg_encoding_and_fresh_resume(self):
        pacer = PreviewPacer(target_fps=30)
        self.assertTrue(pacer.due(0., subscribers=1))
        self.assertFalse(pacer.due(.01, subscribers=0))
        self.assertFalse(pacer.due(10., subscribers=0))
        self.assertTrue(pacer.due(10.001, subscribers=1))

    def test_invalid_configuration_is_rejected(self):
        for fps in (0, -3):
            with self.assertRaises(ValueError):
                PreviewPacer(target_fps=fps)


if __name__ == '__main__':
    unittest.main()
