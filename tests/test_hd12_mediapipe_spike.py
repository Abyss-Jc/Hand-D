"""HD-12: no-camera LIVE_STREAM hand-landmarker contract across MP versions.

Run against isolated MediaPipe 1.1.0 with `uv run --no-project --python 3.13
--with mediapipe==1.1.0 --with numpy python -m unittest ...`.
This does not verify detected human hands, handedness correctness or FPS.
"""
import os
import threading
import unittest
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from handd_core.feature_transform import canonicalize_world_landmarks
from tests.test_feature_transform import landmark_fixture


TASK=Path(__file__).resolve().parents[1]/'models'/'hand_landmarker.task'


class MediaPipeLiveStreamContractTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("HANDD_HD12_MP_LIVE") == "1",
                         "Opt-in LIVE_STREAM graphics probe; set HANDD_HD12_MP_LIVE=1")
    def test_task_opens_and_delivers_timestamps_and_expected_result_fields_without_camera(self):
        self.assertTrue(TASK.is_file(), 'Hand Landmarker task must be available')
        seen=[]
        received=threading.Event()
        def callback(result,image,timestamp):
            seen.append((result,image,timestamp))
            received.set()
        options=vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(TASK)),
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=.7,
            min_hand_presence_confidence=.7,
            min_tracking_confidence=.7,
            result_callback=callback,
        )
        frame=mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=np.zeros((128,128,3),dtype=np.uint8))
        with vision.HandLandmarker.create_from_options(options) as detector:
            detector.detect_async(frame,1000)
            self.assertTrue(received.wait(8), 'Callback failed to arrive for blank frame')
            received.clear()
            detector.detect_async(frame,1020)
            self.assertTrue(received.wait(8),'Second callback failed to arrive')
            self.assertEqual([x[2] for x in seen],[1000,1020])
            for result,output_image,timestamp in seen:
                self.assertTrue(hasattr(result,'hand_landmarks'))
                self.assertTrue(hasattr(result,'hand_world_landmarks'))
                self.assertTrue(hasattr(result,'handedness'))
                self.assertEqual(len(result.hand_landmarks),len(result.hand_world_landmarks))
                self.assertEqual(len(result.hand_world_landmarks),len(result.handedness))
                self.assertEqual(output_image.width,128)
                self.assertEqual(output_image.height,128)
                self.assertIsInstance(timestamp,int)

    def test_transform_contract_has_21_world_points_69_features_and_unmodified_handedness(self):
        original=landmark_fixture()
        for hand in ('Left','Right'):
            with self.subTest(hand=hand):
                points=original.copy()
                features=canonicalize_world_landmarks(points,hand)
                self.assertEqual(features.shape,(69,))
                self.assertEqual(features.dtype,np.float32)
                self.assertTrue(np.isfinite(features).all())
                np.testing.assert_array_equal(points,original)


if __name__=='__main__':unittest.main()
