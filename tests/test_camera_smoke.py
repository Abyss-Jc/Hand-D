"""Legacy model and camera smoke contracts; real hardware is checked separately."""

import subprocess
import sys
import unittest
from pathlib import Path

from handd_core.camera_smoke import load_legacy_checkpoint, classify_hand
from tests.test_feature_transform import landmark_fixture


class CameraSmokeTests(unittest.TestCase):
    def test_existing_checkpoint_is_usable_for_inference(self):
        model_path = Path(__file__).resolve().parents[1] / 'models/gesture_mlp.pth'
        self.assertTrue(model_path.is_file())
        model = load_legacy_checkpoint(model_path)
        result = classify_hand(model, landmark_fixture(), 'Right')
        self.assertIn(result, ('Fist', 'Index_Finger', 'Ruler', 'Thumb_Up', 'Idle'))

    def test_degenerate_landmark_does_not_reach_classifier(self):
        class ForbiddenModel:
            def predict(self, features):
                self.fail('model must not run for invalid landmarks')
        self.assertIsNone(classify_hand(ForbiddenModel(), landmark_fixture()*0, 'Right'))

    def test_help_has_bounded_duration_without_opening_camera(self):
        result = subprocess.run(
            [sys.executable, '-m', 'handd_core.camera_smoke', '--help'],
            capture_output=True, text=True, timeout=12, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--seconds', result.stdout)
        self.assertIn('--camera', result.stdout)
        self.assertIn('--preview', result.stdout)


if __name__ == '__main__':
    unittest.main()
