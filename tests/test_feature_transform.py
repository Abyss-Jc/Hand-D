"""Contract tests for the shared Hand-D Feature Transform v1 seam."""

import unittest
import subprocess
import sys
from pathlib import Path

import numpy as np

from handd_core.feature_transform import canonicalize_world_landmarks
from visualizer_app.gesture_engine import canonicalize as runtime_canonicalize


def landmark_fixture() -> np.ndarray:
    """Non-degenerate, reproducible world-space 21-point hand fixture."""
    return np.array([
        [0, 0, 0], [-.03, -.03, .01], [-.045, -.05, .01],
        [-.05, -.065, .02], [-.052, -.08, .025], [-.04, -.07, .015],
        [-.046, -.106, .018], [-.049, -.135, .025], [-.052, -.161, .034],
        [0, -.082, 0], [0, -.118, .003], [0, -.148, .006],
        [0, -.17, .011], [.037, -.072, -.002], [.042, -.105, -.004],
        [.046, -.13, -.008], [.047, -.153, -.011], [.068, -.056, -.004],
        [.077, -.083, -.008], [.080, -.099, -.011], [.083, -.120, -.014],
    ], dtype=np.float64)


class FeatureTransformV1Tests(unittest.TestCase):
    def test_baseline_right_hand_vector_is_stable(self):
        features = canonicalize_world_landmarks(landmark_fixture(), 'Right')
        self.assertIsNotNone(features)
        self.assertEqual(features.shape, (69,))
        self.assertEqual(features.dtype, np.float32)
        np.testing.assert_allclose(features[:16], [
            0, 0, 0, 0.37462887, 0.3649868, 0.017821716,
            0.55245906, 0.59993517, -0.039342277,
            0.63070434, 0.79185176, 0.047244359,
            0.6638993, 0.9773149, 0.080634855, 0.50266659,
        ], atol=1e-6, rtol=0)
        np.testing.assert_allclose(features[-6:], [
            0.019026982, -0.99272168, 0.11891863,
            0.15684059, 0.12043117, 0.9802537,
        ], atol=1e-6, rtol=0)
        self.assertAlmostEqual(float(np.sum(features.astype(np.float64))), 22.29951075, places=5)

    def test_baseline_left_hand_mirror_correction_is_stable(self):
        features = canonicalize_world_landmarks(landmark_fixture(), 'Left')
        self.assertIsNotNone(features)
        self.assertEqual(features.shape, (69,))
        np.testing.assert_allclose(features[:9], [
            0, 0, 0, 0.37462887, 0.3649868, -0.017821716,
            0.55245906, 0.59993517, 0.039342277,
        ], atol=1e-6, rtol=0)
        np.testing.assert_allclose(features[-6:], [
            -0.019026982, -0.99272168, 0.11891863,
            0.15684059, -0.12043117, -0.9802537,
        ], atol=1e-6, rtol=0)
        self.assertAlmostEqual(float(np.sum(features.astype(np.float64))), 22.79541776, places=5)

    def test_shared_transform_is_the_runtime_transform(self):
        landmarks = landmark_fixture()
        for hand in ('Right', 'Left'):
            with self.subTest(hand=hand):
                self.assertIs(runtime_canonicalize, canonicalize_world_landmarks)
                np.testing.assert_array_equal(runtime_canonicalize(landmarks, hand),
                                              canonicalize_world_landmarks(landmarks, hand))

    def test_does_not_modify_landmarks(self):
        landmarks = landmark_fixture()
        original = landmarks.copy()
        canonicalize_world_landmarks(landmarks, 'Left')
        np.testing.assert_array_equal(landmarks, original)

    def test_float32_landmarks_are_supported_without_mutation(self):
        landmarks = landmark_fixture().astype(np.float32)
        original = landmarks.copy()
        output = canonicalize_world_landmarks(landmarks, 'Right')
        self.assertEqual(output.shape, (69,))
        self.assertEqual(output.dtype, np.float32)
        self.assertTrue(np.isfinite(output).all())
        np.testing.assert_array_equal(landmarks, original)

    def test_degenerate_and_nonfinite_input_are_skipped(self):
        for landmarks in (np.zeros((21, 3)), np.full((21, 3), np.nan),
                          np.full((21, 3), np.inf)):
            with self.subTest(landmarks=landmarks[0, 0]):
                self.assertIsNone(canonicalize_world_landmarks(landmarks, 'Right'))

    def test_shape_and_handedness_contracts_are_explicit(self):
        with self.assertRaises(ValueError):
            canonicalize_world_landmarks(np.zeros((20, 3)), 'Right')
        with self.assertRaises(ValueError):
            canonicalize_world_landmarks(landmark_fixture(), 'Unknown')

    def test_legacy_visualizer_source_launch_keeps_importing(self):
        """Before the uv package cutover, direct source-script launch must work."""
        project_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, '-c', 'import gesture_engine; print("IMPORT_OK")'],
            cwd=project_root / 'visualizer_app',
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('IMPORT_OK', result.stdout)


if __name__ == '__main__':
    unittest.main()
