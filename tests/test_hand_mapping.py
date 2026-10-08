"""A camera handedness calibration must not assume MediaPipe raw labels are physical labels."""
import unittest

from handd_core.hand_mapping import HandMappingCalibration


class HandMappingTests(unittest.TestCase):
    def test_physical_right_raw_left_and_physical_left_raw_right_support_inversion(self):
        calibration = HandMappingCalibration(min_observations=5, min_agreement=0.8)
        for _ in range(8):
            calibration.record('Right', ['Left'])
            calibration.record('Left', ['Right'])
        result = calibration.assess()
        self.assertEqual(result['mapping'], 'inverted')
        self.assertEqual(result['counts']['Right'], {'Left': 8})
        self.assertEqual(result['counts']['Left'], {'Right': 8})

    def test_physical_and_raw_same_support_direct_mapping(self):
        calibration = HandMappingCalibration(min_observations=5)
        for _ in range(6):
            calibration.record('Right', ['Right'])
            calibration.record('Left', ['Left'])
        self.assertEqual(calibration.assess()['mapping'], 'direct')

    def test_missing_hand_or_two_hands_never_certify_mapping(self):
        calibration = HandMappingCalibration(min_observations=3)
        for _ in range(10):
            calibration.record('Right', [])
            calibration.record('Left', ['Right', 'Left'])
        result = calibration.assess()
        self.assertEqual(result['mapping'], 'inconclusive')
        self.assertEqual(result['counts']['Right'], {})
        self.assertEqual(result['counts']['Left'], {})

    def test_unstable_or_mixed_labels_are_inconclusive(self):
        calibration = HandMappingCalibration(min_observations=5, min_agreement=0.8)
        for raw in ['Left', 'Right'] * 5:
            calibration.record('Right', [raw])
            calibration.record('Left', ['Right'])
        self.assertEqual(calibration.assess()['mapping'], 'inconclusive')

    def test_no_camera_required_for_help(self):
        import subprocess
        import sys
        result = subprocess.run([sys.executable, '-m', 'handd_core.hand_mapping', '--help'],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--camera', result.stdout)
        self.assertIn('--hold-seconds', result.stdout)

if __name__ == '__main__':
    unittest.main()
