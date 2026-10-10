"""Live-stream callback adapter: MediaPipe results -> time-based SQLite Capture."""

import tempfile
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from handd_core.capture_sampling import CaptureSampler
from handd_core.dataset_store import DatasetStore
from handd_core.live_collection import LatestCaptureResults
from tests.test_feature_transform import landmark_fixture


def mock_result(raw_hands):
    world = landmark_fixture()
    image = world.copy()
    image[:, :2] = .5 + world[:, :2]
    points = lambda arr: [SimpleNamespace(x=float(x), y=float(y), z=float(z))
                          for x, y, z in arr]
    return SimpleNamespace(
        handedness=[[SimpleNamespace(category_name=hand)] for hand in raw_hands],
        hand_landmarks=[points(image) for _ in raw_hands],
        hand_world_landmarks=[points(world) for _ in raw_hands],
    )


class LatestCaptureResultsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = DatasetStore(Path(self.tmp.name) / 'handd.sqlite')
        self.addCleanup(self.store.close)
        self.store.create_participant('P001')
        self.store.create_session('S001', 'P001')
        self.store.create_capture('C001', 'S001', 'Index_Finger', 'Right', target=4)
        self.sampler = CaptureSampler(
            self.store, 'C001', interval_ms=100, device_id='anonymous',
            camera='camera-0', software={},
        )

    def test_selects_configured_actual_hand_from_two_landmark_sets(self):
        bridge = LatestCaptureResults()
        bridge.register_frame(10, 100)
        bridge.on_result(mock_result(['Right', 'Left']), None, 100)
        sample_id = bridge.drain_to(self.sampler)
        self.assertIsNotNone(sample_id)
        sample = self.store.get_sample(sample_id)
        self.assertEqual(sample['frame_index'], 10)
        self.assertEqual(sample['raw_mp_handedness'], 'Left')
        self.assertEqual(sample['provenance']['actual_handedness'], 'Right')

    def test_latest_only_queue_drops_superseded_callback(self):
        bridge = LatestCaptureResults()
        bridge.register_frame(1, 100)
        bridge.register_frame(2, 200)
        bridge.on_result(mock_result(['Left']), None, 100)
        bridge.on_result(mock_result(['Left']), None, 200)
        sample_id = bridge.drain_to(self.sampler)
        self.assertEqual(self.store.get_sample(sample_id)['frame_index'], 2)
        self.assertIsNone(bridge.drain_to(self.sampler))
        self.assertEqual(self.store.count_samples(), 1)

    def test_no_hands_or_missing_registered_frame_does_not_persist(self):
        bridge = LatestCaptureResults()
        bridge.on_result(mock_result(['Left']), None, 100)
        self.assertIsNone(bridge.drain_to(self.sampler))
        bridge.register_frame(1, 200)
        bridge.on_result(mock_result([]), None, 200)
        self.assertIsNone(bridge.drain_to(self.sampler))
        self.assertEqual(self.store.count_samples(), 0)

    def test_live_diagnostics_distinguish_no_hand_from_filtered_capture(self):
        bridge = LatestCaptureResults()
        bridge.register_frame(10, 100)
        bridge.on_result(mock_result([]), None, 100)
        bridge.drain_to(self.sampler)
        first = bridge.statistics()
        self.assertEqual(first['registered_callbacks'], 1)
        self.assertEqual(first['callbacks_with_hands'], 0)
        self.assertEqual(first['samples_saved'], 0)
        bridge.register_frame(11, 200)
        bridge.on_result(mock_result(['Left']), None, 200)
        bridge.drain_to(self.sampler)
        second = bridge.statistics()
        self.assertEqual(second['registered_callbacks'], 2)
        self.assertEqual(second['callbacks_with_hands'], 1)
        self.assertEqual(second['samples_saved'], 1)

    def test_collection_cli_help_is_available_without_starting_camera(self):
        result = subprocess.run(
            [sys.executable, '-m', 'handd_core.collect_cli', '--help'],
            capture_output=True, text=True, timeout=20, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--workspace', result.stdout)
        self.assertIn('--participant', result.stdout)
        self.assertIn('--gesture', result.stdout)
        self.assertIn('--max-seconds', result.stdout)


if __name__ == '__main__':
    unittest.main()
