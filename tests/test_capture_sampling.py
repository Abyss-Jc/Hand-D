"""Public collection seam: real landmark observations -> durable Samples."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from handd_core.capture_sampling import CaptureSampler
from handd_core.dataset_store import DatasetStore
from tests.test_feature_transform import landmark_fixture


class CaptureSamplingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'handd.sqlite'
        self.store = DatasetStore(self.path)
        self.addCleanup(self.store.close)
        self.store.create_participant('P001')
        self.store.create_session('S001', 'P001')
        self.store.create_capture('C001', 'S001', 'Index_Finger', 'Right', target=3)
        self.world = landmark_fixture()
        self.image = self.world / 2

    def sampler(self, interval_ms=100):
        return CaptureSampler(
            self.store, 'C001', interval_ms=interval_ms,
            device_id='anonymous-device-01', camera='camera-0',
            software={'mediapipe': '0.10.33', 'opencv': '4.13.0'},
        )

    def offer(self, sampler, frame_index, timestamp_ms, hand='Left', image=None, world=None):
        return sampler.offer(
            frame_index=frame_index, timestamp_ms=timestamp_ms,
            raw_mp_handedness=hand,
            image_landmarks=self.image if image is None else image,
            world_landmarks=self.world if world is None else world,
        )

    def test_samples_by_elapsed_time_not_frame_stride_and_stops_at_quota(self):
        sampler = self.sampler(interval_ms=100)
        ids = [self.offer(sampler, i, ms) for i, ms in enumerate((0, 20, 60, 100, 150, 200, 300))]
        self.assertEqual([i is not None for i in ids], [True, False, False, True, False, True, False])
        self.assertTrue(sampler.finished)
        self.assertEqual(sampler.count, 3)
        self.assertEqual(self.store.count_samples(), 3)
        sample = self.store.get_sample(ids[0])
        self.assertEqual(sample['participant_id'], 'P001')
        self.assertEqual(sample['gesture'], 'Index_Finger')
        self.assertEqual(sample['review_status'], 'unreviewed')
        self.assertEqual(sample['lifecycle_status'], 'active')
        self.assertEqual(sample['provenance']['device_id'], 'anonymous-device-01')
        self.assertEqual(sample['provenance']['camera'], 'camera-0')
        self.assertEqual(sample['provenance']['software']['mediapipe'], '0.10.33')
        np.testing.assert_array_equal(sample['world_landmarks'], self.world)

    def test_invalid_or_wrong_hand_observations_do_not_consume_quota_or_interval(self):
        sampler = self.sampler()
        self.assertIsNone(self.offer(sampler, 0, 0, hand='Right'))
        self.assertIsNone(self.offer(sampler, 1, 0, world=np.zeros((21, 3))))
        self.assertIsNone(self.offer(sampler, 2, 0, image=np.full((21, 3), np.nan)))
        sample_id = self.offer(sampler, 3, 0)
        self.assertIsNotNone(sample_id)
        self.assertEqual(sampler.count, 1)
        self.assertEqual(self.store.count_samples(), 1)
        self.assertEqual(self.store.get_sample(sample_id)['provenance']['actual_handedness'], 'Right')

    def test_pause_resume_and_reopening_capture_preserves_time_window_and_quota(self):
        sampler = self.sampler()
        self.assertIsNotNone(self.offer(sampler, 10, 0))
        sampler.pause()
        self.assertIsNone(self.offer(sampler, 11, 150))
        sampler.resume()
        self.assertIsNone(self.offer(sampler, 10, 150))
        self.assertIsNotNone(self.offer(sampler, 11, 150))
        recovered = self.sampler()
        self.assertEqual(recovered.count, 2)
        self.assertIsNone(self.offer(recovered, 12, 180))
        self.assertIsNotNone(self.offer(recovered, 13, 250))
        self.assertIsNone(self.offer(recovered, 14, 350))
        self.assertEqual(recovered.count, 3)

    def test_local_device_provenance_is_generated_once_outside_workspace(self):
        local_config = Path(self.temp.name) / 'outside-workspace' / 'device-id.txt'
        sampler = CaptureSampler.for_local_device(
            self.store, 'C001', interval_ms=100, camera='camera-0',
            device_config_path=local_config,
        )
        self.assertTrue(local_config.is_file())
        self.assertEqual(len(sampler.device_id), 32)
        same = CaptureSampler.for_local_device(
            self.store, 'C001', interval_ms=100, camera='camera-0',
            device_config_path=local_config,
        )
        self.assertEqual(sampler.device_id, same.device_id)
        sample_id = self.offer(sampler, 0, 0)
        metadata = self.store.get_sample(sample_id)['provenance']
        self.assertEqual(metadata['device_id'], same.device_id)
        self.assertIn('mediapipe', metadata['software'])
        self.assertIn('opencv-contrib-python', metadata['software'])
        self.assertNotIn(str(local_config), str(metadata))


if __name__ == '__main__':
    unittest.main()
