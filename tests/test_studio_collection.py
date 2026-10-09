"""Public contract: live Studio collection reuses MediaPipe results, not video."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from handd_core.dataset_store import DatasetStore
from handd_core.studio_collection import StudioCollection
from handd_core.runtime_v2 import GestureRuntime, LatestRuntimeResults
from tests.test_runtime_v2 import observation


class StudioCollectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workspace = Path(self.tmp.name)
        store = DatasetStore(self.workspace / 'handd.sqlite')
        store.close()
        self.controller = StudioCollection(
            self.workspace, device_config=self.workspace / 'anonymous-device.txt')
        self.addCleanup(self.controller.close)

    def test_explicit_start_persists_real_landmarks_until_quota_then_finishes(self):
        self.assertEqual(self.controller.status()['state'], 'idle')
        initial = DatasetStore(self.workspace / 'handd.sqlite')
        self.assertEqual(initial.count_samples(), 0)
        initial.close()
        started = self.controller.start(
            participant='P001', gesture='Index_Finger', hand='Right',
            target=2, interval_ms=100)
        self.assertEqual(started['state'], 'capturing')
        self.assertEqual(started['count'], 0)
        runtime = GestureRuntime()
        mailbox = LatestRuntimeResults()
        for i, ms in enumerate((100, 150, 210), start=1):
            mailbox.on_result(observation('Left'), None, ms)
            output = mailbox.drain_to(runtime, on_observation=self.controller.offer_result)
            self.assertEqual(output['type'], 'runtime.update')
        finished = self.controller.status()
        self.assertEqual(finished['count'], 2)
        self.assertEqual(finished['state'], 'complete')
        store = DatasetStore(self.workspace / 'handd.sqlite')
        self.addCleanup(store.close)
        self.assertEqual(store.count_samples(), 2)
        for row in store.list_samples_overview():
            sample = store.get_sample(row['sample_id'])
            self.assertEqual(sample['review_status'], 'unreviewed')
            self.assertEqual(sample['raw_mp_handedness'], 'Left')
            self.assertEqual(sample['image_landmarks'].shape, (21, 3))
            self.assertEqual(sample['world_landmarks'].shape, (21, 3))
            self.assertNotIn('frame', sample['provenance'])
        self.assertIsNone(self.controller.offer_result(observation('Left'), 300))

    def test_pause_resume_and_manual_finish_preserve_existing_samples(self):
        self.controller.start(participant='P002', gesture='Fist', hand='Left',
                              target=5, interval_ms=100)
        self.assertEqual(self.controller.status()['count'], 0)
        self.assertIsNotNone(self.controller.offer_result(observation('Right'), 100))
        self.controller.pause()
        self.assertIsNone(self.controller.offer_result(observation('Right'), 250))
        self.assertEqual(self.controller.status()['count'], 1)
        self.controller.resume()
        self.assertIsNotNone(self.controller.offer_result(observation('Right'), 360))
        stopped = self.controller.finish()
        self.assertEqual(stopped['state'], 'finished')
        self.assertEqual(stopped['count'], 2)
        self.assertIsNone(self.controller.offer_result(observation('Right'), 410))

    def test_invalid_or_implicit_collection_never_creates_samples(self):
        for fields in (
            dict(participant='P003',gesture='Fist',hand='Right',target=2,interval_ms=100),
            dict(participant='P001',gesture='',hand='Right',target=2,interval_ms=100),
            dict(participant='P001',gesture='Fist',hand='Neither',target=2,interval_ms=100),
            dict(participant='P001',gesture='Fist',hand='Right',target=0,interval_ms=100),
        ):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.controller.start(**fields)
        with self.assertRaises(ValueError):
            self.controller.pause()
        self.assertIsNone(self.controller.offer_result(observation('Left'), 123))
        self.assertEqual(self.controller.status()['state'], 'idle')

    def test_invalid_mediapipe_observation_is_skipped_without_closing_camera(self):
        self.controller.start(participant='P001',gesture='Fist',
                              hand='Right',target=2,interval_ms=100)
        bad=SimpleNamespace(
            handedness=[[SimpleNamespace(category_name='Left')]],
            hand_landmarks=[[SimpleNamespace(x='oops',y=0,z=0)]],
            hand_world_landmarks=[[SimpleNamespace(x=0,y=0,z=0)]],
        )
        self.assertIsNone(self.controller.offer_result(bad, 100))
        self.assertEqual(self.controller.status()['state'], 'capturing')
        self.assertIsNotNone(self.controller.offer_result(observation('Left'), 200))
