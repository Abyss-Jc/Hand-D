"""Snapshot readiness is read-only; review and lifecycle decisions remain human."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from handd_core.dataset_store import DatasetStore
from handd_core.snapshot_readiness import assess_snapshot_readiness
from tests.test_feature_transform import landmark_fixture


class SnapshotReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = DatasetStore(Path(self.temp.name) / 'handd.sqlite')
        self.addCleanup(self.store.close)
        self.store.create_participant('P001')
        self.store.create_session('S001', 'P001')
        self.store.create_capture('C001', 'S001', 'Index_Finger', 'Right')
        self.world = landmark_fixture()

    def add(self, sample_id, *, world=None, capture_id='C001', frame_index=0):
        self.store.add_sample(
            sample_id, capture_id, frame_index, self.world,
            self.world if world is None else world,
            raw_mp_handedness='Left', timestamp_ms=frame_index * 100,
            provenance={'device_id': 'anonymous'},
        )

    def test_no_reviewed_eligible_samples_is_blocker_unreviewed_is_warning(self):
        self.add('A')
        result = assess_snapshot_readiness(self.store)
        self.assertFalse(result.ready)
        self.assertIn('no_eligible_samples', result.blockers)
        self.assertIn('unreviewed_samples', result.warnings)
        self.assertEqual(result.eligible_sample_ids, ())
        self.assertEqual(self.store.get_sample('A')['review_status'], 'unreviewed')

    def test_review_and_drop_are_independent_and_only_accepted_active_enter(self):
        self.add('A', frame_index=1)
        self.add('B', frame_index=2)
        self.add('C', frame_index=3)
        self.store.set_review_status('A', 'accepted', reason='human approved')
        self.store.set_review_status('B', 'rejected', reason='wrong gesture')
        self.store.set_review_status('C', 'accepted')
        self.store.set_lifecycle_status('C', 'dropped')
        result = assess_snapshot_readiness(self.store)
        self.assertTrue(result.ready)
        self.assertEqual(result.eligible_sample_ids, ('A',))
        self.assertIn('coverage_low', result.warnings)
        self.assertEqual(self.store.get_sample('C')['review_status'], 'accepted')
        self.store.set_lifecycle_status('C', 'active', reason='restored')
        self.assertEqual(assess_snapshot_readiness(self.store).eligible_sample_ids, ('A', 'C'))
        self.assertEqual(len(self.store.list_lifecycle_events('C')), 2)

    def test_accepted_but_degenerate_hand_blocks_snapshot(self):
        self.add('A', world=np.zeros((21, 3)))
        self.store.set_review_status('A', 'accepted')
        result = assess_snapshot_readiness(self.store)
        self.assertFalse(result.ready)
        self.assertIn('invalid_feature_transform', result.blockers)

    def test_final_test_session_overlap_is_blocker_not_silent_leakage(self):
        self.add('A')
        self.store.set_review_status('A', 'accepted')
        result = assess_snapshot_readiness(
            self.store, development_session_ids={'S001'}, final_test_session_ids={'S001'}
        )
        self.assertFalse(result.ready)
        self.assertIn('development_final_overlap', result.blockers)


if __name__ == '__main__':
    unittest.main()
