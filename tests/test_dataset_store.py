"""SQLite domain contract for Hand-D v2 (no camera/ML dependencies)."""

import json
from contextlib import closing
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from handd_core.dataset_store import DatasetStore


class DatasetStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'handd.sqlite'
        self.store = DatasetStore(self.path)
        self.addCleanup(self.store.close)
        self.store.create_participant('P001')
        self.store.create_session('S001', 'P001')
        self.store.create_capture('C001', 'S001', 'Index_Finger', 'Right', target=120)
        self.image = np.arange(63, dtype=np.float64).reshape(21, 3) / 100
        self.world = np.arange(63, dtype=np.float64).reshape(21, 3) / 1000

    def sample(self, sample_id='sample-1'):
        self.store.add_sample(sample_id, 'C001', 0, self.image, self.world,
                              raw_mp_handedness='Left', timestamp_ms=1200,
                              provenance={'device_id': 'anonymous-local-1', 'camera': 'demo'})
        return sample_id

    def test_schema_and_sample_roundtrip_preserve_raw_observations(self):
        sample_id = self.sample()
        result = self.store.get_sample(sample_id)
        self.assertEqual(result['capture_id'], 'C001')
        self.assertEqual(result['gesture'], 'Index_Finger')
        self.assertEqual(result['participant_id'], 'P001')
        self.assertEqual(result['session_id'], 'S001')
        self.assertEqual(result['raw_mp_handedness'], 'Left')
        self.assertEqual(result['frame_index'], 0)
        self.assertEqual(result['timestamp_ms'], 1200)
        self.assertEqual(result['review_status'], 'unreviewed')
        self.assertEqual(result['lifecycle_status'], 'active')
        self.assertEqual(result['provenance'], {'device_id': 'anonymous-local-1', 'camera': 'demo'})
        np.testing.assert_array_equal(result['image_landmarks'], self.image)
        np.testing.assert_array_equal(result['world_landmarks'], self.world)
        self.assertFalse(any(k in result for k in ('frame_bgr', 'photo', 'video')))
        with closing(sqlite3.connect(self.path)) as conn:
            self.assertGreaterEqual(conn.execute('PRAGMA user_version').fetchone()[0], 1)

    def test_review_and_lifecycle_are_independent_append_only_events(self):
        sample_id = self.sample()
        self.store.set_review_status(sample_id, 'accepted', reason='quality checked')
        self.store.set_lifecycle_status(sample_id, 'dropped', reason='defer')
        self.assertEqual(self.store.get_sample(sample_id)['review_status'], 'accepted')
        self.assertEqual(self.store.get_sample(sample_id)['lifecycle_status'], 'dropped')
        self.assertEqual(self.store.list_eligible_samples(), [])
        self.store.set_lifecycle_status(sample_id, 'active', reason='restored')
        self.assertEqual([x['sample_id'] for x in self.store.list_eligible_samples()], [sample_id])
        self.store.set_review_status(sample_id, 'rejected')
        self.assertEqual(self.store.list_eligible_samples(), [])
        self.assertEqual([e['new_status'] for e in self.store.list_review_events(sample_id)],
                         ['accepted', 'rejected'])
        self.assertEqual([e['new_status'] for e in self.store.list_lifecycle_events(sample_id)],
                         ['dropped', 'active'])

    def test_invalid_landmarks_and_references_fail_without_partial_persist(self):
        for bad in (np.zeros((20, 3)), np.full((21, 3), np.nan)):
            with self.assertRaises(ValueError):
                self.store.add_sample('bad', 'C001', 2, bad, self.world,
                                      raw_mp_handedness='Left', timestamp_ms=111, provenance={})
        with self.assertRaises(ValueError):
            self.store.add_sample('bad-hand', 'C001', 2, self.image, self.world,
                                  raw_mp_handedness='Unknown', timestamp_ms=111, provenance={})
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.add_sample('bad-ref', 'missing-capture', 0, self.image, self.world,
                                  raw_mp_handedness='Right', timestamp_ms=100, provenance={})
        self.assertEqual(self.store.count_samples(), 0)

    def test_core_observation_fields_cannot_be_rewritten_directly(self):
        sample_id = self.sample()
        with closing(sqlite3.connect(self.path)) as conn:
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute('UPDATE samples SET world_landmarks = ? WHERE sample_id = ?',
                             (json.dumps([[0, 0, 0]] * 21), sample_id))
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute('DELETE FROM samples WHERE sample_id = ?', (sample_id,))
        self.assertEqual(self.store.count_samples(), 1)

    def test_reopened_database_retains_state(self):
        self.sample()
        self.store.set_review_status('sample-1', 'accepted')
        self.store.close()
        reopened = DatasetStore(self.path)
        self.addCleanup(reopened.close)
        self.assertEqual([s['sample_id'] for s in reopened.list_eligible_samples()], ['sample-1'])

    def test_revision_requests_reject_bad_status_and_absent_sample(self):
        self.sample()
        with self.assertRaises(ValueError):
            self.store.set_review_status('sample-1', 'dropped')
        with self.assertRaises(ValueError):
            self.store.set_lifecycle_status('sample-1', 'accepted')
        with self.assertRaises(KeyError):
            self.store.set_review_status('unknown', 'accepted')
        self.assertEqual(len(self.store.list_review_events('sample-1')), 0)

    def test_audit_events_are_immutable_at_database_level(self):
        self.sample()
        self.store.set_review_status('sample-1', 'accepted')
        with closing(sqlite3.connect(self.path)) as conn:
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute('DELETE FROM review_events')
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute('UPDATE review_events SET new_status = ?', ('rejected',))

    def test_direct_sql_review_change_also_generates_audit_event(self):
        self.sample()
        with closing(sqlite3.connect(self.path)) as conn:
            conn.execute("UPDATE samples SET review_status='accepted' WHERE sample_id='sample-1'")
            conn.commit()
        events = self.store.list_review_events('sample-1')
        self.assertEqual([(e['previous_status'], e['new_status']) for e in events],
                         [('unreviewed', 'accepted')])
        self.assertEqual([row['sample_id'] for row in self.store.list_eligible_samples()],
                         ['sample-1'])

    def test_newer_schema_is_not_downgraded_or_written(self):
        self.store.close()
        with closing(sqlite3.connect(self.path)) as conn:
            conn.execute('PRAGMA user_version = 99')
        with self.assertRaisesRegex(ValueError, 'newer'):
            DatasetStore(self.path)
        with closing(sqlite3.connect(self.path)) as conn:
            self.assertEqual(conn.execute('PRAGMA user_version').fetchone()[0], 99)

    def test_legacy_duplicate_capture_frame_is_not_accepted_twice(self):
        self.sample()
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.add_sample('sample-2', 'C001', 0, self.image, self.world,
                                  raw_mp_handedness='Right', timestamp_ms=1300, provenance={})
        self.assertEqual(self.store.count_samples(), 1)


if __name__ == '__main__':
    unittest.main()
