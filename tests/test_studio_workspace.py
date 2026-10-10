"""HD-09: manual Studio curation and snapshot actions use real canonical SQLite."""
import json
import tempfile
import unittest
from pathlib import Path

from handd_core.dataset_store import DatasetStore
from handd_core.studio_workspace import StudioWorkspace
from tests.test_feature_transform import landmark_fixture


class StudioWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.store = DatasetStore(self.path / 'handd.sqlite')
        self.addCleanup(self.store.close)
        self.store.create_participant('P001')
        self.store.create_session('S001', 'P001')
        self.store.create_capture('C001', 'S001', 'Index_Finger', 'Right')
        points = landmark_fixture()
        for i in range(3):
            self.store.add_sample(
                f'SAMPLE{i}', 'C001', i, points, points,
                raw_mp_handedness='Left', timestamp_ms=100*i, provenance={'test': True},
            )

    def test_readonly_requires_explicit_existing_workspace(self):
        with self.assertRaises(ValueError):
            StudioWorkspace(self.path / 'missing')
        self.assertFalse((self.path / 'missing' / 'handd.sqlite').exists())
        summary = StudioWorkspace(self.path).overview()
        self.assertEqual(summary['sample_count'], 3)
        self.assertEqual(summary['review_counts']['unreviewed'], 3)
        self.assertEqual(len(summary['samples']), 3)
        self.assertFalse(summary['snapshot_ready'])
        self.assertNotIn('world_landmarks', json.dumps(summary))

    def test_review_queue_can_reach_samples_after_first_page(self):
        points = landmark_fixture()
        for i in range(3, 47):
            self.store.add_sample(
                f'SAMPLE{i:03d}', 'C001', i, points, points,
                raw_mp_handedness='Left', timestamp_ms=100*i, provenance={},
            )
        studio = StudioWorkspace(self.path)
        first = studio.overview(offset=0)
        second = studio.overview(offset=40)
        self.assertEqual(first['review_page_size'], 40)
        self.assertEqual(first['review_offset'], 0)
        self.assertEqual(len(first['samples']), 40)
        self.assertEqual(second['review_offset'], 40)
        self.assertEqual(len(second['samples']), 7)
        self.assertFalse(
            {row['sample_id'] for row in first['samples']}
            & {row['sample_id'] for row in second['samples']}
        )
        with self.assertRaises(ValueError):
            studio.overview(offset=-1)

    def test_manual_review_drop_restore_and_explicit_snapshot(self):
        studio = StudioWorkspace(self.path)
        studio.transition('SAMPLE0', 'accept')
        studio.transition('SAMPLE1', 'accept')
        studio.transition('SAMPLE1', 'drop')
        self.assertEqual(studio.overview()['review_counts']['accepted'], 1)
        studio.transition('SAMPLE1', 'restore')
        self.assertEqual(studio.overview()['review_counts']['accepted'], 2)
        studio.transition('SAMPLE2', 'reject')
        self.assertEqual(studio.overview()['review_counts']['rejected'], 1)
        self.assertEqual(len(list((self.path / 'snapshots').glob('*')))
                         if (self.path / 'snapshots').exists() else 0, 0)
        artifact = studio.build_snapshot(note='manual Studio snapshot')
        self.assertTrue((self.path / 'snapshots' / artifact['snapshot_id']
                         / 'manifest.json').is_file())
        manifest = json.loads((self.path / 'snapshots' / artifact['snapshot_id']
                               / 'manifest.json').read_text())
        self.assertEqual(manifest['sample_count'], 2)
        self.assertEqual(self.store.list_review_events('SAMPLE2')[-1]['new_status'],
                         'rejected')

    def test_invalid_actions_and_sample_id_do_not_mutate_database(self):
        studio = StudioWorkspace(self.path)
        with self.assertRaises(ValueError):
            studio.transition('SAMPLE0', 'delete')
        with self.assertRaises(KeyError):
            studio.transition('not-found', 'accept')
        self.assertEqual(self.store.get_sample('SAMPLE0')['review_status'], 'unreviewed')

    def test_p003_only_never_enables_a_development_snapshot(self):
        self.store.create_participant('P003')
        self.store.create_session('FINAL-S001','P003')
        self.store.create_capture('FINAL-C001','FINAL-S001','Fist','Right')
        points = landmark_fixture()
        self.store.add_sample('FINAL-SAMPLE','FINAL-C001',0,points,points,
                              raw_mp_handedness='Left',timestamp_ms=10,provenance={})
        studio = StudioWorkspace(self.path)
        studio.transition('FINAL-SAMPLE', 'accept')
        overview = studio.overview()
        self.assertFalse(overview['snapshot_ready'])
        self.assertEqual(overview['eligible_count'], 0,
                         'P003 samples are reserved for the sealed Final Test')
        with self.assertRaises(ValueError):
            studio.build_snapshot()
