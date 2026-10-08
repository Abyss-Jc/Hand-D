"""Human review CLI can curate the same canonical SQLite the collector writes."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from handd_core.dataset_store import DatasetStore
from tests.test_feature_transform import landmark_fixture


class ReviewCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workspace = Path(self.tmp.name)
        self.store = DatasetStore(self.workspace / 'handd.sqlite')
        self.addCleanup(self.store.close)
        self.store.create_participant('P001')
        self.store.create_session('S001', 'P001')
        self.store.create_capture('C001', 'S001', 'Fist', 'Right')
        points = landmark_fixture()
        self.store.add_sample(
            'sample-1', 'C001', 0, points, points,
            raw_mp_handedness='Left', timestamp_ms=100, provenance={},
        )

    def cli(self, *args):
        return subprocess.run(
            [sys.executable, '-m', 'handd_core.review_cli', '--workspace',
             str(self.workspace), *args], capture_output=True, text=True,
            timeout=20, check=False,
        )

    def test_review_requires_explicit_id_and_preserves_drop_history(self):
        listing = self.cli('list', '--status', 'unreviewed')
        self.assertEqual(listing.returncode, 0, listing.stderr)
        self.assertIn('sample-1', listing.stdout)
        self.assertIn('Fist', listing.stdout)
        for operation in ('accept', 'drop', 'restore', 'reject'):
            result = self.cli(operation, 'sample-1')
            self.assertEqual(result.returncode, 0, result.stderr)
        row = self.store.get_sample('sample-1')
        self.assertEqual(row['review_status'], 'rejected')
        self.assertEqual(row['lifecycle_status'], 'active')
        self.assertEqual([x['new_status'] for x in self.store.list_review_events('sample-1')],
                         ['accepted', 'rejected'])
        self.assertEqual([x['new_status'] for x in self.store.list_lifecycle_events('sample-1')],
                         ['dropped', 'active'])
        self.assertEqual(self.store.list_eligible_samples(), [])

    def test_readiness_command_reports_blockers_without_modifying_review(self):
        result = self.cli('readiness')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('no_eligible_samples', result.stdout)
        self.assertIn('unreviewed_samples', result.stdout)
        self.assertEqual(self.store.get_sample('sample-1')['review_status'], 'unreviewed')

    def test_dropped_sample_is_hidden_from_default_list_but_can_be_restored(self):
        self.store.set_lifecycle_status('sample-1', 'dropped')
        default_view = self.cli('list')
        self.assertEqual(default_view.returncode, 0, default_view.stderr)
        self.assertNotIn('sample-1', default_view.stdout)
        all_view = self.cli('list', '--include-dropped')
        self.assertEqual(all_view.returncode, 0, all_view.stderr)
        self.assertIn('sample-1', all_view.stdout)


if __name__ == '__main__':
    unittest.main()
