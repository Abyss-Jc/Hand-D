"""HD-06 public seam: curated SQLite -> standalone immutable Development Snapshot."""

import hashlib
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from handd_core.dataset_store import DatasetStore
from handd_core.snapshot_builder import build_development_snapshot, verify_snapshot
from tests.test_feature_transform import landmark_fixture


class DevelopmentSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workspace = Path(self.tmp.name)
        self.store = DatasetStore(self.workspace / 'handd.sqlite')
        self.addCleanup(self.store.close)
        self.store.create_participant('P001')
        self.store.create_session('S001', 'P001')
        self.store.create_session('S002', 'P001')
        self.store.create_capture('C001', 'S001', 'Index_Finger', 'Right')
        self.store.create_capture('C002', 'S002', 'Fist', 'Right')
        self.world = landmark_fixture()

    def sample(self, sample_id, capture_id, frame_index, *, review='accepted'):
        self.store.add_sample(sample_id, capture_id, frame_index,
                              self.world, self.world, raw_mp_handedness='Left',
                              timestamp_ms=frame_index*100, provenance={'device_id': 'test-device'})
        if review != 'unreviewed':
            self.store.set_review_status(sample_id, review)

    def test_builds_standalone_npz_and_manifest_for_accepted_active_only(self):
        self.sample('a', 'C001', 1)
        self.sample('b', 'C002', 2)
        self.sample('unreviewed', 'C001', 3, review='unreviewed')
        self.sample('rejected', 'C002', 4, review='rejected')
        self.sample('dropped', 'C001', 5)
        self.store.set_lifecycle_status('dropped', 'dropped')
        result = build_development_snapshot(self.store, self.workspace, note='test protocol')
        self.assertEqual(result.path.parent, self.workspace / 'snapshots')
        self.assertTrue((result.path / 'manifest.json').is_file())
        self.assertTrue((result.path / 'dataset.npz').is_file())
        manifest = json.loads((result.path / 'manifest.json').read_text())
        self.assertEqual(manifest['snapshot_id'], result.snapshot_id)
        self.assertEqual(manifest['snapshot_kind'], 'development')
        self.assertEqual(manifest['note'], 'test protocol')
        self.assertEqual(manifest['sample_ids'], ['a', 'b'])
        self.assertEqual(len(manifest['validation_folds']), 2)
        self.assertIn('unreviewed_samples', manifest['warnings'])
        with np.load(result.path / 'dataset.npz', allow_pickle=False) as arrays:
            self.assertEqual(arrays['features'].shape, (2, 69))
            self.assertEqual(arrays['features'].dtype, np.float32)
            self.assertEqual(arrays['sample_ids'].tolist(), ['a', 'b'])
            self.assertEqual(arrays['session_ids'].tolist(), ['S001', 'S002'])
            self.assertEqual([manifest['label_order'][i] for i in arrays['label_indices']],
                             ['Index_Finger', 'Fist'])
        self.assertTrue(verify_snapshot(result.path))

    def test_p003_is_never_included_in_development_even_with_explicit_scope(self):
        self.store.create_participant('P003')
        self.store.create_session('S003', 'P003')
        self.store.create_capture('C003', 'S003', 'Idle', 'Right')
        self.sample('dev', 'C001', 1)
        self.sample('sealed', 'C003', 2)
        result = build_development_snapshot(
            self.store, self.workspace,
            development_session_ids={'S001', 'S003'},
        )
        manifest = json.loads((result.path / 'manifest.json').read_text())
        self.assertEqual(manifest['sample_ids'], ['dev'])
        self.assertNotIn('P003', manifest['participant_ids'])
        self.assertNotIn('S003', manifest['session_ids'])

    def test_curating_later_creates_new_version_without_rewriting_prior_snapshot(self):
        self.sample('a', 'C001', 1)
        self.sample('b', 'C002', 2)
        first = build_development_snapshot(self.store, self.workspace)
        first_manifest = (first.path / 'manifest.json').read_bytes()
        first_data = (first.path / 'dataset.npz').read_bytes()
        self.store.set_review_status('b', 'rejected')
        second = build_development_snapshot(self.store, self.workspace)
        self.assertNotEqual(first.snapshot_id, second.snapshot_id)
        self.assertEqual((first.path / 'manifest.json').read_bytes(), first_manifest)
        self.assertEqual((first.path / 'dataset.npz').read_bytes(), first_data)
        self.assertEqual(json.loads((second.path / 'manifest.json').read_text())['sample_ids'], ['a'])
        self.assertTrue(verify_snapshot(first.path))
        self.assertTrue(verify_snapshot(second.path))

    def test_same_membership_and_config_produce_the_same_materialized_inputs(self):
        self.sample('a', 'C001', 1)
        self.sample('b', 'C002', 2)
        first = build_development_snapshot(self.store, self.workspace, seed=13)
        second = build_development_snapshot(self.store, self.workspace, seed=13)
        manifest1 = json.loads((first.path / 'manifest.json').read_text())
        manifest2 = json.loads((second.path / 'manifest.json').read_text())
        self.assertEqual(manifest1['membership_sha256'], manifest2['membership_sha256'])
        self.assertEqual(manifest1['data_sha256'], manifest2['data_sha256'])
        self.assertEqual(manifest1['validation_folds'], manifest2['validation_folds'])

    def test_explicit_existing_snapshot_id_never_overwrites_bytes(self):
        self.sample('a', 'C001', 1)
        first = build_development_snapshot(self.store, self.workspace, snapshot_id='stable-id')
        before = (first.path / 'manifest.json').read_bytes()
        with self.assertRaises(FileExistsError):
            build_development_snapshot(self.store, self.workspace, snapshot_id='stable-id')
        self.assertEqual((first.path / 'manifest.json').read_bytes(), before)

    def test_tampering_invalidates_integrity_check(self):
        self.sample('a', 'C001', 1)
        result = build_development_snapshot(self.store, self.workspace)
        with (result.path / 'dataset.npz').open('ab') as output:
            output.write(b'TAMPER')
        self.assertFalse(verify_snapshot(result.path))

    def test_corrupt_membership_digest_or_schema_version_is_rejected(self):
        self.sample('a', 'C001', 1)
        result = build_development_snapshot(self.store, self.workspace)
        manifest_path = result.path / 'manifest.json'
        original = json.loads(manifest_path.read_text())
        corrupted = dict(original, membership_sha256='0' * 64)
        manifest_path.write_text(json.dumps(corrupted))
        self.assertFalse(verify_snapshot(result.path))
        manifest_path.write_text(json.dumps(dict(original, snapshot_format_version=999)))
        self.assertFalse(verify_snapshot(result.path))

    def test_blockers_do_not_create_snapshot_directory(self):
        with self.assertRaisesRegex(ValueError, 'no_eligible_samples'):
            build_development_snapshot(self.store, self.workspace)
        self.assertFalse((self.workspace / 'snapshots').exists())

    def test_compatible_legacy_csv_is_frozen_in_separate_partition(self):
        self.sample('a', 'C001', 1)
        self.sample('b', 'C002', 2)
        legacy_path = self.workspace / 'legacy.csv'
        with legacy_path.open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([f'feat_{i}' for i in range(69)] + ['handedness', 'label'])
            writer.writerow([0.05] * 69 + [0.0, 'Fist'])
            writer.writerow([0.07] * 69 + [1.0, 'Idle'])
        result = build_development_snapshot(
            self.store, self.workspace, legacy_csv=legacy_path
        )
        manifest = json.loads((result.path / 'manifest.json').read_text())
        self.assertEqual(manifest['legacy']['count'], 2)
        self.assertTrue(manifest['legacy']['included'])
        self.assertEqual(manifest['legacy']['data_file'], 'legacy.npz')
        with np.load(result.path / 'legacy.npz', allow_pickle=False) as data:
            self.assertEqual(data['features'].shape, (2, 69))
            self.assertEqual([manifest['label_order'][i] for i in data['label_indices']],
                             ['Fist', 'Idle'])
        with np.load(result.path / 'dataset.npz', allow_pickle=False) as data:
            self.assertEqual(data['sample_ids'].tolist(), ['a', 'b'])
        self.assertTrue(verify_snapshot(result.path))

    def test_malformed_legacy_rows_block_without_creating_snapshot(self):
        self.sample('a', 'C001', 1)
        legacy_path = self.workspace / 'legacy-invalid.csv'
        with legacy_path.open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([f'feat_{i}' for i in range(69)] + ['handedness', 'label'])
            writer.writerow([float('nan')] + [0.05] * 68 + [0.0, 'Fist'])
        with self.assertRaisesRegex(ValueError, 'legacy'):
            build_development_snapshot(self.store, self.workspace, legacy_csv=legacy_path)
        self.assertFalse((self.workspace / 'snapshots').exists())

    def test_cli_builds_and_verifies_without_launching_training(self):
        self.sample('a', 'C001', 1)
        self.sample('b', 'C002', 2)
        command = [sys.executable, '-m', 'handd_core.snapshot_cli',
                   '--workspace', str(self.workspace), 'build', '--note', 'demo']
        output = subprocess.run(command, capture_output=True, text=True, timeout=20)
        self.assertEqual(output.returncode, 0, output.stderr)
        snapshots = list((self.workspace / 'snapshots').iterdir())
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(json.loads((snapshots[0] / 'manifest.json').read_text())['note'], 'demo')
        verification = subprocess.run(
            [sys.executable, '-m', 'handd_core.snapshot_cli', '--workspace',
             str(self.workspace), 'verify', snapshots[0].name],
            capture_output=True, text=True, timeout=20,
        )
        self.assertEqual(verification.returncode, 0, verification.stderr)
        self.assertIn('VERIFIED', verification.stdout)


if __name__ == '__main__':
    unittest.main()
