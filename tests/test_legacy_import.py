"""Import transformed historical observations without forging v2 raw Samples."""
import csv
import json
import sqlite3
from contextlib import closing
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from handd_core.dataset_store import DatasetStore
from handd_core.legacy_import import import_legacy_csv, list_legacy_sources, load_legacy_source
from handd_core.snapshot_builder import build_development_snapshot, verify_snapshot
from tests.test_feature_transform import landmark_fixture


class LegacyImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.store = DatasetStore(self.root / 'handd.sqlite')
        self.addCleanup(self.store.close)

    def csv(self, name, records):
        path = self.root / name
        with path.open('w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([f'feat_{i}' for i in range(69)] + ['handedness', 'label'])
            for value, hand, label in records:
                writer.writerow([value] * 69 + [hand, label])
        return path

    def test_import_is_idempotent_feature_only_and_preserves_labels(self):
        source = self.csv('new.csv', [(0.125, 0.0, 'Fist'), (0.375, 1.0, 'Idle')])
        first = import_legacy_csv(self.store, source)
        second = import_legacy_csv(self.store, source)
        self.assertEqual(first.source_id, second.source_id)
        self.assertEqual(first.count, 2)
        self.assertEqual(first.status, 'compatible')
        self.assertEqual(len(list_legacy_sources(self.store)), 1)
        self.assertEqual(self.store.count_samples(), 0)
        with self.store.conn:
            self.assertGreaterEqual(self.store.conn.execute('PRAGMA user_version').fetchone()[0], 2)
        features, labels, hands, row_ids = load_legacy_source(self.store, first.source_id)
        self.assertEqual(features.shape, (2, 69))
        self.assertEqual(features.dtype, np.float32)
        self.assertEqual(labels, ['Fist', 'Idle'])
        self.assertEqual(hands.tolist(), [0., 1.])
        self.assertEqual(len(set(row_ids)), 2)

    def test_legacy_four_class_is_quarantined_without_relabeling(self):
        source = self.csv('older.csv', [(0.25, 0.0, 'Ruler_Gesture')])
        result = import_legacy_csv(self.store, source)
        self.assertEqual(result.status, 'quarantined')
        self.assertEqual(load_legacy_source(self.store, result.source_id, compatible_only=False)[1],
                         ['Ruler_Gesture'])
        with self.assertRaisesRegex(ValueError, 'quarantined'):
            load_legacy_source(self.store, result.source_id)

    def test_malformed_rows_roll_back_entire_import(self):
        source = self.csv('bad.csv', [(0.1, 0, 'Idle'), (float('nan'), 1, 'Fist')])
        with self.assertRaisesRegex(ValueError, 'row'):
            import_legacy_csv(self.store, source)
        self.assertEqual(list_legacy_sources(self.store), [])
        self.assertEqual(self.store.count_samples(), 0)

    def test_core_legacy_rows_are_immutable_even_via_sql(self):
        source = self.csv('new.csv', [(0.125, 0, 'Fist')])
        result = import_legacy_csv(self.store, source)
        with self.assertRaises(sqlite3.IntegrityError):
            with self.store.conn:
                self.store.conn.execute(
                    'UPDATE legacy_rows SET original_label=? WHERE source_id=?',
                    ('Idle', result.source_id),
                )
        with self.assertRaises(sqlite3.IntegrityError):
            with self.store.conn:
                self.store.conn.execute(
                    'DELETE FROM legacy_sources WHERE source_id=?', (result.source_id,)
                )

    def test_real_v1_schema_upgrades_and_preserves_curated_sample(self):
        # This fixture is the original SQL from commit 9458984 (schema v1),
        # not a v2 database with its version number manually changed.
        historical = self.root / 'historical.sqlite'
        ddl = (Path(__file__).parent / 'fixtures' / 'handd_schema_v1.sql').read_text()
        with closing(sqlite3.connect(historical)) as conn, conn:
            conn.executescript(ddl)
            conn.execute('PRAGMA user_version=1')
            conn.execute("INSERT INTO participants VALUES ('P001','2026-10-08')")
            conn.execute("INSERT INTO collection_sessions VALUES ('S001','P001','2026-10-08')")
            conn.execute("INSERT INTO captures VALUES ('C001','S001','Fist','Right',120,'2026-10-08')")
            points = json.dumps(landmark_fixture().tolist())
            conn.execute(
                "INSERT INTO samples(sample_id,capture_id,frame_index,timestamp_ms,"
                "raw_mp_handedness,image_landmarks,world_landmarks,provenance_json,recorded_at)"
                " VALUES(?,?,?,?,?,?,?,?,?)",
                ('sample-v1','C001',0,100,'Left',points,points,'{}','2026-10-08'),
            )
            conn.execute("UPDATE samples SET review_status='accepted' WHERE sample_id='sample-v1'")
        reopened = DatasetStore(historical)
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.conn.execute('PRAGMA user_version').fetchone()[0], 2)
        self.assertEqual(reopened.conn.execute('SELECT count(*) FROM participants').fetchone()[0], 1)
        self.assertEqual(reopened.get_sample('sample-v1')['review_status'], 'accepted')
        self.assertEqual(len(reopened.list_review_events('sample-v1')), 1)
        self.assertTrue(reopened.conn.execute(
            "SELECT name FROM sqlite_master WHERE name='legacy_sources'").fetchone())

    def test_snapshot_can_freeze_imported_compatible_source_without_csv(self):
        source = self.csv('new.csv', [(0.125, 0.0, 'Fist'), (0.375, 1.0, 'Idle')])
        imported = import_legacy_csv(self.store, source)
        source.unlink()
        self.store.create_participant('P001')
        self.store.create_session('S001', 'P001')
        self.store.create_capture('C001', 'S001', 'Fist', 'Right')
        pts = landmark_fixture()
        self.store.add_sample('new-v2', 'C001', 0, pts, pts,
                              raw_mp_handedness='Left', timestamp_ms=100, provenance={})
        self.store.set_review_status('new-v2', 'accepted')
        result = build_development_snapshot(
            self.store, self.root, legacy_source_id=imported.source_id,
        )
        import json
        manifest = json.loads((result.path / 'manifest.json').read_text())
        self.assertEqual(manifest['legacy']['count'], 2)
        self.assertEqual(manifest['legacy']['source_sha256'], imported.source_id)
        self.assertTrue(verify_snapshot(result.path))
        with np.load(result.path / 'dataset.npz', allow_pickle=False) as data:
            self.assertEqual(data['sample_ids'].tolist(), ['new-v2'])
        with np.load(result.path / 'legacy.npz', allow_pickle=False) as data:
            self.assertEqual(data['features'].shape, (2, 69))

    def test_snapshot_rejects_quarantined_source(self):
        source = self.csv('old.csv', [(0.1, 0, 'Ruler_Gesture')])
        quarantined = import_legacy_csv(self.store, source)
        with self.assertRaisesRegex(ValueError, 'quarantined'):
            build_development_snapshot(self.store, self.root,
                                       legacy_source_id=quarantined.source_id)
        self.assertFalse((self.root / 'snapshots').exists())

    def test_import_and_inventory_commands_work_without_camera(self):
        source = self.csv('new.csv', [(0.125, 0, 'Fist')])
        base = [sys.executable, '-m', 'handd_core.legacy_cli',
                '--workspace', str(self.root)]
        imported = subprocess.run(base + ['import', str(source)],
                                  capture_output=True, text=True, timeout=15)
        self.assertEqual(imported.returncode, 0, imported.stderr)
        self.assertIn('compatible', imported.stdout)
        inventory = subprocess.run(base + ['list'], capture_output=True,
                                   text=True, timeout=15)
        self.assertEqual(inventory.returncode, 0, inventory.stderr)
        self.assertIn('new.csv', inventory.stdout)
        self.assertEqual(len(list_legacy_sources(self.store)), 1)
