"""HD-07 public seam: immutable snapshot → grouped OOF → final Model Artifact."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from handd_core.dataset_store import DatasetStore
from handd_core.model_training import TrainingBlocked, run_development_experiment
from handd_core.model_artifact import ModelArtifactError, load_model_artifact
from handd_core.snapshot_builder import build_development_snapshot
from tests.test_feature_transform import landmark_fixture


class DevelopmentTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = DatasetStore(self.root / 'handd.sqlite')
        self.addCleanup(self.store.close)
        self.store.create_participant('P001')
        # Two independent sessions, with multiple examples from each class.
        for sid in ('S001', 'S002'):
            self.store.create_session(sid, 'P001')
        for si, sid in enumerate(('S001', 'S002')):
            for cls, label in enumerate(('Fist', 'Index_Finger', 'Ruler', 'Thumb_Up', 'Idle')):
                cid = f'C-{si}-{cls}'
                self.store.create_capture(cid, sid, label, 'Right')
                for j in range(2):
                    points = landmark_fixture().copy()
                    points[8, 1] -= (cls + 1) * 0.004 + j * 0.001 + si * 0.0002
                    sample_id = f'sample-{si}-{cls}-{j}'
                    self.store.add_sample(
                        sample_id, cid, j, points, points,
                        raw_mp_handedness='Left', timestamp_ms=j * 100,
                        provenance={'device_id': 'synthetic-fixture'},
                    )
                    self.store.set_review_status(sample_id, 'accepted')
        self.snapshot = build_development_snapshot(self.store, self.root)

    def test_session_oof_and_final_refit_artifact_without_leakage(self):
        result = run_development_experiment(
            self.snapshot.path, self.root, epochs=2, seed=31,
            learning_curve_fractions=(0.5, 1.0),
        )
        report = json.loads((result.report_dir / 'metrics.json').read_text())
        self.assertEqual(report['snapshot_id'], self.snapshot.snapshot_id)
        self.assertEqual(report['evaluation_kind'], 'session_grouped_development_oof')
        self.assertEqual(set(report['variants']), {'without_legacy'})
        variant = report['variants']['without_legacy']
        self.assertEqual(len(variant['folds']), 2)
        self.assertEqual(len(variant['oof']), 20)
        self.assertEqual(set(row['sample_id'] for row in variant['oof']),
                         set(json.loads((self.snapshot.path / 'manifest.json').read_text())['sample_ids']))
        self.assertEqual(len(set(row['sample_id'] for row in variant['oof'])), 20)
        for fold in variant['folds']:
            self.assertFalse(set(fold['train_sample_ids']) & set(fold['validation_sample_ids']))
            self.assertTrue(set(fold['validation_session_ids']).isdisjoint(fold['train_session_ids']))
            self.assertNotIn(fold['heldout_session_id'], fold['train_session_ids'])
            self.assertEqual(len(fold['learning_curve']), 2)
            self.assertEqual({entry['fraction'] for entry in fold['learning_curve']}, {0.5, 1.0})
        self.assertEqual(len(variant['confusion_matrix']), 5)
        self.assertEqual(len(variant['per_class']), 5)
        self.assertGreaterEqual(variant['macro_f1'], 0.0)
        self.assertLessEqual(variant['macro_f1'], 1.0)
        model, manifest = load_model_artifact(result.model_dir)
        self.assertEqual(manifest['artifact_role'], 'final_refit_candidate')
        self.assertEqual(manifest['input_features'], 69)
        self.assertEqual(len(manifest['label_order']), 5)
        self.assertFalse(manifest['legacy_in_final_refit'])
        self.assertEqual(model(np.zeros((1, 69), dtype=np.float32)).shape, (1, 5))
        self.assertFalse((self.root / 'active-model.json').exists())
        self.assertFalse((self.root / 'models' / 'active-model.json').exists())

    def test_legacy_comparison_has_identical_v2_folds_and_never_legacy_oof(self):
        csv_path = self.root / 'legacy.csv'
        with csv_path.open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([f'feat_{i}' for i in range(69)] + ['handedness', 'label'])
            for label in ('Fist', 'Index_Finger', 'Ruler', 'Thumb_Up', 'Idle'):
                writer.writerow([0.05] * 69 + [0, label])
        snapshot = build_development_snapshot(self.store, self.root, legacy_csv=csv_path)
        result = run_development_experiment(
            snapshot.path, self.root, epochs=1, seed=9,
            learning_curve_fractions=(1.0,), final_legacy=True,
        )
        report = json.loads((result.report_dir / 'metrics.json').read_text())
        normal = report['variants']['without_legacy']
        augmented = report['variants']['with_legacy']
        self.assertEqual(len(normal['oof']), 20)
        self.assertEqual(len(augmented['oof']), 20)
        self.assertEqual([r['sample_id'] for r in normal['oof']],
                         [r['sample_id'] for r in augmented['oof']])
        for no_legacy, with_legacy in zip(normal['folds'], augmented['folds']):
            self.assertEqual(no_legacy['train_sample_ids'], with_legacy['train_sample_ids'])
            self.assertEqual(no_legacy['validation_sample_ids'], with_legacy['validation_sample_ids'])
            self.assertEqual(no_legacy['legacy_training_sample_count'], 0)
            self.assertEqual(with_legacy['legacy_training_sample_count'], 5)
        _, manifest = load_model_artifact(result.model_dir)
        self.assertTrue(manifest['legacy_in_final_refit'])
        self.assertEqual(manifest['refit_v2_sample_count'], 20)
        self.assertEqual(manifest['refit_legacy_row_count'], 5)

    def test_training_refuses_single_session_and_does_not_create_candidate(self):
        one_session = build_development_snapshot(
            self.store, self.root, development_session_ids={'S001'},
        )
        with self.assertRaisesRegex(TrainingBlocked, 'two distinct'):
            run_development_experiment(one_session.path, self.root, epochs=1)
        self.assertFalse((self.root / 'models').exists())

    def test_training_refuses_tampered_validation_folds(self):
        path = self.snapshot.path / 'manifest.json'
        manifest = json.loads(path.read_text())
        manifest['validation_folds'][0]['train_session_ids'] = ['S001']
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(TrainingBlocked, 'folds'):
            run_development_experiment(self.snapshot.path, self.root, epochs=1)
        self.assertFalse((self.root / 'models').exists())

    def test_artifact_reader_rejects_modified_weights_and_mismatched_transform(self):
        result = run_development_experiment(
            self.snapshot.path, self.root, epochs=1, learning_curve_fractions=(1.0,),
        )
        manifest_file = result.model_dir / 'manifest.json'
        manifest = json.loads(manifest_file.read_text())
        manifest_file.write_text(json.dumps({**manifest, 'feature_transform': 'wrong-contract'}))
        with self.assertRaises(ModelArtifactError):
            load_model_artifact(result.model_dir)
        manifest_file.write_text(json.dumps(manifest))
        with (result.model_dir / 'weights.pth').open('ab') as out:
            out.write(b'tampered')
        with self.assertRaises(ModelArtifactError):
            load_model_artifact(result.model_dir)

    def test_explicit_cli_training_from_snapshot_without_touching_active_model(self):
        command = [
            sys.executable, '-m', 'handd_core.train_cli', '--workspace', str(self.root),
            '--snapshot', self.snapshot.snapshot_id, '--epochs', '1',
            '--fractions', '1.0', '--seed', '15',
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('CANDIDATE', result.stdout)
        self.assertEqual(len(list((self.root / 'models').glob('candidate-*'))), 1)
        self.assertEqual(len(list((self.root / 'reports').glob('experiment-*'))), 1)
        self.assertFalse((self.root / 'models' / 'active-model.json').exists())

    def test_repeat_with_identical_seed_has_identical_predictions_and_weights(self):
        args = dict(epochs=1, seed=21, learning_curve_fractions=(1.0,))
        first = run_development_experiment(self.snapshot.path, self.root, **args)
        second = run_development_experiment(self.snapshot.path, self.root, **args)
        report_a = json.loads((first.report_dir / 'metrics.json').read_text())
        report_b = json.loads((second.report_dir / 'metrics.json').read_text())
        predictions_a = [r['predicted_label'] for r in report_a['variants']['without_legacy']['oof']]
        predictions_b = [r['predicted_label'] for r in report_b['variants']['without_legacy']['oof']]
        self.assertEqual(predictions_a, predictions_b)
        import torch
        state_a = torch.load(first.model_dir / 'weights.pth', weights_only=True)
        state_b = torch.load(second.model_dir / 'weights.pth', weights_only=True)
        self.assertTrue(all(torch.equal(state_a[key], state_b[key]) for key in state_a))


if __name__ == '__main__':
    unittest.main()
