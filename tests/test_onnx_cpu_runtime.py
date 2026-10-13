"""Experimental ONNX CPU backend at the same predictor/Studio/sidecar seams."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from handd_core.dataset_store import DatasetStore
from handd_core.model_artifact import GestureMLP, ModelPredictor, load_model_artifact
from handd_core.studio_models import StudioModels
from handd_core.runtime_v2 import GestureRuntime
from handd_core.sidecar_main import parser
from tests.test_studio_models import create_candidate
from tests.test_runtime_v2 import observation


ONNX_READY = all(importlib.util.find_spec(package) is not None
                 for package in ('onnx', 'onnxruntime', 'onnxscript'))


class OnnxCpuSelectionTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.workspace = Path(tmp.name)
        db = DatasetStore(self.workspace / 'handd.sqlite')
        db.close()
        self.candidate = create_candidate(self.workspace)

    def test_default_stays_pytorch_cpu(self):
        predictor, _ = load_model_artifact(self.candidate)
        self.assertEqual(predictor.backend, 'torch-cpu')

    def test_experimental_sidecar_defaults_to_onnx_but_allows_torch_override(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(parser().parse_args(['--no-camera']).inference_backend, 'onnx-cpu')
            explicit = parser().parse_args(['--inference-backend', 'torch-cpu'])
            self.assertEqual(explicit.inference_backend, 'torch-cpu')
        with patch.dict(os.environ, {'HANDD_INFERENCE_BACKEND': 'torch-cpu'}):
            self.assertEqual(parser().parse_args([]).inference_backend, 'torch-cpu')

    def test_midstroke_onnx_failure_releases_both_hands_then_recovers_on_torch(self):
        from handd_core.onnx_cpu import OnnxCpuPredictor

        labels = ('Fist', 'Index_Finger', 'Ruler', 'Thumb_Up', 'Idle')
        model = GestureMLP(len(labels))
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()
            model.output.bias[0] = 12
        reference = ModelPredictor(model, labels)

        class Session:
            fail = False
            calls = 0

            def run(self, outputs, inputs):
                self.calls += 1
                if self.fail:
                    raise OSError('ONNX CPU execution provider failed')
                return [reference(inputs['features'])]

        session = Session()
        runtime = GestureRuntime(
            predictor=OnnxCpuPredictor(session, reference), stable_frames=2, stable_ms=20,
        )
        frame = observation('Left', 'Right')
        runtime.process_result(frame, 100)
        before = runtime.process_result(frame, 140)
        self.assertEqual(before['payload']['drawing']['action'], 'erase')
        self.assertEqual(before['payload']['health']['inference_backend'], 'onnx-cpu')

        session.fail = True
        failed = runtime.process_result(frame, 180)
        self.assertEqual(failed['payload']['health']['inference_backend'], 'torch-cpu-fallback')
        self.assertEqual(failed['payload']['health']['model'], 'error')
        for role in ('drawing', 'modifier'):
            self.assertIsNone(failed['payload'][role]['action'])
            self.assertIsNone(failed['payload'][role]['stable_gesture'])

        first_recovery = runtime.process_result(frame, 220)
        self.assertEqual(first_recovery['payload']['health']['model'], 'ready')
        self.assertIsNone(first_recovery['payload']['drawing']['action'])
        recovered = runtime.process_result(frame, 260)
        self.assertEqual(recovered['payload']['drawing']['action'], 'erase')
        self.assertEqual(recovered['payload']['health']['inference_backend'], 'torch-cpu-fallback')
        self.assertEqual(session.calls, 5, 'After failover ONNX must not be called again')

    def test_nonfinite_onnx_output_fails_closed_instead_of_selecting_a_gesture(self):
        from handd_core.onnx_cpu import OnnxCpuPredictor
        reference, _ = load_model_artifact(self.candidate)

        class BadSession:
            def run(self, outputs, inputs):
                return [np.full((len(inputs['features']), len(reference.label_order)),
                                np.nan, dtype=np.float32)]

        predictor = OnnxCpuPredictor(BadSession(), reference)
        with self.assertRaisesRegex(RuntimeError, 'release current gestures'):
            predictor.predict(np.zeros((1, 69), dtype=np.float32))
        self.assertEqual(predictor.backend, 'torch-cpu-fallback')
        self.assertEqual(predictor.predict(np.zeros((1, 69), np.float32)),
                         reference.predict(np.zeros((1, 69), np.float32)))

    @unittest.skipUnless(ONNX_READY, 'Optional ONNX dependencies not installed')
    def test_onnx_cpu_matches_verified_candidate_and_rejects_bad_inputs(self):
        from handd_core.onnx_cpu import prepare_predictor
        reference, _ = load_model_artifact(self.candidate)
        active = prepare_predictor(reference)
        self.assertEqual(active.backend, 'onnx-cpu')
        self.assertEqual(active.session.get_providers(), ['CPUExecutionProvider'])
        self.assertEqual(active.label_order, reference.label_order)
        rng = np.random.default_rng(1709)
        for batch in (np.zeros((1, 69), np.float32),
                      rng.normal(0, .25, size=(7, 69)).astype(np.float32)):
            np.testing.assert_allclose(active(batch), reference(batch), rtol=0, atol=1e-4)
            self.assertEqual(active.predict(batch), reference.predict(batch))
        for invalid in (np.ones((1, 68)), np.full((1, 69), np.nan)):
            with self.assertRaisesRegex(ValueError, 'Nx69'):
                active(invalid)
        self.assertEqual(active.backend, 'onnx-cpu',
                         'Invalid features must not disable a healthy ONNX session')

    @unittest.skipUnless(ONNX_READY, 'Optional ONNX dependencies not installed')
    def test_reactivation_reuses_verified_session_but_changed_weights_reexport(self):
        from handd_core.onnx_cpu import prepare_predictor
        first_reference, _ = load_model_artifact(self.candidate)
        with patch('handd_core.onnx_cpu.torch.onnx.export', wraps=torch.onnx.export) as exporter:
            first = prepare_predictor(first_reference)
            same_reference, _ = load_model_artifact(self.candidate)
            second = prepare_predictor(same_reference)
            self.assertIs(first.session, second.session)
            self.assertIsNot(first.reference, second.reference)
            self.assertEqual(exporter.call_count, 1)

            with torch.no_grad():
                same_reference.model.output.bias[0] += 1.5
            updated = prepare_predictor(same_reference)
            self.assertIsNot(first.session, updated.session)
            self.assertEqual(exporter.call_count, 2)

    @unittest.skipUnless(ONNX_READY, 'Optional ONNX dependencies not installed')
    def test_failed_cached_session_is_not_reused_on_next_activation(self):
        from handd_core.onnx_cpu import prepare_predictor
        reference, _ = load_model_artifact(self.candidate)
        first = prepare_predictor(reference)
        with patch.object(first.session, 'run', side_effect=OSError('provider lost')):
            with self.assertRaises(RuntimeError):
                first.predict(np.zeros((1, 69), dtype=np.float32))
        repeated_reference, _ = load_model_artifact(self.candidate)
        another = prepare_predictor(repeated_reference)
        self.assertIsNot(first.session, another.session)
        self.assertEqual(another.backend, 'onnx-cpu')

    @unittest.skipUnless(ONNX_READY, 'Optional ONNX dependencies not installed')
    def test_cache_evicts_oldest_of_four_distinct_model_weights(self):
        from handd_core.onnx_cpu import prepare_predictor
        references = []
        for offset in range(4):
            reference, _ = load_model_artifact(self.candidate)
            with torch.no_grad():
                reference.model.output.bias[0] += (offset + 1) * 2
            references.append(reference)
        with patch('handd_core.onnx_cpu.torch.onnx.export', wraps=torch.onnx.export) as exporter:
            sessions = [prepare_predictor(ref).session for ref in references]
            self.assertEqual(exporter.call_count, 4)
            self.assertIs(prepare_predictor(references[-1]).session, sessions[-1])
            self.assertEqual(exporter.call_count, 4)
            self.assertIsNot(prepare_predictor(references[0]).session, sessions[0])
            self.assertEqual(exporter.call_count, 5)

    def test_invalid_onnx_dependency_or_export_uses_visible_torch_fallback(self):
        from handd_core.onnx_cpu import select_backend
        reference, _ = load_model_artifact(self.candidate)
        with patch('handd_core.onnx_cpu.prepare_predictor', side_effect=RuntimeError('broken export')):
            active = select_backend(reference, 'onnx-cpu')
        self.assertEqual(active.backend, 'torch-cpu-fallback')
        self.assertEqual(active.predict(np.zeros((1, 69), np.float32)),
                         reference.predict(np.zeros((1, 69), np.float32)))

    def test_unrecognized_backend_is_rejected_not_implicitly_activated(self):
        from handd_core.onnx_cpu import select_backend
        reference, _ = load_model_artifact(self.candidate)
        with self.assertRaises(ValueError):
            select_backend(reference, 'mps')

    @unittest.skipUnless(ONNX_READY, 'Optional ONNX dependencies not installed')
    def test_studio_activation_and_restore_both_use_selected_backend(self):
        models = StudioModels(self.workspace, inference_backend='onnx-cpu')
        runtime = GestureRuntime()
        models.activate('candidate-001', runtime)
        self.assertEqual(runtime.snapshot()['health']['inference_backend'], 'onnx-cpu')
        self.assertEqual(runtime.snapshot()['health']['model'], 'ready')
        self.assertEqual(models.active_id(), 'candidate-001')
        restored = GestureRuntime()
        models.restore(restored)
        self.assertEqual(restored.snapshot()['health']['inference_backend'], 'onnx-cpu')
        self.assertEqual(restored.snapshot()['active_model_id'], 'candidate-001')


class OnnxCpuProcessTests(unittest.IsolatedAsyncioTestCase):
    async def test_active_workspace_skips_unneeded_legacy_loading_on_startup(self):
        from aiohttp import ClientSession
        with tempfile.TemporaryDirectory() as folder:
            workspace = Path(folder)
            db = DatasetStore(workspace / 'handd.sqlite')
            db.close()
            create_candidate(workspace, 'candidate-active')
            StudioModels(workspace).activate('candidate-active', GestureRuntime())
            child = await asyncio.create_subprocess_exec(
                sys.executable, '-u', '-m', 'handd_core.sidecar_main',
                '--no-camera', '--workspace', str(workspace),
                '--legacy-checkpoint', str(workspace / 'does-not-exist.pth'),
                '--inference-backend', 'torch-cpu',
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            try:
                line = await asyncio.wait_for(child.stdout.readline(), 15)
                if not line:
                    self.fail('Active Candidate was blocked by legacy: '
                              + (await child.stderr.read()).decode()[:300])
                ready = json.loads(line)
                async with ClientSession() as session:
                    async with session.get(
                        f"http://127.0.0.1:{ready['port']}/health",
                        headers={'Authorization': 'Bearer ' + ready['token']},
                    ) as response:
                        self.assertEqual(response.status, 200)
                        snapshot = (await response.json())['snapshot']
                        self.assertEqual(snapshot['active_model_id'], 'candidate-active')
                        self.assertEqual(snapshot['health']['model'], 'ready')
            finally:
                if child.returncode is None:
                    child.terminate()
                await asyncio.wait_for(child.wait(), 8)

    async def test_broken_active_selection_remains_error_without_legacy_fallback(self):
        from aiohttp import ClientSession
        with tempfile.TemporaryDirectory() as folder:
            workspace = Path(folder)
            db = DatasetStore(workspace / 'handd.sqlite')
            db.close()
            (workspace / 'models').mkdir()
            (workspace / 'models' / '.active-model.json').write_text('invalid-json')
            child = await asyncio.create_subprocess_exec(
                sys.executable, '-u', '-m', 'handd_core.sidecar_main',
                '--no-camera', '--workspace', str(workspace),
                '--legacy-checkpoint', str(workspace / 'missing.pth'),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            try:
                line = await asyncio.wait_for(child.stdout.readline(), 15)
                if not line:
                    self.fail('Corrupt selection crashed sidecar before READY')
                ready = json.loads(line)
                async with ClientSession() as session:
                    async with session.get(
                        f"http://127.0.0.1:{ready['port']}/health",
                        headers={'Authorization': 'Bearer ' + ready['token']},
                    ) as response:
                        self.assertEqual(response.status, 200)
                        snapshot = (await response.json())['snapshot']
                        self.assertIsNone(snapshot['active_model_id'])
                        self.assertEqual(snapshot['health']['model'], 'error')
                        self.assertEqual(snapshot['health']['inference_backend'], 'none')
            finally:
                if child.returncode is None:
                    child.terminate()
                await asyncio.wait_for(child.wait(), 8)

    @unittest.skipUnless(ONNX_READY, 'Optional ONNX dependencies not installed')
    async def test_sidecar_legacy_cpu_onnx_is_default_and_reports_backend(self):
        checkpoint = Path(__file__).resolve().parents[1] / 'models/gesture_mlp.pth'
        env = {key: value for key, value in os.environ.items()
               if key != 'HANDD_INFERENCE_BACKEND'}
        child = await asyncio.create_subprocess_exec(
            sys.executable, '-u', '-m', 'handd_core.sidecar_main',
            '--no-camera', '--legacy-checkpoint', str(checkpoint),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            line = await asyncio.wait_for(child.stdout.readline(), 30)
            if not line:
                child.terminate()
                diagnostics = (await child.stderr.read()).decode()[:500]
                self.fail('Sidecar did not announce READY: ' + diagnostics)
            ready = json.loads(line)
            from aiohttp import ClientSession
            async with ClientSession() as session:
                async with session.get(
                    f"http://127.0.0.1:{ready['port']}/health",
                    headers={'Authorization': 'Bearer ' + ready['token']},
                ) as response:
                    self.assertEqual(response.status, 200)
                    snapshot = (await response.json())['snapshot']
                    self.assertEqual(snapshot['health']['inference_backend'], 'onnx-cpu')
                    self.assertEqual(snapshot['health']['model'], 'legacy_unverified')
        finally:
            if child.returncode is None:
                child.terminate()
            await asyncio.wait_for(child.wait(), 8)


if __name__ == '__main__':
    unittest.main()
