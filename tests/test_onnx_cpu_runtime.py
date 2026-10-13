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

from handd_core.dataset_store import DatasetStore
from handd_core.model_artifact import load_model_artifact
from handd_core.studio_models import StudioModels
from handd_core.runtime_v2 import GestureRuntime
from tests.test_studio_models import create_candidate


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
    @unittest.skipUnless(ONNX_READY, 'Optional ONNX dependencies not installed')
    async def test_sidecar_legacy_cpu_onnx_is_opt_in_and_reports_backend(self):
        checkpoint = Path(__file__).resolve().parents[1] / 'models/gesture_mlp.pth'
        env = {**os.environ, 'HANDD_INFERENCE_BACKEND': 'onnx-cpu'}
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
