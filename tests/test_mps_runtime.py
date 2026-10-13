"""MPS stays explicit on the PyTorch v2 branch and fails back to CPU safely."""
import asyncio
import io
import json
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from handd_core.model_artifact import GestureMLP, ModelPredictor
from handd_core.runtime_v2 import GestureRuntime
from handd_core.sidecar_main import parser
from handd_core.studio_models import StudioModels
from handd_core.dataset_store import DatasetStore
from tests.test_runtime_v2 import observation
from tests.test_studio_models import create_candidate
import tempfile
import sys


def reference_model():
    model=GestureMLP(5)
    with torch.no_grad():
        for param in model.parameters():param.zero_()
        model.output.bias[0]=10
    return ModelPredictor(model, ('Fist','Index_Finger','Ruler','Thumb_Up','Idle'))


class MpsRuntimeTests(unittest.TestCase):
    def test_cpu_is_still_default_on_v2_and_override_is_explicit(self):
        with patch.dict(os.environ,{},clear=True):
            self.assertEqual(parser().parse_args([]).inference_device, 'cpu')
        self.assertEqual(parser().parse_args(['--inference-device','mps']).inference_device, 'mps')
        self.assertEqual(reference_model().backend, 'torch-cpu')

    def test_mps_unavailable_reports_cpu_fallback_without_blocking_camera(self):
        from handd_core.mps_runtime import select_inference_device
        reference=reference_model()
        with patch('handd_core.mps_runtime.prepare_mps_predictor',side_effect=RuntimeError('MPS not available')):
            predictor=select_inference_device(reference,'mps')
        self.assertEqual(predictor.backend,'torch-cpu-mps-fallback')
        runtime=GestureRuntime(predictor=predictor,stable_frames=1,stable_ms=0)
        result=runtime.process_result(observation('Left'),100)
        self.assertEqual(result['payload']['health']['model'],'ready')
        self.assertEqual(result['payload']['health']['inference_backend'],'torch-cpu-mps-fallback')
        self.assertEqual(result['payload']['drawing']['action'],'erase')

    def test_gpu_error_releases_current_erase_frame_and_retrains_stability_on_cpu(self):
        from handd_core.mps_runtime import MpsPredictor
        reference=reference_model()
        class Executor:
            fail=False
            calls=0
            def __call__(self,features):
                self.calls+=1
                if self.fail:raise RuntimeError('Metal execution failed')
                return reference(features)
        executor=Executor()
        runtime=GestureRuntime(predictor=MpsPredictor(reference,executor),
                               stable_frames=2,stable_ms=20)
        frame=observation('Left','Right')
        runtime.process_result(frame,100)
        self.assertEqual(runtime.process_result(frame,140)['payload']['drawing']['action'],'erase')
        executor.fail=True
        failed=runtime.process_result(frame,180)
        self.assertEqual(failed['payload']['health']['model'],'error')
        self.assertEqual(failed['payload']['health']['inference_backend'],'torch-cpu-mps-fallback')
        for role in ('drawing','modifier'):
            self.assertIsNone(failed['payload'][role]['action'])
        first=runtime.process_result(frame,220)
        self.assertIsNone(first['payload']['drawing']['action'])
        self.assertEqual(first['payload']['health']['model'],'ready')
        recovered=runtime.process_result(frame,260)
        self.assertEqual(recovered['payload']['drawing']['action'],'erase')
        self.assertEqual(executor.calls,5,'the failed MPS executor is never called again')

    def test_gpu_nonfinite_result_never_dispatches_destructive_action(self):
        from handd_core.mps_runtime import MpsPredictor
        reference=reference_model()
        predictor=MpsPredictor(reference,lambda features:np.full((len(features),5),np.nan))
        with self.assertRaises(RuntimeError):
            predictor.predict(np.zeros((1,69),np.float32))
        self.assertEqual(predictor.backend,'torch-cpu-mps-fallback')

    def test_studio_restores_verified_candidate_with_mps_fallback_when_unavailable(self):
        with tempfile.TemporaryDirectory() as folder:
            store=DatasetStore(Path(folder)/'handd.sqlite');store.close()
            create_candidate(folder)
            models=StudioModels(folder,inference_device='mps')
            with patch('handd_core.mps_runtime.prepare_mps_predictor',
                       side_effect=RuntimeError('MPS not built')):
                runtime=GestureRuntime()
                models.activate('candidate-001',runtime)
                self.assertEqual(runtime.snapshot()['health']['inference_backend'],
                                 'torch-cpu-mps-fallback')
                restored=GestureRuntime()
                models.restore(restored)
                self.assertEqual(restored.snapshot()['health']['inference_backend'],
                                 'torch-cpu-mps-fallback')
                self.assertEqual(restored.snapshot()['active_model_id'],'candidate-001')

    def test_mps_benchmark_unavailable_is_honest_and_machine_readable(self):
        from handd_core.benchmark_mps import main
        buffer=io.StringIO()
        with patch('handd_core.benchmark_mps.torch.backends.mps.is_available',return_value=False):
            with redirect_stdout(buffer):
                result=main(['--warmup','1','--iterations','2'])
        self.assertEqual(result,2)
        output=json.loads(buffer.getvalue())
        self.assertEqual(output['status'],'unavailable')
        self.assertFalse(output['mps']['available'])

    def test_mps_benchmark_reports_gpu_initialization_failure_as_json(self):
        from handd_core.benchmark_mps import main
        buffer=io.StringIO()
        with (patch('handd_core.benchmark_mps.torch.backends.mps.is_available',return_value=True),
              patch('handd_core.benchmark_mps.load_legacy_checkpoint',
                    return_value=(reference_model(),{})),
              patch('handd_core.benchmark_mps.prepare_mps_predictor',
                    side_effect=RuntimeError('Metal unavailable for model')),
              redirect_stdout(buffer)):
            result=main(['--iterations','2','--warmup','1'])
        self.assertEqual(result,2)
        response=json.loads(buffer.getvalue())
        self.assertEqual(response['status'],'failed')
        self.assertIn('Metal',response['error'])

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS requires a macOS Metal device')
    def test_real_mps_execution_matches_cpu_labels(self):
        from handd_core.mps_runtime import select_inference_device
        reference=reference_model()
        predictor=select_inference_device(reference,'mps')
        self.assertEqual(predictor.backend,'mps')
        for batch_size in (1,8,32):
            points=np.random.default_rng(batch_size).normal(0,.35,(batch_size,69)).astype(np.float32)
            np.testing.assert_allclose(predictor(points),reference(points),rtol=1e-4,atol=1e-4)
            self.assertEqual(predictor.predict(points),reference.predict(points))


class MpsSidecarProcessTests(unittest.IsolatedAsyncioTestCase):
    async def test_sidecar_reports_explicit_mps_or_cpu_fallback(self):
        from aiohttp import ClientSession
        checkpoint=Path(__file__).resolve().parents[1]/'models/gesture_mlp.pth'
        child=await asyncio.create_subprocess_exec(
            sys.executable,'-u','-m','handd_core.sidecar_main',
            '--no-camera','--legacy-checkpoint',str(checkpoint),
            '--inference-device','mps',
            stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE,
        )
        try:
            raw=await asyncio.wait_for(child.stdout.readline(),20)
            if not raw:
                child.terminate()
                self.fail('MPS sidecar did not announce READY: '
                          +(await child.stderr.read()).decode()[:200])
            ready=json.loads(raw)
            async with ClientSession() as session:
                async with session.get(
                    f"http://127.0.0.1:{ready['port']}/health",
                    headers={'Authorization':'Bearer '+ready['token']},
                ) as response:
                    self.assertEqual(response.status,200)
                    snapshot=(await response.json())['snapshot']
                    self.assertEqual(snapshot['health']['model'],'legacy_unverified')
                    self.assertIn(snapshot['health']['inference_backend'],
                                  ('mps','torch-cpu-mps-fallback')
                                  if torch.backends.mps.is_available()
                                  else ('torch-cpu-mps-fallback',))
        finally:
            if child.returncode is None:child.terminate()
            await asyncio.wait_for(child.wait(),8)


if __name__=='__main__':unittest.main()
