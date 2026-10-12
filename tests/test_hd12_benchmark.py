"""HD-12 deployment benchmark at the verified final-refit Candidate seam."""
import hashlib
import io
from contextlib import redirect_stdout
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch

from handd_core.feature_transform import FEATURE_TRANSFORM_ID
from handd_core.model_artifact import ARCHITECTURE_ID, GestureMLP
from handd_core.benchmark_inference import compare_backends, main


def candidate(root: Path, corrupted: bool = False) -> Path:
    root.mkdir(exist_ok=True)
    torch.manual_seed(17)
    model = GestureMLP(3).eval()
    torch.save(model.state_dict(), root / 'weights.pth')
    (root / 'metrics.json').write_text('{}')
    checksum=lambda name:hashlib.sha256((root/name).read_bytes()).hexdigest()
    manifest={'model_format_version':1,'artifact_role':'final_refit_candidate',
        'architecture':ARCHITECTURE_ID,'feature_transform':FEATURE_TRANSFORM_ID,
        'input_features':69,'label_order':['Fist','Idle','Ruler'],
        'weights_sha256':checksum('weights.pth'),'metrics_sha256':checksum('metrics.json')}
    if corrupted:manifest['weights_sha256']='0'*64
    (root / 'manifest.json').write_text(json.dumps(manifest))
    return root


class HD12BenchmarkContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder=candidate(Path(self.tmp.name)/'candidate-demo')

    @unittest.skipUnless(importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime") and importlib.util.find_spec("onnxscript"),
                         "Optional ONNX benchmark dependencies absent (use uv --with)")
    def test_onnx_cpu_matches_pytorch_logits_and_predictions_across_batches(self):
        import onnxruntime  # optional benchmark dependency, supplied by --with
        record=compare_backends(self.folder,batch_sizes=(1,8,32),warmup=2,iterations=8,seed=9)
        self.assertEqual(record['result'],'equivalent')
        self.assertEqual(record['exporter'],'torch.export dynamo=True opset 18')
        self.assertEqual(record['contract']['input_features'],69)
        self.assertEqual(record['contract']['label_order'],['Fist','Idle','Ruler'])
        self.assertIn('CPUExecutionProvider',record['onnxruntime']['providers'])
        self.assertLessEqual(record['numerical']['max_abs_logit_error'],1e-4)
        self.assertEqual(record['numerical']['prediction_mismatches'],0)
        self.assertEqual([x['batch_size'] for x in record['latency_ms']], [1,8,32])
        for sample in record['latency_ms']:
            self.assertGreater(sample['torch']['p95'],0)
            self.assertGreater(sample['onnxruntime']['p95'],0)
        self.assertGreater(record['footprint_bytes']['onnx_model'],0)
        self.assertEqual(record['footprint_bytes']['onnx_model'],
                         record['footprint_bytes']['onnx_graph']
                         +record['footprint_bytes']['onnx_external_data'])
        self.assertTrue(record['input_source'].startswith('seeded_synthetic_'))

    @unittest.skipUnless(importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime")
                         and importlib.util.find_spec("onnxscript"),
                         "Optional ONNX benchmark dependencies absent (use uv --with)")
    def test_cli_writes_only_json_to_stdout(self):
        stream=io.StringIO()
        with redirect_stdout(stream):
            code=main(['--candidate',str(self.folder),'--warmup','1','--iterations','2'])
        self.assertEqual(code,0)
        record=json.loads(stream.getvalue())
        self.assertEqual(record['result'],'equivalent')
        self.assertEqual(record['footprint_bytes']['onnx_model'],
                         record['footprint_bytes']['onnx_graph']
                         +record['footprint_bytes']['onnx_external_data'])

    def test_bad_candidate_is_rejected_before_export(self):
        candidate(self.folder,corrupted=True)
        with self.assertRaises(ValueError):
            compare_backends(self.folder,batch_sizes=(1,),warmup=1,iterations=2)

    def test_invalid_settings_refused_before_export(self):
        for options in ({'batch_sizes':(0,)},{'iterations':0},{'warmup':-1},{'batch_sizes':(1,'8')}):
            with self.subTest(options=options),self.assertRaises(ValueError):
                compare_backends(self.folder,**options)


if __name__=='__main__': unittest.main()
