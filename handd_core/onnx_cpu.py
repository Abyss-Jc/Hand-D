"""Opt-in ONNX CPU inference for an already validated PyTorch predictor.

Training and artifact integrity verification remain PyTorch responsibilities.
The ONNX graph is transient: never rewrite a Candidate or a Workspace.
"""
from __future__ import annotations

from contextlib import redirect_stdout
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch

from handd_core.feature_transform import FEATURE_COUNT


class OnnxCpuPredictor:
    backend = 'onnx-cpu'

    def __init__(self, session, label_order: tuple[str, ...]):
        self.session = session
        self.label_order = label_order

    def __call__(self, features: np.ndarray) -> np.ndarray:
        points = np.asarray(features, dtype=np.float32)
        if (points.ndim != 2 or points.shape[1] != FEATURE_COUNT
                or not np.isfinite(points).all()):
            raise ValueError('inference expects finite float32 Nx69 features')
        return self.session.run(['logits'], {'features': points})[0]

    def predict(self, features: np.ndarray) -> list[str]:
        indices = np.argmax(self(features), axis=1)
        return [self.label_order[int(i)] for i in indices]


def prepare_predictor(reference) -> OnnxCpuPredictor:
    """Export to a temporary ONNX graph and prove CPU parity before activation."""
    try:
        import onnx
        import onnxruntime as ort
        import onnxscript  # noqa: F401 - required by torch's dynamo exporter
    except ImportError as error:
        raise RuntimeError('Optional ONNX packages are missing (uv sync --group onnx-runtime)') from error

    with TemporaryDirectory(prefix='handd-onnx-cpu-') as folder:
        output = Path(folder) / 'model.onnx'
        with redirect_stdout(sys.stderr):
            torch.onnx.export(
                reference.model,
                (torch.zeros(2, FEATURE_COUNT, dtype=torch.float32),),
                str(output),
                input_names=['features'], output_names=['logits'],
                dynamic_shapes=({0: torch.export.Dim('batch', min=1, max=2048)},),
                opset_version=18, dynamo=True,
            )
        onnx.checker.check_model(str(output))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(output), options,
                                       providers=['CPUExecutionProvider'])
        if session.get_providers() != ['CPUExecutionProvider']:
            raise RuntimeError('ONNX Runtime did not select CPUExecutionProvider')
        candidate = OnnxCpuPredictor(session, reference.label_order)
        rng = np.random.default_rng(2026)
        for features in (np.zeros((1, FEATURE_COUNT), np.float32),
                         rng.normal(0, .35, (8, FEATURE_COUNT)).astype(np.float32)):
            expected = reference(features)
            actual = candidate(features)
            if (actual.shape != expected.shape
                    or not np.allclose(actual, expected, rtol=0, atol=1e-4)
                    or not np.array_equal(np.argmax(actual, axis=1),
                                          np.argmax(expected, axis=1))):
                raise RuntimeError('ONNX CPU parity gate failed; PyTorch is retained')
        return candidate


def select_backend(reference, backend: str):
    """Never break an otherwise verified model when optional ONNX is unavailable."""
    if backend == 'torch-cpu':
        return reference
    if backend != 'onnx-cpu':
        raise ValueError(f'Unknown inference backend: {backend}')
    try:
        return prepare_predictor(reference)
    except Exception as error:
        print(f'ONNX CPU initialization failed; using PyTorch CPU: {error}',
              file=sys.stderr, flush=True)
        reference.backend = 'torch-cpu-fallback'
        return reference
