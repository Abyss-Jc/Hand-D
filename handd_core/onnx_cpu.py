"""Experimental ONNX CPU inference for an already validated PyTorch predictor.

Training and artifact integrity verification remain PyTorch responsibilities.
The ONNX graph is transient: never rewrite a Candidate or a Workspace.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import redirect_stdout
from hashlib import sha256
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch

from handd_core.feature_transform import FEATURE_COUNT

_MAX_CACHED_SESSIONS = 3
_cached_sessions = OrderedDict()


def _model_key(reference) -> str:
    """Use actual verified in-memory weights, not an artifact path or ID."""
    checksum = sha256()
    for name, tensor in reference.model.state_dict().items():
        values = tensor.detach().cpu().contiguous()
        checksum.update(name.encode('utf-8') + b'\0')
        checksum.update(f'{values.dtype}:{tuple(values.shape)}'.encode('ascii') + b'\0')
        checksum.update(values.numpy().tobytes())
    return checksum.hexdigest()


class OnnxCpuPredictor:
    def __init__(self, session, reference, *, cache_key=None):
        self.session = session
        self.reference = reference
        self.label_order = reference.label_order
        self._failed = False
        self._cache_key = cache_key

    @property
    def backend(self) -> str:
        return 'torch-cpu-fallback' if self._failed else 'onnx-cpu'

    def __call__(self, features: np.ndarray) -> np.ndarray:
        points = np.asarray(features, dtype=np.float32)
        if (points.ndim != 2 or points.shape[1] != FEATURE_COUNT
                or not np.isfinite(points).all()):
            raise ValueError('inference expects finite float32 Nx69 features')
        if self._failed:
            return self.reference(points)
        try:
            logits = self.session.run(['logits'], {'features': points})[0]
            if (not isinstance(logits, np.ndarray)
                    or logits.shape != (len(points), len(self.label_order))
                    or not np.isfinite(logits).all()):
                raise RuntimeError('ONNX CPU returned invalid logits')
            return logits
        except Exception as error:
            # The current frame must release both hands, never immediately
            # retry a possibly destructive drawing/erasing gesture on Torch.
            self._failed = True
            if (self._cache_key is not None
                    and _cached_sessions.get(self._cache_key) is self.session):
                _cached_sessions.pop(self._cache_key)
            print(f'ONNX CPU inference failed; switching to PyTorch CPU: {error}',
                  file=sys.stderr, flush=True)
            raise RuntimeError('ONNX CPU failed over; release current gestures') from error

    def predict(self, features: np.ndarray) -> list[str]:
        indices = np.argmax(self(features), axis=1)
        return [self.label_order[int(i)] for i in indices]


def prepare_predictor(reference) -> OnnxCpuPredictor:
    """Validate CPU parity; cache at most 3 sessions for identical weights."""
    try:
        import onnx
        import onnxruntime as ort
        import onnxscript  # noqa: F401 - required by torch's dynamo exporter
    except ImportError as error:
        raise RuntimeError('Optional ONNX packages are missing (uv sync --group onnx-runtime)') from error

    cache_key = _model_key(reference)
    session = _cached_sessions.get(cache_key)
    if session is None:
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
    candidate = OnnxCpuPredictor(session, reference, cache_key=cache_key)
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
    _cached_sessions[cache_key] = session
    _cached_sessions.move_to_end(cache_key)
    if len(_cached_sessions) > _MAX_CACHED_SESSIONS:
        _cached_sessions.popitem(last=False)
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
