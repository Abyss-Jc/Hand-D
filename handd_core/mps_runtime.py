"""Experimental PyTorch MPS inference; CPU stays the default and safe fallback.

MPS is only attempted when explicitly requested. All model artifacts and the
training pipeline still use their original verified CPU checkpoints.
"""
from __future__ import annotations

from copy import deepcopy
import sys

import numpy as np
import torch

from handd_core.feature_transform import FEATURE_COUNT


class MpsPredictor:
    def __init__(self, reference, accelerator):
        self.reference = reference
        self.accelerator = accelerator
        self.label_order = reference.label_order
        self._failed = False

    @property
    def backend(self) -> str:
        return 'torch-cpu-mps-fallback' if self._failed else 'mps'

    def __call__(self, features: np.ndarray) -> np.ndarray:
        points = np.asarray(features, dtype=np.float32)
        if (points.ndim != 2 or points.shape[1] != FEATURE_COUNT
                or not np.isfinite(points).all()):
            raise ValueError('inference expects finite float32 Nx69 features')
        if self._failed:
            return self.reference(points)
        try:
            logits = self.accelerator(points)
            if (not isinstance(logits, np.ndarray)
                    or logits.shape != (len(points), len(self.label_order))
                    or not np.isfinite(logits).all()):
                raise RuntimeError('MPS produced invalid logits')
            return logits
        except Exception as error:
            self._failed = True
            # Never redo the failed frame on CPU: both hands must release
            # their actions before stabilizing again on subsequent frames.
            print(f'MPS inference failed; switching to CPU: {error}',file=sys.stderr,flush=True)
            raise RuntimeError('MPS failed over; release current gestures') from error

    def predict(self, features: np.ndarray) -> list[str]:
        return [self.label_order[int(i)] for i in np.argmax(self(features),axis=1)]


def prepare_mps_predictor(reference) -> MpsPredictor:
    if not (torch.backends.mps.is_built() and torch.backends.mps.is_available()):
        raise RuntimeError('PyTorch MPS is not available on this machine')
    # Preserve the verified CPU weights for rollback; do not move the only
    # copy of the model onto the GPU.
    gpu_model = deepcopy(reference.model).eval().to('mps')

    def infer(points):
        with torch.inference_mode():
            logits=gpu_model(torch.from_numpy(points).to('mps'))
            return logits.cpu().numpy()

    candidate=MpsPredictor(reference,infer)
    rng=np.random.default_rng(2026)
    for features in (np.zeros((1,FEATURE_COUNT),np.float32),
                     rng.normal(0,.35,(8,FEATURE_COUNT)).astype(np.float32)):
        expected=reference(features)
        actual=candidate(features)
        if (not np.allclose(actual,expected,rtol=1e-4,atol=1e-4)
                or not np.array_equal(np.argmax(actual,axis=1),
                                      np.argmax(expected,axis=1))):
            raise RuntimeError('MPS parity mismatch; keeping CPU inference')
    return candidate


def select_inference_device(reference, device: str):
    if device == 'cpu':
        return reference
    if device != 'mps':
        raise ValueError(f'Unsupported inference device: {device}')
    try:
        return prepare_mps_predictor(reference)
    except Exception as error:
        print(f'MPS initialization unavailable; using CPU: {error}',
              file=sys.stderr,flush=True)
        reference.backend='torch-cpu-mps-fallback'
        return reference
