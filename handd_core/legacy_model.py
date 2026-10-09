"""Explicitly opt-in legacy MLP predictor for development demonstrations.

The historic checkpoint has neither a verified class-label manifest nor
participant/session provenance. It is never a validated v2 Model Artifact.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import torch

from handd_core.model_artifact import GestureMLP, ModelPredictor

LEGACY_ASSUMED_LABEL_ORDER = ('Fist', 'Index_Finger', 'Ruler', 'Thumb_Up', 'Idle')


def load_legacy_checkpoint(path: str | Path) -> tuple[ModelPredictor, dict]:
    checkpoint = Path(path)
    raw = checkpoint.read_bytes()
    model = GestureMLP(len(LEGACY_ASSUMED_LABEL_ORDER))
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    digest = sha256(raw).hexdigest()
    return (
        ModelPredictor(model, LEGACY_ASSUMED_LABEL_ORDER),
        {
            'kind': 'legacy_diagnostic',
            'artifact_id': f'legacy-unverified-{digest[:12]}',
            'label_order_unverified': True,
            'label_order': list(LEGACY_ASSUMED_LABEL_ORDER),
            'source_sha256': digest,
        },
    )
