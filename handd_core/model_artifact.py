"""Immutable versioned Model Artifact reader for CPU inference.

Historical bare checkpoints need their own explicit legacy contract and are
not accepted as a v2 Model Artifact by this loader.
"""
from __future__ import annotations

from hashlib import sha256
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn

from handd_core.feature_transform import FEATURE_COUNT, FEATURE_TRANSFORM_ID

MODEL_FORMAT_VERSION = 1
ARCHITECTURE_ID = "gesture-mlp-69-128-64-v1"


class ModelArtifactError(ValueError):
    """An artifact lacks the trusted runtime/training compatibility contract."""


class GestureMLP(nn.Module):
    """Same parameter names and dimensions as the five-class legacy MLP."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(FEATURE_COUNT, 128)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.relu2(self.fc2(self.dropout(self.relu1(self.fc1(x))))))


class ModelPredictor:
    def __init__(self, model: GestureMLP, label_order: tuple[str, ...]):
        self.model = model.eval()
        self.label_order = label_order

    def __call__(self, features: np.ndarray) -> np.ndarray:
        points = np.asarray(features, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != FEATURE_COUNT or not np.isfinite(points).all():
            raise ValueError("inference expects finite float32 Nx69 features")
        with torch.inference_mode():
            return self.model(torch.from_numpy(points)).cpu().numpy()

    def predict(self, features: np.ndarray) -> list[str]:
        indices = np.argmax(self(features), axis=1)
        return [self.label_order[int(i)] for i in indices]


def load_model_artifact(path: str | Path) -> tuple[ModelPredictor, dict]:
    """Verify manifest+weights+metrics before loading anything as Active-ready."""
    root = Path(path)
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        labels = manifest["label_order"]
        if (manifest["model_format_version"] != MODEL_FORMAT_VERSION
                or manifest["artifact_role"] != "final_refit_candidate"
                or manifest["architecture"] != ARCHITECTURE_ID
                or manifest["feature_transform"] != FEATURE_TRANSFORM_ID
                or manifest["input_features"] != FEATURE_COUNT
                or not isinstance(labels, list) or len(labels) < 2
                or len(set(labels)) != len(labels)
                or any(not isinstance(label, str) or not label for label in labels)):
            raise ModelArtifactError("incompatible Model Artifact contract")
        weights = (root / "weights.pth").read_bytes()
        metrics = (root / "metrics.json").read_bytes()
        if (sha256(weights).hexdigest() != manifest["weights_sha256"]
                or sha256(metrics).hexdigest() != manifest["metrics_sha256"]):
            raise ModelArtifactError("Model Artifact checksum mismatch")
        model = GestureMLP(len(labels))
        state = torch.load(root / "weights.pth", map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        return ModelPredictor(model, tuple(labels)), manifest
    except (OSError, KeyError, TypeError, json.JSONDecodeError,
            RuntimeError, pickle.UnpicklingError) as exc:
        raise ModelArtifactError(f"invalid or missing Model Artifact: {exc}") from exc
