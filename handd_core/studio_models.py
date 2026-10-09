"""Workspace-scoped Model Artifact catalog and explicit Active selection.

The immutable Candidate must live directly under workspace/models/candidate-*.
Never load a user-supplied arbitrary path via WebSocket. Selection file is
atomic and records only an ID, not executable code or checkpoint bytes.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

from handd_core.model_artifact import ModelArtifactError, load_model_artifact
from handd_core.studio_workspace import StudioWorkspace

CANDIDATE_ID = re.compile(r"candidate-[A-Za-z0-9_-]{1,96}\Z")


class StudioModels:
    def __init__(self, workspace: Path | str):
        self.workspace = StudioWorkspace(workspace).path
        self.model_root = self.workspace / "models"
        self.selection_path = self.model_root / ".active-model.json"

    def _folder(self, artifact_id: str) -> Path:
        if not isinstance(artifact_id, str) or not CANDIDATE_ID.fullmatch(artifact_id):
            raise ValueError("Choose a Model Artifact from this workspace")
        path = self.model_root / artifact_id
        if path.is_symlink() or not path.is_dir() or not (path / "manifest.json").is_file():
            raise ValueError("Model Artifact not found in this workspace")
        if path.resolve() != path.absolute():
            raise ValueError("Model Artifact must not be a symlink")
        for file_name in ("manifest.json", "weights.pth", "metrics.json"):
            item=path/file_name
            if item.is_symlink() or not item.is_file():
                raise ValueError("Model Artifact files must be regular local files")
        return path

    def active_id(self) -> str | None:
        if not self.selection_path.is_file():
            return None
        try:
            selected = json.loads(self.selection_path.read_text(encoding="utf-8"))
            artifact_id = selected["active_model_id"]
            if not isinstance(artifact_id, str) or not CANDIDATE_ID.fullmatch(artifact_id):
                raise ValueError("Invalid persisted Active Model ID")
            return artifact_id
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid persisted Active Model selection") from exc

    def listed_active_id(self) -> str | None:
        """Catalog stays usable if selection metadata must be repaired in Studio."""
        try:
            return self.active_id()
        except ValueError:
            return None

    def list_candidates(self) -> list[dict]:
        if not self.model_root.is_dir():
            return []
        active = self.listed_active_id()
        result = []
        for path in sorted(self.model_root.iterdir()):
            if len(result) >= 50:
                break
            if path.is_symlink() or not path.is_dir() or not CANDIDATE_ID.fullmatch(path.name):
                continue
            try:
                _, manifest = load_model_artifact(self._folder(path.name))
                if manifest["artifact_id"] != path.name:
                    raise ModelArtifactError("candidate ID does not match directory")
                row = {
                    "artifact_id": path.name,
                    "compatible": True,
                    "labels": len(manifest["label_order"]),
                    "source_snapshot": manifest.get("source_snapshot", {}).get("snapshot_id"),
                    "active": active == path.name,
                }
            except (ValueError, OSError) as exc:
                row = {
                    "artifact_id": path.name, "compatible": False,
                    "reason": str(exc)[:160], "active": active == path.name,
                }
            result.append(row)
        return result

    def _load(self, artifact_id: str):
        path = self._folder(artifact_id)
        predictor, manifest = load_model_artifact(path)
        if manifest["artifact_id"] != artifact_id:
            raise ValueError("Model Artifact ID does not match candidate directory")
        return predictor, manifest

    def _write_active(self, artifact_id: str) -> None:
        self.model_root.mkdir(exist_ok=True)
        data = json.dumps({"active_model_id": artifact_id, "version": 1}) + "\n"
        temp = None
        try:
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", prefix=".active-model-",
                suffix=".tmp", dir=self.model_root, delete=False,
            ) as out:
                temp = Path(out.name)
                out.write(data)
                out.flush()
                os.fsync(out.fileno())
            temp.replace(self.selection_path)
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)

    def activate(self, artifact_id: str, runtime) -> dict:
        predictor, manifest = self._load(artifact_id)
        # Neither the running predictor nor the previously selected config is
        # changed if validation/checksum/disk persistence fails.
        self._write_active(artifact_id)
        return runtime.install_loaded_model(predictor, manifest)

    def restore(self, runtime) -> dict | None:
        artifact_id = self.active_id()
        if artifact_id is None:
            return None
        predictor, manifest = self._load(artifact_id)
        return runtime.install_loaded_model(predictor, manifest)
