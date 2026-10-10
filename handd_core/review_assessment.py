"""Read-only, version-attributed review signals for samples outside model training.

Scores are uncalibrated. A model ranks observations for a person to inspect;
it never changes review status, labels, or snapshot membership.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from handd_core.feature_transform import canonicalize_world_landmarks
from handd_core.snapshot_builder import verify_snapshot
from handd_core.studio_models import StudioModels


def rank_suggestions(rows: list[dict], margin_threshold: float | None) -> list[dict]:
    if (margin_threshold is not None and (
        not math.isfinite(margin_threshold) or not 0 <= margin_threshold <= 1)):
        raise ValueError("Invalid development-validation margin threshold")
    flagged = []
    for row in rows:
        if row.get("trained_on_this_sample"):
            continue
        margin = row.get("top2_margin")
        if (not isinstance(margin, (float, int)) or not math.isfinite(margin)
                or margin < 0 or margin > 1):
            raise ValueError("Invalid model assessment margin")
        mismatch = row["predicted_label"] != row["true_label"]
        ambiguous = margin_threshold is not None and margin <= margin_threshold
        if mismatch or ambiguous:
            flagged.append({**row, "reason": "model_disagreement" if mismatch else "low_margin"})
    return sorted(flagged, key=lambda row: (
        row["reason"] != "model_disagreement", row["top2_margin"], row["sample_id"]
    ))


def assess_capture(workspace: Path, rows: list, capture_id: str) -> dict:
    """Only verified Candidate snapshots can provide review suggestions.

    In-snapshot observations are never presented as out-of-sample predictions.
    The historical OOF report supplies a *validation-derived* ambiguity cutoff.
    """
    result = {"assessment_state": "no_model_assessment", "suggested": [],
              "assessed_count": 0, "assessment_model_id": None}
    models = StudioModels(workspace)
    try:
        active = models.active_id()
    except ValueError:
        return {**result, "assessment_state": "assessment_unavailable",
                "assessment_error": "Active Model selection is invalid"}
    if active is None:
        return result
    try:
        predictor, artifact = models._load(active)
        source = artifact["source_snapshot"]
        folder = workspace / "snapshots" / source["snapshot_id"]
        manifest_path = folder / "manifest.json"
        if (source["snapshot_id"] != folder.name
                or (workspace / "snapshots").is_symlink()
                or folder.is_symlink() or not verify_snapshot(folder)
                or sha256(manifest_path.read_bytes()).hexdigest() != source["manifest_sha256"]):
            raise ValueError("Candidate training snapshot provenance failed validation")
        membership = set(json.loads(manifest_path.read_text())["sample_ids"])
        report = json.loads((workspace / "models" / active / "metrics.json").read_text())
        variant = ("with_legacy" if artifact.get("legacy_in_final_refit")
                   else "without_legacy")
        oof = report["variants"][variant]["oof"]
        if (report["snapshot_id"] != source["snapshot_id"]
                or report["snapshot_sha256"] != source["manifest_sha256"]
                or {r["sample_id"] for r in oof} != membership
                or any(r["trained_on_this_sample"] for r in oof)):
            raise ValueError("Candidate OOF evidence is not held out")
        correct_margins = [
            float(row["top2_margin"]) for row in oof
            if row["true_label"] == row["predicted_label"]
        ]
        cutoff = (float(np.percentile(correct_margins, 10))
                  if correct_margins else None)
        labels = list(predictor.label_order)
        if any(row["gesture"] not in labels for row in rows):
            return {**result, "assessment_state": "unsupported_gesture",
                    "assessment_model_id": active}
        current = {row["sample_id"]: row for row in rows if row["lifecycle_status"] == "active"}
        untrained = [row for row in current.values() if row["sample_id"] not in membership]
        cache_identity = {
            "model_artifact_id": active,
            "source_snapshot_id": source["snapshot_id"],
            "weights_sha256": artifact["weights_sha256"],
            "metrics_sha256": artifact["metrics_sha256"],
            "source_manifest_sha256": source["manifest_sha256"],
            "capture_id": capture_id,
            "sample_ids": sorted(row["sample_id"] for row in untrained),
        }
        cache_key = sha256(json.dumps(cache_identity, sort_keys=True).encode()).hexdigest()
        cache_path = workspace / "assessments" / (cache_key + ".json")
        if cache_path.parent.is_symlink() or cache_path.is_symlink():
            raise ValueError("Assessment cache must not be a symlink")
        if cache_path.exists():
            history = json.loads(cache_path.read_text(encoding="utf-8"))
            if any(history.get(key) != value for key, value in cache_identity.items()):
                raise ValueError("Model Assessment provenance does not match current Capture")
        else:
            candidates, features, unassessable = [], [], []
            for row in untrained:
                features_row = canonicalize_world_landmarks(
                    np.array(json.loads(row["world_landmarks"])), row["raw_mp_handedness"])
                if features_row is None:
                    unassessable.append(row["sample_id"])
                    continue
                candidates.append(row)
                features.append(features_row)
            assessments = []
            if features:
                logits = predictor(np.stack(features))
                logits = logits - logits.max(axis=1, keepdims=True)
                scores = np.exp(logits)
                scores /= scores.sum(axis=1, keepdims=True)
                for row, score in zip(candidates, scores):
                    best = np.argsort(score)[-2:][::-1]
                    assessments.append({
                        "sample_id": row["sample_id"],
                        "true_label": row["gesture"],
                        "predicted_label": labels[int(best[0])],
                        "scores_not_calibrated": score.tolist(),
                        "top2_margin": float(score[best[0]] - score[best[1]]),
                        "trained_on_this_sample": False,
                    })
            history = {
                **cache_identity,
                "validation_margin_threshold": cutoff,
                "assessments": assessments,
                "unassessable": unassessable,
            }
            # No database schema mutation: derived, append-only assessments
            # are written as one durable file per model/Capture membership.
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temp = None
            try:
                with NamedTemporaryFile("w", encoding="utf-8",
                                       prefix=".assessment-", suffix=".tmp",
                                       dir=cache_path.parent, delete=False) as handle:
                    temp = Path(handle.name)
                    json.dump(history, handle, sort_keys=True, allow_nan=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                try:
                    os.link(temp, cache_path)  # atomic, never overwrite older history
                except FileExistsError:
                    pass
            finally:
                if temp is not None:
                    temp.unlink(missing_ok=True)
        assessments = [{**row, "review_status": current[row["sample_id"]]["review_status"],
                        "model_artifact_id": active,
                        "source_snapshot_id": source["snapshot_id"]}
                       for row in history["assessments"] if row["sample_id"] in current]
        return {
            "assessment_state": "model_assessed",
            "assessment_model_id": active,
            "assessment_source_snapshot": source["snapshot_id"],
            "assessment_revision": cache_key,
            "validation_margin_threshold": cutoff,
            "assessed_count": len(assessments),
            "unassessable": history["unassessable"],
            "suggested": rank_suggestions(assessments, cutoff),
        }
    except (KeyError, TypeError, ValueError, OSError, IndexError) as exc:
        return {**result, "assessment_state": "assessment_unavailable",
                "assessment_error": str(exc)[:160]}
