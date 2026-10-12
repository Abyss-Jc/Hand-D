"""Validated session-held-out training predictions for historical Studio review."""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np


def verified_historical_oof(snapshot: Path, manifest: dict, report: dict,
                            variant: str) -> dict[str, dict]:
    """Fail closed if OOF rows or validation folds contradict frozen membership."""
    with np.load(snapshot / "dataset.npz", allow_pickle=False) as frozen:
        ids = frozen["sample_ids"].tolist()
        sessions = frozen["session_ids"].tolist()
        indices = frozen["label_indices"].tolist()
    labels = manifest["label_order"]
    expected = {sid: (session, labels[index])
                for sid, session, index in zip(ids, sessions, indices)}
    if len(expected) != len(ids) or report["label_order"] != labels:
        raise ValueError("OOF identities or label order differ from frozen Snapshot")
    folds = report["variants"][variant]["folds"]
    unique_sessions = set(sessions)
    if len(folds) != len(unique_sessions):
        raise ValueError("OOF folds do not cover every development Session")
    folds_by_session = {}
    for fold in folds:
        heldout = fold["heldout_session_id"]
        validation = {sid for sid, (session, _) in expected.items() if session == heldout}
        training = set(expected) - validation
        if (heldout not in unique_sessions or heldout in folds_by_session
                or fold["fold_id"] != f"heldout-{heldout}"
                or len(fold["validation_sample_ids"]) != len(validation)
                or set(fold["validation_sample_ids"]) != validation
                or len(fold["train_sample_ids"]) != len(training)
                or set(fold["train_sample_ids"]) != training
                or set(fold["train_session_ids"]) != unique_sessions - {heldout}
                or fold["validation_session_ids"] != [heldout]):
            raise ValueError("OOF fold has a training/validation Session mismatch")
        folds_by_session[heldout] = fold
    indexed = {}
    for row in report["variants"][variant]["oof"]:
        sample_id = row["sample_id"]
        if sample_id not in expected or sample_id in indexed:
            raise ValueError("OOF has unknown or duplicate Sample IDs")
        session, true_label = expected[sample_id]
        scores = np.asarray(row["scores_not_calibrated"], dtype=np.float64)
        if (session not in folds_by_session
                or row["heldout_session_id"] != session
                or row["fold_id"] != f"heldout-{session}"
                or row["fold_model_file"] !=
                   f"folds/{variant}-{sorted(unique_sessions).index(session)}.pth"
                or not re.fullmatch(r"[a-f0-9]{64}", row["fold_model_sha256"])
                or row["trained_on_this_sample"] is not False
                or row["true_label"] != true_label
                or row["predicted_label"] not in labels
                or scores.shape != (len(labels),)
                or not np.isfinite(scores).all() or np.any(scores < 0)
                or not np.isclose(scores.sum(), 1, atol=1e-4)
                or labels[int(np.argmax(scores))] != row["predicted_label"]
                or not np.isclose(
                    row["top2_margin"], np.sort(scores)[-1] - np.sort(scores)[-2],
                    atol=1e-5)):
            raise ValueError("OOF prediction is not a valid held-out Session score")
        indexed[sample_id] = {
            "sample_id": sample_id, "true_label": true_label,
            "predicted_label": row["predicted_label"],
            "scores_not_calibrated": scores.tolist(),
            "top2_margin": float(row["top2_margin"]),
            "trained_on_this_sample": False,
            "assessment_origin": "heldout_session_oof",
            "sample_session_id": session,
            "heldout_session_id": session,
            "fold_id": row["fold_id"],
            "fold_model_file": row["fold_model_file"],
            "fold_model_sha256": row["fold_model_sha256"],
        }
    if len(indexed) != len(expected):
        raise ValueError("OOF lacks predictions for frozen Samples")
    return indexed
