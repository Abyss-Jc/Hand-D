"""Frozen Development Snapshot -> leakage-safe session CV -> final candidate."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np
import torch

from handd_core.feature_transform import FEATURE_COUNT
from handd_core.model_artifact import ARCHITECTURE_ID, MODEL_FORMAT_VERSION, GestureMLP
from handd_core.snapshot_builder import verify_snapshot
from handd_core.feature_transform import FEATURE_TRANSFORM_ID


class TrainingBlocked(ValueError):
    """The frozen snapshot cannot support reproducible grouped evaluation."""


@dataclass(frozen=True)
class ExperimentResult:
    report_dir: Path
    model_dir: Path


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True, allow_nan=False)


def _metrics(true: np.ndarray, predicted: np.ndarray, labels: list[str]) -> dict:
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for actual, guess in zip(true, predicted):
        cm[int(actual), int(guess)] += 1
    per_class = []
    for i, label in enumerate(labels):
        tp = int(cm[i, i])
        support = int(cm[i].sum())
        positive = int(cm[:, i].sum())
        precision = tp / positive if positive else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"label": label, "support": support, "precision": precision,
                          "recall": recall, "f1": f1})
    return {"count": len(true), "macro_f1": float(np.mean([c["f1"] for c in per_class])),
            "accuracy": float(np.mean(true == predicted)) if len(true) else 0.0,
            "confusion_matrix": cm.tolist(), "per_class": per_class}


def _fit(features: np.ndarray, y: np.ndarray, classes: int, *,
         seed: int, epochs: int, batch_size: int, lr: float) -> GestureMLP:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = GestureMLP(classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        x_tensor = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
        y_tensor = torch.from_numpy(np.asarray(y, dtype=np.int64))
        rng = torch.Generator().manual_seed(seed)
        for _ in range(epochs):
            model.train()
            for idx in torch.randperm(len(x_tensor), generator=rng).split(batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x_tensor[idx]), y_tensor[idx])
                loss.backward()
                optimizer.step()
        return model.eval()


def _predict(model: GestureMLP, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with torch.inference_mode():
        logits = model(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)))
        scores = torch.softmax(logits, dim=1).cpu().numpy()
    return scores.argmax(axis=1), scores


def _read_snapshot(folder: Path):
    if not verify_snapshot(folder):
        raise TrainingBlocked("snapshot integrity/feature contract verification failed")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    with np.load(folder / "dataset.npz", allow_pickle=False) as data:
        x = np.array(data["features"], dtype=np.float32)
        y = np.array(data["label_indices"], dtype=np.int64)
        ids = data["sample_ids"].tolist()
        sessions = data["session_ids"].tolist()
        participants = data["participant_ids"].tolist()
    labels = manifest["label_order"]
    if (x.shape != (len(ids), FEATURE_COUNT) or y.shape != (len(ids),)
            or not np.isfinite(x).all() or len(set(ids)) != len(ids)
            or any(p not in ("P001", "P002") for p in participants)
            or not labels or not np.all((y >= 0) & (y < len(labels)))):
        raise TrainingBlocked("snapshot has invalid labels, observations or sealed participants")
    unique_sessions = sorted(set(sessions))
    if len(unique_sessions) < 2:
        raise TrainingBlocked("at least two distinct Collection Sessions required for CV")
    expected = [
        {"fold_id": f"heldout-{s}", "holdout_session_id": s,
         "train_session_ids": [other for other in unique_sessions if other != s]}
        for s in unique_sessions
    ]
    if manifest["validation_folds"] != expected:
        raise TrainingBlocked("frozen validation folds do not match session membership")
    return manifest, x, y, ids, sessions, labels


def run_development_experiment(
    snapshot_dir: str | Path, workspace: str | Path, *,
    epochs: int = 20, batch_size: int = 32, learning_rate: float = 1e-3,
    seed: int = 42, learning_curve_fractions: tuple[float, ...] = (.25, .5, .75, 1.0),
    final_legacy: bool = False,
) -> ExperimentResult:
    """Evaluates P001/P002 only; legacy is TRAIN-only; never changes Active Model."""
    if (type(epochs) is not int or epochs <= 0 or type(batch_size) is not int
            or batch_size <= 0 or not 0 < learning_rate <= 1):
        raise ValueError("epochs/batch_size must be positive and learning rate must be valid")
    fractions = tuple(float(v) for v in learning_curve_fractions)
    if (not fractions or any(not np.isfinite(f) or not 0 < f <= 1 for f in fractions)
            or len(set(fractions)) != len(fractions) or 1.0 not in fractions):
        raise ValueError("fractions must be unique, in (0,1], including 1.0")
    source = Path(snapshot_dir)
    workspace = Path(workspace)
    manifest, x, y, ids, sessions, labels = _read_snapshot(source)
    legacy_x = np.empty((0, FEATURE_COUNT), dtype=np.float32)
    legacy_y = np.empty((0,), dtype=np.int64)
    if manifest["legacy"]["included"]:
        with np.load(source / "legacy.npz", allow_pickle=False) as data:
            legacy_x = np.asarray(data["features"], dtype=np.float32)
            legacy_y = np.asarray(data["label_indices"], dtype=np.int64)
        if (legacy_x.shape[1:] != (FEATURE_COUNT,) or legacy_y.shape != (len(legacy_x),)
                or not np.isfinite(legacy_x).all()
                or not np.all((legacy_y >= 0) & (legacy_y < len(labels)))):
            raise TrainingBlocked("legacy partition incompatible with frozen label order")
    if final_legacy and not len(legacy_x):
        raise TrainingBlocked("explicit final legacy augmentation needs legacy partition")

    report_root = workspace / "reports"
    model_root = workspace / "models"
    report_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    experiment_id = uuid4().hex
    report_dir = report_root / f"experiment-{experiment_id}"
    model_dir = model_root / f"candidate-{experiment_id}"
    report_dir.mkdir(exist_ok=False)
    model_dir.mkdir(exist_ok=False)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        (report_dir / "folds").mkdir()
        report = {
            "experiment_id": experiment_id, "snapshot_id": manifest["snapshot_id"],
            "snapshot_sha256": _sha(source / "manifest.json"),
            "evaluation_kind": "session_grouped_development_oof",
            "unseen_participant_evaluation": False, "label_order": labels,
            "legacy_provenance_quality": manifest["legacy"]["provenance_quality"],
            "legacy_validation_warning": (
                "Historic legacy rows have no session/participant provenance; "
                "with_legacy is exploratory and possible cross-session overlap is unverified"
                if len(legacy_x) else None
            ),
            "config": {"epochs": epochs, "batch_size": batch_size,
                       "learning_rate": learning_rate, "seed": seed,
                       "learning_curve_fractions": list(fractions)},
            "variants": {},
        }
        sid = np.asarray(sessions)
        for variant_name in (["without_legacy", "with_legacy"] if len(legacy_x)
                             else ["without_legacy"]):
            augment = variant_name == "with_legacy"
            fold_rows = []
            oof = []
            for fold_i, heldout in enumerate(sorted(set(sessions))):
                train_idx = np.flatnonzero(sid != heldout)
                validation_idx = np.flatnonzero(sid == heldout)
                if not len(train_idx) or not len(validation_idx):
                    raise TrainingBlocked("empty training/validation Session fold")
                curve = []
                for fraction_i, fraction in enumerate(fractions):
                    count = max(1, int(np.ceil(len(train_idx) * fraction)))
                    rng = np.random.default_rng(seed + fold_i * 131 + fraction_i)
                    selected = np.sort(rng.choice(train_idx, size=count, replace=False))
                    train_x, train_y = x[selected], y[selected]
                    if augment:
                        train_x = np.concatenate((train_x, legacy_x))
                        train_y = np.concatenate((train_y, legacy_y))
                    # Same v2 subset and initialization for paired variants.
                    model = _fit(train_x, train_y, len(labels),
                                 seed=seed + fold_i * 101 + fraction_i,
                                 epochs=epochs, batch_size=batch_size, lr=learning_rate)
                    predictions, scores = _predict(model, x[validation_idx])
                    metric = _metrics(y[validation_idx], predictions, labels)
                    curve.append({
                        "fraction": fraction, "v2_training_sample_count": len(selected),
                        "legacy_training_sample_count": len(legacy_x) if augment else 0,
                        "macro_f1": metric["macro_f1"], "accuracy": metric["accuracy"],
                    })
                    if fraction == 1.0:
                        checkpoint = f"folds/{variant_name}-{fold_i}.pth"
                        torch.save(model.state_dict(), report_dir / checkpoint)
                        checksum = _sha(report_dir / checkpoint)
                        for offset, index in enumerate(validation_idx):
                            row_scores = scores[offset].tolist()
                            top = sorted(row_scores, reverse=True)
                            oof.append({
                                "sample_id": ids[index], "heldout_session_id": heldout,
                                "fold_id": f"heldout-{heldout}",
                                "true_label": labels[int(y[index])],
                                "predicted_label": labels[int(predictions[offset])],
                                "scores_not_calibrated": row_scores,
                                "top2_margin": float(top[0] - top[1]),
                                "trained_on_this_sample": False,
                                "fold_model_file": checkpoint,
                                "fold_model_sha256": checksum,
                            })
                fold_rows.append({
                    "fold_id": f"heldout-{heldout}", "heldout_session_id": heldout,
                    "validation_session_ids": [heldout],
                    "train_session_ids": sorted(set(sid[train_idx].tolist())),
                    "train_sample_ids": [ids[int(i)] for i in train_idx],
                    "validation_sample_ids": [ids[int(i)] for i in validation_idx],
                    "legacy_training_sample_count": len(legacy_x) if augment else 0,
                    "learning_curve": curve,
                })
            if len(oof) != len(ids) or len(set(r["sample_id"] for r in oof)) != len(ids):
                raise TrainingBlocked("missing or duplicate OOF predictions")
            ordered = {row["sample_id"]: row for row in oof}
            oof_ordered = [ordered[i] for i in ids]
            oof_indices = np.array([labels.index(r["predicted_label"]) for r in oof_ordered])
            report["variants"][variant_name] = {
                **_metrics(y, oof_indices, labels), "folds": fold_rows,
                "oof": oof_ordered, "legacy_training_only": augment,
            }
        final_x = np.concatenate((x, legacy_x)) if final_legacy else x
        final_y = np.concatenate((y, legacy_y)) if final_legacy else y
        final_model = _fit(final_x, final_y, len(labels), seed=seed + 10000,
                           epochs=epochs, batch_size=batch_size, lr=learning_rate)
        torch.save(final_model.state_dict(), model_dir / "weights.pth")
        _write_json(report_dir / "metrics.json", report)
        shutil.copyfile(report_dir / "metrics.json", model_dir / "metrics.json")
        artifact_manifest = {
            "model_format_version": MODEL_FORMAT_VERSION,
            "artifact_role": "final_refit_candidate", "artifact_id": model_dir.name,
            "architecture": ARCHITECTURE_ID, "input_features": FEATURE_COUNT,
            "feature_transform": FEATURE_TRANSFORM_ID, "label_order": labels,
            "training_config": report["config"], "legacy_in_final_refit": final_legacy,
            "refit_v2_sample_count": len(ids),
            "refit_legacy_row_count": len(legacy_x) if final_legacy else 0,
            "source_snapshot": {"snapshot_id": manifest["snapshot_id"],
                                "manifest_sha256": report["snapshot_sha256"],
                                "data_sha256": manifest["data_sha256"]},
            "evaluation_kind": "session_grouped_development_oof",
            "legacy_provenance_quality": manifest["legacy"]["provenance_quality"],
            "sealed_final_test": None, "active_model_changed": False,
            "weights_sha256": _sha(model_dir / "weights.pth"),
            "metrics_sha256": _sha(model_dir / "metrics.json"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(model_dir / "manifest.json", artifact_manifest)
        return ExperimentResult(report_dir=report_dir, model_dir=model_dir)
    except BaseException:
        shutil.rmtree(report_dir)
        shutil.rmtree(model_dir)
        raise
    finally:
        torch.set_num_threads(threads)
