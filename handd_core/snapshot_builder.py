"""Immutable, self-contained Development Snapshot built from curated SQLite.

The model training pipeline reads these frozen arrays and their manifest, never
the mutable SQLite source. Snapshot creation does not launch training.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np

from handd_core.dataset_store import DatasetStore, SCHEMA_VERSION
from handd_core.feature_transform import FEATURE_TRANSFORM_ID, FEATURE_COUNT, canonicalize_world_landmarks
from handd_core.legacy_import import load_legacy_source
from handd_core.snapshot_readiness import assess_snapshot_readiness

SNAPSHOT_FORMAT_VERSION = 1
BASE_LABEL_ORDER = ("Fist", "Index_Finger", "Ruler", "Thumb_Up", "Idle")


@dataclass(frozen=True)
class DevelopmentSnapshot:
    snapshot_id: str
    path: Path


class SnapshotBlocked(ValueError):
    """Creation would violate an integrity/reproducibility contract."""


def _freeze_legacy_csv(path: Path) -> tuple[bytes, list[str], str, int]:
    """Freeze compatible historic feature rows without claiming clean provenance."""
    raw = path.read_bytes()
    legacy_features: list[list[float]] = []
    legacy_labels: list[str] = []
    legacy_hands: list[float] = []
    expected = [f'feat_{i}' for i in range(FEATURE_COUNT)] + ['handedness', 'label']
    with path.open(newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != expected:
            raise SnapshotBlocked('legacy incompatible columns: expected 69 features, handedness, label')
        for line_number, row in enumerate(reader, start=2):
            if line_number > 1000000:
                raise SnapshotBlocked('legacy source exceeds row safety limit')
            try:
                features = [float(row[f'feat_{i}']) for i in range(FEATURE_COUNT)]
                hand = float(row['handedness'])
                label = row['label']
            except (TypeError, ValueError, KeyError) as exc:
                raise SnapshotBlocked(f'legacy invalid row {line_number}') from exc
            if (not np.isfinite(features).all() or hand not in (0.0, 1.0)
                    or label not in BASE_LABEL_ORDER):
                raise SnapshotBlocked(f'legacy invalid values at row {line_number}')
            legacy_features.append(features)
            legacy_labels.append(label)
            legacy_hands.append(hand)
    if not legacy_features:
        raise SnapshotBlocked('legacy CSV has no feature rows')
    serialized = BytesIO()
    label_to_index = {label: idx for idx, label in enumerate(BASE_LABEL_ORDER)}
    np.savez_compressed(
        serialized, features=np.array(legacy_features, dtype=np.float32),
        labels=np.asarray(legacy_labels, dtype=np.str_),
        handedness=np.asarray(legacy_hands, dtype=np.float32),
        label_indices=np.asarray([label_to_index[label] for label in legacy_labels],
                                 dtype=np.int64),
        row_ids=np.asarray([f'legacy-row-{i:06d}' for i in range(1, len(legacy_labels)+1)],
                           dtype=np.str_),
    )
    return serialized.getvalue(), legacy_labels, _checksum(raw), len(legacy_labels)


def _checksum(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _freeze_legacy_store(store: DatasetStore, source_id: str) -> tuple[bytes, list[str], str, int]:
    features, labels, handedness, row_ids = load_legacy_source(store, source_id)
    if not set(labels).issubset(BASE_LABEL_ORDER):
        raise SnapshotBlocked('quarantined legacy labels are not compatible')
    label_to_index = {label: i for i, label in enumerate(BASE_LABEL_ORDER)}
    stream = BytesIO()
    np.savez_compressed(
        stream, features=features,
        labels=np.asarray(labels, dtype=np.str_),
        handedness=handedness,
        label_indices=np.asarray([label_to_index[name] for name in labels], dtype=np.int64),
        row_ids=np.asarray(row_ids, dtype=np.str_),
    )
    return stream.getvalue(), labels, source_id, len(labels)


def _folds(session_ids: list[str]) -> list[dict]:
    sessions = sorted(set(session_ids))
    if len(sessions) < 2:
        return []
    return [
        {"fold_id": f"heldout-{s}", "holdout_session_id": s,
         "train_session_ids": [other for other in sessions if other != s]}
        for s in sessions
    ]


def build_development_snapshot(
    store: DatasetStore, workspace: Path, *,
    note: str | None = None,
    development_session_ids: set[str] | None = None,
    final_test_session_ids: set[str] | None = None,
    seed: int = 42,
    snapshot_id: str | None = None,
    legacy_csv: Path | None = None,
    legacy_source_id: str | None = None,
) -> DevelopmentSnapshot:
    """Freeze accepted + active development inputs, fail without partial artifacts.

    By default, P003 is not included in Development Snapshots regardless of
    session names. To use other participants explicitly, pass scoped session IDs.
    """
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if note is not None and not isinstance(note, str):
        raise ValueError("note must be text")
    if legacy_csv is not None and legacy_source_id is not None:
        raise ValueError("choose either legacy_csv or legacy_source_id, not both")
    if legacy_source_id is not None:
        source = store.conn.execute(
            'SELECT status FROM legacy_sources WHERE source_id=?', (legacy_source_id,),
        ).fetchone()
        if source is None:
            raise KeyError(f'unknown legacy source {legacy_source_id}')
        if source['status'] != 'compatible':
            raise SnapshotBlocked('quarantined legacy source cannot enter Development')
    final = set(final_test_session_ids or ())
    selected = set(development_session_ids) if development_session_ids is not None else None
    if selected is not None and selected & final:
        raise SnapshotBlocked("development_final_overlap")
    # The Development Snapshot is reserved for P001/P002. Explicit session
    # selection must never accidentally unseal P003 or other participants.
    allowed_sessions = {row[0] for row in store.conn.execute(
        "SELECT session_id FROM collection_sessions WHERE participant_id IN ('P001','P002')"
    )}
    if selected is not None:
        allowed_sessions &= selected
    allowed_sessions -= final
    # BEGIN guarantees a coherent view of mutable review state and observations
    # even if a separate SQLite reader/writer is active.
    store.conn.execute("BEGIN")
    try:
        readiness = assess_snapshot_readiness(
            store, development_session_ids=allowed_sessions, final_test_session_ids=final
        )
        samples = [sample for sample in store.list_eligible_samples()
                   if sample["session_id"] in allowed_sessions]
        if not samples:
            raise SnapshotBlocked("no_eligible_samples")
        if readiness.blockers:
            raise SnapshotBlocked(",".join(readiness.blockers))
        features = []
        for sample in samples:
            transformed = canonicalize_world_landmarks(
                sample["world_landmarks"], sample["raw_mp_handedness"]
            )
            if transformed is None or transformed.shape != (FEATURE_COUNT,):
                raise SnapshotBlocked("invalid_feature_transform")
            features.append(transformed)
    finally:
        store.conn.rollback()

    legacy_payload = None
    legacy_labels: list[str] = []
    legacy_source_sha = None
    legacy_count = 0
    if legacy_csv is not None:
        legacy_payload, legacy_labels, legacy_source_sha, legacy_count = _freeze_legacy_csv(
            Path(legacy_csv)
        )
    elif legacy_source_id is not None:
        legacy_payload, legacy_labels, legacy_source_sha, legacy_count = _freeze_legacy_store(
            store, legacy_source_id,
        )
    labels = tuple(dict.fromkeys(
        (*BASE_LABEL_ORDER, *sorted({sample["gesture"] for sample in samples}))
    ))
    # Keep baseline 5-class indices stable even when this snapshot happens
    # to contain only some classes; extra workspace gestures append after it.
    label_map = {name: i for i, name in enumerate(labels)}
    sample_ids = [s["sample_id"] for s in samples]
    participant_ids = [s["participant_id"] for s in samples]
    session_ids = [s["session_id"] for s in samples]
    captures = [s["capture_id"] for s in samples]
    fold_definitions = _folds(session_ids)
    data = BytesIO()
    np.savez_compressed(
        data, features=np.stack(features).astype(np.float32),
        label_indices=np.array([label_map[s["gesture"]] for s in samples], dtype=np.int64),
        sample_ids=np.asarray(sample_ids, dtype=np.str_),
        participant_ids=np.asarray(participant_ids, dtype=np.str_),
        session_ids=np.asarray(session_ids, dtype=np.str_),
        capture_ids=np.asarray(captures, dtype=np.str_),
    )
    payload = data.getvalue()
    if snapshot_id is None:
        snapshot_id = f"dev-{uuid4().hex}"
    if not snapshot_id or snapshot_id in (".", "..") or "/" in snapshot_id or "\\" in snapshot_id:
        raise ValueError("snapshot_id must be a safe single directory name")
    snapshots = Path(workspace) / "snapshots"
    # Final path is reserved exclusively; an existing snapshot can NEVER be replaced.
    snapshots.mkdir(parents=True, exist_ok=True)
    target = snapshots / snapshot_id
    target.mkdir(exist_ok=False)
    manifest = {
        "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
        "snapshot_kind": "development",
        "snapshot_id": snapshot_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": note,
        "seed": seed,
        "feature_transform": FEATURE_TRANSFORM_ID,
        "feature_count": FEATURE_COUNT,
        "source": {"workspace_relative_sqlite": "handd.sqlite",
                   "schema_version": SCHEMA_VERSION},
        "sample_count": len(sample_ids),
        "sample_ids": sample_ids,
        "participant_ids": sorted(set(participant_ids)),
        "session_ids": sorted(set(session_ids)),
        "label_order": list(labels),
        "validation_folds": fold_definitions,
        "warnings": list(readiness.warnings),
        "data_file": "dataset.npz",
        "data_sha256": _checksum(payload),
        "membership_sha256": _checksum(json.dumps(sample_ids, separators=(",", ":")).encode()),
        "legacy": {"included": legacy_payload is not None, "count": legacy_count,
                   "data_file": "legacy.npz" if legacy_payload is not None else None,
                   "data_sha256": _checksum(legacy_payload)
                   if legacy_payload is not None else None,
                   "source_sha256": legacy_source_sha,
                   "source_id": legacy_source_id,
                   "provenance_quality": "legacy_unverified_no_session_groups"
                   if legacy_payload is not None else None},
    }
    try:
        # Exclusive creation, no rewriting any existing snapshot bytes.
        with (target / "dataset.npz").open("xb") as file:
            file.write(payload)
        if legacy_payload is not None:
            with (target / 'legacy.npz').open('xb') as file:
                file.write(legacy_payload)
        with (target / "manifest.json").open("x", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, sort_keys=True)
    except BaseException:
        shutil.rmtree(target)
        raise
    return DevelopmentSnapshot(snapshot_id=snapshot_id, path=target)


def verify_snapshot(path: Path) -> bool:
    """Check integrity and manifest contract without consulting SQLite."""
    try:
        root = Path(path)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if (manifest['snapshot_format_version'] != SNAPSHOT_FORMAT_VERSION
                or manifest['snapshot_kind'] != 'development'):
            return False
        data_name = manifest["data_file"]
        if data_name != "dataset.npz":
            return False
        payload = (root / data_name).read_bytes()
        if _checksum(payload) != manifest["data_sha256"]:
            return False
        if manifest["feature_transform"] != FEATURE_TRANSFORM_ID:
            return False
        with np.load(BytesIO(payload), allow_pickle=False) as arrays:
            actual_ids = arrays["sample_ids"].tolist()
            valid = (actual_ids == manifest["sample_ids"]
                    and len(actual_ids) == manifest["sample_count"]
                    and arrays["features"].shape == (len(actual_ids), FEATURE_COUNT)
                    and arrays["features"].dtype == np.float32
                    and manifest['membership_sha256'] == _checksum(
                        json.dumps(actual_ids, separators=(',', ':')).encode()
                    ))
        legacy = manifest['legacy']
        if legacy['included']:
            legacy_payload = (root / 'legacy.npz').read_bytes()
            if _checksum(legacy_payload) != legacy['data_sha256']:
                return False
            with np.load(BytesIO(legacy_payload), allow_pickle=False) as legacy_arrays:
                valid = (valid
                         and legacy_arrays['features'].shape == (legacy['count'], FEATURE_COUNT)
                         and legacy_arrays['labels'].shape == (legacy['count'],))
        return bool(valid)
    except (KeyError, ValueError, TypeError, OSError, EOFError, json.JSONDecodeError):
        return False
