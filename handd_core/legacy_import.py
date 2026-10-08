"""Idempotent, provenance-aware historical CSV import into canonical SQLite.

Legacy observations have ONLY derived feature vectors. They remain separate
from raw-landmark Samples and are never assigned invented Session identities.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np

from handd_core.dataset_store import DatasetStore
from handd_core.feature_transform import FEATURE_COUNT

BASE_LABELS = frozenset({"Fist", "Index_Finger", "Ruler", "Thumb_Up", "Idle"})
EXPECTED_HEADER = [f"feat_{i}" for i in range(FEATURE_COUNT)] + ["handedness", "label"]


@dataclass(frozen=True)
class LegacyImportResult:
    source_id: str
    count: int
    status: str
    already_imported: bool = False


def list_legacy_sources(store: DatasetStore) -> list[dict]:
    return [dict(row) for row in store.conn.execute(
        "SELECT * FROM legacy_sources ORDER BY imported_at, source_id"
    ).fetchall()]


def import_legacy_csv(store: DatasetStore, csv_path: str | Path) -> LegacyImportResult:
    """Atomic import of the exact CSV bytes, with SHA-256 identity.

    Label-incompatible archives are retained with original labels, but marked
    quarantined and ineligible for Development Snapshot legacy augmentation.
    """
    source = Path(csv_path)
    digest = sha256(source.read_bytes()).hexdigest()
    old = store.conn.execute(
        "SELECT row_count, status FROM legacy_sources WHERE source_id=?", (digest,)
    ).fetchone()
    if old is not None:
        return LegacyImportResult(digest, old["row_count"], old["status"], True)
    rows = []
    labels = set()
    with source.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != EXPECTED_HEADER:
            raise ValueError("legacy CSV requires feat_0..feat_68, handedness, label")
        for row_num, record in enumerate(reader, start=2):
            try:
                vector = np.array([float(record[f"feat_{i}"])
                                   for i in range(FEATURE_COUNT)], dtype=np.float64)
                hand = float(record["handedness"])
                label = record["label"]
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"invalid legacy CSV row {row_num}") from exc
            if (not np.isfinite(vector).all() or hand not in (0., 1.)
                    or not label or not isinstance(label, str)):
                raise ValueError(f"invalid legacy CSV row {row_num}")
            as_f32 = vector.astype("<f4")
            if not np.isfinite(as_f32).all():
                raise ValueError(f"unrepresentable legacy feature at row {row_num}")
            labels.add(label)
            rows.append((digest, row_num, as_f32.tobytes(), label, int(hand)))
    if not rows:
        raise ValueError("legacy CSV has no rows")
    status = "compatible" if labels.issubset(BASE_LABELS) else "quarantined"
    with store.conn:
        store.conn.execute(
            """INSERT INTO legacy_sources
                 (source_id, source_filename, source_sha256, feature_contract,
                  status, row_count, imported_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (digest, source.name, digest, "derived-features-69-f32-v1-unverified-provenance",
             status, len(rows), datetime.now(timezone.utc).isoformat()),
        )
        store.conn.executemany(
            """INSERT INTO legacy_rows
                 (source_id, row_number, features_f32_le, original_label, handedness_binary)
               VALUES (?, ?, ?, ?, ?)""", rows,
        )
    return LegacyImportResult(digest, len(rows), status)


def load_legacy_source(
    store: DatasetStore, source_id: str, *, compatible_only: bool = True
) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    """Retrieve a separate immutable feature-only partition from SQLite."""
    source = store.conn.execute(
        "SELECT status, row_count FROM legacy_sources WHERE source_id=?", (source_id,)
    ).fetchone()
    if source is None:
        raise KeyError(source_id)
    if compatible_only and source["status"] != "compatible":
        raise ValueError("quarantined legacy source is not eligible for training")
    records = store.conn.execute(
        """SELECT row_number, features_f32_le, original_label, handedness_binary
           FROM legacy_rows WHERE source_id=? ORDER BY row_number""", (source_id,)
    ).fetchall()
    if len(records) != source["row_count"]:
        raise ValueError("legacy source row count does not match inventory")
    vectors = np.stack([np.frombuffer(r["features_f32_le"], dtype="<f4")
                        for r in records]).astype(np.float32)
    labels = [r["original_label"] for r in records]
    handedness = np.array([r["handedness_binary"] for r in records], dtype=np.float32)
    ids = [f"legacy-{source_id[:12]}-row-{r['row_number']}" for r in records]
    return vectors, labels, handedness, ids
