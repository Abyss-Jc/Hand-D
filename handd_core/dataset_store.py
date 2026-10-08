"""Canonical SQLite data/curation core for Hand-D v2.

Source observations are immutable; Review Status and Lifecycle Status evolve
independently with append-only audit events. No image/video is stored.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np


SCHEMA_VERSION = 2
REVIEW_STATUSES = frozenset({'unreviewed', 'accepted', 'rejected'})
LIFECYCLE_STATUSES = frozenset({'active', 'dropped'})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _landmarks_json(value: np.ndarray, name: str) -> str:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f'{name} must be numeric 21x3 landmarks') from exc
    if array.shape != (21, 3) or not np.isfinite(array).all():
        raise ValueError(f'{name} must contain 21x3 finite values')
    return json.dumps(array.tolist(), separators=(',', ':'), allow_nan=False)


class DatasetStore:
    """One-writer local SQLite workspace store with explicit transactions."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute('PRAGMA foreign_keys = ON')
        try:
            version = self.conn.execute('PRAGMA user_version').fetchone()[0]
            if version > SCHEMA_VERSION:
                raise ValueError('Workspace database schema is newer than this Hand-D build')
            self._initialize_schema()
        except BaseException:
            self.conn.close()
            raise

    def _initialize_schema(self) -> None:
        self.conn.executescript('''
            CREATE TABLE IF NOT EXISTS participants (
                participant_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS collection_sessions (
                session_id TEXT PRIMARY KEY,
                participant_id TEXT NOT NULL REFERENCES participants(participant_id),
                started_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS captures (
                capture_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES collection_sessions(session_id),
                gesture TEXT NOT NULL CHECK(length(gesture) > 0),
                hand TEXT NOT NULL CHECK(hand IN ('Right', 'Left', 'Any')),
                target INTEGER NOT NULL CHECK(target > 0),
                started_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS samples (
                sample_id TEXT PRIMARY KEY,
                capture_id TEXT NOT NULL REFERENCES captures(capture_id),
                frame_index INTEGER NOT NULL CHECK(frame_index >= 0),
                timestamp_ms INTEGER NOT NULL CHECK(timestamp_ms >= 0),
                raw_mp_handedness TEXT NOT NULL CHECK(raw_mp_handedness IN ('Left','Right')),
                image_landmarks TEXT NOT NULL,
                world_landmarks TEXT NOT NULL,
                provenance_json TEXT NOT NULL,
                review_status TEXT NOT NULL DEFAULT 'unreviewed'
                    CHECK(review_status IN ('unreviewed','accepted','rejected')),
                lifecycle_status TEXT NOT NULL DEFAULT 'active'
                    CHECK(lifecycle_status IN ('active','dropped')),
                recorded_at TEXT NOT NULL,
                UNIQUE(capture_id, frame_index)
            );
            CREATE TABLE IF NOT EXISTS review_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_id TEXT NOT NULL REFERENCES samples(sample_id),
                previous_status TEXT NOT NULL,
                new_status TEXT NOT NULL,
                reason TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS lifecycle_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_id TEXT NOT NULL REFERENCES samples(sample_id),
                previous_status TEXT NOT NULL,
                new_status TEXT NOT NULL,
                reason TEXT,
                created_at TEXT NOT NULL
            );
            -- Historical 69-feature rows CANNOT become canonical Samples:
            -- they lack original landmarks and Session provenance.
            CREATE TABLE IF NOT EXISTS legacy_sources (
                source_id TEXT PRIMARY KEY,
                source_filename TEXT NOT NULL,
                source_sha256 TEXT NOT NULL UNIQUE,
                feature_contract TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('compatible','quarantined')),
                row_count INTEGER NOT NULL CHECK(row_count >= 1),
                imported_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS legacy_rows (
                source_id TEXT NOT NULL REFERENCES legacy_sources(source_id),
                row_number INTEGER NOT NULL CHECK(row_number >= 2),
                features_f32_le BLOB NOT NULL CHECK(length(features_f32_le)=276),
                original_label TEXT NOT NULL,
                handedness_binary INTEGER NOT NULL CHECK(handedness_binary IN (0,1)),
                PRIMARY KEY (source_id, row_number)
            );
            CREATE TRIGGER IF NOT EXISTS legacy_sources_no_update
            BEFORE UPDATE ON legacy_sources
            BEGIN SELECT RAISE(ABORT, 'legacy source is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS legacy_sources_no_delete
            BEFORE DELETE ON legacy_sources
            BEGIN SELECT RAISE(ABORT, 'legacy source is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS legacy_rows_no_update
            BEFORE UPDATE ON legacy_rows
            BEGIN SELECT RAISE(ABORT, 'legacy row is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS legacy_rows_no_delete
            BEFORE DELETE ON legacy_rows
            BEGIN SELECT RAISE(ABORT, 'legacy row is immutable'); END;
            CREATE INDEX IF NOT EXISTS samples_capture_idx ON samples(capture_id);
            CREATE INDEX IF NOT EXISTS samples_eligibility_idx
                ON samples(review_status, lifecycle_status);
            -- Direct SQL state changes must be audited too. Domain API inserts
            -- a reasoned event first in the same transaction; these triggers
            -- add a reasonless event only when no matching latest event exists.
            CREATE TRIGGER IF NOT EXISTS review_changes_audited
            AFTER UPDATE OF review_status ON samples
            WHEN OLD.review_status <> NEW.review_status
            BEGIN
              INSERT INTO review_events
                (sample_id, previous_status, new_status, reason, created_at)
              SELECT NEW.sample_id, OLD.review_status, NEW.review_status, NULL,
                     strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
              WHERE NOT EXISTS (
                SELECT 1 FROM review_events WHERE event_id = (
                  SELECT max(event_id) FROM review_events WHERE sample_id=NEW.sample_id
                ) AND previous_status=OLD.review_status
                  AND new_status=NEW.review_status
              );
            END;
            CREATE TRIGGER IF NOT EXISTS lifecycle_changes_audited
            AFTER UPDATE OF lifecycle_status ON samples
            WHEN OLD.lifecycle_status <> NEW.lifecycle_status
            BEGIN
              INSERT INTO lifecycle_events
                (sample_id, previous_status, new_status, reason, created_at)
              SELECT NEW.sample_id, OLD.lifecycle_status, NEW.lifecycle_status, NULL,
                     strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
              WHERE NOT EXISTS (
                SELECT 1 FROM lifecycle_events WHERE event_id = (
                  SELECT max(event_id) FROM lifecycle_events WHERE sample_id=NEW.sample_id
                ) AND previous_status=OLD.lifecycle_status
                  AND new_status=NEW.lifecycle_status
              );
            END;
            CREATE TRIGGER IF NOT EXISTS samples_core_immutable
            BEFORE UPDATE OF capture_id,frame_index,timestamp_ms,raw_mp_handedness,
                image_landmarks,world_landmarks,provenance_json,recorded_at ON samples
            BEGIN SELECT RAISE(ABORT, 'canonical Sample observation is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS samples_not_deletable
            BEFORE DELETE ON samples
            BEGIN SELECT RAISE(ABORT, 'canonical Sample may not be deleted'); END;
            CREATE TRIGGER IF NOT EXISTS review_events_immutable_update
            BEFORE UPDATE ON review_events
            BEGIN SELECT RAISE(ABORT, 'review event is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS review_events_immutable_delete
            BEFORE DELETE ON review_events
            BEGIN SELECT RAISE(ABORT, 'review event is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS lifecycle_events_immutable_update
            BEFORE UPDATE ON lifecycle_events
            BEGIN SELECT RAISE(ABORT, 'lifecycle event is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS lifecycle_events_immutable_delete
            BEFORE DELETE ON lifecycle_events
            BEGIN SELECT RAISE(ABORT, 'lifecycle event is immutable'); END;
        ''')
        self.conn.execute(f'PRAGMA user_version = {SCHEMA_VERSION}')

    def close(self) -> None:
        self.conn.close()

    def create_participant(self, participant_id: str) -> None:
        with self.conn:
            self.conn.execute(
                'INSERT INTO participants (participant_id, created_at) VALUES (?, ?)',
                (participant_id, _utc_now()),
            )

    def create_session(self, session_id: str, participant_id: str) -> None:
        with self.conn:
            self.conn.execute(
                'INSERT INTO collection_sessions (session_id, participant_id, started_at) VALUES (?, ?, ?)',
                (session_id, participant_id, _utc_now()),
            )

    def create_capture(self, capture_id: str, session_id: str, gesture: str,
                       hand: str, target: int = 120) -> None:
        with self.conn:
            self.conn.execute(
                'INSERT INTO captures (capture_id, session_id, gesture, hand, target, started_at)'
                ' VALUES (?, ?, ?, ?, ?, ?)',
                (capture_id, session_id, gesture, hand, target, _utc_now()),
            )

    def add_sample(self, sample_id: str, capture_id: str, frame_index: int,
                   image_landmarks: np.ndarray, world_landmarks: np.ndarray, *,
                   raw_mp_handedness: str, timestamp_ms: int,
                   provenance: dict[str, Any]) -> None:
        if raw_mp_handedness not in ('Left', 'Right'):
            raise ValueError('raw_mp_handedness must be Left or Right')
        if not isinstance(provenance, dict):
            raise ValueError('provenance must be a metadata dictionary')
        image_json = _landmarks_json(image_landmarks, 'image_landmarks')
        world_json = _landmarks_json(world_landmarks, 'world_landmarks')
        try:
            provenance_json = json.dumps(provenance, separators=(',', ':'),
                                         allow_nan=False, sort_keys=True)
        except (ValueError, TypeError) as exc:
            raise ValueError('provenance must be serializable and finite') from exc
        with self.conn:
            self.conn.execute(
                '''INSERT INTO samples (sample_id, capture_id, frame_index, timestamp_ms,
                    raw_mp_handedness, image_landmarks, world_landmarks,
                    provenance_json, recorded_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (sample_id, capture_id, frame_index, timestamp_ms,
                 raw_mp_handedness, image_json, world_json, provenance_json, _utc_now()),
            )

    @staticmethod
    def _decode_sample(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result['image_landmarks'] = np.asarray(json.loads(result['image_landmarks']),
                                               dtype=np.float64)
        result['world_landmarks'] = np.asarray(json.loads(result['world_landmarks']),
                                               dtype=np.float64)
        result['provenance'] = json.loads(result.pop('provenance_json'))
        return result

    def get_sample(self, sample_id: str) -> dict[str, Any]:
        row = self.conn.execute(
            '''SELECT s.*, c.gesture, c.hand, c.session_id, cs.participant_id
               FROM samples s JOIN captures c ON s.capture_id=c.capture_id
               JOIN collection_sessions cs ON c.session_id=cs.session_id
               WHERE s.sample_id=?''',
            (sample_id,),
        ).fetchone()
        if row is None:
            raise KeyError(sample_id)
        return self._decode_sample(row)

    def count_samples(self) -> int:
        return self.conn.execute('SELECT count(*) FROM samples').fetchone()[0]

    def get_capture_progress(self, capture_id: str) -> dict[str, Any]:
        """Return durable quota/identity/progress for a resumable capture."""
        row = self.conn.execute(
            '''SELECT c.hand, c.gesture, c.target,
                      count(s.sample_id) AS count,
                      max(s.frame_index) AS last_frame_index,
                      max(s.timestamp_ms) AS last_timestamp_ms
               FROM captures c
               LEFT JOIN samples s ON s.capture_id=c.capture_id
               WHERE c.capture_id=? GROUP BY c.capture_id''',
            (capture_id,),
        ).fetchone()
        if row is None:
            raise KeyError(capture_id)
        return dict(row)

    def list_eligible_samples(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            '''SELECT s.*, c.gesture, c.hand, c.session_id, cs.participant_id
               FROM samples s JOIN captures c ON s.capture_id=c.capture_id
               JOIN collection_sessions cs ON c.session_id=cs.session_id
               WHERE s.review_status='accepted' AND s.lifecycle_status='active'
               ORDER BY cs.participant_id, c.session_id, s.capture_id, s.frame_index''',
        ).fetchall()
        return [self._decode_sample(row) for row in rows]

    def get_active_review_counts(self) -> dict[str, int]:
        """Counts for review readiness, excluding soft-dropped samples."""
        counts = {status: 0 for status in REVIEW_STATUSES}
        for row in self.conn.execute(
            '''SELECT review_status, count(*) AS total FROM samples
               WHERE lifecycle_status='active' GROUP BY review_status''',
        ):
            counts[row['review_status']] = row['total']
        return counts

    def list_samples_overview(self, *, review_status: str | None = None,
                              limit: int = 100,
                              include_dropped: bool = False) -> list[dict[str, Any]]:
        """Small, read-only summary for a human review queue (no raw arrays)."""
        if review_status is not None and review_status not in REVIEW_STATUSES:
            raise ValueError('invalid review status filter')
        if limit <= 0:
            raise ValueError('limit must be positive')
        return [dict(row) for row in self.conn.execute(
            '''SELECT s.sample_id, c.gesture, c.session_id, cs.participant_id,
                      s.review_status, s.lifecycle_status, s.timestamp_ms
               FROM samples s JOIN captures c ON c.capture_id=s.capture_id
               JOIN collection_sessions cs ON cs.session_id=c.session_id
               WHERE (? IS NULL OR s.review_status=?)
                 AND (? OR s.lifecycle_status='active')
               ORDER BY s.recorded_at, s.sample_id LIMIT ?''',
            (review_status, review_status, include_dropped, limit),
        ).fetchall()]

    def _transition(self, sample_id: str, new_status: str, *, field: str,
                    allowed: frozenset[str], table: str, reason: str | None) -> None:
        if new_status not in allowed:
            raise ValueError(f'invalid {field}: {new_status}')
        # field and table are private fixed constants set only by public wrappers.
        with self.conn:
            row = self.conn.execute(
                f'SELECT {field} FROM samples WHERE sample_id=?',
                (sample_id,),
            ).fetchone()
            if row is None:
                raise KeyError(sample_id)
            previous = row[field]
            if previous == new_status:
                return
            self.conn.execute(
                f'''INSERT INTO {table}
                    (sample_id, previous_status, new_status, reason, created_at)
                    VALUES (?, ?, ?, ?, ?)''',
                (sample_id, previous, new_status, reason, _utc_now()),
            )
            self.conn.execute(
                f'UPDATE samples SET {field}=? WHERE sample_id=?',
                (new_status, sample_id),
            )

    def set_review_status(self, sample_id: str, new_status: str,
                          reason: str | None = None) -> None:
        self._transition(sample_id, new_status, field='review_status',
                         allowed=REVIEW_STATUSES, table='review_events', reason=reason)

    def set_lifecycle_status(self, sample_id: str, new_status: str,
                             reason: str | None = None) -> None:
        self._transition(sample_id, new_status, field='lifecycle_status',
                         allowed=LIFECYCLE_STATUSES, table='lifecycle_events', reason=reason)

    def _list_events(self, sample_id: str, table: str) -> list[dict[str, Any]]:
        return [dict(row) for row in self.conn.execute(
            f'SELECT * FROM {table} WHERE sample_id=? ORDER BY event_id',
            (sample_id,),
        ).fetchall()]

    def list_review_events(self, sample_id: str) -> list[dict[str, Any]]:
        return self._list_events(sample_id, 'review_events')

    def list_lifecycle_events(self, sample_id: str) -> list[dict[str, Any]]:
        return self._list_events(sample_id, 'lifecycle_events')
