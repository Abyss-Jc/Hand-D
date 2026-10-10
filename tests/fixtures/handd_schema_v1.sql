-- Hand-D schema v1 from commit 9458984 (historical migration fixture).
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
