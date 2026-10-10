"""Explicit Studio capture into existing canonical workspace from one MediaPipe stream.

All methods run on the sidecar's asyncio event-loop thread. The callback thread
never writes SQLite; capture consumes only the latest *processed* observation.
No camera frame or video is retained or persisted.
"""
from __future__ import annotations

from pathlib import Path
from collections import Counter
from uuid import uuid4

from handd_core.capture_sampling import CaptureSampler
from handd_core.collect_cli import _local_device_config
from handd_core.dataset_store import DatasetStore


class StudioCollection:
    def __init__(self, workspace: Path, *, device_config: Path | None = None):
        self.workspace = Path(workspace)
        if not (self.workspace / 'handd.sqlite').is_file():
            raise ValueError('Select an existing workspace before collecting')
        self.device_config = device_config or _local_device_config()
        self._store: DatasetStore | None = None
        self._sampler: CaptureSampler | None = None
        self._frame_index = 0
        self._state = 'idle'
        self._session_id: str | None = None
        self._capture_id: str | None = None
        self._gesture: str | None = None
        self._participant: str | None = None
        self._quality_skips: Counter[str] = Counter()
        self._last_skip_reason: str | None = None

    def status(self) -> dict:
        return {
            'state': self._state,
            'count': self._sampler.count if self._sampler else 0,
            'target': int(self._sampler._progress['target']) if self._sampler else 0,
            'session_id': self._session_id, 'capture_id': self._capture_id,
            'participant': self._participant, 'gesture': self._gesture,
            'quality_skips': dict(self._quality_skips),
            'last_skip_reason': self._last_skip_reason,
        }

    def start(self, *, participant: str, gesture: str, hand: str,
              target: int = 120, interval_ms: int = 100) -> dict:
        if self._state in ('capturing', 'paused'):
            raise ValueError('Finish or resume the active Capture first')
        if participant not in ('P001', 'P002'):
            raise ValueError('Development collection requires P001 or P002')
        if not isinstance(gesture, str) or not (1 <= len(gesture) <= 64) \
                or any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789- '
                       for ch in gesture):
            raise ValueError('Gesture label must be 1-64 safe characters')
        if hand not in ('Right', 'Left', 'Any'):
            raise ValueError('Capture hand must be Right, Left or Any')
        if type(target) is not int or not (1 <= target <= 10000):
            raise ValueError('Target must be 1 to 10000 samples')
        if type(interval_ms) is not int or not (20 <= interval_ms <= 10000):
            raise ValueError('Sample interval must be 20 to 10000 ms')

        store = DatasetStore(self.workspace / 'handd.sqlite')
        try:
            participant_exists = store.conn.execute(
                'SELECT 1 FROM participants WHERE participant_id=?', (participant,),
            ).fetchone()
            if participant_exists is None:
                store.create_participant(participant)
            session_id, capture_id = uuid4().hex, uuid4().hex
            store.create_session(session_id, participant)
            store.create_capture(capture_id, session_id, gesture, hand, target)
            sampler = CaptureSampler.for_local_device(
                store, capture_id, interval_ms=interval_ms, camera='0',
                device_config_path=self.device_config,
            )
        except BaseException:
            store.close()
            raise
        self.close()
        self._store = store
        self._sampler = sampler
        self._frame_index = 0
        self._session_id, self._capture_id = session_id, capture_id
        self._participant, self._gesture = participant, gesture
        self._quality_skips.clear()
        self._last_skip_reason = None
        self._state = 'capturing'
        return self.status()

    def pause(self) -> dict:
        if self._state != 'capturing':
            raise ValueError('No running Capture to pause')
        self._sampler.pause()
        self._state = 'paused'
        return self.status()

    def resume(self) -> dict:
        if self._state != 'paused':
            raise ValueError('No paused Capture to resume')
        self._sampler.resume()
        self._state = 'capturing'
        return self.status()

    def finish(self) -> dict:
        if self._state not in ('capturing','paused'):
            raise ValueError('No active Capture to finish')
        self._state = 'finished'
        state = self.status()
        self.close()
        return state

    def offer_result(self, result, timestamp_ms: int) -> str | None:
        if self._state != 'capturing':
            return None
        self._frame_index += 1
        frame_idx = self._frame_index
        if not result.hand_landmarks:
            self._mark_skip('no_hand')
            return None
        from numpy import array, float64
        reasons = []
        for raw_handedness, image_points, world_points in zip(
            result.handedness, result.hand_landmarks, result.hand_world_landmarks,
        ):
            if not raw_handedness:
                reasons.append('bad_tracking')
                continue
            try:
                make_array = lambda points: array(
                    [[p.x,p.y,p.z] for p in points], dtype=float64)
                sample_id = self._sampler.offer(
                    frame_index=frame_idx, timestamp_ms=timestamp_ms,
                    raw_mp_handedness=raw_handedness[0].category_name,
                    image_landmarks=make_array(image_points),
                    world_landmarks=make_array(world_points),
                )
            except (ValueError, TypeError, AttributeError, OverflowError):
                reasons.append('bad_tracking')
                continue  # bad tracking data does not stop the live camera
            if sample_id is not None:
                self._last_skip_reason = None
                if self._sampler.finished:
                    self._state = 'complete'
                    self.close()
                return sample_id
            reasons.append(self._sampler.last_skip_reason or 'bad_tracking')
        # One discarded observation batch counts once, not once per hand.
        priority=('outside_frame','bad_tracking','wrong_hand','sampling_interval','stale')
        self._mark_skip(next((reason for reason in priority if reason in reasons),'no_hand'))
        return None

    def _mark_skip(self, reason: str) -> None:
        self._last_skip_reason = reason
        self._quality_skips[reason] += 1

    def fail(self) -> dict:
        """A disk/SQLite write error halts capture without crashing inference."""
        self._state = 'error'
        state = self.status()
        state['error'] = 'Capture stopped due to a workspace storage error'
        self.close()
        return state

    def close(self) -> None:
        if self._store:
            self._store.close()
            self._store = None
