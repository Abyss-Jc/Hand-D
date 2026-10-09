"""Hand-D v2 transient tracking/gesture engine.

Python owns inference and temporal gesture state; frontend owns all drawing,
pointer smoothing and document state. Model Artifacts are verified separately.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from threading import Lock
from time import perf_counter
from uuid import uuid4

import numpy as np

from handd_core.feature_transform import canonicalize_world_landmarks
from handd_core.model_artifact import load_model_artifact


BASE_ACTIONS = {
    'drawing': {'Index_Finger': 'draw', 'Fist': 'erase'},
    'modifier': {'Ruler': 'ruler', 'Thumb_Up': 'tool_adjust'},
}


@dataclass
class _GestureState:
    stable: str | None = None
    candidate: str | None = None
    first_seen_ms: int = -1
    count: int = 0

    def reset(self) -> None:
        self.stable = None
        self.candidate = None
        self.first_seen_ms = -1
        self.count = 0

    def update(self, raw: str, at_ms: int, *, frames: int, hold_ms: int) -> str | None:
        if raw != self.candidate:
            self.candidate = raw
            self.first_seen_ms = at_ms
            self.count = 1
        else:
            self.count += 1
        if self.count >= frames and at_ms - self.first_seen_ms >= hold_ms:
            self.stable = raw
        return self.stable


class GestureRuntime:
    """Owner-thread processor for newest MediaPipe results (no camera ownership)."""

    def __init__(self, *, predictor=None, drawing_hand='Right',
                 modifier_hand='Left', stable_frames=3, stable_ms=60):
        if {drawing_hand, modifier_hand} != {'Left', 'Right'}:
            raise ValueError('Drawing/Modifier Hands must be different physical hands')
        if stable_frames < 1 or stable_ms < 0:
            raise ValueError('invalid gesture stabilization parameters')
        self.predictor = predictor
        self.drawing_hand = drawing_hand
        self.modifier_hand = modifier_hand
        self.stable_frames = stable_frames
        self.stable_ms = stable_ms
        self.runtime_session_id = uuid4().hex
        self._states = {'drawing': _GestureState(), 'modifier': _GestureState()}
        self._seq = 0
        self._last_timestamp_ms = -1
        self._active_model_id = None
        self._actions = {role: dict(mapping) for role, mapping in BASE_ACTIONS.items()}
        self._stale_released = False
        self._model_health = 'ready' if predictor is not None else 'unavailable'
        self._success_model_health = self._model_health
        self._camera_health = 'starting'
        self._compute_times_ms = deque(maxlen=120)

    def snapshot(self) -> dict:
        """Transient READY/resync state, never frontend document or stroke data."""
        return {
            'runtime_session_id': self.runtime_session_id, 'seq': self._seq,
            'timestamp_ms': self._last_timestamp_ms,
            'active_model_id': self._active_model_id,
            'roles': {'drawing': self.drawing_hand, 'modifier': self.modifier_hand},
            'health': self._health(),
        }

    def _health(self) -> dict:
        health = {'model': self._model_health, 'camera': self._camera_health}
        if self._compute_times_ms:
            health['compute_ms_last'] = round(self._compute_times_ms[-1], 3)
            health['compute_ms_p95'] = round(
                float(np.percentile(self._compute_times_ms, 95)), 3,
            )
        # This is NOT sensor→IPC→frontend latency.
        return health

    def activate_model(self, path) -> dict:
        """Explicit, atomic Candidate activation after Model Artifact validation."""
        model, manifest = load_model_artifact(path)
        self.predictor = model
        self._model_health = 'ready'
        self._success_model_health = 'ready'
        self._active_model_id = manifest['artifact_id']
        for role in self._states:
            self._states[role].reset()
            # An old custom label must not acquire actions in the new model.
            self._actions[role] = {
                name: action for name, action in self._actions[role].items()
                if name in model.label_order
            }
        return {'artifact_id': self._active_model_id, 'label_order': list(model.label_order)}

    def activate_legacy_checkpoint(self, path) -> dict:
        """Opt-in old checkpoint, clearly marked unverified in every health update."""
        from handd_core.legacy_model import load_legacy_checkpoint
        predictor, metadata = load_legacy_checkpoint(path)
        self.predictor = predictor
        self._active_model_id = metadata['artifact_id']
        self._model_health = 'legacy_unverified'
        self._success_model_health = 'legacy_unverified'
        for state in self._states.values():
            state.reset()
        return metadata

    def set_actions(self, *, drawing=None, modifier=None) -> None:
        """Workspace-configured, role-safe built-in actions (never arbitrary code)."""
        pending = {}
        for role, requested in (('drawing', drawing), ('modifier', modifier)):
            if requested is None:
                continue
            if not isinstance(requested, dict):
                raise ValueError('action mapping must be a dictionary')
            available = getattr(self.predictor, 'label_order', ()) if self.predictor else ()
            if any(label == 'Idle' or label not in available
                   or action not in BASE_ACTIONS[role].values()
                   for label, action in requested.items()):
                raise ValueError('gesture/action is unrecognized, unsafe or role-incompatible')
            pending[role] = dict(requested)
        for role, requested in pending.items():
            self._actions[role] = dict(requested)
            self._states[role].reset()

    def set_roles(self, *, drawing_hand: str, modifier_hand: str) -> None:
        """Only explicit configuration changes physical hand ownership."""
        if {drawing_hand, modifier_hand} != {'Right', 'Left'}:
            raise ValueError('Drawing/Modifier Hands must be distinct Left/Right')
        if (drawing_hand, modifier_hand) != (self.drawing_hand, self.modifier_hand):
            self.drawing_hand = drawing_hand
            self.modifier_hand = modifier_hand
            for state in self._states.values():
                state.reset()

    def process_result(self, result, timestamp_ms: int) -> dict | None:
        """A newer callback advances transient state; stale timestamps do not."""
        if timestamp_ms <= self._last_timestamp_ms:
            return None
        began = perf_counter()
        self._last_timestamp_ms = timestamp_ms
        self._stale_released = False
        self._seq += 1
        hands = {}
        ambiguous = set()
        for categories, image, world in zip(
            result.handedness, result.hand_landmarks, result.hand_world_landmarks
        ):
            if not categories or categories[0].category_name not in ('Left', 'Right'):
                continue
            raw_hand = categories[0].category_name
            physical_hand = 'Right' if raw_hand == 'Left' else 'Left'
            try:
                image_array = np.array([[p.x, p.y, p.z] for p in image], dtype=np.float64)
                world_array = np.array([[p.x, p.y, p.z] for p in world], dtype=np.float64)
                features = canonicalize_world_landmarks(world_array, raw_hand)
                if image_array.shape != (21, 3) or not np.isfinite(image_array).all():
                    continue
                if features is None:
                    continue
            except (TypeError, ValueError, AttributeError):
                continue
            tip = image_array[8, :2]
            if not np.all((tip >= 0) & (tip <= 1)):
                continue
            if physical_hand in hands:
                ambiguous.add(physical_hand)
            else:
                hands[physical_hand] = (image_array, features)
        for side in ambiguous:
            hands.pop(side, None)
        self._camera_health = 'tracking' if hands else 'no_hands'
        role_data = {}
        inference_failed = False
        for role, physical_hand in (('drawing', self.drawing_hand),
                                     ('modifier', self.modifier_hand)):
            item = hands.get(physical_hand)
            state = self._states[role]
            if item is None:
                state.reset()
                role_data[role] = dict(physical_hand=physical_hand, pointer=None,
                                       landmarks=None, raw_gesture=None,
                                       stable_gesture=None, action=None)
                continue
            image, features = item
            tip = image[8, :2]
            pointer = {'x': float(tip[0]), 'y': float(tip[1])}
            try:
                raw = self.predictor.predict(features.reshape(1, 69))[0] if self.predictor else None
                self._model_health = (
                    self._success_model_health if self.predictor else 'unavailable'
                )
            except (ValueError, RuntimeError, TypeError, IndexError):
                raw = None
                inference_failed = True
                self._model_health = 'error'
            if raw:
                stable = state.update(raw, timestamp_ms, frames=self.stable_frames,
                                      hold_ms=self.stable_ms)
            else:
                state.reset()
                stable = None
            landmark_xy = np.clip(image[:, :2], 0., 1.)
            role_data[role] = dict(
                physical_hand=physical_hand, pointer=pointer,
                landmarks=[{'x': float(x), 'y': float(y)} for x, y in landmark_xy],
                raw_gesture=raw,
                stable_gesture=stable,
                action=self._actions[role].get(stable),
            )
        if inference_failed:
            self._model_health = 'error'
            for role, state in self._states.items():
                state.reset()
                role_data[role]['stable_gesture'] = None
                role_data[role]['action'] = None
        self._compute_times_ms.append((perf_counter() - began) * 1000)
        return {
            'runtime_session_id': self.runtime_session_id, 'seq': self._seq,
            'timestamp_ms': timestamp_ms, 'type': 'runtime.update',
            'payload': {**role_data, 'landmark_contract': 'handd.v2.image21.xy.1',
                        'health': self._health()},
        }

    def expire_if_stale(self, now_ms: int, *, max_gap_ms: int = 250) -> dict | None:
        """Release active gestures once when camera callbacks stop arriving."""
        if max_gap_ms <= 0:
            raise ValueError('max_gap_ms must be positive')
        if (self._last_timestamp_ms < 0 or self._stale_released
                or now_ms - self._last_timestamp_ms <= max_gap_ms):
            return None
        self._stale_released = True
        self._last_timestamp_ms = now_ms
        self._seq += 1
        self._camera_health = 'stale'
        for state in self._states.values():
            state.reset()
        return {
            'runtime_session_id': self.runtime_session_id,
            'seq': self._seq, 'timestamp_ms': now_ms, 'type': 'runtime.update',
            'payload': {
                'drawing': {'physical_hand': self.drawing_hand, 'pointer': None,
                            'landmarks': None,
                            'raw_gesture': None, 'stable_gesture': None, 'action': None},
                'modifier': {'physical_hand': self.modifier_hand, 'pointer': None,
                             'landmarks': None,
                             'raw_gesture': None, 'stable_gesture': None, 'action': None},
                'health': self._health(),
            },
        }


class LatestRuntimeResults:
    """Callback-thread latest-only mailbox; owner thread executes inference."""

    def __init__(self):
        self._lock = Lock()
        self._pending = deque(maxlen=1)
        self._last_callback_ms = -1
        self._stats = {'received_callbacks': 0, 'superseded_callbacks': 0,
                       'stale_callbacks': 0, 'processed': 0}

    def on_result(self, result, output_image, timestamp_ms: int) -> None:
        with self._lock:
            self._stats['received_callbacks'] += 1
            if timestamp_ms <= self._last_callback_ms:
                self._stats['stale_callbacks'] += 1
                return
            self._last_callback_ms = timestamp_ms
            if self._pending:
                self._stats['superseded_callbacks'] += 1
            self._pending.clear()
            self._pending.append((result, timestamp_ms))

    def drain_to(self, runtime: GestureRuntime) -> dict | None:
        with self._lock:
            if not self._pending:
                return None
            result, ts = self._pending.pop()
        output = runtime.process_result(result, ts)
        if output is not None:
            with self._lock:
                self._stats['processed'] += 1
        return output

    def statistics(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)


class RuntimeEventGate:
    """A new READY snapshot invalidates every event from an older session."""

    def __init__(self):
        self._runtime_session_id = None
        self._last_seq = -1
        self._last_timestamp_ms = -1

    def install_snapshot(self, snapshot: dict) -> None:
        if (not isinstance(snapshot.get('runtime_session_id'), str)
                or not snapshot['runtime_session_id']
                or not isinstance(snapshot.get('seq'), int)):
            raise ValueError('invalid READY snapshot')
        self._runtime_session_id = snapshot['runtime_session_id']
        self._last_seq = snapshot['seq']
        self._last_timestamp_ms = snapshot.get('timestamp_ms', -1)

    def accept(self, event: dict) -> bool:
        if not isinstance(event, dict):
            return False
        if (event.get('runtime_session_id') != self._runtime_session_id
                or self._runtime_session_id is None
                or event.get('type') != 'runtime.update'
                or not isinstance(event.get('seq'), int)
                or not isinstance(event.get('timestamp_ms'), int)
                or event['seq'] <= self._last_seq
                or event['timestamp_ms'] <= self._last_timestamp_ms):
            return False
        self._last_seq = event['seq']
        self._last_timestamp_ms = event['timestamp_ms']
        return True
