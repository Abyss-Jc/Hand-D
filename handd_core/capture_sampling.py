"""Time-based collection from a MediaPipe observation stream.

No camera ownership or GUI imports: the camera adapter calls `offer` with
raw image/world landmarks and the raw mirrored MediaPipe hand label.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import platform
from uuid import uuid4

import numpy as np

from handd_core.dataset_store import DatasetStore
from handd_core.feature_transform import canonicalize_world_landmarks


class CaptureSampler:
    @classmethod
    def for_local_device(cls, store: DatasetStore, capture_id: str, *,
                         interval_ms: int, camera: str,
                         device_config_path: Path) -> 'CaptureSampler':
        """Create or load a device pseudonym in local application config."""
        path = Path(device_config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            device_id = path.read_text(encoding='utf-8').strip()
        else:
            device_id = uuid4().hex
            with os.fdopen(fd, 'w', encoding='utf-8') as handle:
                handle.write(device_id + '\n')
        if len(device_id) != 32 or any(c not in '0123456789abcdef' for c in device_id):
            raise ValueError('invalid local device pseudonym')
        software = {'python': platform.python_version()}
        for package in ('mediapipe', 'opencv-contrib-python'):
            try:
                software[package] = version(package)
            except PackageNotFoundError:
                software[package] = 'unavailable'
        return cls(store, capture_id, interval_ms=interval_ms,
                   camera=camera, device_id=device_id, software=software)

    def __init__(self, store: DatasetStore, capture_id: str, *, interval_ms: int,
                 device_id: str, camera: str, software: dict[str, str]):
        if type(interval_ms) is not int or interval_ms <= 0:
            raise ValueError('interval_ms must be a positive integer')
        if not device_id or not camera:
            raise ValueError('device_id and camera are required')
        self.store = store
        self.capture_id = capture_id
        self.interval_ms = interval_ms
        self.device_id = device_id
        self.camera = camera
        self.software = dict(software)
        self.paused = False
        self._progress = store.get_capture_progress(capture_id)

    @property
    def count(self) -> int:
        return int(self._progress['count'])

    @property
    def finished(self) -> bool:
        return self.count >= self._progress['target']

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def offer(self, *, frame_index: int, timestamp_ms: int,
              raw_mp_handedness: str, image_landmarks: np.ndarray,
              world_landmarks: np.ndarray) -> str | None:
        """Persist one eligible observation and return its ID; otherwise None."""
        if self.paused or self.finished:
            return None
        if type(frame_index) is not int or frame_index < 0:
            raise ValueError('frame_index must be a nonnegative integer')
        if type(timestamp_ms) is not int or timestamp_ms < 0:
            raise ValueError('timestamp_ms must be a nonnegative integer')
        last_frame = self._progress['last_frame_index']
        last_ms = self._progress['last_timestamp_ms']
        if last_frame is not None and frame_index <= last_frame:
            return None  # stale/replayed result
        if last_ms is not None and timestamp_ms < last_ms + self.interval_ms:
            return None
        if raw_mp_handedness not in ('Left', 'Right'):
            return None
        # Existing mirrored-camera contract, to be independently tested on
        # hardware before trusting collection handedness as real-world fact.
        actual_hand = 'Right' if raw_mp_handedness == 'Left' else 'Left'
        if self._progress['hand'] not in ('Any', actual_hand):
            return None
        try:
            image = np.asarray(image_landmarks, dtype=np.float64)
            world = np.asarray(world_landmarks, dtype=np.float64)
            if image.shape != (21, 3) or not np.isfinite(image).all():
                return None
            if canonicalize_world_landmarks(world, raw_mp_handedness) is None:
                return None
        except (ValueError, TypeError, OverflowError):
            return None
        sample_id = uuid4().hex
        self.store.add_sample(
            sample_id, self.capture_id, frame_index, image, world,
            raw_mp_handedness=raw_mp_handedness, timestamp_ms=timestamp_ms,
            provenance={
                'device_id': self.device_id, 'camera': self.camera,
                'software': self.software, 'sampling_interval_ms': self.interval_ms,
                'actual_handedness': actual_hand,
            },
        )
        self._progress['count'] += 1
        self._progress['last_frame_index'] = frame_index
        self._progress['last_timestamp_ms'] = timestamp_ms
        return sample_id
