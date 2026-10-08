"""MediaPipe LIVE_STREAM callback bridge for the canonical Capture sampler.

MediaPipe invokes callbacks on another thread. Only `drain_to` (called from
the DB owner thread) writes SQLite; dropped detections do not build a backlog.
"""

from __future__ import annotations

from collections import OrderedDict
from queue import Empty, Full, Queue
from threading import Lock

import numpy as np

from handd_core.capture_sampling import CaptureSampler


class LatestCaptureResults:
    def __init__(self, max_inflight: int = 64):
        self._frames: OrderedDict[int, int] = OrderedDict()
        self._lock = Lock()
        self._latest: Queue[tuple[int, object, int]] = Queue(maxsize=1)
        self._max_inflight = max_inflight

    def register_frame(self, frame_index: int, timestamp_ms: int) -> None:
        with self._lock:
            self._frames[timestamp_ms] = frame_index
            while len(self._frames) > self._max_inflight:
                self._frames.popitem(last=False)

    def on_result(self, result: object, output_image: object, timestamp_ms: int) -> None:
        """Thread-safe MediaPipe result_callback; never writes SQLite."""
        with self._lock:
            frame_index = self._frames.pop(timestamp_ms, None)
        if frame_index is None:
            return
        try:
            self._latest.put_nowait((frame_index, result, timestamp_ms))
        except Full:
            try:
                self._latest.get_nowait()
            except Empty:
                pass
            self._latest.put_nowait((frame_index, result, timestamp_ms))

    def drain_to(self, sampler: CaptureSampler) -> str | None:
        """Call on the SQLite owner thread; may persist one configured hand."""
        try:
            frame_index, result, timestamp_ms = self._latest.get_nowait()
        except Empty:
            return None
        if not result.hand_landmarks:
            return None
        for raw_category, image_points, world_points in zip(
            result.handedness, result.hand_landmarks, result.hand_world_landmarks
        ):
            if not raw_category:
                continue
            to_array = lambda lms: np.array(
                [[point.x, point.y, point.z] for point in lms], dtype=np.float64
            )
            sample_id = sampler.offer(
                frame_index=frame_index, timestamp_ms=timestamp_ms,
                raw_mp_handedness=raw_category[0].category_name,
                image_landmarks=to_array(image_points),
                world_landmarks=to_array(world_points),
            )
            if sample_id is not None:
                return sample_id
        return None
