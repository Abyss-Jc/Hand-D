"""Monotonic demand-aware MJPEG pacing, independent from MediaPipe cadence."""


class PreviewPacer:
    def __init__(self, *, target_fps: float = 30):
        if not 0 < target_fps <= 120:
            raise ValueError('target_fps must be between 0 and 120')
        self.interval = 1.0 / target_fps
        self._next_due = None

    def due(self, timestamp_s: float, *, subscribers: int) -> bool:
        """Return True when a preview JPEG is useful, never queue missed frames."""
        if subscribers <= 0:
            self._next_due = None
            return False
        if self._next_due is None:
            self._next_due = timestamp_s + self.interval
            return True
        if timestamp_s < self._next_due:
            return False
        # Do not accumulate work after a long pause or slow image encoding.
        if timestamp_s - self._next_due >= self.interval:
            self._next_due = timestamp_s + self.interval
        else:
            self._next_due += self.interval
        return True
