"""Manual LIVE_STREAM camera collector for the v2 canonical SQLite dataset.

Requires explicit local execution by the operator. No camera/video frames are
stored: only normalized image + world hand landmarks, labels and provenance.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

from handd_core.capture_sampling import CaptureSampler
from handd_core.dataset_store import DatasetStore
from handd_core.live_collection import LatestCaptureResults


def _local_device_config() -> Path:
    if sys.platform == "win32":
        root = Path(os.environ.get("APPDATA", str(Path.home())))
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    return root / "hand-d" / "anonymous-device-id.txt"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Collect Hand-D v2 raw landmarks into a workspace SQLite DB"
    )
    p.add_argument("--workspace", type=Path, required=True, help="Project directory")
    p.add_argument("--participant", required=True, help="Anonymous participant label")
    p.add_argument("--gesture", required=True, help="Intended gesture label")
    p.add_argument("--hand", choices=("Right", "Left", "Any"), default="Right")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--quota", type=int, default=120)
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--max-seconds", type=float, default=None,
                   help="Optional bounded collection window for a safe camera smoke test")
    p.add_argument("--model", type=Path, default=Path(__file__).resolve().parents[1]
                   / "models" / "hand_landmarker.task")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.quota <= 0 or args.interval_ms <= 0:
        parser().error("quota and interval-ms must be positive")
    if args.max_seconds is not None and args.max_seconds <= 0:
        parser().error("max-seconds must be positive")
    if not args.model.is_file():
        parser().error(f"MediaPipe Hand Landmarker task not found: {args.model}")

    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        cap.release()
        raise SystemExit(f"Cannot open camera index {args.camera}")

    bridge = LatestCaptureResults()
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(args.model)),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
        result_callback=bridge.on_result,
    )
    store = None
    try:
        with vision.HandLandmarker.create_from_options(options) as detector:
            store = DatasetStore(args.workspace / "handd.sqlite")
            # Reuse participant identity; every CLI invocation is a new Session.
            if store.conn.execute(
                "SELECT 1 FROM participants WHERE participant_id=?", (args.participant,)
            ).fetchone() is None:
                store.create_participant(args.participant)
            session_id = uuid4().hex
            capture_id = uuid4().hex
            store.create_session(session_id, args.participant)
            store.create_capture(capture_id, session_id, args.gesture, args.hand, args.quota)
            sampler = CaptureSampler.for_local_device(
                store, capture_id, interval_ms=args.interval_ms,
                camera=str(args.camera), device_config_path=_local_device_config(),
            )
            print(f"Session: {session_id} | Capture: {capture_id}")
            print(f"Participant: {args.participant} | Gesture: {args.gesture} | Hand: {args.hand}")
            print("Controls: q=finish early, p=pause/resume, Ctrl+C=finish early")
            frame_index = 0
            last_timestamp_ms = -1
            start_time = time.monotonic()
            while not sampler.finished:
                if args.max_seconds is not None and time.monotonic() - start_time >= args.max_seconds:
                    print("Timed collection window reached; existing Samples remain durable.")
                    break
                ret, frame = cap.read()
                if not ret:
                    print("Camera frame not available; ending this Capture.")
                    break
                frame_index += 1
                # Legacy-compatible selfie mirror; raw handedness is tracked
                # separately from the selected real-world Capture hand.
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp_ms = max(time.time_ns() // 1_000_000, last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms
                bridge.register_frame(frame_index, timestamp_ms)
                detector.detect_async(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms
                )
                sample_id = bridge.drain_to(sampler)
                if sample_id is not None:
                    print(f"Collected: {sampler.count}/{args.quota} ({sample_id[:8]})")
                label = f"Hand-D Collect [{args.gesture}] {sampler.count}/{args.quota}"
                if sampler.paused:
                    label += " [PAUSED]"
                cv2.putText(frame, label, (16, 34), cv2.FONT_HERSHEY_SIMPLEX,
                            0.72, (255, 255, 255), 2)
                cv2.imshow("Hand-D v2 Collector", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("p"):
                    sampler.resume() if sampler.paused else sampler.pause()
            print(f"Capture stored: {sampler.count}/{args.quota} unreviewed Samples")
            print(f"LIVE_CALLBACK_DIAGNOSTICS {bridge.statistics()}")
            return 0
    except KeyboardInterrupt:
        print("Capture interrupted; all stored Samples remain durable and unreviewed.")
        return 130
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if store is not None:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())
