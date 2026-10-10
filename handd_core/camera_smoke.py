"""Bounded, no-recording camera + legacy checkpoint smoke test.

This runs real MediaPipe LIVE_STREAM but DOES NOT write frames, video, Samples,
or Workspace data. An old .pth without a verified label manifest is diagnostic
only; printed predictions are NOT accuracy measurements.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from pathlib import Path
from threading import Lock
import time

import numpy as np

from handd_core.feature_transform import canonicalize_world_landmarks
from handd_core.camera_device import open_camera

ROOT = Path(__file__).resolve().parents[1]


def load_legacy_checkpoint(path: Path):
    """Load the real 5-output legacy MLP, never silently fall back to a stub."""
    from visualizer_app.gesture_engine import _TorchModel
    return _TorchModel(Path(path))


def classify_hand(model, world_landmarks: np.ndarray, raw_hand: str) -> str | None:
    features = canonicalize_world_landmarks(world_landmarks, raw_hand)
    return None if features is None else model.predict(features)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Bounded webcam + MediaPipe LIVE_STREAM + legacy MLP probe, no recording"
    )
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--seconds", type=float, default=8)
    p.add_argument("--preview", action="store_true",
                   help="Show live diagnostics; press q or Escape to close (no recording)")
    p.add_argument("--task", type=Path, default=ROOT / "models/hand_landmarker.task")
    p.add_argument("--model", type=Path, default=ROOT / "models/gesture_mlp.pth")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not 0.5 <= args.seconds <= 30:
        parser().error("--seconds must be 0.5 to 30")
    if not args.model.is_file() or not args.task.is_file():
        parser().error("Model checkpoint and HandLandmarker task must both exist")

    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model = load_legacy_checkpoint(args.model)
    cap = open_camera(cv2, args.camera)
    if not cap.isOpened():
        cap.release()
        print(f"CAMERA_ERROR could not open camera index {args.camera}")
        return 2

    buffered = deque(maxlen=1)
    lock = Lock()
    callbacks = 0
    dropped_callback_batches = 0
    def on_result(result, output_image, timestamp_ms):
        nonlocal callbacks, dropped_callback_batches
        with lock:
            callbacks += 1
            if buffered:
                dropped_callback_batches += 1
            buffered.append((result, timestamp_ms))

    def drain():
        with lock:
            if not buffered:
                return None
            return buffered.popleft()

    read_ok = 0
    attempted = 0
    processed_callbacks = 0
    hands_seen = 0
    invalid_transforms = 0
    predictions = Counter()
    raw_hands = Counter()
    errors = 0
    dimensions = None
    preview_lines = ['Sin mano detectada']
    preview_title = 'Hand-D / prueba de webcam + modelo legacy'
    start_ns = time.monotonic_ns()
    previous_timestamp = -1

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(args.task)),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
        result_callback=on_result,
    )

    def process_callback(item):
        nonlocal processed_callbacks, hands_seen, invalid_transforms, errors, preview_lines
        if item is None:
            return
        result, _ = item
        processed_callbacks += 1
        preview_lines = []
        if result.hand_world_landmarks:
            hands_seen += 1
        else:
            preview_lines = ['Sin mano detectada']
        for handedness, points in zip(result.handedness, result.hand_world_landmarks):
            if not handedness:
                continue
            raw_hand = handedness[0].category_name
            raw_hands[raw_hand] += 1
            landmarks = np.asarray([[p.x, p.y, p.z] for p in points], dtype=np.float64)
            try:
                prediction = classify_hand(model, landmarks, raw_hand)
            except (ValueError, RuntimeError) as exc:
                errors += 1
                print("INFERENCE_ERROR", str(exc))
                continue
            if prediction is None:
                invalid_transforms += 1
            else:
                predictions[prediction] += 1
                # This inversion is legacy behavior, NOT physically verified yet.
                assumed_hand = 'Right' if raw_hand == 'Left' else 'Left'
                preview_lines.append(
                    f'{prediction} | MP={raw_hand} | mano estimada={assumed_hand}'
                )

    try:
        with vision.HandLandmarker.create_from_options(options) as detector:
            while (time.monotonic_ns() - start_ns) / 1e9 < args.seconds:
                worked, frame = cap.read()
                attempted += 1
                if not worked:
                    time.sleep(0.015)
                    continue
                read_ok += 1
                dimensions = (int(frame.shape[1]), int(frame.shape[0]))
                mirror = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(mirror, cv2.COLOR_BGR2RGB)
                elapsed_ms = (time.monotonic_ns() - start_ns) // 1_000_000
                ts_ms = max(previous_timestamp + 1, elapsed_ms)
                previous_timestamp = ts_ms
                detector.detect_async(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)
                process_callback(drain())
                if args.preview:
                    cv2.putText(mirror, 'MODELO ANTIGUO / sin grabacion', (16, 32),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 255, 80), 2)
                    for line_number, line in enumerate(preview_lines[:3]):
                        cv2.putText(mirror, line, (16, 65 + 30 * line_number),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 2)
                    cv2.putText(mirror, 'Q o ESC para cerrar | MP=etiqueta sin verificar',
                                (16, mirror.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX,
                                0.48, (245, 245, 245), 1)
                    cv2.imshow(preview_title, mirror)
                    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                        break
            # A bounded flush of already-delivered callbacks before disposal.
            process_callback(drain())
    finally:
        cap.release()
        if args.preview:
            cv2.destroyAllWindows()

    seconds = (time.monotonic_ns() - start_ns) / 1e9
    print("CAMERA_OPEN=True")
    print("READ_OK=", read_ok, "ATTEMPTED=", attempted, "SIZE=", dimensions)
    print("ELAPSED_SECONDS=", round(seconds, 2), "CAPTURE_FPS=", round(read_ok/seconds, 2))
    print("MEDIAPIPE_CALLBACKS=", callbacks, "PROCESSED=", processed_callbacks,
          "DROPPED_OLD_BATCHES=", dropped_callback_batches)
    print("HAND_CALLBACK_BATCHES=", hands_seen, "RAW_HANDEDNESS=", dict(raw_hands))
    print("LEGACY_MODEL_PREDICTIONS=", dict(predictions), "INVALID_FEATURES=", invalid_transforms,
          "ERRORS=", errors)
    print("NO_VIDEO_OR_SAMPLES_SAVED=True")
    return 0 if read_ok and callbacks and not errors else 3


if __name__ == "__main__":
    raise SystemExit(main())
