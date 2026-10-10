"""Guided physical-hand versus MediaPipe raw-handedness calibration.

Frame data are transient: no photos, video, landmarks, SQLite rows or private
identifiers are persisted. Results are meaningful ONLY if the operator follows
the on-screen physical-hand prompts (right hand, then left hand).
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, deque
import json
from pathlib import Path
from threading import Lock
import time

from handd_core.camera_device import open_camera


class HandMappingCalibration:
    """Aggregate single-hand observations for independently prompted phases."""

    def __init__(self, *, min_observations: int = 8, min_agreement: float = .8):
        if min_observations <= 0 or not 0.5 < min_agreement <= 1.0:
            raise ValueError("invalid calibration requirements")
        self.min_observations = min_observations
        self.min_agreement = min_agreement
        self.counts = {"Right": Counter(), "Left": Counter()}
        self.ignored = Counter()

    def record(self, physical_hand: str, detected_raw_labels: list[str]) -> None:
        if physical_hand not in self.counts:
            raise ValueError("phase must name the physical Right or Left hand")
        if len(detected_raw_labels) != 1:
            self.ignored[physical_hand] += 1
            return
        label = detected_raw_labels[0]
        if label not in ("Left", "Right"):
            self.ignored[physical_hand] += 1
            return
        self.counts[physical_hand][label] += 1

    def assess(self) -> dict:
        majority = {}
        fractions = {}
        for physical in ("Right", "Left"):
            count = self.counts[physical]
            total = sum(count.values())
            dominant, hits = count.most_common(1)[0] if count else (None, 0)
            fractions[physical] = round(hits / total, 3) if total else 0.
            majority[physical] = (dominant if total >= self.min_observations
                                  and hits / total >= self.min_agreement else None)
        mapping = "inconclusive"
        if majority == {"Right": "Left", "Left": "Right"}:
            mapping = "inverted"
        elif majority == {"Right": "Right", "Left": "Left"}:
            mapping = "direct"
        return {
            "mapping": mapping,
            "counts": {hand: dict(self.counts[hand]) for hand in ("Right", "Left")},
            "majority": majority,
            "agreement": fractions,
            "ignored": dict(self.ignored),
            "evidence_is_conditional_on_user_following_hand_prompts": True,
        }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Test physical right/left against raw MediaPipe labels, no recording"
    )
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--hold-seconds", type=float, default=7.0)
    p.add_argument("--task", type=Path, default=Path(__file__).resolve().parents[1]
                   / "models" / "hand_landmarker.task")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not 3 <= args.hold_seconds <= 20:
        parser().error("--hold-seconds must be between 3 and 20")
    if not args.task.is_file():
        parser().error(f"Hand Landmarker task unavailable: {args.task}")
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    camera = open_camera(cv2, args.camera)
    if not camera.isOpened():
        camera.release()
        print("CAMERA_ERROR could not open device", args.camera, flush=True)
        return 2

    # Callback only moves labels to the GUI thread. Do not persist frame data.
    latest = deque(maxlen=1)
    pending_phases: OrderedDict[int, str | None] = OrderedDict()
    lock = Lock()
    callback_count = 0

    def on_result(result, output_image, timestamp_ms):
        nonlocal callback_count
        labels = [group[0].category_name for group in result.handedness if group]
        with lock:
            callback_count += 1
            phase = pending_phases.pop(timestamp_ms, None)
            latest.append((phase, labels))

    phases = [
        ("prepare-right", 5.0, "PREPARATE: MANO DERECHA", None),
        ("right", args.hold_seconds, "MUESTRA SOLO TU MANO DERECHA", "Right"),
        ("prepare-left", 4.0, "CAMBIA A TU MANO IZQUIERDA", None),
        ("left", args.hold_seconds, "MUESTRA SOLO TU MANO IZQUIERDA", "Left"),
        ("summary", 4.0, "RESULTADO / SE CIERRA SOLO", None),
    ]
    calibration = HandMappingCalibration()
    window_title = "Hand-D | Calibracion de mano derecha / izquierda"
    success_reads = 0
    previous_ts = -1
    stopped_early = False

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(args.task)),
        running_mode=vision.RunningMode.LIVE_STREAM, num_hands=2,
        min_hand_detection_confidence=.7, min_hand_presence_confidence=.7,
        min_tracking_confidence=.7, result_callback=on_result,
    )
    try:
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, 980, 670)
        with vision.HandLandmarker.create_from_options(options) as detector:
            beginning = time.monotonic()
            total_duration = sum(d for _, d, _, _ in phases)
            while time.monotonic() - beginning < total_duration:
                elapsed = time.monotonic() - beginning
                remaining = elapsed
                current = phases[-1]
                for phase in phases:
                    if remaining < phase[1]:
                        current = phase
                        break
                    remaining -= phase[1]
                stage, duration, instruction, physical = current
                ok, frame = camera.read()
                if not ok:
                    time.sleep(.025)
                    continue
                success_reads += 1
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp_ms = max(previous_ts + 1, int((time.monotonic() - beginning) * 1000))
                previous_ts = timestamp_ms
                with lock:
                    pending_phases[timestamp_ms] = physical
                    while len(pending_phases) > 64:
                        pending_phases.popitem(last=False)
                detector.detect_async(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
                with lock:
                    item = latest.pop() if latest else None
                    latest.clear()
                if item is not None and item[0] is not None:
                    calibration.record(item[0], item[1])

                # Big visual prompts visible from a normal laptop-camera distance.
                height, width = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (width, 125), (20, 20, 20), -1)
                cv2.putText(frame, instruction, (15, 43),
                            cv2.FONT_HERSHEY_SIMPLEX, .73, (95, 255, 120), 2)
                seconds_left = max(0, int(duration - remaining + .99))
                cv2.putText(frame, f"{seconds_left}s | Solo UNA mano en camara",
                            (15, 80), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
                cv2.rectangle(frame, (0, height - 115), (width, height), (20, 20, 20), -1)
                right = calibration.counts["Right"]
                left = calibration.counts["Left"]
                cv2.putText(frame, f"DERECHA: MP Left={right['Left']} | MP Right={right['Right']}",
                            (12, height - 78), cv2.FONT_HERSHEY_SIMPLEX, .59, (255, 255, 255), 2)
                cv2.putText(frame, f"IZQUIERDA: MP Left={left['Left']} | MP Right={left['Right']}",
                            (12, height - 45), cv2.FONT_HERSHEY_SIMPLEX, .59, (255, 255, 255), 2)
                cv2.putText(frame, "Sin grabacion. ESC o Q para cerrar.",
                            (12, height - 13), cv2.FONT_HERSHEY_SIMPLEX, .49, (95, 255, 120), 1)
                cv2.imshow(window_title, frame)
                if cv2.waitKey(1) & 0xff in (27, ord("q")):
                    stopped_early = True
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()
    result = calibration.assess()
    if stopped_early:
        result["interrupted_before_completion"] = True
        result["mapping"] = "inconclusive"
    result["frames_read"] = success_reads
    result["mediapipe_callbacks"] = callback_count
    result["no_recording"] = True
    print("HAND_MAPPING_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
