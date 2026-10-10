"""Bounded LIVE_STREAM camera + verified v2 Candidate diagnostic (no recording).

This is the deployment-runtime pipeline spike, not the Tauri/HTTP/WS/MJPEG
sidecar. Neither frames nor raw landmarks are written to disk.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from handd_core.model_artifact import ModelArtifactError
from handd_core.camera_device import open_camera
from handd_core.runtime_v2 import GestureRuntime, LatestRuntimeResults


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Hand-D v2 LIVE_STREAM + Model Artifact smoke')
    p.add_argument('--model-artifact', type=Path, required=True,
                   help='Versioned Candidate directory containing verified manifest and weights')
    p.add_argument('--task', type=Path, default=Path(__file__).resolve().parents[1]
                   / 'models' / 'hand_landmarker.task')
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--seconds', type=float, default=8.0)
    p.add_argument('--preview', action='store_true', help='Show live diagnostics; Esc/Q exits')
    return p


def main(argv: list[str] | None = None) -> int:
    p = parser()
    args = p.parse_args(argv)
    if not .5 <= args.seconds <= 30:
        p.error('--seconds must be 0.5 to 30')
    if not args.task.is_file():
        p.error(f'MediaPipe task missing: {args.task}')

    runtime = GestureRuntime()
    try:
        runtime.activate_model(args.model_artifact)
    except ModelArtifactError as exc:
        p.error(f'Candidate compatibility/integrity error: {exc}')

    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    camera = open_camera(cv2, args.camera)
    if not camera.isOpened():
        camera.release()
        print('CAMERA_ERROR: unable to open', args.camera)
        return 2

    bridge = LatestRuntimeResults()
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(args.task)),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=.7,
        min_hand_presence_confidence=.7,
        min_tracking_confidence=.7,
        result_callback=bridge.on_result,
    )
    start = time.monotonic()
    last_ts = -1
    frames = 0
    attempts = 0
    events = 0
    track_updates = 0
    actions = Counter()
    predictions = Counter()
    latest_state = runtime.snapshot()
    window = 'Hand-D v2 | verified Model Artifact (no recording)'
    try:
        with vision.HandLandmarker.create_from_options(options) as detector:
            while time.monotonic() - start < args.seconds:
                ok, frame = camera.read()
                attempts += 1
                now = max(last_ts + 1, int((time.monotonic() - start) * 1000))
                if ok:
                    frames += 1
                    last_ts = now
                    mirrored = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)
                    detector.detect_async(
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), now
                    )
                received = bridge.drain_to(runtime)
                if received is None:
                    received = runtime.expire_if_stale(now)
                if received is not None:
                    latest_state = received
                    events += 1
                    for role in ('drawing', 'modifier'):
                        state = received['payload'][role]
                        if state['pointer'] is not None:
                            track_updates += 1
                        if state['raw_gesture']:
                            predictions[state['raw_gesture']] += 1
                        if state['action']:
                            actions[f'{role}:{state["action"]}'] += 1
                if args.preview and ok:
                    payload = latest_state.get('payload', {})
                    for i, role in enumerate(('drawing', 'modifier')):
                        hand = payload.get(role, {})
                        label = (f'{role.upper()}: {hand.get("stable_gesture") or "-"} '
                                 f'[{hand.get("action") or "no action"}]')
                        cv2.putText(mirrored, label, (16, 40 + i * 35),
                                    cv2.FONT_HERSHEY_SIMPLEX, .60, (255, 255, 255), 2)
                    cv2.putText(mirrored, 'Model v2 | Q/Esc sale | sin grabar',
                                (16, mirrored.shape[0] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, .53, (90, 250, 90), 2)
                    cv2.imshow(window, mirrored)
                    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                        break
                if not ok:
                    time.sleep(.015)
            tail = bridge.drain_to(runtime)
            if tail:
                latest_state = tail
                events += 1
    finally:
        camera.release()
        if args.preview:
            cv2.destroyAllWindows()

    duration = max(.001, time.monotonic() - start)
    summary = {
        'runtime_session_id': runtime.runtime_session_id,
        'active_model_id': runtime.snapshot()['active_model_id'],
        'read_frames': frames, 'attempted_frames': attempts,
        'capture_fps': round(frames / duration, 2),
        'mediapipe': bridge.statistics(),
        'runtime_events': events,
        'tracking_hand_updates': track_updates,
        'predicted_gestures': dict(predictions),
        'stable_role_actions': dict(actions),
        'health': latest_state.get('payload', {}).get('health', runtime.snapshot()['health']),
        'no_frames_video_or_samples_saved': True,
        'latency_scope': 'post-landmarker Python processing only; no Tauri/IPC/front-end',
    }
    print('HD08_RUNTIME_SMOKE=' + json.dumps(summary, sort_keys=True), flush=True)
    return 0 if frames > 0 and bridge.statistics()['received_callbacks'] > 0 else 3


if __name__ == '__main__':
    raise SystemExit(main())
