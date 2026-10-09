"""Supervised Hand-D Python sidecar: transient camera, HTTP+WS+MJPEG loopback.

The process prints exactly one structured sidecar.ready line on stdout for
Rust to parse. The random token is never passed as a command-line argument.
No frames, video, or raw camera samples are written to disk.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import signal
import sys

from handd_core.runtime_v2 import GestureRuntime, LatestRuntimeResults
from handd_core.sidecar_ipc import SidecarServer


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Hand-D v2 private Python sidecar')
    p.add_argument('--no-camera', action='store_true', help='IPC-only diagnostics')
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--task', type=Path, default=Path(__file__).resolve().parents[1]
                   / 'models' / 'hand_landmarker.task')
    p.add_argument('--model-artifact', type=Path, default=None)
    return p


async def _camera_loop(server: SidecarServer, camera_index: int, task: Path,
                       shutdown: asyncio.Event) -> None:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    if not task.is_file():
        print(f'Hand Landmarker asset missing: {task}', file=sys.stderr, flush=True)
        return
    camera = cv2.VideoCapture(camera_index, cv2.CAP_V4L2) if sys.platform.startswith('linux') \
        else cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        print(f'Camera {camera_index} unavailable; Whiteboard stays accessible',
              file=sys.stderr, flush=True)
        return
    mailbox = LatestRuntimeResults()
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(task)),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=.7,
        min_hand_presence_confidence=.7,
        min_tracking_confidence=.7,
        result_callback=mailbox.on_result,
    )
    count = 0
    timestamp = -1
    loop = asyncio.get_running_loop()
    started = loop.time()
    try:
        with vision.HandLandmarker.create_from_options(options) as detector:
            while not shutdown.is_set():
                ok, frame = await asyncio.to_thread(camera.read)
                now = max(timestamp + 1, int((loop.time() - started) * 1000))
                if not ok:
                    released = server.runtime.expire_if_stale(now)
                    if released:
                        await server.publish_event(released)
                    await asyncio.sleep(.1)
                    continue
                timestamp = now
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detector.detect_async(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), now,
                )
                event = mailbox.drain_to(server.runtime)
                if event is not None:
                    await server.publish_event(event)
                else:
                    expired = server.runtime.expire_if_stale(now)
                    if expired is not None:
                        await server.publish_event(expired)
                # Cap preview FPS separately from inference; original BGR frame
                # remains process-local and is never persisted.
                count += 1
                if count % 2 == 0:
                    worked, data = await asyncio.to_thread(
                        cv2.imencode, '.jpg', frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 68],
                    )
                    if worked:
                        await server.publish_jpeg(data.tobytes())
                await asyncio.sleep(0)
    finally:
        camera.release()


async def run(args) -> None:
    runtime = GestureRuntime()
    if args.model_artifact is not None:
        runtime.activate_model(args.model_artifact)
    server = SidecarServer(runtime=runtime)
    port = await server.start()
    print(json.dumps({
        'type': 'sidecar.ready', 'host': server.host, 'port': port,
        'token': server.token, 'runtime_session_id': runtime.runtime_session_id,
    }), flush=True)
    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, shutdown.set)
        except NotImplementedError:
            pass
    camera_task = None
    if not args.no_camera:
        camera_task = asyncio.create_task(
            _camera_loop(server, args.camera, args.task, shutdown),
        )
    try:
        await shutdown.wait()
    finally:
        if camera_task:
            camera_task.cancel()
            try:
                await camera_task
            except asyncio.CancelledError:
                pass
        await server.stop()


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    asyncio.run(run(args))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
