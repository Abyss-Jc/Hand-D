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
import sqlite3
import sys

from handd_core.runtime_v2 import GestureRuntime, LatestRuntimeResults
from handd_core.sidecar_ipc import SidecarServer
from handd_core.preview_pacing import PreviewPacer
from handd_core.studio_models import StudioModels
from handd_core.camera_device import open_camera


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Hand-D v2 private Python sidecar')
    p.add_argument('--no-camera', action='store_true', help='IPC-only diagnostics')
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--workspace', type=Path, default=None,
                   help='Explicit existing project directory; never creates a dataset implicitly')
    p.add_argument('--task', type=Path, default=Path(__file__).resolve().parents[1]
                   / 'models' / 'hand_landmarker.task')
    model = p.add_mutually_exclusive_group()
    model.add_argument('--model-artifact', type=Path, default=None)
    model.add_argument('--legacy-checkpoint', type=Path, default=None,
                       help='Development-only old .pth (label order unverified)')
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
    camera = open_camera(cv2, camera_index)
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
    preview_pacer = PreviewPacer(target_fps=30)
    timestamp = -1
    loop = asyncio.get_running_loop()
    started = loop.time()
    try:
        with vision.HandLandmarker.create_from_options(options) as detector:
            while not shutdown.is_set():
                ok, frame = await asyncio.to_thread(camera.read)
                now = max(timestamp + 1, int((loop.time() - started) * 1000))
                if not ok:
                    server.camera_available = False
                    released = server.runtime.expire_if_stale(now)
                    if released:
                        await server.publish_event(released)
                    await asyncio.sleep(.1)
                    continue
                timestamp = now
                server.camera_available = True
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detector.detect_async(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), now,
                )
                collected = []
                capture_error = []
                def collect_from_same_callback(result, at_ms):
                    if server.collection is not None:
                        try:
                            sample = server.collection.offer_result(result, at_ms)
                        except (sqlite3.Error, OSError, RuntimeError):
                            capture_error.append(server.collection.fail())
                        else:
                            if sample is not None:
                                collected.append(sample)
                event = mailbox.drain_to(
                    server.runtime, on_observation=collect_from_same_callback)
                if event is not None:
                    await server.publish_event(event)
                if collected:
                    # No raw landmarks or frames in UI progress messages.
                    await server.publish_event({
                        'type':'studio.collection',
                        'data':server.collection.status(),
                    })
                if capture_error:
                    await server.publish_event({
                        'type':'studio.collection', 'data':capture_error[0],
                    })
                else:
                    expired = server.runtime.expire_if_stale(now)
                    if expired is not None:
                        await server.publish_event(expired)
                # Cap preview FPS separately from inference; original BGR frame
                # remains process-local and is never persisted.
                if preview_pacer.due(loop.time(), subscribers=server.preview_subscribers):
                    worked, data = await asyncio.to_thread(
                        cv2.imencode, '.jpg', frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 68],
                    )
                    if worked:
                        await server.publish_jpeg(data.tobytes())
                await asyncio.sleep(0)
    finally:
        server.camera_available = False
        camera.release()


async def run(args) -> None:
    runtime = GestureRuntime()
    if args.model_artifact is not None:
        runtime.activate_model(args.model_artifact)
    elif args.legacy_checkpoint is not None:
        runtime.activate_legacy_checkpoint(args.legacy_checkpoint)
    if args.workspace is not None:
        # A previously verified explicit workspace selection takes priority
        # over the development-only legacy checkpoint on every restart.
        models = StudioModels(args.workspace)
        try:
            if models.active_id() is not None:
                models.restore(runtime)
        except (ValueError, OSError) as exc:
            # Also catch corrupt persisted selection JSON: the Whiteboard
            # and camera must remain available in tracking-only mode.
            # Never silently use legacy predictions after an invalid v2 ID.
            runtime.mark_model_load_error()
            print(f'Active Model could not be restored: {exc}', file=sys.stderr)
    server = SidecarServer(runtime=runtime, workspace=args.workspace)
    port = await server.start()
    print(json.dumps({
        'type': 'sidecar.ready', 'host': server.host, 'port': port,
        'token': server.token, 'runtime_session_id': runtime.runtime_session_id,
        'workspace': str(server.studio.path) if server.studio else None,
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
