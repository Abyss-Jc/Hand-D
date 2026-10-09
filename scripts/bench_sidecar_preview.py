"""Bounded no-recording Linux preview/WS cadence probe for manual HD-09 tuning.

From project root: uv run --frozen python scripts/bench_sidecar_preview.py
Starts a disposable sidecar, decodes no JPEGs and prints no URL/token/frame data.
Requires camera to be free. This is not full sensor-to-photon latency.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time

from aiohttp import ClientSession, WSMsgType

DURATION_SECONDS = 8.0
BOUNDARY = b'--frame\r\n'


async def measure() -> None:
    child = await asyncio.create_subprocess_exec(
        sys.executable, '-u', '-m', 'handd_core.sidecar_main',
        '--legacy-checkpoint', 'models/gesture_mlp.pth',
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        ready_line = await asyncio.wait_for(child.stdout.readline(), 12)
        if not ready_line:
            raise RuntimeError('sidecar returned no readiness handshake')
        ready = json.loads(ready_line)
        assert ready['type'] == 'sidecar.ready'
        url = f"http://127.0.0.1:{ready['port']}"
        token = ready['token']
        counts = {'preview_frames': 0, 'ws_updates': 0, 'ws_hand_updates': 0,
                  'preview_bytes': 0}
        async with ClientSession() as client:
            ws = await client.ws_connect(f'{url}/ws?token={token}')
            assert (await ws.receive_json())['type'] == 'runtime.ready'
            response = await client.get(f'{url}/mjpeg?token={token}')
            assert response.status == 200
            beginning = time.monotonic()
            deadline = beginning + DURATION_SECONDS

            async def preview():
                tail = b''
                while time.monotonic() < deadline:
                    try:
                        data = await asyncio.wait_for(response.content.readany(), .6)
                    except asyncio.TimeoutError:
                        continue
                    if not data:
                        break
                    counts['preview_bytes'] += len(data)
                    packet = tail + data
                    counts['preview_frames'] += packet.count(BOUNDARY)
                    tail = packet[-(len(BOUNDARY)-1):]

            async def updates():
                while time.monotonic() < deadline:
                    try:
                        message = await ws.receive(timeout=.6)
                    except asyncio.TimeoutError:
                        continue
                    if message.type != WSMsgType.TEXT:
                        break
                    event = json.loads(message.data)
                    if event.get('type') == 'runtime.update':
                        counts['ws_updates'] += 1
                        if any(event['payload'].get(role, {}).get('pointer') is not None
                               for role in ('drawing', 'modifier')):
                            counts['ws_hand_updates'] += 1

            await asyncio.gather(preview(), updates())
            elapsed = time.monotonic() - beginning
            response.close()
            await ws.close()
            print(json.dumps({
                'probe': 'live sidecar MJPEG+WS, no recording',
                'elapsed_s': round(elapsed, 2),
                'preview_fps_delivered': round(counts['preview_frames']/elapsed, 2),
                'ws_updates_per_s': round(counts['ws_updates']/elapsed, 2),
                **counts,
                'scope': 'delivery to local Python HTTP client, not WebKit render',
            }, sort_keys=True))
    finally:
        if child.returncode is None:
            child.terminate()
        try:
            await asyncio.wait_for(child.wait(), 8)
        except asyncio.TimeoutError:
            child.kill()
            await child.wait()


if __name__ == '__main__':
    asyncio.run(measure())
