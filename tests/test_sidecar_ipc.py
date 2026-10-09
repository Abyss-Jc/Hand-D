"""HD-09: localhost-only, per-launch authenticated HTTP, WS, MJPEG."""
import asyncio
import secrets
import unittest

from aiohttp import ClientSession, WSMsgType

from handd_core.runtime_v2 import GestureRuntime
from handd_core.sidecar_ipc import SidecarServer


class SidecarTransportTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.token = secrets.token_urlsafe(32)
        self.runtime = GestureRuntime()
        self.server = SidecarServer(runtime=self.runtime, token=self.token)
        self.port = await self.server.start()
        self.client = ClientSession()

    async def asyncTearDown(self):
        await self.client.close()
        await self.server.stop()

    def url(self, suffix='health'):
        return f'http://127.0.0.1:{self.port}/{suffix}'

    async def test_bound_loopback_dynamic_port_and_requires_token(self):
        self.assertGreater(self.port, 0)
        self.assertEqual(self.server.host, '127.0.0.1')
        async with self.client.get(self.url()) as response:
            self.assertEqual(response.status, 401)
        async with self.client.get(self.url(f'health?token={self.token}')) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual((await response.json())['runtime_session_id'],
                             self.runtime.runtime_session_id)
            self.assertEqual(response.headers['Cache-Control'], 'no-store')
            self.assertEqual(response.headers['Referrer-Policy'], 'no-referrer')

    async def test_wrong_token_cannot_open_websocket_or_mjpeg(self):
        async with self.client.get(self.url('mjpeg?token=bad')) as response:
            self.assertEqual(response.status, 401)
        async with self.client.get(self.url('ws')) as response:
            self.assertEqual(response.status, 401)
        with self.assertRaises(Exception):
            await self.client.ws_connect(self.url('ws?token=bad'))

    async def test_tauri_dev_origin_can_use_dynamic_frontend_port(self):
        # Tauri CLI selected 1430 on CachyOS; 1420 cannot be hard-coded.
        for origin in ('http://127.0.0.1:1430', 'http://localhost:1430'):
            ws = await self.client.ws_connect(
                self.url(f'ws?token={self.token}'), origin=origin,
            )
            self.assertEqual((await ws.receive_json(timeout=3))['type'], 'runtime.ready')
            await ws.close()

    async def test_foreign_origin_remains_forbidden_even_with_valid_token(self):
        from aiohttp import WSServerHandshakeError
        for origin in ('https://evil.example', 'http://127.0.0.1.evil.example:1430',
                       'http://localhost:1430.evil.example'):
            with self.assertRaises(WSServerHandshakeError) as error:
                await self.client.ws_connect(
                    self.url(f'ws?token={self.token}'), origin=origin,
                )
            self.assertEqual(error.exception.status, 403)

    async def test_websocket_ready_snapshot_events_and_reconnect(self):
        ws = await self.client.ws_connect(self.url(f'ws?token={self.token}'))
        initial = await ws.receive_json(timeout=3)
        self.assertEqual(initial['type'], 'runtime.ready')
        self.assertEqual(initial['snapshot']['runtime_session_id'], self.runtime.runtime_session_id)
        update = {'type': 'runtime.update', 'runtime_session_id': self.runtime.runtime_session_id,
                  'seq': 1, 'timestamp_ms': 100, 'payload': {'drawing': {'action': None}}}
        await self.server.publish_event(update)
        event = await ws.receive_json(timeout=3)
        self.assertEqual(event, update)
        await ws.close()
        reconnected = await self.client.ws_connect(self.url(f'ws?token={self.token}'))
        again = await reconnected.receive_json(timeout=3)
        self.assertEqual(again['type'], 'runtime.ready')
        await reconnected.close()

    async def test_mjpeg_stream_sends_frame_with_multipart_format(self):
        # Minimal JPEG signature for transport checks; no camera or disk use.
        self.assertEqual(self.server.preview_subscribers, 0)
        await self.server.publish_jpeg(b'\xff\xd8EXAMPLE\xff\xd9')
        async with self.client.get(self.url(f'mjpeg?token={self.token}')) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(self.server.preview_subscribers, 1)
            self.assertIn('multipart/x-mixed-replace', response.headers['Content-Type'])
            chunk = await asyncio.wait_for(response.content.read(300), 3)
            self.assertIn(b'Content-Type: image/jpeg', chunk)
            self.assertIn(b'\xff\xd8EXAMPLE\xff\xd9', chunk)

    async def test_preview_subscribers_return_to_zero_on_stream_close(self):
        response = await self.client.get(self.url(f'mjpeg?token={self.token}'))
        self.assertEqual(self.server.preview_subscribers, 1)
        response.close()
        for _ in range(25):
            if self.server.preview_subscribers == 0:
                break
            await asyncio.sleep(.08)
        self.assertEqual(self.server.preview_subscribers, 0,
                         'stream close must stop useless JPEG encoding work')

    async def test_new_process_has_new_token_and_runtime_session(self):
        other = SidecarServer(runtime=GestureRuntime(), token=secrets.token_urlsafe(32))
        port = await other.start()
        try:
            self.assertNotEqual(other.runtime.runtime_session_id, self.runtime.runtime_session_id)
            async with self.client.get(f'http://127.0.0.1:{port}/health?token={self.token}') as r:
                self.assertEqual(r.status, 401)
        finally:
            await other.stop()


if __name__ == '__main__':
    unittest.main()
