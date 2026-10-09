"""Authenticated localhost IPC for the supervised Python runtime.

One ephemeral capability token per sidecar launch. HTTP health, WebSocket and
MJPEG are restricted to loopback and require the token. URLs do not get logged;
future packaged frontend must use Referrer-Policy:no-referrer, and the launch
bootstrap is delivered via the Rust supervisor, never a public web endpoint.
"""
from __future__ import annotations

import asyncio
import hmac
import json
from secrets import token_urlsafe
from urllib.parse import urlsplit

from aiohttp import WSMsgType, web

from handd_core.runtime_v2 import GestureRuntime


class SidecarServer:
    host = '127.0.0.1'

    def __init__(self, *, runtime: GestureRuntime, token: str | None = None):
        self.runtime = runtime
        self.token = token or token_urlsafe(32)
        if len(self.token) < 24:
            raise ValueError('launch token is too short')
        self._runner: web.AppRunner | None = None
        self._clients: set[web.WebSocketResponse] = set()
        self._frame_condition = asyncio.Condition()
        self._latest_jpeg: bytes | None = None
        self._frame_seq = 0
        self.preview_subscribers = 0
        self._stopping = False
        self.port: int | None = None

    @web.middleware
    async def _authentication(self, request: web.Request, handler):
        supplied = request.query.get('token')
        if supplied is None:
            header = request.headers.get('Authorization', '')
            if header.startswith('Bearer '):
                supplied = header[7:]
        if not isinstance(supplied, str) or not hmac.compare_digest(supplied, self.token):
            raise web.HTTPUnauthorized(text='invalid launch token')
        # Do not expose the capability to arbitrary web pages with explicit
        # cross-origin requests; Tauri app origins are decided by supervisor.
        origin = request.headers.get('Origin')
        if origin is not None and not self._trusted_origin(origin):
            raise web.HTTPForbidden(text='unrecognized Origin')
        return await handler(request)

    @staticmethod
    def _trusted_origin(origin: str) -> bool:
        if origin in ('tauri://localhost', 'http://tauri.localhost',
                      'https://tauri.localhost'):
            return True
        try:
            parsed = urlsplit(origin)
            return (
                parsed.scheme == 'http'
                and parsed.hostname in ('127.0.0.1', 'localhost')
                and parsed.port is not None
                and 1 <= parsed.port <= 65535
                and parsed.username is None
                and parsed.password is None
                and parsed.path == ''
                and not parsed.query
                and not parsed.fragment
            )
        except ValueError:
            return False

    async def _headers(self, request: web.Request, response: web.StreamResponse):
        response.headers['Cache-Control'] = 'no-store'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['X-Content-Type-Options'] = 'nosniff'

    async def start(self) -> int:
        if self._runner is not None:
            raise RuntimeError('sidecar already running')
        self._stopping = False
        app = web.Application(middlewares=[self._authentication])
        app.on_response_prepare.append(self._headers)
        app.router.add_get('/health', self._health)
        app.router.add_get('/ws', self._websocket)
        app.router.add_get('/mjpeg', self._mjpeg)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        try:
            site = web.TCPSite(runner, host=self.host, port=0)
            await site.start()
            # Socket is exclusively bound to loopback at OS level.
            self.port = site._server.sockets[0].getsockname()[1]
        except BaseException:
            await runner.cleanup()
            raise
        self._runner = runner
        return self.port

    async def stop(self):
        self._stopping = True
        async with self._frame_condition:
            self._frame_condition.notify_all()
        clients = list(self._clients)
        for client in clients:
            await client.close()
        self._clients.clear()
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
            self.port = None

    async def _health(self, request):
        return web.json_response({
            'status': 'ready', 'runtime_session_id': self.runtime.runtime_session_id,
            'snapshot': self.runtime.snapshot(),
        })

    async def _websocket(self, request):
        ws = web.WebSocketResponse(heartbeat=10)
        await ws.prepare(request)
        self._clients.add(ws)
        await ws.send_json({'type': 'runtime.ready', 'snapshot': self.runtime.snapshot()})
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    # This endpoint publishes runtime status, not unaudited
                    # arbitrary commands. Ignore all client text messages.
                    continue
        finally:
            self._clients.discard(ws)
        return ws

    async def publish_event(self, event: dict) -> None:
        for client in tuple(self._clients):
            if not client.closed:
                try:
                    await client.send_json(event)
                except (ConnectionError, RuntimeError):
                    self._clients.discard(client)

    async def publish_jpeg(self, jpeg: bytes) -> None:
        if not (jpeg.startswith(b'\xff\xd8') and jpeg.endswith(b'\xff\xd9')):
            raise ValueError('preview frame must be JPEG')
        async with self._frame_condition:
            self._latest_jpeg = bytes(jpeg)
            self._frame_seq += 1
            self._frame_condition.notify_all()

    async def _mjpeg(self, request):
        stream = web.StreamResponse(
            status=200, headers={
                'Content-Type': 'multipart/x-mixed-replace; boundary=frame',
                'X-Accel-Buffering': 'no',
            },
        )
        await stream.prepare(request)
        self.preview_subscribers += 1
        last_seq = -1
        try:
            while not self._stopping:
                if request.transport is None or request.transport.is_closing():
                    break
                async with self._frame_condition:
                    try:
                        await asyncio.wait_for(
                            self._frame_condition.wait_for(
                                lambda: self._stopping or (
                                    self._latest_jpeg is not None
                                    and self._frame_seq != last_seq
                                )
                            ), timeout=.25,
                        )
                    except asyncio.TimeoutError:
                        continue
                    if self._stopping:
                        break
                    jpeg = self._latest_jpeg
                    last_seq = self._frame_seq
                await stream.write(
                    b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                    + str(len(jpeg)).encode() + b'\r\n\r\n' + jpeg + b'\r\n'
                )
        except (asyncio.CancelledError, ConnectionError, RuntimeError):
            pass
        finally:
            self.preview_subscribers -= 1
            try:
                await stream.write_eof()
            except (ConnectionError, RuntimeError):
                pass
        return stream
