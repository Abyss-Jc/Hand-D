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
import sqlite3
from secrets import token_urlsafe
from urllib.parse import urlsplit

from aiohttp import WSMsgType, web

from handd_core.runtime_v2 import GestureRuntime
from handd_core.studio_workspace import StudioWorkspace
from handd_core.studio_collection import StudioCollection
from handd_core.studio_models import StudioModels


class SidecarServer:
    host = '127.0.0.1'

    def __init__(self, *, runtime: GestureRuntime, token: str | None = None,
                 workspace=None):
        self.runtime = runtime
        self.token = token or token_urlsafe(32)
        self.studio = StudioWorkspace(workspace) if workspace is not None else None
        self.models = StudioModels(self.studio.path) if self.studio else None
        self.collection = StudioCollection(self.studio.path) if self.studio else None
        self.camera_available = False
        self._studio_lock = asyncio.Lock()
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
        if self.collection is not None:
            self.collection.close()
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
                    if len(msg.data) > 4096:
                        continue
                    try:
                        command = json.loads(msg.data)
                    except (ValueError, TypeError):
                        continue
                    if isinstance(command, dict) and command.get('type') == 'studio.request':
                        response = await self._studio_request(command)
                        await ws.send_json(response)
                        if command.get('action') == 'model_activate' and response.get('ok'):
                            await self.publish_event({
                                'type': 'studio.model',
                                'snapshot': self.runtime.snapshot(),
                                'data': response['data'],
                            })
        finally:
            self._clients.discard(ws)
        return ws

    async def _studio_request(self, command: dict) -> dict:
        request_id = command.get('request_id')
        if not isinstance(request_id, str) or len(request_id) > 64:
            request_id = ''
        result = {'type': 'studio.response', 'request_id': request_id}
        if self.studio is None:
            return {**result, 'ok': False,
                    'error': 'Select an existing Hand-D workspace first'}
        action = command.get('action')
        try:
            async with self._studio_lock:
                if action == 'overview':
                    data = await asyncio.to_thread(
                        self.studio.overview, offset=command.get('offset', 0))
                elif action in ('accept','reject','drop','restore'):
                    data = await asyncio.to_thread(
                        self.studio.transition, command.get('sample_id'), action)
                elif action == 'sample_detail':
                    data = await asyncio.to_thread(
                        self.studio.sample_detail, command.get('sample_id'))
                elif action == 'snapshot':
                    data = await asyncio.to_thread(
                        self.studio.build_snapshot, note=command.get('note', ''))
                elif action == 'models':
                    data = {
                        'active_model_id': self.runtime.snapshot()['active_model_id'],
                        'selected_candidate_id': self.models.listed_active_id(),
                        'candidates': await asyncio.to_thread(self.models.list_candidates),
                        'health': self.runtime.snapshot()['health']['model'],
                    }
                elif action == 'model_activate':
                    data = self.models.activate(command.get('artifact_id'), self.runtime)
                elif action == 'collect_status':
                    data = self.collection.status()
                elif action == 'collect_start':
                    if not self.camera_available:
                        raise ValueError('Camera is not ready for Studio capture')
                    data = self.collection.start(
                        participant=command.get('participant'),
                        gesture=command.get('gesture'), hand=command.get('hand'),
                        target=command.get('target', 120),
                        interval_ms=command.get('interval_ms', 100))
                elif action == 'collect_pause':
                    data = self.collection.pause()
                elif action == 'collect_resume':
                    data = self.collection.resume()
                elif action == 'collect_finish':
                    data = self.collection.finish()
                else:
                    raise ValueError('Unsupported Studio action')
            return {**result, 'ok': True, 'data': data}
        except KeyError:
            return {**result, 'ok': False, 'error': 'Sample not found'}
        except (ValueError, OSError, sqlite3.Error) as exc:
            return {**result, 'ok': False, 'error': str(exc)[:200]}

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
