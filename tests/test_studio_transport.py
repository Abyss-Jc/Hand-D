"""Studio commands are opt-in, authenticated and restricted to one workspace."""
import asyncio
import tempfile
import unittest
from pathlib import Path

from aiohttp import ClientSession

from handd_core.dataset_store import DatasetStore
from handd_core.runtime_v2 import GestureRuntime
from handd_core.sidecar_ipc import SidecarServer
from tests.test_feature_transform import landmark_fixture


class StudioTransportTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name)
        store = DatasetStore(self.workspace / 'handd.sqlite')
        store.create_participant('P001')
        store.create_session('S001','P001')
        store.create_capture('C001','S001','Fist','Right')
        points = landmark_fixture()
        store.add_sample('SAMPLE0','C001',0,points,points,
                         raw_mp_handedness='Left',timestamp_ms=1,provenance={})
        store.close()
        self.server = SidecarServer(runtime=GestureRuntime(), workspace=self.workspace)
        self.port = await self.server.start()
        self.client = ClientSession()

    async def asyncTearDown(self):
        await self.client.close()
        await self.server.stop()
        self.tmp.cleanup()

    async def test_websocket_real_overview_manual_accept_and_snapshot(self):
        ws=await self.client.ws_connect(
            f'http://127.0.0.1:{self.port}/ws?token={self.server.token}')
        await ws.receive_json(timeout=3)

        async def send(action, **kwargs):
            await ws.send_json({'type':'studio.request','request_id':'request-'+action,
                                'action':action, **kwargs})
            reply=await ws.receive_json(timeout=5)
            self.assertEqual(reply['type'],'studio.response')
            self.assertEqual(reply['request_id'],'request-'+action)
            return reply

        overview=await send('overview')
        self.assertTrue(overview['ok'])
        self.assertEqual(overview['data']['review_counts']['unreviewed'],1)
        self.assertEqual(overview['data']['samples'][0]['sample_id'],'SAMPLE0')
        accepted=await send('accept', sample_id='SAMPLE0')
        self.assertTrue(accepted['ok'])
        self.assertEqual(accepted['data']['review_status'],'accepted')
        result=await send('snapshot', note='explicit user request')
        self.assertTrue(result['ok'])
        self.assertTrue((self.workspace/result['data']['workspace_relative_path']
                         / 'manifest.json').is_file())
        forbidden=await send('delete', sample_id='SAMPLE0')
        self.assertFalse(forbidden['ok'])
        await ws.close()

    async def test_without_workspace_is_safe_and_reports_unconfigured(self):
        server=SidecarServer(runtime=GestureRuntime())
        port=await server.start()
        try:
            ws=await self.client.ws_connect(
                f'http://127.0.0.1:{port}/ws?token={server.token}')
            await ws.receive_json(timeout=3)
            await ws.send_json({'type':'studio.request','request_id':'overview',
                                'action':'overview'})
            reply=await ws.receive_json(timeout=3)
            self.assertFalse(reply['ok'])
            self.assertIn('workspace',reply['error'].lower())
            await ws.close()
        finally:
            await server.stop()
