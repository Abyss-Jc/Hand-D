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
from tests.test_runtime_v2 import observation
from tests.test_studio_models import create_candidate


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
        self.server.collection.device_config = self.workspace / 'device-id.txt'
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

    async def test_model_catalog_activation_rollback_and_status_over_authenticated_ws(self):
        create_candidate(self.workspace, "candidate-primary")
        create_candidate(self.workspace, "candidate-invalid", valid=False)
        ws=await self.client.ws_connect(
            f'http://127.0.0.1:{self.port}/ws?token={self.server.token}')
        await ws.receive_json(timeout=3)
        async def request(action, **payload):
            await ws.send_json({'type':'studio.request', 'request_id':action,
                                'action':action,**payload})
            return await ws.receive_json(timeout=5)
        status=await request('models')
        self.assertTrue(status['ok'])
        self.assertIsNone(status['data']['active_model_id'])
        self.assertEqual(len(status['data']['candidates']),2)
        good=await request('model_activate',artifact_id='candidate-primary')
        self.assertTrue(good['ok'])
        self.assertEqual(good['data']['artifact_id'],'candidate-primary')
        self.assertEqual(self.server.runtime.snapshot()['active_model_id'],
                         'candidate-primary')
        notification=await ws.receive_json(timeout=5)
        self.assertEqual(notification['type'],'studio.model')
        fail=await request('model_activate',artifact_id='candidate-invalid')
        self.assertFalse(fail['ok'])
        self.assertEqual(self.server.runtime.snapshot()['active_model_id'],
                         'candidate-primary')
        status=await request('models')
        self.assertEqual(status['data']['active_model_id'],'candidate-primary')
        await ws.close()

    async def test_explicit_collect_start_pause_resume_status_and_finish(self):
        ws=await self.client.ws_connect(
            f'http://127.0.0.1:{self.port}/ws?token={self.server.token}')
        await ws.receive_json(timeout=3)
        async def action(name, **fields):
            await ws.send_json({'type':'studio.request','request_id':name,
                                'action':name,**fields})
            answer=await ws.receive_json(timeout=4)
            self.assertEqual(answer['type'],'studio.response')
            self.assertEqual(answer['request_id'],name)
            return answer
        self.assertEqual((await action('collect_status'))['data']['state'],'idle')
        no_camera=await action('collect_start',participant='P001',
                               gesture='Fist',hand='Right',target=2)
        self.assertFalse(no_camera['ok'])
        self.assertIn('camera', no_camera['error'].lower())
        self.server.camera_available=True  # a synthetic camera in this transport test
        invalid=await action('collect_start',participant='P003',
                             gesture='Fist',hand='Right',target=2)
        self.assertFalse(invalid['ok'])
        begun=await action('collect_start',participant='P001',
                           gesture='Fist',hand='Right',target=2)
        self.assertTrue(begun['ok'])
        self.assertEqual(begun['data']['state'],'capturing')
        self.assertFalse((await action('collect_start',participant='P001',
                                      gesture='Fist',hand='Right',target=2))['ok'])
        paused=await action('collect_pause')
        self.assertEqual(paused['data']['state'],'paused')
        self.assertIsNone(self.server.collection.offer_result(observation('Left'),100))
        resumed=await action('collect_resume')
        self.assertEqual(resumed['data']['state'],'capturing')
        self.assertIsNotNone(self.server.collection.offer_result(observation('Left'),110))
        progress=await action('collect_status')
        self.assertEqual(progress['data']['count'],1)
        stopped=await action('collect_finish')
        self.assertEqual(stopped['data']['state'],'finished')
        self.assertEqual(stopped['data']['count'],1)
        store=DatasetStore(self.workspace/'handd.sqlite')
        try:
            self.assertEqual(store.count_samples(),2) # existing seed + collected
        finally:
            store.close()
        await ws.close()
