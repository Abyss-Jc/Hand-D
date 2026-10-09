"""Real Python subprocess must bootstrap over stdout and exit cleanly."""
import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

from aiohttp import ClientSession
from handd_core.dataset_store import DatasetStore
from handd_core.runtime_v2 import GestureRuntime
from handd_core.studio_models import StudioModels
from tests.test_studio_models import create_candidate


class PythonSidecarProcessTests(unittest.IsolatedAsyncioTestCase):
    async def test_corrupt_persisted_active_selection_does_not_kill_whiteboard(self):
        """HD-09: bad metadata must report a model fault, not kill camera/IPC."""
        with tempfile.TemporaryDirectory() as tmp:
            store=DatasetStore(Path(tmp)/'handd.sqlite')
            store.close()
            model_dir=Path(tmp)/'models'
            model_dir.mkdir()
            (model_dir/'.active-model.json').write_text('{broken json')
            child=await asyncio.create_subprocess_exec(
                sys.executable,'-u','-m','handd_core.sidecar_main',
                '--workspace',tmp,'--no-camera',
                stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE,
            )
            try:
                line=await asyncio.wait_for(child.stdout.readline(),10)
                self.assertTrue(line,'Sidecar must still publish READY with a bad selection')
                ready=json.loads(line)
                async with ClientSession() as session:
                    async with session.get(
                        f"http://127.0.0.1:{ready['port']}/health",
                        headers={'Authorization':'Bearer '+ready['token']},
                    ) as response:
                        self.assertEqual(response.status,200)
                        snapshot=(await response.json())['snapshot']
                        self.assertEqual(snapshot['health']['model'],'error')
                        self.assertIsNone(snapshot['active_model_id'])
            finally:
                if child.returncode is None:
                    child.terminate()
                await asyncio.wait_for(child.wait(),8)

    async def test_workspace_active_v2_model_survives_real_process_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=DatasetStore(Path(tmp)/'handd.sqlite');store.close()
            create_candidate(tmp,'candidate-restart')
            StudioModels(tmp).activate('candidate-restart',GestureRuntime())
            child=await asyncio.create_subprocess_exec(
                sys.executable,'-u','-m','handd_core.sidecar_main',
                '--workspace',tmp,'--no-camera',
                stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
            try:
                ready=json.loads(await asyncio.wait_for(child.stdout.readline(),10))
                async with ClientSession() as client:
                    async with client.get(
                        f"http://127.0.0.1:{ready['port']}/health",
                        headers={'Authorization':'Bearer '+ready['token']}) as resp:
                        self.assertEqual(resp.status,200)
                        snapshot=(await resp.json())['snapshot']
                        self.assertEqual(snapshot['active_model_id'],'candidate-restart')
                        self.assertEqual(snapshot['health']['model'],'ready')
            finally:
                if child.returncode is None:child.terminate()
                await asyncio.wait_for(child.wait(),8)
    async def test_explicit_legacy_model_reports_unverified_health_on_bootstrap(self):
        checkpoint = Path(__file__).resolve().parents[1] / 'models/gesture_mlp.pth'
        process = await asyncio.create_subprocess_exec(
            sys.executable, '-u', '-m', 'handd_core.sidecar_main',
            '--no-camera', '--legacy-checkpoint', str(checkpoint),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            ready = json.loads(await asyncio.wait_for(process.stdout.readline(), 9))
            self.assertEqual(ready['type'], 'sidecar.ready')
            async with ClientSession() as client:
                async with client.get(
                    f"http://127.0.0.1:{ready['port']}/health",
                    headers={'Authorization': 'Bearer ' + ready['token']},
                ) as response:
                    self.assertEqual(response.status, 200)
                    health = (await response.json())['snapshot']
                    self.assertEqual(health['health']['model'], 'legacy_unverified')
                    self.assertTrue(health['active_model_id'].startswith('legacy-unverified-'))
        finally:
            if process.returncode is None:
                process.terminate()
            await asyncio.wait_for(process.wait(), 8)

    async def test_bootstrap_and_restart_have_separate_authentication(self):
        async def launch():
            child = await asyncio.create_subprocess_exec(
                sys.executable, '-u', '-m', 'handd_core.sidecar_main', '--no-camera',
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            try:
                line = await asyncio.wait_for(child.stdout.readline(), timeout=8)
                info = json.loads(line)
                self.assertEqual(info['type'], 'sidecar.ready')
                self.assertGreater(info['port'], 0)
                self.assertGreaterEqual(len(info['token']), 32)
                return child, info
            except BaseException:
                child.kill()
                await child.wait()
                raise

        one, first = await launch()
        try:
            async with ClientSession() as http:
                url = f"http://127.0.0.1:{first['port']}/health?token={first['token']}"
                async with http.get(url) as response:
                    self.assertEqual(response.status, 200)
        finally:
            one.terminate()
            await asyncio.wait_for(one.wait(), 8)
        two, second = await launch()
        try:
            self.assertNotEqual(first['token'], second['token'])
            self.assertNotEqual(first['runtime_session_id'], second['runtime_session_id'])
            async with ClientSession() as http:
                async with http.get(f"http://127.0.0.1:{second['port']}/health?token={first['token']}") as r:
                    self.assertEqual(r.status, 401)
        finally:
            two.terminate()
            await asyncio.wait_for(two.wait(), 8)
