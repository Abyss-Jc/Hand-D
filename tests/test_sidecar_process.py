"""Real Python subprocess must bootstrap over stdout and exit cleanly."""
import asyncio
import json
import sys
import unittest

from aiohttp import ClientSession


class PythonSidecarProcessTests(unittest.IsolatedAsyncioTestCase):
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
