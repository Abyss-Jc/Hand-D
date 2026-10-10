"""HD-10 dry-run: one integrated tracer, explicitly NOT real gesture evidence.

Landmarks and intended labels are synthetic. This tests public Studio WebSocket
commands and the model/runtime contract, not camera hardware or model accuracy.
"""
import asyncio
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
from time import perf_counter
import unittest

from aiohttp import ClientSession
from numpy import percentile

from handd_core.dataset_store import DatasetStore
from handd_core.feature_transform import canonicalize_world_landmarks
from handd_core.runtime_v2 import GestureRuntime, RuntimeEventGate
from handd_core.sidecar_ipc import SidecarServer
from handd_core.snapshot_builder import verify_snapshot
from handd_core.studio_models import StudioModels
from tests.test_feature_transform import landmark_fixture
from tests.test_runtime_v2 import observation


class HD10TracerTests(unittest.IsolatedAsyncioTestCase):
    async def test_synthetic_samples_through_studio_snapshot_candidate_and_live_ipc(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            db = DatasetStore(workspace / "handd.sqlite")
            db.close()
            runtime = GestureRuntime()
            server = SidecarServer(runtime=runtime, workspace=workspace)
            server.collection.device_config = workspace / "test-device-id.txt"
            port = await server.start()
            try:
                async with ClientSession() as client:
                    async with client.ws_connect(
                        f"http://127.0.0.1:{port}/ws?token={server.token}"
                    ) as ws:
                        initial = await ws.receive_json(timeout=5)
                        self.assertEqual(initial["type"], "runtime.ready")
                        gate = RuntimeEventGate()
                        gate.install_snapshot(initial["snapshot"])

                        async def request(action, **fields):
                            await ws.send_json({
                                "type": "studio.request", "request_id": action,
                                "action": action, **fields,
                            })
                            response = await ws.receive_json(timeout=15)
                            self.assertEqual(response["type"], "studio.response")
                            self.assertEqual(response["request_id"], action)
                            self.assertTrue(response["ok"], response.get("error"))
                            return response["data"]

                        # Mimics deliberate operator labels, NOT independently
                        # observed gestures; every frame is the same test fixture.
                        server.camera_available = True
                        labels = ("Fist", "Index_Finger", "Ruler", "Thumb_Up", "Idle")
                        for person_index, participant in enumerate(("P001", "P002")):
                            for label_index, label in enumerate(labels):
                                started = await request(
                                    "collect_start", participant=participant,
                                    gesture=label, hand="Right", target=2,
                                    interval_ms=20,
                                )
                                self.assertEqual(started["state"], "capturing")
                                start_ms = 1000 + 100 * (person_index * 5 + label_index)
                                for i in range(2):
                                    self.assertIsNotNone(server.collection.offer_result(
                                        observation("Left"), start_ms + 25 * i,
                                    ))
                                self.assertEqual(
                                    (await request("collect_status"))["state"], "complete"
                                )
                        overview = await request("overview")
                        self.assertEqual(overview["sample_count"], 20)
                        self.assertEqual(overview["review_counts"]["unreviewed"], 20)
                        samples = [item["sample_id"] for item in overview["samples"]]
                        self.assertEqual(len(set(samples)), 20)

                        # An unreviewed, rejected and dropped observation must
                        # never slip into the immutable Development Snapshot.
                        await request("reject", sample_id=samples[1])
                        await request("drop", sample_id=samples[2])
                        for sample_id in samples[3:]:
                            await request("accept", sample_id=sample_id)
                        frozen = await request("snapshot", note="synthetic HD-10 tracer")
                        snapshot_dir = workspace / frozen["workspace_relative_path"]
                        self.assertTrue(verify_snapshot(snapshot_dir))
                        manifest = json.loads((snapshot_dir / "manifest.json").read_text())
                        self.assertEqual(manifest["sample_count"], 17)
                        self.assertFalse(set(samples[:3]) & set(manifest["sample_ids"]))
                        original_hash = sha256((snapshot_dir / "dataset.npz").read_bytes()).hexdigest()

                        # Editing curation after freezing cannot rewrite history.
                        await request("reject", sample_id=samples[3])
                        self.assertTrue(verify_snapshot(snapshot_dir))
                        self.assertEqual(
                            sha256((snapshot_dir / "dataset.npz").read_bytes()).hexdigest(),
                            original_hash,
                        )
                        later = await request("snapshot", note="one more excluded sample")
                        later_manifest = json.loads((
                            workspace / later["workspace_relative_path"] / "manifest.json"
                        ).read_text())
                        self.assertEqual(later_manifest["sample_count"], 16)

                        # Train only from frozen data, then explicitly activate
                        # the verified Candidate through the Studio IPC contract.
                        trainer = await asyncio.create_subprocess_exec(
                            sys.executable, "-m", "handd_core.train_cli",
                            "--workspace", str(workspace),
                            "--snapshot", frozen["snapshot_id"],
                            "--epochs", "1", "--fractions", "1.0", "--seed", "17",
                            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                        )
                        out, err = await asyncio.wait_for(trainer.communicate(), 60)
                        self.assertEqual(trainer.returncode, 0, err.decode())
                        self.assertIn(b"Active Model unchanged", out)
                        model_dir, = (workspace / "models").glob("candidate-*")
                        report_dir, = (workspace / "reports").glob("experiment-*")
                        report = json.loads((report_dir / "metrics.json").read_text())
                        oof = report["variants"]["without_legacy"]["oof"]
                        self.assertEqual(len(oof), 17)
                        self.assertTrue(all(not row["trained_on_this_sample"] for row in oof))
                        self.assertFalse(report["unseen_participant_evaluation"])
                        active = await request("model_activate", artifact_id=model_dir.name)
                        self.assertEqual(active["artifact_id"], model_dir.name)
                        activated_event = await ws.receive_json(timeout=5)
                        self.assertEqual(activated_event["type"], "studio.model")
                        self.assertEqual(activated_event["snapshot"]["health"]["model"], "ready")

                        # Only NEW observations may be flagged by final-refit
                        # Candidate, never training members from the snapshot.
                        points = landmark_fixture()
                        predictor, _ = StudioModels(workspace)._load(model_dir.name)
                        predicted = predictor.predict(
                            canonicalize_world_landmarks(points, "Left").reshape(1, -1))[0]
                        intended = next(label for label in labels if label != predicted)
                        additional = DatasetStore(workspace / "handd.sqlite")
                        try:
                            additional.create_session("NEW-SESSION", "P001")
                            additional.create_capture("NEW-CAPTURE", "NEW-SESSION",
                                                      intended, "Right")
                            for index in range(2):
                                additional.add_sample(
                                    f"NEW-{index}", "NEW-CAPTURE", index, points, points,
                                    raw_mp_handedness="Left", timestamp_ms=9000 + index,
                                    provenance={"test": "synthetic review probe"},
                                )
                        finally:
                            additional.close()
                        proposed = await request("review_plan", capture_id="NEW-CAPTURE")
                        self.assertEqual(proposed["assessment_state"], "model_assessed")
                        self.assertEqual(proposed["assessment_model_id"], model_dir.name)
                        self.assertEqual(proposed["assessed_count"], 2)
                        self.assertEqual({r["sample_id"] for r in proposed["suggested"]},
                                         {"NEW-0", "NEW-1"})
                        self.assertTrue(all(
                            row["reason"] == "model_disagreement"
                            and not row["trained_on_this_sample"]
                            for row in proposed["suggested"]))
                        self.assertFalse(proposed["can_batch_accept"])
                        history = list((workspace / "assessments").glob("*.json"))
                        self.assertEqual(len(history), 1)
                        provenance = json.loads(history[0].read_text())
                        self.assertEqual(provenance["model_artifact_id"], model_dir.name)
                        self.assertEqual(provenance["source_snapshot_id"], frozen["snapshot_id"])
                        self.assertEqual(len(provenance["assessments"]), 2)
                        immutable_bytes = history[0].read_bytes()
                        history[0].write_text('{"invalid":"cache"}')
                        corrupt = await request("review_plan", capture_id="NEW-CAPTURE")
                        self.assertEqual(corrupt["assessment_state"], "assessment_unavailable")
                        self.assertFalse(corrupt["can_batch_accept"])
                        history[0].write_bytes(immutable_bytes)
                        for sample_id in ("NEW-0", "NEW-1"):
                            await request("accept", sample_id=sample_id)
                        resolved = await request("review_plan", capture_id="NEW-CAPTURE")
                        self.assertEqual(resolved["suggested_pending"], 0)
                        self.assertEqual(history[0].read_bytes(), immutable_bytes)
                        self.assertEqual(len(list((workspace / "assessments").glob("*.json"))), 1)

                        # A novel gesture is collectable before retraining,
                        # but five-class predictions must not condemn it.
                        custom = DatasetStore(workspace / "handd.sqlite")
                        try:
                            custom.create_capture("CUSTOM-CAPTURE", "NEW-SESSION",
                                                  "Future_Gesture", "Right")
                            custom.add_sample(
                                "CUSTOM-0", "CUSTOM-CAPTURE", 0, points, points,
                                raw_mp_handedness="Left", timestamp_ms=9100,
                                provenance={"test": "new gesture"},
                            )
                        finally:
                            custom.close()
                        unsupported = await request("review_plan", capture_id="CUSTOM-CAPTURE")
                        self.assertEqual(unsupported["assessment_state"], "unsupported_gesture")
                        self.assertEqual(unsupported["suggested"], [])
                        self.assertFalse(unsupported["can_batch_accept"])

                        # Real loopback delivery of runtime updates from the new
                        # model; no claim about camera-to-WebKit/browser latency.
                        transport_samples_ms = []
                        for i in range(8):
                            event = runtime.process_result(observation("Left"), 5000 + 70 * i)
                            self.assertEqual(event["payload"]["health"]["model"], "ready")
                            began = perf_counter()
                            await server.publish_event(event)
                            received = await ws.receive_json(timeout=5)
                            transport_samples_ms.append((perf_counter() - began) * 1000)
                            self.assertTrue(gate.accept(received))
                            drawing = received["payload"]["drawing"]
                            self.assertEqual(len(drawing["landmarks"]), 21)
                            self.assertIsNotNone(drawing["pointer"])
                            self.assertIn(drawing["raw_gesture"], labels)
                        self.assertGreaterEqual(runtime.snapshot()["health"]["compute_ms_p95"], 0)
                        self.assertGreaterEqual(min(transport_samples_ms), 0)
                        print(
                            f"HD10_SYNTHETIC_IPC_P95_MS={percentile(transport_samples_ms, 95):.3f} "
                            f"HD10_PYTHON_COMPUTE_P95_MS="
                            f"{runtime.snapshot()['health']['compute_ms_p95']:.3f} "
                            "NOT_CAMERA_OR_FRONTEND_LATENCY",
                        )
            finally:
                await server.stop()

            # Restart the real Python process, not the test-owned runtime.
            # It must restore the Candidate trained above from the workspace.
            child = await asyncio.create_subprocess_exec(
                sys.executable, "-u", "-m", "handd_core.sidecar_main",
                "--no-camera", "--workspace", str(workspace),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            try:
                bootstrap = json.loads(await asyncio.wait_for(child.stdout.readline(), 12))
                self.assertEqual(bootstrap["type"], "sidecar.ready")
                async with ClientSession() as client:
                    async with client.get(
                        f"http://127.0.0.1:{bootstrap['port']}/health",
                        headers={"Authorization": "Bearer " + bootstrap["token"]},
                    ) as response:
                        self.assertEqual(response.status, 200)
                        state = (await response.json())["snapshot"]
                        self.assertEqual(state["active_model_id"], model_dir.name)
                        self.assertEqual(state["health"]["model"], "ready")
            finally:
                if child.returncode is None:
                    child.terminate()
                await asyncio.wait_for(child.wait(), 8)
