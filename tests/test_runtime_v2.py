"""HD-08 contract: verified Model Artifacts -> safe real-time gesture state."""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from handd_core.feature_transform import FEATURE_TRANSFORM_ID
from handd_core.model_artifact import ARCHITECTURE_ID, GestureMLP, ModelArtifactError
from handd_core.runtime_v2 import GestureRuntime, LatestRuntimeResults, RuntimeEventGate
from tests.test_feature_transform import landmark_fixture


def observation(*raw_hands, x=0.31, y=0.64):
    world = landmark_fixture()
    image = np.tile([x, y, 0.0], (21, 1))
    point_list = lambda coordinates: [
        SimpleNamespace(x=float(a), y=float(b), z=float(c))
        for a, b, c in coordinates
    ]
    return SimpleNamespace(
        handedness=[[SimpleNamespace(category_name=hand)] for hand in raw_hands],
        hand_landmarks=[point_list(image) for _ in raw_hands],
        hand_world_landmarks=[point_list(world) for _ in raw_hands],
    )


class FakePredictor:
    label_order = ('Index_Finger', 'Fist', 'Ruler', 'Thumb_Up', 'Idle', 'Custom')

    def __init__(self, label='Index_Finger'):
        self.label = label
        self.calls = []

    def predict(self, features):
        self.calls.append(np.array(features, copy=True))
        return [self.label] * len(features)


class RuntimeTests(unittest.TestCase):
    def test_all_21_landmarks_per_physical_role_and_safe_release(self):
        runtime = GestureRuntime(predictor=FakePredictor(), stable_frames=1, stable_ms=0)
        result = observation('Left', 'Right', x=.34, y=.62)
        first = runtime.process_result(result, 100)
        for role, physical in (('drawing', 'Right'), ('modifier', 'Left')):
            hand = first['payload'][role]
            self.assertEqual(hand['physical_hand'], physical)
            self.assertEqual(len(hand['landmarks']), 21)
            self.assertEqual(hand['landmarks'][8], {'x': .34, 'y': .62})
        self.assertEqual(first['payload']['landmark_contract'], 'handd.v2.image21.xy.1')
        gone = runtime.process_result(observation(), 140)
        self.assertIsNone(gone['payload']['drawing']['landmarks'])
        self.assertIsNone(gone['payload']['modifier']['landmarks'])
        stale = runtime.expire_if_stale(500)
        self.assertIsNone(stale['payload']['drawing']['landmarks'])

    def test_invalid_outlier_image_points_are_safe_for_rendering(self):
        runtime = GestureRuntime(predictor=FakePredictor(), stable_frames=1, stable_ms=0)
        result = observation('Left')
        result.hand_landmarks[0][3].x = 1.08
        packet = runtime.process_result(result, 100)
        self.assertEqual(packet['payload']['drawing']['landmarks'][3]['x'], 1.0)
        invalid = observation('Left')
        invalid.hand_landmarks[0][3].x = float('nan')
        self.assertIsNone(runtime.process_result(invalid, 150)['payload']['drawing']['landmarks'])

    def build_artifact(self, folder: Path) -> Path:
        folder.mkdir()
        model = GestureMLP(5)
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()
            model.output.bias[0] = 12.0
        torch.save(model.state_dict(), folder / 'weights.pth')
        (folder / 'metrics.json').write_text('{"macro_f1":0.0}', encoding='utf-8')
        checksum = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = {
            'model_format_version': 1, 'artifact_role': 'final_refit_candidate',
            'artifact_id': 'candidate-test', 'architecture': ARCHITECTURE_ID,
            'feature_transform': FEATURE_TRANSFORM_ID, 'input_features': 69,
            'label_order': ['Fist', 'Index_Finger', 'Ruler', 'Thumb_Up', 'Idle'],
            'weights_sha256': checksum(folder / 'weights.pth'),
            'metrics_sha256': checksum(folder / 'metrics.json'),
        }
        (folder / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        return folder

    def test_drawing_hand_tracks_immediately_but_actions_need_stable_evidence(self):
        predictor = FakePredictor()
        runtime = GestureRuntime(predictor=predictor, stable_frames=3, stable_ms=60)
        first = runtime.process_result(observation('Left'), 100)
        self.assertEqual(first['type'], 'runtime.update')
        self.assertEqual(first['seq'], 1)
        drawing = first['payload']['drawing']
        self.assertEqual(drawing['physical_hand'], 'Right')
        self.assertEqual(drawing['pointer'], {'x': .31, 'y': .64})
        self.assertEqual(drawing['raw_gesture'], 'Index_Finger')
        self.assertIsNone(drawing['action'])
        runtime.process_result(observation('Left'), 125)
        ready = runtime.process_result(observation('Left'), 165)
        self.assertEqual(ready['payload']['drawing']['stable_gesture'], 'Index_Finger')
        self.assertEqual(ready['payload']['drawing']['action'], 'draw')
        self.assertEqual(ready['payload']['modifier']['action'], None)
        self.assertEqual(predictor.calls[0].shape, (1, 69))
        self.assertEqual(ready['payload']['health']['model'], 'ready')

    def test_transient_wrong_gesture_does_not_erase_or_break_stroke(self):
        predictor = FakePredictor()
        runtime = GestureRuntime(predictor=predictor, stable_frames=3, stable_ms=60)
        for ms in (0, 35, 75):
            runtime.process_result(observation('Left'), ms)
        predictor.label = 'Fist'
        transient = runtime.process_result(observation('Left', x=.48), 90)
        self.assertEqual(transient['payload']['drawing']['action'], 'draw')
        self.assertEqual(transient['payload']['drawing']['pointer']['x'], .48)
        predictor.label = 'Index_Finger'
        recovered = runtime.process_result(observation('Left'), 110)
        self.assertEqual(recovered['payload']['drawing']['action'], 'draw')
        self.assertNotEqual(recovered['payload']['drawing']['action'], 'erase')

    def test_modifier_cannot_take_drawing_role_when_drawing_hand_disappears(self):
        predictor = FakePredictor('Index_Finger')
        runtime = GestureRuntime(predictor=predictor, stable_frames=1, stable_ms=0)
        runtime.process_result(observation('Left', 'Right'), 100)
        no_drawing_hand = runtime.process_result(observation('Right'), 125)
        self.assertIsNone(no_drawing_hand['payload']['drawing']['pointer'])
        self.assertIsNone(no_drawing_hand['payload']['drawing']['action'])
        self.assertEqual(no_drawing_hand['payload']['modifier']['physical_hand'], 'Left')
        runtime.set_roles(drawing_hand='Left', modifier_hand='Right')
        switched = runtime.process_result(observation('Right'), 180)
        self.assertEqual(switched['payload']['drawing']['physical_hand'], 'Left')
        self.assertEqual(switched['payload']['drawing']['action'], 'draw')

    def test_absent_hand_or_invalid_world_geometry_stops_action_immediately(self):
        predictor = FakePredictor('Fist')
        runtime = GestureRuntime(predictor=predictor, stable_frames=1, stable_ms=0)
        self.assertEqual(runtime.process_result(observation('Left'), 100)['payload']['drawing']['action'],
                         'erase')
        self.assertIsNone(runtime.process_result(observation(), 140)['payload']['drawing']['action'])
        self.assertIsNone(runtime.process_result(observation('Left'), 180)['payload']['modifier']['action'])
        invalid = observation('Left')
        invalid.hand_world_landmarks = [[SimpleNamespace(x=0., y=0., z=0.)] * 21]
        self.assertIsNone(runtime.process_result(invalid, 220)['payload']['drawing']['pointer'])
        # A new valid gesture is allowed to act again if configured with 1-frame
        # stabilization; disappearance did not leave an erase action running.
        self.assertEqual(runtime.process_result(observation('Left'), 250)['payload']['drawing']['action'],
                         'erase')

    def test_out_of_order_callback_is_ignored_without_seq_or_action_change(self):
        runtime = GestureRuntime(predictor=FakePredictor(), stable_frames=1, stable_ms=0)
        first = runtime.process_result(observation('Left'), 500)
        self.assertIsNone(runtime.process_result(observation(), 499))
        self.assertIsNone(runtime.process_result(observation(), 500))
        second = runtime.process_result(observation('Left'), 501)
        self.assertEqual(second['seq'], first['seq'] + 1)
        self.assertEqual(second['payload']['drawing']['action'], 'draw')

    def test_custom_unsupported_gestures_are_inert_and_no_model_keeps_pointer(self):
        unknown = GestureRuntime(predictor=FakePredictor('Custom'), stable_frames=1, stable_ms=0)
        self.assertIsNone(unknown.process_result(observation('Left'), 50)['payload']['drawing']['action'])
        no_model = GestureRuntime()
        state = no_model.process_result(observation('Left'), 50)
        self.assertIsNotNone(state['payload']['drawing']['pointer'])
        self.assertIsNone(state['payload']['drawing']['action'])
        self.assertEqual(state['payload']['health']['model'], 'unavailable')

    def test_activation_validates_artifact_and_preserves_active_on_bad_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = self.build_artifact(root / 'candidate')
            runtime = GestureRuntime(stable_frames=1, stable_ms=0)
            info = runtime.activate_model(candidate)
            self.assertEqual(info['artifact_id'], 'candidate-test')
            self.assertEqual(runtime.snapshot()['active_model_id'], 'candidate-test')
            result = runtime.process_result(observation('Left'), 100)
            self.assertEqual(result['payload']['drawing']['raw_gesture'], 'Fist')
            self.assertEqual(result['payload']['drawing']['action'], 'erase')
            modified = json.loads((candidate / 'manifest.json').read_text())
            modified['feature_transform'] = 'not-our-feature-contract'
            (candidate / 'manifest.json').write_text(json.dumps(modified))
            with self.assertRaises(ModelArtifactError):
                runtime.activate_model(candidate)
            self.assertEqual(runtime.snapshot()['active_model_id'], 'candidate-test')
            self.assertEqual(runtime.process_result(observation('Left'), 135)
                             ['payload']['drawing']['action'], 'erase')

    def test_explicit_legacy_checkpoint_has_unverified_metadata_but_predicts(self):
        checkpoint = Path(__file__).resolve().parents[1] / 'models/gesture_mlp.pth'
        runtime = GestureRuntime(stable_frames=1, stable_ms=0)
        self.assertEqual(runtime.snapshot()['health']['model'], 'unavailable')
        metadata = runtime.activate_legacy_checkpoint(checkpoint)
        self.assertEqual(metadata['kind'], 'legacy_diagnostic')
        self.assertTrue(metadata['label_order_unverified'])
        self.assertTrue(runtime.snapshot()['active_model_id'].startswith('legacy-unverified-'))
        state = runtime.process_result(observation('Left'), 100)
        self.assertIn(state['payload']['drawing']['raw_gesture'],
                      ('Fist', 'Index_Finger', 'Ruler', 'Thumb_Up', 'Idle'))
        self.assertEqual(state['payload']['health']['model'], 'legacy_unverified')

    def test_custom_label_only_acts_after_explicit_supported_mapping(self):
        runtime = GestureRuntime(predictor=FakePredictor('Custom'),
                                 stable_frames=1, stable_ms=0)
        self.assertIsNone(runtime.process_result(observation('Left'), 5)
                          ['payload']['drawing']['action'])
        with self.assertRaises(ValueError):
            runtime.set_actions(drawing={'Idle': 'erase'})
        with self.assertRaises(ValueError):
            runtime.set_actions(drawing={'Custom': 'ruler'})
        runtime.set_actions(drawing={'Custom': 'draw'})
        self.assertEqual(runtime.process_result(observation('Left'), 50)
                         ['payload']['drawing']['action'], 'draw')
        runtime.set_actions(drawing={})
        self.assertIsNone(runtime.process_result(observation('Left'), 85)
                          ['payload']['drawing']['action'])

    def test_frontend_gate_discards_old_session_duplicate_and_out_of_order_events(self):
        gate = RuntimeEventGate()
        original = GestureRuntime(predictor=FakePredictor())
        no_hand = observation()
        initial = original.process_result(no_hand, 100)
        self.assertFalse(gate.accept(initial))  # must install READY snapshot first
        gate.install_snapshot(original.snapshot())
        fresh = original.process_result(no_hand, 120)
        self.assertTrue(gate.accept(fresh))
        self.assertFalse(gate.accept(fresh))
        self.assertFalse(gate.accept(initial))
        restarted = GestureRuntime(predictor=FakePredictor())
        gate.install_snapshot(restarted.snapshot())
        self.assertFalse(gate.accept(original.process_result(no_hand, 140)))
        self.assertTrue(gate.accept(restarted.process_result(no_hand, 10)))
        self.assertNotEqual(original.runtime_session_id, restarted.runtime_session_id)

    def test_latest_only_callbacks_do_not_run_inference_until_owner_thread_drains(self):
        predictor = FakePredictor()
        runtime = GestureRuntime(predictor=predictor, stable_frames=1, stable_ms=0)
        bridge = LatestRuntimeResults()
        bridge.on_result(observation('Left'), None, 100)
        bridge.on_result(observation('Left', x=.80), None, 200)
        self.assertEqual(len(predictor.calls), 0)
        applied = bridge.drain_to(runtime)
        self.assertEqual(applied['timestamp_ms'], 200)
        self.assertEqual(applied['payload']['drawing']['pointer']['x'], .8)
        self.assertEqual(len(predictor.calls), 1)
        self.assertIsNone(bridge.drain_to(runtime))
        bridge.on_result(observation(), None, 150)
        self.assertIsNone(bridge.drain_to(runtime))
        self.assertGreaterEqual(bridge.statistics()['superseded_callbacks'], 1)
        self.assertGreaterEqual(bridge.statistics()['stale_callbacks'], 1)

    def test_ambiguous_duplicate_physical_hand_never_chooses_arbitrary_detection(self):
        runtime = GestureRuntime(predictor=FakePredictor('Fist'),
                                 stable_frames=1, stable_ms=0)
        self.assertEqual(runtime.process_result(observation('Left'), 100)
                         ['payload']['drawing']['action'], 'erase')
        ambiguous = runtime.process_result(observation('Left', 'Left'), 130)
        self.assertIsNone(ambiguous['payload']['drawing']['pointer'])
        self.assertIsNone(ambiguous['payload']['drawing']['action'])

    def test_out_of_frame_pointer_must_not_trigger_drawing_or_erase(self):
        runtime = GestureRuntime(predictor=FakePredictor('Fist'),
                                 stable_frames=1, stable_ms=0)
        output = runtime.process_result(observation('Left', x=1.42), 100)
        self.assertIsNone(output['payload']['drawing']['pointer'])
        self.assertIsNone(output['payload']['drawing']['action'])

    def test_camera_gap_forces_safe_release_once_with_health(self):
        runtime = GestureRuntime(predictor=FakePredictor('Fist'),
                                 stable_frames=1, stable_ms=0)
        runtime.process_result(observation('Left'), 100)
        self.assertIsNone(runtime.expire_if_stale(290, max_gap_ms=250))
        released = runtime.expire_if_stale(400, max_gap_ms=250)
        self.assertEqual(released['payload']['health']['camera'], 'stale')
        self.assertIsNone(released['payload']['drawing']['action'])
        self.assertIsNone(released['payload']['drawing']['pointer'])
        self.assertIsNone(runtime.expire_if_stale(420, max_gap_ms=250))
        resumed = runtime.process_result(observation('Left'), 430)
        self.assertEqual(resumed['payload']['drawing']['action'], 'erase')

    def test_model_inference_failure_stops_actions_without_crashing_camera_loop(self):
        class FailingModel:
            def predict(self, features):
                raise RuntimeError('backend was lost')
        runtime = GestureRuntime(predictor=FailingModel(), stable_frames=1, stable_ms=0)
        result = runtime.process_result(observation('Left'), 10)
        self.assertIsNone(result['payload']['drawing']['action'])
        self.assertEqual(result['payload']['health']['model'], 'error')

    def test_role_action_updates_are_atomic_if_one_mapping_is_invalid(self):
        runtime = GestureRuntime(predictor=FakePredictor('Index_Finger'),
                                 stable_frames=1, stable_ms=0)
        before = runtime.process_result(observation('Left'), 10)
        self.assertEqual(before['payload']['drawing']['action'], 'draw')
        with self.assertRaises(ValueError):
            runtime.set_actions(drawing={}, modifier={'Index_Finger': 'erase'})
        after = runtime.process_result(observation('Left'), 20)
        self.assertEqual(after['payload']['drawing']['action'], 'draw')

    def test_one_failed_hand_inference_disables_both_roles_for_that_frame(self):
        class FlakyPredictor:
            label_order = ('Fist', 'Ruler')
            calls = 0

            def predict(self, features):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError('CPU backend unavailable for frame')
                return ['Ruler']
        runtime = GestureRuntime(predictor=FlakyPredictor(),
                                 stable_frames=1, stable_ms=0)
        packet = runtime.process_result(observation('Left', 'Right'), 20)
        self.assertEqual(packet['payload']['health']['model'], 'error')
        self.assertIsNone(packet['payload']['drawing']['action'])
        self.assertIsNone(packet['payload']['modifier']['action'])

    def test_live_runtime_cli_contract_without_opening_camera(self):
        help_result = subprocess.run(
            [sys.executable, '-m', 'handd_core.runtime_cli', '--help'],
            capture_output=True, text=True, timeout=20, check=False,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn('--model-artifact', help_result.stdout)
        self.assertIn('--seconds', help_result.stdout)
        self.assertIn('--camera', help_result.stdout)
        with tempfile.TemporaryDirectory() as tmp:
            candidate = self.build_artifact(Path(tmp) / 'candidate')
            invalid_duration = subprocess.run(
                [sys.executable, '-m', 'handd_core.runtime_cli',
                 '--model-artifact', str(candidate), '--seconds', '0'],
                capture_output=True, text=True, timeout=20, check=False,
            )
            self.assertNotEqual(invalid_duration.returncode, 0)
            self.assertIn('seconds', invalid_duration.stderr.lower())


if __name__ == '__main__':
    unittest.main()
