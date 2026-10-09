"""HD-09: candidate-only model discovery, explicit activation, rollback and restart."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from handd_core.dataset_store import DatasetStore
from handd_core.feature_transform import FEATURE_TRANSFORM_ID
from handd_core.model_artifact import ARCHITECTURE_ID, GestureMLP
from handd_core.runtime_v2 import GestureRuntime
from handd_core.studio_models import StudioModels


def create_candidate(workspace, name="candidate-001", valid=True):
    folder=Path(workspace)/"models"/name
    folder.mkdir(parents=True)
    model=GestureMLP(5)
    torch.save(model.state_dict(), folder/"weights.pth")
    (folder/"metrics.json").write_text('{"macro_f1":0.5}')
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    manifest={
        "model_format_version":1,"artifact_role":"final_refit_candidate",
        "artifact_id":name,"architecture":ARCHITECTURE_ID,
        "feature_transform":FEATURE_TRANSFORM_ID,"input_features":69,
        "label_order":["Fist","Index_Finger","Ruler","Thumb_Up","Idle"],
        "weights_sha256":sha(folder/"weights.pth"),
        "metrics_sha256":sha(folder/"metrics.json"),
    }
    if not valid: manifest["input_features"]=70
    (folder/"manifest.json").write_text(json.dumps(manifest))
    return folder


class StudioModelsTests(unittest.TestCase):
    def test_bad_active_selection_does_not_block_catalog_or_replacement(self):
        create_candidate(self.root,"candidate-recovery")
        selection=self.root/"models"/".active-model.json"
        selection.write_text("{invalid json")
        with self.assertRaises(ValueError):
            self.models.active_id()
        rows=self.models.list_candidates()
        self.assertEqual(len(rows),1)
        self.assertTrue(rows[0]["compatible"])
        self.assertFalse(rows[0]["active"])
        selected=self.models.activate("candidate-recovery",self.runtime)
        self.assertEqual(selected["artifact_id"],"candidate-recovery")
        self.assertEqual(self.models.active_id(),"candidate-recovery")

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        db=DatasetStore(self.root/"handd.sqlite"); db.close()
        self.models=StudioModels(self.root)
        self.runtime=GestureRuntime()

    def test_list_only_workspace_candidates_and_reject_bad(self):
        create_candidate(self.root)
        create_candidate(self.root,"candidate-broken",valid=False)
        (self.root/"models"/"random").mkdir()
        candidates=self.models.list_candidates()
        self.assertEqual([item["artifact_id"] for item in candidates],
                         ["candidate-001","candidate-broken"])
        self.assertTrue(candidates[0]["compatible"])
        self.assertFalse(candidates[1]["compatible"])
        self.assertEqual(candidates[0]["labels"],5)
        self.assertFalse(any("weights" in str(item) for item in candidates))

    def test_explicit_activate_atomic_and_persist_restore(self):
        create_candidate(self.root)
        self.assertIsNone(self.models.active_id())
        self.assertIsNone(self.runtime.snapshot()["active_model_id"])
        activated=self.models.activate("candidate-001",self.runtime)
        self.assertEqual(activated["artifact_id"],"candidate-001")
        self.assertEqual(self.runtime.snapshot()["health"]["model"],"ready")
        self.assertEqual(self.models.active_id(),"candidate-001")
        self.assertTrue((self.root/"models"/".active-model.json").is_file())
        next_runtime=GestureRuntime()
        self.models.restore(next_runtime)
        self.assertEqual(next_runtime.snapshot()["active_model_id"],"candidate-001")
        self.assertEqual(next_runtime.snapshot()["health"]["model"],"ready")

    def test_failed_activation_preserves_live_and_stored_active(self):
        valid=create_candidate(self.root)
        broken=create_candidate(self.root,"candidate-bad",valid=False)
        self.models.activate("candidate-001",self.runtime)
        before=self.runtime.predictor
        for bad in ("candidate-bad","../candidate-001","candidate-nothing","random"):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                self.models.activate(bad,self.runtime)
            self.assertIs(self.runtime.predictor,before)
            self.assertEqual(self.models.active_id(),"candidate-001")
        (valid/"weights.pth").write_bytes(b"tampered")
        with self.assertRaises(ValueError):
            self.models.activate("candidate-001",self.runtime)
        self.assertIs(self.runtime.predictor,before)
        self.assertEqual(self.models.active_id(),"candidate-001")

    def test_untrusted_symlink_is_never_exposed_or_selected(self):
        create_candidate(self.root,"candidate-real")
        outside=Path(self.tmp.name).parent/"outside-model-missing"
        (self.root/"models"/"candidate-link").symlink_to(outside)
        self.assertEqual(len(self.models.list_candidates()),1)
        with self.assertRaises(ValueError):
            self.models.activate("candidate-link",self.runtime)

    def test_individual_artifact_files_must_not_follow_symlinks(self):
        candidate=create_candidate(self.root,"candidate-symlink-file")
        weights=candidate/"weights.pth"
        external=self.root/"unrelated-checkpoint.pth"
        weights.rename(external)
        weights.symlink_to(external)
        listing=self.models.list_candidates()
        self.assertEqual(len(listing),1)
        self.assertFalse(listing[0]["compatible"])
        with self.assertRaises(ValueError):
            self.models.activate("candidate-symlink-file",self.runtime)
        self.assertIsNone(self.runtime.snapshot()["active_model_id"])

    def test_corrupt_but_rehashed_checkpoint_is_not_loadable(self):
        folder=create_candidate(self.root,"candidate-corrupt")
        (folder/"weights.pth").write_bytes(b"not-a-valid-torch-checkpoint")
        manifest=json.loads((folder/"manifest.json").read_text())
        manifest["weights_sha256"]=hashlib.sha256(
            (folder/"weights.pth").read_bytes()).hexdigest()
        (folder/"manifest.json").write_text(json.dumps(manifest))
        result=self.models.list_candidates()
        self.assertEqual(result[0]["artifact_id"],"candidate-corrupt")
        self.assertFalse(result[0]["compatible"])
        with self.assertRaises(ValueError):
            self.models.activate("candidate-corrupt",self.runtime)
