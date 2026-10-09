import tempfile
import unittest
from pathlib import Path

from handd_core.dataset_store import DatasetStore
from handd_core.workspace_init import initialize_workspace


class WorkspaceInitTests(unittest.TestCase):
    def test_create_explicit_empty_workspace_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            initialized=initialize_workspace(path)
            self.assertEqual(initialized,path.resolve())
            store=DatasetStore(path/"handd.sqlite")
            self.assertEqual(store.count_samples(),0)
            store.close()
            with self.assertRaises(ValueError):
                initialize_workspace(path)
            self.assertTrue((path/"handd.sqlite").is_file())

    def test_reject_nonempty_dir_preserving_user_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            (path/"notes.txt").write_text("keep this")
            with self.assertRaises(ValueError):
                initialize_workspace(path)
            self.assertEqual((path/"notes.txt").read_text(),"keep this")
            self.assertFalse((path/"handd.sqlite").exists())
