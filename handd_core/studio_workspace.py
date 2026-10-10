"""Workspace-scoped Studio services; no camera frames or automatic review decisions.

Only existing workspaces are permitted. One short-lived DatasetStore connection per
operation keeps SQLite ownership off the MediaPipe callback thread.
"""
from pathlib import Path

from handd_core.dataset_store import DatasetStore
from handd_core.snapshot_builder import build_development_snapshot
from handd_core.snapshot_readiness import assess_snapshot_readiness


class StudioWorkspace:
    def __init__(self, path: str | Path):
        try:
            workspace = Path(path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError('Select an existing Hand-D workspace with handd.sqlite') from exc
        if not workspace.is_dir() or not (workspace / 'handd.sqlite').is_file():
            raise ValueError('Select an existing Hand-D workspace with handd.sqlite')
        self.path = workspace

    def overview(self, *, offset: int = 0) -> dict:
        if type(offset) is not int or offset < 0:
            raise ValueError('Review offset must be a nonnegative integer')
        store = DatasetStore(self.path / 'handd.sqlite')
        try:
            # Match the actual Snapshot Builder's strict P001/P002 eligibility;
            # P003 must never make a Development Snapshot appear ready.
            allowed_sessions = {
                row[0] for row in store.conn.execute(
                    "SELECT session_id FROM collection_sessions "
                    "WHERE participant_id IN ('P001','P002')")
            }
            readiness = assess_snapshot_readiness(
                store, development_session_ids=allowed_sessions)
            return {
                'workspace': str(self.path),
                'sample_count': store.count_samples(),
                'review_counts': store.get_active_review_counts(),
                'samples': store.list_samples_overview(
                    limit=40, offset=offset, include_dropped=True),
                'review_offset': offset,
                'review_page_size': 40,
                'snapshot_ready': readiness.ready,
                'snapshot_blockers': list(readiness.blockers),
                'snapshot_warnings': list(readiness.warnings),
                'eligible_count': len(readiness.eligible_sample_ids),
            }
        finally:
            store.close()

    def transition(self, sample_id: str, action: str) -> dict:
        if not isinstance(sample_id, str) or not (1 <= len(sample_id) <= 128):
            raise ValueError('Invalid Sample ID')
        allowed = {'accept': 'accepted', 'reject': 'rejected',
                   'drop': 'dropped', 'restore': 'active'}
        if action not in allowed:
            raise ValueError('Unsupported manual Studio review action')
        store = DatasetStore(self.path / 'handd.sqlite')
        try:
            if action in ('accept', 'reject'):
                store.set_review_status(sample_id, allowed[action], reason='Studio manual action')
            else:
                store.set_lifecycle_status(sample_id, allowed[action], reason='Studio manual action')
            sample = store.get_sample(sample_id)
            return {'sample_id': sample_id, 'review_status': sample['review_status'],
                    'lifecycle_status': sample['lifecycle_status']}
        finally:
            store.close()

    def sample_detail(self, sample_id: str) -> dict:
        if not isinstance(sample_id, str) or not (1 <= len(sample_id) <= 128):
            raise ValueError('Invalid Sample ID')
        store = DatasetStore(self.path / 'handd.sqlite')
        try:
            sample = store.get_sample(sample_id)
            return {
                'sample_id': sample['sample_id'], 'gesture': sample['gesture'],
                'hand': sample['hand'], 'raw_mp_handedness': sample['raw_mp_handedness'],
                'review_status': sample['review_status'],
                'lifecycle_status': sample['lifecycle_status'],
                'image_landmarks': sample['image_landmarks'].tolist(),
                'world_landmarks': sample['world_landmarks'].tolist(),
            }
        finally:
            store.close()

    def build_snapshot(self, *, note: str = '') -> dict:
        if not isinstance(note, str) or len(note) > 200:
            raise ValueError('Snapshot note is invalid')
        store = DatasetStore(self.path / 'handd.sqlite')
        try:
            result = build_development_snapshot(store, self.path, note=note)
            return {'snapshot_id': result.snapshot_id, 'kind': 'development',
                    'workspace_relative_path': str(result.path.relative_to(self.path))}
        finally:
            store.close()
