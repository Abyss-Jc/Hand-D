"""Workspace-scoped Studio services; no camera frames or automatic review decisions.

Only existing workspaces are permitted. One short-lived DatasetStore connection per
operation keeps SQLite ownership off the MediaPipe callback thread.
"""
from pathlib import Path
import hashlib
import json
import math
import random

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

    def overview(self, *, offset: int = 0, gesture: str | None = None,
                 review_status: str | None = None, participant: str | None = None) -> dict:
        if type(offset) is not int or offset < 0:
            raise ValueError('Review offset must be a nonnegative integer')
        if (gesture is not None and (not isinstance(gesture, str) or len(gesture)>64)
                or review_status not in (None,'unreviewed','accepted','rejected')
                or participant not in (None,'P001','P002','P003')):
            raise ValueError('Invalid browse filter')
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
            total=store.conn.execute(
                '''SELECT count(*) FROM samples s JOIN captures c USING(capture_id)
                   JOIN collection_sessions cs USING(session_id)
                   WHERE (? IS NULL OR c.gesture=?)
                     AND (? IS NULL OR s.review_status=?)
                     AND (? IS NULL OR cs.participant_id=?)''',
                (gesture,gesture,review_status,review_status,participant,participant),
            ).fetchone()[0]
            captures=[dict(row) for row in store.conn.execute(
                '''SELECT c.capture_id,c.gesture,cs.participant_id,
                          count(s.sample_id) AS sample_count
                   FROM captures c JOIN collection_sessions cs USING(session_id)
                   LEFT JOIN samples s USING(capture_id)
                   WHERE cs.participant_id IN ('P001','P002')
                   GROUP BY c.capture_id ORDER BY c.started_at DESC,c.capture_id DESC''',
            )]
            return {
                'workspace': str(self.path),
                'sample_count': store.count_samples(),
                'review_counts': store.get_active_review_counts(),
                'samples': store.list_samples_overview(
                    limit=40, offset=offset, include_dropped=True,
                    gesture=gesture,review_status=review_status,participant=participant),
                'browse_total': total,
                'captures': captures,
                'review_offset': offset,
                'review_page_size': 40,
                'snapshot_ready': readiness.ready,
                'snapshot_blockers': list(readiness.blockers),
                'snapshot_warnings': list(readiness.warnings),
                'eligible_count': len(readiness.eligible_sample_ids),
            }
        finally:
            store.close()

    @staticmethod
    def _quality_plan(rows: list, capture_id: str) -> dict:
        data=[(row['sample_id'],row['review_status'],row['lifecycle_status']) for row in rows]
        token=hashlib.sha256(json.dumps(data,separators=(',',':')).encode()).hexdigest()
        active=[r for r in rows if r['lifecycle_status']=='active']
        ids=sorted(r['sample_id'] for r in active)
        seed=int.from_bytes(hashlib.sha256(
            (capture_id+'|'+','.join(ids)).encode()).digest()[:8],'big')
        chosen=set(random.Random(seed).sample(
            ids,min(5,max(1,math.ceil(len(ids)*0.05))))) if ids else set()
        qc_ids=sorted(chosen)
        qc_unreviewed=[r['sample_id'] for r in active if
                       r['sample_id'] in chosen and r['review_status']=='unreviewed']
        qc_rejected=[r['sample_id'] for r in active if
                     r['sample_id'] in chosen and r['review_status']=='rejected']
        pending=sum(r['review_status']=='unreviewed' for r in active)
        return {
            'capture_id':capture_id,'token':token,'pending':pending,
            'qc_sample_ids':qc_ids,'qc_unreviewed':qc_unreviewed,
            'qc_rejected':qc_rejected,
            'can_batch_accept':pending>0 and not qc_unreviewed and not qc_rejected,
            'assessment_state':'no_model_assessment','suggested':[],
        }

    @staticmethod
    def _capture_rows(store, capture_id):
        rows=store.conn.execute(
            '''SELECT s.sample_id,s.review_status,s.lifecycle_status
               FROM samples s JOIN captures c USING(capture_id)
               JOIN collection_sessions cs USING(session_id)
               WHERE c.capture_id=? AND cs.participant_id IN ('P001','P002')
               ORDER BY s.sample_id''',(capture_id,),
        ).fetchall()
        if not rows: raise ValueError('Select a nonempty P001/P002 Capture')
        return rows

    def review_plan(self, capture_id: str) -> dict:
        if not isinstance(capture_id,str) or not (1<=len(capture_id)<=128):
            raise ValueError('Invalid Capture ID')
        store=DatasetStore(self.path/'handd.sqlite')
        try:
            return self._quality_plan(self._capture_rows(store,capture_id),capture_id)
        finally:
            store.close()

    def batch_accept(self, capture_id: str, token: str) -> dict:
        if not isinstance(capture_id,str) or not (1<=len(capture_id)<=128) or not isinstance(token,str):
            raise ValueError('Invalid batch request')
        store=DatasetStore(self.path/'handd.sqlite')
        try:
            with store.conn:
                store.conn.execute('BEGIN IMMEDIATE')
                rows=self._capture_rows(store,capture_id)
                plan=self._quality_plan(rows,capture_id)
                if token!=plan['token']:
                    raise ValueError('Capture changed; refresh Quality Check before batch acceptance')
                if not plan['can_batch_accept']:
                    raise ValueError('Resolve random QC; rejected QC requires individual review of the Capture')
                pending=[r['sample_id'] for r in rows if
                         r['lifecycle_status']=='active' and r['review_status']=='unreviewed']
                for sample_id in pending:
                    store.conn.execute(
                        '''INSERT INTO review_events
                           (sample_id,previous_status,new_status,reason,created_at)
                           VALUES (?,'unreviewed','accepted',?,
                           strftime('%Y-%m-%dT%H:%M:%fZ','now'))''',
                        (sample_id,'Studio manual batch accept after random QC'),
                    )
                    store.conn.execute(
                        "UPDATE samples SET review_status='accepted' WHERE sample_id=?",(sample_id,),
                    )
            return {'capture_id':capture_id,'accepted_count':len(pending)}
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
