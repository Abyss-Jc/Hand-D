"""Read-only Snapshot Readiness: hard reproducibility blockers vs soft warnings."""

from __future__ import annotations

from dataclasses import dataclass

from handd_core.dataset_store import DatasetStore
from handd_core.feature_transform import canonicalize_world_landmarks


@dataclass(frozen=True)
class SnapshotReadiness:
    eligible_sample_ids: tuple[str, ...]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.blockers


def assess_snapshot_readiness(
    store: DatasetStore, *,
    development_session_ids: set[str] | None = None,
    final_test_session_ids: set[str] | None = None,
) -> SnapshotReadiness:
    """Inspect canonical samples without updating review, membership, or data."""
    blockers: list[str] = []
    warnings: list[str] = []
    final_sessions = set(final_test_session_ids or ())
    dev_sessions = set(development_session_ids) if development_session_ids is not None else None
    if dev_sessions is not None and dev_sessions & final_sessions:
        blockers.append('development_final_overlap')
    eligible = [s for s in store.list_eligible_samples()
                if (dev_sessions is None or s['session_id'] in dev_sessions)
                and s['session_id'] not in final_sessions]
    if not eligible:
        blockers.append('no_eligible_samples')
    else:
        if any(canonicalize_world_landmarks(s['world_landmarks'], s['raw_mp_handedness'])
               is None for s in eligible):
            blockers.append('invalid_feature_transform')
        if len(eligible) < 100 or len({s['session_id'] for s in eligible}) < 2:
            warnings.append('coverage_low')
    if store.get_active_review_counts()['unreviewed']:
        warnings.append('unreviewed_samples')
    return SnapshotReadiness(
        eligible_sample_ids=tuple(s['sample_id'] for s in eligible),
        blockers=tuple(blockers), warnings=tuple(warnings),
    )
