"""Human-curated review/Drop CLI until the Studio review surface is wired."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from handd_core.dataset_store import DatasetStore
from handd_core.snapshot_readiness import assess_snapshot_readiness


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Review Hand-D v2 Samples manually')
    p.add_argument('--workspace', required=True, type=Path)
    sub = p.add_subparsers(dest='command', required=True)
    listing = sub.add_parser('list', help='List Sample IDs and current states')
    listing.add_argument('--status', choices=('all','unreviewed','accepted','rejected'),
                         default='all')
    listing.add_argument('--limit', type=int, default=100)
    listing.add_argument('--include-dropped', action='store_true',
                         help='Show hidden dropped Samples for restoration')
    for action in ('accept','reject','drop','restore'):
        command = sub.add_parser(action, help=f'Human {action} on an exact Sample ID')
        command.add_argument('sample_id')
        command.add_argument('--reason')
    readiness = sub.add_parser('readiness', help='Inspect snapshot blockers/warnings')
    readiness.add_argument('--development-session', action='append')
    readiness.add_argument('--final-session', action='append')
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    db_path = args.workspace / 'handd.sqlite'
    if not db_path.is_file():
        parser().error(f'Workspace dataset not found: {db_path}')
    store = DatasetStore(db_path)
    try:
        if args.command == 'list':
            status = None if args.status == 'all' else args.status
            for row in store.list_samples_overview(
                review_status=status, limit=args.limit,
                include_dropped=args.include_dropped,
            ):
                print('\t'.join(str(row[k]) for k in (
                    'sample_id', 'gesture', 'participant_id', 'session_id',
                    'review_status', 'lifecycle_status',
                )))
        elif args.command == 'readiness':
            result = assess_snapshot_readiness(
                store,
                development_session_ids=set(args.development_session)
                if args.development_session else None,
                final_test_session_ids=set(args.final_session or ()),
            )
            print(json.dumps(asdict(result), indent=2))
        elif args.command in ('accept','reject'):
            new_status = 'accepted' if args.command == 'accept' else 'rejected'
            store.set_review_status(args.sample_id, new_status, reason=args.reason)
            print(f'{args.sample_id}: review={new_status}')
        else:
            new_status = 'dropped' if args.command == 'drop' else 'active'
            store.set_lifecycle_status(args.sample_id, new_status, reason=args.reason)
            print(f'{args.sample_id}: lifecycle={new_status}')
    finally:
        store.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
