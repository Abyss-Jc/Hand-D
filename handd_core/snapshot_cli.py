"""Manual, explicit Snapshot Builder CLI; never launches model training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from handd_core.dataset_store import DatasetStore
from handd_core.snapshot_builder import (
    SnapshotBlocked, build_development_snapshot, verify_snapshot,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hand-D immutable Development Snapshots")
    p.add_argument("--workspace", type=Path, required=True)
    commands = p.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Freeze eligible rows without training")
    build.add_argument("--note")
    build.add_argument("--legacy-csv", type=Path)
    build.add_argument("--development-session", action="append")
    build.add_argument("--final-test-session", action="append")
    build.add_argument("--seed", type=int, default=42)
    verify = commands.add_parser("verify", help="Verify a frozen Snapshot locally")
    verify.add_argument("snapshot_id")
    return p


def main(argv: list[str] | None = None) -> int:
    p = parser()
    args = p.parse_args(argv)
    workspace = args.workspace
    if args.command == "verify":
        snapshots = workspace / "snapshots"
        path = snapshots / args.snapshot_id
        if path.parent != snapshots or not verify_snapshot(path):
            print("INVALID snapshot: integrity/contract check failed")
            return 3
        print(f"VERIFIED {args.snapshot_id}")
        return 0
    db_path = workspace / "handd.sqlite"
    if not db_path.is_file():
        p.error(f"Canonical SQLite not found: {db_path}")
    store = DatasetStore(db_path)
    try:
        try:
            built = build_development_snapshot(
                store, workspace, note=args.note, legacy_csv=args.legacy_csv,
                development_session_ids=set(args.development_session)
                if args.development_session else None,
                final_test_session_ids=set(args.final_test_session or ()),
                seed=args.seed,
            )
        except SnapshotBlocked as exc:
            print(f"BLOCKED {exc}")
            return 2
    finally:
        store.close()
    manifest = json.loads((built.path / "manifest.json").read_text(encoding="utf-8"))
    print(f"SNAPSHOT {built.snapshot_id} | {manifest['sample_count']} v2 Samples")
    print(f"LOCATION {built.path}")
    print(f"WARNINGS {manifest['warnings']}")
    print("Training was NOT launched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
