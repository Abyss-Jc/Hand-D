"""Explicit CLI for grouped development training from an immutable Snapshot."""
from __future__ import annotations

import argparse
from pathlib import Path

from handd_core.model_training import TrainingBlocked, run_development_experiment


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train Hand-D final-refit Candidate after grouped Development CV"
    )
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--snapshot", required=True,
                   help="Immutable Development Snapshot directory ID in workspace/snapshots")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fractions", default="0.25,0.5,0.75,1.0",
                   help="Unique fractions of Development fold-training rows, including 1.0")
    p.add_argument("--final-legacy", action="store_true",
                   help="Explicitly include frozen legacy rows in FINAL refit")
    return p


def main(argv: list[str] | None = None) -> int:
    p = parser()
    args = p.parse_args(argv)
    snapshots = args.workspace / "snapshots"
    if not args.snapshot.startswith("dev-") or Path(args.snapshot).name != args.snapshot:
        p.error("snapshot must be a Development Snapshot ID, not a filesystem path")
    source = snapshots / args.snapshot
    if not source.is_dir():
        p.error(f"Snapshot not found: {source}")
    try:
        fractions = tuple(float(value) for value in args.fractions.split(","))
    except ValueError:
        p.error("--fractions must be comma-separated numbers")
    try:
        output = run_development_experiment(
            source, args.workspace, epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, seed=args.seed,
            learning_curve_fractions=fractions, final_legacy=args.final_legacy,
        )
    except (TrainingBlocked, ValueError) as exc:
        print(f"BLOCKED: {exc}")
        return 2
    print(f"CANDIDATE {output.model_dir}")
    print(f"DEVELOPMENT_OOF {output.report_dir / 'metrics.json'}")
    print("Active Model unchanged. P003 has NOT been evaluated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
