"""Explicit initialization of a fresh canonical Hand-D Project Workspace."""
from __future__ import annotations

import argparse
from pathlib import Path

from handd_core.dataset_store import DatasetStore


def initialize_workspace(directory: Path | str) -> Path:
    workspace = Path(directory).expanduser().resolve(strict=True)
    if not workspace.is_dir():
        raise ValueError("Choose an existing empty directory")
    if any(workspace.iterdir()):
        raise ValueError("New workspace directory must be empty (existing files preserved)")
    db=workspace/"handd.sqlite"
    store=DatasetStore(db)
    store.close()
    return workspace


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description="Initialize an empty Hand-D Workspace")
    p.add_argument("--workspace",required=True,type=Path)
    args=p.parse_args(argv)
    try:
        print(initialize_workspace(args.workspace))
    except (ValueError,OSError) as exc:
        p.error(str(exc))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
