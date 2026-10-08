"""Historical feature-only CSV inventory and import; does not touch raw Samples."""
from __future__ import annotations

import argparse
from pathlib import Path

from handd_core.dataset_store import DatasetStore
from handd_core.legacy_import import import_legacy_csv, list_legacy_sources


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hand-D historical 69-feature CSV migration")
    p.add_argument("--workspace", required=True, type=Path)
    commands = p.add_subparsers(dest="command", required=True)
    import_cmd = commands.add_parser("import", help="Import CSV with original labels")
    import_cmd.add_argument("csv_path", type=Path)
    commands.add_parser("list", help="List legacy sources and compatibility status")
    return p


def main(argv: list[str] | None = None) -> int:
    p = parser()
    args = p.parse_args(argv)
    if args.command == "list" and not (args.workspace / "handd.sqlite").exists():
        p.error("workspace handd.sqlite does not exist")
    store = DatasetStore(args.workspace / "handd.sqlite")
    try:
        if args.command == "import":
            try:
                result = import_legacy_csv(store, args.csv_path)
            except (OSError, ValueError) as exc:
                p.error(str(exc))
            print(f"IMPORTED {result.count} legacy rows: {result.status}")
            print(f"SOURCE_ID {result.source_id}")
            print(f"ALREADY_IMPORTED {result.already_imported}")
            print("No raw Samples, fake Sessions, or model activation created.")
        elif args.command == "list":
            for item in list_legacy_sources(store):
                print(f"{item['source_filename']}\t{item['row_count']}\t"
                      f"{item['status']}\t{item['source_id']}")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
