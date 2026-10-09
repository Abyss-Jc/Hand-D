"""PyInstaller entrypoint for a portable Hand-D Python sidecar.

Uses the same canonical runtime and workspace_init modules as development.
No arbitrary script runner and no dependency on a repository .venv at runtime.
"""
from __future__ import annotations

import multiprocessing
import sys


def main() -> int:
    multiprocessing.freeze_support()
    if sys.argv[1:2] == ["--init-workspace"]:
        from handd_core.workspace_init import main as workspace_main
        return workspace_main(sys.argv[2:])
    from handd_core.sidecar_main import main as sidecar_main
    return sidecar_main()


if __name__ == "__main__":
    raise SystemExit(main())
