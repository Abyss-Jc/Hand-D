#!/usr/bin/env bash
# Freeze the same tested Python sidecar for bundling inside Tauri v2.
# Run on each target OS/architecture; cross-compiling Python wheels is unsupported.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
out="$repo/desktop/src-tauri/resources/handd-sidecar"
mkdir -p "$out"
uv run --frozen --with pyinstaller pyinstaller \
  --noconfirm --clean --onedir --name handd-sidecar \
  --distpath "$out" --workpath "$repo/build/handd-pyinstaller" \
  --specpath "$repo/build" \
  --paths "$repo" \
  --collect-all mediapipe \
  --collect-submodules handd_core \
  --hidden-import cv2 \
  scripts/packaged_sidecar.py
test -x "$out/handd-sidecar/handd-sidecar" || {
  echo "Freezer did not create the expected executable" >&2; exit 1;
}
"$out/handd-sidecar/handd-sidecar" --help >/dev/null
echo "Frozen sidecar ready: $out/handd-sidecar"
echo "Next: npm --prefix desktop run build -- --bundles deb  (Linux)"
