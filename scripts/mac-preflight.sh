#!/usr/bin/env bash
# macOS Apple Silicon development preflight; intentionally never opens the camera.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "MAC_PREFLIGHT_NOT_RUN: requires macOS; this machine is $(uname -s)"
  exit 2
fi
echo "MAC_HOST_ARCH=$(uname -m)"
for cmd in uv cargo node npm; do
  if ! command -v "$cmd" >/dev/null; then
    echo "MISSING_TOOL=$cmd (install locally before launching)"
    exit 3
  fi
done
uv sync --frozen
npm ci --prefix desktop --no-audit --no-fund
uv lock --check
uv run --frozen python - <<'PY'
import sys, torch, mediapipe, cv2
print("PYTHON_VERSION", sys.version.split()[0])
print("TORCH", torch.__version__, "MPS_AVAILABLE", torch.backends.mps.is_available())
print("MEDIAPIPE", mediapipe.__version__)
print("OPENCV", cv2.__version__)
PY
uv run --frozen python -m unittest discover -s tests -p 'test_*.py' -q
node --test desktop/tests/*.test.mjs
cargo check --manifest-path desktop/src-tauri/Cargo.toml --locked
echo "MAC_PREFLIGHT_PASS: camera not opened, no UI or hardware claim yet."
