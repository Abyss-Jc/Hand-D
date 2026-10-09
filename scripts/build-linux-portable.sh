#!/usr/bin/env bash
# Linux portable archive for Arch/CachyOS and other glibc Linux desktops.
# Build from the tested deb file because its Tauri resource layout has been
# smoke-tested: usr/bin/hand-d-desktop + usr/lib/Hand-D/{models,sidecar}.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "Portable tarball is currently supported only for Linux x86_64" >&2
  exit 1
fi
for dependency in ar tar zstd; do
  command -v "$dependency" >/dev/null || {
    echo "Missing build dependency: $dependency" >&2; exit 1;
  }
done
deb="$repo/desktop/src-tauri/target/release/bundle/deb/Hand-D_0.2.0_amd64.deb"
test -s "$deb" || {
  echo "Missing .deb. Run: npm --prefix desktop run build -- --bundles deb" >&2
  exit 1
}
working="$(mktemp -d)"
trap 'rm -rf "$working"' EXIT
ar p "$deb" data.tar.gz | tar -xz -C "$working"
for resource in \
  "usr/bin/hand-d-desktop" \
  "usr/lib/Hand-D/sidecar/handd-sidecar/handd-sidecar" \
  "usr/lib/Hand-D/models/hand_landmarker.task" \
  "usr/lib/Hand-D/models/gesture_mlp.pth"; do
  test -f "$working/$resource" || {
    echo "Missing expected bundled resource $resource" >&2; exit 1;
  }
done
cat > "$working/launch-handd.sh" <<'LAUNCH'
#!/usr/bin/env bash
# Launch from this extracted directory. No Python environment is required.
set -euo pipefail
bundle="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$bundle/usr/bin/hand-d-desktop" "$@"
LAUNCH
chmod +x "$working/launch-handd.sh"
mkdir -p "$repo/dist"
out="$repo/dist/Hand-D_0.2.0_linux_amd64-portable.tar.zst"
tar --sort=name -C "$working" -I 'zstd -T0 -5' -cf "$out" \
  launch-handd.sh usr/
echo "Portable archive: $out"
echo "Use: tar --zstd -xf $(basename "$out") && ./launch-handd.sh"
