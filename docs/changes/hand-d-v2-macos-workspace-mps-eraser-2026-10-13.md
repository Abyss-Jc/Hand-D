# Hand-D v2: macOS Workspace selection, Eraser compositing, MPS opt-in (2026-10-13)

**Scope:** `feature/hand-d-v2-modernization`. ONNX branch is **not merged**; CPU is still the default PyTorch device. Changes were implemented/tested on CachyOS Linux. **Physical macOS retest is outstanding.** The operator reported that the macOS native picker opened but selecting a directory appeared to do nothing, and that the eraser made Whiteboard feel slow. `/ask-matt` routed the investigation to `/diagnosing-bugs` and `/tdd`.

## Workspace picker: Rust selection and frontend reconnection

**Previously:** `pick_workspace()` both selected and validated a folder, returning an error without its selected path when it wasn't an initialized Workspace (`handd.sqlite` absent). The frontend also used the *raw picker path*, not the canonical result from `select_workspace`, as its `pendingWorkspace` comparison against the restarted sidecar. On macOS, aliases such as `/var` and `/private/var` can refer to the same location while differing as strings, potentially leaving the UI waiting on reconnection. This is a code-derived reproducible case, not confirmed as the exact hardware failure.

**Now:** `pick_workspace` returns the selected path, and the frontend explicitly invokes `select_workspace` for validation; if the directory isn't a Workspace, the selected path and a specific error are shown, with guidance to use Create Workspace for a new empty directory. On success the canonical path returned by Rust drives the input, connection message, and sidecar-restart matching. A *successful* `osascript` with no path is an error, not a silent cancellation. Cancel retains the previous selection.

**Finder metadata:** `.DS_Store` and `.localized` regular files no longer block **Create Workspace** in either Rust or Python. Files remain untouched; any real file/directory/symlink still prevents initialization, and an existing `handd.sqlite` is never overwritten. Verified Python `tests.test_workspace_init`, Rust `new_workspace_accepts_only_macos_folder_metadata_not_user_files`, and frontend tests with distinct picker and canonical returned paths. Source: `desktop/src-tauri/src/main.rs`, `native_files.rs`, `handd_core/workspace_init.py`, `desktop/web/app.js`.

## Eraser: flat chronological masks

**Previously:** every erase nested the entire earlier SVG scene inside a new `g[mask]`. Synthetic `node desktop/bench/eraser-stress.mjs` (alternating one ink stroke and one erase): 20/60/120 erasers caused respective max mask depths of **20/60/120**. A deep compositing tree can force expensive raster work in browsers, but this is not a measured WebKit profiler result.

**Now:** each ink epoch has **one flat SVG mask**. A new eraser path is referenced from earlier epoch masks using SVG `<use href="#...">`; later ink remains outside old masks. Consecutive erase strokes (without new ink between them) reuse one mask. `sync` rebuilds epoch links correctly for Undo/Redo and retains the camera, controls, and later strokes. `desktop/tests/stroke-renderer.test.mjs` verifies depth=1 after 40 alternating erasers, earlier/later chronology, transparency and mask reuse. On the same synthetic benchmark, max depth was **1/1/1** for 20/60/120; references were **190/1770/7140** respectively. Alternating ink/erase in extreme volume can grow the flat mask references quadratically; performance on macOS WebKit and real camera FPS are **not yet verified**. This is an intentional correctness/performance tradeoff to evaluate on hardware.

Eraser input also spaces updates proportional to its configured diameter (4–12 SVG pixels), instead of recording every 2-pixel-ish move; a 160px eraser no longer queues nearly-identical 5px updates. Pen/Wiggly sampling is unchanged. The cursor still follows the pointer continuously. Regression: `desktop/tests/app-ui.test.mjs` large eraser mouse strokes RED→GREEN.

Additional non-browser raster evidence: serialized an ink/erase/ink/erase/ink fixture to SVG, rendered locally using `resvg`, checked RGBA pixels using Pillow: unaffected red/blue/green ink remains opaque and erased segments are transparent `(0,0,0,0)`. This checks standard SVG rendering, **not WebKit**. `desktop/bench/eraser-stress.mjs` is a synthetic DOM/JS benchmark, not GPU paint or FPS.

## PyTorch MPS on macOS (explicit opt-in)

The *base* branch remains `cpu` by default. `HANDD_INFERENCE_DEVICE=mps` (or sidecar flag `--inference-device mps`) loads the existing CPU-validated Candidate or diagnostic legacy model, clones its MLP to `mps` when available, performs float32 logit/label parity checks, and infers on the GPU. The CPU model remains available. If MPS is absent, initialization/parity fails, or inference throws/returns invalid logits, the model uses CPU and reports `torch-cpu-mps-fallback` in runtime health; for an **in-flight GPU exception**, both hands release their actions for that frame and stabilize again before issuing draw/erase. No raw landmarks, photos, models or workspaces are written by this selection. The MPS experiment is separate from ONNX Runtime and does not affect MediaPipe.

`uv run --frozen python -m handd_core.benchmark_mps` performs an optional **no-camera** benchmark on the legacy diagnostic MLP, or `--candidate <verified-folder>` for an artifact. It outputs provider availability, per-batch CPU/MPS medians and p95, numerical error and prediction mismatches; results are model-only, not perceived latency or accuracy. On CachyOS, MPS reported `built=false, available=false` as expected. Real Metal acceleration/parity/performance on Apple Silicon are **pending**.

## TDD and verification

- RED tests: selected folder without an initialized DB must still be visible; raw vs canonical picker path mismatch; successful `osascript` with no output; Finder-only folder must initialize; 40 alternating erasers must never nest 40 masks; consecutive erasers must reuse a mask; MPS CPU default, fallback and safe-release contracts.
- GREEN on Linux after the final input-spacing and MPS diagnostics tests: `uv run --frozen --group training python -m unittest discover -s tests -p 'test_*.py' -q` **162 tests (160 passed, 2 skipped)** (MPS hardware and MediaPipe hardware-opt-in); `node --test desktop/tests/*.test.mjs` **65/65**; `cargo test --locked --manifest-path desktop/src-tauri/Cargo.toml` **8/8**; `npm --prefix desktop run check` **PASS**.
- Outstanding gates: native macOS picker, real folder creation, actual Metal MPS inference and benchmark, camera gesture latency, WebKit eraser paint/FPS, and packaged `.app` validation. The ONNX experiment failing on the user's Mac has **no captured error details yet**; do not claim that MPS resolves it.

## Physical verification and rollback

See [macOS lab runbook](../runbook/mac-lab-oct09.md) for commands, safe temporary workspace and MPS checks. For a CPU-only app run use `HANDD_INFERENCE_DEVICE=cpu npm --prefix desktop run dev`; close the previous app process and save any unsaved Whiteboard first. The MPS benchmark never needs the camera. Do not merge this into the experimental ONNX branch without separate parity/regression work.
