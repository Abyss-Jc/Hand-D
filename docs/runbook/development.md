# Development Runbook

## Precondiciones

- Work from the Hand-D repository.
- For the v2 modernization, use `feature/hand-d-v2-modernization`.
- Do not make direct v2 implementation commits on `main`.
- Use the project-managed Python environment: `uv` + `pyproject.toml` + committed `uv.lock`, with Python 3.13 (verified on CachyOS x86-64, 2026-10-08).

Current environment note:

- The CachyOS development host reported Python 3.14.7.
- `uv 0.12.23` is installed on the CachyOS host (`uv --version`, verified 2026-10-08).
- The global environment did not contain PyTorch when the v2 audit started.
- The legacy `requirements.txt` / `requirementsGPU.txt` remain historical inputs, **not** the dependency source for v2. `pyproject.toml` and `uv.lock` now define the reproducible v2 environment.

## Pasos

```text
1. git fetch origin
2. git switch feature/hand-d-v2-modernization
3. git status
4. uv --version
5. uv python install 3.13
6. uv lock --check
7. uv sync --frozen
8. uv run --frozen python -m unittest discover -s tests -p 'test_*.py' -v
```

The v2 workflow uses `uv run`/`uv sync` without requiring shell activation. Do not install project dependencies globally. `.venv/` is project-local and ignored; `.python-version`, `pyproject.toml` and `uv.lock` must be committed.

### Dependency profiles (verified locally 2026-10-08)

| Profile | Command | Scope |
|---|---|---|
| Runtime only | `uv sync --frozen --no-default-groups` | NumPy, MediaPipe 0.10.33, one OpenCV distribution (`opencv-contrib-python`), Torch CPU on Linux |
| Runtime + development (default) | `uv sync --frozen` | Runtime plus dev tools such as pytest |
| Runtime + training + development | `uv sync --frozen --group training` | Above plus pandas, scikit-learn and training/reporting dependencies |

Python **3.13.16** was installed locally by `uv python install 3.13`. `uv lock --check`, `uv sync --frozen`, `uv pip check --python .venv/bin/python`, and the baseline unit tests succeeded. The runtime-only profile passed 20 tests; the training profile imported pandas **3.0.6**, scikit-learn **1.9.1**, matplotlib **3.11.2**, and passed 20 tests. Returning to default sync removed optional training-only dependencies, confirming separation. MediaPipe pulls some plotting dependencies transitively, so the runtime profile is not guaranteed to contain zero plotting packages.

On this CachyOS x86-64 host, default Torch is **2.14.1+cpu**, MediaPipe **0.10.33**, NumPy **2.5.3**, and OpenCV **4.13.0** (`opencv-contrib-python==4.13.0.92`). `opencv-python` is not installed alongside contrib in `.venv`. Python 3.13 installation on macOS/Windows, GPU providers, camera hardware, and production sidecar packaging are **No verificado**.

To invoke training-only dependencies and then restore the default development profile:

```bash
uv run --frozen --group training python -m unittest discover -s tests -p 'test_*.py' -v
uv sync --frozen
```

The old project-local `venv/` is transitional legacy evidence, not the new `.venv/`. If dependencies intentionally change, regenerate `uv.lock` with `uv lock` and verify it before committing; regular setup must use `uv sync --frozen`.

### Transitional contract tests (verified 2026-10-08)

Before the `uv` migration, no-camera checks run using the preexisting project-local `venv`:

```bash
venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
venv/bin/python -m compileall -q handd_core visualizer_app/gesture_engine.py tests/test_feature_transform.py
```

Verified result: **10 tests passed** for shared Feature Transform v1 and existing GestureEngine checks, including a subprocess import from the legacy source-script working directory. A separate valid-input comparison with the pre-refactor canonicalizer was exact for float32/float64 and Left/Right fixtures. This does not verify `uv sync`, camera operation, or the new data pipeline.

HD-03 adds the no-camera SQLite dataset-store tests. Verified on 2026-10-08: `venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v` passed **20/20**. These tests use temporary SQLite databases and do not alter the team's canonical workspace or legacy CSV data.

### HD-04/05 collection and manual review (2026-10-08)

TDD validation executed:

```bash
uv run --frozen python -m unittest discover -s tests -p 'test_*.py' -q
uv run --frozen python -m handd_core.collect_cli --help
```

The full no-camera suite passed **35/35**. It tests 21-point observations, interval-based quota, pause/resume with persisted progress, invalid/wrong-hand exclusion, local pseudonymous device ID, latest-only LIVE_STREAM callback bridging, explicit human review, and read-only readiness. CLI help was checked via subprocess. The physical camera execution below is **No verificado** (requires camera permission and actual gesture/hand checking):

```bash
uv run --frozen python -m handd_core.collect_cli \
  --workspace /path/to/your/Hand-D-workspace \
  --participant P001 --gesture Index_Finger --hand Right \
  --camera 0 --quota 120 --interval-ms 100
```

The collector creates a fresh Collection Session and Capture, stores **only image/world landmarks and anonymous provenance** in `handd.sqlite`, and shows live preview for framing. Press `p` to pause/resume, `q` to end early, or interrupt to stop; stored Samples remain unreviewed/active. Handedness inversion currently follows the mirrored legacy-camera convention and **needs physical Left/Right smoke verification before trusted data collection**. Use anonymous participant IDs.

Inspect and curate by exact Sample ID (manual CLI, no automatic acceptance):

```bash
uv run --frozen python -m handd_core.review_cli --workspace /path/to/your/Hand-D-workspace list --status unreviewed
uv run --frozen python -m handd_core.review_cli --workspace /path/to/your/Hand-D-workspace accept SAMPLE_ID --reason reviewed
uv run --frozen python -m handd_core.review_cli --workspace /path/to/your/Hand-D-workspace drop SAMPLE_ID
uv run --frozen python -m handd_core.review_cli --workspace /path/to/your/Hand-D-workspace restore SAMPLE_ID
uv run --frozen python -m handd_core.review_cli --workspace /path/to/your/Hand-D-workspace list --include-dropped
uv run --frozen python -m handd_core.review_cli --workspace /path/to/your/Hand-D-workspace readiness
```

Manual review CLI transitions and readiness output were exercised in subprocess tests against temporary SQLite files. Snapshot Readiness **does not generate a snapshot**; immutable manifest/NPZ creation remains HD-06. Avoid running the collector simultaneously with another writer on the same workspace database.

### HD-04 physical webcam + legacy model diagnostic (verified 2026-10-08)

This bounded probe opens the local webcam and uses the **existing** `models/hand_landmarker.task` and `models/gesture_mlp.pth` (PyTorch `weights_only=True`). It runs MediaPipe **LIVE_STREAM**, the shared 69-feature transform and the existing five-output classifier. **No images, video, SQLite Samples, or Workspace data are saved.**

```bash
# Headless, 8-second finite smoke test:
uv run --frozen python -m handd_core.camera_smoke --camera 0 --seconds 8

# Optional interactive diagnostic preview; press q/Esc or let the limit expire:
uv run --frozen python -m handd_core.camera_smoke --camera 0 --seconds 30 --preview
```

Evidence observed on CachyOS laptop `/dev/video0`: 234/234 frames at 640×480 in 8.04 seconds (**29.12 capture FPS**); 234 LIVE_STREAM callbacks, 233 callback batches processed, 12 batches with a detected hand. Raw MediaPipe handedness Left x12, model output Idle x12, zero invalid feature vectors and zero inference errors. A second **executed** visual-preview run at 6 seconds returned 175/175 frames (**28.93 capture FPS**), 175 callbacks, 74 hand-containing callback batches, raw label Left x74 and old-model predictions Idle x32, Index_Finger x29, Fist x13; zero classifier errors. The preview window opened and exited successfully; OpenCV emitted a Qt/Wayland plugin warning, so Wayland-native Qt provider behavior is not verified. The GPU-related EGL initialization log is not proof of GPU-accelerated ML inference; legacy MLP used Torch CPU.

Interpretation: the camera, MediaPipe task, Feature Transform and legacy MLP work together. The checkpoint is a 69→128→64→5 `state_dict`, **without an independently verified label manifest**; predictions are diagnostic rather than demonstrated classification accuracy. Neither physical-hand Left/Right mapping nor the persistent real-image/world landmark Capture through `collect_cli` has been verified. Before collecting trusted real participant data, use the live preview to show a known physical Left hand and then a known Right hand, compare `MP=...` and `mano estimada=...`, and correct the mirror convention if necessary. Never silently label uncertain samples.

`uv run --frozen python -m unittest discover -s tests -p 'test_*.py' -q` now passes **38/38** (includes 3 no-camera legacy-model smoke-contract tests).

### HD-04 guided physical-hand mapping + temporary real SQLite Capture (verified 2026-10-08)

The live calibration GUI requests **PHYSICAL RIGHT**, then **PHYSICAL LEFT** while processing only raw MediaPipe hand labels. It does not save any frames, video, landmarks, samples or identifiers. Use only one clearly visible hand at a time:

```bash
uv run --frozen python -m handd_core.hand_mapping --camera 0 --hold-seconds 8
```

Executed on the laptop webcam with mirrored frames: **866 frames and 866 MediaPipe callbacks**. During the on-screen RIGHT phase, raw MP Left=238 and Right=0 (100%); during the LEFT phase, raw MP Right=188 and Left=30 (86.2%); 18 ambiguous/empty batches in the Left phase were ignored. Aggregate result: `mapping=inverted`, supporting the existing `CaptureSampler` raw-to-physical inversion (`Left` → physical Right, `Right` → physical Left). This is **conditional on the human holding the indicated physical hand**; it cannot independently identify the operator's hand identity, so ask the operator to verify they followed both prompts before treating the mapping as final ground truth. No OpenCV Qt Wayland-native support claim is made.

A separate real `collect_cli` smoke used an **ephemeral temporary workspace** with deliberately non-training gesture label `CameraSmoke_DoNotTrain`, `--hand Any`, `--quota 25`, `--interval-ms 120`, `--max-seconds 12`. The first run saved 0 Samples; the instrumented second run recorded **355 callbacks, 3 hand-containing callbacks, and 1 actual Sample** with raw MediaPipe Right and 21×3 image/world landmarks, persisting `review=unreviewed`, `lifecycle=active` in canonical SQLite. The temporary SQLite/workspace was inspected and then deleted by a shell trap; **no real training workspace changed and no video/photos were written**. This confirms end-to-end camera→LIVE_STREAM→Sampler→SQLite plumbing, but not trained gesture labeling or detection recall.

To repeat a bounded collection smoke in a disposable directory, create a temporary workspace outside the canonical project (avoid using the real Project Workspace), pass `--max-seconds 12`, and inspect/delete that temporary SQLite afterwards. Do not accept or train on samples labeled `CameraSmoke_DoNotTrain`. `LatestCaptureResults.statistics()` reports `registered_callbacks`, `callbacks_with_hands`, `samples_saved`, and `superseded_callbacks` to distinguish absent detections from persistence failures.

TDD evidence: RED→GREEN for `tests.test_hand_mapping` and callback diagnostics; `uv run --frozen python -m unittest discover -s tests -p 'test_*.py' -q` passed **62/62**, plus compiled new code. Model quality, P003 final-test accuracy and Studio integration remain **No verificado**.

### HD-06 immutable Development Snapshot (verified 2026-10-08)

The Snapshot Builder reads **accepted + active** rows from `handd.sqlite`, restricted to P001/P002 Development Sessions. It never includes P003 in a Development Snapshot, even if that participant's Session is explicitly listed. It creates a new `snapshots/dev-*/` directory containing `manifest.json` and `dataset.npz` with exact model-ready 69-feature inputs, Sample/Participant/Session IDs, fixed label order, session holdout folds and SHA256 integrity checks. Existing Snapshot versions are never overwritten; revising human Review creates a new Snapshot. Creation does **not** start training.

```bash
# Inspect blockers and coverage warnings before building:
uv run --frozen python -m handd_core.review_cli --workspace /path/to/Hand-D-workspace readiness

# Freeze current reviewed Development membership (no training is launched):
uv run --frozen python -m handd_core.snapshot_cli --workspace /path/to/Hand-D-workspace build --note 'development baseline'

# Optional: freeze legacy data independently alongside the v2 materialization:
uv run --frozen python -m handd_core.snapshot_cli --workspace /path/to/Hand-D-workspace build --legacy-csv datasets/gesture_dataset.csv

# Verify, replacing the ID with the generated snapshot folder name:
uv run --frozen python -m handd_core.snapshot_cli --workspace /path/to/Hand-D-workspace verify dev-SNAPSHOT_ID
```

Optional legacy CSV must exactly contain columns `feat_0` through `feat_68`, `handedness`, `label`, finite features, handedness encoded as 0/1, and compatible five-class labels. The `legacy.npz` partition is frozen with its own hash and tagged `legacy_unverified_no_session_groups`; **do not** use legacy rows as unseen-participant validation evidence. Without legacy input, the snapshot contains only reviewed v2 data. A one-Session Snapshot is allowed with coverage warnings; the saved `validation_folds` is empty until two or more Development Sessions exist, so grouped cross-validation cannot be claimed yet.

Evidence: RED→GREEN tests for membership, excluded unreviewed/rejected/dropped rows, no P003 leakage, protected prior versions, deterministic materialization, tampered content/manifest, legacy compatibility, and explicit build/verify CLI. `uv run --frozen python -m unittest discover -s tests -p 'test_*.py' -q` passed **49/49**, with no real camera required. A separate temporary-workspace smoke successfully froze all **5,400** rows of the repository's historical `datasets/gesture_dataset.csv` alongside one synthetic accepted v2 Sample and passed `verify_snapshot`, without mutating the dataset or workspaces. This does NOT prove real camera collection, training performance, sealed P003 evaluation, or cross-platform packaging.

### HD-06L legacy CSV migration into SQLite (2026-10-08)

The existing CSV files store **69 derived model-input features**, a binary handedness code and a label; they do NOT store original 21×3 image/world landmarks, Participant IDs, or Collection Sessions. Import never invents these missing facts. SQLite schema **v2** creates separate immutable legacy tables and upgrades a v1 database additively without modifying canonical Samples. Imports are atomic and idempotent using SHA-256 of the original CSV bytes.

```bash
# Run from the repository root, targeting your intentional workspace:
uv run --frozen python -m handd_core.legacy_cli --workspace /path/to/Hand-D-workspace import datasets/gesture_dataset.csv
uv run --frozen python -m handd_core.legacy_cli --workspace /path/to/Hand-D-workspace import datasets/gesture_dataset_old.csv
uv run --frozen python -m handd_core.legacy_cli --workspace /path/to/Hand-D-workspace list

# Pass SOURCE_ID from the preceding list command when freezing a NEW Snapshot.
# A Development Snapshot still requires at least one accepted P001/P002 v2 Sample.
uv run --frozen python -m handd_core.snapshot_cli --workspace /path/to/Hand-D-workspace build --legacy-source SOURCE_ID
```

The first CSV has **5,400** rows and labels compatible with the current five-class catalog. The older CSV has **4,336** rows and an old label `Ruler_Gesture`, so the importer preserves it as **quarantined**, not automatically renamed to `Ruler` or silently inserted into training. These represent 9,736 historical records, **not necessarily 9,736 independent observations**; overlap between CSVs or with future v2 Samples is unknown. Compatible imports can contribute only to the separately flagged legacy TRAIN partition when selected. They are never P001/P002 heldout validation or sealed P003 test membership. Existing immutable Snapshots remain unchanged; build a new one for updated selection.

Verified with TDD and an ephemeral SQLite database: both complete CSVs imported, re-import no-ops, quarantine enforced, 9,736 legacy rows plus zero canonical raw Samples. Tests cover malformed-input rollback, source/row immutability, schema-v1 preservation and snapshot creation **after the original CSV is removed**. No real workspace was modified and no evaluation claim was made.

### HD-07 grouped Development training and Model Artifact (verified 2026-10-08)

Training requires a frozen Development Snapshot with **at least two independent Collection Sessions**. It never queries current SQLite, trains on a P003/final-test Snapshot, or activates the candidate automatically. On CachyOS Linux CPU:

```bash
# Default: Development session-held-out CV, 25/50/75/100% learning curves,
# 20 epochs, seed 42; explicit final refit without legacy augmentation.
uv run --frozen python -m handd_core.train_cli --workspace /path/to/Hand-D-workspace --snapshot dev-SNAPSHOT_ID

# Quick local plumbing check; still evaluates only frozen development sessions:
uv run --frozen python -m handd_core.train_cli --workspace /path/to/Hand-D-workspace --snapshot dev-SNAPSHOT_ID --epochs 1 --fractions 1.0

# Only when the snapshot contains a frozen legacy.npz, and augmentation
# has been explicitly selected for final refit:
uv run --frozen python -m handd_core.train_cli --workspace /path/to/Hand-D-workspace --snapshot dev-SNAPSHOT_ID --final-legacy
```

Outputs are immutable-by-creation `reports/experiment-*/` (fold checkpoint weights and `metrics.json`) and `models/candidate-*/` (`weights.pth`, versioned `manifest.json`, copy of `metrics.json`). The report contains each fold's non-overlapping train/validation Session IDs and Sample IDs, full OOF predictions/uncalibrated scores, Macro F1, per-class precision/recall/F1, confusion matrix and learning-curve points. If the snapshot has legacy data, the trainer runs both **without_legacy** and **with_legacy** over the same Development heldout sessions; **legacy rows never become the heldout evaluation set**. Old legacy provenance has no reliable Collection-Session groups, so its trial may still contain unknowable historical overlap. Treat it as exploratory, not a certified leakage-free generalization result.

The final refit uses **all eligible Development Samples**, plus legacy only if `--final-legacy` was explicitly given. A Candidate is not automatically selected Active; its loader validates architecture, 69-feature contract, label order, weight checksum and metrics checksum. A true unseen-participant P003 evaluation remains separate and **No verificado**.

Executed evidence: TDD module/CLI RED before GREEN, full `uv run --frozen python -m unittest discover -s tests -p 'test_*.py' -q` **56/56 passing**, and ephemeral smoke using **20 synthetic v2 Samples in 2 Sessions** plus the full **5,400-row historical legacy partition**. One-epoch paired training created both OOF variants and a compatible loadable five-class final-refit Candidate. This verifies plumbing only; no real-data classification accuracy, test P003 evidence, deployment FPS or cross-platform performance has been measured for the new Candidate.

### HD-08 v2 runtime with real webcam and Model Artifact (2026-10-08)

The new Python runtime explicitly validates and loads a versioned final-refit Candidate with its declared 69-feature transform and label order; **a bare legacy `.pth` is not a valid Model Artifact**. LIVE_STREAM callbacks enter a latest-only mailbox; the owner thread produces normalized index-tip positions, fixed Drawing/Modifier Hand roles, stable gestures/actions, camera/model health and Runtime Session sequence-enveloped updates. Current stabilization defaults are **3 consecutive predictions over at least 60 ms** (prototype values to tune with real gesture data). A newer READY snapshot resets the event consumer gate, so old-session gesture events cannot be replayed. Canvas strokes, undo and smoothing do not belong to this Python runtime.

```bash
# Run a bounded headless no-recording probe with a verified Candidate:
uv run --frozen python -m handd_core.runtime_cli --model-artifact /path/to/workspace/models/candidate-ID --camera 0 --seconds 8

# Optional framing/gesture preview without image/video capture to disk:
uv run --frozen python -m handd_core.runtime_cli --model-artifact /path/to/workspace/models/candidate-ID --camera 0 --seconds 20 --preview

# All no-camera tests, including latest callback/role/stale-event/model checks:
uv run --frozen python -m unittest discover -s tests -p 'test_*.py' -q
```

Executed evidence on CachyOS webcam `/dev/video0`: (1) with a disposable five-output **diagnostic** Artifact, 175 video frames (~28.58 FPS), 175 MediaPipe callbacks, 173 processed callback batches and ~**0.871 ms p95** for the post-landmarker Python computation window, with no recorded frame/sample. All its predicted Fist labels were intentionally forced and **are not gesture accuracy**. (2) a full disposable **SQLite synthetic P001 sample → Snapshot → one-epoch HD-07 Candidate → HD-08 camera runtime** exercise: 20 synthetic v2 Samples across two sessions, 175 frames (~29 FPS), 175 callbacks, 174 processed batches, 4 hand predictions, zero stabilized tool actions; the hand observations were too sparse to satisfy the safety debounce. Both disposable workspaces and models were deleted.

`uv run --frozen python -m unittest discover -s tests -p 'test_*.py' -q` passed **87/87**; `uv lock --check`, Python compileall and git diff checks passed. The local p95 excludes sensor/landmarker latency, WebSocket, IPC and frontend; **do not claim full end-to-end p95, stable 30Hz recognized-hand cadence, real-model accuracy, production Tauri sidecar, MJPEG or cross-platform validation** based on this smoke.

### Canonical dataset collaboration

For the October milestone, `handd.sqlite` is edited sequentially rather than concurrently:

```text
1. git pull / update the modernization branch
2. confirm no teammate is currently editing the canonical dataset
3. collect / curate using the current handd.sqlite
4. close Studio so database writes are finished
5. commit the dataset change
6. push the branch
7. release dataset editing ownership
```

Git is not expected to merge two independently modified SQLite files. If two contributors need concurrent collection later, add an explicit session import/export workflow rather than relying on binary merges.

## Comprobar

Verified on 2026-10-05:

```text
git branch --show-current
→ feature/hand-d-v2-modernization

git rev-list --left-right --count origin/main...HEAD
→ branch was created from the current origin/main baseline before v2 commits
```

Verified for v2 on CachyOS Linux x86-64 (2026-10-08):

- Python 3.13.16 provisioning and project-local `uv sync --frozen` from `uv.lock`;
- `uv lock --check`, `uv pip check`, and installed dependency versions;
- default, runtime-only, and training dependency profiles; **20/20** no-camera tests in each;
- exactly one installed OpenCV distribution in the v2 `.venv`.

Still **No verificado**:

- reproducibility on a different clean machine and supported non-Linux platforms;
- production-grade Studio collection, real-gesture-trained v2 accuracy/cadence, and Tauri HTTP/WS/MJPEG integration; the Linux camera + legacy classifier, one temporary SQLite Capture, and synthetic-trained HD-08 Python live inference were verified separately above;
- Apple Silicon/MPS execution;
- CUDA/ROCm/XPU execution;
- production desktop-shell packaging.

These items must remain marked `No verificado` until commands/tests are actually run.

### HD-09 Tauri / Python sidecar development spike (verified on CachyOS, 2026-10-08)

This is the **real Linux desktop process**, not the UX prototype HTML. WebKit2GTK 4.1, GTK3, Rust, Node and npm are present. Polkit authorized installation of missing system dependencies. Python dependencies are locked in uv including aiohttp, npm dependencies are pinned in desktop/package-lock.json, and Cargo uses desktop/src-tauri/Cargo.lock.

```bash
# From the Hand-D repository root:
uv sync --frozen
npm ci --prefix desktop
cargo check --manifest-path desktop/src-tauri/Cargo.toml --locked
node --test desktop/tests/*.test.mjs
uv run --frozen python -m unittest discover -s tests -p 'test_*.py' -q

# Open the actual native window:
npm --prefix desktop run dev
```

The native window is titled **Hand-D — Whiteboard + Studio**. Rust supervises the local `.venv/bin/python -u -m handd_core.sidecar_main` (development only). Python prints one bootstrap JSON line to Rust (random loopback port, per-launch capability token, Runtime Session ID) and serves authenticated `GET /health`, `GET /ws` and `GET /mjpeg` bound exclusively to `127.0.0.1`. It uses MediaPipe LIVE_STREAM and the mirrored camera. The preview image and WebSocket include the bootstrap token in their loopback URL because the browser's `<img>` element cannot add Bearer Authorization headers. HTTP access logs are disabled, origins are restricted, no-cache/referrer headers are set, and neither tokens nor image data should be persisted or logged. Packaged Python binaries are **not** implemented.

Whiteboard is the default screen, usable with the mouse without a working camera. It supports drawing, erasing, undo/redo, clear, and a modal enlarged canvas which reuses the same DOM/SVG instead of discarding history. Studio currently exposes an **accurate overview and runtime diagnostics, not yet interactive Collect/Review/Snapshot tools**. The runtime starts without an active v2 Candidate, so unsupported gestures cannot become live tool actions by accident.

Evidence: full **93/93** Python tests (including 6 IPC/subprocess tests), **4/4** Node tests, `uv lock --check`, `cargo check --locked` and a full native `npm --prefix desktop run dev` compilation/launch passed. Niri confirmed the Hand-D desktop window and Rust PID. WebKit opened actual loopback connections to the Python sidecar, and the Python process owned the webcam. After deliberately sending SIGTERM to the supervised Python child, Rust launched a fresh child with a new ephemeral port while the original Tauri window stayed open and WebKit reconnected. Canvas/history preservation on reconnection is unit-tested rather than visually recorded. A real gesture-driven drawing demo, full Studio controls, packaged distribution, explicit on-screen crash/close teardown and end-to-end latency remain **No verificado**.

If preview fails, inspect Studio diagnostics and camera permissions. Check Linux dependency presence using `pkg-config --modversion webkit2gtk-4.1`. Never commit `desktop/node_modules` or `desktop/src-tauri/target`; neither belongs to source control.

### HD-09 camera preview visible, gestures absent, or sidecar repeatedly reconnecting (2026-10-08)

Two development defects were diagnosed on the actual Tauri/Niri shell after the first native smoke. **Cause 1:** the Rust launcher started Python without any gesture model. MediaPipe could track hands but every prediction/action remained null. **Cause 2:** aiohttp accepted WebSocket Origins for Tauri's assumed development port 1420, while the live Tauri instance used port **1430**; camera MJPEG could work even though WebSocket handshakes returned HTTP 403. These were independent failures, not evidence the webcam itself was broken.

TDD reproduced the 1430 WebSocket handshake rejection and missing explicit legacy-checkpoint activation; fixed aiohttp Origin verification to accept only a syntactically valid local development Origin on 127.0.0.1/localhost with a valid dynamic port, while continuing to reject arbitrary/foreign origins and require the per-launch token. The native **development launcher now explicitly opts into** `models/gesture_mlp.pth` using `--legacy-checkpoint`. Python loads the five-output state dict with `weights_only=True` and a fixed 69-feature CPU predictor; its health reports `legacy_unverified` and its ID includes an abbreviated checkpoint SHA256. The five legacy class names/order are only the historical best-known assumption, **not independently certified metadata, a v2 Model Artifact, P003 evidence, or a scientifically validated accuracy result**. For a validated v2 model, use the distinct `--model-artifact` option after generating and verifying a final-refit Candidate; both options cannot be given together.

The Whiteboard now reports separate model and camera statuses. `MODELO ANTIGUO ACTIVO` indicates diagnostic legacy inference, `MODELO NO CARGADO` indicates tracking-only mode, and `SIN MANO DETECTADA` means MediaPipe did not yield a usable hand. Rust supervision and MJPEG still operate independently of WebSocket recovery. On the live Linux desktop, Niri confirmed the window, the supervised Python argv included the explicit legacy flag, that process held `/dev/video0`, and WebKit established **two stable loopback connections** (preview and WS) to the same sidecar across repeated socket checks. User-observed classification reliability and drawing gestures on a real hand still require an in-person interaction; don't infer accuracy from a healthy transport.

```bash
# Native development window with explicit legacy diagnostic inference:
npm --prefix desktop run dev

# No-camera regressions and all project tests:
uv run --frozen python -m unittest tests.test_sidecar_ipc tests.test_sidecar_process tests.test_runtime_v2 -q
node --test desktop/tests/*.test.mjs
uv run --frozen python -m unittest discover -s tests -p 'test_*.py' -q
```

### Whiteboard fluidez: renderer incremental + preview pacing (2026-10-08)

Este corte optimiza la fluidez **sin cambiar Feature Transform, gesto, dataset, ni guardar video**. Se corrigieron dos cuellos de botella medidos: reconstruir el SVG completo por punto y entregar solo uno de cada dos frames MJPEG.

```bash
# Mismo benchmark sintético en Node antes/después, 121 trazos, 1410 puntos:
node desktop/tests/benchmark-strokes.mjs

# Probe de transporte con webcam durante 8 segundos.
# Abre un sidecar temporal con el clasificador legacy solo para diagnóstico;
# no imprime tokens, no decodifica ni guarda los JPEG:
uv run --frozen python scripts/bench_sidecar_preview.py

# Comprobaciones automatizadas:
node --test desktop/tests/*.test.mjs
uv run --frozen python -m unittest discover -s tests -p 'test_*.py' -q
```

**Datos obtenidos:** el probe DOM simulado pasó de **56.265 SVG paths creados / 705 vaciados / 74,99 ms** a **121 paths / 0 vaciados / 7,53 ms**. `desktop/web/stroke-renderer.mjs` conserva nodos y añade segmentos solo al trazo activo, agrupando invalidaciones con `requestAnimationFrame`; tests comprueban undo/redo, erase y callbacks pendientes. **Node mock no mide WebKit real.** El probe sidecar con webcam y un cliente HTTP+WebSocket real pasó de **14,74 a 29,22 FPS MJPEG entregados**; WS pasó de **26,85 a 27,85 actualizaciones/s** en dos corridas de 8 s, con distinto número de manos detectadas por cambios de escena. `PreviewPacer` limita a aproximadamente 30 Hz, sin encoder JPEG cuando no hay espectadores del stream, y `SidecarServer.preview_subscribers` se limpia al desconectar el cliente. Es una medida de **entrega local de MJPEG**, no FPS de render en WebKit ni p95 cámara→pantalla.

Una prueba adicional de cámara legacy de 16 s con preview reportó **29,47 FPS de lectura**, 467 callbacks y 417 callback batches con manos, 0 errores de inferencia. El clasificador antiguo mantiene etiquetas **no validadas independientemente**. La app Tauri optimizada arrancó en Niri y WebKit abrió sockets HTTP al sidecar supervisado con webcam. **102/102 tests Python y 8/8 Node pasaron** además de `uv lock --check`, compileall, sintaxis JS y diff check. Esto no verifica todavía Camera-first con overlay de 21 puntos, Wiggly, Export ni mejora subjetiva con gestos reales.

**Prueba manual solicitada al usuario:** en la app abierta, dibujar un trazo rápido y largo con ratón; comprobar Undo/Redo. Después mostrar manos derecha/izquierda y distinguir claramente si se siente entrecortado el **video**, el **cursor** o el **trazo**. Si persiste jank, capturar telemetría de WebKit y timestamp de IPC antes de otra modificación. Mantener el cambio ajeno de UX fuera de este ticket.

### HD-09 Camera-first + landmarks + Wiggly y preparación de Macs (2026-10-08)

El Whiteboard tiene ahora un único plano de imagen (video MJPEG, esqueleto, SVG y puntero) centrado sin distorsión de aspect ratio. Los controles Sobre cámara / Lienzo limpio conservan trazos y undo, y al ocultar el video desconectan el consumidor MJPEG mientras LIVE_STREAM/WS continúan. Mostrar/Ocultar manos afecta únicamente al overlay. Runtime v2 añade `handd.v2.image21.xy.1` con 21 puntos x/y normalizados por rol y limpia datos al desaparecer la detección; no se guardan en SQLite. Los colores son lima Drawing, azul Modifier; en pérdida de Drawing Hand continúa el overlay de Modifier Hand. Los 21 puntos tienen 20 conexiones del trazado ligero.

El pincel Wiggly original, opt-in, hace oscilar la representación de los trazos azules a ~12 Hz, con como máximo 320 puntos muestreados para la animación de cada trazo; no muta la geometría editable ni cambia la precisión del clasificador. Respeta prefers-reduced-motion y pausa el movimiento al ocultar el documento. Sin sonidos, GIF ni exportación de cámara. Export limpio/Save/Open siguen un slice futuro.

**TDD y ejecución comprobada en Linux:** 104 tests Python + 15 tests JavaScript, benchmark SVG sintético y `cargo check --locked`. Un sidecar temporal con webcam publicó durante ~7 s **203 updates por WS con manos**, **406 roles con arrays válidos de 21 landmarks**, dos manos presentes en 203 updates, sin grabar imágenes ni video. Tauri arrancó en Niri, Rust lanzó Python con el checkpoint legacy etiquetado como no validado y WebKit conectó al puerto localhost de ese sidecar. No se midió p95 de alineación cámara↔puntos en pantalla ni accuracy de gestos. El usuario debe revisar visualmente Cámara/Limpio, overlay, modal y Wiggly.

**Prueba en Mac Apple Silicon:** [guía macOS, checklist y comandos](mac-lab-oct09.md). El preflight `bash scripts/mac-preflight.sh` no abre cámara; `desktop/src-tauri/Info.plist` declara un motivo de acceso a cámara. Preparación del desarrollo **no implica compilación, sandbox/permissions ni webcam ya verificados en macOS**. El empaquetado independiente de la app no existe aún.

### HD-09 hotfix: borrador transparente y botones siempre visibles (2026-10-08)

**Incidente real del usuario:** el borrador anterior creaba un `path` de 32 unidades con `stroke="#fffef9"` sobre el mismo SVG de dibujo, tapando el video en Camera mode; además el SVG tenía `z-index:2` mientras los botones inferiores carecían de una capa explícita, por lo que los trazos podían taparlos. Borrar **solo tinta** requiere composición alfa, nunca pintar color del fondo.

**TDD:** `desktop/tests/stroke-renderer.test.mjs` falló en RED al exigir que el borrador fuese un trazo negro dentro de un `<mask>` SVG de luminancia, que los trazos posteriores estuviesen fuera de esa máscara y que Undo/Redo reconstruyese la estructura sin perder identidad de los trazos anteriores. GREEN: el renderer crea una máscara blanca con trayectoria negra para recortar únicamente el dibujo previo; las capas resultantes son anidadas cronológicamente para permitir volver a pintar encima. Las herramientas y la etiqueta del lienzo tienen `z-index:6` frente a la escena de cámara `z-index:0`, y tests estáticos protegen la prioridad visual. Un test de la integración DOM verifica click de borrador, Undo y Redo.

**Raster real (librsvg local, sin webcam ni guardar foto):** fondo rojo sintético y trazo negro horizontal con borrado central: píxel sobre tinta = `(26,29,28,255)`, píxel de la zona borrada = `(255,0,0,255)`, es decir el rojo original, no blanco. **104/104 Python, 18/18 JavaScript**, `cargo check --locked`, `uv lock --check` y `git diff --check` pasaron. No se modificó cámara, inferencia, SQLite ni imágenes de usuario. **Pendiente prueba humana en WebKit/Tauri real y en Macs**, incluyendo que los botones inferiores puedan pulsarse después de borrar.

### HD-09 English-only UI, stronger Wiggly and real Studio Review/Snapshot

The user explicitly approved **English-only product UI** (ADR 0005, V2-142). The actual Tauri HTML, accessible names, runtime states, Camera/Whiteboard controls, Studio placeholders and modal copy are now all English. Internal docs can remain Spanish. `node --test desktop/tests/ui-language.test.mjs` is the regression contract. Wiggly is opt-in, visual-only, with bounded **8px horizontal / 7px vertical** displacement (previously ~2px), a **65ms repaint timer**, untouched original path geometry, undo/redo and reduced-motion fallback.

**The next HD-09 thin Studio slice is no longer a simulated Dataset overview.** In the Studio workspace field, enter the path to an **existing** Hand-D Project Workspace containing `handd.sqlite` and choose **Open workspace**. This deliberately does not create a database. Rust validates and canonicalizes the directory, restarts the single Python sidecar with a `--workspace` argument, and keeps the Whiteboard document intact. The authenticated existing WebSocket accepts a narrow allowlist of `studio.request` messages only: `overview` (up to 40 real sample summaries + counts + snapshot blockers), `accept` / `reject` / `drop` / `restore` (audited single-Sample manual decisions) and `snapshot` (new immutable Development snapshot, **only on explicit user click**). Invalid Sample IDs, unconfigured workspaces and unsupported commands fail safely; P003 does not appear eligible for Development. Camera frames/landmarks are not saved by this UI.

To prepare data without turning Studio into a fake collector:

```bash
# Existing canonical collection CLI, run with appropriate real labels and participant:
uv run --frozen python -m handd_core.collect_cli --workspace /path/to/project --participant P001 --gesture Index_Finger --hand Right --camera 0 --quota 120
# Return to Studio, open /path/to/project, review the actual Samples, then
# build a Development Snapshot deliberately (no training will auto-start).
```

### HD-09 Studio Collect — same camera, explicit start, no automatic review

New development-only Studio controls (require **an existing selected workspace** with handd.sqlite):

1. Open Studio, type the local workspace directory and choose **Open workspace**. Whiteboard stays usable independently.
2. Keep the camera active. Under **Collect**, choose Participant **P001/P002**, intended gesture, physical hand (Right/Left/Either), target (1–10000, default 120) and sampling interval (20–10000ms, default 100).
3. Click **Start Capture**. This explicitly opens a fresh Session/Capture in that workspace and consumes **only the same latest MediaPipe LIVE_STREAM observation** already processed for Whiteboard by `LatestRuntimeResults.drain_to(on_observation=...)`. No second OpenCV device or detector is created; SQLite writes occur on the event-loop owner thread, not the MediaPipe callback thread.
4. **Pause / Resume / Finish** preserve all previously recorded canonical Samples. A completed quota automatically stops collection; Review lists them as **unreviewed**, and no Snapshot or training starts automatically. Missing camera, P003 or invalid hand/gesture/quota is rejected before any capture starts. Bad tracking observations are skipped. Camera loss and any storage problem must not cause partial video capture (no video is ever persisted).
5. Capture state and count appear in the Studio Collect card. If the sidecar restarts, unreviewed samples already committed to SQLite remain durable; any in-memory active collection is stopped. Start a new capture to resume experimentation.

**Evidence:** tests/test_studio_collection.py validates explicit creation and persisted 21x3 world+image data, mirrored handedness, pause/resume, quota auto-finish, no P003, invalid data; tests/test_studio_transport.py validates authenticated commands and no-camera rejection; desktop/tests/app-ui.test.mjs covers actual UI request messages and statuses. This is an **automated synthetic landmark test**, not a verified real-human camera collection from Tauri. The macOS runbook includes a separate hardware smoke for this.

**Still TODO for HD-09:** native folder picker/Create Workspace UX, richer guided Collection and P003 Final Test workflow, deliberate v2 Active Model selection, native Save/Export, packaging independent of project-local Python, and polished/validated macOS. HD-10 real-data model-performance claims remain TODO.

### HD-09 optional brush — Wiggly three-frame line-boil (2026-10-08)

User-requested repair of the visually rejected sine-wave Wiggly prototype. `desktop/web/line-boil.mjs` (new original Hand-D implementation) derives **three deterministic display-only variants** from the canonical path: 9-unit arc-length display sampling with sharp corners preserved and a **1200-vertex maximum per variant**, seeded smooth perpendicular noise (bounded to 3.3 SVG units) and slightly different stroke widths. This makes the appearance independent of MediaPipe callback frequency without touching editable points. `StrokeRenderer` preserves one canonical path plus three frame paths inside one Wiggly group and regenerates geometry **only on actual stroke-point edits**, using its existing RAF batching. The 12 FPS (`1000/12ms`) display clock changes a **single data attribute** of the root SVG; CSS handles A→B→C, and reduced-motion/hidden Studio mode reverts to the original static path. Pen and ink-only eraser semantics are unchanged.

**TDD and test evidence:** `node --test desktop/tests/*.test.mjs` verifies deterministic output, callback sampling independence, bounded long strokes, frame-only switching with zero SVG path updates, chronological erase masks, undo/redo, static reduced motion and regular Pen. `node desktop/tests/benchmark-line-boil.mjs` created 120 synthetic strokes (112 Wiggly), then toggled the 3-frame clock 360 times with **zero path `d` changes**; Node wall time for the 360 switches ~0.66ms (not browser FPS). Additional synthetic raster test with `rsvg-convert` rendered three visibly different line frames and confirmed that a black erase mask reveals the red background rather than white/black ink; no camera frames or images were saved.

**Not yet verified:** human judgment of appearance in WebKit with camera, real full-stack rendered FPS/p95, macOS hardware. A fresh native Tauri dev session needs the physical visual checklist from [Mac lab](mac-lab-oct09.md). Research and further tuning criteria: [Wiggly line-boil evidence](../changes/hand-d-v2-wiggly-line-boil-research.md).

### HD-09 finish: Active Model, native file operations and Linux portable sidecar (2026-10-09)

**Studio Models (V2-145).** With a canonical existing workspace containing `handd.sqlite`, manually run the immutable training CLI against a curated Development Snapshot (`uv run --frozen python -m handd_core.train_cli --workspace /path/to/project --snapshot dev-SNAPSHOT_ID`). Training writes `models/candidate-*` but **never activates** the candidate. In Studio, use **Model Artifacts → Refresh Models**, select a **compatible** candidate, and explicitly click **Make Active**. Model discovery only accepts immediate workspace-local Candidate directories and regular model files (no symlinks), checks version/role/feature contract, label order, manifest and weights/metrics SHA256. Failed activation preserves the running predictor and the persisted selection; a successful activation writes an atomic selection file (`models/.active-model.json`) and resets runtime temporal state. Sidecar process restarts restore this selection; the Whiteboard reports active artifact ID and model health. The development-only historical checkpoint, if loaded, always reports **legacy_unverified** with uncertified label order.

**Native workspace and Whiteboard files (V2-146).** Studio **Browse…** selects an existing workspace through a native dialog, while **Create workspace…** initializes `handd.sqlite` **only in a selected empty existing folder**, never overwriting user files. Whiteboard **Save drawing…** writes a versioned editable `.handd.json` (canonical pen, Wiggly and Eraser points only); **Open drawing…** validates before replacing live ink, with a user confirmation; **Export SVG…** writes clean static, transparent artwork with chronological erase masks. No frames, MJPEG URLs, camera preview or hand skeleton are saved in these files. Cancel is a no-op and Rust writes are atomic with an 8MB limit. Saving Wiggly exports original static path geometry, not a rasterized frame or captured video.

**Build Linux distribution (V2-147)** from the clone with local `uv` and Tauri dependencies installed:

```bash
bash scripts/build-sidecar-bundle.sh
npm --prefix desktop run build -- --bundles deb
# Optional: a portable Linux x86_64 archive usable on CachyOS/Arch
# without dpkg or a system-wide install:
bash scripts/build-linux-portable.sh
# Generated, not tracked in Git:
ls -lh desktop/src-tauri/target/release/bundle/deb/Hand-D_0.2.0_amd64.deb
ls -lh dist/Hand-D_0.2.0_linux_amd64-portable.tar.zst
```

For a portable launch, move the `.tar.zst` to a writable destination and run:

```bash
mkdir Hand-D-portable && cd Hand-D-portable
tar --zstd -xf /path/to/Hand-D_0.2.0_linux_amd64-portable.tar.zst
./launch-handd.sh
```

This keeps the tested `usr/bin/hand-d-desktop` and `usr/lib/Hand-D/...` resource layout together; it is **not** a sandboxed Flatpak/AppImage and still requires compatible Linux system WebKitGTK/GTK/C-runtime libraries. It does not need Python, uv, a project checkout, dpkg or npm at launch.

The freezer uses PyInstaller **onedir**, bundling canonical Python, Torch CPU, OpenCV and MediaPipe; Tauri packages that directory and the HandLandmarker task plus clearly **unverified legacy diagnostic** .pth. The development launch continues to use the local project `.venv`, while packaged launches take the bundled sidecar from the platform's Tauri resource directory. Do not copy Linux artifacts into Mac/Windows builds. The frozen folder occupies approximately **1.5 GiB uncompressed**, and the October 9 Linux `.deb` is about **485 MiB**. These are local ignored build outputs, not checked-in binaries.

**Linux process/package evidence (2026-10-09):** built the `.deb`, inspected `data.tar.gz` in the archive and found `usr/bin/hand-d-desktop`, `usr/lib/Hand-D/sidecar/handd-sidecar/handd-sidecar`, `usr/lib/Hand-D/models/hand_landmarker.task` and `gesture_mlp.pth`. Extracted the package to a temporary directory (without system-wide installation), launched that native desktop executable from `/tmp` outside the clone: the Tauri window appeared under Niri, WebKit connected via authenticated localhost HTTP/WS, the child process used the **frozen** sidecar path and accessed `/dev/video0`. The frozen binary also passed `--init-workspace` and a no-camera authenticated health endpoint from `/tmp` with an invalid `PYTHONPATH`; a persisted verified v2 Candidate restored with `active_model_id=candidate-frozen-proof` and `health.model=ready`.

**Physical capture evidence:** with a fresh temporary workspace and frozen sidecar, an authenticated Studio WebSocket `collect_start` request (P001 / `Lab_Diagnostic_Unverified` / Any / target=5) reached **complete 5/5** from actual MediaPipe camera callbacks, and SQLite recorded five unreviewed Samples. No image/video files were saved and the temporary workspace was deleted. This demonstrates the real sidecar/Studio IPC and camera persistence path, **not** that these observations are correctly gesture-labeled or that physical UI clicks have been verified.

**Out-of-scope platform evidence:** Mac Apple Silicon permission prompts, packaged .app code signing/notarization and manual Studio clicks remain **not verified**; follow [macOS lab](mac-lab-oct09.md). Windows packaging/hardware remains unverified. No new model accuracy claim or P003 final evaluation is made. Full Sample→Snapshot→Train→Model→Whiteboard with measured latency is **HD-10**, not a hidden claim of HD-09.

**HD-09 final resilience regression (2026-10-09):** corrupted local `models/.active-model.json` previously raised a startup exception before the sidecar could publish READY. RED→GREEN subprocess tests now require a live authenticated `/health` endpoint, tracking availability and model-health `error` even when persisted Active selection JSON is corrupt. `GestureRuntime.mark_model_load_error` preserves that visible diagnostic while processing hand observations and never silently substitutes the legacy predictor. A newly verified candidate can still be manually activated from Studio. The PyInstaller freezer, Tauri `.deb`, and Linux portable archive must be rebuilt after this Python repair; the commands above do so. This is not HD-10 accuracy evidence.

**Recovery proof with the re-frozen sidecar:** launched `resources/handd-sidecar/handd-sidecar/handd-sidecar` from an unrelated temporary working directory with `PYTHONPATH=/definitely/not/the/repository`, a temporary SQLite workspace, a valid synthetic Candidate and deliberately corrupt `.active-model.json`. The frozen process published READY and authenticated health `model=error`. The authenticated WebSocket `models` request still listed the compatible Candidate, `model_activate` returned success, and the `studio.model` notification reported `model=ready`. This establishes package-contained recovery of a corrupt selection, not real model accuracy.

**Final HD-09 Linux evidence (2026-10-09):** after fixing the persisted-model recovery, reran `uv lock --check`, `uv run --frozen python -m unittest discover -s tests -p 'test_*.py' -q` (**128/128** Python), `node --test desktop/tests/*.test.mjs` (**34/34** JavaScript), `cargo test --manifest-path desktop/src-tauri/Cargo.toml --locked -q` (**4/4** Rust), both SVG performance probes, JS/shell syntax, document links and `git diff --check`; all passed. Refroze the sidecar, regenerated the Tauri `.deb` (~485 MiB) and portable Linux x86_64 `.tar.zst` (~406 MiB). Extracted the **rebuilt portable archive** to a clean `/tmp` location, launched `launch-handd.sh` with invalid `PYTHONPATH` outside the checkout; Niri opened Hand-D, a child process launched the package-contained sidecar with bundled task/checkpoint paths, and WebKit established localhost IPC sockets. The temporary archive extraction and GUI smoke process were cleaned up afterward. These verifications finish the specified **Linux HD-09 implementation/package gate**, not HD-10 or Mac/Windows signoff.

### HD-10 integrated synthetic preflight (2026-10-09; not final acceptance)

**Studio visual inspection restoration for HD-10.** Studio → **Collect** now
shows the same authenticated camera preview and live 21-joint drawing/modifier
overlays as Whiteboard. Navigation enables the MJPEG consumer only for the
visible screen; the camera tracker remains alive during either view. It does
not instantiate a second camera or save camera images. Gesture tracking while
Studio is visible never draws on the hidden Whiteboard. Studio → **Review**
selects a canonical Sample, requests its stored image/world landmarks through
the authenticated `sample_detail` WebSocket action, and renders the 2D
image-space hand and a rotatable orthographic projection of its 3D world-space
hand. No photo is reconstructed or persisted. Review navigates datasets in
pages of 40 rather than hiding Samples beyond the previous fixed first 40;
Accept remains disabled until the selected Sample's image landmarks are valid
and displayed. Changes of selection discard the old preview; late responses
for a different Sample are ignored. `tests/test_studio_transport.py`,
`tests/test_studio_workspace.py`, `desktop/tests/app-ui.test.mjs` and
`desktop/tests/world-landmarks.test.mjs` cover transport, paging, rendering,
rotation, valid-preview gating and hidden-Whiteboard protection.
**Linux contract suite: 131 Python / 40 JS / 7 Rust PASS**; the 40th JS test
protects against accidental canvas edits during Studio collection. **Human
visual smoke in native Tauri/WebKit remains
unverified.** The shape alone is not an independent truth label: human review
must compare the intended gesture with observations while collecting; no
stored camera frame exists for retrospective pose verification.

Run `uv run --frozen python -m unittest tests.test_hd10_tracer -v` from the repo root. The test creates an isolated temporary workspace, drives authenticated Studio collection/review/snapshot requests using **synthetic landmark fixtures**, trains a genuine CPU Candidate on frozen inputs, explicitly activates that model, verifies runtime events over a real loopback WebSocket, and restarts the Python sidecar as a real child process to verify Active Model restoration. It removes temporary inputs/artifacts on exit. It does **not** use the camera, Tauri WebKit UI, production datasets or a real operator, and it intentionally does not treat fabricated intended labels as accuracy evidence.

The 2026-10-09 Linux full regression (**130 Python / 35 JavaScript / 7 Rust PASS**) included 17 of 20 synthetic Samples eligible in the original Snapshot; the later review change excluded one more without modifying the immutable earlier Snapshot. The test invokes `python -m handd_core.train_cli` in a real subprocess, not a mocked training routine. Eight runtime updates gave **1.044 ms p95** for in-process event publication to the test's WS client, and **1.183 ms p95** for the separately reported post-landmarker Python compute segment. This is a tiny non-production sample, **not full end-to-end latency**. The verified Candidate was recovered by a new real Python sidecar process with `--no-camera`; no native Tauri or WebKit claim.

**HD-10 remaining acceptance:** On Linux with a cooperating human operator, capture independently verified gesture labels for P001/P002 across distinct Sessions using the existing Studio Collect flow (see above), explicitly review each Sample, build a new immutable Development Snapshot, train with `uv run --frozen python -m handd_core.train_cli --workspace /path/to/test-workspace --snapshot dev-SNAPSHOT_ID`, activate the compatible Candidate in Studio, and observe actual gestures, Whiteboard drawing/erasing and measured full callback→IPC→frontend latency in a real Tauri window. Preserve Sample/source membership hashes, actual Macro F1, per-class errors, runtime/IPC measurements and limitations in the report; do not reuse this synthetic test as model quality evidence. Keep P003 sealed if available and do not claim generalization without a real unseen-participant test. Mac/Windows hardware validation stays separate.

## Recuperación

- If a dependency experiment breaks the environment, remove/recreate `.venv`; do not repair by installing packages globally.
- If a v2 slice is invalid, revert the granular commit on the modernization branch rather than rewriting `main`.
- Preserve legacy datasets and models while migration/import logic is being validated.

## Riesgos y secretos

- Collection provenance must not store hardware serials, MAC addresses, passwords, tokens, or other unnecessary identifying secrets.
- Use an application-generated anonymous device ID for local provenance.
- Never document secret values; document only variable/configuration names when secrets eventually exist.
