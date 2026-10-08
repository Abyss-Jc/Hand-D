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

## Recuperación

- If a dependency experiment breaks the environment, remove/recreate `.venv`; do not repair by installing packages globally.
- If a v2 slice is invalid, revert the granular commit on the modernization branch rather than rewriting `main`.
- Preserve legacy datasets and models while migration/import logic is being validated.

## Riesgos y secretos

- Collection provenance must not store hardware serials, MAC addresses, passwords, tokens, or other unnecessary identifying secrets.
- Use an application-generated anonymous device ID for local provenance.
- Never document secret values; document only variable/configuration names when secrets eventually exist.
