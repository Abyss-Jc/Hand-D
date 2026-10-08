# Hand-D v2 — Execution Tickets (Oct 8–13)

> Ordered, independently testable work against [tracer spec](../spec/hand-d-v2-oct13-tracer.md). **Status definitions:** `TODO` = not implemented; `IN PROGRESS` = tests/code being worked; `DONE` = executable acceptance evidence recorded. A ticket's scope is smaller than the entire long-term v2 feature.

| ID | Pri | Status | Depends | Deliverable and TDD exit gate |
|---|---|---|---|---|
| HD-01 | P0 | DONE | — | `pyproject.toml`, `uv.lock`, supported Python/one OpenCV wheel, runtime/train/dev dependency split; `uv sync` + baseline tests pass without global installs |
| HD-02 | P0 | DONE | — | Single importable **Feature Transform v1** implementation; deterministic golden 69-feature fixtures, left/right semantics, shape/finite/degenerate tests; runtime delegates to it with no valid-input regression |
| HD-03 | P0 | DONE | HD-02 | SQLite canonical participant/session/capture/sample schema + Review/Lifecycle event logs; transaction, provenance and state round-trip tests |
| HD-04 | P0 | IN PROGRESS | HD-03 | Time-based CaptureSampler, resumable quota, automatic provenance and LIVE_STREAM callback bridge; **physical camera + MediaPipe + old MLP smoke verified**, but operator-confirmed real-world handedness and persisted live Capture still **No verificado** |
| HD-05 | P0 | DONE (core/CLI) | HD-03 | Human Accept/Reject/Drop/Restore using audited independent events, eligible accepted+active query, read-only Snapshot Readiness warnings/blockers and CLI; Studio UI integration remains HD-09 |
| HD-06 | P0 | TODO | HD-02, HD-03, HD-05 | Immutable Development Snapshot/NPZ+manifest and compatible legacy partition; content hash, deterministic membership, rejected/unreviewed exclusion, overwrite refusal tests |
| HD-07 | P0 | TODO | HD-01, HD-06 | Reproducible five-class LOSO CV, OOF scores, paired legacy trials, Macro F1/per-class/confusion/learning curve and final-refit Model Artifact; fold leakage and manifest tests |
| HD-08 | P0 | TODO | HD-02, HD-07 | LIVE_STREAM/latest-frame runtime with normalized tracking, stable gestures, fixed hand roles, CPU fallback and health; synthetic stale-frame/lifecycle tests |
| HD-09 | P0 | TODO | HD-04, HD-05, HD-06, HD-08 | Minimal Linux Tauri shell + Python supervised sidecar + dynamic loopback HTTP/WS/MJPEG, camera-first Whiteboard and basic Studio Collect/Review/Snapshot; startup/crash/resync smoke tests |
| HD-10 | P0 | TODO | HD-07, HD-08, HD-09 | One Linux end-to-end collected-Sample→artifact→Whiteboard run with truthful provenance, measured latency, test report and known limitations; document tested/not-tested platform claims |
| HD-11 | P1 | TODO | HD-09 | Expanded Whiteboard visualizer as accessible **in-app modal** using the same canvas/runtime, close/Escape, preserved strokes/undo; frontend interaction test. Do not let polish block HD-10 |
| HD-12 | P1 | TODO | HD-07 | Optional comparison of PyTorch inference vs ONNX Runtime and MediaPipe 1.1.x after contract tests, only if time/hardware allow; no forced provider/library migration |

## Work-in-progress protocol

1. Run `git status`; never touch `main` or overwrite teammate changes.
2. **Red:** write a meaningful focused failing test for a behavior/contract before implementing the ticket.
3. **Green:** minimal code to satisfy the test and preserve compatibility.
4. **Refactor:** remove duplicated behavior, run focused suite plus baseline; log exact commands and results.
5. Document proven implementation changes, commit a coherent slice to the modernization branch, and push only after checks pass.

## Critical-path rules

- HD-02 may begin before HD-01 because the existing project-local `venv` already has NumPy, Torch and MediaPipe; HD-01 becomes mandatory before announcing a reproducible v2 environment.
- HD-03–HD-07 deliver the data/ML backbone; HD-09 should remain a **thin proof**, not a full Studio rewrite.
- HD-11, HD-12 and the complete polished Studio UX are explicitly deferrable if they endanger the October 13 end-to-end gate.
- No P003 means no unseen-participant generalization claim. No real M4/Windows camera means no hardware claim for that target.

## Evidence log

- 2026-10-08: existing `venv/bin/python` can import NumPy 2.4.4, MediaPipe 0.10.33 and Torch 2.14.1+cpu; `uv 0.12.23` installed, no v2 pyproject/lock yet.
- HD-02 (2026-10-08): RED was ModuleNotFoundError for the missing shared core; a second RED proved a direct legacy source-launch import regression. GREEN implemented `handd_core/feature_transform.py`, delegated `visualizer_app/gesture_engine.py`, and added 8 focused tests. `venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v` passed **10/10** total; compileall and diff checks passed. An explicit comparison with the previous runtime implementation produced identical vectors for Left/Right and float32/float64 fixture inputs.
- Legacy limitation: the standalone CSV/plotting collector still has a separate transform; migrate or retire it under HD-04. All new v2 collection/training callers must use the shared transform.
- HD-03 (2026-10-08): RED established missing `handd_core.dataset_store`; second RED established that direct SQLite state mutation must not bypass review auditing. GREEN added canonical participant/session/capture/sample persistence, 21x3 image/world landmarks, schema version and FK validation, immutable Sample observations, Review/Lifecycle transitions and append-only event triggers. Suite now **20/20 passing** (10 dataset-store and 10 legacy/transform), `compileall` and diff checks pass. Later HD-04 adds camera sampling, capture-generated IDs/device metadata; snapshot materialization is HD-06.
- HD-01 (2026-10-08): provisioned Python **3.13.16** using `uv python install 3.13`; created `.python-version`, `pyproject.toml` and `uv.lock`. `uv lock --check`, `uv sync --frozen`, `uv run --frozen python -m unittest discover -s tests -p 'test_*.py' -v` (**20/20 passed**), and `uv pip check --python .venv/bin/python` all succeeded on CachyOS Linux x86-64. Runtime-only `uv sync --frozen --no-default-groups` passed 20 tests; training `uv sync --frozen --group training` imported scikit-learn/pandas/matplotlib and passed 20 tests; base sync removed optional training packages again. The lock/install contains only `opencv-contrib-python`, not two overlapping OpenCV distributions. Linux torch resolved to CPU (`2.14.1+cpu`); accelerator and other-OS installs are **No verificado**. Details: `docs/runbook/development.md`.
- HD-04 (2026-10-08): RED for missing `handd_core.capture_sampling`, missing local-device provenance API, missing `handd_core.live_collection`, and missing CLI module; GREEN added time-interval collection with quota, invalid/wrong-hand skip, pause/restart recovery and local pseudonymous device ID, callback newest-only buffering with main-thread SQLite writes, and `uv run --frozen python -m handd_core.collect_cli --help`. Synthetic MediaPipe 21-point detections and CLI argument contract pass. Real camera/callback timing, physical handedness, and actual frame collection remain **No verificado**; finish this hardware smoke before changing HD-04 to DONE.
- HD-05 (2026-10-08): RED for missing Snapshot Readiness/manual review CLI and a dropped Sample incorrectly visible in default Browse; GREEN added read-only eligibility assessment (no eligible/invalid features/session test overlap hard blockers, incomplete coverage/unreviewed warnings) and explicit per-Sample Accept/Reject/Drop/Restore CLI using independent audit events. Dropped rows are hidden by default and recoverable via `list --include-dropped`. `uv run --frozen python -m unittest discover -s tests -p 'test_*.py' -q` passed **35/35**; all tests use temporary SQLite and synthetic observations. No automatic acceptance/rejection is performed.
- HD-04 hardware smoke (2026-10-08): new `handd_core.camera_smoke` with no recording; TDD RED missing module/`--preview`, GREEN real checkpoint inference and finite-time camera probe. Headless `--seconds 8`: `/dev/video0` read **234/234** 640×480 frames, **29.12 captured FPS**, MediaPipe LIVE_STREAM **234 callbacks**, 12 hand-detection callback batches, raw label Left x12, legacy classifier Idle x12, **0 invalid features/0 inference errors**. Optional visible `--preview --seconds 6`: **175/175** frames, **28.93 captured FPS**, **175 callbacks**, 74 hand callback batches, raw Left x74, predictions Idle x32 / Index_Finger x29 / Fist x13, **0 inference errors**. No frames/video/Samples saved. Latest full no-camera suite now **38/38**. These are **functional smoke results, not classification accuracy, inference p95 or proof of physical handedness**; legacy `.pth` has 69 input / 5 output weights but lacks an independent manifest confirming label order. Qt reported a missing Wayland plugin and used a working window display path; no claim of a Wayland-native HighGUI provider.

Next action: explicit physical Left/Right confirmation and live Capture→SQLite verification to close HD-04, then HD-06 immutable Snapshot Builder. Start the small Tauri/sidecar/MJPEG risk spike early. Avoid initiating new discovery/grill rounds.
