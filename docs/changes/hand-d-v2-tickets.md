# Hand-D v2 — Execution Tickets (Oct 8–13)

> Ordered, independently testable work against [tracer spec](../spec/hand-d-v2-oct13-tracer.md). **Status definitions:** `TODO` = not implemented; `IN PROGRESS` = tests/code being worked; `DONE` = executable acceptance evidence recorded. A ticket's scope is smaller than the entire long-term v2 feature.

| ID | Pri | Status | Depends | Deliverable and TDD exit gate |
|---|---|---|---|---|
| HD-01 | P0 | TODO | — | `pyproject.toml`, `uv.lock`, supported Python/one OpenCV wheel, runtime/train/dev dependency split; `uv sync` + baseline tests pass without global installs |
| HD-02 | P0 | DONE | — | Single importable **Feature Transform v1** implementation; deterministic golden 69-feature fixtures, left/right semantics, shape/finite/degenerate tests; runtime delegates to it with no valid-input regression |
| HD-03 | P0 | DONE | HD-02 | SQLite canonical participant/session/capture/sample schema + Review/Lifecycle event logs; transaction, provenance and state round-trip tests |
| HD-04 | P0 | TODO | HD-03 | Collection core accepts real 21×3 image/world landmark observations with time-based sampling and auto provenance, no image/video; interrupted-capture tests |
| HD-05 | P0 | TODO | HD-03 | Human Accept/Reject/Drop/Restore and eligibility query; audit independence, never auto-reject, snapshot Readiness blocker/warning tests |
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

Next action: HD-01 (`uv` project setup), then HD-04/HD-05 (collector and review integration). Avoid initiating new discovery/grill rounds.
