# Hand-D v2 Roadmap — October 5 to October 13, 2026

## Objetivo

Deliver a technically credible Hand-D v2 tracer/vertical slice by October 13 without pretending the whole target product is complete. The milestone should prove the central path end-to-end: canonical collection -> SQLite/review -> immutable snapshot -> reproducible training/evaluation -> Model Artifact -> fresh real-time inference -> minimal Tauri integration.

### Checkpoint — October 9, 2026

- **HD-01..HD-08:** completados en sus alcances core/CLI/Linux; import legacy HD-06L añadido. Esto no significa accuracy real ni evaluación P003 certificada.
- **HD-09 DONE (Linux implementation and packaged process smoke; external platform validation pending):** Tauri 2 shell + supervised Python/MJPEG/WS, Whiteboard camera-first, landmarks, Wiggly **line-boil accepted visually by user on Linux**, transparent eraser, English-only UI (ADR 0005). Studio supports native Open/Create Workspace selection, Development Collect/Pause/Resume/Finish, audited Review, immutable Development Snapshot and **explicit verified Active Model** with rollback and restart persistence. Whiteboard native Save/Open editable ink and clean SVG export never include camera footage. Linux frozen sidecar without repo Python, Tauri .deb, installed-layout Niri window, WebKit connections, camera and authenticated real 5-Sample SQLite Capture were exercised. **Do not infer macOS/Windows coverage, model accuracy or full manual WebKit UI Collect validation** from this evidence.
- **HD-10 TODO:** un tracer real Sample→snapshot→entrenamiento→Model Artifact→Whiteboard con medición completa y procedencia honesta.
- **Siguiente gate de plataformas:** Macs Apple Silicon el **9 de octubre**, mediante [runbook de laboratorio](../runbook/mac-lab-oct09.md). Son dispositivos **NO VERIFICADOS** hasta ejecutar los pasos; el preflight macOS no certifica cámara, rendimiento ni permisos TCC.

The broader Hand-D v2 architecture remains intentionally larger than this milestone. Confirmed features such as the complete Studio information architecture, Easy Mode polish, extensible gesture UX, release/CD polish, and Tier-1 validation on every desktop OS may continue after October 13 without being considered architectural reversals.

## No-objetivo

- Finish every v2 feature.
- Put training/evaluation inside the Studio GUI.
- Recollect the complete legacy dataset.
- Promise every GPU family.
- Merge directly into `main` during implementation.
- Finish every confirmed Studio/product/release feature before the tracer path works.
- Treat Windows as Tier-1 hardware-validated merely because CI builds it.
- Force MediaPipe 1.1.x or ONNX Runtime if their compatibility spikes fail.

## Milestone cut line

| Area | October 13 gate | Broader v2 target, not required to be fully polished by October 13 |
|---|---|---|
| Data | Canonical Sample/landmarks/provenance + separate Review/Lifecycle state in SQLite | Full Studio browse/filter/analytics polish |
| Collection | One working Participant -> Session -> Capture path | Complete gesture-checklist UX and extensible-gesture authoring polish |
| Curation | Accept/Reject/Drop semantics + snapshot eligibility + minimal review path | Full Suggested-for-Review UI, dashboards, convenience workflows |
| ML | Immutable Development Snapshot, grouped CV, OOF assessments, metrics, final-refit protocol, Model Artifact | Full Studio-owned training UI / MLflow-scale experiment management |
| Runtime | LIVE_STREAM freshness path, normalized runtime result contract, working model inference | Exhaustive provider optimization across every accelerator |
| Desktop | Minimal Tauri shell + Python sidecar lifecycle + HTTP/WS + MJPEG proof | Final packaging/CD/updater and complete Impeccable-polished Studio |
| Platforms | Linux Tier-1 evidence; macOS Tier-1 evidence when lab access exists; Windows native CI/build contract | Windows Tier-1 camera/runtime claim before real hardware evidence |

## Archivos y componentes afectados

Expected areas, subject to design refinement:

- shared data/ML contract and canonicalization currently duplicated across `dataset_extraction_tools/` and `visualizer_app/`;
- `dataset_extraction_tools/`;
- `model_training/`;
- `visualizer_app/` runtime behavior;
- tests and fixtures;
- dependency/packaging configuration;
- `docs/`, `CONTEXT.md`, and `AGENTS.md`;
- a temporary desktop-shell prototype area if needed.

## Diseño

### Sequencing principle

Do not begin with the new UI shell. First stabilize the contracts the UI will consume:

```text
data contract
  → shared transforms / handedness
  → snapshot / reproducible evaluation
  → minimal collection + review workflow
  → runtime result contract
  → Tauri-sidecar vertical integration
  → integration / hardening
```

### October 5 — canonicalize the project plan

Deliverables:

- canonical documentation system;
- requirements and architecture reference;
- accepted ADRs for decisions already made;
- contributor/agent rules;
- milestone roadmap;
- explicit inventory of open decisions.

Gate:

- no implementation work starts from an undocumented or contradictory architecture assumption.

### October 6 — environment and shared data contract

Deliverables:

- establish a project-local Python 3.13 environment managed by `uv` with `pyproject.toml` + `uv.lock`;
- separate runtime/training/dev dependency groups and remove duplicate/conflicting OpenCV distributions;
- freeze current MediaPipe contract tests before testing 1.1.x as a candidate rather than a mandatory migration;
- prototype PyTorch-vs-ONNX Runtime deployment only after the canonical model contract is executable;
- centralize handedness conversion and landmark/feature canonicalization behind one tested implementation;
- define canonical sample/provenance schema with independent Review Status and Sample Lifecycle Status;
- define legacy import boundary without rewriting legacy history.

Verification focus:

- unit tests for handedness normalization;
- unit tests for deterministic feature extraction;
- schema round-trip tests;
- CPU-only tests must pass without camera/GPU hardware.

### October 7 — collector v2 and provenance

Deliverables:

- frame-based Studio collector core;
- automatic session/capture IDs;
- anonymous persistent device provenance;
- participant/gesture/hand selection;
- no image/video persistence;
- two-run collection semantics documented and implemented.

Verification focus:

- interrupted collection does not corrupt prior data;
- provenance is present and internally consistent;
- requested hand semantics match runtime semantics;
- collection can run on CPU.

### October 8 — curator and Suggested for Review

Deliverables:

- reversible accepted/rejected workflow;
- dataset generation excludes rejected samples without deleting source observations;
- model-assessment contract for class scores/margin plus training-membership/out-of-sample context;
- out-of-fold assessments from session-held-out CV for authoritative development Suggested for Review evidence;
- initial Suggested for Review signals are disagreement and uncertainty; geometry hooks remain deferred until justified by observed data;
- confidence threshold remains configurable/experimental rather than an invented fixed truth.

Verification focus:

- rejected samples cannot silently reappear in generated training data;
- dropped samples retain their prior Review Status and remain canonically restorable;
- difficult-but-valid samples can remain accepted;
- changing model version does not overwrite historical sample truth or historical Model Assessments.

### October 9 — training and evaluation repair

Deliverables:

- five-class reproducible training pipeline;
- participant/session-aware split generation;
- legacy dataset allowed for training but not authoritative final testing;
- metrics beyond raw accuracy: confusion matrix and per-class precision/recall/F1;
- model artifact metadata records label order and feature-transform compatibility.
- out-of-fold predictions/score margins are materialized for the development Samples;
- after development choices freeze, the selected configuration is refit on all eligible P001/P002 development data before any P003 final evaluation.

Preferred evaluation:

- P001/P002 provide train/validation sessions;
- genuinely new P003, if available, is final test only;
- if P003 is unavailable, report the limitation explicitly.

Verification focus:

- no row-level random leakage across grouped splits;
- validation loader actually reads validation data;
- repeated run with the same seed/config reproduces the split.
- every development Sample's authoritative review assessment comes from a fold that excluded its Collection Session from training;
- P003 is consulted only after final refit and is not fed back into the milestone model.

### October 10 — real-time runtime v2

Deliverables:

- migrate the webcam path to the already-selected MediaPipe LIVE_STREAM/detect_async contract after compatibility tests pass;
- implement the freshest-result policy for the App path;
- prediction contract returns label plus scores/confidence;
- runtime result sends normalized tracking coordinates; frontend owns canvas mapping/smoothing/drawing state;
- provider resolver supports CPU universally and optional acceleration only when benchmark evidence is positive;
- correct engine pause/stop/lifecycle behavior.

Verification focus:

- stale frames do not accumulate unbounded latency;
- CPU path is functional;
- camera/model failures become explicit runtime states;
- unit tests do not require a physical camera by default.

### October 11 — UX and desktop-shell prototype

Deliverables:

- use the completed UX Shape handoff and interactive static prototype for Whiteboard and Studio rather than continuing open-ended design questioning;
- prototype the minimum runtime-to-frontend contract with the selected Tauri shell;
- validate Python sidecar/process lifecycle, IPC/event flow, startup, shutdown, crash behavior, and packaging constraints;
- validate direct MJPEG preview independently from HTTP/WebSocket control/result traffic.

The prototype is disposable. It must not force production architecture merely because it exists.

### October 12 — integration and cross-platform validation

Deliverables:

- integrate the vertical slice;
- Linux end-to-end Tier-1 test, including CPU fallback and any candidate runtime provider;
- Apple Silicon Tier-1 validation if lab access is available, including MPS training and runtime-provider comparison;
- Windows native CI/build/startup/model-load contract validation without claiming camera Tier-1 status;
- run opposite-hand verification collection;
- update runbook with commands that were actually executed;
- close or explicitly defer open milestone blockers.

If M4 access is unavailable, mark Apple Silicon execution as `No verificado` rather than guessing.

### October 13 — stabilization and milestone evidence

Deliverables:

- regression tests;
- evaluation report/results;
- architecture/reference docs synchronized with implemented reality;
- known limitations;
- demo path;
- final milestone commit/branch state ready for review.

No direct `main` work is part of this roadmap. Merge/release is a separate reviewed action.

## Errores, offline y migración

- Legacy data remains available and is imported/read through an explicit compatibility boundary.
- Dependency upgrades must be reversible through the branch/history until the compatibility set is validated.
- Optional GPU backend failure must degrade to CPU rather than make the product unusable.
- Missing P003 or M4 access changes what can be claimed, not the integrity of the result.

## Rollback

- All v2 work remains on `feature/hand-d-v2-modernization` (or child feature branches if intentionally introduced).
- `main` is not modified directly.
- Major migrations should be committed in granular slices so a failed dependency, data, runtime, or UI experiment can be reverted independently.

## Pruebas y verificación

Required categories by milestone:

- unit: canonicalization, handedness, schema, runtime/provider selection, model metadata;
- data: curation/lifecycle reversibility, grouped split integrity, legacy import;
- ML: reproducible split, OOF assessments, confusion matrix, per-class metrics, final refit, held-out evaluation where possible;
- runtime: newest-result behavior, lifecycle, error states;
- integration: collector → dataset → training → model artifact → runtime;
- platform: Linux Tier-1 required; M4 Tier-1 when physically available; Windows native build/contract Tier-2; acceleration only where hardware/provider evidence exists.

## Definición de terminado

The October 13 vertical slice is complete when:

- canonical docs match the implemented architecture;
- `uv` can recreate the documented Python project environment and dependency groups;
- CPU fallback operation is verified;
- collection and reversible curation use the new data contract;
- training/evaluation is reproducible and five-class;
- grouped evaluation avoids known row-level leakage;
- Suggested-for-Review evidence has an out-of-fold path rather than relying on in-sample confidence;
- the final-training protocol explicitly refits on all development data before a one-time P003 evaluation when P003 exists;
- the App runtime follows the newest-result policy;
- dependency choices are documented and tested on available hardware;
- Tauri + sidecar + HTTP/WS + MJPEG has vertical-slice prototype evidence;
- limitations around participants/platforms are reported explicitly.
