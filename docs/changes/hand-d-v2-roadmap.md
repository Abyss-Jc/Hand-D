# Hand-D v2 Roadmap — October 5 to October 13, 2026

## Objetivo

Deliver a technically credible Hand-D v2 vertical slice by October 13 without pretending the whole rewrite is complete. The milestone should establish reliable data/model contracts, a corrected real-time runtime path, a usable Studio collection/curation slice, reproducible evaluation, refreshed dependencies, documentation, and evidence for the future desktop UI architecture.

## No-objetivo

- Finish every v2 feature.
- Put training/evaluation inside the Studio GUI.
- Recollect the complete legacy dataset.
- Promise every GPU family.
- Merge directly into `main` during implementation.
- Select Tauri/Electron before a measured prototype.

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
  → reproducible evaluation
  → Studio workflow
  → runtime result contract
  → desktop-shell prototype
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

- establish a project-local reproducible Python environment;
- choose and verify the initial compatible Python / MediaPipe / PyTorch dependency set on the Linux development machine;
- centralize handedness conversion and landmark/feature canonicalization behind one tested implementation;
- define canonical sample/provenance/review-state schema;
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
- model-assessment contract for class scores/confidence;
- initial Suggested for Review signals such as model disagreement and tracking/geometry problems;
- confidence threshold remains configurable/experimental rather than an invented fixed truth.

Verification focus:

- rejected samples cannot silently reappear in generated training data;
- difficult-but-valid samples can remain accepted;
- changing model version does not overwrite historical sample truth.

### October 9 — training and evaluation repair

Deliverables:

- five-class reproducible training pipeline;
- participant/session-aware split generation;
- legacy dataset allowed for training but not authoritative final testing;
- metrics beyond raw accuracy: confusion matrix and per-class precision/recall/F1;
- model artifact metadata records label order and feature-transform compatibility.

Preferred evaluation:

- P001/P002 provide train/validation sessions;
- genuinely new P003, if available, is final test only;
- if P003 is unavailable, report the limitation explicitly.

Verification focus:

- no row-level random leakage across grouped splits;
- validation loader actually reads validation data;
- repeated run with the same seed/config reproduces the split.

### October 10 — real-time runtime v2

Deliverables:

- benchmark MediaPipe IMAGE/current behavior against VIDEO/LIVE_STREAM candidates as applicable;
- implement the freshest-result policy for the App path;
- prediction contract returns label plus scores/confidence;
- backend resolver supports CPU universally and optional validated acceleration;
- correct engine pause/stop/lifecycle behavior.

Verification focus:

- stale frames do not accumulate unbounded latency;
- CPU path is functional;
- camera/model failures become explicit runtime states;
- unit tests do not require a physical camera by default.

### October 11 — UX and desktop-shell prototype

Deliverables:

- define the App and Studio primary flows using UI/UX analysis principles;
- prototype the minimum runtime-to-frontend contract with leading shell candidate(s);
- validate Python sidecar/process lifecycle, IPC/event flow, startup, shutdown, crash behavior, and packaging constraints;
- choose a shell only if the evidence is sufficient; otherwise keep the decision open.

The prototype is disposable. It must not force production architecture merely because it exists.

### October 12 — integration and cross-platform validation

Deliverables:

- integrate the vertical slice;
- Linux CPU baseline test;
- Apple Silicon/MPS validation if lab access is available;
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

- unit: canonicalization, handedness, schema, backend selection, model metadata;
- data: curation reversibility, grouped split integrity, legacy import;
- ML: reproducible split, confusion matrix, per-class metrics, held-out evaluation where possible;
- runtime: newest-result behavior, lifecycle, error states;
- integration: collector → dataset → training → model artifact → runtime;
- platform: Linux CPU required; M4/MPS when physically available; optional GPU paths only where hardware exists.

## Definición de terminado

The October 13 vertical slice is complete when:

- canonical docs match the implemented architecture;
- CPU operation is verified;
- collection and reversible curation use the new data contract;
- training/evaluation is reproducible and five-class;
- grouped evaluation avoids known row-level leakage;
- the App runtime follows the newest-result policy;
- dependency choices are documented and tested on available hardware;
- UI shell direction has prototype evidence rather than assumption;
- limitations around participants/platforms are reported explicitly.
