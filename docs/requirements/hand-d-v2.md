# Hand-D v2 Requirements

## Contexto y problema

Hand-D began as a semester project and currently mixes a user-facing whiteboard, data collection utilities, dataset curation scripts, model training, and runtime inference without a single stable contract between them. Parts of the repository have drifted: the runtime uses five gesture classes while older training and specification material still describes four, hardware selection is CUDA-or-CPU only, and the existing dataset lacks participant/session provenance.

The October 13 milestone is not a complete rewrite. It is a vertical slice that establishes reliable boundaries for the Hand-D App, Hand-D Studio, the shared ML/data core, reproducible evaluation, documentation, and a validated direction for the future desktop UI.

## Comportamiento esperado

- The **Hand-D App** remains the user-facing gesture whiteboard.
- **Hand-D Studio** owns developer workflows for collection, inspection, reversible curation, and dataset preparation.
- Training and evaluation must become reproducible, but a training/evaluation GUI is outside the October milestone.
- Runtime inference prioritizes the freshest result and interaction latency rather than processing every camera frame.
- Data collection remains frame-based and deliberate.
- The canonical v2 sample stores raw MediaPipe landmarks, labels, and provenance; feature vectors are versioned derived data.
- Photos and video are not stored as part of the canonical dataset.
- Dataset curation is reversible. Automatic signals may create a Suggested for Review queue, but only a human review action decides whether a sample is accepted or rejected.
- CPU is a functional fallback on every supported platform. GPU acceleration is optional and capability-driven.
- macOS Apple Silicon is a first-class validation target. NVIDIA CUDA, Apple MPS, AMD ROCm, and Intel XPU are only enabled where the selected runtime stack and hardware are actually supported.
- The desktop shell is not selected by assumption. Tauri, Electron, or another candidate must be validated through a focused prototype before becoming a durable architecture decision.
- Project dependencies must be refreshed as a tested compatibility set, not upgraded independently.

## Criterios de aceptación

| ID | Given | When | Then |
|---|---|---|---|
| V2-001 | A supported machine without an available GPU backend | Hand-D inference starts | The runtime uses CPU and the core application remains functional |
| V2-002 | The App runtime cannot keep up with camera input | Newer frames arrive | Stale work may be dropped so the App reacts to the freshest usable result |
| V2-003 | Studio collects a gesture sample | The sample is persisted | Raw landmarks, label, participant/session provenance, device provenance, and timestamps are retained without storing a camera photo/video frame |
| V2-004 | A sample receives low confidence or model disagreement | Studio evaluates review suggestions | The sample is suggested for review but is not automatically deleted, rejected, or relabeled |
| V2-005 | A reviewer rejects a sample | The training dataset is generated | The source observation remains traceable, while rejected samples are excluded from the generated training set |
| V2-006 | A v2 feature transform changes | Existing v2 raw-landmark samples are available | Derived feature data can be regenerated without recollecting those samples |
| V2-007 | Legacy CSV data is used | A model is trained/evaluated | Legacy samples may contribute to training but are not treated as authoritative final-test data because participant/session provenance is unknown |
| V2-008 | A genuinely new third participant is available | Final participant-level evaluation runs | That participant is held out from train/validation and used as the preferred unseen-participant test |
| V2-009 | Only two participants are available by the milestone | Evaluation is reported | Results explicitly distinguish session-level validation from limited participant-level experiments and do not overclaim generalization |
| V2-010 | Primary-hand data has been collected | Handedness invariance is evaluated | A smaller opposite-hand verification set tests whether canonicalization actually generalizes across hands |
| V2-011 | A desktop-shell candidate is proposed | It becomes the production direction | A focused prototype has first validated sidecar lifecycle, IPC/event flow, startup/shutdown behavior, and relevant packaging constraints |
| V2-012 | Dependencies are updated | The migration is accepted | The selected Python, MediaPipe, PyTorch, packaging, and acceleration combination is documented and verified on the available target environments |

## Alcance

- Hand-D App/runtime boundary.
- Hand-D Studio collector and curator.
- Canonical v2 dataset/provenance model.
- Legacy dataset import/use policy.
- Reproducible training and evaluation outside the Studio GUI.
- Confidence/model-assessment data sufficient for Suggested for Review.
- Freshest-result real-time inference path.
- Cross-platform backend selection with universal CPU fallback.
- Desktop-shell prototype and UI/UX direction.
- Dependency modernization.
- Tests and canonical documentation required to support the above.

## Fuera de alcance

- Training and evaluation from the Studio GUI before October 13.
- Remote analytics/telemetry service.
- Storing collection photos or video in the canonical dataset.
- Promising unsupported GPU families merely because a vendor SDK exists.
- Recollecting the full legacy dataset from scratch.
- Selecting Tauri or Electron before the prototype evidence exists.
- A complete production-quality v2 rewrite by October 13.

## Preguntas abiertas y aprobación

Open:

- Exact train/validation/test generation rules after the session schema is finalized.
- Exact machine-generated review signals and confidence thresholds.
- Storage format for v2 sessions/samples.
- Runtime performance budgets.
- Desktop-shell winner after prototype measurements.
- Exact compatible dependency versions and packaging strategy.

Confirmed direction is tracked in `docs/changes/hand-d-v2-modernization.md`. This document should be updated when an open product requirement becomes a confirmed decision.

## Fuentes

- `visualizer_app/gesture_engine.py:90-149` — current 69-feature canonicalization.
- `visualizer_app/gesture_engine.py:181-203` — current CUDA/CPU model wrapper and label-only prediction.
- `visualizer_app/gesture_engine.py:332-374` — current synchronous per-frame detector loop.
- `dataset_extraction_tools/data_extractor.py:43-119` — current collector feature extraction and handedness handling.
- `dataset_extraction_tools/data_extractor.py:197-269` — current CSV persistence and frame-stride collection.
- `model_training/gesture_classifier.py:43-94` — current training data loading/splitting and stale four-class path.
- `docs/changes/hand-d-v2-modernization.md` — accepted v2 planning decisions.
