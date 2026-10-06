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
- Within a capture, the participant should preserve the target gesture while moving naturally through moderate changes in position, distance, and orientation rather than holding one perfect static pose or forcing extreme motion.
- Primary participants should contribute at least two independent collection sessions with moderate natural differences between runs.
- The Collector uses a sample quota rather than a fixed duration. Technically invalid detections do not count toward the quota, and the capture quota includes reserve samples so later human curation can reject observations without immediately requiring recollection.
- The initial collection default is 120 stored observations per gesture/capture, targeting roughly 100 usable observations after curation. The quota is configurable and must be revisited using validation/learning-curve evidence rather than assuming that 100 is universally sufficient.
- Collection sampling uses a configurable time interval between stored observations instead of a fixed frame stride so collection density is not tied to camera FPS.
- The canonical v2 sample stores raw MediaPipe landmarks, labels, and provenance; feature vectors are versioned derived data.
- Each canonical Sample stores both MediaPipe image-normalized hand landmarks and world landmarks. The current 69-feature representation is derived from raw landmarks rather than being the only persisted observation.
- In SQLite, each Sample owns 21 relational landmark rows keyed by landmark index, with image-normalized x/y/z and world x/y/z stored as numeric columns. This keeps raw geometry inspectable and avoids opaque binary encoding at the current dataset scale.
- The canonical working dataset is stored in SQLite. Training does not consume mutable database state implicitly; each training/evaluation run must be tied to an immutable dataset snapshot that records the selected sample IDs, split assignments, transform/version information, labels/order, and reproducibility configuration. Materialized NumPy/NPZ arrays may be generated from that snapshot as derived training artifacts.
- Photos and video are not stored as part of the canonical dataset.
- Dataset curation is reversible. Automatic signals may create a Suggested for Review queue, but only a human review action decides whether a sample is accepted or rejected.
- Newly collected samples begin unreviewed. Studio must support reviewing Suggested for Review samples first and batch-accepting the remaining group so human curation remains meaningful without requiring one click per sample.
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
| V2-013 | A capture is running | The participant maintains the target gesture | Moderate natural motion/position/orientation variation is retained rather than requiring a single idealized pose |
| V2-014 | A frame has no usable hand detection or fails technical landmark checks | The Collector considers the frame | It is skipped and does not consume the capture quota |
| V2-015 | A capture reaches its configured collection quota | Human curation happens later | Reserve observations are available so some samples may be rejected without assuming every collected observation must enter training |
| V2-016 | A collection run uses the initial defaults | A gesture capture completes | Studio stores up to 120 technically valid observations as a starting quota, with the value configurable rather than hard-coded as a model requirement |
| V2-017 | Cameras run at different FPS | The same collection interval is configured | Observation sampling cadence is time-based rather than changing implicitly with camera frame rate |
| V2-018 | A new capture has been collected | The Curator opens it | Samples begin unreviewed, Suggested for Review items can be inspected first, and the remaining group can be batch-accepted by a human action |
| V2-019 | Training/evaluation is started from the canonical SQLite dataset | The run is created | The exact sample membership, split assignment, feature-transform version, label mapping/order, and reproducibility settings are frozen in a snapshot before model training |
| V2-020 | A training snapshot exists | Vectorized model input is needed | Derived NumPy/NPZ arrays may be regenerated from the snapshot without changing the canonical observations or review history |
| V2-021 | MediaPipe returns both normalized image landmarks and world landmarks for a valid observation | Studio persists the Sample | Both landmark sets are stored canonically so later tracking/feature transforms can be regenerated without the camera frame |
| V2-022 | A canonical Sample is persisted | Its landmarks are stored | Exactly 21 landmark rows are associated with the Sample, each carrying image-normalized and world x/y/z coordinates |
| V2-023 | A Capture is paused and resumed within the same active Collection Session | Collection continues | Studio resumes the same Capture and continues toward its configured quota |
| V2-024 | An incomplete Capture has never been referenced by an immutable dataset snapshot | The operator chooses Discard | Studio may permanently delete that Capture and its Samples after explicit confirmation |
| V2-025 | A Collection Session has ended | A partial Capture from that Session still exists | The partial Capture remains historically partial; a later Collection Session creates a new Capture rather than silently continuing the old one |
| V2-026 | A Sample/Capture is referenced by an immutable snapshot | A destructive delete is requested | Studio must preserve snapshot reproducibility and must not silently hard-delete the referenced data |

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
