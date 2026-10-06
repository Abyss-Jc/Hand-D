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
- Suggested for Review is driven by explainable non-destructive signals. Model disagreement/uncertainty and non-fatal geometry/tracking anomalies may prioritize a Sample for review, but they never reject, relabel, or delete it automatically.
- After Suggested for Review items are resolved, Studio should sample a small random subset of the apparently clean remainder before allowing a deliberate batch-accept action. This replaces the legacy one-by-one review workflow while still providing a lightweight check for systematic capture problems.
- Human curation changes are auditable. Review transitions are recorded as immutable Review Events while the current Review Status remains easy to query. Rejected Samples remain in the canonical dataset and are excluded when training snapshots are built; rejection is not a physical delete.
- Model-driven review suggestions are versioned. Every Model Assessment records the explicit model artifact/version that produced its prediction and class scores; newer assessments never overwrite the historical result of an older model.
- The initial uncertainty signal uses class-score margin (difference between the top two predicted classes), optionally combined with maximum score for context. The review threshold is derived from development-validation behavior rather than hard-coded as an arbitrary probability.
- The first implementation of Suggested for Review uses model disagreement and model uncertainty only. Geometry/tracking anomaly hooks remain supported by the design but are added only when a specific signal can be justified and validated from observed data.
- Suggested for Review is ranked rather than defined by one permanent magic cutoff: prediction disagreement receives highest priority, followed by increasing ambiguity from lower top-two class-score margins. Validation evidence may inform a practical review budget/cutoff without discarding the underlying continuous scores.
- Rejecting a Sample remains a low-friction action. Rejection tags are optional. Studio provides a small default tag set for common causes (for example wrong_gesture, bad_tracking, transition_frame, collection_mistake) and allows reviewers to add custom tags when none of the defaults fit; an optional free-form note may also be attached.
- A Sample's intended gesture label is immutable for the milestone. If collection was performed for the wrong gesture or the captured observation does not belong to the intended gesture, the operator rejects/discards that data and recollects rather than relabeling it in place.
- Once persisted, a Sample's core observation data is immutable for the milestone: raw landmarks, participant/session/capture provenance, hand, timestamp/frame identity, and intended gesture are historical facts. Human review state and versioned model assessments may change independently.
- The first v2 training baseline preserves compatibility with the existing 69-feature legacy contract. New raw-landmark Samples may be transformed through that compatible v1 contract and may also be reprocessed through future versioned Feature Transforms; legacy rows can only participate in transforms whose input/output contract is compatible with their already-derived 69 features.
- Feature canonicalization is owned by one versioned Feature Transform module shared by runtime inference, snapshot/training materialization, evaluation, and tests. Collector persistence stores raw Samples and does not duplicate feature-generation logic.
- New-data train/validation splits are session-aware: an individual Collection Session is assigned wholly to train or validation and is never split row-by-row across both sets. If a genuinely new P003 participant is available, all P003 sessions are reserved from model development as the preferred final held-out test set.
- Legacy data is evaluated as optional training augmentation, not as validation/test data. New-v2-only and new-v2-plus-legacy candidates are compared against the same fixed, reproducible v2 session-validation folds. The held-out P003 test remains untouched during model selection and is used only after the model-selection rule is frozen.
- Model selection uses macro F1 as the primary metric, with accuracy, per-class precision/recall/F1, and confusion matrix always reported. A material product-critical class failure may block promotion even when aggregate accuracy is higher.
- Development validation uses Collection Session as the grouping unit. Rather than choosing one permanently privileged validation session, the milestone uses leave-one-collection-session-out cross-validation over the P001/P002 development sessions: each fold holds out one complete Session for validation and trains on the remaining development Sessions. Metrics are aggregated across folds.
- The legacy augmentation comparison is repeated under the same session folds: for every fold, the without-legacy and with-legacy candidates share the exact same v2 training Sessions, held-out validation Session, transform, seed policy, and evaluation code; the only intended difference is whether compatible legacy rows are added to that fold's training data.
- Dataset sufficiency is assessed with learning curves over the development training data. The same session-grouped validation protocol is used while training with increasing fractions of the available training Samples (initially 25%, 50%, 75%, and 100%). If validation performance is still materially improving at 100%, additional collection is justified and the capture quota may be raised.
- The unseen P003 participant remains sealed throughout model development. Feature-transform choice, legacy augmentation, model architecture/hyperparameters, thresholds, and promotion criteria are frozen using only development data before P003 is evaluated once for the final milestone report.
- A training/evaluation snapshot is self-contained for ML execution: it contains both a manifest and materialized NPZ arrays representing the exact features/labels and split-relevant metadata used by the experiment. Reproducing an existing snapshot must not require re-querying the current mutable SQLite database or re-running a potentially changed Feature Transform.
- The P001/P002 development workflow uses one immutable Development Snapshot as the shared evidence base for session-cross-validation, legacy/no-legacy comparison, and learning curves. The snapshot contains the development v2 materialization, compatible legacy materialization, and the fixed session-fold definitions; individual folds and legacy inclusion are experiment configuration over that same frozen snapshot rather than separate independently-created datasets.
- P003 is never bundled into the Development Snapshot. When final evaluation is authorized after development choices are frozen, P003 is materialized into a separate immutable Final Test Snapshot.
- A promoted trained model is stored as a versioned Model Artifact rather than as an unqualified weights file. The artifact contains weights plus a manifest declaring architecture/version, exact Feature Transform identity/version and output contract, input feature count/type, label order, source snapshot identity, training configuration/seed information, and compatibility metadata. Evaluation evidence is stored alongside the artifact.
- Runtime must validate a Model Artifact's declared input/Feature Transform and label contracts before inference. A model whose declared contract does not match the runtime transform/output contract is rejected rather than executed silently.
- Snapshot and Model Artifact builders never overwrite an existing version. Any materially different dataset view, transform materialization, training run, or model output receives a newly allocated artifact version/ID.
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
| V2-020 | Snapshot Builder freezes an approved training/evaluation view | The snapshot is created | The immutable snapshot contains both its manifest and the materialized NPZ model inputs; later experiment subsets/views are derived from that frozen materialization without changing canonical observations or review history |
| V2-021 | MediaPipe returns both normalized image landmarks and world landmarks for a valid observation | Studio persists the Sample | Both landmark sets are stored canonically so later tracking/feature transforms can be regenerated without the camera frame |
| V2-022 | A canonical Sample is persisted | Its landmarks are stored | Exactly 21 landmark rows are associated with the Sample, each carrying image-normalized and world x/y/z coordinates |
| V2-023 | A Capture is paused and resumed within the same active Collection Session | Collection continues | Studio resumes the same Capture and continues toward its configured quota |
| V2-024 | An incomplete Capture has never been referenced by an immutable dataset snapshot | The operator chooses Discard | Studio may permanently delete that Capture and its Samples after explicit confirmation |
| V2-025 | A Collection Session has ended | A partial Capture from that Session still exists | The partial Capture remains historically partial; a later Collection Session creates a new Capture rather than silently continuing the old one |
| V2-026 | A Sample/Capture is referenced by an immutable snapshot | A destructive delete is requested | Studio must preserve snapshot reproducibility and must not silently hard-delete the referenced data |
| V2-027 | A collected Sample is reviewed | Its intended gesture is incorrect or unsuitable | The Sample is rejected/discarded and recollected; the milestone does not relabel it to another gesture |
| V2-028 | A new raw-landmark Sample is used by the legacy-compatible baseline | Features are materialized | The shared v1 Feature Transform produces the same 69-value contract expected by the compatible legacy/model path |
| V2-029 | A future Feature Transform is introduced | New raw Samples are available | The same canonical raw Samples may be reprocessed through the new transform without changing their stored observation data |
| V2-030 | Runtime and training/evaluation need model features | They request feature transformation | They use the same versioned Feature Transform contract rather than maintaining separate normalization/canonicalization implementations |
| V2-031 | A canonical Sample has been persisted | A later review occurs | Core observation/provenance fields remain immutable; only review state and versioned assessments may change |
| V2-032 | Multiple Collection Sessions exist for a participant | Train/validation membership is generated | Each Session is assigned wholly to one split; temporally adjacent Samples from the same Session are not divided across train and validation |
| V2-033 | P003 is genuinely absent from training/development data | Final evaluation is performed | All P003 sessions remain held out until model selection is complete and are then used as the preferred final unseen-participant test |
| V2-034 | Legacy-compatible training is evaluated | The team compares whether legacy rows help | New-v2-only and new-v2-plus-legacy candidates use the same Development Snapshot and fixed session-validation folds; legacy rows never enter validation/test |
| V2-035 | Candidate models are compared | A model is considered for promotion | Macro F1 is primary, while accuracy, per-class precision/recall/F1, and confusion matrix are reviewed before promotion |
| V2-036 | P001/P002 provide multiple Collection Sessions | Development validation runs | Each fold holds out one complete Session and trains on the remaining development Sessions; reported validation metrics aggregate the session folds rather than depending on one arbitrary holdout Session |
| V2-037 | Legacy augmentation is compared | A validation fold is evaluated | With-legacy and without-legacy candidates use the identical v2 train/validation Session membership and evaluation configuration for that fold; legacy is added only to training |
| V2-038 | Collection quota sufficiency is evaluated | A learning curve is generated | Models are trained on increasing fractions of the available development training Samples and evaluated through the same session-grouped validation protocol; continued improvement at full data is evidence to collect more |
| V2-039 | A genuinely unseen P003 test participant is available | Model development is still in progress | P003 is not consulted for feature, model, threshold, legacy-augmentation, or promotion decisions |
| V2-040 | Development decisions have been frozen | Final milestone evaluation runs | The selected final candidate is evaluated against P003 once and the result is reported as the held-out unseen-participant test |
| V2-041 | A Sample has model disagreement, low-confidence/low-margin prediction, or a non-fatal geometry/tracking anomaly | Studio builds Suggested for Review | The Sample is prioritized for human review but is not automatically rejected, relabeled, or deleted |
| V2-042 | Suggested for Review items for a Capture/group have been resolved | The operator considers batch acceptance | Studio presents a small random quality-control subset of the remaining apparently clean Samples before enabling deliberate batch acceptance |
| V2-043 | A human changes a Sample's Review Status | The decision is persisted | An immutable Review Event records the transition and context while the current Review Status is updated for efficient querying |
| V2-044 | A Sample is rejected | A training snapshot is generated | The rejected Sample remains canonically traceable but is excluded from the snapshot's eligible training membership |
| V2-045 | A Sample changes from accepted to rejected after an immutable snapshot already references it | Future data preparation runs | Existing snapshots remain unchanged for reproducibility; only newly generated snapshots apply the newer Review Status |
| V2-046 | A model evaluates a Sample for review assistance | The assessment is persisted | The assessment records the exact model artifact/version, predicted label, class scores, and evaluation time without overwriting assessments from other model versions |
| V2-047 | Studio computes model uncertainty | A Sample is ranked for Suggested for Review | Uncertainty is based primarily on the margin between the two highest class scores, with thresholds selected from development-validation evidence rather than an arbitrary fixed probability |
| V2-048 | Suggested for Review is implemented for the milestone | Automatic prioritization runs | Model disagreement and uncertainty are enabled first; geometry/tracking anomaly rules are not added until each rule has an explicit validated rationale |
| V2-049 | Studio builds a Suggested for Review queue | Multiple suspicious Samples exist | Samples are ranked with disagreement first and then by increasing model ambiguity; the review budget/cutoff may be informed by validation evidence without replacing the stored continuous assessment scores |
| V2-050 | A reviewer rejects a Sample | The rejection is committed | Rejection completes without requiring annotation; the reviewer may optionally apply one or more default/custom rejection tags and add a note for later quality analysis |
| V2-051 | An immutable Development Snapshot is created | Training/evaluation consumes it later | The snapshot includes a manifest plus materialized NPZ data sufficient to reproduce the exact ML inputs without querying the then-current SQLite database or re-running changed feature logic |
| V2-052 | P001/P002 development data and compatible legacy data are frozen | Cross-validation, legacy comparison, or learning curves run | They operate as experiment views over the same Development Snapshot and its fixed session-fold definitions rather than creating unrelated datasets for every fold/variant |
| V2-053 | P003 is still sealed | A Development Snapshot is created | P003 is absent from that snapshot; final unseen-participant data is materialized separately only after development choices are frozen |
| V2-054 | Training produces a candidate model | The candidate is persisted | A versioned Model Artifact stores weights, manifest metadata, and evaluation evidence linking it to its Feature Transform and source snapshot |
| V2-055 | Runtime loads a Model Artifact | The artifact's feature count/transform/label contract conflicts with runtime | Loading fails explicitly instead of attempting inference with a silent semantic mismatch |
| V2-056 | A snapshot or Model Artifact version already exists | Tooling produces a materially different artifact | The existing artifact is preserved and a new version/ID is created; milestone tooling does not overwrite immutable artifacts in place |

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

- Runtime performance budgets.
- Frontend delivery model (browser-hosted web UI vs packaged desktop shell) and, only after that choice, the transport/IPC mechanism between frontend and Python runtime.
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
