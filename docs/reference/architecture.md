# Hand-D Architecture Reference

## Alcance

This reference separates **verified current architecture** from the **target v2 boundaries**. Target sections are design direction, not claims that the code already implements them.

## Estado actual

### Runtime application

`visualizer_app/main.py` owns the Tkinter application, user configuration, UI tick loop, camera/canvas rendering, and lifecycle of `GestureEngine`.

`visualizer_app/gesture_engine.py` currently owns:

- webcam capture;
- MediaPipe Hand Landmarker creation;
- per-frame synchronous `detect(...)`;
- handedness correction and drawing/modifier role assignment;
- landmark canonicalization;
- PyTorch gesture inference;
- delivery of `GestureResult` to the UI.

The current classifier is a 69 → 128 → 64 → 5 MLP. Its runtime backend is CUDA when `torch.cuda.is_available()`, otherwise CPU.

### Data collection

`dataset_extraction_tools/data_extractor.py` captures webcam frames, processes every configured stride, canonicalizes a detected hand, and appends feature rows to a CSV. The CSV stores 69 features plus handedness and label.

The collector currently contains inconsistent handedness semantics: filtering compares the requested hand to MediaPipe's raw label, while the feature function separately computes the actual hand as the opposite label after mirror correction.

### Dataset curation

`dataset_extraction_tools/data_view_3d.py` is an interactive Matplotlib viewer/purger over the CSV representation. Its current save path is not a durable curation model; v2 will replace destructive row editing with review-state-driven dataset generation.

### Training

`model_training/gesture_classifier.py` is stale relative to the runtime:

- it defines four classes rather than the current five;
- it performs row-level random splitting;
- its validation loader points to the training subset;
- its CPU fallback expression is invalid;
- it does not provide a reproducible participant/session-aware evaluation protocol.

This file is evidence of technical debt, not a canonical training specification.

## Target v2 boundaries

```mermaid
flowchart LR
    CAM[Camera] --> RT[Runtime Core]
    RT --> UI[Hand-D App Frontend]

    STUDIO[Hand-D Studio] --> DATA[Dataset Core]
    DATA --> SB[Snapshot Builder]
    SB --> SNAP[Development / Final Test Snapshots]
    SNAP --> TRAIN[Training / Evaluation]
    TRAIN --> MODEL[Versioned Model Artifact]
    MODEL --> RT

    COLLECT[Studio Collector] --> DATA
    CURATE[Studio Curator] --> DATA
```

### Hand-D App

The App is the product surface. It consumes processed runtime results and drawing state. It should not own ML training, dataset curation, or direct knowledge of training storage.

### Runtime Core

The runtime owns camera access, MediaPipe processing, shared feature transformation, gesture inference, temporal gesture behavior, hardware backend selection, and lifecycle/error states. The App frontend consumes results from this boundary.

For the App path, freshness is more important than exhausting every camera frame.

### Hand-D Studio

Studio is a developer-facing surface for:

- participant selection;
- collection sessions/captures;
- dataset inspection;
- Suggested for Review;
- accepted/rejected curation;
- dataset export/preparation.

Training/evaluation should be invokable reproducibly from tooling/CLI for the milestone, but is not required inside the Studio GUI.

### Dataset Core

The canonical observation is raw landmark data plus provenance and human labeling/review state. The canonical working store is SQLite so Studio can query, curate, and update this state transactionally. A versioned transform produces the feature representation used by a specific model.

Conceptual relationships:

```mermaid
flowchart TD
    P[Participant] --> S[Collection Session]
    D[Anonymous Device] --> S
    S --> C[Capture]
    C --> O[Sample / Observation]
    O --> R[Human Review State]
    O --> F[Versioned Feature Transform]
    F --> SB[Snapshot Builder]
    SB --> SNAP[Immutable Snapshot: manifest + NPZ]
    SNAP --> T[Training / Evaluation]
    T --> MA[Versioned Model Artifact]
    M[Model Version] --> A[Model Assessment]
    O --> A
```

The operator should only need to choose participant, gesture, hand, and collection target. Session IDs, capture IDs, timestamps, anonymous device identity, platform, camera, and relevant versions are generated/recorded automatically.

Canonical landmark storage is relational at the current scale:

```text
samples
└── sample_id

landmarks
├── sample_id
├── landmark_index        # 0..20
├── image_x / image_y / image_z
└── world_x / world_y / world_z
```

One Sample therefore owns exactly 21 landmark rows. This representation is intentionally inspectable and testable; vectorized arrays are produced later by the feature/snapshot pipeline.

### Capture lifecycle

A Capture represents one continuous collection context inside a Collection Session. It may be paused and resumed while that Session remains active. If the Session ends, an incomplete Capture remains partial and a later Session starts a new Capture.

An incomplete Capture that has never been referenced by an immutable snapshot may be explicitly discarded and physically deleted with its Samples. Once data is referenced by a snapshot, destructive cleanup must not invalidate the snapshot's ability to reproduce the historical training/evaluation view.

### Feature Transform seam

Feature canonicalization is a shared deep module. Its interface exposes the transform identity/version and the model-ready output contract; mirror correction, translation, scale normalization, canonical-frame rotation, global orientation features, and validation remain implementation details behind the seam.

The current legacy-compatible transform produces the existing 69-value feature contract. New raw Samples can be reprocessed through this transform or through future transforms because their image/world landmarks are preserved. Legacy CSV rows do not have raw landmarks, so they can only participate where their stored 69-feature representation is contract-compatible.

Snapshots reference the Feature Transform contract used to materialize model input; they do not duplicate or expose the transform's internal geometry algorithm.

### Snapshot Builder

Snapshot Builder is the boundary between mutable Dataset Core state and reproducible ML execution. It reads an approved historical view from SQLite, applies the selected Feature Transform exactly once during materialization, and emits an immutable snapshot. Training/evaluation does not query SQLite directly after the snapshot exists.

The milestone snapshot consists conceptually of a manifest.json plus dataset.npz.

The manifest records the snapshot identity, selected Sample membership, participant/session membership, label order, Feature Transform identity/version/output contract, session-validation fold definitions, legacy availability/membership, seed/reproducibility configuration, and source revision/integrity metadata.

The NPZ contains the exact materialized model inputs associated with that manifest, including feature arrays, labels, stable source identifiers, and split/fold metadata needed by training/evaluation. For v2 Samples these arrays are derived from canonical raw landmarks at snapshot creation. Compatible legacy rows remain a separate materialized partition because they have only the existing derived 69-feature contract and cannot be regenerated through future incompatible transforms.

The Development Snapshot is shared across the P001/P002 development protocol. It freezes the v2 development materialization, compatible legacy materialization, and fixed Collection-Session folds. Selecting a fold, choosing legacy=false or legacy=true, and choosing a learning-curve fraction are experiment configuration over this same immutable evidence base rather than separate independently-created datasets.

This guarantees that with-legacy and without-legacy comparisons change the intended variable instead of silently changing validation membership.

P003 is never materialized into the Development Snapshot. Once all development/model-selection decisions are frozen, Snapshot Builder may create a separate Final Test Snapshot for P003 and that snapshot is used for the single held-out final evaluation.

An existing snapshot is self-contained for ML execution. Later SQLite review changes or Feature Transform implementation changes do not alter its manifest or materialized NPZ. New canonical decisions require a new snapshot.

Immutability is enforced by creation semantics rather than filesystem permissions: Snapshot Builder never overwrites an existing snapshot ID. If membership, transform, folds, or materialized content changes, it allocates a new snapshot version.

### Model Artifact

Training output is a versioned bundle rather than an unqualified weights file. The milestone bundle consists conceptually of weights.pth, manifest.json, and metrics.json.

The model manifest declares, at minimum:

- model identity and architecture/version;
- Feature Transform identity/version and expected output contract;
- expected input dtype/feature count;
- label mapping/order;
- source Development Snapshot identity;
- training configuration and seed information;
- legacy inclusion policy and relevant experiment configuration;
- compatibility/runtime metadata.

The metrics document carries the evaluation evidence associated with the artifact, including aggregate session-cross-validation Macro F1, accuracy, per-class precision/recall/F1, confusion-matrix data, learning-curve summary, and other promotion evidence produced by the finalized protocol.

Runtime treats the model manifest as a compatibility contract. It validates the expected Feature Transform/input and label mapping before loading the model for inference. A dimensionally compatible but semantically incompatible model must not be allowed to run merely because its tensor shape happens to match.

Model Artifact creation follows the same rule: an existing model version is never replaced in place. A new training output receives a new model ID/version.

### Model Assessment

Prediction confidence and class scores belong to the pair **sample + model version**, not permanently to the sample itself. Model disagreement or low confidence can prioritize review, but does not imply invalid data.

Each assessment references the exact Model Artifact that produced it. Studio may designate one model version as the current review model for a curation pass, but historical assessments from prior models remain immutable and queryable.

The first uncertainty ranking uses the margin between the top two class scores. A low margin indicates that the model's decision is ambiguous relative to its nearest competing class; the threshold is calibrated from development-validation behavior rather than treated as a universal probability claim.

Suggested for Review is exposed as a ranked queue: explicit model disagreement ranks ahead of otherwise-correct but low-margin predictions. The stored Model Assessment keeps the underlying scores/margin; any UI review budget or cutoff is a view over that evidence rather than a mutation of the Sample.

### Curation state

Human curation is represented by an append-only Review Event history plus an efficiently queryable current Review Status. Automatic signals may prioritize a Sample in Suggested for Review but do not mutate its human review state.

Rejection annotation is optional. Studio exposes a small default tag vocabulary for common causes and permits custom tags when new failure modes appear. Tags are review context rather than Sample truth, and an optional free-form note may accompany them.

Rejected Samples remain part of the canonical historical dataset and are filtered out when eligible snapshot membership is materialized. Physical deletion is reserved for the previously defined explicit discard path for incomplete, snapshot-unreferenced Captures.

Review changes never rewrite an existing immutable snapshot. They affect eligibility only when a later snapshot is created.

### Platform backend policy

CPU is always a valid fallback.

Preferred acceleration is capability-driven:

- NVIDIA: CUDA where supported;
- Apple Silicon: MPS where supported;
- AMD: ROCm only on supported hardware/OS/runtime combinations;
- Intel: PyTorch XPU only on validated hardware/runtime combinations.

Vendor SDK support alone is not enough to claim that Hand-D supports a PyTorch backend on a given machine.

## Interfaces y datos

Target interfaces are conceptual until implementation begins:

- **Runtime result**: gesture labels, confidence/scores when available, hand identity/role, landmarks/tracking position, timing/health state, and optional preview data.
- **Canonical sample**: raw MediaPipe landmarks, target gesture, participant ID, hand, session/capture IDs, local provenance, timestamp/frame index, human review state.
- **Derived feature sample**: source sample ID, transform/version ID, model-compatible feature vector.
- **Model assessment**: source sample ID, model version, predicted label, confidence/class scores, evaluation timestamp.
- **Model artifact**: versioned weights plus manifest and metrics, including label order, input feature contract, Feature Transform version, architecture/version, source snapshot, experiment configuration, and compatibility information.
- **Development snapshot**: immutable manifest + NPZ materialization for P001/P002 development and compatible legacy data, with fixed session-validation folds, feature/label contracts, seed/configuration, and integrity/version information.
- **Final test snapshot**: separate immutable manifest + NPZ materialization for the sealed P003 evaluation after development choices are frozen.

SQLite is the canonical mutable working store. A snapshot freezes the exact training/evaluation view of that store. Its NPZ is the immutable materialized ML input for that snapshot, while SQLite remains the canonical source of collection provenance and evolving curation state. Regenerating a materially different NPZ/manifest creates a new snapshot rather than rewriting an existing one.

## Operación y límites

- No canonical sample stores a photo or video frame.
- Local collection provenance is not remote analytics telemetry.
- Legacy CSV data may be used for training, but cannot prove unseen-participant generalization.
- A third genuinely new participant is the preferred held-out final test participant.
- The primary collection path may use each participant's selected main hand; a smaller opposite-hand verification set checks the left/right normalization assumption.
- The desktop shell remains deliberately undecided until prototype evidence exists.

## Fuentes y verificación

| Afirmación | Fuente | Revisado el | Estado |
|---|---|---|---|
| Runtime performs synchronous `detect(...)` in its camera loop | `visualizer_app/gesture_engine.py:347-374` | 2026-10-05 | Verificado por source |
| Runtime MLP has five outputs and 69 input features | `visualizer_app/gesture_engine.py:157-178` | 2026-10-05 | Verificado por source |
| Runtime device policy is CUDA-or-CPU | `visualizer_app/gesture_engine.py:181-196` | 2026-10-05 | Verificado por source |
| Runtime model excludes handedness from its 69-feature input | `visualizer_app/gesture_engine.py:146-149` | 2026-10-05 | Verificado por source |
| MediaPipe handedness is used separately to assign drawing/modifier roles | `visualizer_app/gesture_engine.py:398-433` | 2026-10-05 | Verificado por source |
| Collector writes 69 features + handedness + label | `dataset_extraction_tools/data_extractor.py:197-207` | 2026-10-05 | Verificado por source |
| Curator drops a row in memory, then concatenates the original file back during save | `dataset_extraction_tools/data_view_3d.py:181-204` | 2026-10-05 | Verificado por source |
| Training script is currently four-class and row-random-split | `model_training/gesture_classifier.py:43-86` | 2026-10-05 | Verificado por source |
| Target App/Studio/data boundaries | `docs/changes/hand-d-v2-modernization.md` | 2026-10-05 | Decisión aprobada, aún no implementada |
