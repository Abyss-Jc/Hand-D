# Hand-D v2 modernization

## Canonical documentation

- Product requirements: `docs/requirements/hand-d-v2.md`
- Architecture reference: `docs/reference/architecture.md`
- Delivery sequencing: `docs/changes/hand-d-v2-roadmap.md`
- Durable decisions: `docs/adr/`
- Developer operations: `docs/runbook/development.md`

## Objective

Evolve the semester project into two maintained surfaces: the **Hand-D App** for real-time gesture drawing and **Hand-D Studio** for the ML/data lifecycle. Preserve CPU as the universal fallback while validating optional hardware acceleration per supported platform.

## Confirmed direction

- The Hand-D App prioritizes low interaction latency and the freshest available gesture result. It does not need to process every camera frame.
- Hand-D Studio collection remains frame-based so collection behavior is deliberate and inspectable.
- The canonical dataset stores raw MediaPipe landmarks plus provenance and labels. The current 69-feature representation becomes a versioned derived artifact so feature extraction can be changed without recollecting samples.
- Hand-D does not store camera photos or video as part of the canonical dataset.
- Dataset curation is reversible. Human review controls whether a sample is accepted or rejected; machine-generated signals only create a Suggested for Review queue and never reject, relabel, or delete a sample automatically.
- Difficult-but-valid samples must be preserved. Low model confidence or disagreement is evidence that a sample may be informative, not evidence that it is bad data.
- Collection provenance is local dataset metadata, not remote product analytics. Studio generates a persistent anonymous device ID automatically and records platform, camera, relevant dependency versions, participant, and capture/session identifiers without requiring the operator to tag the host manually.
- The frontend consumes processed results. Camera capture, MediaPipe tracking, feature extraction, gesture inference, and temporal gesture state belong behind the runtime seam.
- Tkinter is not a constraint for v2. Tauri is the selected v2 desktop-shell direction, with Python packaged/launched as the ML/runtime sidecar. Prototype evidence is still required before treating the sidecar integration as production-ready.
- Tauri/Rust owns sidecar lifecycle; the web frontend does not directly spawn/supervise Python.
- The selected IPC direction is a Python-hosted loopback HTTP + WebSocket service. This changes the transport, not lifecycle ownership: Tauri still starts, monitors, and shuts down the sidecar.
- Sidecar discovery uses 127.0.0.1 with an OS-assigned dynamic port plus a fresh per-launch token. Python reports the assigned endpoint during readiness; no fixed localhost port is part of the v2 contract.
- Live webcam inference migrates from the current default IMAGE + blocking detect loop to MediaPipe LIVE_STREAM + detect_async. Freshness is intentional: when processing is busy, stale camera inputs may be dropped rather than queued.
- VIDEO mode is reserved for recorded/decoded video workloads. Live App and Studio camera paths use the live-stream mode so MediaPipe tracking can be reused without blocking the camera/UI loop.
- Both App and Studio retain optional camera preview. Studio uses it for collection feedback and App may preserve its current camera/dark-mode toggle. Preview encoding/transport is a separate path and never determines inference cadence.
- HTTP MJPEG is the selected v2 preview transport: Python serves the latest preview frames directly over the authenticated loopback HTTP endpoint and the webview consumes the stream without routing frame bytes through control IPC. This adopts the same data-plane separation pattern demonstrated by FaceRay while keeping Hand-D's own control protocol.
- FaceRay is now an explicit architecture inspiration for Tauri + Python/MediaPipe sidecar separation. Hand-D borrows the direct loopback MJPEG preview pattern, not FaceRay's exact stdio control protocol.
- MJPEG is performance-tuned on Tiger Lake/CachyOS and Apple Silicon M4 rather than re-litigated against WebSocket as a co-equal default. A dedicated binary WebSocket JPEG stream remains contingency only if MJPEG later fails an explicit supported-target budget; WebRTC is not a milestone baseline because current Linux WebKitGTK 2.54 disables WebRTC during its backend transition.
- CPU is the fallback on every supported platform.
- macOS Apple Silicon is a first-class target because the project must be testable on the lab's M-series Macs.
- GPU acceleration is capability-driven:
  - NVIDIA: CUDA when available and validated.
  - Apple Silicon: PyTorch MPS when available and validated.
  - AMD: ROCm only on hardware/OS combinations supported by the selected PyTorch/ROCm versions; otherwise CPU.
  - Intel: PyTorch XPU only on validated hardware when available; otherwise CPU. oneAPI support by itself does not imply that Hand-D can use PyTorch XPU on that device.
- Intel Iris Xe / Tiger Lake on the current Linux development laptop is treated as CPU-first unless a later benchmark proves a supported acceleration path.
- Project dependencies will be refreshed as part of v2, but only as a tested compatibility migration. MediaPipe, PyTorch, Python, packaging, and platform-specific acceleration versions must be selected as a mutually compatible set rather than upgraded independently.

## Current evidence

Source observations from the repository:

- The current runtime performs MediaPipe hand detection synchronously for each captured frame in `visualizer_app/gesture_engine.py`.
- The current runtime device selection is CUDA-or-CPU only.
- The checked-in dataset contains five gesture labels, while parts of the training/test tooling still describe four classes.
- The existing UI, collector, dataset viewer/purger, training script, and runtime are separate scripts but do not yet share a single explicit domain/data contract.
- The current project does not have a verified reproducible Python environment on the CachyOS host; the host Python is 3.14.7 and the global environment does not contain PyTorch.

## Delivery boundary

The October 13 milestone is a v2 vertical slice, not a complete rewrite. It should establish the new App/Studio architecture, correct the real-time runtime path, improve collection and dataset review/purge, establish reproducible training/evaluation outside the GUI, document the system, and validate the new UI direction without forcing a production desktop-shell decision prematurely.

Training and evaluation from the Studio GUI are explicitly outside this milestone. The underlying training/evaluation workflow should still become reproducible and documented.

Implementation beyond planning/documentation is deferred until the current grill phase is complete. The desktop-shell choice, IPC mechanism, dataset/session model, training/evaluation protocol, and v2 UX remain open decisions.

## Evaluation direction

- The final evaluation should answer whether Hand-D generalizes to a participant who was not used for training.
- The existing legacy dataset remains usable for training because recollecting an equivalent dataset before the milestone is not realistic.
- Legacy rows only preserve the existing derived 69-feature representation, not the raw landmarks/provenance needed to regenerate a different transform. Therefore a v2 model can combine legacy and new samples only while their feature contract remains compatible; an incompatible future transform must treat legacy data as unavailable for that model.
- The first v2 baseline deliberately preserves the existing 69-feature contract so legacy rows and newly collected raw Samples can participate in one compatible training path. New raw Samples remain cross-version reusable because future Feature Transforms can regenerate alternate feature representations from their stored landmarks; legacy rows remain limited to the compatible v1 path.
- The intended gesture of a collected Sample is immutable for the milestone. Wrong-gesture or incorrectly collected observations are rejected/discarded and recollected rather than relabeled into another class.
- Canonical SampleCore data is immutable after persistence for the milestone: raw landmarks and collection provenance remain historical facts, while ReviewState and versioned ModelAssessment records can evolve.
- Canonicalization becomes one shared versioned Feature Transform seam used by runtime inference and training/evaluation materialization. The seam hides mirror correction, translation, scaling, canonical rotation, global orientation features, and validation from callers.
- The legacy dataset must not be treated as authoritative validation or final-test data because its participant/session provenance is unknown.
- Participant-held-out evaluation must account for possible identity leakage from the legacy dataset. If a participant may already exist inside legacy data, a model trained on that legacy dataset cannot honestly claim that participant was unseen.
- The current realistic collection pool is two participants, with a third participant as a desired milestone improvement.
- If a genuinely new third participant can be collected, that participant is the preferred final held-out test participant.
- Train/validation assignment is session-aware: a Collection Session belongs wholly to one split. P003, when available and genuinely unseen, is held out entirely from model development for the preferred final test.
- Legacy usefulness is tested through two otherwise comparable training candidates: new-v2-only and new-v2-plus-legacy. Both are compared against the same fixed, reproducible v2 session-validation folds; P003 is not repeatedly consulted during that choice.
- Macro F1 is the primary promotion metric, accompanied by accuracy, per-class precision/recall/F1, and confusion-matrix review. A material failure on a product-critical class can prevent promotion even when overall accuracy is higher.
- Development validation uses leave-one-Collection-Session-out cross-validation across P001/P002. Each fold holds out one whole Session and trains on the remaining development Sessions, then metrics are aggregated across folds. This avoids privileging one arbitrary Session and prevents temporally adjacent Samples from leaking across train/validation.
- The legacy experiment is paired fold-by-fold: each with-legacy candidate and its without-legacy counterpart use the same v2 Session membership, transform, evaluation path, and seed policy; compatible legacy rows are added only to training. P003 remains sealed throughout this comparison.
- Learning curves use increasing fractions of each fold's training data (initially 25%, 50%, 75%, 100%) and the same session-grouped validation protocol. If Macro F1 is still materially rising at full data, collect more Samples rather than assuming the initial ~100 accepted Samples per gesture/session are sufficient.
- P003 remains sealed during model development. Feature-transform choice, legacy augmentation, model architecture/hyperparameters, thresholds, and promotion criteria are frozen using only the P001/P002 development protocol; the final selected candidate is then evaluated against P003 once for the milestone report.
- If only two participants are available, the project should report session-level validation and an explicitly limited participant-level experiment rather than overstating generalization.
- New data collection should use at least two independent collection sessions for the primary participants. A session means a separate capture run with the camera/collector restarted and the participant repositioned; it does not require a different day.
- Within each capture, the participant keeps the intended gesture while introducing moderate natural variation in hand position, distance, and orientation. The milestone does not require exaggerated/extreme motion conditions.
- Sessions should differ moderately through repositioning and normal environmental variation; intentionally extreme room/camera/lighting changes are outside the milestone collection protocol.
- Collection is quota-based rather than duration-based. Technically invalid detections do not count toward the quota, and each capture includes extra observations as a reserve for later curation.
- The initial capture quota is 120 observations per gesture/capture, with an approximate post-curation target of 100. This is a configurable pilot default, not a claim that 100 samples are always sufficient; learning-curve/validation evidence will determine whether more collection materially improves the model.
- Stored observations are sampled on a configurable time interval rather than a fixed frame stride, so collection density does not depend directly on whether a camera happens to run at 30 FPS or 60 FPS.
- New samples begin unreviewed. The Curator prioritizes Suggested for Review observations and supports a deliberate batch-accept action for the remaining group so curation does not require inspecting every sample individually.
- Suggested for Review combines explainable model signals (disagreement/uncertainty) with non-fatal geometry/tracking signals. These signals only prioritize human review and never mutate the Sample automatically.
- Before batch-accepting the apparently clean remainder, Studio presents a small random quality-control subset so systematic capture issues can still be noticed without returning to one-by-one review.
- Review decisions are auditable through immutable Review Events. Rejected Samples are excluded from training/snapshot eligibility but remain canonically stored and traceable; rejection is not deletion.
- Review changes are prospective for snapshot generation: an existing immutable snapshot is never rewritten when a Sample's later Review Status changes.
- Model-assisted review is versioned: every Model Assessment references the exact model artifact/version that produced its prediction and scores. Studio may choose a current review model for a curation pass, but older assessments are preserved rather than overwritten.
- Initial uncertainty ranking uses top-two class-score margin, with thresholds derived from development-validation evidence. This is used to prioritize ambiguous Samples, not to claim calibrated real-world probability.
- The milestone starts Suggested for Review with model disagreement and uncertainty only. Geometry/tracking anomaly heuristics remain an extension point and are added only when observed data justifies a specific rule.
- Suggested for Review is a ranked queue: disagreement first, then lower top-two score margin. Validation can inform how much of that ranking to review, but the system stores the underlying assessment scores rather than hard-coding one universal cutoff.
- Reject remains a one-step curation action. A small default rejection-tag vocabulary supports consistent analysis, while reviewers may add custom tags for previously unseen causes; tags and notes remain optional extra context, not a requirement to change Review Status.
- SQLite is the canonical mutable v2 dataset store for collection, provenance, review state, and Studio queries. Training/evaluation is isolated from mutable database state through immutable dataset snapshots that freeze sample membership, folds, feature-transform version, label mapping/order, reproducibility settings, and the exact NPZ materialization consumed by ML. SQLite remains canonical for collection/provenance/curation history; the snapshot NPZ is canonical only for that frozen experiment view.
- Snapshots are self-contained for ML execution: each immutable snapshot consists of a manifest plus materialized NPZ arrays representing the exact model inputs used at that point in time. Existing snapshots do not depend on re-querying future SQLite state or re-running a changed Feature Transform.
- One Development Snapshot freezes the P001/P002 development materialization, compatible legacy partition, and fixed session-validation folds. Cross-validation folds, legacy=false/true experiments, and learning-curve fractions are views/configuration over that same snapshot rather than independently-created datasets.
- P003 is excluded from every Development Snapshot. After development choices are frozen, it is materialized separately as a Final Test Snapshot for the single unseen-participant evaluation.
- Training output becomes a versioned Model Artifact (weights + manifest + metrics) rather than a bare .pth. The manifest declares the architecture, Feature Transform/input contract, label order, source snapshot, training/compatibility configuration, and the metrics file carries promotion/evaluation evidence.
- Runtime validates the Model Artifact contract before inference so a model cannot silently consume a semantically different feature representation just because the tensor dimensions happen to match.
- Canonical Samples retain both MediaPipe image-normalized landmarks and world landmarks. The legacy collector only persisted derived canonical features from world landmarks; v2 preserves the pre-feature geometry so transforms can be changed without recollecting.
- Raw landmarks are stored relationally in SQLite as 21 rows per Sample, each containing the image-normalized and world x/y/z coordinates for one landmark index. At the milestone scale this is small enough to remain simple, inspectable, and queryable.
- A Capture may pause/resume while its Collection Session remains active. If the Session ends, an incomplete Capture remains partial and a later Session creates a new Capture.
- An incomplete Capture that has never entered an immutable snapshot may be explicitly discarded with a destructive delete after confirmation. Snapshot-referenced data must not be silently deleted in a way that breaks experiment reproducibility.
- Dataset collaboration remains sequential for the milestone: one contributor edits the canonical SQLite dataset at a time, starting from the latest pulled version and committing/pushing the completed dataset change before another contributor takes ownership. This preserves the team's existing CSV-era workflow without introducing database replication or binary merge machinery.
- The milestone dataset will be collected primarily from each participant's selected primary/drawing hand to control collection cost. A smaller opposite-hand verification set will be retained to test whether the canonicalization actually provides left/right invariance before relying on that assumption.
- Handedness must be treated separately from gesture classification: MediaPipe supplies hand identity for drawing/modifier role assignment, while the current 69-feature gesture model intentionally omits the handedness bit and relies on canonicalization to normalize hand orientation.
- The current collector's handedness filtering semantics require correction before v2 collection: it filters the raw MediaPipe label against the requested hand while its own feature function interprets the actual hand as the opposite label. v2 should centralize handedness conversion and use one definition everywhere.

## Open decisions

- Exact HTTP/WebSocket protocol schema, reconnection behavior, and MJPEG quality/resolution/FPS tuning.
- Real-time performance budgets for gesture latency, capture FPS, CPU/GPU use, and startup.
- Exact release/platform matrix for the October milestone.
- Migration/version policy for MediaPipe, PyTorch, Python, and packaging.
