# Hand-D

Hand-D is a gesture-driven drawing system composed of a user-facing whiteboard and an advanced Studio for collecting, inspecting, curating, evaluating, and managing gesture data/models.

## Language

**Hand-D App**:
The user-facing real-time whiteboard experience. It consumes gesture results and drawing state; development tooling does not belong in this surface.
_Avoid_: Visualizer App, main GUI

**Hand-D Studio**:
The advanced project/data workspace inside the same Hand-D desktop application. It is used to select/manage a Project Workspace, configure the active model/runtime inputs, collect Samples, inspect/curate data, and access advanced project controls. It is not restricted to software developers.
_Avoid_: Dataset App, admin app

**Project Workspace**:
A user-selected portable writable Hand-D project directory that owns the canonical SQLite dataset plus project-scoped snapshots, generated Model Artifacts, reports, and workspace configuration. Workspace-internal references use relative paths so the directory can be moved, copied, cloned, or used directly as a Git repository without changing its logical structure. It is separate from the installed application bundle and can be opened by both source and packaged Hand-D builds.

**Workspace Format Version**:
The version of the portable workspace layout/config contract. It is independent from App SemVer and from the SQLite Schema Version. Supported older formats migrate forward only; Hand-D never automatically downgrades a workspace.

**SQLite Schema Version**:
The version of the canonical workspace database schema. It can evolve independently from the surrounding Workspace Format Version and is migrated forward transactionally with recoverable backup/checkpoint and validation.

**Easy Mode**:
A progressive-disclosure UI mode, disabled by default, that reduces technical density and exposes safer/high-level controls without changing the underlying workspace, data model, runtime contracts, or artifact compatibility. Advanced controls remain available by switching presentation mode rather than through a different application.

**Drawing Hand**:
The hand whose gesture directly controls drawing and erasing actions.

**Modifier Hand**:
The hand used to modify the Drawing Hand's action, such as changing thickness or activating ruler behavior.

**Participant**:
A pseudonymous person who contributes collection data, identified by a project-local ID such as P001 rather than a real name.

**Sample**:
A single frame-based hand observation stored as raw MediaPipe landmarks plus provenance and its intended gesture label. It does not contain a photo or video frame.

**Curated Sample**:
A Sample whose human review state determines whether it may be used for training. Machine-generated quality signals may place samples into a Suggested for Review queue, but they do not reject or relabel samples automatically.

**Review Status**:
The human quality/curation decision for a Sample: unreviewed, accepted, or rejected. New samples begin unreviewed; Studio may batch-accept reviewed groups so curation does not require approving every observation individually. Rejected means a reviewer determined that the Sample is unsuitable for training.

**Sample Lifecycle Status**:
Whether a Sample participates in normal active dataset views: active or dropped. Dropped is a reversible soft-delete state that preserves the canonical Sample/provenance and its review decision while excluding it from normal active views and future snapshot eligibility until restored.

**Review Event**:
An immutable audit record of a human Review Status transition for a Sample, such as unreviewed → accepted or accepted → rejected, including when the decision occurred and optional review context/reason.

**Lifecycle Event**:
An immutable audit record of a Sample Lifecycle Status transition, such as active → dropped or dropped → active.

**Feature Transform**:
A versioned transformation contract that converts a canonical raw Sample into the model-ready feature representation expected by a compatible model. Callers depend on the transform's identity and output contract, not on its internal normalization/rotation/scaling implementation.

**Development Snapshot**:
An immutable, self-contained ML view of the accepted P001/P002 development data plus optional compatible legacy features. It contains a manifest describing exact membership, session-validation folds, transform/label contracts, and experiment configuration together with materialized NPZ arrays representing the exact model inputs used by training/evaluation.

**Final Test Snapshot**:
An immutable snapshot created for the sealed P003 unseen-participant evaluation after development choices are frozen. P003 is not included in the Development Snapshot.

**Model Artifact**:
A versioned model bundle containing the trained model plus metadata that declares architecture, runtime format(s), input/Feature Transform contract, label order, source snapshot, training configuration, and evaluation evidence. Runtime validates this contract before using a compatible deployment representation.

**Model Assessment**:
A versioned prediction/score record for one Sample produced by one explicit Model Artifact, including whether the Sample was outside that model's training membership. Out-of-sample assessments are preferred for Suggested for Review evidence.

**Collection Provenance**:
Local metadata describing where and how a Sample was collected, including an anonymous device identifier, platform, camera, relevant software versions, participant, and capture/session identifiers. It is stored with the dataset and is not remote analytics telemetry.

**Collection Session**:
An independent collection run for one participant, started after repositioning/restarting the collector so it represents a distinct capture context.

**Capture**:
A contiguous collection segment for one target gesture and hand inside a Collection Session.
