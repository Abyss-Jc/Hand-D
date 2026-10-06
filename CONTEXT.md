# Hand-D

Hand-D is a gesture-driven drawing system composed of a user-facing application and developer tooling for collecting, inspecting, training, and evaluating gesture data.

## Language

**Hand-D App**:
The user-facing real-time whiteboard experience. It consumes gesture results and drawing state; development tooling does not belong in this surface.
_Avoid_: Visualizer App, main GUI

**Hand-D Studio**:
The developer-facing tooling used to collect, inspect, purge, train, evaluate, and benchmark Hand-D gesture data and models.
_Avoid_: Dataset App, admin app

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
The human curation state of a Sample: unreviewed, accepted, or rejected. New samples begin unreviewed; Studio may batch-accept reviewed groups so curation does not require approving every observation individually.

**Review Event**:
An immutable audit record of a human curation transition for a Sample, such as unreviewed → accepted or accepted → rejected, including when the decision occurred and optional review context/reason.

**Feature Transform**:
A versioned transformation contract that converts a canonical raw Sample into the model-ready feature representation expected by a compatible model. Callers depend on the transform's identity and output contract, not on its internal normalization/rotation/scaling implementation.

**Collection Provenance**:
Local metadata describing where and how a Sample was collected, including an anonymous device identifier, platform, camera, relevant software versions, participant, and capture/session identifiers. It is stored with the dataset and is not remote analytics telemetry.

**Collection Session**:
An independent collection run for one participant, started after repositioning/restarting the collector so it represents a distinct capture context.

**Capture**:
A contiguous collection segment for one target gesture and hand inside a Collection Session.
