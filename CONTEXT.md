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

**Sample**:
A single frame-based hand observation stored as raw MediaPipe landmarks plus provenance and its intended gesture label. It does not contain a photo or video frame.

**Curated Sample**:
A Sample whose human review state determines whether it may be used for training. Machine-generated quality signals may place samples into a Suggested for Review queue, but they do not reject or relabel samples automatically.

**Collection Provenance**:
Local metadata describing where and how a Sample was collected, including an anonymous device identifier, platform, camera, relevant software versions, participant, and capture/session identifiers. It is stored with the dataset and is not remote analytics telemetry.
