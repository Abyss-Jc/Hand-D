# Hand-D v2 — October 13 Tracer Specification

> Implementation-ready cut of the approved requirements. **Status:** planned; this document never substitutes for passing tests. Source of truth for broader product behavior remains `docs/requirements/hand-d-v2.md` and `docs/design/hand-d-v2-ux-shape.md`.

## Outcome / non-goals

Demonstrate one traceable flow: **Participant → Collection Session → Capture → canonical SQLite Sample → human Review/Drop → immutable Development Snapshot → grouped development evaluation + final-refit Model Artifact → fresh runtime gesture/tracking result → minimal camera-first Tauri Whiteboard/Studio**. Report real execution evidence, not a v2-complete claim.

Not a gate: polished Studio analytics, full GUI training, flexible gesture-mapping editor, complete native drawing save/export UX, MLflow, arbitrary macros, all GPU backends, or Windows hardware validation. Expanded visualizer modal is in the approved UX target; its Tauri integration may be a follow-up if necessary to protect the vertical slice.

## Functional seams and contract tests

| Seam | Input → Output | Invariant / TDD evidence |
|---|---|---|
| Feature Transform v1 | World landmarks `(21,3)` + raw MediaPipe Left/Right → float32 `(69,)` or invalid | Existing wrist translation / raw-Left X-mirror / MCP scale / palm frame / 63+3+3 features preserved; malformed/NaN/degenerate observations never become training inputs; fixtures agree with legacy valid output |
| Canonical dataset | Participant/Session/Capture + image/world 21×3 + provenance → SQLite Sample | No camera photos or videos stored; new Sample review=unreviewed, lifecycle=active; source observations and provenance immutable |
| Human curation | Sample + Review Event / Lifecycle Event → current state | Review and soft Drop independent/auditable/reversible; no automatic rejection; accepted AND active is the only normal snapshot eligibility |
| Snapshot Builder | SQLite + frozen config + optional compatible legacy partition → immutable manifest/NPZ | Auto-ID, optional note, contract blockers vs quality warnings; no unreviewed/rejected/dropped membership; no overwrites; same source/config reproducible membership |
| ML experiment | frozen Development Snapshot → session-held-out CV, OOF scores, learning curves and final refit | P001/P002 LOSO with no same-session leakage, legacy/no-legacy paired folds, Macro F1/per-class/confusion evidence; refit on all eligible development only; P003 (if present) sealed test once |
| Model Artifact | trained checkpoint + metrics + config → immutable directory | Label order, feature contract, artifact role, training source, manifest and compatibility verified before runtime; new candidate does not displace existing Active Model |
| Runtime | camera + MediaPipe LIVE_STREAM + model → normalized hand position + stable gesture and health | newest result wins, stale frames bounded/dropped, stable hand roles, confidence shown as score/margin rather than calibrated probability, CPU fallback functional |
| Desktop seam | Rust/Tauri supervises sidecar; loopback HTTP/WS + MJPEG → frontend | dynamic localhost port and launch token, restart/resync ignores stale session events, camera preview data separated from control, existing frontend drawing survives sidecar restart |
| Minimum UX | Whiteboard + thin Studio Collect/Review/Snapshot/Models | Camera-first with toggle, nonblocking hint and functional neo-brutalist controls; no forced workspace setup to draw; user-controlled Model activation; no implicit training |

## Demo path / acceptance

1. On Linux, run no-camera contract and schema tests from a reproducible local `uv` environment; record Python/library versions and command evidence.
2. Create/select workspace → participant/session/capture → store actual raw landmark Samples with provenance; inspect one persisted Sample.
3. Accept some Samples, reject/drop others; prove Review and Lifecycle history and snapshot eligibility (unreviewed excluded with warning).
4. Build frozen snapshot, then mutate curation and prove earlier snapshot hash/membership did not change; create a second snapshot for changed membership.
5. Train/evaluate five labels from the frozen snapshot via a reproducible CLI. Record grouped CV, OOF membership assertions, metrics, final-refit artifact; never represent legacy or development folds as unseen-participant final evidence.
6. Load a compatible artifact for real-time inference; verify normalized freshest tracking, camera feedback, and Tauri shell/sidecar IPC on Linux. Confirm Whiteboard canvas remains accessible during camera/sidecar failure.
7. End with passing-test commands, measured/runtime evidence, screenshots or demo notes, verified platforms, and explicitly missing P003/M4/Windows-hardware evidence where applicable.

**Failure handling:** invalid landmarks are rejected before persistence/transform; unsupported artifact or workspace version never silently loads/writes; native app retains document on inference failure; no snapshot or Model Artifact is rewritten in place.

## Sequencing / risk

Implement tests and domain contracts first; then canonical data + snapshots; then training/OOF; then live runtime and Tauri integration. The first production refactor should preserve the exact legacy feature-vector semantics for valid inputs. Dependency changes (MediaPipe 1.1.x, ONNX Runtime provider) are gated by compatibility/latency evidence and are not prerequisites for the CPU Linux tracer.

Only escalate decisions if implementation demonstrates a genuine domain contradiction or blocks the end-to-end acceptance path. Do not reopen approved style, App/Studio, data immutability, model activation, or preview architecture.
