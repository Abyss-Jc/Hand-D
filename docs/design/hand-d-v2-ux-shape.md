# Hand-D v2 — UX Shape v0.1

> Design handoff, not implemented UI. Grounded in decisions Q90–Q115 and the approved intuitive, functional neo-brutalist direction; no new user questionnaire.

## Experience hierarchy

**One desktop application, two spaces:** **Whiteboard** first (default route, usable without a workspace), **Studio** second (Overview, Collect, Dataset, Models, Workspace). Easy Mode simplifies density further but the normal UI must be usable without it.

## 1. Whiteboard — camera-first

```text
┌ Hand-D ─────────────────── Whiteboard | Studio ──────────┐
│ New   Open   Save   Export   Undo   Redo      Settings   │
├─────────────────────────────────────────────────────────┤
│                                                 Camera  │
│   Live camera background + editable drawing      [ON]  │
│                                                         │
│  Tools                                    Hand feedback  │
│  Pen / Erase / Color                       Index • Draw  │
│                                                         │
│                 Visible canvas stays editable           │
├─────────────────────────────────────────────────────────┤
│ Camera: Ready    Model: baseline    Drawing hand: Right  │
└─────────────────────────────────────────────────────────┘
```

- Camera-on drawing overlay is default. An obvious toggle switches to a clean/dark canvas **without modifying strokes or stopping inference**; hiding MJPEG stops unnecessary preview encoding.
- File actions: New / Open / **Save native editable document** / Save As; **Export is separate** (SVG now, other formats later). Avoid the browser term Download for ordinary persistence.
- Tools and core editing also work with mouse/keyboard, including familiar undo/redo shortcuts. Undo history is per session; crash recovery in app-data is separate from a saved drawing.
- Compact, legible status: current stable gesture/action, selected Drawing Hand, active model, camera/sidecar health. Do not crowd the canvas with diagnostic telemetry.
- If the camera/sidecar fails, show a specific recovery action and keep the current drawing editable. No auto-swap between Drawing and Modifier hands; isolated wrong-class frames must not switch eraser/pen.

## 2. Studio Overview — what to do next

```text
┌ Studio: Overview | Collect | Dataset | Models | Workspace ┐
│ Workspace: My project      Active Model: model-04        │
│                                                          │
│ [Collect another session]   [Review pending samples]     │
│                                                          │
│ Sessions/gestures       Coverage gaps       Runtime       │
└──────────────────────────────────────────────────────────┘
```

- Two prominent next actions at most. Compact state summaries, not an analytics wall.
- If no workspace exists, offer Create/Open Workspace without gating Whiteboard.

## 3. Studio Collect — guided session

Select/create Participant → Start/Resume Collection Session → gesture checklist → choose gesture/hand → live camera preview → Start/Pause/Finish Capture → next gesture. Preserve Participant/Session context throughout. Generate IDs/timestamps/device metadata automatically. Default target: 120 observations per Capture, adjustable, with time-based sampling.

Creating a new gesture makes it collectable immediately. Studio can assign a built-in role-aware Whiteboard action, but indicates **Needs compatible trained model** until recognition exists.

## 4. Studio Dataset — inspect, review, freeze

- Browse: compact filters for Gesture, Participant, Session, Review and Lifecycle; open a sample to see tracking/provenance.
- Review: focused Suggested for Review queue with **Accept / Reject / Drop** (Drop is independent of Review), optional reasons, deliberate batch acceptance; never automatic rejection.
- Analytics: only class balance, session/hand coverage and actionable collection/review gaps by default; further breakdowns behind drill-down.
- Build Snapshot: auto-generated immutable ID and optional note, derived protocol defaults; **Blockers** for integrity/contract, **Warnings** for coverage. Only accepted + active Samples are eligible; unreviewed excluded with warning.
- After successful creation show **View Snapshot** and **Prepare Training**. No edit-in-place and no implicit training.

## 5. Studio Models — evidence and intentional activation

- Show Active Model and compatible Candidates, model/source compatibility, and explicit **Set as Active**. Existing active model never changes automatically on training; first compatible model may bootstrap an empty workspace.
- Evaluation defaults to Macro F1, per-class health, confusion matrix, learning curve and performance. Fold history, advanced score/margin, artifact provenance and training history behind details.
- Training & Evaluation shows the selected immutable Snapshot, derived config and reproducible command/copy action; CLI runs training for the October 13 tracer, not a big in-app training control.
- Gesture Mappings: role-aware label → **implemented built-in action** (Draw, Erase, Straight Line, Adjust Tool, No Action; future catalog actions only when implemented). Unmapped recognizable labels are shown but inert; no scripts/macros.
- Import Model validates and copies an artifact into the workspace; bare legacy weights need an explicit migration path.

## 6. Studio Workspace — simple project operations

Create/Open/Recent workspaces via native directory picker; show project health/path, version and model info. Dataset, snapshots, training policy, gesture action mappings and Active Model are portable workspace state. Camera, Easy Mode, window and backend override preferences remain local per device. Unsupported future workspace/schema is never downgraded or modified.

## Interaction and visual design rules

- **Visual direction: functional neo-brutalism.** Off-white/ink foundation, thick high-contrast outlines, crisp rectangular controls, visible offset shadows, bold oversized display typography, electric-lime primary actions and a sparing cobalt/coral secondary palette. Brutalist personality must never reduce legibility, hierarchy, or accessibility.
- **First-glance comprehension:** the Whiteboard's initial state includes a short numbered "Empieza aquí" hint: (1) position the Drawing Hand in view, (2) use Index_Finger to draw / Fist to erase (and the modifier-hand affordances when appropriate), (3) Save for an editable file. Pair instructions with visible controls/icons and an optional dismiss action; do not block the canvas with a forced tutorial.
- **Explain affordances rather than jargon:** descriptive labels on primary controls (Cámara / Lienzo limpio, Guardar, Exportar, Dibujar / Borrar), active-state feedback, a hand-roles legend, and one context-sensitive next step per Studio view.
- **Prototype:** [standalone interactive HTML mock](../../prototypes/hand-d-ux-shape.html) for visual inspection. This is simulated UI, not the Tauri/runtime implementation; no fake camera/training results should be presented as real.
- **Progressive disclosure:** one dominant action per step; advanced controls accessible without blocking normal task completion. Technical IDs in details, not primary labels.
- **Accessible desktop:** high-contrast focus states, keyboard navigation, clear accessible labels, comfortable controls, responsive sidebar collapse for narrow desktop windows.
- **Explicit empty/degraded states:** no workspace, camera denied, runtime restarting, model incompatible, no accepted Samples, unreviewed backlog, snapshot blockers, unmapped new gesture. Every state tells the user the next safe action.
- **Protect invariants:** Save is not Export; Preview toggle changes no drawing data; Snapshot is never editable; training reads snapshot only; Review and Drop are separate; new model activation is deliberate.

## Five short acceptance journeys

| Journey | Pass condition |
|---|---|
| Launch and draw | Whiteboard defaults to camera overlay; camera toggle leaves strokes intact; editable Save is separate from Export |
| Capture | Participant/Session persists through gesture checklist with visible capture progress |
| Curate and freeze | Human-reviewed accepted+active samples build an immutable snapshot; quality gaps warn but integrity failures block |
| Train and select | Prepare Training provides exact command; resulting Candidate does not displace existing Active Model |
| Use custom gesture | After training and activating compatible model, mapped built-in action works; unmapped label is inert |

## October 13 cut

**Tracer:** Tauri Whiteboard shell with camera-on/clean toggle and preserved frontend canvas, Python HTTP/WS/MJPEG feedback, one Studio Collect → Review → Snapshot path, reproducible CLI training/evaluation, and resulting model in runtime. Rich Studio analytics, full document/export UI polish, custom gesture mapping editor, and release-quality cross-platform UX remain broader v2 follow-ups unless needed to prove the tracer.

Next: one brief adversarial consistency review, then **to-spec → to-tickets → TDD**. Reopen product questions only for true implementation blockers.
