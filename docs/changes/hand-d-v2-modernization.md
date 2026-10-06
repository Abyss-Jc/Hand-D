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
- Tkinter is not a constraint for v2. Tauri, Electron, or another desktop shell must earn the choice through a prototype and measured behavior rather than being selected up front.
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
- The legacy dataset must not be treated as authoritative validation or final-test data because its participant/session provenance is unknown.
- Participant-held-out evaluation must account for possible identity leakage from the legacy dataset. If a participant may already exist inside legacy data, a model trained on that legacy dataset cannot honestly claim that participant was unseen.
- The current realistic collection pool is two participants, with a third participant as a desired milestone improvement.
- If a genuinely new third participant can be collected, that participant is the preferred final held-out test participant.
- If only two participants are available, the project should report session-level validation and an explicitly limited participant-level experiment rather than overstating generalization.
- New data collection should use at least two independent collection sessions for the primary participants. A session means a separate capture run with the camera/collector restarted and the participant repositioned; it does not require a different day.
- The milestone dataset will be collected primarily from each participant's selected primary/drawing hand to control collection cost. A smaller opposite-hand verification set will be retained to test whether the canonicalization actually provides left/right invariance before relying on that assumption.
- Handedness must be treated separately from gesture classification: MediaPipe supplies hand identity for drawing/modifier role assignment, while the current 69-feature gesture model intentionally omits the handedness bit and relies on canonicalization to normalize hand orientation.
- The current collector's handedness filtering semantics require correction before v2 collection: it filters the raw MediaPipe label against the requested hand while its own feature function interprets the actual hand as the opposite label. v2 should centralize handedness conversion and use one definition everywhere.

## Open decisions

- Train/validation/test split strategy using participant/session provenance.
- Exact machine-generated review signals and thresholds used to prioritize samples in Studio.
- Desktop prototype comparison and acceptance measurements.
- Real-time performance budgets for gesture latency, capture FPS, CPU/GPU use, and startup.
- Exact release/platform matrix for the October milestone.
- Migration/version policy for MediaPipe, PyTorch, Python, and packaging.
