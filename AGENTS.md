# Hand-D Contributor and Agent Guide

## Branch discipline

- `main` is protected by process: do not implement Hand-D v2 directly on `main`.
- Current v2 work belongs on `feature/hand-d-v2-modernization` unless a deliberately scoped child branch is created.
- Keep commits granular enough that dependency, data, runtime, Studio, or UI experiments can be reverted independently.
- Never discard unrelated user changes.

## Canonical documentation

Start at `docs/README.md`.

- Requirements: `docs/requirements/`
- Current/target technical reference: `docs/reference/`
- Active plans and migrations: `docs/changes/`
- Durable trade-off decisions: `docs/adr/`
- Repeatable operating/setup procedures: `docs/runbook/`
- Domain vocabulary: `CONTEXT.md`

`SPEC.md` is legacy context, not v2 source of truth. Do not update it merely for visual consistency. Move verified facts into the appropriate canonical document.

When documenting:

- cite source as `path:line` or a concrete command/result;
- mark unexecuted or unavailable checks as `No verificado`;
- do not turn an assumption into a fact;
- create an ADR only for a durable choice with real alternatives and meaningful reversal cost.

## Architectural boundaries

- **UI language is English-only** in Whiteboard and Studio, including dynamic model/runtime status, empty states and accessibility labels (ADR 0005, V2-142). Internal planning documents may be bilingual.
- **Hand-D App** is the user-facing whiteboard.
- **Hand-D Studio** is the advanced project/data surface for collection, inspection, curation, model/workspace management, and dataset preparation; it is not developer-only.
- Training/evaluation must be reproducible but is outside the Studio GUI milestone.
- Shared landmark transforms, handedness semantics, data schemas, model metadata, and backend policy must not be duplicated across App and Studio implementations.
- The frontend consumes normalized runtime results and owns canvas/document mapping, smoothing/interpolation, strokes, undo/redo, and tool state; camera/MediaPipe/Feature Transform/model execution belongs behind the runtime boundary.
- Tauri + Python sidecar is the accepted v2 desktop-shell direction. Prototype evidence still gates packaging/runtime details, but do not reopen Tauri vs Electron without new evidence.

## Data and ML rules

- Canonical v2 observations store raw landmarks + provenance + intended label, human Review Status, and independent Sample Lifecycle Status; never store photos/video in the canonical dataset.
- Feature vectors are derived and versioned.
- Preserve difficult-but-valid samples.
- Automatic confidence/disagreement/quality signals may prioritize `Suggested for Review`; they must not automatically reject or relabel data.
- Prefer out-of-sample/OOF Model Assessments for review assistance; do not use predictions from a model trained on the same Sample as authoritative uncertainty evidence.
- Legacy data can help training but cannot be used to claim clean participant-held-out evaluation.
- Prefer participant/session-aware splits over row-random splits.
- Keep model label order, feature-transform version, and architecture compatibility with the model artifact.
- After development choices freeze, refit the selected configuration on all eligible development data before the one-time sealed participant test; do not feed the final-test participant back into the milestone model.
- PyTorch is the training baseline. Treat ONNX Runtime as a deployment candidate until numerical/performance/package evidence selects it.

## Platform rules

- CPU is always a valid fallback.
- Training should use validated acceleration when beneficial. Runtime acceleration/provider choice is benchmark-driven and must preserve CPU fallback.
- Do not infer PyTorch support from the presence of a vendor SDK alone.
- Tests that gate normal development should not require camera or GPU hardware unless explicitly marked as integration/platform tests.

## Implementation workflow

1. Read relevant source and canonical docs before editing.
2. Update/add a failing test when practical for behavioral changes.
3. Make the smallest coherent implementation.
4. Run focused tests first, then broader applicable checks.
5. Update reference/change/runbook docs only with evidence from what changed or what was actually executed.
6. Review the diff before committing.

For v2 Python work, migrate dependency/environment management to `uv` + `pyproject.toml` + `uv.lock`; do not install project packages into system Python. The current host has `uv 0.12.23` available. Until that migration lands, existing legacy environments remain transitional evidence only.
