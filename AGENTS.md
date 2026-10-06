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

- **Hand-D App** is the user-facing whiteboard.
- **Hand-D Studio** is developer tooling for collection, inspection, curation, and dataset preparation.
- Training/evaluation must be reproducible but is outside the Studio GUI milestone.
- Shared landmark transforms, handedness semantics, data schemas, model metadata, and backend policy must not be duplicated across App and Studio implementations.
- The frontend consumes runtime results; camera/MediaPipe/model execution belongs behind the runtime boundary.
- Do not encode Tauri, Electron, or another desktop shell as canonical architecture until the prototype decision is accepted.

## Data and ML rules

- Canonical v2 observations store raw landmarks + provenance + label/review state, not photos/video.
- Feature vectors are derived and versioned.
- Preserve difficult-but-valid samples.
- Automatic confidence/disagreement/quality signals may prioritize `Suggested for Review`; they must not automatically reject or relabel data.
- Legacy data can help training but cannot be used to claim clean participant-held-out evaluation.
- Prefer participant/session-aware splits over row-random splits.
- Keep model label order, feature-transform version, and architecture compatibility with the model artifact.

## Platform rules

- CPU is always a valid fallback.
- Optional acceleration is enabled only when supported and verified for the selected runtime/hardware combination.
- Do not infer PyTorch support from the presence of a vendor SDK alone.
- Tests that gate normal development should not require camera or GPU hardware unless explicitly marked as integration/platform tests.

## Implementation workflow

1. Read relevant source and canonical docs before editing.
2. Update/add a failing test when practical for behavioral changes.
3. Make the smallest coherent implementation.
4. Run focused tests first, then broader applicable checks.
5. Update reference/change/runbook docs only with evidence from what changed or what was actually executed.
6. Review the diff before committing.

For Python on the current CachyOS/Fish development host, use a project virtual environment and `source .venv/bin/activate.fish`; do not install project packages into system Python.
