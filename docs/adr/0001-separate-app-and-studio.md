# ADR: Separate Whiteboard and Studio product spaces

## Estado

`aceptado`

## Contexto

The current repository contains a user-facing gesture whiteboard plus collection, curation, training, evaluation, model-management, and workspace concerns. Flattening all of those concerns into the normal drawing surface would increase coupling and UX complexity.

## Alternativas

| Opción | Ventajas | Costes |
|---|---|---|
| One undifferentiated surface containing every workflow | One navigation model | End-user drawing and ML/data concerns become tightly coupled |
| Two separate installed applications | Strong separation | Duplicates shell/runtime/update lifecycle and makes switching workflows heavier |
| One desktop application with distinct Whiteboard and Studio spaces over shared core contracts | Clear product boundary with one install/runtime lifecycle | Requires explicit navigation and shared interfaces |

## Decisión

Hand-D v2 is one desktop application with two maintained product spaces: **Whiteboard/App** for the user-facing drawing experience and **Hand-D Studio** for advanced project/data/model workflows. Studio is not developer-only; technical detail is progressively disclosed. The two spaces share one Tauri shell/runtime lifecycle plus explicit data/model/runtime contracts rather than becoming one monolithic screen.

## Consecuencias

- Studio complexity does not need to dominate the Whiteboard experience.
- Training/evaluation can remain CLI/tool-driven even while Studio collection/curation receives a GUI.
- Shared transformations and model contracts must move out of UI-specific code.
- Whiteboard remains the default launch destination; Studio remains secondary navigation inside the same installed application.

## Sustituye o es sustituido por

No prior ADR.

## Fuentes

- `CONTEXT.md`
- `docs/changes/hand-d-v2-modernization.md`
