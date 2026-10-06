# ADR: Separate Hand-D App and Hand-D Studio

## Estado

`aceptado`

## Contexto

The current repository contains a user-facing gesture whiteboard plus collection, curation, training, and evaluation concerns. Putting all developer tooling into one end-user UI would increase coupling, dependency weight, and UX complexity.

## Alternativas

| Opción | Ventajas | Costes |
|---|---|---|
| One application containing every workflow | One executable/surface | End-user and ML tooling concerns become tightly coupled |
| Separate App and Studio over shared core contracts | Clear product boundary; lighter App; tooling can evolve independently | Requires explicit shared interfaces |

## Decisión

Hand-D v2 has two maintained surfaces: **Hand-D App** for the user-facing whiteboard and **Hand-D Studio** for developer data/ML workflows. They share data/model/runtime contracts rather than sharing one monolithic UI.

## Consecuencias

- Studio-only dependencies do not need to define the App experience.
- Training/evaluation can remain CLI/tool-driven even while Studio collection/curation receives a GUI.
- Shared transformations and model contracts must move out of UI-specific code.

## Sustituye o es sustituido por

No prior ADR.

## Fuentes

- `CONTEXT.md`
- `docs/changes/hand-d-v2-modernization.md`
