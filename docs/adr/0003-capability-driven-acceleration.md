# ADR: CPU fallback with capability-driven acceleration

## Estado

`aceptado`

## Contexto

Hand-D is developed and tested across machines that may expose different acceleration backends. Treating a particular GPU vendor as mandatory would make the application brittle and would overstate support for hardware combinations that have not been validated.

## Alternativas

| Opción | Ventajas | Costes |
|---|---|---|
| GPU-required runtime | Maximum expected performance on one supported stack | Excludes unsupported/CPU-only machines |
| One vendor-specific GPU path + CPU | Simpler than multi-backend | Poor fit for Apple Silicon and other target machines |
| Capability-driven backends + universal CPU fallback | Portable and honest about support | Requires backend detection, testing, and packaging discipline |

## Decisión

CPU is a supported fallback everywhere. Hand-D may prefer CUDA, MPS, ROCm, or PyTorch XPU only when the selected software stack reports support and the target combination has been validated. A vendor SDK existing on a machine is not sufficient evidence by itself.

## Consecuencias

- Backend selection becomes a tested runtime concern.
- Platform-specific acceleration dependencies may need separate installation/packaging paths.
- Performance claims must name the backend/hardware actually tested.
- The current Tiger Lake/Iris Xe development laptop remains CPU-first unless a supported acceleration path is demonstrated.

## Sustituye o es sustituido por

No prior ADR.

## Fuentes

- `visualizer_app/gesture_engine.py:181-196`
- `docs/changes/hand-d-v2-modernization.md`
