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

CPU is a supported fallback everywhere. Training prefers validated PyTorch acceleration such as CUDA/MPS/ROCm/XPU where it materially helps. Packaged runtime also prefers a validated accelerated provider when it meets or improves the interaction budget, but provider/hardware benchmarks make that selection because availability alone does not guarantee a net win for the small gesture MLP. A vendor SDK or execution provider existing on a machine is not sufficient evidence by itself.

## Consecuencias

- Training-backend selection and deployment-runtime provider selection are related but distinct tested concerns.
- Platform-specific acceleration dependencies may need separate installation/packaging paths.
- Performance claims must name the backend/hardware actually tested.
- The current Tiger Lake/Iris Xe development laptop remains CPU-fallback capable; an accelerated runtime path is used only if benchmark evidence shows a benefit.

## Sustituye o es sustituido por

No prior ADR.

## Fuentes

- `visualizer_app/gesture_engine.py:181-196`
- `docs/changes/hand-d-v2-modernization.md`
