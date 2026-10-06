# ADR: Canonical dataset stores raw landmarks and provenance

## Estado

`aceptado`

## Contexto

The legacy dataset stores model-ready feature rows. That makes feature-transform changes difficult to reproduce and does not preserve participant/session provenance required for honest grouped evaluation. Storing camera photos/video would provide more raw information but adds unnecessary storage and privacy cost for the current landmark-based model.

## Alternativas

| Opción | Ventajas | Costes |
|---|---|---|
| Keep feature-vector CSV as source of truth | Simple and compatible with current scripts | Cannot safely regenerate changed feature transforms; weak provenance |
| Store raw landmarks + provenance as canonical observations | Reproducible transforms; grouped evaluation; compact | Requires a new schema and migration boundary |
| Store video/images + landmarks | Maximum reprocessing flexibility | Large storage/privacy/operational cost outside current need |

## Decisión

The v2 canonical sample stores raw MediaPipe landmarks, target label, hand, participant/session/capture provenance, local device provenance, timestamps/frame identity, and human review state. Photos/video are not part of the canonical dataset. Model-ready feature vectors are versioned derived artifacts.

Human curation is reversible: model signals may suggest review, but do not automatically reject, relabel, or delete source observations.

## Consecuencias

- Feature engineering can change without recollecting v2 samples.
- Train/validation/test generation can group by participant/session.
- Storage format and schema versioning become explicit engineering concerns.
- The legacy CSV remains a compatibility/training source but cannot gain provenance retroactively.

## Sustituye o es sustituido por

No prior ADR.

## Fuentes

- `dataset_extraction_tools/data_extractor.py:197-269`
- `docs/changes/hand-d-v2-modernization.md`
