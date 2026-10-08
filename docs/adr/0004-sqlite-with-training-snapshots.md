# SQLite working dataset with immutable training snapshots

Hand-D v2 uses SQLite as the canonical mutable working dataset because Studio needs transactional collection, provenance queries, reversible curation, and filtered review over a dataset that is currently small enough for an embedded database.

Training does not read an undefined current database as its experiment identity. Before training/evaluation, Snapshot Builder freezes an immutable, self-contained ML snapshot containing:

- a manifest with exact Sample membership, participant/session membership, fixed validation folds, Feature Transform identity/version/output contract, label mapping/order, legacy membership, seed/reproducibility configuration, and integrity/version metadata;
- a materialized NPZ containing the exact feature arrays, labels, source identifiers, and fold/split metadata consumed by the ML workflow.

An existing snapshot therefore remains executable even when later curation changes SQLite or the Feature Transform implementation evolves. Training/evaluation must not re-query current SQLite state to reconstruct a historical snapshot. A materially different membership, transform, or materialization creates a new snapshot rather than rewriting an existing one.

For the milestone, snapshot/model integrity is enforced operationally through immutable version allocation: builders create a new snapshot/model version instead of overwriting an existing artifact. Manifests retain human-readable IDs plus relevant source/provenance information. Standalone cryptographic checksum files are deferred until an external-distribution, corruption-detection, or multi-system artifact-transfer requirement justifies that additional mechanism.

The P001/P002 development protocol uses one Development Snapshot as the common evidence base for session-cross-validation, legacy/no-legacy comparison, and learning curves. Compatible legacy rows are a distinct materialized partition inside that development snapshot. Fold choice, legacy inclusion, and learning-curve fraction are experiment configuration over the same frozen evidence base rather than independently-created datasets.

P003 is intentionally absent from the Development Snapshot. After development/model-selection choices are frozen, the selected configuration is refit on all eligible development data and P003 is materialized separately into a Final Test Snapshot for the single held-out unseen-participant evaluation.

Training outputs versioned Model Artifacts rather than bare weights files. Cross-validation fold artifacts/checkpoints are retained as evaluation evidence for exact OOF assessments and are not promoted as the Active Model. After development choices freeze, one final-refit Model Artifact is trained on all eligible development data and becomes the promotable runtime candidate. Artifacts retain canonical training weights plus a manifest declaring artifact role, architecture, Feature Transform/input contract, label order, source snapshot, training configuration, runtime/deployment formats, and compatibility information, together with role-appropriate evaluation metrics. A validated derived deployment representation such as ONNX may live beside the PyTorch weights without replacing the artifact's provenance contract. Runtime validates this manifest before inference.

This design keeps three responsibilities separate:

1. SQLite — canonical mutable collection/provenance/curation history.
2. Immutable snapshots — frozen, self-contained ML experiment inputs.
3. Model Artifacts — executable trained weights plus the contract/evidence required to use them safely.

It avoids introducing Arrow/Parquet, DVC, MLflow, or a repository abstraction solely for SQLite before dataset scale or collaboration requirements justify that complexity.
