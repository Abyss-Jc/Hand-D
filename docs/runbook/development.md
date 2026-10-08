# Development Runbook

## Precondiciones

- Work from the Hand-D repository.
- For the v2 modernization, use `feature/hand-d-v2-modernization`.
- Do not make direct v2 implementation commits on `main`.
- Use the project-managed Python environment; the v2 target is `uv` + `pyproject.toml` + `uv.lock` with Python 3.13.

Current environment note:

- The CachyOS development host reported Python 3.14.7.
- `uv 0.12.23` is installed on the CachyOS host (`uv --version`, verified 2026-10-08).
- The global environment did not contain PyTorch when the v2 audit started.
- The dependency set is scheduled for compatibility migration; the old README install sequence is therefore not yet considered a verified v2 setup.

## Pasos

```text
1. git fetch origin
2. git switch feature/hand-d-v2-modernization
3. git status
4. uv --version
5. uv python install 3.13           # target command; execute/verify when pyproject migration begins
6. uv sync                          # target command; valid only after pyproject.toml/uv.lock exist
```

The v2 workflow should prefer `uv run ...`/`uv sync` instead of requiring shell activation. Do not install project dependencies globally.

Exact dependency groups/commands remain `No verificado` until the new `pyproject.toml` and lockfile are created and synced successfully. The current requirements files remain legacy migration inputs, not the future dependency source of truth.

### Transitional contract tests (verified 2026-10-08)

Before the `uv` migration, no-camera checks run using the preexisting project-local `venv`:

```bash
venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
venv/bin/python -m compileall -q handd_core visualizer_app/gesture_engine.py tests/test_feature_transform.py
```

Verified result: **10 tests passed** for shared Feature Transform v1 and existing GestureEngine checks, including a subprocess import from the legacy source-script working directory. A separate valid-input comparison with the pre-refactor canonicalizer was exact for float32/float64 and Left/Right fixtures. This does not verify `uv sync`, camera operation, or the new data pipeline.

HD-03 adds the no-camera SQLite dataset-store tests. Verified on 2026-10-08: `venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v` passed **20/20**. These tests use temporary SQLite databases and do not alter the team's canonical workspace or legacy CSV data.

### Canonical dataset collaboration

For the October milestone, `handd.sqlite` is edited sequentially rather than concurrently:

```text
1. git pull / update the modernization branch
2. confirm no teammate is currently editing the canonical dataset
3. collect / curate using the current handd.sqlite
4. close Studio so database writes are finished
5. commit the dataset change
6. push the branch
7. release dataset editing ownership
```

Git is not expected to merge two independently modified SQLite files. If two contributors need concurrent collection later, add an explicit session import/export workflow rather than relying on binary merges.

## Comprobar

Verified on 2026-10-05:

```text
git branch --show-current
→ feature/hand-d-v2-modernization

git rev-list --left-right --count origin/main...HEAD
→ branch was created from the current origin/main baseline before v2 commits
```

Not yet verified for v2:

- clean dependency installation from scratch;
- project Python 3.13 provisioning + `uv sync` from the future lockfile;
- full unit/integration test suite in the new environment;
- Apple Silicon/MPS execution;
- CUDA/ROCm/XPU execution;
- production desktop-shell packaging.

These items must remain marked `No verificado` until commands/tests are actually run.

## Recuperación

- If a dependency experiment breaks the environment, remove/recreate `.venv`; do not repair by installing packages globally.
- If a v2 slice is invalid, revert the granular commit on the modernization branch rather than rewriting `main`.
- Preserve legacy datasets and models while migration/import logic is being validated.

## Riesgos y secretos

- Collection provenance must not store hardware serials, MAC addresses, passwords, tokens, or other unnecessary identifying secrets.
- Use an application-generated anonymous device ID for local provenance.
- Never document secret values; document only variable/configuration names when secrets eventually exist.
