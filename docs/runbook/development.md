# Development Runbook

## Precondiciones

- Work from the Hand-D repository.
- For the v2 modernization, use `feature/hand-d-v2-modernization`.
- Do not make direct v2 implementation commits on `main`.
- Use the project-managed Python environment: `uv` + `pyproject.toml` + committed `uv.lock`, with Python 3.13 (verified on CachyOS x86-64, 2026-10-08).

Current environment note:

- The CachyOS development host reported Python 3.14.7.
- `uv 0.12.23` is installed on the CachyOS host (`uv --version`, verified 2026-10-08).
- The global environment did not contain PyTorch when the v2 audit started.
- The legacy `requirements.txt` / `requirementsGPU.txt` remain historical inputs, **not** the dependency source for v2. `pyproject.toml` and `uv.lock` now define the reproducible v2 environment.

## Pasos

```text
1. git fetch origin
2. git switch feature/hand-d-v2-modernization
3. git status
4. uv --version
5. uv python install 3.13
6. uv lock --check
7. uv sync --frozen
8. uv run --frozen python -m unittest discover -s tests -p 'test_*.py' -v
```

The v2 workflow uses `uv run`/`uv sync` without requiring shell activation. Do not install project dependencies globally. `.venv/` is project-local and ignored; `.python-version`, `pyproject.toml` and `uv.lock` must be committed.

### Dependency profiles (verified locally 2026-10-08)

| Profile | Command | Scope |
|---|---|---|
| Runtime only | `uv sync --frozen --no-default-groups` | NumPy, MediaPipe 0.10.33, one OpenCV distribution (`opencv-contrib-python`), Torch CPU on Linux |
| Runtime + development (default) | `uv sync --frozen` | Runtime plus dev tools such as pytest |
| Runtime + training + development | `uv sync --frozen --group training` | Above plus pandas, scikit-learn and training/reporting dependencies |

Python **3.13.16** was installed locally by `uv python install 3.13`. `uv lock --check`, `uv sync --frozen`, `uv pip check --python .venv/bin/python`, and the baseline unit tests succeeded. The runtime-only profile passed 20 tests; the training profile imported pandas **3.0.6**, scikit-learn **1.9.1**, matplotlib **3.11.2**, and passed 20 tests. Returning to default sync removed optional training-only dependencies, confirming separation. MediaPipe pulls some plotting dependencies transitively, so the runtime profile is not guaranteed to contain zero plotting packages.

On this CachyOS x86-64 host, default Torch is **2.14.1+cpu**, MediaPipe **0.10.33**, NumPy **2.5.3**, and OpenCV **4.13.0** (`opencv-contrib-python==4.13.0.92`). `opencv-python` is not installed alongside contrib in `.venv`. Python 3.13 installation on macOS/Windows, GPU providers, camera hardware, and production sidecar packaging are **No verificado**.

To invoke training-only dependencies and then restore the default development profile:

```bash
uv run --frozen --group training python -m unittest discover -s tests -p 'test_*.py' -v
uv sync --frozen
```

The old project-local `venv/` is transitional legacy evidence, not the new `.venv/`. If dependencies intentionally change, regenerate `uv.lock` with `uv lock` and verify it before committing; regular setup must use `uv sync --frozen`.

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

Verified for v2 on CachyOS Linux x86-64 (2026-10-08):

- Python 3.13.16 provisioning and project-local `uv sync --frozen` from `uv.lock`;
- `uv lock --check`, `uv pip check`, and installed dependency versions;
- default, runtime-only, and training dependency profiles; **20/20** no-camera tests in each;
- exactly one installed OpenCV distribution in the v2 `.venv`.

Still **No verificado**:

- reproducibility on a different clean machine and supported non-Linux platforms;
- camera collection, LIVE_STREAM inference, Tauri HTTP/WS/MJPEG integration;
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
