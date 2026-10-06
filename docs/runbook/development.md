# Development Runbook

## Precondiciones

- Work from the Hand-D repository.
- For the v2 modernization, use `feature/hand-d-v2-modernization`.
- Do not make direct v2 implementation commits on `main`.
- Use a project-local Python virtual environment.

Current environment note:

- The CachyOS development host reported Python 3.14.7.
- The global environment did not contain PyTorch when the v2 audit started.
- The dependency set is scheduled for compatibility migration; the old README install sequence is therefore not yet considered a verified v2 setup.

## Pasos

```text
1. git fetch origin
2. git switch feature/hand-d-v2-modernization
3. git status
4. python --version
5. python -m venv .venv
6. source .venv/bin/activate.fish   # Fish shell on the current development host
```

For POSIX shells other than Fish, use the activation script appropriate for that shell. Do not install project dependencies globally.

Dependency installation commands will be added here only after the v2 compatibility set has been selected and actually verified.

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
