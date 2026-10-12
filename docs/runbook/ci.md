# Hand-D v2 - Continuous Integration (CI-01 / CI-02)

**Scope:** Linux automated checks + a stable merge-status gate for Hand-D v2.

**Sources:** `.github/workflows/ci.yml`, `pyproject.toml`, `uv.lock`, `desktop/package-lock.json`, `desktop/src-tauri/Cargo.lock`, and `tests/test_ci_workflow.py`.

## What is executed in GitHub Actions

On pushes to `feature/hand-d-v2-modernization` or `main`, pull requests targeting either branch, and manual dispatch:

| Check / job | Environment | Locked installation and checks |
|---|---|---|
| `Python tests` (`python`) | Ubuntu 24.04, Python 3.13 via uv | `uv lock --check`; `uv sync --frozen --group training`; Python unittest discovery (including synthetic Studio/ML tests) |
| `Frontend tests` (`frontend`) | Ubuntu 24.04, Node.js 22 | `npm ci --prefix desktop`; syntax check; `node --test desktop/tests/*.test.mjs` |
| `Rust tests` (`rust`) | Ubuntu 24.04, stable Rust and Tauri 2 GTK/WebKit libraries | `cargo test --manifest-path desktop/src-tauri/Cargo.toml --locked -q` |
| **`Required CI`** (`required`) | Ubuntu 24.04 | Succeeds **only if all three jobs succeed**. Failed, cancelled or skipped jobs produce a failing gate. |

CI is headless and synthetic: **no camera hardware, participant data, GPU, native macOS/Windows GUI validation, P003 measurements or releases**. HD-10 real-person acceptance remains separate.

## Workflow-level safety (CI-02)

- **`permissions: contents: read` only**; checkout does not persist GitHub write credentials, and the workflow needs no secrets.
- Never use `pull_request_target` for untrusted changes. Actions are pinned to full commit hashes.
- Dependencies come from their lockfiles (`uv sync --frozen`, `npm ci`, `cargo ... --locked`) so CI cannot silently re-resolve them.
- A concurrency group cancels superseded runs for the same event ref; cached dependencies improve speed but never replace the lockfile.
- The **single named final check `Required CI`** remains stable when the internal test jobs evolve; branch protection should depend on it.

## Admin-only GitHub configuration: Josué

**Publishing a workflow does NOT automatically block merges.** The account currently connected to Magia has **WRITE** permission on `Abyss-Jc/Hand-D`, but not **ADMIN**. No branch protection or rulesets were found during the audit. Only Josué (or another repo admin) can do the following:

1. Open **Repository → Settings → Rules → Rulesets** (or **Settings → Branches → Branch protection**).
2. Target at least **`main`**. If you want pull-request-only contributions on `feature/hand-d-v2-modernization`, also protect that branch — note that doing so will intentionally replace our current direct-push workflow.
3. Enable **Require a pull request before merging**, and **Require status checks to pass**.
4. After the first workflow finishes, select **`Required CI`** (from GitHub Actions) as the required check.
5. Consider requiring an up-to-date branch, blocking force pushes and branch deletion as separate team policies.
6. Verify on a PR: an intentionally broken test must block merge; a clean PR must show `Required CI` green.

**Current CI-02 status:** implemented safety and stable check in YAML; **mandatory enforcement is pending Josué's admin action**. This is not a claim that GitHub currently rejects failed PRs.

## Local commands (no webcam)

On Linux, after installing the native prerequisites described in [setup-v2](setup-v2.md):

```bash
uv lock --check
uv sync --frozen --group training
uv run --frozen --group training python -m unittest discover -s tests -p 'test_*.py' -q
npm ci --prefix desktop --no-audit --no-fund
npm --prefix desktop run check
node --test desktop/tests/*.test.mjs
cargo test --manifest-path desktop/src-tauri/Cargo.toml --locked -q
uv run --frozen python -m unittest tests.test_ci_workflow -q
```

CI contract tests ensure the triggers, lockfile commands, branch protections' expected check name, permissions and cancellation rules are not accidentally removed. They are **static tests**, not evidence that a hosted GitHub runner has already succeeded.

## First hosted run and troubleshooting

In **GitHub → Actions → Hand-D CI**, select the push or pull request commit. Verify the three jobs and final `Required CI` are green; record any first-run discrepancies in the development runbook. If a job fails, inspect its full logs and reproduce locally with the commands above. Do not change passing tests or loosen security rules merely to make a check green.

**Out of scope for CI-01/02:** macOS Apple Silicon CI, Windows native build, signed artifacts, packaged sidecar smoke, runtime/OS webcam permissions and automatic releases. Those can be separate follow-up CI tickets once a real runner and packaging path have been checked.

## Local verification before the first hosted run (2026-10-12)

On the developer machine at `/home/caml/Projects/Hand-D`, the setup and
checks corresponding to all three Linux jobs completed successfully:
`uv lock --check`, `uv sync --frozen --group training`, **142/142 Python
unittests** (including four new CI contract assertions), `npm ci --prefix
desktop`, `npm --prefix desktop run check`, **49/49 JavaScript tests**,
**7/7 Rust tests**, `git diff --check`, and `actionlint v1.7.7` validating
`.github/workflows/ci.yml`. A YAML parser also confirmed the declared
triggers, jobs and read-only permissions. Hosted runner success remains
**unverified until GitHub Actions finishes its first execution**.
