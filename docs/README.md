# Hand-D Documentation System

This directory is the canonical documentation entry point for Hand-D v2. It separates product intent, current/target architecture, active change plans, durable decisions, and operating instructions so that one document does not need to serve every purpose.

## Canonical map

| Question | Canonical document |
|---|---|
| What must Hand-D v2 do? | [Requirements](requirements/hand-d-v2.md) |
| What should approved Whiteboard/Studio UX look and behave like? | [v2 UX Shape](design/hand-d-v2-ux-shape.md) |
| How is Hand-D structured today and what boundaries are we moving toward? | [Architecture reference](reference/architecture.md) |
| What are we changing for the October 13 milestone, and in what order? | [v2 roadmap](changes/hand-d-v2-roadmap.md) |
| What planning decisions have already been confirmed? | [Modernization change record](changes/hand-d-v2-modernization.md) |
| Why did we make durable architectural choices? | [ADRs](adr/) |
| How should a developer prepare and verify the repository? | [Development runbook](runbook/development.md) |
| What rules should coding agents and contributors follow? | [AGENTS.md](../AGENTS.md) |

## Source-of-truth rules

1. **Source code and tests** are evidence for behavior that exists today.
2. **Reference docs** describe verified current behavior and explicitly labeled target architecture.
3. **Requirements docs** define intended product behavior and acceptance criteria.
4. **ADRs** explain durable decisions and trade-offs.
5. **Change docs** describe active migrations, sequencing, and milestone scope.
6. The root `README.md` remains the project landing page.
7. The root `SPEC.md` is a **legacy planning artifact**. It contains useful historical context, but it is not canonical for v2 because several statements no longer match the repository.

Do not silently rewrite legacy documentation to make it appear current. Migrate facts into the appropriate canonical document with source evidence first.

## Documentation status

The v2 system is being formalized on `feature/hand-d-v2-modernization`. `main` remains untouched until the work is reviewed and intentionally merged.
