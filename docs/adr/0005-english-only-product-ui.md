# ADR 0005 — English-only product UI for Hand-D v2

**Status:** Accepted (2026-10-08).

## Context

Hand-D documentation and developer communication may be written in Spanish, but the application itself had a mix of Spanish and English strings in its Whiteboard, Studio, camera diagnostics and accessibility labels. The user explicitly requested an **entirely English UI**.

## Decision

- The canonical product language is **English**. HTML root language: en.
- Apply this to every visible/accessible text surface: navigation, button labels, status messages, hints, modal controls, aria labels, titles, placeholders, disabled/degraded states, errors and Studio sections. No partial English/Spanish screens.
- Technical identifiers and programmatic protocol keys (e.g. runtime_session_id, gesture labels) remain unchanged; this decision governs user-visible text only.
- Internal planning documents, changelogs, comments and developer conversations can remain bilingual. Future languages require an explicit i18n decision rather than ad hoc UI strings.
- Add a regression test that inspects static HTML and dynamic UI strings for mixed-language output; tests may assert English diagnostic vocabulary.

## Consequences

UI test expectations, macOS lab instructions and UX handoff examples must reflect English labels. This is **a product-copy change**, not a change to runtime gesture recognition, database, models or camera frame processing. Preserve unrelated uncommitted UX edits.
