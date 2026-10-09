# HD-09 deferred study: Wiggly brush line-boil renderer

**Status:** Investigated; **implementation deliberately deferred** by user on 2026-10-08. The current sinusoidal animated SVG brush remains a known-quality gap and is **not** considered acceptance-complete.

## Evidence and why the current approach looks wrong

Primary sources, from WigglyPaint's actual author **John Earnest / Internet Janitor**:

- [Some Words on WigglyPaint](https://beyondloom.com/blog/onwigglypaint.html): the author explains three simultaneously edited image buffers with per-segment randomization / varying brush stamps and switching among the buffers at roughly **12 FPS**. Two buffers visibly jitter too much; more than three show diminishing returns. This is the intended "line boil" look rather than rigid oscillation of every point.
- [Author's explanation of brush implementation](https://itch.io/post/16220493): each drawn input segment is rendered into **three frames** with varying endpoints/brush treatment, not a single whole-stroke path re-evaluated with sine displacement.
- Avoid mistaking unaffiliated mirror domains (including wigglypaint.net) for the original author's site. The author explicitly identifies those mirrors as third-party copies/imitations. No original code or art assets were imported into Hand-D.

**Current code** in `desktop/web/stroke-renderer.mjs` at `animateWiggly`:
1. Recomputes one long polyline every ~65ms, shifting its points according to `sin(i*1.3 + phase)` and `cos(i*1.7 + phase*1.1)`.
2. Uses the original **sample index**, rather than *distance along the stroke*, as the spatial variation axis. Therefore a gesture-stream path sampled unevenly in time can produce dense jagged sections and overly long straight interpolations; a mouse and a gesture with the same geometry look different.
3. In long strokes, stride downsampling to ~320 points can unexpectedly remove corners, micro-details and loops. In short strokes, the entire appearance shifts at once and the animation visibly appears like a shaky vector.
4. Increasing amplitude from ~2 SVG units to ~8/7 changes size but not character and may amplify ugly discontinuities. 15 Hz smooth oscillation does not reproduce the intentionally low-frame-count line-boil aesthetic.
5. The current SVG eraser masks correctly remove **ink only**, but the eventual renderer must preserve that stacking and mask operation for each frame, not erase pixels of the live MJPEG camera.

## Proposed original implementation (future small opt-in experiment)

Keep the **immutable canonical normalized path** as the only editable, undoable Whiteboard data. Convert it, for *presentation only*, into three stable seeded variants:

1. **Resample by arc length** for display (e.g., around 6–10 screen-equivalent SVG units), preserving endpoints, sharp turns, and loops. Avoid using MediaPipe callback count as distance; preserve the original geometry when not animating.
2. **Create variants A/B/C once on stroke completion or incrementally on new points** using a locally authored deterministic PRNG seeded from the stroke ID and frame index. Apply *small, correlated perturbations* chiefly perpendicular to the local tangent, with slight independent radius/opacity/brush texture changes, rather than large whole-stroke lateral shifts. Keep initial/final endpoints anchored or only gently displaced so connected segments do not break.
3. **Cycle displayed variants discretely at 10–12 FPS** (A→B→C), not continuous sine motion. Drawing input/cursor/camera stay at their normal high rates; creation is incremental and bounded. Option: one mask/layer per variant, toggling `display` on presentation groups without re-generating all path geometry each tick. Record only the parameters/seed, not the animated polygons, in an eventual native file.
4. **Preserve eraser semantics** by composing completed ink and erase masks in each display variant or by keeping a single canonical eraser mask applied around the three animated variants. Test eraser-then-new-ink ordering, undo/redo, all modes, and Studio navigation. Camera MJPEG must remain unaffected.
5. **Accessibility and budget:** ordinary Pen unchanged by default; reduced motion switches to the canonical static curve; hidden page/Studio pauses the animation; cap visible segments/dirty-region work and measure WebKit DOM/path rebuild cost and camera+inference FPS, not Node mocks alone. Consider raster paint cache or offscreen presentation layer only if measured SVG rendering is too expensive.
6. **Visual acceptance before launch:** run side-by-side video-free live A/B comparison (mouse squiggle, zigzag, handwriting, circles, gestures) with current vs three-variant renderer at camera 640x480; judge line character and stroke recognition, not merely displacement amplitude. Show an explicit Wiggly/Regular toggle and one strong but bounded default.

### Future TDD slices

- **RED:** 3 deterministic variants differ, but are repeatable for equal canonical points/seed; each point remains within bounded displacement, and corner/endpoint continuity passes tests.
- **RED:** equivalent paths with different temporal sample densities render similarly after distance-based resampling.
- **RED:** three-frame clock doesn't mutate geometry or build a new SVG path every 80–100ms; 2D camera/eraser unaffected.
- **RED:** reduced-motion/hidden view yields static canonical geometry and CPU drawing work stops.
- **GREEN:** minimal presenter; real WebKit/MediaPipe manual smoke. Do not import or redistribute source code, bundled artwork or audio from another program.

**Priority:** P2, not on the HD-09→HD-10 critical path. No implementation change made in this investigation.
