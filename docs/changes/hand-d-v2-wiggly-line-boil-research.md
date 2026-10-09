# Wiggly brush: line-boil research, implementation and verification

**Status:** New three-frame implementation completed on 2026-10-08 after the user explicitly requested the fix. Automated geometry, SVG raster and synthetic performance checks passed on Linux; **physical WebKit visual acceptance by the user and macOS verification are still pending**. Do not call the final appearance user-approved until that check.

## Evidence and why the current approach looks wrong

Primary sources, from WigglyPaint's actual author **John Earnest / Internet Janitor**:

- [Some Words on WigglyPaint](https://beyondloom.com/blog/onwigglypaint.html): the author explains three simultaneously edited image buffers with per-segment randomization / varying brush stamps and switching among the buffers at roughly **12 FPS**. Two buffers visibly jitter too much; more than three show diminishing returns. This is the intended "line boil" look rather than rigid oscillation of every point.
- [Author's explanation of brush implementation](https://itch.io/post/16220493): each drawn input segment is rendered into **three frames** with varying endpoints/brush treatment, not a single whole-stroke path re-evaluated with sine displacement.
- Avoid mistaking unaffiliated mirror domains (including wigglypaint.net) for the original author's site. The author explicitly identifies those mirrors as third-party copies/imitations. No original code or art assets were imported into Hand-D.

**Previous code (replaced)** in `desktop/web/stroke-renderer.mjs` at `animateWiggly`:
1. Recomputes one long polyline every ~65ms, shifting its points according to `sin(i*1.3 + phase)` and `cos(i*1.7 + phase*1.1)`.
2. Uses the original **sample index**, rather than *distance along the stroke*, as the spatial variation axis. Therefore a gesture-stream path sampled unevenly in time can produce dense jagged sections and overly long straight interpolations; a mouse and a gesture with the same geometry look different.
3. In long strokes, stride downsampling to ~320 points can unexpectedly remove corners, micro-details and loops. In short strokes, the entire appearance shifts at once and the animation visibly appears like a shaky vector.
4. Increasing amplitude from ~2 SVG units to ~8/7 changes size but not character and may amplify ugly discontinuities. 15 Hz smooth oscillation does not reproduce the intentionally low-frame-count line-boil aesthetic.
5. The current SVG eraser masks correctly remove **ink only**, but the eventual renderer must preserve that stacking and mask operation for each frame, not erase pixels of the live MJPEG camera.

## Implemented original Hand-D approach

The **canonical normalized path** remains the only editable, undoable Whiteboard data. The new implementation creates three seeded display-only variants:

1. **Resample by arc length:** `desktop/web/line-boil.mjs` constructs display points every ~9 SVG units (adaptively coarser for extreme paths), retaining endpoints and high-turn corners. Geometry is bounded to 1,200 vertices per variant and independent of MediaPipe callback density.
2. **Precompute A/B/C on stroke edits:** pure `buildLineBoilFrames(points,{seed})` creates three variant SVG paths using seeded, smooth, perpendicular noise (~3.3 SVG units maximum) and subtly distinct line widths. Endpoints stay anchored. The seed derives from the stroke's starting position unless explicitly provided; the canonical points are never modified. These are **Hand-D's own algorithms**, no imported artwork/code.
3. **Cycle at 12 FPS** using `LINE_BOIL_FRAME_MS=1000/12` and `StrokeRenderer.animateWiggly`. All variants are SVG children of one ink group, under the same existing chronological masks. Rather than modifying each path per tick, a single `data-boil-frame` attribute on the root SVG selects frame 0→1→2 via CSS. On reduced motion, leave Studio or hidden page, the canonical static path is visible, and frame switching stops.
4. **Transparent erasure:** the mask containing the black Eraser only clips ink groups created before the erase action; later Pen/Wiggly ink sits above the masked group. Neither camera pixels nor UI controls are part of these ink masks.
5. **Performance:** three variants are regenerated only when that specific live stroke gets new points, on the preexisting RAF drawing cadence. Once finished, the cost of the 12 FPS clock does not increase with stroke count in JavaScript; browser compositing cost still needs WebKit FPS verification.
6. **Still pending human review:** side-by-side mouse handwriting, circles, zigzags and two-hand tracking in native Tauri, Camera Overlay and Clean Canvas, Undo/Redo, reduced motion, plus macOS device rendering. Color/width/pattern can be tuned from that evidence.

### TDD evidence (RED → GREEN)

- `node --test desktop/tests/line-boil.test.mjs`: 3 deterministic seeded frames differ, endpoint/corner continuity and bounded displacement, equivalent input with different sampling density, pathological long-stroke bound.
- `node --test desktop/tests/wiggly.test.mjs desktop/tests/stroke-renderer.test.mjs desktop/tests/app-ui.test.mjs`: cached paths survive ordinary strokes and erasers, frame switch changes no `d` geometry and exactly one SVG root attribute, real UI undo/redo, reduced motion, old camera-transparent eraser guarantees.
- `node desktop/tests/benchmark-line-boil.mjs`: synthetic 120-stroke (112 Wiggly), 360-frame display-clock test; **0 path geometry writes during clocks**, ~0.66ms total Node simulated clock work; NOT a WebKit/browser FPS measure.
- SVG **real raster** using local `rsvg-convert` with three generated variants and a synthetic red camera background: all three variants had nonidentical pixel renderings, visible ink outside the eraser, red background visible at the erased center; no webcam frames captured.
- Native Tauri process and visual acceptance: **No verificado** until the test run and user feedback below.

**Priority:** P2 optional visual polish; core replacement implemented, hardware visual-acceptance gate remains open without blocking HD-09→HD-10.
