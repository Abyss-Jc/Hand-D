"""
canvas.py
---------
Drawing state machine and stroke management for the Hand-D visualizer.

Responsibilities:
  - Receive GestureResult frames and advance the gesture state machine
  - Maintain an in-memory stroke list (source of truth for both display and SVG export)
  - Map modifier-hand thumb position to stroke thickness
  - Manage ruler mode (preview line → commit on release)
  - Export strokes to SVG

The Canvas class is UI-framework-agnostic: it holds no Tkinter references.
main.py calls `canvas.render(tk_canvas)` to paint the current state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  import tkinter as tk

from gesture_engine import GestureResult

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class DrawState(Enum):
  IDLE = auto()
  DRAWING = auto()
  ERASING = auto()
  THICKNESS_ADJUST = auto()
  RULER_MODE = auto()


@dataclass
class Stroke:
  points: list[tuple[float, float]] = field(default_factory=list)
  color: str = '#000000'
  width: float = 3.0
  is_line: bool = False  # True for ruler-committed straight lines


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

THICKNESS_MIN = 1.0
THICKNESS_MAX = 20.0
ERASER_RADIUS_MIN = 10.0  # px — smallest eraser size
ERASER_RADIUS_MAX = 80.0  # px — largest eraser size
ERASER_RADIUS_DEF = 30.0  # px — default eraser size
SMOOTHING_ALPHA = 0.4  # EMA weight for jitter smoothing (0 = max smooth, 1 = raw)
MIN_POINT_DIST = 4.0  # px — minimum distance between recorded points

COLORS = {
  'red': '#FF0000',
  'black': '#000000',
  'blue': '#0000FF',
  'green': '#00FF00',
}


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------


class Canvas:
  """
  Gesture-driven drawing surface.

  Parameters
  ----------
  width : int
      Pixel width of the drawing area (should match Tkinter canvas size).
  height : int
      Pixel height of the drawing area.
  """

  def __init__(self, width: int, height: int) -> None:
    self.width = width
    self.height = height

    # Drawing state
    self._state: DrawState = DrawState.IDLE
    self._strokes: list[Stroke] = []
    self._active: Stroke | None = None  # stroke currently being drawn
    self._smoothed: tuple[float, float] | None = None  # EMA-smoothed tip position

    # Ruler mode
    self._ruler_start: tuple[float, float] | None = None
    self._ruler_preview: tuple[float, float] | None = None  # current tip while held

    # Drawing settings
    self._color: str = COLORS['black']
    self._thickness: float = 3.0
    self._eraser_radius: float = ERASER_RADIUS_DEF

  # ------------------------------------------------------------------
  # Public properties
  # ------------------------------------------------------------------

  @property
  def state(self) -> DrawState:
    return self._state

  @property
  def thickness(self) -> float:
    return self._thickness

  @property
  def color(self) -> str:
    return self._color

  @property
  def eraser_radius(self) -> float:
    return self._eraser_radius

  # ------------------------------------------------------------------
  # Settings API (called by toolbar)
  # ------------------------------------------------------------------

  def set_color(self, color_name: str) -> None:
    """Set active color by name ('red', 'black', 'blue', 'green')."""
    self._color = COLORS.get(color_name, color_name)

  def set_thickness(self, value: float) -> None:
    """Set stroke thickness directly (clamped to THICKNESS_MIN–MAX)."""
    self._thickness = max(THICKNESS_MIN, min(THICKNESS_MAX, value))

  def set_eraser_radius(self, value: float) -> None:
    """Set eraser radius directly (clamped to ERASER_RADIUS_MIN–MAX)."""
    self._eraser_radius = max(ERASER_RADIUS_MIN, min(ERASER_RADIUS_MAX, value))

  def clear(self) -> None:
    """Clear all strokes and reset state."""
    self._strokes.clear()
    self._active = None
    self._smoothed = None
    self._ruler_start = None
    self._ruler_preview = None
    self._state = DrawState.IDLE

  # ------------------------------------------------------------------
  # Frame update — called every frame from main.py
  # ------------------------------------------------------------------

  def update(self, gr: GestureResult, frame_w: int, frame_h: int) -> None:
    """
    Advance the state machine based on the latest GestureResult.

    Parameters
    ----------
    gr : GestureResult
        Latest result from GestureEngine.
    frame_w : int
        Width of the camera frame (used to normalise tip coords to canvas).
    frame_h : int
        Height of the camera frame.
    """
    drawing = gr.drawing_gesture
    modifier = gr.modifier_gesture
    tip_raw = gr.drawing_tip
    mod_tip = gr.modifier_tip

    # Map tip from frame pixel space → canvas pixel space
    tip = self._map_tip(tip_raw, frame_w, frame_h) if tip_raw else None

    # Smooth tip position to reduce jitter
    if tip:
      tip = self._smooth(tip)

    # ------------------------------------------------------------------
    # Thumb_Up modifier: adjusts thickness while drawing, eraser size while erasing
    # ------------------------------------------------------------------
    if modifier == 'Thumb_Up' and mod_tip is not None:
      mapped_mod = self._map_tip(mod_tip, frame_w, frame_h)
      if drawing == 'Fist':
        self._update_eraser_radius_from_tip(mapped_mod)
      else:
        self._update_thickness_from_tip(mapped_mod)

    # ------------------------------------------------------------------
    # State machine transitions
    # ------------------------------------------------------------------
    
    # Handle Idle state explicitly
    if drawing == 'Idle' or drawing is None:
      self._enter_idle()
      return

    match (drawing, modifier):
      case ('Fist', _):
        self._enter_erasing(tip)

      case ('Index_Finger', 'Ruler'):
        self._enter_or_continue_ruler(tip)

      case ('Index_Finger', _):
        self._enter_or_continue_drawing(tip)

      case _:
        self._enter_idle()

  # ------------------------------------------------------------------
  # State handlers
  # ------------------------------------------------------------------

  def _enter_idle(self) -> None:
    if self._state == DrawState.DRAWING and self._active:
      self._commit_active_stroke()
    if self._state == DrawState.RULER_MODE:
      self._commit_ruler_line()
    self._state = DrawState.IDLE
    self._active = None
    self._smoothed = None

  def _enter_erasing(self, tip: tuple[float, float] | None) -> None:
    if self._state not in (DrawState.IDLE, DrawState.ERASING):
      self._enter_idle()
    self._state = DrawState.ERASING
    if tip:
      self._erase_at(tip)

  def _enter_or_continue_drawing(self, tip: tuple[float, float] | None) -> None:
    if self._state == DrawState.RULER_MODE:
      # Switched away from ruler — commit whatever was there
      self._commit_ruler_line()

    self._state = DrawState.DRAWING

    if tip is None:
      return

    if self._active is None:
      self._active = Stroke(color=self._color, width=self._thickness)

    last = self._active.points[-1] if self._active.points else None
    if last is None or _dist(last, tip) >= MIN_POINT_DIST:
      self._active.points.append(tip)

  def _enter_or_continue_ruler(self, tip: tuple[float, float] | None) -> None:
    if self._state != DrawState.RULER_MODE:
      # Commit any ongoing freehand stroke first
      if self._state == DrawState.DRAWING and self._active:
        self._commit_active_stroke()
      self._state = DrawState.RULER_MODE
      self._ruler_start = tip  # anchor point

    self._ruler_preview = tip  # update end point while held

  # ------------------------------------------------------------------
  # Commit helpers
  # ------------------------------------------------------------------

  def _commit_active_stroke(self) -> None:
    if self._active and len(self._active.points) > 1:
      self._strokes.append(self._active)
    self._active = None

  def _commit_ruler_line(self) -> None:
    if self._ruler_start and self._ruler_preview:
      stroke = Stroke(
        points=[self._ruler_start, self._ruler_preview],
        color=self._color,
        width=self._thickness,
        is_line=True,
      )
      self._strokes.append(stroke)
    self._ruler_start = None
    self._ruler_preview = None

  # ------------------------------------------------------------------
  # Erase logic
  # ------------------------------------------------------------------

  def _erase_at(self, tip: tuple[float, float]) -> None:
    """Remove any stroke that passes within _eraser_radius of tip."""
    self._strokes = [s for s in self._strokes if not _stroke_near(s, tip, self._eraser_radius)]

  # ------------------------------------------------------------------
  # Thickness / eraser radius from thumb position
  # ------------------------------------------------------------------

  def _update_thickness_from_tip(self, tip: tuple[float, float]) -> None:
    """Map horizontal thumb position (canvas coords) to stroke thickness range."""
    norm_x = max(0.0, min(1.0, tip[0] / self.width))
    self._thickness = THICKNESS_MIN + norm_x * (THICKNESS_MAX - THICKNESS_MIN)

  def _update_eraser_radius_from_tip(self, tip: tuple[float, float]) -> None:
    """Map horizontal thumb position (canvas coords) to eraser radius range."""
    norm_x = max(0.0, min(1.0, tip[0] / self.width))
    self._eraser_radius = ERASER_RADIUS_MIN + norm_x * (ERASER_RADIUS_MAX - ERASER_RADIUS_MIN)

  # ------------------------------------------------------------------
  # Coordinate mapping & smoothing
  # ------------------------------------------------------------------

  def _map_tip(
    self,
    tip: tuple[float, float],
    frame_w: int,
    frame_h: int,
  ) -> tuple[float, float]:
    """Scale tip coords from frame pixel space to canvas pixel space."""
    x = tip[0] / frame_w * self.width
    y = tip[1] / frame_h * self.height
    return (x, y)

  def _smooth(self, tip: tuple[float, float]) -> tuple[float, float]:
    """Exponential moving average to reduce hand-tracking jitter."""
    if self._smoothed is None:
      self._smoothed = tip
      return tip
    sx = SMOOTHING_ALPHA * tip[0] + (1 - SMOOTHING_ALPHA) * self._smoothed[0]
    sy = SMOOTHING_ALPHA * tip[1] + (1 - SMOOTHING_ALPHA) * self._smoothed[1]
    self._smoothed = (sx, sy)
    return self._smoothed

  # ------------------------------------------------------------------
  # Render — paint strokes onto a Tkinter Canvas widget
  # ------------------------------------------------------------------

  def render(self, tk_canvas: tk.Canvas) -> None:
    """
    Redraw all committed strokes plus the active/preview stroke.
    Called by main.py on every UI tick.

    Note: This does a full redraw each frame (delete all → repaint).
    Fine for ~60fps with typical stroke counts; optimise with tags if needed.
    """
    tk_canvas.delete('drawing')

    # Committed strokes
    for stroke in self._strokes:
      _render_stroke(tk_canvas, stroke)

    # Active freehand stroke (not yet committed)
    if self._active and len(self._active.points) > 1:
      _render_stroke(tk_canvas, self._active)

    # Eraser cursor — shows radius while in ERASING state
    if self._state == DrawState.ERASING and self._smoothed:
      cx, cy = self._smoothed
      r = self._eraser_radius
      tk_canvas.create_oval(
        cx - r,
        cy - r,
        cx + r,
        cy + r,
        outline='red',
        width=2,
        dash=(4, 3),
        tags='drawing',
      )

    # Ruler preview line
    if self._state == DrawState.RULER_MODE and self._ruler_start and self._ruler_preview:
      tk_canvas.create_line(
        *self._ruler_start,
        *self._ruler_preview,
        fill=self._color,
        width=self._thickness,
        dash=(6, 4),  # dashed preview
        tags='drawing',
      )

  # ------------------------------------------------------------------
  # SVG export
  # ------------------------------------------------------------------

  def save_svg(self, path: Path | None = None) -> Path:
    """
    Serialize all committed strokes to an SVG file.

    Parameters
    ----------
    path : Path | None
        Destination file. Defaults to Documents/drawing_YYYYMMDD_HHMMSS.svg

    Returns
    -------
    Path
        The path where the file was written.
    """
    if path is None:
      timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
      path = Path.home() / 'Documents' / f'drawing_{timestamp}.svg'

    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
      f'<svg xmlns="http://www.w3.org/2000/svg" '
      f'width="{self.width}" height="{self.height}" '
      f'viewBox="0 0 {self.width} {self.height}">',
    ]

    for stroke in self._strokes:
      lines.append(_stroke_to_svg(stroke))

    lines.append('</svg>')
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'[Canvas] Saved SVG → {path}')
    return path


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
  return math.hypot(b[0] - a[0], b[1] - a[1])


def _stroke_near(stroke: Stroke, tip: tuple[float, float], radius: float) -> bool:
  """Return True if any segment of the stroke passes within radius of tip."""
  pts = stroke.points
  if len(pts) == 1:
    return _dist(pts[0], tip) < radius
  for i in range(len(pts) - 1):
    if _point_to_segment_dist(tip, pts[i], pts[i + 1]) < radius:
      return True
  return False


def _point_to_segment_dist(
  p: tuple[float, float],
  a: tuple[float, float],
  b: tuple[float, float],
) -> float:
  """Minimum distance from point p to line segment ab."""
  dx, dy = b[0] - a[0], b[1] - a[1]
  if dx == 0 and dy == 0:
    return _dist(p, a)
  t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)))
  proj = (a[0] + t * dx, a[1] + t * dy)
  return _dist(p, proj)


def _render_stroke(tk_canvas: tk.Canvas, stroke: Stroke) -> None:
  """Draw a single Stroke onto a Tkinter Canvas widget."""
  if len(stroke.points) < 2:  # noqa: PLR2004
    return
  coords: list[float] = []
  for x, y in stroke.points:
    coords.extend([x, y])
  tk_canvas.create_line(
    *coords,
    fill=stroke.color,
    width=stroke.width,
    smooth=True,
    capstyle='round',
    joinstyle='round',
    tags='drawing',
  )


def _stroke_to_svg(stroke: Stroke) -> str:
  """Convert a Stroke to an SVG <polyline> or <line> element."""
  if len(stroke.points) < 2:  # noqa: PLR2004
    return ''
  if stroke.is_line:
    x1, y1 = stroke.points[0]
    x2, y2 = stroke.points[1]
    return (
      f'  <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
      f'stroke="{stroke.color}" stroke-width="{stroke.width:.1f}" '
      f'stroke-linecap="round"/>'
    )
  pts_str = ' '.join(f'{x:.1f},{y:.1f}' for x, y in stroke.points)
  return (
    f'  <polyline points="{pts_str}" '
    f'fill="none" stroke="{stroke.color}" stroke-width="{stroke.width:.1f}" '
    f'stroke-linecap="round" stroke-linejoin="round"/>'
  )