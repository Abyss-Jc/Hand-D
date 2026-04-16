"""
ui/toolbar.py
-------------
Top toolbar and bottom status bar for the Hand-D visualizer.

Toolbar contains:
  - Color swatches (Red, Black, Blue, Green)
  - Clear button
  - Save SVG button
  - Mode toggle (Camera / Dark Mode)

Status bar shows:
  - Current app state (Ready, Drawing, Erasing…)
  - Active stroke thickness
  - Eraser radius
  - Active color name

Both widgets are pure Tkinter — they hold callbacks injected by main.py
and never import canvas.py or gesture_engine.py directly.
"""

from __future__ import annotations

import tkinter as tk
from typing import Callable

# ---------------------------------------------------------------------------
# Palette (matches modal.py)
# ---------------------------------------------------------------------------

_BG = '#1e1e2e'
_SURFACE = '#2a2a3e'
_BORDER = '#45475a'
_TEXT = '#cdd6f4'
_SUBTEXT = '#a6adc8'
_ACCENT = '#89b4fa'
_BTN_BG = '#313244'
_BTN_ACT = '#45475a'
_BTN_DANGER = '#f38ba8'
_BTN_SAVE = '#a6e3a1'
_BTN_FG = '#1e1e2e'

# Color swatches — (display hex, label, color_name key for canvas.set_color)
_COLOR_SWATCHES: list[tuple[str, str, str]] = [
  ('#FF0000', '1  Red', 'red'),
  ('#cdd6f4', '2  Black', 'black'),
  ('#89b4fa', '3  Blue', 'blue'),
  ('#a6e3a1', '4  Green', 'green'),
]

# Mode labels
_MODE_CAMERA = '📷  Camera'
_MODE_DARK = '🌑  Dark'


# ---------------------------------------------------------------------------
# Toolbar
# ---------------------------------------------------------------------------


class Toolbar(tk.Frame):
  """
  Horizontal toolbar packed at the top of the main window.

  Parameters
  ----------
  parent : tk.Widget
      Parent widget (root window or containing frame).
  on_color : Callable[[str], None]
      Called with color name key when a swatch is clicked.
  on_clear : Callable[[], None]
      Called when Clear is clicked.
  on_save : Callable[[], None]
      Called when Save SVG is clicked.
  on_mode_toggle : Callable[[], None]
      Called when the mode toggle button is clicked.
  """

  def __init__(
    self,
    parent: tk.Widget,
    on_color: Callable[[str], None],
    on_clear: Callable[[], None],
    on_save: Callable[[], None],
    on_mode_toggle: Callable[[], None],
  ) -> None:
    super().__init__(parent, bg=_BG, pady=6, padx=10)
    self._on_color = on_color
    self._on_clear = on_clear
    self._on_save = on_save
    self._on_mode_toggle = on_mode_toggle

    self._active_color: str = 'black'
    self._camera_mode: bool = True  # True = Camera, False = Dark
    self._swatch_btns: dict[str, tk.Button] = {}

    self._build()

  # ------------------------------------------------------------------
  # Build
  # ------------------------------------------------------------------

  def _build(self) -> None:
    # ---- Color swatches ----
    swatch_frame = tk.Frame(self, bg=_BG)
    swatch_frame.pack(side='left')

    tk.Label(
      swatch_frame,
      text='Color',
      font=('Segoe UI', 9),
      bg=_BG,
      fg=_SUBTEXT,
    ).pack(side='left', padx=(0, 6))

    for hex_color, label, name in _COLOR_SWATCHES:
      btn = tk.Button(
        swatch_frame,
        text=label,
        font=('Segoe UI', 10),
        bg=_BTN_BG,
        fg=hex_color,
        activebackground=_BTN_ACT,
        activeforeground=hex_color,
        relief='flat',
        padx=10,
        pady=4,
        cursor='hand2',
        command=lambda n=name: self._select_color(n),
      )
      btn.pack(side='left', padx=2)
      self._swatch_btns[name] = btn

    # Highlight default
    self._highlight_swatch('black')

    _vsep(self)

    # ---- Clear ----
    tk.Button(
      self,
      text='🗑  Clear',
      font=('Segoe UI', 10),
      bg=_BTN_BG,
      fg=_BTN_DANGER,
      activebackground=_BTN_ACT,
      activeforeground=_BTN_DANGER,
      relief='flat',
      padx=10,
      pady=4,
      cursor='hand2',
      command=self._on_clear,
    ).pack(side='left', padx=2)

    # ---- Save ----
    tk.Button(
      self,
      text='💾  Save SVG',
      font=('Segoe UI', 10),
      bg=_BTN_BG,
      fg=_BTN_SAVE,
      activebackground=_BTN_ACT,
      activeforeground=_BTN_SAVE,
      relief='flat',
      padx=10,
      pady=4,
      cursor='hand2',
      command=self._on_save,
    ).pack(side='left', padx=2)

    _vsep(self)

    # ---- Mode toggle ----
    self._mode_btn = tk.Button(
      self,
      text=_MODE_CAMERA,
      font=('Segoe UI', 10, 'bold'),
      bg=_BTN_BG,
      fg=_ACCENT,
      activebackground=_BTN_ACT,
      activeforeground=_ACCENT,
      relief='flat',
      padx=10,
      pady=4,
      cursor='hand2',
      command=self._toggle_mode,
    )
    self._mode_btn.pack(side='left', padx=2)

  # ------------------------------------------------------------------
  # Interaction handlers
  # ------------------------------------------------------------------

  def _select_color(self, name: str) -> None:
    self._active_color = name
    self._highlight_swatch(name)
    self._on_color(name)

  def _toggle_mode(self) -> None:
    self._camera_mode = not self._camera_mode
    self._mode_btn.config(text=_MODE_CAMERA if self._camera_mode else _MODE_DARK)
    self._on_mode_toggle()

  # ------------------------------------------------------------------
  # Visual state
  # ------------------------------------------------------------------

  def _highlight_swatch(self, active: str) -> None:
    """Add underline relief to active swatch, flat to all others."""
    for name, btn in self._swatch_btns.items():
      btn.config(relief='sunken' if name == active else 'flat')

  def set_camera_mode(self, camera: bool) -> None:
    """Sync button label if mode is changed externally (e.g. keyboard shortcut)."""
    self._camera_mode = camera
    self._mode_btn.config(text=_MODE_CAMERA if camera else _MODE_DARK)

  @property
  def camera_mode(self) -> bool:
    return self._camera_mode


# ---------------------------------------------------------------------------
# StatusBar
# ---------------------------------------------------------------------------


class StatusBar(tk.Frame):
  """
  Single-line status bar packed at the bottom of the main window.

  Shows: state label | thickness | eraser radius | active color
  """

  def __init__(self, parent: tk.Widget) -> None:
    super().__init__(parent, bg=_SURFACE, pady=4, padx=10)
    self._build()

  def _build(self) -> None:
    self._state_lbl = _status_label(self, 'Ready')
    _vsep(self, vertical=False)
    self._thickness_lbl = _status_label(self, 'Thickness: 3px')
    _vsep(self, vertical=False)
    self._eraser_lbl = _status_label(self, 'Eraser: 30px')
    _vsep(self, vertical=False)
    self._color_lbl = _status_label(self, 'Color: Black')

  # ------------------------------------------------------------------
  # Update API — called from main.py every UI tick
  # ------------------------------------------------------------------

  def update_state(self, state_name: str) -> None:
    self._state_lbl.config(text=f'State: {state_name}')

  def update_thickness(self, value: float) -> None:
    self._thickness_lbl.config(text=f'Thickness: {value:.0f}px')

  def update_eraser(self, value: float) -> None:
    self._eraser_lbl.config(text=f'Eraser: {value:.0f}px')

  def update_color(self, color_name: str) -> None:
    self._color_lbl.config(text=f'Color: {color_name.capitalize()}')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vsep(parent: tk.Widget, vertical: bool = True) -> None:
  """Thin separator line — vertical for toolbar, horizontal for status bar."""
  if vertical:
    tk.Frame(parent, bg=_BORDER, width=1).pack(side='left', fill='y', padx=8, pady=2)
  else:
    tk.Frame(parent, bg=_BORDER, width=1).pack(side='left', fill='y', padx=6, pady=2)


def _status_label(parent: tk.Widget, text: str) -> tk.Label:
  lbl = tk.Label(
    parent,
    text=text,
    font=('Segoe UI', 9),
    bg=_SURFACE,
    fg=_SUBTEXT,
  )
  lbl.pack(side='left')
  return lbl
