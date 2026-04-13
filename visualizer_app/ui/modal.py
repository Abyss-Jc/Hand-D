"""
ui/modal.py
-----------
First-launch welcome modal for the Hand-D visualizer.

Shows:
  - Handedness selection (Right / Left)
  - Gesture reference guide (contextual to chosen hand)
  - "Don't show again" checkbox

Calls `on_confirm(drawing_hand, skip_next)` when the user confirms.
Blocks the parent window until dismissed (modal behaviour via grab_set).
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------

_GESTURES: dict[str, list[tuple[str, str, str]]] = {
  'Right': [
    ('✏️', 'Index Finger  (right)', 'Draw freely on the canvas'),
    ('✊', 'Fist  (right)', 'Erase strokes near your fist'),
    ('👍', 'Thumb Up  (left)', 'Adjust thickness or eraser size\n(move thumb left ↔ right)'),
    ('📏', 'Ruler Gesture  (left)', 'Draw a straight line\n(hold while drawing hand moves)'),
  ],
  'Left': [
    ('✏️', 'Index Finger  (left)', 'Draw freely on the canvas'),
    ('✊', 'Fist  (left)', 'Erase strokes near your fist'),
    ('👍', 'Thumb Up  (right)', 'Adjust thickness or eraser size\n(move thumb left ↔ right)'),
    ('📏', 'Ruler Gesture  (right)', 'Draw a straight line\n(hold while drawing hand moves)'),
  ],
}

_SHORTCUTS = [
  ('M', 'Toggle Camera / Dark Mode'),
  ('C', 'Clear canvas'),
  ('S', 'Save as SVG'),
  ('1', 'Color → Red'),
  ('2', 'Color → Black'),
  ('3', 'Color → Blue'),
  ('4', 'Color → Green'),
]

# ---------------------------------------------------------------------------
# Palette (kept minimal — main app owns the theme)
# ---------------------------------------------------------------------------

_BG = '#1e1e2e'
_SURFACE = '#2a2a3e'
_ACCENT = '#89b4fa'
_TEXT = '#cdd6f4'
_SUBTEXT = '#a6adc8'
_BORDER = '#45475a'
_BTN_BG = '#313244'
_BTN_ACT = '#45475a'
_CONFIRM = '#a6e3a1'
_CONFIRM_FG = '#1e1e2e'


# ---------------------------------------------------------------------------
# WelcomeModal
# ---------------------------------------------------------------------------


class WelcomeModal(tk.Toplevel):
  """
  Modal dialog shown on first launch (or when skip_welcome is False).

  Parameters
  ----------
  parent : tk.Tk
      The root window — modal blocks interaction with it.
  on_confirm : Callable[[str, bool], None]
      Called with (drawing_hand, skip_next) when user clicks Confirm.
  initial_hand : str
      Pre-selected handedness ('Right' or 'Left').
  """

  def __init__(
    self,
    parent: tk.Tk,
    on_confirm: Callable[[str, bool], None],
    initial_hand: str = 'Right',
  ) -> None:
    super().__init__(parent)
    self._on_confirm = on_confirm
    self._hand_var = tk.StringVar(value=initial_hand)
    self._skip_var = tk.BooleanVar(value=False)

    self._build_window()
    self._build_ui()

    # Block parent until this window is closed
    self.grab_set()
    self.focus_force()
    self.wait_window()

  # ------------------------------------------------------------------
  # Window setup
  # ------------------------------------------------------------------

  def _build_window(self) -> None:
    self.title('Welcome to Hand-D')
    self.configure(bg=_BG)
    self.resizable(False, False)

    # Center on screen
    self.update_idletasks()
    w, h = 540, 620
    sw = self.winfo_screenwidth()
    sh = self.winfo_screenheight()
    x = (sw - w) // 2
    y = (sh - h) // 2
    self.geometry(f'{w}x{h}+{x}+{y}')

    # Close button maps to confirm with current selections
    self.protocol('WM_DELETE_WINDOW', self._confirm)

  # ------------------------------------------------------------------
  # UI construction
  # ------------------------------------------------------------------

  def _build_ui(self) -> None:
    root_frame = tk.Frame(self, bg=_BG, padx=24, pady=20)
    root_frame.pack(fill='both', expand=True)

    # --- Title ---
    tk.Label(
      root_frame,
      text='✋  Hand-D',
      font=('Segoe UI', 22, 'bold'),
      bg=_BG,
      fg=_ACCENT,
    ).pack(anchor='w')

    tk.Label(
      root_frame,
      text='Gesture-controlled virtual whiteboard',
      font=('Segoe UI', 10),
      bg=_BG,
      fg=_SUBTEXT,
    ).pack(anchor='w', pady=(0, 16))

    _divider(root_frame)

    # --- Handedness selection ---
    tk.Label(
      root_frame,
      text='Which is your drawing hand?',
      font=('Segoe UI', 11, 'bold'),
      bg=_BG,
      fg=_TEXT,
    ).pack(anchor='w', pady=(14, 8))

    hand_row = tk.Frame(root_frame, bg=_BG)
    hand_row.pack(anchor='w')

    for hand in ('Right', 'Left'):
      tk.Radiobutton(
        hand_row,
        text=hand,
        variable=self._hand_var,
        value=hand,
        font=('Segoe UI', 11),
        bg=_BG,
        fg=_TEXT,
        selectcolor=_SURFACE,
        activebackground=_BG,
        activeforeground=_ACCENT,
        command=self._refresh_gestures,
      ).pack(side='left', padx=(0, 20))

    _divider(root_frame, pady=(14, 0))

    # --- Gesture guide ---
    tk.Label(
      root_frame,
      text='Gestures',
      font=('Segoe UI', 11, 'bold'),
      bg=_BG,
      fg=_TEXT,
    ).pack(anchor='w', pady=(12, 6))

    self._gesture_frame = tk.Frame(root_frame, bg=_BG)
    self._gesture_frame.pack(fill='x')
    self._build_gesture_rows()

    _divider(root_frame, pady=(12, 0))

    # --- Keyboard shortcuts ---
    tk.Label(
      root_frame,
      text='Keyboard shortcuts',
      font=('Segoe UI', 11, 'bold'),
      bg=_BG,
      fg=_TEXT,
    ).pack(anchor='w', pady=(12, 6))

    shortcuts_frame = tk.Frame(root_frame, bg=_BG)
    shortcuts_frame.pack(fill='x')

    for i, (key, desc) in enumerate(_SHORTCUTS):
      col = i % 2
      row = i // 2
      cell = tk.Frame(shortcuts_frame, bg=_BG)
      cell.grid(row=row, column=col, sticky='w', padx=(0, 16), pady=1)
      tk.Label(cell, text=f'[{key}]', font=('Consolas', 10, 'bold'), bg=_BG, fg=_ACCENT, width=4, anchor='w').pack(
        side='left'
      )
      tk.Label(cell, text=desc, font=('Segoe UI', 10), bg=_BG, fg=_SUBTEXT).pack(side='left')

    _divider(root_frame, pady=(14, 0))

    # --- Footer: checkbox + confirm button ---
    footer = tk.Frame(root_frame, bg=_BG)
    footer.pack(fill='x', pady=(14, 0))

    tk.Checkbutton(
      footer,
      text="Don't show this again",
      variable=self._skip_var,
      font=('Segoe UI', 10),
      bg=_BG,
      fg=_SUBTEXT,
      selectcolor=_SURFACE,
      activebackground=_BG,
      activeforeground=_TEXT,
    ).pack(side='left')

    tk.Button(
      footer,
      text="Let's draw  →",
      font=('Segoe UI', 11, 'bold'),
      bg=_CONFIRM,
      fg=_CONFIRM_FG,
      activebackground=_BTN_ACT,
      activeforeground=_TEXT,
      relief='flat',
      padx=16,
      pady=6,
      cursor='hand2',
      command=self._confirm,
    ).pack(side='right')

  # ------------------------------------------------------------------
  # Gesture rows (rebuilt when handedness changes)
  # ------------------------------------------------------------------

  def _build_gesture_rows(self) -> None:
    for widget in self._gesture_frame.winfo_children():
      widget.destroy()

    gestures = _GESTURES[self._hand_var.get()]

    for emoji, name, desc in gestures:
      row = tk.Frame(self._gesture_frame, bg=_SURFACE, pady=6, padx=10)
      row.pack(fill='x', pady=3)

      tk.Label(row, text=emoji, font=('Segoe UI Emoji', 16), bg=_SURFACE, fg=_TEXT, width=3).pack(side='left')

      text_col = tk.Frame(row, bg=_SURFACE)
      text_col.pack(side='left', padx=(6, 0))

      tk.Label(text_col, text=name, font=('Segoe UI', 10, 'bold'), bg=_SURFACE, fg=_TEXT, anchor='w').pack(anchor='w')
      tk.Label(text_col, text=desc, font=('Segoe UI', 9), bg=_SURFACE, fg=_SUBTEXT, anchor='w', justify='left').pack(
        anchor='w'
      )

  def _refresh_gestures(self) -> None:
    """Called when user switches handedness radio — updates gesture rows."""
    self._build_gesture_rows()

  # ------------------------------------------------------------------
  # Confirm
  # ------------------------------------------------------------------

  def _confirm(self) -> None:
    drawing_hand = self._hand_var.get()
    skip_next = self._skip_var.get()
    self.destroy()
    self._on_confirm(drawing_hand, skip_next)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _divider(parent: tk.Frame, pady: tuple[int, int] = (0, 0)) -> None:
  tk.Frame(parent, bg=_BORDER, height=1).pack(fill='x', pady=pady)
