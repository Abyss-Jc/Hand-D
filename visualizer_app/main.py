"""
main.py
-------
Entry point for the Hand-D gesture-controlled virtual whiteboard.

Wires together:
  - Config          : persistent user preferences
  - WelcomeModal    : first-launch handedness + guide
  - GestureEngine   : background thread (MediaPipe + model inference)
  - Canvas          : drawing state machine + stroke management
  - Toolbar         : color, clear, save, mode toggle
  - StatusBar       : live state readout

Threading model
---------------
GestureEngine runs in a daemon thread and deposits GestureResult objects
into `_latest_result` (a one-slot buffer protected by a Lock).
Tkinter's `after(16, _ui_tick)` loop reads that buffer on the main thread
and drives canvas updates + rendering — never touching Tkinter from the
worker thread.
"""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from threading import Lock
from tkinter import filedialog, messagebox

# Ensure visualizer_app/ siblings are importable when run as `python main.py`
sys.path.insert(0, str(Path(__file__).parent))

from canvas import Canvas
from config import Config
from gesture_engine import GestureEngine, GestureResult
from ui.modal import WelcomeModal
from ui.toolbar import StatusBar, Toolbar

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_TITLE = 'Hand-D  —  Gesture Whiteboard'
UI_TICK_MS = 16  # ~60 fps UI refresh
BG_CAMERA = '#000000'  # canvas bg in camera mode (frame fills it)
BG_DARK = '#000000'  # canvas bg in dark mode
LANDMARK_COLOR = '#00FF00'
LANDMARK_RADIUS = 3


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


class App:
  """Main application controller."""

  def __init__(self) -> None:
    self._cfg = Config.load()
    self._root = tk.Tk()
    self._lock = Lock()
    self._latest: GestureResult | None = None

    # Mode flag — True = camera feed visible, False = dark/landmark mode
    self._camera_mode = True

    self._build_window()
    self._build_layout()
    self._bind_shortcuts()

    # Canvas (UI-framework-agnostic drawing logic)
    self._canvas = Canvas(
      width=self._tk_canvas.winfo_reqwidth(),
      height=self._tk_canvas.winfo_reqheight(),
    )
    self._canvas.set_color(self._cfg.default_color)
    self._canvas.set_thickness(self._cfg.default_thickness)
    self._canvas.set_eraser_radius(self._cfg.default_eraser)

    # GestureEngine starts after modal resolves handedness
    self._engine: GestureEngine | None = None

    # Show welcome modal (blocks until confirmed)
    if not self._cfg.skip_welcome:
      WelcomeModal(
        self._root,
        on_confirm=self._on_modal_confirm,
        initial_hand=self._cfg.drawing_hand,
      )
    else:
      self._start_engine(self._cfg.drawing_hand)

    # Kick off UI tick loop
    self._root.after(UI_TICK_MS, self._ui_tick)

  # ------------------------------------------------------------------
  # Window + layout
  # ------------------------------------------------------------------

  def _build_window(self) -> None:
    self._root.title(APP_TITLE)
    self._root.configure(bg='#1e1e2e')
    self._root.state('normal')  # maximized; works on Windows + most Linux WMs
    self._root.protocol('WM_DELETE_WINDOW', self._on_close)

  def _build_layout(self) -> None:
    # Toolbar — top
    self._toolbar = Toolbar(
      self._root,
      on_color=self._on_color,
      on_clear=self._on_clear,
      on_save=self._on_save,
      on_mode_toggle=self._on_mode_toggle,
    )
    self._toolbar.pack(side='top', fill='x')

    # Thin separator
    tk.Frame(self._root, bg='#45475a', height=1).pack(side='top', fill='x')

    # Status bar — bottom
    self._status = StatusBar(self._root)
    self._status.pack(side='bottom', fill='x')

    tk.Frame(self._root, bg='#45475a', height=1).pack(side='bottom', fill='x')

    # Drawing canvas — fills remaining space
    self._tk_canvas = tk.Canvas(
      self._root,
      bg=BG_CAMERA,
      highlightthickness=0,
    )
    self._tk_canvas.pack(side='top', fill='both', expand=True)

  # ------------------------------------------------------------------
  # Keyboard shortcuts
  # ------------------------------------------------------------------

  def _bind_shortcuts(self) -> None:
    self._root.bind('<KeyPress-w>', lambda _e: self._show_welcome_manual())
    self._root.bind('<KeyPress-W>', lambda _e: self._show_welcome_manual())
    self._root.bind('<KeyPress-m>', lambda _e: self._on_mode_toggle())
    self._root.bind('<KeyPress-M>', lambda _e: self._on_mode_toggle())
    self._root.bind('<KeyPress-c>', lambda _e: self._on_clear())
    self._root.bind('<KeyPress-C>', lambda _e: self._on_clear())
    self._root.bind('<KeyPress-s>', lambda _e: self._on_save())
    self._root.bind('<KeyPress-S>', lambda _e: self._on_save())
    self._root.bind('<KeyPress-1>', lambda _e: self._on_color('red'))
    self._root.bind('<KeyPress-2>', lambda _e: self._on_color('black'))
    self._root.bind('<KeyPress-3>', lambda _e: self._on_color('blue'))
    self._root.bind('<KeyPress-4>', lambda _e: self._on_color('green'))

  # ------------------------------------------------------------------
  # Modal callback
  # ------------------------------------------------------------------

  def _on_modal_confirm(self, drawing_hand: str, skip_next: bool) -> None:
    self._cfg.drawing_hand = drawing_hand
    self._cfg.skip_welcome = skip_next
    self._cfg.save()
    self._start_engine(drawing_hand)

  # ------------------------------------------------------------------
  # GestureEngine lifecycle
  # ------------------------------------------------------------------

  def _start_engine(self, drawing_hand: str) -> None:
    if self._engine is not None:
      self._engine.stop()

    self._engine = GestureEngine(
      callback=self._on_gesture_result,
      drawing_hand=drawing_hand,
      camera_index=self._cfg.camera_index,
    )
    self._engine.start()

  def _on_gesture_result(self, gr: GestureResult) -> None:
    """Called from GestureEngine worker thread — only write to buffer."""
    with self._lock:
      self._latest = gr

  # ------------------------------------------------------------------
  # UI tick — runs on main thread at ~60fps
  # ------------------------------------------------------------------

  def _ui_tick(self) -> None:
    with self._lock:
      gr = self._latest
      self._latest = None

    if gr is not None:
      self._process_frame(gr)

    self._root.after(UI_TICK_MS, self._ui_tick)

  def _process_frame(self, gr: GestureResult) -> None:
    # Canvas dimensions may have changed (window resize)
    cw = self._tk_canvas.winfo_width()
    ch = self._tk_canvas.winfo_height()
    self._canvas.width = cw
    self._canvas.height = ch

    # Derive frame dimensions from BGR frame if available
    frame_w, frame_h = cw, ch
    if gr.frame_bgr is not None:
      frame_h, frame_w = gr.frame_bgr.shape[:2]

    # Advance drawing state machine
    self._canvas.update(gr, frame_w, frame_h)

    # --- Background ---
    self._tk_canvas.delete('background')

    if self._camera_mode and gr.frame_bgr is not None:
      self._draw_camera_frame(gr.frame_bgr, cw, ch)
    else:
      self._tk_canvas.configure(bg=BG_DARK)
      if not self._camera_mode:
        self._draw_landmarks(gr, cw, ch, frame_w, frame_h)

    # --- Strokes ---
    self._canvas.render(self._tk_canvas)

    # --- Status bar ---
    self._status.update_state(self._canvas.state.name.replace('_', ' ').title())
    self._status.update_thickness(self._canvas.thickness)
    self._status.update_eraser(self._canvas.eraser_radius)
    self._status.update_color(self._canvas.color)

  # ------------------------------------------------------------------
  # Camera frame rendering
  # ------------------------------------------------------------------

  def _draw_camera_frame(self, frame_bgr, cw: int, ch: int) -> None:  # type: ignore[name-defined]
    """Convert OpenCV BGR frame to a Tkinter PhotoImage and draw it."""
    try:
      import cv2
      from PIL import Image, ImageTk  # pillow

      rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
      img = Image.fromarray(rgb).resize((cw, ch), Image.BILINEAR)
      photo = ImageTk.PhotoImage(img)

      # Keep a reference so GC doesn't collect it before Tkinter renders
      self._camera_photo = photo
      self._tk_canvas.create_image(0, 0, anchor='nw', image=photo, tags='background')
    except Exception as e:  # noqa: BLE001
      print(f'[main] Camera frame render error: {e}')

  # ------------------------------------------------------------------
  # Dark mode landmark rendering
  # ------------------------------------------------------------------

  def _draw_landmarks(
    self,
    gr: GestureResult,
    cw: int,
    ch: int,
    frame_w: int,
    frame_h: int,
  ) -> None:
    from gesture_engine import HAND_CONNECTIONS

    for lms in gr.landmarks_2d:
      # Points
      for lm in lms:
        x = int(lm.x * frame_w / frame_w * cw)
        y = int(lm.y * frame_h / frame_h * ch)
        r = LANDMARK_RADIUS
        self._tk_canvas.create_oval(
          x - r,
          y - r,
          x + r,
          y + r,
          fill=LANDMARK_COLOR,
          outline='',
          tags='background',
        )
      # Connections
      for i, j in HAND_CONNECTIONS:
        if i < len(lms) and j < len(lms):
          x1 = int(lms[i].x * cw)
          y1 = int(lms[i].y * ch)
          x2 = int(lms[j].x * cw)
          y2 = int(lms[j].y * ch)
          self._tk_canvas.create_line(
            x1,
            y1,
            x2,
            y2,
            fill=LANDMARK_COLOR,
            width=1,
            tags='background',
          )

  # ------------------------------------------------------------------
  # Toolbar callbacks
  # ------------------------------------------------------------------

  def _on_color(self, name: str) -> None:
    self._canvas.set_color(name)
    self._toolbar._highlight_swatch(name)  # noqa: SLF001
    self._cfg.default_color = name
    self._cfg.save()

  def _on_clear(self) -> None:
    self._canvas.clear()

  def _on_save(self) -> None:
    path_str = filedialog.asksaveasfilename(
      defaultextension='.svg',
      filetypes=[('SVG files', '*.svg'), ('All files', '*.*')],
      initialfile=f'drawing.svg',
      title='Save drawing as SVG',
    )
    if path_str:
      try:
        saved = self._canvas.save_svg(Path(path_str))
        messagebox.showinfo('Saved', f'Drawing saved to:\n{saved}')
      except Exception as e:  # noqa: BLE001
        messagebox.showerror('Save failed', str(e))

  def _on_mode_toggle(self) -> None:
    self._camera_mode = not self._camera_mode
    self._toolbar.set_camera_mode(self._camera_mode)
    
    # Intelligent Color Switching
    current_color = self._canvas.color
    
    if not self._camera_mode:  # Entering Dark Mode (Black Background)
      if current_color.lower() == 'black':
        self._canvas.set_color('white')
        self._toolbar._highlight_swatch('white')  # noqa: SLF001
    else:  # Returning to Camera Mode (Light/Real Background)
      if current_color.lower() == 'white':
        self._canvas.set_color('black')
        self._toolbar._highlight_swatch('black')  # noqa: SLF001

  def _show_welcome_manual(self) -> None:
    """Allows opening the modal even if 'Don't show again' was checked."""
    # Temporarily pause the engine to avoid thread conflicts
    if self._engine is not None:
      self._engine.pause()
    WelcomeModal(
      self._root,
      on_confirm=self._on_modal_confirm,
      initial_hand=self._cfg.drawing_hand,
    )
    if self._engine is not None:
      self._engine.resume()

  # ------------------------------------------------------------------
  # Shutdown
  # ------------------------------------------------------------------

  def _on_close(self) -> None:
    if self._engine is not None:
      self._engine.stop()
    self._root.destroy()

  # ------------------------------------------------------------------
  # Run
  # ------------------------------------------------------------------

  def run(self) -> None:
    self._root.mainloop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
  App().run()