"""
config.py
---------
Persistent user preferences for the Hand-D visualizer.

Stores and loads a JSON file at ~/.hand-d/config.json with:
  - drawing_hand      : "Right" | "Left"
  - skip_welcome      : bool  (don't show modal on next launch)
  - default_color     : str   (color name key)
  - default_thickness : float
  - default_eraser    : float (eraser radius)
  - camera_index      : int
"""

from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

CONFIG_PATH = Path.home() / '.hand-d' / 'config.json'

DEFAULTS: dict[str, object] = {
  'drawing_hand': 'Right',
  'skip_welcome': False,
  'default_color': 'black',
  'default_thickness': 3.0,
  'default_eraser': 30.0,
  'camera_index': 0,
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class Config:
  """
  Thin wrapper around a JSON config file.

  Usage
  -----
  cfg = Config.load()       # load from disk (or defaults if missing)
  cfg.drawing_hand          # read a value
  cfg.drawing_hand = 'Left' # write a value
  cfg.save()                # persist to disk
  """

  def __init__(self, data: dict[str, object]) -> None:
    self._data = data

  # ------------------------------------------------------------------
  # Persistence
  # ------------------------------------------------------------------

  @classmethod
  def load(cls) -> Config:
    """Load config from disk. Returns defaults if file is missing or corrupt."""
    if CONFIG_PATH.exists():
      try:
        raw = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
        # Merge with defaults so new keys added in future versions are present
        merged = {**DEFAULTS, **raw}
        return cls(merged)
      except (json.JSONDecodeError, OSError) as e:
        print(f'[Config] Failed to read config ({e}), using defaults.')
    return cls(dict(DEFAULTS))

  def save(self) -> None:
    """Persist current config to disk."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(self._data, indent=2), encoding='utf-8')

  # ------------------------------------------------------------------
  # Typed properties
  # ------------------------------------------------------------------

  @property
  def drawing_hand(self) -> str:
    return str(self._data['drawing_hand'])

  @drawing_hand.setter
  def drawing_hand(self, value: str) -> None:
    if value not in ('Right', 'Left'):
      msg = f"drawing_hand must be 'Right' or 'Left', got {value!r}"
      raise ValueError(msg)
    self._data['drawing_hand'] = value

  @property
  def skip_welcome(self) -> bool:
    return bool(self._data['skip_welcome'])

  @skip_welcome.setter
  def skip_welcome(self, value: bool) -> None:
    self._data['skip_welcome'] = value

  @property
  def default_color(self) -> str:
    return str(self._data['default_color'])

  @default_color.setter
  def default_color(self, value: str) -> None:
    self._data['default_color'] = value

  @property
  def default_thickness(self) -> float:
    return float(self._data['default_thickness'])  # type: ignore[arg-type]

  @default_thickness.setter
  def default_thickness(self, value: float) -> None:
    self._data['default_thickness'] = value

  @property
  def default_eraser(self) -> float:
    return float(self._data['default_eraser'])  # type: ignore[arg-type]

  @default_eraser.setter
  def default_eraser(self, value: float) -> None:
    self._data['default_eraser'] = value

  @property
  def camera_index(self) -> int:
    return int(self._data['camera_index'])  # type: ignore[arg-type]

  @camera_index.setter
  def camera_index(self, value: int) -> None:
    self._data['camera_index'] = value