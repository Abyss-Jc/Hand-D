"""
gesture_engine.py
-----------------
Handles MediaPipe hand tracking, feature canonicalization, and gesture inference.
Runs in a background daemon thread; delivers results via a callback.

Expected model: models/gesture_mlp.pth  (state_dict saved with torch.save(model.state_dict(), ...))

GestureMLP architecture:
    Linear(69 -> 128) -> ReLU -> Dropout(0.2)
    -> Linear(128 -> 64) -> ReLU
    -> Linear(64 -> 4)

Output classes (Josué's label encoding order):
    0: Fist
    1: Index_Finger
    2: Ruler_Gesture
    3: Thumb_Up
"""

import time
import urllib.request
from pathlib import Path
from threading import Thread
from typing import Callable

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Torch is optional until the real model arrives — stub kicks in if missing
try:
  import torch
  import torch.nn as nn

  TORCH_AVAILABLE = True
except ImportError:
  TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LANDMARK_MODEL_PATH = Path('models/hand_landmarker.task')
LANDMARK_MODEL_URL = (
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
)
GESTURE_MODEL_PATH = Path('models/gesture_mlp.pth')

# Order must match Josué's LabelEncoder output — verified from prediction_test.py
GESTURE_LABELS = ['Fist', 'Index_Finger', 'Ruler', 'Thumb_Up', 'Idle']

# MediaPipe hand connections (for Dark Mode landmark drawing)
HAND_CONNECTIONS: list[tuple[int, int]] = [
  (0, 1),
  (1, 2),
  (2, 3),
  (3, 4),
  (0, 5),
  (5, 6),
  (6, 7),
  (7, 8),
  (0, 9),
  (9, 10),
  (10, 11),
  (11, 12),
  (0, 13),
  (13, 14),
  (14, 15),
  (15, 16),
  (0, 17),
  (17, 18),
  (18, 19),
  (19, 20),
  (5, 9),
  (9, 13),
  (13, 17),
]

INDEX_TIP = 8  # landmark index for fingertip used in drawing


# ---------------------------------------------------------------------------
# Feature canonicalization  (mirrors data_extractor.py exactly)
# ---------------------------------------------------------------------------


def canonicalize(landmarks_3d: np.ndarray, raw_mp_handedness: str) -> np.ndarray | None:
  """
  Takes raw hand_world_landmarks (21x3 array) and MediaPipe's raw handedness
  string ("Left" or "Right" — mirrored), returns a 69-float feature vector
  identical to what was used at training time.

  Returns None if the hand orientation is unstable / degenerate.
  """
  wrist = landmarks_3d[0]
  pts = landmarks_3d - wrist

  # Mirror correction: MediaPipe flips left/right in selfie mode
  if raw_mp_handedness == 'Left':
    pts[:, 0] = -pts[:, 0]

  # Scale by mean wrist→MCP distance
  mcps = pts[[5, 9, 13, 17]]
  scale = np.mean(np.linalg.norm(mcps - pts[0], axis=1))
  if scale < 1e-3:
    return None
  pts = pts / scale

  # Build canonical frame from palm plane
  p_wrist = pts[0]
  p_index_mcp = pts[5]
  p_pinky_mcp = pts[17]
  p_middle_mcp = pts[9]

  global_y = p_middle_mcp - p_wrist
  norm_y = np.linalg.norm(global_y)
  if norm_y < 1e-3:
    return None
  global_y = global_y / norm_y

  vec1 = p_index_mcp - p_wrist
  vec2 = p_pinky_mcp - p_wrist
  cross = np.cross(vec1, vec2)
  if np.linalg.norm(cross) < 1e-3:
    return None
  global_z = cross / np.linalg.norm(cross)

  global_x = np.cross(global_y, global_z)
  norm_x = np.linalg.norm(global_x)
  if norm_x < 1e-3:
    return None
  global_x = global_x / norm_x

  global_y = np.cross(global_z, global_x)
  global_y = global_y / np.linalg.norm(global_y)

  rotation_matrix = np.stack([global_x, global_y, global_z], axis=1)
  canonical = np.dot(pts, rotation_matrix)

  if np.any(np.abs(canonical) > 4.0):
    return None

  # Note: handedness_binary is NOT included — Josué's model was trained on
  # 63 (canonical coords) + 3 (global_y) + 3 (global_z) = 69 features only.
  features = np.concatenate([canonical.flatten(), global_y, global_z])
  return features.astype(np.float32)  # shape (69,)


# ---------------------------------------------------------------------------
# Gesture classifier wrappers
# ---------------------------------------------------------------------------


class _GestureMLP(nn.Module):
  """
  MLP architecture — must match exactly what Josué used for training.
  Defined here so we can instantiate it before loading state_dict.
  """

  def __init__(self) -> None:
    super().__init__()
    self.fc1     = nn.Linear(69, 128)
    self.relu1   = nn.ReLU()
    self.dropout = nn.Dropout(0.2)
    self.fc2     = nn.Linear(128, 64)
    self.relu2   = nn.ReLU()
    self.output  = nn.Linear(64, 5)

  def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
    x = self.fc1(x)
    x = self.relu1(x)
    x = self.dropout(x)
    x = self.fc2(x)
    x = self.relu2(x)
    return self.output(x)


class _TorchModel:
  """Loads gesture_mlp.pth (state_dict) and wraps inference."""

  def __init__(self, model_path: Path) -> None:
    # Dynamic detection: Use GPU (Cuda) if available, otherwise fall back to CPU.
    self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    self._model = _GestureMLP()
    # map_location is key: load weights correctly regardless of where they were trained (GPU) 
    # and where they run (CPU/GPU).
    state_dict = torch.load(model_path, map_location=self._device, weights_only=True)
    self._model.load_state_dict(state_dict)
    
    self._model.to(self._device)
    self._model.eval()
    print(f'[GestureEngine] Running inference on: {self._device}')

  def predict(self, features: np.ndarray) -> str:
    tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self._device)
    with torch.no_grad():
      logits = self._model(tensor)
      idx = torch.argmax(logits, dim=1).item()
      return GESTURE_LABELS[idx]


class _StubModel:
  """
  Heuristic stub — used when gesture_model.pt is not yet available.
  Good enough to develop and test the full UI pipeline.
  """

  def predict(self, features: np.ndarray) -> str:
    # features[0:63] are canonical 21x3 landmarks
    landmarks = features[:63].reshape(21, 3)
    index_tip_y = landmarks[8, 1]
    index_pip_y = landmarks[6, 1]
    middle_tip_y = landmarks[12, 1]
    ring_tip_y = landmarks[16, 1]
    pinky_tip_y = landmarks[20, 1]
    thumb_tip_x = landmarks[4, 0]

    fingers_curled = (
      middle_tip_y > landmarks[10, 1] and ring_tip_y > landmarks[14, 1] and pinky_tip_y > landmarks[18, 1]
    )

    if index_tip_y < index_pip_y and fingers_curled:
      return 'Index_Finger'
    if index_tip_y > index_pip_y and fingers_curled:
      return 'Fist'
    if abs(thumb_tip_x) > 0.4:
      return 'Thumb_Up'
    return 'Ruler'


def _load_gesture_model() -> _TorchModel | _StubModel:
  if TORCH_AVAILABLE and GESTURE_MODEL_PATH.exists():
    try:
      return _TorchModel(GESTURE_MODEL_PATH)
    except Exception as e:  # noqa: BLE001
      print(f'[GestureEngine] Failed to load model: {e}. Using stub.')
  else:
    print('[GestureEngine] gesture_model.pt not found — running with heuristic stub.')
  return _StubModel()


# ---------------------------------------------------------------------------
# GestureEngine — the main class
# ---------------------------------------------------------------------------


class GestureResult:
  """
  Data object delivered to the callback on every processed frame.

  Attributes
  ----------
  drawing_gesture : str | None
      Gesture on the configured drawing hand, or None if not detected.
  modifier_gesture : str | None
      Gesture on the modifier hand, or None if not detected.
  drawing_tip : tuple[float, float] | None
      (x, y) pixel coords of index fingertip on the drawing hand.
      Coordinates are relative to the frame size — multiply by canvas size.
  modifier_tip : tuple[float, float] | None
      (x, y) pixel coords of index fingertip on the modifier hand.
  frame_bgr : np.ndarray | None
      The (possibly annotated) BGR frame for display.
  landmarks_2d : list
      Raw 2D landmark lists per detected hand — used for Dark Mode drawing.
  handedness_list : list[str]
      "Left" / "Right" (actual, not mirrored) per detected hand.
  """

  def __init__(self) -> None:
    self.drawing_gesture: str | None = None
    self.modifier_gesture: str | None = None
    self.drawing_tip: tuple[float, float] | None = None
    self.modifier_tip: tuple[float, float] | None = None
    self.frame_bgr: np.ndarray | None = None
    self.landmarks_2d: list = []
    self.handedness_list: list[str] = []


class GestureEngine(Thread):
  """
  Background thread that:
    1. Reads frames from the webcam
    2. Runs MediaPipe hand landmarker
    3. Canonicalizes world landmarks and runs gesture inference
    4. Calls `callback(GestureResult)` from the worker thread

  The caller (main.py) should schedule UI updates via Tkinter's `after()`.

  Parameters
  ----------
  callback : Callable[[GestureResult], None]
      Function that accepts a single GestureResult argument.
  drawing_hand : str
      "Right" or "Left" — which actual hand the user draws with.
  camera_index : int
      OpenCV camera index (default 0).
  """

  def __init__(
    self,
    callback: Callable[[GestureResult], None],
    drawing_hand: str = 'Right',
    camera_index: int = 0,
  ) -> None:
    super().__init__(daemon=True)
    self.callback = callback
    self.drawing_hand = drawing_hand
    self.modifier_hand = 'Left' if drawing_hand == 'Right' else 'Right'
    self.camera_index = camera_index
    self.running = False

    self._ensure_landmark_model()
    self._detector = self._build_detector()
    self._gesture_clf = _load_gesture_model()

  # ------------------------------------------------------------------
  # Setup helpers
  # ------------------------------------------------------------------

  def _ensure_landmark_model(self) -> None:
    if not LANDMARK_MODEL_PATH.exists():
      LANDMARK_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
      print('[GestureEngine] Downloading hand_landmarker.task …')
      urllib.request.urlretrieve(LANDMARK_MODEL_URL, LANDMARK_MODEL_PATH)  # noqa: S310
      print('[GestureEngine] Download complete.')

  def _build_detector(self) -> vision.HandLandmarker:
    base_options = python.BaseOptions(model_asset_path=str(LANDMARK_MODEL_PATH))
    options = vision.HandLandmarkerOptions(
      base_options=base_options,
      num_hands=2,
      min_hand_detection_confidence=0.7,
      min_hand_presence_confidence=0.7,
      min_tracking_confidence=0.7,
    )
    return vision.HandLandmarker.create_from_options(options)

  # ------------------------------------------------------------------
  # Thread entry point
  # ------------------------------------------------------------------

  def run(self) -> None:
    self.running = True
    cap = cv2.VideoCapture(self.camera_index)

    if not cap.isOpened():
      print('[GestureEngine] ERROR: Cannot open camera.')
      return

    print('[GestureEngine] Camera opened. Starting capture loop.')

    while self.running:
      ret, frame = cap.read()
      if not ret:
        time.sleep(0.01)
        continue

      # Flip for selfie/mirror mode
      frame = cv2.flip(frame, 1)
      rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
      result = self._detector.detect(mp_image)

      gesture_result = self._parse_result(result, frame)
      self.callback(gesture_result)

    cap.release()
    print('[GestureEngine] Camera released.')

  def stop(self) -> None:
    self.running = False

  def pause(self) -> None:
    self.running = False

  def resume(self) -> None:
    if not self.running:
      self.running = True

  # ------------------------------------------------------------------
  # Result parsing
  # ------------------------------------------------------------------

  def _parse_result(self, detection_result: vision.HandLandmarkerResult, frame: np.ndarray) -> GestureResult:
    gr = GestureResult()
    gr.frame_bgr = frame
    h, w, _ = frame.shape

    if not detection_result.hand_landmarks:
      return gr

    for i, handedness_list in enumerate(detection_result.handedness):
      raw_handedness = handedness_list[0].category_name  # "Left"/"Right" (mirrored)

      # Actual handedness after mirror correction
      actual_handedness = 'Left' if raw_handedness == 'Right' else 'Right'

      # 2D landmarks (screen coords) — for drawing and Dark Mode
      lms_2d = detection_result.hand_landmarks[i]
      gr.landmarks_2d.append(lms_2d)
      gr.handedness_list.append(actual_handedness)

      # 3D world landmarks — for gesture classification
      lms_world = detection_result.hand_world_landmarks[i]
      landmarks_3d = np.array([[lm.x, lm.y, lm.z] for lm in lms_world])

      features = canonicalize(landmarks_3d, raw_handedness)
      if features is None:
        continue

      gesture = self._gesture_clf.predict(features)

      # [Ergonomic KISS] Dynamic landmark selection for tracking
      if gesture == 'Fist':
        tracking_landmark = lms_2d[9]  # Middle Finger MCP - true "center" of fist
      else:
        tracking_landmark = lms_2d[8]  # Index Finger Tip - ideal for pointer/pencil

      tip_x = tracking_landmark.x * w
      tip_y = tracking_landmark.y * h

      if actual_handedness == self.drawing_hand:
        gr.drawing_gesture = gesture
        gr.drawing_tip = (tip_x, tip_y)
      elif actual_handedness == self.modifier_hand:
        gr.modifier_gesture = gesture
        gr.modifier_tip = (tip_x, tip_y)

    return gr