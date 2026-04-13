# Visualizer-App Specification

## Project Context

This application is part of a hand gesture recognition project for a virtual whiteboard. The dataset was collected using custom MediaPipe-based data collection tools (see `dataset_extraction_tools/`).

### References & Inspiration

- **BaranDev/virtual-whiteboard**: https://github.com/BaranDev/virtual-whiteboard
  - Computer vision-powered virtual whiteboard using hand gestures
  - Dual-hand operation (drawing + control hands)
  - Object recognition for drawing tool
  - Touch detection algorithm

- **test28.html** (smm.axtarget.xyz/demo)
  - Black background with green hand landmarks
  - Drawing with index finger gesture
  - Pinch gesture to grab and move drawings

### Dataset

- **Location**: `datasets/gesture_dataset.csv`
- **Gestures**: 4 (~1086 samples each)
  1. Index_Finger - Drawing
  2. Fist - Erase
  3. Thumb_Up - Thickness control
  4. Ruler_Gesture - Straight line drawing

---

## 1. Project Overview

A gesture-controlled virtual whiteboard application that allows users to draw, erase, and adjust drawing settings using hand gestures captured via webcam and MediaPipe hand tracking.

### Purpose
Transform hand gestures into drawing actions on a digital canvas, enabling a touch-free drawing experience.

### Target Users
- Presenters and educators who need hands-free presentation tools
- Users seeking alternative input methods for digital art/diagramming

---

## 2. Technical Architecture

### Technology Stack
- **Language**: Python 3.8+
- **UI Framework**: Tkinter (for cross-platform GUI)
- **Computer Vision**: MediaPipe (Google)
- **ML Framework**: PyTorch (model by Josue)
- **Model Format**: To be determined by model creator (.pt, .h5, .onnx placeholder)
- **Model Path**: `models/hand_landmarker.task` or `models/` directory (placeholder - to be provided by model creator)
- **Model Loading**: MediaPipe Tasks Vision API or custom PyTorch loader

### MediaPipe Integration
```python
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,  # Support both left and right hands simultaneously
    running_mode=vision.RunningMode.LIVE_STREAM
)
detector = vision.HandLandmarker.create_from_options(options)
```

Or custom PyTorch model loading (placeholder):
```python
import torch

model = HandGestureModel()  # Custom model class
model.load_state_dict(torch.load('models/gesture_model.pt'))
model.eval()
```

**Note on Model File**: The application expects either a MediaPipe-compatible hand landmarker model file (`.task` format) or a custom PyTorch model. This is a placeholder - the model creator (Josue) must provide the actual model file for the application to function.

### Dataset Reference
- **Source**: `datasets/gesture_dataset.csv`
- **Total Samples**: ~4336 (balanced across 4 gestures)
- **Samples per Gesture**: ~1086 each
- **Features**: 69 (63 3D landmarks + 6 global features)

---

## 3. Hand Gesture Definitions

### 3.1 Right Hand Gestures (Drawing/Erasing)

| Gesture Name | MediaPipe Equivalent | Detection Criteria |
|--------------|---------------------|---------------------|
| **Index_Finger** (Draw) | `Pointing_Up` | Index finger extended, other fingers curled |
| **Fist** (Erase) | `Closed_Fist` | All fingers curled into palm |

#### Index Finger Detection Logic
- Index finger tip landmark (ID 8) is above index finger PIP landmark (ID 6)
- Middle, Ring, and Pinky finger tips (IDs 12, 16, 20) are below their respective MCP landmarks
- Confidence threshold: 0.7

#### Fist Detection Logic
- All finger tips (IDs 4, 8, 12, 16, 20) are below their PIP landmarks
- OR use MediaPipe's built-in `Closed_Fist` gesture recognition

---

### 3.2 Left Hand Gestures (Modifier Actions)

| Gesture Name | MediaPipe Equivalent | Function |
|--------------|---------------------|----------|
| **Thumb_Up** | `Thumb_Up` | Thickness slider control |
| **Ruler_Gesture** | Custom detection | Draw straight lines |

#### Thickness Slider (Thumb_Up)
- **Activation**: Left hand shows Thumb_Up gesture
- **Control Method**: Horizontal thumb position maps to thickness value
- **Range**: 1px to 20px (configurable)
- **Visual Feedback**: On-screen slider display showing current thickness value
- **Position Mapping**:
  - Thumb extended far left = minimum thickness (1px)
  - Thumb extended far right = maximum thickness (20px)
  - Use normalized x-coordinate of THUMB_TIP (landmark 4) relative to WRIST (landmark 0)

#### Ruler Gesture (Straight Line Mode)
- **Activation**: Left hand shows Ruler_Gesture while right hand draws
- **Detection Criteria**: 
  - Index finger extended horizontally (pointing sideways)
  - Thumb extended at 90° to palm
  - Other three fingers curled
- **Behavior**: 
  - First point: Record start position on Index_Finger tip
  - While gesture held: Show preview line from start to current tip position
  - On gesture release: Commit final straight line to canvas

---

### 3.3 Hand Distinction (Left vs Right)

MediaPipe provides `handedness` classification in detection results:
```python
for i, handedness in enumerate(detection_result.handedness):
    hand_type = handedness[0].category_name  # "Left" or "Right"
    hand_landmarks = detection_result.hand_landmarks[i]
```

**Important**: MediaPipe assumes mirrored (selfie-style) camera input. If using non-mirrored camera, apply `cv2.flip(frame, 1)` before processing.

---

## 4. Two-Hand Operation

### Concept
This application uses **two-hand operation**:
- **Right Hand**: Drawing and erasing actions
- **Left Hand**: Modifier controls (thickness, ruler mode)

### Concurrent Gesture Processing
The application processes both hands simultaneously:
1. Detect all hands in frame (max 2)
2. Classify each hand's handedness (Left/Right)
3. Determine gesture for each hand independently
4. Apply modifier logic if both hands are active

### Modifier Logic
```
IF Right_Hand == Index_Finger AND Left_Hand == Thumb_Up:
    → Draw with variable thickness based on thumb position

IF Right_Hand == Index_Finger AND Left_Hand == Ruler_Gesture:
    → Draw straight line from start to current position

IF Right_Hand == Index_Finger AND Left_Hand == (none):
    → Draw with current thickness (default 3px)

IF Right_Hand == Fist AND Left_Hand == (any):
    → Erase (regardless of left hand)
```

---

## 5. Feature Specifications

### 5.1 Mode Toggle (Camera View / Dark Mode)

| Mode | Description | Background | Landmarks Display |
|------|-------------|------------|-------------------|
| **Camera View** | Shows live webcam feed | Camera feed | Optional overlay |
| **Dark Mode** | Black background + landmark visualization | Pure black (#000000) | Green landmarks drawn on canvas |

- **Toggle**: Button in toolbar or keyboard shortcut (e.g., `M` key)
- **Default**: Camera View on first launch

### 5.2 Toolbar

**Layout**: Horizontal toolbar at top of window

| Button | Function | Keyboard Shortcut |
|--------|----------|-------------------|
| Color: Red | Set stroke color to #FF0000 | `1` |
| Color: Black | Set stroke color to #000000 | `2` |
| Color: Blue | Set stroke color to #0000FF | `3` |
| Color: Green | Set stroke color to #00FF00 | `4` |
| Clear | Clear entire canvas | `C` |
| Save | Save canvas as SVG file | `S` |

**Visual**: Icon-based buttons with tooltip labels

### 5.3 Canvas & Drawing

- **Resolution**: Full screen (entire window)
- **Drawing Method**: Smooth freehand polyline using Bezier interpolation
- **Stroke Smoothing**: Apply simple moving average to reduce jitter
- **Coordinate Mapping**: Normalize MediaPipe landmarks (0-1) to canvas pixel coordinates

### 5.4 Save Format

- **Format**: SVG (Scalable Vector Graphics)
- **Filename**: `drawing_YYYYMMDD_HHMMSS.svg`
- **Default Location**: User's Documents folder or current working directory
- **File Dialog**: Option to choose save location via native file dialog

---

## 6. User Interface

### 6.1 First Launch - User Guide Modal

**Trigger**: Display on first application open only

**Content**:
```
Welcome to Visualizer-App!

HOW TO USE:

RIGHT HAND (Drawing):
• Index Finger extended → Draw
• Fist (all fingers curled) → Erase

LEFT HAND (Modifiers):
• Thumb Up → Adjust thickness
  (move thumb left/right to change stroke width)
• Ruler Gesture → Draw straight line
  (hold ruler gesture while drawing)

TWO-HAND OPERATION:
• Draw with right hand + adjust thickness with left thumb
• Draw with right hand + use ruler with left hand

TOGGLE MODES:
• Press 'M' to switch between Camera View and Dark Mode
• Dark Mode: Black background with colored landmarks

BASIC COLORS (select from toolbar):
• Red • Black • Blue • Green

Enjoy drawing!

[ ] Don't show this again
```

**Features**:
- Modal dialog with "Don't show again" checkbox
- If checkbox selected, save preference to local config file
- "OK" button closes modal and starts application

**Config Storage**: JSON file in user's app data directory (`~/.visualizer-app/config.json`)

### 6.2 Layout

```
+------------------------------------------------------------------+
| [Red] [Black] [Blue] [Green] | [Clear] [Save] | [Mode Toggle]  |
+------------------------------------------------------------------+
|                                                                  |
|                     (Canvas / Camera View)                       |
|                                                                  |
|                                                                  |
+------------------------------------------------------------------+
| Status: Ready | Thickness: 3px | Color: Black                   |
+------------------------------------------------------------------+
```

---

## 7. Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| Windows 10/11 | Supported | Primary target |
| Linux (Ubuntu 20.04+) | Supported | Requires OpenCV compatible camera |

### Requirements
- Python 3.8+
- OpenCV (`opencv-python`)
- MediaPipe (`mediapipe`)
- Tkinter (included with Python)

---

## 8. Implementation Notes

### Performance Considerations
- Process frames at 30 FPS target
- Use MediaPipe's LIVE_STREAM mode for optimal performance
- Implement frame skipping if processing cannot keep up

### Error Handling
- Camera not available: Display error message and exit gracefully
- Model file missing: Display clear error with expected file path
- Hand not detected: Continue without action (no drawing)

### Gesture State Machine
```
States: IDLE, DRAWING, ERASING, THICKNESS_ADJUST, RULER_MODE

Transitions:
IDLE + Right_Index_Finger → DRAWING
IDLE + Right_Fist → ERASING
DRAWING + Left_Thumb_Up → THICKNESS_ADJUST
DRAWING + Left_Ruler_Gesture → RULER_MODE
Any State + No_Gesture → IDLE
```

---

## 9. File Structure (Expected)

```
visualizer-app/
├── main.py                 # Application entry point
├── config.json            # User preferences (auto-generated)
├── models/
│   └── hand_landmarker.task  # MediaPipe model (user-provided)
├── requirements.txt
└── SPEC.md
```

---

## 10. Future Considerations (Out of Scope)

- Custom gesture training/recognition
- Multiple brush types
- Undo/redo functionality
- Tablet stylus support
- Cloud save integration