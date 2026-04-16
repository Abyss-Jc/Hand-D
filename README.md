# Hand-D

A gesture-controlled virtual whiteboard application using hand tracking.

[Placeholder: App screenshot - Camera mode]
[Placeholder: App screenshot - Dark mode]

## Description

Hand-D is a gesture-controlled virtual whiteboard that transforms hand gestures into drawing actions. Draw with your index finger, erase with a fist, adjust stroke thickness with a thumb gesture, and draw straight lines with a ruler gesture - all hands-free. 

We recently upgraded the core engine to use a custom PyTorch model that natively understands when your hand is resting, making the drawing experience much more precise and natural.

## Features

- Draw with your index finger.
- Erase with a fist (ergonomically tracked from your knuckles, not your fingertips).
- Draw straight lines with the ruler gesture.
- Adjust stroke thickness and eraser size dynamically with your thumb.
- **Smart 'Idle' State:** The app knows when you are just resting your hand to prevent accidental scribbles.
- **Dynamic Hardware Support:** Automatically runs on GPU (CUDA) if you have one, or smoothly falls back to CPU (perfect for laptops).
- Camera and Dark mode toggle.
- Left/right-handed support.

## Gestures

[Placeholder: Gesture reference images]

| Gesture | Hand | Action |
|---------|------|--------|
| Index_Finger | Drawing Hand | Draw |
| Fist | Drawing Hand | Erase |
| Ruler | Modifier Hand | Draw Straight Line |
| Thumb_Up | Modifier Hand | Adjust Thickness / Eraser Size |
| Idle | Drawing Hand | Rest / Pause actions |

## Installation

```bash
# Clone the repository
git clone [https://github.com/Abyss-Jc/Hand-D.git](https://github.com/Abyss-Jc/Hand-D.git)
cd Hand-D

# Create virtual environment (Python 3.8+)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install PyTorch based on your hardware (CPU or GPU)
# For CPU (Laptops/Standard PCs):
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)

# For GPU (NVIDIA):
# pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# Install the rest of the dependencies
pip install -r requirements.txt
```

## Usage

Run the app from the root directory:

```bash
python visualizer_app/main.py
```

### First Launch

On the first launch, a welcome modal will ask:
- Which one is your drawing hand?

Once you select it, a quick guide will show you how to use the gestures based on your handedness.

### Keyboard Shortcuts

- **M**: Toggle Camera / Dark mode
- **C**: Clear canvas
- **S**: Save current drawing as SVG
- **W**: Re-open the Preferences Modal (to change your drawing hand)
- **1-4**: Quick color selection (Red, Black, Blue, Green)

## Dataset

The gesture dataset was collected using our custom MediaPipe-based tools:

- **Samples**: ~4,336 total (~1,086 per gesture)
- **Classes**: 5 gestures (Index_Finger, Fist, Thumb_Up, Ruler, Idle)

See `datasets/gesture_dataset.csv`

## Inspiration

This project was inspired by:

- [BaranDev/virtual-whiteboard](https://github.com/BaranDev/virtual-whiteboard) - Computer vision virtual whiteboard
- [test28.html demo](https://smm.axtarget.xyz/test28.html) - Black background hand tracking demo

## Tech Stack

- Python 3.8+
- OpenCV - Computer vision
- MediaPipe (Google) - Hand tracking
- Tkinter - GUI
- PyTorch - Machine Learning model

## Authors

- [Josue Chavez](https://github.com/Abyss-Jc) - Machine Learning & Data Pipeline & Data Recolection 
- [Eduardo Quant](https://github.com/caml07) - UI/UX Architecture & Software Engineering & Data Recolection

## License

MIT

## Acknowledgments

- OpenCV community for computer vision tools.
- MediaPipe team for hand tracking solutions.
- BaranDev for the virtual whiteboard inspiration.
- smm.axtarget.xyz for the dark mode demo inspiration.