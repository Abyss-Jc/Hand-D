# Hand-D

A gesture-controlled virtual whiteboard application using hand tracking.

[Placeholder: App screenshot - Camera mode]
[Placeholder: App screenshot - Dark mode]

## Description

Hand-D is a gesture-controlled virtual whiteboard that transforms hand gestures into drawing actions. Draw with your index finger, erase with a fist, adjust stroke thickness with a thumb gesture, and draw straight lines with a ruler gesture - all hands-free.

## Features

- Draw with index finger gesture
- Erase with fist gesture  
- Draw straight lines with ruler gesture
- Adjust stroke thickness with thumb
- Camera and Dark mode toggle
- Left/right-handed support

## Gestures

[Placeholder: Gesture reference images]

| Gesture | Hand | Action |
|---------|------|--------|
| Index_Finger | Drawing Hand | Draw |
| Fist | Drawing Hand | Erase |
| Thumb_Up | Modifier Hand | Adjust Thickness |
| Ruler_Gesture | Modifier Hand | Draw Straight Line |

## Installation

```bash
# Clone the repository
git clone https://github.com/Abyss-Jc/Hand-D.git
cd Hand-D

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python visualizer-app/main.py
```

### First Launch

On first launch, the app will ask:
- Are you right-handed or left-handed?

Then the user guide modal explains controls.

### Controls

- **M**: Toggle Camera/Dark mode
- **C**: Clear canvas
- **S**: Save as SVG

## Inspiration

This project was inspired by:

- [BaranDev/virtual-whiteboard](https://github.com/BaranDev/virtual-whiteboard) - Computer vision virtual whiteboard
- [test28.html demo](https://smm.axtarget.xyz/test28.html) - Black background hand tracking demo

## Dataset

The gesture dataset was collected using custom MediaPipe-based tools:

- **Samples**: ~4,336 total (~1,086 per gesture)
- **Gestures**: Index_Finger, Fist, Thumb_Up, Ruler_Gesture

See `datasets/gesture_dataset.csv`

## Tech Stack

- Python 3.8+
- MediaPipe (Google) - Hand tracking
- Tkinter - GUI
- PyTorch - ML model (by Josue Chavez)

## Authors

- [Josue Chavez](https://github.com/Abyss-Jc)
- [Eduardo Quant](https://github.com/caml07)

## License

MIT

## Acknowledgments

- OpenCV community for computer vision tools
- MediaPipe team for hand tracking solutions
- BaranDev for the virtual whiteboard inspiration
- smm.axtarget.xyz for the dark mode demo inspiration
