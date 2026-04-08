import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D

# --- CONFIG ---
FILE_NAME = "datasets/gesture_dataset.csv"

# Connections for the hand skeleton
HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),          # thumb
    (0,5), (5,6), (6,7), (7,8),          # index
    (0,9), (9,10), (10,11), (11,12),     # middle
    (0,13), (13,14), (14,15), (15,16),   # ring
    (0,17), (17,18), (18,19), (19,20),   # pinky
    (0,5), (5,9), (9,13), (13,17)        # palm
]

# 1. Load Data
try:
    df = pd.read_csv(FILE_NAME)
    print(f"Loaded {len(df)} samples from {FILE_NAME}")
except FileNotFoundError:
    print(f"Error: {FILE_NAME} not found. Run your collection script first!")
    exit()

# 2. Setup Figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# Make room at the bottom for all the new UI controls
plt.subplots_adjust(bottom=0.25) 

# 3. Visualization Function
def draw_hand(index):
    ax.clear()
    
    if len(df) == 0:
        ax.set_title("Dataset is Empty!")
        ax.set_axis_off()
        return

    # Ensure index is within current bounds (important after deletions)
    index = min(max(int(index), 0), len(df) - 1)
    
    # Extract the first 63 features
    row = df.iloc[index]
    landmarks = row.iloc[:63].values.reshape(21, 3)
    label = row['label']
    handedness = row['handedness']
    
    # Apply your spatial hack to center the hand
    xs, ys, zs = landmarks[:, 0], landmarks[:, 1]-1, landmarks[:, 2]
    
    # Plot points and skeleton
    ax.scatter(xs, ys, zs, c='red', s=50)
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        ax.plot(
            [landmarks[start_idx, 0], landmarks[end_idx, 0]],
            [landmarks[start_idx, 1]-1, landmarks[end_idx, 1]-1],
            [landmarks[start_idx, 2], landmarks[end_idx, 2]],
            color='blue', linewidth=2
        )
    
    # Formatting
    ax.set_title(f"Sample: {index} | Total: {len(df)} | Label: {label} | Hand: {'Left' if handedness == 1 else 'Right'}")
    ax.set_xlim(1, -1); ax.set_ylim(1, -1); ax.set_zlim(-1, 1)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.view_init(elev=-90, azim=-90)

# --- UI CONTROLS SETUP ---

# Slider 
ax_slider = plt.axes([0.25, 0.15, 0.5, 0.03])
slider = Slider(ax_slider, 'Frame', 0, len(df)-1, valinit=0, valfmt='%0.0f')

# Next / Prev Buttons
ax_prev = plt.axes([0.15, 0.15, 0.08, 0.03])
btn_prev = Button(ax_prev, '<< Prev')

ax_next = plt.axes([0.77, 0.15, 0.08, 0.03])
btn_next = Button(ax_next, 'Next >>')

# Delete / Save Buttons
ax_delete = plt.axes([0.3, 0.05, 0.15, 0.05])
btn_delete = Button(ax_delete, 'Trash Sample', color='salmon', hovercolor='red')

ax_save = plt.axes([0.55, 0.05, 0.15, 0.05])
btn_save = Button(ax_save, 'Save CSV', color='lightgreen', hovercolor='lime')

# --- EVENT HANDLERS ---

def update(val):
    draw_hand(int(slider.val))
    fig.canvas.draw_idle()

def go_prev(event):
    slider.set_val(max(slider.val - 1, 0))

def go_next(event):
    slider.set_val(min(slider.val + 1, len(df) - 1))

def delete_current(event):
    global df
    if len(df) == 0: return
    
    idx = int(slider.val)
    idx = min(max(idx, 0), len(df) - 1) 
    
    # Drop the row and reset pandas indices
    df = df.drop(idx).reset_index(drop=True)
    print(f"Deleted sample {idx}. Remaining: {len(df)}")
    
    if len(df) > 0:
        # Update slider visual bounds to match new dataframe length
        new_idx = min(idx, len(df) - 1)
        slider.valmax = len(df) - 1
        slider.ax.set_xlim(0, len(df) - 1)
        slider.set_val(new_idx)
    else:
        draw_hand(0)
        fig.canvas.draw_idle()

def save_data(event):
    if len(df) > 0:
        df.to_csv(FILE_NAME, index=False)
        print(f"Saved {len(df)} cleaned samples back to '{FILE_NAME}'")
    else:
        print("Dataset is empty. Nothing to save.")

# Attach Handlers
slider.on_changed(update)
btn_prev.on_clicked(go_prev)
btn_next.on_clicked(go_next)
btn_delete.on_clicked(delete_current)
btn_save.on_clicked(save_data)

# Initial draw
draw_hand(0)
plt.show()