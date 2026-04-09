import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os

# --- PARSE ARGUMENTS ---
parser = argparse.ArgumentParser(description='3D hand gesture visualizer')
parser.add_argument('--label', type=str, default=None, help='Filter by gesture label')
parser.add_argument('--file', type=str, default='gesture_dataset.csv', help='CSV file')
args = parser.parse_args()

FILE_NAME = args.file

# Connections for the hand skeleton
HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),          # thumb
    (0,5), (5,6), (6,7), (7,8),          # index
    (0,9), (9,10), (10,11), (11,12),     # middle
    (0,13), (13,14), (14,15), (15,16),   # ring
    (0,17), (17,18), (18,19), (19,20),   # pinky
    (0,5), (5,9), (9,13), (13,17)        # palm
]

scatter_artist = None
line_artists = []
quiver_y_artist = None
quiver_z_artist = None
is_initialized = False

# 1. Load Data
try:
    df = pd.read_csv(FILE_NAME)
    print(f"Loaded {len(df)} samples from {FILE_NAME}")
    
    if args.label:
        df = df[df['label'] == args.label].reset_index(drop=True)
        print(f"Filtered to {len(df)} samples with label '{args.label}'")
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
    global scatter_artist, line_artists, quiver_y_artist, quiver_z_artist, is_initialized
    
    if len(df) == 0:
        ax.clear()
        ax.set_title("Dataset is Empty!")
        ax.set_axis_off()
        is_initialized = False # Reset if empty
        return

    # Ensure index is within current bounds
    index = min(max(int(index), 0), len(df) - 1)
    
    # Extract features
    row = df.iloc[index]
    landmarks = row.iloc[:63].values.reshape(21, 3)
    label = row['label']
    handedness = row['handedness']

    global_y = row.iloc[63:66].values.astype(float)
    global_z = row.iloc[66:69].values.astype(float)
    
    # Apply spatial hack
    xs, ys, zs = landmarks[:, 0], landmarks[:, 1]-1, landmarks[:, 2]

    # ==========================================
    # PHASE 1: INITIAL SETUP (Runs only once)
    # ==========================================
    if not is_initialized:
        ax.clear()
        
        # Draw and save the scatter artist
        scatter_artist = ax.scatter(xs, ys, zs, c='red', s=50)
        
        # Draw and save the line artists
        line_artists = []
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            line, = ax.plot(
                [landmarks[start_idx, 0], landmarks[end_idx, 0]],
                [landmarks[start_idx, 1]-1, landmarks[end_idx, 1]-1],
                [landmarks[start_idx, 2], landmarks[end_idx, 2]],
                color='blue', linewidth=2
            )
            line_artists.append(line)

        # Draw and save quivers
        quiver_y_artist = ax.quiver(0, -1, 0, -global_y[0], -global_y[1], -global_y[2], 
                  color='green', linewidth=3, length=0.6, arrow_length_ratio=0.2, label='Global Y (Hand Up)')
        
        quiver_z_artist = ax.quiver(0, -1, 0, -global_z[0], -global_z[1], -global_z[2], 
                  color='purple', linewidth=3, length=0.6, arrow_length_ratio=0.2, label='Global Z (Hand Forward)')
        
        # Formatting (Only needs to be set once now!)
        ax.set_xlim(-1, 1); ax.set_ylim(1, -1); ax.set_zlim(1, -1)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.view_init(elev=-90, azim=-90)
        ax.legend(loc='upper left', bbox_to_anchor=(0.8, 1.05))
        
        is_initialized = True

    # ==========================================
    # PHASE 2: UPDATE DATA (Runs on every click)
    # ==========================================
    else:
        # Update Scatter Points
        scatter_artist._offsets3d = (xs, ys, zs)
        
        # Update Line Positions
        for i, connection in enumerate(HAND_CONNECTIONS):
            start_idx, end_idx = connection
            line_artists[i].set_data_3d(
                [landmarks[start_idx, 0], landmarks[end_idx, 0]],
                [landmarks[start_idx, 1]-1, landmarks[end_idx, 1]-1],
                [landmarks[start_idx, 2], landmarks[end_idx, 2]]
            )
            
        # Update Quivers (Remove the old ones, draw new ones)
        quiver_y_artist.remove()
        quiver_z_artist.remove()
        
        quiver_y_artist = ax.quiver(0, -1, 0, -global_y[0], -global_y[1], -global_y[2], 
                  color='green', linewidth=3, length=0.6, arrow_length_ratio=0.2)
        
        quiver_z_artist = ax.quiver(0, -1, 0, -global_z[0], -global_z[1], -global_z[2], 
                  color='purple', linewidth=3, length=0.6, arrow_length_ratio=0.2)

    # Dynamic Title (Updates every time)
    ax.set_title(f"Sample: {index} | Total: {len(df)} | Label: {label} | Hand: {'Left' if handedness == 1 else 'Right'}")

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
        existing_df = pd.DataFrame()
        if os.path.exists(FILE_NAME):
            existing_df = pd.read_csv(FILE_NAME)
        
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df = combined_df.drop_duplicates()
        combined_df.to_csv(FILE_NAME, index=False)
        print(f"Saved {len(combined_df)} unique samples")
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