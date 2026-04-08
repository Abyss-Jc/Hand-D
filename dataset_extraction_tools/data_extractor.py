import cv2
import mediapipe as mp
import numpy as np
import csv
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import urllib.request
import argparse

# --- PARSE ARGUMENTS ---
parser = argparse.ArgumentParser(description='Hand gesture data collection')
parser.add_argument('--label', type=str, required=True, help='Gesture name')
parser.add_argument('--handedness', type=str, default='Any', help='Expected handedness (Left/Right/Any)')
parser.add_argument('--samples', type=int, default=250, help='Number of samples to collect')
parser.add_argument('--output', type=str, default='datasets/gesture_dataset.csv', help='Output CSV file')
parser.add_argument('--stride', type=int, default=5, help='Frame stride (process every N frames)')
args = parser.parse_args()

# --- DOWNLOAD MODEL IF NOT PRESENT ---
MODEL_PATH = "models/hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Download complete.")

# --- HAND CONNECTIONS ---
HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),          # thumb
    (0,5), (5,6), (6,7), (7,8),          # index
    (0,9), (9,10), (10,11), (11,12),     # middle
    (0,13), (13,14), (14,15), (15,16),   # ring
    (0,17), (17,18), (18,19), (19,20),   # pinky
    (0,5), (5,9), (9,13), (13,17)        # palm
]

# --- FEATURE EXTRACTION ---
def extract_features_with_plot(landmarks_3d, handedness):
    wrist = landmarks_3d[0]
    translated_pts = landmarks_3d - wrist
    handedness_binary = 1.0 if handedness == "Left" else 0.0
    if handedness == "Left":
        translated_pts[:, 0] = -translated_pts[:, 0]

    # More robust scaling factor: average distance from wrist to MCPs
    # (Landmarks 5, 9, 13, 17 are MCPs for index, middle, ring, pinky fingers)
    mcps = translated_pts[[5, 9, 13, 17]]
    scale_factor = np.mean(np.linalg.norm(mcps - translated_pts[0], axis=1)) # Distance from wrist to MCP
    if scale_factor < 1e-3: # Increased threshold for very small hands or bad detection
        print("Debug: Skipping due to small scale_factor")
        return None, None

    scaled_pts = translated_pts / scale_factor

    # Define a more robust canonical coordinate system based on the palm
    # Using wrist (0), MCP_index (5), MCP_pinky (17) to define a plane
    p_wrist = scaled_pts[0] # Should be [0,0,0] after translation
    p_index_mcp = scaled_pts[5]
    p_pinky_mcp = scaled_pts[17]
    p_middle_mcp = scaled_pts[9]

    # Global Y-axis: vector from wrist to middle finger MCP
    global_y = p_middle_mcp - p_wrist
    norm_global_y = np.linalg.norm(global_y)
    if norm_global_y < 1e-3:
        print("Debug: Skipping due to small norm_global_y")
        return None, None
    global_y = global_y / norm_global_y

    # Global Z-axis: Normal of the plane formed by wrist, index MCP, and pinky MCP
    # Ensure points are not collinear by checking cross product norm
    vec1 = p_index_mcp - p_wrist
    vec2 = p_pinky_mcp - p_wrist
    
    cross_product_norm = np.linalg.norm(np.cross(vec1, vec2))
    if cross_product_norm < 1e-3: # Increased threshold for collinear points
        print("Debug: Skipping due to collinear palm points for global_z")
        return None, None

    global_z = np.cross(vec1, vec2)
    global_z = global_z / np.linalg.norm(global_z) # Normalize global_z

    # Global X-axis: cross product of global_y and global_z
    global_x = np.cross(global_y, global_z)
    norm_global_x = np.linalg.norm(global_x)
    if norm_global_x < 1e-3:
        print("Debug: Skipping due to small norm_global_x")
        return None, None
    global_x = global_x / norm_global_x

    # Re-orthogonalize global_y to ensure it's perfectly perpendicular to global_x and global_z
    global_y = np.cross(global_z, global_x)
    global_y = global_y / np.linalg.norm(global_y)


    M = np.stack([global_x, global_y, global_z], axis=1)
    canonical_pts = np.dot(scaled_pts, M)

    # Additional filter: Check if canonical points are within a reasonable range
    if np.any(np.abs(canonical_pts) > 2.0): # Points outside [-2.0, 2.0] range are suspicious
        print("Debug: Skipping due to canonical points out of range")
        return None, None
    
    # Existing part of the function continues from here
    base_features = np.concatenate([canonical_pts.flatten(), global_y, global_z])
    features = np.append(base_features, handedness_binary)
    return features, canonical_pts

    base_features = np.concatenate([canonical_pts.flatten(), global_y, global_z])
    features = np.append(base_features, handedness_binary)
    return features, canonical_pts

# # --- SETUP 3D PLOT (with persistent artists) ---

# # plt.ion()
# # fig = plt.figure("Canonical Hand View", figsize=(6, 6))
# # ax = fig.add_subplot(111, projection='3d')
# # ax.set_title("Normalized Hand (Canonical Frame)")
# # ax.set_xlabel("X")
# # ax.set_ylabel("Y")
# # ax.set_zlabel("Z")
# # ax.set_xlim(-1.5, 1.5)
# # ax.set_ylim(-1.5, 1.5)
# # ax.set_zlim(-1.5, 1.5)

# # Create empty artists that we will update later
# scatter = ax.scatter([], [], [], c='red', s=40)
# line, = ax.plot([], [], [], color='blue', linewidth=2)

def update_canonical_plot(canonical_pts):
    if canonical_pts is None:
        return
    xs, ys, zs = canonical_pts[:, 0], canonical_pts[:, 1], canonical_pts[:, 2]
    # Update scatter points
    scatter._offsets3d = (xs, ys, zs)
    # Update lines: create a list of segments separated by None
    lines_x, lines_y, lines_z = [], [], []
    for (i, j) in HAND_CONNECTIONS:
        lines_x.extend([canonical_pts[i, 0], canonical_pts[j, 0], None])
        lines_y.extend([canonical_pts[i, 1], canonical_pts[j, 1], None])
        lines_z.extend([canonical_pts[i, 2], canonical_pts[j, 2], None])
    line.set_data_3d(lines_x, lines_y, lines_z)
    fig.canvas.draw_idle()
    plt.pause(0.001)

# --- INITIALIZE MEDIAPIPE TASKS HAND LANDMARKER ---
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

# --- COLLECTION CONFIG ---
LABEL = args.label
EXPECTED_HANDEDNESS = args.handedness.capitalize()
TARGET_SAMPLES = args.samples
FILE_NAME = args.output
STRIDE = args.stride

print(f"\n=== Configuration ===")
print(f"Gesture: {LABEL}")
print(f"Handedness: {EXPECTED_HANDEDNESS}")
print(f"Target samples: {TARGET_SAMPLES}")
print(f"Output file: {FILE_NAME}")
print(f"Frame stride: {STRIDE}")
print("========================\n")

cap = cv2.VideoCapture(0)
samples_collected = 0
frame_count = 0
collected_rows = []

# CSV header
try:
    with open(FILE_NAME, 'r') as f:
        file_empty = (f.read(1) == '')
except FileNotFoundError:
    file_empty = True

if file_empty:
    header = [f"feat_{i}" for i in range(69)] + ["handedness", "label"]
    with open(FILE_NAME, mode='w', newline='') as f:
        csv.writer(f).writerow(header)

print(f"Collecting {TARGET_SAMPLES} samples. Press 'r' to reject last, 'q' to quit.")

while samples_collected < TARGET_SAMPLES:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cv2.putText(frame, f"Samples: {samples_collected}/{TARGET_SAMPLES}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Press 'r' to reject last, 'q' to quit", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if frame_count % STRIDE == 0:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)

        if detection_result.hand_landmarks and detection_result.hand_world_landmarks:
            hand_world_landmarks = detection_result.hand_world_landmarks[0]
            handedness = detection_result.handedness[0][0].category_name
            landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_world_landmarks])

            # Handedness filtering
            if EXPECTED_HANDEDNESS != "Any" and handedness != EXPECTED_HANDEDNESS:
                cv2.putText(frame, f"Skipped: Detected {handedness}, Expected {EXPECTED_HANDEDNESS}", (10, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(f"Skipped: Detected {handedness}, Expected {EXPECTED_HANDEDNESS}")
                continue

            features, canonical_pts = extract_features_with_plot(landmarks_array, handedness)
            #update_canonical_plot(canonical_pts)      <------------=======[UNSTABLE]=======\

            if features is not None:
                row = list(features) + [LABEL]
                collected_rows.append(row)
                samples_collected += 1
                print(f"Collected sample {samples_collected}/{TARGET_SAMPLES}")
            else:
                cv2.putText(frame, "Skipped: Unstable hand orientation", (10, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print("Skipped: unstable hand orientation")

            # Draw 2D landmarks on camera feed
            hand_landmarks_2d = detection_result.hand_landmarks[0]
            h, w, _ = frame.shape
            for lm in hand_landmarks_2d:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
            for (i, j) in HAND_CONNECTIONS:
                if i < len(hand_landmarks_2d) and j < len(hand_landmarks_2d):
                    pt1 = (int(hand_landmarks_2d[i].x * w), int(hand_landmarks_2d[i].y * h))
                    pt2 = (int(hand_landmarks_2d[j].x * w), int(hand_landmarks_2d[j].y * h))
                    cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
            cv2.putText(frame, f"{handedness}", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Data Collection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r') and samples_collected > 0:
        collected_rows.pop()
        samples_collected -= 1
        print(f"Rejected last sample. Now at {samples_collected}/{TARGET_SAMPLES}")

if collected_rows:
    with open(FILE_NAME, mode='a', newline='') as f:
        csv.writer(f).writerows(collected_rows)
    print(f"Saved {len(collected_rows)} samples to {FILE_NAME}")
else:
    print("No samples collected.")

cap.release()
cv2.destroyAllWindows()

# plt.ioff()
# plt.close(fig)

detector.close()