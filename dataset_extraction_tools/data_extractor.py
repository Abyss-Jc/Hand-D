import cv2
import mediapipe as mp
import numpy as np
import csv
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import urllib.request

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
    v_y_raw = translated_pts[9]
    scale_factor = np.linalg.norm(v_y_raw)
    if scale_factor < 1e-6:
        return None, None
    scaled_pts = translated_pts / scale_factor
    global_y = scaled_pts[9]
    v_idx = scaled_pts[5]
    v_z = np.cross(global_y, v_idx)
    norm_z = np.linalg.norm(v_z)
    if norm_z < 1e-6:
        return None, None
    global_z = v_z / norm_z
    global_x = np.cross(global_y, global_z)
    M = np.stack([global_x, global_y, global_z], axis=1)
    canonical_pts = np.dot(scaled_pts, M)
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
LABEL = input("Enter gesture name (e.g., Thumbs_Up): ")
TARGET_SAMPLES = int(input(f"How many samples to collect for '{LABEL}': "))
FILE_NAME = "datasets/gesture_dataset.csv"
STRIDE = 5

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

            features, canonical_pts = extract_features_with_plot(landmarks_array, handedness)
            #update_canonical_plot(canonical_pts)      <------------=======[UNSTABLE]=======

            if features is not None:
                row = list(features) + [LABEL]
                collected_rows.append(row)
                samples_collected += 1
                print(f"Collected sample {samples_collected}/{TARGET_SAMPLES}")
            else:
                print("Skipped: unstable hand orientation (cross product too small)")

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