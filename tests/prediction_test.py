import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#Pytorch model definition
class GestureMLP(nn.Module):
    def __init__(self):
        super(GestureMLP, self).__init__()
        self.fc1 = nn.Linear(69, 128)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(64, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.output(x)
        return x

# Labels mapping (Reversed from training)
LABELS = {0: 'Fist', 1: 'Index_Finger', 2: 'Ruler_Gesture', 3: 'Thumb_Up', 4: 'Idle'}

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GestureMLP()
model.load_state_dict(torch.load("models/gesture_mlp.pth", map_location=device))
model.to(device)
model.eval() # CRITICAL: Turns off Dropout for live inference

print("✅ Model loaded successfully!")


# 2. Feature Extraction (Same as collection)

def extract_features(landmarks_3d, handedness):
    wrist = landmarks_3d[0]
    translated_pts = landmarks_3d - wrist
    
    # 1. Save what MediaPipe actually saw (to fix the math)
    raw_mp_detection = handedness 
    
    # 2. Correct the label
    actual_handedness = "Right" if raw_mp_detection == "Left" else "Left"
    handedness_binary = 1.0 if actual_handedness == "Left" else 0.0
    
    # 3. Spatial Normalization: Flip X-axis if MediaPipe saw a "Left" hand
    if raw_mp_detection == "Left":
        translated_pts[:, 0] = -translated_pts[:, 0]

    # Robust scaling factor
    mcps = translated_pts[[5, 9, 13, 17]]
    scale_factor = np.mean(np.linalg.norm(mcps - translated_pts[0], axis=1)) 
    if scale_factor < 1e-3: return None

    scaled_pts = translated_pts / scale_factor

    # Canonical Coordinate System based on palm
    p_wrist = scaled_pts[0]
    p_index_mcp = scaled_pts[5]
    p_pinky_mcp = scaled_pts[17]
    p_middle_mcp = scaled_pts[9]

    # Global Y
    global_y = p_middle_mcp - p_wrist
    norm_global_y = np.linalg.norm(global_y)
    if norm_global_y < 1e-3: return None
    global_y = global_y / norm_global_y

    # Global Z
    vec1 = p_index_mcp - p_wrist
    vec2 = p_pinky_mcp - p_wrist
    cross_product_norm = np.linalg.norm(np.cross(vec1, vec2))
    if cross_product_norm < 1e-3: return None
    global_z = np.cross(vec1, vec2)
    global_z = global_z / np.linalg.norm(global_z)

    # Global X
    global_x = np.cross(global_y, global_z)
    norm_global_x = np.linalg.norm(global_x)
    if norm_global_x < 1e-3: return None
    global_x = global_x / norm_global_x

    # Re-orthogonalize global_y
    global_y = np.cross(global_z, global_x)
    global_y = global_y / np.linalg.norm(global_y)

    M = np.stack([global_x, global_y, global_z], axis=1)
    canonical_pts = np.dot(scaled_pts, M)

    # Sanity Check (Increased threshold to 4.0 for long fingers)
    if np.any(np.abs(canonical_pts) > 4.0): return None
    
    features = np.concatenate([canonical_pts.flatten(), global_y, global_z])
    return features

# ==========================================
# 3. MEDIAPIPE & OPENCV SETUP
# ==========================================
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path="models/hand_landmarker.task") # Update path if needed
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

# ==========================================
# 4. INFERENCE LOOP
# ==========================================
cap = cv2.VideoCapture(0)
print("🎥 Starting webcam... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Flip frame like a mirror
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    # If a hand is detected
    if detection_result.hand_landmarks and detection_result.hand_world_landmarks:
        hand_world_landmarks = detection_result.hand_world_landmarks[0]
        handedness = detection_result.handedness[0][0].category_name
        landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_world_landmarks])

        # 1. Extract the 69 features
        features = extract_features(landmarks_array, handedness)

        if features is not None:
            # 2. Convert to PyTorch Tensor
            x_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)

            # 3. Feed forward through the MLP
            with torch.no_grad():
                outputs = model(x_tensor)
                
                # 4. Apply Softmax to get probabilities (0.0 to 1.0)
                probabilities = F.softmax(outputs, dim=1)
                
                # 5. Get the highest probability and its label index
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                prediction = LABELS[predicted_idx.item()]
                conf_score = confidence.item() * 100

            # 6. Display Result on Screen
            text_color = (0, 255, 0) if conf_score > 80 else (0, 165, 255)
            cv2.putText(frame, f"{prediction} ({conf_score:.1f}%)", (20, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 3)

        # Draw 2D Landmarks (Visual aid)
        h, w, _ = frame.shape
        for lm in detection_result.hand_landmarks[0]:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

    cv2.imshow("Hand-D Live Inference", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()