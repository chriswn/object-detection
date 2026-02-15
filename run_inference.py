import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
import math
from networktables import NetworkTables

# --- Configuration ---
MODEL_PATH = 'models/best.pt'
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.45

# --- NetworkTables Setup (Set to False if testing on laptop without robot) ---
ENABLE_NETWORKTABLES = False 
ROBOT_IP = "10.0.0.2" # CHANGE THIS to your RoboRIO IP (e.g., 10.TE.AM.2)

# --- Robot/Camera Constants ---
CAMERA_HFOV_DEG = 60.0
KNOWN_FUEL_DIAMETER_IN = 5.91
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# --- Logic Settings ---
CLUSTER_PROXIMITY_INCHES = 12.0
MIN_BALLS_FOR_CLUSTER = 1

# Priority Weights: How much we care about Distance vs. Angle
# A higher angle weight means the robot prefers balls straight ahead over closer balls to the side.
PRIORITY_ANGLE_WEIGHT = 0.1 

# --- Initialization ---
print("=" * 60)
print("FRC FUEL OBJECT DETECTION - STARTING")
print("=" * 60)
print(f"Loading model from {MODEL_PATH}...")
try:
    model = YOLO(MODEL_PATH)
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Initialize NetworkTables
if ENABLE_NETWORKTABLES:
    print(f"Connecting to NetworkTables at {ROBOT_IP}...")
    NetworkTables.initialize(server=ROBOT_IP)
    sd = NetworkTables.getTable("SmartDashboard")
    print("✓ NetworkTables Initialized")
else:
    print("ℹ NetworkTables disabled (testing mode)")

print(f"Opening camera {CAMERA_INDEX}...")
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    print(f"❌ Could not open camera at index {CAMERA_INDEX}")
    print("Try running test_camera.py first to verify your camera works")
    exit(1)

print("✓ Camera opened successfully!")

# Focal Length Calculation
focal_length_px = FRAME_WIDTH / (2 * math.tan(math.radians(CAMERA_HFOV_DEG / 2)))
print(f"✓ Focal length calculated: {focal_length_px:.2f} pixels")
print("=" * 60)

def draw_top_down_map(detections, best_target_index):
    """
    Creates a black window (radar view) showing the robot and detected balls.
    """
    # Create a blank black image (500x500 pixels)
    map_img = np.zeros((500, 500, 3), dtype=np.uint8)
    
    # Draw Robot (Green Triangle at bottom center)
    center_x, center_y = 250, 480
    robot_pts = np.array([[center_x, center_y], [center_x-20, center_y+40], [center_x+20, center_y+40]])
    cv2.fillPoly(map_img, [robot_pts], (0, 255, 0))
    
    # Scale: 1 pixel = 0.5 inches
    scale = 2.0 

    for i, det in enumerate(detections):
        # det = [x_dist, y_dist, pixel_x, pixel_y]
        # X is Left/Right, Y is Forward
        
        # Map X/Y to pixel coordinates
        # Map X: Robot Center + (Real X * Scale)
        map_x = int(center_x + (det[0] * scale))
        # Map Y: Robot Center - (Real Y * Scale) (Minus because Y goes down in images)
        map_y = int(center_y - (det[1] * scale))

        # Determine color (Red for Priority, Cyan for others)
        color = (0, 0, 255) if i == best_target_index else (255, 255, 0)
        
        # Draw Ball
        cv2.circle(map_img, (map_x, map_y), 5, color, -1)
        
        # Draw line to priority target
        if i == best_target_index:
            cv2.line(map_img, (center_x, center_y), (map_x, map_y), (0, 0, 255), 1)

    # Add text
    cv2.putText(map_img, "2D RADAR VIEW", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(map_img, "Grid: 50 inches", (10, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    return map_img

print("\n🎥 CAMERA FEED STARTING...")
print("Two windows should open:")
print("  1. 'Camera Feed' - Shows detected objects")
print("  2. '2D Radar' - Shows top-down view")
print("\nPress 'q' in either window to quit.")
print("=" * 60 + "\n")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame from camera")
        break

    frame_count += 1
    
    # Print status every 30 frames (~1 second at 30fps)
    if frame_count == 1:
        print("✓ First frame captured - windows should now be visible!")
    elif frame_count % 30 == 0:
        print(f"Processing frame {frame_count}...")

    results = model(frame, verbose=False)
    
    # [x_dist_in, y_dist_in, pixel_x, pixel_y, angle_deg, distance_total]
    current_frame_detections = [] 

    # --- 1. Process Raw Detections ---
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            name = model.names[cls]

            if conf > CONFIDENCE_THRESHOLD and name.lower() in ['fuel', 'fuels', 'ball']:
                # Math
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                width = x2 - x1
                
                # Angle & Distance
                offset = cx - (FRAME_WIDTH / 2)
                angle_rad = math.atan(offset / focal_length_px)
                angle_deg = math.degrees(angle_rad)
                
                if width > 0:
                    dist_total = (KNOWN_FUEL_DIAMETER_IN * focal_length_px) / width
                else:
                    dist_total = 0

                # Coordinates (Relative to Camera)
                y_in = dist_total * math.cos(angle_rad) # Forward
                x_in = dist_total * math.sin(angle_rad) # Right/Left

                current_frame_detections.append([x_in, y_in, cx, cy, angle_deg, dist_total, x1, y1, x2, y2])

    # --- 2. Determine Priority Target ---
    # We want the ball that minimizes: Distance + (Angle * Weight)
    best_target_index = -1
    lowest_cost = 99999.0

    for i, det in enumerate(current_frame_detections):
        dist = det[5]
        angle = abs(det[4])
        
        # Calculate "Cost"
        cost = dist + (angle * PRIORITY_ANGLE_WEIGHT)
        
        if cost < lowest_cost:
            lowest_cost = cost
            best_target_index = i

    # --- 3. Draw on Camera Feed ---
    for i, det in enumerate(current_frame_detections):
        x1, y1, x2, y2 = det[6], det[7], det[8], det[9]
        
        # Red Box for Priority, Green for others
        if i == best_target_index:
            color = (0, 0, 255) # BGR: Red
            label_prefix = "PRIORITY"
        else:
            color = (0, 255, 0) # BGR: Green
            label_prefix = "Fuel"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label_prefix}: {det[5]:.1f}in", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # --- 4. Draw Top-Down Map ---
    radar_view = draw_top_down_map(current_frame_detections, best_target_index)
    cv2.imshow("2D Radar", radar_view)

    # --- 5. Send to NetworkTables ---
    if ENABLE_NETWORKTABLES and best_target_index != -1:
        target = current_frame_detections[best_target_index]
        # Sending: HasTarget, Forward Distance, Angle
        sd.putBoolean("Fuel/HasTarget", True)
        sd.putNumber("Fuel/Distance", target[5])
        sd.putNumber("Fuel/Angle", target[4])
        sd.putNumber("Fuel/X_Offset", target[0]) # Left/Right inches
    elif ENABLE_NETWORKTABLES:
        sd.putBoolean("Fuel/HasTarget", False)

    cv2.imshow('Camera Feed', frame)

    # Check for 'q' key press to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n✓ Shutting down...")
        break

print("\n" + "=" * 60)
print("CAMERA FEED STOPPED")
print("=" * 60)
cap.release()
cv2.destroyAllWindows()
print("✓ Cleanup complete. Goodbye!")