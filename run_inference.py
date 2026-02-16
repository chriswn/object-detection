import cv2
import numpy as np
from ultralytics import YOLO
import math
import ntcore
import time

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = 'models/best.pt'
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.45

# --- NETWORK SETUP ---
IS_SIMULATION = True   # Set to False when on the real Robot/Kangaroo
TEAM_NUMBER = 2791       # Your FRC team number

# --- ROBOT PHYSICAL CONSTANTS (Measure these!) ---
CAMERA_HFOV_DEG = 60.0         
KNOWN_FUEL_DIAMETER_IN = 5.91  
INCHES_TO_METERS = 0.0254

# --- RADAR SETTINGS ---
RADAR_WIDTH = 500
RADAR_HEIGHT = 500
GRID_SCALE = 4.0 # Pixels per inch

# ==========================================
# INITIALIZATION
# ==========================================
print("Starting FRC Vision System (OpenCV Mode)...")
model = YOLO(MODEL_PATH)

# Initialize NetworkTables 4
inst = ntcore.NetworkTableInstance.getDefault()
if IS_SIMULATION:
    inst.setServer("127.0.0.1")
    print("Connecting to local Simulation...")
else:
    inst.setServerTeam(TEAM_NUMBER)
    print(f"Connecting to Team {TEAM_NUMBER} Robot...")
inst.startClient4("KangarooVision")

# Tables
sd = inst.getTable("SmartDashboard") 
cv_table = inst.getTable("fuelCV")    

# Publishers
pub_num = cv_table.getIntegerTopic("number_of_fuel").publish()
pub_yaw = cv_table.getDoubleArrayTopic("yaw_radians").publish()
pub_dist = cv_table.getDoubleArrayTopic("distance").publish()
pub_x = cv_table.getDoubleArrayTopic("ball_position_x").publish()
pub_y = cv_table.getDoubleArrayTopic("ball_position_y").publish()

pub_has_target = sd.getBooleanTopic("Fuel/HasTarget").publish()
pub_target_angle = sd.getDoubleTopic("Fuel/Angle").publish()

# NetworkTables connection tracking
nt_connected = False
sent_angle = 0.0
sent_dist = 0.0
target_status = "SEARCHING..."

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
focal_length_px = 640 / (2 * math.tan(math.radians(CAMERA_HFOV_DEG / 2)))

def create_radar_frame(detections, best_idx):
    """Draws a top-down view using only OpenCV"""
    # Create black background
    radar = np.zeros((RADAR_HEIGHT, RADAR_WIDTH, 3), dtype=np.uint8)
    
    # Robot origin (Bottom center)
    origin_x = RADAR_WIDTH // 2
    origin_y = RADAR_HEIGHT - 50

    # Draw Robot (Green square)
    cv2.rectangle(radar, (origin_x-10, origin_y-10), (origin_x+10, origin_y+10), (0, 255, 0), -1)
    
    # Draw distance circles (every 50 inches)
    for r in range(50, 250, 50):
        cv2.circle(radar, (origin_x, origin_y), int(r * GRID_SCALE), (50, 50, 50), 1)

    # Draw Balls
    for i, d in enumerate(detections):
        # Map inches to pixels
        px = int(origin_x + (d['x'] * GRID_SCALE))
        py = int(origin_y - (d['y'] * GRID_SCALE))
        
        # Priority ball is Red, others are Yellow
        color = (0, 0, 255) if i == best_idx else (0, 255, 255)
        
        if 0 < px < RADAR_WIDTH and 0 < py < RADAR_HEIGHT:
            cv2.circle(radar, (px, py), 8, color, -1)
            # Draw line to priority
            if i == best_idx:
                cv2.line(radar, (origin_x, origin_y), (px, py), (0, 0, 255), 1)

    return radar

# ==========================================
# MAIN LOOP
# ==========================================
print("Running. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.resize(frame, (640, 480))

    results = model(frame, verbose=False)
    
    # Lists for NetworkTable Arrays
    yaws, dists, xs_m, ys_m = [], [], [], []
    screen_detections = [] 

    # 1. PROCESS DETECTIONS
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = box.conf[0].cpu().numpy()
            name = model.names[int(box.cls[0])]

            if conf > CONFIDENCE_THRESHOLD and name.lower() in ['fuel', 'fuels', 'ball']:
                cx = (x1 + x2) // 2
                w_px = x2 - x1
                
                # Math
                yaw_rad = math.atan((cx - 320) / focal_length_px)
                dist_in = (KNOWN_FUEL_DIAMETER_IN * focal_length_px) / w_px if w_px > 0 else 0
                
                # Relative Coordinates (Inches)
                x_in = dist_in * math.sin(yaw_rad)
                y_in = dist_in * math.cos(yaw_rad)

                # Store for Java (Meters)
                yaws.append(yaw_rad)
                dists.append(dist_in * INCHES_TO_METERS)
                xs_m.append(x_in * INCHES_TO_METERS)
                ys_m.append(y_in * INCHES_TO_METERS)
                
                screen_detections.append({'box': (x1,y1,x2,y2), 'dist': dist_in, 'angle': math.degrees(yaw_rad), 'x': x_in, 'y': y_in})

    # 2. PRIORITY LOGIC
    best_idx = -1
    if screen_detections:
        costs = [d['dist'] + (abs(d['angle']) * 0.2) for d in screen_detections]
        best_idx = np.argmin(costs)

    # 3. PUBLISH TO NETWORKTABLES
    try:
        pub_num.set(len(xs_m))
        pub_yaw.set(yaws)
        pub_dist.set(dists)
        pub_x.set(xs_m)
        pub_y.set(ys_m)
        
        if best_idx != -1:
            pub_has_target.set(True)
            pub_target_angle.set(screen_detections[best_idx]['angle'])
            sent_angle = screen_detections[best_idx]['angle']
            sent_dist = screen_detections[best_idx]['dist']
            target_status = "TARGET LOCKED"
        else:
            pub_has_target.set(False)
            target_status = "SEARCHING..."
        
        nt_connected = True
    except Exception as e:
        nt_connected = False
        print(f"NetworkTables error: {e}")

    # 4. DRAWING
    for i, d in enumerate(screen_detections):
        color = (0, 0, 255) if i == best_idx else (0, 255, 0)
        x1, y1, x2, y2 = d['box']
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = "TARGET" if i == best_idx else "FUEL"
        cv2.putText(frame, f"{label} {d['dist']:.0f}in", (x1, y1-5), 0, 0.5, color, 1)

    # 4. DRAW HUD (Connection & Telemetry)
    # Draw a semi-transparent background box for text
    cv2.rectangle(frame, (0, 0), (280, 110), (0, 0, 0), -1)
    
    # Connection Indicator
    conn_text = "CONNECTED" if nt_connected else "DISCONNECTED"
    conn_color = (0, 255, 0) if nt_connected else (0, 0, 255)  # Green if good, Red if bad
    mode_text = "SIM" if IS_SIMULATION else f"TEAM {TEAM_NUMBER}"
    
    cv2.putText(frame, f"NT4 {mode_text}: {conn_text}", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, conn_color, 2)
    
    # Telemetry Data
    cv2.putText(frame, f"STATUS: {target_status}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"SENT ANGLE: {sent_angle:.2f} deg", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"SENT DIST : {sent_dist:.2f} in", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Show Camera and Radar
    radar_frame = create_radar_frame(screen_detections, best_idx)
    
    cv2.imshow('Robot Camera', frame)
    cv2.imshow('2D Radar Map', radar_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()