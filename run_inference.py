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
CONFIDENCE_THRESHOLD = 0.65

# --- NETWORK SETUP ---
IS_SIMULATION = True   # Set to False when on the real Robot/Kangaroo
TEAM_NUMBER = 2791       

# --- ROBOT PHYSICAL CONSTANTS ---
CAMERA_HFOV_DEG = 60.0         
KNOWN_FUEL_DIAMETER_IN = 5.91  
INCHES_TO_METERS = 0.0254

# --- RADAR SETTINGS ---
RADAR_WIDTH = 500
RADAR_HEIGHT = 500
GRID_SCALE = 4.0 

# ==========================================
# INITIALIZATION
# ==========================================
print("Starting FRC Vision Master Script...")
model = YOLO(MODEL_PATH)

inst = ntcore.NetworkTableInstance.getDefault()
if IS_SIMULATION:
    inst.setServer("127.0.0.1")
else:
    inst.setServerTeam(TEAM_NUMBER)
inst.startClient4("KangarooVision")

sd = inst.getTable("SmartDashboard") 
cv_table = inst.getTable("fuelCV")    

pub_num = cv_table.getIntegerTopic("number_of_fuel").publish()
pub_yaw = cv_table.getDoubleArrayTopic("yaw_radians").publish()
pub_dist = cv_table.getDoubleArrayTopic("distance").publish()
pub_x = cv_table.getDoubleArrayTopic("ball_position_x").publish()
pub_y = cv_table.getDoubleArrayTopic("ball_position_y").publish()

pub_has_target = sd.getBooleanTopic("Fuelcv1/HasTarget").publish()
pub_target_angle = sd.getDoubleTopic("Fuelcv1/Angle").publish()

# Telemetry tracking for HUD
sent_angle = 0.0
sent_dist = 0.0
target_status = "SEARCHING..."

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
focal_length_px = 640 / (2 * math.tan(math.radians(CAMERA_HFOV_DEG / 2)))

def create_radar_frame(detections, best_idx):
    radar = np.zeros((RADAR_HEIGHT, RADAR_WIDTH, 3), dtype=np.uint8)
    origin_x, origin_y = RADAR_WIDTH // 2, RADAR_HEIGHT - 50
    cv2.rectangle(radar, (origin_x-10, origin_y-10), (origin_x+10, origin_y+10), (0, 255, 0), -1)
    for r in range(50, 250, 50):
        cv2.circle(radar, (origin_x, origin_y), int(r * GRID_SCALE), (50, 50, 50), 1)

    for i, d in enumerate(detections):
        # Radar drawing still uses inches for display
        px = int(origin_x - (d['y_in'] * GRID_SCALE)) # Y in code is side-to-side
        py = int(origin_y - (d['x_in'] * GRID_SCALE)) # X in code is forward
        color = (0, 0, 255) if i == best_idx else (0, 255, 255)
        if 0 < px < RADAR_WIDTH and 0 < py < RADAR_HEIGHT:
            cv2.circle(radar, (px, py), 8, color, -1)
    return radar

# ==========================================
# MAIN LOOP
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.resize(frame, (640, 480))
    results = model(frame, verbose=False)
    
    yaws_rad, dists_m, xs_m, ys_m = [], [], [], []
    screen_detections = [] 

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = box.conf[0].cpu().numpy()
            name = model.names[int(box.cls[0])]

            if conf > CONFIDENCE_THRESHOLD and name.lower() in ['fuel', 'fuels', 'ball']:
                cx = (x1 + x2) // 2
                w_px = x2 - x1
                
                # 1. Math for Angle & Total Distance
                yaw_rad = math.atan((cx - 320) / focal_length_px)
                dist_total_in = (KNOWN_FUEL_DIAMETER_IN * focal_length_px) / w_px if w_px > 0 else 0
                
                # 2. COORDINATE MATH (Matches your Java snippet)
                # Forward (X) and Sideways (Y)
                x_meters = (dist_total_in * math.cos(yaw_rad)) * INCHES_TO_METERS
                y_meters = -(dist_total_in * math.sin(yaw_rad)) * INCHES_TO_METERS 

                # 3. Store for NetworkTables
                yaws_rad.append(yaw_rad)
                dists_m.append(dist_total_in * INCHES_TO_METERS)
                xs_m.append(x_meters)
                ys_m.append(y_meters)
                
                # Store for UI Drawing (Inches)
                screen_detections.append({
                    'box': (x1,y1,x2,y2), 
                    'dist_in': dist_total_in, 
                    'angle_deg': math.degrees(yaw_rad),
                    'x_in': dist_total_in * math.cos(yaw_rad),
                    'y_in': dist_total_in * math.sin(yaw_rad)
                })

    # Priority Logic
    best_idx = -1
    if screen_detections:
        costs = [d['dist_in'] + (abs(d['angle_deg']) * 0.2) for d in screen_detections]
        best_idx = np.argmin(costs)

    # Publish
    pub_num.set(len(xs_m))
    pub_yaw.set(yaws_rad)
    pub_dist.set(dists_m)
    pub_x.set(xs_m)
    pub_y.set(ys_m)

    if best_idx != -1:
        target = screen_detections[best_idx]
        pub_has_target.set(True)
        pub_target_angle.set(target['angle_deg'])
        sent_angle, sent_dist = target['angle_deg'], target['dist_in']
        target_status = "LOCKED"
    else:
        pub_has_target.set(False)
        target_status = "SEARCHING..."

    # HUD & UI Drawing
    cv2.rectangle(frame, (0, 0), (260, 100), (0, 0, 0), -1)
    cv2.putText(frame, f"NT4 {'CONNECTED' if inst.isConnected() else 'OFFLINE'}", (10, 20), 0, 0.5, (0,255,0), 1)
    cv2.putText(frame, f"STATUS: {target_status}", (10, 40), 0, 0.5, (255,255,255), 1)
    cv2.putText(frame, f"ANGLE : {sent_angle:.1f} deg", (10, 60), 0, 0.5, (255,255,255), 1)
    cv2.putText(frame, f"DIST  : {sent_dist:.1f} in", (10, 80), 0, 0.5, (255,255,255), 1)

    for i, d in enumerate(screen_detections):
        color = (0, 0, 255) if i == best_idx else (0, 255, 0)
        cv2.rectangle(frame, d['box'][0:2], d['box'][2:4], color, 2)

    cv2.imshow('Robot Camera', frame)
    cv2.imshow('2D Radar Map', create_radar_frame(screen_detections, best_idx))

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()