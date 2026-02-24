import os

os.environ.setdefault("PYTORCH_DISABLE_NNPACK", "1")
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

import cv2
import numpy as np
from ultralytics import YOLO
import math
import time
import sys
import threading
from queue import Queue
from sklearn.cluster import DBSCAN

# OpenVINO 2026+ has simplified imports
try:
    from openvino.runtime import Core
except ImportError:
    from openvino import Core  # OpenVINO 2026+

try:
    from ntcore import NetworkTableInstance
except Exception as exc:
    raise ImportError(
        "WPILib NTCore not available. Install 'pyntcore' and remove conflicting 'ntcore' package."
    ) from exc

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = 'models/best.pt'
OPENVINO_MODEL_PATH = 'models/best_openvino_model'  # Will be generated if needed
USE_OPENVINO = True
USE_GPU = True  # Set to 'NPU' for Neural Processing Unit, True/'GPU' for GPU, False/'CPU' for CPU
USE_FP16 = True  # FP16 is better than INT8 on Cherry Trail (no VNNI support)

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
CAMERA_INDEX_CANDIDATES = [
    int(i.strip())
    for i in os.getenv("CAMERA_INDEX_CANDIDATES", "0,1,2").split(",")
    if i.strip().isdigit()
]

# --- NETWORK SETUP ---
IS_SIMULATION = True   # Set to False when on the real Robot/Kangaroo
TEAM_NUMBER = 3291       

# --- ROBOT PHYSICAL CONSTANTS ---
CAMERA_HFOV_DEG = 60.0         
KNOWN_FUEL_DIAMETER_IN = 5.91  
INCHES_TO_METERS = 0.0254
INTAKE_WIDTH_INCHES = 12.0  # Cluster balls within intake width
MAX_REACH_METERS = 3.0      # Ignore balls beyond 10 feet
MIN_DIST_FROM_WALL_IN = 8.0  # Week 0: Ignore balls too close to Alliance Wall glass

# --- PILE TARGETING SETTINGS ---
ENABLE_CLUSTERING = True     # Set False to disable pile targeting
CLUSTER_RADIUS_IN = 12.0     # Group balls within this distance (matches intake)
PILE_PRIORITY_WEIGHT = 50    # Score = (count * weight) - distance
NT_THROTTLE_FACTOR = 2       # Publish every Nth frame to save bandwidth
CONFIDENCE_FILTER = 0.67     # Minimum detection confidence (lowered for Week 0)

# --- RADAR SETTINGS ---
RADAR_WIDTH = 500
RADAR_HEIGHT = 500
GRID_SCALE = 4.0 

FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# --- ASYNC INFERENCE SETTINGS ---
ASYNC_QUEUE_SIZE = 2  # Keep pipeline short to avoid lag on slow hardware
INFERENCE_THREAD_PRIORITY = -10  # Lower priority to avoid starving camera thread
INFERENCE_DECIMATION = 2  # Run inference every Nth frame (1=every frame, 2=every other)

HEADLESS = True  # Set to True to disable GUI on Kangaroo

# ==========================================
# INITIALIZATION
# ==========================================
print("Starting FRC Vision Master Script (Cherry Trail Optimized)...")
print(f"Using OpenVINO: {USE_OPENVINO} | GPU: {USE_GPU} | FP16: {USE_FP16}")

# Initialize model
if USE_OPENVINO:
    # Setup OpenVINO with GPU acceleration
    core = Core()
    
    # Check if OpenVINO model exists, if not convert from PyTorch
    if not os.path.exists(OPENVINO_MODEL_PATH):
        print(f"Converting PyTorch model to OpenVINO (FP16)...")
        print(f"Target resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
        model = YOLO(MODEL_PATH)
        # Export to OpenVINO with FP16 precision and correct input shape
        # YOLOv8 prefers square inputs, use 320x320 for best compatibility
        model.export(format='openvino', half=USE_FP16, imgsz=320)
        print("Conversion complete.")
        print("Note: Model uses 320x320 input, frames will be padded")
    
    # Load OpenVINO model
    ov_model_xml = os.path.join(OPENVINO_MODEL_PATH, 'best.xml')
    ov_model_bin = os.path.join(OPENVINO_MODEL_PATH, 'best.bin')
    
    # Determine device (support string values like 'NPU', 'GPU', 'CPU')
    # For Kangaroo (Atom x5-Z8500): Force 'GPU' to offload weak CPU
    if isinstance(USE_GPU, str):
        device = USE_GPU.upper()
    elif USE_GPU:
        device = "GPU"
    else:
        device = "CPU"
    
    print(f"Compiling model for device: {device}")
    # Configure OpenVINO for Cherry Trail: single stream, limit threads
    config = {}
    if device == "GPU":
        config["PERFORMANCE_HINT"] = "LATENCY"  # Optimize for single-frame latency
    elif device == "CPU":
        config["NUM_STREAMS"] = "1"  # Single stream for weak CPU
        config["INFERENCE_NUM_THREADS"] = "2"  # Atom x5 has 4 threads, leave 2 for camera
    
    compiled_model = core.compile_model(ov_model_xml, device, config)
    infer_request = compiled_model.create_infer_request()
    
    # Load class names from original model
    original_model = YOLO(MODEL_PATH)
    model_names = original_model.names
    
    print(f"OpenVINO model loaded on {device}")
else:
    model = YOLO(MODEL_PATH)
    original_model = model
    model_names = model.names
    compiled_model = None
    infer_request = None

inst = NetworkTableInstance.getDefault()
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

# Legacy single-target publishers
pub_has_target = sd.getBooleanTopic("Fuelcv1/HasTarget").publish()
pub_target_angle = sd.getDoubleTopic("Fuelcv1/Angle").publish()

# NEW: Pile targeting publishers (for Auto-Collector)
pub_target_x = cv_table.getDoubleTopic("target_x").publish()
pub_target_y = cv_table.getDoubleTopic("target_y").publish()
pub_pile_has_target = cv_table.getBooleanTopic("has_target").publish()
pub_pile_size = cv_table.getIntegerTopic("target_pile_size").publish()
pub_pile_score = cv_table.getDoubleTopic("target_pile_score").publish()

# Telemetry tracking for HUD
sent_angle = 0.0
sent_dist = 0.0
target_status = "SEARCHING..."

# Async inference state
inference_queue = Queue(maxsize=ASYNC_QUEUE_SIZE)
latest_detections = None
detection_lock = threading.Lock()

def open_camera(index):
    backend_candidates = []
    if sys.platform.startswith("linux"):
        backend_candidates = [cv2.CAP_V4L2, cv2.CAP_ANY]
    elif sys.platform.startswith("win"):
        backend_candidates = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        backend_candidates = [cv2.CAP_ANY]

    for backend in backend_candidates:
        cap_local = cv2.VideoCapture(index, backend)
        if cap_local.isOpened():
            return cap_local
        cap_local.release()

    return None


def infer_openvino(frame, infer_request, compiled_model, conf_threshold):
    """Synchronous OpenVINO inference. Returns boxes in YOLO format."""
    # Prepare input (YOLOv8 expects RGB, normalized to 0-1)
    # Model expects 320x320, so resize to square with padding to avoid distortion
    frame_resized = cv2.resize(frame, (320, 320))
    # Convert BGR to RGB and normalize
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    input_tensor = np.expand_dims(frame_rgb.transpose(2, 0, 1), 0).astype(np.float32) / 255.0
    
    # Run inference (OpenVINO 2026 simplified API)
    # Use index-based access instead of names
    output = infer_request.infer(input_tensor)[0]
    
    # YOLOv8 output format: [batch, 84+num_classes, num_predictions]
    # Transpose to [batch, num_predictions, 84+num_classes]
    if len(output.shape) == 3 and output.shape[1] < output.shape[2]:
        output = output.transpose(0, 2, 1)
    
    raw_detections = []
    output = output[0]  # Remove batch dimension: [num_predictions, 84+num_classes]
    
    # YOLOv8 format: [cx, cy, w, h, class0_conf, class1_conf, ...]
    # No objectness score in YOLOv8, classes start at index 4
    for prediction in output:
        # Get class confidences (from index 4 onward)
        class_confs = prediction[4:]
        class_id = np.argmax(class_confs)
        conf = float(class_confs[class_id])
        
        if conf > conf_threshold:
            # Decode bounding box (center format to corner format)
            # Model output is in 320x320 space, need to scale to 320x240 display frame
            cx, cy, w, h = prediction[0:4]
            # Boxes are in pixel coordinates for 320x320 input
            x1 = int((cx - w/2))
            y1 = int((cy - h/2) * (FRAME_HEIGHT / 320))  # Scale Y from 320 to 240
            x2 = int((cx + w/2))
            y2 = int((cy + h/2) * (FRAME_HEIGHT / 320))  # Scale Y from 320 to 240
            
            # Clamp to frame boundaries (320x240)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(FRAME_WIDTH, x2), min(FRAME_HEIGHT, y2)
            
            raw_detections.append({
                'box': (x1, y1, x2, y2),
                'conf': conf,
                'class_id': int(class_id)
            })
    
    # Apply Non-Maximum Suppression (NMS) to remove duplicate detections
    if raw_detections:
        boxes = np.array([d['box'] for d in raw_detections])
        scores = np.array([d['conf'] for d in raw_detections])
        
        # Convert (x1, y1, x2, y2) to format expected by cv2.dnn.NMSBoxes
        boxes_xywh = []
        for box in boxes:
            x1, y1, x2, y2 = box
            boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])
        
        # Apply NMS with IoU threshold of 0.45 (standard for YOLO)
        indices = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), conf_threshold, 0.45)
        
        # Filter detections based on NMS indices
        detections = []
        if len(indices) > 0:
            indices = indices.flatten() if len(indices.shape) > 1 else indices
            for i in indices:
                detections.append(raw_detections[i])
        
        return detections
    
    return raw_detections


def inference_worker():
    """Background thread for async inference on Cherry Trail."""
    global latest_detections
    
    while True:
        try:
            frame, frame_resized = inference_queue.get(timeout=1)
            
            if frame is None:  # Sentinel value to stop thread
                break
            
            if USE_OPENVINO:
                detections = infer_openvino(frame, infer_request, compiled_model, CONFIDENCE_FILTER)
            else:
                results = model(frame_resized, verbose=False)
                detections = []
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0])
                        name = model_names[class_id]
                        
                        if conf > CONFIDENCE_FILTER and name.lower() in ['fuel', 'fuels', 'ball']:
                            detections.append({
                                'box': (x1, y1, x2, y2),
                                'conf': conf,
                                'class_id': class_id
                            })
            
            with detection_lock:
                latest_detections = detections
                
        except Exception as e:
            print(f"Inference worker error: {e}")
            continue


# Open camera with buffer optimization for Kangaroo
camera_indexes = [CAMERA_INDEX] + [idx for idx in CAMERA_INDEX_CANDIDATES if idx != CAMERA_INDEX]
cap = None
for idx in camera_indexes:
    cap = open_camera(idx)
    if cap is not None and cap.isOpened():
        CAMERA_INDEX = idx
        # Optimize camera settings for low-latency on Kangaroo
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag
        print(f"Camera opened at index {CAMERA_INDEX}")
        break

if cap is None or not cap.isOpened():
    raise RuntimeError(f"Unable to open camera. Tried indexes: {camera_indexes}")

focal_length_px = FRAME_WIDTH / (2 * math.tan(math.radians(CAMERA_HFOV_DEG / 2)))
frame_center_x = FRAME_WIDTH / 2

def create_radar_frame(detections, best_idx):
    radar = np.zeros((RADAR_HEIGHT, RADAR_WIDTH, 3), dtype=np.uint8)
    origin_x, origin_y = RADAR_WIDTH // 2, RADAR_HEIGHT - 50
    cv2.rectangle(radar, (origin_x-10, origin_y-10), (origin_x+10, origin_y+10), (0, 255, 0), -1)
    for r in range(50, 250, 50):
        cv2.circle(radar, (origin_x, origin_y), int(r * GRID_SCALE), (50, 50, 50), 1)

    for i, d in enumerate(detections):
        px = int(origin_x - (d['y_in'] * GRID_SCALE))
        py = int(origin_y - (d['x_in'] * GRID_SCALE))
        color = (0, 0, 255) if i == best_idx else (0, 255, 255)
        if 0 < px < RADAR_WIDTH and 0 < py < RADAR_HEIGHT:
            cv2.circle(radar, (px, py), 8, color, -1)
    return radar


def process_detections(raw_detections, frame):
    """Convert raw detections to world coordinates (vectorized for speed)."""
    if not raw_detections:
        return [], [], [], [], [], []
    
    # Pre-filter by confidence and class (vectorized check)
    valid_detections = [
        d for d in raw_detections
        if d['conf'] > CONFIDENCE_FILTER and 
           model_names.get(d['class_id'], '').lower() in ['fuel', 'fuels', 'ball']
    ]
    
    if not valid_detections:
        return [], [], [], [], [], []
    
    # Vectorize coordinate calculations with numpy
    boxes = np.array([d['box'] for d in valid_detections])
    cxs = (boxes[:, 0] + boxes[:, 2]) / 2
    w_pxs = boxes[:, 2] - boxes[:, 0]
    
    # Batch calculate angles and distances
    yaws_rad_arr = np.arctan((cxs - frame_center_x) / focal_length_px)
    dists_in_arr = np.where(w_pxs > 0, (KNOWN_FUEL_DIAMETER_IN * focal_length_px) / w_pxs, 0)
    
    # Robot-relative coordinates (vectorized)
    x_inches_arr = dists_in_arr * np.cos(yaws_rad_arr)
    y_inches_arr = -dists_in_arr * np.sin(yaws_rad_arr)
    
    # Apply filters (distance and wall proximity)
    dists_m_arr = dists_in_arr * INCHES_TO_METERS
    mask = (dists_in_arr > MIN_DIST_FROM_WALL_IN) & (dists_m_arr < MAX_REACH_METERS)
    
    # Extract filtered results
    yaws_rad = yaws_rad_arr[mask].tolist()
    dists_m = dists_m_arr[mask].tolist()
    xs_m = (x_inches_arr[mask] * INCHES_TO_METERS).tolist()
    ys_m = (y_inches_arr[mask] * INCHES_TO_METERS).tolist()
    
    # Build output structures for filtered detections
    screen_detections = []
    balls_in_inches = []
    for i, is_valid in enumerate(mask):
        if is_valid:
            screen_detections.append({
                'box': valid_detections[i]['box'],
                'dist_in': dists_in_arr[i],
                'angle_deg': math.degrees(yaws_rad_arr[i]),
                'x_in': x_inches_arr[i],
                'y_in': y_inches_arr[i]
            })
            balls_in_inches.append([x_inches_arr[i], y_inches_arr[i]])
    
    return yaws_rad, dists_m, xs_m, ys_m, screen_detections, balls_in_inches


def find_best_pile(balls_in_inches):
    """Use DBSCAN clustering to find the best pile of balls."""
    if len(balls_in_inches) == 0:
        return None, 0, 0.0
    
    coords = np.array(balls_in_inches)
    
    # Cluster balls within CLUSTER_RADIUS_IN inches
    db = DBSCAN(eps=CLUSTER_RADIUS_IN, min_samples=1).fit(coords)
    
    best_cluster_center = None
    max_balls = 0
    best_score = -999.0
    
    for label in set(db.labels_):
        if label == -1:  # Noise points (shouldn't happen with min_samples=1)
            continue
        
        cluster_points = coords[db.labels_ == label]
        count = len(cluster_points)
        center = np.mean(cluster_points, axis=0)  # Midpoint of pile
        
        # Scoring: Prefer large piles that are close
        # Score = (Balls * PRIORITY_WEIGHT) - Distance
        dist = np.linalg.norm(center)
        score = (count * PILE_PRIORITY_WEIGHT) - dist
        
        if score > best_score:
            best_score = score
            best_cluster_center = center
            max_balls = count
    
    return best_cluster_center, max_balls, best_score


# Start async inference worker thread
inference_thread = threading.Thread(target=inference_worker, daemon=True)
inference_thread.start()
print("Async inference worker started")
print(f"Clustering enabled: {ENABLE_CLUSTERING} | Intake width: {INTAKE_WIDTH_INCHES}\"")
print(f"Wall filter: {MIN_DIST_FROM_WALL_IN}\" | Max reach: {MAX_REACH_METERS}m | Confidence: {CONFIDENCE_FILTER}")
print(f"Inference decimation: 1/{INFERENCE_DECIMATION} | Network throttle: 1/{NT_THROTTLE_FACTOR}")

# ==========================================
# MAIN LOOP
# ==========================================
frame_count = 0
perf_timer = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        
        # Queue frame for async inference (decimated to save CPU)
        if frame_count % INFERENCE_DECIMATION == 0:
            try:
                inference_queue.put_nowait((frame.copy(), frame))
            except:
                pass  # Queue full, skip this frame
        
        # Process latest detections (from previous frame in async mode)
        with detection_lock:
            raw_detections = latest_detections if latest_detections is not None else []
        
        if raw_detections:
            yaws_rad, dists_m, xs_m, ys_m, screen_detections, balls_in_inches = process_detections(raw_detections, frame)
        else:
            yaws_rad, dists_m, xs_m, ys_m, screen_detections, balls_in_inches = [], [], [], [], [], []
        
        # Priority Logic (Legacy - single ball targeting)
        best_idx = -1
        if screen_detections:
            costs = [d['dist_in'] + (abs(d['angle_deg']) * 0.2) for d in screen_detections]
            best_idx = np.argmin(costs)
        
        # PILE TARGETING (NEW)
        pile_center = None
        pile_size = 0
        pile_score = 0.0
        
        if ENABLE_CLUSTERING and balls_in_inches:
            pile_center, pile_size, pile_score = find_best_pile(balls_in_inches)
        
        # THROTTLED PUBLISHING (Save network bandwidth)
        if frame_count % NT_THROTTLE_FACTOR == 0:
            # Publish all individual balls (for field map visualization)
            pub_num.set(len(xs_m))
            pub_yaw.set(yaws_rad)
            pub_dist.set(dists_m)
            pub_x.set(xs_m)
            pub_y.set(ys_m)
            
            # Legacy single-ball target (for manual drive)
            if best_idx != -1:
                target = screen_detections[best_idx]
                pub_has_target.set(True)
                pub_target_angle.set(target['angle_deg'])
                sent_angle, sent_dist = target['angle_deg'], target['dist_in']
            else:
                pub_has_target.set(False)
            
            # NEW: Publish best pile target (for auto-collector)
            if pile_center is not None:
                pub_pile_has_target.set(True)
                pub_target_x.set(pile_center[0] * INCHES_TO_METERS)  # Convert to meters
                pub_target_y.set(pile_center[1] * INCHES_TO_METERS)
                pub_pile_size.set(pile_size)
                pub_pile_score.set(pile_score)
                target_status = f"PILE:{pile_size}"
                # Update HUD with pile info
                pile_dist = math.sqrt(pile_center[0]**2 + pile_center[1]**2)
                pile_angle = math.degrees(math.atan2(pile_center[1], pile_center[0]))
                sent_angle, sent_dist = pile_angle, pile_dist
            else:
                pub_pile_has_target.set(False)
                target_status = "SEARCHING..."
        
        # HUD & UI Drawing (skip if headless to save CPU)
        if not HEADLESS:
            cv2.rectangle(frame, (0, 0), (280, 140), (0, 0, 0), -1)
            cv2.putText(frame, f"NT4 {'CONNECTED' if inst.isConnected() else 'OFFLINE'}", (10, 20), 0, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"STATUS: {target_status}", (10, 40), 0, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"ANGLE : {sent_angle:.1f} deg", (10, 60), 0, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"DIST  : {sent_dist:.1f} in", (10, 80), 0, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"BALLS : {len(xs_m)} detected", (10, 100), 0, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, f"SCORE : {pile_score:.0f}", (10, 120), 0, 0.5, (255, 0, 255), 1)
            
            # Determine device name for display
            if USE_OPENVINO:
                if isinstance(USE_GPU, str):
                    device_name = USE_GPU.upper()
                elif USE_GPU:
                    device_name = "GPU"
                else:
                    device_name = "CPU"
                mode_text = f"OpenVINO+{device_name}"
            else:
                mode_text = "PyTorch"
            
            clustering_text = "PILE" if ENABLE_CLUSTERING else "SINGLE"
            cv2.putText(frame, f"MODE: {mode_text}|{clustering_text}", (10, 140), 0, 0.45, (0, 255, 255), 1)
            
            # Draw bounding boxes
            for i, d in enumerate(screen_detections):
                color = (0, 0, 255) if i == best_idx else (0, 255, 0)
                cv2.rectangle(frame, d['box'][0:2], d['box'][2:4], color, 2)
            
            # Draw pile size indicator on best pile
            if pile_center is not None and pile_size > 1:
                # Find screen position closest to pile center
                for d in screen_detections:
                    dx = d['x_in'] - pile_center[0]
                    dy = d['y_in'] - pile_center[1]
                    if math.sqrt(dx*dx + dy*dy) < CLUSTER_RADIUS_IN / 2:
                        x1, y1, x2, y2 = d['box']
                        cv2.putText(frame, f"x{pile_size}", (x1, y1-5), 0, 0.6, (255, 0, 255), 2)
                        break
            
            cv2.imshow('Robot Camera (Cherry Trail Optimized)', frame)
            cv2.imshow('2D Radar Map', create_radar_frame(screen_detections, best_idx))
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - perf_timer
            fps = 30 / elapsed if elapsed > 0 else 0
            print(f"FPS: {fps:.1f} | Detections: {len(xs_m)} | Queue: {inference_queue.qsize()}")
            perf_timer = time.time()

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    # Cleanup
    inference_queue.put(None)  # Signal worker to stop
    inference_thread.join(timeout=2)
    cap.release()
    if not HEADLESS:
        cv2.destroyAllWindows()
    print("Vision system shutdown complete")