import os
import json

os.environ.setdefault("PYTORCH_DISABLE_NNPACK", "1")
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

import cv2
import numpy as np
from ultralytics import YOLO
import math
import time
import sys
import threading
from queue import Queue, Empty
from sklearn.cluster import DBSCAN

# OpenVINO 2026+ has simplified imports
try:
    from openvino.runtime import Core
except ImportError:
    from openvino import Core  # OpenVINO 2026+

try:
    import ntcore as _ntcore_module
    from ntcore import NetworkTableInstance
except Exception as exc:
    raise ImportError(
        "WPILib NTCore not available. Install 'pyntcore' and remove conflicting 'ntcore' package."
    ) from exc

def _now_us() -> int:
    """
    Return the current time in microseconds.

    Tries ntcore.now() first (matches the NT server clock used by the RIO
    for latency compensation).  Falls back to time.monotonic_ns() // 1000
    on pyntcore builds that don't expose the module-level function.
    """
    try:
        return int(_ntcore_module.now())
    except AttributeError:
        return time.monotonic_ns() // 1000


# ==========================================
# LOAD EXTERNAL CONFIGURATION
# ==========================================
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

def _load_config(path: str) -> dict:
    """Load config.json, stripping keys that start with '_comment'."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"config.json not found at '{path}'. "
            "Copy config.json next to run_inference.py before starting."
        )
    with open(path, "r") as f:
        raw = json.load(f)

    def _strip_comments(obj):
        if isinstance(obj, dict):
            return {k: _strip_comments(v) for k, v in obj.items() if not k.startswith("_comment")}
        return obj

    return _strip_comments(raw)

cfg = _load_config(_CONFIG_PATH)

# ==========================================
# CONFIGURATION (read from config.json)
# ==========================================

# --- Game / Game-piece ---
# TARGET_CLASSES drives ALL class filtering; update config.json at Kickoff.
TARGET_CLASSES: list[str] = [c.lower() for c in cfg["game"]["target_classes"]]
KNOWN_PIECE_DIAMETER_IN: float = cfg["game"]["known_piece_diameter_in"]
ASPECT_RATIO_MIN: float = cfg["game"]["aspect_ratio_min"]
ASPECT_RATIO_MAX: float = cfg["game"]["aspect_ratio_max"]

# --- Field bounds (for sanity-check filtering) ---
FIELD_WIDTH_M: float  = cfg["field"]["width_m"]
FIELD_HEIGHT_M: float = cfg["field"]["height_m"]

# --- Robot physical ---
CAMERA_HFOV_DEG: float   = cfg["robot"]["camera_hfov_deg"]
INTAKE_WIDTH_INCHES: float = cfg["robot"]["intake_width_inches"]
MAX_REACH_METERS: float    = cfg["robot"]["max_reach_meters"]
MIN_DIST_FROM_WALL_IN: float = cfg["robot"]["min_dist_from_wall_in"]

# --- Pipeline ---
USE_OPENVINO: bool  = cfg["pipeline"]["use_openvino"]
USE_GPU             = cfg["pipeline"]["use_gpu"]   # False | "GPU" | "NPU"
USE_FP16: bool      = cfg["pipeline"]["use_fp16"]
HEADLESS: bool      = cfg["pipeline"]["headless"]
FRAME_WIDTH: int    = cfg["pipeline"]["frame_width"]
FRAME_HEIGHT: int   = cfg["pipeline"]["frame_height"]
CONFIDENCE_FILTER: float = cfg["pipeline"]["confidence_filter"]
INFERENCE_DECIMATION: int = cfg["pipeline"]["inference_decimation"]
NT_THROTTLE_FACTOR: int   = cfg["pipeline"]["nt_throttle_factor"]
ASYNC_QUEUE_SIZE: int     = cfg["pipeline"]["async_queue_size"]

# --- Clustering / TargetSelector ---
ENABLE_CLUSTERING: bool    = cfg["clustering"]["enabled"]
CLUSTER_RADIUS_IN: float   = cfg["clustering"]["cluster_radius_in"]
PILE_PRIORITY_WEIGHT: float = cfg["clustering"]["pile_priority_weight"]

# --- Network ---
IS_SIMULATION: bool = cfg["network"]["is_simulation"]
TEAM_NUMBER: int    = cfg["network"]["team_number"]
NT_CLIENT_NAME: str = cfg["network"]["nt_client_name"]

# --- Camera ---
CAMERA_INDEX: int = int(os.getenv("CAMERA_INDEX", str(cfg["camera"]["index"])))
CAMERA_INDEX_CANDIDATES: list[int] = [
    int(i.strip())
    for i in os.getenv(
        "CAMERA_INDEX_CANDIDATES",
        ",".join(str(x) for x in cfg["camera"]["index_candidates"])
    ).split(",")
    if i.strip().isdigit()
]

# --- Model paths ---
MODEL_PATH: str          = cfg["model"]["pt_path"]
OPENVINO_MODEL_PATH: str = cfg["model"]["openvino_path"]

# --- Radar ---
RADAR_WIDTH: int  = cfg["radar"]["width"]
RADAR_HEIGHT: int = cfg["radar"]["height"]
GRID_SCALE: float = cfg["radar"]["grid_scale"]

# --- Derived constants ---
INCHES_TO_METERS: float = 0.0254
focal_length_px: float = FRAME_WIDTH / (2 * math.tan(math.radians(CAMERA_HFOV_DEG / 2)))
frame_center_x: float  = FRAME_WIDTH / 2

# ==========================================
# RUNTIME STATE (initialized in main())
# ==========================================
model = None
model_names: dict = {}
compiled_model = None
infer_request = None
inference_queue = None

# Each entry: (detections_list, frame_timestamp_us)
latest_detections = None
latest_frame_timestamp_us: int = 0

detection_lock = threading.Lock()


# ==========================================
# HELPERS
# ==========================================

def open_camera(index: int):
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


def _is_valid_aspect_ratio(box: tuple) -> bool:
    """
    Return True if the bounding box's width/height ratio falls within
    [ASPECT_RATIO_MIN, ASPECT_RATIO_MAX].  Filters out tall/thin or
    wide/flat false positives (bumper reflections, field markings, etc.).
    """
    x1, y1, x2, y2 = box
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)
    ratio = w / h
    return ASPECT_RATIO_MIN <= ratio <= ASPECT_RATIO_MAX


# ==========================================
# INFERENCE
# ==========================================

def infer_openvino(frame, infer_request, compiled_model, conf_threshold: float) -> list:
    """Synchronous OpenVINO inference. Returns boxes in YOLO format."""
    # Prepare input (YOLOv8 expects RGB, normalized 0-1, square 320x320)
    frame_resized = cv2.resize(frame, (320, 320))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    input_tensor = np.expand_dims(frame_rgb.transpose(2, 0, 1), 0).astype(np.float32) / 255.0

    # Run inference (OpenVINO 2026 simplified API — index-based output)
    output = infer_request.infer(input_tensor)[0]

    # YOLOv8 output: [batch, 84+classes, predictions] → transpose to [batch, predictions, 84+classes]
    if len(output.shape) == 3 and output.shape[1] < output.shape[2]:
        output = output.transpose(0, 2, 1)

    raw_detections = []
    output = output[0]  # Remove batch dim → [num_predictions, 84+classes]

    # YOLOv8: [cx, cy, w, h, class0_conf, class1_conf, ...]
    for prediction in output:
        class_confs = prediction[4:]
        class_id = int(np.argmax(class_confs))
        conf = float(class_confs[class_id])

        if conf > conf_threshold:
            cx, cy, w, h = prediction[0:4]
            # Scale Y from 320→FRAME_HEIGHT; X stays at FRAME_WIDTH (both 320 wide)
            x1 = int(cx - w / 2)
            y1 = int((cy - h / 2) * (FRAME_HEIGHT / 320))
            x2 = int(cx + w / 2)
            y2 = int((cy + h / 2) * (FRAME_HEIGHT / 320))

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(FRAME_WIDTH, x2), min(FRAME_HEIGHT, y2)

            raw_detections.append({
                'box': (x1, y1, x2, y2),
                'conf': conf,
                'class_id': class_id,
            })

    # Non-Maximum Suppression
    if not raw_detections:
        return raw_detections

    boxes = np.array([d['box'] for d in raw_detections])
    scores = np.array([d['conf'] for d in raw_detections])
    boxes_xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes]
    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), conf_threshold, 0.45)

    if len(indices) == 0:
        return []

    indices = indices.flatten() if len(indices.shape) > 1 else indices
    return [raw_detections[i] for i in indices]


def inference_worker():
    """
    Background thread for async inference on Cherry Trail.

    Queue items: (frame_copy, frame_resized, capture_timestamp_us)
    """
    global latest_detections, latest_frame_timestamp_us

    while True:
        try:
            item = inference_queue.get(timeout=1)

            if item is None:  # Sentinel: shut down
                break

            frame, frame_resized, capture_timestamp_us = item

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
                        # Class-agnostic filter: driven entirely by TARGET_CLASSES from config
                        name = model_names.get(class_id, "").lower()
                        if conf > CONFIDENCE_FILTER and name in TARGET_CLASSES:
                            detections.append({
                                'box': (x1, y1, x2, y2),
                                'conf': conf,
                                'class_id': class_id,
                            })

            with detection_lock:
                latest_detections = detections
                latest_frame_timestamp_us = capture_timestamp_us

        except Empty:
            continue
        except Exception as e:
            print(f"Inference worker error [{type(e).__name__}]: {e}")
            continue


# ==========================================
# TARGET SELECTOR  (formerly "BallLogic")
# ==========================================

def select_targets(raw_detections: list, frame) -> tuple:
    """
    Convert raw detections to robot-relative world coordinates.

    Improvements over legacy BallLogic:
      - Class filtering driven by TARGET_CLASSES (config.json) — game-agnostic.
      - Bounding-box aspect-ratio filtering removes false positives.
      - Returns results as a named tuple-like tuple for clarity.

    Returns: (yaws_rad, dists_m, xs_m, ys_m, screen_detections, pieces_in_inches)
    """
    if not raw_detections:
        return [], [], [], [], [], []

    # --- Class filter (config-driven, game-agnostic) ---
    class_filtered = [
        d for d in raw_detections
        if model_names.get(d['class_id'], '').lower() in TARGET_CLASSES
           and d['conf'] > CONFIDENCE_FILTER
    ]

    # --- Aspect-ratio filter ---
    ar_filtered = [d for d in class_filtered if _is_valid_aspect_ratio(d['box'])]

    if not ar_filtered:
        return [], [], [], [], [], []

    # --- Vectorised coordinate calculations ---
    boxes = np.array([d['box'] for d in ar_filtered])
    cxs   = (boxes[:, 0] + boxes[:, 2]) / 2
    w_pxs = boxes[:, 2] - boxes[:, 0]

    yaws_rad_arr  = np.arctan((cxs - frame_center_x) / focal_length_px)
    dists_in_arr  = np.where(w_pxs > 0, (KNOWN_PIECE_DIAMETER_IN * focal_length_px) / w_pxs, 0)

    x_inches_arr = dists_in_arr * np.cos(yaws_rad_arr)
    y_inches_arr = -dists_in_arr * np.sin(yaws_rad_arr)

    dists_m_arr = dists_in_arr * INCHES_TO_METERS
    mask = (dists_in_arr > MIN_DIST_FROM_WALL_IN) & (dists_m_arr < MAX_REACH_METERS)

    yaws_rad = yaws_rad_arr[mask].tolist()
    dists_m  = dists_m_arr[mask].tolist()
    xs_m     = (x_inches_arr[mask] * INCHES_TO_METERS).tolist()
    ys_m     = (y_inches_arr[mask] * INCHES_TO_METERS).tolist()

    screen_detections = []
    pieces_in_inches  = []
    for i, is_valid in enumerate(mask):
        if is_valid:
            screen_detections.append({
                'box':       ar_filtered[i]['box'],
                'dist_in':   float(dists_in_arr[i]),
                'angle_deg': math.degrees(float(yaws_rad_arr[i])),
                'x_in':      float(x_inches_arr[i]),
                'y_in':      float(y_inches_arr[i]),
            })
            pieces_in_inches.append([float(x_inches_arr[i]), float(y_inches_arr[i])])

    return yaws_rad, dists_m, xs_m, ys_m, screen_detections, pieces_in_inches


def find_best_pile(pieces_in_inches: list) -> tuple:
    """
    Use DBSCAN clustering to find the highest-value pile of game pieces.

    Returns: (center_array_or_None, ball_count, score)
    """
    if not pieces_in_inches:
        return None, 0, 0.0

    coords = np.array(pieces_in_inches)
    db = DBSCAN(eps=CLUSTER_RADIUS_IN, min_samples=1).fit(coords)

    best_center = None
    max_count   = 0
    best_score  = -999.0

    for label in set(db.labels_):
        if label == -1:  # Noise
            continue

        cluster_pts = coords[db.labels_ == label]
        count  = len(cluster_pts)
        center = np.mean(cluster_pts, axis=0)

        dist  = float(np.linalg.norm(center))
        score = float(count * PILE_PRIORITY_WEIGHT) - dist

        if score > best_score:
            best_score  = score
            best_center = center
            max_count   = count

    return best_center, max_count, best_score


# ==========================================
# RADAR VISUALISATION
# ==========================================

def create_radar_frame(detections: list, best_idx: int) -> np.ndarray:
    radar    = np.zeros((RADAR_HEIGHT, RADAR_WIDTH, 3), dtype=np.uint8)
    origin_x = RADAR_WIDTH  // 2
    origin_y = RADAR_HEIGHT - 50
    cv2.rectangle(radar, (origin_x - 10, origin_y - 10), (origin_x + 10, origin_y + 10), (0, 255, 0), -1)
    for r in range(50, 250, 50):
        cv2.circle(radar, (origin_x, origin_y), int(r * GRID_SCALE), (50, 50, 50), 1)

    for i, d in enumerate(detections):
        px = int(origin_x - (d['y_in'] * GRID_SCALE))
        py = int(origin_y - (d['x_in'] * GRID_SCALE))
        color = (0, 0, 255) if i == best_idx else (0, 255, 255)
        if 0 < px < RADAR_WIDTH and 0 < py < RADAR_HEIGHT:
            cv2.circle(radar, (px, py), 8, color, -1)

    return radar


# ==========================================
# MAIN
# ==========================================

def _load_model_names_from_yaml(openvino_model_dir: str) -> dict:
    """
    Read class names from the YAML metadata file Ultralytics writes alongside
    the OpenVINO .xml at export time.  Avoids loading the full PyTorch model
    (~150 MB on the Kangaroo's 2 GB RAM) just to get class labels.

    Falls back to an empty dict if the file is missing; the caller should
    handle that by loading YOLO(MODEL_PATH) instead.
    """
    import yaml
    yaml_candidates = [
        os.path.join(openvino_model_dir, "metadata.yaml"),
        os.path.join(openvino_model_dir, "best.yaml"),
    ]
    for path in yaml_candidates:
        if os.path.exists(path):
            with open(path, "r") as f:
                meta = yaml.safe_load(f)
            # Ultralytics stores names as {0: 'classA', 1: 'classB', ...}
            names = meta.get("names", {})
            if names:
                return {int(k): v for k, v in names.items()}
    return {}


def main():
    global CAMERA_INDEX
    global model, model_names, compiled_model, infer_request
    global inference_queue, latest_detections, latest_frame_timestamp_us

    print("Starting FRC Vision Master Script (Cherry Trail Optimized)...")
    print(f"Config loaded from: {_CONFIG_PATH}")
    print(f"Target classes: {TARGET_CLASSES}")
    print(f"Using OpenVINO: {USE_OPENVINO} | GPU: {USE_GPU} | FP16: {USE_FP16}")

    # --- Model initialisation ---
    if USE_OPENVINO:
        core = Core()

        if not os.path.exists(OPENVINO_MODEL_PATH):
            print("Converting PyTorch model to OpenVINO (FP16)...")
            print(f"Target resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
            model = YOLO(MODEL_PATH)
            model.export(format='openvino', half=USE_FP16, imgsz=320)
            print("Conversion complete. Model uses 320x320 input.")

        ov_model_xml = os.path.join(OPENVINO_MODEL_PATH, 'best.xml')

        if isinstance(USE_GPU, str):
            device = USE_GPU.upper()
        elif USE_GPU:
            device = "GPU"
        else:
            device = "CPU"

        print(f"Compiling model for device: {device}")
        ov_config = {}
        if device == "GPU":
            ov_config["PERFORMANCE_HINT"] = "LATENCY"
        elif device == "CPU":
            ov_config["NUM_STREAMS"] = "1"
            ov_config["INFERENCE_NUM_THREADS"] = "2"

        compiled_model = core.compile_model(ov_model_xml, device, ov_config)
        infer_request  = compiled_model.create_infer_request()

        # Load class names from lightweight YAML — avoids a full PyTorch model
        # load (~150 MB) just to get label strings on the Kangaroo's 2 GB RAM.
        model_names = _load_model_names_from_yaml(OPENVINO_MODEL_PATH)
        if not model_names:
            print("YAML metadata not found — loading PyTorch model for class names (one-time)...")
            _tmp = YOLO(MODEL_PATH)
            model_names = _tmp.names
            del _tmp  # Free RAM immediately after extracting names
        print(f"OpenVINO model loaded on {device} | {len(model_names)} classes: {list(model_names.values())}")
    else:
        model          = YOLO(MODEL_PATH)
        original_model = model
        model_names    = model.names
        compiled_model = None
        infer_request  = None

    # --- NetworkTables ---
    inst = NetworkTableInstance.getDefault()
    if IS_SIMULATION:
        inst.setServer("127.0.0.1")
    else:
        inst.setServerTeam(TEAM_NUMBER)
    inst.startClient4(NT_CLIENT_NAME)

    sd       = inst.getTable("SmartDashboard")
    cv_table = inst.getTable("fuelCV")

    pub_num  = cv_table.getIntegerTopic("number_of_fuel").publish()
    pub_yaw  = cv_table.getDoubleArrayTopic("yaw_radians").publish()
    pub_dist = cv_table.getDoubleArrayTopic("distance").publish()
    pub_x    = cv_table.getDoubleArrayTopic("ball_position_x").publish()
    pub_y    = cv_table.getDoubleArrayTopic("ball_position_y").publish()

    # Latency-compensation timestamp  ← NEW
    # The RIO reads this value and calls drivetrain.getPoseAtTimestamp(ts)
    # to get the historical robot pose that matches this exact camera frame.
    pub_timestamp = cv_table.getIntegerTopic("frame_timestamp_us").publish()

    # Legacy single-target publishers (manual drive assist)
    pub_has_target   = sd.getBooleanTopic("Fuelcv1/HasTarget").publish()
    pub_target_angle = sd.getDoubleTopic("Fuelcv1/Angle").publish()

    # Pile / TargetSelector publishers
    pub_target_x        = cv_table.getDoubleTopic("target_x").publish()
    pub_target_y        = cv_table.getDoubleTopic("target_y").publish()
    pub_pile_has_target = cv_table.getBooleanTopic("has_target").publish()
    pub_pile_size       = cv_table.getIntegerTopic("target_pile_size").publish()
    pub_pile_score      = cv_table.getDoubleTopic("target_pile_score").publish()

    # Telemetry
    sent_angle    = 0.0
    sent_dist     = 0.0
    target_status = "SEARCHING..."

    # Async inference
    inference_queue        = Queue(maxsize=ASYNC_QUEUE_SIZE)
    latest_detections      = None
    latest_frame_timestamp_us = 0

    # Open camera with buffer optimisation for Kangaroo
    camera_indexes = [CAMERA_INDEX] + [i for i in CAMERA_INDEX_CANDIDATES if i != CAMERA_INDEX]
    cap = None
    for idx in camera_indexes:
        cap = open_camera(idx)
        if cap is not None and cap.isOpened():
            CAMERA_INDEX = idx
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"Camera opened at index {CAMERA_INDEX}")
            break

    if cap is None or not cap.isOpened():
        raise RuntimeError(f"Unable to open camera. Tried indexes: {camera_indexes}")

    # Start async inference worker thread
    inference_thread = threading.Thread(target=inference_worker, daemon=True)
    inference_thread.start()
    print("Async inference worker started")
    print(f"Clustering enabled: {ENABLE_CLUSTERING} | Intake width: {INTAKE_WIDTH_INCHES}\"")
    print(f"Wall filter: {MIN_DIST_FROM_WALL_IN}\" | Max reach: {MAX_REACH_METERS}m | Confidence: {CONFIDENCE_FILTER}")
    print(f"Inference decimation: 1/{INFERENCE_DECIMATION} | Network throttle: 1/{NT_THROTTLE_FACTOR}")
    print(f"Aspect ratio filter: [{ASPECT_RATIO_MIN}, {ASPECT_RATIO_MAX}]")

    frame_count = 0
    perf_timer  = time.time()
    # Track the timestamp of the most recently published frame (updated in the publish block)
    current_frame_timestamp_us: int = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # Queue frame for async inference (decimated to save CPU)
            if frame_count % INFERENCE_DECIMATION == 0:
                # Capture a microsecond timestamp at the exact moment of frame
                # acquisition.  _now_us() uses the NT server clock when available
                # (ntcore.now()) so the RIO can correlate it with its pose history.
                capture_time_us: int = _now_us()
                try:
                    inference_queue.put_nowait((frame.copy(), frame, capture_time_us))
                except Exception:
                    pass  # Queue full — skip this frame

            # Retrieve latest detections (from previous async frame)
            with detection_lock:
                raw_detections            = latest_detections if latest_detections is not None else []
                current_frame_timestamp_us = latest_frame_timestamp_us

            if raw_detections:
                yaws_rad, dists_m, xs_m, ys_m, screen_detections, pieces_in_inches = \
                    select_targets(raw_detections, frame)
            else:
                yaws_rad, dists_m, xs_m, ys_m, screen_detections, pieces_in_inches = \
                    [], [], [], [], [], []

            # Legacy single-target priority (cost = distance + angle penalty)
            best_idx = -1
            if screen_detections:
                costs    = [d['dist_in'] + abs(d['angle_deg']) * 0.2 for d in screen_detections]
                best_idx = int(np.argmin(costs))

            # TargetSelector — best pile
            pile_center = None
            pile_size   = 0
            pile_score  = 0.0

            if ENABLE_CLUSTERING and pieces_in_inches:
                pile_center, pile_size, pile_score = find_best_pile(pieces_in_inches)

            # --- THROTTLED PUBLISHING ---
            if frame_count % NT_THROTTLE_FACTOR == 0:
                # Latency-compensation timestamp — publish first so the RIO
                # receives it before the coordinate arrays in the same flush.
                pub_timestamp.set(current_frame_timestamp_us)

                pub_num.set(len(xs_m))
                pub_yaw.set(yaws_rad)
                pub_dist.set(dists_m)
                pub_x.set(xs_m)
                pub_y.set(ys_m)

                # Legacy single-target
                if best_idx != -1:
                    target = screen_detections[best_idx]
                    pub_has_target.set(True)
                    pub_target_angle.set(target['angle_deg'])
                    sent_angle, sent_dist = target['angle_deg'], target['dist_in']
                else:
                    pub_has_target.set(False)

                # Best pile target (for auto-collector / PathPlanner)
                if pile_center is not None:
                    pub_pile_has_target.set(True)
                    pub_target_x.set(float(pile_center[0]) * INCHES_TO_METERS)
                    pub_target_y.set(float(pile_center[1]) * INCHES_TO_METERS)
                    pub_pile_size.set(pile_size)
                    pub_pile_score.set(pile_score)
                    target_status = f"PILE:{pile_size}"
                    pile_dist  = math.sqrt(pile_center[0] ** 2 + pile_center[1] ** 2)
                    pile_angle = math.degrees(math.atan2(pile_center[1], pile_center[0]))
                    sent_angle, sent_dist = pile_angle, pile_dist
                else:
                    pub_pile_has_target.set(False)
                    target_status = "SEARCHING..."

            # --- HUD & UI (skipped in headless mode to save CPU) ---
            if not HEADLESS:
                cv2.rectangle(frame, (0, 0), (280, 150), (0, 0, 0), -1)
                cv2.putText(frame, f"NT4 {'CONNECTED' if inst.isConnected() else 'OFFLINE'}", (10, 20), 0, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"STATUS: {target_status}", (10, 40), 0, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"ANGLE : {sent_angle:.1f} deg", (10, 60), 0, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"DIST  : {sent_dist:.1f} in", (10, 80), 0, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"PIECES: {len(xs_m)} detected", (10, 100), 0, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, f"SCORE : {pile_score:.0f}", (10, 120), 0, 0.5, (255, 0, 255), 1)
                cv2.putText(frame, f"TS_US : {current_frame_timestamp_us}", (10, 140), 0, 0.35, (100, 200, 255), 1)

                if USE_OPENVINO:
                    dev_name  = USE_GPU.upper() if isinstance(USE_GPU, str) else ("GPU" if USE_GPU else "CPU")
                    mode_text = f"OpenVINO+{dev_name}"
                else:
                    mode_text = "PyTorch"

                clustering_text = "PILE" if ENABLE_CLUSTERING else "SINGLE"
                cv2.putText(frame, f"MODE: {mode_text}|{clustering_text}", (10, 155), 0, 0.45, (0, 255, 255), 1)

                for i, d in enumerate(screen_detections):
                    color = (0, 0, 255) if i == best_idx else (0, 255, 0)
                    cv2.rectangle(frame, d['box'][0:2], d['box'][2:4], color, 2)

                if pile_center is not None and pile_size > 1:
                    for d in screen_detections:
                        dx = d['x_in'] - pile_center[0]
                        dy = d['y_in'] - pile_center[1]
                        if math.sqrt(dx * dx + dy * dy) < CLUSTER_RADIUS_IN / 2:
                            x1, y1 = d['box'][0], d['box'][1]
                            cv2.putText(frame, f"x{pile_size}", (x1, y1 - 5), 0, 0.6, (255, 0, 255), 2)
                            break

                cv2.imshow('Robot Camera (Cherry Trail Optimized)', frame)
                cv2.imshow('2D Radar Map', create_radar_frame(screen_detections, best_idx))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - perf_timer
                fps     = 30 / elapsed if elapsed > 0 else 0
                print(f"FPS: {fps:.1f} | Pieces: {len(xs_m)} | Queue: {inference_queue.qsize()} | TS: {current_frame_timestamp_us}")
                perf_timer = time.time()

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        inference_queue.put(None)
        inference_thread.join(timeout=2)
        cap.release()
        if not HEADLESS:
            cv2.destroyAllWindows()
        print("Vision system shutdown complete")


if __name__ == "__main__":
    main()
