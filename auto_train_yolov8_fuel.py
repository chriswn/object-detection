import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
import math
from networktables import NetworkTables
import pygame

# --- Configuration ---
MODEL_PATH = 'models/best.pt'
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.45

# --- Robot/Camera Constants ---
CAMERA_HFOV_DEG = 60.0
KNOWN_FUEL_DIAMETER_IN = 5.91
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# --- Dashboard Settings ---
DASH_WIDTH = 1200
DASH_HEIGHT = 600
BG_COLOR = (30, 30, 30)
TEXT_COLOR = (255, 255, 255)
GRID_COLOR = (50, 50, 50)

# --- Initialization ---
print("Initializing System...")
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
focal_length_px = FRAME_WIDTH / (2 * math.tan(math.radians(CAMERA_HFOV_DEG / 2)))

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((DASH_WIDTH, DASH_HEIGHT))
pygame.display.set_caption("FRC Fuel Detection Dashboard")
clock = pygame.time.Clock()
font_title = pygame.font.SysFont("Arial", 24, bold=True)
font_text = pygame.font.SysFont("Arial", 18)

def draw_grid(surface, rect, scale=2.0):
    """Draws a grid on a specific area of the screen"""
    # Draw background for this panel
    pygame.draw.rect(surface, (0, 0, 0), rect)
    pygame.draw.rect(surface, (100, 100, 100), rect, 2) # Border
    
    # Grid lines (every 10 inches)
    center_x = rect.centerx
    bottom_y = rect.bottom - 20
    
    # Vertical lines
    for i in range(-10, 11):
        x = center_x + (i * 10 * scale)
        if rect.left < x < rect.right:
            pygame.draw.line(surface, GRID_COLOR, (x, rect.top), (x, rect.bottom))
            
    # Horizontal lines
    for i in range(0, 20):
        y = bottom_y - (i * 10 * scale)
        if rect.top < y < rect.bottom:
            pygame.draw.line(surface, GRID_COLOR, (rect.left, y), (rect.right, y))

    # Draw Robot
    robot_poly = [
        (center_x, bottom_y),
        (center_x - 15, bottom_y + 30),
        (center_x + 15, bottom_y + 30)
    ]
    pygame.draw.polygon(surface, (0, 255, 0), robot_poly)
    
    return center_x, bottom_y

def cvimage_to_pygame(image):
    """Converts OpenCV BGR image to Pygame RGB surface"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return pygame.image.frombuffer(image_rgb.tostring(), image_rgb.shape[1::-1], "RGB")

# --- Main Loop ---
running = True
while running:
    # 1. Handle Pygame Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            running = False

    # 2. Capture & Process Frame
    ret, frame = cap.read()
    if not ret: break

    results = model(frame, verbose=False)
    
    # Store detections: [x_in, y_in, angle_deg, dist_total, bbox_coords]
    detections = []
    
    # YOLO Processing
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            name = model.names[cls]

            if conf > CONFIDENCE_THRESHOLD and name.lower() in ['fuel', 'fuels', 'ball']:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                width = x2 - x1
                offset = cx - (FRAME_WIDTH / 2)
                angle_rad = math.atan(offset / focal_length_px)
                angle_deg = math.degrees(angle_rad)
                dist_total = (KNOWN_FUEL_DIAMETER_IN * focal_length_px) / width if width > 0 else 0
                y_in = dist_total * math.cos(angle_rad)
                x_in = dist_total * math.sin(angle_rad)
                detections.append([x_in, y_in, angle_deg, dist_total, (x1, y1, x2, y2)])

    # Determine Priority
    best_idx = -1
    lowest_cost = 9999.0
    for i, det in enumerate(detections):
        cost = det[3] + (abs(det[2]) * 0.1)
        if cost < lowest_cost:
            lowest_cost = cost
            best_idx = i

    # Draw boxes on Camera Frame
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det[4]
        color = (0, 0, 255) if i == best_idx else (0, 255, 0) # Red or Green (BGR for OpenCV)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{det[3]:.1f}in", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # --- 3. Draw Dashboard ---
    screen.fill(BG_COLOR)

    # Panel 1: Camera Feed (Left)
    cam_surface = cvimage_to_pygame(frame)
    # Scale camera feed to fit half screen
    cam_surface = pygame.transform.scale(cam_surface, (DASH_WIDTH // 2 - 20, int(FRAME_HEIGHT * 0.8)))
    screen.blit(cam_surface, (10, 50))
    
    # Header
    title = font_title.render("FRC 2026 Fuel Detection System", True, (255, 255, 0))
    screen.blit(title, (DASH_WIDTH//2 - title.get_width()//2, 10))

    # Panel 2: 2D Radar / Overhead View (Right)
    radar_rect = pygame.Rect(DASH_WIDTH // 2 + 10, 50, DASH_WIDTH // 2 - 20, int(FRAME_HEIGHT * 0.8))
    origin_x, origin_y = draw_grid(screen, radar_rect, scale=3.0)
    
    # Title for Radar
    radar_title = font_text.render("Overhead View (Robot-Relative)", True, TEXT_COLOR)
    screen.blit(radar_title, (radar_rect.x, radar_rect.y - 25))

    # Draw Balls on Radar
    for i, det in enumerate(detections):
        # det[0] is X (Left/Right), det[1] is Y (Forward)
        # Map to screen pixels (scale=3.0)
        px = origin_x + (det[0] * 3.0)
        py = origin_y - (det[1] * 3.0)
        
        # Color: Red if priority, Yellow otherwise
        color = (255, 0, 0) if i == best_idx else (255, 255, 0)
        
        # Draw dot
        pygame.draw.circle(screen, color, (int(px), int(py)), 8)
        
        # Draw line to priority
        if i == best_idx:
            pygame.draw.line(screen, (255, 0, 0), (origin_x, origin_y), (int(px), int(py)), 2)
            
            # Display stats for priority target
            stats_text = f"TARGET LOCKED: Dist: {det[3]:.1f}in  Angle: {det[2]:.1f} deg"
            stats_surf = font_text.render(stats_text, True, (0, 255, 0))
            screen.blit(stats_surf, (DASH_WIDTH // 2 + 20, radar_rect.bottom + 10))

    pygame.display.flip()
    clock.tick(30) # Limit to 30 FPS

cap.release()
pygame.quit()