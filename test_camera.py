import cv2
import sys


def open_camera(index):
    backend_candidates = []
    if sys.platform.startswith("linux"):
        backend_candidates = [cv2.CAP_V4L2, cv2.CAP_ANY]
    elif sys.platform.startswith("win"):
        backend_candidates = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        backend_candidates = [cv2.CAP_ANY]

    for backend in backend_candidates:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            return cap, backend
        cap.release()

    return None, None

print("=" * 50)
print("Camera Test Script Starting...")
print(f"OpenCV Version: {cv2.__version__}")
print("=" * 50)

def test_camera_index(index):
    print(f"\n📹 Testing Camera Index {index}...")
    
    try:
        cap, backend = open_camera(index)
        
        if cap is None:
            print(f" Could not open camera at index {index}.")
            return False
        
        print(f"✓ Camera device opened at index {index} (backend={backend})")
        
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f" Camera opened at index {index}, but could not read a frame.")
            cap.release()
            return False
        
        print(f" Successfully read frame: {frame.shape}")
        print(f" SUCCESS! Camera working at index {index}.")
        print("Press 'q' in the window to close the test.")
        print("Opening camera window...")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Lost camera connection")
                break
            
            # Add frame counter
            frame_count += 1
            cv2.putText(frame, f"Frame: {frame_count} | Press 'q' to quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(f"Camera Test - Index {index}", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                print("\nClosing camera...")
                break
                
        cap.release()
        cv2.destroyAllWindows()
        print("Camera closed successfully")
        return True
        
    except Exception as e:
        print(f" Error testing camera {index}: {e}")
        import traceback
        traceback.print_exc()
        return False

# Main execution
print("\nStarting camera detection...")

# Try multiple camera indices
found = False
for i in range(3):  # Try indices 0, 1, 2
    if test_camera_index(i):
        found = True
        break
    
if not found:
    print("\n" + "=" * 50)
    print(" No working camera found!")
    print("=" * 50)
    print("\nTroubleshooting tips:")
    print("1. Make sure your camera is physically connected")
    print("2. Check if any other app is using the camera")
    print("3. Try unplugging and replugging your camera")
    print("4. Check Windows Camera app to verify camera works")
else:
    print("\n" + "=" * 50)
    print(" Camera test completed successfully!")
    print("=" * 50)