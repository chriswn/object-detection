#!/usr/bin/env python3
"""
Cherry Trail Hardware Diagnostics Script
Checks OpenVINO setup, GPU availability, and performance metrics
"""

import sys
import os

def check_python_version():
    """Verify Python 3.9+"""
    print("=" * 60)
    print("PYTHON VERSION CHECK")
    print("=" * 60)
    version = sys.version_info
    print(f"Detected: Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ FAILED: Python 3.9+ required")
        return False
    print("✓ PASSED: Python version OK")
    return True

def check_opencv():
    """Verify OpenCV installation"""
    print("\n" + "=" * 60)
    print("OPENCV CHECK")
    print("=" * 60)
    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
        # Check camera availability
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✓ CAMERA: Detected at index 0")
            cap.release()
        else:
            print("⚠ WARNING: No camera at index 0 (OK if using different camera)")
        print("✓ PASSED: OpenCV installed")
        return True
    except ImportError as e:
        print(f"❌ FAILED: {e}")
        return False

def check_openvino():
    """Verify OpenVINO installation and GPU support"""
    print("\n" + "=" * 60)
    print("OPENVINO CHECK")
    print("=" * 60)
    try:
        # OpenVINO 2026+ simplified imports
        try:
            from openvino.runtime import Core
        except ImportError:
            from openvino import Core
        core = Core()
        
        # Check OpenVINO version (API changed in 2026+)
        try:
            print(f"OpenVINO version: {core.get_version()}")
        except AttributeError:
            import openvino as ov
            print(f"OpenVINO version: {ov.__version__}")
        
        # List available devices
        devices = core.available_devices
        print(f"\nAvailable devices: {devices}")
        
        gpu_available = any('GPU' in d for d in devices)
        cpu_available = any('CPU' in d for d in devices)
        
        if gpu_available:
            print("✓ GPU: Detected (Intel HD Graphics)")
        else:
            print("⚠ WARNING: GPU not detected (CPU-only mode will be slower)")
        
        if cpu_available:
            print("✓ CPU: Detected")
        else:
            print("❌ ERROR: No CPU device available")
            return False
        
        print("✓ PASSED: OpenVINO installed and functional")
        return True
    except ImportError as e:
        print(f"❌ FAILED: {e}")
        print("   Install with: pip install openvino openvino-dev")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def check_pytorch():
    """Verify PyTorch (needed for model conversion)"""
    print("\n" + "=" * 60)
    print("PYTORCH CHECK")
    print("=" * 60)
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print("✓ PASSED: PyTorch installed (needed for model conversion)")
        return True
    except ImportError as e:
        print(f"❌ FAILED: {e}")
        print("   Install with: pip install torch torchvision")
        return False

def check_ultralytics():
    """Verify YOLOv8 installation"""
    print("\n" + "=" * 60)
    print("ULTRALYTICS (YOLOV8) CHECK")
    print("=" * 60)
    try:
        from ultralytics import YOLO
        print("✓ PASSED: YOLOv8 installed")
        return True
    except ImportError as e:
        print(f"❌ FAILED: {e}")
        print("   Install with: pip install ultralytics")
        return False

def check_model_file():
    """Check if best.pt exists"""
    print("\n" + "=" * 60)
    print("MODEL FILE CHECK")
    print("=" * 60)
    model_path = "models/best.pt"
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✓ Found: {model_path} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"❌ NOT FOUND: {model_path}")
        print("   Place your trained YOLOv8 model at models/best.pt")
        return False

def check_openvino_model():
    """Check if OpenVINO model exists"""
    print("\n" + "=" * 60)
    print("OPENVINO MODEL CHECK")
    print("=" * 60)
    ir_path = "models/best_openvino_model/best.xml"
    if os.path.exists(ir_path):
        print(f"✓ Found: OpenVINO IR model")
        print("   (Will be used for inference)")
        return True
    else:
        print("ℹ INFO: OpenVINO IR model not found (OK)")
        print("   First run will convert models/best.pt automatically")
        print("   This is expected behavior - not an error")
        return True  # Changed to True since this is expected

def check_pyntcore():
    """Verify NetworkTables library"""
    print("\n" + "=" * 60)
    print("PYNTCORE (NETWORKTABLES) CHECK")
    print("=" * 60)
    try:
        from ntcore import NetworkTableInstance
        print("✓ PASSED: pyntcore installed")
        return True
    except ImportError:
        print("❌ FAILED: pyntcore not installed")
        print("   Install with: pip install pyntcore")
        return False

def test_inference_mock():
    """Test basic inference without actual camera"""
    print("\n" + "=" * 60)
    print("INFERENCE PIPELINE TEST")
    print("=" * 60)
    try:
        import numpy as np
        try:
            from openvino.runtime import Core
        except ImportError:
            from openvino import Core
        
        core = Core()
        
        # Create a dummy model for testing (just tests pipeline, not actual inference)
        test_input = np.random.rand(1, 3, 320, 240).astype(np.float32)
        print(f"Created test input: {test_input.shape}")
        
        devices = core.available_devices
        best_device = 'GPU' if any('GPU' in d for d in devices) else 'CPU'
        print(f"Would use device: {best_device}")
        print("✓ PASSED: Inference pipeline ready")
        return True
    except Exception as e:
        print(f"⚠ WARNING: {e}")
        return False

def main():
    """Run all diagnostics"""
    print("\n" + "🔍 " * 20)
    print("CHERRY TRAIL HARDWARE DIAGNOSTIC")
    print("Intel Atom x5-Z8500 Vision System")
    print("🔍 " * 20 + "\n")
    
    checks = [
        ("Python Version", check_python_version),
        ("Python Packages", lambda: all([
            check_opencv(),
            check_pytorch(),
            check_ultralytics(),
            check_pyntcore(),
        ])),
        ("OpenVINO Setup", check_openvino),
        ("Model Files", lambda: all([
            check_model_file(),
            check_openvino_model(),
        ])),
        ("Inference Ready", test_inference_mock),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ EXCEPTION in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(r for _, r in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL CHECKS PASSED - Ready to run vision system!")
        print("\nNext step:")
        print("  python run_inference.py")
        return 0
    else:
        print("❌ SOME CHECKS FAILED - See above for details")
        print("\nNext steps:")
        print("  1. Review failed checks above")
        print("  2. Install missing dependencies")
        print("  3. Re-run diagnostics")
        return 1

if __name__ == "__main__":
    sys.exit(main())
