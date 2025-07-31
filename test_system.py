#!/usr/bin/env python3
"""
Test script for Medical Emergency Detection System
Tests the system without requiring camera or trained model
"""

import cv2
import numpy as np
import time
import os
from pathlib import Path
from detection_system import MedicalEmergencyDetector

def create_test_image():
    """Create a test image for detection"""
    # Create a simple test image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some shapes that might trigger detections
    # Draw a "person" shape (simplified)
    cv2.rectangle(img, (200, 150), (300, 400), (255, 255, 255), -1)  # Body
    cv2.circle(img, (250, 100), 50, (255, 255, 255), -1)  # Head
    
    # Add some text
    cv2.putText(img, "Test Person", (180, 450), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return img

def test_detection_system():
    """Test the detection system with a mock image"""
    
    print("🧪 Testing Medical Emergency Detection System")
    print("=" * 50)
    
    # Create test directories
    Path("models").mkdir(exist_ok=True)
    Path("detections").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Create a dummy model file if it doesn't exist
    model_path = Path("models/medical_emergency.pt")
    if not model_path.exists():
        print("⚠️  No trained model found. Using YOLOv8n as fallback.")
        print("   This will detect general objects, not medical emergencies.")
        print("   Train a custom model for medical emergency detection.")
    
    try:
        # Initialize detector
        print("📡 Initializing detection system...")
        detector = MedicalEmergencyDetector(
            model_path="models/medical_emergency.pt",
            confidence_threshold=0.3,  # Lower threshold for testing
            detection_interval=0.1
        )
        
        # Create test image
        print("🖼️  Creating test image...")
        test_img = create_test_image()
        
        # Save test image
        cv2.imwrite("test_image.jpg", test_img)
        print("💾 Test image saved as 'test_image.jpg'")
        
        # Process the test image
        print("🔍 Processing test image for detections...")
        detections = detector.process_frame(test_img)
        
        if detections:
            print(f"✅ Found {len(detections)} detection(s)!")
            for i, detection in enumerate(detections):
                print(f"   Detection {i+1}:")
                print(f"     Label: {detection.label}")
                print(f"     Confidence: {detection.confidence:.2f}")
                print(f"     BBox: {detection.bbox}")
                print(f"     Image saved: {detection.image_path}")
        else:
            print("ℹ️  No detections found in test image.")
            print("   This is normal if using a general-purpose model.")
        
        # Test statistics
        print("\n📊 Testing statistics...")
        stats = detector.get_detection_stats()
        print(f"   Total detections: {stats.get('total_detections', 0)}")
        print(f"   Detection counts: {stats.get('detection_counts', {})}")
        
        # Test CSV logging
        csv_path = Path("logs/detections.csv")
        if csv_path.exists():
            print(f"✅ Detection logs saved to: {csv_path}")
        else:
            print("ℹ️  No detection logs created (no detections)")
        
        print("\n🎉 System test completed successfully!")
        print("\nNext steps:")
        print("1. Train a custom model for medical emergency detection")
        print("2. Connect a camera for real-time testing")
        print("3. Run the web interface: python web_interface.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check if all dependencies are installed")
        print("2. Ensure you have write permissions in the current directory")
        print("3. Try installing ultralytics: pip install ultralytics")
        return False

def test_web_interface():
    """Test if the web interface can start"""
    
    print("\n🌐 Testing Web Interface...")
    print("=" * 30)
    
    try:
        # Test if Flask and SocketIO can be imported
        from flask import Flask
        from flask_socketio import SocketIO
        
        print("✅ Flask and SocketIO imports successful")
        
        # Test if the template exists
        template_path = Path("templates/index.html")
        if template_path.exists():
            print("✅ HTML template found")
        else:
            print("❌ HTML template not found")
            return False
        
        # Test if the JavaScript file exists
        js_path = Path("static/js/app.js")
        if js_path.exists():
            print("✅ JavaScript file found")
        else:
            print("❌ JavaScript file not found")
            return False
        
        print("✅ Web interface components are ready")
        print("   Run 'python web_interface.py' to start the web server")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Install missing dependencies: pip install flask flask-socketio")
        return False
    except Exception as e:
        print(f"❌ Web interface test failed: {e}")
        return False

def main():
    """Run all tests"""
    
    print("🚁 Medical Emergency Detection System - Test Suite")
    print("=" * 60)
    
    # Test detection system
    detection_success = test_detection_system()
    
    # Test web interface
    web_success = test_web_interface()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 20)
    print(f"Detection System: {'✅ PASS' if detection_success else '❌ FAIL'}")
    print(f"Web Interface:   {'✅ PASS' if web_success else '❌ FAIL'}")
    
    if detection_success and web_success:
        print("\n🎉 All tests passed! The system is ready to use.")
        print("\nTo get started:")
        print("1. Train a medical emergency model or use a pre-trained one")
        print("2. Connect a camera")
        print("3. Run: python web_interface.py")
        print("4. Open http://localhost:5000 in your browser")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
        print("   Install missing dependencies or fix configuration issues.")

if __name__ == "__main__":
    main() 