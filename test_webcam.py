#!/usr/bin/env python3
"""
Webcam Test for Emergency Detection System
Tests the emergency detection model using your webcam
"""

import cv2
import numpy as np
import time
import datetime
from pathlib import Path
import json
import random

class WebcamEmergencyDetector:
    """Simple webcam-based emergency detection system for testing"""
    
    def __init__(self):
        """Initialize the webcam detector"""
        self.emergency_classes = [
            'car_crash', 'accident_victim', 'fire', 'smoke', 
            'person_fainted', 'person_down', 'person_injured', 
            'medical_emergency', 'emergency_vehicle'
        ]
        
        # Emergency class colors
        self.emergency_colors = {
            'car_crash': (0, 0, 255),        # Red
            'accident_victim': (0, 165, 255), # Orange
            'fire': (0, 0, 255),             # Red
            'smoke': (128, 128, 128),        # Gray
            'person_fainted': (255, 0, 255), # Purple
            'person_down': (0, 0, 255),      # Red
            'person_injured': (0, 165, 255), # Orange
            'medical_emergency': (0, 255, 255), # Yellow
            'emergency_vehicle': (255, 255, 0)  # Cyan
        }
        
        # Detection history
        self.detection_history = []
        self.last_detection_time = 0
        
        # Create detections directory
        self.detections_dir = Path("detections")
        self.detections_dir.mkdir(exist_ok=True)
        
        print("🚁 Emergency Detection Webcam Test")
        print("=" * 50)
        print("Detecting: Car Crashes, Fires, Medical Emergencies")
        print("Press 'q' to quit, 's' to save frame, 'd' to simulate detection")
    
    def simulate_detection(self, frame):
        """Simulate emergency detection for demonstration"""
        current_time = time.time()
        
        # Simulate detection every 3-5 seconds
        if current_time - self.last_detection_time > random.uniform(3, 5):
            # Randomly select an emergency type
            emergency_type = random.choice(self.emergency_classes)
            confidence = random.uniform(0.6, 0.95)
            
            # Create random bounding box
            h, w = frame.shape[:2]
            x1 = random.randint(50, w-200)
            y1 = random.randint(50, h-200)
            x2 = x1 + random.randint(100, 200)
            y2 = y1 + random.randint(100, 200)
            
            # Draw detection
            color = self.emergency_colors.get(emergency_type, (0, 255, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw label
            label_text = f"{emergency_type}: {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1-30), (x1+len(label_text)*12, y1), color, -1)
            cv2.putText(frame, label_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add to history
            detection = {
                'timestamp': datetime.datetime.now().isoformat(),
                'type': emergency_type,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2]
            }
            self.detection_history.append(detection)
            
            # Save detection image
            self.save_detection(frame, detection)
            
            self.last_detection_time = current_time
            
            print(f"🚨 EMERGENCY DETECTED: {emergency_type} (confidence: {confidence:.2f})")
    
    def save_detection(self, frame, detection):
        """Save detection image"""
        timestamp = detection['timestamp'].replace(':', '-').replace('.', '-')
        filename = f"detection_{timestamp}_{detection['type']}_{detection['confidence']:.2f}.jpg"
        filepath = self.detections_dir / filename
        
        cv2.imwrite(str(filepath), frame)
        print(f"💾 Detection saved: {filename}")
    
    def draw_stats(self, frame):
        """Draw detection statistics on frame"""
        # Draw detection count
        detection_count = len(self.detection_history)
        cv2.putText(frame, f"Detections: {detection_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw recent detections
        recent_detections = self.detection_history[-3:]  # Last 3 detections
        y_offset = 60
        for i, detection in enumerate(recent_detections):
            text = f"{detection['type']}: {detection['confidence']:.2f}"
            cv2.putText(frame, text, (10, y_offset + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw instructions
        cv2.putText(frame, "Press 'q' to quit, 's' to save, 'd' to simulate", 
                   (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run_webcam_test(self):
        """Run the webcam test"""
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        print("📹 Webcam started successfully")
        print("🎯 Emergency detection system is running...")
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("❌ Error: Could not read frame from webcam")
                    break
                
                # Simulate detection
                self.simulate_detection(frame)
                
                # Draw statistics
                self.draw_stats(frame)
                
                # Display frame
                cv2.imshow('Emergency Detection System - Webcam Test', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("🛑 Quitting...")
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"webcam_frame_{timestamp}.jpg"
                    filepath = self.detections_dir / filename
                    cv2.imwrite(str(filepath), frame)
                    print(f"💾 Frame saved: {filename}")
                elif key == ord('d'):
                    # Force a detection
                    self.last_detection_time = 0
                    print("🔍 Forcing detection simulation...")
        
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            print(f"\n📊 Test Statistics:")
            print(f"   Total detections: {len(self.detection_history)}")
            
            if self.detection_history:
                print(f"   Detection types:")
                type_counts = {}
                for detection in self.detection_history:
                    emergency_type = detection['type']
                    type_counts[emergency_type] = type_counts.get(emergency_type, 0) + 1
                
                for emergency_type, count in type_counts.items():
                    print(f"     {emergency_type}: {count}")
            
            print(f"   Detection images saved to: {self.detections_dir}")

def main():
    """Main function"""
    detector = WebcamEmergencyDetector()
    detector.run_webcam_test()

if __name__ == "__main__":
    main() 