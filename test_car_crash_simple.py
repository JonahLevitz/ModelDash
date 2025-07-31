#!/usr/bin/env python3
"""
Simplified Car Crash Detection Test
Tests car crash detection logic without requiring PyTorch
"""

import cv2
import numpy as np
import time
import datetime
from pathlib import Path

class SimpleCarCrashDetector:
    """Simplified car crash detection for testing"""
    
    def __init__(self):
        """Initialize the simple detector"""
        self.confidence_threshold = 0.5
        self.process_interval = 2.0
        self.last_process_time = 0
        
        # Create detections directory
        self.detections_dir = Path("detections")
        self.detections_dir.mkdir(exist_ok=True)
        
        print("🚗 Simple Car Crash Detection Test")
        print("=" * 50)
        print("🔍 Testing Car Crash Detection Logic:")
        print("   🚗 Multiple vehicles close together")
        print("   🚗 Single overturned/damaged vehicles")
        print("   🚗 Vehicle aspect ratio analysis")
        print("Press 'q' to quit, 's' to save frame, 'p' to scan now")
    
    def detect_car_crash(self, frame, mock_detections):
        """Detect car crashes using mock vehicle detections"""
        vehicles = []
        
        # Simulate vehicle detections for testing
        # In a real scenario, these would come from YOLO model
        for detection in mock_detections:
            if detection['class'] in ['car', 'truck', 'bus'] and detection['confidence'] > 0.6:
                vehicles.append(detection)
        
        # Debug: Print detected vehicles
        if vehicles:
            print(f"🚗 Detected {len(vehicles)} vehicles: {[v['class'] for v in vehicles]}")
        
        # If multiple vehicles close together, might be crash
        if len(vehicles) >= 2:
            # Check if vehicles are close to each other
            for i, vehicle1 in enumerate(vehicles):
                for j, vehicle2 in enumerate(vehicles[i+1:], i+1):
                    bbox1 = vehicle1['bbox']
                    bbox2 = vehicle2['bbox']
                    
                    # Calculate distance between vehicle centers
                    center1 = ((bbox1[0] + bbox1[2])/2, (bbox1[1] + bbox1[3])/2)
                    center2 = ((bbox2[0] + bbox2[2])/2, (bbox2[1] + bbox2[3])/2)
                    
                    distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                    
                    print(f"📏 Distance between vehicles: {distance:.1f}px")
                    
                    # If vehicles are very close, might be crash
                    if distance < 150:  # Close vehicles
                        return {
                            'type': 'car_crash',
                            'confidence': max(vehicle1['confidence'], vehicle2['confidence']),
                            'bbox': bbox1,  # Use first vehicle bbox
                            'reason': f'Multiple vehicles in close proximity ({distance:.1f}px apart)'
                        }
        
        # Check for single vehicle crashes (overturned, damaged)
        if len(vehicles) == 1:
            vehicle = vehicles[0]
            bbox = vehicle['bbox']
            x1, y1, x2, y2 = bbox
            
            # Calculate aspect ratio to detect overturned vehicles
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height if height > 0 else 0
            
            print(f"📐 Vehicle aspect ratio: {aspect_ratio:.2f} (width/height)")
            
            # If vehicle is very tall relative to width, might be overturned
            if aspect_ratio < 0.8:  # Tall and narrow vehicle
                return {
                    'type': 'car_crash',
                    'confidence': vehicle['confidence'],
                    'bbox': bbox,
                    'reason': f'Vehicle appears overturned (aspect ratio: {aspect_ratio:.2f})'
                }
        
        return None
    
    def create_mock_detections(self, frame):
        """Create mock vehicle detections for testing"""
        height, width = frame.shape[:2]
        
        # Create some mock vehicle detections
        mock_detections = []
        
        # Mock detection 1: Car in top-left area
        car1 = {
            'class': 'car',
            'confidence': 0.85,
            'bbox': [50, 100, 200, 250]  # Normal car aspect ratio
        }
        mock_detections.append(car1)
        
        # Mock detection 2: Car close to first car (simulating crash)
        car2 = {
            'class': 'car',
            'confidence': 0.78,
            'bbox': [180, 120, 330, 270]  # Close to first car
        }
        mock_detections.append(car2)
        
        # Mock detection 3: Overturned car (tall aspect ratio)
        overturned_car = {
            'class': 'car',
            'confidence': 0.72,
            'bbox': [400, 150, 450, 300]  # Tall and narrow
        }
        mock_detections.append(overturned_car)
        
        return mock_detections
    
    def draw_mock_detections(self, frame, detections):
        """Draw mock vehicle detections on frame"""
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = map(int, detection['bbox'])
            label = detection['class']
            confidence = detection['confidence']
            
            # Draw bounding box
            color = (0, 255, 0)  # Green for vehicles
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label_text = f"{label}: {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1-25), (x1+len(label_text)*10, y1), color, -1)
            cv2.putText(frame, label_text, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw vehicle number
            cv2.putText(frame, f"#{i+1}", (x1, y2+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def draw_crash_detection(self, frame, crash_detection):
        """Draw car crash detection on frame"""
        if crash_detection:
            x1, y1, x2, y2 = map(int, crash_detection['bbox'])
            label = crash_detection['type']
            confidence = crash_detection['confidence']
            reason = crash_detection.get('reason', '')
            
            # Draw red bounding box for crash
            color = (0, 0, 255)  # Red for crash
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw crash label
            label_text = f"🚨 {label}: {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1-30), (x1+len(label_text)*12, y1), color, -1)
            cv2.putText(frame, label_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw reason
            if reason:
                cv2.putText(frame, reason, (x1, y2+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def save_detection(self, frame, detection):
        """Save car crash detection image"""
        timestamp = datetime.datetime.now().isoformat().replace(':', '-').replace('.', '-')
        filename = f"CAR_CRASH_{detection['type']}_{timestamp}_{detection['confidence']:.2f}.jpg"
        filepath = self.detections_dir / filename
        
        cv2.imwrite(str(filepath), frame)
        print(f"💾 Car crash detection saved: {filename}")
    
    def run_test(self):
        """Run the car crash detection test"""
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        print("📹 Webcam started successfully")
        print("🚗 Testing car crash detection logic...")
        print("🔍 Using mock vehicle detections for testing...")
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("❌ Error: Could not read frame from webcam")
                    break
                
                current_time = time.time()
                
                # Only process every 2 seconds
                if current_time - self.last_process_time >= self.process_interval:
                    # Create mock detections
                    mock_detections = self.create_mock_detections(frame)
                    
                    # Test car crash detection
                    crash_detection = self.detect_car_crash(frame, mock_detections)
                    
                    if crash_detection:
                        print(f"🚨 CAR CRASH DETECTED: {crash_detection['reason']}")
                        self.save_detection(frame, crash_detection)
                    else:
                        print("✅ No car crash detected")
                    
                    self.last_process_time = current_time
                
                # Draw mock detections
                mock_detections = self.create_mock_detections(frame)
                self.draw_mock_detections(frame, mock_detections)
                
                # Test and draw crash detection
                crash_detection = self.detect_car_crash(frame, mock_detections)
                self.draw_crash_detection(frame, crash_detection)
                
                # Draw instructions
                cv2.putText(frame, "Press 'q' to quit, 's' to save, 'p' to scan now", 
                           (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display frame
                cv2.imshow('Car Crash Detection Test', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("🛑 Quitting...")
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"car_crash_test_{timestamp}.jpg"
                    filepath = self.detections_dir / filename
                    cv2.imwrite(str(filepath), frame)
                    print(f"💾 Frame saved: {filename}")
                elif key == ord('p'):
                    # Force process now
                    self.last_process_time = 0
                    print("🔍 Forcing immediate car crash scan...")
        
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Car crash detection test completed")

def main():
    """Main function"""
    detector = SimpleCarCrashDetector()
    detector.run_test()

if __name__ == "__main__":
    main() 