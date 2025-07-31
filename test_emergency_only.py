#!/usr/bin/env python3
"""
Emergency-Only Detection System
Only detects: Passed out people, Car crashes, Fires
Uses the trained emergency detection model
"""

import cv2
import numpy as np
import time
import datetime
from pathlib import Path
import json
from ultralytics import YOLO

class EmergencyOnlyDetector:
    """Emergency-only detection system"""
    
    def __init__(self, model_path: str = "models/emergency_detection.pt"):
        """Initialize the emergency-only detector"""
        self.model_path = model_path
        self.confidence_threshold = 0.6  # Higher threshold for accuracy
        self.process_interval = 2.0  # Process every 2 seconds
        self.last_process_time = 0
        
        # Only the 3 emergency types you need
        self.emergency_classes = [
            'car_crash',      # Car crashes
            'person_fainted', # Passed out people  
            'fire'           # Fires
        ]
        
        # Emergency class colors
        self.emergency_colors = {
            'car_crash': (0, 0, 255),        # Red
            'person_fainted': (255, 0, 255), # Purple
            'fire': (0, 0, 255),             # Red
        }
        
        # Detection history
        self.detection_history = []
        self.current_detections = []
        
        # Create detections directory
        self.detections_dir = Path("detections")
        self.detections_dir.mkdir(exist_ok=True)
        
        # Load the trained emergency model
        print("🤖 Loading trained emergency detection model...")
        try:
            self.model = YOLO(model_path)
            print(f"✅ Emergency model loaded successfully from {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        print("🚁 Emergency-Only Detection System")
        print("=" * 50)
        print("🔍 ONLY Detecting:")
        print("   🚑 Passed out people")
        print("   🚗 Car crashes") 
        print("   🔥 Fires")
        print("Press 'q' to quit, 's' to save frame, 'p' to process now")
    
    def process_frame(self, frame):
        """Process frame with emergency-only model"""
        current_time = time.time()
        
        # Only process every 2 seconds
        if current_time - self.last_process_time < self.process_interval:
            return self.current_detections
        
        try:
            # Run inference with trained emergency model
            results = self.model(frame, verbose=False)
            
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    
                    for box in boxes:
                        confidence = float(box.conf[0])
                        
                        if confidence >= self.confidence_threshold:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            bbox = [float(x1), float(y1), float(x2), float(y2)]
                            
                            # Get class label
                            class_id = int(box.cls[0])
                            if class_id < len(self.model.names):
                                label = self.model.names[class_id]
                            else:
                                label = f"class_{class_id}"
                            
                            # Only process if it's one of our 3 emergency types
                            if label in self.emergency_classes:
                                # Create detection object
                                detection = {
                                    'timestamp': datetime.datetime.now().isoformat(),
                                    'type': label,
                                    'confidence': confidence,
                                    'bbox': bbox
                                }
                                
                                detections.append(detection)
                                
                                # Add to history
                                self.detection_history.append(detection)
                                
                                # Emergency alert
                                if label == 'person_fainted':
                                    print(f"🚨 EMERGENCY: Passed out person detected! (confidence: {confidence:.2f})")
                                elif label == 'car_crash':
                                    print(f"🚨 EMERGENCY: Car crash detected! (confidence: {confidence:.2f})")
                                elif label == 'fire':
                                    print(f"🚨 EMERGENCY: Fire detected! (confidence: {confidence:.2f})")
                                
                                # Save detection image
                                self.save_detection(frame, detection)
            
            # Update current detections and timestamp
            self.current_detections = detections
            self.last_process_time = current_time
            
            return detections
            
        except Exception as e:
            print(f"❌ Error processing frame: {e}")
            return self.current_detections
    
    def draw_detections(self, frame):
        """Draw emergency detections on frame"""
        for detection in self.current_detections:
            x1, y1, x2, y2 = map(int, detection['bbox'])
            label = detection['type']
            confidence = detection['confidence']
            
            # Get color
            color = self.emergency_colors.get(label, (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw label
            label_text = f"{label}: {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1-30), (x1+len(label_text)*12, y1), color, -1)
            cv2.putText(frame, label_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def save_detection(self, frame, detection):
        """Save emergency detection image"""
        timestamp = detection['timestamp'].replace(':', '-').replace('.', '-')
        filename = f"EMERGENCY_{detection['type']}_{timestamp}_{detection['confidence']:.2f}.jpg"
        filepath = self.detections_dir / filename
        
        cv2.imwrite(str(filepath), frame)
        print(f"💾 EMERGENCY detection saved: {filename}")
    
    def draw_stats(self, frame):
        """Draw emergency statistics on frame"""
        # Draw emergency count
        emergency_count = len(self.detection_history)
        cv2.putText(frame, f"EMERGENCIES: {emergency_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Draw current emergencies count
        current_count = len(self.current_detections)
        cv2.putText(frame, f"Current: {current_count}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw recent emergencies
        recent_emergencies = self.detection_history[-3:]  # Last 3 emergencies
        y_offset = 90
        for i, detection in enumerate(recent_emergencies):
            text = f"{detection['type']}: {detection['confidence']:.2f}"
            cv2.putText(frame, text, (10, y_offset + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw processing status
        time_since_last = time.time() - self.last_process_time
        status = f"Next scan in: {max(0, self.process_interval - time_since_last):.1f}s"
        cv2.putText(frame, status, (10, frame.shape[0]-50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw instructions
        cv2.putText(frame, "Press 'q' to quit, 's' to save, 'p' to scan now", 
                   (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run_webcam_test(self):
        """Run the emergency-only webcam test"""
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        print("📹 Webcam started successfully")
        print("🚨 Emergency detection system is running...")
        print("🔍 Scanning for: Passed out people, Car crashes, Fires")
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("❌ Error: Could not read frame from webcam")
                    break
                
                # Process frame (only every 2 seconds)
                detections = self.process_frame(frame)
                
                # Draw emergency detections
                self.draw_detections(frame)
                
                # Draw statistics
                self.draw_stats(frame)
                
                # Display frame
                cv2.imshow('Emergency-Only Detection System', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("🛑 Quitting...")
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"emergency_frame_{timestamp}.jpg"
                    filepath = self.detections_dir / filename
                    cv2.imwrite(str(filepath), frame)
                    print(f"💾 Frame saved: {filename}")
                elif key == ord('p'):
                    # Force process now
                    self.last_process_time = 0
                    print("🔍 Forcing immediate emergency scan...")
        
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            print(f"\n📊 Emergency Detection Statistics:")
            print(f"   Total emergencies: {len(self.detection_history)}")
            
            if self.detection_history:
                print(f"   Emergency types:")
                type_counts = {}
                for detection in self.detection_history:
                    emergency_type = detection['type']
                    type_counts[emergency_type] = type_counts.get(emergency_type, 0) + 1
                
                for emergency_type, count in type_counts.items():
                    print(f"     {emergency_type}: {count}")
            
            print(f"   Emergency images saved to: {self.detections_dir}")

def main():
    """Main function"""
    detector = EmergencyOnlyDetector()
    detector.run_webcam_test()

if __name__ == "__main__":
    main() 