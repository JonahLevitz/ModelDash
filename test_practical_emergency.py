#!/usr/bin/env python3
"""
Practical Emergency Detection System
Detects real-world emergencies using multiple approaches
"""

import cv2
import numpy as np
import time
import datetime
from pathlib import Path
import json
from ultralytics import YOLO
import requests
import threading

class PracticalEmergencyDetector:
    """Practical emergency detection system"""
    
    def __init__(self):
        """Initialize the practical detector"""
        self.confidence_threshold = 0.4  # Lower threshold for better detection
        self.process_interval = 2.0
        self.last_process_time = 0
        
        # Load pre-trained models
        print("🤖 Loading detection models...")
        try:
            # Try to load the newly trained car crash detection model first
            car_crash_model_path = "models/car_crash_detection2/weights/best.pt"
            if Path(car_crash_model_path).exists():
                self.emergency_model = YOLO(car_crash_model_path)
                print("✅ New car crash detection model loaded successfully")
                print("📊 Model trained for 1 epoch with car crash dataset")
            else:
                # Fallback to original emergency detection model
                emergency_model_path = "models/emergency_detection.pt"
                if Path(emergency_model_path).exists():
                    self.emergency_model = YOLO(emergency_model_path)
                    print("✅ Emergency detection model loaded successfully")
                else:
                    print("⚠️ Emergency detection model not found, using default YOLO model")
                    self.emergency_model = YOLO('yolov8n.pt')
            
            # Also load default model for person detection
            self.person_model = YOLO('yolov8n.pt')
            print("✅ Models loaded successfully")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
        
        # Emergency detection rules
        self.emergency_rules = {
            'person_down': {
                'detection': 'person',
                'conditions': ['motionless', 'lying_down', 'unconscious'],
                'color': (0, 0, 255)  # Red
            },
            'person_fainted': {
                'detection': 'person_fainted',
                'conditions': ['unconscious', 'fainted'],
                'color': (0, 0, 255)  # Red
            },
            'person_injured': {
                'detection': 'person_injured',
                'conditions': ['injured', 'hurt'],
                'color': (0, 0, 255)  # Red
            },
            'car_crash': {
                'detection': 'car_crash',
                'conditions': ['damaged', 'overturned', 'accident'],
                'color': (0, 165, 255)  # Orange
            },
            'fire': {
                'detection': 'fire',
                'conditions': ['bright_orange', 'smoke_plume'],
                'color': (0, 0, 255)  # Red
            },
            'smoke': {
                'detection': 'smoke',
                'conditions': ['smoke_plume', 'smoke_detected'],
                'color': (128, 128, 128)  # Gray
            },
            'medical_emergency': {
                'detection': 'medical_emergency',
                'conditions': ['medical', 'emergency'],
                'color': (255, 0, 255)  # Magenta
            },
            'accident_victim': {
                'detection': 'accident_victim',
                'conditions': ['victim', 'accident'],
                'color': (0, 255, 255)  # Cyan
            }
        }
        
        # Detection history
        self.detection_history = []
        self.current_detections = []
        
        # Create detections directory
        self.detections_dir = Path("detections")
        self.detections_dir.mkdir(exist_ok=True)
        
        # Dashboard integration
        self.dashboard_url = "http://localhost:5000"
        self.drone_id = "drone-001"
        self.dashboard_enabled = True
        
        # Register drone with dashboard
        self.register_drone_with_dashboard()
        
        print("🚁 Practical Emergency Detection System")
        print("=" * 50)
        print("🔍 Detecting Real-World Emergencies:")
        print("   🚑 People lying down/unconscious/fainted")
        print("   🚗 Car crashes and accident victims")
        print("   🔥 Fires and smoke")
        print("   🏥 Medical emergencies and injuries")
        print("📊 Dashboard integration enabled")
        print("🌐 Dashboard URL: http://localhost:5000")
        print("Press 'q' to quit, 's' to save frame, 'p' to scan now")
    
    def detect_person_down(self, frame, detections):
        """Detect if someone is lying down or unconscious"""
        for detection in detections:
            if detection['class'] == 'person' and detection['confidence'] > 0.7:
                # Check if person is in unusual position (lying down)
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                
                # Calculate aspect ratio (width/height)
                width = x2 - x1
                height = y2 - y1
                aspect_ratio = width / height if height > 0 else 0
                
                # If person is very wide relative to height, might be lying down
                if aspect_ratio > 2.0:  # Very wide person
                    return {
                        'type': 'person_down',
                        'confidence': detection['confidence'],
                        'bbox': bbox,
                        'reason': 'Person appears to be lying down'
                    }
        
        return None
    
    def detect_car_crash(self, frame, detections):
        """Detect car crashes or damaged vehicles"""
        vehicles = []
        for detection in detections:
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
                    
                    # If vehicles are very close, might be crash
                    if distance < 150:  # Increased threshold for better detection
                        return {
                            'type': 'car_crash',
                            'confidence': max(vehicle1['confidence'], vehicle2['confidence']),
                            'bbox': bbox1,  # Use first vehicle bbox
                            'reason': 'Multiple vehicles in close proximity'
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
            
            # If vehicle is very tall relative to width, might be overturned
            if aspect_ratio < 0.8:  # Tall and narrow vehicle
                return {
                    'type': 'car_crash',
                    'confidence': vehicle['confidence'],
                    'bbox': bbox,
                    'reason': 'Vehicle appears overturned or damaged'
                }
        
        return None
    
    def detect_fire(self, frame):
        """Detect fire using color analysis"""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define fire color ranges (orange/red)
        lower_fire = np.array([0, 100, 100])
        upper_fire = np.array([20, 255, 255])
        
        # Create mask for fire colors
        fire_mask = cv2.inRange(hsv, lower_fire, upper_fire)
        
        # Count fire-colored pixels
        fire_pixels = cv2.countNonZero(fire_mask)
        total_pixels = frame.shape[0] * frame.shape[1]
        fire_ratio = fire_pixels / total_pixels
        
        # If significant portion is fire-colored, might be fire
        if fire_ratio > 0.05:  # 5% of frame is fire-colored
            return {
                'type': 'fire',
                'confidence': min(fire_ratio * 10, 0.95),  # Scale ratio to confidence
                'bbox': [0, 0, frame.shape[1], frame.shape[0]],  # Full frame
                'reason': f'Fire-colored pixels detected ({fire_ratio:.2%})'
            }
        
        return None
    
    def register_drone_with_dashboard(self):
        """Register this drone with the dashboard"""
        try:
            response = requests.post(f"{self.dashboard_url}/api/register_drone", json={
                "drone_id": self.drone_id,
                "name": "Emergency Detection Drone",
                "location": [40.7128, -74.0060]  # Default location (NYC)
            })
            if response.status_code == 200:
                print("✅ Drone registered with dashboard successfully")
            else:
                print("⚠️ Failed to register drone with dashboard")
        except Exception as e:
            print(f"⚠️ Dashboard not available: {e}")
            self.dashboard_enabled = False
    
    def send_detection_to_dashboard(self, detection, frame=None):
        """Send detection to dashboard"""
        if not self.dashboard_enabled:
            return
        
        try:
            # Convert frame to base64 if provided
            image_data = None
            if frame is not None:
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                image_data = buffer.tobytes()
            
            # Prepare detection data
            detection_data = {
                "drone_id": self.drone_id,
                "type": detection['type'],
                "confidence": detection['confidence'],
                "reason": detection.get('reason', ''),
                "bbox": detection.get('bbox', None)
            }
            
            # Send to dashboard
            response = requests.post(f"{self.dashboard_url}/api/add_detection", json=detection_data)
            
            if response.status_code == 200:
                print(f"📊 Sent to dashboard: {detection['type']} (confidence: {detection['confidence']:.2f})")
            else:
                print(f"❌ Failed to send to dashboard: {detection['type']}")
                
        except Exception as e:
            print(f"❌ Error sending to dashboard: {e}")
    
    def update_drone_status(self, battery_level=None, location=None):
        """Update drone status on dashboard"""
        if not self.dashboard_enabled:
            return
        
        try:
            status_data = {
                "drone_id": self.drone_id,
                "status": "active",
                "battery_level": battery_level or 85,  # Default battery level
                "location": location or [40.7128, -74.0060]
            }
            
            response = requests.post(f"{self.dashboard_url}/api/update_drone", json=status_data)
            
            if response.status_code != 200:
                print("❌ Failed to update drone status")
                
        except Exception as e:
            print(f"❌ Error updating drone status: {e}")
    
    def process_frame(self, frame):
        """Process frame for emergency detection"""
        current_time = time.time()
        
        # Only process every 2 seconds
        if current_time - self.last_process_time < self.process_interval:
            return self.current_detections
        
        try:
            # Run emergency detection with trained model
            emergency_results = self.emergency_model(frame, verbose=False)
            
            # Run person detection with default model
            person_results = self.person_model(frame, verbose=False)
            
            detections = []
            emergencies = []
            
            # Process emergency detections from trained model
            for result in emergency_results:
                if result.boxes is not None:
                    boxes = result.boxes
                    
                    for box in boxes:
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = self.emergency_model.names[class_id]
                        
                        if confidence >= self.confidence_threshold:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            bbox = [float(x1), float(y1), float(x2), float(y2)]
                            
                            # Debug: Print all detections (excluding smoke)
                            if class_name != 'smoke':  # Skip smoke detection entirely
                                print(f"🔍 Detected: {class_name} (confidence: {confidence:.2f})")
                            
                            # Check if this is an emergency class (excluding smoke for now)
                            if class_name in ['car_crash', 'fire', 'person_fainted', 'person_down', 'person_injured', 'medical_emergency']:
                                emergency_type = class_name
                                if class_name == 'person_fainted':
                                    emergency_type = 'person_down'  # Map to person_down for consistency
                                
                                emergencies.append({
                                    'type': emergency_type,
                                    'confidence': confidence,
                                    'bbox': bbox,
                                    'reason': f'Detected {class_name} with trained model'
                                })
                            
                            detections.append({
                                'class': class_name,
                                'confidence': confidence,
                                'bbox': bbox
                            })
            
            # Process person detections for additional person_down detection
            for result in person_results:
                if result.boxes is not None:
                    boxes = result.boxes
                    
                    for box in boxes:
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = self.person_model.names[class_id]
                        
                        if confidence >= self.confidence_threshold and class_name == 'person':
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            bbox = [float(x1), float(y1), float(x2), float(y2)]
                            
                            # Check for person down using aspect ratio
                            person_emergency = self.detect_person_down(frame, [{'class': 'person', 'confidence': confidence, 'bbox': bbox}])
                            if person_emergency:
                                emergencies.append(person_emergency)
            
            # Note: Car crash detection is handled by the trained model above
            # The trained model already includes 'car_crash' class detection
            
            # Update current detections and timestamp
            self.current_detections = emergencies
            self.last_process_time = current_time
            
            # Add to history and save
            for emergency in emergencies:
                self.detection_history.append(emergency)
                print(f"🚨 EMERGENCY: {emergency['type']} - {emergency['reason']} (confidence: {emergency['confidence']:.2f})")
                self.save_detection(frame, emergency)
                
                # Send to dashboard
                self.send_detection_to_dashboard(emergency, frame)
            
            return emergencies
            
        except Exception as e:
            print(f"❌ Error processing frame: {e}")
            return self.current_detections
    
    def draw_detections(self, frame):
        """Draw emergency detections on frame"""
        for detection in self.current_detections:
            x1, y1, x2, y2 = map(int, detection['bbox'])
            label = detection['type']
            confidence = detection['confidence']
            reason = detection.get('reason', '')
            
            # Get color
            color = self.emergency_rules.get(label, {}).get('color', (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw label
            label_text = f"{label}: {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1-30), (x1+len(label_text)*12, y1), color, -1)
            cv2.putText(frame, label_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw reason
            if reason:
                cv2.putText(frame, reason, (x1, y2+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def save_detection(self, frame, detection):
        """Save emergency detection image"""
        timestamp = datetime.datetime.now().isoformat().replace(':', '-').replace('.', '-')
        filename = f"EMERGENCY_{detection['type']}_{timestamp}_{detection['confidence']:.2f}.jpg"
        filepath = self.detections_dir / filename
        
        cv2.imwrite(str(filepath), frame)
        print(f"💾 Emergency detection saved: {filename}")
    
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
        """Run the practical webcam test"""
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        print("📹 Webcam started successfully")
        print("🚨 Practical emergency detection system is running...")
        print("🔍 Scanning for real-world emergencies...")
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("❌ Error: Could not read frame from webcam")
                    break
                
                # Process frame (only every 2 seconds)
                detections = self.process_frame(frame)
                
                # Update drone status periodically
                if time.time() % 30 < 1:  # Every 30 seconds
                    self.update_drone_status()
                
                # Draw emergency detections
                self.draw_detections(frame)
                
                # Draw statistics
                self.draw_stats(frame)
                
                # Display frame
                cv2.imshow('Practical Emergency Detection System', frame)
                
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
            print(f"\n📊 Practical Emergency Detection Statistics:")
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
    detector = PracticalEmergencyDetector()
    detector.run_webcam_test()

if __name__ == "__main__":
    main() 