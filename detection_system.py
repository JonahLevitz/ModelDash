#!/usr/bin/env python3
"""
Emergency Detection System
Detects: Car Crashes, Fires, Medical Emergencies
Uses combined model trained on multiple emergency datasets
"""

import cv2
import numpy as np
import time
import datetime
import csv
import os
import json
from pathlib import Path
from ultralytics import YOLO
import logging
from typing import List, Dict, Tuple, Optional
import threading
from dataclasses import dataclass
from collections import deque

@dataclass
class Detection:
    """Data class for storing detection results"""
    timestamp: str
    label: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    image_path: str
    frame_data: Optional[np.ndarray] = None

class EmergencyDetector:
    """Main detection system for multiple emergency types"""
    
    def __init__(self, model_path: str = "models/emergency_detection.pt", 
                 confidence_threshold: float = 0.5,
                 detection_interval: float = 0.5):
        """
        Initialize the detection system
        
        Args:
            model_path: Path to the trained emergency detection model
            confidence_threshold: Minimum confidence for detections
            detection_interval: Time between detections in seconds
        """
        self.confidence_threshold = confidence_threshold
        self.detection_interval = detection_interval
        self.last_detection_time = 0
        
        # Create directories
        self.detections_dir = Path("detections")
        self.detections_dir.mkdir(exist_ok=True)
        
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Initialize logging
        self._setup_logging()
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Detection history for tracking
        self.detection_history = deque(maxlen=100)
        
        # Emergency class mapping for combined model
        self.emergency_classes = {
            0: "car_crash",
            1: "accident_victim", 
            2: "fire",
            3: "smoke",
            4: "person_fainted",
            5: "person_down",
            6: "person_injured",
            7: "medical_emergency",
            8: "emergency_vehicle"
        }
        
        # Emergency class colors (for visualization)
        self.emergency_colors = {
            "car_crash": (0, 0, 255),        # Red
            "accident_victim": (0, 165, 255), # Orange
            "fire": (0, 0, 255),             # Red
            "smoke": (128, 128, 128),        # Gray
            "person_fainted": (255, 0, 255), # Purple
            "person_down": (0, 0, 255),      # Red
            "person_injured": (0, 165, 255), # Orange
            "medical_emergency": (0, 255, 255), # Yellow
            "emergency_vehicle": (255, 255, 0)  # Cyan
        }
        
        # Critical emergency types (trigger alerts)
        self.critical_emergencies = [
            "car_crash", "accident_victim", "fire", 
            "person_fainted", "person_down", "person_injured"
        ]
        
        # Initialize CSV logging
        self._setup_csv_logging()
        
        self.logger.info("Emergency Detection System initialized")
        self.logger.info(f"Detecting: {list(self.emergency_classes.values())}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.logs_dir / 'emergency_detection.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_csv_logging(self):
        """Setup CSV file for logging detections"""
        self.csv_path = self.logs_dir / 'emergency_detections.csv'
        csv_exists = self.csv_path.exists()
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not csv_exists:
                writer.writerow(['timestamp', 'label', 'confidence', 'bbox', 'image_path', 'emergency_type'])
    
    def _load_model(self, model_path: str) -> YOLO:
        """Load the YOLO model"""
        try:
            if os.path.exists(model_path):
                self.logger.info(f"Loading emergency detection model from {model_path}")
                return YOLO(model_path)
            else:
                self.logger.warning(f"Emergency model not found at {model_path}, using YOLOv8n as fallback")
                return YOLO('yolov8n.pt')
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def process_frame(self, frame: np.ndarray) -> List[Detection]:
        """
        Process a single frame and return detections
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of Detection objects
        """
        current_time = time.time()
        
        # Skip if not enough time has passed since last detection
        if current_time - self.last_detection_time < self.detection_interval:
            return []
        
        try:
            # Run inference
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
                            label = self.emergency_classes.get(class_id, f"class_{class_id}")
                            
                            # Determine emergency type
                            emergency_type = self._get_emergency_type(label)
                            
                            # Create detection object
                            detection = Detection(
                                timestamp=datetime.datetime.now().isoformat(),
                                label=label,
                                confidence=confidence,
                                bbox=bbox,
                                image_path="",
                                frame_data=frame.copy()
                            )
                            
                            detections.append(detection)
                            
                            # Log critical emergencies
                            if label in self.critical_emergencies:
                                self.logger.warning(f"CRITICAL EMERGENCY DETECTED: {label} (confidence: {confidence:.2f})")
            
            if detections:
                self.last_detection_time = current_time
                self._save_detections(detections)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            return []
    
    def _get_emergency_type(self, label: str) -> str:
        """Categorize emergency type"""
        if label in ["car_crash", "accident_victim"]:
            return "traffic_accident"
        elif label in ["fire", "smoke"]:
            return "fire_emergency"
        elif label in ["person_fainted", "person_down", "person_injured", "medical_emergency"]:
            return "medical_emergency"
        else:
            return "other"
    
    def _save_detections(self, detections: List[Detection]):
        """Save detection images and log to CSV"""
        for detection in detections:
            try:
                # Save image
                timestamp_str = detection.timestamp.replace(':', '-').replace('.', '-')
                image_filename = f"{timestamp_str}_{detection.label}_{detection.confidence:.2f}.jpg"
                image_path = self.detections_dir / image_filename
                
                # Draw bounding box on image
                annotated_frame = self._draw_bounding_box(
                    detection.frame_data, 
                    detection.bbox, 
                    detection.label, 
                    detection.confidence
                )
                
                cv2.imwrite(str(image_path), annotated_frame)
                detection.image_path = str(image_path)
                
                # Get emergency type
                emergency_type = self._get_emergency_type(detection.label)
                
                # Log to CSV
                with open(self.csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        detection.timestamp,
                        detection.label,
                        detection.confidence,
                        json.dumps(detection.bbox),
                        detection.image_path,
                        emergency_type
                    ])
                
                # Add to history
                self.detection_history.append(detection)
                
                self.logger.info(f"Emergency detected: {detection.label} (confidence: {detection.confidence:.2f})")
                
            except Exception as e:
                self.logger.error(f"Error saving detection: {e}")
    
    def _draw_bounding_box(self, frame: np.ndarray, bbox: List[float], 
                          label: str, confidence: float) -> np.ndarray:
        """Draw bounding box and label on frame"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Get color for this emergency type
        color = self.emergency_colors.get(label, (0, 255, 0))
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label_text = f"{label}: {confidence:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
        
        # Draw label text
        cv2.putText(frame, label_text, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def get_detection_stats(self) -> Dict:
        """Get detection statistics"""
        stats = {
            'total_detections': len(self.detection_history),
            'recent_detections': list(self.detection_history)[-10:],
            'detection_counts': {},
            'emergency_types': {
                'traffic_accident': 0,
                'fire_emergency': 0,
                'medical_emergency': 0,
                'other': 0
            }
        }
        
        for detection in self.detection_history:
            # Count by label
            stats['detection_counts'][detection.label] = \
                stats['detection_counts'].get(detection.label, 0) + 1
            
            # Count by emergency type
            emergency_type = self._get_emergency_type(detection.label)
            stats['emergency_types'][emergency_type] += 1
        
        return stats

def main():
    """Main function to run the detection system"""
    # Initialize detector
    detector = EmergencyDetector(
        model_path="models/emergency_detection.pt",
        confidence_threshold=0.5,
        detection_interval=1.0
    )
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Use default camera
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Emergency Detection System Started")
    print("Detecting: Car Crashes, Fires, Medical Emergencies")
    print("Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame
            detections = detector.process_frame(frame)
            
            # Draw detections on frame
            for detection in detections:
                frame = detector._draw_bounding_box(
                    frame, detection.bbox, detection.label, detection.confidence
                )
            
            # Display frame
            cv2.imshow('Emergency Detection System', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 