#!/usr/bin/env python3
"""
Simplified Medical Emergency Detection System
Uses OpenCV for basic object detection (demonstration version)
For full functionality, install ultralytics and PyTorch
"""

import cv2
import numpy as np
import time
import datetime
import csv
import os
from pathlib import Path
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

class SimpleMedicalEmergencyDetector:
    """Simplified detection system using OpenCV"""
    
    def __init__(self, confidence_threshold: float = 0.5,
                 detection_interval: float = 0.5):
        """
        Initialize the detection system
        
        Args:
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
        
        # Load OpenCV's pre-trained models for demonstration
        self._load_opencv_models()
        
        # Detection history for tracking
        self.detection_history = deque(maxlen=100)
        
        # Emergency class mapping (simplified)
        self.emergency_classes = {
            0: "person",
            1: "car", 
            2: "truck",
            3: "bus",
            4: "motorcycle"
        }
        
        # Initialize CSV logging
        self._setup_csv_logging()
        
        self.logger.info("Simple Medical Emergency Detection System initialized")
        self.logger.warning("This is a demonstration version. For full medical emergency detection, install ultralytics and PyTorch.")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.logs_dir / 'detection_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_csv_logging(self):
        """Setup CSV file for logging detections"""
        self.csv_path = self.logs_dir / 'detections.csv'
        csv_exists = self.csv_path.exists()
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not csv_exists:
                writer.writerow(['timestamp', 'label', 'confidence', 'bbox', 'image_path'])
    
    def _load_opencv_models(self):
        """Load OpenCV pre-trained models for demonstration"""
        try:
            # Load COCO dataset classes
            self.coco_classes = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                'toothbrush'
            ]
            
            # Load YOLO model files (if available)
            model_path = Path("yolov3.weights")
            config_path = Path("yolov3.cfg")
            
            if model_path.exists() and config_path.exists():
                self.net = cv2.dnn.readNet(str(model_path), str(config_path))
                self.logger.info("Loaded YOLO model")
            else:
                self.net = None
                self.logger.warning("YOLO model files not found. Using basic OpenCV detection.")
                
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            self.net = None
    
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
            detections = []
            
            if self.net is not None:
                # Use YOLO model if available
                detections = self._detect_with_yolo(frame)
            else:
                # Use basic OpenCV detection
                detections = self._detect_with_opencv(frame)
            
            if detections:
                self.last_detection_time = current_time
                self._save_detections(detections)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            return []
    
    def _detect_with_yolo(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects using YOLO model"""
        detections = []
        
        # Prepare image for YOLO
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Get detections
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        outputs = self.net.forward(output_layers)
        
        # Process detections
        height, width = frame.shape[:2]
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x1 = int(center_x - w / 2)
                    y1 = int(center_y - h / 2)
                    x2 = int(center_x + w / 2)
                    y2 = int(center_y + h / 2)
                    
                    label = self.coco_classes[class_id] if class_id < len(self.coco_classes) else f"class_{class_id}"
                    
                    detection_obj = Detection(
                        timestamp=datetime.datetime.now().isoformat(),
                        label=label,
                        confidence=float(confidence),
                        bbox=[x1, y1, x2, y2],
                        image_path="",
                        frame_data=frame.copy()
                    )
                    
                    detections.append(detection_obj)
        
        return detections
    
    def _detect_with_opencv(self, frame: np.ndarray) -> List[Detection]:
        """Basic OpenCV detection for demonstration"""
        detections = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Simple edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        min_area = 1000
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Simple confidence based on area
                confidence = min(area / 10000, 0.9)
                
                if confidence > self.confidence_threshold:
                    detection_obj = Detection(
                        timestamp=datetime.datetime.now().isoformat(),
                        label="object",
                        confidence=confidence,
                        bbox=[x, y, x + w, y + h],
                        image_path="",
                        frame_data=frame.copy()
                    )
                    
                    detections.append(detection_obj)
        
        return detections
    
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
                
                # Log to CSV
                with open(self.csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        detection.timestamp,
                        detection.label,
                        detection.confidence,
                        str(detection.bbox),
                        detection.image_path
                    ])
                
                # Add to history
                self.detection_history.append(detection)
                
                self.logger.info(f"Detection saved: {detection.label} (confidence: {detection.confidence:.2f})")
                
            except Exception as e:
                self.logger.error(f"Error saving detection: {e}")
    
    def _draw_bounding_box(self, frame: np.ndarray, bbox: List[float], 
                          label: str, confidence: float) -> np.ndarray:
        """Draw bounding box and label on frame"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Draw label background
        label_text = f"{label}: {confidence:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 0, 255), -1)
        
        # Draw label text
        cv2.putText(frame, label_text, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def get_detection_stats(self) -> Dict:
        """Get detection statistics"""
        stats = {
            'total_detections': len(self.detection_history),
            'recent_detections': list(self.detection_history)[-10:],
            'detection_counts': {}
        }
        
        for detection in self.detection_history:
            stats['detection_counts'][detection.label] = \
                stats['detection_counts'].get(detection.label, 0) + 1
        
        return stats

def main():
    """Main function to run the detection system"""
    # Initialize detector
    detector = SimpleMedicalEmergencyDetector(
        confidence_threshold=0.3,
        detection_interval=1.0
    )
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Use default camera
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Simple Medical Emergency Detection System Started")
    print("This is a demonstration version using OpenCV")
    print("For full medical emergency detection, install ultralytics and PyTorch")
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
            cv2.imshow('Simple Medical Emergency Detection', frame)
            
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