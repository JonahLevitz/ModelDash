#!/usr/bin/env python3
"""
Drone Emergency Detection Dashboard - Webcam Testing Version
Real-time monitoring dashboard for drone emergency detections with webcam testing
"""

from flask import Flask, render_template, jsonify, request, Response
from flask_socketio import SocketIO, emit
import json
import sqlite3
import datetime
import threading
import time
import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import io
from PIL import Image

app = Flask(__name__)
app.config['SECRET_KEY'] = 'drone_dashboard_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

class DroneDashboard:
    """Dashboard for monitoring drone emergency detections with webcam testing"""
    
    def __init__(self):
        """Initialize the dashboard"""
        self.db_path = "drone_detections.db"
        self.clear_database()  # Clear database for testing
        self.init_database()
        self.drones = {}
        self.detection_history = []
        
        # Lazy load detection model
        self.model = None
        self.model_loaded = False
        
        # Webcam variables
        self.webcam = None
        self.webcam_active = False
        
        print("🚁 Drone Emergency Detection Dashboard - Webcam Testing")
        print("=" * 60)
        print("📊 Features:")
        print("   🚨 Real-time emergency alerts")
        print("   📹 Webcam testing interface")
        print("   📈 Detection analytics")
        print("   🎯 Live detection testing")
    
    def clear_database(self):
        """Clear the database for testing"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Drop existing tables
            cursor.execute('DROP TABLE IF EXISTS detections')
            cursor.execute('DROP TABLE IF EXISTS drones')
            
            conn.commit()
            conn.close()
            print("🗑️ Database cleared for testing")
        except Exception as e:
            print(f"⚠️ Error clearing database: {e}")
    
    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create detections table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                drone_id TEXT NOT NULL,
                detection_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                location_lat REAL,
                location_lng REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                image_path TEXT,
                bbox TEXT,
                reason TEXT
            )
        ''')
        
        # Create drones table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS drones (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT DEFAULT 'active',
                battery_level INTEGER DEFAULT 100,
                last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                location_lat REAL,
                location_lng REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Database initialized")
    
    def load_detection_model(self):
        """Load the detection model (lazy loading)"""
        if self.model_loaded:
            return self.model
            
        try:
            print("🔄 Loading detection model...")
            # Try to load from a local path first
            try:
                model_path = "models/emergency_detection.pt"
                if os.path.exists(model_path):
                    self.model = YOLO(model_path)
                else:
                    # Use a lightweight model for testing
                    self.model = YOLO('yolov8n.pt')  # Use nano model for speed
                    print("⚠️ Using default YOLO model for testing")
            except Exception as e:
                print(f"⚠️ Error loading custom model: {e}")
                # Fallback to default model
                self.model = YOLO('yolov8n.pt')
                print("✅ Using fallback YOLO model")
            
            self.model_loaded = True
            print("✅ Detection model loaded successfully")
            return self.model
            
        except Exception as e:
            print(f"❌ Error loading detection model: {e}")
            self.model = None
            self.model_loaded = False
            return None
    
    def start_webcam(self):
        """Start webcam capture"""
        try:
            self.webcam = cv2.VideoCapture(0)
            if self.webcam.isOpened():
                self.webcam_active = True
                print("📹 Webcam started")
                return True
            else:
                print("❌ Failed to start webcam")
                return False
        except Exception as e:
            print(f"❌ Error starting webcam: {e}")
            return False
    
    def stop_webcam(self):
        """Stop webcam capture"""
        if self.webcam:
            self.webcam.release()
        self.webcam_active = False
        print("📹 Webcam stopped")
    
    def process_webcam_frame(self, frame):
        """Process a webcam frame for detections"""
        try:
            # Load model if not loaded
            model = self.load_detection_model()
            if not model:
                return [], frame
            
            # Run detection
            results = model(frame)
            
            detections = []
            processed_frame = frame.copy()
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = result.names[class_id]
                        
                        # Filter for emergency-related classes
                        emergency_classes = ['person', 'fire', 'smoke', 'car', 'truck', 'bus']
                        if any(emergency in class_name.lower() for emergency in emergency_classes):
                            detection = {
                                'type': class_name,
                                'confidence': confidence,
                                'bbox': [int(x1), int(y1), int(x2), int(y2)]
                            }
                            detections.append(detection)
                            
                            # Draw bounding box
                            cv2.rectangle(processed_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                            cv2.putText(processed_frame, f"{class_name} {confidence:.2f}", 
                                      (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            
                            print(f"🚨 Detection: {class_name} by webcam_test (confidence: {confidence:.2f})")
            
            return detections, processed_frame
            
        except Exception as e:
            print(f"❌ Error processing frame: {e}")
            return [], frame
    
    def add_detection(self, drone_id, detection_type, confidence, location=None, image=None, bbox=None, reason=""):
        """Add a detection to the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Save image if provided
            image_path = None
            if image is not None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = f"static/detections/detection_{timestamp}.jpg"
                os.makedirs("static/detections", exist_ok=True)
                cv2.imwrite(image_path, image)
            
            # Insert detection
            cursor.execute('''
                INSERT INTO detections (drone_id, detection_type, confidence, location_lat, location_lng, image_path, bbox, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (drone_id, detection_type, confidence, 
                  location.get('lat') if location else None,
                  location.get('lng') if location else None,
                  image_path, json.dumps(bbox) if bbox else None, reason))
            
            detection_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            # Add to history
            detection = {
                'id': detection_id,
                'drone_id': drone_id,
                'type': detection_type,
                'confidence': confidence,
                'timestamp': datetime.datetime.now().isoformat(),
                'image_path': image_path,
                'bbox': bbox,
                'reason': reason
            }
            self.detection_history.append(detection)
            
            # Emit real-time update
            socketio.emit('new_detection', detection)
            
            return detection
            
        except Exception as e:
            print(f"❌ Error adding detection: {e}")
            return None
    
    def get_detections(self, limit=100):
        """Get recent detections from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, drone_id, detection_type, confidence, timestamp, image_path, bbox, reason
                FROM detections
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            detections = []
            for row in rows:
                detection = {
                    'id': row[0],
                    'drone_id': row[1],
                    'type': row[2],
                    'confidence': row[3],
                    'timestamp': row[4],
                    'image_path': row[5],
                    'bbox': json.loads(row[6]) if row[6] else None,
                    'reason': row[7]
                }
                detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"❌ Error getting detections: {e}")
            return []
    
    def get_detection_stats(self):
        """Get detection statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total detections
            cursor.execute('SELECT COUNT(*) FROM detections')
            total_detections = cursor.fetchone()[0]
            
            # Detections by type
            cursor.execute('''
                SELECT detection_type, COUNT(*) 
                FROM detections 
                GROUP BY detection_type
            ''')
            detections_by_type = dict(cursor.fetchall())
            
            # Recent detections (last 24 hours)
            cursor.execute('''
                SELECT COUNT(*) FROM detections 
                WHERE timestamp >= datetime('now', '-1 day')
            ''')
            recent_detections = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'total_detections': total_detections,
                'detections_by_type': detections_by_type,
                'recent_detections': recent_detections,
                'webcam_tests': len([d for d in self.detection_history if d['drone_id'] == 'webcam_test'])
            }
            
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
            return {
                'total_detections': 0,
                'detections_by_type': {},
                'recent_detections': 0,
                'webcam_tests': 0
            }
    
    def register_drone(self, drone_id, name, location=None):
        """Register a new drone"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO drones (id, name, location_lat, location_lng)
                VALUES (?, ?, ?, ?)
            ''', (drone_id, name, 
                  location.get('lat') if location else None,
                  location.get('lng') if location else None))
            
            conn.commit()
            conn.close()
            
            self.drones[drone_id] = {
                'id': drone_id,
                'name': name,
                'status': 'active',
                'location': location
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Error registering drone: {e}")
            return False
    
    def update_drone_status(self, drone_id, status, location=None, battery_level=None):
        """Update drone status"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE drones 
                SET status = ?, location_lat = ?, location_lng = ?, battery_level = ?, last_seen = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (status, 
                  location.get('lat') if location else None,
                  location.get('lng') if location else None,
                  battery_level,
                  drone_id))
            
            conn.commit()
            conn.close()
            
            if drone_id in self.drones:
                self.drones[drone_id].update({
                    'status': status,
                    'location': location,
                    'battery_level': battery_level
                })
            
            return True
            
        except Exception as e:
            print(f"❌ Error updating drone status: {e}")
            return False

# Initialize dashboard
dashboard = DroneDashboard()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/webcam')
def webcam_page():
    """Webcam testing page"""
    return render_template('webcam.html')

@app.route('/api/detections')
def api_detections():
    """API endpoint for getting detections"""
    detections = dashboard.get_detections()
    return jsonify(detections)

@app.route('/api/stats')
def api_stats():
    """API endpoint for getting statistics"""
    stats = dashboard.get_detection_stats()
    return jsonify(stats)

@app.route('/api/start_webcam', methods=['POST'])
def api_start_webcam():
    """API endpoint for starting webcam"""
    success = dashboard.start_webcam()
    return jsonify({'success': success})

@app.route('/api/stop_webcam', methods=['POST'])
def api_stop_webcam():
    """API endpoint for stopping webcam"""
    dashboard.stop_webcam()
    return jsonify({'success': True})

@app.route('/api/process_frame', methods=['POST'])
def api_process_frame():
    """API endpoint for processing webcam frame"""
    try:
        data = request.json
        frame_data = data.get('frame')
        
        if not frame_data:
            return jsonify({'success': False, 'error': 'No frame data provided'})
        
        # Decode base64 frame
        frame_data = frame_data.split(',')[1]  # Remove data URL prefix
        frame_bytes = base64.b64decode(frame_data)
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        # Process frame
        detections, processed_frame = dashboard.process_webcam_frame(frame)
        
        # Convert processed frame back to base64
        _, buffer = cv2.imencode('.jpg', processed_frame)
        processed_image = base64.b64encode(buffer).decode('utf-8')
        
        # Add detections to database if any found
        for detection in detections:
            dashboard.add_detection(
                drone_id="webcam_test",
                detection_type=detection['type'],
                confidence=detection['confidence'],
                bbox=detection['bbox'],
                reason=f"Webcam detection: {detection['type']}"
            )
        
        return jsonify({
            'success': True,
            'processed_image': f"data:image/jpeg;base64,{processed_image}",
            'detections': detections
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/add_detection', methods=['POST'])
def api_add_detection():
    """API endpoint for adding a detection"""
    data = request.json
    drone_id = data.get('drone_id')
    detection_type = data.get('type')
    confidence = data.get('confidence')
    location = data.get('location')
    bbox = data.get('bbox')
    reason = data.get('reason', '')
    
    # Convert base64 image if provided
    image = None
    if 'image' in data:
        image_data = base64.b64decode(data['image'])
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    
    detection = dashboard.add_detection(drone_id, detection_type, confidence, location, image, bbox, reason)
    return jsonify(detection)

@app.route('/api/clear_detections', methods=['POST'])
def api_clear_detections():
    """API endpoint for clearing all detections"""
    try:
        conn = sqlite3.connect(dashboard.db_path)
        cursor = conn.cursor()
        
        # Clear all detections
        cursor.execute('DELETE FROM detections')
        
        # Clear detection images
        import shutil
        detections_dir = Path("static/detections")
        if detections_dir.exists():
            shutil.rmtree(detections_dir)
            detections_dir.mkdir(exist_ok=True)
        
        conn.commit()
        conn.close()
        
        # Clear detection history
        dashboard.detection_history.clear()
        
        print("🗑️ All detections cleared")
        return jsonify({'success': True})
        
    except Exception as e:
        print(f"❌ Error clearing detections: {e}")
        return jsonify({'success': False, 'error': str(e)})

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f"Client connected: {request.sid}")
    emit('connected', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"Client disconnected: {request.sid}")

def run_dashboard():
    """Run the dashboard server"""
    print("🚀 Starting Drone Dashboard Server...")
    print("📱 Dashboard available at: http://localhost:5000")
    print("📹 Webcam testing at: http://localhost:5000/webcam")
    print("🔌 WebSocket enabled for real-time updates")
    
    # Use environment variable for port in production
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    socketio.run(app, host='0.0.0.0', port=port, debug=debug)

if __name__ == '__main__':
    run_dashboard() 