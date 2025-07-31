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
        
        # Load detection model
        self.load_detection_model()
        
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
        """Load the detection model"""
        try:
            # Try to load the trained car crash detection model
            car_crash_model_path = "models/car_crash_detection2/weights/best.pt"
            if Path(car_crash_model_path).exists():
                self.detection_model = YOLO(car_crash_model_path)
                print("✅ Detection model loaded successfully")
            else:
                print("⚠️ Detection model not found")
                self.detection_model = None
        except Exception as e:
            print(f"❌ Error loading detection model: {e}")
            self.detection_model = None
    
    def start_webcam(self):
        """Start webcam for testing"""
        try:
            self.webcam = cv2.VideoCapture(0)
            if not self.webcam.isOpened():
                print("❌ Could not open webcam")
                return False
            
            self.webcam_active = True
            print("📹 Webcam started successfully")
            return True
        except Exception as e:
            print(f"❌ Error starting webcam: {e}")
            return False
    
    def stop_webcam(self):
        """Stop webcam"""
        if self.webcam:
            self.webcam.release()
            self.webcam = None
        self.webcam_active = False
        print("📹 Webcam stopped")
    
    def process_webcam_frame(self, frame):
        """Process a webcam frame for detections"""
        if not self.detection_model:
            return None, []
        
        try:
            # Run detection on frame
            results = self.detection_model(frame)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get detection info
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        coords = box.xyxy[0].cpu().numpy()
                        
                        # Get class name
                        class_name = result.names[cls]
                        
                        if conf > 0.3:  # Confidence threshold
                            detection = {
                                'type': class_name,
                                'confidence': conf,
                                'bbox': coords.tolist(),
                                'class_id': cls
                            }
                            detections.append(detection)
            
            return frame, detections
            
        except Exception as e:
            print(f"❌ Error processing frame: {e}")
            return frame, []
    
    def add_detection(self, drone_id, detection_type, confidence, location=None, image=None, bbox=None, reason=""):
        """Add a new detection to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Save detection to database
        cursor.execute('''
            INSERT INTO detections (drone_id, detection_type, confidence, location_lat, location_lng, bbox, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (drone_id, detection_type, confidence, location[0] if location else None, 
              location[1] if location else None, json.dumps(bbox) if bbox else None, reason))
        
        detection_id = cursor.lastrowid
        
        # Save image if provided
        image_path = None
        if image is not None:
            image_path = f"static/detections/detection_{detection_id}.jpg"
            Path("static/detections").mkdir(exist_ok=True)
            cv2.imwrite(image_path, image)
            
            # Update database with image path
            cursor.execute('UPDATE detections SET image_path = ? WHERE id = ?', (image_path, detection_id))
        
        conn.commit()
        conn.close()
        
        # Create detection record
        detection = {
            'id': detection_id,
            'drone_id': drone_id,
            'type': detection_type,
            'confidence': confidence,
            'location': location,
            'timestamp': datetime.datetime.now().isoformat(),
            'image_path': image_path,
            'bbox': bbox,
            'reason': reason
        }
        
        # Add to history
        self.detection_history.append(detection)
        
        # Emit to connected clients
        socketio.emit('new_detection', detection)
        
        print(f"🚨 Detection: {detection_type} by {drone_id} (confidence: {confidence:.2f})")
        return detection
    
    def get_detections(self, limit=100):
        """Get recent detections from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, drone_id, detection_type, confidence, location_lat, location_lng, 
                   timestamp, image_path, bbox, reason
            FROM detections 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        detections = []
        for row in cursor.fetchall():
            detection = {
                'id': row[0],
                'drone_id': row[1],
                'type': row[2],
                'confidence': row[3],
                'location': (row[4], row[5]) if row[4] and row[5] else None,
                'timestamp': row[6],
                'image_path': row[7],
                'bbox': json.loads(row[8]) if row[8] else None,
                'reason': row[9]
            }
            detections.append(detection)
        
        conn.close()
        return detections
    
    def get_detection_stats(self):
        """Get detection statistics"""
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
            SELECT COUNT(*) 
            FROM detections 
            WHERE timestamp > datetime('now', '-1 day')
        ''')
        recent_detections = cursor.fetchone()[0]
        
        # Active drones
        cursor.execute('''
            SELECT COUNT(*) 
            FROM drones 
            WHERE status = 'active'
        ''')
        active_drones = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_detections': total_detections,
            'detections_by_type': detections_by_type,
            'recent_detections': recent_detections,
            'active_drones': active_drones
        }
    
    def register_drone(self, drone_id, name, location=None):
        """Register a new drone"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO drones (id, name, location_lat, location_lng)
            VALUES (?, ?, ?, ?)
        ''', (drone_id, name, location[0] if location else None, location[1] if location else None))
        
        conn.commit()
        conn.close()
        
        # Update drones dict
        self.drones[drone_id] = {
            'id': drone_id,
            'name': name,
            'status': 'active',
            'location': location,
            'last_seen': datetime.datetime.now()
        }
        
        socketio.emit('drone_registered', self.drones[drone_id])
        print(f"✅ Drone registered: {name} ({drone_id})")
    
    def update_drone_status(self, drone_id, status, location=None, battery_level=None):
        """Update drone status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE drones 
            SET status = ?, location_lat = ?, location_lng = ?, battery_level = ?, last_seen = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (status, location[0] if location else None, location[1] if location else None, 
              battery_level, drone_id))
        
        conn.commit()
        conn.close()
        
        # Update drones dict
        if drone_id in self.drones:
            self.drones[drone_id].update({
                'status': status,
                'location': location,
                'battery_level': battery_level,
                'last_seen': datetime.datetime.now()
            })
        
        socketio.emit('drone_status_update', {
            'drone_id': drone_id,
            'status': status,
            'location': location,
            'battery_level': battery_level
        })

# Global dashboard instance
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
    limit = request.args.get('limit', 100, type=int)
    detections = dashboard.get_detections(limit)
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
    """API endpoint for processing a webcam frame"""
    try:
        # Get base64 image data
        data = request.json
        image_data = base64.b64decode(data['image'].split(',')[1])
        
        # Convert to OpenCV format
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process frame
        processed_frame, detections = dashboard.process_webcam_frame(frame)
        
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