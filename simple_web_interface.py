#!/usr/bin/env python3
"""
Simplified Web Interface for Medical Emergency Detection System
Works with OpenCV-based detection (demonstration version)
"""

from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import json
import threading
import time
import base64
from datetime import datetime
import os
from pathlib import Path
from simple_detection_system import SimpleMedicalEmergencyDetector
import logging

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
detector = None
video_thread = None
is_detecting = False
detection_stats = {}

class VideoStream:
    """Handles video streaming and detection"""
    
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.detector = None
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
    
    def start(self, camera_index=0):
        """Start video capture"""
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise Exception("Could not open camera")
        
        self.is_running = True
        self.detector = SimpleMedicalEmergencyDetector(
            confidence_threshold=0.3,
            detection_interval=0.5
        )
        
        # Start detection thread
        detection_thread = threading.Thread(target=self._detection_loop)
        detection_thread.daemon = True
        detection_thread.start()
    
    def stop(self):
        """Stop video capture"""
        self.is_running = False
        if self.cap:
            self.cap.release()
    
    def _detection_loop(self):
        """Main detection loop"""
        while self.is_running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # Process frame for detections
                    detections = self.detector.process_frame(frame)
                    
                    # Update stats
                    stats = self.detector.get_detection_stats()
                    
                    # Emit detection results
                    if detections:
                        for detection in detections:
                            socketio.emit('detection', {
                                'timestamp': detection.timestamp,
                                'label': detection.label,
                                'confidence': detection.confidence,
                                'bbox': detection.bbox,
                                'image_path': detection.image_path
                            })
                    
                    # Emit stats update
                    socketio.emit('stats_update', stats)
                    
                    # Calculate FPS
                    self.frame_count += 1
                    current_time = time.time()
                    if current_time - self.last_fps_time >= 1.0:
                        self.fps = self.frame_count
                        self.frame_count = 0
                        self.last_fps_time = current_time
                        
                        socketio.emit('fps_update', {'fps': self.fps})
            
            time.sleep(0.1)  # Small delay to prevent excessive CPU usage

# Initialize video stream
video_stream = VideoStream()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/start_detection')
def start_detection():
    """Start detection system"""
    global is_detecting
    
    try:
        if not is_detecting:
            video_stream.start()
            is_detecting = True
            return jsonify({'status': 'success', 'message': 'Detection started'})
        else:
            return jsonify({'status': 'error', 'message': 'Detection already running'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/stop_detection')
def stop_detection():
    """Stop detection system"""
    global is_detecting
    
    try:
        if is_detecting:
            video_stream.stop()
            is_detecting = False
            return jsonify({'status': 'success', 'message': 'Detection stopped'})
        else:
            return jsonify({'status': 'error', 'message': 'Detection not running'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/stats')
def get_stats():
    """Get current detection statistics"""
    if video_stream.detector:
        return jsonify(video_stream.detector.get_detection_stats())
    return jsonify({'error': 'Detector not initialized'})

@app.route('/api/detections')
def get_recent_detections():
    """Get recent detections"""
    if video_stream.detector:
        detections = list(video_stream.detector.detection_history)[-20:]
        return jsonify([{
            'timestamp': d.timestamp,
            'label': d.label,
            'confidence': d.confidence,
            'image_path': d.image_path
        } for d in detections])
    return jsonify([])

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('status', {'message': 'Connected to detection system'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

@socketio.on('request_stats')
def handle_stats_request():
    """Handle stats request"""
    if video_stream.detector:
        emit('stats_update', video_stream.detector.get_detection_stats())

if __name__ == '__main__':
    # Create necessary directories
    Path("models").mkdir(exist_ok=True)
    Path("detections").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    print("Starting Simple Medical Emergency Detection Web Interface...")
    print("This is a demonstration version using OpenCV")
    print("For full medical emergency detection, install ultralytics and PyTorch")
    print("Open http://localhost:5000 in your browser")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True) 