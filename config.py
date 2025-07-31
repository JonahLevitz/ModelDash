#!/usr/bin/env python3
"""
Configuration file for Medical Emergency Detection System
Centralized settings for easy customization
"""

import os
from pathlib import Path

class Config:
    """Configuration class for the detection system"""
    
    # Model Configuration
    MODEL_PATH = "models/medical_emergency.pt"
    CONFIDENCE_THRESHOLD = 0.5
    DETECTION_INTERVAL = 1.0  # seconds
    DEVICE = "cpu"  # "cpu" or "cuda"
    
    # Camera Configuration
    CAMERA_INDEX = 0  # 0 for default camera
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    FPS_TARGET = 30
    
    # Web Interface Configuration
    WEB_HOST = "0.0.0.0"
    WEB_PORT = 5000
    WEB_DEBUG = True
    SECRET_KEY = "your-secret-key-change-this-in-production"
    
    # File Paths
    BASE_DIR = Path(__file__).parent
    MODELS_DIR = BASE_DIR / "models"
    DETECTIONS_DIR = BASE_DIR / "detections"
    LOGS_DIR = BASE_DIR / "logs"
    TEMPLATES_DIR = BASE_DIR / "templates"
    STATIC_DIR = BASE_DIR / "static"
    
    # Emergency Classes
    EMERGENCY_CLASSES = {
        0: "person_down",
        1: "person_injured", 
        2: "person_unconscious",
        3: "medical_emergency",
        4: "accident_victim"
    }
    
    # Emergency Class Colors (for visualization)
    EMERGENCY_COLORS = {
        "person_down": (0, 0, 255),      # Red
        "person_injured": (0, 165, 255),  # Orange
        "person_unconscious": (255, 0, 255), # Purple
        "medical_emergency": (0, 255, 255),  # Yellow
        "accident_victim": (0, 0, 139)       # Dark Red
    }
    
    # Logging Configuration
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = LOGS_DIR / "detection_system.log"
    CSV_LOG_FILE = LOGS_DIR / "detections.csv"
    
    # Detection Configuration
    MAX_DETECTION_HISTORY = 100
    SAVE_DETECTION_IMAGES = True
    ANNOTATE_DETECTIONS = True
    
    # Performance Configuration
    ENABLE_FPS_MONITORING = True
    ENABLE_STATISTICS = True
    ENABLE_REAL_TIME_UPDATES = True
    
    # Alert Configuration
    ENABLE_EMERGENCY_ALERTS = True
    CRITICAL_EMERGENCIES = ["person_down", "person_unconscious"]
    ALERT_SOUND_ENABLED = False
    ALERT_SOUND_FILE = "alert.wav"
    
    # Training Configuration (for train_medical_emergency.py)
    TRAINING_CONFIG = {
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'device': 'cpu',
        'workers': 4,
        'patience': 20,
        'save_period': 10,
        'project': 'medical_emergency_training',
        'name': 'medical_emergency_model',
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 2.0,
        'label_smoothing': 0.0,
        'nbs': 64,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'plots': True
    }
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        directories = [
            cls.MODELS_DIR,
            cls.DETECTIONS_DIR,
            cls.LOGS_DIR,
            cls.TEMPLATES_DIR,
            cls.STATIC_DIR,
            cls.STATIC_DIR / "css",
            cls.STATIC_DIR / "js"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_model_path(cls):
        """Get the model path, with fallback to YOLOv8n"""
        if cls.MODEL_PATH and Path(cls.MODEL_PATH).exists():
            return cls.MODEL_PATH
        return "yolov8n.pt"  # Fallback to default YOLO model
    
    @classmethod
    def validate_config(cls):
        """Validate configuration settings"""
        errors = []
        
        # Check if required directories can be created
        try:
            cls.create_directories()
        except Exception as e:
            errors.append(f"Cannot create directories: {e}")
        
        # Validate confidence threshold
        if not 0 <= cls.CONFIDENCE_THRESHOLD <= 1:
            errors.append("Confidence threshold must be between 0 and 1")
        
        # Validate detection interval
        if cls.DETECTION_INTERVAL <= 0:
            errors.append("Detection interval must be positive")
        
        # Validate web port
        if not 1024 <= cls.WEB_PORT <= 65535:
            errors.append("Web port must be between 1024 and 65535")
        
        return errors

# Environment-specific overrides
if os.getenv("MEDICAL_DETECTION_ENV") == "production":
    Config.WEB_DEBUG = False
    Config.SECRET_KEY = os.getenv("SECRET_KEY", Config.SECRET_KEY)
    Config.LOG_LEVEL = "WARNING"

if os.getenv("MEDICAL_DETECTION_ENV") == "development":
    Config.WEB_DEBUG = True
    Config.LOG_LEVEL = "DEBUG"
    Config.CONFIDENCE_THRESHOLD = 0.3  # Lower threshold for testing

# Raspberry Pi specific optimizations
if os.path.exists("/proc/device-tree/model") and "Raspberry Pi" in open("/proc/device-tree/model").read():
    Config.DEVICE = "cpu"
    Config.FPS_TARGET = 15  # Lower FPS for Pi
    Config.DETECTION_INTERVAL = 2.0  # Slower detection for Pi
    Config.TRAINING_CONFIG['batch'] = 8  # Smaller batch size for Pi 