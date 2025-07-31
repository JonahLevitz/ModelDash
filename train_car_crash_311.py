#!/usr/bin/env python3.11
"""
Train Car Crash Detection with Python 3.11
Train car crash detection for 1 epoch using the existing dataset
"""

from ultralytics import YOLO
import yaml
from pathlib import Path

def create_car_crash_dataset_config():
    """Create dataset configuration for car crash training"""
    
    # Check if medical_emergency_dataset exists (contains car crash data)
    dataset_path = Path("medical_emergency_dataset")
    if not dataset_path.exists():
        print("❌ Error: medical_emergency_dataset not found")
        return False
    
    # Create car crash specific dataset config
    car_crash_config = {
        'path': str(dataset_path.absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'names': {
            0: 'car_crash',
            1: 'accident_victim',
            2: 'fire',
            3: 'smoke',
            4: 'person_fainted',
            5: 'person_down',
            6: 'person_injured',
            7: 'medical_emergency',
            8: 'emergency_vehicle'
        },
        'nc': 9
    }
    
    # Save config
    config_path = Path("car_crash_dataset.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(car_crash_config, f, default_flow_style=False)
    
    print(f"✅ Created dataset config: {config_path}")
    return str(config_path)

def train_car_crash_detection():
    """Train car crash detection for 1 epoch"""
    
    print("🚗 Retraining Car Crash Detection")
    print("=" * 50)
    
    # Create dataset config
    config_path = create_car_crash_dataset_config()
    if not config_path:
        return False
    
    # Training parameters
    model_size = "n"  # YOLOv8n for fast training
    epochs = 1
    imgsz = 640
    batch_size = 16
    
    print(f"📊 Training Parameters:")
    print(f"   Model: YOLOv8{model_size}")
    print(f"   Epochs: {epochs}")
    print(f"   Image Size: {imgsz}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Dataset Config: {config_path}")
    
    try:
        # Load base model
        print("🤖 Loading YOLOv8n model...")
        model = YOLO(f'yolov8{model_size}.pt')
        
        # Train for car crash detection
        print("🚀 Starting training...")
        results = model.train(
            data=config_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            name='car_crash_detection',
            project='models',
            save=True,
            verbose=True
        )
        
        print("✅ Car crash detection training completed!")
        print(f"Results saved to: models/car_crash_detection/")
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def main():
    """Main function"""
    success = train_car_crash_detection()
    
    if success:
        print("\n🎉 Car crash detection retraining completed!")
        print("📁 Check models/car_crash_detection/ for results")
        print("🔧 You can now use the retrained model for better car crash detection")
    else:
        print("\n❌ Training failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 