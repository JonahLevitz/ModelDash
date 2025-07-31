#!/usr/bin/env python3
"""
Train Emergency Detection Model
Combined model for: Car Crashes, Fires, Medical Emergencies
Trains for 10 epochs on the combined dataset
"""

from ultralytics import YOLO
import os
import yaml
from pathlib import Path
import shutil

def train_emergency_detection_model():
    """Train the combined emergency detection model"""
    
    print("🚁 Training Emergency Detection Model")
    print("=" * 50)
    print("Combined Emergency Types:")
    print("  🚗 Car Crashes & Accident Victims")
    print("  🔥 Fires & Smoke")
    print("  🚑 Fainted/Unconscious People")
    print("  🏥 Medical Emergencies")
    print("=" * 50)
    
    # Check if combined dataset exists
    dataset_path = Path("medical_emergency_dataset")
    if not dataset_path.exists():
        print("❌ Combined dataset not found!")
        print("Please run combine_datasets.py first")
        return False
    
    # Check data.yaml
    data_yaml = dataset_path / "data.yaml"
    if not data_yaml.exists():
        print("❌ data.yaml not found in combined dataset!")
        return False
    
    # Load dataset info
    with open(data_yaml, 'r') as f:
        dataset_info = yaml.safe_load(f)
    
    print(f"📊 Dataset Info:")
    print(f"   Classes: {dataset_info['names']}")
    print(f"   Number of classes: {dataset_info['nc']}")
    print(f"   Train path: {dataset_info['train']}")
    print(f"   Validation path: {dataset_info['val']}")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Initialize model
    print("\n🤖 Initializing YOLOv8 model...")
    model = YOLO('yolov8n.pt')  # Start with YOLOv8 nano
    
    # Training configuration for 10 epochs
    training_config = {
        'data': str(data_yaml),
        'epochs': 10,  # Train for 10 epochs as requested
        'imgsz': 640,
        'batch': 16,
        'device': 'cpu',  # Use CPU for compatibility
        'workers': 4,
        'patience': 5,  # Early stopping patience
        'save': True,
        'save_period': 2,  # Save every 2 epochs
        'cache': False,
        'project': 'emergency_detection_training',
        'name': 'emergency_detection_model',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'auto',
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'single_cls': False,
        'rect': False,
        'cos_lr': False,
        'close_mosaic': 10,
        'resume': False,
        'amp': True,
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 1.0,  # Shorter warmup for 10 epochs
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
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
    
    print(f"\n🎯 Training Configuration:")
    print(f"   Epochs: {training_config['epochs']}")
    print(f"   Image size: {training_config['imgsz']}")
    print(f"   Batch size: {training_config['batch']}")
    print(f"   Device: {training_config['device']}")
    print(f"   Learning rate: {training_config['lr0']}")
    
    try:
        print("\n🚀 Starting training...")
        print("This will train for 10 epochs on the combined emergency dataset")
        print("Training may take 30-60 minutes depending on your system")
        
        # Start training
        results = model.train(**training_config)
        
        print("\n✅ Training completed successfully!")
        print(f"📁 Best model saved at: {results.save_dir}")
        
        # Copy the best model to models directory
        best_model_path = Path(results.save_dir) / 'weights' / 'best.pt'
        if best_model_path.exists():
            final_model_path = models_dir / 'emergency_detection.pt'
            shutil.copy2(best_model_path, final_model_path)
            print(f"🎯 Best model copied to: {final_model_path}")
        
        # Show training results
        print(f"\n📊 Training Results:")
        print(f"   Final mAP50: {results.box.map50:.3f}")
        print(f"   Final mAP50-95: {results.box.map:.3f}")
        print(f"   Training completed in: {results.epoch}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def validate_model():
    """Validate the trained model"""
    
    model_path = Path("models/emergency_detection.pt")
    
    if not model_path.exists():
        print("❌ Trained model not found. Please train the model first.")
        return False
    
    print("\n🔍 Validating emergency detection model...")
    
    try:
        model = YOLO(str(model_path))
        
        # Run validation
        results = model.val()
        
        print("✅ Validation completed!")
        print(f"📊 Validation Results:")
        print(f"   mAP50: {results.box.map50:.3f}")
        print(f"   mAP50-95: {results.box.map:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

def main():
    """Main training function"""
    
    print("🚁 Emergency Detection Model Training")
    print("=" * 60)
    
    # Step 1: Combine datasets (if not already done)
    if not Path("medical_emergency_dataset/data.yaml").exists():
        print("📁 Combining datasets...")
        from combine_datasets import combine_datasets
        combine_datasets()
    
    # Step 2: Train the model
    if train_emergency_detection_model():
        print("\n🎉 Training completed successfully!")
        
        # Step 3: Validate the model
        print("\n🔍 Validating model...")
        validate_model()
        
        print("\n🚀 Next steps:")
        print("1. Test the model: python3 test_system.py")
        print("2. Run web interface: python3 web_interface.py")
        print("3. Deploy to Raspberry Pi")
        
    else:
        print("❌ Training failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 