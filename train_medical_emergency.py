#!/usr/bin/env python3
"""
Training script for Medical Emergency Detection Model
Uses YOLOv8 for training on medical emergency scenarios
"""

from ultralytics import YOLO
import os
import yaml
from pathlib import Path
import shutil

def create_medical_emergency_dataset():
    """Create medical emergency dataset structure"""
    
    # Create dataset directories
    dataset_dir = Path("medical_emergency_dataset")
    dataset_dir.mkdir(exist_ok=True)
    
    for split in ['train', 'valid', 'test']:
        for subdir in ['images', 'labels']:
            (dataset_dir / split / subdir).mkdir(parents=True, exist_ok=True)
    
    # Create data.yaml for medical emergencies
    medical_data = {
        'path': str(dataset_dir.absolute()),
        'train': 'train/images',
        'val': 'valid/images', 
        'test': 'test/images',
        'nc': 5,  # Number of classes
        'names': [
            'person_down',      # Person lying on ground
            'person_injured',   # Person with visible injuries
            'person_unconscious', # Unconscious person
            'medical_emergency', # General medical emergency
            'accident_victim'    # Person in accident
        ]
    }
    
    with open(dataset_dir / 'data.yaml', 'w') as f:
        yaml.dump(medical_data, f, default_flow_style=False)
    
    print(f"Created medical emergency dataset structure at {dataset_dir}")
    return dataset_dir

def prepare_training_data():
    """Prepare training data from existing car accident dataset"""
    
    # Use existing car accident data as starting point
    car_accident_dir = Path("CarAccident")
    medical_dataset_dir = Path("medical_emergency_dataset")
    
    if not car_accident_dir.exists():
        print("CarAccident dataset not found. Please ensure the dataset is available.")
        return False
    
    # Copy and adapt the data
    print("Preparing medical emergency training data...")
    
    # For now, we'll use the car accident data as a base
    # In a real scenario, you would need to:
    # 1. Collect medical emergency images
    # 2. Annotate them with medical emergency labels
    # 3. Convert annotations to YOLO format
    
    # Create a simple mapping for demonstration
    # This is a placeholder - you'll need real medical emergency data
    print("Note: This is a placeholder. You need to:")
    print("1. Collect medical emergency images")
    print("2. Annotate them with medical emergency labels")
    print("3. Place them in the medical_emergency_dataset directory")
    
    return True

def train_medical_emergency_model():
    """Train the medical emergency detection model"""
    
    # Create dataset structure
    dataset_dir = create_medical_emergency_dataset()
    
    # Prepare training data
    if not prepare_training_data():
        print("Failed to prepare training data")
        return
    
    # Initialize model
    model = YOLO('yolov8n.pt')  # Start with YOLOv8 nano
    
    # Training configuration
    training_config = {
        'data': str(dataset_dir / 'data.yaml'),
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'device': 'cpu',  # Use 'cuda' if GPU available
        'workers': 4,
        'patience': 20,
        'save': True,
        'save_period': 10,
        'cache': False,
        'project': 'medical_emergency_training',
        'name': 'medical_emergency_model',
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
        'warmup_epochs': 3.0,
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
    
    print("Starting medical emergency model training...")
    print(f"Training configuration: {training_config}")
    
    try:
        # Start training
        results = model.train(**training_config)
        
        print("Training completed successfully!")
        print(f"Best model saved at: {results.save_dir}")
        
        # Copy the best model to models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        best_model_path = Path(results.save_dir) / 'weights' / 'best.pt'
        if best_model_path.exists():
            shutil.copy2(best_model_path, models_dir / 'medical_emergency.pt')
            print(f"Best model copied to: {models_dir / 'medical_emergency.pt'}")
        
        return True
        
    except Exception as e:
        print(f"Training failed: {e}")
        return False

def validate_model():
    """Validate the trained model"""
    
    model_path = Path("models/medical_emergency.pt")
    
    if not model_path.exists():
        print("Trained model not found. Please train the model first.")
        return
    
    print("Validating medical emergency model...")
    
    try:
        model = YOLO(str(model_path))
        
        # Run validation
        results = model.val()
        
        print("Validation completed!")
        print(f"mAP50: {results.box.map50}")
        print(f"mAP50-95: {results.box.map}")
        
        return True
        
    except Exception as e:
        print(f"Validation failed: {e}")
        return False

def main():
    """Main training function"""
    
    print("Medical Emergency Detection Model Training")
    print("=" * 50)
    
    # Create models directory
    Path("models").mkdir(exist_ok=True)
    
    # Train the model
    if train_medical_emergency_model():
        print("\nTraining completed successfully!")
        
        # Validate the model
        print("\nValidating model...")
        validate_model()
        
        print("\nNext steps:")
        print("1. Test the model with real medical emergency scenarios")
        print("2. Fine-tune if needed")
        print("3. Deploy to Raspberry Pi")
        
    else:
        print("Training failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 