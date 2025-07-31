#!/usr/bin/env python3
"""
Combine multiple emergency datasets into one unified dataset
Combines: Car Accident, Fire, and Fainted Detection datasets
"""

import os
import shutil
from pathlib import Path
import yaml
from collections import defaultdict

def combine_datasets():
    """Combine all three emergency datasets"""
    
    print("🚁 Combining Emergency Detection Datasets...")
    print("=" * 50)
    
    # Create combined dataset directory
    combined_dir = Path("medical_emergency_dataset")
    combined_dir.mkdir(exist_ok=True)
    
    for split in ['train', 'valid', 'test']:
        for subdir in ['images', 'labels']:
            (combined_dir / split / subdir).mkdir(parents=True, exist_ok=True)
    
    # Dataset paths
    datasets = {
        'car_accident': Path("CarAccident"),
        'fire': Path("fires"), 
        'fainted': Path("Fainted detection")
    }
    
    # Class mapping for each dataset
    class_mappings = {
        'car_accident': {
            '0': 'car_crash',
            '1': 'accident_victim', 
            '2': 'car_crash',
            '3': 'accident_victim',
            '4': 'car_crash',
            'car': 'car_crash'
        },
        'fire': {
            'fire': 'fire',
            'smoke': 'smoke'
        },
        'fainted': {
            'Fainted': 'person_fainted'
        }
    }
    
    # Combined class list
    combined_classes = [
        'car_crash',
        'accident_victim', 
        'fire',
        'smoke',
        'person_fainted',
        'person_down',
        'person_injured',
        'medical_emergency',
        'emergency_vehicle'
    ]
    
    # Track file counts
    total_files = defaultdict(int)
    
    # Process each dataset
    for dataset_name, dataset_path in datasets.items():
        print(f"\n📁 Processing {dataset_name} dataset...")
        
        if not dataset_path.exists():
            print(f"❌ Dataset {dataset_name} not found at {dataset_path}")
            continue
            
        # Read original data.yaml
        original_yaml = dataset_path / "data.yaml"
        if original_yaml.exists():
            with open(original_yaml, 'r') as f:
                original_data = yaml.safe_load(f)
                original_classes = original_data.get('names', [])
                print(f"   Original classes: {original_classes}")
        
        # Process each split
        for split in ['train', 'valid', 'test']:
            images_dir = dataset_path / split / "images"
            labels_dir = dataset_path / split / "labels"
            
            if not images_dir.exists() or not labels_dir.exists():
                print(f"   ⚠️  {split} split not found for {dataset_name}")
                continue
            
            # Copy images and update labels
            image_files = list(images_dir.glob("*"))
            label_files = list(labels_dir.glob("*.txt"))
            
            print(f"   📊 {split}: {len(image_files)} images, {len(label_files)} labels")
            
            for img_file in image_files:
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    # Copy image
                    new_img_path = combined_dir / split / "images" / f"{dataset_name}_{img_file.name}"
                    shutil.copy2(img_file, new_img_path)
                    total_files[f"{split}_images"] += 1
                    
                    # Process corresponding label file
                    label_file = labels_dir / f"{img_file.stem}.txt"
                    if label_file.exists():
                        new_label_path = combined_dir / split / "labels" / f"{dataset_name}_{label_file.name}"
                        
                        # Update class IDs in label file
                        with open(label_file, 'r') as f:
                            lines = f.readlines()
                        
                        updated_lines = []
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                old_class_id = int(parts[0])
                                if old_class_id < len(original_classes):
                                    old_class_name = original_classes[old_class_id]
                                    new_class_name = class_mappings[dataset_name].get(old_class_name, old_class_name)
                                    new_class_id = combined_classes.index(new_class_name)
                                    parts[0] = str(new_class_id)
                                    updated_lines.append(' '.join(parts) + '\n')
                        
                        with open(new_label_path, 'w') as f:
                            f.writelines(updated_lines)
                        
                        total_files[f"{split}_labels"] += 1
    
    # Create combined data.yaml
    combined_yaml = {
        'path': str(combined_dir.absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(combined_classes),
        'names': combined_classes
    }
    
    with open(combined_dir / "data.yaml", 'w') as f:
        yaml.dump(combined_yaml, f, default_flow_style=False)
    
    print("\n✅ Dataset combination completed!")
    print("=" * 50)
    print(f"📊 Combined Dataset Statistics:")
    print(f"   Train images: {total_files['train_images']}")
    print(f"   Train labels: {total_files['train_labels']}")
    print(f"   Valid images: {total_files['valid_images']}")
    print(f"   Valid labels: {total_files['valid_labels']}")
    print(f"   Test images: {total_files['test_images']}")
    print(f"   Test labels: {total_files['test_labels']}")
    print(f"   Total classes: {len(combined_classes)}")
    print(f"   Classes: {combined_classes}")
    print(f"\n📁 Combined dataset saved to: {combined_dir}")
    
    return combined_dir

if __name__ == "__main__":
    combine_datasets() 