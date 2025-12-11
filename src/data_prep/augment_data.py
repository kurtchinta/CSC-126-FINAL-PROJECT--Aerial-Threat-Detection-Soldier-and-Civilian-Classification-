"""
Data Augmentation and Preprocessing Script
Augments training data to improve model generalization
"""

import os
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
from tqdm import tqdm
import yaml
import shutil
from typing import List, Tuple


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


class DataAugmenter:
    """Augment aerial surveillance images for better model training"""
    
    def __init__(self, augmentation_factor: int = 3):
        """
        Initialize augmenter
        
        Args:
            augmentation_factor: Number of augmented versions per image
        """
        self.augmentation_factor = augmentation_factor
        
        # Define augmentation pipeline
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.MotionBlur(blur_limit=5, p=0.2),
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.RandomScale(scale_limit=0.1, p=0.3),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.3
            ),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
    
    def read_yolo_labels(self, label_path: Path) -> Tuple[List[int], List[List[float]]]:
        """Read YOLO format labels"""
        class_labels = []
        bboxes = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_labels.append(int(parts[0]))
                        bboxes.append([float(x) for x in parts[1:]])
        
        return class_labels, bboxes
    
    def write_yolo_labels(self, label_path: Path, class_labels: List[int], bboxes: List[List[float]]):
        """Write YOLO format labels"""
        with open(label_path, 'w') as f:
            for cls, bbox in zip(class_labels, bboxes):
                f.write(f"{cls} {' '.join(map(str, bbox))}\n")
    
    def augment_image(self, image_path: Path, label_path: Path, output_dir: Path, index: int):
        """Augment a single image with its labels"""
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            return False
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read labels
        class_labels, bboxes = self.read_yolo_labels(label_path)
        
        # Generate augmented versions
        for i in range(self.augmentation_factor):
            try:
                # Apply augmentation
                if len(bboxes) > 0:
                    transformed = self.transform(
                        image=image,
                        bboxes=bboxes,
                        class_labels=class_labels
                    )
                else:
                    transformed = {'image': image, 'bboxes': [], 'class_labels': []}
                
                # Save augmented image
                aug_image = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)
                output_image_path = output_dir / f"{image_path.stem}_aug_{index}_{i}{image_path.suffix}"
                cv2.imwrite(str(output_image_path), aug_image)
                
                # Save augmented labels
                output_label_path = output_dir.parent / "labels" / f"{image_path.stem}_aug_{index}_{i}.txt"
                self.write_yolo_labels(
                    output_label_path,
                    transformed['class_labels'],
                    transformed['bboxes']
                )
                
            except Exception as e:
                print(f"Error augmenting {image_path.name}: {str(e)}")
                continue
        
        return True
    
    def process_dataset(self, dataset_dir: Path, output_dir: Path):
        """Process an entire dataset"""
        print(f"\nProcessing: {dataset_dir.name}")
        
        # Setup output directories
        for split in ['train', 'valid', 'test']:
            (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Process each split
        for split in ['train', 'valid', 'test']:
            split_dir = dataset_dir / split
            if not split_dir.exists():
                continue
            
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            
            if not images_dir.exists():
                continue
            
            # Get all images
            image_files = list(images_dir.glob('*.jpg')) + \
                         list(images_dir.glob('*.png')) + \
                         list(images_dir.glob('*.jpeg'))
            
            print(f"  {split}: {len(image_files)} images")
            
            # Process images
            for idx, image_path in enumerate(tqdm(image_files, desc=f"  Augmenting {split}")):
                label_path = labels_dir / f"{image_path.stem}.txt"
                
                # Copy original
                output_image_dir = output_dir / split / 'images'
                output_label_dir = output_dir / split / 'labels'
                
                shutil.copy(image_path, output_image_dir / image_path.name)
                if label_path.exists():
                    shutil.copy(label_path, output_label_dir / label_path.name)
                
                # Augment only training data
                if split == 'train':
                    self.augment_image(image_path, label_path, output_image_dir, idx)


def combine_datasets(processed_dir: Path, output_dir: Path, class_mapping: dict):
    """Combine multiple datasets into one unified dataset"""
    print("\n" + "="*60)
    print("Combining datasets...")
    print("="*60)
    
    # Create output structure
    for split in ['train', 'valid', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Combine datasets
    dataset_dirs = [d for d in processed_dir.iterdir() if d.is_dir()]
    
    for dataset_dir in dataset_dirs:
        print(f"\nCombining: {dataset_dir.name}")
        
        for split in ['train', 'valid', 'test']:
            src_images = dataset_dir / split / 'images'
            src_labels = dataset_dir / split / 'labels'
            dst_images = output_dir / split / 'images'
            dst_labels = output_dir / split / 'labels'
            
            if not src_images.exists():
                continue
            
            # Copy images
            for img_file in src_images.iterdir():
                if img_file.is_file():
                    new_name = f"{dataset_dir.name}_{img_file.name}"
                    shutil.copy(img_file, dst_images / new_name)
            
            # Copy and remap labels if needed
            if src_labels.exists():
                for label_file in src_labels.iterdir():
                    if label_file.is_file():
                        new_name = f"{dataset_dir.name}_{label_file.name}"
                        shutil.copy(label_file, dst_labels / new_name)
    
    print("\n✓ Datasets combined successfully!")


def create_dataset_yaml(output_dir: Path):
    """Create YOLO dataset configuration file"""
    yaml_content = {
        'path': str(output_dir.absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'names': {
            0: 'soldier',
            1: 'civilian',
            2: 'person'
        },
        'nc': 3
    }
    
    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"\n✓ Created dataset.yaml at {yaml_path}")


def main():
    """Main execution"""
    print("\n" + "="*60)
    print("AERIAL SURVEILLANCE DATA AUGMENTATION")
    print("="*60)
    
    # Check if raw data exists
    if not RAW_DATA_DIR.exists() or not list(RAW_DATA_DIR.iterdir()):
        print("\n⚠ No raw data found!")
        print("  Run: python src/data_prep/download_datasets.py")
        return
    
    # Initialize augmenter
    augmenter = DataAugmenter(augmentation_factor=3)
    
    # Process each dataset
    dataset_dirs = [d for d in RAW_DATA_DIR.iterdir() if d.is_dir()]
    
    print(f"\nFound {len(dataset_dirs)} datasets to process")
    
    for dataset_dir in dataset_dirs:
        output_dir = PROCESSED_DATA_DIR / dataset_dir.name
        augmenter.process_dataset(dataset_dir, output_dir)
    
    # Combine datasets
    combined_dir = PROJECT_ROOT / "data" / "combined"
    class_mapping = {
        'soldier': 0,
        'civilian': 1,
        'person': 2
    }
    
    combine_datasets(PROCESSED_DATA_DIR, combined_dir, class_mapping)
    
    # Create dataset YAML
    create_dataset_yaml(combined_dir)
    
    print("\n" + "="*60)
    print("AUGMENTATION COMPLETE")
    print("="*60)
    print("\nNext step: python src/training/train_yolo.py")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
