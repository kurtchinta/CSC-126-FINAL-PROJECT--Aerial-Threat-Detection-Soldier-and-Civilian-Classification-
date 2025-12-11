"""
Utility functions for data preparation
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import yaml


def visualize_annotations(image_path: Path, label_path: Path, class_names: dict):
    """
    Visualize YOLO format annotations on an image
    
    Args:
        image_path: Path to image
        label_path: Path to YOLO format label file
        class_names: Dictionary mapping class IDs to names
    """
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    # Read labels
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center, y_center, w, h = map(float, parts[1:])
                    
                    # Convert YOLO format to pixel coordinates
                    x1 = int((x_center - w/2) * width)
                    y1 = int((y_center - h/2) * height)
                    x2 = int((x_center + w/2) * width)
                    y2 = int((y_center + h/2) * height)
                    
                    # Draw bounding box
                    color = (255, 0, 0) if class_id == 0 else (0, 255, 0) if class_id == 1 else (0, 0, 255)
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label
                    label = class_names.get(class_id, f"Class {class_id}")
                    cv2.putText(image, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Display
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Annotations: {image_path.name}")
    plt.tight_layout()
    plt.show()


def analyze_dataset(dataset_dir: Path):
    """
    Analyze dataset statistics
    
    Args:
        dataset_dir: Path to dataset directory
    """
    stats = {
        'total_images': 0,
        'total_annotations': 0,
        'class_distribution': {},
        'image_sizes': []
    }
    
    for split in ['train', 'valid', 'test']:
        images_dir = dataset_dir / split / 'images'
        labels_dir = dataset_dir / split / 'labels'
        
        if not images_dir.exists():
            continue
        
        image_files = list(images_dir.glob('*.jpg')) + \
                     list(images_dir.glob('*.png')) + \
                     list(images_dir.glob('*.jpeg'))
        
        stats['total_images'] += len(image_files)
        
        for image_path in image_files:
            # Get image size
            img = cv2.imread(str(image_path))
            if img is not None:
                stats['image_sizes'].append(img.shape[:2])
            
            # Count annotations
            label_path = labels_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            stats['total_annotations'] += 1
                            stats['class_distribution'][class_id] = \
                                stats['class_distribution'].get(class_id, 0) + 1
    
    # Print statistics
    print("\n" + "="*60)
    print(f"Dataset Analysis: {dataset_dir.name}")
    print("="*60)
    print(f"Total Images: {stats['total_images']}")
    print(f"Total Annotations: {stats['total_annotations']}")
    print(f"\nClass Distribution:")
    for class_id, count in stats['class_distribution'].items():
        percentage = (count / stats['total_annotations']) * 100
        print(f"  Class {class_id}: {count} ({percentage:.2f}%)")
    
    if stats['image_sizes']:
        avg_height = np.mean([s[0] for s in stats['image_sizes']])
        avg_width = np.mean([s[1] for s in stats['image_sizes']])
        print(f"\nAverage Image Size: {avg_width:.0f} x {avg_height:.0f}")
    
    print("="*60 + "\n")
    
    return stats


def check_dataset_integrity(dataset_dir: Path):
    """
    Check dataset for missing labels or corrupted images
    
    Args:
        dataset_dir: Path to dataset directory
    """
    issues = []
    
    for split in ['train', 'valid', 'test']:
        images_dir = dataset_dir / split / 'images'
        labels_dir = dataset_dir / split / 'labels'
        
        if not images_dir.exists():
            continue
        
        image_files = list(images_dir.glob('*.jpg')) + \
                     list(images_dir.glob('*.png')) + \
                     list(images_dir.glob('*.jpeg'))
        
        for image_path in image_files:
            # Check if image can be read
            img = cv2.imread(str(image_path))
            if img is None:
                issues.append(f"Corrupted image: {image_path}")
                continue
            
            # Check if label exists
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                issues.append(f"Missing label: {label_path}")
    
    if issues:
        print("\n⚠ Dataset Integrity Issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    else:
        print("\n✓ Dataset integrity check passed!")
    
    return issues


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data" / "combined"
    
    if DATA_DIR.exists():
        analyze_dataset(DATA_DIR)
        check_dataset_integrity(DATA_DIR)
