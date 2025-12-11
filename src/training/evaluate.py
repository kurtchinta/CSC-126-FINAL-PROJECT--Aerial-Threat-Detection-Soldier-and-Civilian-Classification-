"""
Model Evaluation Script
Comprehensive evaluation of trained YOLO model
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
import json


PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "combined"
RESULTS_DIR = PROJECT_ROOT / "tests" / "results"


class ModelEvaluator:
    """Evaluate YOLO model performance"""
    
    def __init__(self, model_path: Path, device: str = '0'):
        """Initialize evaluator"""
        self.model_path = model_path
        self.device = device
        self.model = None
        self.results_dir = RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Class names
        self.class_names = {
            0: 'soldier',
            1: 'civilian',
            2: 'person'
        }
    
    def load_model(self):
        """Load trained model"""
        print(f"\nLoading model: {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = YOLO(str(self.model_path))
        print("✓ Model loaded successfully")
    
    def evaluate_on_testset(self):
        """Evaluate model on test set"""
        print("\n" + "="*60)
        print("EVALUATING ON TEST SET")
        print("="*60)
        
        # Run validation
        results = self.model.val(
            data=str(DATA_DIR / "dataset.yaml"),
            split='test',
            batch=16,
            imgsz=640,
            device=self.device,
            plots=True,
            save_json=True,
            save_hybrid=True,
            conf=0.001,
            iou=0.6,
            max_det=300,
            verbose=True,
        )
        
        # Print metrics
        print("\n" + "="*60)
        print("TEST SET METRICS")
        print("="*60)
        print(f"mAP50: {results.box.map50:.4f}")
        print(f"mAP50-95: {results.box.map:.4f}")
        print(f"Precision: {results.box.mp:.4f}")
        print(f"Recall: {results.box.mr:.4f}")
        print("="*60)
        
        # Per-class metrics
        print("\nPer-Class Metrics:")
        for i, (ap, p, r) in enumerate(zip(results.box.ap50, results.box.p, results.box.r)):
            class_name = self.class_names.get(i, f"Class {i}")
            print(f"  {class_name}:")
            print(f"    AP50: {ap:.4f}")
            print(f"    Precision: {p:.4f}")
            print(f"    Recall: {r:.4f}")
        
        return results
    
    def test_inference_speed(self, num_iterations: int = 100):
        """Test model inference speed"""
        print("\n" + "="*60)
        print("TESTING INFERENCE SPEED")
        print("="*60)
        
        # Create dummy image
        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Warmup
        print("Warming up...")
        for _ in range(10):
            self.model(dummy_img, verbose=False)
        
        # Measure speed
        print(f"Running {num_iterations} iterations...")
        import time
        
        times = []
        for _ in tqdm(range(num_iterations)):
            start = time.time()
            self.model(dummy_img, verbose=False)
            end = time.time()
            times.append(end - start)
        
        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = 1 / avg_time
        
        print("\n" + "="*60)
        print("INFERENCE SPEED RESULTS")
        print("="*60)
        print(f"Average Time: {avg_time*1000:.2f} ms")
        print(f"Std Dev: {std_time*1000:.2f} ms")
        print(f"FPS: {fps:.2f}")
        print(f"Min Time: {np.min(times)*1000:.2f} ms")
        print(f"Max Time: {np.max(times)*1000:.2f} ms")
        print("="*60)
        
        # Save results
        speed_results = {
            'avg_time_ms': float(avg_time * 1000),
            'std_time_ms': float(std_time * 1000),
            'fps': float(fps),
            'min_time_ms': float(np.min(times) * 1000),
            'max_time_ms': float(np.max(times) * 1000),
            'num_iterations': num_iterations,
            'image_size': '640x640',
            'device': self.device
        }
        
        with open(self.results_dir / 'inference_speed.json', 'w') as f:
            json.dump(speed_results, f, indent=2)
        
        return speed_results
    
    def visualize_predictions(self, num_samples: int = 10):
        """Visualize predictions on sample images"""
        print("\n" + "="*60)
        print("GENERATING PREDICTION VISUALIZATIONS")
        print("="*60)
        
        test_images_dir = DATA_DIR / "test" / "images"
        if not test_images_dir.exists():
            print("⚠ Test images directory not found")
            return
        
        # Get sample images
        image_files = list(test_images_dir.glob('*.jpg'))[:num_samples]
        
        if not image_files:
            print("⚠ No test images found")
            return
        
        # Create output directory
        vis_dir = self.results_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        print(f"Processing {len(image_files)} images...")
        
        for img_path in tqdm(image_files):
            # Run prediction
            results = self.model(str(img_path), verbose=False)[0]
            
            # Plot and save
            result_img = results.plot()
            output_path = vis_dir / f"pred_{img_path.name}"
            cv2.imwrite(str(output_path), result_img)
        
        print(f"✓ Visualizations saved to: {vis_dir}")
    
    def analyze_confidence_distribution(self):
        """Analyze confidence score distribution"""
        print("\n" + "="*60)
        print("ANALYZING CONFIDENCE DISTRIBUTION")
        print("="*60)
        
        test_images_dir = DATA_DIR / "test" / "images"
        if not test_images_dir.exists():
            print("⚠ Test images directory not found")
            return
        
        image_files = list(test_images_dir.glob('*.jpg'))
        
        all_confidences = {class_id: [] for class_id in self.class_names.keys()}
        
        print(f"Processing {len(image_files)} images...")
        
        for img_path in tqdm(image_files):
            results = self.model(str(img_path), verbose=False)[0]
            
            if results.boxes is not None:
                for box in results.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    all_confidences[class_id].append(confidence)
        
        # Plot distribution
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for i, (class_id, confidences) in enumerate(all_confidences.items()):
            if confidences:
                axes[i].hist(confidences, bins=50, edgecolor='black', alpha=0.7)
                axes[i].set_title(f'{self.class_names[class_id].title()} Confidence')
                axes[i].set_xlabel('Confidence')
                axes[i].set_ylabel('Count')
                axes[i].axvline(np.mean(confidences), color='r', linestyle='--', 
                              label=f'Mean: {np.mean(confidences):.3f}')
                axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'confidence_distribution.png', dpi=300)
        print(f"✓ Confidence distribution saved")
        
        # Print statistics
        print("\nConfidence Statistics:")
        for class_id, confidences in all_confidences.items():
            if confidences:
                print(f"  {self.class_names[class_id].title()}:")
                print(f"    Mean: {np.mean(confidences):.4f}")
                print(f"    Std: {np.std(confidences):.4f}")
                print(f"    Min: {np.min(confidences):.4f}")
                print(f"    Max: {np.max(confidences):.4f}")
    
    def generate_report(self):
        """Generate comprehensive evaluation report"""
        print("\n" + "="*60)
        print("GENERATING EVALUATION REPORT")
        print("="*60)
        
        report = {
            'model_path': str(self.model_path),
            'device': self.device,
            'timestamp': str(Path.ctime(self.model_path)),
        }
        
        # Save report
        report_path = self.results_dir / 'evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Report saved to: {report_path}")


def main():
    """Main execution"""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Load model
    model_path = MODELS_DIR / "best.pt"
    
    if not model_path.exists():
        print(f"\n✗ Model not found: {model_path}")
        print("  Train a model first: python src/training/train_yolo.py")
        sys.exit(1)
    
    # Initialize evaluator
    device = '0' if torch.cuda.is_available() else 'cpu'
    evaluator = ModelEvaluator(model_path, device=device)
    
    # Load model
    evaluator.load_model()
    
    # Run evaluations
    print("\n[1/5] Evaluating on test set...")
    evaluator.evaluate_on_testset()
    
    print("\n[2/5] Testing inference speed...")
    evaluator.test_inference_speed()
    
    print("\n[3/5] Generating visualizations...")
    evaluator.visualize_predictions(num_samples=20)
    
    print("\n[4/5] Analyzing confidence distribution...")
    evaluator.analyze_confidence_distribution()
    
    print("\n[5/5] Generating report...")
    evaluator.generate_report()
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
