"""
YOLOv8 Model Training Script for Aerial Surveillance Classification
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import torch
import yaml
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "combined"
MODELS_DIR = PROJECT_ROOT / "models"
RUNS_DIR = PROJECT_ROOT / "runs"

# Training configuration
CONFIG = {
    'model': os.getenv('MODEL_SIZE', 'yolov8m.pt'),  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    'epochs': int(os.getenv('EPOCHS', 100)),
    'batch': int(os.getenv('BATCH_SIZE', 16)),
    'imgsz': int(os.getenv('IMAGE_SIZE', 640)),
    'device': os.getenv('DEVICE', '0'),  # '0' for GPU, 'cpu' for CPU
    'workers': int(os.getenv('WORKERS', 8)),
    'optimizer': 'SGD',  # SGD, Adam, AdamW
    'lr0': 0.01,  # initial learning rate
    'lrf': 0.01,  # final learning rate
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'box': 7.5,  # box loss gain
    'cls': 0.5,  # cls loss gain
    'dfl': 1.5,  # dfl loss gain
    'patience': 50,  # early stopping patience
    'save': True,
    'save_period': 10,  # save checkpoint every x epochs
    'cache': False,  # cache images for faster training
    'pretrained': True,
    'verbose': True,
    'seed': 0,
    'deterministic': True,
    'single_cls': False,
    'rect': False,
    'cos_lr': False,
    'close_mosaic': 10,
    'resume': False,
    'amp': True,  # Automatic Mixed Precision
    'fraction': 1.0,  # dataset fraction to train on
    'profile': False,
    'overlap_mask': True,
    'mask_ratio': 4,
    'dropout': 0.0,
    'val': True,
    'plots': True,
}


class TrainingLogger:
    """Custom training logger"""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
    def log(self, message: str):
        """Log a message to console and file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')


def check_environment():
    """Check training environment and requirements"""
    logger = TrainingLogger(PROJECT_ROOT / "logs")
    
    logger.log("="*60)
    logger.log("ENVIRONMENT CHECK")
    logger.log("="*60)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        logger.log(f"✓ CUDA available")
        logger.log(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.log(f"  CUDA Version: {torch.version.cuda}")
        logger.log(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.log("⚠ CUDA not available - training will use CPU (slower)")
    
    # Check dataset
    dataset_yaml = DATA_DIR / "dataset.yaml"
    if not dataset_yaml.exists():
        logger.log(f"✗ Dataset configuration not found: {dataset_yaml}")
        logger.log("  Run: python src/data_prep/download_datasets.py")
        logger.log("  Then: python src/data_prep/augment_data.py")
        sys.exit(1)
    
    logger.log(f"✓ Dataset configuration found: {dataset_yaml}")
    
    # Load and verify dataset
    with open(dataset_yaml, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    train_path = Path(dataset_config['path']) / dataset_config['train']
    val_path = Path(dataset_config['path']) / dataset_config['val']
    
    if train_path.exists():
        train_images = len(list(train_path.glob('*')))
        logger.log(f"✓ Training images: {train_images}")
    else:
        logger.log(f"✗ Training path not found: {train_path}")
        sys.exit(1)
    
    if val_path.exists():
        val_images = len(list(val_path.glob('*')))
        logger.log(f"✓ Validation images: {val_images}")
    else:
        logger.log(f"⚠ Validation path not found: {val_path}")
    
    logger.log("="*60 + "\n")
    
    return logger


def train_model(logger: TrainingLogger):
    """Train YOLOv8 model"""
    
    logger.log("="*60)
    logger.log("STARTING TRAINING")
    logger.log("="*60)
    logger.log(f"Model: {CONFIG['model']}")
    logger.log(f"Epochs: {CONFIG['epochs']}")
    logger.log(f"Batch Size: {CONFIG['batch']}")
    logger.log(f"Image Size: {CONFIG['imgsz']}")
    logger.log(f"Device: {CONFIG['device']}")
    logger.log("="*60 + "\n")
    
    try:
        # Load model
        logger.log(f"Loading model: {CONFIG['model']}")
        model = YOLO(CONFIG['model'])
        
        # Dataset configuration
        dataset_yaml = DATA_DIR / "dataset.yaml"
        
        # Train model
        logger.log("Starting training...")
        results = model.train(
            data=str(dataset_yaml),
            epochs=CONFIG['epochs'],
            batch=CONFIG['batch'],
            imgsz=CONFIG['imgsz'],
            device=CONFIG['device'],
            workers=CONFIG['workers'],
            optimizer=CONFIG['optimizer'],
            lr0=CONFIG['lr0'],
            lrf=CONFIG['lrf'],
            momentum=CONFIG['momentum'],
            weight_decay=CONFIG['weight_decay'],
            warmup_epochs=CONFIG['warmup_epochs'],
            warmup_momentum=CONFIG['warmup_momentum'],
            warmup_bias_lr=CONFIG['warmup_bias_lr'],
            box=CONFIG['box'],
            cls=CONFIG['cls'],
            dfl=CONFIG['dfl'],
            patience=CONFIG['patience'],
            save=CONFIG['save'],
            save_period=CONFIG['save_period'],
            cache=CONFIG['cache'],
            pretrained=CONFIG['pretrained'],
            verbose=CONFIG['verbose'],
            seed=CONFIG['seed'],
            deterministic=CONFIG['deterministic'],
            single_cls=CONFIG['single_cls'],
            rect=CONFIG['rect'],
            cos_lr=CONFIG['cos_lr'],
            close_mosaic=CONFIG['close_mosaic'],
            resume=CONFIG['resume'],
            amp=CONFIG['amp'],
            fraction=CONFIG['fraction'],
            profile=CONFIG['profile'],
            overlap_mask=CONFIG['overlap_mask'],
            mask_ratio=CONFIG['mask_ratio'],
            dropout=CONFIG['dropout'],
            val=CONFIG['val'],
            plots=CONFIG['plots'],
            project=str(RUNS_DIR),
            name='aerial_surveillance',
            exist_ok=True,
        )
        
        logger.log("\n" + "="*60)
        logger.log("TRAINING COMPLETE")
        logger.log("="*60)
        
        # Save best model
        best_model_path = Path(results.save_dir) / "weights" / "best.pt"
        if best_model_path.exists():
            output_path = MODELS_DIR / "best.pt"
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            
            import shutil
            shutil.copy(best_model_path, output_path)
            logger.log(f"✓ Best model saved to: {output_path}")
        
        # Print metrics
        logger.log("\nFinal Metrics:")
        logger.log(f"  mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        logger.log(f"  mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        logger.log(f"  Precision: {results.results_dict.get('metrics/precision(B)', 'N/A')}")
        logger.log(f"  Recall: {results.results_dict.get('metrics/recall(B)', 'N/A')}")
        
        logger.log("\nResults saved to: " + str(results.save_dir))
        logger.log("="*60 + "\n")
        
        return results
        
    except Exception as e:
        logger.log(f"\n✗ Training failed: {str(e)}")
        raise


def main():
    """Main execution"""
    print("\n" + "="*60)
    print("AERIAL SURVEILLANCE MODEL TRAINING")
    print("YOLOv8 - Soldier/Civilian Classification")
    print("="*60 + "\n")
    
    # Check environment
    logger = check_environment()
    
    # Train model
    try:
        results = train_model(logger)
        
        print("\n✓ Training completed successfully!")
        print("\nNext steps:")
        print("  1. Evaluate model: python src/training/evaluate.py")
        print("  2. Test detection: python src/detection/detect_video.py")
        print("  3. Launch app: cd electron-app && npm start")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Training failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
