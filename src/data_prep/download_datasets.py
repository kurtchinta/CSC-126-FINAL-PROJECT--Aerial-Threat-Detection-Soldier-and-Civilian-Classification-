"""
Dataset Download Script for Aerial Surveillance Classification
Downloads UAV Person dataset from Roboflow for training YOLO models
Dataset: https://universe.roboflow.com/militarypersons/uav-person-3
"""

import os
import sys
from pathlib import Path
from roboflow import Roboflow
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Roboflow API Key (set in .env file)
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")

# Dataset configuration - UAV Person Dataset
DATASET_CONFIG = {
    "name": "uav-person",
    "workspace": "militarypersons",
    "project": "uav-person-3",
    "version": 2
}


def download_dataset(workspace: str, project: str, version: int, name: str):
    """
    Download a dataset from Roboflow
    
    Args:
        workspace: Roboflow workspace name
        project: Project name
        version: Dataset version
        name: Local name for the dataset
    """
    print(f"\n{'='*60}")
    print(f"Downloading: {name}")
    print(f"Workspace: {workspace}/{project} (v{version})")
    print(f"{'='*60}\n")
    
    try:
        # Initialize Roboflow
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        
        # Get project
        project_obj = rf.workspace(workspace).project(project)
        
        # Download dataset in YOLO format
        dataset = project_obj.version(version).download(
            model_format="yolov8",
            location=str(DATA_DIR / name)
        )
        
        print(f"✓ Successfully downloaded {name} to {DATA_DIR / name}")
        return True
        
    except Exception as e:
        print(f"✗ Error downloading {name}: {str(e)}")
        return False


def setup_directories():
    """Create necessary directory structure"""
    directories = [
        DATA_DIR,
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "data" / "combined"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def create_env_template():
    """Create .env template file if it doesn't exist"""
    env_file = PROJECT_ROOT / ".env"
    env_template = PROJECT_ROOT / ".env.example"
    
    template_content = """# Roboflow API Configuration
ROBOFLOW_API_KEY=qFOIJQ6mzgDiG8HSWAmw

# Get your API key from: https://app.roboflow.com/settings/api

# Model Configuration
MODEL_SIZE=yolov8m.pt
BATCH_SIZE=16
IMAGE_SIZE=640
EPOCHS=50

# Training Configuration
DEVICE=0  # GPU device (0, 1, 2...) or 'cpu'
WORKERS=8
"""
    
    if not env_file.exists():
        with open(env_template, 'w') as f:
            f.write(template_content)
        print(f"\n✓ Created .env.example template")
        print(f"  Copy it to .env and add your Roboflow API key")
        print(f"  Get your key from: https://app.roboflow.com/settings/api\n")


def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("AERIAL SURVEILLANCE DATASET DOWNLOADER")
    print("UAV Person Dataset - militarypersons/uav-person-3")
    print("="*60)
    
    # Setup
    print("\n[1/3] Setting up directories...")
    setup_directories()
    
    print("\n[2/3] Checking environment configuration...")
    create_env_template()
    
    if not ROBOFLOW_API_KEY:
        print("\n⚠ WARNING: Roboflow API key not configured!")
        print("  1. Login to Roboflow: roboflow login")
        print("  2. Or get your API key from: https://app.roboflow.com/settings/api")
        print("  3. Create a .env file in the project root")
        print("  4. Add: ROBOFLOW_API_KEY=your_actual_key")
        print("\nExiting...")
        sys.exit(1)
    
    print(f"\n✓ Using API key: {ROBOFLOW_API_KEY[:10]}...")
    
    # Download dataset
    print("\n[3/3] Downloading UAV Person dataset...")
    print(f"Dataset URL: https://universe.roboflow.com/militarypersons/uav-person-3")
    
    success = download_dataset(
        workspace=DATASET_CONFIG["workspace"],
        project=DATASET_CONFIG["project"],
        version=DATASET_CONFIG["version"],
        name=DATASET_CONFIG["name"]
    )
    
    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    
    if success:
        print(f"✓ Success: {DATASET_CONFIG['name']}")
        print(f"\n✓ Dataset downloaded successfully to: {DATA_DIR / DATASET_CONFIG['name']}")
        print("\nNext steps:")
        print("  1. Run augmentation: python src/data_prep/augment_data.py")
        print("  2. Start training: python src/training/train_yolo.py")
    else:
        print(f"✗ Failed: {DATASET_CONFIG['name']}")
        print("\n⚠ Dataset download failed. Check errors above.")
        print("\nTroubleshooting:")
        print("  1. Ensure you're logged in: roboflow login")
        print("  2. Check your API key in .env file")
        print("  3. Verify dataset URL: https://universe.roboflow.com/militarypersons/uav-person-3")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
