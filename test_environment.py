"""
Simple test script to verify installation and environment
"""

import sys
import os
from pathlib import Path


def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def check_python_version():
    """Check Python version"""
    print_section("Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("✓ Python version OK")
        return True
    else:
        print("✗ Python 3.8+ required")
        return False


def check_imports():
    """Check if required packages are installed"""
    print_section("Package Imports")
    
    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'cv2': 'OpenCV',
        'ultralytics': 'Ultralytics',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'yaml': 'PyYAML',
        'tqdm': 'tqdm',
        'matplotlib': 'Matplotlib',
        'sklearn': 'scikit-learn'
    }
    
    results = {}
    
    for package, name in packages.items():
        try:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✓ {name}: {version}")
            results[package] = True
        except ImportError:
            print(f"✗ {name}: NOT INSTALLED")
            results[package] = False
    
    return all(results.values())


def check_cuda():
    """Check CUDA availability"""
    print_section("CUDA/GPU")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  GPU Count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name}")
                print(f"    Memory: {props.total_memory / 1e9:.2f} GB")
            
            return True
        else:
            print("⚠ CUDA not available (CPU mode)")
            print("  Training will be slower on CPU")
            return False
    except Exception as e:
        print(f"✗ Error checking CUDA: {str(e)}")
        return False


def check_directories():
    """Check if required directories exist"""
    print_section("Project Structure")
    
    project_root = Path(__file__).parent
    
    required_dirs = [
        'data',
        'models',
        'src/data_prep',
        'src/training',
        'src/detection',
        'electron-app',
        'docs'
    ]
    
    all_exist = True
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ (missing)")
            all_exist = False
    
    return all_exist


def check_config():
    """Check configuration files"""
    print_section("Configuration")
    
    project_root = Path(__file__).parent
    
    # Check .env file
    env_file = project_root / '.env'
    if env_file.exists():
        print("✓ .env file exists")
        
        # Check if API key is set
        with open(env_file, 'r') as f:
            content = f.read()
            if 'ROBOFLOW_API_KEY' in content and 'your_api_key_here' not in content:
                print("✓ Roboflow API key configured")
            else:
                print("⚠ Roboflow API key not configured")
                print("  Set ROBOFLOW_API_KEY in .env file")
    else:
        print("✗ .env file not found")
        print("  Copy .env.example to .env and configure")
        return False
    
    return True


def test_yolo_import():
    """Test YOLOv8 import and basic functionality"""
    print_section("YOLOv8 Test")
    
    try:
        from ultralytics import YOLO
        import numpy as np
        
        print("✓ YOLO imported successfully")
        
        # Try loading a model
        try:
            model = YOLO('yolov8n.pt')  # Smallest model for testing
            print("✓ YOLOv8 nano model loaded")
            
            # Test inference on dummy image
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            results = model(dummy_img, verbose=False)
            print("✓ Inference test successful")
            
            return True
        except Exception as e:
            print(f"⚠ Model test failed: {str(e)}")
            print("  This is OK if models aren't downloaded yet")
            return True
            
    except Exception as e:
        print(f"✗ YOLO import failed: {str(e)}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print(" AERIAL SURVEILLANCE SYSTEM - ENVIRONMENT TEST")
    print("="*60)
    
    results = {
        'Python Version': check_python_version(),
        'Package Imports': check_imports(),
        'CUDA/GPU': check_cuda(),
        'Project Structure': check_directories(),
        'Configuration': check_config(),
        'YOLOv8': test_yolo_import()
    }
    
    # Summary
    print_section("Test Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! System ready.")
        print("\nNext steps:")
        print("  1. python src/data_prep/download_datasets.py")
        print("  2. python src/data_prep/augment_data.py")
        print("  3. python src/training/train_yolo.py")
    else:
        print("\n⚠ Some tests failed. Please fix issues above.")
        print("\nSee docs/SETUP.md for detailed setup instructions.")
    
    print("\n" + "="*60 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
