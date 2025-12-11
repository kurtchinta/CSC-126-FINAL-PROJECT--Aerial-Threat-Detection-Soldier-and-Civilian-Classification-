"""
Quick test to verify the trained model loads correctly
"""
from pathlib import Path
from ultralytics import YOLO

# Model path
model_path = Path("backend/civilian_soldier_working/yolov8n.pt")

print(f"Testing model: {model_path}")
print(f"Model exists: {model_path.exists()}")

if model_path.exists():
    try:
        # Load model
        model = YOLO(str(model_path))
        
        print("\n✓ Model loaded successfully!")
        print(f"  Model type: {type(model)}")
        print(f"  Classes: {model.names}")
        print(f"  Number of classes: {len(model.names)}")
        
        # Show model info
        print("\nModel Summary:")
        model.info()
        
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
else:
    print("\n✗ Model file not found!")
