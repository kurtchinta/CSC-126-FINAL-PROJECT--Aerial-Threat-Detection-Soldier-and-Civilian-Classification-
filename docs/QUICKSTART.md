# Quick Start Guide

## Fast Track Setup (15 minutes)

### 1. Install Dependencies (5 min)

```bash
# Navigate to project
cd "c:\Users\Kurt\Desktop\FINAL-CSC"

# Create Python virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install Python packages
pip install -r requirements.txt

# Install Node.js packages for Electron app
cd electron-app
npm install
cd ..
```

### 2. Configure Roboflow API (2 min)

```bash
# Create .env file
echo "ROBOFLOW_API_KEY=your_api_key_here" > .env
```

Get your API key from: https://app.roboflow.com/settings/api

### 3. Download Datasets (5 min)

```bash
python src/data_prep/download_datasets.py
```

### 4. Prepare Data (3 min)

```bash
python src/data_prep/augment_data.py
```

---

## Training Options

### Option A: Full Training (4-8 hours on GPU)

```bash
python src/training/train_yolo.py
```

### Option B: Use Pre-trained Model (Skip training)

If you have a pre-trained model:
1. Place `best.pt` in `models/` folder
2. Skip to Testing section

### Option C: Google Colab Training (Recommended for no GPU)

Use Google Colab for free GPU training - see `docs/COLAB_TRAINING.md`

---

## Testing

### Test on Video

```bash
python src/detection/detect_video.py --source test_video.mp4
```

### Test on Webcam

```bash
python src/detection/detect_stream.py --source 0
```

### Launch Desktop App

```bash
cd electron-app
npm start
```

---

## Quick Commands Reference

```bash
# Activate virtual environment
.\venv\Scripts\activate

# Download datasets
python src/data_prep/download_datasets.py

# Augment data
python src/data_prep/augment_data.py

# Train model
python src/training/train_yolo.py

# Evaluate model
python src/training/evaluate.py

# Detect on video
python src/detection/detect_video.py --source video.mp4

# Detect on stream
python src/detection/detect_stream.py --source 0

# Launch Electron app
cd electron-app && npm start
```

---

## Expected Results

### Dataset Stats
- Training images: ~3,000-5,000
- Validation images: ~500-1,000
- Test images: ~500-1,000
- Classes: Soldier, Civilian, Person

### Model Performance (Expected)
- mAP50: 0.70-0.85
- Precision: 0.75-0.90
- Recall: 0.70-0.85
- Inference Speed: 20-60 FPS (GPU)

### Output Locations
- Trained models: `models/best.pt`
- Training logs: `runs/aerial_surveillance/`
- Detection output: `output/videos/` or `output/streams/`
- Test results: `tests/results/`

---

## Troubleshooting Quick Fixes

**CUDA not found?**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Out of memory?**
Edit `.env` file: `BATCH_SIZE=4`

**Roboflow API error?**
Check `.env` file has correct API key

**Slow training?**
Use Google Colab or smaller model: `MODEL_SIZE=yolov8s.pt`

---

## Project Structure Overview

```
FINAL-CSC/
├── data/               # Datasets
├── models/             # Trained models
├── src/               # Source code
│   ├── data_prep/     # Data preparation
│   ├── training/      # Training scripts
│   └── detection/     # Detection scripts
├── electron-app/      # Desktop application
├── runs/              # Training runs
├── output/            # Detection outputs
├── tests/             # Test results
└── docs/              # Documentation
```

---

## Next Steps After Setup

1. ✅ Review dataset statistics
2. ✅ Monitor training progress
3. ✅ Evaluate model performance
4. ✅ Test on sample videos
5. ✅ Create project documentation
6. ✅ Prepare presentation/report

---

## Need Help?

See detailed guides in `docs/` folder:
- `SETUP.md` - Detailed setup instructions
- `TRAINING.md` - Training guide
- `DEPLOYMENT.md` - Deployment instructions
- `API.md` - API documentation

---

## Educational Use Only

⚠️ This project is for **educational purposes only**. Real-world military or surveillance applications require proper ethical review, legal compliance, and government authorization.
