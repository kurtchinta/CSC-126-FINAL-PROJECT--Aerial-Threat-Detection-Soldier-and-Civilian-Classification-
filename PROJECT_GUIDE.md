# Aerial Surveillance Classification System
## Complete Project Guide - From Setup to Deployment

---

## 📋 Table of Contents

1. [Quick Overview](#quick-overview)
2. [Project Structure](#project-structure)
3. [Step-by-Step Workflow](#step-by-step-workflow)
4. [Command Reference](#command-reference)
5. [Troubleshooting](#troubleshooting)
6. [Project Deliverables](#project-deliverables)
7. [Grading Criteria Checklist](#grading-criteria-checklist)

---

## 🎯 Quick Overview

**What You're Building:**
An AI-powered system that can detect and classify soldiers, civilians, and persons in aerial/drone footage using YOLOv8 deep learning.

**What You'll Learn:**
- Computer vision and object detection
- Deep learning model training
- Real-time video processing
- Desktop application development
- Dataset preparation and augmentation

**Time Estimate:**
- Setup: 30 minutes
- Data preparation: 1 hour
- Training: 4-8 hours (GPU) or 24-48 hours (CPU)
- Testing & deployment: 2 hours
- Documentation: 2-4 hours

**Hardware Requirements:**
- Minimum: 8GB RAM, any modern CPU
- Recommended: 16GB RAM, NVIDIA GPU with 4GB+ VRAM

---

## 📁 Project Structure

```
FINAL-CSC/
│
├── 📂 data/                         # All datasets
│   ├── raw/                         # Downloaded from Roboflow
│   ├── processed/                   # Augmented datasets
│   └── combined/                    # Final combined dataset
│       └── dataset.yaml             # YOLO config file
│
├── 📂 models/                       # Trained model weights
│   └── best.pt                      # Best performing model
│
├── 📂 src/                          # Source code
│   ├── 📂 data_prep/
│   │   ├── download_datasets.py    # Download from Roboflow
│   │   ├── augment_data.py         # Data augmentation
│   │   └── utils.py                # Helper functions
│   │
│   ├── 📂 training/
│   │   ├── train_yolo.py           # Train YOLOv8 model
│   │   └── evaluate.py             # Evaluate performance
│   │
│   └── 📂 detection/
│       ├── detect_video.py         # Process video files
│       └── detect_stream.py        # Live stream detection
│
├── 📂 electron-app/                 # Desktop application
│   ├── main.js                      # Electron main process
│   ├── renderer.js                  # UI logic
│   ├── index.html                   # User interface
│   └── package.json                 # Node dependencies
│
├── 📂 runs/                         # Training outputs
│   └── aerial_surveillance/        # Training logs, plots
│
├── 📂 output/                       # Detection outputs
│   ├── videos/                      # Processed videos
│   └── streams/                     # Stream recordings
│
├── 📂 tests/                        # Test results
│   └── results/                     # Evaluation metrics
│
├── 📂 docs/                         # Documentation
│   ├── SETUP.md                     # Detailed setup guide
│   ├── QUICKSTART.md                # Quick start guide
│   └── REPORT_TEMPLATE.md           # Project report template
│
├── 📄 README.md                     # Main documentation
├── 📄 requirements.txt              # Python dependencies
├── 📄 .env.example                  # Config template
├── 📄 .gitignore                    # Git ignore rules
├── 📄 LICENSE                       # License file
└── 📄 test_environment.py           # Environment checker
```

---

## 🚀 Step-by-Step Workflow

### Phase 1: Environment Setup (30 minutes)

#### Step 1: Open PowerShell/Terminal

```powershell
# Navigate to project
cd "c:\Users\Kurt\Desktop\FINAL-CSC"
```

#### Step 2: Create Python Virtual Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1

# If you get execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Step 3: Install Python Dependencies

```powershell
# Upgrade pip
pip install --upgrade pip

# Install all packages (this takes 5-10 minutes)
pip install -r requirements.txt
```

#### Step 4: Test Environment

```powershell
python test_environment.py
```

**Expected output:** All checks should pass ✓

#### Step 5: Configure Roboflow API

```powershell
# Copy template
copy .env.example .env

# Edit .env file and add your API key
notepad .env
```

Get API key from: https://app.roboflow.com/settings/api

Your `.env` should look like:
```env
ROBOFLOW_API_KEY=YOUR_ACTUAL_KEY_HERE
MODEL_SIZE=yolov8m.pt
BATCH_SIZE=16
IMAGE_SIZE=640
EPOCHS=100
DEVICE=0
WORKERS=8
```

#### Step 6: Setup Electron App

```powershell
cd electron-app
npm install
cd ..
```

---

### Phase 2: Data Preparation (1-2 hours)

#### Step 7: Download Datasets

```powershell
python src/data_prep/download_datasets.py
```

**What happens:**
- Downloads 4 datasets from Roboflow
- ~3,000-5,000 images total
- Takes 10-20 minutes depending on internet speed

**Output location:** `data/raw/`

#### Step 8: Augment and Combine Data

```powershell
python src/data_prep/augment_data.py
```

**What happens:**
- Applies augmentations (rotation, flip, brightness, etc.)
- Creates 3x more training images
- Combines all datasets
- Creates YOLO format labels
- Takes 20-40 minutes

**Output location:** `data/combined/`

#### Step 9: Verify Dataset

```powershell
# Check dataset statistics
python -c "from src.data_prep.utils import analyze_dataset; from pathlib import Path; analyze_dataset(Path('data/combined'))"
```

---

### Phase 3: Model Training (4-48 hours)

#### Step 10: Start Training

```powershell
python src/training/train_yolo.py
```

**Training duration:**
- **RTX 3080**: 1-3 hours
- **GTX 1660**: 4-8 hours  
- **CPU only**: 24-48 hours

**What to monitor:**
- Loss should decrease over time
- mAP should increase
- Watch for overfitting (validation loss increases)

**Training outputs:**
- Live progress in terminal
- TensorBoard logs: `runs/aerial_surveillance/`
- Checkpoints saved every 10 epochs
- Best model: `models/best.pt`

#### Step 11: Monitor Training (Optional)

Open new terminal:
```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Start TensorBoard
tensorboard --logdir runs

# Open browser to: http://localhost:6006
```

#### Step 12: Training Tips

**If training is too slow:**
```env
# Edit .env file
BATCH_SIZE=8          # Reduce batch size
MODEL_SIZE=yolov8s.pt # Use smaller model
EPOCHS=50             # Fewer epochs
```

**If running out of memory:**
```env
BATCH_SIZE=4          # Smaller batches
CACHE=false           # Don't cache images
WORKERS=4             # Fewer workers
```

**Alternative: Use Google Colab**
- Free GPU access
- Upload dataset to Google Drive
- Run training notebook
- Download trained model

---

### Phase 4: Model Evaluation (30 minutes)

#### Step 13: Evaluate Model

```powershell
python src/training/evaluate.py --weights models/best.pt
```

**Evaluation includes:**
- Test set metrics (mAP, precision, recall)
- Inference speed (FPS)
- Confusion matrix
- Confidence distribution
- Sample predictions

**Results saved to:** `tests/results/`

#### Step 14: Review Results

```powershell
# Open results folder
explorer tests\results

# Check files:
# - evaluation_report.json
# - inference_speed.json
# - confidence_distribution.png
# - visualizations/*.jpg
```

**Good model indicators:**
- mAP50 > 0.70 (70%)
- Precision > 0.75 (75%)
- Recall > 0.70 (70%)
- FPS > 20 on GPU

---

### Phase 5: Detection Testing (1 hour)

#### Step 15: Test on Sample Video

```powershell
# Download sample drone video or use your own
# Place in project folder

python src/detection/detect_video.py --source sample_video.mp4 --weights models/best.pt
```

**Output:**
- Processed video with bounding boxes
- Detection statistics
- Saved to: `output/videos/`

#### Step 16: Test on Webcam

```powershell
python src/detection/detect_stream.py --source 0 --weights models/best.pt
```

**Controls:**
- **Q**: Quit
- **R**: Start/stop recording
- **S**: Save screenshot

#### Step 17: Launch Desktop Application

```powershell
cd electron-app
npm start
```

**Features:**
- Video mode: Select and process video files
- Stream mode: Live webcam/RTSP detection
- Adjust confidence/IOU thresholds
- Real-time statistics
- Easy-to-use interface

---

### Phase 6: Documentation (2-4 hours)

#### Step 18: Create Project Report

Use the template:
```powershell
# Copy template
copy docs\REPORT_TEMPLATE.md PROJECT_REPORT.md

# Edit and fill in your results
notepad PROJECT_REPORT.md
```

**Include:**
- Introduction and objectives
- Methodology and dataset details
- Training configuration
- Results and metrics
- Sample detection images
- Analysis and discussion
- Conclusion and future work

#### Step 19: Create Presentation

**Suggested slides:**
1. Title & Introduction
2. Problem Statement
3. Dataset Overview
4. YOLOv8 Architecture
5. Training Process
6. Results & Metrics
7. Live Demo
8. Challenges & Solutions
9. Ethical Considerations
10. Conclusion & Future Work

#### Step 20: Record Demo Video

**Show:**
1. Desktop application interface
2. Loading video file
3. Real-time detection
4. Classification results
5. Statistics display
6. Saving output

---

## 📝 Command Reference

### Essential Commands

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

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

# Detect on webcam
python src/detection/detect_stream.py --source 0

# Launch app
cd electron-app && npm start

# Test environment
python test_environment.py
```

### Advanced Commands

```powershell
# Custom training
python src/training/train_yolo.py --epochs 150 --batch 32

# Video with custom settings
python src/detection/detect_video.py --source video.mp4 --conf 0.3 --iou 0.5

# TensorBoard
tensorboard --logdir runs

# Check dataset stats
python -c "from src.data_prep.utils import analyze_dataset; from pathlib import Path; analyze_dataset(Path('data/combined'))"

# Build Electron app
cd electron-app && npm run build:win
```

---

## 🔧 Troubleshooting

### Common Issues & Solutions

#### 1. "CUDA not available"
**Solution:**
```powershell
# Install CUDA-enabled PyTorch
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 2. "Out of memory" during training
**Solution:** Edit `.env`:
```env
BATCH_SIZE=4
CACHE=false
```

#### 3. "Roboflow API error"
**Solution:**
- Check `.env` has correct API key
- Verify key at https://app.roboflow.com/settings/api
- Check internet connection

#### 4. "No module named 'ultralytics'"
**Solution:**
```powershell
# Ensure venv is activated
.\venv\Scripts\Activate.ps1
pip install ultralytics
```

#### 5. Electron app won't start
**Solution:**
```powershell
cd electron-app
rm -rf node_modules
npm install
npm start
```

#### 6. Python not found
**Solution:**
- Install Python 3.8+ from python.org
- Add to PATH during installation
- Restart terminal

#### 7. Slow training on CPU
**Solutions:**
- Use Google Colab (free GPU)
- Reduce epochs: `EPOCHS=50`
- Use smaller model: `MODEL_SIZE=yolov8n.pt`
- Consider cloud GPU (AWS, Azure)

---

## 📦 Project Deliverables

### What to Submit

#### 1. GitHub Repository
```
✅ All source code
✅ README.md
✅ Documentation
✅ .gitignore (no large files)
✅ requirements.txt
✅ LICENSE
```

#### 2. Trained Model
```
✅ Upload to Google Drive/OneDrive
✅ Share link in README
✅ Include model metadata
```

#### 3. Project Report (PDF)
```
✅ 10-15 pages
✅ Introduction & objectives
✅ Methodology
✅ Results with visualizations
✅ Analysis & discussion
✅ Conclusion
✅ References
```

#### 4. Presentation (PPT/PDF)
```
✅ 10-15 slides
✅ Clear visualizations
✅ Demo screenshots
✅ Key metrics highlighted
```

#### 5. Demo Video (Optional)
```
✅ 3-5 minutes
✅ Show application in action
✅ Explain features
✅ Upload to YouTube/Drive
```

---

## ✅ Grading Criteria Checklist

### Technical Implementation (40%)

- [ ] Dataset successfully downloaded and prepared
- [ ] Data augmentation applied
- [ ] Model trained on combined dataset
- [ ] Achieves reasonable performance (mAP > 0.60)
- [ ] Real-time detection working
- [ ] Desktop application functional
- [ ] Code is well-organized and documented

### Model Performance (20%)

- [ ] mAP50 > 0.70
- [ ] Precision > 0.75
- [ ] Recall > 0.70
- [ ] Inference speed documented
- [ ] Performance analyzed across classes

### Documentation (20%)

- [ ] Comprehensive README
- [ ] Setup instructions clear
- [ ] Code comments present
- [ ] Report follows template
- [ ] Results well-documented

### Presentation (10%)

- [ ] Clear and professional
- [ ] Technical content accurate
- [ ] Demo effective
- [ ] Questions answered well

### Innovation & Ethics (10%)

- [ ] Unique improvements/features
- [ ] Ethical considerations addressed
- [ ] Future work discussed
- [ ] Real-world applications considered

---

## 🎓 Tips for Success

### Before Submission

1. **Test Everything**
   - Run `test_environment.py`
   - Test all detection modes
   - Verify Electron app works
   - Check all documentation links

2. **Clean Up**
   - Remove large data files from git
   - Organize output folders
   - Update .gitignore
   - Check for hardcoded paths

3. **Document Results**
   - Screenshot key metrics
   - Save sample detections
   - Export training plots
   - Record demo video

4. **Review Checklist**
   - All requirements met
   - Code runs without errors
   - Documentation complete
   - Ethical considerations included

### During Demo/Presentation

1. **Have Backup Plan**
   - Pre-recorded video if live demo fails
   - Screenshots of results
   - Printed report

2. **Practice Demo**
   - Test on presentation computer
   - Know keyboard shortcuts
   - Prepare sample videos

3. **Anticipate Questions**
   - Why YOLOv8?
   - How does detection work?
   - What are limitations?
   - Ethical implications?

---

## 🌟 Going Above and Beyond

### Extra Credit Ideas

1. **Advanced Features**
   - Multi-object tracking
   - Heatmap visualization
   - Export to different formats
   - REST API for detection

2. **Additional Datasets**
   - Collect own drone footage
   - Annotate custom data
   - Test on different conditions

3. **Performance Optimization**
   - Model quantization
   - TensorRT optimization
   - Edge device deployment (Jetson)

4. **Enhanced UI**
   - More statistics
   - Graphical analytics
   - Configuration presets
   - Batch processing

5. **Research Component**
   - Compare model architectures
   - Ablation studies
   - Novel augmentation techniques

---

## 📞 Support

If you get stuck:

1. Check `docs/` folder for guides
2. Review error messages carefully
3. Test environment: `python test_environment.py`
4. Consult course materials
5. Ask instructor/TA

---

**Good luck with your project! 🚀**

Remember: This is an educational project. Focus on learning the concepts and building something you're proud of. The journey is more important than perfection.

---

*Last updated: December 2025*
