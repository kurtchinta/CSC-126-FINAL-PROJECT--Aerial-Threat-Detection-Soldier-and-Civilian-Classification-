# CSC 126 Final Project: Aerial Threat Detection - Soldier and Civilian Classification

## Team 6
- **RESERVA**
- **FELISILDA**
- **NONAN**

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Requirements](#system-requirements)
3. [Complete Installation Guide](#complete-installation-guide)
4. [Running the Application](#running-the-application)
5. [Project Structure](#project-structure)
6. [Features](#features)
7. [Dependencies](#dependencies)
8. [Troubleshooting](#troubleshooting)
9. [Dataset Sources](#dataset-sources)
10. [Ethical Considerations](#ethical-considerations)

---

## Project Overview

This project develops a computer vision system that distinguishes soldiers from civilians in aerial imagery captured by drones, supporting reconnaissance and humanitarian operations. The system uses YOLOv8 deep learning architecture for real-time object detection and classification.

### Key Capabilities
- **Real-time Detection**: Process video files and live camera streams
- **Classification**: Distinguish between soldiers and civilians with bounding boxes
- **Desktop Application**: Professional Electron-based interface
- **Cumulative Statistics**: Track total detections throughout processing

---

## System Requirements

### Minimum Requirements
| Component | Requirement |
|-----------|-------------|
| **Operating System** | Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+) |
| **Python** | 3.8, 3.9, 3.10, or 3.11 (3.12+ may have compatibility issues) |
| **Node.js** | 16.x or higher (LTS version recommended) |
| **RAM** | 8 GB minimum (16 GB recommended) |
| **Storage** | 5 GB free space |
| **GPU** | Optional but recommended for faster processing |

### Verify Your System
```bash
# Check Python version
python --version

# Check Node.js version
node --version

# Check npm version
npm --version
```

---

## Complete Installation Guide

### Step 1: Clone or Download the Project

**Option A: Clone with Git**
```bash
git clone https://github.com/kurtchinta/CSC-126-FINAL-PROJECT--Aerial-Threat-Detection-Soldier-and-Civilian-Classification-.git
cd CSC-126-FINAL-PROJECT--Aerial-Threat-Detection-Soldier-and-Civilian-Classification-
```

**Option B: Download ZIP**
1. Download the project ZIP file
2. Extract to your desired location
3. Open terminal/command prompt and navigate to the extracted folder

### Step 2: Set Up Python Environment

**Windows (Command Prompt or PowerShell):**
```bash
# Navigate to project directory
cd "path\to\FINAL-CSC"

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate

# Verify activation (should show (venv) in prompt)
```

**macOS/Linux (Terminal):**
```bash
# Navigate to project directory
cd path/to/FINAL-CSC

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify activation (should show (venv) in prompt)
```

### Step 3: Install Python Dependencies

```bash
# Ensure virtual environment is activated (you should see (venv) in your prompt)

# Upgrade pip first
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

**Expected installation time:** 5-15 minutes depending on internet speed

**Verify installation:**
```bash
python -c "import torch; import ultralytics; import cv2; import flask; print('All packages installed successfully!')"
```

### Step 4: Install Node.js Dependencies for Desktop App

```bash
# Navigate to electron-app folder
cd electron-app

# Install Node.js packages
npm install

# Return to project root
cd ..
```

**Expected installation time:** 2-5 minutes

### Step 5: Verify Model File Exists

The trained model file should be located at:
```
backend/civilian_soldier_working/runs/train/custom_aerial_detection/weights/best.pt
```

**To verify:**
```bash
# Windows
dir backend\civilian_soldier_working\runs\train\custom_aerial_detection\weights\

# macOS/Linux
ls backend/civilian_soldier_working/runs/train/custom_aerial_detection/weights/
```

If `best.pt` is missing, check `data/best.pt` as an alternative location.

---

## Running the Application

### Option 1: Desktop Application (Recommended)

This is the primary way to use the system with a graphical interface.

```bash
# Make sure you're in the project root directory
# Activate virtual environment first
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Navigate to electron-app
cd electron-app

# Start the application
npm start
```

**Using the Desktop App:**
1. **Select Video/Stream Tab**: Choose between video file or live stream
2. **Browse Video**: Click "Browse" to select a video file (.mp4, .avi, .mov)
3. **Adjust Settings**: Set confidence threshold (default: 0.55) and IOU threshold (default: 0.40)
4. **Start Detection**: Click "Start Detection" to begin processing
5. **View Results**: Watch real-time detection with bounding boxes
6. **Stop/Continue**: Use Stop to pause, then Continue or New Detection

### Option 2: Command Line - Video File Detection

```bash
# Activate virtual environment
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Run detection on a video file
python src/detection/detect_video.py --source "path/to/your/video.mp4" --conf 0.5
```

### Option 3: Command Line - Live Stream/Webcam

```bash
# Activate virtual environment
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Run detection on webcam (0 = default camera)
python src/detection/detect_stream.py --source 0 --conf 0.5

# Run detection on RTSP stream
python src/detection/detect_stream.py --source "rtsp://your-stream-url" --conf 0.5
```

---

## Project Structure

```
FINAL-CSC/
├── backend/                          # Backend models and datasets
│   └── civilian_soldier_working/
│       ├── yolo11n.pt                # Base YOLO11n model
│       ├── yolov8n.pt                # Alternative YOLOv8 model
│       ├── dataset.yaml              # Dataset configuration
│       ├── FINAL_CSC_AERIAL.ipynb    # Training notebook
│       ├── runs/                     # Training outputs
│       │   └── train/
│       │       └── custom_aerial_detection/
│       │           └── weights/
│       │               └── best.pt   # ★ TRAINED MODEL FILE
│       ├── train/                    # Training images and labels
│       ├── val/                      # Validation images and labels
│       └── test/                     # Test images and labels
│
├── data/                             # Additional data files
│   ├── best.pt                       # Alternative model location
│   └── last.pt                       # Last checkpoint
│
├── src/                              # Source code
│   ├── __init__.py
│   ├── data_prep/                    # Data preparation scripts
│   │   ├── __init__.py
│   │   ├── download_datasets.py
│   │   ├── augment_data.py
│   │   └── utils.py
│   ├── training/                     # Model training scripts
│   │   ├── __init__.py
│   │   ├── train_yolo.py
│   │   └── evaluate.py
│   └── detection/                    # Detection system
│       ├── __init__.py
│       ├── detect_video.py           # Video file processing
│       └── detect_stream.py          # Live stream processing
│
├── electron-app/                     # Desktop application
│   ├── main.js                       # Electron main process
│   ├── renderer.js                   # UI logic and interactions
│   ├── index.html                    # User interface
│   └── package.json                  # Node.js dependencies
│
├── docs/                             # Documentation
│   ├── QUICKSTART.md
│   ├── SETUP.md
│   └── REPORT_TEMPLATE.md
│
├── output/                           # Output files
│   └── videos/                       # Processed video outputs
│
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── LICENSE                           # License information
└── test_*.py                         # Test scripts
```

---

## Features

### Desktop Application Features
- **Video Tab**: Load and process video files (.mp4, .avi, .mov, .mkv)
- **Stream Tab**: Connect to webcam (0, 1) or RTSP streams
- **Real-time Display**: Side-by-side original and detected frames
- **Statistics Panel**: Live count of civilians, soldiers, and total detections
- **Adjustable Parameters**: Confidence and IOU thresholds
- **Detection Controls**: Start, Stop, Continue, New Detection options
- **Log Output**: Real-time processing logs and status updates

### Detection Capabilities
- **Soldier Detection**: Red bounding boxes with confidence scores
- **Civilian Detection**: Green bounding boxes with confidence scores
- **Cumulative Counting**: Running total of all detections
- **FPS Display**: Real-time processing speed indicator
- **Progress Tracking**: Frame count and completion percentage

---

## Dependencies

### Python Libraries (requirements.txt)

| Category | Package | Version | Purpose |
|----------|---------|---------|---------|
| **Deep Learning** | torch | ≥2.0.0 | PyTorch framework |
| | torchvision | ≥0.15.0 | Image processing |
| | ultralytics | ≥8.0.0 | YOLO implementation |
| **Computer Vision** | opencv-python | ≥4.8.0 | Video processing |
| | opencv-contrib-python | ≥4.8.0 | Additional CV functions |
| **Data Processing** | numpy | ≥1.24.0 | Numerical operations |
| | pandas | ≥2.0.0 | Data manipulation |
| | Pillow | ≥10.0.0 | Image handling |
| **Visualization** | matplotlib | ≥3.7.0 | Plotting |
| | seaborn | ≥0.12.0 | Statistical visualization |
| **Web Framework** | flask | ≥2.3.0 | Video streaming server |
| | flask-cors | ≥4.0.0 | Cross-origin support |
| **Utilities** | tqdm | ≥4.65.0 | Progress bars |
| | pyyaml | ≥6.0 | Configuration files |
| | requests | ≥2.31.0 | HTTP requests |

### Node.js Packages (electron-app/package.json)

| Package | Version | Purpose |
|---------|---------|---------|
| electron | ^28.0.0 | Desktop application framework |
| electron-builder | ^24.9.0 | Application packaging |
| electron-store | ^8.1.0 | Local storage |

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: "Python not found" or wrong version
```bash
# Check installed Python versions
python --version
python3 --version

# Use specific version if needed
python3.10 -m venv venv
```

#### Issue 2: Virtual environment not activating
```bash
# Windows PowerShell - may need execution policy change
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then try activation again
.\venv\Scripts\activate
```

#### Issue 3: Package installation fails
```bash
# Upgrade pip first
pip install --upgrade pip

# Install packages one at a time if bulk install fails
pip install torch torchvision
pip install ultralytics
pip install opencv-python flask
```

#### Issue 4: "Model not found" error
- Verify model exists at: `backend/civilian_soldier_working/runs/train/custom_aerial_detection/weights/best.pt`
- Alternative: Copy `data/best.pt` to the expected location
- Or update model path in the application settings

#### Issue 5: Electron app won't start
```bash
# Clear node_modules and reinstall
cd electron-app
rm -rf node_modules  # Linux/Mac
rmdir /s /q node_modules  # Windows
npm install
npm start
```

#### Issue 6: CUDA/GPU errors
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, the system will use CPU (slower but functional)
```

#### Issue 7: Video not displaying
- Ensure video format is supported (.mp4, .avi, .mov)
- Check if Flask server started (look for port 5000 messages in log)
- Try a different video file

### Getting Help
1. Check the log output in the application for specific error messages
2. Run test scripts to verify installation:
   ```bash
   python test_environment.py
   python test_model.py
   ```

---

---

## Dataset Sources

The project utilizes publicly available datasets from Roboflow:

| Dataset | URL | Purpose |
|---------|-----|---------|
| UAV Person Dataset | https://universe.roboflow.com/militarypersons/uav-person-3 | Primary aerial person detection |
| Combatant Dataset | https://universe.roboflow.com/minwoo/combatant-dataset | Military personnel detection |
| Soldiers Detection | https://universe.roboflow.com/xphoenixua-nlncq/soldiers-detection-spf | Soldier-specific detection |
| Look Down Folks | https://universe.roboflow.com/folks/look-down-folks | Top-down civilian detection |

---

## Training the Model (Optional)

If you need to retrain the model:

### Option A: Using Python Script
```bash
python src/training/train_yolo.py --epochs 100 --batch 16 --img 640
```

### Option B: Using Jupyter Notebook
```bash
jupyter notebook backend/civilian_soldier_working/FINAL_CSC_AERIAL.ipynb
```

### Option C: Google Colab (Recommended for free GPU)
1. Upload `FINAL_CSC_AERIAL.ipynb` to Google Colab
2. Enable GPU runtime (Runtime → Change runtime type → GPU)
3. Run all cells

---

## Ethical Considerations

**IMPORTANT**: This project is strictly educational and conceptual.

- Developed solely for academic learning and demonstration purposes
- Not intended for real-life military application without proper ethical evaluation
- Requires government oversight, legal compliance, and ethical review before any deployment
- Must respect privacy rights and human rights considerations
- Primary focus on humanitarian applications (search and rescue, disaster response)

---

## Project Deliverables

| Deliverable | Location |
|-------------|----------|
| Trained YOLOv8 Model | `backend/civilian_soldier_working/runs/train/custom_aerial_detection/weights/best.pt` |
| Desktop Application | `electron-app/` |
| Source Code | `src/` |
| Training Notebook | `backend/civilian_soldier_working/FINAL_CSC_AERIAL.ipynb` |
| Documentation | `docs/`, `README.md` |

---

## Quick Reference Commands

```bash
# === SETUP ===
python -m venv venv                    # Create virtual environment
.\venv\Scripts\activate                # Activate (Windows)
source venv/bin/activate               # Activate (Mac/Linux)
pip install -r requirements.txt        # Install Python packages
cd electron-app && npm install         # Install Node packages

# === RUN APPLICATION ===
cd electron-app && npm start           # Start desktop app

# === COMMAND LINE DETECTION ===
python src/detection/detect_video.py --source video.mp4 --conf 0.5
python src/detection/detect_stream.py --source 0 --conf 0.5

# === TESTING ===
python test_environment.py             # Test Python environment
python test_model.py                   # Test model loading
```

---

## License

This project is developed for educational purposes as part of CSC 126 coursework. See LICENSE file for details.

---

## Team Contact

**Team 6 - CSC 126 Final Project**
- RESERVA, Kurt Daniel M.
- FELISILDA
- NONAN

**Repository:** https://github.com/kurtchinta/CSC-126-FINAL-PROJECT--Aerial-Threat-Detection-Soldier-and-Civilian-Classification-.git

---

## Disclaimer

This system is a prototype developed for educational demonstration purposes only. It is not intended for operational military or surveillance use. The developers acknowledge the sensitive nature of this technology and emphasize its educational purpose in learning computer vision, deep learning, and system integration concepts.
