# Setup Guide - Aerial Surveillance Classification System

## Prerequisites

### Software Requirements
- **Python 3.8+** - [Download](https://www.python.org/downloads/)
- **Node.js 16+** - [Download](https://nodejs.org/)
- **Git** - [Download](https://git-scm.com/)
- **CUDA Toolkit 11.8+** (Optional, for GPU acceleration) - [Download](https://developer.nvidia.com/cuda-downloads)

### Hardware Requirements
- **Minimum**: 8GB RAM, Intel i5/AMD Ryzen 5, Integrated Graphics
- **Recommended**: 16GB RAM, Intel i7/AMD Ryzen 7, NVIDIA GPU (4GB+ VRAM)
- **Storage**: 10GB free space

## Installation Steps

### 1. Navigate to Project Directory

```bash
cd "c:\Users\Kurt\Desktop\FINAL-CSC"
```

### 2. Python Environment Setup

#### Create Virtual Environment
```bash
python -m venv venv
```

#### Activate Virtual Environment
**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.\venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

#### Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Roboflow API Configuration

1. Create a Roboflow account: [https://app.roboflow.com/](https://app.roboflow.com/)
2. Get your API key: [https://app.roboflow.com/settings/api](https://app.roboflow.com/settings/api)
3. Create `.env` file in project root:

```bash
# Copy template
cp .env.example .env

# Edit .env and add your API key
ROBOFLOW_API_KEY=your_actual_api_key_here
```

**.env file contents:**
```env
# Roboflow API Configuration
ROBOFLOW_API_KEY=your_api_key_here

# Model Configuration
MODEL_SIZE=yolov8m.pt
BATCH_SIZE=16
IMAGE_SIZE=640
EPOCHS=100

# Training Configuration
DEVICE=0  # GPU device (0, 1, 2...) or 'cpu'
WORKERS=8
```

### 4. Electron Application Setup

```bash
cd electron-app
npm install
cd ..
```

## Dataset Preparation

### Download Datasets from Roboflow

```bash
python src/data_prep/download_datasets.py
```

This will download:
- UAV Person Dataset
- Combatant Dataset
- Soldiers Detection Dataset
- Look Down Folks Dataset

### Augment and Process Data

```bash
python src/data_prep/augment_data.py
```

This creates augmented training data and combines all datasets.

## Model Training

### Basic Training

```bash
python src/training/train_yolo.py
```

### Training with Custom Parameters

```bash
# Edit .env file to change parameters
# Or use command line (modify script to accept args)
```

Training typically takes:
- **CPU**: 24-48 hours
- **GPU (GTX 1660)**: 4-8 hours
- **GPU (RTX 3080)**: 1-3 hours

### Monitor Training

Training progress is saved in:
- `runs/aerial_surveillance/` - TensorBoard logs, plots, weights
- `logs/` - Training logs
- `models/best.pt` - Best model weights

### View TensorBoard

```bash
tensorboard --logdir runs
```

Open browser to `http://localhost:6006`

## Model Evaluation

### Evaluate Trained Model

```bash
python src/training/evaluate.py --weights models/best.pt
```

Results saved to `tests/results/`:
- Performance metrics (mAP, precision, recall)
- Inference speed analysis
- Prediction visualizations
- Confidence distribution plots

## Running Detection

### Video File Detection

```bash
python src/detection/detect_video.py --source path/to/video.mp4 --weights models/best.pt
```

**Parameters:**
- `--source`: Path to video file
- `--weights`: Path to model weights
- `--conf`: Confidence threshold (default: 0.25)
- `--iou`: IOU threshold (default: 0.45)
- `--output`: Output video path
- `--no-display`: Disable video display

### Live Stream Detection

```bash
# Webcam
python src/detection/detect_stream.py --source 0

# RTSP Stream
python src/detection/detect_stream.py --source "rtsp://camera_url"
```

**Controls during live stream:**
- `Q` - Quit
- `R` - Start/Stop recording
- `S` - Save screenshot

## Electron Desktop Application

### Launch Application

```bash
cd electron-app
npm start
```

### Build Executable

**Windows:**
```bash
npm run build:win
```

**Mac:**
```bash
npm run build:mac
```

**Linux:**
```bash
npm run build:linux
```

Executable will be in `electron-app/dist/`

## Troubleshooting

### CUDA Not Found
- Install CUDA Toolkit from NVIDIA
- Verify: `nvidia-smi` in terminal
- Reinstall PyTorch with CUDA support:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```

### Out of Memory (GPU)
- Reduce batch size in `.env`: `BATCH_SIZE=8` or `BATCH_SIZE=4`
- Use smaller model: `MODEL_SIZE=yolov8s.pt` or `yolov8n.pt`

### Roboflow Download Fails
- Check API key in `.env`
- Verify internet connection
- Check Roboflow quota limits

### Electron App Won't Start
- Ensure Python is in PATH
- Verify virtual environment is activated
- Check model exists at `models/best.pt`

### Slow Inference
- Use GPU if available
- Reduce image size: lower resolution videos
- Use smaller model (yolov8n.pt)
- Close other applications

## Using Pre-trained Models

If you don't want to train from scratch:

1. Download pre-trained model from Google Drive/GitHub
2. Place `best.pt` in `models/` directory
3. Skip training steps
4. Go directly to detection

## Google Colab Training (Alternative)

For cloud training without local GPU:

1. Open Google Colab
2. Upload project files
3. Run training notebook (create from `notebooks/`)
4. Download trained model
5. Use locally for inference

## Next Steps

After successful setup:

1. **Test with sample video** - Use detection scripts
2. **Fine-tune model** - Adjust confidence thresholds
3. **Create documentation** - Document your findings
4. **Prepare presentation** - Showcase results
5. **GitHub repository** - Push code to version control

## Support Resources

- **Ultralytics YOLOv8 Docs**: [https://docs.ultralytics.com/](https://docs.ultralytics.com/)
- **Roboflow Documentation**: [https://docs.roboflow.com/](https://docs.roboflow.com/)
- **OpenCV Documentation**: [https://docs.opencv.org/](https://docs.opencv.org/)
- **Electron Documentation**: [https://www.electronjs.org/docs](https://www.electronjs.org/docs)

## Ethical Reminder

⚠️ This is an **educational project only**. Real-world deployment requires:
- Ethical review board approval
- Legal compliance verification
- Privacy impact assessment
- Government oversight
- Proper authorization

Use responsibly and within legal/ethical boundaries.
