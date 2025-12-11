# Defense Reconnaissance System: Soldier & Civilian Classification

## Project Overview

**Mission Critical Application for National Defense and Humanitarian Operations**

As tensions escalate and conflict looms, the ability to identify and classify individuals from aerial surveillance becomes a crucial defense capability. This computer vision system distinguishes soldiers from civilians in aerial imagery captured by drones, supporting reconnaissance and humanitarian operations.

This project demonstrates how Computer Science and Information Technology can contribute to national defense through intelligent aerial surveillance using deep learning and real-time video processing.

## Strategic Importance

- **Defense Operations**: Identify military personnel in conflict zones
- **Humanitarian Aid**: Distinguish civilians for protection and assistance
- **Reconnaissance**: Real-time intelligence gathering from drone footage
- **Border Security**: Monitor and classify individuals in sensitive areas
- **Search & Rescue**: Locate personnel in disaster zones

## Features

- **✅ Soldier/Civilian Classification**: Binary classification with YOLOv8 deep learning
- **✅ Real-time Bounding Boxes**: Visual identification with class labels ("Soldier", "Civilian")
- **✅ Video Stream Integration**: Process drone footage simulation and live streams
- **✅ Electron Desktop Application**: Military-grade interface for reconnaissance operations
- **✅ Performance Metrics**: Precision, recall, and mAP evaluation for mission readiness

## Project Structure

```
FINAL-CSC/
├── data/                      # Dataset directory
│   ├── raw/                   # Raw downloaded datasets
│   ├── processed/             # Augmented and preprocessed data
│   └── datasets.yaml          # Dataset configuration for YOLO
├── models/                    # Trained model weights
│   └── best.pt                # Best performing model
├── src/                       # Source code
│   ├── data_prep/             # Data preparation scripts
│   ├── training/              # Model training scripts
│   ├── detection/             # Real-time detection system
│   └── utils/                 # Utility functions
├── electron-app/              # Electron desktop application
│   ├── main.js
│   ├── renderer.js
│   └── index.html
├── notebooks/                 # Jupyter notebooks for experimentation
├── tests/                     # Test scripts and results
├── docs/                      # Documentation and reports
├── requirements.txt           # Python dependencies
├── package.json               # Node.js dependencies
└── README.md                  # This file
```

## Dataset Sources

1. **UAV Person Dataset**: https://universe.roboflow.com/militarypersons/uav-person-3
2. **Combatant Dataset**: https://universe.roboflow.com/minwoo/combatant-dataset
3. **Soldiers Detection**: https://universe.roboflow.com/xphoenixua-nlncq/soldiers-detection-spf
4. **Look Down Folks**: https://universe.roboflow.com/folks/look-down-folks

## Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- CUDA-capable GPU (recommended)
- Git

### Setup

1. Clone or navigate to the project directory:
```bash
cd "c:\Users\Kurt\Desktop\FINAL-CSC"
```

2. Create Python virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Install Node.js dependencies for Electron app:
```bash
cd electron-app
npm install
cd ..
```

## Usage

### 1. Dataset Preparation

Download datasets from Roboflow:
```bash
python src/data_prep/download_datasets.py
```

Augment and preprocess data:
```bash
python src/data_prep/augment_data.py
```

### 2. Model Training

Train YOLOv8 model:
```bash
python src/training/train_yolo.py --epochs 100 --batch 16 --img 640
```

Evaluate model performance:
```bash
python src/training/evaluate.py --weights models/best.pt
```

### 3. Real-time Detection

Run detection on video file:
```bash
python src/detection/detect_video.py --source path/to/video.mp4 --weights models/best.pt
```

Run detection on webcam/drone stream:
```bash
python src/detection/detect_stream.py --source 0 --weights models/best.pt
```

### 4. Electron Application

Launch the desktop application:
```bash
cd electron-app
npm start
```

## Model Performance Metrics

- **Precision**: Accuracy of positive predictions
- **Recall**: Ability to find all positive instances
- **mAP (mean Average Precision)**: Overall detection quality
- **Inference Speed**: FPS on target hardware

Results will be saved in `tests/results/`

## Technologies Used

- **YOLOv8**: Real-time object detection
- **Roboflow**: Dataset management and augmentation
- **OpenCV**: Video processing and visualization
- **PyTorch**: Deep learning framework
- **Electron**: Cross-platform desktop application
- **Python**: Backend processing
- **Node.js**: Application runtime

## Ethical Considerations

⚠️ **IMPORTANT**: This project is strictly educational and conceptual.

- Not intended for real-life military application without proper ethical evaluation
- Requires government oversight and ethical review for deployment
- Respects privacy and human rights considerations
- Acknowledges potential for misuse and dual-use concerns
- Emphasizes humanitarian applications (search & rescue, disaster response)

## Future Enhancements

- [ ] Multi-class detection (vehicles, weapons, etc.)
- [ ] Thermal imaging integration
- [ ] Edge deployment for on-device processing
- [ ] Real drone integration with DJI SDK
- [ ] Advanced tracking and trajectory prediction
- [ ] Cloud-based processing pipeline

## Contributing

This is an educational project. Contributions should align with ethical guidelines and educational purposes.

## License

For educational purposes only. See LICENSE file for details.

## Acknowledgments

- Roboflow community for dataset curation
- Ultralytics for YOLOv8 framework
- OpenCV community
- Academic institutions supporting computer vision research

## Contact

For questions or collaboration: [Your Contact Information]

---

**Disclaimer**: This system is a prototype for educational demonstration. Real-world deployment requires extensive validation, ethical review, legal compliance, and professional oversight.
