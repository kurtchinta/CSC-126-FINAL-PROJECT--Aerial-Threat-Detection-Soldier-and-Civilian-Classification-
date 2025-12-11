# CSC 126 Final Project: Aerial Threat Detection - Soldier and Civilian Classification

## Team 6
- **RESERVA, Kurt Daniel M.**
- **FELISILDA**
- **NONAN**

---

## Project Overview

As tensions escalate and conflict looms, the ability to identify and classify individuals from aerial surveillance becomes a crucial defense capability. This project develops a computer vision system that distinguishes soldiers from civilians in aerial imagery captured by drones, supporting reconnaissance and humanitarian operations.

As students of Computer Science and Information Technology, this project demonstrates how our field can contribute to national defense through intelligent aerial surveillance using deep learning and real-time video processing.

## Objectives

1. **Build an image classification model** to distinguish soldiers from civilians in aerial images
2. **Utilize drone footage** from publicly available datasets (Roboflow)
3. **Integrate the trained model** with video stream processing (drone footage simulation)
4. **Provide a system prototype** that visualizes classifications in real-time through an Electron desktop application

## Strategic Importance

This system addresses critical needs in:
- **Defense Operations**: Identifying military personnel in conflict zones
- **Humanitarian Aid**: Distinguishing civilians for protection and assistance
- **Reconnaissance**: Real-time intelligence gathering from drone footage
- **Border Security**: Monitoring and classifying individuals in sensitive areas
- **Search and Rescue**: Locating personnel in disaster zones

## Key Features

- Soldier/Civilian Classification using YOLO11n deep learning architecture
- Real-time bounding boxes with class labels ("Soldier", "Civilian")
- Video stream integration for drone footage simulation and live camera feeds
- Electron desktop application with professional interface
- Custom trained model on UAV aerial surveillance datasets
- Performance evaluation using precision, recall, and mAP metrics
- Adjustable confidence thresholds for different operational scenarios

## Project Structure

```
FINAL-CSC/
├── backend/                   # Backend models and datasets
│   └── civilian_soldier_working/
│       ├── yolo11n.pt         # YOLO11n model (default)
│       ├── dataset.yaml       # Dataset configuration
│       ├── train/             # Training images and labels
│       ├── val/               # Validation images and labels
│       └── test/              # Test images and labels
├── data/                      # Additional datasets
│   ├── raw/                   # Raw downloaded datasets
│   ├── processed/             # Augmented and preprocessed data
│   ├── best.pt                # Alternative trained model
│   └── last.pt                # Last checkpoint model
├── src/                       # Source code
│   ├── data_prep/             # Data preparation scripts
│   ├── training/              # Model training scripts
│   └── detection/             # Real-time detection system
│       ├── detect_video.py    # Video file processing
│       └── detect_stream.py   # Live stream processing
├── electron-app/              # Electron desktop application
│   ├── main.js                # Main process (Python integration)
│   ├── renderer.js            # Renderer process (UI logic)
│   ├── index.html             # User interface (mint green theme)
│   └── package.json           # Node.js dependencies
├── docs/                      # Documentation and guides
│   ├── QUICKSTART.md          # Quick start guide
│   ├── SETUP.md               # Setup instructions
│   └── REPORT_TEMPLATE.md     # Project report template
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── DEFENSE_OPERATIONS_GUIDE.md # Operational manual
```

## Dataset Sources

The project utilizes publicly available datasets from Roboflow:

1. **UAV Person Dataset**: https://universe.roboflow.com/militarypersons/uav-person-3
   - Primary dataset for aerial person detection
   
2. **Combatant Dataset**: https://universe.roboflow.com/minwoo/combatant-dataset
   - Specialized military personnel detection
   
3. **Soldiers Detection**: https://universe.roboflow.com/xphoenixua-nlncq/soldiers-detection-spf
   - Soldier-specific detection dataset
   
4. **Look Down Folks**: https://universe.roboflow.com/folks/look-down-folks
   - Top-down perspective civilian detection

These datasets were augmented using techniques including rotation, flipping, and scaling to improve model generalization across various lighting and altitude conditions.

## Installation

### Prerequisites
- Python 3.8 or higher
- Node.js 16 or higher
- CUDA-capable GPU (recommended for training)
- Git for version control

### Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/kurtchinta/CSC-126-FINAL-PROJECT--Aerial-Threat-Detection-Soldier-and-Civilian-Classification-.git
cd CSC-126-FINAL-PROJECT--Aerial-Threat-Detection-Soldier-and-Civilian-Classification-
```

2. **Create and activate Python virtual environment:**
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install Node.js dependencies for Electron application:**
```bash
cd electron-app
npm install
cd ..
```

## Project Implementation

### Phase 1: Dataset Preparation

**Objective:** Collect and prepare labeled datasets for training

```bash
# Download datasets from Roboflow
python src/data_prep/download_datasets.py
```

**Data Augmentation:**
```bash
# Apply augmentation techniques (rotate, flip, scale)
python src/data_prep/augment_data.py
```

The dataset preparation process includes:
- Collecting labeled images of soldiers and civilians
- Applying augmentation to improve model generalization
- Organizing data into train/validation/test splits
- Ensuring balanced class distribution

### Phase 2: Model Selection and Training

**Objective:** Train YOLO11n model for real-time object detection

**Training via Python script:**
```bash
python src/training/train_yolo.py --epochs 100 --batch 16 --img 640
```

**Training via Jupyter Notebook:**
```bash
jupyter notebook backend/civilian_soldier_working/FINAL_CSC_AERIAL.ipynb
```

**Model Evaluation:**
```bash
python src/training/evaluate.py --weights backend/civilian_soldier_working/yolo11n.pt
```

The training process utilizes:
- YOLO11n architecture for optimal speed and accuracy balance
- Annotated drone-like aerial images
- Evaluation metrics: Precision, Recall, mAP (mean Average Precision)
- Custom confidence thresholds (0.50 general, 0.65 for civilians)

### Phase 3: System Development

**Objective:** Integrate trained model with video stream processing

**Video File Detection:**
```bash
python src/detection/detect_video.py --source path/to/video.mp4 --weights backend/civilian_soldier_working/yolo11n.pt --conf 0.5
```

**Live Stream Detection:**
```bash
python src/detection/detect_stream.py --source 0 --weights backend/civilian_soldier_working/yolo11n.pt --conf 0.5
```

**Electron Desktop Application:**
```bash
cd electron-app
npm start
```

System features include:
- Real-time bounding boxes with class labels
- Video file upload and processing
- Live camera/RTSP stream support
- Adjustable confidence and IOU thresholds
- Real-time statistics (civilian count, soldier count, total detections)

### Phase 4: Testing and Evaluation

**Testing Approach:**
- Test model on unseen aerial images and video feeds
- Assess accuracy across various lighting conditions
- Evaluate performance at different altitude perspectives
- Measure inference speed and real-time processing capability

**Performance Metrics:**
- **Precision**: Accuracy of positive predictions (minimize false positives)
- **Recall**: Ability to detect all positive instances (minimize false negatives)
- **mAP@0.5**: Mean Average Precision at 0.5 IoU threshold
- **mAP@0.5:0.95**: Mean Average Precision across multiple IoU thresholds
- **Inference Speed**: Frames per second on target hardware

Training results and detailed metrics are saved in:
`backend/civilian_soldier_working/runs/train/`

## Tools and Technologies

**Programming Languages:**
- Python 3.8+ (Backend processing and AI/ML pipeline)
- JavaScript/Node.js (Electron application interface)

**Deep Learning Frameworks:**
- YOLO11n (Primary model for real-time object detection)
- YOLOv8 (Alternative detection framework)
- PyTorch (Deep learning framework for model training)
- Ultralytics (YOLO implementation and training tools)

**Computer Vision:**
- OpenCV (Video processing and visualization)
- Roboflow (Dataset management and augmentation)

**Application Development:**
- Electron (Cross-platform desktop application framework)
- HTML/CSS/JavaScript (User interface)

**Development Environment:**
- Google Colab / Local GPU (Model training)
- Visual Studio Code (Code editor)
- Git/GitHub (Version control)

## Expected Output

This project delivers:

1. **Working Prototype System**
   - Electron desktop application capable of detecting and classifying individuals in aerial footage
   - Real-time video processing with bounding box visualization
   - Support for both video file upload and live stream processing
   - Available on GitHub: https://github.com/kurtchinta/CSC-126-FINAL-PROJECT--Aerial-Threat-Detection-Soldier-and-Civilian-Classification-.git

2. **Model Performance Documentation**
   - Trained YOLO11n model with evaluation metrics (precision, recall, mAP)
   - Training logs and performance graphs
   - Testing results on various lighting and altitude conditions

3. **Technical Report and Presentation**
   - Model design and architecture explanation
   - Performance analysis and evaluation
   - Recommendations for real-world deployment
   - Discussion of ethical considerations and limitations

## Real-World Deployment Considerations

**Technical Requirements:**
- High-performance computing infrastructure for real-time processing
- Reliable communication systems for drone-to-ground transmission
- Robust error handling and fail-safe mechanisms
- Regular model updates and retraining with new data

**Operational Factors:**
- Validation across diverse environments (urban, rural, forest, desert)
- Testing under various weather conditions
- Integration with existing military/humanitarian systems
- Operator training and standard operating procedures

**Limitations:**
- Model accuracy depends on image quality and viewing angle
- Performance may vary in extreme lighting conditions
- Requires continuous monitoring and human oversight
- Cannot replace human judgment in critical decision-making

## Ethical Considerations

**IMPORTANT**: This project is strictly educational and conceptual.

**Ethical Guidelines:**
- This system is developed solely for academic learning and demonstration purposes
- Not intended for real-life military application without proper ethical evaluation
- Requires government oversight, legal compliance, and ethical review before any deployment
- Must respect privacy rights and human rights considerations
- Acknowledges potential for misuse and dual-use technology concerns

**Humanitarian Emphasis:**
- Primary focus on humanitarian applications (search and rescue, disaster response)
- Should prioritize civilian protection over military objectives
- Must include human oversight in all operational scenarios
- Requires transparent decision-making processes

**Responsible AI Development:**
- Model bias testing and mitigation strategies
- Accountability frameworks for system errors
- Clear documentation of limitations and failure modes
- Ongoing ethical assessment throughout system lifecycle

## Future Enhancements

**Technical Improvements:**
- Multi-class detection (vehicles, weapons, equipment)
- Thermal imaging integration for night operations
- Edge deployment for on-device processing
- Real drone hardware integration (DJI SDK)
- Advanced tracking and trajectory prediction algorithms
- Cloud-based processing pipeline for distributed systems

**Operational Enhancements:**
- Integration with geographic information systems (GIS)
- Multi-drone coordination and data fusion
- Automated alert and notification systems
- Historical data analysis and pattern recognition
- Mobile application for field operations

## Project Deliverables

**Repository Contents:**
- Complete source code for all components
- Trained YOLO11n model weights
- Dataset preparation and augmentation scripts
- Electron desktop application
- Documentation and user guides
- Training logs and performance metrics

**Documentation:**
- README.md (this file)
- DEFENSE_OPERATIONS_GUIDE.md (operational manual)
- QUICKSTART.md (quick start guide)
- SETUP.md (detailed setup instructions)
- REPORT_TEMPLATE.md (project report structure)

## Contributing

This is an educational project developed for CSC 126. Contributions should align with ethical guidelines and educational purposes. For suggestions or improvements, please open an issue or submit a pull request on GitHub.

## License

This project is developed for educational purposes as part of CSC 126 coursework. See LICENSE file for details.

## Acknowledgments

**Academic Resources:**
- Roboflow community for dataset curation and management tools
- Ultralytics for YOLO11n and YOLOv8 frameworks
- OpenCV community for computer vision libraries
- Academic institutions supporting computer vision research

**Development Team:**
- Team 6 members for collaborative development and testing
- Course instructors for guidance and project requirements
- Open-source community for tools and resources

## Team Contact Information

**Team 6 - CSC 126 Final Project**

Team Members:
- RESERVA, Kurt Daniel M.
- FELISILDA
- NONAN

**Project Repository:**
https://github.com/kurtchinta/CSC-126-FINAL-PROJECT--Aerial-Threat-Detection-Soldier-and-Civilian-Classification-.git

For questions, issues, or collaboration inquiries, please open an issue on the GitHub repository.

---

## Disclaimer

This system is a prototype developed for educational demonstration purposes only. It is not intended for operational military or surveillance use. Real-world deployment would require:
- Extensive validation and testing across diverse scenarios
- Comprehensive ethical review and approval
- Legal compliance with national and international laws
- Professional oversight and quality assurance
- Ongoing monitoring and accountability mechanisms

The developers acknowledge the sensitive nature of this technology and emphasize its educational purpose in learning computer vision, deep learning, and system integration concepts.
