# Defense Reconnaissance System - Operations Guide

## Mission Briefing

**System Purpose**: Real-time classification of soldiers and civilians from aerial drone surveillance to support defense operations and humanitarian missions.

## Strategic Context

As tensions escalate and conflict looms, this AI-powered reconnaissance system provides crucial intelligence capabilities:

- **Threat Assessment**: Identify military personnel in operational zones
- **Civilian Protection**: Distinguish non-combatants for humanitarian aid
- **Real-time Intelligence**: Process drone footage for immediate tactical decisions
- **Bounding Box Visualization**: Clear visual identification with class labels

---

## System Capabilities

### ✅ Core Features (Requirements Met)

1. **Image Classification Model**
   - Distinguishes soldiers from civilians in aerial images
   - YOLOv8n architecture trained on UAV surveillance datasets
   - Model location: `backend/civilian_soldier_working/yolov8n.pt`

2. **Video Stream Integration**
   - Processes drone footage simulation
   - Supports live webcam feeds
   - RTSP stream capability for remote drones

3. **Bounding Box Detection**
   - Red boxes: Soldiers (military personnel)
   - Green boxes: Civilians (non-combatants)
   - Class labels displayed on each detection

4. **Electron Desktop Application**
   - Military-grade interface
   - Real-time statistics dashboard
   - Video and stream processing modes

---

## Quick Start - Reconnaissance Operations

### 1. System Activation

```powershell
# Navigate to base
cd "C:\Users\Kurt\Desktop\FINAL-CSC"

# Activate reconnaissance system
cd electron-app
npm start
```

### 2. Mission Configuration

**In the Electron App:**

1. **Model Path**: Already set to `backend/civilian_soldier_working/yolov8n.pt`
2. **Confidence Threshold**: 0.25 (adjust for mission requirements)
3. **IOU Threshold**: 0.45 (non-maximum suppression)

### 3. Operational Modes

#### Mode A: Video Analysis (Drone Footage)
1. Click "Video" tab
2. Browse and select drone footage file
3. Click "Start Detection"
4. Monitor real-time classification
5. Results saved to `output/videos/`

#### Mode B: Live Stream (Real-time Surveillance)
1. Click "Stream" tab
2. Select source:
   - **Webcam (0)**: Local camera simulation
   - **RTSP**: Remote drone feed URL
3. Click "Start Detection"
4. Live bounding boxes appear
5. Press 'Q' to terminate

---

## Detection Output

### Visual Indicators

```
🟥 RED BOUNDING BOX + "Soldier" label
   → Military personnel detected
   → Threat assessment required
   
🟩 GREEN BOUNDING BOX + "Civilian" label  
   → Non-combatant identified
   → Humanitarian consideration
```

### Statistics Dashboard

- **Soldier Count**: Total military personnel detected
- **Civilian Count**: Total non-combatants identified  
- **Total Persons**: Combined count
- **Confidence Scores**: Detection certainty (0.0 - 1.0)

---

## Command Line Operations (Advanced)

### Video Detection
```powershell
# Activate environment
.\venv\Scripts\Activate.ps1

# Process drone footage
python src/detection/detect_video.py --source drone_footage.mp4 --weights backend/civilian_soldier_working/yolov8n.pt --conf 0.25

# Output: Annotated video with bounding boxes
```

### Live Stream Detection
```powershell
# Webcam surveillance
python src/detection/detect_stream.py --source 0 --weights backend/civilian_soldier_working/yolov8n.pt

# RTSP drone feed
python src/detection/detect_stream.py --source rtsp://192.168.1.100:8554/stream --weights backend/civilian_soldier_working/yolov8n.pt
```

---

## Dataset Intelligence

**Training Data**: UAV Person Dataset
- Source: Roboflow Military Persons collection
- Images: High-altitude aerial surveillance
- Classes: 2 (Soldier, Civilian)
- Format: YOLO annotation format
- Location: `backend/civilian_soldier_working/`

**Dataset Structure**:
```
civilian_soldier_working/
├── train/
│   ├── images/       # Training aerial images
│   └── labels/       # YOLO format annotations
├── val/
│   ├── images/       # Validation images
│   └── labels/       # Validation annotations
├── test/
│   ├── images/       # Test images
│   └── labels/       # Test annotations
├── dataset.yaml      # YOLO configuration
└── yolov8n.pt       # Trained model weights
```

---

## Mission Requirements Checklist

### ✅ Academic Requirements Met

- [x] **Build image classification model** - YOLOv8n trained on aerial images
- [x] **Distinguish soldiers from civilians** - Binary classification implemented
- [x] **Integrate with video stream** - Both video files and live streams supported
- [x] **Draw bounding boxes** - Color-coded boxes (Red: Soldier, Green: Civilian)
- [x] **Class labels displayed** - "Soldier" and "Civilian" labels on detections
- [x] **Electron application** - Desktop GUI with video/stream integration
- [x] **Object detection and classification** - Real-time YOLOv8 inference

---

## Operational Parameters

### Recommended Settings

**Standard Reconnaissance**:
- Confidence: 0.25
- IOU: 0.45
- Image Size: 640x640

**High-Precision Mode**:
- Confidence: 0.40
- IOU: 0.50
- Image Size: 640x640

**Fast Surveillance**:
- Confidence: 0.20
- IOU: 0.40
- Image Size: 416x416

---

## Performance Metrics

### Model Specifications

- **Architecture**: YOLOv8n (Nano - optimized for speed)
- **Input Size**: 640x640 pixels
- **Classes**: 2 (Soldier, Civilian)
- **Inference Speed**: ~50-100 FPS (GPU) / ~10-20 FPS (CPU)
- **Parameters**: ~3.2M

### Expected Accuracy
- **mAP50**: Target >0.70
- **Precision**: Target >0.75
- **Recall**: Target >0.70

---

## Troubleshooting

### Issue: Model not found
**Solution**: Verify path
```powershell
dir backend\civilian_soldier_working\yolov8n.pt
# Should show file exists
```

### Issue: Low FPS
**Solution**: 
- Use GPU if available
- Reduce confidence threshold
- Lower image resolution

### Issue: Poor detection accuracy
**Solution**:
- Increase confidence threshold (0.30-0.40)
- Ensure good lighting in footage
- Verify camera angle similar to training data

---

## Ethical Considerations

**IMPORTANT**: This is an educational project demonstrating computer vision for defense applications.

- Use only for lawful defense and humanitarian purposes
- Respect privacy and international law
- Classification may not be 100% accurate
- Human oversight required for all operational decisions
- Consider civilian safety in all deployments

---

## Project Deliverables

For academic submission:

1. ✅ Trained model: `backend/civilian_soldier_working/yolov8n.pt`
2. ✅ Detection system: `src/detection/*.py`
3. ✅ Desktop application: `electron-app/`
4. ✅ Dataset configuration: `backend/civilian_soldier_working/dataset.yaml`
5. ✅ Documentation: This guide + README.md

---

## Support & Contact

For technical issues:
1. Check system logs in Electron app
2. Review error messages
3. Verify Python environment active
4. Consult PROJECT_GUIDE.md

---

**Status**: ✅ OPERATIONAL
**Last Updated**: December 2025
**Classification**: Educational Project - Defense Technology

*"Contributing to national defense through Computer Science and Information Technology"*
