# 🚁 Aerial Surveillance System - Quick Reference Card

## One-Page Cheat Sheet

### 🎯 Project Goal
Build an AI system to detect & classify soldiers/civilians in aerial footage using YOLOv8

---

### ⚡ Quick Setup (Copy-Paste)

```powershell
# 1. Setup Environment
cd "c:\Users\Kurt\Desktop\FINAL-CSC"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Configure API
copy .env.example .env
notepad .env  # Add your Roboflow API key

# 3. Install Electron
cd electron-app
npm install
cd ..

# 4. Test
python test_environment.py
```

---

### 📊 Complete Workflow

```powershell
# Step 1: Download Data (10-20 min)
python src/data_prep/download_datasets.py

# Step 2: Augment Data (20-40 min)
python src/data_prep/augment_data.py

# Step 3: Train Model (4-8 hours GPU)
python src/training/train_yolo.py

# Step 4: Evaluate (5 min)
python src/training/evaluate.py

# Step 5: Test Detection (1 min)
python src/detection/detect_video.py --source video.mp4

# Step 6: Launch App (instant)
cd electron-app && npm start
```

---

### 🔧 Essential Commands

| Task | Command |
|------|---------|
| **Activate venv** | `.\venv\Scripts\Activate.ps1` |
| **Download data** | `python src/data_prep/download_datasets.py` |
| **Train model** | `python src/training/train_yolo.py` |
| **Evaluate** | `python src/training/evaluate.py` |
| **Detect video** | `python src/detection/detect_video.py --source file.mp4` |
| **Webcam** | `python src/detection/detect_stream.py --source 0` |
| **Launch app** | `cd electron-app && npm start` |
| **TensorBoard** | `tensorboard --logdir runs` |

---

### 📁 Key Files & Locations

| What | Where |
|------|-------|
| **Config** | `.env` |
| **Datasets** | `data/combined/` |
| **Model** | `models/best.pt` |
| **Training logs** | `runs/aerial_surveillance/` |
| **Results** | `tests/results/` |
| **Videos out** | `output/videos/` |
| **App** | `electron-app/` |
| **Docs** | `docs/` |

---

### 🐛 Quick Fixes

**CUDA not found?**
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Out of memory?**
Edit `.env`: `BATCH_SIZE=4`

**Slow training?**
Edit `.env`: `MODEL_SIZE=yolov8s.pt` and `EPOCHS=50`

**Roboflow error?**
Check API key in `.env` file

**Electron won't start?**
```powershell
cd electron-app; rm -rf node_modules; npm install
```

---

### 📈 Expected Performance

| Metric | Target | Good |
|--------|--------|------|
| **mAP50** | >0.60 | >0.75 |
| **Precision** | >0.70 | >0.80 |
| **Recall** | >0.65 | >0.75 |
| **FPS (GPU)** | >20 | >40 |

---

### 🎓 Project Deliverables

✅ GitHub repository with code
✅ Trained model (Google Drive link)
✅ Project report (10-15 pages PDF)
✅ Presentation (10-15 slides)
✅ Demo video (optional, 3-5 min)

---

### ⚙️ Configuration (.env)

```env
ROBOFLOW_API_KEY=your_key_here
MODEL_SIZE=yolov8m.pt
BATCH_SIZE=16
IMAGE_SIZE=640
EPOCHS=100
DEVICE=0
WORKERS=8
CONFIDENCE_THRESHOLD=0.25
IOU_THRESHOLD=0.45
```

---

### 🎯 Classes

| ID | Class | Color |
|----|-------|-------|
| 0 | Soldier | 🔴 Red |
| 1 | Civilian | 🟢 Green |
| 2 | Person | 🔵 Blue |

---

### 📚 Documentation

- **Full Guide**: `PROJECT_GUIDE.md`
- **Setup**: `docs/SETUP.md`
- **Quick Start**: `docs/QUICKSTART.md`
- **Report Template**: `docs/REPORT_TEMPLATE.md`

---

### 🔗 Useful Links

- **Roboflow**: https://app.roboflow.com/
- **YOLOv8 Docs**: https://docs.ultralytics.com/
- **PyTorch**: https://pytorch.org/
- **Electron**: https://www.electronjs.org/

---

### 📞 Getting Help

1. Run: `python test_environment.py`
2. Check `docs/` folder
3. Review error messages
4. Ask instructor/TA

---

### ⏱️ Time Estimates

| Phase | Time |
|-------|------|
| Setup | 30 min |
| Data prep | 1 hour |
| Training | 4-8 hours (GPU) |
| Testing | 1 hour |
| Documentation | 2-4 hours |
| **Total** | **~10-15 hours** |

---

### 💡 Pro Tips

1. **Start early** - Training takes time
2. **Use GPU** - Much faster than CPU
3. **Save often** - Checkpoints every 10 epochs
4. **Document as you go** - Screenshots, notes
5. **Test frequently** - Don't wait until the end

---

### ⚠️ Remember

**Educational Use Only**
- Not for real military use
- Ethical considerations required
- Privacy and rights important

---

**Print this page for quick reference!**

Last updated: December 2025
