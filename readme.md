# NUAI Lab - LoCoBot & Image Classification

## Overview

This repository contains two main projects:

1. **LoCoBot Navigation & Object Detection** - Autonomous robot control for duck detection and approach using YOLO
2. **Battery Image Classifier** - Deep learning model for real-time battery/no-battery classification (side project)

## Projects

### 1. LoCoBot Duck Detection & Approach

**Files:**
- `t10.py` - Main duck detection and approach script
- `t7.py` - Red box detection with camera tilt feedback control

**Features:**
- YOLO-based duck detection
- Camera pan/tilt centering before motion
- Depth-based distance estimation
- Re-acquisition logic when object is lost

**Requirements:**
- InterbotixLocobotXS robot (px100)
- ROS + RealSense camera
- YOLO model (`best.pt` or similar)

**Usage:**
```bash
roslaunch interbotix_xslocobot_control xslocobot_python.launch robot_model:=locobot_px100
python t10.py
```

### 2. Battery Image Classifier

**Files:**
- `image_classifier.ipynb` - Training notebook (MobileNetV2 transfer learning)
- `test_classifier_webcam.py` - Real-time webcam inference app
- `classifier.keras` - Trained model (binary classification)
- `classifier.tflite` - Optimized model for mobile deployment

**Features:**
- Transfer learning with MobileNetV2
- 80/10/10 train/validation/test split
- TFLite export for edge devices
- Real-time webcam inference with OpenCV

**Requirements:**
```bash
pip install tensorflow keras opencv-python numpy matplotlib
```

**Usage:**

*Training:*
```bash
jupyter notebook image_classifier.ipynb
```

*Testing with Webcam:*
```bash
python test_classifier_webcam.py
```

Press `q` to quit, `s` to save frames.

## Dataset

Battery classifier expects images organized as:
```
data/battery/data/
├── battery/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── no_battery/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## Model Performance

- **Test Accuracy**: Evaluated on held-out test set
- **Model Size**: ~36 MB (Keras), ~9 MB (TFLite)
- **Inference Speed**: Real-time on CPU with OpenCV

## Directory Structure

```
NUAI lab/
├── readme.md
├── image_classifier.ipynb
├── test_classifier_webcam.py
├── classifier.keras
├── classifier.tflite
├── t10.py
├── t7.py
├── projects/
│   ├── Diwali_final_v1.py
│   └── ...
└── data/
    └── battery/
        └── data/
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train classifier (optional):**
   ```bash
   jupyter notebook image_classifier.ipynb
   ```

3. **Test with webcam:**
   ```bash
   python test_classifier_webcam.py
   ```

4. **Run robot navigation:**
   ```bash
   python t10.py
   ```

## Notes

- Models trained on custom dataset
- LoCoBot scripts require active ROS and robot connection
- Webcam app requires `classifier.keras` in working directory
