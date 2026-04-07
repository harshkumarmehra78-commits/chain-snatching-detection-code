---
title: Chain Snatching Detection
emoji: ""
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 3.50.2
app_file: app.py
pinned: false
---

# Chain Snatching Detection System

A comprehensive AI-powered computer vision solution for detecting chain snatching activity in surveillance videos using multiple machine learning models (CNN-LSTM, SVM, and Random Forest).

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Architecture & Workflow](#architecture--workflow)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [Training the Model](#training-the-model)
- [Model Details](#model-details)
- [Output Files](#output-files)
- [File Documentation](#file-documentation)
- [Troubleshooting](#troubleshooting)
- [Tips & Best Practices](#tips--best-practices)

---

## 📌 Project Overview

**Chain Snatching Detection System** is a video surveillance AI application that automatically detects chain snatching incidents using an ensemble of machine learning models. The system combines:

- **CNN-LSTM**: Deep learning model for spatio-temporal analysis (primary model)
- **SVM (RBF)**: Support Vector Machine for classical pattern recognition
- **Random Forest**: Ensemble learning approach for robust classification

The system processes video files frame-by-frame and outputs predictions with confidence scores.

---

## ✨ Features

- **Multi-Model Approach**: Trains and compares CNN-LSTM, SVM, and Random Forest simultaneously
- **Web UI Interface**: Gradio-based interface for easy video upload and inference
- **Fast Processing**: Processes videos in 1-30 seconds depending on model
- **Automatic Diagram Generation**: Creates 8+ publication-ready training diagrams
- **Multiple Model Formats**: Saves models as .h5 (Keras), .pkl (SVM, Random Forest)
- **Flexible Input**: Supports MP4 and AVI video formats
- **Comprehensive Metrics**: Generates confusion matrices, ROC curves, classification reports

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Prepare Your Data

```
Create folders structure:
dataset/
├── normal/          (add normal activity videos here)
└── snatching/       (add chain snatching videos here)
```

### 3. Train the Model

```bash
python train.py
```

Output: `model.h5`, `svm_model.pkl`, `rf_model.pkl` + 8 diagrams

### 4. Run the Web Interface

```bash
python app.py
```

Open browser: **http://localhost:7860**

---

## 🏗️ Architecture & Workflow

### Data Flow Pipeline

```
Video Input
    ↓
[Frame Extraction] → Extract 10 frames per video
    ↓
[Preprocessing] → Resize to 32×32, Normalize [0,1]
    ↓
[Feature Representation] → Shape: (10, 32, 32, 3) = 30,720 dimensions
    ↓
┌────────────────────────────────────────┐
│         THREE PARALLEL MODELS          │
├────────────────────────────────────────┤
│ • CNN-LSTM (deep learning, temporal)   │
│ • SVM with RBF kernel (classical ML)   │
│ • Random Forest (ensemble voting)      │
└────────────────────────────────────────┘
    ↓
[Softmax Classification Layer]
    ↓
[Output] Class: 0=Normal, 1=Chain Snatching
         Confidence: [0.0 - 1.0]
```

### Training Workflow

```
1. LOAD DATASET
   ├─ Read videos from dataset/normal/
   └─ Read videos from dataset/snatching/
        ↓
2. PREPARE DATA
   ├─ Extract 10 frames per video
   ├─ Resize to 32×32 pixels
   ├─ Normalize to [0, 1] range
   └─ Labels: 0=normal, 1=snatching (binary classification)
        ↓
3. SPLIT DATA
   ├─ Training: 80% of data
   └─ Testing: 20% of data
        ↓
4. TRAIN MODELS (SEQUENTIAL)
   ├─ CNN-LSTM: ~10 minutes on CPU
   ├─ SVM (RBF): ~2 minutes on CPU
   └─ Random Forest: ~1 minute on CPU
        ↓
5. EVALUATE PERFORMANCE
   ├─ Confusion Matrices for each model
   ├─ Classification Reports (Precision, Recall, F1)
   ├─ ROC-AUC Scores
   └─ Generate 8+ publication-ready diagrams
        ↓
6. SAVE TRAINED MODELS
   ├─ model.h5 (CNN-LSTM in Keras format)
   ├─ svm_model.pkl (SVM in pickle format)
   └─ rf_model.pkl (Random Forest in pickle format)
```

---

## 📦 System Requirements

### Minimum Requirements

- **CPU**: Intel i5 or AMD Ryzen 5 equivalent
- **RAM**: 4GB
- **Storage**: 2GB
- **Python**: 3.10.13 (as specified in runtime.txt)

### Recommended Requirements

- **CPU**: Intel i7 / AMD Ryzen 7 or better
- **RAM**: 8GB or more
- **GPU**: NVIDIA with CUDA support (for 5-10x faster training)
- **Storage**: 5GB

### Software Dependencies

```
gradio                          # Web interface
numpy                          # Numerical computing
opencv-python-headless        # Video processing
tensorflow                     # Deep learning
scikit-learn                   # Classical ML (SVM, Random Forest)
Pillow                         # Image processing
huggingface_hub               # Model hub integration
fastapi                       # Web framework
pydantic                      # Data validation
typing_extensions             # Type hints
```

---

## 🔧 Installation

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd chain-snatch-main
```

### Step 2: Create Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import tensorflow, gradio, cv2; print('✅ All dependencies installed!')"
```

### Step 5: Optional - Install FFmpeg

For advanced video processing:

**Windows (Chocolatey):**

```bash
choco install ffmpeg
```

**Linux (Ubuntu/Debian):**

```bash
sudo apt-get install ffmpeg
```

**macOS (Homebrew):**

```bash
brew install ffmpeg
```

---

## 📁 Project Structure

```
chain-snatch-main/
│
├── app.py                      Gradio web interface (inference only)
├── train.py                    Main training script
├── predict.py                  Standalone prediction script
├── feature_extraction.py       Extract features from videos
├── utils.py                    Helper functions (frame extraction)
├── svm_model.py                Standalone SVM training
├── rf_model.py                 Standalone RF training
├── cnn_only.py                 CNN-only model variant
├── comparison_graph.py         Model comparison visualization
│
├── requirements.txt            Python dependencies
├── runtime.txt                 Python version = 3.10.13
├── README.md                   This file
│
├── model.h5                    ✅ Trained CNN-LSTM model
├── svm_model.pkl               ✅ Trained SVM model (after train.py)
├── rf_model.pkl                ✅ Trained RF model (after train.py)
├── X.npy                       Feature matrix (after feature_extraction.py)
├── y.npy                       Labels array (after feature_extraction.py)
│
├── dataset/                    📂 Training data (you create this)
│   ├── normal/                 Add MP4/AVI videos here
│   └── snatching/              Add MP4/AVI videos here
│
└── diagrams_from_training/     📊 Generated automatically by train.py
    ├── Diagram_1_System_Architecture.png
    ├── Diagram_2_Algorithm_Comparison.png
    ├── Diagram_3_Confusion_Matrix.png
    ├── Diagram_4_Complexity_Comparison.png
    ├── Diagram_5_Training_Metrics.png
    ├── Diagram_6_Performance_Metrics.png
    ├── Diagram_7_ROC_Curves.png
    └── Diagram_8_Confusion_Matrices.png
```

---

## 📚 Usage Guide

### Phase 1: Prepare Dataset

```bash
# Create folder structure
mkdir dataset
mkdir dataset/normal
mkdir dataset/snatching
```

**Add videos:**

- `dataset/normal/` → Videos of normal activity (people walking, shopping, etc.)
- `dataset/snatching/` → Videos of chain snatching incidents

**Recommended Dataset Size:**

- Minimum: 100-150 videos per category
- Ideal: 300-500 videos per category
- Video length: 3-30 seconds each
- Resolution: 480p or higher
- Format: MP4 or AVI

**Example structure:**

```
dataset/
├── normal/
│   ├── normal_video_001.mp4
│   ├── normal_video_002.avi
│   ├── normal_video_003.mp4
│   └── ...
└── snatching/
    ├── snatching_video_001.mp4
    ├── snatching_video_002.avi
    ├── snatching_video_003.mp4
    └── ...
```

### Phase 2: Train the Model

```bash
python train.py
```

**What happens:**

1. ✅ Loads all videos from `dataset/` folder
2. ✅ Extracts 10 frames per video
3. ✅ Normalizes frame dimensions to 32×32 pixels
4. ✅ Splits data: 80% training, 20% testing
5. ✅ Trains 3 models simultaneously:
   - CNN-LSTM: ~10 minutes (CPU) or ~2 minutes (GPU)
   - SVM: ~2 minutes (CPU)
   - Random Forest: ~1 minute (CPU)
6. ✅ Generates 8+ evaluation diagrams
7. ✅ Saves models: `model.h5`, `svm_model.pkl`, `rf_model.pkl`

**Output in terminal:**

```
Loading dataset...
Snatching: 250 videos
Normal: 250 videos

[MODEL 1/3] Training CNN-LSTM...
Epoch 1/10... Accuracy: 78.5%
...
[MODEL 2/3] Training SVM RBF Kernel...
[MODEL 3/3] Training Random Forest...

[DIAGRAMS] Generating publication-ready diagrams...
✓ Saved: Diagram_1_System_Architecture.png
✓ Saved: Diagram_2_Algorithm_Comparison.png
...

All models trained successfully!
```

### Phase 3: Run the Web Interface

```bash
python app.py
```

**Output:**

```
Running on local URL:  http://0.0.0.0:7860
To create a public link, set `share=True` in `launch()`.
```

**Access:** Open browser and go to `http://localhost:7860`

**Using the interface:**

1. Click "Upload Video" button
2. Select MP4 or AVI file
3. Click "Submit"
4. Get prediction with confidence score

### Phase 4: Alternative - Command Line Prediction

```bash
# Edit predict.py and change video_path to your video
python predict.py
```

---

## 📊 Model Details

### CNN-LSTM Architecture (Primary Model)

**Model Structure:**

```
Input: (batch, 10, 32, 32, 3)
    ↓
TimeDistributed(Conv2D-32, kernel=3×3, activation=ReLU)
    ↓
TimeDistributed(MaxPooling2D, pool=2×2)
    ↓
TimeDistributed(Conv2D-64, kernel=3×3, activation=ReLU)
    ↓
TimeDistributed(MaxPooling2D, pool=2×2)
    ↓
TimeDistributed(Flatten)
    ↓
Dropout(0.5)
    ↓
LSTM(64 units)
    ↓
Dropout(0.5)
    ↓
Dense(2, activation=Softmax)
    ↓
Output: [prob_normal, prob_snatching]
```

**Training Hyperparameters:**

- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Epochs: 10
- Batch Size: 4
- Validation Split: 20%
- Early Stopping: Not used

**Advantages:**

- Captures temporal dependencies in video sequences
- Deep learning approach with high accuracy
- Bidirectional feature learning

---

### SVM Model (RBF Kernel)

**Configuration:**

- Kernel: RBF (Radial Basis Function)
- C (Regularization): 1.0
- Gamma: scale (auto)
- Input: Flattened 30,720 dimension vectors

**Advantages:**

- Fast training (~2 minutes on CPU)
- Memory efficient
- Good for moderate dataset sizes

**Training Time:** ~2 minutes (CPU)
**Inference Time:** ~0.2 seconds per video

---

### Random Forest Model

**Configuration:**

- Number of Estimators: 100 trees
- Max Depth: 15
- Input: Flattened 30,720 dimension vectors
- Random State: 42

**Advantages:**

- Very fast inference (~0.3 seconds per video)
- No GPU required
- Interpretable feature importance scores
- Highly parallelizable

**Training Time:** ~1 minute (CPU)
**Inference Time:** ~0.3 seconds per video

---

### Performance Comparison

| Metric           | CNN-LSTM | SVM      | Random Forest |
| ---------------- | -------- | -------- | ------------- |
| Typical Accuracy | 93-96%   | 88-92%   | 85-90%        |
| Training Time    | ~10 min  | ~2 min   | ~1 min        |
| Inference Time   | ~1.0 sec | ~0.2 sec | ~0.3 sec      |
| Model Size       | ~50 MB   | ~6.5 MB  | ~10 MB        |
| GPU Support      | Yes      | No       | No            |
| CPU Inference    | Moderate | Fast     | Fast          |

_Times are approximate on CPU (Intel i5). GPU times are 5-10x faster._

---

## 📦 Output Files

### Model Files (created by train.py)

- **model.h5** (~50 MB)
  - CNN-LSTM Keras model
  - Primary inference model
  - Spatio-temporal learning

- **svm_model.pkl** (~6.5 MB)
  - SVM with RBF kernel
  - Classical ML approach
  - Fast inference

- **rf_model.pkl** (~10 MB)
  - Random Forest classifier
  - Ensemble voting approach
  - Very fast inference

### Data Files (created by feature_extraction.py)

- **X.npy**
  - Feature matrix
  - Shape: (n_videos, 30720)
  - Contains flattened frame features

- **y.npy**
  - Label array
  - Shape: (n_videos,)
  - Values: 0=normal, 1=snatching

### Generated Diagrams (created by train.py)

Located in `diagrams_from_training/`:

1. **Diagram_1_System_Architecture.png**
   - Shows complete data pipeline
   - All processing steps

2. **Diagram_2_Algorithm_Comparison.png**
   - Model performance comparison
   - Accuracy across models

3. **Diagram_3_Confusion_Matrix.png**
   - Individual confusion matrices
   - TP, TN, FP, FN visualization

4. **Diagram_4_Complexity_Comparison.png**
   - Training time comparison
   - Memory requirements
   - Model size comparison

5. **Diagram_5_Training_Metrics.png**
   - Training/validation accuracy
   - Loss curves per epoch

6. **Diagram_6_Performance_Metrics.png**
   - Accuracy, Precision, Recall, F1
   - Performance table
   - ROC-AUC scores

7. **Diagram_7_ROC_Curves.png**
   - ROC curves for each model
   - True Positive Rate vs False Positive Rate

8. **Diagram_8_Confusion_Matrices.png**
   - All 3 confusion matrices side-by-side
   - Normalized and non-normalized

---

## 📖 File Documentation

### app.py

**Purpose:** Web interface using Gradio

**Main Function:** `predict_video(video_file)`

- Input: Video file (MP4 or AVI)
- Process: Extracts frames → model inference
- Output: Prediction text with confidence %

**Usage:**

```bash
python app.py
# Access at http://localhost:7860
```

---

### train.py

**Purpose:** Main training script - trains 3 models and generates diagrams

**Key Steps:**

1. Loads videos from `dataset/normal/` and `dataset/snatching/`
2. Extracts 10 frames per video
3. Trains CNN-LSTM (10 epochs)
4. Trains SVM (RBF kernel)
5. Trains Random Forest (100 estimators)
6. Generates 8+ evaluation diagrams
7. Saves: model.h5, svm_model.pkl, rf_model.pkl

**Usage:**

```bash
python train.py
```

---

### utils.py

**Purpose:** Utility functions for video processing

**Main Function:** `extract_frames(video_path, max_frames=10, size=(32, 32))`

**Parameters:**

- `video_path`: Path to video file
- `max_frames`: Number of frames to extract (default: 10)
- `size`: Resize dimensions (default: 32×32)

**Returns:** NumPy array of shape `(max_frames, height, width, 3)`

**Example:**

```python
from utils import extract_frames
frames = extract_frames("video.mp4")
print(frames.shape)  # (10, 32, 32, 3)
```

---

### predict.py

**Purpose:** Standalone prediction script without web interface

**Usage:**

```bash
# Edit video_path in predict.py, then:
python predict.py
```

**Output:**

```
Frames shape: (10, 32, 32, 3)
⚠️ Chain Snatching Detected
# or
✅ Normal Activity
```

---

### feature_extraction.py

**Purpose:** Extract features from all dataset videos

**What it does:**

1. Reads all videos from `dataset/normal/` and `dataset/snatching/`
2. Extracts 10 frames per video
3. Resizes to 32×32 pixels
4. Flattens to 1D vectors (30,720 dimensions)
5. Saves as X.npy and y.npy

**Usage:**

```bash
python feature_extraction.py
```

**Output:**

```
X.npy - Feature matrix (n_videos, 30720)
y.npy - Labels (n_videos,)
Feature extraction complete: (500, 30720)
```

---

### svm_model.py

**Purpose:** Standalone SVM training (requires X.npy, y.npy)

**Configuration:**

- Kernel: Linear (can be changed)
- Test size: 20%
- Outputs confusion matrix and classification report

---

### rf_model.py

**Purpose:** Standalone Random Forest training (requires X.npy, y.npy)

**Configuration:**

- n_estimators: 100
- Test size: 20%
- Outputs confusion matrix and classification report

---

### cnn_only.py

**Purpose:** CNN-only model variant (no LSTM)

**Advantages:**

- Faster training and inference
- Simpler architecture
- Good for single frames

---

## 🔍 Troubleshooting

### Issue: "Dataset folder not found"

**Solution:**

```bash
mkdir dataset
mkdir dataset/normal
mkdir dataset/snatching
# Add your videos to these folders
```

### Issue: "Video not readable / OpenCV error"

**Solution:**

1. Check video format (must be MP4 or AVI)
2. Verify videos exist in correct folders
3. Try converting video:
   ```bash
   ffmpeg -i input.mov -c:v libx264 output.mp4
   ```
4. Check video is not corrupted

### Issue: "Out of Memory" during training

**Solution:**

1. Start with smaller dataset (50-100 videos)
2. Reduce batch size: `batch_size=2` in train.py
3. Use smaller frame size: `size=(16, 16)` in utils.py
4. Add more RAM or use GPU

### Issue: TensorFlow import errors

**Solution:**

```bash
pip uninstall tensorflow -y
pip install tensorflow==2.13.0
```

### Issue: "model.h5 not found" when running app.py

**Solution:**

1. Train model: `python train.py`
2. Check model exists: `ls model.h5`
3. Verify permissions on file

### Issue: Gradio not at localhost:7860

**Solution:**

1. Check port availability
2. Change port in app.py: `server_port=7861`
3. Check firewall settings
4. Restart the script

### Issue: Poor accuracy after training

**Solution:**

1. Use more training data (300+ videos)
2. Ensure balanced dataset (equal normal/snatching)
3. Improve video quality/diversity
4. Retrain with more epochs
5. Check dataset for mislabeled videos

### Issue: Slow inference on CPU

**Solution:**

1. Use GPU: `pip install tensorflow-gpu`
2. Use faster model: Try `model.h5` vs SVM
3. Reduce frame resolution: `(16, 16)`
4. Use SVM or Random Forest (faster)

---

## 💡 Tips & Best Practices

### Dataset Preparation

- Use diverse videos (different lighting, angles, scenes)
- Balance classes (equal normal/snatching samples)
- Include edge cases and challenging scenarios
- Minimum 100 videos per category
- Ideally 300-500 videos per category

### Training Best Practices

- Monitor loss and accuracy metrics
- Check confusion matrices for misclassifications
- Save model checkpoints
- Use validation data for early stopping
- Consider data augmentation

### Deployment Best Practices

- Use GPU for production inference
- Cache model to avoid reloading
- Set appropriate confidence thresholds
- Monitor prediction quality
- Log predictions for analysis

### Improvement Strategies

- Collect more diverse data
- Try different frame extraction methods
- Experiment with model architectures
- Ensemble predictions from multiple models
- Regularly retrain with new data

---

## ❓ FAQ

**Q: Why 10 frames per video?**
A: Balance between temporal information and computational efficiency. More frames = better temporal understanding but slower processing.

**Q: Can I use MOV format?**
A: Convert first: `ffmpeg -i input.mov -c:v libx264 output.mp4`

**Q: What if my videos are very short (< 1 second)?**
A: Script extracts available frames and pads with zeros to reach 10 frames.

**Q: How do I improve accuracy?**
A: More data, balanced classes, better video quality, longer training, data diversity.

**Q: Can I use this model outside Gradio?**
A: Yes! Load model.h5 with TensorFlow and use directly in any Python application.

**Q: How long does training take?**
A: ~10 min (CNN-LSTM) + ~2 min (SVM) + ~1 min (RF) on CPU. Much faster on GPU.

---

**Last Updated:** April 2026  
**Python Version:** 3.10.13  
**Status:** Production Ready  
**Gradio Version:** 3.50.2
# chain-snatching-detection-code
