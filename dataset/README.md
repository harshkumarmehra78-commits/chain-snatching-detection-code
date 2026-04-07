# Dataset Folder

This folder contains the training data for the Chain Snatching Detection model.

## Structure

```
dataset/
├── normal/           # Normal activity videos
│   └── *.avi, *.mp4  # Video files
└── snatching/        # Chain snatching videos
    └── *.avi, *.mp4  # Video files
```

## How to Use

### Add Your Videos

1. **Normal Activity Videos:**
   - Place videos showing normal market/street activity in `dataset/normal/`
   - Examples: People walking, shopping, normal pedestrian movement

2. **Chain Snatching Videos:**
   - Place videos showing chain snatching incidents in `dataset/snatching/`
   - Examples: Actual chain snatching cases, theft incidents

### Video Requirements

- **Format:** MP4 or AVI
- **Resolution:** Minimum 480p (recommended)
- **Duration:** 3-30 seconds per video
- **Frame Rate:** 24-30 FPS recommended
- **Codec:** H.264 or similar

### Recommended Dataset Size

- **Minimum:** 100-150 videos per category
- **Ideal:** 300-500 videos per category
- **Total:** 200-1000 videos for good results

## Dataset Preparation

### Step 1: Organize Videos
```bash
# Create folders if not present
mkdir dataset/normal
mkdir dataset/snatching

# Move your videos
cp path/to/normal/videos/* dataset/normal/
cp path/to/snatching/videos/* dataset/snatching/
```

### Step 2: Extract Features
```bash
python feature_extraction.py
```
This will:
- Load all videos from both folders
- Extract 10 frames per video
- Create feature vectors (30,720 dimensions)
- Save as `X.npy` and `y.npy`

### Step 3: Train Model
```bash
python train.py
```
This will train 3 models:
- CNN-LSTM (primary)
- SVM (RBF kernel)
- Random Forest

## Tips for Better Results

1. **Balanced Dataset:**
   - Keep equal number of videos in both folders
   - Minimum 50 videos per category

2. **Data Diversity:**
   - Use videos from different:
     - Times of day (day, night, evening)
     - Locations (markets, streets, malls)
     - Camera angles
     - Lighting conditions

3. **Quality:**
   - Ensure videos are not corrupted
   - Clear video without heavy compression
   - Reasonable frame rate (24-30 FPS)

4. **Edge Cases:**
   - Include challenging scenarios
   - Include similar activities to chain snatching
   - Include different types of normal activities

## File Size Notes

⚠️ **Large Video Files:**
- `.avi` and `.mp4` files are NOT tracked by Git (see `.gitignore`)
- These files should be managed separately:
  - Store locally in this folder
  - Use cloud storage (Google Drive, AWS S3, etc.)
  - Use Git LFS if you need GitHub to store them

## Labels

The model uses binary classification:
- **Label 0:** Normal Activity (folder: `dataset/normal/`)
- **Label 1:** Chain Snatching (folder: `dataset/snatching/`)

## Example Video Counts

For optimal training:
```
dataset/
├── normal/
│   ├── Normal001.avi     (200 KB)
│   ├── Normal002.avi     (220 KB)
│   ├── Normal003.avi     (180 KB)
│   └── ... (300-400 more files)
│
└── snatching/
    ├── Snatching001.avi  (210 KB)
    ├── Snatching002.avi  (190 KB)
    ├── Snatching003.avi  (230 KB)
    └── ... (300-400 more files)
```

## Troubleshooting

**Issue:** "Dataset folder not found"
```bash
mkdir -p dataset/normal
mkdir -p dataset/snatching
```

**Issue:** "No videos found"
- Check that videos are in correct folders
- Verify video format (.avi or .mp4)
- Check file extensions (case-sensitive on Linux)

**Issue:** "Corrupted video file"
- Convert video to standard format:
  ```bash
  ffmpeg -i input.mov -c:v libx264 output.mp4
  ```

## Next Steps

1. Add your videos to the respective folders
2. Run `python feature_extraction.py`
3. Run `python train.py`
4. Run `python app.py` to start the web interface

For detailed instructions, see [README.md](../README.md)
