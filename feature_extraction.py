import os
import numpy as np
from utils import extract_frames

data = []
labels = []

dataset_path = "dataset"

for category in ["snatching", "normal"]:
    folder = os.path.join(dataset_path, category)

    for file in os.listdir(folder):
        video_path = os.path.join(folder, file)

        if file.endswith(".mp4") or file.endswith(".avi"):
            frames = extract_frames(video_path)

            # 🔥 Convert video → single feature vector
            features = np.mean(frames, axis=0)   # average frames

            features = features.flatten()       # convert to 1D

            data.append(features)

            if category == "snatching":
                labels.append(1)
            else:
                labels.append(0)

X = np.array(data)
y = np.array(labels)

np.save("X.npy", X)
np.save("y.npy", y)

print("Feature extraction complete:", X.shape)