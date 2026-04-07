import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from utils import extract_frames

# =========================
# LOAD DATA
# =========================

data = []
labels = []

dataset_path = "dataset"

for category in ["snatching", "normal"]:
    folder = os.path.join(dataset_path, category)

    for file in os.listdir(folder):
        video_path = os.path.join(folder, file)

        if file.endswith(".mp4") or file.endswith(".avi"):
            frames = extract_frames(video_path)

            frame = frames[0]   # take first frame only
            data.append(frame)

            if category == "snatching":
                labels.append(1)
            else:
                labels.append(0)

X = np.array(data)
y = np.array(labels)

# =========================
# SPLIT DATA
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_train_cat = to_categorical(y_train, 2)
y_test_cat = to_categorical(y_test, 2)

# =========================
# BUILD MODEL
# =========================

model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# TRAIN
# =========================

history = model.fit(X_train, y_train_cat, epochs=5, batch_size=8)

# =========================
# EVALUATE
# =========================

loss, acc = model.evaluate(X_test, y_test_cat)
print("\nCNN Accuracy:", acc)

# =========================
# CONFUSION MATRIX
# =========================

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)

print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred_classes))

# =========================
# PLOT CONFUSION MATRIX
# =========================

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("CNN Confusion Matrix")
plt.show()