import numpy as np
from tensorflow.keras.models import load_model
import utils
from utils import extract_frames

# 🔍 Check which utils file is being used
print("Using utils from:", utils.__file__)

# 🔹 Load trained model
model = load_model("model.h5")

# 🔍 Check model input shape
print("Model expects input shape:", model.input_shape)


def predict_video(video_path):
    # 🔹 Extract frames
    frames = extract_frames(video_path)

    # 🔍 DEBUG: Check frame shape
    print("Frames shape BEFORE expand:", frames.shape)

    # ❌ SAFETY CHECK (VERY IMPORTANT)
    if frames.shape[1] != 32 or frames.shape[2] != 32:
        print("❌ ERROR: Frames are NOT 32x32. Fix utils.py!")
        return

    # 🔹 Add batch dimension
    frames = np.expand_dims(frames, axis=0)

    print("Frames shape AFTER expand:", frames.shape)

    # 🔹 Prediction
    prediction = model.predict(frames)
    class_id = np.argmax(prediction)

    # 🔹 Output result
    if class_id == 1:
        print("⚠️ Chain Snatching Detected")
    else:
        print("✅ Normal Activity")


# =========================
# 🔹 TEST VIDEO
# =========================
predict_video("test_videos/Normal_Test001.avi")