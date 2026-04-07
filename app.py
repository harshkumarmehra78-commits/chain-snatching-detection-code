import os
import inspect
from pathlib import Path

# Set TensorFlow options before importing TensorFlow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model

from utils import extract_frames

model = None
MODEL_PATH = Path("model.h5")


def get_model():
    """Load the model once and reuse it for all requests."""
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH.resolve()}")
        model = load_model(str(MODEL_PATH), compile=False)
    return model


def _resolve_uploaded_path(video_file):
    if video_file is None:
        return None

    if isinstance(video_file, str):
        return video_file

    if isinstance(video_file, (bytes, bytearray)):
        raise ValueError("Binary upload not supported. Please upload as a file.")

    name = getattr(video_file, "name", None)
    if isinstance(name, str):
        return name

    raise ValueError("Unsupported file input received from Gradio.")


def predict_video(video_file):
    """Process uploaded video and return a prediction message."""
    if video_file is None:
        return "Please upload a video file."

    try:
        model_instance = get_model()
        video_path = _resolve_uploaded_path(video_file)

        if not video_path or not Path(video_path).exists():
            return "Uploaded video could not be read. Please upload again."

        frames = extract_frames(video_path)
        frames = np.expand_dims(frames.astype(np.float32), axis=0)

        prediction = model_instance.predict(frames, verbose=0)
        prediction = np.asarray(prediction)

        if prediction.ndim == 2 and prediction.shape[1] >= 2:
            class_id = int(np.argmax(prediction[0]))
            confidence = float(prediction[0][class_id])
        else:
            score = float(prediction.reshape(-1)[0])
            class_id = 1 if score >= 0.5 else 0
            confidence = score if class_id == 1 else (1.0 - score)

        if class_id == 1:
            return f"Chain snatching detected (confidence: {confidence:.2%})"
        return f"Normal activity (confidence: {confidence:.2%})"

    except Exception as e:
        return f"Error processing video: {str(e)}"


interface_kwargs = {}
if "flagging_mode" in inspect.signature(gr.Interface).parameters:
    interface_kwargs["flagging_mode"] = "never"
else:
    interface_kwargs["allow_flagging"] = "never"


demo = gr.Interface(
    fn=predict_video,
    inputs=gr.File(label="Upload Video", file_types=[".mp4", ".avi", ".mov"]),
    outputs=gr.Textbox(label="Prediction Result"),
    title="Chain Snatching Detection",
    description="Upload a video to detect chain snatching activity using AI.",
    **interface_kwargs,
)


if __name__ == "__main__":
    demo.launch(
    server_name="0.0.0.0",
    server_port=7860
)

