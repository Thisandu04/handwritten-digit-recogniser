"""Model loading and prediction (no preprocessing and no UI code here)."""
from pathlib import Path

import numpy as np
from tensorflow import keras

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "digit_cnn.keras"


def load_digit_model(path=DEFAULT_MODEL_PATH):
    """Load the saved CNN. In the app, call this once and reuse the object."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}. Run: python -m src.export_app_model")
    model = keras.models.load_model(path)
    if tuple(model.input_shape[1:]) != (28, 28, 1):
        raise ValueError(f"Unexpected model input shape: {model.input_shape}")
    return model


def predict_digit(model, tensor):
    """tensor: (1, 28, 28, 1) float32 in [0, 1]. Returns the digit and the model scores."""
    tensor = np.asarray(tensor, dtype="float32")
    if tensor.shape != (1, 28, 28, 1):
        raise ValueError(f"Expected shape (1, 28, 28, 1), got {tensor.shape}")
    scores = np.asarray(model(tensor, training=False))[0]
    order = np.argsort(-scores)
    return {
        "digit": int(order[0]),
        "scores": scores,   # model scores (softmax outputs), not calibrated probabilities
        "top3": [(int(d), float(scores[d])) for d in order[:3]],
    }