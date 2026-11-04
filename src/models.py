"""Model definitions."""
from tensorflow import keras
from tensorflow.keras import layers


def build_dense_model(hidden_units=128):
    """Fully connected baseline: flatten -> hidden ReLU layer -> 10-class softmax."""
    return keras.Sequential(
        [
            layers.Input(shape=(28, 28)),
            layers.Flatten(),
            layers.Dense(hidden_units, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ],
        name="dense_baseline",
    )

def build_cnn_model():
    """Compact CNN: 2 x (conv -> max-pool), then a dense classification head."""
    return keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ],
        name="cnn_baseline",
    )