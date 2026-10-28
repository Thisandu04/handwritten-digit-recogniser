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