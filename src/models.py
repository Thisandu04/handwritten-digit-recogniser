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


def build_cnn_model(use_pooling=True):
    """Compact CNN: 2 x (conv [-> max-pool]), then a dense classification head."""
    blocks = [layers.Input(shape=(28, 28, 1))]
    for filters in (32, 64):
        blocks.append(layers.Conv2D(filters, (3, 3), activation="relu"))
        if use_pooling:
            blocks.append(layers.MaxPooling2D((2, 2)))
    blocks += [
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
    return keras.Sequential(blocks, name="cnn_baseline" if use_pooling else "cnn_no_pool")