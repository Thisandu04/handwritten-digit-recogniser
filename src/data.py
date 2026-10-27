"""Data loading, splitting and preprocessing for MNIST."""
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist

SEED = 42
VAL_FRACTION = 0.1  # 6,000 validation images


def load_raw_mnist():
    """Return the official MNIST train/test arrays (uint8, 0-255)."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return x_train, y_train, x_test, y_test


def scale_pixels(x):
    """Scale uint8 pixels [0, 255] to float32 [0, 1]."""
    return x.astype("float32") / 255.0


def make_train_val_split(x_train, y_train, val_fraction=VAL_FRACTION, seed=SEED):
    """Stratified split of the TRAINING portion only. Test set is never touched."""
    return train_test_split(
        x_train, y_train,
        test_size=val_fraction,
        random_state=seed,
        stratify=y_train,
    )