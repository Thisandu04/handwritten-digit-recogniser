"""Sanity checks before CNN training. Usage: python -m src.check_cnn_setup"""
import numpy as np

from src.data import (add_channel_axis, load_raw_mnist,
                      make_train_val_split, scale_pixels)
from src.models import build_cnn_model

x_train, y_train, _, _ = load_raw_mnist()  # test set deliberately unused
x_scaled = scale_pixels(x_train)

# Week 5 split (no channel axis) vs the CNN split (with channel axis)
_, x_val_old, _, y_val_old = make_train_val_split(x_scaled, y_train)
x_tr, x_val, y_tr, y_val = make_train_val_split(add_channel_axis(x_scaled), y_train)

print("CNN train shape:", x_tr.shape, x_tr.dtype)
print("CNN val shape  :", x_val.shape)
print("Value range    :", x_tr.min(), "-", x_tr.max())
print("Same validation labels as Week 5:", np.array_equal(y_val, y_val_old))
print("Same validation pixels as Week 5:", np.array_equal(x_val[..., 0], x_val_old))

print()
build_cnn_model().summary()