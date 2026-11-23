"""Preview augmentation. Usage: python -m src.preview_augmentation"""
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

from src.data import (add_channel_axis, load_raw_mnist,
                      make_train_val_split, scale_pixels)

x_train, y_train, _, _ = load_raw_mnist()  # test set deliberately unused
x_tr, _, y_tr, _ = make_train_val_split(add_channel_axis(scale_pixels(x_train)), y_train)

aug = keras.Sequential([
    layers.RandomRotation(0.04, fill_mode="constant", fill_value=0.0),
    layers.RandomTranslation(0.1, 0.1, fill_mode="constant", fill_value=0.0),
    layers.RandomZoom(0.1, fill_mode="constant", fill_value=0.0),
])
keras.utils.set_random_seed(0)
x = x_tr[:8]

fig, axes = plt.subplots(4, 8, figsize=(12, 6.5))
rows = [("original", x)] + [(f"augmented {k}", aug(x, training=True).numpy()) for k in (1, 2, 3)]
for r, (name, imgs) in enumerate(rows):
    for c in range(8):
        axes[r, c].imshow(imgs[c, :, :, 0], cmap="gray", vmin=0, vmax=1)
        axes[r, c].axis("off")
    axes[r, 0].set_title(name, fontsize=8, loc="left")
fig.tight_layout()
fig.savefig("results/week7_improvement/augmentation_preview.png", dpi=150)