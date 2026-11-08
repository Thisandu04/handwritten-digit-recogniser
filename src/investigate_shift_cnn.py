"""Week 6: translation sensitivity, CNN vs dense (validation set only).
Usage: python -m src.investigate_shift_cnn
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from tensorflow import keras

from src.data import (add_channel_axis, load_raw_mnist,
                      make_train_val_split, scale_pixels)
from src.investigate_flattening import shift_horizontal  # reuse the Week 5 function

OUT = Path("results/week6_shift")
OUT.mkdir(parents=True, exist_ok=True)

x_train, y_train, _, _ = load_raw_mnist()  # test set deliberately unused
_, x_val, _, y_val = make_train_val_split(scale_pixels(x_train), y_train)

dense = keras.models.load_model("models/dense_baseline_best.keras")
cnn = keras.models.load_model("models/cnn_baseline_best.keras")

shifts = list(range(-4, 5))
dense_acc, cnn_acc = [], []
for k in shifts:
    xs = shift_horizontal(x_val, k)
    dense_acc.append(float(dense.evaluate(xs, y_val, batch_size=512, verbose=0)[1]))
    cnn_acc.append(float(cnn.evaluate(add_channel_axis(xs), y_val,
                                      batch_size=512, verbose=0)[1]))

print(f"{'shift':>6}{'dense acc':>12}{'cnn acc':>10}")
for k, d, c in zip(shifts, dense_acc, cnn_acc):
    print(f"{k:>+6d}{d:>12.4f}{c:>10.4f}")

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(shifts, dense_acc, marker="o", label="Dense baseline")
ax.plot(shifts, cnn_acc, marker="s", label="CNN baseline")
ax.set_xlabel("Horizontal shift (pixels; negative = left)")
ax.set_ylabel("Validation accuracy")
ax.set_title("Sensitivity to translation (validation set)")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout()
fig.savefig(OUT / "shift_sensitivity_dense_vs_cnn.png", dpi=150)
plt.close(fig)

with open(OUT / "shift_results.json", "w") as f:
    json.dump({"shifts": shifts, "dense_val_accuracy": dense_acc,
               "cnn_val_accuracy": cnn_acc}, f, indent=2)