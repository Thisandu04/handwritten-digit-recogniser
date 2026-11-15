"""Week 7: training-validation gap for the selected CNN (no new test evaluation).
Usage: python -m src.investigate_overfitting
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from src.data import (add_channel_axis, load_raw_mnist,
                      make_train_val_split, scale_pixels)

OUT = Path("results/week7")
RUNS = ["dense_baseline", "cnn_baseline", "cnn_seed_1", "cnn_seed_2", "cnn_seed_3"]

records = {}
for r in RUNS:
    p = Path("results") / r / "run_record.json"
    if p.exists():
        records[r] = json.load(open(p))
    else:
        print(f"(skipping {r}: no run record)")

# ---- 1. From run records (Keras training metrics are running averages) ----
print("=== From run records ===")
print(f"{'run':<15}{'best ep':>8}{'train acc':>10}{'val acc':>9}{'train loss':>11}"
      f"{'val loss':>9}{'final val loss':>15}{'rise':>8}")
for name, rec in records.items():
    h, b = rec["history"], rec["best_epoch"] - 1
    rise = h["val_loss"][-1] - h["val_loss"][b]
    print(f"{name:<15}{b + 1:>8}{h['accuracy'][b]:>10.4f}{h['val_accuracy'][b]:>9.4f}"
          f"{h['loss'][b]:>11.4f}{h['val_loss'][b]:>9.4f}{h['val_loss'][-1]:>15.4f}{rise:>8.4f}")

# ---- 2. Frozen CNN in inference mode: training vs validation partition ----
x_train, y_train, _, _ = load_raw_mnist()  # test set deliberately unused
x_tr, x_val, y_tr, y_val = make_train_val_split(
    add_channel_axis(scale_pixels(x_train)), y_train)
model = keras.models.load_model("models/cnn_baseline_best.keras")
tr_loss, tr_acc = model.evaluate(x_tr, y_tr, batch_size=512, verbose=0)
va_loss, va_acc = model.evaluate(x_val, y_val, batch_size=512, verbose=0)
print("\n=== Frozen CNN, inference mode ===")
print(f"Training   : loss={tr_loss:.4f}  acc={tr_acc:.4f}  ({int((1 - tr_acc) * len(y_tr))} errors of {len(y_tr)})")
print(f"Validation : loss={va_loss:.4f}  acc={va_acc:.4f}  ({int(round((1 - va_acc) * len(y_val)))} errors of {len(y_val)})")
print(f"Accuracy gap (train - val): {tr_acc - va_acc:+.4f}")

# ---- 3. Validation vs test ----
t = json.load(open(OUT / "test_results.json"))
print("\n=== Validation vs test ===")
print(f"Validation acc={va_acc:.4f}   Test acc={t['accuracy']:.4f}   "
      f"(test 95% half-width +-{t['accuracy_ci95_halfwidth']:.4f})")

# ---- 4. Gap curves ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
for name, rec in records.items():
    h = rec["history"]
    e = range(1, len(h["loss"]) + 1)
    ax1.plot(e, np.array(h["val_loss"]) - np.array(h["loss"]), label=name)
    ax2.plot(e, np.array(h["accuracy"]) - np.array(h["val_accuracy"]), label=name)
ax1.axhline(0, color="grey", lw=0.8)
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Validation loss - training loss"); ax1.legend()
ax2.axhline(0, color="grey", lw=0.8)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Training accuracy - validation accuracy")
fig.suptitle("Generalisation gap per epoch")
fig.tight_layout()
fig.savefig(OUT / "generalisation_gap.png", dpi=150)
plt.close(fig)
print(f"\nSaved {OUT / 'generalisation_gap.png'}")