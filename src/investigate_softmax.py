"""Week 5 investigation: is a softmax score a calibrated probability?

Uses the validation set and the saved baseline. Usage:
    python -m src.investigate_softmax
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from src.data import SEED, load_raw_mnist, make_train_val_split, scale_pixels

OUT = Path("results/week5_softmax")
OUT.mkdir(parents=True, exist_ok=True)

x_train, y_train, _, _ = load_raw_mnist()  # test set deliberately unused
_, x_val, _, y_val = make_train_val_split(scale_pixels(x_train), y_train)

model = keras.models.load_model("models/dense_baseline_best.keras")
probs = model.predict(x_val, batch_size=512, verbose=0)
pred = probs.argmax(axis=1)
conf = probs.max(axis=1)
correct = pred == y_val

# --- 1. Reliability table ---
print(f"Validation accuracy: {correct.mean():.4f}")
print(f"Mean confidence (correct):   {conf[correct].mean():.4f}")
print(f"Mean confidence (incorrect): {conf[~correct].mean():.4f}\n")

edges = [0.0, 0.5, 0.7, 0.9, 0.99, 1.0001]
print(f"{'confidence bin':<16}{'n':>6}{'mean conf':>12}{'accuracy':>10}")
rows = []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (conf >= lo) & (conf < hi)
    if m.sum() == 0:
        continue
    rows.append({"bin": f"{lo:.2f}-{min(hi, 1.0):.2f}", "n": int(m.sum()),
                 "mean_conf": float(conf[m].mean()), "accuracy": float(correct[m].mean())})
    print(f"{rows[-1]['bin']:<16}{rows[-1]['n']:>6}{rows[-1]['mean_conf']:>12.4f}"
          f"{rows[-1]['accuracy']:>10.4f}")

# --- 2. Confident mistakes ---
wrong_confident = int(((conf > 0.9) & ~correct).sum())
print(f"\nWrong predictions with confidence > 0.9: {wrong_confident} "
      f"of {int((~correct).sum())} total errors")

# --- 3. Non-digit inputs ---
rng = np.random.default_rng(SEED)
blank = np.zeros((1, 28, 28), dtype="float32")
noise = rng.random((100, 28, 28)).astype("float32")
pb = model.predict(blank, verbose=0)[0]
pn = model.predict(noise, verbose=0)
print(f"\nBlank image: predicted {pb.argmax()} with score {pb.max():.4f}")
print(f"100 random-noise images: mean top score {pn.max(axis=1).mean():.4f}, "
      f"predicted-class counts {np.bincount(pn.argmax(axis=1), minlength=10)}")

# --- Plot: confidence histograms ---
fig, ax = plt.subplots(figsize=(7, 4))
bins = np.linspace(0, 1, 26)
ax.hist(conf[correct], bins=bins, alpha=0.7, label="Correct")
ax.hist(conf[~correct], bins=bins, alpha=0.7, label="Incorrect")
ax.set_yscale("log")
ax.set_xlabel("Max softmax score"); ax.set_ylabel("Count (log scale)")
ax.set_title("Baseline: softmax score on validation set")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "confidence_histogram.png", dpi=150)
plt.close(fig)

with open(OUT / "softmax_results.json", "w") as f:
    json.dump({"reliability": rows, "wrong_confident_gt_0.9": wrong_confident,
               "blank_top_score": float(pb.max()),
               "noise_mean_top_score": float(pn.max(axis=1).mean())}, f, indent=2)