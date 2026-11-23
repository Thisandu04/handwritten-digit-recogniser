"""Week 7 Part B: exploratory robustness test of the FROZEN CNN.

Uses the VALIDATION set (the test set stays reserved for the final report).
Transformations and severities are fixed here BEFORE running.
These transformed images are NOT an independent real-world benchmark.
Usage: python -m src.robustness
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from tensorflow import keras

from src.data import (SEED, add_channel_axis, load_raw_mnist,
                      make_train_val_split, scale_pixels)

_p = argparse.ArgumentParser()
_p.add_argument("--model", default="models/cnn_baseline_best.keras")
_p.add_argument("--out", default="results/week7_robustness")
_args = _p.parse_args()
OUT = Path(_args.out)
OUT.mkdir(parents=True, exist_ok=True)


def rotate(x, deg):
    return ndimage.rotate(x, deg, axes=(1, 2), reshape=False, order=1,
                          mode="constant", cval=0.0)


def rescale(x, s):
    z = ndimage.zoom(x, (1, s, s), order=1)
    h = z.shape[1]
    if h >= 28:  # zoom in: centre-crop
        o = (h - 28) // 2
        return z[:, o:o + 28, o:o + 28]
    out = np.zeros_like(x)  # zoom out: centre-pad
    o = (28 - h) // 2
    out[:, o:o + h, o:o + h] = z
    return out


def vshift(x, k):
    return ndimage.shift(x, (0, k, 0), order=0, mode="constant", cval=0.0)


def blur(x, sigma):
    return ndimage.gaussian_filter(x, sigma=(0, sigma, sigma))


def add_noise(x, std):
    rng = np.random.default_rng(SEED)
    return np.clip(x + rng.normal(0, std, x.shape), 0, 1).astype("float32")


def contrast(x, f):
    return x * f


def invert(x, _):
    return 1.0 - x


# ---- Pre-defined plan: family, identity level, severity levels ----
PLAN = [
    {"family": "rotation (deg)",      "identity": 0,   "levels": [-30, -15, 15, 30],  "fn": rotate},
    {"family": "scale (factor)",      "identity": 1.0, "levels": [0.6, 0.8, 1.2, 1.4], "fn": rescale},
    {"family": "vertical shift (px)", "identity": 0,   "levels": [-4, -2, 2, 4],     "fn": vshift},
    {"family": "blur (sigma)",        "identity": 0,   "levels": [0.5, 1.0, 1.5],    "fn": blur},
    {"family": "noise (std)",         "identity": 0,   "levels": [0.1, 0.2, 0.4],    "fn": add_noise},
    {"family": "contrast (factor)",   "identity": 1.0, "levels": [0.6, 0.3, 0.15],   "fn": contrast},
    {"family": "inversion (0=orig)",  "identity": 0,   "levels": [1],               "fn": invert},
]
with open(OUT / "robustness_plan.json", "w") as f:
    json.dump([{k: v for k, v in p.items() if k != "fn"} for p in PLAN], f, indent=2)

# ---- Data and frozen model (test set unused) ----
x_train, y_train, _, _ = load_raw_mnist()
_, x_val, _, y_val = make_train_val_split(scale_pixels(x_train), y_train)
model = keras.models.load_model(_args.model)


def evaluate(x):
    probs = model.predict(add_channel_axis(x.astype("float32")), batch_size=512, verbose=0)
    return float((probs.argmax(axis=1) == y_val).mean()), float(probs.max(axis=1).mean())


base_acc, base_conf = evaluate(x_val)
print(f"Original validation accuracy: {base_acc:.4f}  (mean top score {base_conf:.4f})\n")
print(f"{'transformation':<22}{'level':>7}{'accuracy':>10}{'change':>9}{'mean top score':>16}")

results = []
for fam in PLAN:
    for level in fam["levels"]:
        acc, mean_conf = evaluate(fam["fn"](x_val, level))
        results.append({"family": fam["family"], "level": level,
                        "accuracy": acc, "mean_top_score": mean_conf})
        print(f"{fam['family']:<22}{level:>7}{acc:>10.4f}{acc - base_acc:>+9.4f}{mean_conf:>16.4f}")

# ---- Non-digit probes ----
print("\n=== Non-digit inputs ===")
rng = np.random.default_rng(SEED)
probes = {"blank (all 0)": np.zeros((1, 28, 28), "float32"),
          "all white (all 1)": np.ones((1, 28, 28), "float32")}
probe_results = {}
for name, img in probes.items():
    pr = model.predict(add_channel_axis(img), verbose=0)[0]
    probe_results[name] = {"pred": int(pr.argmax()), "top_score": float(pr.max())}
    print(f"{name:<18} predicted {pr.argmax()} with score {pr.max():.4f}")
pn = model.predict(add_channel_axis(rng.random((200, 28, 28)).astype("float32")), verbose=0)
probe_results["uniform noise (200)"] = {"mean_top_score": float(pn.max(axis=1).mean()),
                                        "pred_counts": np.bincount(pn.argmax(axis=1), minlength=10).tolist()}
print(f"uniform noise x200: mean top score {pn.max(axis=1).mean():.4f}, "
      f"predicted-class counts {np.bincount(pn.argmax(axis=1), minlength=10)}")

# ---- Figures ----
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for ax, fam in zip(axes.ravel(), PLAN):
    pts = [(fam["identity"], base_acc)] + [
        (r["level"], r["accuracy"]) for r in results if r["family"] == fam["family"]]
    pts.sort()
    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o")
    ax.axhline(base_acc, color="grey", ls="--", lw=0.8)
    ax.set_xlabel(fam["family"]); ax.set_ylabel("Validation accuracy"); ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
axes.ravel()[-1].axis("off")
fig.suptitle("Frozen CNN under pre-defined transformations (validation set, exploratory)")
fig.tight_layout()
fig.savefig(OUT / "robustness_curves.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(len(PLAN), 5, figsize=(8, 1.7 * len(PLAN)))
for r, fam in enumerate(PLAN):
    levels = [fam["identity"]] + fam["levels"]
    for c in range(5):
        ax = axes[r, c]
        ax.axis("off")
        if c < len(levels):
            img = x_val[:1] if levels[c] == fam["identity"] else fam["fn"](x_val[:1], levels[c])
            ax.imshow(img[0], cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{fam['family']}\n{levels[c]}", fontsize=6)
fig.tight_layout()
fig.savefig(OUT / "transformation_examples.png", dpi=150)
plt.close(fig)

with open(OUT / "robustness_results.json", "w") as f:
    json.dump({"set": "validation", "baseline_accuracy": base_acc, "results": results,
               "non_digit_probes": probe_results,
               "note": "exploratory; not an independent real-world benchmark"}, f, indent=2)
print(f"\nSaved to {OUT}")