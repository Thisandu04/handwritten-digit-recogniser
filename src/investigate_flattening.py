"""Week 5 investigation: how much does the dense network use spatial structure?

A: train on pixel-permuted images (same permutation for every image).
B: shift validation images horizontally and evaluate the baseline.

Usage (from project root):
    python -m src.investigate_flattening
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from src.data import SEED, load_raw_mnist, make_train_val_split, scale_pixels
from src.models import build_dense_model

OUT = Path("results/week5_flattening")
OUT.mkdir(parents=True, exist_ok=True)


def permute_pixels(x, perm):
    """Apply one fixed permutation to the 784 pixel positions of every image."""
    return x.reshape(len(x), -1)[:, perm].reshape(x.shape)


def shift_horizontal(x, k):
    """Shift images k pixels right (k > 0) or left (k < 0), filling with zeros."""
    out = np.zeros_like(x)
    if k > 0:
        out[:, :, k:] = x[:, :, :-k]
    elif k < 0:
        out[:, :, :k] = x[:, :, -k:]
    else:
        out = x.copy()
    return out


def main():
    x_train, y_train, _, _ = load_raw_mnist()  # test set deliberately unused
    x_tr, x_val, y_tr, y_val = make_train_val_split(scale_pixels(x_train), y_train)

    # ---------- Experiment A: pixel permutation ----------
    perm = np.random.default_rng(SEED).permutation(28 * 28)
    xp_tr, xp_val = permute_pixels(x_tr, perm), permute_pixels(x_val, perm)

    # Visual: original vs permuted
    fig, axes = plt.subplots(2, 6, figsize=(9, 3.4))
    for i in range(6):
        axes[0, i].imshow(x_val[i], cmap="gray"); axes[0, i].axis("off")
        axes[1, i].imshow(xp_val[i], cmap="gray"); axes[1, i].axis("off")
    axes[0, 0].set_title("Original", fontsize=9, loc="left")
    axes[1, 0].set_title("Permuted", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "permuted_examples.png", dpi=150)
    plt.close(fig)

    # Same settings as the baseline run: seed 42, 128 units, Adam 1e-3, 15 epochs
    keras.utils.set_random_seed(SEED)
    model = build_dense_model(hidden_units=128)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    hist = model.fit(xp_tr, y_tr, validation_data=(xp_val, y_val),
                     epochs=15, batch_size=128, verbose=0).history
    best = int(np.argmin(hist["val_loss"]))  # same selection rule as baseline

    with open("results/dense_baseline/run_record.json") as f:
        base = json.load(f)

    result_a = {
        "baseline_best_val_accuracy": base["best_val_accuracy"],
        "baseline_best_val_loss": base["best_val_loss"],
        "permuted_best_val_accuracy": float(hist["val_accuracy"][best]),
        "permuted_best_val_loss": float(hist["val_loss"][best]),
        "permuted_best_epoch": best + 1,
    }
    print("\n=== Experiment A: pixel permutation (validation set) ===")
    print(f"Baseline : val_acc={result_a['baseline_best_val_accuracy']:.4f}  "
          f"val_loss={result_a['baseline_best_val_loss']:.4f}")
    print(f"Permuted : val_acc={result_a['permuted_best_val_accuracy']:.4f}  "
          f"val_loss={result_a['permuted_best_val_loss']:.4f}  (epoch {best + 1})")

    # ---------- Experiment B: horizontal shifts on the baseline ----------
    baseline = keras.models.load_model("models/dense_baseline_best.keras")
    shifts = list(range(-4, 5))
    accs = []
    for k in shifts:
        _, acc = baseline.evaluate(shift_horizontal(x_val, k), y_val, verbose=0)
        accs.append(float(acc))

    print("\n=== Experiment B: horizontal shift of validation images ===")
    for k, a in zip(shifts, accs):
        print(f"shift {k:+d} px: val_acc={a:.4f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(shifts, accs, marker="o")
    ax.set_xlabel("Horizontal shift (pixels; negative = left)")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Dense baseline: sensitivity to translation")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "shift_sensitivity_dense.png", dpi=150)
    plt.close(fig)

    with open(OUT / "flattening_results.json", "w") as f:
        json.dump({"experiment_A": result_a,
                   "experiment_B": {"shifts": shifts, "val_accuracy": accs}}, f, indent=2)


if __name__ == "__main__":
    main()