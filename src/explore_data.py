import numpy as np
import matplotlib.pyplot as plt
from src.data import load_raw_mnist, scale_pixels, make_train_val_split

x_train, y_train, x_test, y_test = load_raw_mnist()

# --- 1. Verify shapes, dtypes, labels ---
print("x_train:", x_train.shape, x_train.dtype)
print("y_train:", y_train.shape, y_train.dtype)
print("x_test :", x_test.shape, x_test.dtype)
print("labels :", np.unique(y_train))
print("pixel range (raw):", x_train.min(), "-", x_train.max())

# --- 2. Sample grid: 10 examples per class ---
fig, axes = plt.subplots(10, 10, figsize=(10, 10))
for digit in range(10):
    idxs = np.where(y_train == digit)[0][:10]
    for col, idx in enumerate(idxs):
        ax = axes[digit, col]
        ax.imshow(x_train[idx], cmap="gray")
        ax.axis("off")
    axes[digit, 0].set_title(f"Digit {digit}", fontsize=8, loc="left")
plt.tight_layout()
plt.savefig("results/week5_samples_per_class.png", dpi=150)
plt.close()

# --- 3. Class frequencies ---
counts = np.bincount(y_train)
for d, c in enumerate(counts):
    print(f"digit {d}: {c} ({c / len(y_train):.2%})")

plt.bar(range(10), counts)
plt.xticks(range(10))
plt.xlabel("Digit class"); plt.ylabel("Number of training images")
plt.title("MNIST training class frequencies")
plt.savefig("results/week5_class_frequencies.png", dpi=150)
plt.close()

# --- 4. Scaling + split ---
x_train_s = scale_pixels(x_train)
x_tr, x_val, y_tr, y_val = make_train_val_split(x_train_s, y_train)
print("\nAfter scaling:", x_tr.dtype, x_tr.min(), "-", x_tr.max())
print("Train:", x_tr.shape, "| Val:", x_val.shape)
print("Val class counts:", np.bincount(y_val))