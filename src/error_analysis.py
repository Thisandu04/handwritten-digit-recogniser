"""Week 7 Part B: reproducible error analysis of the frozen model (test set).

Uses saved predictions from Part A. Diagnosis only; nothing here changes the model.
Usage (from project root):
    python -m src.error_analysis
"""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data import SEED, load_raw_mnist, scale_pixels

OUT = Path("results/week7")
GALLERY = OUT / "error_gallery"
GALLERY.mkdir(parents=True, exist_ok=True)

_, _, x_test, y_test = load_raw_mnist()
x_test = scale_pixels(x_test)  # (10000, 28, 28), display and descriptive stats only

data = np.load(OUT / "test_predictions.npz")
probs, y_saved = data["probs"], data["y_true"]
assert np.array_equal(y_saved, y_test), "test labels do not match saved predictions"
pred = probs.argmax(axis=1)
conf = probs.max(axis=1)
wrong = np.where(pred != y_test)[0]
correct = np.where(pred == y_test)[0]

with open(OUT / "test_results.json") as f:
    reported = json.load(f)
assert len(wrong) == reported["errors"], "error count differs from test_results.json"
print(f"Errors: {len(wrong)} of {len(y_test)} (matches test_results.json)\n")

# ---- 1. Full error table ----
ranks = np.array([int((probs[i] > probs[i, y_test[i]]).sum()) + 1 for i in wrong])
with open(OUT / "errors.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["test_index", "true", "pred", "pred_score", "true_class_score",
                "true_class_rank"] + [f"score_{d}" for d in range(10)])
    for i, r in zip(wrong, ranks):
        w.writerow([int(i), int(y_test[i]), int(pred[i]), f"{conf[i]:.4f}",
                    f"{probs[i, y_test[i]]:.4f}", int(r)] + [f"{p:.4f}" for p in probs[i]])

# ---- 2. Score summary ----
print("=== Scores ===")
print(f"Mean top score  correct: {conf[correct].mean():.4f} | errors: {conf[wrong].mean():.4f}")
print(f"Errors with top score > 0.90 : {int((conf[wrong] > 0.90).sum())}")
print(f"Errors with top score > 0.99 : {int((conf[wrong] > 0.99).sum())}")
print(f"True class ranked 2nd: {int((ranks == 2).sum())} | within top 3: {int((ranks <= 3).sum())}")

# ---- 3. Measured features: errors vs correct (descriptive only) ----
print("\n=== Measured features (median) ===")
ink = x_test.sum(axis=(1, 2))
rows_with_ink = (x_test > 0.5).any(axis=2).sum(axis=1)
cols_with_ink = (x_test > 0.5).any(axis=1).sum(axis=1)
for name, v in [("ink amount", ink), ("rows with ink", rows_with_ink),
                ("cols with ink", cols_with_ink)]:
    print(f"{name:<14} correct={np.median(v[correct]):7.1f}   errors={np.median(v[wrong]):7.1f}")
print("(Classes differ in ink, e.g. digit 1 has little, so this is a rough comparison.)")


# ---- 4. Galleries ----
def show(indices, path, cols=6, title=None):
    n_rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(n_rows, cols, figsize=(cols * 1.7, n_rows * 1.9), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), indices):
        ax.imshow(x_test[i], cmap="gray")
        ax.set_title(f"#{i}  {y_test[i]}->{pred[i]} ({conf[i]:.2f})", fontsize=7)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# 4a. Seeded random sample (reproducible)
rng = np.random.default_rng(SEED)
sample = np.sort(rng.choice(wrong, size=min(24, len(wrong)), replace=False))
with open(OUT / "error_sample_indices.json", "w") as f:
    json.dump({"seed": SEED, "test_indices": sample.tolist()}, f)
show(sample, GALLERY / "random_sample_24.png",
     title="Random sample of errors  (#index  true->pred (top score))")

# 4b. Most confident errors, with score distributions
top = wrong[np.argsort(-conf[wrong])][:8]
fig, axes = plt.subplots(2, len(top), figsize=(2.0 * len(top), 4.2), squeeze=False)
for k, i in enumerate(top):
    axes[0, k].imshow(x_test[i], cmap="gray")
    axes[0, k].axis("off")
    axes[0, k].set_title(f"#{i}\ntrue {y_test[i]} / pred {pred[i]}", fontsize=8)
    colours = ["tab:green" if d == y_test[i] else "tab:red" if d == pred[i] else "lightgrey"
               for d in range(10)]
    axes[1, k].bar(range(10), probs[i], color=colours)
    axes[1, k].set_xticks(range(10))
    axes[1, k].set_ylim(0, 1)
    axes[1, k].tick_params(labelsize=7)
axes[1, 0].set_ylabel("Score", fontsize=8)
fig.suptitle("Most confident errors (green = true class, red = predicted class)", fontsize=10)
fig.tight_layout()
fig.savefig(GALLERY / "most_confident_errors.png", dpi=150)
plt.close(fig)

# 4c. The three most frequent confusion cells
for pair in reported["top_confusions"][:3]:
    t, p_, c = pair["true"], pair["pred"], pair["count"]
    idx = np.where((y_test == t) & (pred == p_))[0][:12]
    show(idx, GALLERY / f"confusion_{t}_as_{p_}.png",
         title=f"True {t} predicted as {p_}  ({c} cases)")

# 4d. All errors, paged (for annotation)
for start in range(0, len(wrong), 30):
    show(wrong[start:start + 30], GALLERY / f"all_errors_page_{start // 30 + 1}.png")

# ---- 5. Annotation template (never overwrites your work) ----
ann = OUT / "error_annotation.csv"
if not ann.exists():
    with open(ann, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test_index", "true", "pred", "category", "notes"])
        for i in wrong:
            w.writerow([int(i), int(y_test[i]), int(pred[i]), "", ""])
    print(f"\nCreated annotation template: {ann}")
else:
    print(f"\nAnnotation file already exists, left untouched: {ann}")
print(f"Galleries saved in {GALLERY}")