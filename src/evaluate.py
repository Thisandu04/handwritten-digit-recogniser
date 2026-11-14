"""One-time held-out test evaluation of the FROZEN model.

Usage (from project root):
    python -m src.evaluate
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from tensorflow import keras

from src.data import add_channel_axis, load_raw_mnist, scale_pixels
from src.freeze_model import sha256

OUT = Path("results/week7")
LABELS = list(range(10))


def plot_confusion(matrix, path, title, mask_diagonal=False):
    data = matrix.copy()
    if mask_diagonal:
        np.fill_diagonal(data, 0)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(data, cmap="Blues")
    ax.set_xticks(LABELS); ax.set_yticks(LABELS)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_title(title)
    cutoff = data.max() / 2
    for i in LABELS:
        for j in LABELS:
            if mask_diagonal and i == j:
                continue
            ax.text(j, i, str(data[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if data[i, j] > cutoff else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frozen", default="results/week7/frozen_config.json")
    p.add_argument("--force", action="store_true",
                   help="overwrite existing test results (document why in your report)")
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "test_results.json"
    if result_path.exists() and not args.force:
        raise SystemExit("Test results already exist. The test set is a one-time "
                         "evaluation; use --force only for a documented re-run.")

    with open(args.frozen) as f:
        cfg = json.load(f)
    if sha256(cfg["model_path"]) != cfg["model_sha256"]:
        raise SystemExit("Model file has changed since it was frozen. Aborting.")

    # Same preprocessing as training: /255, float32, channel axis
    _, _, x_test, y_test = load_raw_mnist()
    x_test = add_channel_axis(scale_pixels(x_test))
    n = len(y_test)

    model = keras.models.load_model(cfg["model_path"])
    test_loss, _ = model.evaluate(x_test, y_test, batch_size=512, verbose=0)
    probs = model.predict(x_test, batch_size=512, verbose=0)
    pred = probs.argmax(axis=1)

    errors = int((pred != y_test).sum())
    acc = 1 - errors / n
    half_width = 1.96 * np.sqrt(acc * (1 - acc) / n)  # approximate 95% interval

    prec, rec, f1, support = precision_recall_fscore_support(
        y_test, pred, labels=LABELS, zero_division=0)
    macro = precision_recall_fscore_support(
        y_test, pred, labels=LABELS, average="macro", zero_division=0)[:3]
    weighted = precision_recall_fscore_support(
        y_test, pred, labels=LABELS, average="weighted", zero_division=0)[:3]
    cm = confusion_matrix(y_test, pred, labels=LABELS)

    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = sorted(((int(i), int(j), int(off[i, j]))
                    for i in LABELS for j in LABELS if off[i, j] > 0),
                   key=lambda t: -t[2])

    # ---- Console summary ----
    print(f"Model: {cfg['run_id']}  (SHA-256 verified)")
    print(f"Test images: {n}   Errors: {errors}")
    print(f"Test accuracy: {acc:.4f}  (approx. 95% interval +-{half_width:.4f})")
    print(f"Test loss    : {test_loss:.4f}")
    print(f"Validation   : acc={cfg['validation_accuracy']:.4f}  "
          f"loss={cfg['validation_loss']:.4f}  (for comparison)\n")
    print(f"{'digit':>5}{'precision':>11}{'recall':>9}{'F1':>8}{'support':>9}")
    for d in LABELS:
        print(f"{d:>5}{prec[d]:>11.4f}{rec[d]:>9.4f}{f1[d]:>8.4f}{int(support[d]):>9}")
    print(f"{'macro':>5}{macro[0]:>11.4f}{macro[1]:>9.4f}{macro[2]:>8.4f}{n:>9}")
    print(f"{'wtd':>5}{weighted[0]:>11.4f}{weighted[1]:>9.4f}{weighted[2]:>8.4f}{n:>9}")
    print("\nMost frequent confusions (true -> predicted : count)")
    for t, pr, c in pairs[:8]:
        print(f"  {t} -> {pr} : {c}")
    print(f"\nSanity: matrix total={cm.sum()}, diagonal/total={np.trace(cm) / cm.sum():.4f}")

    # ---- Saved artefacts ----
    (OUT / "classification_report.txt").write_text(
        classification_report(y_test, pred, labels=LABELS, digits=4), encoding="utf-8")
    np.savetxt(OUT / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    np.savez_compressed(OUT / "test_predictions.npz", probs=probs, y_true=y_test)
    plot_confusion(cm, OUT / "confusion_matrix.png",
                   f"Confusion matrix (test set, n={n})")
    plot_confusion(cm, OUT / "confusion_matrix_errors_only.png",
                   "Misclassifications only (diagonal hidden)", mask_diagonal=True)

    results = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "model_run_id": cfg["run_id"],
        "model_sha256": cfg["model_sha256"],
        "n_test": n,
        "errors": errors,
        "accuracy": acc,
        "accuracy_ci95_halfwidth": float(half_width),
        "test_loss": float(test_loss),
        "validation_accuracy": cfg["validation_accuracy"],
        "validation_loss": cfg["validation_loss"],
        "per_class": {str(d): {"precision": float(prec[d]), "recall": float(rec[d]),
                               "f1": float(f1[d]), "support": int(support[d])}
                      for d in LABELS},
        "macro_avg": dict(zip(["precision", "recall", "f1"], map(float, macro))),
        "weighted_avg": dict(zip(["precision", "recall", "f1"], map(float, weighted))),
        "top_confusions": [{"true": t, "pred": pr, "count": c} for t, pr, c in pairs[:10]],
        "evaluation_script": "src/evaluate.py",
        "forced_rerun": bool(args.force),
    }
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {OUT}")


if __name__ == "__main__":
    main()