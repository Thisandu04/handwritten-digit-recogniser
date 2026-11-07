"""Week 6: compare runs on the SAME validation set (no test data used).

Usage:
    python -m src.compare_models --runs dense_baseline cnn_baseline
    python -m src.compare_models --runs dense_baseline cnn_baseline cnn_no_pool
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from src.data import (add_channel_axis, load_raw_mnist,
                      make_train_val_split, scale_pixels)

OUT = Path("results/week6_comparison")
OUT.mkdir(parents=True, exist_ok=True)


def single_image_latency_ms(model, x_one, n_warmup=20, n_runs=200):
    """Median and 95th-percentile time to classify ONE image, in milliseconds."""
    for _ in range(n_warmup):
        np.asarray(model(x_one, training=False))
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        np.asarray(model(x_one, training=False))
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times)), float(np.percentile(times, 95))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    args = p.parse_args()

    x_train, y_train, _, _ = load_raw_mnist()  # test set deliberately unused
    _, x_val, _, y_val = make_train_val_split(scale_pixels(x_train), y_train)

    rows, records = [], {}
    for run_id in args.runs:
        with open(Path("results") / run_id / "run_record.json") as f:
            rec = json.load(f)
        records[run_id] = rec

        model = keras.models.load_model(rec["model_path"])
        x = add_channel_axis(x_val) if rec["architecture"]["type"] == "cnn" else x_val

        loss, _ = model.evaluate(x, y_val, batch_size=128, verbose=0)

        model.predict(x[:512], verbose=0)  # warm-up
        t0 = time.perf_counter()
        probs = model.predict(x, batch_size=512, verbose=0)
        batch_ms = (time.perf_counter() - t0) / len(x) * 1000
        errors = int((probs.argmax(axis=1) != y_val).sum())

        median_ms, p95_ms = single_image_latency_ms(model, x[:1])

        if abs(loss - rec["best_val_loss"]) > 1e-3:
            print(f"WARNING: {run_id} re-evaluated loss differs from its record")

        rows.append({
            "run_id": run_id,
            "parameters": rec["parameter_count"],
            "train_seconds": rec["training_seconds"],
            "best_epoch": rec["best_epoch"],
            "val_loss": float(loss),
            "val_accuracy": 1 - errors / len(y_val),
            "val_errors": errors,
            "n_val": int(len(y_val)),
            "latency_median_ms": median_ms,
            "latency_p95_ms": p95_ms,
            "batch_ms_per_image": batch_ms,
        })

    print(f"\n{'run':<16}{'params':>10}{'train s':>9}{'best ep':>8}{'val loss':>10}"
          f"{'val acc':>9}{'errors':>8}{'1-img ms':>10}{'batch ms/img':>14}")
    for r in rows:
        print(f"{r['run_id']:<16}{r['parameters']:>10,}{r['train_seconds']:>9.1f}"
              f"{r['best_epoch']:>8}{r['val_loss']:>10.4f}{r['val_accuracy']:>9.4f}"
              f"{r['val_errors']:>8}{r['latency_median_ms']:>10.2f}"
              f"{r['batch_ms_per_image']:>14.4f}")

    with open(OUT / "comparison.json", "w") as f:
        json.dump(rows, f, indent=2)

    # Overlay of validation curves from the run records
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for run_id, rec in records.items():
        h = rec["history"]
        epochs = range(1, len(h["val_loss"]) + 1)
        ax1.plot(epochs, h["val_loss"], label=run_id)
        ax2.plot(epochs, h["val_accuracy"], label=run_id)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Validation loss"); ax1.legend()
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Validation accuracy"); ax2.legend()
    fig.suptitle("Validation curves: " + " vs ".join(args.runs))
    fig.tight_layout()
    fig.savefig(OUT / "validation_curves_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()