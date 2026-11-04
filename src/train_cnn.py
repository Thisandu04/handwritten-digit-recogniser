"""Train the CNN.

Usage (from project root):
    python -m src.train_cnn --run-id cnn_baseline
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.data import (SEED, VAL_FRACTION, add_channel_axis, load_raw_mnist,
                      make_train_val_split, scale_pixels)
from src.models import build_cnn_model
from src.train_dense import plot_curves  # reuse the same plotting code


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def main():
    args = parse_args()
    keras.utils.set_random_seed(args.seed)

    run_dir = Path("results") / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path("models") / f"{args.run_id}_best.keras"
    model_path.parent.mkdir(exist_ok=True)

    # Test set returned by the loader but deliberately unused.
    # Split uses the fixed SEED so it matches the Week 5 partition exactly.
    x_train, y_train, _, _ = load_raw_mnist()
    x_tr, x_val, y_tr, y_val = make_train_val_split(
        add_channel_axis(scale_pixels(x_train)), y_train
    )

    model = build_cnn_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # Same selection rule as the dense baseline: lowest validation loss.
    checkpoint = keras.callbacks.ModelCheckpoint(
        str(model_path), monitor="val_loss", mode="min", save_best_only=True
    )

    start = time.perf_counter()
    history = model.fit(
        x_tr, y_tr,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[checkpoint],
        verbose=2,
    )
    duration = time.perf_counter() - start

    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    best_idx = int(np.argmin(hist["val_loss"]))

    record = {
        "run_id": args.run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "split": {"val_fraction": VAL_FRACTION, "split_seed": SEED, "stratified": True,
                  "n_train": int(len(x_tr)), "n_val": int(len(x_val))},
        "preprocessing": "pixels / 255.0 -> float32 in [0, 1]; channel axis added -> (28, 28, 1)",
        "architecture": {
            "type": "cnn",
            "blocks": "2 x (Conv2D 3x3 ReLU -> MaxPool 2x2); filters 32, 64",
            "head": "Flatten -> Dense(128, ReLU) -> Dense(10, softmax)",
            "padding": "valid",
        },
        "parameter_count": int(model.count_params()),
        "optimiser": {"name": "Adam", "learning_rate": args.lr},
        "loss": "sparse_categorical_crossentropy",
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "training_seconds": round(duration, 1),
        "selection_criterion": "lowest validation loss",
        "best_epoch": best_idx + 1,
        "best_val_loss": hist["val_loss"][best_idx],
        "best_val_accuracy": hist["val_accuracy"][best_idx],
        "model_path": str(model_path),
        "versions": {"tensorflow": tf.__version__, "keras": keras.__version__},
        "history": hist,
    }
    with open(run_dir / "run_record.json", "w") as f:
        json.dump(record, f, indent=2)

    plot_curves(hist, best_idx + 1, args.run_id, run_dir / "learning_curves.png")

    print(f"\nRun {args.run_id} finished in {duration:.1f}s")
    print(f"Best epoch: {best_idx + 1} | val_loss={hist['val_loss'][best_idx]:.4f} "
          f"| val_acc={hist['val_accuracy'][best_idx]:.4f}")
    print(f"Saved: {model_path}  and  {run_dir}")


if __name__ == "__main__":
    main()