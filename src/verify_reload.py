"""Verify that a saved CNN reloads in a fresh process and behaves consistently.

Run as TWO separate commands (two separate Python processes):
    python -m src.verify_reload --run-id cnn_baseline --save
    python -m src.verify_reload --run-id cnn_baseline
"""
import argparse
import json
from pathlib import Path

import numpy as np
from tensorflow import keras

from src.data import (add_channel_axis, load_raw_mnist,
                      make_train_val_split, scale_pixels)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default="cnn_baseline")
    p.add_argument("--save", action="store_true",
                   help="store reference outputs instead of comparing")
    args = p.parse_args()

    run_dir = Path("results") / args.run_id
    with open(run_dir / "run_record.json") as f:
        record = json.load(f)

    x_train, y_train, _, _ = load_raw_mnist()  # test set deliberately unused
    _, x_val, _, y_val = make_train_val_split(
        add_channel_axis(scale_pixels(x_train)), y_train
    )

    model = keras.models.load_model(record["model_path"])

    # Fixed sample: the split is deterministic, so these are always the same 10 images
    sample = x_val[:10]
    probs = model.predict(sample, verbose=0)
    ref_path = run_dir / "reload_reference.npy"

    print("True labels     :", y_val[:10].tolist())
    print("Predicted labels:", probs.argmax(axis=1).tolist())

    if args.save:
        np.save(ref_path, probs)
        lines = []
        model.summary(print_fn=lines.append)
        (run_dir / "model_summary.txt").write_text("\n".join(lines), encoding="utf-8")
        print(f"\nSaved reference outputs -> {ref_path}")
        print(f"Saved layer summary     -> {run_dir / 'model_summary.txt'}")
    else:
        ref = np.load(ref_path)
        diff = float(np.abs(probs - ref).max())
        same_labels = np.array_equal(probs.argmax(axis=1), ref.argmax(axis=1))
        print(f"\nCheck 1 (fresh process vs saved reference)")
        print(f"  Same predicted labels: {same_labels}")
        print(f"  Max absolute difference in outputs: {diff:.2e}  "
              f"-> {'PASS' if diff < 1e-6 else 'CHECK'}")

    # Check 2: reloaded model vs metrics recorded during training
    loss, acc = model.evaluate(x_val, y_val, batch_size=128, verbose=0)
    print("\nCheck 2 (reloaded model vs training-time record)")
    print(f"  Reloaded : val_loss={loss:.4f}  val_acc={acc:.4f}")
    print(f"  Recorded : val_loss={record['best_val_loss']:.4f}  "
          f"val_acc={record['best_val_accuracy']:.4f}")


if __name__ == "__main__":
    main()