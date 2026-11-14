"""Freeze the selected model BEFORE the test set is used.

Usage (from project root):
    python -m src.freeze_model --run-id cnn_baseline
"""
import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import tensorflow as tf
from tensorflow import keras


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unavailable"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default="cnn_baseline")
    args = p.parse_args()

    with open(Path("results") / args.run_id / "run_record.json") as f:
        rec = json.load(f)

    frozen = {
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": args.run_id,
        "model_path": rec["model_path"],
        "model_sha256": sha256(rec["model_path"]),
        "git_commit_at_freeze": git_commit(),
        "seed": rec["seed"],
        "architecture": rec["architecture"],
        "parameter_count": rec["parameter_count"],
        "preprocessing": rec["preprocessing"],
        "optimiser": rec["optimiser"],
        "batch_size": rec["batch_size"],
        "epochs": rec["epochs"],
        "selection_criterion": rec["selection_criterion"],
        "selected_epoch": rec["best_epoch"],
        "validation_loss": rec["best_val_loss"],
        "validation_accuracy": rec["best_val_accuracy"],
        "versions": {"tensorflow": tf.__version__, "keras": keras.__version__},
        "evaluation_script": "src/evaluate.py",
        "test_set_used_before_freeze": False,
    }

    out = Path("results/week7")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "frozen_config.json", "w") as f:
        json.dump(frozen, f, indent=2)

    print(f"Frozen: {frozen['model_path']}")
    print(f"SHA-256: {frozen['model_sha256']}")
    print(f"Validation: loss={frozen['validation_loss']:.4f} "
          f"acc={frozen['validation_accuracy']:.4f} (epoch {frozen['selected_epoch']})")
    print(f"Saved: {out / 'frozen_config.json'}")


if __name__ == "__main__":
    main()