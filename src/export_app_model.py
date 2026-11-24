"""Copy the chosen model to the stable app path and record its provenance.

Usage (from project root):
    python -m src.export_app_model --run-id cnn_aug_seed_42
"""
import argparse
import json
import shutil
from pathlib import Path

from src.freeze_model import sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default="cnn_aug_seed_42",
                   help="use cnn_baseline if you kept the baseline in Week 7")
    args = p.parse_args()

    with open(Path("results") / args.run_id / "run_record.json") as f:
        rec = json.load(f)

    src = Path(rec["model_path"])
    dst = Path("models/digit_cnn.keras")
    shutil.copyfile(src, dst)

    info = {
        "source_run_id": args.run_id,
        "source_model_path": str(src),
        "app_model_path": str(dst),
        "sha256": sha256(dst),
        "size_mb": round(dst.stat().st_size / 1e6, 2),
        "parameter_count": rec["parameter_count"],
        "architecture": rec["architecture"],
        "training_preprocessing": rec["preprocessing"],
        "validation_accuracy": rec["best_val_accuracy"],
        "validation_loss": rec["best_val_loss"],
        "versions": rec["versions"],
    }
    with open("models/digit_cnn_info.json", "w") as f:
        json.dump(info, f, indent=2)

    # If this model was frozen in Week 7, confirm the bytes are identical
    frozen = Path("results/week7_improved/frozen_config.json")
    if frozen.exists():
        cfg = json.load(open(frozen))
        if cfg["run_id"] == args.run_id:
            print("Matches the frozen model hash:", cfg["model_sha256"] == info["sha256"])

    print(f"Exported {src} -> {dst} ({info['size_mb']} MB)")
    print(f"SHA-256: {info['sha256']}")


if __name__ == "__main__":
    main()