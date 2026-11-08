"""Summarise validation results across seeds. Usage: python -m src.summarise_seeds"""
import json
from pathlib import Path

import numpy as np

GROUPS = {
    "dense": ["dense_baseline", "dense_seed_1", "dense_seed_2", "dense_seed_3"],
    "cnn": ["cnn_baseline", "cnn_seed_1", "cnn_seed_2", "cnn_seed_3"],
}

summary = {}
for name, runs in GROUPS.items():
    recs = []
    for r in runs:
        with open(Path("results") / r / "run_record.json") as f:
            recs.append(json.load(f))
    acc = np.array([r["best_val_accuracy"] for r in recs])
    loss = np.array([r["best_val_loss"] for r in recs])

    print(f"\n=== {name} ({len(recs)} seeds) ===")
    for r in recs:
        print(f"  seed {r['seed']:>3}: val_acc={r['best_val_accuracy']:.4f}  "
              f"val_loss={r['best_val_loss']:.4f}  best_epoch={r['best_epoch']}")
    print(f"  accuracy: mean={acc.mean():.4f}  std={acc.std(ddof=1):.4f}  "
          f"min={acc.min():.4f}  max={acc.max():.4f}")
    print(f"  loss    : mean={loss.mean():.4f}  std={loss.std(ddof=1):.4f}  "
          f"min={loss.min():.4f}  max={loss.max():.4f}")
    summary[name] = {"seeds": [r["seed"] for r in recs],
                     "val_accuracy": acc.tolist(), "val_loss": loss.tolist(),
                     "acc_mean": float(acc.mean()), "acc_std": float(acc.std(ddof=1))}

Path("results/week6_comparison").mkdir(parents=True, exist_ok=True)
with open("results/week6_comparison/seed_summary.json", "w") as f:
    json.dump(summary, f, indent=2)