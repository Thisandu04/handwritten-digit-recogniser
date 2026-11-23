"""Week 7 Part C: baseline vs augmented CNN on VALIDATION data only.
Applies the pre-registered decision rule. Usage: python -m src.compare_improvement
"""
import json
from pathlib import Path

import numpy as np

OUT = Path("results/week7_improvement")
OUT.mkdir(parents=True, exist_ok=True)
GROUPS = {
    "baseline": ["cnn_baseline", "cnn_seed_1", "cnn_seed_2", "cnn_seed_3"],
    "augmented": ["cnn_aug_seed_42", "cnn_aug_seed_1", "cnn_aug_seed_2", "cnn_aug_seed_3"],
}
N_VAL = 6000

stats = {}
for name, runs in GROUPS.items():
    recs = [json.load(open(Path("results") / r / "run_record.json")) for r in runs]
    acc = np.array([r["best_val_accuracy"] for r in recs])
    loss = np.array([r["best_val_loss"] for r in recs])
    ep = [r["best_epoch"] for r in recs]
    stats[name] = {"runs": runs, "acc": acc, "loss": loss, "epochs": ep,
                   "train_seconds": [r["training_seconds"] for r in recs]}

    print(f"\n=== {name} ===")
    for r in recs:
        print(f"  seed {r['seed']:>3}: val_acc={r['best_val_accuracy']:.4f} "
              f"({round((1 - r['best_val_accuracy']) * N_VAL)} errors)  "
              f"val_loss={r['best_val_loss']:.4f}  best_epoch={r['best_epoch']}")
    print(f"  accuracy: mean={acc.mean():.4f} std={acc.std(ddof=1):.4f} "
          f"min={acc.min():.4f} max={acc.max():.4f}")
    print(f"  loss    : mean={loss.mean():.4f} std={loss.std(ddof=1):.4f} "
          f"min={loss.min():.4f} max={loss.max():.4f}")

b, a = stats["baseline"], stats["augmented"]
loss_better = bool(a["loss"].mean() < b["loss"].mean())
acc_better = bool(a["acc"].mean() > b["acc"].mean())
adopt = bool(loss_better and acc_better)
overlap = bool(a["acc"].min() <= b["acc"].max() and b["acc"].min() <= a["acc"].max())

print("\n=== Pre-registered decision rule ===")
print(f"Mean val loss lower  : {loss_better}  ({a['loss'].mean():.4f} vs {b['loss'].mean():.4f})")
print(f"Mean val accuracy higher: {acc_better}  ({a['acc'].mean():.4f} vs {b['acc'].mean():.4f})")
print(f"Mean accuracy change : {a['acc'].mean() - b['acc'].mean():+.4f} "
      f"(about {(a['acc'].mean() - b['acc'].mean()) * N_VAL:+.0f} images of {N_VAL})")
print(f"Accuracy ranges overlap across seeds: {overlap}")
print(f"Mean train time: baseline {np.mean(b['train_seconds']):.0f}s | "
      f"augmented {np.mean(a['train_seconds']):.0f}s")
print(f"\nDECISION: {'ADOPT augmentation (freeze seed-42 augmented model)' if adopt else 'KEEP baseline (augmentation reported as a negative or inconclusive result)'}")

# ---- Secondary evidence: robustness (only if both runs exist) ----
rb = Path("results/week7_robustness/robustness_results.json")
ra = Path("results/week7_robustness_aug/robustness_results.json")
if rb.exists() and ra.exists():
    jb, ja = json.load(open(rb)), json.load(open(ra))
    print("\n=== Robustness on validation (secondary, exploratory) ===")
    print(f"Original validation: baseline {jb['baseline_accuracy']:.4f} | augmented {ja['baseline_accuracy']:.4f}")
    print(f"{'transformation':<22}{'level':>7}{'baseline':>10}{'augmented':>11}{'diff':>8}")
    for x, y in zip(jb["results"], ja["results"]):
        print(f"{x['family']:<22}{x['level']:>7}{x['accuracy']:>10.4f}{y['accuracy']:>11.4f}"
              f"{y['accuracy'] - x['accuracy']:>+8.4f}")

with open(OUT / "improvement_decision.json", "w") as f:
    json.dump({"decision_adopt": adopt, "loss_better": loss_better, "acc_better": acc_better,
               "ranges_overlap": overlap,
               "baseline": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in b.items()},
               "augmented": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in a.items()}},
              f, indent=2)