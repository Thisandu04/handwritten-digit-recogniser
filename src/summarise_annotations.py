"""Summarise manual error annotations. Usage: python -m src.summarise_annotations"""
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path("results/week7")
ALLOWED = ["ambiguous_stroke", "unusual_shape", "slanted_or_rotated", "thin_or_faint",
           "broken_or_cut_off", "label_questionable", "other"]

with open(OUT / "error_annotation.csv", newline="") as f:
    rows = list(csv.DictReader(f))

labelled = [r for r in rows if r["category"].strip()]
print(f"Annotated {len(labelled)} of {len(rows)} errors\n")

unknown = {r["category"] for r in labelled} - set(ALLOWED)
if unknown:
    print(f"WARNING: unrecognised categories (typos?): {sorted(unknown)}\n")

counts = Counter(r["category"].strip() for r in labelled)
for cat in ALLOWED:
    n = counts.get(cat, 0)
    print(f"{cat:<22}{n:>4}  ({n / max(len(labelled), 1):.0%})")

print("\nMost common confusion pair per category (true->pred)")
for cat in ALLOWED:
    pairs = Counter(f"{r['true']}->{r['pred']}" for r in labelled if r["category"].strip() == cat)
    if pairs:
        print(f"  {cat:<22}{pairs.most_common(2)}")

with open(OUT / "annotation_summary.json", "w") as f:
    json.dump({"n_errors": len(rows), "n_annotated": len(labelled),
               "counts": {c: counts.get(c, 0) for c in ALLOWED}}, f, indent=2)

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.barh(ALLOWED[::-1], [counts.get(c, 0) for c in ALLOWED[::-1]])
ax.set_xlabel("Number of annotated errors")
ax.set_title("Manual grouping of test errors (one primary category each)")
fig.tight_layout()
fig.savefig(OUT / "error_categories.png", dpi=150)
plt.close(fig)