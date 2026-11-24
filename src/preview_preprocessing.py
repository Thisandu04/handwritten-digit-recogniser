"""Show what preprocessing does to an image.

Usage:  python -m src.preview_preprocessing path\\to\\digit.png [--polarity auto]
"""
import argparse
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from src.preprocessing import InvalidImageError, preprocess_image

p = argparse.ArgumentParser()
p.add_argument("image")
p.add_argument("--polarity", default="auto", choices=["auto", "dark_on_light", "light_on_dark"])
args = p.parse_args()

data = Path(args.image).read_bytes()
try:
    result = preprocess_image(data, args.polarity)
except InvalidImageError as exc:
    raise SystemExit(f"Rejected: {exc}")

print(f"Inverted: {result.inverted}")
print("Warnings:", result.warnings or "none")

fig, (a, b) = plt.subplots(1, 2, figsize=(7, 3.6))
a.imshow(Image.open(BytesIO(data)).convert("RGB"))
a.set_title("Uploaded image")
b.imshow(result.image28, cmap="gray", vmin=0, vmax=1)
b.set_title("Model input (28x28)")
for ax in (a, b):
    ax.axis("off")
out = Path("results/week8")
out.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(out / f"preview_{Path(args.image).stem}.png", dpi=150)
plt.show()