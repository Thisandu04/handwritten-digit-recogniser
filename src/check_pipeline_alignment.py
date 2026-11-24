"""Does the preprocessing pipeline preserve accuracy on MNIST-origin images?
Validation set only. Usage: python -m src.check_pipeline_alignment
"""
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from src.data import load_raw_mnist, make_train_val_split
from src.inference import load_digit_model
from src.preprocessing import InvalidImageError, preprocess_image

N = 2000
x_train, y_train, _, _ = load_raw_mnist()  # test set deliberately unused
_, x_val, _, y_val = make_train_val_split(x_train, y_train)  # same split indices (uint8)
x_val, y_val = x_val[:N], y_val[:N]
model = load_digit_model()


def to_png(im):
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def native(a):      # unchanged 28x28, white on black
    return Image.fromarray(a)


def drawn(a):       # 10x upscaled, black ink on white paper, with a wide margin
    big = Image.fromarray(a).resize((280, 280), Image.Resampling.BICUBIC)
    return ImageOps.expand(ImageOps.invert(big), border=60, fill=255)


def small_offcentre(a):   # small digit in the top-right of a large white page
    small = Image.fromarray(a).resize((84, 84), Image.Resampling.BICUBIC)
    page = Image.new("L", (300, 300), 0)
    page.paste(small, (180, 30))
    return ImageOps.invert(page)


def run(render):
    tensors, keep, rejected = [], [], 0
    for i, a in enumerate(x_val):
        try:
            tensors.append(preprocess_image(to_png(render(a))).tensor[0])
            keep.append(i)
        except InvalidImageError:
            rejected += 1
    probs = model.predict(np.stack(tensors), batch_size=256, verbose=0)
    return float((probs.argmax(axis=1) == y_val[keep]).mean()), rejected


direct_probs = model.predict((x_val.astype("float32") / 255.0)[..., None], batch_size=256, verbose=0)
direct = float((direct_probs.argmax(axis=1) == y_val).mean())

results = {"direct_28x28": {"accuracy": direct, "rejected": 0}}
for name, fn in [("pipeline_native_28x28", native), ("pipeline_drawn_10x_dark_on_light", drawn),
                 ("pipeline_small_off_centre", small_offcentre)]:
    acc, rej = run(fn)
    results[name] = {"accuracy": acc, "rejected": rej}

print(f"Validation images used: {N}\n")
print(f"{'input route':<36}{'accuracy':>10}{'vs direct':>11}{'rejected':>10}")
for name, r in results.items():
    print(f"{name:<36}{r['accuracy']:>10.4f}{r['accuracy'] - direct:>+11.4f}{r['rejected']:>10}")

Path("results/week8").mkdir(parents=True, exist_ok=True)
with open("results/week8/pipeline_alignment.json", "w") as f:
    json.dump({"n": N, "set": "validation", "results": results,
               "note": "images rendered from MNIST; a consistency check, not a real-world benchmark"}, f, indent=2)