"""Create the manual test images for the app.  Usage: python -m src.make_test_images"""
from pathlib import Path

import numpy as np
from PIL import Image

from src import synthetic_images as si
from src.data import load_raw_mnist, make_train_val_split

OUT = Path("test_images/generated")
OUT.mkdir(parents=True, exist_ok=True)

x_train, y_train, _, _ = load_raw_mnist()  # test set deliberately unused
_, x_val, _, y_val = make_train_val_split(x_train, y_train)


def first(label):
    return x_val[int(np.where(y_val == label)[0][0])]


files = {
    "01_clear_digit_dark_on_light.png": (si.png_bytes(si.drawn(first(7))), "ok, true label 7"),
    "02_small_off_centre.png": (si.png_bytes(si.small_off_centre(first(3))), "ok, true label 3"),
    "03_white_on_black.png": (si.png_bytes(si.white_on_black_large(first(2))), "ok, not inverted, true label 2"),
    "04_blank_white.png": (si.png_bytes(si.blank_white()), "rejected: blank"),
    "05_faint_noise.png": (si.png_bytes(si.faint_noise()), "rejected: blank or low contrast"),
    "06_two_digits.png": (si.png_bytes(si.two_digits(first(5), first(8))), "warning: more than one digit"),
    "07_non_digit_stripes.png": (si.png_bytes(si.stripes()), "no crash; observe and document the result"),
    "08_corrupt_truncated.png": (si.truncated_png(first(4)), "rejected: could not be read"),
    "09_fake_image.png": (b"this is plain text, not an image", "rejected: could not be read"),
    "10_too_many_pixels.png": (si.png_bytes(Image.new("L", (6000, 6000), 255)), "rejected: too many pixels"),
}
for name, (data, expect) in files.items():
    (OUT / name).write_bytes(data)
    print(f"{name:<34} expected: {expect}")

# Larger than the 5 MB limit: Streamlit itself blocks it before the app sees it.
noise = np.random.default_rng(0).integers(0, 256, (2600, 2600, 3), dtype=np.uint8)
Image.fromarray(noise).save(OUT / "11_oversized_noise.png")
print(f"{'11_oversized_noise.png':<34} expected: blocked by the 5 MB upload limit")

(OUT / "12_unsupported.txt").write_text("not an image")
print(f"{'12_unsupported.txt':<34} expected: upload control refuses the file type")