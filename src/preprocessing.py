"""Image preprocessing for the digit recogniser (no model or UI code here).

Contract (each step is tested in tests/test_preprocessing.py):
 1. Decode bytes, enforce size limits, fix EXIF orientation
 2. Composite transparency onto white, convert to grayscale
 3. Detect polarity from the border; convert to light digit on dark background
 4. Subtract the background level; reject blank / low-contrast images
 5. Threshold to find strokes; reject images with almost no ink; collect warnings
 6. Crop to the strokes; resize to fit a 20x20 box, keeping the aspect ratio
 7. Paste into a 28x28 canvas and centre by centre of mass (as MNIST was prepared)
 8. Return float32 values in [0, 1] with shape (1, 28, 28, 1)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

MAX_FILE_BYTES = 5 * 1024 * 1024   # 5 MB
MAX_PIXELS = 25_000_000            # 25 megapixels
MIN_SIDE = 8                       # pixels
MIN_CONTRAST = 40.0                # on the 0-255 scale, after background removal
INK_THRESHOLD = 0.3                # fraction of the peak that counts as stroke
MIN_INK_PIXELS = 12
MIN_INK_FRACTION = 0.0005          # of all pixels
MULTI_SHARE = 0.2                  # a component holding >= 20% of the ink counts as a digit
BOX = 20                           # digit is fitted into a 20x20 box
CANVAS = 28
POLARITIES = ("auto", "dark_on_light", "light_on_dark")


class InvalidImageError(ValueError):
    """Input cannot be used. The message is safe to show to the user."""


@dataclass
class PreprocessResult:
    tensor: np.ndarray      # (1, 28, 28, 1) float32 in [0, 1], ready for the model
    image28: np.ndarray     # (28, 28) float32, for display
    inverted: bool          # True if the image was converted from dark-on-light
    warnings: list = field(default_factory=list)


def decode_image(data: bytes) -> Image.Image:
    """Bytes -> PIL image, with size limits and safe failure."""
    if not data:
        raise InvalidImageError("The file is empty.")
    if len(data) > MAX_FILE_BYTES:
        raise InvalidImageError(
            f"The file is too large ({len(data) / 1e6:.1f} MB). "
            f"The maximum is {MAX_FILE_BYTES / 1e6:.0f} MB.")
    try:
        Image.open(BytesIO(data)).verify()   # structural check; invalidates the object
        img = Image.open(BytesIO(data))
        if img.width * img.height > MAX_PIXELS:
            raise InvalidImageError(
                f"The image has too many pixels ({img.width}x{img.height}). "
                f"The maximum is {MAX_PIXELS // 1_000_000} megapixels.")
        img.load()
        img = ImageOps.exif_transpose(img)
    except InvalidImageError:
        raise
    except Exception as exc:  # corrupt, truncated, unsupported format, decompression bomb...
        raise InvalidImageError(
            "The file could not be read as an image. It may be corrupt "
            "or in an unsupported format.") from exc
    return img


def _to_gray(img: Image.Image) -> np.ndarray:
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white, rgba)   # transparency -> white background
    return np.asarray(img.convert("L"), dtype=np.float32)


def preprocess_image(source, polarity: str = "auto") -> PreprocessResult:
    """Turn uploaded bytes (or a PIL image) into a model-ready tensor.

    polarity: "auto" (detect from the border), "dark_on_light" or "light_on_dark".
    Raises InvalidImageError for unusable input.
    """
    if polarity not in POLARITIES:
        raise ValueError(f"polarity must be one of {POLARITIES}")

    if isinstance(source, Image.Image):
        img = source
    else:
        img = decode_image(bytes(source))
    if min(img.size) < MIN_SIDE:
        raise InvalidImageError(
            f"The image is too small (minimum {MIN_SIDE}x{MIN_SIDE} pixels).")

    gray = _to_gray(img)
    h, w = gray.shape

    # ---- polarity: the border should be background ----
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    border_median = float(np.median(border))
    inverted = (border_median > 127.5) if polarity == "auto" else (polarity == "dark_on_light")
    if inverted:
        gray = 255.0 - gray
        border_median = 255.0 - border_median

    # ---- background removal and blank check ----
    signal = np.clip(gray - border_median, 0.0, None)
    peak = float(signal.max())
    if peak < MIN_CONTRAST:
        raise InvalidImageError(
            "The image looks blank or has too little contrast to contain a digit.")
    signal /= peak

    # ---- strokes ----
    mask = signal > INK_THRESHOLD
    if int(mask.sum()) < max(MIN_INK_PIXELS, int(MIN_INK_FRACTION * mask.size)):
        raise InvalidImageError(
            "Almost no ink was found in the image. Please upload a clearer digit.")

    warnings = []
    labels, n = ndimage.label(mask, structure=np.ones((3, 3)))   # 8-connectivity
    if n > 1:
        sizes = np.bincount(labels.ravel())[1:]
        if int((sizes >= MULTI_SHARE * sizes.sum()).sum()) >= 2:
            warnings.append("The image may contain more than one digit. Only a single "
                            "digit is supported, so the result is unreliable.")

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0, r1, c0, c1 = int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])
    if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
        warnings.append("The digit touches the image edge and may be cut off.")

    # ---- crop, resize keeping aspect ratio, centre by centre of mass ----
    crop = signal[r0:r1 + 1, c0:c1 + 1]
    ch, cw = crop.shape
    scale = BOX / max(ch, cw)
    new_h, new_w = max(1, round(ch * scale)), max(1, round(cw * scale))
    small = Image.fromarray((crop * 255).astype(np.uint8)).resize(
        (new_w, new_h), Image.Resampling.LANCZOS)
    small = np.asarray(small, dtype=np.float32) / 255.0

    canvas = np.zeros((CANVAS, CANVAS), dtype=np.float32)
    top, left = (CANVAS - new_h) // 2, (CANVAS - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = small
    if canvas.max() <= 0:
        raise InvalidImageError("No usable digit could be extracted from the image.")

    cy, cx = ndimage.center_of_mass(canvas)
    rr = np.where(canvas.max(axis=1) > 0)[0]
    cc = np.where(canvas.max(axis=0) > 0)[0]
    shift_y = int(np.clip(round(13.5 - cy), -rr[0], CANVAS - 1 - rr[-1]))  # never push the
    shift_x = int(np.clip(round(13.5 - cx), -cc[0], CANVAS - 1 - cc[-1]))  # digit off-canvas
    canvas = ndimage.shift(canvas, (shift_y, shift_x), order=0, mode="constant", cval=0.0)

    image28 = np.clip(canvas, 0.0, 1.0).astype(np.float32)
    return PreprocessResult(tensor=image28[None, :, :, None], image28=image28,
                            inverted=bool(inverted), warnings=warnings)