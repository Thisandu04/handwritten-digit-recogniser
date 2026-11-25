"""Synthetic test inputs rendered from MNIST-origin digits (validation set only)."""
import io

import numpy as np
from PIL import Image, ImageOps


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def native(a):
    """Unchanged 28x28, white digit on black (uint8 array in)."""
    return Image.fromarray(a)


def drawn(a, margin=60):
    """10x upscaled, black ink on white paper with a margin."""
    big = Image.fromarray(a).resize((280, 280), Image.Resampling.BICUBIC)
    return ImageOps.expand(ImageOps.invert(big), border=margin, fill=255)


def small_off_centre(a):
    """Small dark digit in the top-right of a large white page."""
    small = Image.fromarray(a).resize((84, 84), Image.Resampling.BICUBIC)
    page = Image.new("L", (300, 300), 0)
    page.paste(small, (180, 30))
    return ImageOps.invert(page)


def white_on_black_large(a):
    """Large white digit on a black background."""
    return Image.fromarray(a).resize((400, 400), Image.Resampling.BICUBIC)


def two_digits(a, b):
    """Two digits side by side, dark on light."""
    page = Image.new("L", (440, 240), 0)
    page.paste(Image.fromarray(a).resize((200, 200), Image.Resampling.BICUBIC), (10, 20))
    page.paste(Image.fromarray(b).resize((200, 200), Image.Resampling.BICUBIC), (230, 20))
    return ImageOps.invert(page)


def stripes():
    """A non-digit pattern with plenty of ink."""
    y, x = np.indices((280, 280))
    return Image.fromarray((((x + y) // 20) % 2 * 255).astype("uint8"))


def blank_white():
    return Image.new("L", (280, 280), 255)


def faint_noise():
    rng = np.random.default_rng(0)
    arr = np.clip(255 + rng.normal(0, 3, (280, 280)), 0, 255).astype("uint8")
    return Image.fromarray(arr)


def truncated_png(a):
    data = png_bytes(drawn(a))
    return data[: len(data) // 2]