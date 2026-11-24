"""Tests for src/preprocessing.py. Run from the project root:  python -m pytest -v"""
import io

import numpy as np
import pytest
from PIL import Image, ImageDraw
from scipy import ndimage

from src.preprocessing import MAX_FILE_BYTES, InvalidImageError, preprocess_image


# ---------- helpers: synthetic digits ----------
def draw_seven(size=280, margin=60, dark_on_light=True):
    img = Image.new("L", (size, size), 255 if dark_on_light else 0)
    ink = 0 if dark_on_light else 255
    t = max(3, size // 14)
    x0, x1, y0, y1 = margin, size - margin, margin, size - margin
    ImageDraw.Draw(img).line([(x0, y0), (x1, y0), (x0 + (x1 - x0) // 3, y1)], fill=ink, width=t)
    return img


def draw_one(size=280, margin=60):
    img = Image.new("L", (size, size), 255)
    ImageDraw.Draw(img).line([(size // 2, margin), (size // 2, size - margin)],
                             fill=0, width=max(3, size // 14))
    return img


def draw_two_ones(size=280):
    img = Image.new("L", (size, size), 255)
    d = ImageDraw.Draw(img)
    for x in (size // 4, 3 * size // 4):
        d.line([(x, 60), (x, size - 60)], fill=0, width=20)
    return img


def png(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def ink_box(image28):
    ys, xs = np.where(image28 > 0.3)
    return ys.max() - ys.min() + 1, xs.max() - xs.min() + 1


# ---------- clear single-digit image ----------
def test_output_contract():
    r = preprocess_image(png(draw_seven()))
    assert r.tensor.shape == (1, 28, 28, 1)
    assert r.tensor.dtype == np.float32
    assert 0.0 <= r.tensor.min() and r.tensor.max() <= 1.0
    assert r.tensor.max() > 0.5            # a digit is actually present
    assert r.warnings == []


def test_repeatable():
    data = png(draw_seven())
    assert np.array_equal(preprocess_image(data).tensor, preprocess_image(data).tensor)


def test_pil_and_bytes_inputs_agree():
    img = draw_seven()
    assert np.array_equal(preprocess_image(img).tensor, preprocess_image(png(img)).tensor)


# ---------- polarity ----------
def test_polarity_auto_detects_both_styles():
    dark_on_light = preprocess_image(png(draw_seven(dark_on_light=True)))
    light_on_dark = preprocess_image(png(draw_seven(dark_on_light=False)))
    assert dark_on_light.inverted is True
    assert light_on_dark.inverted is False
    assert np.allclose(dark_on_light.tensor, light_on_dark.tensor, atol=0.02)


def test_polarity_override_matches_auto_when_correct():
    data = png(draw_seven(dark_on_light=True))
    assert np.array_equal(preprocess_image(data, "dark_on_light").tensor,
                          preprocess_image(data, "auto").tensor)


def test_invalid_polarity_value():
    with pytest.raises(ValueError):
        preprocess_image(png(draw_seven()), polarity="sideways")


# ---------- unusual margins, size, aspect ratio ----------
@pytest.mark.parametrize("size,margin", [(280, 60), (500, 20), (100, 30), (640, 200)])
def test_size_and_margin_normalised(size, margin):
    r = preprocess_image(png(draw_seven(size, margin)))
    h, w = ink_box(r.image28)
    assert 17 <= max(h, w) <= 21                       # fitted to the 20x20 box
    cy, cx = ndimage.center_of_mass(r.image28)
    assert abs(cy - 13.5) <= 1.0 and abs(cx - 13.5) <= 1.0   # centred by mass


def test_aspect_ratio_preserved():
    h, w = ink_box(preprocess_image(png(draw_one())).image28)
    assert h >= 17          # tall
    assert 1 <= w <= 6      # still narrow, not stretched to a square


def test_jpeg_input():
    buf = io.BytesIO()
    draw_seven().convert("RGB").save(buf, format="JPEG", quality=50)
    r = preprocess_image(buf.getvalue())
    assert r.tensor.shape == (1, 28, 28, 1)
    assert 17 <= max(ink_box(r.image28)) <= 21


def test_transparent_png():
    img = Image.new("RGBA", (280, 280), (0, 0, 0, 0))
    ImageDraw.Draw(img).line([(60, 60), (220, 60), (113, 220)], fill=(0, 0, 0, 255), width=20)
    r = preprocess_image(png(img))
    assert r.inverted is True and r.tensor.max() > 0.5


# ---------- blank and near-blank ----------
@pytest.mark.parametrize("value", [255, 0, 128])
def test_uniform_images_rejected(value):
    with pytest.raises(InvalidImageError, match="blank|contrast"):
        preprocess_image(png(Image.new("L", (200, 200), value)))


def test_faint_noise_rejected():
    rng = np.random.default_rng(0)
    arr = np.clip(255 + rng.normal(0, 3, (200, 200)), 0, 255).astype(np.uint8)
    with pytest.raises(InvalidImageError):
        preprocess_image(png(Image.fromarray(arr)))


def test_tiny_speck_rejected():
    img = Image.new("L", (280, 280), 255)
    ImageDraw.Draw(img).rectangle([140, 140, 141, 141], fill=0)
    with pytest.raises(InvalidImageError, match="ink"):
        preprocess_image(png(img))


# ---------- multiple digits and edge contact: documented warnings ----------
def test_multiple_digits_warn():
    r = preprocess_image(png(draw_two_ones()))
    assert any("more than one digit" in w for w in r.warnings)


def test_single_digit_has_no_multi_warning():
    r = preprocess_image(png(draw_seven()))
    assert not any("more than one digit" in w for w in r.warnings)


def test_digit_touching_edge_warns():
    r = preprocess_image(png(draw_seven(margin=0)))
    assert any("edge" in w for w in r.warnings)


# ---------- corrupt, unsupported, oversized ----------
def test_empty_bytes():
    with pytest.raises(InvalidImageError, match="empty"):
        preprocess_image(b"")


def test_not_an_image():
    with pytest.raises(InvalidImageError, match="could not be read"):
        preprocess_image(b"this is definitely not an image")


def test_truncated_png():
    data = png(draw_seven())
    with pytest.raises(InvalidImageError):
        preprocess_image(data[: len(data) // 2])


def test_oversized_file():
    with pytest.raises(InvalidImageError, match="too large"):
        preprocess_image(b"\x00" * (MAX_FILE_BYTES + 1))


def test_too_many_pixels():
    with pytest.raises(InvalidImageError, match="too many pixels"):
        preprocess_image(png(Image.new("L", (6000, 6000), 255)))


def test_too_small_image():
    with pytest.raises(InvalidImageError, match="too small"):
        preprocess_image(png(Image.new("L", (4, 4), 255)))