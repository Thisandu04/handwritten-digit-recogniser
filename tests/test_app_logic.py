"""Functional and negative tests for src/app_logic.py (uses the real saved model).
Run from the project root:  python -m pytest -v
"""
import numpy as np
import pytest

from src import synthetic_images as si
from src.app_logic import analyse_image
from src.data import load_raw_mnist, make_train_val_split
from src.inference import DEFAULT_MODEL_PATH, load_digit_model
from src.preprocessing import MAX_FILE_BYTES

N = 40  # validation images used per consistency check
AGREEMENT = 0.90  # sanity threshold chosen for these tests, not a performance claim


@pytest.fixture(scope="module")
def model():
    if not DEFAULT_MODEL_PATH.exists():
        pytest.skip("models/digit_cnn.keras not found; run: python -m src.export_app_model")
    return load_digit_model()


@pytest.fixture(scope="module")
def val():
    x_train, y_train, _, _ = load_raw_mnist()  # test set deliberately unused
    _, x_val, _, y_val = make_train_val_split(x_train, y_train)
    return x_val[:N], y_val[:N]


@pytest.fixture(scope="module")
def direct_labels(model, val):
    x, _ = val
    return model.predict(x.astype("float32")[..., None] / 255.0, verbose=0).argmax(axis=1)


def label_of(model, img):
    r = analyse_image(model, si.png_bytes(img))
    return None if r.status == "rejected" else r.digit


# ---------- clear single-digit image ----------
def test_clear_digit_returns_valid_result(model, val):
    r = analyse_image(model, si.png_bytes(si.drawn(val[0][0])))
    assert r.status in ("ok", "warning")
    assert r.digit in range(10)
    assert r.scores.shape == (10,) and abs(float(r.scores.sum()) - 1.0) < 1e-4
    assert r.image28.shape == (28, 28)
    assert [d for d, _ in r.top3][0] == r.digit


def test_repeatable(model, val):
    data = si.png_bytes(si.drawn(val[0][0]))
    a, b = analyse_image(model, data), analyse_image(model, data)
    assert a.digit == b.digit and np.array_equal(a.scores, b.scores)


# ---------- unusual margins and size, and polarity ----------
@pytest.mark.parametrize("name,render", [
    ("drawn_margin_60", lambda a: si.drawn(a, 60)),
    ("drawn_margin_20", lambda a: si.drawn(a, 20)),
    ("drawn_margin_120", lambda a: si.drawn(a, 120)),
    ("small_off_centre", si.small_off_centre),
    ("white_on_black_large", si.white_on_black_large),
])
def test_render_variants_agree_with_direct_input(model, val, direct_labels, name, render):
    x, _ = val
    labels = [label_of(model, render(a)) for a in x]
    agreement = np.mean([l == d for l, d in zip(labels, direct_labels)])
    print(f"{name}: agreement with direct 28x28 prediction = {agreement:.3f}")
    assert agreement >= AGREEMENT


def test_polarity_flags(model, val):
    a = val[0][0]
    assert analyse_image(model, si.png_bytes(si.drawn(a))).inverted is True
    assert analyse_image(model, si.png_bytes(si.white_on_black_large(a))).inverted is False


# ---------- blank and near-blank ----------
@pytest.mark.parametrize("img", [si.blank_white(), si.faint_noise()])
def test_blank_and_near_blank_rejected_without_prediction(model, img):
    r = analyse_image(model, si.png_bytes(img))
    assert r.status == "rejected"
    assert r.digit is None and r.scores is None
    assert r.message


# ---------- multiple digits ----------
def test_multiple_digits_warn(model, val):
    x, y = val
    a, b = x[int(np.where(y == 5)[0][0])], x[int(np.where(y == 8)[0][0])]
    r = analyse_image(model, si.png_bytes(si.two_digits(a, b)))
    assert r.status == "warning"
    assert any("more than one digit" in w for w in r.warnings)
    assert r.digit is not None   # a result is shown, with a warning


# ---------- non-digit: documented limitation ----------
def test_non_digit_does_not_crash(model):
    r = analyse_image(model, si.png_bytes(si.stripes()))
    print(f"stripes -> status={r.status}, digit={r.digit}, "
          f"top score={None if r.scores is None else float(r.scores.max()):.3f}")
    assert r.status in ("ok", "warning", "rejected")   # no exception is the requirement


# ---------- corrupt, unsupported, oversized ----------
def test_corrupt_and_unreadable_inputs_rejected(model, val):
    for data in (b"", b"this is not an image", si.truncated_png(val[0][0])):
        r = analyse_image(model, data)
        assert r.status == "rejected" and r.digit is None


def test_oversized_file_rejected(model):
    r = analyse_image(model, b"\x00" * (MAX_FILE_BYTES + 1))
    assert r.status == "rejected" and "too large" in r.message