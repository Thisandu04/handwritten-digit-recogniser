"""Streamlit interface. Run from the project root:  streamlit run app/app.py"""
import json
import logging
import sys
from io import BytesIO
from pathlib import Path

# Streamlit puts app/ on sys.path, not the project root, so add the root for `src` imports.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
from PIL import Image

from src.app_logic import LOW_SCORE_THRESHOLD, analyse_image
from src.inference import load_digit_model

st.set_page_config(page_title="Handwritten Digit Recogniser", layout="centered")

POLARITY_OPTIONS = {
    "Detect automatically": "auto",
    "Dark digit on light background": "dark_on_light",
    "Light digit on dark background": "light_on_dark",
}


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    """Loaded once per server process, not on every interaction."""
    return load_digit_model()


def show_original(data: bytes):
    """Display the uploaded image if it can be decoded; ignore failures."""
    try:
        st.image(Image.open(BytesIO(data)), caption="Uploaded image", use_container_width=True)
    except Exception:
        st.caption("The uploaded file cannot be displayed.")


def show_processed(image28):
    arr = (image28 * 255).astype("uint8")
    big = Image.fromarray(arr).resize((224, 224), Image.Resampling.NEAREST)
    st.image(big, caption="Processed 28x28 model input (enlarged)")


st.title("Handwritten Digit Recogniser")
st.write("Upload one image containing a single handwritten digit (0-9).")

with st.sidebar:
    st.header("About")
    info_path = ROOT / "models" / "digit_cnn_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        st.write(f"Model: CNN from run `{info['source_run_id']}`, "
                 f"{info['parameter_count']:,} parameters.")
        st.write(f"Validation accuracy during training: {info['validation_accuracy']:.2%} "
                 "(MNIST validation set, not a measure of accuracy on your images).")
    st.markdown(
        "**Limitations**\n"
        "- Trained on MNIST only. Photos, shadows and unusual writing may fail.\n"
        "- One digit per image.\n"
        "- The app cannot verify that the image really contains a digit.\n"
        "- Scores are model outputs, not calibrated probabilities or guarantees.\n\n"
        "**Privacy:** uploads are processed in memory and are not saved by this app."
    )

uploaded = st.file_uploader("Image (PNG or JPG, max 5 MB)", type=["png", "jpg", "jpeg"])
polarity_label = st.radio("Ink colour", list(POLARITY_OPTIONS), horizontal=True)

if uploaded is None:
    st.info("Choose an image to begin.")
    st.stop()

data = uploaded.getvalue()

try:
    model = get_model()
except Exception:
    logging.exception("Model failed to load")
    st.error("The model could not be loaded. Check that models/digit_cnn.keras exists "
             "(run: python -m src.export_app_model).")
    st.stop()

try:
    result = analyse_image(model, data, POLARITY_OPTIONS[polarity_label])
except Exception:
    logging.exception("Unexpected error during analysis")
    st.error("Something unexpected went wrong while processing this image. "
             "Please try a different image.")
    st.stop()

left, right = st.columns(2)
with left:
    show_original(data)

if result.status == "rejected":
    st.error(result.message)
    st.stop()

with right:
    show_processed(result.image28)

if result.inverted:
    st.caption("The image was detected as a dark digit on a light background and was inverted.")
for w in result.warnings:
    st.warning(w)
if result.low_score:
    st.warning(f"The model's highest score is below {LOW_SCORE_THRESHOLD:.2f}, so the model is "
               "not clearly favouring one digit. Treat this result with extra caution.")

st.metric("Model's top prediction", str(result.digit))
st.caption("This is the digit with the highest model score. It is not a guarantee that the "
           "prediction is correct.")

scores = pd.DataFrame({"digit": [str(d) for d in range(10)],
                       "model score": [float(s) for s in result.scores]}).set_index("digit")
st.bar_chart(scores)
st.write("Top 3 model scores: " + ", ".join(f"{d} ({s:.3f})" for d, s in result.top3))
st.caption("Model scores are softmax outputs. They are not calibrated probabilities, and "
           "the app cannot check that the image contains a digit at all.")