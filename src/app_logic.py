"""Application logic: combines preprocessing and prediction into one result.

No Streamlit code here, so it can be tested without a browser.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.inference import predict_digit
from src.preprocessing import InvalidImageError, preprocess_image

# Heuristic for a visible caution, NOT a validated rejection rule. Week 5 and Week 7
# showed that wrong predictions can have high scores, and non-digits can too.
LOW_SCORE_THRESHOLD = 0.60


@dataclass
class AnalysisResult:
    status: str                       # "ok", "warning" or "rejected"
    message: str | None = None        # reason, only when rejected
    warnings: list = field(default_factory=list)
    digit: int | None = None
    scores: np.ndarray | None = None  # model scores (softmax outputs)
    top3: list = field(default_factory=list)
    image28: np.ndarray | None = None
    inverted: bool | None = None
    low_score: bool = False


def analyse_image(model, data: bytes, polarity: str = "auto") -> AnalysisResult:
    """Preprocess uploaded bytes and classify them. Never raises for bad user input."""
    try:
        pre = preprocess_image(data, polarity)
    except InvalidImageError as exc:
        return AnalysisResult(status="rejected", message=str(exc))

    pred = predict_digit(model, pre.tensor)
    low = bool(pred["scores"].max() < LOW_SCORE_THRESHOLD)
    return AnalysisResult(
        status="warning" if (pre.warnings or low) else "ok",
        warnings=list(pre.warnings),
        digit=pred["digit"],
        scores=pred["scores"],
        top3=pred["top3"],
        image28=pre.image28,
        inverted=pre.inverted,
        low_score=low,
    )