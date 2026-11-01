# Handwritten Digit Recogniser

A CNN-based handwritten digit classifier trained on MNIST, with a Streamlit app for inference.

## Progress

- [x] **Week 5:** Dataset analysis and dense baseline
- [ ] **Week 6:** CNN design, training and baseline comparison
- [ ] **Week 7:** Test evaluation, error analysis and report
- [ ] **Week 8:** Streamlit app, tests, documentation and demo of the project

---

## Week 5: Neural-network foundations and dense baseline

### Objective

Understand and inspect the MNIST dataset, then build a reproducible dense (fully connected) baseline.

### Data

- **Source:** MNIST, loaded with `keras.datasets.mnist`.
- **Size:** 60,000 training and 10,000 test images, 28x28 grayscale, labels 0-9.
- **Validation split:** 10% of the training portion (6,000 images), stratified, `random_state=42`. Defined once in `src/data.py` and shared by every run.
- **Test set:** not used for model selection, tuning or choosing epochs.
- **Preprocessing:** pixels divided by 255, giving float32 values in [0, 1]. This keeps inputs in a small range, which generally helps gradient-based optimisation.

**Observations from the data:**
- *(Class balance: which classes are largest/smallest, and by how much.)*
- *(Visual variation: stroke width, slant, alignment, ambiguous pairs such as 4/9 or 1/7.)*

### Project structure

```
handwritten-digit-recogniser/
├── src/            # Reusable code: data, models, training and investigation scripts
├── notebooks/      # Exploratory work only
├── app/            # Streamlit interface (Week 8)
├── tests/          # Automated checks (Week 8)
├── results/        # Plots, run records (JSON), reports
├── models/         # Saved model files
├── requirements.txt
└── README.md
```

Training, evaluation and app code are kept separate so that nothing depends on notebook execution order.

### Setup

```powershell
git clone https://github.com/Thisandu04/handwritten-digit-recogniser.git
cd handwritten-digit-recogniser
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Versions used:** Python `3.14.4`, TensorFlow `2.22.0-rc0`, Keras `3.16.0.dev2026102704`.

### Commands

Run from the project root.

| Task | Command |
|---|---|
| Explore the dataset | `python -m src.explore_data` |
| Train the dense baseline | `python -m src.train_dense --run-id dense_baseline` |
| Learning-rate runs | `python -m src.train_dense --run-id dense_lr_0.01 --lr 0.01` and `... --run-id dense_lr_0.0001 --lr 0.0001` |
| Flattening investigation | `python -m src.investigate_flattening` |
| Softmax investigation | `python -m src.investigate_softmax` |

### Dense baseline model

`Flatten -> Dense(128, ReLU) -> Dense(10, softmax)`

- **Parameters:** 101,770, computed as (784x128 + 128) + (128x10 + 10).
- **Loss:** sparse categorical cross-entropy (the same as categorical cross-entropy, but accepting integer labels).
- **Optimiser:** Adam, learning rate 0.001, batch size 128, 15 epochs, seed 42.
- **Selection rule:** checkpoint with the lowest validation loss.
- **Iterations per epoch:** 54,000 / 128 = 422 (rounded up).

Each run writes `results/<run_id>/run_record.json` (seed, split, architecture, parameter count, optimiser settings, epoch-wise metrics, training time, selected checkpoint) and `learning_curves.png`.

### Results (validation set, measured)

| Run | Learning rate | Best epoch | Val loss | Val accuracy | Training time (s) |
|---|---|---|---|---|---|
| dense_lr_0.0001 | 0.0001 | 15 | 0.1663 | 0.9736 | 19.2 |
| dense_baseline | 0.001 | 9 | 0.0989 | 0.9536 | 20.4 |
| dense_lr_0.01 | 0.01 | 2 | 0.1195 | 0.9649 | 19.5 |

**Learning curves:** `results/dense_baseline/learning_curves.png`

**Interpretation:** *(2-3 sentences: does the smaller rate learn more slowly, is the larger rate less stable, is there evidence of overfitting?)*

**Caveat:** each configuration was run once with one seed, so small differences may be noise.

### Investigations

**Flattening (`results/week5_flattening/`)**
- *Pixel permutation:* baseline val accuracy `0.9737` vs permuted `0.9713`. *(What this implies about whether the dense model uses spatial layout.)*
- *Translation sensitivity:* *(how accuracy changes from 0 to ±4 pixel shifts; see `shift_sensitivity_dense.png`.)*

**Softmax scores (`results/week5_softmax/`)**
- *(Is accuracy close to mean confidence in each bin?)*
- *(How many wrong predictions had a score above 0.9?)*
- *(Score on a blank image and on random noise.)*
- **Conclusion:** a softmax score is a model score, not a calibrated probability or a guarantee of correctness.

### Limitations

- Results are validation-only and single-seed; the test set has not been used.
- The dense baseline treats pixels as an unordered vector, so it ignores local structure.
- Softmax scores can be high for non-digit inputs.

### Documents

- Technical note: `results/week5_technical_note.md`

### References

- TensorFlow CNN tutorial: https://www.tensorflow.org/tutorials/images/cnn
- MNIST (TFDS catalogue): https://www.tensorflow.org/datasets/catalog/mnist
- Keras: https://keras.io/
- scikit-learn metrics: https://scikit-learn.org/stable/modules/model_evaluation.html