# Handwritten Digit Recogniser

A convolutional neural network (CNN) that classifies handwritten digits (0-9), trained and evaluated on MNIST, with a Streamlit app that accepts an uploaded image of a single digit. Built as Project 2 of a 12-week AI/ML student development programme (Weeks 5-8).

The focus is on justified technical decisions, reproducible experiments and honest analysis of limitations, not only on accuracy.

## Screenshots

| Valid digit | Handled error (blank image) |
|---|---|
| ![Valid digit](results\screenshots\01_valid_digit.png) | ![Rejected blank image](results\screenshots\04_rejected_blank.png) |

More: white-on-black input (`02_`), multiple-digit warning (`03_`), corrupt file (`05_`), upload limit (`06_`) and limitations sidebar (`07_`) in `results/screenshots/`.

## Key results

| Model | Params | Val accuracy | Test accuracy | Notes |
|---|---|---|---|---|
| Dense baseline | 101,770 | 97.37% | not evaluated | Week 5 baseline, validation only |
| CNN (frozen in Week 7) | 225,034 | 98.65% | 99.03% | Evaluated once on the 10,000 test images |
| CNN + augmentation (app model) | 225,034 | 98.98% (seed 42) | 99.19% | Adopted by a pre-registered validation rule; evaluated once |

- Validation numbers are for seed 42. Across four seeds the augmented CNN averaged 99.02% against 98.74% for the baseline CNN.
- The augmented app model's test accuracy has an approximate 95% interval of ±0.18 percentage points (test-set size only; it ignores seed variation).
- On the frozen baseline test set, the weakest class was digit 9 (recall 97.92%); the most frequent confusion was 9 predicted as 4 (8 cases).
- Full evaluation: `results/week7_evaluation_report.md`.

## Quick start

Tested with Python 3.14.4 on Windows. Package versions are listed below.

```powershell
git clone https://github.com/Thisandu04/handwritten-digit-recogniser.git
cd handwritten-digit-recogniser
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run app/app.py
```

On macOS/Linux, activate with `source .venv/bin/activate`. The trained model (`models/digit_cnn.keras`) is included in the repository, so no training is needed to run the app. Then open http://localhost:8501 and upload a PNG or JPG of a single digit.

**Versions used:** Python 3.14.4, TensorFlow 2.22.0-rc0, Keras 3.16.0.dev2026102704, Streamlit 1.66.1. See `requirements.txt`.

## How the app behaves

| Situation | Behaviour |
|---|---|
| Valid single digit | Shows the processed 28x28 input, the model's top prediction and all ten model scores |
| Warning cases (several digits, digit cut off, low top score) | Shows the result and a visible warning |
| Blank, near-blank, corrupt, too large or too small | Shows an error and no prediction |
| Non-digit image with plenty of ink | Not detectable; may return a prediction (documented limitation) |

Limits: PNG or JPG, at most 5 MB, at most 25 megapixels, one digit per image. Uploads are processed in memory and not saved. Scores are softmax outputs, not calibrated probabilities.

## Project structure

```
handwritten-digit-recogniser/
├── app/               # Streamlit interface (app.py)
├── src/               # Reusable code and scripts
│   ├── data.py            # loading, scaling, fixed train/validation split
│   ├── models.py          # dense baseline and CNN definitions
│   ├── preprocessing.py   # upload preprocessing contract (no model/UI code)
│   ├── inference.py       # model loading and prediction (no UI code)
│   ├── app_logic.py       # combines preprocessing + prediction for the app
│   └── ...                # training, evaluation and investigation scripts
├── tests/             # automated tests (preprocessing and app logic)
├── models/            # saved models (digit_cnn.keras is the app model)
├── results/           # run records, plots, reports, screenshots
├── notebooks/         # unused (see notebooks/README.md)
├── .streamlit/        # Streamlit config (5 MB upload limit)
├── requirements.txt
└── README.md
```

**Why this layout:** training, evaluation, preprocessing and the interface are separate, so nothing depends on notebook execution order and the preprocessing can be tested without the UI or the model.

## Data provenance

- **Dataset:** MNIST (LeCun, Cortes and Burges), loaded through `keras.datasets.mnist`.
- **Size:** 60,000 training and 10,000 test images, 28x28 grayscale, labels 0-9.
- **Split:** 10% of the training portion (6,000 images) for validation, stratified, `random_state=42`, defined once in `src/data.py` and shared by every run.
- **Test set:** used only after the model configuration was frozen (Week 7); never for model selection or tuning.
- **Preprocessing for training:** pixels divided by 255 (float32 in [0, 1]); a channel axis is added for the CNN.

## Model architecture

| # | Layer | Output shape | Parameters |
|---|---|---|---|
| 1 | Input | (28, 28, 1) | 0 |
| 2 | Training-only augmentation: RandomRotation, RandomTranslation, RandomZoom | (28, 28, 1) | 0 |
| 3 | Conv2D 32 filters, 3x3, ReLU | (26, 26, 32) | 320 |
| 4 | MaxPooling2D 2x2 | (13, 13, 32) | 0 |
| 5 | Conv2D 64 filters, 3x3, ReLU | (11, 11, 64) | 18,496 |
| 6 | MaxPooling2D 2x2 | (5, 5, 64) | 0 |
| 7 | Flatten | 1,600 | 0 |
| 8 | Dense 128, ReLU | 128 | 204,928 |
| 9 | Dense 10, softmax | 10 | 1,290 |
| | **Total** | | **225,034** |

Design rationale: two convolution + pooling blocks extract local features with shared weights, max-pooling reduces computation and gives some shift tolerance, and a small dense head classifies. Removing pooling made the model 21x larger and about 4.7x slower to train with no accuracy benefit (Week 6).

## Experimental configuration

| Setting | Value |
|---|---|
| Optimiser | Adam, learning rate 0.001 |
| Loss | Sparse categorical cross-entropy |
| Batch size / epochs | 128 / 15 |
| Model selection rule | Lowest validation loss (checkpoint) |
| Seeds | 42 (main), plus 1, 2, 3 for stability |
| Augmentation (adopted) | Rotation about ±14°, translation up to 10%, zoom ±10%, black fill, training only |

Every run saves `results/<run_id>/run_record.json` with its seed, split, preprocessing, architecture, parameters, optimiser settings, epoch-wise metrics, training time and model path. Seeds are fixed, but exact reproducibility can still vary with hardware and library versions.

## Commands

All commands run from the project root.

| Task | Command |
|---|---|
| Explore the data | `python -m src.explore_data` |
| Train dense baseline | `python -m src.train_dense --run-id dense_baseline` |
| Train CNN | `python -m src.train_cnn --run-id cnn_baseline` |
| Train augmented CNN | `python -m src.train_cnn --run-id cnn_aug_seed_42 --augment` |
| Compare models | `python -m src.compare_models --runs dense_baseline cnn_baseline` |
| Freeze a model | `python -m src.freeze_model --run-id cnn_baseline` |
| Evaluate on the test set (once) | `python -m src.evaluate` |
| Export the model for the app | `python -m src.export_app_model --run-id cnn_aug_seed_42` |
| Run the app | `python -m streamlit run app/app.py` |
| Run the tests | `python -m pytest -v` |
| Generate manual test images | `python -m src.make_test_images` |

## Preprocessing contract (uploads)

1. Decode the bytes, enforce size limits, fix EXIF orientation.
2. Composite transparency onto white; convert to grayscale.
3. Detect polarity from the image border and convert to a light digit on a dark background (the model was trained on white-on-black).
4. Subtract the background level; reject blank or low-contrast images.
5. Threshold to find strokes; reject images with almost no ink; warn on several digits or edge contact.
6. Crop to the strokes and resize to fit a 20x20 box, keeping the aspect ratio (squashing a narrow digit would change the prediction).
7. Paste into a 28x28 canvas and centre by centre of mass, as MNIST itself was prepared.
8. Output float32 in [0, 1] with shape (1, 28, 28, 1).

## Testing

- `tests/test_preprocessing.py`: 24 tests covering the output contract, repeatability, polarity, margins and sizes, aspect ratio, JPEG and transparent PNG, blank and near-blank images, multiple digits, edge contact, and corrupt, truncated, oversized and tiny files.
- `tests/test_app_logic.py`: 17 tests running the real saved model on rendered variants of validation digits (100% agreement with direct 28x28 predictions across five variants, 40 validation digits per variant), negative inputs and a non-digit image.
- Manual app test log: `results/week8/app_test_log.md`.
- Pipeline consistency check (validation images rendered as uploads): `results/week8/pipeline_alignment.json`.

Run the automated tests with `python -m pytest -v`. The first run downloads MNIST into the Keras cache.

## Limitations

- Trained on MNIST only: clean, centred digits. In the single recorded phone-photo trial, the app predicted 5 without a warning; this is not a real-world robustness benchmark.
- One digit per image. Several digits produce a warning, not a correct reading.
- The app cannot confirm that an image contains a digit; the recorded stripes test returned a prediction of 8 with a warning.
- Softmax scores are model outputs, not calibrated probabilities. Confident mistakes exist (Weeks 5 and 7).
- Polarity detection assumes the border is mostly background.
- Robustness tests were exploratory and are not a real-world benchmark.
- Error categories came from a single annotator's visual judgement (first pass drafted with AI assistance, then reviewed).
- Timings were measured on one CPU.

## Improvement opportunities

1. **Better preprocessing for real photos** (uneven-lighting correction, adaptive thresholding): benefit is robustness to phone photos; cost is more code and tuning; risk is that aggressive cleanup removes real strokes.
2. **Training on more varied data** (for example handwriting sets other than MNIST): benefit is wider style coverage; cost is new data handling and relabelling checks; risk is label noise.
3. **Out-of-distribution rejection** (a "not a digit" class or a learned rejection score): benefit is rejecting doodles and stripes; cost is extra training data; risk is rejecting valid but unusual digits.
4. **Improve digit-9 discrimination**: digit 9 had the lowest baseline test recall (97.92%) and was most often confused with 4 (8 cases). Cost: targeted data and validation experiments. Risk: improving recall may reduce precision for visually similar digits.

## Attribution

- **Implemented by me:** the project code in `src/`, `app/` and `tests/`, the experiments, the analysis and the written reports.
- **Supplied by libraries:** TensorFlow/Keras (layers, optimiser, training), scikit-learn (split, metrics), SciPy and Pillow (image operations), Streamlit (interface), pytest.
- **References:** TensorFlow CNN tutorial (https://www.tensorflow.org/tutorials/images/cnn), MNIST (https://www.tensorflow.org/datasets/catalog/mnist), Keras (https://keras.io/), scikit-learn metrics (https://scikit-learn.org/stable/modules/model_evaluation.html), Streamlit (https://docs.streamlit.io/).
- **AI assistance:** AI was used to draft the first-pass test-error annotations, which I reviewed.

---
## Progress

- [x] **Week 5:** Dataset analysis and dense baseline
- [x] **Week 6:** CNN design, training and baseline comparison
- [x] **Week 7:** Test evaluation, error analysis and report
- [x] **Week 8:** Streamlit app, tests, documentation and demo of the project

---
## Development log (Weeks 5-8)

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
- Training class counts ranged from 5,421 for digit 5 to 6,742 for digit 1, a difference of 1,321 across 60,000 images.
- In the later test-error review, 35 of 97 errors were annotated as unusual shapes and 31 as ambiguous strokes; these are subjective categories, not population-wide measurements.

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
| dense_lr_0.0001 | 0.0001 | 15 | 0.1663 | 0.9537 | 19.2 |
| dense_baseline | 0.001 | 9 | 0.0989 | 0.9737 | 20.4 |
| dense_lr_0.01 | 0.01 | 2 | 0.1195 | 0.9649 | 19.5 |

**Learning curves:** `results/dense_baseline/learning_curves.png`

**Interpretation:** The 0.0001 learning rate improved more slowly and reached 95.37% validation accuracy only at epoch 15, with validation loss still decreasing. The 0.01 run reached its lowest validation loss at epoch 2, then its validation loss rose substantially; the 0.001 baseline peaked at epoch 9 and also showed later overfitting. Each setting was run once, so these curves are descriptive rather than evidence of seed-independent effects.

**Caveat:** each configuration was run once with one seed, so small differences may be noise.

### Investigations

**Flattening (`results/week5_flattening/`)**
- *Pixel permutation:* baseline validation accuracy was 97.37% versus 97.13% with a fixed permutation. The dense model can learn with consistently reordered inputs; this small difference does not show that spatial layout is irrelevant.
- *Translation sensitivity:* accuracy fell from 97.37% with no shift to 36.22% at -4 pixels and 34.58% at +4 pixels. At ±1 pixel it remained 94.03%-95.05%, showing that larger shifts are much more disruptive.

**Softmax scores (`results/week5_softmax/`)**
- In the 0.99-1.00 confidence bin, mean confidence was 99.90% and accuracy was 99.65%; confidence tracked accuracy but was not perfectly calibrated.
- 54 incorrect validation predictions had a top score above 0.90.
- A blank image received a top score of 0.187; random noise had a mean top score of 0.882.
- **Conclusion:** a softmax score is a model score, not a calibrated probability or a guarantee of correctness.

### Limitations

- Results are validation-only and single-seed; the test set has not been used.
- The dense baseline treats pixels as an unordered vector, so it ignores local structure.
- Softmax scores can be high for non-digit inputs.

### Documents

- Technical note: `results/week5_technical_note.md`

---

## Week 6: CNN design and controlled training

### Objective

Design a compact CNN, train it under the same protocol as the Week 5 dense baseline, compare the two fairly on the same validation set, and verify that the saved model reloads independently of the training session.

### CNN architecture

| # | Layer | Output shape | Parameters |
|---|---|---|---|
| 1 | Input | (28, 28, 1) | 0 |
| 2 | Conv2D (32 filters, 3x3, ReLU) | (26, 26, 32) | 320 |
| 3 | MaxPooling2D (2x2) | (13, 13, 32) | 0 |
| 4 | Conv2D (64 filters, 3x3, ReLU) | (11, 11, 64) | 18,496 |
| 5 | MaxPooling2D (2x2) | (5, 5, 64) | 0 |
| 6 | Flatten | (1600) | 0 |
| 7 | Dense (128, ReLU) | (128) | 204,928 |
| 8 | Dense (10, softmax) | (10) | 1,290 |
| | **Total** | | **225,034** |

Output sizes use `floor((N + 2P - K) / S) + 1` with P = 0 and S = 1; convolution parameters are `(kh x kw x in_channels + 1) x filters`. The values were calculated by hand and then checked against `model.summary()` (saved in `results/cnn_baseline/model_summary.txt`).

**Design rationale**
- **Input (28, 28, 1):** 28x28 pixels with one grayscale channel; the spatial layout is kept, unlike the dense model.
- **Two conv + pool blocks:** the first layer detects small local patterns and the second combines them; enough for MNIST while keeping the model compact.
- **Max pooling:** reduces computation and gives some tolerance to small shifts, at the cost of spatial detail.
- **Dense(128) head and 10-unit softmax output:** the same head width as the baseline, and one output per mutually exclusive digit class.

### Training protocol

Identical to Week 5 except for the architecture and the added channel axis:

| Setting | Value |
|---|---|
| Split | Same stratified 90/10 split, seed 42 (verified identical to Week 5) |
| Preprocessing | Pixels / 255, shape (28, 28, 1) |
| Optimiser | Adam, learning rate 0.001 |
| Batch size / epochs | 128 / 15 |
| Selection rule | Lowest validation loss |
| Seed | 42 (extra seeds 1, 2, 3 for stability) |

### Commands

| Task | Command |
|---|---|
| Check CNN setup (shapes, same split) | `python -m src.check_cnn_setup` |
| Train the CNN | `python -m src.train_cnn --run-id cnn_baseline` |
| Train the no-pooling ablation | `python -m src.train_cnn --run-id cnn_no_pool --no-pool` |
| Verify reload (two separate processes) | `python -m src.verify_reload --run-id cnn_baseline --save` then `python -m src.verify_reload --run-id cnn_baseline` |
| Compare models | `python -m src.compare_models --runs dense_baseline cnn_baseline cnn_no_pool` |
| Seed stability summary | `python -m src.summarise_seeds` |
| Translation test (dense vs CNN) | `python -m src.investigate_shift_cnn` |

### Results (validation set, measured, seed 42, single run)

| Metric | Dense baseline | CNN baseline | CNN (no pooling) |
|---|---|---|---|
| Parameters | 101,770 | 225,034 | 4,738,826 |
| Training time | 20.4 s | 188.7 s | 880.8 s |
| Best epoch | 9 | 5 | 3 |
| Validation loss | 0.0989 | 0.0486 | 0.0563 |
| Validation accuracy | 97.37% | 98.65% | 98.45% |
| Validation errors (of 6,000) | 158 | 81 | 93 |
| Latency, single image | 6.19 ms | 22.42 ms | 93.66 ms |
| Latency, batch (per image) | 0.0253 ms | 0.0755 ms | 0.1879 ms |

Plots: `results/cnn_baseline/learning_curves.png` and `results/week6_comparison/validation_curves_comparison.png`.

**Observations**
- **Accuracy:** the CNN made 77 fewer errors than the dense model (about 49% fewer), and its validation loss was about half.
- **Cost:** the CNN has 2.2x the parameters but took 9.3x longer to train and was 3.6x slower on a single image, so parameter count understates its computational cost.
- **Pooling ablation:** removing both pooling layers raised parameters about 21x (the flatten layer grew from 1,600 to 36,864 values), slowed training about 4.7x and gave slightly more errors (93 vs 81). The data supports "no measurable benefit for a large cost"; the 12-error gap alone is not evidence that pooling improves accuracy.
- **Early best epochs:** the CNN peaked at epoch 5 and the no-pool model at epoch 3 of 15, which suggests later epochs overfit (see the learning curves).

### Stability across seeds

Four seeds per model (42, 1, 2, 3); the split stays fixed and only initialisation and batch order change.

| Model | Mean val accuracy | Std | 
|---|---|---|
| Dense |0.9736|0.0007|
| CNN |0.9874|0.0009|

The four-seed ranges do not overlap: Dense was 97.25%-97.43%, while the CNN was 98.65%-98.83%. With only four seeds, these ranges and standard deviations are still crude estimates of run-to-run variation.

### Reload verification

The saved CNN was reloaded in a separate Python process. Predictions on a fixed 10-image validation sample matched the saved reference outputs, and the reloaded model's validation loss and accuracy matched the training-time record.

### Translation sensitivity (exploratory, validation set)

Horizontal shifts of -4 to +4 pixels with zero fill, defined in advance. These transformed images are not an independent real-world benchmark.

| Shift (px) | Dense accuracy | CNN accuracy |
|---|---|---|
| -4 |0.3621|0.7438|
| -2 |0.7400|0.9583|
| -1 |0.9403|0.9818|
| 0 |0.9736|0.9865|
| +1 |0.9505|0.9825|
| +2 |0.8289|0.9606|
| +4 |0.3458|0.6971|

Plot: `results/week6_shift/shift_sensitivity_dense_vs_cnn.png`. At ±4 pixels, dense accuracy fell to 36.22% and 34.58%, while CNN accuracy was 74.38% and 69.71%. At ±1 pixel, the CNN stayed near 98.2% versus 94.0%-95.1% for the dense model. The CNN is more tolerant of these shifts, but not invariant; it was not trained on shifted digits.

### Repository notes

- `models/cnn_no_pool_best.keras` (54 MB) and the `*_seed_*.keras` checkpoints are excluded by `.gitignore`; their `run_record.json` files are committed. Recreate any of them with the commands above.
- Large or reproducible model files are kept out of Git to keep the repository small.

### Limitations (Week 6)

- All comparisons use the validation set, which was also used to select checkpoints, so the numbers are slightly optimistic. The test set is untouched until Week 7.
- Only four seeds per model; small differences should not be over-interpreted.
- Latency and training times were measured on one CPU while other programs were running.
- The shift test is exploratory and uses a single transformation type.

### Documents

- Week 6 notes: `results/week6_notes.md`

---

## Week 7: Evaluation, error analysis and limitations

### Objective

Freeze the selected CNN, evaluate it once on the held-out MNIST test set, analyse per-class performance and misclassified images, check generalisation and robustness, and test one pre-registered improvement using validation data only.

### Evaluation protocol

- **Frozen model:** `models/cnn_baseline_best.keras` (seed 42, selected by lowest validation loss at epoch 5).
- **Freeze record:** `results/week7/frozen_config.json` stores the model SHA-256, settings, validation metrics and git commit. It was committed and tagged `model-frozen` before the test set was used.
- **Safeguards:** `src/evaluate.py` aborts if the model file no longer matches the frozen SHA-256 and refuses to overwrite existing test results without `--force`.
- **Test set use:** one evaluation of the frozen model. No architecture, hyperparameter or epoch decision was made using test data.

### Commands

| Task | Command |
|---|---|
| Freeze the model | `python -m src.freeze_model --run-id cnn_baseline` |
| Test evaluation (once) | `python -m src.evaluate` |
| Error analysis (galleries, scores) | `python -m src.error_analysis` |
| Summarise manual error categories | `python -m src.summarise_annotations` |
| Training-validation gap check | `python -m src.investigate_overfitting` |
| Robustness test (validation set) | `python -m src.robustness` |
| Preview augmentation | `python -m src.preview_augmentation` |
| Train augmented CNN (4 seeds) | `python -m src.train_cnn --run-id cnn_aug_seed_42 --augment` and `python -m src.train_cnn --run-id cnn_aug_seed_<n> --augment --seed <n>` |
| Compare baseline vs augmented | `python -m src.compare_improvement` |
| Freeze augmented model | `python -m src.freeze_model --run-id cnn_aug_seed_42 --out results/week7_improved` |
| Evaluate augmented model (once) | `python -m src.evaluate --frozen results/week7_improved/frozen_config.json --out results/week7_improved` |

### Final test results (frozen CNN)

| Metric | Value |
|---|---|
| Test images | 10,000 |
| Errors | 97 |
| Test accuracy | 99.03% (approximate 95% interval 98.84%-99.22%) |
| Test loss | 0.030765 |
| Validation accuracy (for comparison) | 98.65% |
| Validation loss (for comparison) | 0.048649 |

The interval reflects finite test-set size. It does not include seed variation or the effect of checkpoint selection on validation data.

**Per-class metrics**

| Digit | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 | 0.9928 | 0.9908 | 0.9918 | 980 |
| 1 | 0.9921 | 0.9956 | 0.9938 | 1135 |
| 2 | 0.9942 | 0.9884 | 0.9913 | 1032 |
| 3 | 0.9921 | 0.9921 | 0.9921 | 1010 |
| 4 | 0.9889 | 0.9980 | 0.9934 | 982 |
| 5 | 0.9866 | 0.9888 | 0.9877 | 892 |
| 6 | 0.9917 | 0.9937 | 0.9927 | 958 |
| 7 | 0.9798 | 0.9893 | 0.9845 | 1028 |
| 8 | 0.9907 | 0.9867 | 0.9887 | 974 |
| 9 | 0.9940 | 0.9792 | 0.9865 | 1009 |
| Macro average | 0.9903 | 0.9902 | 0.9903 | 10,000 |
| Weighted average | 0.9903 | 0.9903 | 0.9903 | 10,000 |

Digit 9 has the lowest recall (97.92%), making it the main class-level weakness. Digit 7 has the lowest precision (97.98%). These two measures can identify different problem directions.

**Confusion matrix:** [`results/week7/confusion_matrix.png`](results/week7/confusion_matrix.png), with true class on rows and predicted class on columns. Errors only: [`results/week7/confusion_matrix_errors_only.png`](results/week7/confusion_matrix_errors_only.png).

The three largest confusions are 9→4 (8), 2→7 (6), and 9→7 (6). They are not symmetric: 4→9 has 0, 7→2 has 4, and 7→9 has 1.

### Error analysis

- **Sample:** all test errors are listed in [`results/week7/errors.csv`](results/week7/errors.csv) with true label, predicted label and all ten scores. A seeded random sample and the most confident errors are in [`results/week7/error_gallery/`](results/week7/error_gallery/).
- **Scores:** 32 errors had a top score above 0.90 and 9 had a top score above 0.99. The true class was ranked 2nd in 80 cases. The first-pass annotations were drafted with AI assistance from the gallery and reviewed by a single annotator.
- **Manual grouping** (`results/week7/error_annotation.csv`, one primary category per error, subjective):

| Category | Count | Share |
|---|---:|---:|
| ambiguous_stroke | 31 | 32.0% |
| unusual_shape | 35 | 36.1% |
| slanted_or_rotated | 9 | 9.3% |
| thin_or_faint | 10 | 10.3% |
| broken_or_cut_off | 6 | 6.2% |
| label_questionable | 4 | 4.1% |
| other | 2 | 2.1% |

- **Observed:** unusual or highly atypical strokes appeared in 35 errors, including looped 2s and blob-like 1s/4s/6s. Ambiguous stroke boundaries appeared in 31 errors, especially 3/5 and 9/4 pairs. Thin, faint, broken and strongly slanted samples accounted for another 25 examples. Four examples were questionable MNIST labels.
- **Plausible but untested explanations:** unusual samples may be underrepresented relative to standard MNIST strokes, and stroke loss or translation may move a digit outside the learned shape pattern. These mechanisms are hypotheses, not established causes.

Confident mistakes confirm that softmax output is not a guarantee. Examples include 8→0 at 0.9996, 8→0 at 0.9994, 8→9 at 0.9890, and 5→6 at 0.9983.

### Generalisation gap

| Measure | Value |
|---|---|
| Frozen CNN, training partition (inference mode) | accuracy 99.385%, loss 0.021902 |
| Frozen CNN, validation partition | accuracy 98.65%, loss 0.048649 |
| Accuracy gap (train - validation) | 0.735 percentage points |
| Validation loss rise after best epoch | 0.005973 from epoch 5 to epoch 15 |
| Test accuracy vs validation accuracy | 99.03% vs 98.65% |

The gap is small at the selected checkpoint. Validation loss rises after epoch 5 while training loss continues to fall, providing modest evidence of later-epoch overfitting. The test result is better than the validation result and should not be interpreted as evidence that the model is guaranteed to generalize; the interval and held-out sampling are still finite.

### Robustness (exploratory, validation set)

Transformations and severities were defined before running. These transformed images are **not** an independent real-world benchmark.

| Transformation | Level | Accuracy | Change |
|---|---|---|---|
| Original | - | 98.65% | - |
| Rotation | ±15° | 97.17% / 96.75% | -1.48 / -1.90 points |
| Rotation | ±30° | 83.47% / 85.17% | -15.18 / -13.48 points |
| Scale | 0.6 / 1.4 | 63.68% / 89.80% | -34.97 / -8.85 points |
| Vertical shift | ±2 px | 92.15% / 92.72% | -6.50 / -5.93 points |
| Vertical shift | ±4 px | 49.58% / 48.87% | -49.07 / -49.78 points |
| Blur | sigma 0.5 / 1.5 | 98.55% / 92.75% | -0.10 / -5.90 points |
| Noise | std 0.1 / 0.4 | 98.40% / 85.72% | -0.25 / -12.93 points |
| Contrast | 0.6 / 0.15 | 98.65% / 97.12% | 0.00 / -1.53 points |
| Inversion | on | 27.32% | -71.33 points |

**Non-digit inputs:** blank -> predicted 1 (score 0.1455); all-white -> predicted 0 (score 0.7111); random noise -> mean top score 0.4153. The app must reject, confirm or route low-confidence/non-digit input rather than treating a predicted digit as valid. Softmax score alone cannot reliably detect out-of-distribution input.

Plots: [`results/week7_robustness/robustness_curves.png`](results/week7_robustness/robustness_curves.png), [`results/week7_robustness/transformation_examples.png`](results/week7_robustness/transformation_examples.png).

### Improvement experiment: data augmentation

**Pre-registered plan** (`results/week7_improvement/plan.md`, committed before training):

- Training-time augmentation: `RandomRotation(0.04)` (about ±14°), `RandomTranslation(0.1, 0.1)`, `RandomZoom(0.1)`, black fill.
- Parameter count unchanged at 225,034.
- Everything else identical to the baseline CNN, with seeds 42, 1, 2 and 3.
- **Decision rule (validation only):** adopt if mean validation loss is lower and mean validation accuracy is higher than baseline across four seeds.

**Validation results (measured)**

| | Baseline CNN | Augmented CNN |
|---|---:|---:|
| Mean validation accuracy | 98.74% | 99.02% |
| Accuracy range (4 seeds) | 98.65%-98.83% | 98.98%-99.10% |
| Mean validation loss | 0.0460 | 0.0337 |
| Validation errors per seed (of 6,000) | 70-81 | 54-61 |
| Best epochs | 5, 9, 5, 7 | 12, 12, 15, 15 |
| Mean training time | 186 s | 256 s |

**Decision:** both conditions held, so augmentation was **adopted** under the rule. Accuracy ranges do not overlap across seeds, and training took about 38% longer. The gain is about 17 validation images of 6,000 (+0.28 percentage points). Both augmented runs reaching epoch 15 means the true gain may have been affected by the fixed training budget.

**Candidate chosen in advance:** the seed-42 augmented model, frozen as `model-aug-frozen` before its single test evaluation.

**Test result for the augmented model (evaluated once, reported next to the original):**

| | Frozen CNN (baseline) | Augmented CNN |
|---|---:|---:|
| Test accuracy | 99.03% | 99.19% |
| Approximate 95% interval | 98.84%-99.22% | 99.01%-99.37% |
| Test errors | 97 | 81 |
| Test loss | 0.030765 | 0.026569 |

The approximate intervals overlap, so the test-set improvement of 16 correct images should not be described as statistically distinguished with this procedure. The original frozen CNN result is preserved rather than replaced.

**Robustness comparison (secondary, exploratory):** augmentation improved the validation-set vertical shift from 49.58%/48.87% to 95.53%/94.42% at ±4 px, and improved scale, blur and normal noise. The ±30° rotation result improved from roughly 83%-85% to 92%-93%. It did not improve every perturbation: strong noise (std 0.4) fell from 85.72% to 46.92%, and inversion fell from 27.32% to 21.68%; these transformations were outside the training augmentation.

### Limitations (Week 7)

- Everything is MNIST-only; performance on phone photos, coloured backgrounds or other writing styles is not measured.
- Error categories are a single annotator's visual judgement, partly drafted with AI assistance.
- Improvement proposals were informed by test errors. Diagnosis does not mean test data was used for model tuning, but test observations informed which candidate to analyse.
- The augmented model's test evaluation is a second use of the test set for one pre-registered candidate.
- Four seeds per model provide limited evidence of sampling variability.
- Robustness results are exploratory, use one severity set and are not a real-world benchmark.
- `src/freeze_model.py` and `src/evaluate.py` gained an `--out` option after the original freeze; default behaviour is unchanged.
- Timings come from one CPU.

### Prioritised improvements

1. **Preprocessing alignment for app uploads**: centring, scaling and polarity handling address scale, shift and inversion failures. Cost: extra preprocessing code and UI testing. Failure mode: careless cropping or automatic inversion can change otherwise valid predictions.
2. **Retain data augmentation**: the candidate improved validation and test accuracy and vertical/scale robustness. Cost: about 38% longer mean training time. Failure mode: it does not correct ambiguous digits or questionable labels and can lose accuracy at extreme untrained perturbations.
3. **Add out-of-distribution and class-9 checks**: reject blank/noisy inputs and inspect hard 9→4/9→7 examples. Cost: requires calibration/OOD data and additional validation. Failure mode: strict thresholds may reject valid handwriting, while weak thresholds can leave confident false predictions.

### Documents

- Technical evaluation report: `results/week7_technical_note.md`
- Part B notes: `results/week7_partB_notes.md`
- Freeze records: `results/week7/frozen_config.json`, `results/week7_improved/frozen_config.json`
- Git tags: `model-frozen`, `model-aug-frozen`, `week7-partA`, `week7-partB`


---

## Week 8: Application, tests and delivery

### What was built

- `src/preprocessing.py`: the preprocessing contract above, with no model or UI code.
- `src/inference.py`: model loading and prediction. The app loads the model once with Streamlit caching.
- `src/app_logic.py`: combines preprocessing and prediction and returns a status (`ok`, `warning`, `rejected`) without raising on bad user input.
- `app/app.py`: the Streamlit interface (upload, ink-colour option, processed input, model's top prediction, score chart, warnings and errors, limitations sidebar).
- `src/export_app_model.py`: copies the chosen model to `models/digit_cnn.keras` and records its hash and source run in `models/digit_cnn_info.json`.

### Results

| Check | Result |
|---|---|
| Automated tests | 41 passed |
| Pipeline consistency (2,000 rendered validation images) | direct 28x28 99.4%, native 99.2%, drawn 10x 99.1%, small off-centre 99.2% |
| Manual app tests (14 inputs) | 5 of 14 met the logged expectations; see `results/week8/app_test_log.md` |
| Phone photo | Predicted 5 without a warning (one trial) |

### Observations

- Selecting the wrong ink colour caused the image to be rejected as blank or too low-contrast. The stripes input was classified as 8 with a warning.
- The recorded manual UI tests also found unhandled errors/crashes for corrupt or fake images, oversized uploads, and unsupported text files.

### Documents

- Test log: `results/week8/app_test_log.md`
- Screenshots: `results/screenshots/`
- Demo notes: `results/week8/demo_plan.md`


### References

- TensorFlow CNN tutorial: https://www.tensorflow.org/tutorials/images/cnn
- MNIST (TFDS catalogue): https://www.tensorflow.org/datasets/catalog/mnist
- Keras: https://keras.io/
- scikit-learn metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
