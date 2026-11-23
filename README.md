# Handwritten Digit Recogniser

A CNN-based handwritten digit classifier trained on MNIST, with a Streamlit app for inference.

## Progress

- [x] **Week 5:** Dataset analysis and dense baseline
- [x] **Week 6:** CNN design, training and baseline comparison
- [x] **Week 7:** Test evaluation, error analysis and report
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

*(State whether the two ranges overlap. With four seeds the standard deviation is crude, so report the range too.)*

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

Plot: `results/week6_shift/shift_sensitivity_dense_vs_cnn.png`. *(Describe how each model degrades. Pooling gives some tolerance to small shifts, not full invariance, and the CNN was never trained on shifted digits.)*

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

### Next step

Week 8: build the Streamlit upload app, implement and test the preprocessing contract (including blank, inverted and corrupt inputs), complete this README and prepare the five-minute demonstration.


### References

- TensorFlow CNN tutorial: https://www.tensorflow.org/tutorials/images/cnn
- MNIST (TFDS catalogue): https://www.tensorflow.org/datasets/catalog/mnist
- Keras: https://keras.io/
- scikit-learn metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
