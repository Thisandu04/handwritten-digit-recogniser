# Handwritten Digit Recogniser

A CNN-based handwritten digit classifier trained on MNIST, with a Streamlit app for inference.

## Progress

- [x] **Week 5:** Dataset analysis and dense baseline
- [x] **Week 6:** CNN design, training and baseline comparison
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

### References

- TensorFlow CNN tutorial: https://www.tensorflow.org/tutorials/images/cnn
- MNIST (TFDS catalogue): https://www.tensorflow.org/datasets/catalog/mnist
- Keras: https://keras.io/
- scikit-learn metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
