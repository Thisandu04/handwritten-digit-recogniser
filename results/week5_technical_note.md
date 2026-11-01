# Week 5 Technical Note: Dense Baseline for MNIST

## 1. Dataset
- Source, sizes, shapes, dtype, label set (from your explore_data output)
- Class balance: (your observation)
- Visual variation: (stroke width, slant, ambiguous pairs, from your sample image)

## 2. Preprocessing and split
- Scaling method and why: 
- Split: stratified 90/10 of the training portion, seed 42; test set untouched
- Resulting shapes and ranges:

## 3. Baseline model
- Architecture: Flatten -> Dense(128, ReLU) -> Dense(10, softmax)
- Parameter count and how it is computed: 
- Why ReLU in the hidden layer, why softmax at the output:

## 4. Training setup
- Loss: (what it measures, and why this one)
- Optimiser: Adam, lr 0.001, batch size 128, 15 epochs
- Epoch vs mini-batch vs iteration (use your numbers: how many iterations per epoch?)
- Selection rule: lowest validation loss

## 5. Results
- Baseline: best epoch, val loss, val accuracy (measured)
- Learning-rate experiment table (3 runs) and what the curves show
- Overfitting evidence, or lack of it:
- Limits of the comparison (single seed per config)

## 6. Flattening investigation
- Experiment A result and interpretation
- Experiment B result and interpretation
- What information flattening discards:

## 7. Softmax investigation
- Reliability table summary
- Confident mistakes and non-digit behaviour
- Conclusion: is a softmax score a calibrated probability / a guarantee?

## 8. Limitations and next step
- Evidence in this note is validation-only and single-seed
- Why a CNN is the natural next step