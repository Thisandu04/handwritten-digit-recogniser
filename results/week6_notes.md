# Week 6 Notes: CNN Design and Comparison

## 1. Architecture and rationale
- Layer table (output shapes + parameters) - hand-calculated, verified against model_summary.txt
- Why conv + ReLU + pool + dense head; why (28, 28, 1) input and 10 outputs

## 2. Training protocol
- Same split, seed, optimiser, batch size, epochs and selection rule as the dense baseline

## 3. Results (validation, measured)
| Run | Params | Train s | Best epoch | Val loss | Val acc | Val errors | Latency (ms) |
|---|---|---|---|---|---|---|---|
| dense_baseline |101,770|20.4|9|0.0989|0.9737|158|0.0253 |
| cnn_baseline |225,034 |188.7 |5 |0.0486 |0.9865 |81 |0.0755 |
| cnn_no_pool (optional) |4,738,826 |880.8 |3 |0.0563 |0.9845 |93 |0.1879 |

## 4. Observations from the plots (at least two, grounded in the figures)
1.
2.

## 5. Stability across seeds
- Mean, std, min-max for both models; do the ranges overlap?

## 6. Translation test
- Dense vs CNN at shifts of +-1, +-2, +-4; what it does and does not show

## 7. Trade-offs: accuracy vs complexity vs latency

All values are measured on the same 6,000 validation images (seed 42, single run per model). Ratios and percentages are calculated from the measured values.

### Comparison table

| Metric | Dense baseline | CNN baseline | CNN vs dense | CNN (no pooling) | No-pool vs CNN |
|---|---|---|---|---|---|
| Validation accuracy | 97.37% | 98.65% | +1.28 points | 98.45% | -0.20 points |
| Validation errors (of 6,000) | 158 | 81 | 77 fewer (about 49%) | 93 | 12 more |
| Validation loss | 0.0989 | 0.0486 | about 51% lower | 0.0563 | about 16% higher |
| Parameters | 101,770 | 225,034 | 2.2x | 4,738,826 | 21.1x |
| Training time | 20.4 s | 188.7 s | 9.3x | 880.8 s | 4.7x |
| Latency, single image | 6.19 ms | 22.42 ms | 3.6x | 93.66 ms | 4.2x |
| Latency, batch (per image) | 0.0253 ms | 0.0755 ms | 3.0x | 0.1879 ms | 2.5x |
| Best epoch | 9 | 5 | | 3 | |

### Accuracy

The CNN made 81 errors against the dense model's 158, so it roughly halved the number of mistakes. The accuracy change looks small (97.37% to 98.65%), but in terms of errors it is a large improvement. Validation loss also fell by about half, so the CNN assigns better scores to the correct class, not only more correct labels.

### Complexity

The CNN has about 2.2x as many parameters as the dense model but took about 9x longer to train (188.7 s vs 20.4 s). Parameter count therefore understates the CNN's computational cost. Convolution reuses each small kernel at every image position, which keeps the parameter count low, but every reuse is a computation.

### Latency

The CNN was about 3.6x slower on a single image (22.42 ms vs 6.19 ms) and about 3x slower in batches. Both are small in absolute terms for a one-image app, but the relative cost is clear. Timings were measured on my CPU only, and the single-image figures include Keras call overhead.

### Pooling ablation

Removing both max-pooling layers (nothing else changed) made the model larger, slower and no better.

- The flatten layer received 24 x 24 x 64 = 36,864 values instead of 5 x 5 x 64 = 1,600, so the dense layer grew enormously and parameters rose about 21x.
- Training was about 4.7x slower and single-image latency about 4.2x higher.
- Accuracy was slightly lower (93 vs 81 errors) and validation loss was higher.

In this experiment, keeping full spatial resolution gave no measurable benefit, while pooling cut the cost sharply. Pooling trades spatial detail for lower computation and some tolerance to small shifts.

### Overall

For MNIST the pooled CNN is the better trade-off. About 9x the training time and 3.6x the single-image latency buy roughly half the errors, and the absolute latency is still small for a one-image application. The no-pooling model costs far more and returns nothing measurable.

### Caveats

- **Single seed.** The 12-error gap between the pooled and no-pool CNN is within plausible random variation, so the data supports "removing pooling gave no benefit and cost a lot", not "pooling improves accuracy". The 77-error gap between dense and CNN is much larger; see Section 5 for the multi-seed ranges.
- **Validation data used for selection.** The best epoch was chosen on this same set, so these numbers are slightly optimistic. Final test results come in Week 7.
- **Early best epochs.** The CNN peaked at epoch 5 and the no-pool model at epoch 3 of 15, which suggests overfitting in later epochs (confirm in the learning curves). Recorded training times include these unnecessary later epochs.
- **Timing noise.** Training time and latency depend on other programs running during measurement.
- **Hardware.** All timings are CPU-only on a single machine.

## 8. Reload verification
- What was checked, and the result

## 9. Limitations
- Validation-only; 4 seeds; latency measured on one CPU; shifts are exploratory