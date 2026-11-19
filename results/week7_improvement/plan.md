# Improvement experiment plan (written before training)

Candidate: data augmentation (training only)
- RandomRotation(0.04)  -> about +-14 degrees
- RandomTranslation(0.1, 0.1) -> up to about 2.8 px
- RandomZoom(0.1) -> +-10%
- fill mode constant (black), so no reflected borders

Everything else identical to cnn_baseline: architecture, Adam lr 0.001, batch 128,
15 epochs, validation split, selection rule (lowest validation loss).

Runs: seeds 42, 1, 2, 3 (same four seeds as the baseline CNN).

Decision rule (validation only):
Adopt augmentation if, across the four seeds, BOTH
 - mean best validation loss is lower than the baseline's, AND
 - mean best validation accuracy is higher than the baseline's.
Otherwise keep the frozen baseline and report augmentation as a negative result.

If adopted: the seed-42 augmented model (chosen in advance, not the best seed) is
frozen and evaluated once on the test set, reported alongside the original result.
The original frozen CNN's test result is NOT replaced.

Secondary evidence (not part of the rule): robustness test on validation data.