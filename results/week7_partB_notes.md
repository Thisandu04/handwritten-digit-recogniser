# Week 7 Part B Notes

## 1. Error sample (reproducible)
- Number of test errors, seed for the random sample, location of errors.csv
- Score distribution: confident errors, true-class rank (2nd? top 3?)

## 2. Error groups (manual, single annotator)
- Category counts; which confusion pairs dominate
- For each main category: what I OBSERVED / what I SUPPOSE (kept separate)

## 3. Confusion cells inspected
- The three most frequent cells and what the images show

## 4. Training-validation gap
- Gap at the selected checkpoint, rise in validation loss after it
- Validation vs test; sampling variability; data quality (questionable labels)

## 5. Robustness (exploratory, validation set)
- Transformation plan, results table, weakest and strongest cases
- Non-digit inputs; what this means for the Week 8 app
- Limitation: not an independent real-world benchmark

## 6. Review questions
- Can a high-accuracy model still fail systematically on one digit? (Use your lowest-recall digit.)
- Why does a confident prediction not guarantee correctness? (Cite your confident errors.)
- What evidence would support a claim of improved robustness?