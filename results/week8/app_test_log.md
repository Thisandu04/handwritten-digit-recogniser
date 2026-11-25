# App test log

| # | Input | Expected | Observed (what I actually saw) | Pass? |
|---|---|---|---|---|
| 01 | 01_clear_digit_dark_on_light.png | Prediction 7, no warning | Predicted 7, no warnings | pass |
| 02 | 02_small_off_centre.png | Prediction 3 | Predicted 3, no warnings| pass |
| 03 | 03_white_on_black.png | Prediction 2, no inversion note |Predicted 2, no warnings | pass |
| 04 | 04_blank_white.png | Error, no prediction | Displayed an error message| no |
| 05 | 05_faint_noise.png | Error, no prediction | Displayed an error message| no |
| 06 | 06_two_digits.png | Result plus warning | Predicted 7 as with top score with two warnings| no |
| 07 | 07_non_digit_stripes.png | No crash; describe result | Predicted 8 with top score with one warning| no |
| 08 | 08_corrupt_truncated.png | Error: could not be read |Crashed with an error message | no |
| 09 | 09_fake_image.png | Error: could not be read |crashed with an error message | no |
| 10 | 10_too_many_pixels.png | Error: too many pixels |Displayed an error message as too many pixels | no |
| 11 | 11_oversized_noise.png | Blocked by 5 MB limit | crashed, because the oversize| no, crashed|
| 12 | 12_unsupported.txt | Upload control refuses |Crasged, because unsupported format |no, crashed |
| 13 | my Paint digit | A sensible prediction |Predicted 5 without warnings |pass |
| 14 | my phone photo | May fail; describe |Predicted 5 without warnings | pass |

## Extra checks
- Wrong Ink colour setting on file 01: Displayed an error message "The image looks blank or has too little contrast to contain a digit."
- Two uploads in a row: Can't upload two uploads in a row, app takes only one upload at a time

## Notes and limitations I found
- Only JPG,PNG file formats are supported
- The file size limit is 5MB