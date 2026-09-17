# Laterality and coordinate frames

Two frames appear in this code, and the difference between them is the single
most common source of confusion when reading it.

**On-screen frame.** A position on the printed report, left to right as the page
is viewed. Clock hours `rnfl_h01` … `rnfl_h12` are read in this frame: the OCR
step (`ocr_rnfl_detail.py`) takes them from fixed page coordinates and does not
know which eye it is looking at, so `rnfl_h03` means "three o'clock on the page"
for both eyes.

**Anatomical (OD-normalised) frame.** Temporal, superior, nasal, inferior
relative to the eye. Quadrants `rnfl_q_t`, `rnfl_q_s`, `rnfl_q_n`, `rnfl_q_i`
are read in this frame: `ocr_rnfl_detail.py` reads the temporal quadrant from
x ≈ 0.040 for OD and from x ≈ 0.985 for OS, so both eyes come out of OCR with
`_t` already meaning temporal.

## What follows for the OS mirroring step

`make_rnfl_flip.py` mirrors OS rows into the OD frame:

* **clock hours are swapped** — `HOUR_SWAP` pairs h01↔h11, h02↔h10, h03↔h09,
  h04↔h08, h05↔h07, with h06 and h12 fixed on the vertical axis;
* **quadrants are not** — `QUAD_SWAP` is empty, because OCR already resolved
  them anatomically.

An earlier version also swapped the quadrants. That double application put nasal
values into `rnfl_q_t` for OS rows. It was found by a row-level identity check —
each quadrant should equal the mean of its own three clock hours — and corrected
upstream in `make_rnfl_flip.py` rather than downstream, so the corrected column
is what every later stage reads. After the correction both eyes satisfy
`q_t ≈ mean(h08, h09, h10)` and `q_n ≈ mean(h02, h03, h04)` to a median of
0.33 µm.

The correction moved the reported numbers by less than 0.01 dB in each direction
(out-of-fold XGB 8.653 → 8.660, out-of-fold late fusion 8.064 → 8.069; the fitted
fusion weights were unchanged). The values in `paper/results_frozen/` are the
post-correction ones.

## Ganglion cell analysis sectors

A second defect of the same kind was found later, in the ganglion cell analysis
columns. `apply_vf_neg1_to_zero.py` applies a temporal/nasal swap to four OS
superior and inferior sector columns that have already been mapped upstream, so
the swap is applied twice and the two eyes end up in opposite frames for those
four columns. It was found by a fellow-eye check: the correlation between a
sector and its counterpart in the other eye should not depend on laterality, and
it did (t = +5.02).

**This correction is not applied to the reported results.** The reason is that
its effect cannot be measured. Re-running the whole pipeline with the four
columns corrected moves the summary branch, but so does re-running it with the
columns untouched: the right-eye rows, whose features do not change by a single
value, still move by +0.134 dB, which is the size of the refitting perturbation
alone. The difference in differences — the left-eye shift minus the right-eye
shift, which is the only part attributable to the frame — is +0.029 dB with
Welch p = 0.819. The defect is real; its effect on the reported numbers is not
distinguishable from noise.

The numbers in `paper/results_frozen/`, and the numbers in the manuscript, are
therefore the uncorrected ones: out-of-fold XGB 8.660 dB RMSE, out-of-fold late
fusion 8.069, fusion weight w = 0.47. For the record, applying the correction and
re-running everything gives 8.831, 8.142 and w = 0.43; the image branch does not
move at all (out-of-fold CNN 8.540, five-backbone ensemble 8.115), which is
direct evidence that the two branches are independent. Reporting the uncorrected
values is the more conservative choice, because the correction widens the gap
this study reports between the image and summary representations rather than
narrowing it.

## Visual field points

`p26` and `p35` are the two physiological blind-spot locations of the 24-2
pattern. They are dropped from the 54-point grid, leaving the 52 points every
model predicts (`image_preprocessing.py`, `BLIND_POINTS`). The manuscript's
per-point figures are drawn in the OD-normalised frame.
