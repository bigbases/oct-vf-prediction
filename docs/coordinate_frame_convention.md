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

## Visual field points

`p26` and `p35` are the two physiological blind-spot locations of the 24-2
pattern. They are dropped from the 54-point grid, leaving the 52 points every
model predicts (`image_preprocessing.py`, `BLIND_POINTS`). The manuscript's
per-point figures are drawn in the OD-normalised frame.
