# Revision round 1 — analysis plan for R1-1 (representation vs model family)

Written before any R1-1 model was trained. Settings fixed in §0 of
`prereg_R1-3_R1-4.md` apply unchanged.

## Question

Does the image branch's advantage come from the map representation, or from the model
family? Each comparison below holds the model family fixed and changes only the input
representation.

## CNN arm — original map vs sector-averaged map

**Manipulated variable.** Each thickness map is replaced, after cropping and before the
transform, by its sector-averaged version (`experiments/revision_r1/sector_maps.py`):

1. colour is inverted to value with the ramp read from the RNFL panel's own colorbar
   (350/175/0 µm). The GCA panel has no colorbar and uses the same ramp, so its decoded
   values are proportional to thickness with an unknown scale; the scale cancels because
   the map is re-rendered with the ramp it was decoded with;
2. the centre is detected. RNFL: the optic disc, as the centroid of the grey disc overlay
   (`experiments/sector_mask/`); no failures on this cohort. GCA: the fovea, by fitting the
   two annulus ellipses of known size to the drawn lines (`annulus_fit.fit_annulus`: line
   pixels = more than 30 RGB units from their 5 x 5 median, coarse vote over +-2.5 mm,
   refined at 0.5 px). Maps whose ellipses run along a drawn line at fewer than 0.35 of their
   points are flagged (16 of 280) and adjudicated one by one, on images kept in the
   repository's working directory, into three classes (`cache/fit_adjudication.json`):
   - **(a) bad scan** (6): the fovea lies at or beyond the crop edge, or the map is distorted;
   - **(b) bad fit** (8): fovea and ring are visible but the fit misses the ring. These are
     refitted (`annulus_fit.refit_annulus_v2`): start at the fovea - the darkest point of
     the smoothed luminance map, or the centre of the visible inner ring, searched only
     where the 264 unflagged fits put the fovea (their 0.5-99.5 percentile box widened by
     0.03 of the crop) - with the scale taken from the crop's longer side (crops cut along
     one side keep the true scale on the other); the inner ellipse is fitted first, then the
     outer one's two semi-axes independently (0.8-1.6 of nominal), each maximising the
     absolute contrast between the line and its two sides (the ring can be darker or, on thin
     advanced maps, lighter than the tissue). A refit is accepted when it lies on the drawn
     ring (4 accepted, 4 rejected);
   - **normal** (2): low line contrast on a thin map, but the first fit lies on the ring.
   A GCA map in class (a), or in class (b) with a rejected refit, has no usable annulus. In
   the sector-averaged condition it keeps only no-data black as original pixels; the frame
   band takes the fixed band colour of step 4, every other pixel takes the map-wide mean of
   its thickness pixels and no ring is drawn. This gives that eye less GCA information than a sector map, i.e. it works against
   the sector-averaged condition. Classification, refits and acceptance were fixed on
   2026-10-13 before any R1-1 result existed;
3. the whole map is divided into regions around that centre, every pixel in exactly one:
   RNFL — 12 clock-hour wedges of 30°; GCA — the Cirrus GCA geometry, i.e. the elliptical
   annulus (inner 1.2 × 1.0 mm, outer 4.8 × 4.0 mm) cut into its 6 sectors of 60° in the
   elliptical angle, plus the centre inside the annulus and everything outside it as two
   further regions (8 in all), so that the GCA map carries at least the information of the
   printed GCA sectors;
4. non-thickness pixels are classified explicitly and never enter a mean
   (`sector_maps.overlay_masks`). The panel frame band (RNFL 9/271, GCA 4/343 of the crop
   width; also holds the colorbar sliver) is painted one fixed colour per map type in the
   sector-averaged map, the channel-wise median band colour of all original crops (GCA RGB
   86, 133, 195 over 280 maps; RNFL 53, 169, 224 over 277): left original, the tissue
   pixels inside it showed through at the map edge. Kept as the original pixel in both
   conditions: the optic-disc overlay and cup (desaturated: saturation < 0.12, 40 < max < 250; RNFL maps only - on GCA
   maps the desaturated pixels are mostly light-grey stretches of the drawn ring, 91% of them
   on the rings, and fall in the ring band or the off-ramp class), data gaps (max < 40; on
   GCA maps black inside the inner ellipse is the foveal pit and counts as thickness), and
   the GCA annulus lines (a band of +-1.5 px around the fitted ellipses). The ring pixels are
   blends of the line and the tissue under it (their colour tracks the adjacent thickness,
   within-map r = +0.93), so in the sector-averaged map the band is painted one fixed colour,
   the median ring colour of 80 maps (RGB 101, 113, 155). Left out of the mean but filled
   with the region's value: a one-pixel fringe
   around disc, gaps and lines, and any other pixel further than 42 RGB units from the ramp.
   Rationale, measured on 40 eyes before the rule existed: the ramp ends in black, so every
   data-gap pixel decoded as 0 um, and part of the drawn lines matched the ramp;
5. every remaining pixel of a region is filled with the region mean, re-rendered with the
   ramp.

Images are not mirrored, as submitted (original and sector-averaged alike), the frame of the
primary analysis. Everything else in training is unchanged.

**Maps without a usable annulus.** Every eye stays in training and in evaluation in both
conditions, so the sample is that of the main analysis and the original condition is the
submitted model. A **sensitivity analysis** repeats the comparisons without the eyes whose
GCA map is a map-mean map (class a and rejected refits), and is reported next to the main
result; if the two differ, both are reported. Their number and MD distribution are reported.

**Runs.** Inception-ResNet-v2 first. After it, Inception-v3, VGG16, Xception and
DenseNet121.

**Comparisons** (per-eye RMSE and MAE; paired Wilcoxon and patient-clustered bootstrap
interval, as in §0):

1. **Primary for R1-1: Inception-ResNet-v2, sector-averaged vs original, OOF per-eye
   RMSE.** Original = the submitted run (the current code reproduces it exactly).
2. Other backbones, sector vs original. For Inception-v3 the original is the submitted run.
   For VGG16, Xception and DenseNet121 the original is the unmirrored rerun with the current
   code queued under R1-3 (the submitted runs are not reproducible; see the R1-3 decision
   log).
3. Five-backbone ensemble, sector vs original.
4. Fusion with the summary branch, using the sector-averaged image branch: fusion vs summary
   and fusion vs the submitted fusion, with the weight conventions of §0.
5. Held-out versions of 1–4.

Equivalence is not inferred from a non-significant difference; differences are reported
with their intervals.

## XGBoost arm — printed parameters vs the same parameters measured on the map

**Manipulated variable.** The 25 input features (the 26 printed summary parameters minus
vertical cup-to-disc ratio, which a thickness map does not carry):

* printed: the device's printed values;
* map-derived: the same 25 quantities measured on the decoded maps with the Cirrus geometry
  validated in `experiments/sector_mask/` — RNFL: 3.46 mm calculation circle, 0.20 mm band,
  centred on the detected disc → 12 clock hours, 4 quadrants, average; GCA: elliptical
  annulus (inner 1.2 × 1.0 mm, outer 4.8 × 4.0 mm) centred on the detected fovea → 6
  sectors, average, and minimum (lowest of 360 one-degree spoke means). Left-eye maps are
  mirrored before measurement so that every feature is in the right-eye frame. GCA values
  stay in decoded units; tree models are invariant to the unknown scale.

XGBoost settings, folds and per-location models as in §0.

**Guards.** Through the same code path: printed 26 must reproduce the frozen 8.6603 /
6.3975 dB (OOF pooled RMSE / MAE), and printed 25 the frozen value of the
"no vertical C/D" row of `forest_rows.json`.

**Primary map-derived set: map23** — the 20 region means of the sector-averaged map (12 RNFL
wedges + 8 GCA regions) plus three disc measures read from the map's disc overlay: disc area
and cup area (mm^2; cup = the lighter grey inside the filled disc, mean RGB > 160) and
vertical cup-to-disc ratio. The sector-averaged map keeps the disc pixels, so without these
the CNN would see information the XGBoost input lacks. **Secondary: map26** — the Cirrus-
geometry 25 above plus the map's vertical C/D, i.e. the 26 printed quantities measured on the
map. Both use the thickness pixels of step 4 and the ellipses of the sector-map cache; GCA
features are replaced by the map-wide mean for maps without a usable annulus, matching their
image. For XGBoost, left-eye features are put in the right-eye frame of the printed
parameters and labels: regions are re-ordered as the mirror image (clock hour h <-> 12 - h;
GCA sectors swap nasal and temporal). The CNN input is not mirrored.

**Comparisons.** (i) **CNN on the sector-averaged map vs XGBoost on map23** — the same
information, different model family (primary for the model-family question); (ii) XGBoost
map23 vs printed 26 and map26 vs printed 26 — the same model family, map-derived vs printed;
(iii) the CNN comparisons above. OOF and held-out, per-eye RMSE and MAE, paired as above,
on all eyes, and as the sensitivity analysis without the map-mean eyes.

**Known difference between the two XGBoost inputs.** The printed GCA sector columns of
left eyes carry the frame defect described in `docs/coordinate_frame_convention.md`; the
map-derived features do not. Its measured effect on the summary branch was not
distinguishable from refitting noise.

## Reported as a 2 × 2 (+1) table

| Model \ input | printed parameters | sector values from the map | original map |
|---|---|---|---|
| XGBoost | submitted (26) | map23 (primary); map26 | — |
| CNN | — | sector-averaged map | submitted |

## Decision log

* 2026-10-13, authors. GCA regions changed from 6 whole-map wedges to the Cirrus annulus
  sectors plus centre and outside (8 regions); RNFL kept at 12 wedges. Samples regenerated.
* 2026-10-13, build of the sector-averaged maps (after the sample-QC rules): 557 maps (GCA
  280, RNFL 277); ramp of 271 entries (1.30 um per step), identical on 40 panels; centre
  detected on every map; thickness-pixel fraction median 0.90 (GCA) / 0.81 (RNFL). Three RNFL
  maps carry large data gaps (29-47% black); in one of them two wedges had fewer than 30
  thickness pixels and took the map mean. Round-trip error at most 0.65 um (half a ramp
  step). Pixels the ramp alone would have accepted but that belong to an overlay class, now
  kept out of the means: black 412,891, GCA contour 47,637, fringe 394,920 (all maps).
* 2026-10-13, guard: `--map_mode identity` on Inception-ResNet-v2, fold 0, three epochs,
  reproduces the submitted history exactly (every crop found in the cache).
* 2026-10-13, authors (second QC round). R1-1 stays unmirrored in both conditions, the frame
  of the primary analysis; the reviewer-raised concern that the sector-averaged map carries
  fewer laterality cues than the original is recorded as a limitation. XGBoost primary set
  map23 (adds the disc measures, since the disc pixels are kept), secondary map26 vs printed
  26. Ring painted one colour. GCA fit replaced by `annulus_fit.py` with the failure
  threshold above; failed maps handled as stated, eyes excluded from evaluation only.
* 2026-10-13, authors (third QC round, before any R1-1 result). Fixed-colour ring band and
  the review of the 21 maps with support 0.35-0.55 approved. Flagged maps split into bad
  scans (a), bad fits (b) and normal; (b) refitted as stated in step 2. Maps without a usable
  annulus are no longer kept original in the sector-averaged condition nor dropped from
  evaluation: they take the map mean (working against the sector condition), stay in
  training and evaluation, and their exclusion is a sensitivity analysis reported alongside.
  The grey rule is restricted to RNFL maps. This supersedes the failure handling of the
  previous entry.
* 2026-10-13, after the third QC round (before any R1-1 result). GCA maps: 266 first fits,
  4 accepted refits (#01, #07, #12, #14; #12 and #14 were first labelled normal and moved to
  class b on the labelled grid), 10 map-mean maps (6 class a, 4 rejected refits). Class a: 6
  eye-visits, all out-of-fold, MD median -19.5 dB (-24.1 to -2.6); all map-mean maps: 10
  eye-visits (9 out-of-fold, 1 held-out), MD median -17.5 dB (-24.1 to -1.9), cohort median
  -4.0 dB. No grey-rule pixel on any GCA map. Round-trip error at most 0.65 um. Identity guard
  reproduces the submitted history.
* 2026-10-13: of the items listed for review before launch (updated samples, labelled
  adjudication grid, QC table), the authors approved the four sample sheets qc_01-qc_04
  (fourth QC round, at the end of this log).
* 2026-10-13, authors. Primary map-derived XGBoost set map20, secondary map25. The CNN vs
  XGBoost comparison on the same 20 values is required.
* 2026-10-13, authors (sample QC). Non-thickness pixels kept as the original in both
  conditions; exclusion rules of step 4 introduced, after measuring that data gaps and part
  of the contour lines had entered the means. Left-eye XGBoost features re-ordered to the
  right-eye frame; on a left-eye sample the 12 RNFL wedge means correlate with the printed
  clock hours at r = +0.51 before and +0.96 after re-ordering. Samples regenerated.
* 2026-10-13, authors (fourth QC round, before any R1-1 result). Sheets qc_01-qc_04
  approved with one change: the frame band of the sector-averaged map - RNFL and GCA, the
  map-mean GCA maps included - is painted one fixed colour per map type (step 4) instead of
  being kept original, because tissue pixels inside it showed through (on qc_03 the superior
  RNFL bundle as a yellow streak along the top edge, visible at the 161 px model input too).
  The band stays out of every mean; the original condition is unchanged. The four rejected
  refits stay map-mean. Rebuild of all 557 maps: the band is exactly the fixed colour on
  every map (one colour per map type over 1,599,316 GCA and 2,612,664 RNFL band pixels);
  every pixel outside the band and every per-map diagnostic is bit-identical to the
  previous build. The cache summary's GCA contour / fringe counts read 193,808 / 265,314
  instead of 195,163 / 265,718: the previous summary had not been recomputed after the
  adjudicated maps were re-rendered (`patch_adjudicated.py`), and the new values are the
  sums of the unchanged per-map diagnostics. Identity guard (Inception-ResNet-v2, fold 0,
  three epochs) reproduces the submitted history exactly. The authors allowed the launch
  without a further review once both checks passed: Inception-ResNet-v2 first, then the
  XGBoost arm.
