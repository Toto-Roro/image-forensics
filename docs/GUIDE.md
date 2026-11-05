# Step-by-step guide

Contents
1. How the system works (overview)
2. How each algorithm works
3. Installing dependencies
4. Running the application
5. Testing the application
6. Adding a research dataset
7. Interpreting the results
8. Tuning parameters

---

## 1. How the system works

```
 upload ─► validate ─► preprocess ─► keypoints ─► self-match ─► RANSAC groups
                                     (ORB/SIFT)   (g2NN +        (+ dense NCC
                                                   spatial        check)
                                                   filter)
        ─► verdict (+ repetitive-pattern guard) ─► localization ─► visuals / report
```

A copy-move forgery duplicates content inside one image. Duplicated content produces
nearly identical local descriptors in two places, related by a single geometric
transform. The system finds such descriptor pairs, keeps only those that agree on one
transform, and then marks the regions that the transform maps onto each other.

Code entry point: `modules/pipeline.py::run_analysis(image_bgr, config)`.

---

## 2. How each algorithm works

### 2.1 Validation (`preprocessing.py`)
Checks extension (JPG/JPEG/PNG), empty file, size (≤ 20 MB), structural integrity
(Pillow `verify()` catches truncated/corrupt files), decodability (OpenCV) and
dimensions (64 px – 12 000 px). Any failure raises `ImageValidationError` with a
readable message, which the app shows instead of crashing.

### 2.2 Preprocessing
| Step | Why |
|---|---|
| Resize to ≤ 1024 px (aspect preserved, never upscaled) | bounded runtime; masks are scaled back to full size |
| BGR → RGB (display) and → grayscale (analysis) | descriptors work on intensity |
| Gaussian blur 3×3 (or bilateral / none) | removes sensor/JPEG noise that creates unstable keypoints |
| CLAHE (clip 2.0, 8×8 tiles) | local contrast boost → more keypoints in dark or flat areas |

### 2.3 Feature extraction (`feature_extraction.py`)
- **ORB** — FAST corner detector on an 8-level image pyramid, orientation by intensity
  centroid, rotated BRIEF descriptor (256 bits = 32 bytes). Compared with **Hamming
  distance** (number of differing bits). Fast; rotation invariant; partially scale
  invariant via the pyramid.
- **SIFT** — Difference-of-Gaussians blob detector across scales, dominant gradient
  orientation, 4×4 grid × 8-bin gradient histograms = 128-D descriptor. Compared with
  **Euclidean (L2) distance**. Slower, but more robust to rotation, scaling and
  compression — therefore the default.

The app's "Compare ORB and SIFT" panel shows keypoint count and time for both.

### 2.4 Self-matching (`feature_matching.py`)
Each descriptor is matched against **all descriptors of the same image** (k-NN,
k = 9). The trivial self-match (distance 0 to itself) is removed.

*Lowe's ratio test* accepts the nearest neighbour only if d₁/d₂ < T. With copy-move,
an object pasted twice has *two* genuine partners, so the plain test fails. The
**generalised 2NN test (g2NN, Amerini et al. 2011)** walks the sorted distances and
accepts neighbour *i* while dᵢ/dᵢ₊₁ < T, stopping at the first big jump.

Then:
- **spatial filter** — pairs closer than 4 % of the image diagonal are dropped (they
  come from the same structure or duplicate keypoints at one spot);
- **de-duplication** — (i, j) and (j, i) are the same pair;
- for ORB an absolute cap (Hamming ≤ 80) removes very weak matches.

Matchers: `BFMatcher(NORM_HAMMING)` for ORB, `BFMatcher(NORM_L2)` for SIFT, or
**FLANN** (KD-tree) for SIFT when there are more than 3 000 keypoints.

### 2.5 RANSAC geometric verification (`geometric_verification.py`)
A copy is one **similarity transform** (rotation θ, uniform scale s, translation t):

```
[x']   [ s·cosθ  -s·sinθ ] [x]   [tx]
[y'] = [ s·sinθ   s·cosθ ] [y] + [ty]
```

`cv2.estimateAffinePartial2D` with RANSAC repeatedly samples 2 matches, builds a
candidate transform, counts how many matches it maps to within `ransac_thresh` pixels
(inliers) and keeps the best candidate, then refits on the inliers.

- **Direction ambiguity:** a pair does not say which side is the original, so each
  pair is given to RANSAC in both orientations and re-oriented afterwards.
- **Multiple copies:** inliers are removed and RANSAC is run again, giving several
  *match groups*.
- **Validity checks:** at least `min_inliers` inliers, scale within 0.33–3, and the two
  regions must be clearly apart (no near-identity transforms).
- **Dense confirmation:** a group with only 4–7 inliers is accepted only if the image,
  warped by the group's transform, agrees with itself over a textured area. Agreement
  is measured with local **zero-mean normalised cross-correlation (NCC)** in 9×9
  windows (NCC > 0.8, on pixels with local std > 6). Real copies reach 40–80 %
  agreement; wrong transforms stay around 10 %.

### 2.6 Verdict and detection score (`forgery_detection.py`)
- **Potentially manipulated** — at least one valid group that does not look like
  natural repetition.
- **Inconclusive** — evidence is weak (a group with half the required inliers), or the
  duplication looks like a repeating texture.
- **No manipulation detected** — no geometrically consistent duplicate.

Repetitive-pattern guard (tiles, bricks, windows):
1. *self-overlap* — a texture shifted by one period overlaps itself, so the convex hulls
   of the two sides overlap (IoU > 0.25). A pasted object goes somewhere else.
2. *lattice* — the pure-translation shifts of the groups are integer combinations of
   one or two basis vectors (the tile size).
3. *too many* — more than 4 independent valid groups.

**Detection score** (heuristic, not a probability):
`score = 100 × min(1, verified matches / (3 × min_inliers))`.
It measures the amount of geometric evidence; it is not calibrated on any dataset.

### 2.7 Localization (`localization.py`)
Two levels, always labelled in the output:

1. **Keypoint-region estimate** — inliers of each side are clustered with single-linkage
   hierarchical clustering (distance 8 % of diagonal); clusters with < 3 points are
   dropped. Each cluster → filled convex hull + a disc per keypoint →
   **morphological closing** (fills gaps) → **dilation**.
2. **Transform-verified (pixel-level)** — the image is warped onto itself with the
   group's transform (Pan & Lyu style). Pixels whose 7×7 neighbourhood matches the
   warped copy (adaptive threshold learned at the verified keypoints) are candidates.
   The search is limited to a band around the keypoint region, cleaned with
   **opening** (removes speckle) and **closing**, then **connected-component analysis**
   keeps only components that contain verified keypoints; holes are filled.

Finally contours (`findContours`) and bounding boxes (`boundingRect`) are extracted.
Regions are labelled **A** (cyan) and **B** (red): the two sides of one duplicated pair.

### 2.8 Experimental splicing indicators (`splicing_detection.py`)
Per 32×32 block: noise level (MAD of the median-filter residual), error-level analysis
(difference after JPEG re-compression at q=90), edge sharpness (mean gradient on Canny
edges) and residual variance. Each map becomes a robust z-score (median/MAD); a block is
flagged when at least two features exceed |z| > 3.5. Flat and clipped blocks are
excluded. Output is a heatmap and a flagged-block overlay — indicators, not proof.

---

## 3. Installing dependencies

```bash
python3 --version                    # 3.10 or newer
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

All packages are pure pip wheels (no system libraries). Total download ≈ 150 MB.
If you prefer not to install anything locally, use Streamlit Community Cloud (see
README).

## 4. Running the application

```bash
streamlit run app.py
```

Open the URL shown (normally http://localhost:8501). For a quick demo pick
**Use a demo sample** and step through `01_…` to `08_…`:

| Sample | Expected result |
|---|---|
| 01_original_untouched | No manipulation detected |
| 02_copied_object_translate | Potentially manipulated, pair A1/B1 |
| 03_rotated_copy | Potentially manipulated, rotation shown in group table |
| 04_resized_copy | Potentially manipulated, scale shown in group table |
| 05_compressed_copy_q75 | Potentially manipulated (survives JPEG) |
| 06_repeated_pattern_tiles | Inconclusive – possible repetitive pattern |
| 07_natural_pattern_bricks | No manipulation detected |
| 08_corrupted_file | Clear error message, no crash |

Regenerate samples and the synthetic dataset any time with `python -m modules.synthetic`.

## 5. Testing the application

```bash
python -m pytest -q              # whole suite
python -m pytest tests/test_detection.py -v
```

| File | Covers |
|---|---|
| `test_preprocessing.py` | valid PNG/JPEG, wrong extension, empty, corrupt, truncated, fake, tiny, oversize; resize; grayscale/RGBA |
| `test_feature_matching.py` | ORB/SIFT descriptors, flat image, g2NN invariants, matches inside GT, RANSAC recovers rotation+scale |
| `test_detection.py` | 4 untouched photos (ORB, SIFT, High sensitivity), translate/rotate/scale/rotate+scale/JPEG copies, re-compressed q=60, tiles, bricks, score formula |
| `test_localization.py` | IoU with GT > 0.5, both sides found, keypoint-only labelling, empty mask on clean image, metrics edge cases, mask-name lookup, dataset evaluation, reports, splicing |

## 6. Adding a research dataset

1. Download the dataset from its official page (CoMoFoD, CASIA v2.0, DEFACTO).
2. Copy files into `dataset/original/`, `dataset/forged/`, `dataset/masks/`
   (see `dataset/README.md` for the per-dataset mapping and mask naming).
3. Run `python -m modules.evaluation --dataset dataset` or use the app's
   **Dataset evaluation** tab. Use `--limit 50` for a quick trial.
4. Results are written to `results/eval_dataset_<time>/`.

## 7. Interpreting the results

- **Verdict banner** — red = potentially manipulated, amber = inconclusive,
  green = nothing found. "Nothing found" does not mean authentic: it means no
  *copy-move* evidence; splicing or retouching are not covered by the main method.
- **Verified matches / groups** — the actual evidence. A group of 30+ verified matches
  with low mean error is strong; 4–7 matches confirmed only by the dense check is weaker.
- **Group table** — rotation and scale tell you *how* the region was copied.
  `dense_agreement_pct` shows how much of the neighbourhood really matches.
- **Mask & boxes** — A and B are the two sides of a duplicate; either may be the
  original. Check the "Localization method" line: pixel-level vs keypoint estimate.
- **Detection score** — amount of evidence, not probability.
- **Image-level vs pixel-level metrics** — image-level asks *is this image forged?*
  (accuracy, precision, recall, F1, confusion matrix). Pixel-level asks *are the right
  pixels marked?* (pixel precision, recall, F1, IoU), computed only on forged images
  with ground-truth masks.

## 8. Tuning parameters

| Want | Change |
|---|---|
| fewer false alarms | Sensitivity Low (ratio 0.5, ≥ 12 inliers) |
| catch small / weak copies | Sensitivity High (ratio 0.7, ≥ 6 inliers) |
| speed | ORB, lower working resolution, fewer keypoints |
| large images with small copies | raise working resolution to 1600 |
| copies placed very close to the source | lower "Min distance between copies" |
