# Image Tampering / Forgery Detection and Localization Using Computer Vision
### Technical write-up for the mini-project viva

---

## 1. Problem statement

Digital images are easy to edit and are routinely used as evidence in news, insurance,
legal and academic contexts. One of the most common manipulations is **copy-move
forgery**: a region of an image is copied and pasted elsewhere in the *same* image to
duplicate an object (e.g. extra people in a crowd) or to hide one (pasting background
over it). Because the copied pixels come from the same photo, colour, noise and lighting
match naturally, so the edit is hard to see.

The task: given a single image and no reference, decide whether it shows evidence of
copy-move manipulation and **localize** the duplicated regions.

## 2. Objectives

1. Accept and validate JPG/PNG uploads; reject invalid or corrupt files gracefully.
2. Detect duplicated regions with classical CV — keypoints, descriptor matching,
   RANSAC — robust to translation, rotation, scaling and JPEG compression.
3. Avoid flagging natural repetition (tiles, bricks, windows) as definite forgery.
4. Localize both sides of the duplicated pair with a binary mask, contours and boxes.
5. Present results in a dashboard with downloadable HTML/PDF/JSON reports.
6. Evaluate with image-level and pixel-level metrics on datasets with ground truth.
7. Provide an experimental, clearly-labelled splicing-indicator module.

## 3. Literature / technical background

Copy-move detection methods fall into two families:

**Block-based methods** split the image into overlapping blocks, compute a feature per
block (DCT coefficients — Fridrich et al., 2003; PCA — Popescu & Farid, 2004; Zernike
moments — Ryu et al., 2010), sort the features lexicographically and look for similar
blocks at a consistent shift. Accurate for plain translations, but slow and weak against
rotation and scaling.

**Keypoint-based methods** detect local interest points and match their descriptors
within the image. Huang et al. (2008) used SIFT; **Amerini et al. (2011)** introduced
the generalised 2NN test and hierarchical clustering with geometric-transform estimation;
**Pan & Lyu (2010)** estimated the affine transform between matched regions with RANSAC
and obtained dense masks by correlating the image with its warped copy. ORB (Rublee et
al., 2011) offers a faster binary alternative to SIFT (Lowe, 2004).

Benchmarks: **CoMoFoD** (Tralic et al., 2013), **CASIA v2.0** (Dong et al., 2013),
**DEFACTO** (Mahfoudi et al., 2019). Christlein et al. (2012) compared 15 methods and
found keypoint methods efficient and robust to geometric changes, but weak on flat
regions — a trade-off this project inherits.

This project follows the keypoint family (Amerini + Pan & Lyu ideas), adds a dense NCC
confirmation step for weak groups and an explicit repetitive-pattern guard.

## 4. Methodology

1. **Preprocessing** — resize (≤ 1024 px, aspect preserved), grayscale, 3×3 Gaussian
   denoise, CLAHE.
2. **Feature extraction** — ORB (binary, Hamming) or SIFT (128-D float, L2).
3. **Self-matching** — k-NN of every descriptor in the same image; g2NN ratio test
   (T = 0.6); spatial filter (pairs closer than 4 % of the diagonal dropped);
   BFMatcher or FLANN.
4. **Geometric verification** — iterative RANSAC fitting of a 4-DoF similarity
   transform, both pair orientations, multiple groups; validity checks on inlier
   count, scale and shift; dense NCC confirmation for groups with 4–7 inliers.
5. **Decision** — verdict + repetitive-pattern guard (self-overlap, lattice, count) +
   heuristic detection score.
6. **Localization** — single-linkage clustering of inliers → hull + morphology
   (keypoint estimate) → transform-verified pixel map (warp, local difference,
   opening/closing, connected components seeded by keypoints) → contours, boxes.
7. **Evaluation** — accuracy/precision/recall/F1/confusion matrix (image level);
   pixel precision/recall/F1 and IoU (pixel level).

## 5. Algorithms (key equations)

**g2NN**: for sorted neighbour distances d₁ ≤ d₂ ≤ … ≤ d_k accept neighbour i while
dᵢ / dᵢ₊₁ < T.

**Similarity transform**: p' = s·R(θ)·p + t, 4 parameters, 2 point pairs determine it.

**RANSAC**: repeat N times — sample 2 pairs, fit transform, count inliers with
‖M(p) − q‖ < ε — keep the best; number of iterations for confidence c with inlier
ratio w and sample size 2: N = log(1 − c) / log(1 − w²).

**Local NCC** over window W:
NCC = Σ(I − μ_I)(J − μ_J) / √(Σ(I − μ_I)² · Σ(J − μ_J)²), with J the warped image.

**Detection score**: 100 · min(1, V / (3 · min_inliers)), V = verified matches.

**Metrics**: Precision = TP/(TP+FP); Recall = TP/(TP+FN);
F1 = 2PR/(P+R); IoU = TP/(TP+FP+FN). Zero denominators return 0 (both masks empty → 1).

## 6. System architecture

```
                     ┌──────────────── Streamlit app (app.py) ────────────────┐
                     │ upload · settings · results · summary · downloads · eval│
                     └───────────────┬─────────────────────────────────────────┘
                                     │ run_analysis()
  modules/  preprocessing ─► feature_extraction ─► feature_matching ─► geometric_verification
                                                                           │
            report_generator ◄─ visualization ◄─ localization ◄─ forgery_detection
            evaluation (CLI + tab)   synthetic (test data)   splicing_detection (experimental)
```

Each module has one responsibility; the UI contains no image-processing code. All
parameters live in `AnalysisConfig`, all paths are relative to the project root.

## 7. Results

Measured on the built-in synthetic set (5 untouched photos + 25 copy-move forgeries:
translation, rotation 20–45°, scaling ×0.8/×1.2–1.3, rotation + scaling, JPEG q=75),
default settings:

| Method | Accuracy | Precision | Recall | F1 | Pixel F1 | IoU (all forged) | IoU (detected) | Time |
|---|---|---|---|---|---|---|---|---|
| SIFT | 0.900 | 1.000 | 0.880 | 0.936 | 0.818 | 0.764 | 0.868 | 0.18 s |
| ORB  | 0.867 | 1.000 | 0.840 | 0.913 | 0.723 | 0.640 | 0.762 | 0.23 s |

Observations:
- No false positives on untouched photos; one untouched photo (`rocket`, with repeated
  tower structures) is honestly reported as *Inconclusive*.
- The tiled-texture test image is reported *Inconclusive – possible repetitive pattern*
  with both ORB and SIFT; the natural brick wall gives no detection.
- Missed cases are scaled copies in a low-contrast photo, where too few keypoints
  survive.
- SIFT localizes better than ORB (IoU 0.76 vs 0.64) because it finds more matches on
  rotated/scaled copies.

These numbers are from a small synthetic set. Results on CoMoFoD/CASIA must be produced
with `python -m modules.evaluation --dataset dataset` after downloading them; they will
be lower, especially for small or texture-less forgeries.

## 8. Limitations

- **Flat regions**: no keypoints → copy of sky or wall is invisible to the method.
- **Natural repetition** is genuinely duplicated content; the guard reduces false alarms
  but can also hide a real copy placed on a periodic background.
- **Small copies** (< ~2 % of the image) or strong blur / heavy compression give too
  few matches.
- **Source vs copy** cannot be distinguished from one image.
- **Similarity model**: perspective or non-rigid warps of the copy are not modelled.
- The **detection score** is a heuristic, not a calibrated probability.
- **Splicing** module is experimental; natural content triggers its indicators.

## 9. Future scope

- Hybrid block + keypoint matching (e.g. Zernike / PCT block features in flat areas).
- Full affine or homography models and local multi-scale refinement.
- Learned descriptors or a lightweight CNN for splicing, trained on CASIA/DEFACTO.
- Calibrate the detection score into a probability using a held-out dataset.
- JPEG double-quantisation and camera-noise (PRNU) analysis for splicing.
- Batch processing API and GPU acceleration for large archives.

## 10. Likely viva questions

**Why match the image against itself?** The copy and the source are in the same image,
so duplicated content appears as two similar descriptors at different places.

**Why g2NN instead of Lowe's ratio test?** If an object is pasted twice, the 2nd nearest
neighbour is also a true match; Lowe's test would reject both.

**Why RANSAC?** Thousands of descriptor matches contain many false ones. A genuine copy
moves all its points with one transform; RANSAC finds that transform despite outliers.

**Why not just draw a box around the matched keypoints?** Keypoints are sparse; the box
would include unrelated pixels and miss parts of the copy. The transform-verified step
checks every pixel.

**How do you avoid flagging tiles or bricks?** Overlap between the two sides, lattice
structure of the shifts, and the number of groups — such cases become *Inconclusive*.

**Why SIFT as default if ORB is faster?** SIFT gave higher recall and IoU on rotated and
scaled copies in our evaluation; ORB stays available for speed.

**What does the detection score mean?** Only how much verified geometric evidence was
found, normalised to 0–100; it is not a probability.
