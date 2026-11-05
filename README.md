# IMAGE FORENSICS — Image Tampering Detection System

**Image Tampering / Forgery Detection and Localization Using Computer Vision**
(college mini-project)

Detects and localizes **copy-move forgery** — a region of an image copied and pasted
elsewhere in the same image, optionally rotated, resized or JPEG-compressed — using
classical computer vision: ORB / SIFT keypoints, self-matching with the generalised
ratio test, RANSAC geometric verification, dense correlation checks and morphological
localization. No deep-learning model, no pretrained weights, no dataset download needed.

An **experimental** splicing module reports statistical inconsistencies (noise level,
error-level analysis, edge sharpness, residual texture) as indicators only.

> Automated image-forensics results are indicative and should not be treated as
> definitive proof of image manipulation.

---

## Quick start

```bash
cd Image-Tampering-Detection
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m modules.synthetic          # optional: (re)create demo samples + synthetic dataset
streamlit run app.py                 # opens http://localhost:8501
```

Python 3.10+ is required (tested on 3.13 with OpenCV 5.0, Streamlit 1.66).
The `samples/` folder and `dataset/synthetic/` are already included, so the app can be
demonstrated immediately: choose **Use a demo sample** in the app.

### Run without installing anything locally (browser only)

1. Push this folder to a GitHub repository.
2. Go to <https://share.streamlit.io>, sign in with GitHub, click **Create app**, pick
   the repo and set the main file to `app.py`.
3. Streamlit Community Cloud installs `requirements.txt` and gives you a public URL.
   (`opencv-python-headless` is used precisely so this works without system libraries.)

---

## Running the tests

```bash
python -m pytest -q
```

69 tests cover validation (incl. corrupt files), preprocessing, ORB/SIFT extraction,
g2NN matching, RANSAC transform recovery, verdicts for all eight required test cases,
localization IoU against ground truth, metric edge cases, dataset evaluation, report
generation and the splicing module. Last run: **69 passed in ~11 s**.

## Evaluating on a dataset

```bash
python -m modules.evaluation --dataset dataset/synthetic            # built-in set
python -m modules.evaluation --dataset dataset --method ORB --limit 100
```

Outputs `results/eval_<name>_<time>/per_image.csv`, `summary.json` and
`confusion_matrix.png`. The same evaluation is available in the app's
**Dataset evaluation** tab. See [`dataset/README.md`](dataset/README.md) for how to
lay out CoMoFoD, CASIA v2.0 and DEFACTO.

### Measured results on the built-in synthetic set

5 untouched photos + 25 forgeries (translate, rotate, scale, rotate+scale, JPEG q=75),
default settings (Medium sensitivity). These are real outputs of the command above,
**on synthetic data only** — they are not CoMoFoD/CASIA results.

| Method | Accuracy | Precision | Recall | F1 | Pixel F1 (all forged) | IoU (all forged) | IoU (detected only) | Time / image |
|---|---|---|---|---|---|---|---|---|
| SIFT | 0.900 | 1.000 | 0.880 | 0.936 | 0.818 | 0.764 | 0.868 | 0.18 s |
| ORB  | 0.867 | 1.000 | 0.840 | 0.913 | 0.723 | 0.640 | 0.762 | 0.23 s |

SIFT confusion matrix: TP 22, FP 0, FN 3, TN 5. Misses: two scaled copies in the
low-contrast `coffee` photo and one rotate+scale copy in `rocket`. `rocket` (untouched)
is reported *Inconclusive* because it really contains repeated tower structures.

---

## Project structure

```
Image-Tampering-Detection/
├── app.py                      Streamlit frontend (UI only)
├── modules/
│   ├── config.py               paths, limits, sensitivity presets, AnalysisConfig
│   ├── preprocessing.py        upload validation, resize, grayscale, denoise, CLAHE
│   ├── feature_extraction.py   ORB / SIFT keypoints + descriptors, method comparison
│   ├── feature_matching.py     self-matching, g2NN ratio test, spatial filter (BF / FLANN)
│   ├── geometric_verification.py  multi-group RANSAC + dense NCC confirmation
│   ├── forgery_detection.py    verdict, detection score, repetitive-pattern guard
│   ├── localization.py         clustering, hull + morphology, transform-verified mask
│   ├── pipeline.py             runs the whole chain -> AnalysisResult
│   ├── splicing_detection.py   EXPERIMENTAL noise / ELA / edge / texture indicators
│   ├── visualization.py        the nine visual outputs + splicing heatmaps
│   ├── report_generator.py     JSON, self-contained HTML, PDF reports
│   ├── evaluation.py           image- and pixel-level metrics, dataset evaluation CLI
│   └── synthetic.py            synthetic copy-move generator with ground-truth masks
├── tests/                      pytest suite (conftest + 4 test files)
├── samples/                    demo images for the 8 test cases (+ GT masks)
├── dataset/                    original/ forged/ masks/  +  synthetic/
├── results/                    saved reports and evaluations
├── docs/
│   ├── GUIDE.md                step-by-step guide: how it works, how to use it
│   └── VIVA.md                 technical write-up for the viva
├── .streamlit/config.toml      theme + 20 MB upload limit
└── requirements.txt
```

## Using the app

1. **Image source** — upload a JPG/PNG (drag and drop) or pick a demo sample.
2. **Sidebar** — detection mode (Copy-Move / Splicing experimental), feature method
   (ORB / SIFT), sensitivity (Low / Medium / High) and advanced parameters.
3. **Results** — verdict banner, key numbers, the nine visual outputs, forensic summary,
   RANSAC group table, and (for demo samples) pixel metrics against ground truth.
4. **Downloads** — HTML report, PDF report, tampering mask, highlighted image, match
   visualization, JSON results.

Full explanations of every step are in [`docs/GUIDE.md`](docs/GUIDE.md).

## Limitations (short)

- Texture-less copied regions (sky, plain walls) give no keypoints and are missed.
- Naturally repeated content (tiles, windows, bricks) is genuinely duplicated; the system
  reports it as *Inconclusive* rather than guessing.
- Strong blur, heavy compression or large scale changes reduce the number of matches.
- Which of the two duplicated regions is the original cannot be decided from one image.
- Splicing detection is experimental and expected to produce false positives.
