"""
IMAGE FORENSICS - Image Tampering Detection System (Streamlit frontend).

Run with:   streamlit run app.py

The app only handles the user interface. All image processing lives in
the `modules` package.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from modules.config import (ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB, SAMPLES_DIR,
                            SENSITIVITY_PRESETS, SYNTHETIC_DIR, AnalysisConfig)
from modules.evaluation import evaluate_dataset, pixel_metrics
from modules.feature_extraction import available_methods, compare_methods
from modules.forgery_detection import STATUS_INCONCLUSIVE, STATUS_MANIPULATED
from modules.pipeline import run_analysis
from modules.preprocessing import ImageValidationError, validate_image_bytes
from modules.report_generator import DISCLAIMER, build_html, build_pdf, json_bytes
from modules.splicing_detection import STATUS_INDICATORS, analyze_splicing
from modules.visualization import (build_copy_move_visuals, build_splicing_visuals,
                                   heatmap_overlay, to_png_bytes)

st.set_page_config(page_title="Image Forensics", page_icon="🔍", layout="wide")

# --------------------------------------------------------------------------
# Styling: minimal forensic dashboard - white, navy text, blue accents.
# --------------------------------------------------------------------------
st.markdown("""
<style>
.block-container {padding-top: 2.2rem; max-width: 1280px;}
.kicker {font-size: .74rem; letter-spacing: .2em; font-weight: 700; color: #1d4ed8;}
.title {font-size: 2.0rem; font-weight: 700; color: #0f1b33; margin: .1rem 0 .3rem 0;}
.lede {color: #475569; font-size: .98rem; max-width: 780px;}
.card {border: 1px solid #e2e8f0; border-radius: 6px; padding: 14px 16px; background: #fff;}
.card .k {font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; color: #64748b;}
.card .v {font-size: 1.45rem; font-weight: 700; color: #0f1b33; margin-top: 2px;}
.verdict {border-radius: 6px; padding: 16px 20px; color: #fff; margin: .4rem 0 1rem 0;}
.verdict .s {font-size: 1.25rem; font-weight: 700;}
.verdict .d {font-size: .9rem; opacity: .92; margin-top: 2px;}
.v-red {background: #b91c1c;} .v-amber {background: #b45309;} .v-green {background: #15803d;}
.disclaimer {border-left: 3px solid #f59e0b; background: #fffbeb; padding: 10px 14px;
             font-size: .86rem; color: #78350f; margin: .6rem 0;}
.step {display:flex; gap:10px; margin:.25rem 0; color:#334155; font-size:.9rem;}
.step b {color:#1d4ed8; min-width: 18px;}
section[data-testid="stSidebar"] .kicker {margin-bottom: .2rem;}
</style>
""", unsafe_allow_html=True)


def card(col, label: str, value) -> None:
    col.markdown(f'<div class="card"><div class="k">{label}</div>'
                 f'<div class="v">{value}</div></div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Cached helpers
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def ensure_samples() -> list[str]:
    """Create the demo samples once if the folder is missing."""
    if not SAMPLES_DIR.exists() or not any(SAMPLES_DIR.iterdir()):
        from modules.synthetic import generate_demo_samples
        generate_demo_samples(SAMPLES_DIR)
    return sorted(p.name for p in SAMPLES_DIR.iterdir()
                  if p.suffix.lower() in ALLOWED_EXTENSIONS and "_mask" not in p.stem)


@st.cache_resource(show_spinner=False, max_entries=8)
def cached_analysis(data_hash: str, data: bytes, filename: str, cfg_json: str):
    """Validation + full analysis, cached per (image, settings)."""
    img, info = validate_image_bytes(data, filename)
    cfg = AnalysisConfig(**json.loads(cfg_json))
    result = run_analysis(img, cfg, info)
    visuals = build_copy_move_visuals(result)
    splicing_visuals = None
    if cfg.mode.startswith("Splicing"):
        result.splicing = analyze_splicing(result.pre.original_bgr)
        splicing_visuals = build_splicing_visuals(result.pre.rgb, result.splicing)
    return result, visuals, splicing_visuals


@st.cache_resource(show_spinner=False, max_entries=4)
def cached_method_comparison(data_hash: str, gray: np.ndarray):
    return compare_methods(gray)


# --------------------------------------------------------------------------
# Sidebar: analysis settings
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="kicker">ANALYSIS SETTINGS</div>', unsafe_allow_html=True)
    mode = st.radio("Detection mode", ["Copy-Move", "Splicing (Experimental)"],
                    help="Copy-move is the main, verified method. Splicing shows "
                         "statistical indicators only.")
    methods = available_methods()
    method = st.selectbox("Feature method", methods,
                          index=methods.index("SIFT") if "SIFT" in methods else 0,
                          help="SIFT: more robust to rotation/scale/JPEG. ORB: faster, binary.")
    sens = st.select_slider("Detection sensitivity", list(SENSITIVITY_PRESETS), value="Medium",
                            help="Higher sensitivity accepts weaker evidence "
                                 "(more detections, more false alarms).")
    preset = SENSITIVITY_PRESETS[sens]
    with st.expander("Feature matching parameters"):
        ratio = st.slider("g2NN ratio threshold", 0.3, 0.9, preset["ratio_thresh"], 0.05,
                          key=f"ratio_{sens}")
        min_inliers = st.slider("Min RANSAC inliers per group", 4, 30, preset["min_inliers"],
                                key=f"inl_{sens}")
        ransac = st.slider("RANSAC reprojection threshold (px)", 1.0, 10.0,
                           preset["ransac_thresh"], 0.5, key=f"ran_{sens}")
        min_dist = st.slider("Min distance between copies (% of diagonal)", 1, 15, 4) / 100
        n_feat = st.slider("Max keypoints", 1000, 12000, 6000, 500)
        max_dim = st.select_slider("Working resolution (longest side)",
                                   [512, 768, 1024, 1280, 1600], value=1024)
    with st.expander("Preprocessing & localization"):
        denoise = st.selectbox("Noise reduction", ["gaussian", "bilateral", "none"])
        clahe = st.checkbox("CLAHE contrast enhancement", True)
        refine = st.checkbox("Pixel-level refinement (transform-verified)", True,
                             help="Off: show only the keypoint-cluster region estimate.")

cfg = AnalysisConfig.from_sensitivity(
    sens, method=method, mode=mode, ratio_thresh=ratio, min_inliers=min_inliers,
    ransac_thresh=ransac, min_spatial_dist=min_dist, n_features=n_feat,
    max_dim=max_dim, denoise=denoise, use_clahe=clahe, refine_localization=refine)

# --------------------------------------------------------------------------
# Header / dashboard
# --------------------------------------------------------------------------
st.markdown('<div class="kicker">IMAGE FORENSICS</div>'
            '<div class="title">Image Tampering Detection System</div>'
            '<div class="lede">Detects and localizes <b>copy-move forgery</b> - a part of '
            'the image copied and pasted elsewhere in the same image, optionally rotated '
            'or resized - using keypoint matching and RANSAC geometric verification. '
            'An experimental splicing module reports statistical inconsistencies.</div>',
            unsafe_allow_html=True)

tab_analyze, tab_eval, tab_how = st.tabs(["Analyze an image", "Dataset evaluation",
                                          "How it works"])

# ==========================================================================
# TAB 1 - ANALYZE
# ==========================================================================
def render_analyze() -> None:
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        source = st.radio("Image source", ["Upload an image", "Use a demo sample"],
                          horizontal=True, label_visibility="collapsed")
        data, filename, gt_mask = None, None, None
        if source == "Upload an image":
            up = st.file_uploader("Drag and drop an image",
                                  type=[e.strip(".") for e in ALLOWED_EXTENSIONS],
                                  help=f"JPG, JPEG or PNG, up to {MAX_FILE_SIZE_MB} MB")
            if up is not None:
                data, filename = up.getvalue(), up.name
        else:
            samples = ensure_samples()
            choice = st.selectbox("Demo sample", samples,
                                  help="Generated copy-move test cases with known "
                                       "ground truth, plus negative and invalid cases.")
            p = SAMPLES_DIR / choice
            data, filename = p.read_bytes(), p.name
            mp = SAMPLES_DIR / f"{p.stem}_mask.png"
            if mp.exists():
                gt_mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
    with right:
        st.markdown("**Detection process**")
        for i, s in enumerate(["Validate and preprocess (resize, grayscale, denoise, CLAHE)",
                               f"Extract {method} keypoints and descriptors",
                               "Match the image against itself (g2NN ratio test)",
                               "RANSAC: keep matches agreeing on one transform",
                               "Dense check + verdict",
                               "Localize regions: mask, contours, boxes"], 1):
            st.markdown(f'<div class="step"><b>{i}</b><span>{s}</span></div>',
                        unsafe_allow_html=True)
        st.caption("Supported formats: JPG, JPEG, PNG · max "
                   f"{MAX_FILE_SIZE_MB} MB · min 64 px per side")

    if data is None:
        st.info("Upload an image or pick a demo sample to start.")
        return

    digest = hashlib.sha1(data).hexdigest()
    try:
        with st.spinner("Analysing image..."):
            result, visuals, spl_visuals = cached_analysis(
                digest, data, filename, json.dumps(cfg.to_dict()))
    except ImageValidationError as exc:
        st.error(f"**{filename}** cannot be analysed: {exc}")
        return

    info, det, loc = result.info, result.detection, result.localization

    # ---- Upload preview -------------------------------------------------
    st.markdown("---")
    pc, ic = st.columns([1, 2], gap="large")
    pc.image(result.pre.rgb, caption="Uploaded image", width="stretch")
    with ic:
        st.markdown("**Image details**")
        c1, c2, c3, c4 = st.columns(4)
        card(c1, "Filename", f'<span style="font-size:.85rem">{info.filename}</span>')
        card(c2, "Dimensions", f"{info.width}×{info.height}")
        card(c3, "File size", f"{info.file_size_kb:,.0f} KB")
        card(c4, "Format", info.format)
        if result.pre.scale < 1:
            st.caption(f"Analysed at {result.pre.gray.shape[1]}×{result.pre.gray.shape[0]} "
                       "(aspect ratio preserved); masks are scaled back to full size.")

    # ---- Experimental splicing view -------------------------------------
    if result.splicing is not None:
        sp = result.splicing
        st.subheader("Experimental splicing indicators")
        cls = "v-amber" if sp.status == STATUS_INDICATORS else "v-green"
        st.markdown(f'<div class="verdict {cls}"><div class="s">{sp.status}</div>'
                    f'<div class="d">Flagged area: {100 * sp.suspicious_fraction:.1f}% of '
                    f'usable blocks · block size {sp.block_size}px</div></div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="disclaimer">These are forensic indicators (noise, ELA, '
                    'edge and texture statistics). Natural content also changes them, so '
                    'false positives are expected. They do not prove splicing.</div>',
                    unsafe_allow_html=True)
        cols = st.columns(3)
        for i, (title, img) in enumerate(spl_visuals.items()):
            cols[i % 3].image(img, caption=title, width="stretch")
        for n in sp.notes:
            st.caption("• " + n)
        st.markdown("#### Copy-move result for the same image")

    # ---- Verdict --------------------------------------------------------
    cls = {STATUS_MANIPULATED: "v-red", STATUS_INCONCLUSIVE: "v-amber"}.get(det.status, "v-green")
    st.markdown(f'<div class="verdict {cls}"><div class="s">{det.status}</div>'
                f'<div class="d">Suspected type: {det.forgery_type} · '
                f'{det.n_valid_groups} duplicated region pair(s) · detection score '
                f'{det.score:.0f}/100</div></div>', unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    card(m1, "Keypoints", f"{result.features.count:,}")
    card(m2, "Candidate matches", f"{result.matches.count:,}")
    card(m3, "Verified matches", det.n_verified_matches)
    card(m4, "Suspicious regions", loc.n_regions)
    card(m5, "Processing time", f"{result.total_time_s:.2f} s")

    # ---- Visual results -------------------------------------------------
    st.subheader("Analysis results")
    items = list(visuals.items())
    for row in range(0, len(items), 3):
        cols = st.columns(3)
        for col, (title, img) in zip(cols, items[row:row + 3]):
            col.image(img, caption=title, width="stretch")
    st.caption(f"Localization method: **{loc.method}**. " + " ".join(loc.notes))

    # ---- Forensic summary ----------------------------------------------
    st.subheader("Forensic summary")
    s1, s2 = st.columns([1, 1], gap="large")
    with s1:
        summ = result.summary()
        table = pd.DataFrame(
            [(k.replace("_", " ").capitalize(), str(v)) for k, v in summ.items()],
            columns=["Field", "Value"])
        st.dataframe(table, hide_index=True, width="stretch")
    with s2:
        st.markdown("**Findings**")
        for n in det.notes:
            st.markdown(f"- {n}")
        with st.expander("How is the detection score calculated?"):
            st.markdown("The detection score is a **heuristic measure of the amount of "
                        "geometric evidence**, not a probability:")
            st.code(det.score_explanation)
            st.caption("It is not calibrated on any dataset; use it to compare images "
                       "analysed with the same settings.")
        if result.verification.groups:
            st.markdown("**RANSAC match groups**")
            st.dataframe(pd.DataFrame([g.summary() for g in result.verification.groups]),
                         hide_index=True, width="stretch")
        if gt_mask is not None:
            pm = pixel_metrics(gt_mask, result.mask_original_size())
            st.markdown("**Against this sample's ground-truth mask**")
            g1, g2, g3, g4 = st.columns(4)
            card(g1, "Pixel precision", f"{pm['pixel_precision']:.2f}")
            card(g2, "Pixel recall", f"{pm['pixel_recall']:.2f}")
            card(g3, "Pixel F1", f"{pm['pixel_f1']:.2f}")
            card(g4, "IoU", f"{pm['iou']:.2f}")
    st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)

    with st.expander("Compare ORB and SIFT on this image"):
        st.dataframe(pd.DataFrame(cached_method_comparison(digest, result.pre.processed)),
                     hide_index=True, width="stretch")
        st.caption("Counts and timings for feature extraction only. Switch the method in "
                   "the sidebar to compare full detection results.")

    # ---- Downloads ------------------------------------------------------
    st.subheader("Download report")
    stem = Path(filename).stem
    d1, d2, d3, d4, d5, d6 = st.columns(6)
    d1.download_button("HTML report", build_html(result, visuals, spl_visuals),
                       f"{stem}_report.html", "text/html", width="stretch")
    d2.download_button("PDF report", build_pdf(result, visuals), f"{stem}_report.pdf",
                       "application/pdf", width="stretch")
    d3.download_button("Tampering mask", to_png_bytes(result.mask_original_size()),
                       f"{stem}_mask.png", "image/png", width="stretch")
    d4.download_button("Highlighted image", to_png_bytes(items[7][1]),
                       f"{stem}_highlighted.png", "image/png", width="stretch")
    d5.download_button("Match visualization", to_png_bytes(items[5][1]),
                       f"{stem}_matches.png", "image/png", width="stretch")
    d6.download_button("JSON results", json_bytes(result), f"{stem}_analysis.json",
                       "application/json", width="stretch")
    if result.splicing is not None:
        st.download_button("Splicing heatmap (PNG)",
                           to_png_bytes(heatmap_overlay(result.pre.rgb,
                                                        result.splicing.combined_map)),
                           f"{stem}_splicing_heatmap.png", "image/png")


with tab_analyze:
    render_analyze()

# ==========================================================================
# TAB 2 - DATASET EVALUATION
# ==========================================================================
with tab_eval:
    st.markdown("Evaluate the detector on a folder with `original/`, `forged/` and "
                "`masks/` sub-folders (CoMoFoD, CASIA v2.0 or DEFACTO after sorting, "
                "or the built-in synthetic set). Uses the settings in the sidebar.")
    e1, e2, e3 = st.columns([2, 1, 1])
    ds_path = e1.text_input("Dataset folder (relative to the project or absolute)",
                            str(SYNTHETIC_DIR.relative_to(Path(__file__).parent)))
    limit = e2.number_input("Max images per folder (0 = all)", 0, 5000, 0)
    incl = e3.checkbox("Count 'Inconclusive' as positive", False)

    ds = Path(ds_path)
    if not ds.is_absolute():
        ds = Path(__file__).parent / ds
    if not (ds / "forged").is_dir():
        st.warning(f"No `forged/` folder found in {ds}.")
        if ds.resolve() == SYNTHETIC_DIR.resolve() and st.button("Generate synthetic dataset"):
            from modules.synthetic import generate_suite
            with st.spinner("Generating 25 forged images with ground-truth masks..."):
                generate_suite(SYNTHETIC_DIR)
            st.rerun()
    elif st.button("Run evaluation", type="primary"):
        bar = st.progress(0.0, "Starting...")
        rep = evaluate_dataset(ds, cfg, limit or None, incl,
                               progress=lambda i, n, f: bar.progress(i / n, f"{i}/{n} · {f}"))
        bar.empty()
        st.session_state["eval_report"] = rep

    rep = st.session_state.get("eval_report")
    if rep is not None:
        im = rep.image_metrics
        st.markdown("#### Image-level detection (is the image forged?)")
        a, b, c, d, e = st.columns(5)
        card(a, "Images", im["n_images"])
        card(b, "Accuracy", f"{im['accuracy']:.3f}")
        card(c, "Precision", f"{im['precision']:.3f}")
        card(d, "Recall", f"{im['recall']:.3f}")
        card(e, "F1-score", f"{im['f1']:.3f}")
        cm = im["confusion_matrix"]
        st.dataframe(pd.DataFrame({"Predicted authentic": [cm["TN"], cm["FN"]],
                                   "Predicted manipulated": [cm["FP"], cm["TP"]]},
                                  index=["Actually authentic", "Actually forged"]),
                     width="content")
        st.markdown("#### Pixel-level localization (where is the forgery?)")
        pm, pd_ = rep.pixel_metrics_mean, rep.pixel_metrics_detected_only
        if pm.get("n_images"):
            rows = []
            for name, src in (("All forged images with masks", pm),
                              ("Only images the detector flagged", pd_)):
                rows.append({"Subset": name, "Images": src["n_images"],
                             "Pixel precision": round(src.get("pixel_precision", 0), 3),
                             "Pixel recall": round(src.get("pixel_recall", 0), 3),
                             "Pixel F1": round(src.get("pixel_f1", 0), 3),
                             "IoU": round(src.get("iou", 0), 3)})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("If a dataset's masks mark only the pasted copy (not the source), "
                       "pixel precision is underestimated, because this detector marks both.")
        else:
            st.caption("No ground-truth masks were found, so pixel metrics are unavailable.")
        st.markdown(f"Mean processing time: **{rep.per_image['time_s'].mean():.2f} s**/image")
        st.dataframe(rep.per_image, hide_index=True, width="stretch")
        st.download_button("Download per-image CSV", rep.per_image.to_csv(index=False),
                           "evaluation_per_image.csv", "text/csv")
        st.download_button("Download summary JSON", json.dumps(rep.to_dict(), indent=2),
                           "evaluation_summary.json", "application/json")
        for err in rep.errors:
            st.warning(err)

# ==========================================================================
# TAB 3 - HOW IT WORKS
# ==========================================================================
with tab_how:
    st.markdown("""
#### What is detected
**Copy-move forgery**: part of an image is copied and pasted elsewhere in the *same*
image (to duplicate or hide objects), possibly rotated, resized or JPEG-compressed.

#### Pipeline
1. **Preprocessing** - resize (aspect preserved), grayscale, light Gaussian denoise and
   CLAHE so that low-contrast areas still produce keypoints.
2. **Feature extraction** - ORB (FAST corners + binary BRIEF, Hamming distance) or
   SIFT (Difference-of-Gaussians + 128-D gradient histograms, L2 distance).
3. **Self-matching** - every descriptor is compared with all others in the same image.
   The generalised ratio test (g2NN) accepts neighbour *i* while
   d<sub>i</sub>/d<sub>i+1</sub> < T, so multiple copies are found. Matches between
   nearby keypoints are discarded.
4. **RANSAC** - fits a similarity transform (rotation + scale + translation) and keeps
   matches that agree with it; repeated to find several copied regions.
5. **Dense check** - groups with few keypoint matches are accepted only if local
   normalised cross-correlation between the image and its warped copy confirms the
   duplication over a textured area.
6. **Verdict** - groups whose two sides overlap, or whose shifts form a regular lattice,
   are treated as natural repetition (tiles, bricks) -> *Inconclusive*.
7. **Localization** - clustered keypoints give a region estimate (hull + morphology);
   the image is then compared with itself warped by the transform to get a pixel-level
   mask, cleaned with opening/closing and connected-component analysis.

#### Limitations
- Flat, texture-less copied regions (sky, walls) produce no keypoints -> missed.
- Naturally repeated objects can be indistinguishable from a copy -> *Inconclusive*.
- Heavy blur, strong compression or large scale changes reduce matches.
- Splicing from another image is out of scope for the main method; the experimental
  module only reports statistical indicators.
""", unsafe_allow_html=True)
