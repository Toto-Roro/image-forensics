"""Localization quality, mask format, and supporting modules (metrics, reports, splicing)."""
import json

import numpy as np
import pytest

from modules.config import AnalysisConfig
from modules.evaluation import (evaluate_dataset, find_mask, image_level_metrics,
                                pixel_metrics)
from modules.localization import cluster_points
from modules.pipeline import run_analysis
from modules.report_generator import build_html, build_json, build_pdf
from modules.splicing_detection import analyze_splicing
from modules.synthetic import generate_suite
from modules.visualization import build_copy_move_visuals, build_splicing_visuals


@pytest.mark.parametrize("kind", ["translate", "rotate", "scale", "jpeg"])
def test_localization_overlaps_ground_truth(forged_cases, iou, kind):
    forged, gt = forged_cases[("astronaut", kind)]
    r = run_analysis(forged, AnalysisConfig())
    mask = r.mask_original_size()
    assert mask.shape == gt.shape
    assert set(np.unique(mask)) <= {0, 255}
    assert iou(mask, gt) > 0.5


def test_both_sides_of_copy_are_localized(forged_cases):
    r = run_analysis(forged_cases[("camera", "translate")][0], AnalysisConfig())
    roles = {reg.role for reg in r.localization.regions}
    assert roles == {"A", "B"}
    for reg in r.localization.regions:
        x, y, w, h = reg.bbox
        assert w > 0 and h > 0 and reg.area_px > 0
        assert reg.contour is not None and len(reg.contour) >= 3


def test_keypoint_only_localization_is_labelled(forged_cases, iou):
    forged, gt = forged_cases[("astronaut", "translate")]
    r = run_analysis(forged, AnalysisConfig(refine_localization=False))
    assert r.localization.method == "keypoint-region estimate"
    assert iou(r.mask_original_size(), gt) > 0.3


def test_refined_is_tighter_than_keypoint_estimate(forged_cases, iou):
    forged, gt = forged_cases[("astronaut", "rotate")]
    refined = run_analysis(forged, AnalysisConfig())
    coarse = run_analysis(forged, AnalysisConfig(refine_localization=False))
    assert iou(refined.mask_original_size(), gt) >= iou(coarse.mask_original_size(), gt)


def test_clean_image_gives_empty_mask(base_images):
    r = run_analysis(base_images["chelsea"])
    assert r.localization.n_regions == 0 and r.localization.coverage == 0


def test_cluster_points_separates_far_groups():
    pts = np.array([[0, 0], [5, 5], [3, 1], [200, 200], [205, 198], [201, 207]], float)
    assert len(cluster_points(pts, eps=30)) == 2
    assert cluster_points(pts[:2], eps=30) == []          # below min_size


# ---- metrics ------------------------------------------------------------
def test_image_metrics_and_zero_division():
    m = image_level_metrics([1, 1, 0, 0], [1, 0, 0, 1])
    assert m["confusion_matrix"] == {"TP": 1, "FP": 1, "FN": 1, "TN": 1}
    assert m["accuracy"] == 0.5 and m["f1"] == 0.5
    none = image_level_metrics([0, 0], [0, 0])            # no positives at all
    assert none["precision"] == 0.0 and none["recall"] == 0.0 and none["accuracy"] == 1.0


def test_pixel_metrics_edge_cases():
    gt = np.zeros((10, 10), np.uint8)
    gt[2:6, 2:6] = 255
    assert pixel_metrics(gt, gt)["iou"] == 1.0
    assert pixel_metrics(gt, np.zeros_like(gt))["pixel_recall"] == 0.0
    empty = np.zeros_like(gt)
    assert pixel_metrics(empty, empty)["iou"] == 1.0       # both empty: agree
    assert pixel_metrics(empty, gt)["pixel_precision"] == 0.0
    half = np.zeros((20, 20), np.uint8)
    half[4:12, 4:12] = 255
    assert pixel_metrics(gt, half)["iou"] == 1.0           # resized to GT size


def test_find_mask_conventions(tmp_path):
    (tmp_path / "masks").mkdir()
    for n in ("a_mask.png", "b_gt.png", "001_B.png", "c.png"):
        (tmp_path / "masks" / n).write_bytes(b"x")
    m = tmp_path / "masks"
    assert find_mask(tmp_path / "a.jpg", m).name == "a_mask.png"
    assert find_mask(tmp_path / "b.tif", m).name == "b_gt.png"
    assert find_mask(tmp_path / "001_F.png", m).name == "001_B.png"
    assert find_mask(tmp_path / "001_F_JC3.png", m).name == "001_B.png"
    assert find_mask(tmp_path / "c.png", m).name == "c.png"
    assert find_mask(tmp_path / "zzz.png", m) is None


def test_dataset_evaluation_end_to_end(tmp_path):
    generate_suite(tmp_path, seed=3, kinds=("translate", "rotate"),
                   names=("astronaut", "chelsea"))
    rep = evaluate_dataset(tmp_path, AnalysisConfig())
    assert rep.image_metrics["n_images"] == 6
    assert rep.pixel_metrics_mean["n_images"] == 4
    assert 0 <= rep.pixel_metrics_mean["iou"] <= 1
    assert rep.image_metrics["recall"] >= 0.75
    json.dumps(rep.to_dict())


def test_dataset_evaluation_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluate_dataset(tmp_path)


# ---- reports & experimental splicing -------------------------------------
def test_reports_generate(forged_cases):
    r = run_analysis(forged_cases[("astronaut", "translate")][0])
    vis = build_copy_move_visuals(r)
    assert len(vis) == 9 and all(v.ndim == 3 for v in vis.values())
    j = build_json(r)
    json.dumps(j)
    assert j["summary"]["status"] == r.detection.status
    assert "disclaimer" in j
    html = build_html(r, vis)
    assert "data:image/png;base64" in html and "indicative" in html
    assert build_pdf(r, vis)[:4] == b"%PDF"


def test_splicing_module_runs_and_flags_pasted_noise(base_images):
    img = base_images["coffee"].copy()
    rng = np.random.default_rng(0)
    patch = img[120:248, 200:328].astype(float) + rng.normal(0, 14, (128, 128, 3))
    img[120:248, 200:328] = np.clip(patch, 0, 255).astype(np.uint8)
    res = analyze_splicing(img)
    assert res.combined_map.shape == img.shape[:2]
    # The noisy pasted block should score higher than the rest of the image.
    inside = res.combined_map[136:232, 216:312].mean()
    assert inside > 2 * res.combined_map.mean()
    vis = build_splicing_visuals(img[..., ::-1].copy(), res)
    assert "Combined inconsistency heatmap" in vis
    assert "experimental" in res.status.lower()
