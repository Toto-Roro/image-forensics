"""Feature extraction, self-matching and RANSAC verification."""
import math

import numpy as np
import pytest

from modules.config import AnalysisConfig
from modules.feature_extraction import (available_methods, compare_methods,
                                        extract_features, sift_available)
from modules.feature_matching import match_self
from modules.geometric_verification import decompose_similarity, verify_matches
from modules.preprocessing import preprocess


@pytest.mark.parametrize("method", available_methods())
def test_extracts_keypoints_and_descriptors(astronaut, method):
    pre = preprocess(astronaut)
    f = extract_features(pre.processed, method, 3000)
    assert f.count > 200
    assert f.descriptors.shape[0] == f.count == len(f.coords)
    assert f.descriptors.shape[1] == (32 if method == "ORB" else 128)


def test_flat_image_gives_no_features():
    f = extract_features(np.full((200, 200), 128, np.uint8), "ORB")
    assert f.count == 0 and f.descriptors is None
    assert match_self(f).count == 0


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        extract_features(np.zeros((100, 100), np.uint8), "SURF")


def test_compare_methods_lists_all(astronaut):
    rows = compare_methods(preprocess(astronaut).processed)
    assert {r["Method"] for r in rows} == set(available_methods())


@pytest.mark.parametrize("method", available_methods())
def test_self_matching_finds_copied_region(forged_cases, method):
    forged, gt = forged_cases[("astronaut", "translate")]
    pre = preprocess(forged)
    f = extract_features(pre.processed, method, 6000)
    min_dist = 0.04 * math.hypot(*pre.gray.shape)
    m = match_self(f, 0.6, min_dist)
    assert m.count >= 10
    # No self matches, no duplicates, and every pair respects the spatial filter.
    assert np.all(m.pairs[:, 0] < m.pairs[:, 1])
    assert len({tuple(p) for p in m.pairs}) == m.count
    assert np.all(np.linalg.norm(m.src_pts - m.dst_pts, axis=1) >= min_dist)
    # Most matches have both endpoints inside the ground-truth mask.
    inside = [gt[int(y), int(x)] > 0 and gt[int(v), int(u)] > 0
              for (x, y), (u, v) in zip(m.src_pts, m.dst_pts)]
    assert np.mean(inside) > 0.6
    assert m.n_after_ratio >= m.n_after_spatial


def test_ransac_recovers_rotation_and_scale(forged_cases):
    if not sift_available():
        pytest.skip("SIFT unavailable")
    forged, _ = forged_cases[("camera", "rotate_scale")]
    cfg = AnalysisConfig()
    pre = preprocess(forged, cfg)
    f = extract_features(pre.processed, "SIFT", cfg.n_features)
    m = match_self(f, cfg.ratio_thresh, cfg.min_spatial_dist * math.hypot(*pre.gray.shape))
    ver = verify_matches(m, pre.gray, cfg)
    g = ver.valid_groups[0]
    # The generator rotated by 20-45 deg and scaled by 0.78-0.85 or 1.2-1.3;
    # RANSAC may report the A->B or B->A direction.
    rot, sc = abs(g.rotation_deg), g.scale if g.scale >= 1 else 1 / g.scale
    assert 18 <= rot <= 47
    assert 1.15 <= sc <= 1.32
    assert g.mean_error_px < cfg.ransac_thresh


def test_decompose_similarity():
    a = math.radians(30)
    M = np.array([[2 * math.cos(a), -2 * math.sin(a), 5],
                  [2 * math.sin(a), 2 * math.cos(a), -7]])
    rot, scale, t = decompose_similarity(M)
    assert rot == pytest.approx(30) and scale == pytest.approx(2) and t == (5, -7)


def test_verification_with_too_few_matches(astronaut):
    cfg = AnalysisConfig()
    pre = preprocess(astronaut, cfg)
    f = extract_features(pre.processed, "ORB", 50)
    m = match_self(f, 0.1, 10)
    ver = verify_matches(m, pre.gray, cfg)
    assert ver.n_verified == 0
