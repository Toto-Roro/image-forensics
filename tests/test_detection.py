"""
Image-level verdicts for the required test cases:
 1 untouched photo, 2 copied object, 3 translated, 4 rotated, 5 resized,
 6 compressed, 7 repeated natural pattern.  (Case 8, invalid files, is in
 test_preprocessing.py.)
"""
import pytest

from modules.config import AnalysisConfig
from modules.feature_extraction import available_methods
from modules.forgery_detection import (STATUS_CLEAN, STATUS_INCONCLUSIVE,
                                       STATUS_MANIPULATED)
from modules.pipeline import run_analysis
from modules.synthetic import jpeg_roundtrip

CLEAN_PHOTOS = ("astronaut", "coffee", "chelsea", "camera")


@pytest.mark.parametrize("name", CLEAN_PHOTOS)
@pytest.mark.parametrize("method", available_methods())
def test_untouched_photo_not_flagged(base_images, name, method):
    r = run_analysis(base_images[name], AnalysisConfig(method=method))
    assert r.detection.status == STATUS_CLEAN
    assert r.localization.n_regions == 0
    assert r.mask_original_size().max() == 0


@pytest.mark.parametrize("name", CLEAN_PHOTOS)
def test_untouched_photo_not_flagged_at_high_sensitivity(base_images, name):
    r = run_analysis(base_images[name], AnalysisConfig.from_sensitivity("High"))
    assert r.detection.status != STATUS_MANIPULATED


@pytest.mark.parametrize("kind", ["translate", "rotate", "scale", "rotate_scale", "jpeg"])
@pytest.mark.parametrize("name", ["astronaut", "camera"])
def test_copy_move_detected_sift(forged_cases, name, kind):
    forged, _ = forged_cases[(name, kind)]
    r = run_analysis(forged, AnalysisConfig(method="SIFT"))
    assert r.detection.status == STATUS_MANIPULATED, r.detection.notes
    assert r.detection.forgery_type == "Copy-move"
    assert r.detection.n_valid_groups >= 1
    assert 0 < r.detection.score <= 100


@pytest.mark.parametrize("kind", ["translate", "rotate", "jpeg"])
def test_copy_move_detected_orb(forged_cases, kind):
    forged, _ = forged_cases[("astronaut", kind)]
    r = run_analysis(forged, AnalysisConfig(method="ORB"))
    assert r.detection.status == STATUS_MANIPULATED


def test_low_quality_recompressed_copy(forged_cases):
    """Case 6: copy + heavy JPEG (q=60) on top of the q=75 forgery."""
    forged, _ = forged_cases[("astronaut", "translate")]
    r = run_analysis(jpeg_roundtrip(forged, 60), AnalysisConfig())
    assert r.detection.status == STATUS_MANIPULATED


def test_low_quality_clean_photo_not_flagged(base_images):
    r = run_analysis(jpeg_roundtrip(base_images["coffee"], 50), AnalysisConfig())
    assert r.detection.status == STATUS_CLEAN


@pytest.mark.parametrize("method", available_methods())
def test_tiled_pattern_is_not_called_manipulated(tiles, method):
    r = run_analysis(tiles, AnalysisConfig(method=method))
    assert r.detection.status in (STATUS_INCONCLUSIVE, STATUS_CLEAN)
    if r.detection.status == STATUS_INCONCLUSIVE:
        assert r.detection.repetitive_pattern


def test_natural_brick_wall_not_called_manipulated(bricks):
    r = run_analysis(bricks, AnalysisConfig())
    assert r.detection.status != STATUS_MANIPULATED


def test_score_formula_is_documented(forged_cases):
    r = run_analysis(forged_cases[("astronaut", "translate")][0], AnalysisConfig())
    assert "min_inliers" in r.detection.score_explanation
    expected = min(100.0, 100 * r.detection.n_verified_matches / (3 * 8))
    assert r.detection.score == pytest.approx(expected, abs=0.1)


def test_summary_has_required_fields(forged_cases):
    s = run_analysis(forged_cases[("camera", "rotate")][0]).summary()
    for k in ("status", "suspected_forgery_type", "verified_matches", "suspicious_regions",
              "method", "detection_score", "processing_time_s"):
        assert k in s
