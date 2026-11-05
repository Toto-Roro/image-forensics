"""
Orchestrates the full copy-move pipeline:

  preprocess -> extract features -> self-match (g2NN) -> RANSAC groups
             -> verdict -> localization

`run_analysis` is the single entry point used by the Streamlit app, the
report generator, the dataset evaluator and the tests.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .config import AnalysisConfig
from .feature_extraction import FeatureResult, extract_features
from .feature_matching import MatchResult, match_self
from .forgery_detection import STATUS_MANIPULATED, DetectionResult, decide
from .geometric_verification import VerificationResult, verify_matches
from .localization import LocalizationResult, localize, mask_to_original_size
from .preprocessing import ImageInfo, PreprocessResult, preprocess


@dataclass
class AnalysisResult:
    config: AnalysisConfig
    info: ImageInfo | None
    pre: PreprocessResult
    features: FeatureResult
    matches: MatchResult
    verification: VerificationResult
    detection: DetectionResult
    localization: LocalizationResult
    timings: dict = field(default_factory=dict)
    splicing: object | None = None      # SplicingResult when that mode is run

    @property
    def total_time_s(self) -> float:
        return sum(self.timings.values())

    def mask_original_size(self) -> np.ndarray:
        return mask_to_original_size(self.localization.mask, self.pre.original_shape)

    def summary(self) -> dict:
        d, f, m = self.detection, self.features, self.matches
        return {
            "status": d.status,
            "suspected_forgery_type": d.forgery_type,
            "detection_score": d.score,
            "detection_score_formula": d.score_explanation,
            "method": f.method,
            "matcher": m.matcher,
            "keypoints": f.count,
            "candidate_matches": m.n_after_ratio,
            "matches_after_spatial_filter": m.n_after_spatial,
            "verified_matches": d.n_verified_matches,
            "valid_match_groups": d.n_valid_groups,
            "weak_match_groups": d.n_weak_groups,
            "suspicious_regions": self.localization.n_regions,
            "localization_method": self.localization.method,
            "mask_coverage_pct": round(100 * self.localization.coverage, 2),
            "repetitive_pattern_flag": d.repetitive_pattern,
            "processing_time_s": round(self.total_time_s, 3),
        }


def run_analysis(img_bgr: np.ndarray, cfg: AnalysisConfig | None = None,
                 info: ImageInfo | None = None) -> AnalysisResult:
    cfg = cfg or AnalysisConfig()
    timings = {}

    t = time.perf_counter()
    pre = preprocess(img_bgr, cfg)
    timings["preprocessing"] = time.perf_counter() - t

    t = time.perf_counter()
    feats = extract_features(pre.processed, cfg.method, cfg.n_features)
    timings["feature_extraction"] = time.perf_counter() - t

    t = time.perf_counter()
    h, w = pre.gray.shape
    min_dist_px = cfg.min_spatial_dist * math.hypot(h, w)
    matches = match_self(feats, cfg.ratio_thresh, min_dist_px,
                         cfg.max_neighbours, cfg.max_hamming)
    timings["feature_matching"] = time.perf_counter() - t

    t = time.perf_counter()
    ver = verify_matches(matches, pre.gray, cfg)
    timings["geometric_verification"] = time.perf_counter() - t

    t = time.perf_counter()
    det = decide(ver, pre.gray.shape, cfg)
    timings["decision"] = time.perf_counter() - t

    t = time.perf_counter()
    # Localize the copy-move evidence. If the image was judged manipulated,
    # groups that look like natural repetition are left out; otherwise all
    # valid groups are shown so the user can inspect them.
    groups = ver.valid_groups
    if det.status == STATUS_MANIPULATED:
        groups = [g for g in groups if not g.pattern_like]
    loc = localize(pre, groups, cfg)
    timings["localization"] = time.perf_counter() - t

    return AnalysisResult(cfg, info, pre, feats, matches, ver, det, loc, timings)
