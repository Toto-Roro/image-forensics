"""
Step 7: localization of the suspicious regions.

Two levels of localization are produced and always labelled as such:

1. Keypoint-region estimate (always available)
   The verified inlier points of each match group are clustered
   (single-linkage hierarchical clustering). Each cluster becomes a region by
   filling its convex hull, stamping a disc on every keypoint and applying
   morphological closing + dilation. This shows WHERE the evidence is, but
   its boundary is only as good as the keypoint coverage.

2. Transform-verified refinement (when enabled and it succeeds)
   Using the RANSAC transform M (region A -> region B) we warp the image
   onto itself and compare it pixel by pixel (cf. Pan & Lyu, 2010). Pixels
   whose neighbourhood matches the warped copy closely are part of the
   duplicated content. The search is restricted to a band around the
   keypoint region, cleaned with opening/closing, and only connected
   components that actually contain verified keypoints are kept. This gives
   a much tighter, pixel-level boundary - but it is still an estimate, not a
   ground-truth segmentation.

Note: we cannot know which of the two regions is the original and which is
the pasted copy - both are reported as a pair (A, B).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.cluster.hierarchy import fclusterdata

from .config import AnalysisConfig
from .forgery_detection import hull_mask
from .geometric_verification import MatchGroup
from .preprocessing import PreprocessResult


@dataclass
class Region:
    region_id: str            # e.g. "A1" / "B1"
    role: str                 # "A" (one side of the copy pair) or "B" (other side)
    group_id: int
    bbox: tuple               # (x, y, w, h) at working resolution
    area_px: int
    contour: np.ndarray
    method: str


@dataclass
class LocalizationResult:
    mask: np.ndarray                     # uint8 0/255, working resolution
    mask_a: np.ndarray
    mask_b: np.ndarray
    keypoint_mask: np.ndarray            # level-1 estimate only
    regions: list = field(default_factory=list)
    method: str = "none"
    notes: list = field(default_factory=list)

    @property
    def n_regions(self) -> int:
        return len(self.regions)

    @property
    def coverage(self) -> float:
        return float((self.mask > 0).mean())


def empty_localization(shape: tuple) -> LocalizationResult:
    z = np.zeros(shape[:2], np.uint8)
    return LocalizationResult(z, z.copy(), z.copy(), z.copy(), [], "none",
                              ["No valid match group - nothing to localize."])


def cluster_points(pts: np.ndarray, eps: float, min_size: int = 3) -> list[np.ndarray]:
    """Single-linkage clustering: points closer than eps end up together."""
    if len(pts) < min_size:
        return []
    if len(pts) == 1:
        return [pts]
    labels = fclusterdata(pts.astype(np.float64), t=eps, criterion="distance",
                          method="single")
    clusters = [pts[labels == lab] for lab in np.unique(labels)]
    return [c for c in clusters if len(c) >= min_size]


def keypoint_region_mask(clusters: list[np.ndarray], shape: tuple,
                         radius: int) -> np.ndarray:
    """Hull + per-keypoint discs, then morphological closing and dilation."""
    mask = np.zeros(shape[:2], np.uint8)
    for c in clusters:
        mask |= hull_mask(c, shape)
        for x, y in c.astype(int):
            cv2.circle(mask, (int(x), int(y)), radius, 255, -1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return mask


def _similarity_map(gray: np.ndarray, M: np.ndarray, pts_at: np.ndarray):
    """Local mean |I - warp(I, M)| and an adaptive threshold.

    warp(I, M) places the content of region A at region B, so low values
    in the map mark pixels of B that are copies of A.
    """
    h, w = gray.shape
    g = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0)
    warped = cv2.warpAffine(g, M, (w, h), flags=cv2.INTER_LINEAR)
    inside = cv2.warpAffine(np.ones_like(g), M, (w, h), flags=cv2.INTER_NEAREST) > 0.5
    diff = cv2.blur(np.abs(g - warped), (7, 7))
    # Threshold learned from the image itself: at verified keypoints we KNOW
    # the content is copied, so the typical difference there tells us how
    # much compression / interpolation noise to tolerate.
    xs = np.clip(pts_at[:, 0].astype(int), 0, w - 1)
    ys = np.clip(pts_at[:, 1].astype(int), 0, h - 1)
    ref = float(np.median(diff[ys, xs])) if len(xs) else 5.0
    thr = float(np.clip(2.5 * ref + 4.0, 6.0, 30.0))
    return diff, inside, thr


def refine_side(gray: np.ndarray, M: np.ndarray, pts: np.ndarray,
                kp_mask: np.ndarray, radius: int) -> np.ndarray | None:
    """Pixel-level region on the side that M maps INTO."""
    diff, inside, thr = _similarity_map(gray, M, pts)
    roi = cv2.dilate(kp_mask, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (6 * radius + 1,) * 2)) > 0
    cand = ((diff < thr) & inside & roi).astype(np.uint8) * 255
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    # Keep only connected components that contain verified keypoints.
    n, labels = cv2.connectedComponents(cand)
    h, w = gray.shape
    xs = np.clip(pts[:, 0].astype(int), 0, w - 1)
    ys = np.clip(pts[:, 1].astype(int), 0, h - 1)
    hit = np.bincount(labels[ys, xs], minlength=n)
    keep = [lab for lab in range(1, n) if hit[lab] >= 2]
    if not keep:
        return None
    out = np.isin(labels, keep).astype(np.uint8) * 255
    # Fill interior holes (flat areas inside the copied object).
    contours, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, 255, -1)
    return out


def extract_regions(mask: np.ndarray, role: str, group_id: int, method: str,
                    min_area: int, start: int = 1) -> list[Region]:
    """Connected-component analysis -> contours and bounding boxes."""
    regions = []
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    idx = start
    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = (labels == lab).astype(np.uint8) * 255
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        regions.append(Region(f"{role}{idx}", role, group_id,
                              tuple(int(v) for v in cv2.boundingRect(contour)),
                              area, contour, method))
        idx += 1
    return regions


def localize(pre: PreprocessResult, groups: list[MatchGroup],
             cfg: AnalysisConfig) -> LocalizationResult:
    shape = pre.gray.shape
    if not groups:
        return empty_localization(shape)

    h, w = shape
    diag = math.hypot(h, w)
    eps = max(25.0, cfg.cluster_dist * diag)
    radius = int(max(6, 0.012 * diag))
    min_area = int(max(30, 0.0004 * h * w))

    mask_a = np.zeros(shape, np.uint8)
    mask_b = np.zeros(shape, np.uint8)
    kp_total = np.zeros(shape, np.uint8)
    regions: list[Region] = []
    refined_any, kp_only_any = False, False
    notes = []
    counters = {"A": 1, "B": 1}

    for g in groups:
        M_ab = g.transform
        M_ba = cv2.invertAffineTransform(M_ab)
        for role, pts, M_into in (("A", g.src_pts, M_ba), ("B", g.dst_pts, M_ab)):
            clusters = cluster_points(pts, eps)
            if not clusters:
                continue
            core = np.vstack(clusters)
            kp_mask = keypoint_region_mask(clusters, shape, radius)
            kp_total |= kp_mask
            side, method = kp_mask, "keypoint-region estimate"
            if cfg.refine_localization:
                refined = refine_side(pre.gray, M_into, core, kp_mask, radius)
                if refined is not None:
                    side, method = refined, "transform-verified (pixel-level)"
                    refined_any = True
                else:
                    kp_only_any = True
            else:
                kp_only_any = True
            new = extract_regions(side, role, g.group_id, method, min_area,
                                  counters[role])
            counters[role] += len(new)
            regions.extend(new)
            if role == "A":
                mask_a |= side
            else:
                mask_b |= side

    mask = mask_a | mask_b
    if refined_any and kp_only_any:
        method = "mixed (pixel-level where verified, keypoint estimate elsewhere)"
    elif refined_any:
        method = "transform-verified (pixel-level)"
    elif kp_only_any:
        method = "keypoint-region estimate"
    else:
        method = "none"
    if method.startswith("keypoint"):
        notes.append("Boundaries are estimated from keypoint clusters; they show "
                     "where evidence lies, not an exact segmentation.")
    if refined_any:
        notes.append("Pixel-level boundaries come from comparing the image with "
                     "itself warped by the estimated transform.")
    notes.append("Region A and B are the two sides of a duplicated pair; which "
                 "one is the original cannot be determined from the image alone.")
    return LocalizationResult(mask, mask_a, mask_b, kp_total, regions, method, notes)


def mask_to_original_size(mask: np.ndarray, original_shape: tuple) -> np.ndarray:
    h, w = original_shape[:2]
    if mask.shape[:2] == (h, w):
        return mask
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
