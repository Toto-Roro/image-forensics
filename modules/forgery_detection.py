"""
Step 6: turn verified match groups into an image-level verdict.

Verdicts
--------
* "Potentially manipulated"  - at least one valid group and no sign that the
                                matches come from a naturally repeating pattern.
* "Inconclusive"             - evidence exists but is weak, or the matches
                                look like a repeating texture (tiles, bricks,
                                windows), which genuinely duplicates content.
* "No manipulation detected" - no geometrically consistent duplicated region.

Detection score (heuristic, NOT a probability)
----------------------------------------------
    score = 100 * min(1, V / (3 * min_inliers))
where V is the total number of RANSAC-verified matches in valid groups. A
single group with exactly the minimum number of inliers scores ~33; three
times that amount saturates at 100. It measures the *amount* of geometric
evidence, it is not calibrated against any dataset.

Repetitive-pattern guard
------------------------
Three independent signs of a naturally periodic texture:
1. Self-overlap: a texture shifted by one period overlaps itself, so the
   source and destination hulls of a group overlap. A pasted object is put
   somewhere else, so its hulls are (mostly) disjoint.
2. Lattice: in a tiled pattern, the pure-translation shifts of all groups
   are integer combinations of one or two basis vectors (the tile size).
3. Too many independent valid groups at once.
Groups with sign 1 are marked `pattern_like` and are excluded from the
copy-move evidence and from localization.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import AnalysisConfig
from .geometric_verification import VerificationResult

STATUS_MANIPULATED = "Potentially manipulated"
STATUS_CLEAN = "No manipulation detected"
STATUS_INCONCLUSIVE = "Inconclusive"


@dataclass
class DetectionResult:
    status: str
    forgery_type: str
    score: float
    score_explanation: str
    n_valid_groups: int
    n_weak_groups: int
    n_verified_matches: int
    repetitive_pattern: bool
    notes: list = field(default_factory=list)


def hull_mask(points: np.ndarray, shape: tuple) -> np.ndarray:
    """Filled convex hull of points (or empty mask if < 3 points)."""
    mask = np.zeros(shape[:2], np.uint8)
    if len(points) >= 3:
        hull = cv2.convexHull(points.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
    return mask


def hull_overlap_iou(src: np.ndarray, dst: np.ndarray, shape: tuple) -> float:
    a = hull_mask(src, shape) > 0
    b = hull_mask(dst, shape) > 0
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def lattice_fit_count(groups, tol_px: float = 6.0) -> int:
    """How many pure-translation groups lie on a common 2-D lattice.

    Basis b1 = shortest shift, b2 = shortest shift not parallel to b1.
    A shift T fits if T = a*b1 + c*b2 with a, c (close to) integers.
    """
    shifts = [np.array(g.translation) for g in groups
              if abs(g.rotation_deg) < 2 and abs(g.scale - 1) < 0.02]
    shifts = [v for v in shifts if np.linalg.norm(v) > 1]
    if len(shifts) < 3:
        return 0
    shifts.sort(key=np.linalg.norm)
    b1 = shifts[0]
    b2 = next((v for v in shifts[1:]
               if abs(b1[0] * v[1] - b1[1] * v[0]) / (np.linalg.norm(b1) * np.linalg.norm(v)) > 0.35),
              None)
    fits = 0
    for v in shifts:
        if b2 is not None:
            coef = np.linalg.solve(np.column_stack([b1, b2]), v)
        else:
            coef = np.array([v @ b1 / (b1 @ b1), 0.0])
        rounded = np.round(coef)
        recon = rounded[0] * b1 + (rounded[1] * b2 if b2 is not None else 0)
        if np.any(rounded != 0) and np.linalg.norm(recon - v) < tol_px:
            fits += 1
    return fits


def decide(verification: VerificationResult, image_shape: tuple,
           cfg: AnalysisConfig) -> DetectionResult:
    valid = verification.valid_groups
    weak = verification.weak_groups
    V = verification.n_verified
    score = round(100.0 * min(1.0, V / (3.0 * cfg.min_inliers)), 1)
    explanation = (f"100 x min(1, verified matches / (3 x min_inliers)) = "
                   f"100 x min(1, {V} / {3 * cfg.min_inliers}) = {score}")
    notes = []

    overlaps = []
    for g in valid:
        o = hull_overlap_iou(g.src_pts, g.dst_pts, image_shape)
        g.pattern_like = o > cfg.repetitive_overlap_iou
        overlaps.append(o)
    overlapping = [o for o in overlaps if o > cfg.repetitive_overlap_iou]
    copy_like = [g for g in valid if not g.pattern_like]
    too_many = len(valid) > cfg.max_valid_groups_before_repetitive
    n_lattice = lattice_fit_count(verification.groups, tol_px=2 * cfg.ransac_thresh)
    lattice = n_lattice >= 4
    repetitive = bool(valid) and (too_many or lattice or not copy_like)

    if not valid:
        strongest_weak = max((g.n_inliers for g in weak), default=0)
        if strongest_weak >= max(4, cfg.min_inliers // 2):
            status = STATUS_INCONCLUSIVE
            ftype = "Weak duplicated-region evidence"
            notes.append(f"Best match group has {strongest_weak} consistent matches, "
                         f"below the {cfg.min_inliers} required.")
        else:
            status, ftype = STATUS_CLEAN, "None"
            notes.append("No geometrically consistent duplicated region was found.")
    elif repetitive:
        status = STATUS_INCONCLUSIVE
        ftype = "Possible repetitive pattern"
        if too_many:
            notes.append(f"{len(valid)} independent duplicate groups - typical of "
                         "periodic textures (tiles, bricks, windows).")
        if lattice:
            notes.append(f"{n_lattice} duplicate shifts are integer multiples of a "
                         "common tile size - a periodic lattice.")
        if overlapping:
            notes.append("Matched regions overlap their own copy (hull IoU "
                         f"{max(overlapping):.2f}) - typical of a texture shifted "
                         "by one period, not a pasted object.")
    else:
        status, ftype = STATUS_MANIPULATED, "Copy-move"
        for g in copy_like:
            kind = []
            if abs(g.rotation_deg) > 3:
                kind.append(f"rotated {g.rotation_deg:.0f} deg")
            if abs(g.scale - 1) > 0.05:
                kind.append(f"scaled x{g.scale:.2f}")
            notes.append(f"Group {g.group_id}: {g.n_inliers} verified matches, "
                         f"{', '.join(kind) or 'pure translation'}.")
        if overlapping:
            notes.append(f"{len(overlapping)} group(s) overlap their own copy and "
                         "were treated as natural repetition (excluded).")

    return DetectionResult(status=status, forgery_type=ftype, score=score,
                           score_explanation=explanation,
                           n_valid_groups=len(valid), n_weak_groups=len(weak),
                           n_verified_matches=V, repetitive_pattern=repetitive,
                           notes=notes)
