"""
Step 5: RANSAC geometric verification of the candidate matches.

A genuine copy-move maps a whole region onto another with ONE geometric
transform (translation, possibly with rotation and uniform scaling). Random
false matches do not agree on any single transform. So we:

1. Fit a 4-DoF similarity transform  [s*cos(t) -s*sin(t) tx; s*sin(t) s*cos(t) ty]
   with RANSAC (cv2.estimateAffinePartial2D). RANSAC repeatedly samples 2
   matches, builds a candidate transform, counts how many other matches it
   explains within `ransac_thresh` pixels, and keeps the best one.
2. Remove those inliers and repeat, which finds multiple independent copies
   ("match groups").

Direction ambiguity: a match (p, q) does not tell us which point is the
original. For a copy A->B some pairs arrive as (a, b) and others as (b, a).
We therefore feed RANSAC both orientations; a pair is a member of the group
if either orientation fits, and it is then re-oriented consistently.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import AnalysisConfig
from .feature_matching import MatchResult


@dataclass
class MatchGroup:
    group_id: int
    transform: np.ndarray       # 2x3 similarity matrix mapping region A -> region B
    src_pts: np.ndarray         # (n, 2) inlier points in region A
    dst_pts: np.ndarray         # (n, 2) corresponding points in region B
    pair_indices: np.ndarray    # rows of MatchResult.pairs belonging to this group
    rotation_deg: float
    scale: float
    translation: tuple
    mean_error_px: float
    valid: bool
    reason: str
    dense_area_px: int = 0          # textured pixels confirmed by dense check
    dense_ratio: float = 0.0        # confirmed / textured pixels near keypoints
    pattern_like: bool = False      # set by the detector (repetitive texture)

    @property
    def n_inliers(self) -> int:
        return len(self.src_pts)

    def summary(self) -> dict:
        return {
            "group": self.group_id,
            "inliers": self.n_inliers,
            "rotation_deg": round(self.rotation_deg, 1),
            "scale": round(self.scale, 3),
            "shift_px": round(float(np.hypot(*self.translation)), 1),
            "mean_error_px": round(self.mean_error_px, 2),
            "dense_support_px": self.dense_area_px,
            "dense_agreement_pct": round(100 * self.dense_ratio, 1),
            "valid": self.valid,
            "note": self.reason,
        }


@dataclass
class VerificationResult:
    groups: list = field(default_factory=list)       # every group found
    n_input_matches: int = 0

    @property
    def valid_groups(self) -> list:
        return [g for g in self.groups if g.valid]

    @property
    def weak_groups(self) -> list:
        return [g for g in self.groups if not g.valid]

    @property
    def n_verified(self) -> int:
        return int(sum(g.n_inliers for g in self.valid_groups))


def decompose_similarity(M: np.ndarray) -> tuple[float, float, tuple]:
    """Return (rotation degrees, scale, translation) of a 2x3 similarity."""
    a, b = M[0, 0], M[1, 0]
    scale = float(math.hypot(a, b))
    rotation = float(math.degrees(math.atan2(b, a)))
    return rotation, scale, (float(M[0, 2]), float(M[1, 2]))


def _local_ncc(a: np.ndarray, b: np.ndarray, k: int = 9):
    """Zero-mean normalised cross-correlation of a and b in k x k windows.

    Returns (ncc, std_a, std_b). NCC is insensitive to brightness/contrast
    changes and JPEG noise, but drops sharply when the content differs.
    """
    mu_a, mu_b = cv2.blur(a, (k, k)), cv2.blur(b, (k, k))
    var_a = np.maximum(cv2.blur(a * a, (k, k)) - mu_a ** 2, 0)
    var_b = np.maximum(cv2.blur(b * b, (k, k)) - mu_b ** 2, 0)
    cov = cv2.blur(a * b, (k, k)) - mu_a * mu_b
    sa, sb = np.sqrt(var_a), np.sqrt(var_b)
    return cov / (sa * sb + 1e-6), sa, sb


def _side_support(g: np.ndarray, M: np.ndarray, pts: np.ndarray,
                  radius: int, ncc_thresh: float, min_std: float) -> tuple[int, float]:
    """Agreement between the image and its warped copy around `pts`.

    ROI = discs of `radius` around the keypoints of this side. Inside the ROI
    we look only at textured pixels (local std > min_std in both images) and
    count those whose local NCC exceeds ncc_thresh.
    Returns (agreeing pixel count, agreeing / textured ratio).
    """
    h, w = g.shape
    warped = cv2.warpAffine(g, M, (w, h), flags=cv2.INTER_LINEAR)
    inside = cv2.warpAffine(np.ones_like(g), M, (w, h), flags=cv2.INTER_NEAREST) > 0.5
    ncc, sa, sb = _local_ncc(g, warped)
    roi = np.zeros((h, w), np.uint8)
    for x, y in pts.astype(int):
        cv2.circle(roi, (int(x), int(y)), radius, 1, -1)
    textured = (roi > 0) & inside & (sa > min_std) & (sb > min_std)
    agree = textured & (ncc > ncc_thresh)
    n_tex = int(textured.sum())
    n_agree = int(agree.sum())
    return n_agree, (n_agree / n_tex if n_tex else 0.0)


def dense_support(gray: np.ndarray, group: "MatchGroup",
                  ncc_thresh: float = 0.8, min_std: float = 6.0) -> tuple[int, float]:
    """Dense (pixel-wise) confirmation of a candidate transform.

    Warp the image with the group's transform and measure, around the matched
    keypoints, how many textured pixels have a high local normalised
    cross-correlation with the warped copy. A wrong transform agrees on only a
    few scattered pixels; a real copy agrees over the whole copied area. Flat
    pixels (sky, walls) are ignored because they would agree with anything.
    Returns (min area over both sides, min agreement ratio over both sides).
    """
    g = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0)
    radius = int(max(12, 0.05 * math.hypot(*g.shape)))
    M = group.transform
    area_b, ratio_b = _side_support(g, M, group.dst_pts, radius, ncc_thresh, min_std)
    area_a, ratio_a = _side_support(g, cv2.invertAffineTransform(M), group.src_pts,
                                    radius, ncc_thresh, min_std)
    return min(area_a, area_b), min(ratio_a, ratio_b)


def verify_matches(matches: MatchResult, gray: np.ndarray,
                   cfg: AnalysisConfig) -> VerificationResult:
    """Iterative multi-group RANSAC followed by dense confirmation."""
    result = VerificationResult(n_input_matches=matches.count)
    if matches.count < 3:
        return result

    cv2.setRNGSeed(42)  # deterministic RANSAC for reproducible results
    image_shape = gray.shape
    h, w = image_shape[:2]
    min_shift = cfg.min_spatial_dist * math.hypot(h, w)

    P = matches.src_pts.astype(np.float32)
    Q = matches.dst_pts.astype(np.float32)
    remaining = np.arange(matches.count)

    for gid in range(1, cfg.max_groups * 2 + 1):
        if len(remaining) < 3 or len(result.valid_groups) >= cfg.max_groups:
            break
        p, q = P[remaining], Q[remaining]
        n = len(remaining)
        # Both orientations of every pair go into RANSAC.
        src = np.vstack([p, q])
        dst = np.vstack([q, p])
        M, inl = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=cfg.ransac_thresh,
            maxIters=4000, confidence=0.995, refineIters=10)
        if M is None or inl is None:
            break
        inl = inl.ravel().astype(bool)
        fwd, bwd = inl[:n], inl[n:]
        member = fwd | bwd
        if member.sum() < 3:
            break

        # Orient every member so that it maps A -> B under M.
        a_pts = np.where(fwd[:, None], p, q)[member]
        b_pts = np.where(fwd[:, None], q, p)[member]

        # Least-squares refit on the oriented inliers (no outliers left).
        M_ref, _ = cv2.estimateAffinePartial2D(a_pts, b_pts, method=cv2.LMEDS)
        if M_ref is not None:
            M = M_ref
        proj = a_pts @ M[:, :2].T + M[:, 2]
        err = float(np.mean(np.linalg.norm(proj - b_pts, axis=1)))

        rot, scale, t = decompose_similarity(M)
        centroid_shift = float(np.linalg.norm(b_pts.mean(0) - a_pts.mean(0)))

        group = MatchGroup(
            group_id=gid, transform=M.astype(np.float64), src_pts=a_pts,
            dst_pts=b_pts, pair_indices=remaining[member], rotation_deg=rot,
            scale=scale, translation=t, mean_error_px=err, valid=False, reason="")

        # Validity checks on the estimated transform.
        n_in = int(member.sum())
        min_dense = int(cfg.dense_min_area_frac * h * w)
        if not (cfg.min_scale <= scale <= cfg.max_scale):
            group.reason = f"implausible scale {scale:.2f}"
        elif centroid_shift < min_shift:
            group.reason = "regions too close (near-identity transform)"
        elif n_in >= cfg.min_inliers:
            group.valid, group.reason = True, "geometrically consistent copy"
            group.dense_area_px, group.dense_ratio = dense_support(gray, group)
        elif n_in >= cfg.min_inliers_dense:
            # Few keypoint matches: accept only if the dense pixel-wise check
            # confirms that a sizeable textured area really is duplicated.
            group.dense_area_px, group.dense_ratio = dense_support(gray, group)
            if (group.dense_area_px >= min_dense
                    and group.dense_ratio >= cfg.dense_min_ratio):
                group.valid = True
                group.reason = (f"{n_in} inliers confirmed by dense check "
                                f"({group.dense_area_px} px, "
                                f"{100 * group.dense_ratio:.0f}% agreement)")
            else:
                group.reason = (f"only {n_in} inliers; dense check failed "
                                f"({group.dense_area_px} px, "
                                f"{100 * group.dense_ratio:.0f}% agreement)")
        else:
            group.reason = f"only {n_in} inliers (< {cfg.min_inliers})"
        result.groups.append(group)
        remaining = remaining[~member]
        if n_in < 3:
            break

    # Re-number valid groups first so "Group 1" is the strongest evidence.
    result.groups.sort(key=lambda g: (not g.valid, -g.n_inliers))
    for i, g in enumerate(result.groups, 1):
        g.group_id = i
    return result
