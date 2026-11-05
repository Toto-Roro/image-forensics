"""
Step 4: matching an image's descriptors *against themselves*.

In copy-move forgery the copied region lives in the same image, so every
descriptor is compared with all other descriptors of the same image.

Generalised 2-Nearest-Neighbour test (g2NN, Amerini et al., 2011)
-----------------------------------------------------------------
Lowe's ratio test accepts the best match only if d1 / d2 < T. If a region
was pasted several times, the 2nd neighbour is *also* a genuine copy, so the
plain test wrongly rejects it. g2NN walks the sorted neighbour distances
d1 <= d2 <= ... and accepts neighbour i while d_i / d_{i+1} < T. The first
large jump in distance separates true copies from unrelated descriptors.

Spatial filter: matches between keypoints that are physically close are
dropped -- they come from the same structure (or duplicate keypoints with
different orientation), not from a copy elsewhere in the image.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .feature_extraction import FeatureResult


@dataclass
class MatchResult:
    pairs: np.ndarray          # (M, 2) int: indices (i, j) into the keypoints, i < j
    src_pts: np.ndarray        # (M, 2) coords of keypoint i
    dst_pts: np.ndarray        # (M, 2) coords of keypoint j
    distances: np.ndarray      # (M,) descriptor distance
    n_candidates: int          # neighbour pairs examined (before ratio test)
    n_after_ratio: int         # surviving the g2NN ratio test
    n_after_spatial: int       # surviving the spatial-distance filter
    matcher: str
    time_s: float

    @property
    def count(self) -> int:
        return len(self.pairs)


def _empty(matcher: str, t: float = 0.0) -> MatchResult:
    z = np.zeros((0, 2), np.float32)
    return MatchResult(np.zeros((0, 2), int), z, z.copy(), np.zeros(0, np.float32),
                       0, 0, 0, matcher, t)


def _knn_self(desc: np.ndarray, method: str, k: int):
    """k-NN of each descriptor among all descriptors (including itself)."""
    if method == "ORB":
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        name = "BFMatcher (Hamming)"
    elif len(desc) > 3000:
        # FLANN KD-tree is much faster than brute force for many float descriptors.
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
        name = "FLANN (KD-tree, L2)"
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        name = "BFMatcher (L2)"
    desc = desc if method == "ORB" else desc.astype(np.float32)
    return matcher.knnMatch(desc, desc, k=k), name


def match_self(features: FeatureResult, ratio_thresh: float = 0.6,
               min_spatial_dist: float = 20.0, max_neighbours: int = 8,
               max_hamming: int = 80) -> MatchResult:
    """Find descriptor pairs that likely correspond to duplicated content.

    min_spatial_dist is in pixels (callers convert from a diagonal fraction).
    """
    t0 = time.perf_counter()
    method = features.method
    desc = features.descriptors
    if desc is None or len(desc) < 3:
        return _empty("n/a", time.perf_counter() - t0)

    k = int(min(max_neighbours + 1, len(desc)))
    knn, matcher_name = _knn_self(desc, method, k)

    coords = features.coords
    pair_set: dict[tuple[int, int], float] = {}
    n_candidates = 0
    n_after_ratio = 0

    for qi, neighbours in enumerate(knn):
        # Drop the trivial self-match (distance 0 to itself).
        nb = [m for m in neighbours if m.trainIdx != qi]
        n_candidates += len(nb)
        # g2NN: accept neighbour i while d_i / d_{i+1} < T.
        for a, b in zip(nb, nb[1:]):
            if b.distance <= 0:
                break
            if a.distance / b.distance >= ratio_thresh:
                break
            if method == "ORB" and a.distance > max_hamming:
                break
            n_after_ratio += 1
            j = a.trainIdx
            # Spatial filter: copies must be physically separated.
            if np.hypot(*(coords[qi] - coords[j])) < min_spatial_dist:
                continue
            key = (min(qi, j), max(qi, j))   # de-duplicate (i,j) / (j,i)
            if key not in pair_set or a.distance < pair_set[key]:
                pair_set[key] = float(a.distance)

    if not pair_set:
        r = _empty(matcher_name, time.perf_counter() - t0)
        r.n_candidates, r.n_after_ratio = n_candidates, n_after_ratio
        return r

    pairs = np.array(list(pair_set.keys()), dtype=int)
    dists = np.array(list(pair_set.values()), dtype=np.float32)
    return MatchResult(pairs=pairs, src_pts=coords[pairs[:, 0]].copy(),
                       dst_pts=coords[pairs[:, 1]].copy(), distances=dists,
                       n_candidates=n_candidates, n_after_ratio=n_after_ratio,
                       n_after_spatial=len(pairs), matcher=matcher_name,
                       time_s=time.perf_counter() - t0)
