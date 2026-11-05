"""
Step 3: keypoint detection and descriptor extraction (ORB and SIFT).

ORB  = FAST corners + oriented BRIEF binary descriptors (256 bits).
       Fast, rotation invariant, partially scale invariant (image pyramid).
SIFT = Difference-of-Gaussians blobs + 128-D gradient-histogram descriptors.
       Slower but more robust to rotation, scaling and JPEG compression.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FeatureResult:
    method: str
    keypoints: list            # list[cv2.KeyPoint]
    descriptors: np.ndarray | None
    coords: np.ndarray         # (N, 2) float32, x/y of every keypoint
    sizes: np.ndarray          # (N,) keypoint diameters
    time_s: float

    @property
    def count(self) -> int:
        return len(self.keypoints)


def sift_available() -> bool:
    """SIFT is in mainline OpenCV since 4.4; older builds lack it."""
    return hasattr(cv2, "SIFT_create")


def available_methods() -> list[str]:
    return ["ORB", "SIFT"] if sift_available() else ["ORB"]


def _make_detector(method: str, n_features: int):
    method = method.upper()
    if method == "ORB":
        # Low FAST threshold + small edge threshold -> many keypoints, which
        # copy-move detection needs because each copied region is small.
        return cv2.ORB_create(nfeatures=n_features, scaleFactor=1.2, nlevels=8,
                              edgeThreshold=15, patchSize=31, fastThreshold=7)
    if method == "SIFT":
        if not sift_available():
            raise RuntimeError("SIFT is not available in this OpenCV build; use ORB.")
        # Lower contrast threshold than default (0.04) keeps weaker blobs.
        return cv2.SIFT_create(nfeatures=n_features, contrastThreshold=0.02)
    raise ValueError(f"Unknown feature method '{method}'. Use ORB or SIFT.")


def extract_features(gray: np.ndarray, method: str = "ORB",
                     n_features: int = 6000) -> FeatureResult:
    """Detect keypoints and compute descriptors on a grayscale image."""
    if gray.ndim != 2:
        raise ValueError("extract_features expects a single-channel image.")
    detector = _make_detector(method, n_features)
    t0 = time.perf_counter()
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    elapsed = time.perf_counter() - t0

    keypoints = list(keypoints or [])
    if keypoints:
        coords = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        sizes = np.array([kp.size for kp in keypoints], dtype=np.float32)
    else:
        coords = np.zeros((0, 2), np.float32)
        sizes = np.zeros((0,), np.float32)
        descriptors = None
    return FeatureResult(method=method.upper(), keypoints=keypoints,
                         descriptors=descriptors, coords=coords, sizes=sizes,
                         time_s=elapsed)


def compare_methods(gray: np.ndarray, n_features: int = 6000) -> list[dict]:
    """Run every available detector once and report count + timing."""
    rows = []
    for m in available_methods():
        fr = extract_features(gray, m, n_features)
        desc_dim = 0 if fr.descriptors is None else fr.descriptors.shape[1]
        rows.append({
            "Method": m,
            "Keypoints": fr.count,
            "Descriptor": f"{desc_dim} {'bytes (binary)' if m == 'ORB' else 'floats'}",
            "Distance": "Hamming" if m == "ORB" else "Euclidean (L2)",
            "Time (ms)": round(fr.time_s * 1000, 1),
        })
    return rows
