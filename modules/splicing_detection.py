"""
EXPERIMENTAL secondary module: splicing (cut-and-paste from another image).

A spliced region comes from a different camera / exposure / compression
history, so its low-level statistics often differ from the rest of the
image. We divide the image into blocks and measure, per block:

  1. Noise level     - robust std (MAD) of the high-pass residual
                       (image - median-filtered image).
  2. ELA             - Error Level Analysis: mean |I - JPEG90(I)|. Regions
                       with a different compression history re-compress
                       differently.
  3. Edge sharpness  - mean gradient magnitude on Canny edges. Pasted
                       objects often have unnaturally sharp or soft edges.
  4. Texture / local statistics - local variance of the residual.

Each feature map is converted to a robust z-score (median / MAD over all
blocks). Blocks that deviate strongly in several features are flagged.

IMPORTANT: these are forensic *indicators*. Natural content (sky vs. foliage,
saturated highlights, depth-of-field blur) also changes these statistics, so
false positives are expected. This module never declares an image spliced.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

STATUS_INDICATORS = "Inconsistency indicators present (experimental)"
STATUS_NO_INDICATORS = "No strong inconsistency indicators (experimental)"


@dataclass
class SplicingResult:
    status: str
    block_size: int
    feature_maps: dict          # name -> block-level float map
    zscore_maps: dict           # name -> block-level robust z map
    combined_map: np.ndarray    # pixel-level 0..1 heatmap (working size)
    suspicious_mask: np.ndarray  # uint8 0/255, pixel level
    suspicious_fraction: float
    largest_cluster_blocks: int
    notes: list = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "status": self.status,
            "block_size": self.block_size,
            "suspicious_area_pct": round(100 * self.suspicious_fraction, 2),
            "largest_suspicious_cluster_blocks": self.largest_cluster_blocks,
        }


def _robust_z(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    vals = x[valid]
    if vals.size < 4:
        return np.zeros_like(x)
    med = np.median(vals)
    mad = 1.4826 * np.median(np.abs(vals - med)) + 1e-6
    z = (x - med) / mad
    z[~valid] = 0
    return z


def _block_reduce(img: np.ndarray, bs: int, fn) -> np.ndarray:
    h, w = img.shape[:2]
    by, bx = h // bs, w // bs
    out = np.zeros((by, bx), np.float32)
    for i in range(by):
        for j in range(bx):
            out[i, j] = fn(img[i * bs:(i + 1) * bs, j * bs:(j + 1) * bs])
    return out


def ela_image(img_bgr: np.ndarray, quality: int = 90) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    rec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return np.abs(img_bgr.astype(np.float32) - rec.astype(np.float32)).mean(axis=2)


def analyze_splicing(img_bgr: np.ndarray, block_size: int = 32,
                     z_thresh: float = 3.5) -> SplicingResult:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    bs = int(max(16, min(block_size, min(h, w) // 4)))
    g = gray.astype(np.float32)

    residual = g - cv2.medianBlur(gray, 3).astype(np.float32)
    ela = ela_image(img_bgr)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    grad = np.hypot(gx, gy)
    edges = cv2.Canny(gray, 60, 150) > 0

    def mad_std(b):
        return 1.4826 * np.median(np.abs(b - np.median(b)))

    maps = {
        "noise_level": _block_reduce(residual, bs, mad_std),
        "ela": _block_reduce(ela, bs, np.mean),
        "edge_sharpness": _block_reduce(np.where(edges, grad, np.nan), bs,
                                        lambda b: np.nanmean(b) if np.isfinite(b).any() else 0.0),
        "texture_variance": _block_reduce(residual, bs, np.var),
    }
    content_std = _block_reduce(g, bs, np.std)
    edge_frac = _block_reduce(edges.astype(np.float32), bs, np.mean)
    mean_int = _block_reduce(g, bs, np.mean)

    # Flat or clipped blocks give meaningless noise / ELA values: exclude them.
    valid = (content_std > 3) & (mean_int > 8) & (mean_int < 247)
    edge_valid = valid & (edge_frac > 0.02)

    zmaps = {k: _robust_z(v, edge_valid if k == "edge_sharpness" else valid)
             for k, v in maps.items()}
    stacked = np.stack([np.abs(z) for z in zmaps.values()])
    # A block is suspicious if at least two features are strongly abnormal.
    n_abnormal = (stacked > z_thresh).sum(0)
    block_flag = (n_abnormal >= 2) & valid
    combined = np.clip(np.sort(stacked, axis=0)[-2] / (2 * z_thresh), 0, 1)

    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(
        block_flag.astype(np.uint8), connectivity=8)
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if n_lab > 1 else 0

    by, bx = block_flag.shape
    combined_px = np.zeros((h, w), np.float32)
    combined_px[:by * bs, :bx * bs] = cv2.resize(combined, (bx * bs, by * bs),
                                                 interpolation=cv2.INTER_NEAREST)
    combined_px = cv2.GaussianBlur(combined_px, (0, 0), bs / 3)
    mask = np.zeros((h, w), np.uint8)
    mask[:by * bs, :bx * bs] = cv2.resize(block_flag.astype(np.uint8) * 255,
                                          (bx * bs, by * bs),
                                          interpolation=cv2.INTER_NEAREST)
    frac = float(block_flag.sum() / max(1, valid.sum()))

    notes = ["Experimental: indicators only - natural content changes also "
             "trigger these features.",
             f"{int(valid.sum())} of {by * bx} blocks usable (flat or clipped "
             "blocks excluded)."]
    status = STATUS_INDICATORS if largest >= 2 else STATUS_NO_INDICATORS
    if status == STATUS_INDICATORS:
        notes.append(f"A cluster of {largest} adjacent blocks deviates in at "
                     "least two independent statistics.")
    return SplicingResult(status, bs, maps, zmaps, combined_px, mask, frac,
                          largest, notes)
