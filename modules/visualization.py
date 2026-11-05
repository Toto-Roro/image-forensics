"""
Step 8: visual outputs. Every function returns an RGB uint8 image so the
same picture can be shown in Streamlit, embedded in reports and saved.
"""
from __future__ import annotations

import io
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image

# Colours (RGB)
RED = (230, 57, 70)
BLUE = (37, 99, 235)
CYAN = (6, 182, 212)
AMBER = (245, 158, 11)
GREEN = (34, 197, 94)
GROUP_COLOURS = [(37, 99, 235), (230, 57, 70), (16, 185, 129), (245, 158, 11),
                 (139, 92, 246), (236, 72, 153), (20, 184, 166), (234, 88, 12)]


def _thickness(img: np.ndarray) -> int:
    return max(1, round(max(img.shape[:2]) / 500))


def gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def draw_keypoints(rgb: np.ndarray, keypoints, max_points: int = 4000) -> np.ndarray:
    """Rich keypoints: circle size = scale, line = orientation."""
    out = (rgb * 0.6).astype(np.uint8)
    kps = sorted(keypoints, key=lambda k: -k.response)[:max_points]
    return cv2.drawKeypoints(out, kps, None, color=(0, 255, 140),
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


def draw_match_lines(rgb: np.ndarray, src: np.ndarray, dst: np.ndarray,
                     color=AMBER, dim: float = 0.55, max_lines: int = 1500) -> np.ndarray:
    """Copy-move matches live in one image, so lines are drawn on the image itself."""
    out = (rgb * dim).astype(np.uint8)
    t = _thickness(rgb)
    for (x1, y1), (x2, y2) in list(zip(src, dst))[:max_lines]:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.line(out, p1, p2, color, t, cv2.LINE_AA)
        cv2.circle(out, p1, 2 * t + 1, color, -1, cv2.LINE_AA)
        cv2.circle(out, p2, 2 * t + 1, color, -1, cv2.LINE_AA)
    return out


def draw_verified_groups(rgb: np.ndarray, groups) -> np.ndarray:
    """RANSAC inliers, one colour per group; A ends hollow, B ends filled.
    Groups judged to be natural repetition are drawn in grey."""
    out = (rgb * 0.55).astype(np.uint8)
    t = _thickness(rgb)
    for i, g in enumerate(groups):
        c = (150, 150, 150) if g.pattern_like else GROUP_COLOURS[i % len(GROUP_COLOURS)]
        for (x1, y1), (x2, y2) in zip(g.src_pts, g.dst_pts):
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.line(out, p1, p2, c, t, cv2.LINE_AA)
            cv2.circle(out, p1, 3 * t + 1, c, t, cv2.LINE_AA)
            cv2.circle(out, p2, 3 * t + 1, c, -1, cv2.LINE_AA)
        cx, cy = g.dst_pts.mean(0).astype(int)
        _label(out, f"G{g.group_id}" + (" pattern" if g.pattern_like else ""),
               (int(cx), int(cy)), c)
    return out


def _label(img, text, org, color):
    scale = max(0.45, max(img.shape[:2]) / 1400)
    t = _thickness(img)
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, t)
    x, y = org
    x = int(np.clip(x, 0, img.shape[1] - tw - 6))
    y = int(np.clip(y, th + 6, img.shape[0] - 2))
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + base - 2), color, -1)
    cv2.putText(img, text, (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), t, cv2.LINE_AA)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color=RED, alpha: float = 0.45) -> np.ndarray:
    """Semi-transparent red fill + solid outline of the suspicious regions."""
    out = rgb.copy()
    m = mask > 0
    if m.any():
        tint = np.zeros_like(rgb)
        tint[:] = color
        out[m] = (rgb[m] * (1 - alpha) + tint[m] * alpha).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, _thickness(rgb) + 1, cv2.LINE_AA)
    return out


def draw_regions(rgb: np.ndarray, regions, groups=()) -> np.ndarray:
    """Bounding boxes: cyan = side A, red = side B, arrow A -> B per group."""
    out = rgb.copy()
    t = _thickness(rgb) + 1
    for r in regions:
        c = CYAN if r.role == "A" else RED
        x, y, w, h = r.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), c, t)
        _label(out, f"{r.region_id} (G{r.group_id})", (x, y), c)
    for g in groups:
        a = tuple(int(v) for v in g.src_pts.mean(0))
        b = tuple(int(v) for v in g.dst_pts.mean(0))
        cv2.arrowedLine(out, a, b, (255, 255, 255), t, cv2.LINE_AA, tipLength=0.04)
    return out


def heatmap_overlay(rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """0..1 heatmap rendered with the inferno colour map on top of the image."""
    h8 = (np.clip(heat, 0, 1) * 255).astype(np.uint8)
    colored = cv2.cvtColor(cv2.applyColorMap(h8, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    a = (np.clip(heat, 0, 1) * alpha)[..., None]
    return (rgb * (1 - a) + colored * a).astype(np.uint8)


def block_map_image(zmap: np.ndarray, size: tuple) -> np.ndarray:
    """Block-level |z| map scaled to the image size, as a colour image."""
    v = np.clip(np.abs(zmap) / 6.0, 0, 1)
    h8 = cv2.resize((v * 255).astype(np.uint8), (size[1], size[0]),
                    interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(cv2.applyColorMap(h8, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def build_copy_move_visuals(result) -> "OrderedDict[str, np.ndarray]":
    """All nine required visual outputs, in pipeline order."""
    pre, ver, loc = result.pre, result.verification, result.localization
    groups_shown = [g for g in ver.valid_groups
                    if g.group_id in {r.group_id for r in loc.regions}] or ver.valid_groups
    v = OrderedDict()
    v["1. Original image"] = pre.rgb
    v["2. Grayscale image"] = gray_to_rgb(pre.gray)
    v["3. Preprocessed image"] = gray_to_rgb(pre.processed)
    v[f"4. {result.features.method} keypoints ({result.features.count})"] = \
        draw_keypoints(pre.rgb, result.features.keypoints)
    v[f"5. Feature matches after ratio + spatial filter ({result.matches.count})"] = \
        draw_match_lines(pre.rgb, result.matches.src_pts, result.matches.dst_pts)
    v[f"6. RANSAC-verified matches ({ver.n_verified})"] = \
        draw_verified_groups(pre.rgb, ver.valid_groups)
    v["7. Binary tampering mask"] = mask_to_rgb(loc.mask)
    v["8. Suspicious regions highlighted"] = overlay_mask(pre.rgb, loc.mask)
    v["9. Bounding boxes (A = cyan, B = red)"] = draw_regions(pre.rgb, loc.regions,
                                                              groups_shown)
    return v


def build_splicing_visuals(rgb: np.ndarray, sres) -> "OrderedDict[str, np.ndarray]":
    v = OrderedDict()
    v["Original image"] = rgb
    v["Combined inconsistency heatmap"] = heatmap_overlay(rgb, sres.combined_map)
    v["Flagged blocks"] = overlay_mask(rgb, sres.suspicious_mask, AMBER, 0.4)
    titles = {"noise_level": "Noise level |z|", "ela": "Error level analysis |z|",
              "edge_sharpness": "Edge sharpness |z|", "texture_variance": "Residual variance |z|"}
    for k, z in sres.zscore_maps.items():
        v[titles.get(k, k)] = block_map_image(z, rgb.shape[:2])
    return v


def to_png_bytes(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()
