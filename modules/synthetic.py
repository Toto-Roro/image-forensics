"""
Synthetic copy-move generator with exact ground-truth masks.

Lets the project be demonstrated and evaluated immediately, without
downloading a research dataset. Base photographs come from scikit-image's
bundled sample images (no network access needed).

Manipulations produced:
  translate     - region copied and moved
  rotate        - copied region rotated 20-45 degrees
  scale         - copied region resized (x0.8 or x1.2-1.3)
  rotate_scale  - both
  jpeg          - translated copy, then the whole image JPEG-compressed (q=75)

The ground-truth mask marks BOTH the source region and the pasted copy,
because a copy-move detector finds the duplicated pair.

CLI:
  python -m modules.synthetic                 # dataset/synthetic + samples/
  python -m modules.synthetic --out my_folder --seed 3
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from .config import SAMPLES_DIR, SYNTHETIC_DIR

KINDS = ("translate", "rotate", "scale", "rotate_scale", "jpeg")
BASE_NAMES = ("astronaut", "coffee", "chelsea", "rocket", "camera")


def load_base_images(names=BASE_NAMES) -> dict[str, np.ndarray]:
    """Bundled scikit-image photos as BGR uint8."""
    from skimage import data
    out = {}
    for n in names:
        img = getattr(data, n)()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2BGR)
        out[n] = np.ascontiguousarray(img)
    return out


def _texture_map(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    return cv2.blur(np.hypot(gx, gy), (31, 31))


def _ellipse_mask(shape, center, axes, angle=0.0) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    cv2.ellipse(m, (int(center[0]), int(center[1])), (int(axes[0]), int(axes[1])),
                angle, 0, 360, 255, -1)
    return m


def jpeg_roundtrip(img: np.ndarray, quality: int = 75) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def make_copy_move(img: np.ndarray, kind: str = "translate", seed: int = 0,
                   region_frac: float = 0.11) -> tuple[np.ndarray, np.ndarray, dict]:
    """Copy a textured elliptical region elsewhere in the image.

    Returns (forged_bgr, gt_mask uint8 0/255, metadata).
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    r = region_frac * min(h, w)
    axes = (r * rng.uniform(1.0, 1.35), r * rng.uniform(0.75, 1.0))
    ell_angle = float(rng.uniform(0, 180))

    angle, scale = 0.0, 1.0
    if kind in ("rotate", "rotate_scale"):
        angle = float(rng.uniform(20, 45) * rng.choice([-1, 1]))
    if kind in ("scale", "rotate_scale"):
        scale = float(rng.choice([rng.uniform(0.78, 0.85), rng.uniform(1.2, 1.3)]))

    reach = max(axes) * max(1.0, scale) + 4      # radius of the pasted copy
    margin_src = max(axes) + 4
    tex = _texture_map(img)

    # Pick the source centre among the most textured candidate positions.
    ys, xs = np.mgrid[int(margin_src):int(h - margin_src):8,
                      int(margin_src):int(w - margin_src):8]
    cand = np.stack([xs.ravel(), ys.ravel()], 1)
    vals = tex[cand[:, 1], cand[:, 0]]
    top = cand[np.argsort(vals)[-max(20, len(vals) // 5):]]

    for _ in range(400):
        cs = top[rng.integers(len(top))].astype(float)
        cd = np.array([rng.uniform(reach, w - reach), rng.uniform(reach, h - reach)])
        # Copy must not overlap the source region.
        if np.linalg.norm(cd - cs) > (max(axes) + reach) * 1.15:
            break
    else:
        raise RuntimeError("Could not place a non-overlapping copy; image too small.")

    src_mask = _ellipse_mask(img.shape, cs, axes, ell_angle)
    # Transform: rotate/scale about the source centre, then move to cd.
    M = cv2.getRotationMatrix2D((float(cs[0]), float(cs[1])), angle, scale)
    M[:, 2] += cd - cs
    warped = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)
    alpha = cv2.warpAffine(src_mask.astype(np.float32) / 255.0, M, (w, h))
    alpha_soft = cv2.GaussianBlur(alpha, (5, 5), 0)[..., None]   # feathered seam
    forged = (warped * alpha_soft + img * (1 - alpha_soft)).round().astype(np.uint8)

    dst_mask = (alpha > 0.5).astype(np.uint8) * 255
    gt = cv2.bitwise_or(src_mask, dst_mask)

    if kind == "jpeg":
        forged = jpeg_roundtrip(forged, 75)

    meta = {"kind": kind, "src_center": [round(float(v), 1) for v in cs],
            "dst_center": [round(float(v), 1) for v in cd],
            "rotation_deg": round(angle, 1), "scale": round(scale, 3),
            "region_axes": [round(float(a), 1) for a in axes]}
    return forged, gt, meta


def make_repetitive_pattern(size: int = 512, tile: int = 64, seed: int = 0) -> np.ndarray:
    """A tiled texture (wallpaper / tiles): genuine repetition, NOT a forgery."""
    rng = np.random.default_rng(seed)
    base = load_base_images(("coffee",))["coffee"]
    y = int(rng.integers(100, base.shape[0] - tile - 20))
    x = int(rng.integers(150, base.shape[1] - tile - 20))
    motif = base[y:y + tile, x:x + tile]
    reps = size // tile + 1
    img = np.tile(motif, (reps, reps, 1))[:size, :size].astype(np.float32)
    # Smooth lighting gradient + sensor noise, as in a real photo of tiles.
    gy, gx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    img *= (0.8 + 0.3 * gx + 0.1 * gy)[..., None]
    img += rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def natural_repetitive_image() -> np.ndarray:
    """scikit-image's brick wall photo - a natural repeated pattern."""
    from skimage import data
    return cv2.cvtColor(data.brick(), cv2.COLOR_GRAY2BGR)


def corrupted_jpeg_bytes(seed: int = 0) -> bytes:
    """Bytes that start like a JPEG but are garbage."""
    rng = np.random.default_rng(seed)
    return b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + rng.integers(0, 256, 4000, dtype=np.uint8).tobytes()


def _write(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise IOError(f"Could not write {path}")


def generate_suite(out_dir: Path = SYNTHETIC_DIR, seed: int = 7,
                   kinds=KINDS, names=BASE_NAMES) -> list[dict]:
    """Write original/, forged/ and masks/ plus manifest.csv to out_dir."""
    out_dir = Path(out_dir)
    rows = []
    for bi, (name, img) in enumerate(load_base_images(names).items()):
        _write(out_dir / "original" / f"{name}.png", img)
        for ki, kind in enumerate(kinds):
            forged, gt, meta = make_copy_move(img, kind, seed=seed * 1000 + bi * 10 + ki)
            ext = ".jpg" if kind == "jpeg" else ".png"
            fname = f"{name}_{kind}{ext}"
            if ext == ".jpg":
                # Already JPEG-compressed in memory; store losslessly as the
                # final JPEG bytes so the file equals what we analysed.
                ok, buf = cv2.imencode(".jpg", forged, [cv2.IMWRITE_JPEG_QUALITY, 95])
                (out_dir / "forged").mkdir(parents=True, exist_ok=True)
                (out_dir / "forged" / fname).write_bytes(buf.tobytes())
            else:
                _write(out_dir / "forged" / fname, forged)
            _write(out_dir / "masks" / f"{name}_{kind}_mask.png", gt)
            rows.append({"file": fname, "base": name, **{k: str(v) for k, v in meta.items()}})
    with open(out_dir / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def generate_demo_samples(samples_dir: Path = SAMPLES_DIR, seed: int = 11) -> list[Path]:
    """A small, labelled set of files for the live demo (one per test case)."""
    samples_dir = Path(samples_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)
    imgs = load_base_images(("coffee", "astronaut", "chelsea", "rocket"))
    written = []

    def save(name, img):
        p = samples_dir / name
        _write(p, img)
        written.append(p)

    save("01_original_untouched.png", imgs["coffee"])
    f, m, _ = make_copy_move(imgs["astronaut"], "translate", seed)
    save("02_copied_object_translate.png", f)
    save("02_copied_object_translate_mask.png", m)
    f, m, _ = make_copy_move(imgs["rocket"], "rotate", seed + 1)
    save("03_rotated_copy.png", f)
    save("03_rotated_copy_mask.png", m)
    f, m, _ = make_copy_move(imgs["chelsea"], "scale", seed + 2)
    save("04_resized_copy.png", f)
    save("04_resized_copy_mask.png", m)
    f, m, _ = make_copy_move(imgs["coffee"], "jpeg", seed + 3)
    p = samples_dir / "05_compressed_copy_q75.jpg"
    ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 95])
    p.write_bytes(buf.tobytes())
    written.append(p)
    save("05_compressed_copy_q75_mask.png", m)
    save("06_repeated_pattern_tiles.png", make_repetitive_pattern(seed=seed))
    save("07_natural_pattern_bricks.png", natural_repetitive_image())
    p = samples_dir / "08_corrupted_file.jpg"
    p.write_bytes(corrupted_jpeg_bytes(seed))
    written.append(p)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic copy-move test data.")
    ap.add_argument("--out", type=Path, default=SYNTHETIC_DIR,
                    help="output folder for original/forged/masks")
    ap.add_argument("--samples", type=Path, default=SAMPLES_DIR,
                    help="output folder for demo samples")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rows = generate_suite(args.out, args.seed)
    demo = generate_demo_samples(args.samples)
    print(f"Wrote {len(rows)} forged images + masks to {args.out}")
    print(f"Wrote {len(demo)} demo files to {args.samples}")


if __name__ == "__main__":
    main()
