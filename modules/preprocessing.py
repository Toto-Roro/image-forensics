"""
Step 1 + 2: upload validation and image preprocessing.

Validation rejects anything that is not a readable JPG/PNG of sensible size.
Preprocessing produces the grayscale "working" image used for feature
extraction: resize (aspect preserved) -> grayscale -> denoise -> CLAHE.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from .config import (ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB, MAX_IMAGE_DIM,
                     MIN_IMAGE_DIM, AnalysisConfig)


class ImageValidationError(ValueError):
    """Raised when an uploaded file cannot be analysed."""


@dataclass
class ImageInfo:
    filename: str
    width: int
    height: int
    channels: int
    file_size_kb: float
    format: str


@dataclass
class PreprocessResult:
    """All intermediate images. Everything is at *working* resolution."""
    original_bgr: np.ndarray         # resized colour image (BGR)
    rgb: np.ndarray                  # same, RGB order for display
    gray: np.ndarray                 # plain grayscale
    processed: np.ndarray            # grayscale after denoise + CLAHE
    scale: float                     # working size / original size
    original_shape: tuple            # (h, w) of the uploaded image
    steps: list = field(default_factory=list)


def validate_image_bytes(data: bytes, filename: str) -> tuple[np.ndarray, ImageInfo]:
    """Validate raw upload bytes and decode them into a BGR image.

    Checks, in order: extension, empty file, size limit, decodability
    (Pillow verify + OpenCV decode), and dimension limits.
    """
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ImageValidationError(
            f"Unsupported file type '{ext or 'none'}'. Allowed: JPG, JPEG, PNG.")
    if not data:
        raise ImageValidationError("The file is empty.")
    size_mb = len(data) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ImageValidationError(
            f"File is {size_mb:.1f} MB; the limit is {MAX_FILE_SIZE_MB} MB.")

    # Pillow's verify() catches truncated / structurally broken files.
    try:
        with Image.open(io.BytesIO(data)) as pil:
            fmt = pil.format or ext.strip(".").upper()
            pil.verify()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ImageValidationError(f"The file is corrupt or not an image ({exc}).") from exc

    if fmt not in {"JPEG", "PNG", "MPO"}:
        raise ImageValidationError(f"File content is {fmt}, not JPG/PNG.")

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ImageValidationError("OpenCV could not decode the image data.")

    img = to_bgr(img)
    h, w = img.shape[:2]
    if min(h, w) < MIN_IMAGE_DIM:
        raise ImageValidationError(
            f"Image is {w}x{h}px; at least {MIN_IMAGE_DIM}px per side is needed.")
    if max(h, w) > MAX_IMAGE_DIM:
        raise ImageValidationError(
            f"Image is {w}x{h}px; the maximum is {MAX_IMAGE_DIM}px per side.")

    info = ImageInfo(filename=Path(filename).name, width=w, height=h,
                     channels=3, file_size_kb=round(len(data) / 1024, 1),
                     format="JPEG" if fmt == "MPO" else fmt)
    return img, info


def load_image(path: str | Path) -> tuple[np.ndarray, ImageInfo]:
    """Load and validate an image from disk."""
    path = Path(path)
    if not path.exists():
        raise ImageValidationError(f"File not found: {path}")
    return validate_image_bytes(path.read_bytes(), path.name)


def to_bgr(img: np.ndarray) -> np.ndarray:
    """Normalise grayscale / BGRA / 16-bit images to 8-bit, 3-channel BGR."""
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def resize_keep_aspect(img: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    """Downscale so the longest side is <= max_dim. Never upscales."""
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / float(max(h, w)))
    if scale < 1.0:
        img = cv2.resize(img, (round(w * scale), round(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img, scale


def preprocess(img_bgr: np.ndarray, cfg: AnalysisConfig | None = None) -> PreprocessResult:
    """Run the preprocessing chain and keep every intermediate for display."""
    cfg = cfg or AnalysisConfig()
    steps = []
    original_shape = img_bgr.shape[:2]

    work, scale = resize_keep_aspect(img_bgr, cfg.max_dim)
    if scale < 1.0:
        steps.append(f"Resized by {scale:.3f} to {work.shape[1]}x{work.shape[0]}")

    rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    steps.append("Converted BGR -> grayscale")

    processed = gray
    # Light smoothing suppresses sensor / JPEG noise that would otherwise
    # create unstable keypoints. Kernel kept small so real texture survives.
    if cfg.denoise == "gaussian":
        processed = cv2.GaussianBlur(processed, (3, 3), 0)
        steps.append("Gaussian blur 3x3 (noise reduction)")
    elif cfg.denoise == "bilateral":
        processed = cv2.bilateralFilter(processed, 5, 30, 5)
        steps.append("Bilateral filter (edge-preserving noise reduction)")

    # CLAHE boosts local contrast so that low-contrast regions still yield
    # keypoints; clip limit keeps noise amplification bounded.
    if cfg.use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed = clahe.apply(processed)
        steps.append("CLAHE contrast enhancement (clip 2.0, 8x8 tiles)")

    return PreprocessResult(original_bgr=work, rgb=rgb, gray=gray,
                            processed=processed, scale=scale,
                            original_shape=original_shape, steps=steps)
