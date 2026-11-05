"""Upload validation and preprocessing (test case 8: invalid / corrupt files)."""
import numpy as np
import pytest

from modules.config import AnalysisConfig
from modules.preprocessing import (ImageValidationError, preprocess,
                                   resize_keep_aspect, to_bgr, validate_image_bytes)
from modules.synthetic import corrupted_jpeg_bytes


def test_valid_png_and_jpeg(astronaut, png_bytes):
    for ext in (".png", ".jpg"):
        img, info = validate_image_bytes(png_bytes(astronaut, ext), f"a{ext}")
        assert img.shape == astronaut.shape
        assert (info.width, info.height) == (512, 512)
        assert info.format in {"PNG", "JPEG"}


def test_rejects_wrong_extension(astronaut, png_bytes):
    with pytest.raises(ImageValidationError, match="Unsupported"):
        validate_image_bytes(png_bytes(astronaut), "image.gif")


def test_rejects_empty_file():
    with pytest.raises(ImageValidationError, match="empty"):
        validate_image_bytes(b"", "x.png")


def test_rejects_corrupted_file():
    with pytest.raises(ImageValidationError, match="corrupt"):
        validate_image_bytes(corrupted_jpeg_bytes(), "broken.jpg")


def test_rejects_text_renamed_as_png():
    with pytest.raises(ImageValidationError):
        validate_image_bytes(b"hello, this is not an image" * 50, "fake.png")


def test_rejects_truncated_png(astronaut, png_bytes):
    data = png_bytes(astronaut)
    with pytest.raises(ImageValidationError):
        validate_image_bytes(data[: len(data) // 3], "cut.png")


def test_rejects_tiny_image(png_bytes):
    with pytest.raises(ImageValidationError, match="at least"):
        validate_image_bytes(png_bytes(np.zeros((30, 30, 3), np.uint8)), "tiny.png")


def test_rejects_oversized_file(monkeypatch, astronaut, png_bytes):
    monkeypatch.setattr("modules.preprocessing.MAX_FILE_SIZE_MB", 0.01)
    with pytest.raises(ImageValidationError, match="limit"):
        validate_image_bytes(png_bytes(astronaut), "big.png")


def test_grayscale_and_rgba_are_normalised():
    assert to_bgr(np.zeros((70, 80), np.uint8)).shape == (70, 80, 3)
    assert to_bgr(np.zeros((70, 80, 4), np.uint8)).shape == (70, 80, 3)
    assert to_bgr(np.zeros((70, 80), np.uint16)).dtype == np.uint8


def test_resize_preserves_aspect_and_never_upscales():
    img = np.zeros((1000, 2000, 3), np.uint8)
    out, s = resize_keep_aspect(img, 1024)
    assert out.shape[:2] == (512, 1024) and s == pytest.approx(0.512)
    small, s2 = resize_keep_aspect(np.zeros((100, 200, 3), np.uint8), 1024)
    assert small.shape[:2] == (100, 200) and s2 == 1.0


def test_preprocess_outputs(astronaut):
    pre = preprocess(astronaut, AnalysisConfig())
    assert pre.gray.ndim == 2 and pre.processed.shape == pre.gray.shape
    assert pre.rgb[..., 0].sum() == astronaut[..., 2].sum()      # BGR -> RGB
    assert any("CLAHE" in s for s in pre.steps)
    pre2 = preprocess(astronaut, AnalysisConfig(denoise="none", use_clahe=False))
    assert np.array_equal(pre2.processed, pre2.gray)
