"""Shared fixtures: synthetic test images generated in memory (no downloads)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.config import AnalysisConfig  # noqa: E402
from modules.synthetic import (load_base_images, make_copy_move,  # noqa: E402
                               make_repetitive_pattern, natural_repetitive_image)


@pytest.fixture(scope="session")
def base_images():
    return load_base_images()


@pytest.fixture(scope="session")
def astronaut(base_images):
    return base_images["astronaut"]


@pytest.fixture(scope="session")
def forged_cases(base_images):
    """(kind -> (forged, gt_mask)) on two different photos."""
    cases = {}
    for name in ("astronaut", "camera"):
        for i, kind in enumerate(("translate", "rotate", "scale", "rotate_scale", "jpeg")):
            cases[(name, kind)] = make_copy_move(base_images[name], kind, seed=100 + i)[:2]
    return cases


@pytest.fixture(scope="session")
def tiles():
    return make_repetitive_pattern(seed=1)


@pytest.fixture(scope="session")
def bricks():
    return natural_repetitive_image()


@pytest.fixture
def cfg():
    return AnalysisConfig()


def mask_iou(a, b):
    a, b = a > 0, b > 0
    u = (a | b).sum()
    return (a & b).sum() / u if u else 1.0


@pytest.fixture
def iou():
    return mask_iou


@pytest.fixture
def png_bytes():
    import cv2

    def _enc(img, ext=".png"):
        ok, buf = cv2.imencode(ext, img)
        assert ok
        return buf.tobytes()
    return _enc


@pytest.fixture
def rng():
    return np.random.default_rng(0)
