"""
Central configuration: paths, validation limits and analysis parameters.

All paths are resolved relative to the project root, so the project can be
moved anywhere without editing code.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DATASET_DIR = PROJECT_ROOT / "dataset"
SYNTHETIC_DIR = DATASET_DIR / "synthetic"
SAMPLES_DIR = PROJECT_ROOT / "samples"

# ---- Upload validation limits -------------------------------------------
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE_MB = 20
MIN_IMAGE_DIM = 64          # px, smaller images carry too few features
MAX_IMAGE_DIM = 12000       # px, guards against decompression bombs

# ---- Sensitivity presets ---------------------------------------------------
# ratio_thresh : g2NN ratio threshold (higher = more matches accepted)
# min_inliers  : RANSAC inliers needed for a match group to count as evidence
# ransac_thresh: max reprojection error (px) for a match to be a RANSAC inlier
SENSITIVITY_PRESETS = {
    "Low": {"ratio_thresh": 0.5, "min_inliers": 12, "ransac_thresh": 3.0},
    "Medium": {"ratio_thresh": 0.6, "min_inliers": 8, "ransac_thresh": 4.0},
    "High": {"ratio_thresh": 0.7, "min_inliers": 6, "ransac_thresh": 5.0},
}


@dataclass
class AnalysisConfig:
    """Every tunable parameter of the pipeline in one place."""

    method: str = "SIFT"                 # "ORB" or "SIFT"
    mode: str = "Copy-Move"              # "Copy-Move" or "Splicing (Experimental)"
    sensitivity: str = "Medium"

    # Preprocessing
    max_dim: int = 1024                  # longest side of the working image
    denoise: str = "gaussian"            # "none", "gaussian", "bilateral"
    use_clahe: bool = True               # contrast-limited histogram equalisation

    # Feature extraction
    n_features: int = 6000

    # Matching
    ratio_thresh: float = 0.6
    min_spatial_dist: float = 0.04       # fraction of image diagonal
    max_neighbours: int = 8              # k for g2NN search
    max_hamming: int = 80                # ORB only: absolute descriptor cap

    # Geometric verification
    ransac_thresh: float = 4.0
    min_inliers: int = 8
    max_groups: int = 6
    min_inliers_dense: int = 4           # fewer inliers accepted if dense check passes
    dense_min_area_frac: float = 0.003   # min confirmed area, fraction of image
    dense_min_ratio: float = 0.35        # min share of textured ROI pixels agreeing
    min_scale: float = 0.33
    max_scale: float = 3.0

    # Detection
    max_valid_groups_before_repetitive: int = 4
    repetitive_overlap_iou: float = 0.25

    # Localization
    refine_localization: bool = True
    cluster_dist: float = 0.08           # fraction of diagonal for clustering

    @classmethod
    def from_sensitivity(cls, sensitivity: str = "Medium", **overrides) -> "AnalysisConfig":
        """Build a config from a named preset, then apply explicit overrides."""
        if sensitivity not in SENSITIVITY_PRESETS:
            raise ValueError(f"Unknown sensitivity '{sensitivity}'. "
                             f"Choose from {list(SENSITIVITY_PRESETS)}")
        params = dict(SENSITIVITY_PRESETS[sensitivity])
        params.update(overrides)
        return cls(sensitivity=sensitivity, **params)

    def to_dict(self) -> dict:
        return asdict(self)
