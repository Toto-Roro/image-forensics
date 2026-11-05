"""
Dataset evaluation: image-level detection metrics and pixel-level
localization metrics against ground-truth masks.

Expected layout (works for CoMoFoD, CASIA v2.0, DEFACTO after sorting files):

    <dataset>/
        original/   authentic images              (label 0)
        forged/     tampered images                (label 1)
        masks/      ground-truth masks for forged/ (white = tampered)

Mask lookup for forged/<name>.<ext> tries, in order:
    masks/<name>.*, <name>_mask.*, <name>_gt.*, <name>_GT.*,
    and the CoMoFoD convention  001_F.png / 001_F_JC3.png -> 001_B.png

Image level   : Accuracy, Precision, Recall, F1, confusion matrix.
                Positive = "Potentially manipulated". "Inconclusive" counts as
                negative by default (configurable).
Pixel level   : Precision, Recall, F1, IoU between predicted and GT masks,
                computed on forged images that have a mask. A missed image
                contributes an empty prediction (all metrics 0).

CLI:
    python -m modules.evaluation --dataset dataset/synthetic
    python -m modules.evaluation --dataset dataset --method ORB --limit 50
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import RESULTS_DIR, AnalysisConfig
from .forgery_detection import STATUS_INCONCLUSIVE, STATUS_MANIPULATED
from .pipeline import run_analysis

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def image_level_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, bool)
    y_pred = np.asarray(y_pred, bool)
    tp = int((y_true & y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "n_images": int(len(y_true)),
        "accuracy": safe_div(tp + tn, len(y_true)),
        "precision": precision,
        "recall": recall,
        "f1": safe_div(2 * precision * recall, precision + recall),
        "confusion_matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
    }


def pixel_metrics(gt_mask: np.ndarray, pred_mask: np.ndarray) -> dict:
    """Pixel precision / recall / F1 / IoU. Masks are resized to GT size.

    Edge cases never divide by zero: if both masks are empty the prediction
    agrees perfectly (all 1.0); if exactly one is empty, the scores are 0.
    """
    gt = gt_mask > 127 if gt_mask.max() > 1 else gt_mask > 0
    if pred_mask.shape[:2] != gt.shape[:2]:
        pred_mask = cv2.resize(pred_mask, (gt.shape[1], gt.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
    pred = pred_mask > 0
    tp = int((gt & pred).sum())
    fp = int((~gt & pred).sum())
    fn = int((gt & ~pred).sum())
    if tp + fp + fn == 0:
        return {"pixel_precision": 1.0, "pixel_recall": 1.0, "pixel_f1": 1.0, "iou": 1.0}
    p = safe_div(tp, tp + fp)
    r = safe_div(tp, tp + fn)
    return {"pixel_precision": p, "pixel_recall": r,
            "pixel_f1": safe_div(2 * p * r, p + r), "iou": safe_div(tp, tp + fp + fn)}


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def find_mask(forged: Path, mask_dir: Path) -> Path | None:
    if not mask_dir.is_dir():
        return None
    stem = forged.stem
    candidates = [stem, f"{stem}_mask", f"{stem}_gt", f"{stem}_GT"]
    m = re.match(r"^(.*)_F(_[A-Za-z]+\d*)?$", stem)
    if m:                                          # CoMoFoD: 001_F, 001_F_JC3 -> 001_B
        candidates.append(m.group(1) + "_B")
    by_stem = {}
    for p in mask_dir.iterdir():
        if p.suffix.lower() in IMAGE_EXTS:
            by_stem.setdefault(p.stem, p)
    for c in candidates:
        if c in by_stem:
            return by_stem[c]
    return None


def read_image(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


@dataclass
class EvaluationReport:
    per_image: pd.DataFrame
    image_metrics: dict
    pixel_metrics_mean: dict
    pixel_metrics_detected_only: dict
    config: dict
    dataset: str
    inconclusive_as_positive: bool
    errors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"dataset": self.dataset, "config": self.config,
                "inconclusive_as_positive": self.inconclusive_as_positive,
                "image_level": self.image_metrics,
                "pixel_level_mean_over_forged_with_masks": self.pixel_metrics_mean,
                "pixel_level_mean_over_detected_only": self.pixel_metrics_detected_only,
                "runtime_mean_s": float(self.per_image["time_s"].mean()) if len(self.per_image) else 0.0,
                "errors": self.errors}


def evaluate_dataset(dataset_dir: Path, cfg: AnalysisConfig | None = None,
                     limit: int | None = None, inconclusive_as_positive: bool = False,
                     progress=None) -> EvaluationReport:
    """Run the detector on every image in the dataset and score it.

    progress: optional callable(done, total, filename) for UI progress bars.
    """
    cfg = cfg or AnalysisConfig()
    dataset_dir = Path(dataset_dir)
    originals = list_images(dataset_dir / "original")
    forged = list_images(dataset_dir / "forged")
    if limit:
        originals, forged = originals[:limit], forged[:limit]
    jobs = [(p, 0) for p in originals] + [(p, 1) for p in forged]
    if not jobs:
        raise FileNotFoundError(
            f"No images found in {dataset_dir}/original or {dataset_dir}/forged.")

    rows, errors = [], []
    positive = {STATUS_MANIPULATED} | ({STATUS_INCONCLUSIVE} if inconclusive_as_positive else set())
    for i, (path, label) in enumerate(jobs, 1):
        if progress:
            progress(i, len(jobs), path.name)
        img = read_image(path)
        if img is None:
            errors.append(f"Unreadable: {path.name}")
            continue
        t0 = time.perf_counter()
        res = run_analysis(img, cfg)
        elapsed = time.perf_counter() - t0
        pred = res.detection.status in positive
        row = {"file": path.name, "label": label, "status": res.detection.status,
               "predicted": int(pred), "keypoints": res.features.count,
               "matches": res.matches.count,
               "verified_matches": res.detection.n_verified_matches,
               "groups": res.detection.n_valid_groups,
               "score": res.detection.score, "time_s": round(elapsed, 3)}
        if label == 1:
            mpath = find_mask(path, dataset_dir / "masks")
            row["mask"] = mpath.name if mpath else ""
            if mpath is not None:
                gt = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
                if gt is None:
                    errors.append(f"Unreadable mask: {mpath.name}")
                else:
                    pm = res.mask_original_size() if pred else np.zeros_like(gt)
                    row.update(pixel_metrics(gt, pm))
        rows.append(row)

    df = pd.DataFrame(rows)
    img_m = image_level_metrics(df["label"], df["predicted"]) if len(df) else {}
    pix_cols = ["pixel_precision", "pixel_recall", "pixel_f1", "iou"]
    if len(df) and set(pix_cols) <= set(df.columns):
        scored = df.dropna(subset=["iou"])
        pix_mean = {c: float(scored[c].mean()) for c in pix_cols}
        pix_mean["n_images"] = int(len(scored))
        det = scored[scored["predicted"] == 1]
        pix_det = {c: (float(det[c].mean()) if len(det) else 0.0) for c in pix_cols}
        pix_det["n_images"] = int(len(det))
    else:
        pix_mean, pix_det = {"n_images": 0}, {"n_images": 0}
    return EvaluationReport(df, img_m, pix_mean, pix_det, cfg.to_dict(),
                            str(dataset_dir), inconclusive_as_positive, errors)


def save_evaluation(report: EvaluationReport, out_dir: Path | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    name = Path(report.dataset).name or "dataset"
    out = Path(out_dir or RESULTS_DIR) / f"eval_{name}_{dt.datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    report.per_image.to_csv(out / "per_image.csv", index=False)
    (out / "summary.json").write_text(json.dumps(report.to_dict(), indent=2))

    cm = report.image_metrics.get("confusion_matrix")
    if cm:
        fig, ax = plt.subplots(figsize=(4, 3.4))
        mat = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
        ax.imshow(mat, cmap="Blues")
        for (r, c), v in np.ndenumerate(mat):
            ax.text(c, r, str(v), ha="center", va="center",
                    color="white" if v > mat.max() / 2 else "black", fontsize=13)
        ax.set_xticks([0, 1], ["Pred authentic", "Pred manipulated"])
        ax.set_yticks([0, 1], ["Authentic", "Forged"])
        ax.set_title("Image-level confusion matrix", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "confusion_matrix.png", dpi=150)
        plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate copy-move detection on a dataset.")
    ap.add_argument("--dataset", type=Path, required=True,
                    help="folder containing original/, forged/, masks/")
    ap.add_argument("--method", default="SIFT", choices=["ORB", "SIFT"])
    ap.add_argument("--sensitivity", default="Medium", choices=["Low", "Medium", "High"])
    ap.add_argument("--limit", type=int, default=None, help="max images per folder")
    ap.add_argument("--inconclusive-positive", action="store_true",
                    help="count 'Inconclusive' as a positive detection")
    args = ap.parse_args()

    cfg = AnalysisConfig.from_sensitivity(args.sensitivity, method=args.method)
    rep = evaluate_dataset(args.dataset, cfg, args.limit, args.inconclusive_positive,
                           progress=lambda i, n, f: print(f"[{i}/{n}] {f}", flush=True))
    out = save_evaluation(rep)
    print(json.dumps(rep.to_dict(), indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
