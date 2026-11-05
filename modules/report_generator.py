"""
Report generation: JSON (machine-readable), HTML (self-contained, images
embedded as base64) and PDF (matplotlib, no extra dependency).
"""
from __future__ import annotations

import base64
import datetime as dt
import html
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from .config import RESULTS_DIR  # noqa: E402
from .visualization import to_png_bytes  # noqa: E402

DISCLAIMER = ("Automated image-forensics results are indicative and should not be "
              "treated as definitive proof of image manipulation.")


def build_json(result) -> dict:
    """Everything needed to reproduce / audit the analysis."""
    info = result.info
    data = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "image": None if info is None else {
            "filename": info.filename, "width": info.width, "height": info.height,
            "format": info.format, "file_size_kb": info.file_size_kb},
        "working_resolution": list(result.pre.gray.shape[::-1]),
        "preprocessing_steps": result.pre.steps,
        "summary": result.summary(),
        "notes": result.detection.notes + result.localization.notes,
        "match_groups": [g.summary() for g in result.verification.groups],
        "regions": [{"id": r.region_id, "group": r.group_id, "role": r.role,
                     "bbox_xywh_working": list(r.bbox),
                     "bbox_xywh_original": [round(v / result.pre.scale) for v in r.bbox],
                     "area_px": r.area_px, "method": r.method}
                    for r in result.localization.regions],
        "timings_s": {k: round(v, 4) for k, v in result.timings.items()},
        "parameters": result.config.to_dict(),
        "disclaimer": DISCLAIMER,
    }
    if result.splicing is not None:
        data["splicing_experimental"] = {**result.splicing.summary(),
                                         "notes": result.splicing.notes}
    return data


def json_bytes(result) -> bytes:
    return json.dumps(build_json(result), indent=2).encode()


def _b64(rgb) -> str:
    return base64.b64encode(to_png_bytes(rgb)).decode()


def build_html(result, visuals: dict, splicing_visuals: dict | None = None) -> str:
    d = build_json(result)
    s = d["summary"]
    status = s["status"]
    colour = {"Potentially manipulated": "#c0262d",
              "Inconclusive": "#b45309"}.get(status, "#15803d")
    rows = "".join(f"<tr><th>{html.escape(k.replace('_', ' ').capitalize())}</th>"
                   f"<td>{html.escape(str(v))}</td></tr>" for k, v in s.items())
    groups = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in g.values()) + "</tr>"
        for g in d["match_groups"])
    gh = "".join(f"<th>{html.escape(k)}</th>" for k in (d["match_groups"][0] if d["match_groups"] else {}))
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in d["notes"])
    figs = "".join(f'<figure><img src="data:image/png;base64,{_b64(img)}"/>'
                   f"<figcaption>{html.escape(t)}</figcaption></figure>"
                   for t, img in visuals.items())
    spl = ""
    if splicing_visuals and result.splicing is not None:
        sp = d["splicing_experimental"]
        spl = (f"<h2>Experimental splicing indicators</h2><p><b>{html.escape(sp['status'])}</b>"
               f" &middot; suspicious area {sp['suspicious_area_pct']}%</p>"
               + "<div class='grid'>" + "".join(
                   f'<figure><img src="data:image/png;base64,{_b64(img)}"/>'
                   f"<figcaption>{html.escape(t)}</figcaption></figure>"
                   for t, img in splicing_visuals.items()) + "</div>")
    fname = html.escape(d["image"]["filename"]) if d["image"] else "image"
    params = html.escape(json.dumps(d["parameters"], indent=1))
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Image Forensics Report - {fname}</title>
<style>
body{{font-family:Inter,Segoe UI,Helvetica,Arial,sans-serif;color:#0f1b33;margin:0;background:#f6f8fb}}
main{{max-width:1100px;margin:auto;padding:32px 20px}}
h1{{font-size:24px;margin:0}} h2{{font-size:17px;margin-top:32px;border-bottom:1px solid #d9e0ea;padding-bottom:6px}}
.kicker{{color:#2563eb;font-size:12px;letter-spacing:.14em;font-weight:600}}
.status{{display:inline-block;margin-top:14px;padding:6px 14px;border-radius:4px;color:#fff;background:{colour};font-weight:600}}
table{{border-collapse:collapse;width:100%;background:#fff;font-size:13px}}
th,td{{border:1px solid #e3e8ef;padding:6px 9px;text-align:left;vertical-align:top}} th{{background:#f1f4f9;font-weight:600;width:32%}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px}}
figure{{margin:0;background:#fff;border:1px solid #e3e8ef;padding:8px}} img{{width:100%;display:block}}
figcaption{{font-size:12px;color:#475569;margin-top:6px}}
.disc{{background:#fff7e6;border-left:3px solid #f59e0b;padding:10px 14px;font-size:13px;margin-top:20px}}
pre{{background:#fff;border:1px solid #e3e8ef;padding:10px;font-size:11px;overflow:auto}}
</style></head><body><main>
<div class="kicker">IMAGE FORENSICS</div>
<h1>Copy-move analysis report &mdash; {fname}</h1>
<div style="color:#64748b;font-size:13px">Generated {d['generated_at']}</div>
<div class="status">{html.escape(status)}</div>
<div class="disc">{DISCLAIMER}</div>
<h2>Forensic summary</h2><table>{rows}</table>
<h2>Findings</h2><ul>{notes}</ul>
<h2>Match groups (RANSAC)</h2><table><tr>{gh}</tr>{groups}</table>
<h2>Visual analysis</h2><div class="grid">{figs}</div>
{spl}
<h2>Analysis parameters</h2><pre>{params}</pre>
</main></body></html>"""


def build_pdf(result, visuals: dict) -> bytes:
    d = build_json(result)
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.07, 0.95, "IMAGE FORENSICS", color="#2563eb", fontsize=9, weight="bold")
        fig.text(0.07, 0.92, "Copy-move analysis report", fontsize=17, weight="bold")
        name = d["image"]["filename"] if d["image"] else "image"
        fig.text(0.07, 0.895, f"{name}  |  {d['generated_at']}", fontsize=9, color="#64748b")
        y = 0.85
        for k, v in d["summary"].items():
            if k == "detection_score_formula":
                continue
            fig.text(0.07, y, k.replace("_", " ").capitalize(), fontsize=9, weight="bold")
            fig.text(0.45, y, str(v), fontsize=9)
            y -= 0.022
        y -= 0.01
        fig.text(0.07, y, "Score formula: " + d["summary"]["detection_score_formula"],
                 fontsize=8, color="#475569", wrap=True)
        y -= 0.04
        fig.text(0.07, y, "Findings", fontsize=11, weight="bold")
        y -= 0.025
        for n in d["notes"]:
            fig.text(0.08, y, "- " + n, fontsize=8, wrap=True)
            y -= 0.03
        fig.text(0.07, 0.05, DISCLAIMER, fontsize=8, color="#b45309", wrap=True)
        pdf.savefig(fig)
        plt.close(fig)

        items = list(visuals.items())
        for i in range(0, len(items), 4):
            fig, axes = plt.subplots(2, 2, figsize=(8.27, 11.69))
            for ax in axes.ravel():
                ax.axis("off")
            for ax, (title, img) in zip(axes.ravel(), items[i:i + 4]):
                ax.imshow(img)
                ax.set_title(title, fontsize=8)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    return buf.getvalue()


def save_report(result, visuals: dict, out_dir: Path | None = None,
                splicing_visuals: dict | None = None) -> Path:
    """Write JSON, HTML, PDF, mask and key images to results/<name>_<time>/."""
    stem = Path(result.info.filename).stem if result.info else "image"
    out = Path(out_dir or RESULTS_DIR) / f"{stem}_{dt.datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_bytes(json_bytes(result))
    (out / "report.html").write_text(build_html(result, visuals, splicing_visuals))
    (out / "report.pdf").write_bytes(build_pdf(result, visuals))
    from PIL import Image
    Image.fromarray(result.mask_original_size()).save(out / "mask.png")
    for title, img in visuals.items():
        safe = "".join(c if c.isalnum() else "_" for c in title.split("(")[0]).strip("_")
        Image.fromarray(img).save(out / f"{safe}.png")
    return out
