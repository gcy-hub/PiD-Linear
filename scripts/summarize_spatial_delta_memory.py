"""Summarize image-level collision associations and real generation timings."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pid._src.linear_pid.data import atomic_json


def bootstrap(values):
    x = np.asarray([v for v in values if v is not None and np.isfinite(v)])
    if not len(x):
        return dict(images=0, median=None, mean=None, mean_ci95=None)
    means = np.random.default_rng(42).choice(x, (2000, len(x)), replace=True).mean(1)
    return dict(
        images=len(x),
        median=float(np.median(x)),
        mean=float(x.mean()),
        mean_ci95=np.quantile(means, [0.025, 0.975]).tolist(),
    )


def display_number(value):
    return "—" if value is None else f"{value:.2f}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", default="outputs/research/spatial-delta-memory")
    args = p.parse_args()
    root = Path(args.output_root)
    records = [json.loads(f.read_text()) for f in (root / "diagnosis").glob("row_*_t*_l*.json")]
    records = [r for r in records if r.get("diagnosis_schema", 1) >= 2]
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["layer"], row["timestep"])].append(row)
    table = []
    for (layer, t), rows in sorted(grouped.items()):
        table.append(
            dict(
                layer=layer,
                timestep=t,
                **{
                    k: bootstrap([r["metrics"].get(k) for r in rows])
                    for k in [
                        "conflict_error_spearman",
                        "conflict_error_partial_spearman",
                        "weighted_conflict_error_spearman",
                        "effective_key_dimension_mean",
                        "centered_effective_key_dimension_mean",
                        "key_mean_energy_fraction",
                        "centered_high_key_fraction",
                        "centered_conflict_error_spearman",
                        "high_key_similarity_fraction",
                        "mean_attention_mse",
                        "mean_conflict",
                    ]
                },
            )
        )
    speed = [json.loads(f.read_text()) for f in (root / "speed").glob("*/*/measurement.json")]
    timing = []
    for row in speed:
        cfg = row["settings"]
        timing.append(
            dict(
                model=cfg["model"],
                resolution=cfg["resolution"],
                network_ms=row["network_median_ms"],
                layer8_ms=row["layer8_attention_median_ms"],
                generation_seconds=row.get("decode_seconds"),
                peak_gib=row.get("peak_memory_gib"),
                mmdit_seconds=row.get("decode_parts", {}).get("mmdit_ms", 0) / 1000,
                pit_seconds=row.get("decode_parts", {}).get("pit_ms", 0) / 1000,
                untrained_prototype=cfg["memory_prototype_untrained"],
            )
        )
    sources = [json.loads(f.read_text()) for f in (root / "diagnosis").glob("row_*.source.json")]
    coverage = dict(
        images=len(sources),
        records=len(records),
        person_caption=sum(s["tags"]["person"] for s in sources),
        text_caption=sum(s["tags"]["text"] for s in sources),
        texture_caption=sum(s["tags"]["texture"] for s in sources),
        images_with_face_candidates=sum(bool(s["rois"]["face"]) for s in sources),
        images_with_manual_text_rois=sum(bool(s["rois"]["text"]) for s in sources),
    )
    manual_path = root / "manual-rois.json"
    manual = json.loads(manual_path.read_text()) if manual_path.exists() else {"rows": {}}
    roi_rows = []
    for r in records:
        for kind, boxes in manual["rows"].get(str(r["row"]), {}).items():
            f = root / "diagnosis" / f"row_{r['row']}_t{r['timestep']}_l{r['layer']}.npz"
            with np.load(f) as a:
                ix = a["indices"]
                x = (ix % r["grid"][1] + 0.5) * 16
                y = (ix // r["grid"][1] + 0.5) * 16
                use = np.zeros(len(ix), dtype=bool)
                for x1, y1, x2, y2 in boxes:
                    use |= (x >= x1) & (x < x2) & (y >= y1) & (y < y2)
                if use.sum() < 4 or (~use).sum() < 4:
                    continue
                roi_rows.append(
                    dict(
                        row=r["row"],
                        layer=r["layer"],
                        timestep=r["timestep"],
                        kind=kind,
                        roi_anchors=int(use.sum()),
                        error_ratio=float(a["error"][use].mean() / max(a["error"][~use].mean(), 1e-12)),
                        conflict_ratio=float(a["conflict"][use].mean() / max(a["conflict"][~use].mean(), 1e-12)),
                    )
                )
    roi_groups = []
    for kind in ["face", "text"]:
        for key in ["error_ratio", "conflict_ratio"]:
            per_image = defaultdict(list)
            for r in roi_rows:
                if r["kind"] == kind:
                    per_image[r["row"]].append(r[key])
            roi_groups.append(dict(kind=kind, metric=key, **bootstrap([np.mean(v) for v in per_image.values()])))
    result = dict(
        coverage=coverage,
        associations=table,
        speed=timing,
        manual_roi_supplement=dict(selection_note=manual.get("selection_note"), groups=roi_groups, records=roi_rows),
        limitations=[
            "Within-image associations are summarized over independent images; tokens are not independent replicates.",
            "Same student hidden input control is not a full teacher trajectory or a causal image intervention.",
            "Caption tags are image-level weak labels; Haar detections are candidates, not verified face annotations. Text ROIs require manual annotation.",
            "Effective dimension and collision proxy measure representation redundancy, not proof of image quality loss.",
            "Uncentered effective dimension can be dominated by the shared key mean. Centering is a diagnostic control, not the key representation used by the model.",
            "Global bank prototypes are untrained, use zero initial states and self writes, omit raster decay and text memory, preserve text through local Softmax. Timing is not a quality-preserving speedup claim.",
        ],
    )
    atomic_json(root / "summary.json", result)
    md = [
        "# Spatially Disambiguated Delta Memory: phase-one report",
        "",
        f"Completed: {coverage['images']} real held-out images, {len(records)} layer/time records.",
        "",
        "| Model | Resolution | Layer 8 attention ms | Network ms | 25-step generation s | Peak GiB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(timing, key=lambda r: (r["resolution"], r["model"])):
        md.append(
            f"|{r['model']}|{r['resolution']}|{r['layer8_ms']:.2f}|{r['network_ms']:.2f}|{display_number(r['generation_seconds'])}|{display_number(r['peak_gib'])}|"
        )
    md += [
        "",
        "| Layer | t | Images | Mean Spearman (95% image bootstrap CI) | Partial Spearman | Effective dimension / 64 | Centered dimension / 64 | Mean energy fraction |",
        "|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for r in table:
        a = r["conflict_error_spearman"]
        c = r["conflict_error_partial_spearman"]
        d = r["effective_key_dimension_mean"]
        md.append(
            f"|{r['layer']}|{r['timestep']}|{a['images']}|{a['mean']} {a['mean_ci95']}|{c['mean']}|{d['mean']}|{r['centered_effective_key_dimension_mean']['mean']}|{r['key_mean_energy_fraction']['mean']}|"
        )
    md += [
        "",
        "## Manually checked source ROIs (small supplementary sample)",
        "",
        "| Kind | Metric (ROI / background) | Images | Mean | 95% image bootstrap CI |",
        "|---|---|---:|---:|---|",
    ]
    for r in roi_groups:
        md.append(f"|{r['kind']}|{r['metric']}|{r['images']}|{r['mean']}|{r['mean_ci95']}|")
    md += ["", "## Limits", "", *["- " + s for s in result["limitations"]]]
    (root / "report.md").write_text("\n".join(md) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
