"""Stage-one observational diagnosis on held-out real images; no training or teacher copy."""

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pid._src.linear_pid.data import RawImageDataset, atomic_json, sha256_file
from pid._src.linear_pid.runtime import Conditioning
from pid._src.vdn_pid.attention import execution_mode
from pid._src.vdn_pid.checkpoint import load_delta, load_state, verify_base
from pid._src.vdn_pid.config import VDNConfig
from pid._src.vdn_pid.memory_diagnosis import collision_metrics
from pid._src.vdn_pid.training import build_student

PATTERNS = {
    "person": r"\b(person|people|man|woman|girl|boy|face|portrait)\b",
    "text": r"\b(text|letter|word|sign|logo|writing|poster)\b",
    "texture": r"\b(texture|fabric|fur|hair|wood|brick|pattern)\b",
}


def record_complete(path):
    return path.exists() and json.loads(path.read_text()).get("diagnosis_schema", 1) >= 2


def publish_experiment(path, value, rank=0):
    if path.exists() and json.loads(path.read_text())["identity"] != value["identity"]:
        raise ValueError("Diagnosis identity changed; use another output directory")
    if rank == 0:
        atomic_json(path, value)


def conditioning_provenance(dataset, weights_root):
    import hashlib

    implementation = hashlib.sha256()
    for p in [Path(__file__), Path(__file__).parents[1] / "pid/_src/vdn_pid/memory_diagnosis.py"]:
        implementation.update(p.name.encode())
        implementation.update(p.read_bytes())
    details = dict(
        schema=2,
        text_cache_fingerprint=dataset.text_cache.fingerprint,
        vae_sha256=sha256_file(Path(weights_root) / "PiD/checkpoints/ae.safetensors"),
        diagnosis_implementation_sha256=implementation.hexdigest(),
    )
    return dict(identity=hashlib.sha256(json.dumps(details, sort_keys=True).encode()).hexdigest(), details=details)


def tags(caption):
    return {k: bool(re.search(v, caption.lower())) for k, v in PATTERNS.items()}


class Selected(Dataset):
    def __init__(self, dataset, rows):
        self.dataset, self.rows = dataset, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        item = self.dataset[self.rows[i]]
        return {"selection_index": i, **item}


def faces(image):
    cv2.setNumThreads(1)
    h, w = image.shape[:2]
    scale = min(1, 1024 / max(h, w))
    gray = cv2.cvtColor(cv2.resize(image, (round(w * scale), round(h * scale))), cv2.COLOR_RGB2GRAY)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return [
        [int(x / scale), int(y / scale), int((x + ww) / scale), int((y + hh) / scale)]
        for x, y, ww, hh in detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(14, 14))
    ]


def draw_maps(image, grid, error, arrays, rois, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Source; Haar face candidates / manual ROIs")
    for boxes in rois.values():
        for x1, y1, x2, y2 in boxes:
            axes[0].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime"))
    axes[1].imshow(error.reshape(grid), cmap="magma")
    axes[1].set_title("Same-hidden Attention MSE")
    ix = arrays["indices"]
    sc = axes[2].scatter(ix % grid[1], ix // grid[1], c=arrays["conflict"], s=10, cmap="magma")
    axes[2].set_ylim(grid[0], 0)
    axes[2].set_title("Sampled conflict (not a dense map)")
    fig.colorbar(sc, ax=axes[2], fraction=0.04)
    for a in axes:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--weights-root", required=True)
    p.add_argument("--index-root", required=True)
    p.add_argument("--text-cache-root", required=True)
    p.add_argument("--output-root", default="outputs/research/spatial-delta-memory/diagnosis")
    p.add_argument("--images", type=int, default=100)
    p.add_argument("--layers", nargs="+", type=int, default=[0, 7, 13])
    p.add_argument("--timesteps", nargs="+", type=int, default=[100, 500, 900])
    p.add_argument("--anchors", type=int, default=512)
    p.add_argument("--candidates", type=int, default=64)
    p.add_argument("--tau", type=float, default=0.8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plots", type=int, default=8)
    p.add_argument("--roi-json", help="Sample-id -> face/text/texture boxes in cropped 2K pixel coordinates")
    p.add_argument("--max-seconds", type=int, default=1200)
    p.add_argument("--swanlab-mode", choices=["offline", "cloud", "disabled"], default="offline")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    rank, world = int(os.getenv("RANK", 0)), int(os.getenv("WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.getenv("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    dataset = RawImageDataset(args.index_root, text_cache_root=args.text_cache_root)
    validation = np.load(Path(args.index_root) / "validation.npy")
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(validation)
    groups = {k: [] for k in ["person", "text", "texture", "other"]}
    for row in order:
        record = dataset.record(int(row))
        labels = tags(record["caption"])
        for k in groups:
            if labels.get(k, not any(labels.values())):
                groups[k].append(int(row))
    chosen = []
    while len(chosen) < args.images:
        before = len(chosen)
        for group in groups.values():
            while group and group[-1] in chosen:
                group.pop()
            if group and len(chosen) < args.images:
                chosen.append(group.pop())
        if len(chosen) == before:
            raise ValueError("Not enough distinct held-out images")
    payload = load_state(args.checkpoint)
    config = VDNConfig(**payload["config"])
    config.weights_root = args.weights_root
    verify_base(config, payload)
    settings = {k: v for k, v in vars(args).items() if k not in ["workers", "threads", "max_seconds", "swanlab_mode"]}
    settings.update(
        checkpoint_sha256=sha256_file(Path(args.checkpoint) / "state.pt"),
        data_fingerprint=dataset.metadata["fingerprint"],
        rows=chosen,
        source="real held-out x0 mixed with fixed Gaussian; actual NoPE SiLU+DWConv+L2 VDA keys; same student hidden original Attention control",
    )
    settings["roi_sha256"] = sha256_file(args.roi_json) if args.roi_json else None
    identity = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    old = out / "experiment.json"
    provenance_path = out / "conditioning-provenance.json"
    if old.exists() and not provenance_path.exists():
        raise ValueError(
            "Legacy diagnosis has no conditioning provenance; use a new directory or explicitly audit its immutable inputs"
        )
    publish_experiment(provenance_path, conditioning_provenance(dataset, args.weights_root), rank)
    selection = [dict(row=row, **{k: dataset.record(row)[k] for k in ["id", "caption", "bucket"]}) for row in chosen]
    publish_experiment(old, dict(identity=identity, settings=settings, selection=selection), rank)
    manual = json.loads(Path(args.roi_json).read_text()) if args.roi_json else {}
    net = build_student(config, payload["stage"], device)
    load_delta(net, payload["delta"])
    net.eval().requires_grad_(False)
    conditioning = Conditioning(args.weights_root, device, text_cache_root=args.text_cache_root)
    tracker = None
    if args.swanlab_mode != "disabled":
        import swanlab

        tracker = swanlab.init(
            project="PiD-Spatial-Memory-Diagnosis",
            experiment_name=f"phase1-rank{rank}",
            mode=args.swanlab_mode,
            logdir=str(out / "swanlog"),
            config=settings,
        )
    started = time.monotonic()
    rows = chosen[rank::world]
    # DataLoader keeps uint8 images and BF16 cached text; no full dataset metadata per worker.
    loader = DataLoader(Selected(dataset, rows), batch_size=None, num_workers=args.workers)
    try:
        for item in tqdm(loader, desc=f"Diagnosis rank{rank}"):
            if args.max_seconds and time.monotonic() - started > args.max_seconds:
                break
            if "error" in item:
                atomic_json(out / f"invalid_{item['row']}.json", dict(error=item["error"]))
                continue
            index = chosen.index(int(item["row"]))
            record = dataset.record(int(item["row"]))
            required = [out / f"row_{item['row']}_t{t}_l{layer}.json" for t in args.timesteps for layer in args.layers]
            if all(record_complete(f) for f in required):
                continue
            image = item["image"].permute(1, 2, 0).numpy()
            rois = manual.get(item["id"], {})
            rois = {
                "face": rois.get("face", faces(image) if tags(record["caption"])["person"] else []),
                "text": rois.get("text", []),
                "texture": rois.get("texture", []),
            }
            thumb = Image.fromarray(image)
            thumb.thumbnail((768, 768))
            thumb.save(out / f"row_{item['row']}.jpg")
            atomic_json(
                out / f"row_{item['row']}.source.json",
                dict(
                    id=item["id"],
                    row=item["row"],
                    caption=record["caption"],
                    tags=tags(record["caption"]),
                    shape=list(image.shape),
                    rois=rois,
                    face_roi_source="Haar candidate; may miss small/profile faces",
                ),
            )
            x0 = item["image"][None].to(device, dtype=torch.float32) / 127.5 - 1
            embs = item["text_embs"][None].to(device)
            mask = item["text_mask"][None].to(device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                latent, _ = conditioning.encode_image(x0)
            noise = torch.randn(
                x0.shape,
                device=device,
                generator=torch.Generator(device=device).manual_seed(args.seed + int(item["row"])),
            )
            gray = x0.mean(1, keepdim=True)
            gx = F.pad((gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs(), (0, 1))
            gy = F.pad((gray[:, :, 1:] - gray[:, :, :-1]).abs(), (0, 0, 0, 1))
            texture = F.avg_pool2d(gx + gy, 16).flatten()
            for timestep in args.timesteps:
                pending = [
                    layer
                    for layer in args.layers
                    if not record_complete(out / f"row_{item['row']}_t{timestep}_l{layer}.json")
                ]
                if not pending:
                    continue
                handles = []
                for layer in pending:

                    def hook(module, inputs, kwargs, outputs, layer=layer):
                        x, y, pos = inputs[:3]
                        pt = inputs[3] if len(inputs) > 3 else None
                        grid = tuple(map(int, kwargs["grid_size"]))
                        teacher = module.orig(x, y, pos, pt, kwargs.get("attn_mask"))[0]
                        error = (outputs[0].float() - teacher.float()).square().mean(-1)[0]
                        energy = teacher.float().square().mean(-1)[0]
                        qkv, tqkv = module.raw_qkv(x, y)
                        _, k, v = module.linear_attention.features(qkv, grid)
                        beta = torch.sigmoid(module.linear_attention.beta_proj(x))
                        metrics, arrays = collision_metrics(
                            k,
                            v,
                            beta,
                            error,
                            energy,
                            texture,
                            grid,
                            seed=args.seed + int(item["row"]) + layer,
                            anchors=args.anchors,
                            candidates=args.candidates,
                            tau=args.tau,
                        )
                        ix = arrays["indices"]
                        cy = (ix // grid[1] + 0.5) * 16
                        cx = (ix % grid[1] + 0.5) * 16
                        for kind, boxes in rois.items():
                            selected = np.zeros(len(ix), dtype=bool)
                            for x1, y1, x2, y2 in boxes:
                                selected |= (cx >= x1) & (cx < x2) & (cy >= y1) & (cy < y2)
                            arrays[kind + "_roi"] = selected
                            metrics[kind + "_roi_anchors"] = int(selected.sum())
                            metrics[kind + "_roi_error"] = (
                                float(arrays["error"][selected].mean()) if selected.any() else None
                            )
                            metrics[kind + "_roi_conflict"] = (
                                float(arrays["conflict"][selected].mean()) if selected.any() else None
                            )
                        prefix = out / f"row_{item['row']}_t{timestep}_l{layer}"
                        np.savez_compressed(str(prefix) + ".npz", **arrays)
                        if index < args.plots and timestep == 500 and layer == args.layers[len(args.layers) // 2]:
                            draw_maps(image, grid, error.cpu().numpy(), arrays, rois, str(prefix) + ".png")
                        result = dict(
                            diagnosis_schema=2,
                            identity=identity,
                            row=int(item["row"]),
                            sample_id=item["id"],
                            layer=layer,
                            timestep=timestep,
                            grid=grid,
                            tags=tags(record["caption"]),
                            metrics=metrics,
                        )
                        atomic_json(str(prefix) + ".json", result)
                        if tracker:
                            tracker.log(
                                {
                                    f"layer{layer}/t{timestep}/{k}": v
                                    for k, v in metrics.items()
                                    if v is not None and isinstance(v, (float, int))
                                },
                                step=index,
                            )

                    handles.append(net.patch_blocks[layer].attn.register_forward_hook(hook, with_kwargs=True))
                try:
                    noisy = x0 * (1 - timestep / 1000) + noise * (timestep / 1000)
                    with (
                        torch.no_grad(),
                        execution_mode(net, inference=True),
                        torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False),
                    ):
                        net(
                            noisy,
                            torch.tensor([timestep], device=device, dtype=torch.float32),
                            embs,
                            lq_latent=latent,
                            degrade_sigma=torch.zeros(1, device=device),
                            text_valid_mask=mask,
                        )
                finally:
                    for h in handles:
                        h.remove()
            del x0, latent, noise, embs, mask
    finally:
        if tracker:
            tracker.finish()
    atomic_json(
        out / f"rank{rank}.status.json",
        dict(
            identity=identity,
            completed_records=len(list(out.glob("row_*_t*_l*.json"))),
            expected_records=args.images * len(args.layers) * len(args.timesteps),
        ),
    )


if __name__ == "__main__":
    main()
