"""Prepare resumable fixed real/generated latent assets before starting training."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm

from pid._src.datasets.utils import IMAGE_RES_SIZE_INFO
from pid._src.linear_pid.data import RawImageDataset, atomic_json, resize_crop, sha256_file
from pid._src.linear_pid.evaluation import save_png, tensor_image
from pid._src.linear_pid.runtime import Conditioning
from pid._src.linear_pid.training import broadcast_object, init_distributed

PROMPTS = [
    "A red fox standing in snowy woodland at sunrise, fine fur, soft light.",
    "A portrait of an elderly woman with silver hair, natural skin, window lighting.",
    "Two dogs, a black dog on the left and a white dog on the right, on green grass.",
    "A blue cup to the left of a red apple on a wooden table, daylight.",
    "A city street with brick buildings, bicycles, and people under warm evening light.",
    "A mountain lake reflecting snow-covered peaks, pine trees, crisp fine details.",
    "A close-up photograph of colorful woven fabric with intricate repeating patterns.",
    "A stone bridge across a river beneath a castle, clear architectural details.",
    "Three yellow lemons beside a glass pitcher, bright kitchen still life.",
    "A small child and a golden retriever walking together along a beach.",
    "A green parrot on a wooden branch, vivid plumage, natural color.",
    "A wooden bookshelf filled with books, a small plant on the right.",
    "A white ceramic bowl of strawberries on a blue tablecloth.",
    "A lighthouse on a rocky coastline during a dramatic cloudy sunset.",
    "A macro photograph of a butterfly resting on a purple flower.",
    "A busy outdoor market with fruit stalls, people, and colorful awnings.",
    "A silver train crossing a steel bridge through autumn woodland.",
    "A watercolor illustration of a cottage surrounded by flowering trees.",
    "A full-body portrait of a dancer in a flowing red dress, studio lighting.",
    "A brown tabby cat sitting in a sunlit window, detailed whiskers.",
    "A modern living room with a yellow sofa and a blue armchair.",
    "A sailing boat on turquoise water beside a tropical island.",
    "A farmer holding a basket of vegetables beside a field.",
    "A snowy alpine village at night with warm glowing windows.",
    "A bowl of noodles with vegetables and chopsticks, food photography.",
    "A futuristic city with flying vehicles, neon signs and layered architecture.",
    "A close-up of a watch with a polished metal case and a leather strap.",
    "A medieval knight riding a white horse through a green valley.",
    "A library interior with tall shelves and intricate wooden carvings.",
    "A bright orange bicycle leaning against a pale blue wall.",
    "A group of four friends sitting at a cafe table, natural expressions.",
    "A detailed ink illustration of a dragon above a forest.",
]


def prepare_generated_latents(cases, output, weights_root, rank, world, device):
    inputs = {
        "schema": 1,
        "weights_root": str(Path(weights_root).resolve()),
        "cases": [
            {key: case[key] for key in ("kind", "ordinal", "resolution", "seed", "caption")}
            for case in cases if case["kind"] == "generated"
        ],
    }
    inputs_path = output / "generated_inputs.json"
    if inputs_path.exists() and json.loads(inputs_path.read_text()) != inputs:
        raise ValueError("Generated latent inputs changed; choose a fresh --output-root")
    if rank == 0:
        atomic_json(inputs_path, inputs)
    pending = []
    for case in cases:
        if case["kind"] != "generated":
            continue
        identity = f"{case['kind']}_{case['ordinal']:03d}_{case['resolution']}"
        if (
            not (output / (identity + ".latent.pt")).exists()
            and not (output / (identity + ".pt")).exists()
        ):
            pending.append(case)
    pending = broadcast_object(pending if rank == 0 else None, rank)
    if pending:
        from diffusers import ZImagePipeline

        pipeline = ZImagePipeline.from_pretrained(
            Path(weights_root) / "Z-Image-Turbo", torch_dtype=torch.bfloat16, local_files_only=True
        ).to(device)
        for case in tqdm(pending[rank::world], desc="Z-Image-Turbo latents"):
            identity = f"{case['kind']}_{case['ordinal']:03d}_{case['resolution']}"
            size = 512 if case["resolution"] == "2048" else 1024
            latent = pipeline(
                case["caption"], height=size, width=size, num_inference_steps=9,
                guidance_scale=0.0,
                generator=torch.Generator(device=device).manual_seed(case["seed"]),
                output_type="latent",
            ).images
            if latent.ndim == 5:
                latent = latent.squeeze(2)
            temp = output / (identity + ".latent.tmp")
            torch.save(latent.detach().cpu(), temp)
            temp.replace(output / (identity + ".latent.pt"))
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def prepare_conditions(cases, output, weights_root, rank, world, device):
    pending = [case for case in cases if not (output / case["file"]).exists()]
    pending = broadcast_object(pending if rank == 0 else None, rank)
    if not pending:
        return
    conditioning = Conditioning(weights_root, device)
    for case in tqdm(pending[rank::world], desc="Fixed PiD conditions"):
        destination = output / case["file"]
        caption_embs, caption_mask = conditioning.encode_text([case["caption"]])
        if case["kind"] == "real":
            record = case["record"]
            with Image.open(record["path"]) as image:
                image = resize_crop(ImageOps.exif_transpose(image).convert("RGB"), record["bucket"], case["resolution"])
                array = np.array(image, copy=True)
            pixels = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device).float() / 127.5 - 1
            latent, lq = conditioning.encode_image(pixels)
            h, w = pixels.shape[-2:]
        else:
            latent = torch.load(output / (destination.stem + ".latent.pt"), weights_only=True).to(device)
            h, w = latent.shape[-2] * 32, latent.shape[-1] * 32
            lq = conditioning.vae.decode(latent)
        preview = output / (destination.stem + ".lq.png")
        save_png(tensor_image(lq), preview)
        asset = {
            "latent": latent.cpu(),
            "caption_embs": caption_embs.cpu(),
            "caption_mask": caption_mask.cpu(),
            "null_embs": conditioning.null_embs.cpu(),
            "null_mask": conditioning.null_mask.cpu(),
            "height": h,
            "width": w,
            "seed": case["seed"],
            "caption": case["caption"],
            "source_image": case.get("record", {}).get("path"),
        }
        temp = destination.with_suffix(".tmp")
        torch.save(asset, temp)
        temp.replace(destination)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-root", default="/home/ganchangyi/huggingface_ckpts")
    parser.add_argument("--index-root", default="/home/ganchangyi/dataset/MultiAspect-4K-1M/linear_pid_index")
    parser.add_argument("--output-root", default="/home/ganchangyi/code/PiD-Linear/outputs/linear-pid/assets")
    parser.add_argument("--real-count", type=int, default=64)
    parser.add_argument("--generated-count", type=int, default=32)
    parser.add_argument("--four-k-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument(
        "--generated-latents-only", action="store_true",
        help="Prepare generated latents while the full image index is still being built",
    )
    stage.add_argument(
        "--generated-conditions-only", action="store_true",
        help="Prepare generated captions and latents before the full image index is available",
    )
    args = parser.parse_args()
    if args.real_count < 0 or args.four_k_count < 0:
        parser.error("real-count and four-k-count must be nonnegative")
    if not 0 <= args.generated_count <= len(PROMPTS):
        parser.error("generated-count must be between 0 and 32")
    rank, world, device = init_distributed()
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    if args.generated_latents_only or args.generated_conditions_only:
        cases = [
            {"kind": "generated", "ordinal": i, "resolution": resolution,
             "seed": args.seed + 1000 + i, "caption": PROMPTS[i]}
            for resolution, count in [
                ("2048", args.generated_count),
                ("4096", min(args.four_k_count, args.generated_count)),
            ]
            for i in range(count)
        ]
        for case in cases:
            case["file"] = f"generated_{case['ordinal']:03d}_{case['resolution']}.pt"
        prepare_generated_latents(cases, output, args.weights_root, rank, world, device)
        if args.generated_conditions_only:
            prepare_conditions(cases, output, args.weights_root, rank, world, device)
        if rank == 0:
            print(f"Generated assets ready in {output}; the image index is still required for final assets")
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        return
    dataset = RawImageDataset(args.index_root)
    validation = np.load(Path(args.index_root) / "validation.npy")
    grouped = {}
    for row in validation:
        record = dataset.record(row)
        grouped.setdefault(record["bucket"], []).append(record)
    # Round-robin buckets for qualitative diversity; no random re-selection on resume.
    records = []
    while grouped and len(records) < args.real_count:
        for bucket in list(grouped):
            if grouped[bucket]:
                records.append(grouped[bucket].pop(0))
                if len(records) == args.real_count:
                    break
            else:
                del grouped[bucket]
    if len(records) < args.real_count:
        raise ValueError("Not enough held-out images for the requested gallery")
    four_k = []
    for record in records:
        w, h = IMAGE_RES_SIZE_INFO["4096"][record["bucket"]]
        if min(record["width"] / w, record["height"] / h) > 0.9:
            four_k.append(record)
    if len(four_k) < args.four_k_count:
        raise ValueError(
            "Not enough selected validation images with native 4K detail; increase real-count or inspect validation split"
        )
    cases = []
    for resolution, real in [("2048", records), ("4096", four_k[: args.four_k_count])]:
        for i, record in enumerate(real):
            cases.append(
                {
                    "kind": "real",
                    "ordinal": records.index(record),
                    "resolution": resolution,
                    "seed": args.seed + records.index(record),
                    "caption": record["caption"],
                    "record": record,
                }
            )
        count = args.generated_count if resolution == "2048" else min(args.four_k_count, args.generated_count)
        for i in range(count):
            cases.append(
                {
                    "kind": "generated",
                    "ordinal": i,
                    "resolution": resolution,
                    "seed": args.seed + 1000 + i,
                    "caption": PROMPTS[i],
                }
            )
    expected = {
        "schema": 1,
        "data_fingerprint": dataset.metadata["fingerprint"],
        "weights_root": args.weights_root,
        "seed": args.seed,
        "real_count": args.real_count,
        "generated_count": args.generated_count,
        "four_k_count": args.four_k_count,
    }
    build_path = output / "build.json"
    if build_path.exists() and json.loads(build_path.read_text()) != expected:
        raise ValueError("Asset inputs changed; choose a fresh --output-root")
    complete = output / "assets.json"
    if complete.exists():
        previous = json.loads(complete.read_text())
        valid = all(
            (output / c["file"]).exists() and sha256_file(output / c["file"]) == c["sha256"] for c in previous["cases"]
        )
        valid = broadcast_object(valid if rank == 0 else None, rank)
        if valid:
            if rank == 0:
                print(f"Complete fixed assets already exist in {output}")
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            return
        raise ValueError("Published assets were corrupted; use a fresh output-root")
    if rank == 0:
        atomic_json(build_path, expected)
    for case in cases:
        identity = f"{case['kind']}_{case['ordinal']:03d}_{case['resolution']}"
        case["file"] = f"{identity}.pt"
    prepare_generated_latents(cases, output, args.weights_root, rank, world, device)
    prepare_conditions(cases, output, args.weights_root, rank, world, device)
    if rank == 0:
        published = []
        for case in cases:
            published.append(
                {k: v for k, v in case.items() if k not in {"record", "caption"}}
                | {
                    "sha256": sha256_file(output / case["file"]),
                    "preview": str(output / (Path(case["file"]).stem + ".lq.png")),
                }
            )
        atomic_json(output / "assets.json", expected | {"cases": published})
        print(f"Published {len(published)} fixed assets in {output}")
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
