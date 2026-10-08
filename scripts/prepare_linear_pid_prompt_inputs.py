"""Prepare prompt-only PiD inputs: fresh Z-Image latents and fresh Gemma embeddings.

Read captions and aspect buckets from a saved unseen-sample selection. Never read
the dataset source images or their VAE latents. Conditions are shared across the
original PiD and KDA decoder for a paired comparison.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import torch
from tqdm import tqdm

from pid._src.datasets.utils import IMAGE_RES_SIZE_INFO
from pid._src.linear_pid.checkpoint import load_checkpoint, resolve_checkpoint
from pid._src.linear_pid.compare_inference import validate_asset
from pid._src.linear_pid.data import atomic_json, sha256_file
from pid._src.linear_pid.text_encoder import GemmaTextEncoder, NEGATIVE_PROMPT
from pid._src.linear_pid.training import init_distributed


def prompt_cases(selection, seed):
    return [dict(id=f"prompt_{i:03d}_{resolution}", caption=record["caption"],
                 source_caption_id=record["id"], resolution=resolution, seed=seed + i,
                 width=IMAGE_RES_SIZE_INFO[resolution][record["bucket"]][0],
                 height=IMAGE_RES_SIZE_INFO[resolution][record["bucket"]][1])
            for resolution in ["2048", "4096"] for i, record in enumerate(selection["records"])]


def save_tensor(value, destination):
    temporary = destination.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    payload = load_checkpoint(resolve_checkpoint(args.checkpoint))
    selection = json.loads(Path(args.selection).read_text())
    if selection["signature"]["checkpoint_progress"] != payload["progress"]:
        raise ValueError("Unseen selection belongs to a different checkpoint")
    if selection["signature"]["data_fingerprint"] != payload["metadata"]["data_fingerprint"]:
        raise ValueError("Unseen selection belongs to a different training dataset")
    cases = prompt_cases(selection, args.seed)
    weights_root = Path(payload["config"]["weights_root"])
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    signature = {"cases": cases, "selection_sha256": sha256_file(args.selection),
                 "weights_root": str(weights_root.resolve()), "base_model": "Z-Image-Turbo",
                 "base_steps": 9, "base_guidance": 0.0, "condition_source": "prompt_only",
                 "fresh_gemma_encoding": True, "dataset_images_used": False}
    signature_path = output / "inputs.json"
    rank, world, device = init_distributed()
    try:
        if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
            raise ValueError("Prompt-generation settings changed; use a different output root")
        if rank == 0:
            atomic_json(signature_path, signature)
        if world > 1:
            torch.distributed.barrier()
        assigned = cases[rank::world]
        pending = [case for case in assigned if not (
            (output / (case["id"] + ".pt")).exists()
            and (output / (case["id"] + ".json")).exists())]
        need_latents = [case for case in pending if not (
            (output / (case["id"] + ".latent.pt")).exists()
            and (output / (case["id"] + ".latent.json")).exists())]
        if need_latents:
            from diffusers import ZImagePipeline

            pipeline = ZImagePipeline.from_pretrained(
                weights_root / "Z-Image-Turbo", torch_dtype=torch.bfloat16, local_files_only=True
            ).to(device)
            with torch.inference_mode():
                for case in tqdm(need_latents, desc=f"Prompt-generated latents GPU {rank}"):
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    latent = pipeline(
                        case["caption"], height=case["height"] // 4, width=case["width"] // 4,
                        num_inference_steps=9, guidance_scale=0.0, output_type="latent",
                        generator=torch.Generator(device=device).manual_seed(case["seed"]),
                    ).images
                    if latent.ndim == 5:
                        latent = latent.squeeze(2)
                    torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - start
                    destination = output / (case["id"] + ".latent.pt")
                    save_tensor(latent.cpu(), destination)
                    atomic_json(destination.with_suffix(".json"), {
                        "case": case, "latent_generation_seconds": elapsed,
                        "latent_sha256": sha256_file(destination), "dataset_images_used": False,
                    })
            del pipeline, latent
            gc.collect()
            torch.cuda.empty_cache()
        if pending:
            # Re-encode from prompt text; do not read the training Gemma cache.
            encoder = GemmaTextEncoder(weights_root, device)
            with torch.inference_mode():
                null_embs, null_mask = encoder.encode_text([NEGATIVE_PROMPT])
                for case in tqdm(pending, desc=f"Fresh prompt encoding GPU {rank}"):
                    latent_path = output / (case["id"] + ".latent.pt")
                    latent_meta = json.loads(latent_path.with_suffix(".json").read_text())
                    if sha256_file(latent_path) != latent_meta["latent_sha256"]:
                        raise ValueError(f"Corrupt generated latent: {latent_path}")
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    embs, mask = encoder.encode_text([case["caption"]])
                    torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - start
                    asset = {"latent": torch.load(latent_path, map_location="cpu", weights_only=True),
                             "caption_embs": embs.cpu(), "caption_mask": mask.cpu(),
                             "null_embs": null_embs.cpu(), "null_mask": null_mask.cpu(),
                             "caption": case["caption"], "height": case["height"], "width": case["width"],
                             "seed": case["seed"], "condition_source": "prompt_only",
                             "latent_source": "Z-Image-Turbo", "source_caption_id": case["source_caption_id"]}
                    validate_asset(asset)
                    destination = output / (case["id"] + ".pt")
                    save_tensor(asset, destination)
                    atomic_json(destination.with_suffix(".json"), latent_meta | {
                        "gemma_encoding_seconds": elapsed, "asset_sha256": sha256_file(destination),
                        "fresh_gemma_encoding": True,
                    })
        for case in assigned:
            path = output / (case["id"] + ".pt")
            metadata = json.loads(path.with_suffix(".json").read_text())
            if sha256_file(path) != metadata["asset_sha256"]:
                raise ValueError(f"Corrupt prompt condition: {path}")
            validate_asset(torch.load(path, map_location="cpu", weights_only=True))
        if world > 1:
            torch.distributed.barrier()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
