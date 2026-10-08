#!/usr/bin/env python3
"""Download the model files used by Linear-PiD training and inference."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


# Keep the directory names in sync with linear_pid/runtime.py.
# Revisions pin the official repositories rather than a moving main branch.
MODEL_DOWNLOADS = {
    "pid": {
        "repo_id": "nvidia/PiD",
        "revision": "1b9b0872cef786e527ecc5f12b5ab2e6a02d55a4",
        "directory": "PiD",
        "allow_patterns": [
            "checkpoints/PiD_v1pt5_res2kto4k_sr4x_official_flux_undistilled/model_ema_bf16.pth",
        ],
    },
    "vae": {
        "repo_id": "nvidia/PiD",
        "revision": "1b9b0872cef786e527ecc5f12b5ab2e6a02d55a4",
        "directory": "PiD",
        "allow_patterns": ["checkpoints/ae.safetensors"],
    },
    "gemma": {
        "repo_id": "google/gemma-2-2b-it",
        "revision": "299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8",
        "directory": "gemma-2-2b-it",
        "allow_patterns": ["*.json", "*.safetensors", "tokenizer.model", "README.md", "LICENSE*"],
    },
    "zimage": {
        "repo_id": "Tongyi-MAI/Z-Image-Turbo",
        "revision": "f332072aa78be7aecdf3ee76d5c247082da564a6",
        "directory": "Z-Image-Turbo",
        "allow_patterns": [
            "model_index.json", "scheduler/*", "text_encoder/*", "tokenizer/*",
            "transformer/*", "vae/*", "README.md", "LICENSE*",
        ],
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=["all", *MODEL_DOWNLOADS], default=["all"])
    parser.add_argument("--weights-root", type=Path, default=Path("weights"))
    parser.add_argument("--workers", type=int, default=4, help="Concurrent Hugging Face file downloads")
    args = parser.parse_args()
    names = list(MODEL_DOWNLOADS) if "all" in args.models else list(dict.fromkeys(args.models))
    for name in names:
        spec = MODEL_DOWNLOADS[name]
        destination = args.weights_root / spec["directory"]
        print(f"Downloading {name}: {spec['repo_id']} -> {destination}", flush=True)
        snapshot_download(
            repo_id=spec["repo_id"],
            revision=spec["revision"],
            local_dir=destination,
            allow_patterns=spec["allow_patterns"],
            max_workers=args.workers,
            endpoint="https://huggingface.co",
        )
    print("Model downloads complete. Rerun the same command to resume interrupted downloads.")


if __name__ == "__main__":
    main()
