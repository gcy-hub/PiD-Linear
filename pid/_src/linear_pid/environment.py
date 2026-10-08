"""Pinned operator provenance and recorded runtime versions."""

import importlib.metadata
import json
import os

import torch

from pid._src.linear_pid.attention import FLA_COMMIT


def check_runtime():
    versions = {
        package: importlib.metadata.version(package)
        for package in ["torch", "triton", "transformers", "diffusers", "swanlab", "flash-linear-attention"]
    }
    distribution = importlib.metadata.distribution("flash-linear-attention")
    origin = json.loads(distribution.read_text("direct_url.json") or "{}")
    if origin.get("vcs_info", {}).get("commit_id") != FLA_COMMIT:
        raise RuntimeError(f"FLA must be installed from commit {FLA_COMMIT}; run scripts/setup_linear_pid_env.sh")
    if torch.__version__.split("+")[0] != "2.10.0":
        raise RuntimeError("Linear-PiD v1 requires the separate Python 3.12 / PyTorch 2.10 environment")
    from fla.ops.kda import chunk_kda  # noqa: F401
    from fla.utils import FLA_DISABLE_TENSOR_CACHE

    versions["fla_commit"] = FLA_COMMIT
    versions["cuda"] = torch.version.cuda
    versions["fla_disable_tensor_cache"] = bool(FLA_DISABLE_TENSOR_CACHE)
    versions["cuda_launch_blocking"] = os.environ.get("CUDA_LAUNCH_BLOCKING") == "1"
    return versions
