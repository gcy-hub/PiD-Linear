# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared official PiD resolution grids and aspect classification."""
import math
from typing import List, Tuple
from .utils import IMAGE_RES_SIZE_INFO

_MULTI_RES_LEVELS: List[int] = [2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096]
_MULTI_RES_MAX_RESOLUTIONS: tuple[str, ...] = ("3072", "3840", "4096")
_ALIGN: int = 64

# (ar_string, w_ratio, h_ratio). Ordering kept consistent with InferAspectRatio.
_AR_RATIOS: List[Tuple[str, int, int]] = [
    ("1,1", 1, 1),
    ("4,3", 4, 3),
    ("3,4", 3, 4),
    ("16,9", 16, 9),
    ("9,16", 9, 16),
]


def _round_to_multiple(x: float, m: int) -> int:
    return int(round(x / m)) * m


def _normalize_max_resolution(max_resolution: str) -> str:
    max_resolution = str(max_resolution)
    if max_resolution not in _MULTI_RES_MAX_RESOLUTIONS:
        raise ValueError(
            f"Unsupported multi-resolution max_resolution={max_resolution}. "
            f"Expected one of {_MULTI_RES_MAX_RESOLUTIONS}."
        )
    return max_resolution


def _compute_target(
    level: int,
    ar: str,
    w_r: int,
    h_r: int,
    ar_max_wh: dict[str, Tuple[int, int]],
) -> Tuple[int, int]:
    """Compute (W, H) for a given level and AR via area-scale formula, snap-64,
    then clip independently by the selected per-AR OOM ceiling.
    """
    r = w_r / h_r
    w = level * math.sqrt(r)
    h = level / math.sqrt(r)
    w = _round_to_multiple(w, _ALIGN)
    h = _round_to_multiple(h, _ALIGN)
    w_max, h_max = ar_max_wh[ar]
    w = min(w, w_max)
    h = min(h, h_max)
    return (w, h)


def _build_grid(max_resolution: str) -> dict:
    """Returns dict[ar_string -> list[(level, W, H)]] sorted descending by level
    (so largest-fit search walks from biggest to smallest).
    """
    max_resolution = _normalize_max_resolution(max_resolution)
    max_level = int(max_resolution)
    levels = [level for level in _MULTI_RES_LEVELS if level <= max_level]
    ar_max_wh = IMAGE_RES_SIZE_INFO[max_resolution]

    grid = {}
    for ar, w_r, h_r in _AR_RATIOS:
        entries = []
        for L in levels:
            w, h = _compute_target(L, ar, w_r, h_r, ar_max_wh)
            entries.append((L, w, h))
        entries.sort(key=lambda e: e[0], reverse=True)
        grid[ar] = entries
    return grid


# Module-level constants. Each grid is dict[ar -> list[(level, W, H)]], sorted
# descending by level. Keep MULTI_RES_GRID as the legacy 4096-grid alias.
MULTI_RES_GRID_BY_MAX_RESOLUTION: dict[str, dict] = {
    max_resolution: _build_grid(max_resolution) for max_resolution in _MULTI_RES_MAX_RESOLUTIONS
}
MULTI_RES_GRID: dict = MULTI_RES_GRID_BY_MAX_RESOLUTION["4096"]


# -----------------------------------------------------------------------------
# AR classification (mirrors InferAspectRatio at add_aspect_ratio.py:80)
# -----------------------------------------------------------------------------

_PREDEFINED_AR_TO_RATIO: List[Tuple[str, float]] = [
    ("16,9", 16 / 9),
    ("4,3", 4 / 3),
    ("1,1", 1.0),
    ("3,4", 3 / 4),
    ("9,16", 9 / 16),
]


def _classify_aspect_ratio(w: int, h: int) -> str:
    """Classify (w, h) into the closest predefined AR via log-space distance."""
    log_ratio = math.log(w / h)
    best_ar = _PREDEFINED_AR_TO_RATIO[0][0]
    best_dist = float("inf")
    for ar, ref_ratio in _PREDEFINED_AR_TO_RATIO:
        dist = abs(log_ratio - math.log(ref_ratio))
        if dist < best_dist:
            best_dist = dist
            best_ar = ar
    return best_ar
