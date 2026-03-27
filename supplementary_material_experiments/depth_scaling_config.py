"""Depth-scaling experiment configuration.

Tests compositional depths 1 through 8 to demonstrate that the COGITAO
generator supports arbitrary composition depth, addressing the reviewer
concern that "the main benchmark only evaluates compositions up to depth 3".

Design choices:
- **Size-preserving transforms only** (rot90, mirror_horizontal, mirror_vertical,
  change_shape_color) so that deeper chains don't require ever-larger grids.
  This isolates the effect of *compositional depth* from spatial scaling.
- **Grid size 20x20** — same as Setting 3 in the main compositionality
  benchmark, so results are directly comparable.
- **2 shapes per grid**, small shapes (< 6 rows/cols, fully connected).
- For each depth d, we enumerate all ordered d-tuples from the 4 transforms
  (with repetition). This means depth 1 has 4 combos, depth 2 has 16, etc.
  To keep generation tractable at higher depths we cap at 64 randomly sampled
  combos for d >= 4.
"""

import itertools
import random

from arcworld.config import DatasetConfig
from experiment_configs.entry import ExperimentEntry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIZE_PRESERVING_TRANSFORMS = [
    "rot90",
    "mirror_horizontal",
    "translate_up",
    "change_shape_color",
]

_BASE = dict(
    min_n_shapes_per_grid=2,
    max_n_shapes_per_grid=2,
    n_examples=1,
    min_grid_size=20,
    max_grid_size=20,
    shape_compulsory_conditionals=[
        "is_shape_less_than_6_rows",
        "is_shape_less_than_6_cols",
        "is_shape_fully_connected",
    ],
)

MAX_COMBOS_PER_DEPTH = 64  # cap to keep generation time reasonable
DEPTHS = list(range(1, 9))  # 1 through 8

# ---------------------------------------------------------------------------
# Build configs
# ---------------------------------------------------------------------------

# Fix seed so the subset of combos is reproducible across runs
_rng = random.Random(42)


def _combos_for_depth(depth: int) -> list[list[str]]:
    """Return transformation combinations for a given depth.

    For small depths (<=3) we enumerate all ordered tuples.
    For larger depths we sample a reproducible random subset.
    """
    all_combos = [list(c) for c in itertools.product(SIZE_PRESERVING_TRANSFORMS, repeat=depth)]
    if len(all_combos) <= MAX_COMBOS_PER_DEPTH:
        return all_combos
    return [list(c) for c in _rng.sample(all_combos, MAX_COMBOS_PER_DEPTH)]


def make_entry(combos, depth):
    cfg = DatasetConfig(
        **_BASE,
        allowed_combinations=combos,
    )
    return ExperimentEntry(
        cfg=cfg,
        setting=1,
        experiment=depth,  # experiment number = depth
        split="train",
        subdir=f"depth_{depth}",
    )


depth_scaling_configs: list[ExperimentEntry] = []

for d in DEPTHS:
    combos = _combos_for_depth(d)
    depth_scaling_configs.append(make_entry(combos, d))
