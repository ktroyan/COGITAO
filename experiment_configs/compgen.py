from arcworld.config import DatasetConfig
from experiment_configs.entry import ExperimentEntry

from itertools import product

# -------------------------------
# Helpers
# -------------------------------
def generate_combinations(atomics, depth, excluded_ordered=None, excluded_unordered=None):
    """
    Generate all depth-n ordered combinations of atomics.
    excluded_ordered: exact combos (as lists) to exclude by sequence.
    excluded_unordered: element-sets — exclude any combo whose set contains any such set as a subset.
    """
    all_combos = [list(c) for c in product(atomics, repeat=depth)]

    if excluded_ordered:
        all_combos = [c for c in all_combos if c not in excluded_ordered]

    if excluded_unordered:
        def is_unordered_excluded(combo):
            combo_set = set(combo)
            return any(set(excl).issubset(combo_set) for excl in excluded_unordered)
        all_combos = [c for c in all_combos if not is_unordered_excluded(c)]

    return all_combos


# -------------------------------
# Atomic Transformation Families
# -------------------------------
TRANSLATE  = ["translate_up", "translate_left", "translate_right", "translate_down"]
PAD        = ["pad_top", "pad_left", "pad_right", "pad_bottom"]
MIRROR     = ["mirror_horizontal", "mirror_vertical"]
ROT90      = ["rot90"]
CROP       = ["crop_top_side", "crop_right_side", "crop_left_side", "crop_bottom_side"]
FILL       = ["fill_holes_same_color", "fill_holes_different_color"]
EXTEND     = ["extend_contours_same_color", "extend_contours_different_color"]


# -------------------------------
# Configs
# -------------------------------
SHAPE_CONDITIONALS = [
    "is_shape_less_than_6_rows",
    "is_shape_less_than_6_cols",
    "is_shape_more_than_2_cell",
    "is_shape_fully_connected",
    "is_shape_evenly_colored",
    "is_shape_cross_or_rectangle"   # this is a strong constraint over the shapes, forcing simpler shapes
]


def _base_cfg(n_examples: int, transformations: list[list[str]]) -> DatasetConfig:
    return DatasetConfig(
        env_format="grid",
        n_examples=n_examples,
        batch_size=16,
        min_grid_size=15,
        max_grid_size=15,
        min_n_shapes_per_grid=2,
        max_n_shapes_per_grid=2,
        allowed_combinations=transformations,
        min_transformation_depth=None,
        max_transformation_depth=None,
        shape_compulsory_conditionals=SHAPE_CONDITIONALS,
    )


# -------------------------------
# Experiments
# -------------------------------
compgen_experiments_s1: list[ExperimentEntry] = []


# ------------------------------
# Setting 1: Atomic and Composite Transformations
# ------------------------------

# ------------------------------
# Exp 1-1
# ID:  atomic translate_* + all depth-2 except [translate_up, translate_right]
# OOD: [translate_up, translate_right]
# Goal: task recognition & permutation invariance — single family, the seen equivalent [translate_right, translate_up] is in ID.
# ------------------------------
_atomics_1_1 = TRANSLATE
_id_1_1 = (
    [[a] for a in _atomics_1_1]
    + generate_combinations(_atomics_1_1, depth=2, excluded_ordered=[["translate_up", "translate_right"]])
)
_ood_1_1 = [["translate_up", "translate_right"]]

compgen_experiments_s1.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_id_1_1),  setting=1, experiment=1, split="train", paired_splits=True))
compgen_experiments_s1.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_ood_1_1), setting=1, experiment=1, split="test",  paired_splits=True))


# ------------------------------
# Exp 1-2
# ID:  atomic translate_*, mirror_*, rot90 + all depth-2 except [rot90, translate_up]
# OOD: [rot90, translate_up]
# Goal: task recognition & permutation invariance — multi-family, the seen equivalent [translate_up, rot90] is in ID.
# ------------------------------
_atomics_1_2 = TRANSLATE + MIRROR + ROT90
_id_1_2 = (
    [[a] for a in _atomics_1_2]
    + generate_combinations(_atomics_1_2, depth=2, excluded_ordered=[["rot90", "translate_up"]])
)
_ood_1_2 = [["rot90", "translate_up"]]

compgen_experiments_s1.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_id_1_2),  setting=1, experiment=2, split="train", paired_splits=True))
compgen_experiments_s1.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_ood_1_2), setting=1, experiment=2, split="test",  paired_splits=True))


# ------------------------------
# Exp 2-1
# ID:  atomic translate_*, mirror_*, rot90 + all depth-2 except [rot90, mirror_horizontal]
# OOD: [rot90, mirror_horizontal]
# Goal: semi-novel composite (depth 2) — non-commutative, only the reverted order [mirror_horizontal, rot90] is in ID.
# ------------------------------
_atomics_2_1 = TRANSLATE + MIRROR + ROT90
_id_2_1 = (
    [[a] for a in _atomics_2_1]
    + generate_combinations(_atomics_2_1, depth=2, excluded_ordered=[["rot90", "mirror_horizontal"]])
)
_ood_2_1 = [["rot90", "mirror_horizontal"]]

compgen_experiments_s1.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_id_2_1),  setting=2, experiment=1, split="train", paired_splits=True))
compgen_experiments_s1.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_ood_2_1), setting=2, experiment=1, split="test",  paired_splits=True))


# ------------------------------
# Exp 3-1
# ID:  atomic translate_* + all depth-2 except {translate_up, translate_right} in any order
# OOD: [translate_up, translate_right]
# Goal: novel composite (depth 2) — single family, neither ordering seen during training.
# ------------------------------
_atomics_3_1 = TRANSLATE
_id_3_1 = (
    [[a] for a in _atomics_3_1]
    + generate_combinations(_atomics_3_1, depth=2, excluded_unordered=[["translate_up", "translate_right"]])
)
_ood_3_1 = [["translate_up", "translate_right"]]

compgen_experiments_s1.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_id_3_1),  setting=3, experiment=1, split="train", paired_splits=True))
compgen_experiments_s1.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_ood_3_1), setting=3, experiment=1, split="test",  paired_splits=True))


# ------------------------------
# Exp 3-2
# ID:  atomic pad_* + all depth-2 except {pad_top, pad_right} in any order
# OOD: [pad_top, pad_right]
# Goal: novel composite (depth 2) — single family, neither ordering seen during training.
# ------------------------------
_atomics_3_2 = PAD
_id_3_2 = (
    [[a] for a in _atomics_3_2]
    + generate_combinations(_atomics_3_2, depth=2, excluded_unordered=[["pad_top", "pad_right"]])
)
_ood_3_2 = [["pad_top", "pad_right"]]

compgen_experiments_s1.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_id_3_2),  setting=3, experiment=2, split="train", paired_splits=True))
compgen_experiments_s1.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_ood_3_2), setting=3, experiment=2, split="test",  paired_splits=True))


# ------------------------------
# Exp 3-3
# ID:  atomic translate_*, mirror_*, rot90 + all depth-2 except {rot90, mirror_vertical} in any order
# OOD: [rot90, mirror_vertical]
# Goal: novel composite (depth 2) — multi-family, neither ordering seen during training.
# ------------------------------
_atomics_3_3 = TRANSLATE + MIRROR + ROT90
_id_3_3 = (
    [[a] for a in _atomics_3_3]
    + generate_combinations(_atomics_3_3, depth=2, excluded_unordered=[["rot90", "mirror_vertical"]])
)
_ood_3_3 = [["rot90", "mirror_vertical"]]

compgen_experiments_s1.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_id_3_3),  setting=3, experiment=3, split="train", paired_splits=True))
compgen_experiments_s1.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_ood_3_3), setting=3, experiment=3, split="test",  paired_splits=True))


# ------------------------------
# Exp 4-1
# ID:  atomic translate_*, mirror_*, rot90 + all depth-2 + all depth-3 excluding those containing {translate_up, mirror_horizontal, rot90} in any order
# OOD: [translate_up, mirror_horizontal, rot90]
# Goal: novel composite (depth 3) — depth-2 sub-transformations seen, but not this depth-3 combination.
# ------------------------------
_atomics_4_1 = TRANSLATE + MIRROR + ROT90
_id_4_1 = (
    [[a] for a in _atomics_4_1]
    + generate_combinations(_atomics_4_1, depth=2)
    + generate_combinations(_atomics_4_1, depth=3, excluded_unordered=[["translate_up", "mirror_horizontal", "rot90"]])
)
_ood_4_1 = [["translate_up", "mirror_horizontal", "rot90"]]

compgen_experiments_s1.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_id_4_1),  setting=4, experiment=1, split="train", paired_splits=True))
compgen_experiments_s1.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_ood_4_1), setting=4, experiment=1, split="test",  paired_splits=True))


# ------------------------------
# Exp 5-1
# ID:  atomic translate_*, mirror_* + all depth-2 except {translate_up, mirror_horizontal} in any order
# OOD: [translate_up, mirror_horizontal]
# Goal: novel composite — low ID variety (2 families).
# ------------------------------
_atomics_5_1 = TRANSLATE + MIRROR
_id_5_1 = (
    [[a] for a in _atomics_5_1]
    + generate_combinations(_atomics_5_1, depth=2, excluded_unordered=[["translate_up", "mirror_horizontal"]])
)
_ood_5_1 = [["translate_up", "mirror_horizontal"]]

compgen_experiments_s1.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_id_5_1),  setting=5, experiment=1, split="train", paired_splits=True))
compgen_experiments_s1.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_ood_5_1), setting=5, experiment=1, split="test",  paired_splits=True))


# ------------------------------
# Exp 5-2
# ID:  atomic translate_*, mirror_*, rot90, crop_* + all depth-2 except {translate_up, mirror_horizontal} in any order
# OOD: [translate_up, mirror_horizontal]
# Goal: novel composite — medium ID variety (4 families).
# ------------------------------
_atomics_5_2 = TRANSLATE + MIRROR + ROT90 + CROP
_id_5_2 = (
    [[a] for a in _atomics_5_2]
    + generate_combinations(_atomics_5_2, depth=2, excluded_unordered=[["translate_up", "mirror_horizontal"]])
)
_ood_5_2 = [["translate_up", "mirror_horizontal"]]

compgen_experiments_s1.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_id_5_2),  setting=5, experiment=2, split="train", paired_splits=True))
compgen_experiments_s1.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_ood_5_2), setting=5, experiment=2, split="test",  paired_splits=True))


# ------------------------------
# Exp 5-3
# ID:  atomic translate_*, mirror_*, rot90, crop_*, fill_*, extend_* + all depth-2 except {translate_up, mirror_horizontal} in any order
# OOD: [translate_up, mirror_horizontal]
# Goal: novel composite — high ID variety (6 families).
# ------------------------------
_atomics_5_3 = TRANSLATE + MIRROR + ROT90 + CROP + FILL + EXTEND
_id_5_3 = (
    [[a] for a in _atomics_5_3]
    + generate_combinations(_atomics_5_3, depth=2, excluded_unordered=[["translate_up", "mirror_horizontal"]])
)
_ood_5_3 = [["translate_up", "mirror_horizontal"]]

compgen_experiments_s1.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_id_5_3),  setting=5, experiment=3, split="train", paired_splits=True))
compgen_experiments_s1.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_ood_5_3), setting=5, experiment=3, split="test",  paired_splits=True))

# ------------------------------
# Setting 2: Depth Extrapolation
# ------------------------------
compgen_experiments_s2: list[ExperimentEntry] = []

# ------------------------------
# Exp 1-1
# ID:  atomic translate_*, mirror_*, rot90 + all depth-2 (no exclusions)
# OOD: [translate_up, mirror_horizontal, rot90]
# Goal: depth extrapolation from depth 1 & 2 to 3.
# NOTE: implementation should use max task-token sequence length of 3 with an identity token.
# ------------------------------
_atomics_s2_1_1 = TRANSLATE + MIRROR + ROT90
_id_s2_1_1 = (
    [[a] for a in _atomics_s2_1_1]
    + generate_combinations(_atomics_s2_1_1, depth=2)
)
_ood_s2_1_1 = [["translate_up", "mirror_horizontal", "rot90"]]

compgen_experiments_s2.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_id_s2_1_1),  setting=1, experiment=1, split="train", paired_splits=True))
compgen_experiments_s2.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_ood_s2_1_1), setting=1, experiment=1, split="test",  paired_splits=True))


# ------------------------------
# Exp 2-1
# ID:  atomic translate_*, mirror_*, rot90 + all depth-2 + all depth-3 (no exclusions)
# OOD: [translate_up, mirror_horizontal, rot90, translate_up]
# Goal: depth extrapolation from depth 1, 2 & 3 to 4.
# NOTE: implementation should use max task-token sequence length of 4 with an identity token.
# ------------------------------
_atomics_s2_2_1 = TRANSLATE + MIRROR + ROT90
_id_s2_2_1 = (
    [[a] for a in _atomics_s2_2_1]
    + generate_combinations(_atomics_s2_2_1, depth=2)
    + generate_combinations(_atomics_s2_2_1, depth=3)
)
_ood_s2_2_1 = [["translate_up", "mirror_horizontal", "rot90", "translate_up"]]

compgen_experiments_s2.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_id_s2_2_1),  setting=2, experiment=1, split="train", paired_splits=True))
compgen_experiments_s2.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_ood_s2_2_1), setting=2, experiment=1, split="test",  paired_splits=True))
