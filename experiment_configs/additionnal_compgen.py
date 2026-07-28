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
# NOTE: We use the same shape conditionals as in the old original experiments.
#       In the old original experiments, we did NOT use the two last stronger constraints that made the
#       objects simpler and thus better for analysis, which we did in the recent experiments.
SHAPE_CONDITIONALS = [
    "is_shape_less_than_6_rows",
    "is_shape_less_than_6_cols",
    "is_shape_more_than_2_cell",
    "is_shape_fully_connected",
    # "is_shape_evenly_colored",
    # "is_shape_cross_or_rectangle"   # this is a strong constraint over the shapes, forcing simpler shapes
]


def _base_cfg(n_examples: int, transformations: list[list[str]]) -> DatasetConfig:
    return DatasetConfig(
        env_format="grid",
        n_examples=n_examples,
        batch_size=16,
        min_grid_size=20,
        max_grid_size=20,
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
# NOTE: Depending on the generator and models' failure/success, use a different set of atomic transformations.
#       Currently we use: TRANSLATE + MIRROR + ROT90.
#
#       The choice of families matches that of the old original setting and experiment 
#       C3-1 "From Composite Tasks to Deeper Composite Tasks". However, here we use almost all of the
#       atomic transformations part of those families.
#
#       If we want to create more instances of experiments, we would transformations from different families.




# ------------------------------
# Setting 5: Depth Distribution Variety
# ------------------------------
compgen_experiments_s5: list[ExperimentEntry] = []

# ------------------------------
# Setting 5, Exp 1-1
# ID:  atomic translate_*, mirror_*, rot90 + all depth-2
# OOD: ["translate_up", "rot90", "mirror_horizontal"]
# Goal: depth extrapolation from depths 1-2 --> 3.
# NOTE: implementation should use max task-token sequence length of 3 with an identity token for filling.
# ------------------------------
_atomics_s5_1_1 = TRANSLATE + MIRROR + ROT90
_id_s5_1_1 = (
    [[a] for a in _atomics_s5_1_1]
    + generate_combinations(_atomics_s5_1_1, depth=2)
)
_ood_s5_1_1 = [["translate_up", "rot90", "mirror_horizontal"]]

compgen_experiments_s5.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_id_s5_1_1),  setting=5, experiment=1, split="train", paired_splits=True))
compgen_experiments_s5.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_ood_s5_1_1), setting=5, experiment=1, split="test",  paired_splits=True))


# ------------------------------
# Setting 5, Exp 2-1
# ID: atomic translate_*, mirror_*, rot90 + all depth-2 + all depth-3
# OOD: ["translate_up", "rot90", "mirror_horizontal", "translate_right"]
# Goal: depth extrapolation from depths 1-3 --> 4.
# NOTE: implementation should use max task-token sequence length of 4 with an identity token for filling.
# ------------------------------
_atomics_s5_2_1 = TRANSLATE + MIRROR + ROT90
_id_s5_2_1 = (
    [[a] for a in _atomics_s5_2_1]
    + generate_combinations(_atomics_s5_2_1, depth=2)
    + generate_combinations(_atomics_s5_2_1, depth=3)
)
_ood_s5_2_1 = [["translate_up", "rot90", "mirror_horizontal", "translate_right"]]

compgen_experiments_s5.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_id_s5_2_1),  setting=5, experiment=2, split="train", paired_splits=True))
compgen_experiments_s5.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_ood_s5_2_1), setting=5, experiment=2, split="test",  paired_splits=True))

# ------------------------------
# Setting 5, Exp 3-1
# ID: atomic translate_*, mirror_*, rot90 + all depth-2 + all depth-3 + all depth-4
# OOD: ["translate_up", "rot90", "mirror_horizontal", "translate_right", "rot90"]
# Goal: depth extrapolation from depths 1-4 --> 5.
# NOTE: implementation should use max task-token sequence length of 5 with an identity token for filling.
# ------------------------------
_atomics_s5_3_1 = TRANSLATE + MIRROR + ROT90
_id_s5_3_1 = (
    [[a] for a in _atomics_s5_3_1]
    + generate_combinations(_atomics_s5_3_1, depth=2)
    + generate_combinations(_atomics_s5_3_1, depth=3)
    + generate_combinations(_atomics_s5_3_1, depth=4)
)
_ood_s5_3_1 = [["translate_up", "rot90", "mirror_horizontal", "translate_right", "rot90"]]

compgen_experiments_s5.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_id_s5_3_1),  setting=5, experiment=3, split="train", paired_splits=True))
compgen_experiments_s5.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_ood_s5_3_1), setting=5, experiment=3, split="test",  paired_splits=True))

# ------------------------------
# Setting 5, Exp 4-1
# ID: atomic translate_*, mirror_*, rot90 + all depth-2 + all depth-3 + all depth-4 + all depth-5
# OOD: ["translate_up", "rot90", "mirror_horizontal", "translate_right", "rot90", "mirror_vertical"]
# Goal: depth extrapolation from depths 1-5 --> 6.
# NOTE: implementation should use max task-token sequence length of 6 with an identity token for filling.
# ------------------------------
_atomics_s5_4_1 = TRANSLATE + MIRROR + ROT90
_id_s5_4_1 = (
    [[a] for a in _atomics_s5_4_1]
    + generate_combinations(_atomics_s5_4_1, depth=2)
    + generate_combinations(_atomics_s5_4_1, depth=3)
    + generate_combinations(_atomics_s5_4_1, depth=4)
    + generate_combinations(_atomics_s5_4_1, depth=5)
)
_ood_s5_4_1 = [["translate_up", "rot90", "mirror_horizontal", "translate_right", "rot90", "mirror_vertical"]]

compgen_experiments_s5.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_id_s5_4_1),  setting=5, experiment=4, split="train", paired_splits=True))
compgen_experiments_s5.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_ood_s5_4_1), setting=5, experiment=4, split="test",  paired_splits=True))




# ------------------------------
# Setting 6: Further Depth Extrapolation
# ------------------------------
compgen_experiments_s6: list[ExperimentEntry] = []

# ------------------------------
# Setting 6, Exp 1-1
# ID:  atomic translate_*, mirror_*, rot90 + all depth-2
# OOD: ["translate_up", "rot90", "mirror_horizontal"]
# Goal: depth extrapolation from depths 1-2 --> 3.
# NOTE: implementation should use max task-token sequence length of 3 with an identity token for filling.
# NOTE: This experiment would be identical to Setting 5, Exp 1-1, but we keep it here for completeness, as
#       to not make it confusing. However, in practice, except if we want to see how two randomly generated
#       datasets with the exactsame overall config compare, it can be useful to run experiments on it too.
#       Otherwise, simply report the same model evaluation results as those we will get for Setting 5, Exp 1-1.
# ------------------------------
_atomics_s6_1_1 = TRANSLATE + MIRROR + ROT90
_id_s6_1_1 = (
    [[a] for a in _atomics_s6_1_1]
    + generate_combinations(_atomics_s6_1_1, depth=2)
)
_ood_s6_1_1 = [["translate_up", "rot90", "mirror_horizontal"]]

compgen_experiments_s6.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_id_s6_1_1),  setting=6, experiment=1, split="train", paired_splits=True))
compgen_experiments_s6.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_ood_s6_1_1), setting=6, experiment=1, split="test",  paired_splits=True))

# ------------------------------
# Setting 6, Exp 2-1
# ID: atomic translate_*, mirror_*, rot90 + all depth-2 + all depth-3
# OOD: ["translate_up", "rot90", "mirror_horizontal", "translate_right", "rot90"]
# Goal: depth extrapolation from depths 1-3 --> 5.
# NOTE: implementation should use max task-token sequence length of 5 with an identity token for filling.
# ------------------------------
_atomics_s6_2_1 = TRANSLATE + MIRROR + ROT90
_id_s6_2_1 = (
    [[a] for a in _atomics_s6_2_1]
    + generate_combinations(_atomics_s6_2_1, depth=2)
    + generate_combinations(_atomics_s6_2_1, depth=3)
)
_ood_s6_2_1 = [["translate_up", "rot90", "mirror_horizontal", "translate_right", "rot90"]]

compgen_experiments_s6.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_id_s6_2_1),  setting=6, experiment=2, split="train", paired_splits=True))
compgen_experiments_s6.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_ood_s6_2_1), setting=6, experiment=2, split="test",  paired_splits=True))

# ------------------------------
# Setting 6, Exp 3-1
# ID: atomic translate_*, mirror_*, rot90 + all depth-2 + all depth-3 + all depth-4
# OOD: ["translate_up", "rot90", "mirror_horizontal", "translate_right", "rot90", "mirror_vertical", "translate_down"]
# Goal: depth extrapolation from depths 1-4 --> 7.
# NOTE: implementation should use max task-token sequence length of 7 with an identity token for filling.
# ------------------------------
_atomics_s6_3_1 = TRANSLATE + MIRROR + ROT90
_id_s6_3_1 = (
    [[a] for a in _atomics_s6_3_1]
    + generate_combinations(_atomics_s6_3_1, depth=2)
    + generate_combinations(_atomics_s6_3_1, depth=3)
    + generate_combinations(_atomics_s6_3_1, depth=4)
)
_ood_s6_3_1 = [["translate_up", "rot90", "mirror_horizontal", "translate_right", "rot90", "mirror_vertical", "translate_down"]]

compgen_experiments_s6.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_id_s6_3_1),  setting=6, experiment=3, split="train", paired_splits=True))
compgen_experiments_s6.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_ood_s6_3_1), setting=6, experiment=3, split="test",  paired_splits=True))

# ------------------------------
# Setting 6, Exp 4-1
# ID: atomic translate_*, mirror_*, rot90 + all depth-2 + all depth-3 + all depth-4 + all depth-5
# OOD: ["translate_up", "rot90", "mirror_horizontal", "translate_right", "rot90", "mirror_vertical", "translate_down", "rot90", "mirror_horizontal"]
# Goal: depth extrapolation from depths 1-5 --> 9.
# NOTE: implementation should use max task-token sequence length of 9 with an identity token for filling.
# ------------------------------
_atomics_s6_4_1 = TRANSLATE + MIRROR + ROT90
_id_s6_4_1 = (
    [[a] for a in _atomics_s6_4_1]
    + generate_combinations(_atomics_s6_4_1, depth=2)
    + generate_combinations(_atomics_s6_4_1, depth=3)
    + generate_combinations(_atomics_s6_4_1, depth=4)
    + generate_combinations(_atomics_s6_4_1, depth=5)
)
_ood_s6_4_1 = [["translate_up", "rot90", "mirror_horizontal", "translate_right", "rot90", "mirror_vertical", "translate_down", "rot90", "mirror_horizontal"]]

compgen_experiments_s6.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_id_s6_4_1),  setting=6, experiment=4, split="train", paired_splits=True))
compgen_experiments_s6.append(ExperimentEntry(cfg=_base_cfg(n_examples=1, transformations=_ood_s6_4_1), setting=6, experiment=4, split="test",  paired_splits=True))
