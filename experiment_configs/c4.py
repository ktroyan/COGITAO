import itertools

from arcworld.config import DatasetConfig

from .entry import ExperimentEntry

_BASE = dict(
    min_n_shapes_per_grid=2,
    max_n_shapes_per_grid=2,
    n_examples=1,
    shape_compulsory_conditionals=[
        "is_shape_less_than_6_rows",
        "is_shape_less_than_6_cols",
        "is_shape_fully_connected",
    ],
)


def make_entry(combos, setting, exp_number, split, min_size=20, max_size=20):
    cfg = DatasetConfig(
        **_BASE,
        allowed_combinations=combos,
        min_grid_size=min_size,
        max_grid_size=max_size,
    )
    return ExperimentEntry(cfg=cfg, setting=setting, experiment=exp_number, split=split)


def all_single_and_double(transforms):
    singles = [[t] for t in transforms]
    doubles = [list(pair) for pair in itertools.combinations(transforms, 2)]
    return singles + doubles


base_transforms = [
    "mirror_horizontal",
    "rot90",
    "double_right",
]

extra_transforms = [
    "translate_up",
    "crop_top_side",
    "mirror_vertical",
    "pad_top",
    "translate_right",
    "pad_right",
    "fill_holes_same_color",
    "fill_holes_different_color",
    "change_shape_color",
    "empty_inside_pixels",
    "double_down",
    "extend_contours_same_color",
]

test_combo = [["double_right", "rot90"]]

exp_sizes = [3, 6, 9, 12, 15]

c4_configs: list[ExperimentEntry] = []

for exp_num, size in enumerate(exp_sizes, start=1):
    pool = base_transforms + extra_transforms[: size - len(base_transforms)]
    train_combos = all_single_and_double(pool)
    if ["double_right", "rot90"] in train_combos:
        train_combos.remove(["double_right", "rot90"])
    if ["fill_holes_different_color", "empty_inside_pixels"] in train_combos:
        train_combos.remove(["fill_holes_different_color", "empty_inside_pixels"])
    if ["fill_holes_same_color", "empty_inside_pixels"] in train_combos:
        train_combos.remove(["fill_holes_same_color", "empty_inside_pixels"])

    c4_configs.append(make_entry(train_combos, setting=4, exp_number=exp_num, split="train"))
    c4_configs.append(make_entry(test_combo, setting=4, exp_number=exp_num, split="test"))
