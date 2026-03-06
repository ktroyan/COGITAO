from arcworld.config import DatasetConfig

from .entry import ExperimentEntry

_BASE = dict(
    min_n_shapes_per_grid=2,
    max_n_shapes_per_grid=2,
    n_examples=1,
    min_grid_size=15,
    max_grid_size=15,
    shape_compulsory_conditionals=[
        "is_shape_less_than_6_rows",
        "is_shape_less_than_6_cols",
        "is_shape_fully_connected",
    ],
)


def make_entry(combos, setting, exp_number, split):
    cfg = DatasetConfig(**_BASE, allowed_combinations=combos)
    return ExperimentEntry(cfg=cfg, setting=setting, experiment=exp_number, split=split)


sample_efficiency_configs: list[ExperimentEntry] = [
    make_entry([["translate_up"]], 1, 1, "train"),
    make_entry([["rot90"]], 1, 2, "train"),
    make_entry([["mirror_horizontal"]], 1, 3, "train"),
    make_entry([["extend_contours_different_color"]], 1, 4, "train"),
    make_entry([["empty_inside_pixels"]], 1, 5, "train"),
    make_entry([["crop_top_side"]], 1, 6, "train"),
    make_entry([["fill_holes_different_color"]], 1, 7, "train"),
    make_entry([["double_up"]], 1, 8, "train"),
    make_entry([["change_shape_color"]], 1, 9, "train"),
    make_entry([["pad_shape"]], 1, 10, "train"),
]
