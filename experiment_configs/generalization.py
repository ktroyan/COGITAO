from arcworld.config import DatasetConfig

from .entry import ExperimentEntry

_BASE = dict(
    n_examples=1,
    shape_compulsory_conditionals=[
        "is_shape_less_than_6_rows",
        "is_shape_less_than_6_cols",
        "is_shape_fully_connected",
    ],
)


def make_entry(transform, grid_size_min, grid_size_max, n_obj_min, n_obj_max,
               setting, exp_number, split, compulsory_conditions=None):
    kw = dict(**_BASE)
    if compulsory_conditions is not None:
        kw["shape_compulsory_conditionals"] = compulsory_conditions
    cfg = DatasetConfig(
        allowed_combinations=transform,
        min_n_shapes_per_grid=n_obj_min,
        max_n_shapes_per_grid=n_obj_max,
        min_grid_size=grid_size_min,
        max_grid_size=grid_size_max,
        **kw,
    )
    return ExperimentEntry(cfg=cfg, setting=setting, experiment=exp_number, split=split)


generalization_configs: list[ExperimentEntry] = []

# === Setting 1 === N shapes generalization

exps_1 = [
    ([["translate_up"]],      15, 15, 1, 2, 1, 1, "train"),
    ([["translate_up"]],      15, 15, 3, 4, 1, 1, "test"),
    ([["rot90"]],             15, 15, 1, 2, 1, 2, "train"),
    ([["rot90"]],             15, 15, 3, 4, 1, 2, "test"),
    ([["mirror_horizontal"]], 15, 15, 1, 2, 1, 3, "train"),
    ([["mirror_horizontal"]], 15, 15, 3, 4, 1, 3, "test"),
    ([["crop_top_side"]],     15, 15, 1, 2, 1, 4, "train"),
    ([["crop_top_side"]],     15, 15, 3, 4, 1, 4, "test"),
    ([["extend_contours_same_color"]], 15, 15, 1, 2, 1, 5, "train"),
    ([["extend_contours_same_color"]], 15, 15, 3, 4, 1, 5, "test"),
]

for transform, grid_min, grid_max, n_min, n_max, setting, exp, split in exps_1:
    generalization_configs.append(make_entry(
        transform, grid_min, grid_max, n_min, n_max, setting, exp, split))

# === Setting 2 === Grid size generalization

exps_2 = [
    ([["translate_up"]],      10, 15, 2, 2, 2, 1, "train"),
    ([["translate_up"]],      16, 20, 2, 2, 2, 1, "test"),
    ([["rot90"]],             10, 15, 2, 2, 2, 2, "train"),
    ([["rot90"]],             16, 20, 2, 2, 2, 2, "test"),
    ([["mirror_horizontal"]], 10, 15, 2, 2, 2, 3, "train"),
    ([["mirror_horizontal"]], 16, 20, 2, 2, 2, 3, "test"),
    ([["crop_top_side"]],     10, 15, 2, 2, 2, 4, "train"),
    ([["crop_top_side"]],     16, 20, 2, 2, 2, 4, "test"),
    ([["extend_contours_same_color"]], 10, 15, 2, 2, 2, 5, "train"),
    ([["extend_contours_same_color"]], 16, 20, 2, 2, 2, 5, "test"),
]

for transform, grid_min, grid_max, n_min, n_max, setting, exp, split in exps_2:
    generalization_configs.append(make_entry(
        transform, grid_min, grid_max, n_min, n_max, setting, exp, split))

# === Setting 3 === Object Dimension Generalization

exps_3 = [
    ([["translate_up"]],      15, 15, 2, 2, 3, 1, "train",
     ["is_shape_less_than_5_rows", "is_shape_less_than_5_cols", "is_shape_fully_connected"]),
    ([["translate_up"]],      15, 15, 2, 2, 3, 1, "test",
     ["is_shape_less_than_9_rows", "is_shape_less_than_9_cols", "is_shape_more_than_5_rows", "is_shape_more_than_5_cols", "is_shape_fully_connected"]),
    ([["rot90"]],             15, 15, 2, 2, 3, 2, "train",
     ["is_shape_less_than_5_rows", "is_shape_less_than_5_cols", "is_shape_fully_connected"]),
    ([["rot90"]],             15, 15, 2, 2, 3, 2, "test",
     ["is_shape_less_than_9_rows", "is_shape_less_than_9_cols", "is_shape_more_than_5_rows", "is_shape_more_than_5_cols", "is_shape_fully_connected"]),
    ([["mirror_horizontal"]], 15, 15, 2, 2, 3, 3, "train",
     ["is_shape_less_than_5_rows", "is_shape_less_than_5_cols", "is_shape_fully_connected"]),
    ([["mirror_horizontal"]], 15, 15, 2, 2, 3, 3, "test",
     ["is_shape_less_than_9_rows", "is_shape_less_than_9_cols", "is_shape_more_than_5_rows", "is_shape_more_than_5_cols", "is_shape_fully_connected"]),
    ([["crop_top_side"]],     15, 15, 2, 2, 3, 4, "train",
     ["is_shape_less_than_5_rows", "is_shape_less_than_5_cols", "is_shape_fully_connected"]),
    ([["crop_top_side"]],     15, 15, 2, 2, 3, 4, "test",
     ["is_shape_less_than_9_rows", "is_shape_less_than_9_cols", "is_shape_more_than_5_rows", "is_shape_more_than_5_cols", "is_shape_fully_connected"]),
    ([["extend_contours_same_color"]], 15, 20, 2, 2, 3, 5, "train",
     ["is_shape_less_than_5_rows", "is_shape_less_than_5_cols", "is_shape_fully_connected"]),
    ([["extend_contours_same_color"]], 15, 20, 2, 2, 3, 5, "test",
     ["is_shape_less_than_9_rows", "is_shape_less_than_9_cols", "is_shape_more_than_5_rows", "is_shape_more_than_5_cols", "is_shape_fully_connected"]),
]

for transform, grid_min, grid_max, n_min, n_max, setting, exp, split, conds in exps_3:
    generalization_configs.append(make_entry(
        transform, grid_min, grid_max, n_min, n_max, setting, exp, split, conds))

# === Setting 4 === Object Complexity Generalization

exps_4 = [
    ([["translate_up"]],      15, 15, 2, 2, 4, 1, "train",
     ["is_shape_symmetric", "is_shape_evenly_colored", "is_shape_fully_connected"]),
    ([["translate_up"]],      15, 15, 2, 2, 4, 1, "test",
     ["is_shape_not_symmetric", "is_shape_not_evenly_colored", "is_shape_fully_connected"]),
    ([["rot90"]],             15, 15, 2, 2, 4, 2, "train",
     ["is_shape_symmetric", "is_shape_evenly_colored", "is_shape_fully_connected"]),
    ([["rot90"]],             15, 15, 2, 2, 4, 2, "test",
     ["is_shape_not_symmetric", "is_shape_not_evenly_colored", "is_shape_fully_connected"]),
    ([["mirror_horizontal"]], 15, 15, 2, 2, 4, 3, "train",
     ["is_shape_symmetric", "is_shape_evenly_colored", "is_shape_fully_connected"]),
    ([["mirror_horizontal"]], 15, 15, 2, 2, 4, 3, "test",
     ["is_shape_not_symmetric", "is_shape_not_evenly_colored", "is_shape_fully_connected"]),
    ([["crop_top_side"]],     15, 15, 2, 2, 4, 4, "train",
     ["is_shape_symmetric", "is_shape_evenly_colored", "is_shape_fully_connected"]),
    ([["crop_top_side"]],     15, 15, 2, 2, 4, 4, "test",
     ["is_shape_not_symmetric", "is_shape_not_evenly_colored", "is_shape_fully_connected"]),
    ([["extend_contours_same_color"]], 15, 15, 2, 2, 4, 5, "train",
     ["is_shape_symmetric", "is_shape_evenly_colored", "is_shape_fully_connected"]),
    ([["extend_contours_same_color"]], 15, 15, 2, 2, 4, 5, "test",
     ["is_shape_not_symmetric", "is_shape_not_evenly_colored", "is_shape_fully_connected"]),
]

for transform, grid_min, grid_max, n_min, n_max, setting, exp, split, conds in exps_4:
    generalization_configs.append(make_entry(
        transform, grid_min, grid_max, n_min, n_max, setting, exp, split, conds))

# === Setting 5 === All rules mixed Generalization

exps_5 = [
    ([["translate_up"]],      10, 15, 1, 2, 5, 1, "train",
     ["is_shape_symmetric", "is_shape_evenly_colored", "is_shape_less_than_5_rows", "is_shape_less_than_5_cols", "is_shape_fully_connected"]),
    ([["translate_up"]],      16, 20, 3, 4, 5, 1, "test",
     ["is_shape_not_symmetric", "is_shape_not_evenly_colored", "is_shape_less_than_9_rows", "is_shape_less_than_9_cols", "is_shape_more_than_5_rows", "is_shape_more_than_5_cols", "is_shape_fully_connected"]),
    ([["rot90"]],             10, 15, 1, 2, 5, 2, "train",
     ["is_shape_symmetric", "is_shape_evenly_colored", "is_shape_less_than_5_rows", "is_shape_less_than_5_cols", "is_shape_fully_connected"]),
    ([["rot90"]],             16, 20, 3, 4, 5, 2, "test",
     ["is_shape_not_symmetric", "is_shape_not_evenly_colored", "is_shape_less_than_9_rows", "is_shape_less_than_9_cols", "is_shape_more_than_5_rows", "is_shape_more_than_5_cols", "is_shape_fully_connected"]),
    ([["mirror_horizontal"]], 10, 15, 1, 2, 5, 3, "train",
     ["is_shape_symmetric", "is_shape_evenly_colored", "is_shape_less_than_5_rows", "is_shape_less_than_5_cols", "is_shape_fully_connected"]),
    ([["mirror_horizontal"]], 16, 20, 3, 4, 5, 3, "test",
     ["is_shape_not_symmetric", "is_shape_not_evenly_colored", "is_shape_less_than_9_rows", "is_shape_less_than_9_cols", "is_shape_more_than_5_rows", "is_shape_more_than_5_cols", "is_shape_fully_connected"]),
    ([["crop_top_side"]],     10, 15, 1, 2, 5, 4, "train",
     ["is_shape_symmetric", "is_shape_evenly_colored", "is_shape_less_than_5_rows", "is_shape_less_than_5_cols", "is_shape_fully_connected"]),
    ([["crop_top_side"]],     16, 20, 3, 4, 5, 4, "test",
     ["is_shape_not_symmetric", "is_shape_not_evenly_colored", "is_shape_less_than_9_rows", "is_shape_less_than_9_cols", "is_shape_more_than_5_rows", "is_shape_more_than_5_cols", "is_shape_fully_connected"]),
    ([["extend_contours_same_color"]], 10, 15, 1, 2, 5, 5, "train",
     ["is_shape_symmetric", "is_shape_evenly_colored", "is_shape_less_than_5_rows", "is_shape_less_than_5_cols", "is_shape_fully_connected"]),
    ([["extend_contours_same_color"]], 17, 20, 3, 3, 5, 5, "test",
     ["is_shape_not_symmetric", "is_shape_not_evenly_colored", "is_shape_less_than_9_rows", "is_shape_less_than_9_cols", "is_shape_more_than_5_rows", "is_shape_more_than_5_cols", "is_shape_fully_connected"]),
]

for transform, grid_min, grid_max, n_min, n_max, setting, exp, split, conds in exps_5:
    generalization_configs.append(make_entry(
        transform, grid_min, grid_max, n_min, n_max, setting, exp, split, conds))
