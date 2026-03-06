from arcworld.config import DatasetConfig

from .entry import ExperimentEntry

_BASE = dict(
    min_n_shapes_per_grid=2,
    max_n_shapes_per_grid=2,
    n_examples=1,
    shape_compulsory_conditionals=[
        "is_shape_less_than_9_rows",
        "is_shape_less_than_9_cols",
        "is_shape_fully_connected",
    ],
)


def make_entry(combos, setting, exp_number, split, min_size=15, max_size=15):
    cfg = DatasetConfig(
        **_BASE,
        allowed_combinations=combos,
        min_grid_size=min_size,
        max_grid_size=max_size,
    )
    return ExperimentEntry(
        cfg=cfg, setting=setting, experiment=exp_number, split=split,
        subdir=f"grid_size_{max_size}",
    )


# --- C1 combination definitions ---

c11_train = [["translate_up"], ["rot90"], ["mirror_horizontal"],
    ["translate_up", "translate_up"], ["rot90", "rot90"],
    ["mirror_horizontal", "mirror_horizontal"],
    ["translate_up", "mirror_horizontal"],
    ["rot90", "mirror_horizontal"],
    ["rot90", "translate_up"],
    ["mirror_horizontal", "rot90"]]
c11_test = [["translate_up", "rot90"]]

c12_train = [["pad_right"], ["fill_holes_different_color"], ["change_shape_color"],
    ["pad_right", "pad_right"],
    ["fill_holes_different_color", "fill_holes_different_color"],
    ["change_shape_color", "change_shape_color"],
    ["pad_right", "fill_holes_different_color"],
    ["fill_holes_different_color", "change_shape_color"],
    ["change_shape_color", "fill_holes_different_color"],
    ["change_shape_color", "pad_right"]]
c12_test = [["pad_right", "change_shape_color"]]

c13_train = [["rot90"], ["pad_top"], ["crop_bottom_side"],
    ["rot90", "rot90"],
    ["pad_top", "pad_top"],
    ["crop_bottom_side", "crop_bottom_side"],
    ["rot90", "pad_top"],
    ["pad_top", "crop_bottom_side"],
    ["crop_bottom_side", "rot90"]]
c13_test = [["rot90", "crop_bottom_side"]]

c14_train = [["double_right"], ["crop_contours"], ["change_shape_color"],
    ["double_right", "double_right"],
    ["crop_contours", "crop_contours"],
    ["change_shape_color", "change_shape_color"],
    ["double_right", "change_shape_color"],
    ["crop_contours", "change_shape_color"],
    ["change_shape_color", "double_right"]]
c14_test = [["double_right", "crop_contours"]]

c15_train = [["extend_contours_same_color"], ["pad_left"], ["mirror_vertical"],
    ["extend_contours_same_color", "extend_contours_same_color"],
    ["pad_left", "pad_left"],
    ["mirror_vertical", "mirror_vertical"],
    ["mirror_vertical", "pad_left"],
    ["extend_contours_same_color", "pad_left"],
    ["extend_contours_same_color", "mirror_vertical"]]
c15_test = [["pad_left", "extend_contours_same_color"]]

# --- C3 combination definitions ---

c31_train = [["translate_up"], ["mirror_horizontal"], ["rot90"],
    ["translate_up", "translate_up"], ["mirror_horizontal", "mirror_horizontal"], ["rot90", "rot90"],
    ["translate_up", "mirror_horizontal"], ["translate_up", "rot90"],
    ["mirror_horizontal", "rot90"]]

c31_test = [["translate_up", "translate_up", "translate_up"],
    ["translate_up", "translate_up", "mirror_horizontal"],
    ["translate_up", "translate_up", "rot90"],
    ["translate_up", "mirror_horizontal", "mirror_horizontal"],
    ["translate_up", "mirror_horizontal", "rot90"],
    ["translate_up", "rot90", "rot90"],
    ["mirror_horizontal", "mirror_horizontal", "mirror_horizontal"],
    ["mirror_horizontal", "mirror_horizontal", "rot90"],
    ["mirror_horizontal", "rot90", "rot90"],
    ["rot90", "rot90", "rot90"],
    ["translate_up", "mirror_horizontal", "translate_up"],
    ["translate_up", "rot90", "translate_up"],
    ["mirror_horizontal", "rot90", "translate_up"],
    ["mirror_horizontal", "translate_up", "rot90"],
    ["rot90", "translate_up", "mirror_horizontal"],
    ["rot90", "mirror_horizontal", "translate_up"]]

c32_train = [["pad_right"], ["fill_holes_different_color"], ["change_shape_color"],
    ["pad_right", "pad_right"], ["fill_holes_different_color", "fill_holes_different_color"], ["change_shape_color", "change_shape_color"],
    ["pad_right", "fill_holes_different_color"], ["pad_right", "change_shape_color"],
    ["fill_holes_different_color", "change_shape_color"]]

c32_test = [["pad_right", "pad_right", "fill_holes_different_color"],
    ["pad_right", "pad_right", "change_shape_color"],
    ["pad_right", "fill_holes_different_color", "fill_holes_different_color"],
    ["pad_right", "fill_holes_different_color", "change_shape_color"],
    ["pad_right", "change_shape_color", "change_shape_color"],
    ["fill_holes_different_color", "fill_holes_different_color", "fill_holes_different_color"],
    ["fill_holes_different_color", "fill_holes_different_color", "change_shape_color"],
    ["fill_holes_different_color", "change_shape_color", "change_shape_color"],
    ["change_shape_color", "change_shape_color", "change_shape_color"],
    ["pad_right", "fill_holes_different_color", "pad_right"],
    ["pad_right", "change_shape_color", "pad_right"],
    ["fill_holes_different_color", "change_shape_color", "pad_right"],
    ["fill_holes_different_color", "pad_right", "change_shape_color"],
    ["change_shape_color", "pad_right", "fill_holes_different_color"],
    ["change_shape_color", "fill_holes_different_color", "pad_right"]]

c33_train = [["rot90"], ["pad_top"], ["crop_bottom_side"],
    ["rot90", "rot90"], ["pad_top", "pad_top"], ["crop_bottom_side", "crop_bottom_side"],
    ["rot90", "pad_top"], ["rot90", "crop_bottom_side"],
    ["pad_top", "crop_bottom_side"]]

c33_test = [["rot90", "rot90", "rot90"],
    ["rot90", "rot90", "pad_top"],
    ["rot90", "rot90", "crop_bottom_side"],
    ["rot90", "pad_top", "pad_top"],
    ["rot90", "pad_top", "crop_bottom_side"],
    ["rot90", "crop_bottom_side", "crop_bottom_side"],
    ["pad_top", "pad_top", "pad_top"],
    ["pad_top", "pad_top", "crop_bottom_side"],
    ["pad_top", "crop_bottom_side", "crop_bottom_side"],
    ["crop_bottom_side", "crop_bottom_side", "crop_bottom_side"],
    ["rot90", "pad_top", "rot90"],
    ["rot90", "crop_bottom_side", "rot90"],
    ["pad_top", "crop_bottom_side", "rot90"],
    ["pad_top", "rot90", "crop_bottom_side"],
    ["crop_bottom_side", "rot90", "pad_top"],
    ["crop_bottom_side", "pad_top", "rot90"]]

c34_train = [["double_right"], ["crop_contours"], ["change_shape_color"],
    ["double_right", "double_right"], ["crop_contours", "crop_contours"], ["change_shape_color", "change_shape_color"],
    ["double_right", "crop_contours"], ["double_right", "change_shape_color"],
    ["crop_contours", "change_shape_color"]]

c34_test = [["double_right", "double_right", "crop_contours"],
    ["double_right", "double_right", "change_shape_color"],
    ["double_right", "crop_contours", "crop_contours"],
    ["double_right", "crop_contours", "change_shape_color"],
    ["double_right", "change_shape_color", "change_shape_color"],
    ["crop_contours", "crop_contours", "change_shape_color"],
    ["crop_contours", "change_shape_color", "change_shape_color"],
    ["change_shape_color", "change_shape_color", "change_shape_color"],
    ["double_right", "crop_contours", "double_right"],
    ["double_right", "change_shape_color", "double_right"],
    ["crop_contours", "change_shape_color", "double_right"],
    ["crop_contours", "double_right", "change_shape_color"],
    ["change_shape_color", "double_right", "crop_contours"],
    ["change_shape_color", "crop_contours", "double_right"]]

c35_train = [["extend_contours_same_color"], ["pad_left"], ["mirror_vertical"],
    ["extend_contours_same_color", "extend_contours_same_color"], ["pad_left", "pad_left"], ["mirror_vertical", "mirror_vertical"],
    ["extend_contours_same_color", "pad_left"], ["extend_contours_same_color", "mirror_vertical"],
    ["pad_left", "mirror_vertical"]]

c35_test = [["extend_contours_same_color", "extend_contours_same_color", "pad_left"],
    ["extend_contours_same_color", "extend_contours_same_color", "mirror_vertical"],
    ["extend_contours_same_color", "pad_left", "pad_left"],
    ["extend_contours_same_color", "pad_left", "mirror_vertical"],
    ["extend_contours_same_color", "mirror_vertical", "mirror_vertical"],
    ["pad_left", "pad_left", "pad_left"],
    ["pad_left", "pad_left", "mirror_vertical"],
    ["pad_left", "mirror_vertical", "mirror_vertical"],
    ["mirror_vertical", "mirror_vertical", "mirror_vertical"],
    ["extend_contours_same_color", "pad_left", "extend_contours_same_color"],
    ["extend_contours_same_color", "mirror_vertical", "extend_contours_same_color"],
    ["pad_left", "mirror_vertical", "extend_contours_same_color"],
    ["pad_left", "extend_contours_same_color", "mirror_vertical"],
    ["mirror_vertical", "extend_contours_same_color", "pad_left"],
    ["mirror_vertical", "pad_left", "extend_contours_same_color"]]


# --- Build configs across grid sizes ---

compositionality_gridsize_config: list[ExperimentEntry] = []
grid_sizes = [30, 40, 50]

for gd in grid_sizes:
    # C1
    compositionality_gridsize_config.append(make_entry(c11_train, 1, 1, "train", gd, gd))
    compositionality_gridsize_config.append(make_entry(c11_test, 1, 1, "test", gd, gd))
    compositionality_gridsize_config.append(make_entry(c12_train, 1, 2, "train", gd, gd))
    compositionality_gridsize_config.append(make_entry(c12_test, 1, 2, "test", gd, gd))
    compositionality_gridsize_config.append(make_entry(c13_train, 1, 3, "train", gd, gd))
    compositionality_gridsize_config.append(make_entry(c13_test, 1, 3, "test", gd, gd))
    compositionality_gridsize_config.append(make_entry(c14_train, 1, 4, "train", gd, gd))
    compositionality_gridsize_config.append(make_entry(c14_test, 1, 4, "test", gd, gd))
    compositionality_gridsize_config.append(make_entry(c15_train, 1, 5, "train", gd, gd))
    compositionality_gridsize_config.append(make_entry(c15_test, 1, 5, "test", gd, gd))
    # C3
    compositionality_gridsize_config.append(make_entry(c31_train, 3, 1, "train", gd, gd))
    compositionality_gridsize_config.append(make_entry(c31_test, 3, 1, "test", gd, gd))
    compositionality_gridsize_config.append(make_entry(c32_train, 3, 2, "train", gd, gd))
    compositionality_gridsize_config.append(make_entry(c32_test, 3, 2, "test", gd, gd))
    compositionality_gridsize_config.append(make_entry(c33_train, 3, 3, "train", gd, gd))
    compositionality_gridsize_config.append(make_entry(c33_test, 3, 3, "test", gd, gd))
    compositionality_gridsize_config.append(make_entry(c34_train, 3, 4, "train", gd, gd))
    compositionality_gridsize_config.append(make_entry(c34_test, 3, 4, "test", gd, gd))
    compositionality_gridsize_config.append(make_entry(c35_train, 3, 5, "train", gd, gd))
    compositionality_gridsize_config.append(make_entry(c35_test, 3, 5, "test", gd, gd))
