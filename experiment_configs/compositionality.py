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


def make_entry(combos, setting, exp_number, split, min_size=15, max_size=15):
    cfg = DatasetConfig(
        **_BASE,
        allowed_combinations=combos,
        min_grid_size=min_size,
        max_grid_size=max_size,
    )
    return ExperimentEntry(cfg=cfg, setting=setting, experiment=exp_number, split=split)


compositionality_configs: list[ExperimentEntry] = []

# === Setting 1 ===

# Exp 1
compositionality_configs.append(make_entry(
    [["translate_up"], ["rot90"], ["mirror_horizontal"],
     ["translate_up", "translate_up"], ["rot90", "rot90"],
     ["mirror_horizontal", "mirror_horizontal"],
     ["translate_up", "mirror_horizontal"],
     ["rot90", "mirror_horizontal"],
     ["mirror_horizontal", "rot90"]],
    1, 1, "train"))

compositionality_configs.append(make_entry(
    [["translate_up", "rot90"], ["rot90", "translate_up"]],
    1, 1, "test"))

# Exp 2
compositionality_configs.append(make_entry(
    [["pad_right"], ["fill_holes_different_color"], ["change_shape_color"],
     ["pad_right", "pad_right"],
     ["fill_holes_different_color", "fill_holes_different_color"],
     ["change_shape_color", "change_shape_color"],
     ["pad_right", "fill_holes_different_color"],
     ["fill_holes_different_color", "change_shape_color"],
     ["change_shape_color", "fill_holes_different_color"],
     ["change_shape_color", "pad_right"]],
    1, 2, "train"))

compositionality_configs.append(make_entry(
    [["pad_right", "change_shape_color"], ["change_shape_color", "pad_right"]],
    1, 2, "test"))

# Exp 3
compositionality_configs.append(make_entry(
    [["rot90"], ["pad_top"], ["crop_bottom_side"],
     ["rot90", "rot90"],
     ["pad_top", "pad_top"],
     ["crop_bottom_side", "crop_bottom_side"],
     ["rot90", "pad_top"],
     ["pad_top", "crop_bottom_side"],
     ["crop_bottom_side", "rot90"]],
    1, 3, "train"))

compositionality_configs.append(make_entry(
    [["rot90", "crop_bottom_side"], ["crop_bottom_side", "rot90"]],
    1, 3, "test"))

# Exp 4
compositionality_configs.append(make_entry(
    [["double_right"], ["crop_contours"], ["change_shape_color"],
     ["double_right", "double_right"],
     ["crop_contours", "crop_contours"],
     ["change_shape_color", "change_shape_color"],
     ["double_right", "change_shape_color"],
     ["crop_contours", "change_shape_color"],
     ["change_shape_color", "double_right"]],
    1, 4, "train"))

compositionality_configs.append(make_entry(
    [["double_right", "crop_contours"], ["crop_contours", "double_right"]],
    1, 4, "test"))

# Exp 5
compositionality_configs.append(make_entry(
    [["extend_contours_same_color"], ["pad_left"], ["mirror_vertical"],
     ["extend_contours_same_color", "extend_contours_same_color"],
     ["pad_left", "pad_left"],
     ["mirror_vertical", "mirror_vertical"],
     ["mirror_vertical", "pad_left"],
     ["extend_contours_same_color", "pad_left"],
     ["extend_contours_same_color", "mirror_vertical"]],
    1, 5, "train"))

compositionality_configs.append(make_entry(
    [["pad_left", "extend_contours_same_color"], ["extend_contours_same_color", "pad_left"]],
    1, 5, "test"))

# === Setting 2 ===

# Exp 1
compositionality_configs.append(make_entry(
    [["translate_up", "translate_up"],
     ["rot90", "rot90"],
     ["mirror_horizontal", "mirror_horizontal"],
     ["translate_up", "mirror_horizontal"],
     ["rot90", "mirror_horizontal"],
     ["mirror_horizontal", "rot90"]],
    2, 1, "train"))

compositionality_configs.append(make_entry(
    [["translate_up", "rot90"], ["rot90", "translate_up"]],
    2, 1, "test"))

# Exp 2
compositionality_configs.append(make_entry(
    [["pad_right", "pad_right"],
     ["fill_holes_different_color", "fill_holes_different_color"],
     ["change_shape_color", "change_shape_color"],
     ["pad_right", "fill_holes_different_color"],
     ["fill_holes_different_color", "change_shape_color"],
     ["change_shape_color", "fill_holes_different_color"],
     ["change_shape_color", "pad_right"]],
    2, 2, "train"))

compositionality_configs.append(make_entry(
    [["pad_right", "change_shape_color"], ["change_shape_color", "pad_right"]],
    2, 2, "test"))

# Exp 3
compositionality_configs.append(make_entry(
    [["rot90", "rot90"],
     ["pad_top", "pad_top"],
     ["crop_bottom_side", "crop_bottom_side"],
     ["rot90", "pad_top"],
     ["pad_top", "crop_bottom_side"],
     ["crop_bottom_side", "rot90"]],
    2, 3, "train"))

compositionality_configs.append(make_entry(
    [["rot90", "crop_bottom_side"], ["crop_bottom_side", "rot90"]],
    2, 3, "test"))

# Exp 4
compositionality_configs.append(make_entry(
    [["double_right", "double_right"],
     ["crop_contours", "crop_contours"],
     ["change_shape_color", "change_shape_color"],
     ["double_right", "change_shape_color"],
     ["crop_contours", "change_shape_color"],
     ["change_shape_color", "double_right"]],
    2, 4, "train"))

compositionality_configs.append(make_entry(
    [["double_right", "crop_contours"], ["crop_contours", "double_right"]],
    2, 4, "test"))

# Exp 5
compositionality_configs.append(make_entry(
    [["extend_contours_same_color", "extend_contours_same_color"],
     ["pad_left", "pad_left"],
     ["mirror_vertical", "mirror_vertical"],
     ["mirror_vertical", "pad_left"],
     ["extend_contours_same_color", "pad_left"],
     ["extend_contours_same_color", "mirror_vertical"]],
    2, 5, "train"))

compositionality_configs.append(make_entry(
    [["pad_left", "extend_contours_same_color"], ["extend_contours_same_color", "pad_left"]],
    2, 5, "test"))

# === Setting 3 ===

# Exp 1
compositionality_configs.append(make_entry(
    [["translate_up"], ["mirror_horizontal"], ["rot90"],
     ["translate_up", "translate_up"], ["mirror_horizontal", "mirror_horizontal"], ["rot90", "rot90"],
     ["translate_up", "mirror_horizontal"], ["translate_up", "rot90"],
     ["mirror_horizontal", "rot90"]],
    3, 1, "train", 20, 20))

compositionality_configs.append(make_entry(
    [["translate_up", "translate_up", "translate_up"],
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
     ["rot90", "mirror_horizontal", "translate_up"]],
    3, 1, "test", 20, 20))

# Exp 2
compositionality_configs.append(make_entry(
    [["pad_right"], ["fill_holes_different_color"], ["change_shape_color"],
     ["pad_right", "pad_right"], ["fill_holes_different_color", "fill_holes_different_color"], ["change_shape_color", "change_shape_color"],
     ["pad_right", "fill_holes_different_color"], ["pad_right", "change_shape_color"],
     ["fill_holes_different_color", "change_shape_color"]],
    3, 2, "train", 20, 20))

compositionality_configs.append(make_entry(
    [["pad_right", "pad_right", "pad_right"],
     ["pad_right", "pad_right", "fill_holes_different_color"],
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
     ["change_shape_color", "fill_holes_different_color", "pad_right"]],
    3, 2, "test", 20, 20))

# Exp 3
compositionality_configs.append(make_entry(
    [["rot90"], ["pad_top"], ["crop_bottom_side"],
     ["rot90", "rot90"], ["pad_top", "pad_top"], ["crop_bottom_side", "crop_bottom_side"],
     ["rot90", "pad_top"], ["rot90", "crop_bottom_side"],
     ["pad_top", "crop_bottom_side"]],
    3, 3, "train", 20, 20))

compositionality_configs.append(make_entry(
    [["rot90", "rot90", "rot90"],
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
     ["crop_bottom_side", "pad_top", "rot90"]],
    3, 3, "test", 20, 20))

# Exp 4
compositionality_configs.append(make_entry(
    [["double_right"], ["crop_contours"], ["change_shape_color"],
     ["double_right", "double_right"], ["crop_contours", "crop_contours"],
     ["change_shape_color", "change_shape_color"],
     ["double_right", "crop_contours"], ["double_right", "change_shape_color"],
     ["crop_contours", "change_shape_color"]],
    3, 4, "train", 20, 20))

compositionality_configs.append(make_entry(
    [["double_right", "double_right", "crop_contours"],
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
     ["change_shape_color", "crop_contours", "double_right"]],
    3, 4, "test", 20, 20))

# Exp 5
compositionality_configs.append(make_entry(
    [["extend_contours_same_color"], ["pad_left"], ["mirror_vertical"],
     ["extend_contours_same_color", "extend_contours_same_color"], ["pad_left", "pad_left"], ["mirror_vertical", "mirror_vertical"],
     ["extend_contours_same_color", "pad_left"], ["extend_contours_same_color", "mirror_vertical"],
     ["pad_left", "mirror_vertical"]],
    3, 5, "train", 20, 20))

compositionality_configs.append(make_entry(
    [["extend_contours_same_color", "extend_contours_same_color", "extend_contours_same_color"],
     ["extend_contours_same_color", "extend_contours_same_color", "pad_left"],
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
     ["mirror_vertical", "pad_left", "extend_contours_same_color"]],
    3, 5, "test", 20, 20))
