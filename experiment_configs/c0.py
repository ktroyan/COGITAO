from arcworld.config import DatasetConfig

from .entry import ExperimentEntry

# Default parameters common to all experiments
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


# All single ops
single_ops = [
    ["translate_up"],
    ["translate_down"],
    ["translate_left"],
    ["translate_right"],
]

# All two-operation compositions (ordered)
double_ops = [
    ["translate_up", "translate_up"],
    ["translate_up", "translate_down"],
    ["translate_up", "translate_left"],
    ["translate_down", "translate_up"],
    ["translate_down", "translate_down"],
    ["translate_down", "translate_left"],
    ["translate_down", "translate_right"],
    ["translate_left", "translate_up"],
    ["translate_left", "translate_down"],
    ["translate_left", "translate_left"],
    ["translate_left", "translate_right"],
    ["translate_right", "translate_down"],
    ["translate_right", "translate_left"],
    ["translate_right", "translate_right"],
]

# All ordered triples
triple_ops = [[a[0], b[0], c[0]] for a in single_ops for b in single_ops for c in single_ops]

# Pick exactly ONE pair to hold out for test
held_out = [["translate_up", "translate_right"], ["translate_right", "translate_up"]]

# TRAIN = all combinations except held-out
c10_train_ops = single_ops + double_ops
c20_train_ops = list(double_ops)
c30_train_ops = single_ops + double_ops + held_out

c0_configs: list[ExperimentEntry] = [
    # === Setting 1 ===
    make_entry(c10_train_ops, 1, 0, "train"),
    make_entry(held_out, 1, 0, "test"),
    # === Setting 2 ===
    make_entry(c20_train_ops, 2, 0, "train"),
    make_entry(held_out, 2, 0, "test"),
    # === Setting 3 ===
    make_entry(c30_train_ops, 3, 0, "train"),
    make_entry(triple_ops, 3, 0, "test"),
]
