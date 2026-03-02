import copy

# Default parameters common to all experiments
cp_path = "c0_million"

cp_base_config = {
    "min_n_shapes_per_grid": 2,
    "max_n_shapes_per_grid": 2,
    "n_examples": 1,
    "min_grid_size": 20,
    "max_grid_size": 20,
    "allowed_combinations": None,
    "allowed_transformations": None,
    "min_transformation_depth": None,
    "max_transformation_depth": None,
    "shape_compulsory_conditionals": ["is_shape_less_than_6_rows", 
                                      "is_shape_less_than_6_cols", 
                                      "is_shape_fully_connected"],
    "saving_path": None,
}

def make_config(combos, setting, exp_number, split, min_size = 15, max_size = 15):
    config = copy.deepcopy(cp_base_config)
    config["allowed_combinations"] = combos
    config["saving_path"] = f"{cp_path}/exp_setting_{setting}/experiment_{exp_number}/{split}.json"
    config["min_grid_size"] = min_size
    config["max_grid_size"] = max_size
    return config

compositionality_configs = []


# # === Setting 1 ===

# Exp 1

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

# Pick exactly ONE pair to hold out for test:
held_out = [["translate_up", "translate_right"], ["translate_right", "translate_up"]]

# TRAIN = all combinations except held-out
c10_train_ops = single_ops + [op for op in double_ops]
c20_train_ops = [op for op in double_ops]
c30_train_ops = single_ops + [op for op in double_ops] + held_out


compositionality_configs.append(make_config(
    c10_train_ops,
    1, 0, "train"
))

# TEST = only the held-out combination
compositionality_configs.append(make_config(
    held_out,
    1, 0, "test"
))

compositionality_configs.append(make_config(
    c20_train_ops,
    2, 0, "train"
))

# # TEST = only the held-out combination
compositionality_configs.append(make_config(
    held_out,
    2, 0, "test"
))

compositionality_configs.append(make_config(
    c30_train_ops,
    3, 0, "train",
))

compositionality_configs.append(make_config(
    triple_ops,
    3, 0, "test",
))
