from arcworld.config import DatasetConfig
from experiment_configs.entry import ExperimentEntry

SHAPE_CONDITIONALS = [
    "is_shape_less_than_6_rows",
    "is_shape_less_than_6_cols",
    "is_shape_more_than_2_cell",
    "is_shape_fully_connected",
    "is_shape_evenly_colored",
    "is_shape_not_hollow",
    # "is_shape_not_symmetric",
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


# ID splits
id_transformations_setting_1_exp_1 = [
    # Elementary transformations (4)
    ["translate_up"],
    ["translate_left"],
    ["translate_right"],
    ["translate_down"],
    # Composite transformations (depth 2) (15)
    ["translate_up", "translate_left"],
    ["translate_up", "translate_down"],
    ["translate_up", "translate_up"],
    ["translate_left", "translate_up"],
    ["translate_left", "translate_left"],
    ["translate_left", "translate_right"],
    ["translate_left", "translate_down"],
    ["translate_right", "translate_up"],
    ["translate_right", "translate_left"],
    ["translate_right", "translate_right"],
    ["translate_right", "translate_down"],
    ["translate_down", "translate_up"],
    ["translate_down", "translate_left"],
    ["translate_down", "translate_right"],
    ["translate_down", "translate_down"],
]

ood_transformations_setting_1_exp_1 = [
    # See whether the model can recognize task tokens and use them in a basic scenario with basic composite transformations.
    ["translate_up", "translate_right"]
]

id_transformations_setting_1_exp_2 = [
    # NOTE: "double_*" is "duplicate_*" in the paper.
    # Elementary transformations (4)
    ["double_up"],
    ["double_left"],
    ["double_right"],
    ["double_down"],
    # Composite transformations (depth 2) (15)
    ["double_up", "double_left"],
    ["double_up", "double_down"],
    ["double_up", "double_up"],
    ["double_left", "double_up"],
    ["double_left", "double_left"],
    ["double_left", "double_right"],
    ["double_left", "double_down"],
    ["double_right", "double_up"],
    ["double_right", "double_left"],
    ["double_right", "double_right"],
    ["double_right", "double_down"],
    ["double_down", "double_up"],
    ["double_down", "double_left"],
    ["double_down", "double_right"],
    ["double_down", "double_down"],
]

ood_transformations_setting_1_exp_2 = [
    # See whether the model can recognize task tokens and use them in a basic scenario with basic composite transformations.
    # This set of transformation may be more difficult than the one in Experiment 1 which only involves translations.
    ["double_up", "double_right"]
]


# Define the experiments' datasets
compgen_basics_experiments: list[ExperimentEntry] = []

# ------------------------------
# Basic Setting 1, Experiment 1
# ------------------------------
# ID splits
compgen_basics_experiments.append(
    ExperimentEntry(
        cfg=_base_cfg(n_examples=1, transformations=id_transformations_setting_1_exp_1),
        setting=1,
        experiment=1,
        split="train",
    )
)

# OOD splits
compgen_basics_experiments.append(
    ExperimentEntry(
        cfg=_base_cfg(n_examples=1, transformations=ood_transformations_setting_1_exp_1),
        setting=1,
        experiment=1,
        split="test",
    )
)

# ------------------------------
# Basic Setting 1, Experiment 2
# ------------------------------
# ID splits
compgen_basics_experiments.append(
    ExperimentEntry(
        cfg=_base_cfg(n_examples=1, transformations=id_transformations_setting_1_exp_2),
        setting=1,
        experiment=2,
        split="train",
    )
)

# OOD splits
compgen_basics_experiments.append(
    ExperimentEntry(
        cfg=_base_cfg(n_examples=1, transformations=ood_transformations_setting_1_exp_2),
        setting=1,
        experiment=2,
        split="test",
    )
)
