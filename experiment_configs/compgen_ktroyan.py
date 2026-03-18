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
        min_grid_size=15,
        max_grid_size=15,
        min_n_shapes_per_grid=2,
        max_n_shapes_per_grid=2,
        allowed_combinations=transformations,
        min_transformation_depth=None,
        max_transformation_depth=None,
        shape_compulsory_conditionals=SHAPE_CONDITIONALS,
    )


# ID splits
id_transformations = [
    # Elementary transformations (7)
    ["translate_up"],
    ["translate_left"],
    ["translate_right"],
    ["translate_down"],
    ["rot90"],
    ["mirror_horizontal"],
    ["mirror_vertical"],
    # Composite transformations (depth 2) (35)
    ["translate_up", "translate_left"],
    ["translate_up", "translate_right"],
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
    ["translate_up", "rot90"],
    ["translate_left", "rot90"],
    ["translate_right", "rot90"],
    ["translate_down", "rot90"],
    ["translate_up", "mirror_horizontal"],
    ["translate_left", "mirror_horizontal"],
    ["translate_right", "mirror_horizontal"],
    ["translate_down", "mirror_horizontal"],
    ["mirror_horizontal", "mirror_horizontal"],
    ["mirror_horizontal", "translate_up"],
    ["mirror_horizontal", "translate_left"],
    ["mirror_horizontal", "translate_right"],
    ["mirror_horizontal", "translate_down"],
    ["mirror_horizontal", "rot90"],
    ["mirror_vertical", "mirror_vertical"],
    ["mirror_vertical", "translate_up"],
    ["mirror_vertical", "translate_left"],
    ["mirror_vertical", "translate_right"],
    ["mirror_vertical", "translate_down"],
]

id_transformations_setting_1 = id_transformations.copy() + [["rot90", "mirror_horizontal"], ["rot90", "mirror_vertical"], ["mirror_vertical", "rot90"]]
id_transformations_setting_2 = id_transformations.copy() + [["rot90", "translate_up"], ["rot90", "mirror_vertical"], ["mirror_vertical", "rot90"]]
id_transformations_setting_3 = id_transformations.copy() + [["rot90", "translate_up"], ["rot90", "mirror_horizontal"]]
id_transformations_setting_4 = id_transformations.copy() + [["rot90", "translate_up"], ["rot90", "mirror_horizontal"], ["rot90", "mirror_vertical"], ["mirror_vertical", "rot90"]]

ood_transformations_setting_1 = [
    # OOD Type 1: Baseline for commutativity. The model has seen the commutated version which is equivalent, so it should do well if it can recognize task tokens and relate them to the underlying transformations.
    ["rot90", "translate_up"]
]

ood_transformations_setting_2 = [
    # OOD Type 2: Non-commutative composite transformation. The model needs to understand sequence order in order to predict this correctly since it is not commutative.
    ["rot90", "mirror_horizontal"]
]

ood_transformations_setting_3 = [
    # OOD Type 3: Novel composite transformation (depth 2). The model has not seen this specific combination nor its commutated version (even though they are not equivalent).
    ["rot90", "mirror_vertical"]
]

ood_transformations_setting_4 = [
    # OOD Type 4: Novel composite transformation (depth 3). The model has not seen this specific combination depth, but it has seen the individual transformations and their depth 2 combinations.
    ["translate_up", "mirror_horizontal", "rot90"]
]

# Define the experiments' datasets
compgen_ktroyan_experiments: list[ExperimentEntry] = []

# ------------------------------
# Setting 1, Experiment 1
# ------------------------------
# ID splits
compgen_ktroyan_experiments.append(
    ExperimentEntry(
        cfg=_base_cfg(n_examples=1, transformations=id_transformations_setting_1),
        setting=1,
        experiment=1,
        split="train",
    )
)

# OOD splits
compgen_ktroyan_experiments.append(
    ExperimentEntry(
        cfg=_base_cfg(n_examples=1, transformations=ood_transformations_setting_1),
        setting=1,
        experiment=1,
        split="test",
    )
)

# ------------------------------
# Setting 2, Experiment 1
# ------------------------------
# ID splits
compgen_ktroyan_experiments.append(
    ExperimentEntry(
        cfg=_base_cfg(n_examples=1, transformations=id_transformations_setting_2),
        setting=2,
        experiment=1,
        split="train",
    )
)

# OOD splits
compgen_ktroyan_experiments.append(
    ExperimentEntry(
        cfg=_base_cfg(n_examples=1, transformations=ood_transformations_setting_2),
        setting=2,
        experiment=1,
        split="test",
    )
)

# ------------------------------
# Setting 3, Experiment 1
# ------------------------------
# ID splits
compgen_ktroyan_experiments.append(
    ExperimentEntry(
        cfg=_base_cfg(n_examples=1, transformations=id_transformations_setting_3),
        setting=3,
        experiment=1,
        split="train",
    )
)

# OOD splits
compgen_ktroyan_experiments.append(
    ExperimentEntry(
        cfg=_base_cfg(n_examples=1, transformations=ood_transformations_setting_3),
        setting=3,
        experiment=1,
        split="test",
    )
)

# ------------------------------
# Setting 4, Experiment 1
# ------------------------------
# ID splits
compgen_ktroyan_experiments.append(
    ExperimentEntry(
        cfg=_base_cfg(n_examples=1, transformations=id_transformations_setting_4),
        setting=4,
        experiment=1,
        split="train",
    )
)

# OOD splits
compgen_ktroyan_experiments.append(
    ExperimentEntry(
        cfg=_base_cfg(n_examples=1, transformations=ood_transformations_setting_4),
        setting=4,
        experiment=1,
        split="test",
    )
)
