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


test_config_for_klim: list[ExperimentEntry] = []

# Train: translate_up
test_config_for_klim.append(
    ExperimentEntry(
        cfg=_base_cfg(n_examples=1, transformations=[["translate_up"]]),
        setting=1,
        experiment=1,
        split="train",
    )
)

# Test: translate_down
test_config_for_klim.append(
    ExperimentEntry(
        cfg=_base_cfg(n_examples=1, transformations=[["translate_down"]]),
        setting=1,
        experiment=1,
        split="test",
    )
)
