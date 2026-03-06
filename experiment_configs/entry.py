from dataclasses import dataclass

from arcworld.config import DatasetConfig


@dataclass
class ExperimentEntry:
    """A single experiment split configuration.

    Attributes:
        cfg: Dataset generation parameters (transformations, grid size, shapes, etc.)
        setting: Experiment setting number.
        experiment: Experiment number within the setting.
        split: Role declaration — "train" or "test".
              "train" entries expand to train/val/test splits.
              "test" entries expand to val_ood/test_ood splits.
        subdir: Optional extra subdirectory inserted into the output path
                (e.g. "grid_size_30" for grid-size experiments).
    """

    cfg: DatasetConfig
    setting: int
    experiment: int
    split: str  # "train" or "test"
    subdir: str | None = None
