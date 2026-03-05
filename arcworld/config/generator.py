from pydantic import BaseModel

from .data import DatasetConfig


class GeneratorConfig(BaseModel):
    """Configuration for the Task Generator

    This config defines how to generate synthetic tasks using the COGITAO generator.
    It controls grid size, shapes, transformations, and output formatting.
    """

    dataset: DatasetConfig

    # Output settings
    output_file: str
    output_dir: str = "."

    # Generation
    num_workers: int = 16
