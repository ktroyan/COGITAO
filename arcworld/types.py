"""Shared TypedDicts for Generator and Dataset outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import TypedDict

if TYPE_CHECKING:
    import torch
    from .transformations.shape_transformations import TransformationType
else:
    TransformationType = object  # Placeholder for type checking only


class TaskPair(TypedDict):
    """A single input/output example within a task."""

    input: np.ndarray
    output: np.ndarray
    n_shapes: int
    grid_size: tuple[int, int]
    full_grid_sequence: list[np.ndarray | None]


class Task(TypedDict):
    """Output of Generator.generate_single_task()."""

    pairs: list[TaskPair]
    transformation_suite: list[TransformationType]


class CogitaoSampleNumpy(TypedDict):
    """Raw sample returned by HDF5CogitaoStore (numpy arrays)."""

    inputs: np.ndarray
    outputs: np.ndarray
    grid_sizes: np.ndarray
    n_shapes: np.ndarray
    transformation_suite: list[TransformationType]
    full_grid_sequence: np.ndarray
    seq_len: int


class CogitaoSample(TypedDict):
    """Sample returned by CogitaoDataset (torch tensors)."""

    inputs: torch.Tensor
    outputs: torch.Tensor
    grid_sizes: torch.Tensor
    n_shapes: torch.Tensor
    transformation_suite: list[TransformationType]
    full_grid_sequence: torch.Tensor
    seq_len: torch.Tensor
