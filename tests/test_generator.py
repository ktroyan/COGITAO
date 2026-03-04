import os

import pytest

from arcworld import GeneratorConfig, ParallelGenerator, CogitaoDataset
from arcworld.wrapper.dataset import HDF5CogitaoStore


def test_generator_batch_size_config(temp_h5_path, mock_dataset_cfg):
    """Test that ParallelGenerator respects batch_size configuration."""
    cfg = GeneratorConfig(
        dataset=mock_dataset_cfg,
        output_file=temp_h5_path,
        num_workers=2,
    )

    generator = ParallelGenerator(cfg)
    generator.generate(num_samples=2)

    assert os.path.exists(temp_h5_path)

    store = HDF5CogitaoStore(temp_h5_path)
    assert store.cfg.batch_size == mock_dataset_cfg.batch_size


def test_single_mode_generation(temp_h5_path, mock_dataset_cfg):
    """Test dataset generation and validation.

    ParallelGenerator workers each generate tasks with n_examples=1 (forced
    internally), so num_samples=2 → exactly 2 rows in the store.
    """
    cfg = GeneratorConfig(
        dataset=mock_dataset_cfg,
        output_file=temp_h5_path,
        num_workers=1,
    )

    generator = ParallelGenerator(cfg)
    num_samples = 2
    generator.generate(num_samples=num_samples)

    torch = pytest.importorskip("torch")

    dataset = CogitaoDataset(temp_h5_path)
    # Workers use n_examples=1 internally → each queued task = 1 pair = 1 row
    assert len(dataset) == num_samples

    sample = dataset[0]
    assert isinstance(sample, dict)
    assert "inputs" in sample
    assert "outputs" in sample

    # Each row is a single padded (H, W) grid
    assert isinstance(sample["inputs"], torch.Tensor)
    assert sample["inputs"].shape == (
        mock_dataset_cfg.max_grid_size,
        mock_dataset_cfg.max_grid_size,
    )

    for item in dataset:
        assert item["inputs"].shape == (
            mock_dataset_cfg.max_grid_size,
            mock_dataset_cfg.max_grid_size,
        )
